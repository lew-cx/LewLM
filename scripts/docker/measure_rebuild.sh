#!/usr/bin/env bash
# Measure clean, no-change, and app-only-change image builds.
#
# Produces the build-cache evidence the modernization roadmap asks for
# (step 03): three timed BuildKit builds with their full plain-progress logs,
# plus a count of native compile steps in each log so "app-only edits perform
# zero llama.cpp/CUDA compilation" is a measured statement, not a belief.
#
# Every build runs on a dedicated buildx builder (`lewlm-measure`) so its layer
# cache and cache mounts (pip, ccache) are isolated from the operator's default
# builder. `--reset` deletes and recreates that builder for a genuinely cold
# start; the default builder and any user caches are never touched.
#
# Usage:
#   scripts/docker/measure_rebuild.sh [--dockerfile Dockerfile] [--flavor full]
#       [--out docs/validation/evidence/rebuild-<stamp>] [--reset]
#       [-- extra docker build args...]
#
# Example (CPU serving image, cold):
#   scripts/docker/measure_rebuild.sh --flavor serving --reset -- --build-arg CONVERSION_TOOLS=disabled
#
# Requires: docker with buildx, git. The app-only edit is made in a disposable
# `git worktree`, never in the working checkout.

set -euo pipefail

dockerfile="Dockerfile"
flavor="full"
out=""
reset=0
builder="lewlm-measure"
extra_args=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dockerfile) dockerfile="$2"; shift 2 ;;
        --flavor) flavor="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --reset) reset=1; shift ;;
        --builder) builder="$2"; shift 2 ;;
        --) shift; extra_args=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo_root"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out=${out:-"docs/validation/evidence/rebuild-${stamp}"}
mkdir -p "$out"
tag="lewlm:measure-${flavor}"

if ! docker buildx version >/dev/null 2>&1; then
    echo "docker buildx is required" >&2
    exit 2
fi
if [ "$reset" -eq 1 ]; then
    docker buildx rm "$builder" >/dev/null 2>&1 || true
fi
if ! docker buildx inspect "$builder" >/dev/null 2>&1; then
    docker buildx create --name "$builder" --driver docker-container >/dev/null
fi

count_native() {
    # Ninja per-object lines ("[12/345] Building CXX object ...") cover both
    # llama-cpp-python's scikit-build compile and the conversion-tool build;
    # "Building wheel for llama-cpp-python" catches a rebuild that hid its
    # object lines behind pip's own progress.
    grep -cE '\] Building (C|CXX|CUDA) object|Building wheel for llama[-_]cpp[-_]python' "$1" || true
}

run_build() {
    local label="$1" context="$2" log="$out/$label.log"
    shift 2
    local start end elapsed
    echo "== $label (context: $context)"
    start=$(date +%s)
    docker buildx build --builder "$builder" --load --progress=plain "$@" \
        -f "$context/$dockerfile" -t "$tag" \
        --build-arg "IMAGE_FLAVOR=$flavor" \
        "${extra_args[@]}" "$context" 2>&1 | tee "$log" >/dev/null
    end=$(date +%s)
    elapsed=$((end - start))
    printf '%s\t%ss\tnative_compile_steps=%s\n' "$label" "$elapsed" "$(count_native "$log")" | tee -a "$out/summary.tsv"
}

: > "$out/summary.tsv"
{
    echo "dockerfile=$dockerfile"
    echo "flavor=$flavor"
    echo "builder=$builder reset=$reset"
    echo "revision=$(git rev-parse HEAD) dirty=$([ -n "$(git status --porcelain)" ] && echo yes || echo no)"
    echo "host=$(uname -srm)"
    echo "docker=$(docker --version)"
    echo "extra_args=${extra_args[*]:-}"
    echo "captured=$stamp"
} > "$out/context.txt"

# `clean` ignores the layer cache; pip/ccache cache mounts are cold only when
# the builder was just created (--reset), and context.txt records which.
run_build clean "$repo_root" --no-cache
run_build no-change "$repo_root"

worktree=$(mktemp -d "${TMPDIR:-/tmp}/lewlm-app-edit.XXXXXX")
rmdir "$worktree"
git worktree add --detach "$worktree" HEAD >/dev/null
trap 'git worktree remove --force "$worktree" >/dev/null 2>&1 || true' EXIT
printf '\n# app-only edit for rebuild measurement %s\n' "$stamp" >> "$worktree/src/lewlm/__init__.py"
run_build app-only "$worktree"

echo
echo "summary ($out/summary.tsv):"
cat "$out/summary.tsv"
app_only_native=$(count_native "$out/app-only.log")
if [ "$app_only_native" != "0" ]; then
    echo "FAIL: the app-only rebuild performed $app_only_native native compile step(s); the dependency layer was invalidated" >&2
    exit 1
fi
echo "OK: the app-only rebuild performed no native compilation"
