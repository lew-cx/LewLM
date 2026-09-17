#!/usr/bin/env bash
# Resolve a pinned, hashed dependency lock for one image flavor on the image's
# own platform.
#
# The dependency inputs in requirements/<flavor>.txt are exported from
# pyproject.toml and deliberately unpinned. A lock turns that moving resolution
# into a fixed input: exact versions plus artifact hashes, resolved inside the
# same base image the Dockerfile uses, so the dependency layer is keyed on a
# file that only changes when someone re-locks. Point the build at it with
# `--build-arg DEPENDENCY_INPUT=requirements/locks/<name>.txt`.
#
# Locks are platform-specific. This script runs pip-tools inside a container
# matching the target (Linux, the selected Python, and for the CUDA image the
# CUDA-matched torch index), so it produces a valid Linux lock even from macOS
# or Windows. It never installs anything into the host environment.
#
# Usage:
#   scripts/docker/lock_dependencies.sh --flavor serving
#   scripts/docker/lock_dependencies.sh --flavor full --torch-index https://download.pytorch.org/whl/cpu
#   scripts/docker/lock_dependencies.sh --flavor full --image nvidia/cuda:12.6.2-devel-ubuntu24.04 \
#       --python python3 --torch-index https://download.pytorch.org/whl/cu126 --label cuda126
#
# Output: requirements/locks/<flavor>[-<label>]-linux-<arch>-py<ver>.txt

set -euo pipefail

flavor="serving"
image="python:3.11-slim"
python_bin="python"
torch_index=""
label=""
platform="linux/amd64"

while [ $# -gt 0 ]; do
    case "$1" in
        --flavor) flavor="$2"; shift 2 ;;
        --image) image="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --torch-index) torch_index="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --platform) platform="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo_root"
input="requirements/${flavor}.txt"
[ -f "$input" ] || { echo "missing dependency input $input (run scripts/export_dependency_inputs.py)" >&2; exit 2; }

python scripts/export_dependency_inputs.py --check

extra_index_args=()
if [ -n "$torch_index" ]; then
    extra_index_args=(--extra-index-url "$torch_index")
fi

mkdir -p requirements/locks
docker run --rm --platform "$platform" \
    -v "$repo_root:/src" -w /src \
    -e "PIP_DISABLE_PIP_VERSION_CHECK=1" \
    "$image" sh -euc '
        py="$1"; input="$2"; flavor="$3"; label="$4"; shift 4
        if ! command -v "$py" >/dev/null 2>&1; then
            apt-get update >/dev/null && apt-get install -y --no-install-recommends python3 python3-venv python3-pip >/dev/null
        fi
        "$py" -m venv /tmp/lock-venv
        . /tmp/lock-venv/bin/activate
        pip install -q --upgrade pip pip-tools
        ver=$(python -c "import sys; print(f\"{sys.version_info[0]}{sys.version_info[1]}\")")
        arch=$(uname -m)
        name="${flavor}${label:+-$label}-linux-${arch}-py${ver}"
        out="requirements/locks/${name}.txt"
        pip-compile --quiet --strip-extras --generate-hashes --allow-unsafe \
            --output-file "$out" "$@" "$input"
        {
            echo "# Locked by scripts/docker/lock_dependencies.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "# Base image: '"$image"'  Python: $(python --version 2>&1)  Platform: $(uname -sm)"
            echo "# Input: $input  Torch index: '"${torch_index:-default}"'"
            cat "$out"
        } > "$out.tmp" && mv "$out.tmp" "$out"
        echo "wrote $out"
    ' sh "$python_bin" "$input" "$flavor" "$label" "${extra_index_args[@]}"
