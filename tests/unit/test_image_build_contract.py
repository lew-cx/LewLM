"""Static contract for the container build files.

These checks run on every OS without Docker. They pin the layering rules that
make an app-only rebuild cheap (dependencies keyed on ``requirements/``,
application wheel installed last with ``--no-deps``, caches in BuildKit
mounts) so a later edit cannot quietly reintroduce ``COPY . .`` before the
dependency install or ``PIP_NO_CACHE_DIR=1`` in the builder. Actual build
timings are produced by ``scripts/docker/measure_rebuild.sh`` on a Docker host.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILES = (REPO_ROOT / "Dockerfile", REPO_ROOT / "Dockerfile.cuda")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _stage(text: str, name: str) -> str:
    """Return the body of the ``FROM ... AS name`` stage."""

    match = re.search(rf"^FROM [^\n]+ AS {re.escape(name)}\n(.*?)(?=^FROM |\Z)", text, re.M | re.S)
    assert match is not None, f"stage {name!r} missing"
    return match.group(1)


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.name)
def test_dependencies_install_before_application_source(path: Path) -> None:
    text = _text(path)
    deps = _stage(text, "deps")
    app = _stage(text, "app")

    assert "COPY . ." not in text, "copying the whole tree into a build stage invalidates the dependency layer"
    assert re.search(r"^COPY \$\{DEPENDENCY_INPUT\} \./requirements\.txt", deps, re.M)
    assert "COPY src" not in deps and "COPY pyproject.toml" not in deps
    assert "pip install -r requirements.txt" in deps
    assert re.search(r"^COPY pyproject\.toml README\.md LICENSE MANIFEST\.in \./", app, re.M)
    assert "pip wheel --no-deps" in app
    assert 'pip install --no-deps "${wheel}"' in app


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.name)
def test_native_builds_are_bounded_and_cached(path: Path) -> None:
    text = _text(path)

    assert "PIP_NO_CACHE_DIR" not in text
    assert "$(nproc)" not in text, "unbounded parallelism exhausts memory on small runners"
    assert re.search(r"^ARG BUILD_JOBS=4$", text, re.M)
    assert "CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}" in text
    assert '-j "${BUILD_JOBS}"' in text
    assert "--mount=type=cache" in text
    assert "target=/root/.cache/ccache" in text
    assert "-DCMAKE_C_COMPILER_LAUNCHER=ccache" in text
    # FORCE_CMAKE belongs to the source-build branch only, never stage-wide.
    assert not re.search(r"^\s*ENV [^\n]*FORCE_CMAKE", text, re.M)
    assert "FORCE_CMAKE=1" in _stage(text, "deps")
    # The wheel-index path and the source path are both present and exclusive.
    assert "LLAMA_CPP_PYTHON_WHEEL_INDEX" in text
    assert "--only-binary llama-cpp-python" in text
    assert "--no-binary llama-cpp-python" in text


def test_cpu_image_stays_portable_and_verifies_itself() -> None:
    text = _text(REPO_ROOT / "Dockerfile")

    assert 'ARG LLAMA_CMAKE_ARGS="-DGGML_NATIVE=OFF"' in text
    assert "-DGGML_NATIVE=OFF" in _stage(text, "llamacpp-tools-enabled")
    assert 'ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"' in text
    assert 'ARG LLAMA_BUILD_EXPECT="cpu"' in text
    assert 'verify_llamacpp_build.py --expect "${LLAMA_BUILD_EXPECT}"' in text
    for flavor in ("bridge", "serving", "full"):
        assert f"{flavor})" in _stage(text, "app")
    assert "LEWLM_IMAGE_FLAVOR=${IMAGE_FLAVOR}" in text


def test_cuda_image_validates_architectures_and_proves_offload() -> None:
    text = _text(REPO_ROOT / "Dockerfile.cuda")
    deps = _stage(text, "deps")

    assert "validate_cuda_archs.sh" in deps
    assert deps.index("validate_cuda_archs.sh \"${CUDA_ARCHITECTURES}\"") < deps.index("pip install"), (
        "architectures must be validated before any compilation"
    )
    assert "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache" in text
    assert 'ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"' in text
    assert "12.6.2-devel-ubuntu24.04" in text and "12.6.2-runtime-ubuntu24.04" in text
    assert "verify_llamacpp_build.py --expect gpu --hint cuda" in _stage(text, "app")
    assert 'ARG CUDA_ARCHITECTURES="75;80;86;89"' in text


def test_compose_and_ci_use_the_flavor_contract() -> None:
    compose = _text(REPO_ROOT / "docker-compose.yml")
    ci = _text(REPO_ROOT / ".github" / "workflows" / "ci.yml")
    env_example = _text(REPO_ROOT / ".env.example")

    assert compose.count('IMAGE_FLAVOR: "${LEWLM_DOCKER_IMAGE_FLAVOR:-full}"') == 2
    assert "BUILD_JOBS" in compose and "DEPENDENCY_INPUT" in compose
    assert "TORCH_INDEX_URL" in compose
    assert "LEWLM_DOCKER_IMAGE_FLAVOR" in env_example
    # Fast CI boots the bridge flavor without conversion tools; the full image
    # has its own gate that runs the rebuild measurement.
    assert "--build-arg IMAGE_FLAVOR=bridge" in ci
    assert "--build-arg CONVERSION_TOOLS=disabled" in ci
    assert "measure_rebuild.sh --flavor full" in ci
    assert "storage_access" in ci


def test_dockerignore_keeps_wheel_inputs_and_drops_the_rest() -> None:
    ignore = _text(REPO_ROOT / ".dockerignore").splitlines()

    assert "tests/" in ignore and "docs/" in ignore and "examples/" in ignore
    assert "*.md" in ignore and "!README.md" in ignore
    for needed in ("requirements", "src", "scripts", "pyproject.toml", "LICENSE", "MANIFEST.in"):
        assert needed not in ignore and f"{needed}/" not in ignore


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sh script")
def test_cuda_arch_validator_rejects_unsupported_and_native(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "docker" / "validate_cuda_archs.sh"
    toolkit = tmp_path / "nvcc-list.txt"
    toolkit.write_text("compute_75\ncompute_80\ncompute_86\ncompute_89\ncompute_90\n", encoding="utf-8")

    def run(archs: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["sh", str(script), archs, "--list-file", str(toolkit)],
            capture_output=True,
            text=True,
            check=False,
        )

    assert run("75;80;86;89").returncode == 0
    assert run("89-real,90-virtual").returncode == 0
    blackwell = run("120")
    assert blackwell.returncode == 1
    assert "compute_120 is NOT supported" in blackwell.stderr
    assert "compute_90" in blackwell.stderr, "the supported list must be shown"
    assert run("native").returncode == 1
    assert run("all-major").returncode == 1
    assert run("abc").returncode == 1
    assert run("").returncode == 2


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX container build shell")
@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("requirement,expected", [
    ("llama-cpp-python>=0.3.0,<1.0.0\n", "llama-cpp-python>=0.3.0,<1.0.0"),
    ("llama-cpp-python==0.3.16 \\\n    --hash=sha256:abc\n", "llama-cpp-python==0.3.16"),
    ("llama-cpp-python==0.3.16 ; python_version >= '3.11' \\\n    --hash=sha256:abc\n", "llama-cpp-python==0.3.16"),
    ("httpx==0.28.1\n", ""),
])
def test_native_requirement_extraction_accepts_hashed_locks(path, requirement, expected, tmp_path):
    # Execute the actual Docker build's extraction command against pip-compile
    # output. A trailing continuation slash makes the native pip build fail.
    command = next(line.strip() for line in _stage(_text(path), "deps").splitlines()
                   if line.strip().startswith('spec="$('))
    command = command.removesuffix("\\").rstrip()
    (tmp_path / "requirements.txt").write_text(requirement, encoding="utf-8")
    result = subprocess.run(["sh", "-eu", "-c", command + '\nprintf "%s" "$spec"'],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout == expected
