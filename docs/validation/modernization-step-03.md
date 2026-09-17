# Modernization step 03 validation

Implemented on 2026-09-17 on the baseline Apple Silicon host (Darwin 25.2.0
arm64, Python 3.14.7 venv plus a Homebrew Python 3.11.15 used for clean-venv
simulations). The Docker daemon on this host is unavailable, so every
container build, cached rebuild, CUDA compile, and Linux ISA measurement is
**deferred** with its exact command below. Everything that can be proven
without a daemon was proven here.

## What changed

| Area | Change |
| --- | --- |
| Packaging | New `llamacpp_runtime` extra: `llama-cpp-python` (+ Windows CMake/Ninja helpers) and nothing else. `llamacpp` and `gguf_conversion` are unchanged, so existing installs keep conversion. |
| Dependency inputs | `requirements/{bridge,serving,full}.txt` are exported from `pyproject.toml` by `scripts/export_dependency_inputs.py`; a unit test fails when they drift. `scripts/docker/lock_dependencies.sh` resolves pinned, hashed locks inside the target base image into `requirements/locks/`. |
| `Dockerfile` / `Dockerfile.cuda` | Four stages (toolchain → deps → app → runtime). The deps layer copies only the requirements input; the app layer copies `pyproject.toml`, `README.md`, `LICENSE`, `MANIFEST.in`, `src/` and installs the LewLM wheel `--no-deps`. `IMAGE_FLAVOR=bridge|serving|full` (default `full`). `BUILD_JOBS=4` drives `CMAKE_BUILD_PARALLEL_LEVEL`, `MAX_JOBS`, and `-j`; `$(nproc)` is gone. `ccache` is the C/C++/CUDA compiler launcher with BuildKit cache mounts; `PIP_NO_CACHE_DIR=1` is gone. `llama-cpp-python` is either downloaded as a prebuilt wheel from `LLAMA_CPP_PYTHON_WHEEL_INDEX` (`--only-binary`) or built with `pip wheel` into a cache directory keyed on scope + CMake flags + Python ABI, so a stale wheel can never satisfy a changed flag set; `FORCE_CMAKE=1` exists only on that source path. Torch resolves only from an explicit index (`whl/cpu` or `whl/cu126`) and only in `full`. |
| Build-time proofs | The app stage asserts the flavor's package set (`bridge` has no `llama_cpp`/torch; `serving` has no torch/transformers/weasyprint), runs `scripts/verify_llamacpp_build.py` (`--expect cpu` on the CPU image, `--expect gpu --hint cuda` on the CUDA image), runs `pip check`, and runs `lewlm --help`. `Dockerfile.cuda` validates every requested SM against `nvcc --list-gpu-arch` before compiling and refuses `native`/`all*`. |
| Runtime evidence | Images stamp `LEWLM_IMAGE_FLAVOR`; `install_profiles.container.image_flavor` reports it and lean flavors explain that conversion/documents are absent by design. `install_profiles.storage_access` is a real create/delete write probe of the data directory as the runtime user; `lewlm doctor` prints `data dir writable: yes/NO`. `/v1/health` carries both. |
| Build-flavor detection | `MTL` added as a Metal marker: current llama.cpp reports backend sections (`MTL : …`, `CUDA : …`), which the parser previously missed for Metal. Observed on this host. |
| Compose / env / CI | Compose and `.env.example` expose `IMAGE_FLAVOR`, `BUILD_JOBS`, `LLAMA_CPP_PYTHON_WHEEL_INDEX`, `DEPENDENCY_INPUT`, `TORCH_INDEX_URL`. The fast CI docker job builds `bridge` with `CONVERSION_TOOLS=disabled`, asserts no native compile line in the log, and checks `storage_access`. A new blocking `docker-full` job runs `scripts/docker/measure_rebuild.sh --flavor full` (clean, no-change, app-only) and fails if the app-only rebuild performs any native compile step, then proves the image is GGUF-ready with conversion tools present. |
| Docs | `docs/operations/docker.md` (flavors, build args, layering, locks, measurement, CUDA per-device recipe, portability check), installation guide and README rows for `llamacpp_runtime`, `requirements/README.md`. |

Compatibility: `docker build -t lewlm:cpu .`, `docker build -f Dockerfile.cuda …`,
`docker compose up --build`, and `--build-arg EXTRAS=…` keep working. `EXTRAS`
is now layered on top of the flavor (`dev` still adds pytest). The removed
default `EXTRAS=llamacpp,documents` is exactly the `full` flavor. No LewLM
setting, route, model ID, or response field changed; `image_flavor` and
`storage_access` are additive.

## Portable results on this host

| Command / observation | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_install_profiles.py tests/unit/test_container.py tests/unit/test_dependency_inputs.py tests/unit/test_image_build_contract.py tests/unit/test_verify_llamacpp_build.py tests/unit/test_llamacpp_build_flavor.py tests/unit/test_dependency_audit.py tests/unit/test_cli.py` | **117 passed** |
| Roadmap focused regression command (step 00 list) | see the handoff record in [modernization-baseline.md](modernization-baseline.md) |
| `python scripts/export_dependency_inputs.py --check` | in sync |
| `sh scripts/docker/validate_cuda_archs.sh "75;80;86;89" --list-file <saved cu12.6 list>` / `"120"` / `"native"` / `"89-real,90-virtual"` | 0 / 1 with supported list / 1 / 0 |
| `python scripts/verify_llamacpp_build.py --expect any` on this host's Metal build | `OK: … GPU-offload-capable (accelerator hints: metal)` after the `MTL` fix; `--expect cpu` correctly fails with exit 1 |
| `docker compose config` (no daemon required) | valid; `DEPENDENCY_INPUT` resolves to `requirements/full.txt` by default and `requirements/serving.txt` under `LEWLM_DOCKER_IMAGE_FLAVOR=serving` |
| App-stage simulation, Python 3.11 clean venv: copy exactly `pyproject.toml README.md LICENSE MANIFEST.in src/`, `pip wheel --no-deps .` | `lewlm-0.4.2-py3-none-any.whl` with 185 module files, `LICENSE`, `METADATA`, `entry_points.txt` |
| `bridge` flavor simulation: `pip install -r requirements/bridge.txt` (2.6 s warm cache), `pip install --no-deps <wheel>`, `pip check`, package-honesty assertion, `lewlm --help`, `LEWLM_IN_CONTAINER=true LEWLM_IMAGE_FLAVOR=bridge lewlm doctor --json` | all pass; doctor reports `image_flavor=bridge`, `storage_access.writable=true`, GGUF profile not installed |
| `pip install --dry-run "<wheel>[dev]"` (legacy `EXTRAS` path) | resolves pytest + pytest-asyncio on top of the wheel |
| `serving` flavor simulation: the Dockerfile's exact `pip --cache-dir <keyed> wheel --no-deps --no-binary llama-cpp-python` with `CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_METAL=OFF"`, `CMAKE_BUILD_PARALLEL_LEVEL=8`, then install, `pip check`, honesty assertion, `verify_llamacpp_build.py --expect cpu` | wheel built in 45.8 s wall (513 % CPU, so the parallel level was honored), stored under the keyed cache directory; `pip check` clean; no torch/transformers/weasyprint; **`--expect cpu` failed with exit 1** — see below. [serving-sim.log](evidence/modernization-step-03/serving-sim.log) |
| Same wheel, `verify_llamacpp_build.py --expect gpu --hint metal` | `OK: … GPU-offload-capable (accelerator hints: metal)`, exit 0 |

The serving simulation is a macOS arm64 build of the same pip command line;
it proves the command shape and the keyed wheel cache. It is **not** the Linux
x86_64 portable-ISA evidence, which is deferred below.

The `--expect cpu` failure is the verifier working as designed, and it is the
clearest evidence this step produced: pip reported success, yet the wheel
came out Metal-capable (`MTL : EMBED_LIBRARY = 1`) because
`llama-cpp-python` 0.3.35's own `CMakeLists.txt` force-sets `GGML_METAL=ON`
on Apple hosts, overriding `CMAKE_ARGS`. A build-time check that trusted pip's
exit code would have shipped an image whose flavor did not match its label.
This upstream behavior is macOS-only and does not affect the Linux images;
on them the same check is what refuses a CPU-only wheel in the CUDA image.

## Deferred measurements and exact prerequisites

| Measurement | Missing prerequisite | Follow-up command / expected observation |
| --- | --- | --- |
| Clean / no-change / app-only CPU builds with logs and times | Running Docker daemon with buildx | `scripts/docker/measure_rebuild.sh --flavor full --reset`; expected: `app-only` row reports `native_compile_steps=0` and the script exits 0; retain `docs/validation/evidence/rebuild-<stamp>/`. Repeat with `--flavor serving -- --build-arg CONVERSION_TOOLS=disabled`. Target from the roadmap: app-only wall time ≥ 50 % below the baseline `docker build` on the same machine; record the measured number even if unmet. |
| `BUILD_JOBS` honored | Docker daemon | Two `measure_rebuild.sh --reset` runs with `-- --build-arg BUILD_JOBS=1` and `BUILD_JOBS=4`; expected: materially different `clean` times and `[n/m]` ninja progress in both logs. |
| Lean image contents | Docker daemon | `docker build --build-arg IMAGE_FLAVOR=serving --build-arg CONVERSION_TOOLS=disabled -t lewlm:serving .`; the build itself asserts no torch/transformers/weasyprint and a CPU-only llama.cpp; then `docker run --rm lewlm:serving doctor --json` → `gguf_fallback_backend.ready == true`, `container.image_flavor == "serving"`. |
| Full image converts and quantizes a fixture | Docker daemon + a small licensed HF checkpoint on the host | `docker run --rm -v <models>:/data/models lewlm:cpu convert <model-id> --authorize model_conversion` then `runtime probe --model <model-id>_converted --mode generate`; expected: a GGUF under `/data/cache/conversions` and a generated reply. |
| Pinned locks | Docker daemon (runs pip-tools inside `python:3.11-slim` / the CUDA devel image) | `scripts/docker/lock_dependencies.sh --flavor serving`, `--flavor full --torch-index https://download.pytorch.org/whl/cpu`, and `--flavor full --image nvidia/cuda:12.6.2-devel-ubuntu24.04 --python python3 --torch-index https://download.pytorch.org/whl/cu126 --label cuda126`; then rebuild with `--build-arg DEPENDENCY_INPUT=requirements/locks/<name>.txt`. Locks are not committed until generated. |
| Portable CPU ISA | Linux x86_64 container runtime and the oldest CPU to support | `docker run --rm lewlm:cpu doctor --json` → inspect `install_profiles.llamacpp_build.system_info` (`CPU : SSE3 = 1 \| AVX = 1 …`) and run `lewlm runtime probe --model <gguf> --mode generate` on that CPU. `GGML_NATIVE=OFF` is retained; it is not by itself proof for every instruction. |
| CUDA compile, SM validation on a real toolkit, offload, generation | Linux host with NVIDIA driver, NVIDIA Container Toolkit | `docker build -f Dockerfile.cuda --build-arg CUDA_ARCHITECTURES=<sm> -t lewlm:cuda .` (the build fails fast on an unsupported SM and fails on a CPU-only wheel); `docker run --rm --gpus all lewlm:cuda doctor --json` → `llamacpp_build.gpu_offload_supported == true`, `accelerator_hints` contains `cuda`; then `runtime probe --model <gguf> --mode generate` with `LEWLM_GPU_OFFLOAD_LAYERS=-1` and observe VRAM in `nvidia-smi`. |
| Prebuilt wheel path | Docker daemon; a release of `llama-cpp-python` with a wheel for `cp311` `linux_x86_64` on the chosen index | `docker build --build-arg LLAMA_CPP_PYTHON_WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cpu -t lewlm:cpu-wheel .`; expected: no `Building wheel for llama-cpp-python` in the log and the verifier passes. A missing wheel must fail the build, not fall back to compiling. |
| Native Windows wheel/import safety | Windows host | `py -3.11 -m pip install -e ".[llamacpp_runtime]"` then `python scripts/verify_llamacpp_build.py --expect any`; expected exit 0, or exit 2 with the import-guard reason on a CPU lacking the wheel's instructions, which is the documented signal to use the container image. |
| GitHub Actions `docker` and `docker-full` jobs | Push to CI | Both jobs green; `docker-full-rebuild-evidence` artifact contains `summary.tsv` with `app-only … native_compile_steps=0`. |

## Rollback

Revert the step commit. `requirements/` and the scripts are additive; the
previous Dockerfiles copied the whole tree and installed `.[EXTRAS]`, and
restoring them restores that behavior. No stored data, setting, or model ID is
affected. `image_flavor` and `storage_access` are optional fields that
disappear with the revert.
