# Running LewLM in Docker

Docker is the **promoted way to run LewLM on every non-Apple host**. Apple
Silicon keeps MLX natively — Metal is unavailable to containers — but on Linux
and Windows the image is where LewLM's packaged runtime family actually comes
together:

- a portable llama.cpp build (`GGML_NATIVE=OFF`), so the backend does not bake
  in CPU instructions the host may lack — the direct fix for the Windows
  `0xc000001d` (`STATUS_ILLEGAL_INSTRUCTION`) crash on `llama.dll`
- a CUDA llama.cpp build on NVIDIA hosts, without needing a CUDA toolkit or a
  C++ compiler on the host
- the two llama.cpp tools LewLM's conversion path shells out to, so
  `lewlm convert` is executable rather than reporting `requires_install`

A native install remains supported and is fully documented in the
[installation guide](../getting-started/installation.md), but it leaves the
GPU-capable llama.cpp build and the conversion tools for you to supply.
`lewlm doctor` reports which of the two shapes it is running in.

## Why the image carries llama.cpp tools

`lewlm convert` does not implement GGUF export itself. It shells out to
llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize`, and the
`llama-cpp-python` wheel ships **neither** — it installs only the shared
libraries (`libllama`, `libggml*`, `libmtmd`). Installing the `llamacpp` or
`gguf_conversion` extra therefore satisfies the converter's *Python
dependencies* while leaving the converter itself absent.

The images build both tools from a pinned llama.cpp revision
(`LLAMA_CPP_REF`) into `/opt/llamacpp-tools` and point LewLM at them with
`LEWLM_LLAMACPP_CONVERT_HF_TO_GGUF_PATH` and `LEWLM_LLAMACPP_QUANTIZE_PATH`.
The converter, `gguf-py/`, and `conversion/` stay siblings in that directory
because `convert_hf_to_gguf.py` imports the last two relative to its own
location.

## What runs in a container, and what does not

| Path | Container support |
| --- | --- |
| Core middleware (CLI, HTTP API, registry, routing, readiness, documents) | ✅ Fully supported on the Linux container |
| `llamacpp` GGUF chat, streaming, embeddings, rerank, structured output | ✅ Supported (CPU image, or NVIDIA via `Dockerfile.cuda`) |
| HF→GGUF conversion (incl. JANG normalization) | ✅ Supported — the image builds llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize` and points LewLM at them. Build with `--build-arg CONVERSION_TOOLS=disabled` to omit them |
| Apple MLX runtimes | ❌ Not containerizable — Apple Metal is unavailable to containers; run MLX natively on macOS |
| Vision / audio | Bridge-only on non-Apple, same as native: front a loopback server via the external-accelerator bridge. Containerising does not change this — it is a runtime-adapter gap, not a packaging one |

## Image flavors

Every image is built from one of three flavors (`--build-arg IMAGE_FLAVOR=...`):

| Flavor | Contents | Use it when |
| --- | --- | --- |
| `bridge` | base package only (`requirements/bridge.txt`) | LewLM fronts Ollama or another loopback server; nothing native is compiled |
| `serving` | `bridge` + llama.cpp GGUF runtime (`llamacpp_runtime` extra) | packaged local GGUF inference without conversion; no Torch/Transformers, no document libraries |
| `full` | `serving` + HF→GGUF conversion dependencies + documents/OCR (`llamacpp` + `documents` extras) | **default** — batteries included, unchanged behavior for `docker build -t lewlm:cpu .` |

`lewlm doctor` reports the flavor (`install_profiles.container.image_flavor`) and,
for `bridge`/`serving`, says that conversion and document workflows are not
installed by design rather than leaving you to discover a missing package.
Pair the lean flavors with `--build-arg CONVERSION_TOOLS=disabled`: they have
no converter Python dependencies, so building `llama-quantize` for them is
wasted time.

## Quick start (CPU)

```bash
# Build the portable CPU image (batteries included: GGUF + conversion + documents)
docker build -t lewlm:cpu .

# Lean GGUF serving image (no torch, no conversion, no documents)
docker build -t lewlm:serving --build-arg IMAGE_FLAVOR=serving --build-arg CONVERSION_TOOLS=disabled .

# Run it, persisting state + models in a named volume
docker run -d --name lewlm -p 8080:8080 -v lewlm-data:/data lewlm:cpu

# Confirm readiness
curl -s http://127.0.0.1:8080/v1/health | python -m json.tool

# Run any CLI command in the same image
docker run --rm -v lewlm-data:/data lewlm:cpu doctor --json
```

Or with Compose:

```bash
docker compose up --build                       # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # NVIDIA
```

There is one `lewlm` service; `docker-compose.gpu.yml` swaps it onto the
CUDA image rather than adding a second service, so the CPU and GPU images
never run side by side and contend for the host port and the SQLite
registry in `lewlm-data`. To make plain `docker compose` commands use the
GPU image, set `COMPOSE_FILE` in `.env` (see `.env.example`).

Upgrading from the old `gpu` profile: its `lewlm-cuda` container keeps
running and holding the port after the service is gone. Remove it once:

```bash
docker compose up -d --remove-orphans     # add -f ... -f docker-compose.gpu.yml for the GPU image
```

None of the external engines (Ollama, vLLM, SGLang, TabbyAPI) is part of
this compose project or needed to run LewLM. Their recipes under
`examples/backends/` are opt-in: an image is only pulled when you run that
recipe's own compose file.

Compose reads a `.env` file beside `docker-compose.yml`. Copy `.env.example`
and set at least `LEWLM_DOCKER_MODELS_DIR` to the model directory you already have;
every other value has a working default, so a fresh clone runs without one.

```ini
LEWLM_DOCKER_MODELS_DIR=C:/Users/you/.lewlm/models

# Blackwell / RTX 50-series needs CUDA >= 12.8 and its own architecture number.
LEWLM_DOCKER_CUDA_ARCHITECTURES=120
LEWLM_DOCKER_CUDA_DEVEL_IMAGE=12.8.0-devel-ubuntu24.04
LEWLM_DOCKER_CUDA_RUNTIME_IMAGE=12.8.0-runtime-ubuntu24.04
```

Check your GPU's number with
`nvidia-smi --query-gpu=compute_cap --format=csv`; a build whose
`CUDA_ARCHITECTURES` does not cover your card produces a binary it cannot run.

## Using models you already have on the host

State lives at `/data` inside the container (`LEWLM_DATA_DIR=/data`), and models
are scanned under `/data/models`. Bind-mount your host `~/.lewlm` to reuse an
existing registry and model files instead of the named volume:

```bash
# Linux / macOS
docker run -d -p 8080:8080 -v "$HOME/.lewlm:/data" lewlm:cpu

# Windows PowerShell
docker run -d -p 8080:8080 -v "$env:USERPROFILE\.lewlm:/data" lewlm:cpu
```

Then scan and chat against a GGUF model:

```bash
docker exec lewlm lewlm models scan
docker exec lewlm lewlm chat --model <model-id> --prompt "Hello"
```

## NVIDIA GPU

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host.

```bash
docker build -f Dockerfile.cuda -t lewlm:cuda .
docker run -d --gpus all -p 8080:8080 -v "$HOME/.lewlm:/data" lewlm:cuda
# or:  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The default `CUDA_ARCHITECTURES=75;80;86;89` is the broad **compatibility
profile** (Turing through Ada). It produces one image that runs on any of
those cards at the cost of compiling every kernel four times. A **per-device
build** is the recommended local recipe: find your card's number and build for
it alone.

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. 8.9 -> 89

docker build -f Dockerfile.cuda --build-arg CUDA_ARCHITECTURES=89 -t lewlm:cuda .
```

The build checks every requested SM against the toolkit in the selected base
image *before* compiling (`scripts/docker/validate_cuda_archs.sh`), so an
unsupported combination fails in seconds with the supported list instead of
twenty minutes in. `native` is refused: the image must not depend on which GPU
was visible on the build host. Blackwell / RTX 50-series (`120`) needs a CUDA
≥ 12.8 base image and the matching torch index:

```bash
docker build -f Dockerfile.cuda \
  --build-arg CUDA_ARCHITECTURES=120 \
  --build-arg CUDA_DEVEL_IMAGE=12.8.0-devel-ubuntu24.04 \
  --build-arg CUDA_RUNTIME_IMAGE=12.8.0-runtime-ubuntu24.04 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  -t lewlm:cuda .
```

The CUDA image proves at build time that the installed `llama-cpp-python` has
the CUDA backend compiled in (`scripts/verify_llamacpp_build.py --expect
gpu-build --hint cuda`); a CPU-only wheel — from a wrong wheel index or a
source build that silently lost its toolkit — fails the build. A build has no
GPU: the CUDA library needs the driver's `libcuda.so.1` just to load, so the
toolkit's stub stands in for that one command (it never reaches the runtime
image), and with no device llama.cpp cannot report offload. Real offload and
generation are verified on hardware:

```bash
docker run --rm --gpus all lewlm:cuda doctor --json | python -c "import json,sys; b=json.load(sys.stdin)['install_profiles']['llamacpp_build']; print(b['gpu_offload_supported'], b['accelerator_hints'])"
docker run --rm --gpus all -v "$HOME/.lewlm:/data" lewlm:cuda runtime probe --model <gguf-model-id> --mode generate
```

## Configuration

The image is configured entirely through `LEWLM_*` environment variables (see
the [configuration reference](../reference/configuration.md)). The most relevant:

| Variable | Default in image | Purpose |
| --- | --- | --- |
| `LEWLM_HOST` | `0.0.0.0` | Bind address (set so the server is reachable outside the container) |
| `LEWLM_PORT` | `8080` | Bind port |
| `LEWLM_DATA_DIR` | `/data` | State + model root (mount a volume here) |
| `LEWLM_EXTERNAL_ACCELERATOR_ENABLED` | unset | Set `true` to front a local OpenAI-compatible server |
| `LEWLM_EXTERNAL_ACCELERATOR_BASE_URL` | unset | e.g. `http://host.docker.internal:1234` to reach an LM Studio / llama.cpp server on the host |
| `LEWLM_IN_CONTAINER` | `true` | Marks the deployment shape so `lewlm doctor` reports it and tailors its guidance |
| `LEWLM_GPU_OFFLOAD_LAYERS` | `-1` (CUDA image only) | Layers offloaded to the GPU. Without a value llama.cpp is asked for **zero** offloaded layers, so a CUDA build would run CPU-only. Lower it for models that exceed your VRAM |
| `LEWLM_LLAMACPP_CONVERT_HF_TO_GGUF_PATH` | `/opt/llamacpp-tools/convert_hf_to_gguf.py` | HF→GGUF converter the conversion path shells out to |
| `LEWLM_LLAMACPP_QUANTIZE_PATH` | `/opt/llamacpp-tools/llama-quantize` | GGUF quantizer used for every non-`f16` profile |

## Build arguments

| Arg | Default | Purpose |
| --- | --- | --- |
| `IMAGE_FLAVOR` | `full` | `bridge`, `serving`, or `full` — see [Image flavors](#image-flavors) |
| `DEPENDENCY_INPUT` | `requirements/<flavor>.txt` | The dependency-layer input. Point it at a lock from `scripts/docker/lock_dependencies.sh` for pinned, hashed installs |
| `BUILD_JOBS` | `4` | Parallel compile jobs for llama.cpp and the conversion tools (`CMAKE_BUILD_PARALLEL_LEVEL`, `MAX_JOBS`, `-j`). Deliberately not `nproc`: raise it on a large build host, lower it on a memory-constrained one |
| `LLAMA_CPP_PYTHON_WHEEL_INDEX` | *(empty)* | When set, install `llama-cpp-python` only as a prebuilt wheel from that index (e.g. `https://abetlen.github.io/llama-cpp-python/whl/cpu`, `/whl/cu126`) and skip the source build. Wheel availability is release-specific; a missing wheel fails the build rather than silently compiling. The flavor is still verified |
| `LLAMA_CMAKE_ARGS` | `-DGGML_NATIVE=OFF` (CPU and CUDA) | llama.cpp build flags. Keep `GGML_NATIVE=OFF` for a distributable image; `-DGGML_NATIVE=ON` is the explicit host-tuned local option. On the CUDA image this governs the CPU half of the build (appended after the CUDA flags) |
| `LLAMA_BUILD_EXPECT` (CPU) | `cpu` | What the installed llama.cpp must report: `cpu`, `gpu` (when `LLAMA_CMAKE_ARGS` enabled an accelerator such as Vulkan), or `any` |
| `TORCH_INDEX_URL` | CPU: `…/whl/cpu`; CUDA: `…/whl/cu126` | Where torch resolves from in the `full` flavor. Never inferred from PyPI's default build; keep it on the image's CUDA version |
| `PYTHON_VERSION` (CPU) | `3.11` | Base Python version |
| `CUDA_ARCHITECTURES` (CUDA) | `75;80;86;89` | Target GPU compute capabilities, validated against the toolkit before compiling |
| `CACHE_SCOPE` | CPU: `cpu-py<ver>`; CUDA: `cuda-<devel image>` | Name of the BuildKit cache mounts for ccache and native wheels. Different toolchains never share a cache; different flags within one scope are separated by a hash of the flags |
| `LLAMA_CPP_REF` | `b10698` | Pinned llama.cpp revision used to build the conversion tools |
| `CONVERSION_TOOLS` | `enabled` | `disabled` skips building `convert_hf_to_gguf.py` and `llama-quantize` |
| `EXTRAS` | *(empty)* | Legacy knob: extra pyproject extras layered on top of the flavor (`dev` adds pytest). The flavor already carries its dependencies, so only the difference resolves |

`CONVERSION_TOOLS=disabled` leaves the tool paths configured but absent, so
`lewlm convert` reports the converter as not found at its configured path
rather than silently dropping the capability.

## How the layers are arranged, and why rebuilds are cheap

Both Dockerfiles share one shape:

1. **toolchain** — compilers, `ccache`, an empty venv. Changes almost never.
2. **deps** — copies *only* `requirements/<flavor>.txt` (or the lock named by
   `DEPENDENCY_INPUT`) and installs it. `llama-cpp-python` is handled first and
   on its own: either downloaded as a verified prebuilt wheel or compiled once
   with `ccache` as the compiler launcher. The compiled wheel is cached in a
   directory keyed on cache scope + CMake flags + Python ABI, so pip can never
   hand back a wheel built with different flags.
3. **app** — copies `pyproject.toml`, `README.md`, `LICENSE`, `MANIFEST.in`, and
   `src/`, builds the LewLM wheel, installs it `--no-deps`, then verifies the
   image: the flavor's package set is honest (no torch in `serving`, no
   llama.cpp in `bridge`) and the native backend is the intended flavor.
4. **runtime** — slim base with the venv, the llama.cpp tools, and only the
   shared libraries the flavor needs.

Only step 2's inputs (the requirements file and the build arguments) feed the
dependency layer, so editing anything under `src/` rebuilds step 3 alone: a
wheel build and a `pip install --no-deps`, no compilation. Package downloads
and `ccache` output live in BuildKit cache mounts (`--mount=type=cache`), which
persist across builds on the same builder and never land in the image.

`requirements/*.txt` are exported from `pyproject.toml` by
`scripts/export_dependency_inputs.py`; a unit test fails when they drift, so
`pyproject.toml` stays the single source of truth. They are deliberately
unpinned inputs. For a pinned build, resolve a lock **on the image's own
platform** (the script runs pip-tools inside the matching base image, so this
works from macOS or Windows too) and point the build at it:

```bash
scripts/docker/lock_dependencies.sh --flavor serving
scripts/docker/lock_dependencies.sh --flavor full --torch-index https://download.pytorch.org/whl/cpu
docker build --build-arg IMAGE_FLAVOR=serving \
  --build-arg DEPENDENCY_INPUT=requirements/locks/serving-linux-x86_64-py311.txt .
```

### Measuring it

`scripts/docker/measure_rebuild.sh` runs three builds — clean (`--no-cache`),
no-change, and app-only-change (made in a disposable `git worktree`) — on a
dedicated buildx builder so your own build cache is never touched, keeps all
three plain-progress logs, and fails if the app-only build performed any native
compile step:

```bash
scripts/docker/measure_rebuild.sh --flavor full --reset            # cold caches
scripts/docker/measure_rebuild.sh --flavor serving -- --build-arg CONVERSION_TOOLS=disabled
scripts/docker/measure_rebuild.sh --dockerfile Dockerfile.cuda --flavor full -- --build-arg CUDA_ARCHITECTURES=89
```

Output lands in `docs/validation/evidence/rebuild-<stamp>/` (`summary.tsv`,
`context.txt`, and one log per build). CI runs this for the `full` CPU flavor
on every push and uploads the logs; there is no wall-clock assertion, only the
compile-step count. To confirm `BUILD_JOBS` is honored, build once with
`--build-arg BUILD_JOBS=1` and once with `4` and compare the `clean` times in
the two summaries on the same machine.

### Portability of the CPU build

`GGML_NATIVE=OFF` removes `-march=native`; it does not by itself guarantee that
every instruction llama.cpp enables is present on every old CPU. To see what a
build actually assumes, read the backend's own report from the image
(`CPU : SSE3 = 1 | AVX = 1 | AVX2 = 1 | …`):

```bash
docker run --rm lewlm:cpu doctor --json | python -c "import json,sys; print(json.load(sys.stdin)['install_profiles']['llamacpp_build']['system_info'])"
```

and test generation on the oldest CPU you intend to support with
`lewlm runtime probe --model <gguf-model-id> --mode generate`. A build tuned to
the build host is an explicit local choice: `--build-arg
LLAMA_CMAKE_ARGS=-DGGML_NATIVE=ON`; do not distribute that image.

Measured on 2026-09-24 (llama.cpp in `llama-cpp-python` 0.3.35, both images):
`CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1`.
With `GGML_NATIVE=OFF` llama.cpp still enables AVX2/FMA/F16C/BMI2 by default,
so the practical floor of these images is **x86-64-v3** (Intel Haswell / AMD
Excavator, 2013–2015, or newer). Older CPUs need a build with those features
turned off in `LLAMA_CMAKE_ARGS`. `lewlm doctor` and
`scripts/verify_llamacpp_build.py` compare the build against the host
(`llamacpp_build.missing_cpu_features`) and say so before the first model
load crashes with an illegal instruction.

## Converting models in the container

```bash
docker compose exec lewlm lewlm scan
docker compose exec lewlm lewlm convert <model-id> --authorize model_conversion
docker compose exec lewlm lewlm runtime probe --model <model-id>_converted --mode generate
```

Conversion output goes to `/data/cache/conversions`, which is inside the
`lewlm-data` **named volume**, not the bind-mounted model tree. That is why the
model bind mount can stay `read_only: true`. It also means converted GGUFs are
not visible from the host by default; uncomment the conversions bind mount in
`docker-compose.yml` (and set `LEWLM_DOCKER_CONVERSIONS_DIR`) to land them on the host
instead.

Multimodal sources convert too, as a **text-only** artifact: llama.cpp exports the
text tower of many vision/audio architectures, and LewLM checks the local
converter's architecture registry rather than refusing the whole modality class.
The plan's notes name the architecture being exported and state that the vision
and audio towers are dropped. No `mmproj` artifact is produced yet, so this does
not add packaged vision or audio.

A bundle copied from macOS often arrives with AppleDouble sidecars (`._config.json`,
`._model.safetensors`) beside the real files. LewLM skips those when it looks for
weight shards. What it cannot repair is a bundle whose *subdirectories* were lost
in the copy: a sentence-transformers model whose `modules.json` names `1_Pooling`
will fail conversion if that directory is missing, because llama.cpp's converter
reads `1_Pooling/config.json` directly. Re-download such bundles rather than
converting them in place.

## Notes

- The default (`full`) image bundles a CPU build of `torch` (pulled in by the
  conversion extra) from the explicit CPU wheel index. For a smaller
  serving-only image use `--build-arg IMAGE_FLAVOR=serving`, which installs no
  torch at all.
- The container runs as a non-root `lewlm` user; the mounted volume must be
  writable by uid `10001`. `lewlm doctor` proves this with a real write probe
  (`install_profiles.storage_access.writable`) and prints `data dir writable:
  NO …` with the reason when a bind mount is not, so a bad mount is visible
  before the first request fails. `/v1/health` exposes the same field.
- A `HEALTHCHECK` polls `/v1/health`, so `docker ps` reports container health
  directly. It also requires a network interface other than loopback: a
  container that was never attached to its network (for example because its
  published host port was already taken) reports `unhealthy` instead of
  passing a loopback-only probe.
- Building the conversion tools compiles llama.cpp a second time (only the
  `llama-quantize` target, roughly 20-30 seconds cold, near-instant with a warm
  `ccache`). Pass `--build-arg CONVERSION_TOOLS=disabled` to skip it.
- Docker Desktop on Windows gives the VM a fraction of host RAM by default, so
  `lewlm doctor` inside the container reports less memory than the host has.
  Raise it in `.wslconfig` if model residency needs more.
- Windows checkouts: `.gitattributes` keeps the shell scripts, Dockerfiles, and
  `requirements/` inputs LF even with `core.autocrlf=true`; a CRLF copy breaks
  the build (`sh` rejects the scripts, pip rejects the native requirement). If
  you copied the tree some other way, run `git add --renormalize .` or convert
  those files to LF before building.
- Windows bind mounts (Docker Desktop, NTFS) are fine for model folders and
  data, but they refuse some directory renames that Linux filesystems allow.
  Engine caches that stage and rename (SGLang's kernel JIT) belong on a named
  volume, as the recipes do.
- Validated end to end on Windows 11 + Docker Desktop 4.74 (WSL2 kernel 6.6.87)
  with an RTX 5090 Laptop GPU on 2026-09-24: the CPU `bridge`, `serving`, and
  `full` images, the rebuild contract, and the CUDA image for SM 120 with full
  offload. See the [Windows/Linux validation record](../validation/modernization-windows-linux.md).
