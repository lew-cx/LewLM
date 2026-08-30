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

## Quick start (CPU)

```bash
# Build the portable CPU image (batteries included: GGUF + conversion + documents)
docker build -t lewlm:cpu .

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
docker compose --profile gpu up --build lewlm-cuda   # NVIDIA
```

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
# or:  docker compose --profile gpu up --build lewlm-cuda
```

Set the GPU compute capability for your card if the default set does not cover
it (e.g. Blackwell / RTX 50-series is `120`, and needs a recent CUDA base
image):

```bash
docker build -f Dockerfile.cuda \
  --build-arg CUDA_ARCHITECTURES=120 \
  --build-arg CUDA_DEVEL_IMAGE=12.8.0-devel-ubuntu22.04 \
  --build-arg CUDA_RUNTIME_IMAGE=12.8.0-runtime-ubuntu22.04 \
  -t lewlm:cuda .
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
| `EXTRAS` | `llamacpp,documents` | Which install extras to bake in. Use `llamacpp` for a leaner serving-only image, or `dev` for a CI image with no torch/llama build |
| `LLAMA_CMAKE_ARGS` (CPU) | `-DGGML_NATIVE=OFF` | llama.cpp build flags. Keep `GGML_NATIVE=OFF` for portability |
| `PYTHON_VERSION` (CPU) | `3.11` | Base Python version |
| `CUDA_ARCHITECTURES` (CUDA) | `75;80;86;89` | Target GPU compute capabilities |
| `LLAMA_CPP_REF` | `b10698` | Pinned llama.cpp revision used to build the conversion tools |
| `CONVERSION_TOOLS` | `enabled` | `disabled` skips building `convert_hf_to_gguf.py` and `llama-quantize`, for a leaner CI image |

`CONVERSION_TOOLS=disabled` leaves the tool paths configured but absent, so
`lewlm convert` reports the converter as not found at its configured path
rather than silently dropping the capability.

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

- The default image bundles a CPU build of `torch` (pulled in by the
  conversion/llamacpp extras) to keep it lean. For a smaller serving-only image,
  build with `--build-arg EXTRAS=llamacpp` and skip documents.
- The container runs as a non-root `lewlm` user; the mounted volume must be
  writable by uid `10001`.
- A `HEALTHCHECK` polls `/v1/health`, so `docker ps` reports container health
  directly.
- Building the conversion tools compiles llama.cpp a second time (only the
  `llama-quantize` target, roughly 20-30 seconds). Pass
  `--build-arg CONVERSION_TOOLS=disabled` to skip it.
- Docker Desktop on Windows gives the VM a fraction of host RAM by default, so
  `lewlm doctor` inside the container reports less memory than the host has.
  Raise it in `.wslconfig` if model residency needs more.
