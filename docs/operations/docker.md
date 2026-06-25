# Running LewLM in Docker

Docker is the recommended way to run LewLM consistently across operating
systems, and the cleanest fix for hosts that cannot load a prebuilt
`llama-cpp-python` wheel — most notably the Windows `0xc000001d`
(`STATUS_ILLEGAL_INSTRUCTION`) crash on `llama.dll`, which happens when the
installed wheel was compiled with CPU instructions the host does not have.

The provided image compiles llama.cpp with `GGML_NATIVE=OFF`, so the runtime
backend does not bake in CPU instructions that may be missing on the host. You
get a working, portable GGUF runtime without fighting platform wheels.

## What runs in a container, and what does not

| Path | Container support |
| --- | --- |
| Core middleware (CLI, HTTP API, registry, routing, readiness, documents) | ✅ Fully supported on the Linux container |
| `llamacpp` GGUF chat, streaming, embeddings, rerank, structured output | ✅ Supported (CPU image, or NVIDIA via `Dockerfile.cuda`) |
| HF→GGUF conversion (incl. JANG normalization) | ✅ Supported (image includes the conversion stack) |
| Apple MLX runtimes | ❌ Not containerizable — Apple Metal is unavailable to containers; run MLX natively on macOS |
| Vision / audio | Bridge-only on non-Apple, same as native: front a loopback server via the external-accelerator bridge |

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
docker compose up --build        # CPU
```

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

## Build arguments

| Arg | Default | Purpose |
| --- | --- | --- |
| `EXTRAS` | `llamacpp,documents` | Which install extras to bake in. Use `llamacpp` for a leaner serving-only image, or `dev` for a CI image with no torch/llama build |
| `LLAMA_CMAKE_ARGS` (CPU) | `-DGGML_NATIVE=OFF` | llama.cpp build flags. Keep `GGML_NATIVE=OFF` for portability |
| `PYTHON_VERSION` (CPU) | `3.11` | Base Python version |
| `CUDA_ARCHITECTURES` (CUDA) | `75;80;86;89` | Target GPU compute capabilities |

## Notes

- The default image bundles a CPU build of `torch` (pulled in by the
  conversion/llamacpp extras) to keep it lean. For a smaller serving-only image,
  build with `--build-arg EXTRAS=llamacpp` and skip documents.
- The container runs as a non-root `lewlm` user; the mounted volume must be
  writable by uid `10001`.
- A `HEALTHCHECK` polls `/v1/health`, so `docker ps` reports container health
  directly.
