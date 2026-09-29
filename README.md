# LewLM

LewLM is a **local-first AI middleware and shared runtime** for applications that run models on your own machine. It scans local model folders, picks a compatible runtime, and exposes one backend contract through a CLI, a local HTTP API, SSE/WebSocket event streams, and an embeddable Python interface.

On top of that contract LewLM provides model discovery and routing, chat and streaming, structured output, embeddings and rerank, vision and audio, sessions, local tools, and deterministic document workflows.

**Status:** alpha / pre-release. It runs on **macOS, Windows, and Linux**. Every platform in the table below has been run end to end on real hardware.

## Platform support

| Platform | How LewLM runs there | Tested |
| --- | --- | --- |
| macOS, Apple Silicon | native `.[mlx]` (text, vision, audio) and `.[llamacpp]`; oMLX as a bridge | MLX packaged path; oMLX recipe `validated` on an M2 Max |
| Windows 11, native | `.[llamacpp_runtime]` from the prebuilt CPU wheel; Ollama as a bridge | llama.cpp and Ollama acceptance suites pass; full test suite passes |
| Windows + WSL2 | LewLM native on Windows; a GPU engine in Docker Desktop | vLLM, SGLang, and TabbyAPI acceptance suites pass; vLLM also passes rollout/rollback and the Chap smoke |
| Linux, CPU | the `full` Docker image | acceptance suite and full test suite pass; HF→GGUF conversion works inside the image |
| Linux, NVIDIA | the CUDA image (`docker-compose.gpu.yml`) | llama.cpp CUDA acceptance suite passes with full GPU offload |

CI runs the portable test suite and the Chap contract smoke on Ubuntu, Windows, and macOS on every push, and builds the bridge, full, and CUDA images. Per-engine status lives in [`examples/backends/compatibility.json`](examples/backends/compatibility.json).

Not yet proven:

- **Bare-metal Linux.** The Linux results above ran in Linux userspace on Docker Desktop's WSL2 kernel. vLLM, SGLang, and TabbyAPI stay `deferred` for bare-metal Linux + NVIDIA.
- **Native Windows llama.cpp with CUDA.** The prebuilt Windows CUDA wheels need AVX-512, which most consumer Intel CPUs lack. `lewlm doctor` detects the mismatch. With an NVIDIA GPU on Windows, use the CUDA container.
- **Engines running natively on Windows**, such as Ollama for Windows. The Ollama results used Ollama in Docker.
- **Hardware breadth.** The Windows and Linux results come from a single laptop (Core Ultra 9 285HX, RTX 5090 Laptop). Other GPUs, drivers, and CPUs have not been exercised.
- **Packaged vision and audio outside Apple Silicon.** On other platforms they go through the bridge.

## What LewLM does

- **Model discovery and routing:** scans GGUF files, MLX folders, local Hugging Face-style bundles, and audio and multimodal bundles, then routes each request to a compatible runtime with an explanation.
- **One interface over many runtimes:** native **MLX** on Apple Silicon, **llama.cpp/GGUF** everywhere (the first-class packaged runtime), and loopback-only engines (vLLM, SGLang, TabbyAPI, Ollama, oMLX) as opt-in bridges.
- **Multiple public surfaces:** the same backend through the CLI, the local HTTP API, SSE/WebSocket streams, the `LewLM` Python facade, and the lighter `LewLMAppClient`.
- **Shared runtime:** one long-lived process owns models, scheduling, caches, and residency for every connected application.
- **Documents:** with `.[documents]`, ingest and render TXT, Markdown, PDF, DOCX, CSV, XLSX, and OCR-style image flows, plus built-in transform skills.
- **Operator controls:** `doctor`, runtime and cache stats, benchmark artifacts, serving profiles, warm/unload controls, and audit-friendly request metadata.
- **Honest capability reporting:** every model and path reports what it supports, what falls back, and why.

On the Apple Silicon MLX text path, LewLM also owns serving-control layers such as batching, paged-KV accounting, prefix reuse, speculation control, and benchmark-backed defaults. Elsewhere it relies on the underlying runtime for low-level execution and says so.

LewLM is **not** a GUI, vector database, workflow engine, or universal serving engine. It is meant to sit under other applications.

## Install

**Pick your host first. The two paths are different.**

| Host | Path | Why |
| --- | --- | --- |
| macOS on Apple Silicon | native install with `.[mlx]` | MLX needs Metal, which containers cannot reach |
| Linux / Windows | **Docker** (see [below](#docker-the-recommended-path-on-linux-and-windows)) | the image already has a GPU-capable llama.cpp build and the HF→GGUF conversion tools |

A native install on Linux and Windows is still supported. It leaves you to supply two things the image already has: a llama.cpp build compiled for your GPU, and llama.cpp's `convert_hf_to_gguf.py` plus `llama-quantize`, which `lewlm convert` uses and the `llama-cpp-python` wheel does not ship. `lewlm doctor` tells you which setup it is running in.

```bash
git clone https://github.com/lew-cx/LewLM.git
cd LewLM
```

**macOS / Linux**

```bash
python3 -m venv .venv
. .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Then choose an install profile:

| Profile | Install command | What it gives you |
| --- | --- | --- |
| Core only | `python -m pip install -e .` | CLI, local API, model registry, routing, and readiness surfaces; no inference runtime |
| Apple MLX | `python -m pip install -e ".[mlx]"` | MLX text, vision, and audio serving on Apple Silicon |
| GGUF (llama.cpp) | `python -m pip install -e ".[llamacpp]"` | GGUF chat, embeddings, and rerank on macOS, Linux, and Windows, plus HF→GGUF conversion dependencies |
| GGUF, serving only | `python -m pip install -e ".[llamacpp_runtime]"` | `llama-cpp-python` only, without Torch/Transformers; the fastest GGUF install |
| ONNX Runtime GenAI | `python -m pip install -e ".[onnx_genai]"` | Windows-native ONNX GenAI bundles (CPU, DirectML, CUDA) and HF→ONNX conversion |
| External engine bridge | `python -m pip install -e .` | LewLM in front of a local OpenAI-compatible server you already run |
| Documents add-on | `python -m pip install -e ".[documents]"` | PDF, DOCX, XLSX, OCR-oriented ingest, render, and transform |

You need **at least one runtime profile** (MLX, GGUF, ONNX GenAI, or the bridge) for chat and generation. Documents is an add-on. Common combinations are `.[mlx,documents]`, `.[llamacpp,documents]`, and `.[dev,documents]` for development and tests.

On **native Windows**, PyPI ships `llama-cpp-python` only as source, and a source build fails on a default install because of the 260-character path limit. Install the prebuilt CPU wheel instead:

```powershell
python -m pip install -e ".[llamacpp_runtime]" --only-binary llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python scripts/verify_llamacpp_build.py --expect cpu
```

For NVIDIA GPUs on Windows, use the CUDA container. The [installation guide](docs/getting-started/installation.md) covers every profile in detail, including ONNX conversion settings and bridge configuration.

### Docker: the recommended path on Linux and Windows

The image carries a portable llama.cpp build, a CUDA build for NVIDIA hosts (no CUDA toolkit or compiler needed on the host), and the llama.cpp conversion tools, so `lewlm convert` works out of the box.

```bash
cp .env.example .env                          # set LEWLM_DOCKER_MODELS_DIR to your model tree
docker compose up --build                     # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # NVIDIA
curl -s http://127.0.0.1:8080/v1/health       # confirm readiness
```

Converting inside the container:

```bash
docker compose exec lewlm lewlm scan
docker compose exec lewlm lewlm convert <model-id> --authorize model_conversion
```

Apple MLX cannot run in a container, so run it natively on macOS. See [Running LewLM in Docker](docs/operations/docker.md) for GPU builds, `.env` settings, bind mounts, and build args.

## Quick start

With a runtime profile installed:

```bash
lewlm doctor                                  # check the install and see where models go
lewlm scan
lewlm list-models
lewlm capabilities "<model name or id>"
lewlm runtime probe --model "<model name or id>" --mode load
lewlm warm "<model name or id>"
lewlm chat "Hello from LewLM"
lewlm serve
```

By default LewLM stores state under `~/.lewlm` and scans `~/.lewlm/models` (`%USERPROFILE%\.lewlm` on Windows). Model weights never need to live in this repository.

To front a local OpenAI-compatible server instead of loading models directly:

```text
LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true
LEWLM_EXTERNAL_ACCELERATOR_BASE_URL=http://127.0.0.1:8000
LEWLM_EXTERNAL_ACCELERATOR_PROFILE=vllm_local
```

To run several bridges at once, set `LEWLM_EXTERNAL_ENDPOINTS` to a JSON array of named loopback endpoints instead (see [Configuration](docs/reference/configuration.md)). Pinned recipes for oMLX, vLLM, SGLang, and TabbyAPI are in [`examples/backends/`](examples/backends/).

For documents:

```bash
lewlm list-skills
lewlm transform --input examples/receipt-transform.json --output ./receipt.md
```

OCR flows also need a local OCR engine such as `tesseract`.

## One shared runtime for many apps

For production, run one long-lived LewLM process and connect each application with `LewLMAppClient.from_http()`. The server owns model objects, scheduling, caches, and residency. Applications own their own schemas, workflows, and templates. Closing a client never unloads a shared model.

```python
from lewlm import LewLMAppClient

rag_client = LewLMAppClient.from_http("http://127.0.0.1:8080", application_id="rag-chat")
document_client = LewLMAppClient.from_http("http://127.0.0.1:8080", application_id="document-generator")

assert rag_client.runtime_info().runtime_instance_id == document_client.runtime_info().runtime_instance_id
```

- Run **one server worker per model replica**. `uvicorn --workers 4` creates four processes and can load four copies of a model.
- Operator and administrator API keys separate lifecycle access. Operators can inspect residency, warm models, and request drains; administrators can also unload models and run disruptive diagnostics.
- Runtime statistics include per-application request, lease, wait, and contention summaries without recording prompts or document content.

See [ADR-001](docs/architecture/adr-001-shared-runtime-residency.md) and [`examples/shared_runtime_clients.py`](examples/shared_runtime_clients.py).

## Recommended paths by platform

| Platform | Chat | Embeddings / rerank | Vision | Audio | Structured output |
| --- | --- | --- | --- | --- | --- |
| macOS | MLX on Apple Silicon; GGUF on other Macs | MLX on Apple Silicon; bridge on other Macs | MLX on Apple Silicon; bridge on other Macs | MLX on Apple Silicon; bridge on other Macs | GGUF for decode-time enforcement; MLX is prompt-guided |
| Linux | GGUF (Docker image) | GGUF with embedding-capable models | bridge | bridge | GGUF |
| Windows | GGUF (Docker image); ONNX GenAI/DirectML is probe-gated | GGUF with embedding-capable models | bridge | bridge | GGUF |

`lewlm doctor` and `GET /v1/health` report the recommended paths for the current host under `install_profiles`.

## Capability reporting

LewLM reports how each feature reaches execution rather than claiming equal support everywhere. `GET /v1/health`, `GET /v1/runtime/stats`, and `GET /v1/models/{model_id}/capabilities` expose:

- a **support path** for each feature (`support_path`: `packaged`, `bridge`, and so on) and how it was verified (`verification_method`, such as `host_probe`), with `benchmark_backed` flags on measured evidence
- an **acceptance state** for each standards term in `standards_acceptance_contract` (terms such as `kv_offload` through `local_agent_sandbox`): `lewlm_owned`, `backend_native`, `partial`, `fallback`, `unsupported`, or `unverified`
- explicit **fallback metadata** when a runtime cannot honor a request, for example prompt-guided JSON instead of decode-time enforcement

A few boundaries to know:

- **Structured output depends on the runtime.** GGUF/llama.cpp enforces JSON schemas and grammars at decode time; other runtimes report a prompt-guided fallback.
- **Determinism is reported per engine.** llama.cpp evaluates seeded requests from an empty KV state and vLLM salts its prefix cache per seed. SGLang and TabbyAPI report `seed` as unsupported at their pinned versions, and Ollama reports `deterministic: false`.
- **Bridges stay bridges.** External engines are promoted per platform only after their acceptance suite passes there, and never count as packaged LewLM support.
- **Distributed serving is experimental.** It is a proof-oriented pipeline, not a production tensor-parallel engine.
- **Frontier-architecture reporting is partly metadata.** LewLM detects hybrid SSM/MoE traits but does not run a custom execution core for them.

The full matrix is in the [runtime and capability matrix](docs/reference/runtime-capability-matrix.md).

## Chap: a real app built on LewLM

[**Chap**](https://github.com/lew-cx/Chap) is a chat and operations GUI built entirely on LewLM's public HTTP contract. It uses one base URL and no engine names, SDKs, or engine-specific stream parsers:

- about 700 hand-written lines of TypeScript integration code in [`packages/lewlm/src/`](https://github.com/lew-cx/Chap/tree/main/packages/lewlm/src), covering SSE streaming, cancellation, identity headers, errors, and the `/v1/events` subscription
- everything else (routes, schemas, event types, error codes) generated from LewLM's published contract, with a check that fails when the contract drifts

To build your own UI without a model, run `python -m lewlm.testing.fake_backend`. `examples/chap_backend_smoke.py` checks the backend half of the contract over HTTP, and the [Chap validation guide](docs/guides/chap-validation.md) has the UI checklist.

## Docs and examples

- [Documentation index](docs/index.md)
- [Getting started](docs/getting-started/index.md)
- [Running LewLM in Docker](docs/operations/docker.md)
- [Host-app integration](docs/guides/host-app-integration.md)
- [Chap validation](docs/guides/chap-validation.md)
- [Engine rollout and rollback](docs/operations/backends/rollout-and-rollback.md): oMLX, vLLM, SGLang, TabbyAPI
- [Chat and responses](docs/guides/chat-and-responses.md)
- [Documents guide](docs/guides/documents.md)
- [CLI reference](docs/reference/cli.md)
- [HTTP API reference](docs/reference/http-api.md)
- [Python API reference](docs/reference/python-api.md)
- [Release and validation reference](docs/reference/release-and-validation.md)
- [Security notes](docs/security.md)
- [`examples/python_app_client.py`](examples/python_app_client.py) and [`examples/http_api_integration.py`](examples/http_api_integration.py)

## Community

- [Contributing guide](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
