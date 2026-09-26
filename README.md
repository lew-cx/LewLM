# LewLM

LewLM is a **local-first AI middleware and shared runtime** that provides model discovery, routing, serving, multimodal capabilities, structured output, retrieval primitives, sessions, tools, and deterministic document infrastructure to application clients. It scans local model folders, selects a compatible runtime, and exposes one backend contract through a CLI, a local HTTP API, streaming events, and an embeddable Python interface.

For production applications, run one long-lived model-owning LewLM process and connect independent applications with `LewLMAppClient.from_http()`. That process owns model objects, scheduling, caches, and residency. Application clients own product schemas, workflows, templates, and artifact lifecycle. Closing a client does not unload shared models.

Lifecycle access can be separated with operator and administrator API keys. Operators may inspect residency, warm models, and request drains; administrators may also unload models and run disruptive diagnostics. Long drains can be started as asynchronous operations and polled without keeping an HTTP connection open. Runtime statistics expose bounded per-application request, lease, wait, failure, model-use, and contention summaries without recording prompts or document content.

An ordinary `uvicorn --workers 4` deployment creates four independent Python processes and can load four copies of a model. Run **one server worker per intended model replica**; scale with explicit replicas, not web-worker count.

```bash
lewlm serve
```

```python
from lewlm import LewLMAppClient

rag_client = LewLMAppClient.from_http(
    "http://127.0.0.1:8080",
    application_id="rag-chat",
)
document_client = LewLMAppClient.from_http(
    "http://127.0.0.1:8080",
    application_id="document-generator",
)

assert rag_client.runtime_info().runtime_instance_id == document_client.runtime_info().runtime_instance_id
```

See [ADR-001](docs/architecture/adr-001-shared-runtime-residency.md), the [action plan](docs/architecture/shared-runtime-action-plan.md), and the [shared-client example](examples/shared_runtime_clients.py).

LewLM runs on **macOS, Windows, and Linux**. Each platform has its own packaged path, and each has been validated for real on actual hardware:

- **Apple Silicon:** native **MLX** for text, vision, and audio. This is also where LewLM owns the most serving-performance layers.
- **Every platform:** **llama.cpp/GGUF** is the first-class packaged runtime. On Linux and Windows it is promoted through a CPU or CUDA Docker image.
- **Loopback-only engines** (vLLM, SGLang, TabbyAPI, Ollama, oMLX) run behind the same contract as opt-in bridges.

**Status:** alpha / pre-release. See [Platform status](#platform-status) for what has been proven where, and what has not.

## Platform status

Each platform lane below was run end to end: LewLM's common acceptance suite, a repeated-prefix benchmark, and, where noted, the full test suite. The table lists what passed and the host it passed on. Nothing broader is claimed.

| Platform | How LewLM runs there | Result | Record |
| --- | --- | --- | --- |
| macOS, Apple Silicon | native `.[mlx]` and `.[llamacpp]`; oMLX bridge recipe | MLX packaged path; oMLX recipe `validated` (M2 Max, 2026-09-17) | [step 05](docs/validation/modernization-step-05.md) |
| Windows 11, native | `.[llamacpp_runtime]` from the prebuilt CPU wheel; Ollama as a bridge | llama.cpp lane 11/11; Ollama lane 11/11; full suite **1,201 passed, 0 failed**; graceful Ctrl+Break shutdown | [Windows/Linux](docs/validation/modernization-windows-linux.md) |
| Windows + WSL2 | LewLM native on Windows; the GPU engine in Docker Desktop | vLLM 11/11 plus real-engine rollout/rollback and the Chap smoke (12/12 real-model checks); SGLang 11/11; TabbyAPI 10/10 (tools inconclusive on a 0.5B model) | [Windows/Linux](docs/validation/modernization-windows-linux.md) |
| Linux, CPU | the `full` Docker image | lane 11/11; full suite on `python:3.11-slim` **1,202 passed, 0 failed**; HF→GGUF conversion inside the image | [Windows/Linux](docs/validation/modernization-windows-linux.md) |
| Linux, NVIDIA | the CUDA image (`docker-compose.gpu.yml`) | llama.cpp CUDA lane 11/11 with full offload on an RTX 5090 (SM 12.0) | [Windows/Linux](docs/validation/modernization-windows-linux.md) |

CI runs the portable suite and the Chap contract smoke on Ubuntu, Windows, and macOS on every push. It also builds the bridge, full, and CUDA images.

What is **not** proven yet:

- **Bare-metal Linux.** The Linux lanes ran in Linux userspace on Docker Desktop's WSL2 kernel, and each record says so (`host.wsl: true`). vLLM, SGLang, and TabbyAPI stay `deferred` for bare-metal Linux + NVIDIA in [`examples/backends/compatibility.json`](examples/backends/compatibility.json).
- **Native Windows llama.cpp with CUDA.** The prebuilt Windows CUDA wheels need AVX-512, which most consumer Intel CPUs lack. `lewlm doctor` detects that mismatch. On Windows with an NVIDIA GPU, use the CUDA container.
- **Engines running natively on Windows**, such as Ollama for Windows. The Ollama lane used Ollama in Docker.
- **One host per lane.** Each Windows and Linux lane ran on a single laptop (Core Ultra 9 285HX, RTX 5090 Laptop). Other GPUs, drivers, and CPUs have not been exercised.
- **Packaged vision and audio outside Apple Silicon.** On other platforms these go through the bridge (see [Parity acceptance contract](#parity-acceptance-contract)).

## What LewLM is today

LewLM is a **middleware backend with selective performance ownership**. It is **not** trying to become a universal inference engine on every backend.

Right now, LewLM owns:

- the stable developer contract across CLI, local HTTP API, events, and Python embedding
- model discovery, registry, routing, and serving-profile selection
- local API, CLI, session, document, and typed app-helper surfaces
- capability reporting, fallback reporting, measured benchmark artifacts, and telemetry
- serving-core state, cache orchestration, and request coalescing
- on the primary Apple Silicon MLX text path: LewLM-owned serving-control layers such as batching, paged-KV accounting, prefix reuse, selective speculation control, and benchmark-backed default adoption

LewLM still relies on **MLX**, **MLX-VLM**, **MLX-Audio**, **llama.cpp**, and optional **loopback-only OpenAI-compatible engines** for the low-level runtime execution details that actually run the model.

That means LewLM already owns meaningful middleware and performance behavior on its first-class path, while still staying honest about backend-dependent execution details on secondary paths.

## Parity acceptance contract

Full parity in LewLM means **stable local-first middleware surfaces plus explicit support-path reporting**, not that every backend is equally packaged or equally LewLM-owned.

| Feature class | Accepted packaged path | Accepted bridge path | Explicit boundary |
| --- | --- | --- | --- |
| Chat + streaming | Apple MLX on Apple Silicon; GGUF/llama.cpp on Darwin, Linux, and Windows | loopback-only external accelerator bridge when another local server already owns execution | bridge wins do not become LewLM-owned packaged parity |
| Semantic text | Apple MLX on Apple Silicon; GGUF/llama.cpp with embedding-capable semantic GGUF models; rerank stays packaged through LewLM's embedding-similarity fallback when the backend lacks a native rerank API | bridge-backed semantic routes remain supported when another local server already owns `/v1/embeddings` and `/v1/rerank` | compatible semantic GGUF models stay the packaged non-Apple default, while adapter-backed semantic bridges remain explicit instead of becoming packaged parity |
| Vision | Apple MLX vision on Apple Silicon | external bridge with OpenAI-style image content blocks | no packaged non-Apple vision parity is claimed today |
| Audio | Apple MLX audio on Apple Silicon | external bridge with compatible local `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints | bridge-only non-Apple audio parity is accepted today; no packaged non-Apple audio path is claimed |
| Structured output | GGUF/llama.cpp is the first-class packaged decode-time enforcement path; MLX keeps prompt-guided fallback | bridge may preserve request shape but is not the portable decode-time enforcement default | LewLM reports fallback explicitly instead of overclaiming equal runtime parity |
| Documents | `.[documents]` stays additive across supported hosts | not a bridge-backed runtime class | LewLM remains a middleware backend, not a vector database or workflow framework |
| Performance-core evidence | strongest on the Apple MLX text path, with selective benchmark-backed GGUF reporting | bridge paths can report preserved backend-native or partial behavior | LewLM does not claim universal performance-core ownership on every backend |

The machine-readable acceptance signals are already exposed through `install_profiles.recommended_feature_paths[].support_path`, target-platform `verification_method` values such as `host_probe`, and runtime support strategy or performance-core reports with `benchmark_backed` evidence flags. The full acceptance matrix lives in [docs/reference/runtime-capability-matrix.md](docs/reference/runtime-capability-matrix.md).

LewLM also now ships a shared `standards_acceptance_contract` through `GET /v1/health` under `install_profiles`, `GET /v1/runtime/stats`, and `GET /v1/models/{model_id}/capabilities`. That contract defines the Milestone 120 acceptance states `lewlm_owned`, `backend_native`, `partial`, `fallback`, `unsupported`, and `unverified`, and reserves the 2026 vocabulary keys without pretending that every path already implements them.

- `memory and context`: `kv_offload`, `kv_quantization`, `hybrid_memory`, `pd_disaggregation`, `distributed_kv_transfer`
- `structured output and reasoning`: `strict_tool_parser`, `reasoning_tags`, `parallel_tool_calls`, `streaming_tool_calls`, `responses_api_events`
- `speculation`: `mtp_speculation`, `eagle_speculation`, `dflash_speculation`, `ngram_draft_speculation`, `reasoning_budget_speculation`
- `dependency baselines`: `transformers_v5_ready`, `cuda13_ready`, `pytorch211_ready`, `cxx20_ready`
- `multimodal, document, and semantic`: `multimodal_omni`, `document_ocr_transformer`, `long_context_embedding`
- `agent interoperability`: `local_agent_sandbox`

Release artifacts now carry the same closure contract. `release-manifest.json` includes `install_profiles`, `dependency_audit.compatibility_gates`, and `standards_refresh_acceptance`, and `validate_release_candidate.py` fails when a target manifest is missing the completed Milestones 121-132 standards-refresh summary alongside the existing host, dependency, frontier, optimization, and performance-core proof checks.

## Install

**Pick your host first — the two paths are genuinely different.**

| Host | Path | Why |
| --- | --- | --- |
| macOS on Apple Silicon | native install with `.[mlx]` | MLX needs Metal, which containers cannot reach |
| Linux / Windows | **Docker** (see [below](#docker-the-promoted-path-on-non-apple-hosts)) | the image is where a GPU-capable llama.cpp build and the HF→GGUF conversion tools are already assembled |

A native install on Linux and Windows stays supported and is documented in full
below, but it leaves two things for you to supply that the image already has: a
llama.cpp build compiled for your GPU, and llama.cpp's `convert_hf_to_gguf.py`
plus `llama-quantize`, which `lewlm convert` shells out to and which the
`llama-cpp-python` wheel does not ship. `lewlm doctor` reports which of the two
shapes it is running in and tailors its guidance accordingly.

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

If you prefer not to activate the environment, use `.venv/bin/python -m pip ...` on macOS/Linux or `.venv\Scripts\python -m pip ...` on Windows for the install commands below.

Choose the install profile that matches what you want LewLM to do:

| Profile | Install command | Best for | What it gives you | Notes |
| --- | --- | --- | --- | --- |
| Core only | `python -m pip install -e .` | contract-first integration, registry, health, and config work | CLI, local API, model registry, routing, readiness surfaces | No inference runtime or documents extras |
| Apple MLX local backend | `python -m pip install -e ".[mlx]"` | Apple Silicon local text, vision, and audio serving | MLX runtime adapters on supported hosts | First-class packaged runtime on Apple Silicon |
| Cross-platform GGUF backend | `python -m pip install -e ".[llamacpp]"` | packaged local serving on macOS, Linux, and Windows | llama.cpp-backed GGUF runtime for chat, embeddings, and packaged rerank fallback, plus HF→GGUF conversion dependencies | First-class non-Apple runtime family; non-Apple audio still uses the bridge path |
| Cross-platform GGUF backend, serving only | `python -m pip install -e ".[llamacpp_runtime]"` | GGUF serving without model conversion | `llama-cpp-python` only — no Torch/Transformers | Fastest GGUF install; what the lean `serving` container flavor uses |
| ONNX Runtime GenAI backend | `python -m pip install -e ".[onnx_genai]"` | Windows-native prepared ONNX bundles, provider probing, and HF-to-ONNX conversion | ONNX GenAI discovery plus CPU, DirectML, and CUDA provider metadata, and executable HF-to-ONNX conversion via the onnxruntime-genai model builder | Load/generate adapter for compatible ONNX bundles; HF-to-ONNX conversion is executable when this extra is installed |
| Cross-platform external accelerator bridge | `python -m pip install -e .` | LewLM in front of another local loopback server | base package plus the built-in bridge runtime pack | Requires `LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true` and `LEWLM_EXTERNAL_ACCELERATOR_BASE_URL`; LewLM does not install the external server |
| Documents add-on | `python -m pip install -e ".[documents]"` | local ingest, render, and transform workflows | PDF, DOCX, XLSX, OCR-oriented, and deterministic artifact packages | Additive; pair with a runtime profile when you also want inference |

**You need at least one runtime profile** such as Apple MLX, cross-platform GGUF, or the external accelerator bridge for chat and generation. The documents profile is additive.

Common combinations:

- `python -m pip install -e ".[mlx,documents]"` for Apple Silicon plus document workflows
- `python -m pip install -e ".[llamacpp,documents]"` for cross-platform GGUF plus document workflows
- `python -m pip install -e ".[dev,documents]"` for repository development and tests

On **Linux** and **Windows**, start with `.[llamacpp]` when you want packaged local model execution. This is now LewLM's first-class non-Apple path for text workloads and semantic GGUF models. Embeddings stay packaged there for compatible GGUF models, and rerank stays honest by using LewLM's packaged embedding-similarity fallback when llama.cpp does not expose a native rerank API. If you already run a local OpenAI-compatible server, including an NVIDIA-oriented Linux/Windows service, the external accelerator bridge remains the intended path for that topology and the current bridge-only non-Apple audio parity path. LewLM does not bundle the server itself; bridge-backed semantic routes still require compatible local `/v1/embeddings` and `/v1/rerank` endpoints or equivalent extensions, and audio requires compatible local `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints that LewLM probes separately.

On native Windows, PyPI ships `llama-cpp-python` only as source. A source build needs the Microsoft C++ Build Tools, and it fails on a default Windows install because of the 260-character path limit. Install the prebuilt CPU wheel instead. This is the command the Windows lane was validated with:

```powershell
python -m pip install -e ".[llamacpp_runtime]" --only-binary llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Then check it with `python scripts/verify_llamacpp_build.py --expect cpu`. For NVIDIA GPUs on Windows, the CUDA container is the validated path. The [installation guide](docs/getting-started/installation.md) explains why the prebuilt CUDA wheels may not run on your CPU.

The new `.[onnx_genai]` profile is LewLM's Windows-native ONNX/DirectML path for ONNX GenAI bundles. It can load and generate through compatible `onnxruntime-genai` Python packages, and — when this extra is installed — it makes HF-to-ONNX conversion executable through the official `onnxruntime-genai` model builder. Conversion precision follows the conversion policy (`max_quality` → fp16, otherwise int4) and the execution provider follows `LEWLM_ONNX_GENAI_CONVERSION_EXECUTION_PROVIDER` (`cpu` default, or `dml`/`cuda`). Use `lewlm convert <model-id> --target onnx_genai` to run it.

`lewlm doctor` and `GET /v1/health` expose an `install_profiles` summary so you can confirm which profile is active on the current host, plus `recommended_feature_paths` for the current host's default operator routes. `lewlm doctor` and `GET /v1/runtime/stats` also report detected host memory when available, or an explicit unavailability reason when the host probe cannot determine it.

### Docker: the promoted path on non-Apple hosts

On Linux and Windows the image is where LewLM's packaged runtime family actually comes together. It carries a portable llama.cpp build (`GGML_NATIVE=OFF`, the fix for the Windows `0xc000001d` / `STATUS_ILLEGAL_INSTRUCTION` crash on `llama.dll`), a CUDA build on NVIDIA hosts without needing a CUDA toolkit or C++ compiler on the host, and the two llama.cpp tools `lewlm convert` shells out to — so conversion is executable rather than reporting `requires_install`.

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

Apple MLX is **not** containerizable — run MLX natively on macOS. See [docs/operations/docker.md](docs/operations/docker.md) for GPU builds, `.env` settings, bind mounts, build args, and the full container support matrix.

## Recommended feature paths by platform

| Platform | Chat | Semantic text | Vision | Audio | Structured output |
| --- | --- | --- | --- | --- | --- |
| macOS | Apple MLX on Apple Silicon; GGUF on non-MLX Macs | Apple MLX on Apple Silicon; external bridge on non-MLX Macs | Apple MLX vision on Apple Silicon; external bridge on non-MLX Macs | Apple MLX audio on Apple Silicon; external bridge on non-MLX Macs | GGUF/llama.cpp when you need decode-time enforcement; MLX remains prompt-guided fallback |
| Linux | GGUF/llama.cpp packaged default, promoted through the Docker image | GGUF/llama.cpp packaged default for embedding-capable semantic models; bridge remains optional | external accelerator bridge with OpenAI-style image content blocks | external accelerator bridge with compatible local `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints; bridge-only audio parity | GGUF/llama.cpp packaged default |
| Windows | GGUF/llama.cpp packaged default, promoted through the Docker image; ONNX GenAI/DirectML is probe-gated candidate work | GGUF/llama.cpp packaged default for embedding-capable semantic models; bridge remains optional | external accelerator bridge with OpenAI-style image content blocks | external accelerator bridge with compatible local `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints; bridge-only audio parity | GGUF/llama.cpp packaged default |

`semantic text` covers embeddings and rerank, and `structured output` here means requests that need decode-time JSON-schema or grammar enforcement. On non-Apple hosts, GGUF keeps the packaged semantic default while the bridge remains explicit adapter guidance when another local server owns semantic execution, and `lewlm doctor` plus `GET /v1/health` surface the current-host mapping directly.

## Quick start

Use the quick path that matches the profile you installed:

1. **Core only**

   ```bash
   lewlm doctor
   lewlm config
   lewlm scan
   ```

   This is enough to validate paths, registry state, and readiness wiring, but not to run inference.

2. **Apple MLX local backend** or **Cross-platform GGUF backend**

   ```bash
   lewlm doctor
   # put compatible local models there
   lewlm scan
   lewlm list-models
   lewlm capabilities "<model name or id>"
   lewlm runtime probe --model "<model name or id>" --mode load
   lewlm warm "<model name or id>"
   lewlm chat "Hello from LewLM"
   lewlm serve
   ```

   Use **Apple MLX** only on Apple Silicon macOS. On Linux, Windows, and non-MLX Mac hosts, use the **Cross-platform GGUF backend** instead. It is the first-class non-Apple path LewLM now productizes. `runtime probe --mode load` upgrades a model from routing-only evidence to a persisted load smoke test without generating text; use `--mode generate` when you want to verify output too.

   On native Windows, install llama.cpp from the prebuilt CPU wheel shown under [Install](#install) before this step. If `lewlm doctor` reports that the native library does not load, it will say why (for example `missing_cpu_features`); switch to the Docker image.

3. **Cross-platform external accelerator bridge**

   Use this when LewLM should front a loopback-only local OpenAI-compatible server instead of importing a runtime package directly. This is where NVIDIA-oriented Linux/Windows local servers fit conceptually. LewLM does not replace the packaged GGUF runtime path for LewLM-managed execution, and benchmark wins on this bridge do not promote it over the first-class packaged runtime when that path is available. The bridge remains a documented cross-platform path for adapter-backed chat, streaming, embeddings, and rerank when the local server exposes the needed endpoints, even though non-Apple semantic defaults now stay packaged on compatible GGUF models.

   ```text
   LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true
   LEWLM_EXTERNAL_ACCELERATOR_BASE_URL=http://127.0.0.1:8000
   LEWLM_EXTERNAL_ACCELERATOR_PROFILE=vllm_local
   ```

   Those settings remain the single-server compatibility form. To run more
   than one bridge concurrently, set `LEWLM_EXTERNAL_ENDPOINTS` to a JSON array
   of named loopback endpoints as documented in
   [Configuration](docs/reference/configuration.md). Do not enable the legacy
   singular endpoint at the same time. Named endpoint IDs become part of
   runtime, residency, cache, and benchmark identity so equal upstream model
   names on different servers cannot collide.

   Then run:

   ```bash
   lewlm doctor
   lewlm scan
   lewlm list-models
   ```

4. **Documents add-on**

   ```bash
   lewlm list-skills
   lewlm transform --input examples/receipt-transform.json --output ./receipt.md
   ```

   Pair this with Apple MLX, cross-platform GGUF, or the external accelerator bridge when you also want local model execution. OCR-oriented flows still require a working local OCR engine such as `tesseract`; the Python extra only installs the LewLM-side packages.

LewLM creates its state directories on first use. By default, it stores state under `~/.lewlm` and scans `~/.lewlm/models` (`%USERPROFILE%\.lewlm` and `%USERPROFILE%\.lewlm\models` on Windows).

The repository itself does **not** need to contain model weights. Keep local models under `~/.lewlm/models` or another external directory you point LewLM at.

## What you get

- **Model discovery and routing:** scan GGUF files, MLX folders, local Hugging Face-style bundles, audio bundles, and multimodal bundles with capability metadata.
- **Multiple public surfaces:** the same backend is available through the CLI, local HTTP API, SSE/WebSocket streams, the `LewLM` Python facade, and the lighter-weight `LewLMAppClient`.
- **Document workflows:** with `.[documents]`, LewLM can ingest and render TXT, Markdown, PDF, DOCX, CSV, XLSX, and OCR-style image flows.
- **Operator controls:** `doctor`, runtime and cache stats, explainable routing, benchmark artifacts, serving profiles, warm/unload controls, and audit-friendly request metadata.
- **Selective performance ownership:** on the first-class MLX text path, LewLM owns serving-control layers such as batching, paged-KV accounting, prefix reuse, speculation control, and benchmark-backed default adoption.

## Honest current boundaries

LewLM's strongest performance-ownership claim is still the first-class Apple Silicon MLX text path. Other runtimes may remain adapter-backed or backend-native as long as LewLM reports their capabilities, fallbacks, and measured evidence honestly.

A few other boundaries matter too:

- **Structured output and constrained decoding are runtime-dependent.** The packaged llama.cpp/GGUF path is LewLM's first-class cross-platform decode-time enforcement path for JSON-schema and grammar contracts today. Other runtimes still return explicit prompt-guided fallback metadata when the selected backend cannot honor the requested constraint mode.
- **Performance-core ownership is selective, not universal.** LewLM can own serving-core state, batching, paged-KV accounting, prefix reuse, speculation control, and benchmark-backed defaults where it has a first-class path, without claiming parity across every backend.
- **Optional modules stay optional.** Documents and local-tooling surfaces add real value, but they should not be mistaken for LewLM's always-on core identity.
- **Frontier architecture reporting is partly planning and diagnostics.** LewLM can detect and annotate hybrid SSM/MoE traits, but some frontier reporting is still metadata-driven rather than proof of a custom execution core.
- **Distributed serving is experimental.** The current distributed path is a proof-oriented pipeline surface, not a production tensor-parallel engine.
- **Cross-platform productization stays selective.** LewLM uses GGUF via llama.cpp as its single first-class runtime family outside Apple Silicon. External accelerators remain bridges rather than LewLM-owned packaged parity. Each bridge engine is promoted per platform lane only after that lane passes on real hardware (see [Platform status](#platform-status)).
- **Engine determinism is reported, not assumed.** Sampling support is reported per engine, from what the engine actually does. On llama.cpp, seeded requests are evaluated from an empty KV state. On vLLM, seeded requests get a fresh prefix-cache salt. SGLang and TabbyAPI report `seed` as unsupported at their pinned versions. Ollama forwards the seed but reports `deterministic: false`.
- **LewLM is not a GUI, vector database, workflow engine, or universal multi-backend serving engine.** It is meant to sit under other applications.

## What LewLM is best at right now

LewLM is strongest today when you want:

1. a **local-first middleware backend** instead of a GUI app
2. **one interface** over MLX, llama.cpp, and loopback engines (vLLM, SGLang, TabbyAPI, Ollama, oMLX), on macOS, Windows, and Linux
3. **honest capability reporting** with explicit fallback reasons
4. **simple local startup** that can grow into benchmarking, serving profiles, multimodal routing, and document workflows

## Chap: a real app built on LewLM

[**Chap**](https://github.com/lew-cx/Chap) is a chat and operations GUI built entirely on LewLM's public HTTP contract. It uses one base URL and no engine names, SDKs, or engine-specific stream parsers. It is the best place to see a real client in practice:

- about 700 hand-written lines of TypeScript integration code in [`packages/lewlm/src/`](https://github.com/lew-cx/Chap/tree/main/packages/lewlm/src), covering SSE streaming for both chat surfaces, cancellation, identity headers, one error type, and the `/v1/events` subscription
- everything else it knows about LewLM (routes, schemas, event types, error codes) generated from LewLM's published contract, with a check that fails when the contract drifts
- a browser-driven run of LewLM's [UI checklist](docs/guides/chap-validation.md) against the fake backend and a real engine
- a record of the contract gaps it found against LewLM and how each was closed

In this repository, `python -m lewlm.testing.fake_backend` gives you a working LewLM with no model to build a UI against. `examples/chap_backend_smoke.py` proves the backend half of that contract over HTTP.

## Docs and examples

- [Documentation index](docs/index.md)
- [Getting started](docs/getting-started/index.md)
- [Running LewLM in Docker](docs/operations/docker.md) — the promoted path on Linux and Windows, CPU and CUDA
- [Host-app integration](docs/guides/host-app-integration.md)
- [Chap validation](docs/guides/chap-validation.md) — the contract a chat UI builds against, the backend proof, and the UI checklist
- [Rollout and rollback](docs/operations/backends/rollout-and-rollback.md) — opt-in engine recipes (oMLX, vLLM, SGLang, TabbyAPI), `lewlm doctor` guidance, one-entry rollback
- [Windows/Linux validation record](docs/validation/modernization-windows-linux.md) — every lane, the defects found and fixed, and what remains deferred
- [Chat and responses](docs/guides/chat-and-responses.md)
- [Documents guide](docs/guides/documents.md)
- [CLI reference](docs/reference/cli.md)
- [HTTP API reference](docs/reference/http-api.md)
- [Python API reference](docs/reference/python-api.md)
- [Release and validation reference](docs/reference/release-and-validation.md)
- [Security notes](docs/security.md)
- [`examples/python_app_client.py`](examples/python_app_client.py)
- [`examples/http_api_integration.py`](examples/http_api_integration.py)

## Community

- [Contributing guide](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
