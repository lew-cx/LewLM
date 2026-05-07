# LewLM

LewLM is a **local-first middleware backend** for multimodal model discovery, routing, serving, and document workflows. It scans local model folders, selects a compatible runtime, and exposes one backend contract through a CLI, a local HTTP API, streaming events, and an embeddable Python interface.

LewLM is optimized first for **Apple Silicon + MLX**, with **llama.cpp/GGUF** as the **first-class non-Apple packaged runtime family** and **loopback-only external accelerators** as a bridge path.

**Status:** alpha / pre-release.

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

## Install

```bash
git clone https://github.com/Lewted/LewLM.git
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
| Cross-platform GGUF backend | `python -m pip install -e ".[llamacpp]"` | packaged local serving on macOS, Linux, and Windows | llama.cpp-backed GGUF runtime | First-class non-Apple runtime family |
| Cross-platform external accelerator bridge | `python -m pip install -e .` | LewLM in front of another local loopback server | base package plus the built-in bridge runtime pack | Requires `LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true` and `LEWLM_EXTERNAL_ACCELERATOR_BASE_URL`; LewLM does not install the external server |
| Documents add-on | `python -m pip install -e ".[documents]"` | local ingest, render, and transform workflows | PDF, DOCX, XLSX, OCR-oriented, and deterministic artifact packages | Additive; pair with a runtime profile when you also want inference |

**You need at least one runtime profile** such as Apple MLX, cross-platform GGUF, or the external accelerator bridge for chat and generation. The documents profile is additive.

Common combinations:

- `python -m pip install -e ".[mlx,documents]"` for Apple Silicon plus document workflows
- `python -m pip install -e ".[llamacpp,documents]"` for cross-platform GGUF plus document workflows
- `python -m pip install -e ".[dev,documents]"` for repository development and tests

On **Linux** and **Windows**, start with `.[llamacpp]` when you want packaged local model execution. This is now LewLM's first-class non-Apple path. If you already run a local OpenAI-compatible server, including an NVIDIA-oriented Linux/Windows service, the external accelerator bridge is the intended path for that topology. LewLM does not bundle the server itself; embeddings require a compatible local `/v1/embeddings` endpoint, and rerank requires a compatible local `/v1/rerank` endpoint or equivalent extension.

`lewlm doctor` and `GET /v1/health` expose an `install_profiles` summary so you can confirm which profile is active on the current host. `lewlm doctor` and `GET /v1/runtime/stats` also report detected host memory when available, or an explicit unavailability reason when the host probe cannot determine it.

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
   lewlm warm "<model name or id>"
   lewlm chat "Hello from LewLM"
   lewlm serve
   ```

   Use **Apple MLX** only on Apple Silicon macOS. On Linux, Windows, and non-MLX Mac hosts, use the **Cross-platform GGUF backend** instead. It is the first-class non-Apple path LewLM now productizes.

3. **Cross-platform external accelerator bridge**

   Use this when LewLM should front a loopback-only local OpenAI-compatible server instead of importing a runtime package directly. This is where NVIDIA-oriented Linux/Windows local servers fit conceptually. LewLM does not replace the packaged GGUF runtime path for LewLM-managed execution, and benchmark wins on this bridge do not promote it over the first-class packaged runtime when that path is available. The bridge remains a documented cross-platform path for adapter-backed chat, streaming, embeddings, and rerank when the local server exposes the needed endpoints.

   ```text
   LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true
   LEWLM_EXTERNAL_ACCELERATOR_BASE_URL=http://127.0.0.1:8000
   LEWLM_EXTERNAL_ACCELERATOR_PROFILE=vllm_local
   ```

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

- **Structured output and constrained decoding are runtime-dependent.** LewLM enforces JSON-schema and grammar contracts at decode time on supported paths and returns explicit prompt-guided fallback metadata when a selected runtime cannot honor the requested constraint mode.
- **Performance-core ownership is selective, not universal.** LewLM can own serving-core state, batching, paged-KV accounting, prefix reuse, speculation control, and benchmark-backed defaults where it has a first-class path, without claiming parity across every backend.
- **Optional modules stay optional.** Documents and local-tooling surfaces add real value, but they should not be mistaken for LewLM's always-on core identity.
- **Frontier architecture reporting is partly planning and diagnostics.** LewLM can detect and annotate hybrid SSM/MoE traits, but some frontier reporting is still metadata-driven rather than proof of a custom execution core.
- **Distributed serving is experimental.** The current distributed path is a proof-oriented pipeline surface, not a production tensor-parallel engine.
- **Cross-platform productization stays selective.** LewLM now chooses GGUF via llama.cpp as its single first-class non-Apple runtime family. External accelerators remain a bridge rather than LewLM-owned packaged parity.
- **LewLM is not a GUI, vector database, workflow engine, or universal multi-backend serving engine.** It is meant to sit under other applications.

## What LewLM is best at right now

LewLM is strongest today when you want:

1. a **local-first middleware backend** instead of a GUI app
2. **one interface** over MLX, llama.cpp, and loopback accelerator bridges
3. **honest capability reporting** with explicit fallback reasons
4. **simple local startup** that can grow into benchmarking, serving profiles, multimodal routing, and document workflows

## Docs and examples

- [Documentation index](docs/index.md)
- [Getting started](docs/getting-started/index.md)
- [Host-app integration](docs/guides/host-app-integration.md)
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
