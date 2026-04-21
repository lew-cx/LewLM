# LewLM

LewLM is a **local-first middleware backend** for multimodal model discovery, routing, serving, and document workflows. It scans local model folders, selects a compatible runtime, and exposes one backend contract through a CLI, a local HTTP API, streaming events, and an embeddable Python interface.

LewLM is optimized first for **Apple Silicon + MLX**, with **llama.cpp/GGUF** as the main fallback path.

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

python3 -m venv .venv
. .venv/bin/activate
```

Choose the install profile that matches what you want LewLM to do:

| Profile | Install command | Best for | What it gives you |
| --- | --- | --- | --- |
| Core only | `pip install -e .` | contract-first integration, registry, health, and config work | CLI, local API, model registry, routing, readiness surfaces |
| MLX local app backend | `pip install -e ".[mlx]"` | Apple Silicon local text, vision, and audio serving | MLX runtime adapters on supported hosts |
| GGUF fallback backend | `pip install -e ".[llamacpp]"` | GGUF serving and broader host coverage | llama.cpp-backed fallback runtime |
| Documents-enabled backend | `pip install -e ".[documents]"` | local ingest, render, and transform workflows | PDF, DOCX, XLSX, OCR-oriented, and deterministic artifact packages |
| Common Apple Silicon app backend | `pip install -e ".[mlx,documents]"` | local-first app backends that need both inference and documents | MLX plus document workflows |
| Full dev/test environment | `pip install -e ".[dev,documents]"` | repository development and tests | dev tooling plus document packages |

**You need at least one runtime profile** such as `mlx` or `llamacpp` for chat and generation. The `documents` profile is additive.

`lewlm doctor` and `GET /v1/health` now both expose an `install_profiles` summary so you can confirm which profile is active on the current host.

## Quick start

Use the quick path that matches the profile you installed:

1. **Core only**

   ```bash
   lewlm doctor
   lewlm config
   lewlm scan
   ```

   This is enough to validate paths, registry state, and readiness wiring, but not to run inference.

2. **MLX local app backend** or **GGUF fallback backend**

   ```bash
   lewlm doctor
   mkdir -p ~/.lewlm/models
   # put compatible local models there
   lewlm scan
   lewlm list-models
   lewlm capabilities "<model name or id>"
   lewlm warm "<model name or id>"
   lewlm chat "Hello from LewLM"
   lewlm serve
   ```

3. **Documents-enabled backend**

   ```bash
   lewlm list-skills
   lewlm transform --input examples/receipt-transform.json --output ./receipt.md
   ```

   Pair this with `.[mlx]` or `.[llamacpp]` when you also want local model execution.

By default, LewLM stores state under `~/.lewlm` and scans `~/.lewlm/models`.

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
- **Cross-platform proof is still incomplete.** The primary supported path is still the local Apple Silicon host with optional llama.cpp fallback.
- **LewLM is not a GUI, vector database, workflow engine, or universal multi-backend serving engine.** It is meant to sit under other applications.

## What LewLM is best at right now

LewLM is strongest today when you want:

1. a **local-first middleware backend** instead of a GUI app
2. **one interface** over MLX, llama.cpp, and related local runtimes
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
