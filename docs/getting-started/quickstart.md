# Quickstart

Choose the first-use flow that matches the install profile you picked in [Installation](installation.md).

## Core only

Use this when you want to validate configuration, registry paths, and public contracts before committing to a runtime package.

```bash
lewlm doctor
lewlm config
lewlm scan
```

This is enough to confirm:

- the base package is installed
- the configured data and model roots are correct
- `install_profiles` and capability-readiness surfaces are wired

Core-only does **not** enable chat or generation.

On Linux and Windows, this remains a diagnostics-only setup until you add a runtime profile such as `.[llamacpp]`.

## Apple MLX local backend

Use this on Apple Silicon when you want the preferred local runtime path.

```bash
lewlm doctor
# put compatible MLX or other supported local bundles there
lewlm scan
lewlm list-models
lewlm capabilities <model-id>
lewlm warm <model-id>
lewlm chat "Hello from LewLM"
lewlm serve
```

If `lewlm doctor` reports the MLX profile as host-blocked, stop here and switch to the Cross-platform GGUF backend instead. MLX remains Apple Silicon macOS only.

## Cross-platform GGUF backend

Use this on Linux, Windows, and non-MLX Mac hosts when you want the documented local runtime path. It is LewLM's first-class non-Apple runtime family.

```bash
lewlm doctor
# put GGUF files there
lewlm scan
lewlm list-models
lewlm capabilities <model-id>
lewlm warm <model-id>
lewlm chat "Hello from LewLM"
lewlm serve
```

After `lewlm doctor`, confirm the recommended runtime profile matches this install and note the `host memory` line. If LewLM cannot determine total host memory, it reports that explicitly instead of implying a precise budget.

## Cross-platform external accelerator bridge

Use this when LewLM should front a loopback-only local OpenAI-compatible server instead of importing a runtime package directly.

This is where Linux/Windows operators with NVIDIA-backed local servers fit conceptually. LewLM does **not** install that server for you, and this bridge does **not** replace the first-class packaged GGUF runtime path for LewLM-managed execution. Treat it as an adapter-backed path: bridge-backed embeddings require a compatible local `/v1/embeddings` endpoint, bridge-backed rerank requires a compatible local `/v1/rerank` endpoint or equivalent extension, and bridge-only non-Apple audio parity depends on compatible local `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints that LewLM probes separately.

For image-conditioned chat on this path, use the same public LewLM surfaces as macOS: `lewlm chat --attach-image`, `/v1/chat/completions` or `/v1/responses` with `input_image`, `LewLM.chat()` / `LewLM.chat_sync()` with image attachments, or the typed `LewLMAppClient` helpers with `InputImagePart`.

**macOS / Linux**

```bash
export LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true
export LEWLM_EXTERNAL_ACCELERATOR_BASE_URL=http://127.0.0.1:8000
export LEWLM_EXTERNAL_ACCELERATOR_PROFILE=vllm_local
```

**Windows PowerShell**

```powershell
$env:LEWLM_EXTERNAL_ACCELERATOR_ENABLED="true"
$env:LEWLM_EXTERNAL_ACCELERATOR_BASE_URL="http://127.0.0.1:8000"
$env:LEWLM_EXTERNAL_ACCELERATOR_PROFILE="vllm_local"
```

Then run:

```bash
lewlm doctor
lewlm scan
lewlm list-models
```

## Platform default feature guide

Use `lewlm doctor` as the current-host source of truth: it now prints the same recommended feature paths exposed by `GET /v1/health.install_profiles.recommended_feature_paths`.

| Platform | Chat | Semantic text | Vision | Audio | Structured output |
| --- | --- | --- | --- | --- | --- |
| macOS | Apple MLX on Apple Silicon; GGUF on non-MLX Macs | Apple MLX on Apple Silicon; external bridge on non-MLX Macs | Apple MLX vision on Apple Silicon; external bridge on non-MLX Macs | Apple MLX audio on Apple Silicon; external bridge on non-MLX Macs | GGUF/llama.cpp when you need decode-time enforcement |
| Linux / Windows | GGUF/llama.cpp packaged default | GGUF/llama.cpp packaged default for compatible semantic GGUF models; bridge remains optional | external accelerator bridge | external accelerator bridge; bridge-only audio parity | GGUF/llama.cpp packaged default |

`structured output` is intentionally split from the default chat path: on Apple Silicon macOS, LewLM still recommends GGUF when the request needs decode-time JSON-schema or grammar enforcement.

`lewlm doctor --json` and `GET /v1/runtime/stats` now also expose `standards_acceptance_contract`, which carries the shared Milestone 120 vocabulary keys from `kv_offload` through `local_agent_sandbox` plus the acceptance-state legend that later milestones reuse.

## Documents add-on

Use this when you want local document ingest, render, or transform workflows, with or without model execution.

```bash
lewlm doctor
lewlm list-skills
lewlm transform --input examples/receipt-transform.json --output ./receipt.md
```

If you also want chat or responses work, pair `.[documents]` with Apple MLX, cross-platform GGUF, or the external accelerator bridge.

The documents extra installs LewLM's Python dependencies, but OCR-style flows still need a working local OCR engine such as `tesseract`.

LewLM creates the default model root on first use, so you can copy local models into `~/.lewlm/models` or `%USERPROFILE%\.lewlm\models` on Windows before or after the first `lewlm scan`.

## Shared follow-up checks

Once the API is running, the main HTTP entry points are:

- `GET /v1/health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/events`

The Python facade follows the same profile split:

```python
from lewlm import LewLM

with LewLM() as lewlm:
    print(lewlm.health()["install_profiles"]["active_profile_ids"])
    print(
        {
            item["feature_class"]: f'{item["label"]} [{item["support_path"]}]'
            for item in lewlm.health()["install_profiles"]["recommended_feature_paths"]
        }
    )
    lewlm.scan_models()
```

See [Python API reference](../reference/python-api.md) for the full surface.
