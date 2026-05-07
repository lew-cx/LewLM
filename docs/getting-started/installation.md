# Installation

LewLM is packaged as a Python project. Start from a source checkout unless you specifically want to pin a Git reference from another project.

## Base setup

```bash
git clone <repo-url>
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

## Install profiles

Choose the profile that matches what you want from the current environment:

| Profile | Install command | Best for | What it gives you | Notes |
| --- | --- | --- | --- | --- |
| Core only | `python -m pip install -e .` | registry, health, config, and contract-first integration work | CLI, local API, model registry, routing, readiness surfaces | No inference runtime or document packages |
| Apple MLX local backend | `python -m pip install -e ".[mlx]"` | Apple Silicon text, vision, and audio serving | MLX runtime adapters | First-class packaged runtime; requires macOS on arm64/aarch64 |
| Cross-platform GGUF backend | `python -m pip install -e ".[llamacpp]"` | packaged local serving on macOS, Linux, and Windows | llama.cpp runtime | LewLM's first-class non-Apple runtime family |
| Cross-platform external accelerator bridge | `python -m pip install -e .` | LewLM in front of another local loopback server | base package plus the built-in bridge runtime pack | Requires `LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true` and `LEWLM_EXTERNAL_ACCELERATOR_BASE_URL`; text and streaming use `/v1/chat/completions`, vision uses OpenAI-style image content blocks on that route, and audio requires compatible local `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints. LewLM does not install the external server |
| Documents add-on | `python -m pip install -e ".[documents]"` | ingest, render, and transform workflows | PDF, DOCX, XLSX, OCR-oriented, and deterministic artifact packages | Additive; combine with a runtime profile when you also want inference |

**You need at least one runtime profile** such as Apple MLX, cross-platform GGUF, or the external accelerator bridge before expecting chat, vision, embeddings, rerank, or audio execution. The documents profile is separate on purpose so LewLM can stay modular for middleware hosts that do not need local parsing/rendering.

Common combinations:

- `python -m pip install -e ".[mlx,documents]"` for Apple Silicon plus document workflows
- `python -m pip install -e ".[llamacpp,documents]"` for cross-platform GGUF plus document workflows
- `python -m pip install -e ".[dev,documents]"` for repository development and tests

Operator guidance by host:

- **Apple Silicon macOS:** use `.[mlx]` or `.[mlx,documents]` when you want LewLM's first-class local runtime path.
- **Linux / Windows / non-MLX Macs:** use `.[llamacpp]` for the documented packaged runtime path. This is the first-class non-Apple runtime family LewLM now productizes. If you install `.[mlx]` there, `lewlm doctor` reports it as installed but host-blocked.
- **Linux / Windows with another local server:** the external accelerator bridge is the intended path when you already run a loopback-only OpenAI-compatible server, including NVIDIA-backed local servers. LewLM does not bundle that server; text and streaming use `/v1/chat/completions`, image-conditioned chat requires a compatible server that accepts OpenAI-style image content blocks there, audio requires compatible local `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints, and semantic extras still depend on compatible local `/v1/embeddings` and `/v1/rerank` support. Treat this as an adapter-backed bridge path rather than a packaged-parity claim, and do not treat bridge benchmark wins as a replacement for the first-class packaged GGUF default.
- **Documents profile:** `.[documents]` installs the Python-side packages only. OCR still depends on a working local engine such as `tesseract`.

## Git install

If you want another project to install a pinned LewLM version directly from Git:

```bash
python -m pip install "lewlm @ git+https://github.com/<owner>/LewLM.git@<tag-or-commit>"
```

Profile-oriented examples:

```bash
python -m pip install "lewlm @ git+https://github.com/<owner>/LewLM.git@<tag-or-commit>"
python -m pip install "lewlm[llamacpp] @ git+https://github.com/<owner>/LewLM.git@<tag-or-commit>"
python -m pip install "lewlm[mlx,documents] @ git+https://github.com/<owner>/LewLM.git@<tag-or-commit>"
```

## Confirm the active profile

After install, use either surface below to confirm what the current host can actually use:

```bash
lewlm doctor
curl http://127.0.0.1:8080/v1/health
```

Both now include an `install_profiles` summary with the active profile ids plus per-profile readiness notes.

`lewlm doctor` also prints the detected host memory when the probe succeeds. If it cannot determine total memory on the current host, it reports that explicitly so routing/readiness output stays honest.

## Install behavior and local state

Installing LewLM does **not** overwrite `~/.lewlm`. LewLM creates missing directories on first use and reuses the existing state for the current OS user unless you override `LEWLM_DATA_DIR`. On Windows, the default location is `%USERPROFILE%\.lewlm`.

## Next step

Run the [Quickstart](quickstart.md) flow that matches the profile you selected.
