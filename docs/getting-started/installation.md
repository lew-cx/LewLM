# Installation

LewLM is packaged as a Python project. Start from a source checkout unless you specifically want to pin a Git reference from another project.

## Base setup

```bash
git clone <repo-url>
cd LewLM

python3 -m venv .venv
. .venv/bin/activate
```

## Install profiles

Choose the profile that matches what you want from the current environment:

| Profile | Install command | Best for | What it gives you | Notes |
| --- | --- | --- | --- | --- |
| Core only | `pip install -e .` | registry, health, config, and contract-first integration work | CLI, local API, model registry, routing, readiness surfaces | No inference runtime or document packages |
| MLX local app backend | `pip install -e ".[mlx]"` | Apple Silicon text, vision, and audio serving | MLX runtime adapters | Requires macOS on arm64/aarch64 |
| GGUF fallback backend | `pip install -e ".[llamacpp]"` | GGUF serving and broader host coverage | llama.cpp runtime | Main cross-platform fallback |
| Documents-enabled backend | `pip install -e ".[documents]"` | ingest, render, and transform workflows | PDF, DOCX, XLSX, OCR-oriented, and deterministic artifact packages | Additive; combine with a runtime profile when you also want inference |
| Common Apple Silicon app backend | `pip install -e ".[mlx,documents]"` | local-first app backends on Apple Silicon | MLX plus document workflows | Good default local package shape on supported Macs |
| Full dev/test environment | `pip install -e ".[dev,documents]"` | repository development and test runs | dev tooling plus document packages | Add `mlx` or `llamacpp` separately when you need a real runtime |

**You need at least one runtime profile** such as `mlx` or `llamacpp` before expecting chat, embeddings, rerank, or audio execution. The `documents` profile is separate on purpose so LewLM can stay modular for middleware hosts that do not need local parsing/rendering.

## Git install

If you want another project to install a pinned LewLM version directly from Git:

```bash
pip install "lewlm @ git+https://github.com/<owner>/LewLM.git@<tag-or-commit>"
```

Profile-oriented examples:

```bash
pip install "lewlm[llamacpp] @ git+https://github.com/<owner>/LewLM.git@<tag-or-commit>"
pip install "lewlm[mlx,documents] @ git+https://github.com/<owner>/LewLM.git@<tag-or-commit>"
```

## Confirm the active profile

After install, use either surface below to confirm what the current host can actually use:

```bash
lewlm doctor
curl http://127.0.0.1:8080/v1/health
```

Both now include an `install_profiles` summary with the active profile ids plus per-profile readiness notes.

## Install behavior and local state

Installing LewLM does **not** overwrite `~/.lewlm`. LewLM creates missing directories on first use and reuses the existing state for the current OS user unless you override `LEWLM_DATA_DIR`.

## Next step

Run the [Quickstart](quickstart.md) flow that matches the profile you selected.
