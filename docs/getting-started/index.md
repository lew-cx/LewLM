# Getting started

LewLM is a middleware-first backend package designed to be usable in three ways:

1. as a **CLI** (`lewlm ...`)
2. as a **local HTTP service** (`lewlm serve`)
3. as an **embeddable Python package** (`from lewlm import LewLM`)

## Before you install

LewLM is optimized first for:

- **Apple Silicon + MLX** for text, vision, and audio runtimes
- **GGUF + llama.cpp** as the main fallback path

The package code does **not** bundle model weights. By default LewLM stores state under `~/.lewlm` and scans `~/.lewlm/models`.

## Start here

- Choose an install profile in [Installation](installation.md)
- Run the first-use flow in [Quickstart](quickstart.md)
- Use [Configuration](../guides/configuration.md) if you need custom paths, API keys, or runtime tuning

## Public defaults

| Setting | Default |
| --- | --- |
| Data directory | `~/.lewlm` |
| Default model roots | `~/.lewlm/models` |
| API host | `127.0.0.1` |
| API port | `8080` |
| OpenAPI document | `/v1/openapi.json` |

## Main workflows

- **Registry and routing**: scan local bundles, inspect capabilities, warm and unload models
- **Inference**: chat, responses, embeddings, rerank, audio
- **Documents**: ingest local files, render deterministic artifacts, run built-in transforms
- **Operations**: inspect cache/runtime stats, benchmark, autotune, capture release artifacts
