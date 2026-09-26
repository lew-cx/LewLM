# Getting started

LewLM is a middleware-first backend package designed to be usable in three ways:

1. as a **CLI** (`lewlm ...`)
2. as a **local HTTP service** (`lewlm serve`)
3. as an **embeddable Python package** (`from lewlm import LewLM`)

## Before you install

LewLM runs on macOS, Windows, and Linux, with a different packaged path on each:

- **Apple Silicon macOS:** native **MLX** for text, vision, and audio. This is also where LewLM owns the most serving-performance layers.
- **Linux and Windows:** **GGUF + llama.cpp**, promoted through the CPU or CUDA **Docker image**. Native Windows installs use the prebuilt CPU wheel.
- **Any platform:** **loopback engines** (vLLM, SGLang, TabbyAPI, Ollama, oMLX) as opt-in bridges when another local server owns execution.

Every platform lane has been run for real on a single host per lane. The [README's platform status](../../README.md#platform-status) lists what passed where, and what is still deferred (notably bare-metal Linux + NVIDIA for the GPU engines).

The package code does **not** bundle model weights. By default LewLM stores state under `~/.lewlm` and scans `~/.lewlm/models` (`%USERPROFILE%\.lewlm` and `%USERPROFILE%\.lewlm\models` on Windows).

## Start here

- Choose an install profile in [Installation](installation.md)
- Run the first-use flow in [Quickstart](quickstart.md)
- Use [Configuration](../guides/configuration.md) if you need custom paths, API keys, or runtime tuning

## Public defaults

| Setting | Default |
| --- | --- |
| Data directory | `~/.lewlm` (`%USERPROFILE%\.lewlm` on Windows) |
| Default model roots | `~/.lewlm/models` (`%USERPROFILE%\.lewlm\models` on Windows) |
| API host | `127.0.0.1` |
| API port | `8080` |
| OpenAPI document | `/v1/openapi.json` |

## Main workflows

- **Registry and routing**: scan local bundles, inspect capabilities, warm and unload models
- **Inference**: chat, responses, embeddings, rerank, audio
- **Documents**: ingest local files, render deterministic artifacts, run built-in transforms
- **Operations**: inspect cache/runtime stats, benchmark, autotune, capture release artifacts
