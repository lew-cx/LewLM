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

## MLX local app backend

Use this on Apple Silicon when you want the preferred local runtime path.

```bash
lewlm doctor
mkdir -p ~/.lewlm/models
# put compatible MLX or other supported local bundles there
lewlm scan
lewlm list-models
lewlm capabilities <model-id>
lewlm warm <model-id>
lewlm chat "Hello from LewLM"
lewlm serve
```

## GGUF fallback backend

Use this when you want the main cross-platform fallback path.

```bash
lewlm doctor
mkdir -p ~/.lewlm/models
# put GGUF files there
lewlm scan
lewlm list-models
lewlm capabilities <model-id>
lewlm warm <model-id>
lewlm chat "Hello from LewLM"
lewlm serve
```

## Documents-enabled backend

Use this when you want local document ingest, render, or transform workflows, with or without model execution.

```bash
lewlm doctor
lewlm list-skills
lewlm transform --input examples/receipt-transform.json --output ./receipt.md
```

If you also want chat or responses work, pair `.[documents]` with `.[mlx]` or `.[llamacpp]`.

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
    lewlm.scan_models()
```

See [Python API reference](../reference/python-api.md) for the full surface.
