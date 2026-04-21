# CLI reference

LewLM's CLI is grouped around serving, model management, documents, operations, and experimental cluster flows.

## Command map

| Group | Commands |
| --- | --- |
| Server and config | `serve`, `doctor`, `config`, `cache` |
| Model registry | `scan`, `list-models`, `capabilities`, `warm`, `unload` |
| Conversion and tuning | `convert`, `benchmark`, `autotune` |
| Documents | `generate-doc`, `transform` |
| Tools and skills | `list-skills`, `show-skill`, `list-tools`, `show-tool`, `run-tool` |
| Sessions | `list-sessions`, `show-session`, `export-session`, `import-session`, `delete-session` |
| Chat | `chat` |
| Cluster | `cluster status`, `cluster issue-token`, `cluster join`, `cluster heartbeat`, `cluster plan`, `cluster benchmark` |

## Shared patterns

### JSON-oriented commands

Several commands support JSON output for automation-friendly use.

### Authorization-aware commands

When tool authorization is required, document and conversion flows can take:

```bash
--authorize <action>
```

### Idempotent commands

Document and conversion-oriented commands can also carry:

```bash
--idempotency-key <key>
```

## Notable commands

### `serve`

Starts the local FastAPI service.

### `doctor`

Operator diagnostics for:

- active install-profile summary
- runtime-pack and feature-pack status
- resolved configuration
- runtime availability
- storage readiness
- target-platform and capability hints
- measured capability probe registry counts and per-category status on the current host

### `scan`

Scans configured or explicit model roots and updates the local registry.

### `list-models`

Shows a grouped human-facing model view by default so converted variants stay under one source model. Use `--all` to inspect every registered artifact row, or `--json` for the raw machine-readable inventory.

### `cache`

Shows managed cache stats by default. Use `lewlm cache clear-conversions` to remove cached conversion artifacts, clear their conversion-cache metadata, and rescan the configured model roots plus the conversions cache root so stale converted entries disappear from the local registry.

### `capabilities`

Shows per-model capability reporting, measured routing preference, downgrade notes, fallback guidance, and per-host measured probe summaries for batching, cache reuse, constrained decoding, compile/kernels, speculation, and adapter preservation.

### `convert`

Queues or resolves a conversion job with:

- conversion policy
- optional custom bit width
- optional structured quantization profile
- optional layer overrides

### `benchmark`

Runs benchmark flows and emits artifact-backed summaries, including when measured adapter comparisons are persisted but downgraded instead of adopted.

### `autotune`

Benchmarks serving-profile candidates and persists the recommended profile.

### `chat`

The richest interactive command. It can use:

- direct prompt text
- system/developer prompt overrides
- prompt files
- structured tool metadata
- MCP-style tool metadata
- local attachments
- reasoning visibility controls

### `generate-doc` and `transform`

These are the main document-authoring commands:

- `generate-doc` renders a `DocumentIR`
- `transform` applies a built-in deterministic skill

Both commands depend on the `documents` feature pack being loaded.

### `run-tool`

Executes a local tool request from JSON, returning a structured envelope with a trace and result.

## Cluster subcommands

The cluster namespace is experimental and currently covers:

- status inspection
- token issue and enrollment
- worker heartbeat
- distributed plan creation
- distributed benchmark invocation

## Best companions

- [HTTP API reference](http-api.md)
- [Python API reference](python-api.md)
- [Troubleshooting](../operations/troubleshooting.md)
