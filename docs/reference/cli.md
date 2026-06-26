# CLI reference

LewLM's CLI is grouped around serving, model management, documents, operations, and experimental cluster flows.

## Command map

| Group | Commands |
| --- | --- |
| Server and config | `serve`, `doctor`, `config`, `cache` |
| Model registry | `scan`, `list-models`, `models scan`, `models list`, `models import`, `models artifacts`, `capabilities`, `warm`, `unload` |
| Runtime evidence | `runtime probe`, `bridges test` |
| Conversion and tuning | `convert`, `benchmark`, `bench`, `autotune`, `optimize` |
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
- current-host recommended feature paths for chat, semantic text, vision, audio, and structured output
- runtime-pack and feature-pack status
- resolved configuration
- runtime availability
- storage readiness
- target-platform and capability hints
- measured capability probe registry counts and per-category status on the current host
- runtime-support strategy, including the first-class non-Apple path and bridge-only boundaries

### `scan`

Scans configured or explicit model roots and updates the local registry.

`lewlm models scan` is the namespaced alias for the same operation.

### `list-models`

Shows a grouped human-facing model view by default so converted variants stay under one source model. Use `--all` to inspect every registered artifact row, or `--json` for the raw machine-readable inventory.

`lewlm models list` is the namespaced alias. `lewlm models import <path>` indexes an existing local file or directory without copying it, and `lewlm models artifacts <model>` shows lineage, conversion artifacts, latest benchmark evidence, and capability evidence for one model.

### `cache`

Shows managed cache stats by default. Use `lewlm cache clear-conversions` to remove cached conversion artifacts, clear their conversion-cache metadata, and rescan the configured model roots plus the conversions cache root so stale converted entries disappear from the local registry.

### `capabilities`

Shows per-model capability reporting, measured routing preference, downgrade notes, fallback guidance, capability evidence, and per-host measured probe summaries for batching, cache reuse, constrained decoding, compile/kernels, speculation, and adapter preservation. Runtime and benchmark payloads now also surface portable performance-core ownership modes such as `lewlm_owned`, `backend_native`, and `partial`.

### `runtime probe`

Runs a capability probe and emits LewLM's evidence vocabulary: `discovered`, `requires_install`, `requires_conversion`, `load_passed`, `generate_passed`, `benchmark_passed`, `probe_failed`, or `unsupported`.

By default, `lewlm runtime probe` uses `--mode routing`, which does not load a model or generate text. Use `--mode load --model <model-id>` to run an explicit runtime load smoke test for any routeable capability, or `--mode generate --model <model-id> --prompt "..."` to verify a chat-like generation path. These execution modes are opt-in so capability truth can be upgraded without surprising operators. Successful smoke probes persist runtime evidence and `lewlm models artifacts <model>` shows the stored `runtime_probe_records`. Non-chat generation requests are rejected as `probe_failed` rather than treated as proof.

### `bridges test`

Reports configured bridge runtime providers separately from packaged runtimes, including provider family, availability, ownership, and advertised capabilities.

### `convert`

Queues or resolves a conversion job with:

- conversion policy
- optional custom bit width
- optional structured quantization profile
- optional layer overrides

Use `lewlm convert <model-id> --plan` to inspect target options without queueing a job. The plan reports executable targets such as GGUF/llama.cpp, Mac-oriented MLX targets when available, and ONNX Runtime GenAI for Windows-native DirectML/CUDA/CPU work. The ONNX target is executable when the `onnx_genai` extra is installed; otherwise it is reported with state `requires_install` rather than as a fake conversion claim. Use `--target <target-id>` with the ids from that plan when you want to force a specific executable target; install-gated or unsupported targets return an explicit compatibility report rather than queueing fake work.

### `benchmark`

Runs benchmark flows and emits artifact-backed summaries, including when measured adapter comparisons are persisted but downgraded instead of adopted. Benchmark feature records preserve ownership-mode evidence so cross-platform paths can report truthful backend-native or partial preservation without claiming universal parity, and external-adapter wins now stay bridge-only when they would otherwise replace a first-class packaged runtime.

`lewlm bench` is a shorter alias for the common managed benchmark path.

### `autotune`

Benchmarks serving-profile candidates and persists the recommended profile.

`lewlm optimize` is the operator-facing alias for the same benchmark-backed serving-profile recommendation flow.

### `chat`

The richest interactive command. It can use:

- direct prompt text
- system/developer prompt overrides
- prompt files
- structured tool metadata
- MCP-style tool metadata
- local attachments
- reasoning visibility controls

When a chat request includes `--response-format-file` or `--output-schema-file`, `lewlm chat --json` now includes the same `structured_output` payload exposed by the HTTP and Python surfaces. The default human-readable output also prints a concise structured-output status line so decode-time enforcement and prompt-guided fallback are distinguishable without inspecting internal metadata.

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
