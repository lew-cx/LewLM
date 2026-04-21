# Python API reference

LewLM exposes an embeddable `LewLM` facade for in-process use.

## Construction

```python
from lewlm import LewLM, LewLMSettings

with LewLM(LewLMSettings()) as lewlm:
    ...
```

The facade also supports async context-manager usage and `create_app()` for binding a FastAPI app to the same service container.

## Method groups

### Configuration and lifecycle

| Method | Purpose |
| --- | --- |
| `create_app()` | build a FastAPI app bound to the facade |
| `app_client()` | typed app-helper surface for embedded callers |
| `config_snapshot()` | redacted settings snapshot |
| `health()` | in-process install-profile, health, and readiness snapshot |
| `close()` / `aclose()` | release owned resources |

### Models and capabilities

| Method | Purpose |
| --- | --- |
| `scan_models()` | scan roots and refresh the registry |
| `inventory()` | full inventory envelope |
| `list_models()` | manifest list |
| `get_model()` | single manifest |
| `model_capabilities()` | capability, readiness, and measured probe report |

### Skills and tools

| Method | Purpose |
| --- | --- |
| `list_skills()` / `get_skill()` | skill catalog access |
| `list_tools()` / `get_tool()` | tool catalog access |
| `execute_tool()` | run a local tool request |

### Sessions

| Method | Purpose |
| --- | --- |
| `create_session()` | create a persisted session |
| `list_sessions()` | enumerate sessions |
| `get_session_detail()` | session plus turns |
| `session_messages()` | flattened conversation |
| `export_session()` / `import_session()` | portable bundles |
| `delete_session()` | remove a session |

### Conversion and jobs

| Method | Purpose |
| --- | --- |
| `submit_conversion()` | queue a conversion job |
| `get_job()` | read a job |
| `wait_for_job()` / `wait_for_job_async()` | poll for completion |

### Inference

| Method | Purpose |
| --- | --- |
| `chat()` / `chat_sync()` | non-streaming text generation |
| `stream_chat()` | streaming generation |
| `warm_model()` / `warm_model_sync()` | warm a model |
| `unload_model()` / `unload_model_sync()` | unload a model |

`chat()` and `chat_sync()` return a `ChatExecution` object. Alongside `.response`, it exposes `.metadata`, which mirrors the HTTP execution metadata envelope with request identity, resolved model/runtime details, routing summary, queue/load/execute timing, and the final chat serving-core summary. The returned `.request_metadata` also includes a `serving` block with the runtime adapter shape, queue residency entries, and phase history for that request.

### Documents and telemetry

| Method | Purpose |
| --- | --- |
| `generate_document()` | render from `DocumentIR` |
| `ingest_documents()` | parse files into structured output |
| `transform_document()` | apply built-in skill |
| `cache_stats()` | cache snapshot |
| `runtime_stats()` / `runtime_stats_sync()` | readiness, runtime, scheduler, serving-core, performance-feature, and measured capability-registry diagnostics |
| `subscribe_events()` | in-process event subscription with normalized event envelopes |

## Useful companion types

You will commonly work with:

- `LewLMSettings`
- `LewLMAppClient`
- `ChatExecution`
- `PromptCompilationRequest`
- `ConversionJobRequest`
- `DocumentIR`
- `DocumentTransformRequest`
- `ToolExecutionRequest`

## Typed app helpers

`LewLMAppClient` is the thin helper layer for common host-app calls. It supports the same surface in two modes:

1. embedded: `lewlm.app_client()` or `LewLMAppClient.from_lewlm(lewlm)`
2. local server: `LewLMAppClient.from_http("http://127.0.0.1:8080")`

Helper methods:

| Method | Purpose |
| --- | --- |
| `list_tools()` / `get_tool()` | tool catalog helper access |
| `execute_tool()` | local-tool execution with the shared API envelope |
| `health()` | typed health response |
| `runtime_stats()` | typed runtime diagnostics, including first-class performance-feature metrics plus the measured capability registry summary for the current host |
| `chat_completion()` | chat-completions request/response helper |
| `responses()` | responses-style request/response helper |
| `embeddings()` | embeddings helper |
| `retrieve_context()` | stateless retrieval helper over caller-provided chunks |
| `rerank()` | rerank helper |
| `ingest_documents()` | document-ingest helper |

The helper methods accept either the request model directly or the common keyword arguments for that call. `chat_completion()` and `responses()` also accept the optional `citation_context` package used by the HTTP chat and responses surfaces, so embedded apps can pass known `sources[]` / `chunks[]` packages and consume returned `citations[]` without rebuilding LewLM's source-linking metadata.

These helpers stay intentionally narrow and do not replace the raw `LewLM` facade or the raw HTTP API when you need a lower-level path.

For host-app adoption, two details matter in particular:

- `chat_completion()` and `responses()` accept `response_format` plus `include_prompt_trace` so apps can request structured output and inspect the returned contract; `response_format_path` is available when you want the same contract loaded from a local file, and the legacy `output_schema` / `output_schema_path` aliases remain available for compatibility.
- `chat_completion()` and `responses()` return `structured_output` metadata so host apps can distinguish prompt-guided fallback from future decode-time enforcement and inspect post-generation validation details.
- `chat_completion()` and `responses()` accept `citation_context` so host apps can round-trip LewLM ingest packaging into grounded answers and receive stable `citations[]` back.
- `ingest_documents()` returns the same `sources[]` and `chunks[]` packages as the HTTP API, including `source_id`, `chunk_id`, `section_id`, `source_label`, and `section_label` fields that are useful for citation packaging.
- `retrieve_context()` consumes those same caller-provided `sources[]` and `chunks[]` packages and returns ranked context items plus stage metadata for embedding and rerank passes.

## Example

The repository includes:

- `examples/python_api_workflow.py` for the broader embeddable facade workflow
- `examples/python_app_client.py` for the thin typed helper surface
- `examples/app_starter_proofs.py` for chat-app, grounded-answer, document-ingest, and local-tool starter proofs
- `docs/guides/host-app-integration.md` for the app-facing structured-output and citation-ready adoption path

`examples/python_api_workflow.py` demonstrates:

- scanning
- conversion job submission and polling
- chat execution
- document generation
- tool execution
- session export
- runtime stats
