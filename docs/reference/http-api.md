# HTTP API reference

LewLM serves a local FastAPI app with OpenAPI at:

```text
/v1/openapi.json
```

## Route groups

### Health and operations

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/health` | service, storage, configuration, install-profile, pack, and capability-readiness health |
| `GET` | `/v1/cache/stats` | cache and performance-feature snapshot |
| `GET` | `/v1/runtime/stats` | readiness, runtime, scheduler, and residency stats |
| `GET` | `/v1/jobs/{job_id}` | background job status |
| `POST` | `/v1/benchmarks/autotune` | serving-profile recommendation |
| `GET` | `/v1/cluster/stats` | experimental cluster status |

### Models

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | list registry manifests |
| `GET` | `/v1/models/{model_id}/capabilities` | per-model capability, readiness, and runtime report |
| `POST` | `/v1/models/scan` | scan roots and refresh registry |
| `POST` | `/v1/models/convert` | queue or replay a conversion job |
| `POST` | `/v1/models/{model_id}/warm` | warm a model |
| `POST` | `/v1/models/{model_id}/unload` | unload a model |

### Chat and responses

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | chat-style completion API |
| `POST` | `/v1/responses` | responses-style completion API |

Features:

- JSON and multipart request modes
- optional streaming over `text/event-stream`
- session integration
- `response_format` structured-output contracts plus `structured_output` fallback/validation metadata
- prompt overrides, tools, and MCP-style tool metadata
- prompt trace output

### Multimodal

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/embeddings` | embeddings |
| `POST` | `/v1/retrieval/context` | stateless ranked retrieval over caller-provided chunks |
| `POST` | `/v1/rerank` | rerank candidate documents |
| `POST` | `/v1/audio/transcriptions` | audio transcription via JSON or multipart |
| `POST` | `/v1/audio/speech` | speech synthesis |

### Documents, tools, and skills

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/documents/generate` | render artifact from `DocumentIR` |
| `POST` | `/v1/documents/ingest` | ingest local files to structured output |
| `POST` | `/v1/documents/transform` | apply built-in document skill |
| `GET` | `/v1/tools` | list local tools |
| `GET` | `/v1/tools/{tool_name}` | tool descriptor |
| `POST` | `/v1/tools/execute` | execute local tool |
| `GET` | `/v1/skills` | list built-in skills |
| `GET` | `/v1/skills/{skill_name}` | skill descriptor |

These surfaces are owned by the `documents` feature pack. When that pack is disabled, `/v1/tools` and `/v1/skills` return empty catalogs and the document execution routes fail with `pack_unavailable`.

### Sessions and events

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/sessions` | create session |
| `GET` | `/v1/sessions` | list sessions |
| `GET` | `/v1/sessions/{session_id}` | session detail |
| `GET` | `/v1/sessions/{session_id}/messages` | flattened messages |
| `GET` | `/v1/sessions/{session_id}/export` | portable bundle |
| `POST` | `/v1/sessions/import` | import bundle |
| `DELETE` | `/v1/sessions/{session_id}` | delete session |
| `GET` | `/v1/events` | SSE event stream |
| `WS` | `/v1/events` | WebSocket event stream |

### Cluster

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/cluster/status` | coordinator or worker status |
| `POST` | `/v1/cluster/tokens` | issue enrollment token |
| `POST` | `/v1/cluster/workers/enroll` | enroll worker |
| `POST` | `/v1/cluster/workers/heartbeat` | refresh worker heartbeat |
| `POST` | `/v1/cluster/plans` | distributed plan |
| `POST` | `/v1/cluster/worker/pipeline-stage` | experimental stage handoff |

## Request modes

| Mode | Where used | Notes |
| --- | --- | --- |
| JSON | most routes | standard request bodies |
| Multipart | chat, responses, audio transcription | uploads are staged in secure workspaces |
| SSE | chat streaming, responses streaming, event stream | event-stream framing and keep-alives |
| WebSocket | `/v1/events` | normalized event payloads as JSON |

## Shared execution metadata

The main execution responses for chat, responses, embeddings, retrieval, rerank, audio, and document routes include a top-level `metadata` object with a stable machine-readable envelope.

| Field | Meaning |
| --- | --- |
| `version` | metadata contract version (`v1`) |
| `request_id`, `created` | request identity and creation timestamp |
| `result_origin` | `runtime`, `cache_hit`, `coalesced`, `tool_execution`, or `idempotent_replay` |
| `model` | requested/resolved model IDs plus runtime details |
| `routing` | shared routing summary, including route kind and reason |
| `timing` | queue, load, execute, and total durations in milliseconds |
| `serving` | chat/responses serving-core summary, including final phase, adapter kind, queue residency, and batching shape when applicable |
| `idempotency_key`, `idempotent_replay` | present on replay-capable document/tool-style flows |

For streaming chat and responses APIs, the envelope is attached to the final SSE chunk so consumers can read completed timing data without a separate lookup.

The retrieval surface also includes per-stage `embedding_stage` and `rerank_stage` summaries so host apps can distinguish the overall helper request from the underlying scoring passes.

## Consumer-ready fields

For host applications, the main machine-readable readiness fields are:

- `/v1/health.readiness`
- `/v1/health.configuration.runtime_packs[]`
- `/v1/health.configuration.feature_packs[]`
- `/v1/runtime/stats.readiness`
- `/v1/runtime/stats.runtime_packs[]`
- `/v1/runtime/stats.feature_packs[]`
- `/v1/runtime/stats.runtimes[].readiness_state`
- `/v1/runtime/stats.measured_capability_registry`
- `/v1/models/{model_id}/capabilities.runtime_candidates[].readiness_state`
- `/v1/models/{model_id}/capabilities.capabilities[].readiness_state`
- `/v1/models/{model_id}/capabilities.measured_capabilities[]`
- `/v1/events` top-level `request_id`, `capability`, `operation`, `stage`, and `status`

## Guards and expectations

Request handling is subject to:

- request size limits
- content-type validation per endpoint
- optional API-key enforcement
- rate limiting

See [Security](../security.md) for the detailed guardrail list.
