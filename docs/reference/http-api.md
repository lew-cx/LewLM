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
| `GET` | `/v1/runtime` | stable process identity and compact live counts |
| `GET` | `/v1/runtime/stats` | readiness, runtime, scheduler, residency, and runtime-strategy stats |
| `GET` | `/v1/runtime/residencies` | live model residency snapshots |
| `GET` | `/v1/model-lifecycle/operations/{operation_id}` | poll an asynchronous lifecycle operation |
| `DELETE` | `/v1/model-lifecycle/operations/{operation_id}` | cancel a pending/running lifecycle operation |
| `POST` | `/v1/requests/{request_id}/cancel` | cancel an in-flight request by its `x-request-id` handle |
| `GET` | `/v1/jobs/{job_id}` | background job status |
| `POST` | `/v1/benchmarks/autotune` | serving-profile recommendation |
| `GET` | `/v1/serving-profiles` | stored serving profiles, newest first (`model`, `capability`, `limit` from 1 to 500) |
| `GET` | `/v1/cluster/stats` | experimental cluster status |

### LewLM middleware evidence

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/lewlm/capabilities` | host-level middleware capability evidence, runtime providers, and readiness |
| `POST` | `/v1/lewlm/probes` | host/model routing probe, or opt-in model load/generation smoke probe |
| `POST` | `/v1/lewlm/conversions/plan` | read-only conversion target planning without queueing a job |
| `POST` | `/v1/lewlm/conversions` | conversion job alias in the LewLM namespace |
| `GET` | `/v1/lewlm/conversions/{job_id}` | conversion job status alias in the LewLM namespace |
| `POST` | `/v1/lewlm/benchmarks` | benchmark run through the LewLM namespace |
| `GET` | `/v1/lewlm/models/{model_id}/artifacts` | artifact lineage, conversion artifacts, latest benchmark, and capability evidence |

`POST /v1/lewlm/probes` defaults to `{"mode": "routing"}`, which does not load or generate from a model. Use `{"mode": "load", "model_id": "..."}` for runtime load evidence on any routeable capability, or `{"mode": "generate", "model_id": "...", "prompt": "...", "max_tokens": 1}` for chat-like generation evidence. Successful smoke probes return and persist `load_passed` or `generate_passed`; backend and policy failures return and persist `probe_failed` with the reason. Stored smoke evidence appears in `GET /v1/lewlm/models/{model_id}/artifacts.runtime_probe_records`.

### Models

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | list registry manifests, annotated with per-model serving readiness |
| `GET` | `/v1/models/{model_id}` | one manifest plus its readiness annotation |
| `GET` | `/v1/models/{model_id}/capabilities` | per-model capability, readiness, and runtime report |
| `POST` | `/v1/models/scan` | scan roots and refresh registry |
| `POST` | `/v1/models/convert` | queue or replay a conversion job |
| `GET` | `/v1/models/{model_id}/residency` | one model's live residency, if present |
| `POST` | `/v1/models/{model_id}/warm` | load once and run the backend warm hook |
| `POST` | `/v1/models/{model_id}/drain` | refuse new leases, wait up to `model_drain_timeout_seconds`, and unload |
| `POST` | `/v1/models/{model_id}/drain-operations` | return 202 and drain in a pollable background operation |
| `POST` | `/v1/models/{model_id}/unload` | unload only when no usage lease is active |

Scoped lifecycle authorization uses operator credentials for residency inspection, warm, and drain, and administrator credentials for unload and disruptive diagnostics. Application identity headers are audit context, not credentials. Async drain records use `pending`, `running`, `succeeded`, `failed`, and `cancelled` states and accept an application-scoped `idempotency_key`.

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
- prompt trace output, on the response body and on a stream's terminal chunk alike
- message `role` is a closed set — `system`, `developer`, `user`, `assistant`, `tool` — so an unrecognized role is rejected rather than serialized into the prompt as an unknown tag

### Multimodal

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/embeddings` | embeddings |
| `POST` | `/v1/retrieval/context` | stateless ranked retrieval over caller-provided chunks |
| `POST` | `/v1/rerank` | rerank candidate documents |
| `POST` | `/v1/audio/transcriptions` | audio transcription via JSON or multipart |
| `POST` | `/v1/audio/speech` | speech synthesis |
| `GET` | `/v1/audio/voices` | synthesis voices resolvable for a model on this host |
| `POST` | `/v1/tokenize/count` | model-accurate token count and deterministic truncation boundary |

Audio capability is per model, not per runtime: a manifest carries `audio_roles`, and the inventory's `ready_capabilities` names only the side a model serves, so a synthesis request against a transcription model is a `routing_error` rather than a `500`. `GET /v1/audio/voices` reports the voices a synthesis model can resolve **on this host**, since a backend may keep its voice packs in its own download cache rather than in the model directory. A listed voice is a guarantee; an absent one is not a refusal, because a backend may fetch a name on demand.

### Documents, tools, and skills

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/documents/generate` | render artifact from `DocumentIR` |
| `POST` | `/v1/documents/ingest` | ingest local paths **or uploaded bytes** to structured output |
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

Both surfaces accept the same filters as query parameters: `types`, `scope`, `request_id` and `model_id`. Each may be repeated or comma-separated, values within one parameter are alternatives, and the parameters combine — `?types=token.delta&request_id=req-1` is one request's tokens and nothing else. Filtering is applied before an event is queued for the connection, so an excluded event is never serialized or sent. A value that names no known event type or scope is refused with `invalid_request` (422 on SSE, close code `1008` before the WebSocket handshake is accepted) rather than silently ignored, because an ignored filter returns an empty stream that looks exactly like a quiet server.

Replay is not available: a reconnecting client resumes from the moment it reconnects, and `Last-Event-ID` is not honoured.

The WebSocket handshake is guarded like every other route. When `api_key_required` is set, send the key as an `x-api-key` header or, from a browser that cannot set handshake headers, as a `lewlm.api-key.<key>` entry in `Sec-WebSocket-Protocol`. The key is never echoed back as the accepted subprotocol. An unauthenticated handshake is closed with code `1008` before it is accepted, and a rate-limited one with `1013`.

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
| `version` | envelope schema version (`v1`). This does **not** identify the code that ran — see `components` |
| `request_id`, `created` | request identity and creation timestamp |
| `correlation_id` | caller-supplied correlation identifier, or `null`. LewLM never generates one |
| `components[]` | named, versioned parser/renderer/chunker/OCR/scoring/tokenizer records that produced this result |
| `result_origin` | `runtime`, `cache_hit`, `coalesced`, `tool_execution`, or `idempotent_replay` |
| `model` | requested/resolved model IDs plus runtime details |
| `routing` | shared routing summary, including route kind and reason |
| `timing` | queue, load, execute, and total durations in milliseconds |
| `serving` | chat/responses serving-core summary, including final phase, adapter kind, queue residency, and batching shape when applicable |
| `idempotency_key`, `idempotent_replay` | present on replay-capable document/tool-style flows |

For streaming chat and responses APIs, the envelope is attached to the final SSE chunk so consumers can read completed timing data without a separate lookup.

The retrieval surface also includes per-stage `embedding_stage` and `rerank_stage` summaries so host apps can distinguish the overall helper request from the underlying scoring passes.

### Component provenance

`metadata.version` identifies the envelope, not the implementation. Each entry in `metadata.components[]` names one component that actually ran:

| Field | Meaning |
| --- | --- |
| `kind` | `parser`, `renderer`, `chunker`, `ocr`, `scoring_policy`, or `tokenizer` |
| `name` | stable LewLM-owned component name, not a package name |
| `version` | version of the LewLM-owned behaviour; bumped when observable output changes |
| `implementation` / `implementation_version` | third-party distribution doing the work, when one does |
| `deterministic` | whether the same input reproduces the same output |

### Request identifiers

Every response carries `x-request-id`. Send your own URL-safe handle and LewLM echoes it verbatim; omit it and LewLM mints one. Handles are 1–128 characters from `A-Z`, `a-z`, `0-9`, `.`, `_`, `~`, and `-`; other values are refused with `invalid_request` rather than being silently trimmed into a different cancellation identity. The header is set on error responses too, so a failed call is still traceable. This is distinct from `x-lewlm-correlation-id`: the request ID identifies one HTTP call, the correlation ID ties a caller's workflow together across many.

### Cancelling an in-flight request

A caller awaiting its own HTTP response can cancel by disconnecting. A separate process — an API that dispatched work to a worker, an orchestrator supervising a queue — cannot: it holds no socket for that request. `POST /v1/requests/{request_id}/cancel` addresses the request by the `x-request-id` handle it was sent with, so cancellation does not depend on owning the connection.

```
POST /v1/responses            x-request-id: 5f1c…   x-lewlm-application-id: docktizo
POST /v1/requests/5f1c…/cancel                      x-lewlm-application-id: docktizo
```

The call is idempotent and returns the current state of the handle:

| `state` | Meaning |
| --- | --- |
| `cancelling` | a request with this handle is in flight and has been signalled |
| `pending` | no request with this handle is known. The intent is held until `expires_at`, and a request arriving with it is stopped immediately — so an orchestrator may cancel before the worker has even issued the call |
| `cancelled` | the request stopped at a checkpoint |
| `completed` | the request reached a terminal state without observing the signal |

Repeat the call to observe the outcome; the state moves from `cancelling` to `cancelled` or `completed`. A cancelled request fails with `request_cancelled` (499) and `details.stage` names the checkpoint that stopped it.

Use an unpredictable UUID or comparable high-entropy handle within the URL-safe alphabet above, and do not reuse it while a request is active. Concurrent reuse is ambiguous and is refused with `request_handle_conflict` (409), leaving the original request registered.

When API-key authentication is enabled, LewLM binds the handle to a keyed fingerprint of the credential that issued or first cancelled it; the raw key is never retained in the cancellation record. A different credential is refused with `tool_authorization_error`. `x-lewlm-application-id` is untrusted observability metadata, not authorization: changing it neither grants nor removes cancellation authority. In an open deployment there is no authenticated owner to compare, so possession of the unpredictable handle is the capability to cancel it.

What LewLM does **not** claim:

- **Cooperative, not pre-emptive.** The request stops at its next checkpoint: runtime admission (including while queued behind it), the entry to a tool or document operation, and between sources during multi-source ingestion. Work already committed to one bounded backend call runs to completion.
- **Process-local.** Only the instance named by `runtime_instance_id` on the record can act on the handle. Behind a load balancer, cancel against the instance that accepted the request.
- **Batched requests share an admission.** On a runtime that batches continuously, several requests pass admission together; once such a batch is dispatched, its members are no longer individually cancellable.
- **Streaming remains cooperative.** The handle stays active until the response body closes, and LewLM checks it before the first frame and between stream items or heartbeats. Work already awaiting one bounded backend chunk is not pre-empted. Disconnecting remains the immediate option for the process that owns the connection, and LewLM closes the source stream deterministically on the way out.
- **Sandboxed tool and document work is stopped before it starts**, not during: with `tool_sandbox_enabled`, the operation runs in a subprocess that no checkpoint reaches.

Handles are remembered in bounded, in-memory maps (`request_cancellation_max_tracked_requests`, `request_cancellation_intent_ttl_seconds`); an old handle eventually reports `pending` again rather than `completed`.

### Sampling controls

Chat and responses requests accept a `sampling` object: `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, and `stop`.

Backends differ in what they expose, so LewLM never silently drops a control. `metadata.sampling` reports what happened:

| Field | Meaning |
| --- | --- |
| `runtime` | which runtime handled the request |
| `requested` | the controls the caller set |
| `applied` | the controls that reached the backend |
| `unsupported` | controls this backend — or this installed build of it — cannot honor |
| `deterministic` | true only when a `seed` was requested **and** actually applied |

`unsupported` reflects the running system, not the documented API: a control the backend family nominally supports but the installed build does not accept is still reported as unsupported.

### Streaming usage

The final streaming chunk carries `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`). Earlier chunks have `usage: null`, since the totals are not knowable before the stream ends. `usage.measured` is `true` when the counts came from the model's own tokenizer and `false` when the backend exposed none and LewLM had to estimate. `usage.cached_tokens` is present only when the backend itself reported prompt tokens served from its prefix cache (OpenAI-style `prompt_tokens_details.cached_tokens`, e.g. SGLang with `--enable-cache-report` or vLLM with `--enable-prompt-tokens-details`); absent means unknown, never zero — LewLM does not infer cache hits.

Abandoning a stream closes it deterministically: LewLM closes the source stream on the way out rather than waiting for garbage collection, so the backend learns the consumer is gone and stops generating.

### Correlation identifiers

Send `x-lewlm-correlation-id`, or set `correlation_id` in the request body on chat, responses, and document routes. The header wins when both are present. LewLM echoes the header on the response, threads the value through `metadata.correlation_id`, and stamps it onto every event emitted for the request. LewLM never invents a correlation ID, so an absent value stays `null` and a caller can always distinguish its own identifier from a LewLM `request_id`. Values are trimmed and bounded to 128 characters.

## Error envelope

Non-success responses use one top-level machine-readable shape:

```json
{
  "error": {
    "code": "runtime_unavailable",
    "message": "The selected runtime is not ready on this host.",
    "details": {
      "support_path": "bridge",
      "fallback_guidance": ["Configure a loopback-only local bridge endpoint."]
    }
  }
}
```

`details` is where LewLM surfaces parity-specific guidance such as `support_path`, `feature_class`, `available_support_paths`, `bridge_only`, and `fallback_guidance`.

Common codes include:

| Code | Status | Meaning |
| --- | --- | --- |
| `model_load_failed` | 503 | The runtime is installed, but this model could not be loaded on this host. `details` carries `runtime`, `architecture_family`, `cause_type`, and `cause` so a host app can explain the failure. |
| `runtime_unavailable` | 503 | The selected runtime is not ready on this host. |
| `model_lifecycle_conflict` | 409 | A lifecycle action would interrupt active model use. |
| `invalid_request` | 422 | A request body or parameter failed validation. `details.fields[]` names each offending field with a message and type. |
| `internal_error` | 500 | An unexpected failure. Carries `details.cause_type` so a host app can report something actionable; LewLM never returns a bare, envelope-less 500. |
| `not_found` / `method_not_allowed` | 404 / 405 | Framework routing errors, normalized onto the same envelope. |
| `request_cancelled` | 499 | The request stopped because its `x-request-id` handle was cancelled. `details.stage` names the checkpoint. Not retryable: the caller withdrew the work. |
| `request_handle_conflict` | 409 | The handle is already active or reserved by another authenticated caller. Use a fresh high-entropy request ID. |
| `backend_contract_violation` | 502 | A backend returned a result LewLM cannot trust — a rerank index outside the candidate range, a duplicate index, an unscored candidate, or a non-finite score. Expected mainly on bridge-backed runtimes. |

Backend load failures are always returned in this envelope, including on the streaming chat and responses routes.

## Consumer-ready fields

For host applications, the main machine-readable readiness fields are:

- `/v1/health.install_profiles.recommended_feature_paths[]`
- `/v1/health.install_profiles.standards_acceptance_contract`
- `/v1/health.install_profiles.backend_inventory[]` (installed backend module versions; inventory evidence only, never a capability claim)
- `/v1/health.install_profiles.backend_feature_probes[]` (import-cheap API-presence probes per backend: speculation, KV cache controls, grammar enforcement, multimodal surfaces; presence is inventory evidence only)
- `/v1/health.install_profiles.llamacpp_build` (feature-detected llama.cpp build flavor: GPU offload support, heuristic accelerator hints, and backend system info; inventory evidence only)
- `/v1/models.capability_availability[]` (per-model `servable`, `chat_ready`, `ready_capabilities`, `blocked_capabilities`; pick a usable model without one request per model)
- `/v1/models.chat_ready_count` and `/v1/models.servable_count`
- `/v1/documents/ingest.sources[]` (upload documents as bytes with a caller-owned `source_id`, `expected_sha256`, and bounded `metadata`; no shared filesystem mount required, and `path` is `null` on every uploaded source)
- `/v1/documents/ingest.source_results[]` (one outcome per requested source: `status`, stable `error_code`, `retryable`, `chunk_count`, `content_sha256`, `provider_reference`, and per-source `components[]`), plus `ingested_count`, `failed_count`, and `partial`
- `/v1/retrieval/context.scoring_policy` (named, versioned ranking rules: `primary_signal`, `tie_break_signal`, `final_tie_break`, `normalization`, `missing_score_behaviour`, `deduplication`)
- `/v1/tokenize/count.token_count` and `.truncated_text` (exact counts from the selected model's tokenizer; truncation lands on a reproducible token boundary)
- `/v1/runtime.build` (`package_version`, `api_schema_version`, `source_commit`, `source_dirty`, `distribution_digest`, `install_kind`, `release_build`) — enough to prove which implementation produced an artifact
- `metadata.components[]` on every execution response (which renderer, parser, chunker, OCR engine, scoring policy, or tokenizer ran, and at what version)
- `metadata.correlation_id` on every execution response, and `correlation_id` on every event
- `metadata.sampling` (`requested` / `applied` / `unsupported` / `deterministic`) on chat and responses
- `usage` on the final streaming chunk, with `measured` distinguishing tokenizer counts from estimates
- `x-request-id` on every response, echoed from the caller when supplied
- `PATCH /v1/sessions/{session_id}` to rename a session or adjust metadata without touching turn history
- `/v1/chat/completions.tool_calls` and `/v1/responses.tool_calls` (strict, schema-validated model-emitted tool calls; `status` is one of `no_tool_calls`, `parsed`, `partial`, `failed`, with an explicit `issues[]` reason for anything not accepted, and it is `null` when the request declared no tools). Declaring `tools` also injects the accepted invocation shape into the compiled prompt, so callers do not spend their own `system_prompt` slot on it — see [Tool calling](../guides/host-app-integration.md#tool-calling)
- `/v1/health.readiness`
- `/v1/health.readiness.capabilities[].available_support_paths`
- `/v1/health.readiness.capabilities[].bridge_only`
- `/v1/health.configuration.runtime_packs[]`
- `/v1/health.configuration.feature_packs[]`
- `/v1/runtime/stats.readiness`
- `/v1/runtime/stats.standards_acceptance_contract`
- `/v1/runtime/stats.runtime_packs[]`
- `/v1/runtime/stats.feature_packs[]`
- `/v1/runtime/stats.runtimes[].readiness_state`
- `/v1/runtime/stats.measured_capability_registry`
- `/v1/runtime/stats.runtime_support_strategy`
- `/v1/models/{model_id}/capabilities.standards_acceptance_contract`
- `/v1/models/{model_id}/capabilities.runtime_candidates[].support_path`
- `/v1/models/{model_id}/capabilities.runtime_candidates[].readiness_state`
- `/v1/models/{model_id}/capabilities.capabilities[].support_path`
- `/v1/models/{model_id}/capabilities.capabilities[].readiness_state`
- `/v1/models/{model_id}/capabilities.capability_evidence[]`
- `/v1/models/{model_id}/capabilities.measured_capabilities[]`
- `/v1/models/{model_id}/capabilities.structured_output` (whether a `response_format` will be enforced at decode time or fall back to `prompt_guided`, per contract mode, before you spend a generation finding out)
- `/v1/lewlm/capabilities.capability_evidence[]`
- `/v1/lewlm/capabilities.runtime_providers[]`
- `/v1/lewlm/models/{model_id}/artifacts.capability_evidence[]`
- `/v1/events` top-level `request_id`, `capability`, `operation`, `stage`, and `status`

## Guards and expectations

Request handling is subject to:

- request size limits
- content-type validation per endpoint
- optional API-key enforcement
- rate limiting

See [Security](../security.md) for the detailed guardrail list.
