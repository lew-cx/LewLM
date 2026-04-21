# Sessions and events

LewLM supports both persisted conversation state and live event streaming.

## Sessions

Session surfaces include:

- `lewlm list-sessions`
- `lewlm show-session <session-id>`
- `lewlm export-session <session-id>`
- `lewlm import-session <bundle-path>`
- `lewlm delete-session <session-id>`

HTTP surfaces:

- `POST /v1/sessions`
- `GET /v1/sessions`
- `GET /v1/sessions/{session_id}`
- `GET /v1/sessions/{session_id}/messages`
- `GET /v1/sessions/{session_id}/export`
- `POST /v1/sessions/import`
- `DELETE /v1/sessions/{session_id}`

Python surfaces:

- `create_session()`
- `list_sessions()`
- `get_session_detail()`
- `export_session()`
- `import_session()`
- `delete_session()`

## Context policies

LewLM supports three built-in context policies:

| Policy | Behavior |
| --- | --- |
| `full_history` | use all prior turns |
| `last_turn` | use only the last exchange |
| `summary_and_last_turn` | compact history plus the final exchange |

## Event streaming

LewLM emits runtime and lifecycle events through:

- `GET /v1/events` for SSE
- `WebSocket /v1/events`
- Python `subscribe_events()` for in-process consumers

Each emitted event now includes shared top-level fields when LewLM can resolve them:

- `request_id`
- `model_id`
- `runtime`
- `capability`
- `operation`
- `stage`
- `status`

For chat and streaming request events, the `payload` also includes a `serving` object with the current serving-core view for that request: active phase, queue residency so far, runtime adapter kind, streaming ownership, and any cancellation request LewLM has observed.

## Event categories

Current event families include:

- request lifecycle
- token and reasoning deltas
- speculation lifecycle
- model scan and loading
- audio transcription and speech
- document parse, render, and transform
- tool lifecycle
- cluster lifecycle
- autotune completion

### Speculation events

When LewLM runs speculative decoding on a compatible path, it emits:

- `speculation.started` when draft/verify work begins
- `speculation.accepted` when the request completes with accepted, rejected, verified, and fallback counts

Those payloads include the speculation `mode` and an `execution_path` so host apps can distinguish LewLM-owned controller execution from explicit backend passthrough behavior.

## SSE shape

SSE events use standard event-stream framing:

```text
event: request.completed
data: {"event_id":"...","type":"request.completed","scope":"request","request_id":"...","capability":"chat","operation":"text.generation","status":"completed","payload":{...}}
```

LewLM also emits keep-alive comments when needed to keep the connection active.
