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

## Narrowing the stream

Both HTTP surfaces accept the same filters as query parameters, and the Python
facade takes the same filter as an object:

| Parameter | Narrows by |
| --- | --- |
| `types` | event type, e.g. `token.delta` |
| `scope` | `system`, `request`, or `job` |
| `request_id` | the request that produced the event |
| `model_id` | the model the event is about |

Each may be repeated or comma-separated. Values within one parameter are
alternatives; the parameters combine.

```text
GET /v1/events?types=token.delta,request.completed&request_id=req-1
```

```python
from lewlm.events.filters import EventFilter
from lewlm.events.schema import EventType

subscription = lewlm.subscribe_events(
    EventFilter(types=frozenset({EventType.TOKEN_DELTA}), request_ids=frozenset({"req-1"})),
)
```

The filter is applied on the bus, before an event is queued for the subscriber,
so an excluded event is never queued, serialized, or sent — a filtered
subscriber's queue stays shallow while an unfiltered one grows. A streamed
generation emits one `token.delta` per token to every listener, so this is the
difference between a busy host and an unusable one for a client that only wants
lifecycle events.

A value naming no known event type or scope is refused with `invalid_request`
rather than ignored: an ignored filter returns an empty stream that is
indistinguishable from a quiet server. SSE answers 422; the WebSocket closes
`1008` before accepting the handshake.

`exclude_types` is the one negative filter, for the common "everything except
the token flood" ask.

Every event carries a `cursor` (the SSE `id:` line) and the bus retains the
last `LEWLM_EVENT_REPLAY_BUFFER_SIZE` events (default 4096). A client that
drops sends the last cursor it saw back — `Last-Event-ID`, or `?after=` when it
cannot set headers, or `subscribe_events(after=...)` in process — and the
stream begins with an `events.resumed` marker saying how many events were
`replayed` and how many were `lost` (`null` for a cursor from another server
lifetime), then the replayed events in order, then live ones. A client can
therefore present a continuous timeline, or say exactly where its gap is.

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
