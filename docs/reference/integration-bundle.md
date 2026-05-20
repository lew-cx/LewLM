# Integration bundle reference

LewLM publishes a compact checked-in integration bundle for host applications at:

```text
examples/integration-bundle.json
```

Use it alongside the live OpenAPI document at `/v1/openapi.json`.

## What the bundle contains

| Surface | Path | Bundle schema refs | Example payloads |
| --- | --- | --- | --- |
| chat | `POST /v1/chat/completions` | `#/schemas/chat.request`, `#/schemas/chat.response`, `#/schemas/chat.stream` | request, response, SSE chunk |
| responses | `POST /v1/responses` | `#/schemas/responses.request`, `#/schemas/responses.response`, `#/schemas/responses.stream` | request, response, SSE chunk |
| embeddings | `POST /v1/embeddings` | `#/schemas/embeddings.request`, `#/schemas/embeddings.response` | request, response |
| retrieval | `POST /v1/retrieval/context` | `#/schemas/retrieval.request`, `#/schemas/retrieval.response` | request, response |
| rerank | `POST /v1/rerank` | `#/schemas/rerank.request`, `#/schemas/rerank.response` | request, response |
| documents | `POST /v1/documents/ingest`, `POST /v1/documents/generate`, `POST /v1/documents/transform` | `#/schemas/documents.*` | request, response |
| events | `GET /v1/events` | `#/schemas/events.stream` | SSE frame, WebSocket event |

## How to use it

1. Read `examples/integration-bundle.json` to map a host-app request or response onto LewLM's stable shapes.
2. Use `/v1/openapi.json` for the live route index, operation IDs, and content-type metadata, and `GET /v1/health` when you need the current-host `recommended_feature_paths` / readiness contract that sits beside these request and response shapes.
3. Start from `examples/http_api_integration.py` for a minimal standard-library client that calls chat, responses, embeddings, retrieval, rerank, documents, and SSE events.
4. Use `examples/app_starter_proofs.py` when you want app-shaped proofs for structured chat, grounded answers, document ingest, and local tools instead of single-surface calls.

## Host-app prove-out details

- The chat and responses examples now show `citation_context` plus returned `citations[]`, alongside `response_format`, `structured_output`, and `include_prompt_trace`, so both grounded-answer and structured-output contracts are visible in the checked-in bundle.
- The retrieval example shows the stateless `candidate_sources[]` / `candidate_chunks[]` request shape and the ranked context package LewLM returns.
- The document-ingest example includes `source_id`, `chunk_id`, `section_id`, `source_label`, and `section_label` so the source packages used for citations and retrieval are visible without reading implementation code.
- For the broader adoption path, see [Host-app integration](../guides/host-app-integration.md).

## Notes

- The bundle is intentionally app-agnostic and does not assume any orchestration framework.
- Chat and responses accept both `application/json` and `multipart/form-data`; multipart requests send a JSON-encoded `payload_json` field plus uniquely named upload parts referenced by `upload_name`.
- The events surface is modeled as `StreamEvent` payloads over SSE or WebSocket. The checked-in SSE examples show the framing LewLM actually emits.
