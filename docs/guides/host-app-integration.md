# Host-app integration

LewLM's host-app surface is meant to stay small, typed, and app-agnostic.

Start with three public entry points:

1. `examples/integration-bundle.json` for checked-in schemas and example payloads
2. `/v1/openapi.json` for the live route index and content-type metadata
3. `LewLMAppClient` for thin typed Python helpers over the same public contracts

## Starter proofs

The repository now includes `examples/app_starter_proofs.py` for four small app-shaped prove-outs:

- `chat-app` for a structured chat UI contract
- `grounded-answer-app` for ingest plus citation-aware answers
- `document-ingest-app` for source and chunk packaging reuse
- `local-tool-app` for deterministic local-tool execution through the shared tool contract

Run the same proof either against an embedded instance or a local LewLM server:

```bash
python examples/app_starter_proofs.py local-tool-app
python examples/app_starter_proofs.py --base-url http://127.0.0.1:8080 local-tool-app
```

For event-surface prove-out, reuse `examples/http_api_integration.py events --count 3` to watch the SSE framing a host app would consume for progress badges, audit panes, or request-status views.

## Structured output contracts

Host apps can request structured output on both text-generation APIs:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `LewLMAppClient.chat_completion(...)`
- `LewLMAppClient.responses(...)`

Prefer:

- `response_format`
- `response_format_path` on local file-backed prompt surfaces

LewLM also still accepts the legacy aliases:

- `output_schema`
- `output_schema_path`

Set `include_prompt_trace=true` when you want LewLM to return the applied output contract and prompt-template selection in a machine-readable `prompt_trace`.

Read the response-side `structured_output` block when you need stable enforcement, fallback, and validation metadata:

- `enforcement` and `decoder_enforced` tell you whether LewLM actually enforced the contract or fell back to prompt guidance
- `validation` reports grammar enforcement state plus JSON parse or schema-validation results
- `parsed_output` surfaces the parsed JSON value when LewLM could decode it successfully

See:

- `examples/http_api_integration.py chat-structured`
- `examples/python_app_client.py`

**Current boundary:** LewLM now enforces JSON-schema and grammar contracts at decode time on supported runtime paths, and returns explicit prompt-guided fallback metadata when the selected runtime cannot honor the requested constraint mode.

## Citation-ready source and chunk packaging

`POST /v1/documents/ingest` and `LewLMAppClient.ingest_documents()` return app-facing packaging that lines up with grounded-answer and citation flows:

- `sources[]` with stable `source_id` and display-ready `source_label`
- `chunks[]` with `chunk_id`, `section_id`, `source_id`, `source_label`, and `section_label`

Those fields let a host app keep its own retrieval or answer-rendering logic while reusing LewLM's ingest output directly.

```json
{
  "source_id": "src-001",
  "chunk_id": "src-001-sec-0001-chunk-0001",
  "section_id": "src-001-sec-0001",
  "source_label": "source.md",
  "section_label": "source.md / Summary"
}
```

The checked-in ingest example in `examples/integration-bundle.json` shows the full response shape.

## Citation-aware chat and response packaging

LewLM also exposes a generic grounded-answer contract on both text-generation APIs:

- `POST /v1/chat/completions`
- `POST /v1/responses`

Pass `citation_context` with caller-supplied `sources[]` and `chunks[]` packages. LewLM teaches the model to emit stable citation markers, strips valid markers from the visible text, and returns machine-readable `citations[]` aligned with:

- `source_id`
- `chunk_id`
- `section_id`
- `source_label`
- `section_label`

That lets a host app keep its own rendering model while still receiving stable references instead of scraping inline citation text.

## Stateless retrieval helper

LewLM now exposes a separate stateless retrieval helper on top of the same source and chunk packages:

- `POST /v1/retrieval/context`
- `LewLMAppClient.retrieve_context(...)`

The request takes:

- `query`
- `candidate_sources[]`
- `candidate_chunks[]`
- optional `embedding_model` / `rerank_model`
- strategy controls such as `use_embeddings`, `use_rerank`, and `top_k`

The response returns:

- ranked `items[]` with the selected `chunk` and optional `source`
- per-item `embedding_score` / `rerank_score`
- stage metadata for embedding and rerank passes
- a top-level `metadata` envelope aligned with the rest of the API

This keeps retrieval app-agnostic and reusable while still avoiding a LewLM-managed collection layer.

**Current boundary:** LewLM does not manage persistent vector storage, collection CRUD, delete-by-source flows, or app-owned memory policy here. The caller still owns candidate selection and persistence.

## Related references

- [Integration bundle reference](../reference/integration-bundle.md)
- [HTTP API reference](../reference/http-api.md)
- [Python API reference](../reference/python-api.md)
- [Tools and skills](tools-and-skills.md)
- [Sessions and events](sessions-and-events.md)
- [Documents](documents.md)
- [Chat and responses](chat-and-responses.md)
