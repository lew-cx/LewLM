# Host-app integration

LewLM's host-app surface is meant to stay small, typed, and app-agnostic.

Start with three public entry points:

1. `examples/integration-bundle.json` for checked-in schemas and example payloads
2. `/v1/openapi.json` for the live route index and content-type metadata
3. `LewLMAppClient` for thin typed Python helpers over the same public contracts

## The chat-UI contract in one place

A chat host app (Chap) needs one base URL and these routes, and nothing
engine-specific: `GET /v1/health`, `GET /v1/models`,
`GET /v1/models/{id}/capabilities`, `GET /v1/runtime`, `POST /v1/chat/completions`
(or `POST /v1/responses`), `POST /v1/requests/{x-request-id}/cancel`, and
optionally `GET /v1/events`. Exact payloads for every state a UI has to render
— streaming, tool calls, JSON output, usage, cancellation, an engine outage,
an interrupted stream — are in `examples/integration-bundle.json` under
`chap`, and a smoke script proves them over HTTP against a fake engine on any
machine:

```bash
python -m lewlm.testing.fake_backend --port 8080          # LewLM + fake engine to build a UI against, no model needed
python examples/chap_backend_smoke.py --fixture           # the same checks CI runs on Linux, macOS, and Windows
```

Read service, engine, and model state from three different places and never
infer one from another: `health.status` (this service), `health.engines[]` and
`runtime.startup.engines[]` (each engine's cached inventory state, no probe),
`runtime.startup.warm_models[]` (process-local residency). The
[Chap validation guide](chap-validation.md) has the full field map and the UI
checklist.

## Shared server versus embedded mode

Use embedded `LewLM()` for tests, notebooks, CLIs, and deliberate single-process applications. Production products should normally connect to one long-running server so RAG Chat, document-generation applications, and future tools share the same runtime-owned model objects and caches.

```python
from lewlm import LewLMAppClient

rag = LewLMAppClient.from_http("http://127.0.0.1:8080", application_id="rag-chat")
document_app = LewLMAppClient.from_http("http://127.0.0.1:8080", application_id="document-generator")
assert rag.runtime_info().runtime_instance_id == document_app.runtime_info().runtime_instance_id
```

Application identity is operational metadata, not authentication, and does not create a separate model residency. `list_model_residencies()` reports live state. `warm_model()` is an operator action; `unload_model()` is rejected with 409 while the model is used; `drain_model()` stops new leases and waits up to the configured `model_drain_timeout_seconds` limit (30 seconds by default).

For drains that may outlive a practical request, an HTTP-backed client can call `create_drain_operation()`, retain its `operation_id`, and poll `get_lifecycle_operation()`. The operation reports `pending`, `running`, `succeeded`, `failed`, or `cancelled`; repeating an identical submission with the same application-scoped idempotency key returns the original operation, while changing the model or timeout returns a conflict. The synchronous embedded backend completes its recorded operation before returning because it has no long-lived event loop for background work. The synchronous `drain_model()` method remains available.

Configure `lifecycle_operator_api_keys` for residency inspection, warm, and drain access. Configure `lifecycle_administrator_api_keys` for unload and disruptive diagnostics. When scoped keys are configured they are authoritative; `application_id` and `client_instance_id` add audit context but grant no permission.

Runtime stats contain bounded `request_metrics.applications` summaries. Use stable deployment-controlled application IDs as labels. Client instance IDs, request IDs, session IDs, user IDs, prompts, and document content belong only in appropriately protected logs or traces, never metric labels.

Run one server worker per intended model replica. Multiple Uvicorn workers are multiple operating-system processes with independent model memory. Horizontal replicas must be deliberate.

LewLM owns generic ingestion, retrieval, structured output, `DocumentIR`, validation, and deterministic rendering. A document product owns status-report or contract schemas, templates, workflow, approval, and artifact lifecycle. See `examples/shared_runtime_clients.py` for that adapter boundary.

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

Set `include_prompt_trace=true` when you want LewLM to return the applied output contract and prompt-template selection in a machine-readable `prompt_trace`. It works on streaming requests too: the trace arrives on the terminal chunk, exactly as `usage` does, so inspecting the compiled prompt never costs you the stream.

To know what a contract will get *before* spending a generation, read `structured_output` on `GET /v1/models/{model_id}/capabilities`. It reports the same `StructuredOutputRuntimeStatus` shape the response carries afterwards, per contract mode, so you can warn that a request will fall back to `prompt_guided` instead of discovering it from the result.

Read the response-side `structured_output` block when you need stable enforcement, fallback, and validation metadata:

- `enforcement` and `decoder_enforced` tell you whether LewLM actually enforced the contract or fell back to prompt guidance
- `grammar_relaxations` names bounds too large for the decoder's grammar parser, which LewLM enforced through `validation` after generation instead
- `validation` reports grammar enforcement state plus JSON parse or schema-validation results
- `parsed_output` surfaces the parsed JSON value when LewLM could decode it successfully

See:

- `examples/http_api_integration.py chat-structured`
- `examples/python_app_client.py`

**Current boundary:** LewLM now enforces JSON-schema and grammar contracts at decode time on supported runtime paths, and returns explicit prompt-guided fallback metadata when the selected runtime cannot honor the requested constraint mode.

## Remote-safe document ingestion

Path-based ingestion requires the caller and LewLM to share an identical absolute mount, and makes source identity depend on deployment paths. For a remote caller, upload bytes instead:

```python
from lewlm import LewLMAppClient

client = LewLMAppClient.from_http("http://127.0.0.1:8080", application_id="document-generator")

source = client.upload_source(
    "caller-owned-id",              # opaque identity the caller controls
    pdf_bytes,
    file_name="contract.pdf",
    media_type="application/pdf",   # expected_sha256 is computed for you
)
result = client.ingest_documents(sources=[source], correlation_id="job-42")
```

LewLM stages the bytes inside its own sandboxed workspace, verifies `expected_sha256` before parsing, and returns `path: null` on every uploaded source. The caller's `source_id` flows through to every chunk, so citation packaging never depends on where LewLM stored anything.

Uploads are bounded: 64 MiB per source, 32 metadata entries, and 1024 characters per metadata value.

### Per-source outcomes

Multi-source ingestion returns exactly one entry in `source_results[]` for every requested source, in request order, alongside `ingested_count`, `failed_count`, and `partial`:

| Field | Meaning |
| --- | --- |
| `source_id` | caller-provided ID for uploads, LewLM-derived for paths |
| `status` | `ingested` or `failed` |
| `error_code` | stable code: `checksum_mismatch`, `empty_source`, `source_too_large`, `unsupported_source_type`, `corrupt_source`, `parser_failed`, `parser_timeout`, `ocr_unavailable`, `access_denied`, `internal_error` |
| `retryable` | whether retrying this source unchanged could plausibly succeed |
| `chunk_count`, `section_count`, `content_sha256` | what LewLM actually produced |
| `provider_reference` | LewLM-side reference for correlating with logs and events |
| `components[]` | parser and OCR provenance for this specific source |

One bad source no longer discards the sources that parsed cleanly. A request only fails outright when no source survives, and the per-source detail is still carried in `details.source_results`.

## Typed document rendering

The typed client covers rendering, not just ingestion, so a host app does not need a second raw HTTP transport:

```python
response = client.generate_document(document=document_ir, output_format="pdf")
pdf_bytes = LewLMAppClient.document_bytes(response)   # base64 decoding included

transformed = client.transform_document(contract_replacement_request)
```

Both carry authorization, idempotency, correlation, structured errors, and renderer provenance on the same envelope as every other surface.

## Asynchronous client

`LewLMAppClient` is synchronous. An async host app should use `LewLMAsyncClient`, which pools connections and propagates cancellation into the HTTP request itself rather than abandoning a waiter while the server keeps working:

```python
from lewlm import LewLMAsyncClient

async with LewLMAsyncClient(
    "http://127.0.0.1:8080",
    application_id="document-generator",
    correlation_id="job-42",
) as client:
    result = await client.count_tokens(text=body, model=model_id)
    response = await client.generate_document(request, timeout_seconds=120)

    async for chunk in client.stream_chat_completion(chat_request):
        ...
```

Every operation accepts a per-operation `timeout_seconds`. `with_correlation_id()` returns a view that stamps a different correlation ID while sharing the same connection pool. Call `aclose()` (or use the context manager) when finished; it is idempotent.

## Cancelling from another process

Cancelling the task that awaits a call is enough when one process owns the request. It is not enough when the process that decides to stop the work is not the one holding the connection — an API accepting a cancel while a durable worker runs the generation, for example. Give the operation a handle you choose, and cancel it by name from wherever the decision is made:

```python
handle = str(uuid4())

# worker process
answer = await client.responses(request, request_id=handle)

# API or orchestrator process, no task or socket for that request
record = await orchestrator_client.cancel_request(handle)
```

`request_id` is accepted by the generation, retrieval, tool, and document operations, and travels as the `x-request-id` header. Use an unpredictable UUID or comparable high-entropy value made of 1–128 URL-safe letters, digits, `.`, `_`, `~`, or `-`, and never reuse one while a request is active; invalid handles fail locally in the typed clients or as `invalid_request` over HTTP, while concurrent reuse fails with `request_handle_conflict` (HTTP 409). `cancel_request()` is idempotent, may be called before the worker has issued its request at all, and returns the state of the handle (`cancelling`, `pending`, `cancelled`, `completed`) — call it again to see whether the request actually stopped. A stopped request fails with `request_cancelled` (HTTP 499), which is a withdrawal, not a failure to retry.

Cancellation is cooperative and process-local: LewLM stops at its next checkpoint, so work already inside one bounded backend call finishes, and only the instance named by `runtime_instance_id` on the record can act on the handle. When API-key authentication is enabled, the target and cancel calls must use the same credential. `x-lewlm-application-id` remains observability metadata and is never authority; in an open deployment, possession of the unpredictable handle is the only capability boundary. The full boundaries — batching, streaming, and sandboxed tool work — are in [the HTTP API reference](../reference/http-api.md).

## Tokenizer-aware counting

Estimating one token per four bytes is safe but wastes context. `POST /v1/tokenize/count` and `LewLMAppClient.count_tokens(...)` use the selected model's own tokenizer:

```python
result = client.count_tokens(text=long_text, model=model_id, max_tokens=3000)
if result.truncated:
    long_text = result.truncated_text     # cut at an exact token boundary
```

Re-counting `truncated_text` returns exactly `max_tokens`, so the boundary is reproducible rather than approximate.

## Correlation identifiers

Pass a correlation ID and LewLM threads it through execution metadata, events, and logs:

- header `x-lewlm-correlation-id` on any request, or `correlation_id` in the body of chat, responses, and document requests (the header wins)
- echoed back as a response header, so even an unparseable response is correlatable
- present as `metadata.correlation_id` on every execution response and `correlation_id` on every event

LewLM never generates one. An absent value stays `null`, so a caller can always tell its own identifier apart from a LewLM `request_id`.

## Tool calling

Declaring `tools` on a chat or responses request does two things: it lists the tools in the compiled prompt, **and** it tells the model the exact output shape LewLM's strict parser accepts. You do not need to spend your own `system_prompt` slot restating the format.

The accepted shapes — advertised to the model and enforced by the parser from the same constants in `lewlm/tool_call_contract.py`:

```json
{"name": "<tool name>", "arguments": {...}}
{"tool_call": {"name": "<tool name>", "arguments": {...}}}
{"tool_calls": [{"name": "<tool name>", "arguments": {...}}]}
```

- `name` is required and must match a declared tool exactly. **A reply containing only arguments is not a tool call** — it parses as `no_tool_calls`.
- `arguments` must be a JSON object satisfying that tool's `input_schema`; `input` is accepted as an alias.
- The object may stand alone or sit inside a ```` ```json ```` fenced block.

Results arrive on `tool_calls` with `status` of `no_tool_calls`, `parsed`, `partial`, or `failed`, plus an explicit `issues[]` entry for anything rejected. LewLM never silently repairs model output, so `partial` is reported as `partial` rather than flattened into success. `tool_calls` is `null` when the request declared no tools, so ordinary JSON output never produces spurious issues.

## Failure shape across HTTP and typed helpers

When a local-server request fails, LewLM keeps one machine-readable error shape:

- HTTP returns a top-level `error` object with `code`, `message`, and `details`
- `LewLMAppClient.from_http()` raises `LewLMAppClientHTTPError` with matching `code`, `status_code`, and `details`
- support-path-related failures can include fields such as `support_path`, `feature_class`, and `fallback_guidance`
- `backend_contract_violation` (502) means a backend returned a result LewLM will not silently repair — most often a bridge-backed rerank server returning a duplicate index, an out-of-range index, an unscored candidate, or a non-finite score
- both clients bound response bodies (`max_response_bytes`, default 128 MiB) and raise `LewLMAppClientResponseTooLargeError` rather than buffering without limit; error bodies are truncated to 8 KiB for diagnostics
- that ceiling is also settable per operation with `max_response_bytes_by_operation={"generate_document": …}`, keyed by the method names in `APP_CLIENT_OPERATIONS` / `ASYNC_CLIENT_OPERATIONS`, so a rendered artifact can have the room it needs without loosening the bound on every `health` poll; an unknown key raises rather than silently never applying, and the raised error names the operation whose limit was hit
- `examples/integration-bundle.json` publishes the full error-code catalog (`code`, `http_status`, `retryable`, `description`) so a host app can build its own switch without scraping LewLM source

That keeps embedded and local-server host apps aligned on the same fallback and support-path diagnostics instead of forcing callers to scrape raw HTTP text.

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
- a `scoring_policy` object naming and versioning the exact ranking rules that were applied

### Versioned scoring policy

Ranking is rerank-primary with embedding tie-breaking and a stable original-input-order fallback. That policy is now named and versioned so a result is reproducible:

| Field | Meaning |
| --- | --- |
| `name` / `version` | `rerank_primary_embedding_tiebreak`, versioned independently of the envelope |
| `primary_signal` | `rerank`, `embedding`, or `none` |
| `tie_break_signal` / `final_tie_break` | `embedding` then `original_order` |
| `normalization` | `cosine` when embeddings ran, else `none` |
| `missing_score_behaviour` | `rejected` — an unscored candidate fails the request rather than silently ranking last |
| `deduplication` | `chunk_id` |

Candidate identity is validated before any model work: duplicate `chunk_id` or `source_id` values are rejected with `configuration_error`. A rerank backend that returns a duplicate index, an out-of-range index, a non-finite score, or fails to score every candidate is rejected with `backend_contract_violation` instead of having its response silently repaired.

This keeps retrieval app-agnostic and reusable while still avoiding a LewLM-managed collection layer.

**Current boundary:** LewLM does not manage persistent vector storage, collection CRUD, delete-by-source flows, or app-owned memory policy here. The caller still owns candidate selection and persistence.

## Proving which build produced an artifact

`GET /v1/runtime` returns a `build` object so a caller can prove which implementation ran, rather than trusting a package version that an editable checkout, a patched wheel, and a release all report identically:

| Field | Meaning |
| --- | --- |
| `package_version` | installed LewLM version |
| `api_schema_version` | public contract shape, versioned independently of the package |
| `source_commit` / `source_dirty` | git commit of the running tree, and whether it had uncommitted changes |
| `distribution_digest` | stable digest of installed package metadata, for comparing two deployments |
| `install_kind` | `editable`, `installed`, or `unknown` |
| `release_build` | true only for a clean, non-editable install with no local modifications |

## Related references

- [Integration bundle reference](../reference/integration-bundle.md)
- [HTTP API reference](../reference/http-api.md)
- [Python API reference](../reference/python-api.md)
- [Tools and skills](tools-and-skills.md)
- [Sessions and events](sessions-and-events.md)
- [Documents](documents.md)
- [Chat and responses](chat-and-responses.md)
