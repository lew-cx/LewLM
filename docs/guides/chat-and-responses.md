# Chat and responses

LewLM exposes two primary text-generation APIs:

- `POST /v1/chat/completions`
- `POST /v1/responses`

The CLI `lewlm chat` and the Python `LewLM.chat_sync()` / `LewLM.stream_chat()` methods sit on the same core orchestration path.

## Core request controls

Both chat-style APIs support:

- optional `model`
- optional `session_id`
- optional `citation_context` source/chunk packages for grounded answers
- `max_tokens` or `max_output_tokens`
- `temperature`
- `apply_serving_profile`
- `stream`
- `reasoning_visibility`
- prompt overrides and prompt-trace inspection

## Structured output contracts

Both routes also accept a host-app-facing `response_format` contract.

- `type: "json_schema"` records a JSON-schema request
- `type: "grammar"` records a grammar request
- `structured_output` on the response reports whether LewLM enforced the contract at decode time, whether it had to fall back to prompt guidance, and what JSON/grammar validation state LewLM observed

On Linux and Windows, the packaged llama.cpp/GGUF path is the first-class non-Apple route that can report `enforcement: "decode_time"` for structured output. Non-enforcing routes keep the same public shape, but they report `enforcement: "prompt_guided"` with `fallback_used: true` instead of implying decoder enforcement.

When the caller is using local file-backed prompt assets, `response_format_path` can load that same contract from disk. Legacy `output_schema` and `output_schema_path` are still accepted for prompt-oriented JSON-schema input, but `response_format` is the stable public contract for host applications.

## Message content parts

Chat and responses inputs can contain:

- text parts
- image parts
- file parts
- audio parts

File-like parts can reference either:

- a validated local `path`
- an uploaded multipart file via `upload_name`

## Citation-aware grounding

Both routes can optionally accept a `citation_context` object containing the same `sources[]` and `chunks[]` packages returned by document ingest.

When the selected model emits LewLM citation markers for those records, LewLM strips the markers back out of the visible text and returns machine-readable `citations[]` alongside the normal chat or responses payload. Each citation entry lines up with ingest packaging fields such as:

- `source_id`
- `chunk_id`
- `section_id`
- `source_label`
- `section_label`

That keeps the backend app-agnostic: LewLM handles the stable reference packaging, while the host application still decides how to render links, chips, tooltips, or inline footnotes.

## Streaming behavior

When `stream=true`, LewLM returns `text/event-stream` responses:

- chat completions emit chat chunk objects
- responses emit response chunk objects
- streams include keep-alive heartbeats
- session persistence can be finalized when the stream completes

## Sessions

If you pass `session_id`, LewLM merges the new request with persisted context according to the session policy.

Session-aware surfaces include:

- API requests with `session_id`
- CLI session management commands
- Python session helpers such as `create_session()`, `export_session()`, and `delete_session()`

## Reasoning visibility

LewLM models reasoning output with three visibility levels:

- `hidden`
- `summarized`
- `raw_model_emitted`

Support depends on the selected runtime and model behavior.

## Response examples

### HTTP chat

```json
{
  "messages": [
    {"role": "user", "content": "Summarize the local runtime state with grounding."}
  ],
  "response_format": {
    "type": "json_schema",
    "name": "runtime_summary",
    "schema": {
      "type": "object",
      "properties": {
        "summary": {"type": "string"}
      },
      "required": ["summary"],
      "additionalProperties": false
    }
  },
  "citation_context": {
    "sources": [
      {
        "source_id": "src-runtime",
        "path": "/tmp/runtime.md",
        "source_type": "markdown",
        "source_name": "runtime.md",
        "source_label": "Runtime Notes",
        "metadata": {}
      }
    ],
    "chunks": [
      {
        "chunk_id": "src-runtime-sec-0001-chunk-0001",
        "text": "LewLM exposes chat, responses, embeddings, rerank, and document workflows through one local backend contract.",
        "source_id": "src-runtime",
        "section_id": "src-runtime-sec-0001",
        "source_label": "Runtime Notes",
        "section_label": "Runtime Notes / Summary",
        "metadata": {}
      }
    ]
  },
  "stream": false
}
```

### Python

```python
from lewlm import LewLM

with LewLM() as lewlm:
    result = lewlm.chat_sync(prompt="Summarize the local runtime state.")
    print(result.response.output_text)
    print(result.structured_output.validation.state if result.structured_output else "text")
```

## Related pages

- [Prompting and MCP-style tools](prompting-and-mcp-tools.md)
- [Sessions and events](sessions-and-events.md)
- [HTTP API reference](../reference/http-api.md)
