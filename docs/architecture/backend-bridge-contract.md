# Backend bridge contract

LewLM's external-engine bridge is an optional local transport. Native MLX,
llama.cpp, ONNX, and Ollama selection remain available when a bridge is
disabled or cannot serve a compatible registered model.

Each configured endpoint owns one reusable asynchronous HTTP connection pool.
The pool has bounded total and keep-alive connections, ignores ambient proxy
settings, does not follow redirects, and is closed with the LewLM service
container. Endpoint URLs may be written as a server root or with a trailing
`/v1`; LewLM canonicalizes both forms before it builds API paths. Only loopback
HTTP(S) endpoints are accepted.

Set `api_key_env` on an endpoint to name an environment variable containing
that backend's credential. This credential is separate from caller
authentication. LewLM sends it only to the configured endpoint and redacts its
value from transport errors and endpoint snapshots. A missing variable makes
that endpoint unavailable without preventing LewLM from starting.

For chat requests, the bridge preserves supported sampling controls, stop
sequences, native tool declarations and choices, and structured output
contracts. Unsupported profile-specific controls are listed in
`sampling_controls.unsupported`; unset optional fields stay absent. When LewLM sends native
tool or JSON-schema fields, it removes the equivalent compiler-generated system
scaffolding so the upstream model receives each instruction once. JSON schema
uses the OpenAI `response_format` shape. Grammar forwarding is currently
enabled only for the vLLM, vLLM-MLX, and SGLang profiles through their
`structured_outputs` request field; other profiles retain prompt-guided grammar
fallback.

Capable runtimes can emit structured stream events containing content,
reasoning, tool-call fragments, usage, and a finish reason. Existing string
stream runtimes are adapted automatically. The chat endpoint exposes native
tool fragments on `choices[].delta.tool_calls`; the responses endpoint exposes
them as `tool_call_delta`. LewLM emits those fragments once and retains the
assembled call internally for its existing validation path. Usage and prompt
trace remain terminal metadata. Reasoning continues to follow the configured
visibility policy.

The SSE reader accepts fragmented UTF-8, LF or CRLF framing, comments,
multiline `data` fields, empty-choice usage events, and `[DONE]`. EOF before
`[DONE]`, malformed UTF-8/JSON, and unexpected payloads are errors. Closing a
client stream closes the upstream response immediately; the orchestrator's
existing `finally` path then releases admission and residency state. This
proves transport cancellation only. Engine-specific GPU work cancellation must
be recorded as unverified until a real backend demonstrates it.

Bridge errors retain a machine-readable `error_kind` in the normal LewLM error
details: `authentication`, `invalid_request`, `rate_limited`,
`model_not_found`, `redirect`, `unavailable`, `timeout`, `malformed_response`,
`malformed_stream`, or `premature_eof`. Generation is never retried after
submission or after a streamed fragment.
