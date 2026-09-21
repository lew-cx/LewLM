#!/usr/bin/env python3
"""Regenerate the generated parts of `examples/integration-bundle.json`.

    python scripts/export_integration_bundle.py            # rewrite schemas + errors + chap examples
    python scripts/export_integration_bundle.py --check    # exit 1 if schemas/errors are out of date

`schemas` and `errors` are derived from the Pydantic models and the error
catalog in this checkout, so they are never hand-edited. The `chap` section
holds exact payloads captured from a LewLM fixture server fronting the fake
engine (`lewlm.testing`): a streamed reply with its terminal usage chunk, a
native tool call, JSON output, an engine outage, a cancelled request, and an
interrupted stream. Volatile identifiers and timestamps are normalized so the
file only changes when a contract changes. The hand-written `surfaces` and
`notes` are left exactly as they are.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "examples" / "integration-bundle.json"
sys.path.insert(0, str(ROOT / "src"))


def build_schemas() -> dict[str, Any]:
    from pydantic import TypeAdapter

    from lewlm.api.schemas.chat import (
        ChatCompletionChunk,
        ChatCompletionRequest,
        ChatCompletionResponse,
        ResponseChunk,
        ResponseCreateRequest,
        ResponseCreateResponse,
    )
    from lewlm.api.schemas.documents import (
        DocumentGenerateRequest,
        DocumentGenerateResponse,
        DocumentIngestRequest,
        DocumentIngestResponse,
        DocumentTransformResponse,
    )
    from lewlm.api.schemas.multimodal import (
        EmbeddingCreateRequest,
        EmbeddingCreateResponse,
        RerankCreateRequest,
        RerankCreateResponse,
        RetrievalContextRequest,
        RetrievalContextResponse,
    )
    from lewlm.documents.skills.models import DocumentTransformRequest
    from lewlm.events.schema import StreamEvent

    return {
        "chat.request": ChatCompletionRequest.model_json_schema(),
        "chat.response": ChatCompletionResponse.model_json_schema(),
        "chat.stream": ChatCompletionChunk.model_json_schema(),
        "responses.request": ResponseCreateRequest.model_json_schema(),
        "responses.response": ResponseCreateResponse.model_json_schema(),
        "responses.stream": ResponseChunk.model_json_schema(),
        "embeddings.request": EmbeddingCreateRequest.model_json_schema(),
        "embeddings.response": EmbeddingCreateResponse.model_json_schema(),
        "retrieval.request": RetrievalContextRequest.model_json_schema(),
        "retrieval.response": RetrievalContextResponse.model_json_schema(),
        "rerank.request": RerankCreateRequest.model_json_schema(),
        "rerank.response": RerankCreateResponse.model_json_schema(),
        "documents.ingest.request": DocumentIngestRequest.model_json_schema(),
        "documents.ingest.response": DocumentIngestResponse.model_json_schema(),
        "documents.generate.request": DocumentGenerateRequest.model_json_schema(),
        "documents.generate.response": DocumentGenerateResponse.model_json_schema(),
        "documents.transform.request": TypeAdapter(DocumentTransformRequest).json_schema(),
        "documents.transform.response": DocumentTransformResponse.model_json_schema(),
        "events.stream": StreamEvent.model_json_schema(),
    }


def build_errors() -> list[dict[str, Any]]:
    from lewlm.core.errors import error_code_catalog

    return error_code_catalog()


_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_VOLATILE_KEYS = {
    "request_id": "req-example", "correlation_id": "conv-example", "runtime_instance_id": "runtime-example",
    "operation_id": "op-example", "profile_id": "profile-example", "id": "chatcmpl-example", "hostname": "host-example",
    "data_dir": "/path/to/lewlm/state", "database_path": "/path/to/lewlm/state/metadata.sqlite3",
    "base_url": "http://127.0.0.1:PORT/v1", "public_base_url": "http://127.0.0.1:PORT", "source_path": "external://fixture/fixture-chat",
}
_PATH_LIST_KEYS = {"models_dir": "/path/to/lewlm/models"}
_ZERO_KEYS = {"created", "process_id", "queue_milliseconds", "load_milliseconds", "execute_milliseconds", "total_milliseconds",
              "scheduler_wait_milliseconds", "queue_residency_milliseconds", "inventory_age_seconds", "elapsed_ms",
              "lewlm_ready_seconds", "residency_wait_milliseconds"}


def normalize(value: Any, key: str | None = None) -> Any:
    """Replace ids, timestamps, and timings so captured examples are stable."""

    if isinstance(value, dict):
        return {k: normalize(v, k) for k, v in value.items()}
    if isinstance(value, list):
        if key in _PATH_LIST_KEYS:
            return [_PATH_LIST_KEYS[key] for _ in value]
        return [normalize(item, key) for item in value]
    if key in _VOLATILE_KEYS and isinstance(value, str):
        return _VOLATILE_KEYS[key]
    if key in _ZERO_KEYS and isinstance(value, int | float) and not isinstance(value, bool):
        return 0
    if isinstance(value, str):
        if _UUID.match(value) or _HEX32.match(value):
            return "id-example"
        if _ISO.match(value):
            return "2026-01-01T00:00:00+00:00"
        if key == "model" or key == "model_id" or key == "resolved_model_id" or key == "requested_model_id":
            return "fixture-chat-fixture-example" if value.startswith("fixture-chat-") else value
    return value


def _sse_frames(response) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    buffer = ""
    for raw in response.iter_text():
        buffer += raw
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            for line in event.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        frames.append({"_done": True})
                    else:
                        frames.append(json.loads(data))
    return frames


def capture_chap_examples() -> dict[str, Any]:
    """Exact payloads from the fixture server, normalized. Requires `httpx`."""

    import httpx

    from lewlm.testing import FakeBackendFixture, FakeOpenAIEngine

    examples: dict[str, Any] = {}
    with FakeBackendFixture(engine=FakeOpenAIEngine(stream_delay_seconds=0.01)) as fixture:
        client = httpx.Client(base_url=fixture.base_url, timeout=60, headers={"x-lewlm-application-id": "chap"})
        model = fixture.model_id
        chat = lambda **body: client.post("/v1/chat/completions", json={"model": model, "max_tokens": 32, **body})  # noqa: E731

        examples["health"] = client.get("/v1/health").json()
        examples["runtime"] = client.get("/v1/runtime").json()
        models = client.get("/v1/models").json()
        examples["models.capability_availability"] = models["capability_availability"]
        examples["model.capabilities"] = client.get(f"/v1/models/{model}/capabilities").json()

        # Text, streamed: first content chunk and the terminal chunk with usage.
        with client.stream("POST", "/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "count"}], "max_tokens": 32, "stream": True}) as response:
            frames = _sse_frames(response)
        content_frames = [f for f in frames if any((c.get("delta") or {}).get("content") for c in f.get("choices", []))]
        terminal = [f for f in frames if any(c.get("finish_reason") for c in f.get("choices", []))]
        examples["chat.stream.text"] = {"first_content_chunk": content_frames[0], "terminal_chunk": terminal[-1], "ends_with": "[DONE]"}

        examples["chat.native_tool_call"] = chat(
            messages=[{"role": "user", "content": "What is the weather in Lisbon? Use the tool."}],
            tools=[{"name": "get_weather", "description": "Get the current weather for a city.", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
            tool_choice="auto",
        ).json()
        examples["chat.json_output"] = chat(
            messages=[{"role": "user", "content": "Give the city Paris and a population estimate as JSON."}],
            response_format={"type": "json_schema", "name": "city", "strict": True, "schema": {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}}, "required": ["city", "population"], "additionalProperties": False}},
        ).json()

        # Cancellation: cancel after the first chunk, keep the record and the terminal chunk.
        request_id = f"chap-{uuid.uuid4().hex[:8]}"
        cancel_record: dict[str, Any] = {}
        with client.stream("POST", "/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "Write a long story."}], "max_tokens": 256, "stream": True}, headers={"x-request-id": request_id}) as response:
            buffer = ""
            frames = []
            for raw in response.iter_text():
                buffer += raw
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.splitlines():
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            frames.append({"_done": True} if data == "[DONE]" else json.loads(data))
                if not cancel_record and any((c.get("delta") or {}).get("content") for f in frames if isinstance(f, dict) for c in f.get("choices", [])):
                    cancel_record = client.post(f"/v1/requests/{request_id}/cancel").json()
        final_record = client.post(f"/v1/requests/{request_id}/cancel").json()
        examples["chat.cancellation"] = {
            "cancel_response_after_first_chunk": cancel_record,
            "cancel_response_after_stream_ended": final_record,
            "terminal_chunk": next((f for f in reversed(frames) if isinstance(f, dict) and any(c.get("finish_reason") for c in f.get("choices", []))), None),
        }

        # Interrupted stream: engine dies after the first chunk.
        fixture.engine.die_mid_stream.clear()
        with client.stream("POST", "/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "Write a long story."}], "max_tokens": 256, "stream": True}) as response:
            buffer = ""
            frames = []
            for raw in response.iter_text():
                buffer += raw
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.splitlines():
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            frames.append({"_done": True} if data == "[DONE]" else json.loads(data))
                if len(frames) >= 1:
                    fixture.engine.die_mid_stream.set()
        fixture.engine.die_mid_stream.clear()
        examples["chat.stream.interrupted"] = {"terminal_chunk": next((f for f in reversed(frames) if isinstance(f, dict) and any(c.get("finish_reason") for c in f.get("choices", []))), None), "ends_with": "[DONE]"}

        # Engine outage: 503 naming the endpoint; health stays ok with the engine failed.
        with fixture.engine_stopped():
            client.post("/v1/models/scan", json={})
            outage = chat(messages=[{"role": "user", "content": "hi"}])
            examples["chat.unavailable_engine"] = {"http_status": outage.status_code, "body": outage.json()}
            examples["health.engine_down"] = client.get("/v1/health").json()
        client.post("/v1/models/scan", json={})
        examples["chat.model_not_found"] = {"http_status": 404, "body": client.post("/v1/chat/completions", json={"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]}).json()}

        # Responses surface: a reply cut off by `max_output_tokens` says so in
        # `finish_reason`, exactly as the chat surface does.
        fixture.engine.long_reply_words = 12
        examples["responses.truncated"] = client.post("/v1/responses", json={"model": model, "input": "Write a long story.", "max_output_tokens": 256}).json()
    return normalize(examples)


CHAP_FIELD_NOTES = {
    "identity": "One base URL. `x-request-id` (optional, caller-chosen) is echoed on every response and is the cancellation handle; `x-lewlm-correlation-id` is echoed and lands in `metadata.correlation_id`; `x-lewlm-application-id: chap` labels metrics and audit and grants nothing.",
    "service_vs_engine_vs_model": "`GET /v1/health.status` is this service. Engine reachability is `health.engines[].state` and `GET /v1/runtime.startup.engines[]` (cached inventory: advertised | stale | failed | unknown; no probe). Model warmth is `runtime.startup.warm_models[]` / `GET /v1/runtime/residencies`. A 200 from health never implies an engine is up or a model is warm.",
    "model_picker": "`GET /v1/models.capability_availability[]` gives, per model: `chat_ready`, `reason`, `endpoint_id`/`engine_profile`/`execution_locality` (null for packaged runtimes), and `engine_state` (`packaged`, or the endpoint's cached state). `GET /v1/models/{id}/capabilities.structured_output` predicts enforcement before a request is spent.",
    "chat_metadata": "`metadata.model` names `resolved_model_id`, `runtime_name`, `endpoint_id`, `engine_profile`, `execution_locality`; `metadata.routing.fallback_from_model_id`/`fallback_reason` are set only when an explicit fallback alias substituted the model before generation; `metadata.serving.runtime_adapter_kind` is `backend_native_batch` for engines that batch themselves.",
    "finish_reason": "Both surfaces publish why generation stopped from one vocabulary: chat in `choices[0].finish_reason`, responses in `finish_reason` on the sync body and on the terminal chunk (`done: true`). `length` means the reply hit the output limit and is truncated; `stop` means the model finished; `tool_calls` means it stopped to call a tool.",
    "usage": "`usage` is on the non-streaming body and on the terminal streaming chunk only; earlier chunks carry `usage: null`. `usage.measured` is false when counts were estimated. `usage.cached_tokens` is present only when the backend reported prefix-cache hits; absent means unknown.",
    "streaming": "SSE `data:` frames of `chat.stream` chunks, then `data: [DONE]`. Exactly one chunk carries a non-null `finish_reason` (`stop`, `length`, `tool_calls`, `cancelled`, or `error`). Native tool-call fragments arrive as `delta.tool_calls[]` with `index`; arguments are concatenated per index.",
    "incomplete_stream": "If generation fails after output has started, the stream still returns HTTP 200: the last chunk has `finish_reason: \"error\"` and an `error` envelope (`code`, `message`, redacted `details`, `partial_output: true`), followed by `[DONE]`. Delivered text stands; LewLM never replays. A cancelled stream ends with `finish_reason: \"cancelled\"`.",
    "errors": "Every non-2xx body is `{\"error\": {code, message, details}}`; codes and fixed statuses are in `errors`. An unreachable engine behind an explicitly requested model is `503 runtime_unavailable` with `details.endpoint_id`; an unknown model is `404 model_not_found`. Raw backend payloads and credentials are never included.",
    "absent_fields": "Optional fields are omitted or null rather than fabricated: `usage.cached_tokens`, `structured_output` (only when a contract was requested), `tool_calls` (only when tools were offered), `prompt_trace` (only with `include_prompt_trace`), `metadata.routing.fallback_*`, `startup.engines[].first_advertised_at` (until first inventory read).",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Verify schemas and errors match the checkout; do not write")
    parser.add_argument("--skip-capture", action="store_true", help="Do not run the fixture; keep the existing chap examples")
    parser.add_argument("--bundle", default=str(BUNDLE_PATH), help="Bundle path (default: examples/integration-bundle.json)")
    args = parser.parse_args(argv)
    bundle_path = Path(args.bundle)

    import pydantic
    import pydantic_core

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    schemas = build_schemas()
    errors = build_errors()
    generated_with = {"pydantic": pydantic.VERSION, "pydantic_core": pydantic_core.__version__}
    if args.check:
        drift = [name for name, schema in schemas.items() if bundle.get("schemas", {}).get(name) != schema]
        if bundle.get("errors") != errors:
            drift.append("errors")
        recorded = bundle.get("generated_with") or {}
        if drift:
            print("integration bundle is out of date for: " + ", ".join(drift), file=sys.stderr)
            if recorded and recorded != generated_with:
                print(f"note: the bundle was generated with {recorded}, this environment has {generated_with}; "
                      "regenerate under the pinned versions before deciding it is a contract change", file=sys.stderr)
            print("run: python scripts/export_integration_bundle.py", file=sys.stderr)
            return 1
        print(f"integration bundle schemas and errors match the checkout (pydantic {generated_with['pydantic']})")
        return 0

    bundle["schemas"] = schemas
    bundle["errors"] = errors
    bundle["generated_with"] = generated_with
    chap = bundle.get("chap") or {}
    chap["generated_by"] = "scripts/export_integration_bundle.py"
    chap["smoke_script"] = "examples/chap_backend_smoke.py"
    chap["fixture"] = "python -m lewlm.testing.fake_backend --port 8080"
    chap["field_notes"] = CHAP_FIELD_NOTES
    if not args.skip_capture:
        chap["examples"] = capture_chap_examples()
    bundle["chap"] = chap
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {bundle_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
