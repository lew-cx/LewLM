from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from lewlm.config.endpoints import ExternalEndpoint
from lewlm.core.contracts import GenerateMessage, GenerateRequest, SamplingControls
from lewlm.core.errors import RuntimeUnavailableError
from lewlm.runtime.adapters.http_transport import AsyncBridgeTransport, _SSEDecoder
from lewlm.runtime.adapters.openai_compatible import LocalOpenAICompatibleAdapterRuntime
from lewlm.config.settings import LewLMSettings
from lewlm.structured_output import JSONSchemaResponseFormat


def _endpoint(*, api_key_env: str | None = None) -> ExternalEndpoint:
    return ExternalEndpoint(
        endpoint_id="test",
        base_url="http://127.0.0.1:18080/v1",
        api_key_env=api_key_env,
        connect_timeout_seconds=0.1,
        read_timeout_seconds=0.1,
        pool_timeout_seconds=0.1,
    )


def test_sse_decoder_handles_fragmented_utf8_crlf_comments_and_multiline_data() -> None:
    decoder = _SSEDecoder()
    payload = 'data: {"text":"café"}\r\ndata: second\r\n: ignored\r\n\r\n'.encode()
    events = []
    for byte in payload:
        events.extend(decoder.feed(bytes([byte])))
    events.extend(decoder.finish())

    assert [event.data for event in events] == ['{"text":"café"}\nsecond']


class _FragmentStream(httpx.AsyncByteStream):
    def __init__(self, fragments: list[bytes]) -> None:
        self.fragments = fragments
        self.closed = False

    async def __aiter__(self):
        for fragment in self.fragments:
            yield fragment

    async def aclose(self) -> None:
        self.closed = True


def test_transport_stream_closes_upstream_when_consumer_cancels() -> None:
    async def run() -> None:
        body = _FragmentStream([b'data: {"choices":[]}\n\n', b"data: [DONE]\n\n"])

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

        transport = AsyncBridgeTransport(endpoint=_endpoint(), runtime_name="test-runtime")
        await transport._client.aclose()
        transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(handler),
            trust_env=False,
            follow_redirects=False,
        )
        stream = transport.stream_sse("POST", "/v1/chat/completions", payload={"stream": True})
        event = await anext(stream)
        assert event.data == '{"choices":[]}'
        await stream.aclose()
        assert body.closed is True
        await transport.aclose()

    asyncio.run(run())


def test_cancelled_stream_does_not_poison_the_next_stream() -> None:
    async def run() -> None:
        streams = [
            _FragmentStream([b"data: first\n\n", b"data: [DONE]\n\n"]),
            _FragmentStream([b"data: second\n\n", b"data: [DONE]\n\n"]),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=streams.pop(0))

        transport = AsyncBridgeTransport(endpoint=_endpoint(), runtime_name="test-runtime")
        await transport._client.aclose()
        transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(handler),
            trust_env=False,
        )
        first = transport.stream_sse("POST", "/v1/chat/completions", payload={"stream": True})
        assert (await anext(first)).data == "first"
        await first.aclose()
        second = [event.data async for event in transport.stream_sse(
            "POST",
            "/v1/chat/completions",
            payload={"stream": True},
        )]
        assert second == ["second", "[DONE]"]
        await transport.aclose()

    asyncio.run(run())


class _BrokenBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"partial": '
        raise httpx.ReadError("connection reset mid-body")


@pytest.mark.parametrize(
    ("answers", "marks_unreachable"),
    [(False, True), (True, False)],
    ids=["dropped-before-answer", "dropped-mid-body"],
)
def test_nonstreaming_request_marks_unreachable_only_before_the_engine_answers(answers: bool, marks_unreachable: bool) -> None:
    async def run() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if not answers:
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
            return httpx.Response(200, headers={"content-type": "application/json"}, stream=_BrokenBody())

        reasons: list[str] = []
        transport = AsyncBridgeTransport(endpoint=_endpoint(), runtime_name="test-runtime", on_unreachable=reasons.append)
        await transport._client.aclose()
        transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(handler),
            trust_env=False,
            follow_redirects=False,
        )
        with pytest.raises(RuntimeUnavailableError):
            await transport.request_json("POST", "/v1/chat/completions", payload={})
        assert bool(reasons) is marks_unreachable
        await transport.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, "authentication"), (400, "invalid_request"), (404, "model_not_found"), (429, "rate_limited"), (503, "unavailable")],
)
def test_transport_maps_http_statuses(status: int, kind: str) -> None:
    async def run() -> None:
        transport = AsyncBridgeTransport(endpoint=_endpoint(), runtime_name="test-runtime")
        await transport._client.aclose()
        transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(lambda request: httpx.Response(status, text="failure")),
            trust_env=False,
            follow_redirects=False,
        )
        with pytest.raises(RuntimeUnavailableError) as exc_info:
            await transport.request_json("GET", "/v1/models")
        assert exc_info.value.details["error_kind"] == kind
        await transport.aclose()

    asyncio.run(run())


def test_transport_rejects_redirects_and_does_not_expose_api_key(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setenv("LEWLM_TEST_KEY", "super-secret")
        seen_headers: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.append(request.headers)
            return httpx.Response(
                307,
                headers={"location": "http://127.0.0.1:9999/v1/models"},
                text="super-secret",
            )

        transport = AsyncBridgeTransport(
            endpoint=_endpoint(api_key_env="LEWLM_TEST_KEY"),
            runtime_name="test-runtime",
        )
        await transport._client.aclose()
        transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(handler),
            trust_env=False,
            follow_redirects=False,
            headers={"Authorization": "Bearer super-secret"},
        )
        with pytest.raises(RuntimeUnavailableError) as exc_info:
            await transport.request_json("GET", "/v1/models")
        assert exc_info.value.details["status_code"] == 307
        assert exc_info.value.details["error_kind"] == "redirect"
        assert "super-secret" not in str(exc_info.value.details)
        assert seen_headers[0]["authorization"] == "Bearer super-secret"
        await transport.aclose()

    asyncio.run(run())


def test_missing_backend_credential_is_reported_without_breaking_construction(monkeypatch) -> None:
    monkeypatch.delenv("LEWLM_TEST_KEY", raising=False)
    transport = AsyncBridgeTransport(
        endpoint=_endpoint(api_key_env="LEWLM_TEST_KEY"),
        runtime_name="test-runtime",
    )

    async def run() -> None:
        with pytest.raises(RuntimeUnavailableError) as exc_info:
            await transport.request_json("GET", "/v1/models")
        assert exc_info.value.details["error_kind"] == "authentication"
        assert exc_info.value.details["api_key_env"] == "LEWLM_TEST_KEY"
        await transport.aclose()

    asyncio.run(run())


def test_transport_maps_timeouts_and_malformed_json() -> None:
    async def run_timeout() -> None:
        def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow upstream", request=request)

        transport = AsyncBridgeTransport(endpoint=_endpoint(), runtime_name="test-runtime")
        await transport._client.aclose()
        transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(timeout_handler),
            trust_env=False,
        )
        with pytest.raises(RuntimeUnavailableError) as exc_info:
            await transport.request_json("GET", "/v1/models")
        assert exc_info.value.details["error_kind"] == "timeout"
        await transport.aclose()

    async def run_malformed() -> None:
        transport = AsyncBridgeTransport(endpoint=_endpoint(), runtime_name="test-runtime")
        await transport._client.aclose()
        transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"not-json")),
            trust_env=False,
        )
        with pytest.raises(RuntimeUnavailableError) as exc_info:
            await transport.request_json("GET", "/v1/models")
        assert exc_info.value.details["error_kind"] == "malformed_response"
        await transport.aclose()

    asyncio.run(run_timeout())
    asyncio.run(run_malformed())


def test_bridge_payload_preserves_sampling_tools_and_json_schema_without_duplicate_prompts(tmp_path) -> None:
    runtime = LocalOpenAICompatibleAdapterRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            external_accelerator_enabled=True,
            external_accelerator_base_url="http://127.0.0.1:18080",
            external_accelerator_profile="vllm_local",
        ),
    )
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    request = GenerateRequest(
        model_id="demo",
        messages=[
            GenerateMessage(role="system", content="Declared tools:\n- lookup"),
            GenerateMessage(role="system", content="To call a tool, reply with a JSON object in one of these shapes and nothing else:"),
            GenerateMessage(role="system", content="Structured output contract:\nReturn JSON"),
            GenerateMessage(role="user", content="hello"),
        ],
        sampling=SamplingControls(
            top_p=0.9,
            top_k=20,
            min_p=0.05,
            repetition_penalty=1.1,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            seed=7,
            stop=["END"],
        ),
        structured_output=JSONSchemaResponseFormat(name="answer", schema=schema),
        metadata={
            "bridge_tools": [
                {"name": "lookup", "description": "Look up a value", "input_schema": {"type": "object"}},
            ],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        },
    )

    payload = runtime._chat_payload(remote_model_id="upstream", request=request, stream=True)

    for name in ("top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty", "seed", "stop"):
        assert name in payload
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert payload["response_format"]["json_schema"]["schema"] == schema
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == [{"type": "text", "text": "hello"}]
    assert request.metadata["sampling_controls"]["unsupported"] == []
    asyncio.run(runtime.aclose())


def test_bridge_payload_keeps_unset_optional_fields_absent(tmp_path) -> None:
    runtime = LocalOpenAICompatibleAdapterRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            external_accelerator_enabled=True,
            external_accelerator_base_url="http://127.0.0.1:18080",
        ),
    )
    payload = runtime._chat_payload(
        remote_model_id="upstream",
        request=GenerateRequest(
            model_id="demo",
            messages=[GenerateMessage(role="user", content="hello")],
        ),
        stream=False,
    )

    for field in ("top_p", "top_k", "min_p", "seed", "stop", "tools", "tool_choice", "response_format", "stream_options"):
        assert field not in payload
    asyncio.run(runtime.aclose())


def test_bridge_stream_accumulates_tool_fragments_usage_and_finish_reason(tmp_path) -> None:
    async def run() -> None:
        runtime = LocalOpenAICompatibleAdapterRuntime(
            settings=LewLMSettings(
                data_dir=tmp_path / "state",
                external_accelerator_enabled=True,
                external_accelerator_base_url="http://127.0.0.1:18080",
            ),
        )
        frames = [
            {"choices": [{"delta": {"reasoning_content": "checking"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "look", "arguments": '{"q"'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "up", "arguments": ':"x"}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
        ]
        fragments = [f"data: {json.dumps(frame)}\r\n\r\n".encode() for frame in frames]
        fragments.append(b"data: [DONE]\r\n\r\n")
        await runtime._transport._client.aclose()
        runtime._transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=_FragmentStream(fragments), headers={"content-type": "text/event-stream"}),
            ),
            trust_env=False,
            follow_redirects=False,
        )

        events = [event async for event in runtime._stream_chat_completion({"stream": True})]
        assert "".join(event.reasoning or "" for event in events) == "checking"
        assert "".join(event.tool_call.name or "" for event in events if event.tool_call is not None) == "lookup"
        assert next(event.usage for event in events if event.usage) == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        assert next(event.finish_reason for event in events if event.finish_reason) == "tool_calls"
        await runtime.aclose()

    asyncio.run(run())


def test_bridge_stream_rejects_premature_eof(tmp_path) -> None:
    async def run() -> None:
        runtime = LocalOpenAICompatibleAdapterRuntime(
            settings=LewLMSettings(
                data_dir=tmp_path / "state",
                external_accelerator_enabled=True,
                external_accelerator_base_url="http://127.0.0.1:18080",
            ),
        )
        await runtime._transport._client.aclose()
        runtime._transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    stream=_FragmentStream([b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n']),
                ),
            ),
            trust_env=False,
            follow_redirects=False,
        )
        with pytest.raises(RuntimeUnavailableError, match=r"before the `\[DONE\]` marker"):
            _ = [event async for event in runtime._stream_chat_completion({"stream": True})]
        await runtime.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("status", "kind"),
    [(307, "redirect"), (401, "authentication"), (400, "invalid_request"),
     (404, "model_not_found"), (429, "rate_limited"), (503, "unavailable")],
)
def test_stream_http_failure_preserves_status_and_closes_unread_body(status, kind):
    async def run():
        body = _FragmentStream([b'private upstream diagnostic'])
        transport = AsyncBridgeTransport(endpoint=_endpoint(), runtime_name="test-runtime")
        await transport._client.aclose()
        transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(lambda request: httpx.Response(status, stream=body)),
        )
        try:
            with pytest.raises(RuntimeUnavailableError) as caught:
                await anext(transport.stream_sse("POST", "/v1/chat/completions", payload={}))
            assert caught.value.details["status_code"] == status
            assert caught.value.details["error_kind"] == kind
            assert "private upstream diagnostic" not in str(caught.value.details)
            assert body.closed
        finally:
            await transport.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("frame,kind", [
    ({"error": {"message": "private diagnostic"}}, "upstream_error"),
    ({"choices": [None]}, "malformed_stream"),
    ({"choices": [{"delta": None}]}, "malformed_stream"),
    ({"choices": "invalid"}, "malformed_stream"),
])
def test_bridge_rejects_invalid_frames_and_closes_transport(tmp_path, frame, kind):
    async def run():
        runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(
            data_dir=tmp_path / "state", external_accelerator_enabled=True,
            external_accelerator_base_url="http://127.0.0.1:18080",
        ))
        body = _FragmentStream([f"data: {json.dumps(frame)}\n\n".encode(), b"data: [DONE]\n\n"])
        await runtime._transport._client.aclose()
        runtime._transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=body)),
        )
        try:
            with pytest.raises(RuntimeUnavailableError) as caught:
                _ = [event async for event in runtime._stream_chat_completion({})]
            assert caught.value.details["error_kind"] == kind
            assert "private diagnostic" not in str(caught.value.details)
            assert body.closed
        finally:
            await runtime.aclose()

    asyncio.run(run())


def test_bridge_done_closes_transport_before_returning(tmp_path):
    async def run():
        runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(
            data_dir=tmp_path / "state", external_accelerator_enabled=True,
            external_accelerator_base_url="http://127.0.0.1:18080",
        ))
        body = _FragmentStream([b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
                                b"data: [DONE]\n\n"])
        await runtime._transport._client.aclose()
        runtime._transport._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:18080",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=body)),
        )
        try:
            events = [event async for event in runtime._stream_chat_completion({})]
            assert events[-1].finish_reason == "stop"
            assert body.closed
        finally:
            await runtime.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("choices", [[None], [{"message": None}], [{"message": "invalid"}]])
def test_bridge_rejects_malformed_nonstreaming_choices(tmp_path, monkeypatch, choices):
    async def run():
        runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(
            data_dir=tmp_path / "state", external_accelerator_enabled=True,
            external_accelerator_base_url="http://127.0.0.1:18080",
        ))
        monkeypatch.setattr(runtime, "_require_remote_model_id", lambda manifest: "demo")

        async def response(*args):
            return {"choices": choices}

        monkeypatch.setattr(runtime, "_request_json_async", response)
        try:
            with pytest.raises(RuntimeUnavailableError, match="invalid chat completion"):
                await runtime._generate_with_manifest(None, GenerateRequest(
                    model_id="demo", messages=[GenerateMessage(role="user", content="hello")],
                ))
        finally:
            await runtime.aclose()

    asyncio.run(run())
