"""Coverage for the async client and the client-side response bounds."""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from lewlm.app_helpers import (
    APP_CLIENT_OPERATIONS,
    MAX_ERROR_BODY_CHARACTERS,
    LewLMAppClientHTTPError,
    LewLMAppClientResponseTooLargeError,
    _HttpAppClientBackend,
    _read_bounded,
    _truncated_error_body,
    validate_operation_limits,
)
from lewlm.async_helpers import ASYNC_CLIENT_OPERATIONS, LewLMAsyncClient
from lewlm.api.schemas.chat import ChatCompletionRequest, ChatMessage
from lewlm.api.schemas.multimodal import TokenCountRequest


# --- bounded reads on the synchronous client ---------------------------------


class _FakeStream:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, size: int = -1) -> bytes:
        return self._payload if size < 0 else self._payload[:size]


def test_bounded_read_returns_a_body_within_the_limit() -> None:
    assert _read_bounded(_FakeStream(b"hello"), limit=16, url="http://x") == b"hello"


def test_bounded_read_accepts_a_body_exactly_at_the_limit() -> None:
    assert _read_bounded(_FakeStream(b"12345"), limit=5, url="http://x") == b"12345"


def test_bounded_read_refuses_an_oversized_body() -> None:
    with pytest.raises(LewLMAppClientResponseTooLargeError) as exc_info:
        _read_bounded(_FakeStream(b"x" * 100), limit=10, url="http://x/y")
    assert exc_info.value.limit_bytes == 10
    assert exc_info.value.code == "response_too_large"


def test_error_bodies_are_truncated_not_retained_whole() -> None:
    body = "e" * (MAX_ERROR_BODY_CHARACTERS + 500)
    truncated = _truncated_error_body(body)
    assert len(truncated) < len(body)
    assert "truncated 500 characters" in truncated


def test_short_error_bodies_pass_through_unchanged() -> None:
    assert _truncated_error_body("boom") == "boom"
    assert _truncated_error_body(None) is None


# --- async client ------------------------------------------------------------


def _json_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def test_async_client_sends_identity_headers() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return _json_response({"status": "ok"})

    async with LewLMAsyncClient(
        "http://lewlm.test",
        api_key="secret",
        application_id="document-generator",
        client_instance_id="instance-1",
        authorized_actions=["document_generate"],
        correlation_id="trace-1",
        transport=_transport(handler),
    ) as client:
        await client._request_raw("GET", "/v1/health")

    assert seen["x-api-key"] == "secret"
    assert seen["x-lewlm-application-id"] == "document-generator"
    assert seen["x-lewlm-client-instance-id"] == "instance-1"
    assert seen["x-lewlm-authorized-actions"] == "document_generate"
    assert seen["x-lewlm-correlation-id"] == "trace-1"


async def test_async_client_parses_typed_responses() -> None:
    payload = {
        "request_id": "tok-1",
        "created": 1,
        "model": "m",
        "token_count": 7,
        "character_count": 20,
        "truncated": False,
        "routing": {
            "model_id": "m",
            "runtime_name": "fake",
            "runtime_affinity": "mlx_text",
            "reason": "test",
        },
        "metadata": {
            "request_id": "tok-1",
            "created": 1,
            "routing": {"kind": "model_router"},
        },
    }

    async with LewLMAsyncClient(
        "http://lewlm.test",
        transport=_transport(lambda request: _json_response(payload)),
    ) as client:
        result = await client.count_tokens(text="hello")

    assert result.token_count == 7
    assert result.model == "m"


async def test_async_client_raises_a_typed_error_envelope() -> None:
    error_payload = {
        "error": {
            "code": "model_load_failed",
            "message": "Model type gemma4 not supported.",
            "details": {"architecture_family": "gemma4"},
        },
    }

    async with LewLMAsyncClient(
        "http://lewlm.test",
        transport=_transport(lambda request: _json_response(error_payload, status_code=503)),
    ) as client:
        with pytest.raises(LewLMAppClientHTTPError) as exc_info:
            await client.health()

    assert exc_info.value.code == "model_load_failed"
    assert exc_info.value.status_code == 503


async def test_async_client_refuses_an_oversized_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 5000)

    async with LewLMAsyncClient(
        "http://lewlm.test",
        max_response_bytes=1000,
        transport=_transport(handler),
    ) as client:
        with pytest.raises(LewLMAppClientResponseTooLargeError):
            await client._request_raw("GET", "/v1/health")


async def test_async_client_propagates_cancellation_to_the_request() -> None:
    started = asyncio.Event()
    cancelled_inside_request = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal cancelled_inside_request
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            # The in-flight request itself is cancelled, not merely a waiter.
            cancelled_inside_request = True
            raise
        return _json_response({"status": "ok"})

    async with LewLMAsyncClient(
        "http://lewlm.test",
        transport=httpx.MockTransport(handler),
    ) as client:
        task = asyncio.create_task(client.health())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert cancelled_inside_request is True


async def test_async_client_reuses_one_connection_pool() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _json_response({"status": "ok"})

    client = LewLMAsyncClient("http://lewlm.test", transport=_transport(handler))
    try:
        await asyncio.gather(*(client._request_raw("GET", "/v1/health") for _ in range(5)))
    finally:
        await client.aclose()

    assert call_count == 5
    assert client._client.is_closed


async def test_aclose_is_idempotent() -> None:
    client = LewLMAsyncClient("http://lewlm.test", transport=_transport(lambda r: _json_response({})))
    await client.aclose()
    await client.aclose()


async def test_with_correlation_id_shares_the_pool_and_restamps_the_header() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-lewlm-correlation-id"))
        return _json_response({"status": "ok"})

    async with LewLMAsyncClient(
        "http://lewlm.test",
        correlation_id="base-trace",
        transport=_transport(handler),
    ) as client:
        scoped = client.with_correlation_id("per-request-trace")
        await client._request_raw("GET", "/v1/health")
        await scoped._request_raw("GET", "/v1/health")
        assert scoped._client is client._client

    assert seen == ["base-trace", "per-request-trace"]


async def test_async_client_refuses_streaming_through_the_unary_helper() -> None:
    async with LewLMAsyncClient(
        "http://lewlm.test",
        transport=_transport(lambda r: _json_response({})),
    ) as client:
        request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")], stream=True)
        with pytest.raises(ValueError, match="does not support stream=True"):
            await client.chat_completion(request)


async def test_async_client_streams_chat_deltas() -> None:
    frames = b'data: {"delta": "a"}\n\ndata: {"delta": "b"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=frames, headers={"content-type": "text/event-stream"})

    async with LewLMAsyncClient(
        "http://lewlm.test",
        transport=_transport(handler),
    ) as client:
        request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        chunks = [chunk async for chunk in client.stream_chat_completion(request)]

    assert [json.loads(chunk)["delta"] for chunk in chunks] == ["a", "b"]


async def test_streaming_surfaces_an_error_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"error": {"code": "runtime_unavailable", "message": "no runtime"}}, status_code=503)

    async with LewLMAsyncClient("http://lewlm.test", transport=_transport(handler)) as client:
        request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(LewLMAppClientHTTPError) as exc_info:
            [chunk async for chunk in client.stream_chat_completion(request)]

    assert exc_info.value.code == "runtime_unavailable"


async def test_per_operation_timeout_overrides_the_client_default() -> None:
    seen_timeouts: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_timeouts.append(request.extensions.get("timeout"))
        return _json_response({"status": "ok"})

    async with LewLMAsyncClient(
        "http://lewlm.test",
        timeout_seconds=30.0,
        transport=_transport(handler),
    ) as client:
        await client._request_raw("GET", "/v1/health")
        await client._request_raw("GET", "/v1/health", timeout_seconds=1.5)

    assert seen_timeouts[0]["read"] == 30.0
    assert seen_timeouts[1]["read"] == 1.5


def test_upload_source_helper_computes_the_digest() -> None:
    import hashlib

    body = b"# Title\n\nBody.\n"
    source = LewLMAsyncClient.upload_source("s1", body, file_name="doc.md", media_type="text/markdown")

    assert source.source_id == "s1"
    assert source.expected_sha256 == hashlib.sha256(body).hexdigest()
    assert base64.b64decode(source.content_base64) == body


def test_upload_source_helper_can_skip_verification() -> None:
    source = LewLMAsyncClient.upload_source("s1", b"body", file_name="doc.md", verify=False)
    assert source.expected_sha256 is None


# --- per-operation response bounds -------------------------------------------


def test_operation_limits_reject_a_name_that_is_not_an_operation() -> None:
    """A mistyped key would otherwise be accepted and simply never apply."""

    with pytest.raises(ValueError) as exc_info:
        validate_operation_limits({"generate_docs": 10})
    assert "generate_docs" in str(exc_info.value)

    with pytest.raises(ValueError):
        validate_operation_limits({"generate_document": 0})

    assert validate_operation_limits(None) == {}
    assert validate_operation_limits({"generate_document": 10}) == {"generate_document": 10}


def test_every_bounded_operation_is_addressable_on_both_clients() -> None:
    assert "generate_document" in APP_CLIENT_OPERATIONS
    assert "health" in APP_CLIENT_OPERATIONS
    # The async surface is a subset of the sync one plus nothing new.
    assert set(ASYNC_CLIENT_OPERATIONS) <= set(APP_CLIENT_OPERATIONS)


async def test_async_client_applies_a_per_operation_limit_over_the_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 5000)

    async with LewLMAsyncClient(
        "http://lewlm.test",
        max_response_bytes=1000,
        max_response_bytes_by_operation={"generate_document": 10_000},
        transport=_transport(handler),
    ) as client:
        # The raised operation-specific ceiling admits a body the default refuses.
        raw = await client._send(
            "POST",
            "/v1/documents/generate",
            operation="generate_document",
            payload=None,
            timeout_seconds=None,
        )
        assert len(raw) == 5000

        with pytest.raises(LewLMAppClientResponseTooLargeError) as exc_info:
            await client._send("GET", "/v1/health", operation="health", payload=None, timeout_seconds=None)

    # The error names the operation, so a caller knows which limit to raise.
    assert exc_info.value.operation == "health"
    assert exc_info.value.limit_bytes == 1000


def test_sync_client_applies_a_per_operation_limit_over_the_default() -> None:
    backend = _HttpAppClientBackend(
        base_url="http://lewlm.test",
        api_key=None,
        timeout_seconds=1.0,
        application_id=None,
        client_instance_id="instance-1",
        authorized_actions=None,
        max_response_bytes=1000,
        max_response_bytes_by_operation={"generate_document": 10_000},
    )

    assert backend._limit_for("generate_document") == 10_000
    assert backend._limit_for("health") == 1000


def test_oversized_response_error_names_the_operation() -> None:
    with pytest.raises(LewLMAppClientResponseTooLargeError) as exc_info:
        _read_bounded(_FakeStream(b"x" * 100), limit=10, url="http://x/y", operation="generate_document")
    assert exc_info.value.operation == "generate_document"
    assert "`generate_document`" in str(exc_info.value)
