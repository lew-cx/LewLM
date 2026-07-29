"""First-class asynchronous client for a running LewLM HTTP server.

`LewLMAppClient` is synchronous, so an async caller has to offload it to a
thread. That breaks cancellation: cancelling the caller's task only abandons its
waiter while the underlying HTTP request keeps running to completion, and it
opens a fresh connection per call. This client is natively async, pools
connections, and propagates cancellation to the request itself.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import hashlib
import inspect
from typing import Any, Self, TypeVar
from uuid import uuid4

import httpx
from pydantic import BaseModel

from lewlm.api.routes.models import AsyncDrainRequest, ModelLifecycleResponse
from lewlm.api.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ResponseCreateRequest,
    ResponseCreateResponse,
)
from lewlm.api.schemas.documents import (
    DocumentGenerateRequest,
    DocumentGenerateResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentTransformResponse,
    DocumentUploadSource,
)
from lewlm.api.schemas.health import HealthResponse
from lewlm.api.schemas.multimodal import (
    AudioSpeechCreateRequest,
    AudioSpeechCreateResponse,
    AudioTranscriptionCreateRequest,
    AudioTranscriptionCreateResponse,
    EmbeddingCreateRequest,
    EmbeddingCreateResponse,
    RerankCreateRequest,
    RerankCreateResponse,
    RetrievalContextRequest,
    RetrievalContextResponse,
    TokenCountRequest,
    TokenCountResponse,
)
from lewlm.api.schemas.history import SessionUpdateRequest
from lewlm.api.schemas.tools import ToolListResponse
from lewlm.history.models import SessionRecord
from lewlm.app_helpers import (
    DEFAULT_MAX_RESPONSE_BYTES,
    LewLMAppClientHTTPError,
    LewLMAppClientResponseTooLargeError,
    _parse_http_error_payload,
    _truncated_error_body,
    validate_operation_limits,
)
from lewlm.documents.skills.models import DocumentTransformRequest
from lewlm.runtime.identity import RuntimeInfo
from lewlm.runtime.operations import LifecycleOperationRecord
from lewlm.runtime.residency import ModelResidencySnapshot
from lewlm.telemetry.stats import RuntimeStats
from lewlm.tools.models import LocalToolDescriptor, ToolExecutionEnvelope, ToolExecutionRequest

ResponseT = TypeVar("ResponseT", bound=BaseModel)

#: Default connection-pool sizing for one long-lived client.
DEFAULT_MAX_CONNECTIONS = 20
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 10


class LewLMAsyncClient:
    """Async typed client for a LewLM server, with pooling and real cancellation.

    Use it as an async context manager, or call `aclose()` when finished. The
    underlying connection pool is reused across calls, so one client per
    application beats one per request.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        application_id: str | None = None,
        client_instance_id: str | None = None,
        authorized_actions: Sequence[str] | None = None,
        correlation_id: str | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_response_bytes_by_operation: Mapping[str, int] | None = None,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections: int = DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_response_bytes = max_response_bytes
        # Keyed by the method names in `ASYNC_CLIENT_OPERATIONS`, so a caller
        # can give `generate_document` room without raising the ceiling on
        # every poll of `health`.
        self._max_response_bytes_by_operation = validate_operation_limits(
            max_response_bytes_by_operation,
            operations=ASYNC_CLIENT_OPERATIONS,
        )
        self._correlation_id = correlation_id
        self._default_headers = _client_headers(
            api_key=api_key,
            application_id=application_id,
            client_instance_id=client_instance_id or str(uuid4()),
            authorized_actions=authorized_actions,
            correlation_id=correlation_id,
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout_seconds,
            headers=self._default_headers,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the connection pool. Safe to call more than once."""

        await self._client.aclose()

    @property
    def correlation_id(self) -> str | None:
        return self._correlation_id

    def with_correlation_id(self, correlation_id: str) -> "LewLMAsyncClient":
        """Return a view of this client that stamps a different correlation ID.

        The connection pool is shared, so per-request correlation costs nothing.
        """

        clone = object.__new__(LewLMAsyncClient)
        clone._base_url = self._base_url
        clone._max_response_bytes = self._max_response_bytes
        clone._max_response_bytes_by_operation = self._max_response_bytes_by_operation
        clone._correlation_id = correlation_id
        clone._default_headers = {**self._default_headers, "x-lewlm-correlation-id": correlation_id}
        clone._client = self._client
        return clone

    # --- operational surfaces -------------------------------------------------

    async def health(self, *, timeout_seconds: float | None = None) -> HealthResponse:
        return await self._request(
            "GET",
            "/v1/health",
            HealthResponse,
            operation="health",
            timeout_seconds=timeout_seconds,
        )

    async def runtime_info(self, *, timeout_seconds: float | None = None) -> RuntimeInfo:
        return await self._request(
            "GET",
            "/v1/runtime",
            RuntimeInfo,
            operation="runtime_info",
            timeout_seconds=timeout_seconds,
        )

    async def runtime_stats(self, *, timeout_seconds: float | None = None) -> RuntimeStats:
        return await self._request(
            "GET",
            "/v1/runtime/stats",
            RuntimeStats,
            operation="runtime_stats",
            timeout_seconds=timeout_seconds,
        )

    async def list_model_residencies(self, *, timeout_seconds: float | None = None) -> list[ModelResidencySnapshot]:
        payload = await self._request_raw(
            "GET",
            "/v1/runtime/residencies",
            operation="list_model_residencies",
            timeout_seconds=timeout_seconds,
        )
        return [ModelResidencySnapshot.model_validate(item) for item in payload]

    async def warm_model(self, model_id: str, *, timeout_seconds: float | None = None) -> ModelLifecycleResponse:
        return await self._request(
            "POST",
            f"/v1/models/{model_id}/warm",
            ModelLifecycleResponse,
            operation="warm_model",
            timeout_seconds=timeout_seconds,
        )

    async def unload_model(self, model_id: str, *, timeout_seconds: float | None = None) -> ModelLifecycleResponse:
        return await self._request(
            "POST",
            f"/v1/models/{model_id}/unload",
            ModelLifecycleResponse,
            operation="unload_model",
            timeout_seconds=timeout_seconds,
        )

    async def create_drain_operation(
        self,
        model_id: str,
        *,
        timeout_seconds: float | None = None,
        drain_timeout_seconds: float | None = None,
        idempotency_key: str | None = None,
    ) -> LifecycleOperationRecord:
        return await self._request(
            "POST",
            f"/v1/models/{model_id}/drain-operations",
            LifecycleOperationRecord,
            payload=AsyncDrainRequest(timeout_seconds=drain_timeout_seconds, idempotency_key=idempotency_key),
            operation="create_drain_operation",
            timeout_seconds=timeout_seconds,
        )

    async def get_lifecycle_operation(
        self,
        operation_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> LifecycleOperationRecord:
        return await self._request(
            "GET",
            f"/v1/model-lifecycle/operations/{operation_id}",
            LifecycleOperationRecord,
            operation="get_lifecycle_operation",
            timeout_seconds=timeout_seconds,
        )

    async def list_tools(self, *, timeout_seconds: float | None = None) -> ToolListResponse:
        return await self._request(
            "GET",
            "/v1/tools",
            ToolListResponse,
            operation="list_tools",
            timeout_seconds=timeout_seconds,
        )

    async def get_tool(self, tool_name: str, *, timeout_seconds: float | None = None) -> LocalToolDescriptor:
        return await self._request(
            "GET",
            f"/v1/tools/{tool_name}",
            LocalToolDescriptor,
            operation="get_tool",
            timeout_seconds=timeout_seconds,
        )

    async def execute_tool(
        self,
        payload: ToolExecutionRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolExecutionEnvelope:
        return await self._request(
            "POST",
            "/v1/tools/execute",
            ToolExecutionEnvelope,
            payload=payload,
            operation="execute_tool",
            timeout_seconds=timeout_seconds,
        )

    # --- generation surfaces --------------------------------------------------

    async def chat_completion(
        self,
        payload: ChatCompletionRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> ChatCompletionResponse:
        if payload.stream:
            raise ValueError(
                "LewLMAsyncClient.chat_completion does not support stream=True. "
                "Use stream_chat_completion() for server-sent events.",
            )
        return await self._request(
            "POST",
            "/v1/chat/completions",
            ChatCompletionResponse,
            payload=payload,
            operation="chat_completion",
            timeout_seconds=timeout_seconds,
        )

    async def stream_chat_completion(
        self,
        payload: ChatCompletionRequest,
        *,
        timeout_seconds: float | None = None,
    ):
        """Yield raw server-sent event payloads for a streaming chat request.

        Cancelling the consuming task closes the underlying response, so the
        server stops generating instead of running to completion unobserved.
        """

        streaming_payload = payload.model_copy(update={"stream": True})
        request_timeout = httpx.USE_CLIENT_DEFAULT if timeout_seconds is None else timeout_seconds
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            content=_encoded(streaming_payload),
            headers={"content-type": "application/json", **self._request_headers()},
            timeout=request_timeout,
        ) as response:
            if response.status_code >= 400:
                body = _truncated_error_body((await response.aread()).decode("utf-8", errors="replace"))
                raise LewLMAppClientHTTPError(
                    url=str(response.url),
                    status_code=response.status_code,
                    body=body,
                    api_error=_parse_http_error_payload(body, status_code=response.status_code),
                )
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line[6:]

    async def responses(
        self,
        payload: ResponseCreateRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> ResponseCreateResponse:
        if payload.stream:
            raise ValueError("LewLMAsyncClient.responses does not support stream=True.")
        return await self._request(
            "POST",
            "/v1/responses",
            ResponseCreateResponse,
            payload=payload,
            operation="responses",
            timeout_seconds=timeout_seconds,
        )

    async def embeddings(
        self,
        payload: EmbeddingCreateRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> EmbeddingCreateResponse:
        return await self._request(
            "POST",
            "/v1/embeddings",
            EmbeddingCreateResponse,
            payload=payload,
            operation="embeddings",
            timeout_seconds=timeout_seconds,
        )

    async def rerank(
        self,
        payload: RerankCreateRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> RerankCreateResponse:
        return await self._request(
            "POST",
            "/v1/rerank",
            RerankCreateResponse,
            payload=payload,
            operation="rerank",
            timeout_seconds=timeout_seconds,
        )

    async def retrieve_context(
        self,
        payload: RetrievalContextRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> RetrievalContextResponse:
        return await self._request(
            "POST",
            "/v1/retrieval/context",
            RetrievalContextResponse,
            payload=payload,
            operation="retrieve_context",
            timeout_seconds=timeout_seconds,
        )

    async def count_tokens(
        self,
        payload: TokenCountRequest | None = None,
        *,
        text: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> TokenCountResponse:
        if payload is None:
            if text is None:
                raise ValueError("text is required when payload is not provided.")
            payload = TokenCountRequest(text=text, model=model, max_tokens=max_tokens)
        return await self._request(
            "POST",
            "/v1/tokenize/count",
            TokenCountResponse,
            payload=payload,
            operation="count_tokens",
            timeout_seconds=timeout_seconds,
        )

    async def transcribe_audio(
        self,
        payload: AudioTranscriptionCreateRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> AudioTranscriptionCreateResponse:
        return await self._request(
            "POST",
            "/v1/audio/transcriptions",
            AudioTranscriptionCreateResponse,
            payload=payload,
            operation="transcribe_audio",
            timeout_seconds=timeout_seconds,
        )

    async def synthesize_speech(
        self,
        payload: AudioSpeechCreateRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> AudioSpeechCreateResponse:
        return await self._request(
            "POST",
            "/v1/audio/speech",
            AudioSpeechCreateResponse,
            payload=payload,
            operation="synthesize_speech",
            timeout_seconds=timeout_seconds,
        )

    async def rename_session(
        self,
        session_id: str,
        title: str,
        *,
        timeout_seconds: float | None = None,
    ) -> SessionRecord:
        """Rename a session without touching its metadata or turn history."""

        return await self.update_session(
            session_id,
            SessionUpdateRequest(title=title),
            timeout_seconds=timeout_seconds,
        )

    async def update_session(
        self,
        session_id: str,
        payload: SessionUpdateRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> SessionRecord:
        return await self._request(
            "PATCH",
            f"/v1/sessions/{session_id}",
            SessionRecord,
            payload=payload,
            operation="update_session",
            timeout_seconds=timeout_seconds,
        )

    # --- document surfaces ----------------------------------------------------

    async def ingest_documents(
        self,
        payload: DocumentIngestRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> DocumentIngestResponse:
        return await self._request(
            "POST",
            "/v1/documents/ingest",
            DocumentIngestResponse,
            payload=payload,
            operation="ingest_documents",
            timeout_seconds=timeout_seconds,
        )

    async def generate_document(
        self,
        payload: DocumentGenerateRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> DocumentGenerateResponse:
        return await self._request(
            "POST",
            "/v1/documents/generate",
            DocumentGenerateResponse,
            payload=payload,
            operation="generate_document",
            timeout_seconds=timeout_seconds,
        )

    async def transform_document(
        self,
        payload: DocumentTransformRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> DocumentTransformResponse:
        return await self._request(
            "POST",
            "/v1/documents/transform",
            DocumentTransformResponse,
            payload=payload,
            operation="transform_document",
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def upload_source(
        source_id: str,
        content: bytes,
        *,
        file_name: str,
        media_type: str | None = None,
        expected_sha256: str | None = None,
        verify: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentUploadSource:
        """Build an uploadable source, computing the integrity digest by default."""

        if verify and expected_sha256 is None:
            expected_sha256 = hashlib.sha256(content).hexdigest()
        return DocumentUploadSource(
            source_id=source_id,
            file_name=file_name,
            content_base64=base64.b64encode(content).decode("ascii"),
            media_type=media_type,
            expected_sha256=expected_sha256,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def document_bytes(response: DocumentGenerateResponse) -> bytes:
        """Decode a rendered artifact without hand-rolling base64 handling."""

        return base64.b64decode(response.content_base64, validate=True)

    # --- transport ------------------------------------------------------------

    def _request_headers(self) -> dict[str, str]:
        if self._correlation_id:
            return {"x-lewlm-correlation-id": self._correlation_id}
        return {}

    async def _request(
        self,
        method: str,
        path: str,
        response_type: type[ResponseT],
        *,
        operation: str,
        payload: BaseModel | None = None,
        timeout_seconds: float | None = None,
    ) -> ResponseT:
        raw = await self._send(
            method,
            path,
            operation=operation,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return response_type.model_validate_json(raw)

    async def _request_raw(
        self,
        method: str,
        path: str,
        *,
        # An unnamed operation falls back to the client-wide ceiling.
        operation: str = "",
        payload: BaseModel | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        import json

        raw = await self._send(
            method,
            path,
            operation=operation,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return json.loads(raw)

    def _limit_for(self, operation: str) -> int:
        return self._max_response_bytes_by_operation.get(operation, self._max_response_bytes)

    async def _send(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        payload: BaseModel | None,
        timeout_seconds: float | None,
    ) -> bytes:
        limit = self._limit_for(operation)
        headers = self._request_headers()
        content = None
        if payload is not None:
            headers = {**headers, "content-type": "application/json"}
            content = _encoded(payload)
        request_timeout = httpx.USE_CLIENT_DEFAULT if timeout_seconds is None else timeout_seconds

        # Streaming the response lets an over-long body be refused before it is
        # fully buffered, rather than after.
        async with self._client.stream(
            method,
            path,
            content=content,
            headers=headers,
            timeout=request_timeout,
        ) as response:
            if response.status_code >= 400:
                body = _truncated_error_body((await response.aread()).decode("utf-8", errors="replace").strip() or None)
                raise LewLMAppClientHTTPError(
                    url=str(response.url),
                    status_code=response.status_code,
                    body=body,
                    api_error=_parse_http_error_payload(body, status_code=response.status_code),
                )
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > limit:
                    raise LewLMAppClientResponseTooLargeError(
                        url=str(response.url),
                        limit_bytes=limit,
                        operation=operation or None,
                    )
                chunks.append(chunk)
        return b"".join(chunks)


def _encoded(payload: BaseModel) -> bytes:
    return payload.model_dump_json(exclude_none=True, by_alias=True).encode("utf-8")


def _client_headers(
    *,
    api_key: str | None,
    application_id: str | None,
    client_instance_id: str,
    authorized_actions: Sequence[str] | None,
    correlation_id: str | None,
) -> dict[str, str]:
    headers = {
        "accept": "application/json",
        "x-lewlm-client-instance-id": client_instance_id,
    }
    if api_key:
        headers["x-api-key"] = api_key
    if application_id:
        headers["x-lewlm-application-id"] = application_id
    if authorized_actions:
        headers["x-lewlm-authorized-actions"] = ",".join(authorized_actions)
    if correlation_id:
        headers["x-lewlm-correlation-id"] = correlation_id
    return headers


#: `stream_chat_completion` yields frames rather than buffering a body, so no
#: response-size limit applies to it; `rename_session` delegates to
#: `update_session` and is bounded under that name.
_UNBOUNDED_ASYNC_OPERATIONS = frozenset({"aclose", "rename_session"})

#: Operation names accepted by `max_response_bytes_by_operation`, derived from
#: the client's own methods so the registry cannot drift from the surface.
ASYNC_CLIENT_OPERATIONS: tuple[str, ...] = tuple(
    name
    for name, member in vars(LewLMAsyncClient).items()
    if not name.startswith("_")
    and inspect.iscoroutinefunction(member)
    and name not in _UNBOUNDED_ASYNC_OPERATIONS
)
