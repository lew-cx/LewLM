"""Thin typed helper layer for embedded apps and local-server clients."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import hashlib
from http import HTTPStatus
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import RootModel

from lewlm.api.message_normalization import normalize_chat_messages
from lewlm.api.routes.chat import (
    _completion_usage,
    _merge_session_messages,
    _persist_session_turn,
    _prompt_request_from_payload,
    _reasoning_visibility_from_request,
)
from lewlm.api.routes.multimodal import _decode_audio_bytes
from lewlm.api.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ResponseCreateRequest,
    ResponseCreateResponse,
    ResponseInputMessage,
    ResponseOutputText,
)
from lewlm.api.schemas.documents import (
    DocumentGenerateRequest,
    DocumentGenerateResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentTransformResponse,
    DocumentUploadSource,
)
from lewlm.api.schemas.history import SessionUpdateRequest
from lewlm.api.schemas.tools import ToolListResponse
from lewlm.history.models import SessionRecord
from lewlm.core.citations import CitationContextPackage
from lewlm.api.schemas.health import HealthResponse
from lewlm.api.routes.models import AsyncDrainRequest, ModelLifecycleResponse
from lewlm.api.schemas.multimodal import (
    AudioSpeechCreateRequest,
    AudioSpeechCreateResponse,
    AudioTranscriptionCreateRequest,
    AudioTranscriptionCreateResponse,
    AudioTranscriptionSegment,
    EmbeddingCreateRequest,
    EmbeddingCreateResponse,
    EmbeddingDatum,
    RetrievalContextItem,
    RetrievalContextRequest,
    RetrievalContextResponse,
    RetrievalStageSummary,
    RerankCreateRequest,
    RerankCreateResponse,
    RerankResultItem,
    TokenCountRequest,
    TokenCountResponse,
)
from lewlm.core.contracts import ReasoningVisibility
from lewlm.core.errors import LewLMError, error_from_dict
from lewlm.core.execution_metadata import build_tool_execution_metadata
from lewlm.core.provenance import ComponentProvenance, renderer_provenance
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.documents.skills.models import DocumentTransformRequest
from lewlm.documents.ingest.models import DocumentChunk, IngestedDocumentSource
from lewlm.structured_output import StructuredOutputRequest
from lewlm.telemetry.stats import RuntimeStats
from lewlm.runtime.cancellation import RequestCancellationRecord, validate_request_handle
from lewlm.runtime.identity import RuntimeInfo
from lewlm.runtime.operations import LifecycleOperationRecord
from lewlm.runtime.residency import ModelResidencySnapshot
from lewlm.tools.models import (
    DocumentGenerateToolRequest,
    DocumentIngestToolRequest,
    DocumentTransformToolRequest,
    GenerateDocumentToolInput,
    IngestDocumentToolInput,
    LocalToolDescriptor,
    ToolExecutionEnvelope,
    ToolExecutionRequest,
)

if TYPE_CHECKING:
    from lewlm.library import LewLM


#: Default ceiling on a single response body. Document artifacts are returned
#: base64-encoded, so this has to accommodate a large rendered PDF while still
#: refusing an unbounded read.
DEFAULT_MAX_RESPONSE_BYTES = 128 * 1024 * 1024

#: Error bodies are only ever read for diagnostics, so they stay small.
MAX_ERROR_BODY_CHARACTERS = 8 * 1024


def validate_operation_limits(
    limits: Mapping[str, int] | None,
    *,
    operations: Sequence[str] | None = None,
) -> dict[str, int]:
    """Normalize a per-operation response-size map, rejecting unknown names.

    A mistyped operation name would otherwise be accepted and simply never
    apply, which is the worst way to learn that a limit was not in force.
    """

    if not limits:
        return {}
    known = tuple(operations) if operations is not None else APP_CLIENT_OPERATIONS
    unknown = sorted(set(limits) - set(known))
    if unknown:
        raise ValueError(
            f"Unknown LewLM client operation(s): {', '.join(unknown)}. "
            f"Valid operations: {', '.join(known)}.",
        )
    invalid = sorted(name for name, limit in limits.items() if not isinstance(limit, int) or limit <= 0)
    if invalid:
        raise ValueError(f"Response-size limits must be positive integers; got non-positive for: {', '.join(invalid)}.")
    return dict(limits)


class _ModelResidencyList(RootModel[list[ModelResidencySnapshot]]):
    pass


class _OptionalModelResidency(RootModel[ModelResidencySnapshot | None]):
    pass


class LewLMAppClientHTTPError(LewLMError):
    """Raised when the HTTP-backed helper receives a non-success response."""

    def __init__(
        self,
        *,
        url: str,
        status_code: int,
        body: str | None = None,
        api_error: LewLMError | None = None,
    ) -> None:
        if api_error is not None:
            message = str(api_error)
            code = api_error.code
            details = api_error.details
        else:
            message = f"LewLM app client request failed with HTTP {status_code} for {url}."
            if body:
                message = f"{message} {body}"
            code = "http_error"
            details = {}
        super().__init__(message, code=code, status_code=status_code, details=details)
        self.url = url
        self.body = body
        self.api_error = api_error


class LewLMAppClientResponseTooLargeError(LewLMError):
    """Raised when a response exceeds the client's configured size limit.

    Failing beats buffering: an unbounded read lets a misconfigured or hostile
    endpoint exhaust the calling process's memory.
    """

    def __init__(self, *, url: str, limit_bytes: int, operation: str | None = None) -> None:
        scope = f"`{operation}` " if operation else ""
        super().__init__(
            f"LewLM app client {scope}response from {url} exceeded the {limit_bytes}-byte limit.",
            code="response_too_large",
            status_code=HTTPStatus.INSUFFICIENT_STORAGE,
            details={"url": url, "limit_bytes": limit_bytes, "operation": operation},
        )
        self.url = url
        self.limit_bytes = limit_bytes
        self.operation = operation


class _AppClientBackend(Protocol):
    def health(self) -> HealthResponse: ...

    def runtime_stats(self) -> RuntimeStats: ...

    def runtime_info(self) -> RuntimeInfo: ...

    def list_model_residencies(self) -> list[ModelResidencySnapshot]: ...

    def get_model_residency(self, model_id: str, runtime: str | None = None) -> ModelResidencySnapshot | None: ...

    def warm_model(self, model_id: str) -> ModelLifecycleResponse: ...

    def unload_model(self, model_id: str) -> ModelLifecycleResponse: ...

    def drain_model(self, model_id: str) -> ModelLifecycleResponse: ...

    def create_drain_operation(
        self,
        model_id: str,
        *,
        timeout_seconds: float | None = None,
        idempotency_key: str | None = None,
    ) -> LifecycleOperationRecord: ...

    def get_lifecycle_operation(self, operation_id: str) -> LifecycleOperationRecord: ...

    def cancel_lifecycle_operation(self, operation_id: str) -> LifecycleOperationRecord: ...

    def cancel_request(self, request_id: str) -> RequestCancellationRecord: ...

    def list_tools(self) -> ToolListResponse: ...

    def get_tool(self, tool_name: str) -> LocalToolDescriptor: ...

    def execute_tool(self, payload: ToolExecutionRequest) -> ToolExecutionEnvelope: ...

    def chat_completion(self, payload: ChatCompletionRequest) -> ChatCompletionResponse: ...

    def responses(self, payload: ResponseCreateRequest) -> ResponseCreateResponse: ...

    def embeddings(self, payload: EmbeddingCreateRequest) -> EmbeddingCreateResponse: ...

    def rerank(self, payload: RerankCreateRequest) -> RerankCreateResponse: ...

    def retrieve_context(self, payload: RetrievalContextRequest) -> RetrievalContextResponse: ...

    def transcribe_audio(self, payload: AudioTranscriptionCreateRequest) -> AudioTranscriptionCreateResponse: ...

    def synthesize_speech(self, payload: AudioSpeechCreateRequest) -> AudioSpeechCreateResponse: ...

    def ingest_documents(self, payload: DocumentIngestRequest) -> DocumentIngestResponse: ...

    def update_session(self, session_id: str, payload: SessionUpdateRequest) -> SessionRecord: ...

    def generate_document(self, payload: DocumentGenerateRequest) -> DocumentGenerateResponse: ...

    def transform_document(self, payload: DocumentTransformRequest) -> DocumentTransformResponse: ...

    def count_tokens(self, payload: TokenCountRequest) -> TokenCountResponse: ...


#: Every operation the client can perform, in declaration order. Derived from
#: the backend protocol so a new surface cannot be added without becoming
#: addressable by a per-operation response-size limit.
APP_CLIENT_OPERATIONS: tuple[str, ...] = tuple(
    name for name in vars(_AppClientBackend) if not name.startswith("_")
)


class LewLMAppClient:
    """Typed helper surface for host apps embedding LewLM or calling the local server."""

    def __init__(self, backend: _AppClientBackend) -> None:
        self._backend = backend

    @classmethod
    def from_lewlm(cls, lewlm: LewLM) -> Self:
        """Bind the helper to an in-process LewLM facade."""

        return cls(_EmbeddedAppClientBackend(lewlm))

    @classmethod
    def from_http(
        cls,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        application_id: str | None = None,
        client_instance_id: str | None = None,
        authorized_actions: Sequence[str] | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_response_bytes_by_operation: Mapping[str, int] | None = None,
        correlation_id: str | None = None,
    ) -> Self:
        """Bind the helper to a running local LewLM HTTP server.

        `max_response_bytes` is the ceiling every operation inherits.
        `max_response_bytes_by_operation` overrides it per operation, keyed by
        the method names in `APP_CLIENT_OPERATIONS`, so a caller can leave a
        tight bound on `health` while allowing `generate_document` the room a
        rendered artifact actually needs. Unknown names are rejected rather
        than silently ignored.
        """

        return cls(
            _HttpAppClientBackend(
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                application_id=application_id,
                client_instance_id=client_instance_id or str(uuid4()),
                authorized_actions=authorized_actions,
                max_response_bytes=max_response_bytes,
                max_response_bytes_by_operation=max_response_bytes_by_operation,
                correlation_id=correlation_id,
            ),
        )

    def health(self) -> HealthResponse:
        """Return the typed health snapshot."""

        return self._backend.health()

    def runtime_stats(self) -> RuntimeStats:
        """Return typed runtime diagnostics."""

        return self._backend.runtime_stats()

    def runtime_info(self) -> RuntimeInfo:
        """Return stable identity and a compact live summary for the connected runtime."""

        return self._backend.runtime_info()

    def list_model_residencies(self) -> list[ModelResidencySnapshot]:
        """List live model residencies owned by the connected runtime."""

        return self._backend.list_model_residencies()

    def get_model_residency(self, model_id: str, runtime: str | None = None) -> ModelResidencySnapshot | None:
        """Return residency state for one model, if it is tracked."""

        return self._backend.get_model_residency(model_id, runtime)

    def warm_model(self, model_id: str) -> ModelLifecycleResponse:
        """Load and warm a model through the operator lifecycle surface."""

        return self._backend.warm_model(model_id)

    def unload_model(self, model_id: str) -> ModelLifecycleResponse:
        """Unload an unused model through the administrative lifecycle surface."""

        return self._backend.unload_model(model_id)

    def drain_model(self, model_id: str) -> ModelLifecycleResponse:
        """Stop new leases, wait for active usage, and unload a model."""

        return self._backend.drain_model(model_id)

    def create_drain_operation(
        self,
        model_id: str,
        *,
        timeout_seconds: float | None = None,
        idempotency_key: str | None = None,
    ) -> LifecycleOperationRecord:
        """Create a recorded drain operation.

        HTTP-backed clients return the live polling record immediately. The
        synchronous embedded backend completes the operation before returning
        because it does not own a long-lived event loop for background tasks.
        """

        return self._backend.create_drain_operation(
            model_id,
            timeout_seconds=timeout_seconds,
            idempotency_key=idempotency_key,
        )

    def get_lifecycle_operation(self, operation_id: str) -> LifecycleOperationRecord:
        """Poll an asynchronous model lifecycle operation."""

        return self._backend.get_lifecycle_operation(operation_id)

    def cancel_lifecycle_operation(self, operation_id: str) -> LifecycleOperationRecord:
        """Cancel an asynchronous model lifecycle operation when still active."""

        return self._backend.cancel_lifecycle_operation(operation_id)

    def cancel_request(self, request_id: str) -> RequestCancellationRecord:
        """Cancel an in-flight request by its `x-request-id` handle.

        For orchestrating requests this client did not issue: this client sends
        one request at a time on the calling thread, and does not stamp per-call
        handles, so the handles it cancels come from elsewhere — typically a
        worker using `LewLMAsyncClient(..., request_id=...)`. Cancellation is
        best-effort and only affects the LewLM instance holding the request.
        """

        return self._backend.cancel_request(request_id)

    def list_tools(self) -> ToolListResponse:
        """Return the typed local-tool catalog."""

        return self._backend.list_tools()

    def get_tool(self, tool_name: str) -> LocalToolDescriptor:
        """Return one typed local-tool descriptor."""

        return self._backend.get_tool(tool_name)

    def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionEnvelope:
        """Execute one local tool request with the shared API envelope."""

        return self._backend.execute_tool(request)

    def chat_completion(
        self,
        request: ChatCompletionRequest | None = None,
        *,
        model: str | None = None,
        session_id: str | None = None,
        messages: Sequence[ChatMessage] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        apply_serving_profile: bool = True,
        stream: bool = False,
        reasoning_visibility: ReasoningVisibility | None = None,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        citation_context: CitationContextPackage | None = None,
        pretext_path: str | None = None,
        skills_path: str | None = None,
        response_format: StructuredOutputRequest | None = None,
        response_format_path: str | None = None,
        output_schema: dict[str, Any] | None = None,
        output_schema_path: str | None = None,
        tools: Sequence[Any] | None = None,
        tools_path: str | None = None,
        mcp_tools: Sequence[Any] | None = None,
        mcp_tools_path: str | None = None,
        include_prompt_trace: bool = False,
    ) -> ChatCompletionResponse:
        """Create one non-streaming chat completion with API-shaped models."""

        if request is not None and any(
            (
                model is not None,
                session_id is not None,
                messages is not None,
                max_tokens != 512,
                temperature != 0.7,
                not apply_serving_profile,
                stream,
                reasoning_visibility is not None,
                system_prompt is not None,
                developer_prompt is not None,
                citation_context is not None,
                pretext_path is not None,
                skills_path is not None,
                response_format is not None,
                response_format_path is not None,
                output_schema is not None,
                output_schema_path is not None,
                tools is not None,
                tools_path is not None,
                mcp_tools is not None,
                mcp_tools_path is not None,
                include_prompt_trace,
            ),
        ):
            raise ValueError("Pass either `request` or keyword arguments to chat_completion(), not both.")
        if request is None and messages is None:
            raise ValueError("messages is required when request is not provided.")
        payload = request or ChatCompletionRequest(
            model=model,
            session_id=session_id,
            messages=list(messages or ()),
            max_tokens=max_tokens,
            temperature=temperature,
            apply_serving_profile=apply_serving_profile,
            stream=stream,
            reasoning_visibility=reasoning_visibility,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            citation_context=citation_context,
            pretext_path=pretext_path,
            skills_path=skills_path,
            response_format=response_format,
            response_format_path=response_format_path,
            output_schema=output_schema,
            output_schema_path=output_schema_path,
            tools=list(tools or ()),
            tools_path=tools_path,
            mcp_tools=list(mcp_tools or ()),
            mcp_tools_path=mcp_tools_path,
            include_prompt_trace=include_prompt_trace,
        )
        return self._backend.chat_completion(payload)

    def responses(
        self,
        request: ResponseCreateRequest | None = None,
        *,
        model: str | None = None,
        session_id: str | None = None,
        input: str | Sequence[ResponseInputMessage] | None = None,
        max_output_tokens: int = 512,
        temperature: float = 0.7,
        apply_serving_profile: bool = True,
        stream: bool = False,
        reasoning_visibility: ReasoningVisibility | None = None,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        citation_context: CitationContextPackage | None = None,
        pretext_path: str | None = None,
        skills_path: str | None = None,
        response_format: StructuredOutputRequest | None = None,
        response_format_path: str | None = None,
        output_schema: dict[str, Any] | None = None,
        output_schema_path: str | None = None,
        tools: Sequence[Any] | None = None,
        tools_path: str | None = None,
        mcp_tools: Sequence[Any] | None = None,
        mcp_tools_path: str | None = None,
        include_prompt_trace: bool = False,
    ) -> ResponseCreateResponse:
        """Create one non-streaming responses-style completion with API-shaped models."""

        if request is not None and any(
            (
                model is not None,
                session_id is not None,
                input is not None,
                max_output_tokens != 512,
                temperature != 0.7,
                not apply_serving_profile,
                stream,
                reasoning_visibility is not None,
                system_prompt is not None,
                developer_prompt is not None,
                citation_context is not None,
                pretext_path is not None,
                skills_path is not None,
                response_format is not None,
                response_format_path is not None,
                output_schema is not None,
                output_schema_path is not None,
                tools is not None,
                tools_path is not None,
                mcp_tools is not None,
                mcp_tools_path is not None,
                include_prompt_trace,
            ),
        ):
            raise ValueError("Pass either `request` or keyword arguments to responses(), not both.")
        if request is None and input is None:
            raise ValueError("input is required when request is not provided.")
        payload = request or ResponseCreateRequest(
            model=model,
            session_id=session_id,
            input=input if isinstance(input, str) or input is None else list(input),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            apply_serving_profile=apply_serving_profile,
            stream=stream,
            reasoning_visibility=reasoning_visibility,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            citation_context=citation_context,
            pretext_path=pretext_path,
            skills_path=skills_path,
            response_format=response_format,
            response_format_path=response_format_path,
            output_schema=output_schema,
            output_schema_path=output_schema_path,
            tools=list(tools or ()),
            tools_path=tools_path,
            mcp_tools=list(mcp_tools or ()),
            mcp_tools_path=mcp_tools_path,
            include_prompt_trace=include_prompt_trace,
        )
        return self._backend.responses(payload)

    def embeddings(
        self,
        request: EmbeddingCreateRequest | None = None,
        *,
        model: str | None = None,
        inputs: str | Sequence[str] | None = None,
    ) -> EmbeddingCreateResponse:
        """Create embeddings with API-shaped requests and responses."""

        if request is not None and any((model is not None, inputs is not None)):
            raise ValueError("Pass either `request` or keyword arguments to embeddings(), not both.")
        if request is None and inputs is None:
            raise ValueError("inputs is required when request is not provided.")
        payload = request or EmbeddingCreateRequest(
            model=model,
            input=inputs if isinstance(inputs, str) or inputs is None else list(inputs),
        )
        return self._backend.embeddings(payload)

    def rerank(
        self,
        request: RerankCreateRequest | None = None,
        *,
        model: str | None = None,
        query: str | None = None,
        documents: Sequence[str] | None = None,
        top_n: int | None = None,
    ) -> RerankCreateResponse:
        """Rerank candidate documents with API-shaped requests and responses."""

        if request is not None and any((model is not None, query is not None, documents is not None, top_n is not None)):
            raise ValueError("Pass either `request` or keyword arguments to rerank(), not both.")
        if request is None and (query is None or documents is None):
            raise ValueError("query and documents are required when request is not provided.")
        payload = request or RerankCreateRequest(
            model=model,
            query=query,
            documents=list(documents),
            top_n=top_n,
        )
        return self._backend.rerank(payload)

    def retrieve_context(
        self,
        request: RetrievalContextRequest | None = None,
        *,
        query: str | None = None,
        candidate_chunks: Sequence[DocumentChunk] | None = None,
        candidate_sources: Sequence[IngestedDocumentSource] | None = None,
        top_k: int = 8,
        use_embeddings: bool = True,
        use_rerank: bool = True,
        embedding_model: str | None = None,
        rerank_model: str | None = None,
    ) -> RetrievalContextResponse:
        """Rank caller-provided chunks into reusable retrieval context packages."""

        if request is not None and any(
            (
                query is not None,
                candidate_chunks is not None,
                candidate_sources is not None,
                top_k != 8,
                not use_embeddings,
                not use_rerank,
                embedding_model is not None,
                rerank_model is not None,
            ),
        ):
            raise ValueError("Pass either `request` or keyword arguments to retrieve_context(), not both.")
        if request is None and (query is None or candidate_chunks is None):
            raise ValueError("query and candidate_chunks are required when request is not provided.")
        payload = request or RetrievalContextRequest(
            query=query,
            candidate_chunks=list(candidate_chunks or ()),
            candidate_sources=list(candidate_sources or ()),
            top_k=top_k,
            use_embeddings=use_embeddings,
            use_rerank=use_rerank,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
        )
        return self._backend.retrieve_context(payload)

    def transcribe_audio(
        self,
        request: AudioTranscriptionCreateRequest | None = None,
        *,
        model: str | None = None,
        audio_bytes: bytes | None = None,
        file_name: str = "audio.wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> AudioTranscriptionCreateResponse:
        """Transcribe audio with API-shaped requests and responses."""

        if request is not None and any(
            (
                model is not None,
                audio_bytes is not None,
                file_name != "audio.wav",
                language is not None,
                prompt is not None,
            ),
        ):
            raise ValueError("Pass either `request` or keyword arguments to transcribe_audio(), not both.")
        if request is None and audio_bytes is None:
            raise ValueError("audio_bytes is required when request is not provided.")
        payload = request or AudioTranscriptionCreateRequest(
            model=model,
            audio_base64=base64.b64encode(audio_bytes or b"").decode("ascii"),
            file_name=file_name,
            language=language,
            prompt=prompt,
        )
        return self._backend.transcribe_audio(payload)

    def synthesize_speech(
        self,
        request: AudioSpeechCreateRequest | None = None,
        *,
        model: str | None = None,
        input_text: str | None = None,
        voice: str | None = None,
        audio_format: str = "wav",
    ) -> AudioSpeechCreateResponse:
        """Synthesize speech with API-shaped requests and responses."""

        if request is not None and any(
            (
                model is not None,
                input_text is not None,
                voice is not None,
                audio_format != "wav",
            ),
        ):
            raise ValueError("Pass either `request` or keyword arguments to synthesize_speech(), not both.")
        if request is None and input_text is None:
            raise ValueError("input_text is required when request is not provided.")
        payload = request or AudioSpeechCreateRequest(
            model=model,
            input=input_text or "",
            voice=voice,
            format=audio_format,
        )
        return self._backend.synthesize_speech(payload)

    def ingest_documents(
        self,
        request: DocumentIngestRequest | None = None,
        *,
        paths: Sequence[Path | str] | Path | str | None = None,
        sources: Sequence[DocumentUploadSource] | None = None,
        title: str | None = None,
        authorized_actions: Sequence[str] | None = None,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
    ) -> DocumentIngestResponse:
        """Ingest local paths or uploaded bytes with the API's typed shapes.

        Prefer `sources` for a remote LewLM: uploaded bytes carry caller-owned
        identity and need no shared filesystem mount.
        """

        keyword_arguments = (paths, sources, title, authorized_actions, idempotency_key, correlation_id)
        if request is not None and any(item is not None for item in keyword_arguments):
            raise ValueError("Pass either `request` or keyword arguments to ingest_documents(), not both.")
        if request is None and paths is None and sources is None:
            raise ValueError("paths or sources is required when request is not provided.")
        payload = request or DocumentIngestRequest(
            paths=_normalize_paths(paths) if paths is not None else [],
            sources=list(sources or ()),
            title=title,
            authorized_actions=list(authorized_actions or ()),
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
        )
        return self._backend.ingest_documents(payload)

    def upload_source(
        self,
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

    def generate_document(
        self,
        request: DocumentGenerateRequest | None = None,
        *,
        document: DocumentIR | None = None,
        output_format: DocumentOutputFormat | str | None = None,
        file_name: str | None = None,
        authorized_actions: Sequence[str] | None = None,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
    ) -> DocumentGenerateResponse:
        """Render a document artifact using the API's typed request and response."""

        keyword_arguments = (document, output_format, file_name, authorized_actions, idempotency_key, correlation_id)
        if request is not None and any(item is not None for item in keyword_arguments):
            raise ValueError("Pass either `request` or keyword arguments to generate_document(), not both.")
        if request is None and (document is None or output_format is None):
            raise ValueError("document and output_format are required when request is not provided.")
        payload = request or DocumentGenerateRequest(
            output_format=DocumentOutputFormat(output_format),
            document=document,
            file_name=file_name,
            authorized_actions=list(authorized_actions or ()),
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
        )
        return self._backend.generate_document(payload)

    def transform_document(self, request: DocumentTransformRequest) -> DocumentTransformResponse:
        """Run a built-in document skill and return its rendered artifact."""

        return self._backend.transform_document(request)

    def rename_session(self, session_id: str, title: str) -> SessionRecord:
        """Rename a session without touching its metadata or turn history."""

        return self._backend.update_session(session_id, SessionUpdateRequest(title=title))

    def update_session(
        self,
        session_id: str,
        request: SessionUpdateRequest | None = None,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        replace_metadata: bool = False,
    ) -> SessionRecord:
        """Apply a partial update to a session. Omitted fields stay unchanged."""

        if request is not None and any(item is not None for item in (title, metadata)):
            raise ValueError("Pass either `request` or keyword arguments to update_session(), not both.")
        payload = request or SessionUpdateRequest(
            title=title,
            metadata=metadata,
            replace_metadata=replace_metadata,
        )
        return self._backend.update_session(session_id, payload)

    @staticmethod
    def document_bytes(response: DocumentGenerateResponse) -> bytes:
        """Decode a rendered artifact, so callers never hand-roll base64 handling."""

        return base64.b64decode(response.content_base64, validate=True)

    def count_tokens(
        self,
        request: TokenCountRequest | None = None,
        *,
        text: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        correlation_id: str | None = None,
    ) -> TokenCountResponse:
        """Count tokens with the selected model's tokenizer instead of estimating."""

        if request is not None and any(item is not None for item in (text, model, max_tokens, correlation_id)):
            raise ValueError("Pass either `request` or keyword arguments to count_tokens(), not both.")
        if request is None and text is None:
            raise ValueError("text is required when request is not provided.")
        payload = request or TokenCountRequest(
            text=text or "",
            model=model,
            max_tokens=max_tokens,
            correlation_id=correlation_id,
        )
        return self._backend.count_tokens(payload)


class _EmbeddedAppClientBackend:
    def __init__(self, lewlm: LewLM) -> None:
        self._lewlm = lewlm

    def health(self) -> HealthResponse:
        return HealthResponse.model_validate(self._lewlm.health())

    def runtime_stats(self) -> RuntimeStats:
        return self._lewlm.runtime_stats_sync()

    def runtime_info(self) -> RuntimeInfo:
        from lewlm.library import _run_sync

        async def resolve() -> RuntimeInfo:
            services = self._lewlm.services
            residencies = await services.model_residency_manager.list_residencies()
            return RuntimeInfo(
                **services.runtime_instance.model_dump(),
                loaded_model_count=sum(1 for item in residencies if item.state.value == "ready"),
                active_request_count=int(services.runtime_request_scheduler.snapshot()["active_requests"]),
            )

        return _run_sync(resolve, helper_name="LewLMAppClient.runtime_info", async_name="runtime_info")

    def list_model_residencies(self) -> list[ModelResidencySnapshot]:
        from lewlm.library import _run_sync

        return _run_sync(
            self._lewlm.services.model_residency_manager.list_residencies,
            helper_name="LewLMAppClient.list_model_residencies",
            async_name="ModelResidencyManager.list_residencies",
        )

    def get_model_residency(self, model_id: str, runtime: str | None = None) -> ModelResidencySnapshot | None:
        from lewlm.library import _run_sync

        return _run_sync(
            lambda: self._lewlm.services.model_residency_manager.get_residency(model_id, runtime_name=runtime),
            helper_name="LewLMAppClient.get_model_residency",
            async_name="ModelResidencyManager.get_residency",
        )

    def warm_model(self, model_id: str) -> ModelLifecycleResponse:
        return self._lifecycle(model_id, operation="warm")

    def unload_model(self, model_id: str) -> ModelLifecycleResponse:
        return self._lifecycle(model_id, operation="unload")

    def drain_model(self, model_id: str) -> ModelLifecycleResponse:
        return self._lifecycle(model_id, operation="drain")

    def create_drain_operation(
        self,
        model_id: str,
        *,
        timeout_seconds: float | None = None,
        idempotency_key: str | None = None,
    ) -> LifecycleOperationRecord:
        from lewlm.library import _run_sync

        services = self._lewlm.services
        return _run_sync(
            lambda: services.lifecycle_operation_manager.submit_drain_and_wait(
                model_id,
                timeout_seconds=(
                    timeout_seconds
                    if timeout_seconds is not None
                    else float(services.settings.model_drain_timeout_seconds)
                ),
                application_id="embedded",
                client_instance_id=None,
                idempotency_key=idempotency_key,
            ),
            helper_name="LewLMAppClient.create_drain_operation",
            async_name="LifecycleOperationManager.submit_drain",
        )

    def get_lifecycle_operation(self, operation_id: str) -> LifecycleOperationRecord:
        from lewlm.library import _run_sync

        return _run_sync(
            lambda: self._lewlm.services.lifecycle_operation_manager.get(operation_id),
            helper_name="LewLMAppClient.get_lifecycle_operation",
            async_name="LifecycleOperationManager.get",
        )

    def cancel_lifecycle_operation(self, operation_id: str) -> LifecycleOperationRecord:
        from lewlm.library import _run_sync

        return _run_sync(
            lambda: self._lewlm.services.lifecycle_operation_manager.cancel(operation_id),
            helper_name="LewLMAppClient.cancel_lifecycle_operation",
            async_name="LifecycleOperationManager.cancel",
        )

    def cancel_request(self, request_id: str) -> RequestCancellationRecord:
        # Embedded callers share the process with the registry, so no event loop
        # is involved: the handle is cancelled directly. The host owns the
        # process and has no HTTP application identity, so it may cancel any
        # handle its own server issued.
        return self._lewlm.services.request_cancellation_registry.cancel(request_id, trusted_caller=True)

    def _lifecycle(self, model_id: str, *, operation: str) -> ModelLifecycleResponse:
        from lewlm.library import _run_sync

        async def execute() -> ModelLifecycleResponse:
            router = self._lewlm.services.model_router
            if operation == "warm":
                decision, result = await router.warm_model_lifecycle(model_id)
                status = "warmed"
            else:
                decision, result = await router.unload_model_lifecycle(model_id, drain=operation == "drain")
                status = "drained" if operation == "drain" else "unloaded"
            return ModelLifecycleResponse(
                status=status,
                model_id=decision.model_id,
                runtime=decision.runtime_name,
                reason=result.reason,
                **result.model_dump(exclude={"model_id", "runtime", "reason"}),
            )

        return _run_sync(execute, helper_name=f"LewLMAppClient.{operation}_model", async_name=operation)

    def list_tools(self) -> ToolListResponse:
        tools = self._lewlm.list_tools()
        return ToolListResponse(count=len(tools), items=tools)

    def get_tool(self, tool_name: str) -> LocalToolDescriptor:
        return self._lewlm.get_tool(tool_name)

    def execute_tool(self, payload: ToolExecutionRequest) -> ToolExecutionEnvelope:
        return self._lewlm.execute_tool(
            payload,
            actor="api",
            allowed_file_roots=self._lewlm.settings.file_access_roots,
        )

    def chat_completion(self, payload: ChatCompletionRequest) -> ChatCompletionResponse:
        if payload.stream:
            raise ValueError(
                "LewLMAppClient.chat_completion does not support stream=True. "
                "Use LewLM.stream_chat() or the raw /v1/chat/completions SSE API.",
            )
        from lewlm.library import _run_sync

        services = self._lewlm.services
        prompt_request = _prompt_request_from_payload(
            actor="api",
            system_prompt=payload.system_prompt,
            developer_prompt=payload.developer_prompt,
            pretext_path=payload.pretext_path,
            skills_path=payload.skills_path,
            response_format=payload.response_format,
            response_format_path=payload.response_format_path,
            output_schema=payload.output_schema,
            output_schema_path=payload.output_schema_path,
            tools=payload.tools,
            tools_path=payload.tools_path,
            mcp_tools=payload.mcp_tools,
            mcp_tools_path=payload.mcp_tools_path,
            include_trace=payload.include_prompt_trace,
        )

        async def run_completion() -> ChatCompletionResponse:
            input_messages = await normalize_chat_messages(payload.messages, services)
            messages = _merge_session_messages(services, payload.session_id, input_messages)
            execution = await services.chat_orchestrator.complete(
                model_id=payload.model,
                messages=messages,
                citation_context=payload.citation_context,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                apply_serving_profile=payload.apply_serving_profile,
                reasoning_visibility=_reasoning_visibility_from_request(
                    payload.reasoning_visibility,
                    services.settings.reasoning_visibility,
                ),
                prompt_request=prompt_request,
            )
            _persist_session_turn(
                services,
                session_id=payload.session_id,
                request_kind="chat.completions",
                input_messages=input_messages,
                output_text=execution.response.output_text,
                requested_model_id=payload.model,
                resolved_model_id=execution.response.model_id,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                finish_reason=execution.response.finish_reason,
                usage=execution.response.usage,
                metadata=execution.request_metadata,
            )
            return ChatCompletionResponse(
                id=execution.request_id,
                created=execution.created_at,
                model=execution.response.model_id,
                session_id=payload.session_id,
                choices=[
                    ChatCompletionChoice(
                        message=ChatCompletionChoiceMessage(
                            role="assistant",
                            content=execution.response.output_text,
                            reasoning=execution.response.reasoning,
                        ),
                        finish_reason=execution.response.finish_reason,
                    ),
                ],
                usage=_completion_usage(execution.response.usage),
                metadata=execution.metadata,
                citations=execution.response.citations,
                structured_output=getattr(execution, "structured_output", None),
                tool_calls=getattr(execution, "tool_calls", None),
                prompt_trace=execution.prompt_trace if payload.include_prompt_trace else None,
                serving_profile=execution.serving_profile,
            )

        return _run_sync(
            run_completion,
            helper_name="LewLMAppClient.chat_completion",
            async_name="LewLM.chat",
        )

    def responses(self, payload: ResponseCreateRequest) -> ResponseCreateResponse:
        if payload.stream:
            raise ValueError(
                "LewLMAppClient.responses does not support stream=True. "
                "Use LewLM.stream_chat() or the raw /v1/responses SSE API.",
            )
        from lewlm.library import _run_sync

        services = self._lewlm.services
        prompt_request = _prompt_request_from_payload(
            actor="api",
            system_prompt=payload.system_prompt,
            developer_prompt=payload.developer_prompt,
            pretext_path=payload.pretext_path,
            skills_path=payload.skills_path,
            response_format=payload.response_format,
            response_format_path=payload.response_format_path,
            output_schema=payload.output_schema,
            output_schema_path=payload.output_schema_path,
            tools=payload.tools,
            tools_path=payload.tools_path,
            mcp_tools=payload.mcp_tools,
            mcp_tools_path=payload.mcp_tools_path,
            include_trace=payload.include_prompt_trace,
        )

        async def run_completion() -> ResponseCreateResponse:
            from lewlm.api.message_normalization import normalize_response_input

            input_messages = await normalize_response_input(payload.input, services)
            messages = _merge_session_messages(services, payload.session_id, input_messages)
            execution = await services.chat_orchestrator.complete(
                model_id=payload.model,
                messages=messages,
                citation_context=payload.citation_context,
                max_tokens=payload.max_output_tokens,
                temperature=payload.temperature,
                apply_serving_profile=payload.apply_serving_profile,
                reasoning_visibility=_reasoning_visibility_from_request(
                    payload.reasoning_visibility,
                    services.settings.reasoning_visibility,
                ),
                prompt_request=prompt_request,
            )
            _persist_session_turn(
                services,
                session_id=payload.session_id,
                request_kind="responses",
                input_messages=input_messages,
                output_text=execution.response.output_text,
                requested_model_id=payload.model,
                resolved_model_id=execution.response.model_id,
                max_tokens=payload.max_output_tokens,
                temperature=payload.temperature,
                finish_reason=execution.response.finish_reason,
                usage=execution.response.usage,
                metadata=execution.request_metadata,
            )
            return ResponseCreateResponse(
                id=execution.request_id,
                created=execution.created_at,
                model=execution.response.model_id,
                session_id=payload.session_id,
                output=[
                    ResponseOutputText(
                        text=execution.response.output_text,
                        reasoning=execution.response.reasoning,
                    ),
                ],
                output_text=execution.response.output_text,
                usage=_completion_usage(execution.response.usage),
                metadata=execution.metadata,
                citations=execution.response.citations,
                structured_output=execution.structured_output,
                tool_calls=execution.tool_calls,
                prompt_trace=execution.prompt_trace if payload.include_prompt_trace else None,
                serving_profile=execution.serving_profile,
            )

        return _run_sync(
            run_completion,
            helper_name="LewLMAppClient.responses",
            async_name="LewLM.chat",
        )

    def embeddings(self, payload: EmbeddingCreateRequest) -> EmbeddingCreateResponse:
        from lewlm.library import _run_sync

        inputs = [payload.input] if isinstance(payload.input, str) else payload.input

        async def run_embeddings() -> EmbeddingCreateResponse:
            execution = await self._lewlm.services.multimodal_orchestrator.embed(
                model_id=payload.model,
                inputs=inputs,
            )
            return EmbeddingCreateResponse(
                request_id=execution.request_id,
                created=execution.created_at,
                data=[EmbeddingDatum(index=item.index, embedding=item.embedding) for item in execution.response.data],
                model=execution.response.model_id,
                usage=_completion_usage(execution.response.usage),
                routing=execution.routing,
                metadata=execution.metadata,
            )

        return _run_sync(
            run_embeddings,
            helper_name="LewLMAppClient.embeddings",
            async_name="LewLM.services.multimodal_orchestrator.embed",
        )

    def rerank(self, payload: RerankCreateRequest) -> RerankCreateResponse:
        from lewlm.library import _run_sync

        async def run_rerank() -> RerankCreateResponse:
            execution = await self._lewlm.services.multimodal_orchestrator.rerank(
                model_id=payload.model,
                query=payload.query,
                documents=payload.documents,
                top_n=payload.top_n,
            )
            return RerankCreateResponse(
                request_id=execution.request_id,
                created=execution.created_at,
                model=execution.response.model_id,
                results=[
                    RerankResultItem(
                        index=item.index,
                        relevance_score=item.relevance_score,
                        document=item.document,
                    )
                    for item in execution.response.results
                ],
                routing=execution.routing,
                metadata=execution.metadata,
            )

        return _run_sync(
            run_rerank,
            helper_name="LewLMAppClient.rerank",
            async_name="LewLM.services.multimodal_orchestrator.rerank",
        )

    def retrieve_context(self, payload: RetrievalContextRequest) -> RetrievalContextResponse:
        from lewlm.library import _run_sync

        async def run_retrieval() -> RetrievalContextResponse:
            execution = await self._lewlm.services.multimodal_orchestrator.retrieve_context(
                query=payload.query,
                candidate_chunks=payload.candidate_chunks,
                candidate_sources=payload.candidate_sources,
                top_k=payload.top_k,
                use_embeddings=payload.use_embeddings,
                use_rerank=payload.use_rerank,
                embedding_model_id=payload.embedding_model,
                rerank_model_id=payload.rerank_model,
            )
            return RetrievalContextResponse(
                request_id=execution.request_id,
                created=execution.created_at,
                query=execution.query,
                strategy=execution.strategy,
                candidate_count=len(payload.candidate_chunks),
                returned_count=len(execution.items),
                items=[
                    RetrievalContextItem(
                        rank=item.rank,
                        score=item.score,
                        embedding_score=item.embedding_score,
                        rerank_score=item.rerank_score,
                        chunk=item.chunk,
                        source=item.source,
                    )
                    for item in execution.items
                ],
                sources=execution.sources,
                scoring_policy=execution.scoring_policy,
                embedding_stage=_retrieval_stage_summary(execution.embedding_stage),
                rerank_stage=_retrieval_stage_summary(execution.rerank_stage),
                metadata=execution.metadata,
            )

        return _run_sync(
            run_retrieval,
            helper_name="LewLMAppClient.retrieve_context",
            async_name="LewLM.services.multimodal_orchestrator.retrieve_context",
        )

    def transcribe_audio(self, payload: AudioTranscriptionCreateRequest) -> AudioTranscriptionCreateResponse:
        from lewlm.library import _run_sync

        async def run_transcription() -> AudioTranscriptionCreateResponse:
            execution = await self._lewlm.services.multimodal_orchestrator.transcribe_audio(
                model_id=payload.model,
                audio_bytes=_decode_audio_bytes(payload.audio_base64, file_name=payload.file_name),
                file_name=payload.file_name,
                language=payload.language,
                prompt=payload.prompt,
            )
            return AudioTranscriptionCreateResponse(
                request_id=execution.request_id,
                created=execution.created_at,
                model=execution.response.model_id,
                text=execution.response.text,
                language=execution.response.language,
                duration_seconds=execution.response.duration_seconds,
                segments=[
                    AudioTranscriptionSegment(
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        text=segment.text,
                    )
                    for segment in execution.response.segments
                ],
                routing=execution.routing,
                metadata=execution.metadata,
            )

        return _run_sync(
            run_transcription,
            helper_name="LewLMAppClient.transcribe_audio",
            async_name="LewLM.services.multimodal_orchestrator.transcribe_audio",
        )

    def synthesize_speech(self, payload: AudioSpeechCreateRequest) -> AudioSpeechCreateResponse:
        from lewlm.library import _run_sync

        async def run_speech() -> AudioSpeechCreateResponse:
            execution = await self._lewlm.services.multimodal_orchestrator.synthesize_speech(
                model_id=payload.model,
                input_text=payload.input,
                voice=payload.voice,
                audio_format=payload.format,
            )
            return AudioSpeechCreateResponse(
                request_id=execution.request_id,
                created=execution.created_at,
                model=execution.response.model_id,
                media_type=execution.response.media_type,
                content_type=execution.response.media_type,
                audio_base64=base64.b64encode(execution.response.audio_bytes).decode("ascii"),
                voice=execution.response.voice,
                duration_seconds=execution.response.duration_seconds,
                routing=execution.routing,
                metadata=execution.metadata,
            )

        return _run_sync(
            run_speech,
            helper_name="LewLMAppClient.synthesize_speech",
            async_name="LewLM.services.multimodal_orchestrator.synthesize_speech",
        )

    def ingest_documents(self, payload: DocumentIngestRequest) -> DocumentIngestResponse:
        services = self._lewlm.services
        envelope = services.tool_execution_service.execute(
            DocumentIngestToolRequest(
                input=IngestDocumentToolInput(
                    paths=payload.paths,
                    sources=payload.sources,
                    title=payload.title,
                    authorized_actions=payload.authorized_actions,
                    idempotency_key=payload.idempotency_key,
                    correlation_id=payload.correlation_id,
                ),
            ),
            actor="app_client",
            allowed_file_roots=services.settings.file_access_roots,
            emit_tool_events=False,
        )
        return DocumentIngestResponse.model_validate(
            {
                **envelope.result,
                "request_id": envelope.request_id,
                "idempotency_key": envelope.idempotency_key,
                "idempotent_replay": envelope.idempotent_replay,
                "metadata": self._tool_metadata(
                    envelope,
                    correlation_id=payload.correlation_id,
                    components=envelope.result.get("components"),
                ).model_dump(mode="json"),
            },
        )

    def update_session(self, session_id: str, payload: SessionUpdateRequest) -> SessionRecord:
        return self._lewlm.services.session_history_service.update_session(
            session_id,
            title=payload.title,
            metadata=payload.metadata,
            context_policy=payload.context_policy,
            merge_metadata=not payload.replace_metadata,
        )

    def generate_document(self, payload: DocumentGenerateRequest) -> DocumentGenerateResponse:
        services = self._lewlm.services
        envelope = services.tool_execution_service.execute(
            DocumentGenerateToolRequest(
                input=GenerateDocumentToolInput(
                    output_format=payload.output_format,
                    document=payload.document,
                    file_name=payload.file_name,
                    authorized_actions=payload.authorized_actions,
                    idempotency_key=payload.idempotency_key,
                    correlation_id=payload.correlation_id,
                ),
            ),
            actor="app_client",
            allowed_file_roots=services.settings.file_access_roots,
            emit_tool_events=False,
        )
        return self._render_response(
            DocumentGenerateResponse,
            envelope,
            correlation_id=payload.correlation_id,
        )

    def transform_document(self, payload: DocumentTransformRequest) -> DocumentTransformResponse:
        services = self._lewlm.services
        envelope = services.tool_execution_service.execute(
            DocumentTransformToolRequest(input=payload),
            actor="app_client",
            allowed_file_roots=services.settings.file_access_roots,
            emit_tool_events=False,
        )
        return self._render_response(
            DocumentTransformResponse,
            envelope,
            correlation_id=payload.correlation_id,
            extra={"skill": payload.skill},
        )

    def count_tokens(self, payload: TokenCountRequest) -> TokenCountResponse:
        from lewlm.library import _run_sync

        async def run_count() -> TokenCountResponse:
            execution = await self._lewlm.services.tokenization_service.count_tokens(
                model_id=payload.model,
                text=payload.text,
                max_tokens=payload.max_tokens,
                correlation_id=payload.correlation_id,
            )
            return TokenCountResponse(
                request_id=execution.request_id,
                created=execution.created_at,
                model=execution.model_id,
                token_count=execution.token_count,
                character_count=execution.character_count,
                truncated=execution.truncated,
                truncated_text=execution.truncated_text,
                truncated_token_count=execution.truncated_token_count,
                routing=execution.routing,
                metadata=execution.metadata,
            )

        return _run_sync(
            run_count,
            helper_name="LewLMAppClient.count_tokens",
            async_name="LewLM.services.tokenization_service.count_tokens",
        )

    @staticmethod
    def _tool_metadata(envelope, *, correlation_id: str | None, components=None):
        return build_tool_execution_metadata(
            request_id=envelope.request_id,
            created=int(envelope.trace.started_at.timestamp()),
            tool_name=envelope.tool,
            duration_milliseconds=envelope.trace.duration_ms,
            idempotency_key=envelope.idempotency_key,
            idempotent_replay=envelope.idempotent_replay,
            correlation_id=correlation_id,
            components=[ComponentProvenance.model_validate(item) for item in (components or ())],
        )

    def _render_response(self, response_type, envelope, *, correlation_id: str | None, extra: dict | None = None):
        return response_type.model_validate(
            {
                **(extra or {}),
                "request_id": envelope.request_id,
                "idempotency_key": envelope.idempotency_key,
                "idempotent_replay": envelope.idempotent_replay,
                "file_name": envelope.result["file_name"],
                "output_format": envelope.result["output_format"],
                "media_type": envelope.result["media_type"],
                "size_bytes": envelope.result["size_bytes"],
                "content_base64": envelope.result["content_base64"],
                "metadata": self._tool_metadata(
                    envelope,
                    correlation_id=correlation_id,
                    components=[renderer_provenance(str(envelope.result["output_format"])).model_dump(mode="json")],
                ).model_dump(mode="json"),
            },
        )


def _read_bounded(stream: Any, *, limit: int, url: str, operation: str | None = None) -> bytes:
    """Read at most `limit` bytes, failing rather than buffering without bound."""

    # Reading limit+1 makes an over-long body detectable without reading it all.
    payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise LewLMAppClientResponseTooLargeError(url=url, limit_bytes=limit, operation=operation)
    return payload


def _truncated_error_body(body_text: str | None) -> str | None:
    """Bound an error body so an unexpected payload cannot be retained whole."""

    if body_text is None:
        return None
    if len(body_text) <= MAX_ERROR_BODY_CHARACTERS:
        return body_text
    return f"{body_text[:MAX_ERROR_BODY_CHARACTERS]}… [truncated {len(body_text) - MAX_ERROR_BODY_CHARACTERS} characters]"


class _HttpAppClientBackend:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        application_id: str | None,
        client_instance_id: str,
        authorized_actions: Sequence[str] | None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_response_bytes_by_operation: Mapping[str, int] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._application_id = application_id
        self._client_instance_id = client_instance_id
        self._authorized_actions = tuple(authorized_actions or ())
        self._max_response_bytes = max_response_bytes
        self._max_response_bytes_by_operation = validate_operation_limits(max_response_bytes_by_operation)
        self._correlation_id = correlation_id

    def _limit_for(self, operation: str) -> int:
        return self._max_response_bytes_by_operation.get(operation, self._max_response_bytes)

    def health(self) -> HealthResponse:
        return self._request_json("GET", "/v1/health", operation="health", response_type=HealthResponse)

    def runtime_stats(self) -> RuntimeStats:
        return self._request_json("GET", "/v1/runtime/stats", operation="runtime_stats", response_type=RuntimeStats)

    def runtime_info(self) -> RuntimeInfo:
        return self._request_json("GET", "/v1/runtime", operation="runtime_info", response_type=RuntimeInfo)

    def list_model_residencies(self) -> list[ModelResidencySnapshot]:
        payload = self._request_json(
            "GET",
            "/v1/runtime/residencies",
            operation="list_model_residencies",
            response_type=_ModelResidencyList,
        )
        return payload.root

    def get_model_residency(self, model_id: str, runtime: str | None = None) -> ModelResidencySnapshot | None:
        suffix = f"?runtime={runtime}" if runtime else ""
        return self._request_json(
            "GET",
            f"/v1/models/{model_id}/residency{suffix}",
            operation="get_model_residency",
            response_type=_OptionalModelResidency,
        ).root

    def warm_model(self, model_id: str) -> ModelLifecycleResponse:
        return self._request_json(
            "POST",
            f"/v1/models/{model_id}/warm",
            operation="warm_model",
            response_type=ModelLifecycleResponse,
        )

    def unload_model(self, model_id: str) -> ModelLifecycleResponse:
        return self._request_json(
            "POST",
            f"/v1/models/{model_id}/unload",
            operation="unload_model",
            response_type=ModelLifecycleResponse,
        )

    def drain_model(self, model_id: str) -> ModelLifecycleResponse:
        return self._request_json(
            "POST",
            f"/v1/models/{model_id}/drain",
            operation="drain_model",
            response_type=ModelLifecycleResponse,
        )

    def create_drain_operation(
        self,
        model_id: str,
        *,
        timeout_seconds: float | None = None,
        idempotency_key: str | None = None,
    ) -> LifecycleOperationRecord:
        return self._request_json(
            "POST",
            f"/v1/models/{model_id}/drain-operations",
            payload=AsyncDrainRequest(timeout_seconds=timeout_seconds, idempotency_key=idempotency_key),
            operation="create_drain_operation",
            response_type=LifecycleOperationRecord,
        )

    def get_lifecycle_operation(self, operation_id: str) -> LifecycleOperationRecord:
        return self._request_json(
            "GET",
            f"/v1/model-lifecycle/operations/{operation_id}",
            operation="get_lifecycle_operation",
            response_type=LifecycleOperationRecord,
        )

    def cancel_lifecycle_operation(self, operation_id: str) -> LifecycleOperationRecord:
        return self._request_json(
            "DELETE",
            f"/v1/model-lifecycle/operations/{operation_id}",
            operation="cancel_lifecycle_operation",
            response_type=LifecycleOperationRecord,
        )

    def cancel_request(self, request_id: str) -> RequestCancellationRecord:
        request_id = validate_request_handle(request_id)
        return self._request_json(
            "POST",
            f"/v1/requests/{request_id}/cancel",
            operation="cancel_request",
            response_type=RequestCancellationRecord,
        )

    def list_tools(self) -> ToolListResponse:
        return self._request_json("GET", "/v1/tools", operation="list_tools", response_type=ToolListResponse)

    def get_tool(self, tool_name: str) -> LocalToolDescriptor:
        return self._request_json(
            "GET",
            f"/v1/tools/{tool_name}",
            operation="get_tool",
            response_type=LocalToolDescriptor,
        )

    def execute_tool(self, payload: ToolExecutionRequest) -> ToolExecutionEnvelope:
        return self._request_json(
            "POST",
            "/v1/tools/execute",
            payload=payload,
            operation="execute_tool",
            response_type=ToolExecutionEnvelope,
        )

    def chat_completion(self, payload: ChatCompletionRequest) -> ChatCompletionResponse:
        if payload.stream:
            raise ValueError(
                "LewLMAppClient.chat_completion does not support stream=True. "
                "Use the raw /v1/chat/completions SSE API when you need streaming.",
            )
        return self._request_json(
            "POST",
            "/v1/chat/completions",
            payload=payload,
            operation="chat_completion",
            response_type=ChatCompletionResponse,
        )

    def responses(self, payload: ResponseCreateRequest) -> ResponseCreateResponse:
        if payload.stream:
            raise ValueError(
                "LewLMAppClient.responses does not support stream=True. "
                "Use the raw /v1/responses SSE API when you need streaming.",
            )
        return self._request_json(
            "POST",
            "/v1/responses",
            payload=payload,
            operation="responses",
            response_type=ResponseCreateResponse,
        )

    def embeddings(self, payload: EmbeddingCreateRequest) -> EmbeddingCreateResponse:
        return self._request_json(
            "POST",
            "/v1/embeddings",
            payload=payload,
            operation="embeddings",
            response_type=EmbeddingCreateResponse,
        )

    def rerank(self, payload: RerankCreateRequest) -> RerankCreateResponse:
        return self._request_json(
            "POST",
            "/v1/rerank",
            payload=payload,
            operation="rerank",
            response_type=RerankCreateResponse,
        )

    def retrieve_context(self, payload: RetrievalContextRequest) -> RetrievalContextResponse:
        return self._request_json(
            "POST",
            "/v1/retrieval/context",
            payload=payload,
            operation="retrieve_context",
            response_type=RetrievalContextResponse,
        )

    def transcribe_audio(self, payload: AudioTranscriptionCreateRequest) -> AudioTranscriptionCreateResponse:
        return self._request_json(
            "POST",
            "/v1/audio/transcriptions",
            payload=payload,
            operation="transcribe_audio",
            response_type=AudioTranscriptionCreateResponse,
        )

    def synthesize_speech(self, payload: AudioSpeechCreateRequest) -> AudioSpeechCreateResponse:
        return self._request_json(
            "POST",
            "/v1/audio/speech",
            payload=payload,
            operation="synthesize_speech",
            response_type=AudioSpeechCreateResponse,
        )

    def ingest_documents(self, payload: DocumentIngestRequest) -> DocumentIngestResponse:
        return self._request_json(
            "POST",
            "/v1/documents/ingest",
            payload=payload,
            operation="ingest_documents",
            response_type=DocumentIngestResponse,
        )

    def update_session(self, session_id: str, payload: SessionUpdateRequest) -> SessionRecord:
        return self._request_json(
            "PATCH",
            f"/v1/sessions/{session_id}",
            payload=payload,
            operation="update_session",
            response_type=SessionRecord,
        )

    def generate_document(self, payload: DocumentGenerateRequest) -> DocumentGenerateResponse:
        return self._request_json(
            "POST",
            "/v1/documents/generate",
            payload=payload,
            operation="generate_document",
            response_type=DocumentGenerateResponse,
        )

    def transform_document(self, payload: DocumentTransformRequest) -> DocumentTransformResponse:
        return self._request_json(
            "POST",
            "/v1/documents/transform",
            payload=payload,
            operation="transform_document",
            response_type=DocumentTransformResponse,
        )

    def count_tokens(self, payload: TokenCountRequest) -> TokenCountResponse:
        return self._request_json(
            "POST",
            "/v1/tokenize/count",
            payload=payload,
            operation="count_tokens",
            response_type=TokenCountResponse,
        )

    def _request_json(self, method: str, path: str, *, operation: str, response_type, payload=None):
        url = f"{self._base_url}{path}"
        limit = self._limit_for(operation)
        headers = {"accept": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        if self._application_id:
            headers["x-lewlm-application-id"] = self._application_id
        headers["x-lewlm-client-instance-id"] = self._client_instance_id
        if self._authorized_actions:
            headers["x-lewlm-authorized-actions"] = ",".join(self._authorized_actions)
        if self._correlation_id:
            headers["x-lewlm-correlation-id"] = self._correlation_id
        body = None
        if payload is not None:
            headers["content-type"] = "application/json"
            body = payload.model_dump_json(exclude_none=True, by_alias=True).encode("utf-8")
        request = Request(url=url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                raw = _read_bounded(response, limit=limit, url=url, operation=operation)
        except HTTPError as exc:
            # An error body is diagnostic only, so it is read under a much
            # tighter bound than a successful response.
            body_text = (
                _read_bounded(exc, limit=MAX_ERROR_BODY_CHARACTERS * 4, url=url)
                .decode("utf-8", errors="replace")
                .strip()
                or None
            )
            raise LewLMAppClientHTTPError(
                url=url,
                status_code=exc.code,
                body=_truncated_error_body(body_text),
                api_error=_parse_http_error_payload(body_text, status_code=exc.code),
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"LewLM app client could not reach {url}: {exc.reason}") from exc
        return response_type.model_validate_json(raw)


def _parse_http_error_payload(body_text: str | None, *, status_code: int) -> LewLMError | None:
    if not body_text:
        return None
    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error_payload = payload.get("error")
    if not isinstance(error_payload, dict):
        return None
    normalized_payload = dict(error_payload)
    normalized_payload.setdefault("status_code", status_code)
    return error_from_dict(normalized_payload)


def _normalize_paths(paths: Sequence[Path | str] | Path | str | None) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [str(paths)]
    return [str(path) for path in paths]


def _retrieval_stage_summary(stage) -> RetrievalStageSummary | None:
    if stage is None:
        return None
    return RetrievalStageSummary(
        request_id=stage.request_id,
        created=stage.created_at,
        model=stage.model_id,
        routing=stage.routing,
        metadata=stage.metadata,
        usage=_completion_usage(stage.usage) if stage.usage is not None else None,
    )


__all__ = ["LewLMAppClient", "LewLMAppClientHTTPError"]
