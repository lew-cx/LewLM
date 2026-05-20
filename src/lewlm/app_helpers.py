"""Thin typed helper layer for embedded apps and local-server clients."""

from __future__ import annotations

import base64
from collections.abc import Sequence
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
from lewlm.api.schemas.documents import DocumentIngestRequest, DocumentIngestResponse
from lewlm.api.schemas.tools import ToolListResponse
from lewlm.core.citations import CitationContextPackage
from lewlm.api.schemas.health import HealthResponse
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
)
from lewlm.core.contracts import ReasoningVisibility
from lewlm.core.errors import LewLMError, error_from_dict
from lewlm.core.execution_metadata import build_tool_execution_metadata
from lewlm.documents.ingest.models import DocumentChunk, IngestedDocumentSource
from lewlm.structured_output import StructuredOutputRequest
from lewlm.telemetry.stats import RuntimeStats
from lewlm.tools.models import (
    DocumentIngestToolRequest,
    IngestDocumentToolInput,
    LocalToolDescriptor,
    ToolExecutionEnvelope,
    ToolExecutionRequest,
)

if TYPE_CHECKING:
    from lewlm.library import LewLM


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


class _AppClientBackend(Protocol):
    def health(self) -> HealthResponse: ...

    def runtime_stats(self) -> RuntimeStats: ...

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
    ) -> Self:
        """Bind the helper to a running local LewLM HTTP server."""

        return cls(
            _HttpAppClientBackend(
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            ),
        )

    def health(self) -> HealthResponse:
        """Return the typed health snapshot."""

        return self._backend.health()

    def runtime_stats(self) -> RuntimeStats:
        """Return typed runtime diagnostics."""

        return self._backend.runtime_stats()

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
        title: str | None = None,
        authorized_actions: Sequence[str] | None = None,
        idempotency_key: str | None = None,
    ) -> DocumentIngestResponse:
        """Ingest local files using the same typed request and response shape as the API."""

        if request is not None and any((paths is not None, title is not None, authorized_actions is not None, idempotency_key is not None)):
            raise ValueError("Pass either `request` or keyword arguments to ingest_documents(), not both.")
        if request is None and paths is None:
            raise ValueError("paths is required when request is not provided.")
        payload = request or DocumentIngestRequest(
            paths=_normalize_paths(paths),
            title=title,
            authorized_actions=list(authorized_actions or ()),
            idempotency_key=idempotency_key,
        )
        return self._backend.ingest_documents(payload)


class _EmbeddedAppClientBackend:
    def __init__(self, lewlm: LewLM) -> None:
        self._lewlm = lewlm

    def health(self) -> HealthResponse:
        return HealthResponse.model_validate(self._lewlm.health())

    def runtime_stats(self) -> RuntimeStats:
        return self._lewlm.runtime_stats_sync()

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
                    title=payload.title,
                    authorized_actions=payload.authorized_actions,
                    idempotency_key=payload.idempotency_key,
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
                "metadata": build_tool_execution_metadata(
                    request_id=envelope.request_id,
                    created=int(envelope.trace.started_at.timestamp()),
                    tool_name=envelope.tool,
                    duration_milliseconds=envelope.trace.duration_ms,
                    idempotency_key=envelope.idempotency_key,
                    idempotent_replay=envelope.idempotent_replay,
                ).model_dump(mode="json"),
            },
        )


class _HttpAppClientBackend:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def health(self) -> HealthResponse:
        return self._request_json("GET", "/v1/health", response_type=HealthResponse)

    def runtime_stats(self) -> RuntimeStats:
        return self._request_json("GET", "/v1/runtime/stats", response_type=RuntimeStats)

    def list_tools(self) -> ToolListResponse:
        return self._request_json("GET", "/v1/tools", response_type=ToolListResponse)

    def get_tool(self, tool_name: str) -> LocalToolDescriptor:
        return self._request_json("GET", f"/v1/tools/{tool_name}", response_type=LocalToolDescriptor)

    def execute_tool(self, payload: ToolExecutionRequest) -> ToolExecutionEnvelope:
        return self._request_json(
            "POST",
            "/v1/tools/execute",
            payload=payload,
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
            response_type=ResponseCreateResponse,
        )

    def embeddings(self, payload: EmbeddingCreateRequest) -> EmbeddingCreateResponse:
        return self._request_json(
            "POST",
            "/v1/embeddings",
            payload=payload,
            response_type=EmbeddingCreateResponse,
        )

    def rerank(self, payload: RerankCreateRequest) -> RerankCreateResponse:
        return self._request_json(
            "POST",
            "/v1/rerank",
            payload=payload,
            response_type=RerankCreateResponse,
        )

    def retrieve_context(self, payload: RetrievalContextRequest) -> RetrievalContextResponse:
        return self._request_json(
            "POST",
            "/v1/retrieval/context",
            payload=payload,
            response_type=RetrievalContextResponse,
        )

    def transcribe_audio(self, payload: AudioTranscriptionCreateRequest) -> AudioTranscriptionCreateResponse:
        return self._request_json(
            "POST",
            "/v1/audio/transcriptions",
            payload=payload,
            response_type=AudioTranscriptionCreateResponse,
        )

    def synthesize_speech(self, payload: AudioSpeechCreateRequest) -> AudioSpeechCreateResponse:
        return self._request_json(
            "POST",
            "/v1/audio/speech",
            payload=payload,
            response_type=AudioSpeechCreateResponse,
        )

    def ingest_documents(self, payload: DocumentIngestRequest) -> DocumentIngestResponse:
        return self._request_json(
            "POST",
            "/v1/documents/ingest",
            payload=payload,
            response_type=DocumentIngestResponse,
        )

    def _request_json(self, method: str, path: str, *, response_type, payload=None):
        url = f"{self._base_url}{path}"
        headers = {"accept": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        body = None
        if payload is not None:
            headers["content-type"] = "application/json"
            body = payload.model_dump_json(exclude_none=True, by_alias=True).encode("utf-8")
        request = Request(url=url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace").strip() or None
            raise LewLMAppClientHTTPError(
                url=url,
                status_code=exc.code,
                body=body_text,
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
