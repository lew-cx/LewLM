"""Embeddings, rerank, and audio routes."""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, Request
from starlette.datastructures import UploadFile

from lewlm.api.dependencies import get_services
from lewlm.api.schemas.chat import CompletionUsage
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
from lewlm.core.errors import ConfigurationError, UnsupportedMediaTypeError
from lewlm.security.files import validate_audio_bytes


router = APIRouter(tags=["multimodal"])

_RETRIEVAL_JSON_EXAMPLE = {
    "query": "typed helper for host applications",
    "candidate_sources": [
        {
            "source_id": "source-1",
            "path": "/tmp/app-notes.md",
            "source_type": "markdown",
            "source_name": "app-notes.md",
            "source_label": "app-notes.md",
        },
    ],
    "candidate_chunks": [
        {
            "chunk_id": "chunk-1",
            "text": "LewLM exposes typed helper methods for host applications.",
            "source_id": "source-1",
            "section_id": "section-1",
            "source_label": "app-notes.md",
            "section_label": "app-notes.md / Section 1",
        },
        {
            "chunk_id": "chunk-2",
            "text": "This unrelated note is about weather forecasts.",
            "source_id": "source-1",
            "section_id": "section-2",
            "source_label": "app-notes.md",
            "section_label": "app-notes.md / Section 2",
        },
    ],
    "top_k": 1,
}


@router.post("/v1/embeddings", response_model=EmbeddingCreateResponse)
async def create_embeddings(payload: EmbeddingCreateRequest, request: Request) -> EmbeddingCreateResponse:
    """Create embeddings with a compatible local model."""

    services = get_services(request)
    inputs = [payload.input] if isinstance(payload.input, str) else payload.input
    execution = await services.multimodal_orchestrator.embed(model_id=payload.model, inputs=inputs)
    return EmbeddingCreateResponse(
        request_id=execution.request_id,
        created=execution.created_at,
        data=[
            EmbeddingDatum(index=item.index, embedding=item.embedding)
            for item in execution.response.data
        ],
        model=execution.response.model_id,
        usage=_completion_usage(execution.response.usage),
        routing=execution.routing,
        metadata=execution.metadata,
    )


@router.post("/v1/rerank", response_model=RerankCreateResponse)
async def rerank_documents(payload: RerankCreateRequest, request: Request) -> RerankCreateResponse:
    """Rerank candidate documents with a compatible local model."""

    services = get_services(request)
    execution = await services.multimodal_orchestrator.rerank(
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


@router.post(
    "/v1/retrieval/context",
    response_model=RetrievalContextResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": RetrievalContextRequest.model_json_schema(),
                    "example": _RETRIEVAL_JSON_EXAMPLE,
                },
            },
        },
    },
)
async def retrieve_context(payload: RetrievalContextRequest, request: Request) -> RetrievalContextResponse:
    """Rank caller-provided candidate chunks into reusable retrieval context."""

    services = get_services(request)
    execution = await services.multimodal_orchestrator.retrieve_context(
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


@router.post("/v1/audio/transcriptions", response_model=AudioTranscriptionCreateResponse)
async def transcribe_audio(request: Request) -> AudioTranscriptionCreateResponse:
    """Transcribe a JSON or multipart audio payload with a compatible local model."""

    services = get_services(request)
    content_type = request.headers.get("content-type", "")
    media_type = content_type.partition(";")[0].strip().casefold()
    if media_type == "multipart/form-data":
        model_id, audio_bytes, file_name, language, prompt = await _parse_multipart_transcription_request(request)
    else:
        payload = AudioTranscriptionCreateRequest.model_validate_json(await request.body())
        model_id = payload.model
        file_name = payload.file_name
        language = payload.language
        prompt = payload.prompt
        audio_bytes = _decode_audio_bytes(payload.audio_base64, file_name=file_name)
    execution = await services.multimodal_orchestrator.transcribe_audio(
        model_id=model_id,
        audio_bytes=audio_bytes,
        file_name=file_name,
        language=language,
        prompt=prompt,
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


@router.post("/v1/audio/speech", response_model=AudioSpeechCreateResponse)
async def synthesize_speech(payload: AudioSpeechCreateRequest, request: Request) -> AudioSpeechCreateResponse:
    """Generate speech audio from text with a compatible local model."""

    services = get_services(request)
    execution = await services.multimodal_orchestrator.synthesize_speech(
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


def _completion_usage(raw_usage: dict[str, int]) -> CompletionUsage:
    prompt_tokens = raw_usage.get("prompt_tokens", 0)
    completion_tokens = raw_usage.get("completion_tokens", 0)
    total_tokens = raw_usage.get("total_tokens", prompt_tokens + completion_tokens)
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


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


def _decode_audio_bytes(encoded: str, *, file_name: str) -> bytes:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise UnsupportedMediaTypeError(
            "Audio input must be valid base64.",
            details={"file_name": file_name},
        ) from exc
    validate_audio_bytes(raw, purpose="Audio input", file_name=file_name)
    return raw


async def _parse_multipart_transcription_request(request: Request) -> tuple[str | None, bytes, str, str | None, str | None]:
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise ConfigurationError("Multipart audio transcription requests require a `file` upload.")
    raw = await upload.read()
    await upload.close()
    file_name = upload.filename or "audio"
    validate_audio_bytes(raw, purpose="Audio input", file_name=file_name)
    model_id = form.get("model")
    language = form.get("language")
    prompt = form.get("prompt")
    if model_id is not None and not isinstance(model_id, str):
        raise ConfigurationError("The multipart `model` field must be a string when provided.")
    if language is not None and not isinstance(language, str):
        raise ConfigurationError("The multipart `language` field must be a string when provided.")
    if prompt is not None and not isinstance(prompt, str):
        raise ConfigurationError("The multipart `prompt` field must be a string when provided.")
    return model_id, raw, file_name, language, prompt
