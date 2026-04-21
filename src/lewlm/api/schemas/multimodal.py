"""Schemas for embeddings, rerank, and audio APIs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from lewlm.core.contracts import RoutingDecision
from lewlm.core.execution_metadata import ExecutionMetadata
from lewlm.documents.ingest.models import DocumentChunk, IngestedDocumentSource

from .chat import CompletionUsage


class EmbeddingCreateRequest(BaseModel):
    model: str | None = None
    input: str | list[str]


class EmbeddingDatum(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingCreateResponse(BaseModel):
    request_id: str
    created: int
    object: Literal["list"] = "list"
    data: list[EmbeddingDatum]
    model: str
    usage: CompletionUsage = Field(default_factory=CompletionUsage)
    routing: RoutingDecision
    metadata: ExecutionMetadata


class RerankCreateRequest(BaseModel):
    model: str | None = None
    query: str
    documents: list[str]
    top_n: int | None = Field(default=None, ge=1)


class RerankResultItem(BaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankCreateResponse(BaseModel):
    request_id: str
    created: int
    model: str
    results: list[RerankResultItem]
    routing: RoutingDecision
    metadata: ExecutionMetadata


class RetrievalContextRequest(BaseModel):
    query: str
    candidate_chunks: list[DocumentChunk]
    candidate_sources: list[IngestedDocumentSource] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1)
    use_embeddings: bool = True
    use_rerank: bool = True
    embedding_model: str | None = None
    rerank_model: str | None = None

    @model_validator(mode="after")
    def _validate_strategy(self) -> "RetrievalContextRequest":
        if not self.use_embeddings and not self.use_rerank:
            raise ValueError("Retrieval requests must enable embeddings, rerank, or both.")
        if not self.candidate_chunks:
            raise ValueError("Retrieval requests require at least one candidate chunk.")
        return self


class RetrievalStageSummary(BaseModel):
    request_id: str
    created: int
    model: str
    routing: RoutingDecision
    metadata: ExecutionMetadata
    usage: CompletionUsage | None = None


class RetrievalContextItem(BaseModel):
    rank: int
    score: float
    embedding_score: float | None = None
    rerank_score: float | None = None
    chunk: DocumentChunk
    source: IngestedDocumentSource | None = None


class RetrievalContextResponse(BaseModel):
    request_id: str
    created: int
    query: str
    strategy: Literal["hybrid", "embeddings", "rerank"]
    candidate_count: int
    returned_count: int
    items: list[RetrievalContextItem]
    sources: list[IngestedDocumentSource] = Field(default_factory=list)
    embedding_stage: RetrievalStageSummary | None = None
    rerank_stage: RetrievalStageSummary | None = None
    metadata: ExecutionMetadata


class AudioTranscriptionCreateRequest(BaseModel):
    model: str | None = None
    audio_base64: str
    file_name: str = "audio.wav"
    language: str | None = None
    prompt: str | None = None


class AudioTranscriptionSegment(BaseModel):
    start_seconds: float | None = None
    end_seconds: float | None = None
    text: str


class AudioTranscriptionCreateResponse(BaseModel):
    request_id: str
    created: int
    model: str
    text: str
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[AudioTranscriptionSegment] = Field(default_factory=list)
    routing: RoutingDecision
    metadata: ExecutionMetadata


class AudioSpeechCreateRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    format: str = "wav"


class AudioSpeechCreateResponse(BaseModel):
    request_id: str
    created: int
    model: str
    media_type: str
    content_type: str
    audio_base64: str
    voice: str | None = None
    duration_seconds: float | None = None
    routing: RoutingDecision
    metadata: ExecutionMetadata
