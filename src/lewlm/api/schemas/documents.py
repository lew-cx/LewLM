"""Document API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from lewlm.core.execution_metadata import ExecutionMetadata
from lewlm.documents.ingest.models import DocumentIngestResult
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.documents.skills.models import DocumentTransformRequest


class DocumentIngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    title: str | None = None
    authorized_actions: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None


class DocumentGenerateRequest(BaseModel):
    output_format: DocumentOutputFormat
    document: DocumentIR
    file_name: str | None = None
    authorized_actions: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None


class DocumentGenerateResponse(BaseModel):
    request_id: str
    idempotency_key: str | None = None
    idempotent_replay: bool = False
    file_name: str
    output_format: DocumentOutputFormat
    media_type: str
    size_bytes: int
    content_base64: str = Field(description="Base64-encoded artifact payload.")
    metadata: ExecutionMetadata


class DocumentTransformResponse(DocumentGenerateResponse):
    skill: str


class DocumentIngestResponse(DocumentIngestResult):
    request_id: str
    idempotency_key: str | None = None
    idempotent_replay: bool = False
    metadata: ExecutionMetadata
