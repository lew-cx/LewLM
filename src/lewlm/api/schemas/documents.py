"""Document API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from lewlm.core.execution_metadata import ExecutionMetadata
from lewlm.documents.ingest.models import DocumentIngestResult
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.documents.skills.models import DocumentTransformRequest
from lewlm.tools.models import UploadedSourceToolInput


class DocumentUploadSource(UploadedSourceToolInput):
    """A document uploaded as bytes, requiring no shared filesystem mount."""


class DocumentIngestRequest(BaseModel):
    paths: list[str] = Field(
        default_factory=list,
        description="Server-local paths. Only usable when the caller shares LewLM's filesystem.",
    )
    sources: list[DocumentUploadSource] = Field(
        default_factory=list,
        description="Uploaded byte sources with caller-owned identity. Preferred for remote callers.",
    )
    title: str | None = None
    authorized_actions: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None
    correlation_id: str | None = Field(
        default=None,
        description="Caller correlation identifier echoed back through metadata and events.",
    )

    @model_validator(mode="after")
    def _require_a_source(self) -> "DocumentIngestRequest":
        if not self.paths and not self.sources:
            raise ValueError("Document ingest requires at least one path or uploaded source.")
        return self


class DocumentGenerateRequest(BaseModel):
    output_format: DocumentOutputFormat
    document: DocumentIR
    file_name: str | None = None
    authorized_actions: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None
    correlation_id: str | None = Field(
        default=None,
        description="Caller correlation identifier echoed back through metadata and events.",
    )


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
