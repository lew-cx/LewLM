"""Structured results for document ingestion."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from lewlm.core.provenance import ComponentProvenance
from lewlm.documents.ir.models import DocumentIR


class DocumentSourceType(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    CSV = "csv"
    XLSX = "xlsx"
    DOCX = "docx"
    PDF = "pdf"
    IMAGE = "image"
    IMAGE_BUNDLE = "image_bundle"


class IngestedDocumentSource(BaseModel):
    source_id: str = Field(
        description=(
            "Stable source identifier. Caller-provided for uploaded sources, "
            "otherwise derived from the local source path."
        ),
    )
    path: str | None = Field(
        default=None,
        description=(
            "Server-local path. Always null for uploaded sources, which never "
            "make the caller depend on a shared filesystem mount."
        ),
    )
    source_type: DocumentSourceType
    source_name: str = Field(description="Basename of the local source path.")
    source_label: str = Field(description="Human-readable label for reuse in app UIs and citations.")
    media_type: str | None = Field(default=None, description="Detected media type when LewLM can determine it.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str = Field(description="Stable chunk identifier derived from source and section identity.")
    text: str
    source_id: str = Field(description="Stable source identifier that owns this chunk.")
    section_id: str = Field(description="Stable section identifier that owns this chunk.")
    source_label: str = Field(description="Human-readable source label for display and citation packaging.")
    section_label: str = Field(description="Human-readable section label for display and citation packaging.")
    section_heading: str | None = None
    section_level: int | None = None
    source_name: str | None = None
    source_path: str | None = None
    source_type: DocumentSourceType | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestErrorCode(str, Enum):
    """Stable, machine-readable reasons a single source failed to ingest."""

    UNSUPPORTED_SOURCE_TYPE = "unsupported_source_type"
    CORRUPT_SOURCE = "corrupt_source"
    EMPTY_SOURCE = "empty_source"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    SOURCE_TOO_LARGE = "source_too_large"
    PARSER_FAILED = "parser_failed"
    PARSER_TIMEOUT = "parser_timeout"
    OCR_UNAVAILABLE = "ocr_unavailable"
    ACCESS_DENIED = "access_denied"
    INTERNAL_ERROR = "internal_error"


#: Failures worth retrying unchanged; everything else needs caller intervention.
_RETRYABLE_ERROR_CODES = frozenset(
    {
        DocumentIngestErrorCode.PARSER_TIMEOUT,
        DocumentIngestErrorCode.INTERNAL_ERROR,
    },
)


class DocumentSourceIngestOutcome(BaseModel):
    """The per-source result of a multi-source ingest request.

    Multi-source ingestion previously surfaced only the sources that succeeded,
    leaving a caller to infer which of its inputs went missing and with no way
    to recover the reason. Every requested source now gets exactly one outcome.
    """

    source_id: str = Field(description="Caller-provided ID for uploads, else the LewLM-derived ID.")
    status: Literal["ingested", "failed"]
    source_label: str | None = None
    source_type: DocumentSourceType | None = None
    media_type: str | None = None
    chunk_count: int = 0
    section_count: int = 0
    content_sha256: str | None = Field(
        default=None,
        description="SHA-256 of the bytes LewLM actually parsed.",
    )
    error_code: DocumentIngestErrorCode | None = None
    error_message: str | None = None
    retryable: bool = Field(
        default=False,
        description="Whether retrying this source unchanged could plausibly succeed.",
    )
    provider_reference: str | None = Field(
        default=None,
        description="LewLM-side reference for correlating this source with logs and events.",
    )
    components: list[ComponentProvenance] = Field(
        default_factory=list,
        description="Parser, OCR, and chunker components applied to this source.",
    )

    @property
    def succeeded(self) -> bool:
        return self.status == "ingested"


def retryable_for(code: DocumentIngestErrorCode) -> bool:
    """Whether a failure code is worth retrying with an unchanged request."""

    return code in _RETRYABLE_ERROR_CODES


class DocumentIngestResult(BaseModel):
    document: DocumentIR
    sources: list[IngestedDocumentSource] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)
    source_results: list[DocumentSourceIngestOutcome] = Field(
        default_factory=list,
        description="One outcome per requested source, in request order.",
    )
    ingested_count: int = 0
    failed_count: int = 0
    partial: bool = Field(
        default=False,
        description="True when at least one requested source failed while others succeeded.",
    )
    components: list[ComponentProvenance] = Field(
        default_factory=list,
        description="Distinct components that contributed to this ingest.",
    )
