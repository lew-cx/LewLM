"""Structured results for document ingestion."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

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
    source_id: str = Field(description="Stable source identifier derived from the local source path.")
    path: str
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


class DocumentIngestResult(BaseModel):
    document: DocumentIR
    sources: list[IngestedDocumentSource] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)
