"""Document ingest services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lewlm.documents.ingest.models import DocumentChunk, DocumentIngestResult, DocumentSourceType, IngestedDocumentSource

if TYPE_CHECKING:
    from lewlm.documents.ingest.service import DocumentIngestService


def __getattr__(name: str):
    if name == "DocumentIngestService":
        from lewlm.documents.ingest.service import DocumentIngestService

        return DocumentIngestService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["DocumentChunk", "DocumentIngestResult", "DocumentIngestService", "DocumentSourceType", "IngestedDocumentSource"]
