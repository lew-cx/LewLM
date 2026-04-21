"""Document service exports."""

from __future__ import annotations

__all__ = ["DocumentGenerationService", "DocumentIngestService", "DocumentTransformService"]


def __getattr__(name: str):
    if name == "DocumentGenerationService":
        from lewlm.documents.service import DocumentGenerationService

        return DocumentGenerationService
    if name == "DocumentIngestService":
        from lewlm.documents.ingest.service import DocumentIngestService

        return DocumentIngestService
    if name == "DocumentTransformService":
        from lewlm.documents.skills.service import DocumentTransformService

        return DocumentTransformService
    raise AttributeError(name)
