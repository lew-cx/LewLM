"""Renderer contracts for document artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat


class DocumentRenderer(ABC):
    """Render a document IR payload into a concrete artifact."""

    output_format: DocumentOutputFormat
    media_type: str
    file_extension: str

    @abstractmethod
    def render(self, document: DocumentIR) -> bytes:
        """Render a document artifact."""
