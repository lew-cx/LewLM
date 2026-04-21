"""JSON document renderer."""

from __future__ import annotations

from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.documents.render.base import DocumentRenderer
from lewlm.documents.validators.ir import DocumentIRValidator


class JsonDocumentRenderer(DocumentRenderer):
    output_format = DocumentOutputFormat.JSON
    media_type = "application/json"
    file_extension = ".json"

    def __init__(self, validator: DocumentIRValidator) -> None:
        self.validator = validator

    def render(self, document: DocumentIR) -> bytes:
        self.validator.validate(document)
        return document.model_dump_json(indent=2).encode("utf-8")
