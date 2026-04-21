"""Document renderer registry."""

from __future__ import annotations

from dataclasses import dataclass
import json

from lewlm.core.errors import DocumentGenerationError
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.documents.render.base import DocumentRenderer
from lewlm.documents.render.csv_builder import CsvDocumentRenderer
from lewlm.documents.render.docx_builder import DocxDocumentRenderer
from lewlm.documents.render.json_builder import JsonDocumentRenderer
from lewlm.documents.render.markdown_builder import MarkdownDocumentRenderer
from lewlm.documents.render.pdf_builder import PdfDocumentRenderer
from lewlm.documents.render.text_builder import TextDocumentRenderer
from lewlm.documents.render.xlsx_builder import XlsxDocumentRenderer
from lewlm.documents.validators.ir import DocumentIRValidator


@dataclass(slots=True)
class GeneratedDocumentArtifact:
    output_format: DocumentOutputFormat
    media_type: str
    file_extension: str
    file_name: str
    content: bytes

    @property
    def size_bytes(self) -> int:
        return len(self.content)


class DocumentRendererRegistry:
    """Registry of deterministic document renderers."""

    def __init__(self, validator: DocumentIRValidator | None = None) -> None:
        self.validator = validator or DocumentIRValidator()
        self._renderers: dict[DocumentOutputFormat, DocumentRenderer] = {
            DocumentOutputFormat.TEXT: TextDocumentRenderer(self.validator),
            DocumentOutputFormat.MARKDOWN: MarkdownDocumentRenderer(self.validator),
            DocumentOutputFormat.JSON: JsonDocumentRenderer(self.validator),
            DocumentOutputFormat.CSV: CsvDocumentRenderer(self.validator),
            DocumentOutputFormat.DOCX: DocxDocumentRenderer(self.validator),
            DocumentOutputFormat.PDF: PdfDocumentRenderer(self.validator),
            DocumentOutputFormat.XLSX: XlsxDocumentRenderer(self.validator),
        }

    def get_renderer(self, output_format: DocumentOutputFormat) -> DocumentRenderer:
        try:
            return self._renderers[output_format]
        except KeyError as exc:
            raise DocumentGenerationError(
                "Unsupported document output format.",
                details={"output_format": output_format.value},
            ) from exc

    def render(
        self,
        document: DocumentIR,
        *,
        output_format: DocumentOutputFormat,
        file_name: str,
    ) -> GeneratedDocumentArtifact:
        renderer = self.get_renderer(output_format)
        content = renderer.render(document)
        self._validate_rendered_artifact(output_format=output_format, content=content)
        return GeneratedDocumentArtifact(
            output_format=output_format,
            media_type=renderer.media_type,
            file_extension=renderer.file_extension,
            file_name=file_name,
            content=content,
        )

    def _validate_rendered_artifact(self, *, output_format: DocumentOutputFormat, content: bytes) -> None:
        if not content:
            raise DocumentGenerationError(
                "Document renderer produced an empty artifact.",
                details={"output_format": output_format.value},
            )
        if output_format in {DocumentOutputFormat.TEXT, DocumentOutputFormat.MARKDOWN, DocumentOutputFormat.CSV}:
            decoded = content.decode("utf-8")
            if not decoded.strip():
                raise DocumentGenerationError(
                    "Text-based document renderer produced only empty content.",
                    details={"output_format": output_format.value},
                )
            return
        if output_format == DocumentOutputFormat.JSON:
            decoded = content.decode("utf-8")
            try:
                json.loads(decoded)
            except json.JSONDecodeError as exc:
                raise DocumentGenerationError(
                    "JSON document renderer produced invalid JSON.",
                    details={"output_format": output_format.value},
                ) from exc
            return
        if output_format == DocumentOutputFormat.PDF and not content.startswith(b"%PDF"):
            raise DocumentGenerationError("PDF renderer produced an invalid PDF signature.")
        if output_format in {DocumentOutputFormat.DOCX, DocumentOutputFormat.XLSX} and not content.startswith(b"PK"):
            raise DocumentGenerationError(
                "Office document renderer produced an invalid ZIP-based artifact signature.",
                details={"output_format": output_format.value},
            )
