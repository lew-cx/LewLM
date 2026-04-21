"""Plain-text document renderer."""

from __future__ import annotations

from lewlm.documents.ir.models import CalloutBlock, DocumentIR, DocumentOutputFormat, ImageBlock, ListBlock, ParagraphBlock, TableBlock
from lewlm.documents.render.base import DocumentRenderer
from lewlm.documents.validators.ir import DocumentIRValidator


class TextDocumentRenderer(DocumentRenderer):
    output_format = DocumentOutputFormat.TEXT
    media_type = "text/plain"
    file_extension = ".txt"

    def __init__(self, validator: DocumentIRValidator) -> None:
        self.validator = validator

    def render(self, document: DocumentIR) -> bytes:
        self.validator.validate(document)
        lines: list[str] = [document.title]
        if document.header:
            lines.extend(_render_header_footer("Header", document.header))
        for section in document.sections:
            lines.append("")
            if section.heading:
                lines.append(section.heading)
            for block in section.blocks:
                lines.extend(self._render_block(block))
        if document.citations:
            lines.append("")
            lines.append(document.references_title)
            for citation in document.citations:
                text = citation.text if citation.url is None else f"{citation.text} ({citation.url})"
                lines.append(f"[{citation.label}] {text}")
        if document.footer:
            lines.append("")
            lines.extend(_render_header_footer("Footer", document.footer))
        return "\n".join(line for line in lines if line is not None).strip().encode("utf-8")

    def _render_block(self, block) -> list[str]:
        if isinstance(block, ParagraphBlock):
            return [block.text]
        if isinstance(block, ListBlock):
            return [f"{index}. {item}" if block.ordered else f"- {item}" for index, item in enumerate(block.items, start=1)]
        if isinstance(block, TableBlock):
            lines = [block.caption] if block.caption else []
            if block.headers:
                lines.append(" | ".join(block.headers))
            lines.extend(" | ".join(row) for row in block.rows)
            return lines
        if isinstance(block, CalloutBlock):
            prefix = block.kind.upper()
            if block.title:
                prefix = f"{prefix}: {block.title}"
            return [f"{prefix} {block.body}"]
        if isinstance(block, ImageBlock):
            caption = f" - {block.caption}" if block.caption else ""
            return [f"[Image] {block.alt_text}{caption}"]
        return []


def _render_header_footer(label: str, content) -> list[str]:
    values = [value for value in (content.left, content.center, content.right) if value]
    if not values:
        return []
    return [f"{label}: " + " | ".join(values)]
