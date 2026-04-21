"""Markdown document renderer."""

from __future__ import annotations

import json

from lewlm.documents.ir.models import CalloutBlock, DocumentIR, DocumentOutputFormat, ImageBlock, ListBlock, ParagraphBlock, TableBlock
from lewlm.documents.render.base import DocumentRenderer
from lewlm.documents.validators.ir import DocumentIRValidator


class MarkdownDocumentRenderer(DocumentRenderer):
    output_format = DocumentOutputFormat.MARKDOWN
    media_type = "text/markdown"
    file_extension = ".md"

    def __init__(self, validator: DocumentIRValidator) -> None:
        self.validator = validator

    def render(self, document: DocumentIR) -> bytes:
        self.validator.validate(document)
        lines: list[str] = [f"# {document.title}"]
        if document.header:
            header_values = [value for value in (document.header.left, document.header.center, document.header.right) if value]
            if header_values:
                lines.extend(["", f"_Header: {' | '.join(header_values)}_"])
        for section in document.sections:
            lines.append("")
            if section.heading:
                level = max(2, min(section.level + 1, 6))
                lines.append(f"{'#' * level} {section.heading}")
            for block in section.blocks:
                lines.extend(self._render_block(block))
        if document.citations:
            lines.extend(["", f"## {document.references_title}"])
            for citation in document.citations:
                text = citation.text if citation.url is None else f"{citation.text} ({citation.url})"
                lines.append(f"1. [{citation.label}] {text}")
        if document.footer:
            footer_values = [value for value in (document.footer.left, document.footer.center, document.footer.right) if value]
            if footer_values:
                lines.extend(["", f"_Footer: {' | '.join(footer_values)}_"])
        return "\n".join(lines).strip().encode("utf-8")

    def _render_block(self, block) -> list[str]:
        if isinstance(block, ParagraphBlock):
            return ["", block.text]
        if isinstance(block, ListBlock):
            prefix_template = "{index}. {item}" if block.ordered else "- {item}"
            return [""] + [
                prefix_template.format(index=index, item=item)
                for index, item in enumerate(block.items, start=1)
            ]
        if isinstance(block, TableBlock):
            headers = block.headers or [f"column_{index + 1}" for index in range(len(block.rows[0]))]
            lines = [""]
            if block.caption:
                lines.append(f"**{block.caption}**")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            lines.extend("| " + " | ".join(row) + " |" for row in block.rows)
            return lines
        if isinstance(block, CalloutBlock):
            title = f" **{block.title}**" if block.title else ""
            return ["", f"> [{block.kind.upper()}]{title} {block.body}"]
        if isinstance(block, ImageBlock):
            alt = block.alt_text
            if block.path:
                line = f"![{alt}]({block.path})"
            else:
                metadata = json.dumps({"width": block.width, "height": block.height}, separators=(",", ":"))
                line = f"![{alt}](#embedded-image \"{metadata}\")"
            if block.caption:
                return ["", line, "", f"*{block.caption}*"]
            return ["", line]
        return []
