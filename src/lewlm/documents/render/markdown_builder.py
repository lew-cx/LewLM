"""Markdown document renderer."""

from __future__ import annotations

import json

from lewlm.documents.ir.models import CalloutBlock, DocumentIR, DocumentOutputFormat, ImageBlock, ListBlock, ParagraphBlock, TableBlock
from lewlm.documents.ir.style import (
    DocumentStyleSheet,
    ResolvedStyle,
    StyleElement,
    element_for_block,
)
from lewlm.documents.render.base import DocumentRenderer
from lewlm.documents.validators.ir import DocumentIRValidator


class MarkdownDocumentRenderer(DocumentRenderer):
    """Render Markdown.

    Markdown carries no typography or colour vocabulary, so only the `emphasis`
    style token affects this format; colour, font, and size tokens are inert
    here by contract.
    """

    output_format = DocumentOutputFormat.MARKDOWN
    media_type = "text/markdown"
    file_extension = ".md"

    def __init__(self, validator: DocumentIRValidator) -> None:
        self.validator = validator

    def render(self, document: DocumentIR) -> bytes:
        self.validator.validate(document)
        stylesheet = DocumentStyleSheet.from_document(document)
        lines: list[str] = [f"# {document.title}"]
        if document.header:
            header_values = [value for value in (document.header.left, document.header.center, document.header.right) if value]
            if header_values:
                header_style = stylesheet.for_element(StyleElement.HEADER, token_names=document.header.style_tokens)
                lines.extend(["", f"_Header: {_emphasize(' | '.join(header_values), header_style)}_"])
        for section in document.sections:
            lines.append("")
            if section.heading:
                level = max(2, min(section.level + 1, 6))
                heading_style = stylesheet.for_element(StyleElement.HEADING, token_names=section.style_tokens)
                lines.append(f"{'#' * level} {_emphasize(section.heading, heading_style)}")
            for block in section.blocks:
                block_style = stylesheet.for_element(
                    element_for_block(block),
                    token_names=[*section.style_tokens, *block.style_tokens],
                )
                lines.extend(self._render_block(block, block_style))
        if document.citations:
            lines.extend(["", f"## {document.references_title}"])
            for citation in document.citations:
                citation_style = stylesheet.for_element(StyleElement.CITATION, token_names=citation.style_tokens)
                text = citation.text if citation.url is None else f"{citation.text} ({citation.url})"
                lines.append(f"1. [{citation.label}] {_emphasize(text, citation_style)}")
        if document.footer:
            footer_values = [value for value in (document.footer.left, document.footer.center, document.footer.right) if value]
            if footer_values:
                footer_style = stylesheet.for_element(StyleElement.FOOTER, token_names=document.footer.style_tokens)
                lines.extend(["", f"_Footer: {_emphasize(' | '.join(footer_values), footer_style)}_"])
        return "\n".join(lines).strip().encode("utf-8")

    def _render_block(self, block, style: ResolvedStyle) -> list[str]:
        if isinstance(block, ParagraphBlock):
            return ["", _emphasize(block.text, style)]
        if isinstance(block, ListBlock):
            prefix_template = "{index}. {item}" if block.ordered else "- {item}"
            return [""] + [
                prefix_template.format(index=index, item=_emphasize(item, style))
                for index, item in enumerate(block.items, start=1)
            ]
        if isinstance(block, TableBlock):
            headers = block.headers or [f"column_{index + 1}" for index in range(len(block.rows[0]))]
            lines = [""]
            if block.caption:
                lines.append(f"**{block.caption}**")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            lines.extend("| " + " | ".join(_emphasize(cell, style) for cell in row) + " |" for row in block.rows)
            return lines
        if isinstance(block, CalloutBlock):
            title = f" **{block.title}**" if block.title else ""
            return ["", f"> [{block.kind.upper()}]{title} {_emphasize(block.body, style)}"]
        if isinstance(block, ImageBlock):
            alt = block.alt_text
            if block.path:
                line = f"![{alt}]({block.path})"
            else:
                metadata = json.dumps({"width": block.width, "height": block.height}, separators=(",", ":"))
                line = f"![{alt}](#embedded-image \"{metadata}\")"
            if block.caption:
                return ["", line, "", f"*{_emphasize(block.caption, style)}*"]
            return ["", line]
        return []


def _emphasize(text: str, style: ResolvedStyle) -> str:
    if not text.strip() or not (style.bold or style.italic):
        return text
    if style.bold:
        text = f"**{text}**"
    if style.italic:
        text = f"*{text}*"
    return text
