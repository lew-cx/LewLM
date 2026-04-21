"""PDF document renderer."""

from __future__ import annotations

import contextlib
import html
from io import BytesIO, StringIO
from pathlib import Path

from lewlm.core.errors import DocumentGenerationError
from lewlm.documents.ir.models import (
    CalloutBlock,
    DocumentIR,
    DocumentOutputFormat,
    ImageBlock,
    ListBlock,
    ParagraphBlock,
    TableBlock,
)
from lewlm.documents.render.base import DocumentRenderer
from lewlm.documents.validators.ir import DocumentIRValidator


class PdfDocumentRenderer(DocumentRenderer):
    output_format = DocumentOutputFormat.PDF
    media_type = "application/pdf"
    file_extension = ".pdf"

    def __init__(self, validator: DocumentIRValidator) -> None:
        self.validator = validator

    def render(self, document: DocumentIR) -> bytes:
        self.validator.validate(document)
        try:
            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                from weasyprint import HTML
        except (ImportError, OSError):
            return self._render_with_reportlab(document)
        return HTML(string=self._to_html(document), base_url=str(Path.cwd())).write_pdf()

    def _to_html(self, document: DocumentIR) -> str:
        body_parts: list[str] = [f"<h1>{html.escape(document.title)}</h1>"]
        for section in document.sections:
            if section.heading:
                level = max(1, min(section.level + 1, 6))
                body_parts.append(f"<h{level}>{html.escape(section.heading)}</h{level}>")
            for block in section.blocks:
                body_parts.append(self._render_block(block))

        if document.citations:
            body_parts.append(f"<h2>{html.escape(document.references_title)}</h2><ol>")
            for citation in document.citations:
                text = citation.text if citation.url is None else f"{citation.text} ({citation.url})"
                body_parts.append(f"<li>[{html.escape(citation.label)}] {html.escape(text)}</li>")
            body_parts.append("</ol>")

        header_html = self._header_footer_html(document.header)
        footer_html = self._header_footer_html(document.footer)
        return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <style>
      @page {{
        size: A4;
        margin: 20mm;
      }}
      body {{ font-family: sans-serif; color: #1f2937; }}
      h1, h2, h3, h4, h5, h6 {{ color: #111827; }}
      table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
      th, td {{ border: 1px solid #d1d5db; padding: 0.4rem; text-align: left; }}
      .callout {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 0.75rem; margin: 1rem 0; }}
      .header-footer {{ color: #6b7280; font-size: 0.85rem; }}
      .image-caption {{ color: #4b5563; font-size: 0.9rem; }}
    </style>
  </head>
  <body>
    {header_html}
    {''.join(body_parts)}
    {footer_html}
  </body>
</html>
"""

    def _render_block(self, block) -> str:
        if isinstance(block, ParagraphBlock):
            return f"<p>{html.escape(block.text)}</p>"
        if isinstance(block, ListBlock):
            tag = "ol" if block.ordered else "ul"
            items = "".join(f"<li>{html.escape(item)}</li>" for item in block.items)
            return f"<{tag}>{items}</{tag}>"
        if isinstance(block, TableBlock):
            header_html = ""
            if block.headers:
                header_cells = "".join(f"<th>{html.escape(cell)}</th>" for cell in block.headers)
                header_html = f"<thead><tr>{header_cells}</tr></thead>"
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in block.rows
            )
            caption_html = f"<caption>{html.escape(block.caption)}</caption>" if block.caption else ""
            return f"<table>{caption_html}{header_html}<tbody>{rows_html}</tbody></table>"
        if isinstance(block, CalloutBlock):
            title = f"<strong>{html.escape(block.title)}</strong><br>" if block.title else ""
            return f"<div class='callout'>{title}{html.escape(block.body)}</div>"
        if isinstance(block, ImageBlock):
            if block.path is not None:
                image_uri = Path(block.path).expanduser().resolve(strict=False).as_uri()
                image_html = f"<img src='{html.escape(image_uri)}' alt='{html.escape(block.alt_text)}' style='max-width: 100%;'>"
            else:
                image_html = f"<div>[Image] {html.escape(block.alt_text)}</div>"
            caption = f"<div class='image-caption'>{html.escape(block.caption)}</div>" if block.caption else ""
            return f"{image_html}{caption}"
        raise DocumentGenerationError("Encountered an unknown document block during PDF rendering.")

    def _header_footer_html(self, content) -> str:
        if content is None:
            return ""
        values = [value for value in (content.left, content.center, content.right) if value]
        if not values:
            return ""
        joined = " | ".join(html.escape(value) for value in values)
        return f"<div class='header-footer'>{joined}</div>"

    def _render_with_reportlab(self, document: DocumentIR) -> bytes:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                Image,
                ListFlowable,
                ListItem,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError as exc:
            raise DocumentGenerationError(
                "PDF generation requires either a working WeasyPrint install or the `reportlab` dependency.",
                details={"renderer": self.output_format.value},
            ) from exc

        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        heading_style = styles["Heading1"]
        body_style = styles["BodyText"]
        callout_style = ParagraphStyle(
            "Callout",
            parent=body_style,
            backColor="#EFF6FF",
            borderColor="#2563EB",
            borderWidth=1,
            borderPadding=6,
            spaceAfter=10,
            leftIndent=6,
        )

        story = [Paragraph(html.escape(document.title), title_style), Spacer(1, 0.2 * inch)]
        for section in document.sections:
            if section.heading:
                story.append(Paragraph(html.escape(section.heading), heading_style))
            for block in section.blocks:
                if isinstance(block, ParagraphBlock):
                    story.append(Paragraph(html.escape(block.text), body_style))
                    story.append(Spacer(1, 0.12 * inch))
                elif isinstance(block, ListBlock):
                    items = [ListItem(Paragraph(html.escape(item), body_style)) for item in block.items]
                    story.append(ListFlowable(items, bulletType="1" if block.ordered else "bullet"))
                    story.append(Spacer(1, 0.12 * inch))
                elif isinstance(block, TableBlock):
                    table_data = [block.headers] if block.headers else []
                    table_data.extend(block.rows)
                    table = Table(table_data, hAlign="LEFT")
                    table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")) if block.headers else ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                                ("PADDING", (0, 0), (-1, -1), 6),
                            ],
                        ),
                    )
                    story.append(table)
                    if block.caption:
                        story.append(Paragraph(html.escape(block.caption), body_style))
                    story.append(Spacer(1, 0.12 * inch))
                elif isinstance(block, CalloutBlock):
                    title = f"<b>{html.escape(block.title)}</b><br/>" if block.title else ""
                    story.append(Paragraph(f"{title}{html.escape(block.body)}", callout_style))
                    story.append(Spacer(1, 0.12 * inch))
                elif isinstance(block, ImageBlock):
                    if block.path is not None:
                        story.append(Image(str(Path(block.path).expanduser()), width=4 * inch, preserveAspectRatio=True, hAlign="LEFT"))
                    else:
                        story.append(Paragraph(f"[Image] {html.escape(block.alt_text)}", body_style))
                    if block.caption:
                        story.append(Paragraph(html.escape(block.caption), body_style))
                    story.append(Spacer(1, 0.12 * inch))

        if document.citations:
            story.append(Paragraph(html.escape(document.references_title), heading_style))
            for citation in document.citations:
                text = citation.text if citation.url is None else f"{citation.text} ({citation.url})"
                story.append(Paragraph(f"[{html.escape(citation.label)}] {html.escape(text)}", body_style))

        buffer = BytesIO()
        pdf = SimpleDocTemplate(buffer, pagesize=A4, title=document.title)

        def draw_header_footer(canvas, _doc) -> None:
            canvas.saveState()
            header_values = document.header and [value for value in (document.header.left, document.header.center, document.header.right) if value] or []
            footer_values = document.footer and [value for value in (document.footer.left, document.footer.center, document.footer.right) if value] or []
            canvas.setFont("Helvetica", 9)
            if header_values:
                canvas.drawString(pdf.leftMargin, A4[1] - 30, " | ".join(header_values))
            if footer_values:
                canvas.drawString(pdf.leftMargin, 20, " | ".join(footer_values))
            canvas.restoreState()

        pdf.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
        return buffer.getvalue()
