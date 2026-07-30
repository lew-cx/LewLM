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
from lewlm.documents.ir.style import (
    DocumentStyleSheet,
    ResolvedStyle,
    StyleElement,
    element_for_block,
    style_css,
)
from lewlm.documents.render.base import DocumentRenderer
from lewlm.documents.validators.ir import DocumentIRValidator

DEFAULT_ACCENT_COLOR = "#2563EB"
DEFAULT_SURFACE_COLOR = "#EFF6FF"


class PdfDocumentRenderer(DocumentRenderer):
    """Render PDF.

    The WeasyPrint path honours every reserved style token. The ReportLab
    fallback honours colour, point size, and emphasis tokens but ignores font
    family tokens because it can only lay out its built-in font set.
    """

    output_format = DocumentOutputFormat.PDF
    media_type = "application/pdf"
    file_extension = ".pdf"

    def __init__(self, validator: DocumentIRValidator) -> None:
        self.validator = validator

    def render(self, document: DocumentIR) -> bytes:
        self.validator.validate(document)
        stylesheet = DocumentStyleSheet.from_document(document)
        try:
            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                from weasyprint import HTML
        except (ImportError, OSError):
            return self._render_with_reportlab(document, stylesheet)
        return HTML(string=self._to_html(document, stylesheet), base_url=str(Path.cwd())).write_pdf()

    def _to_html(self, document: DocumentIR, stylesheet: DocumentStyleSheet) -> str:
        title_style = stylesheet.for_element(StyleElement.HEADING)
        body_parts: list[str] = [f"<h1{_style_attribute(title_style)}>{html.escape(document.title)}</h1>"]
        for section in document.sections:
            if section.heading:
                level = max(1, min(section.level + 1, 6))
                heading_style = stylesheet.for_element(StyleElement.HEADING, token_names=section.style_tokens)
                body_parts.append(
                    f"<h{level}{_style_attribute(heading_style)}>{html.escape(section.heading)}</h{level}>",
                )
            for block in section.blocks:
                token_names = [*section.style_tokens, *block.style_tokens]
                block_element = element_for_block(block)
                block_style = stylesheet.for_element(
                    block_element,
                    token_names=token_names,
                )
                body_parts.append(
                    self._render_block(
                        block,
                        block_style,
                        stylesheet.accent_color_for(block_element, token_names=token_names),
                    ),
                )

        if document.citations:
            body_parts.append(f"<h2>{html.escape(document.references_title)}</h2><ol>")
            for citation in document.citations:
                citation_style = stylesheet.for_element(StyleElement.CITATION, token_names=citation.style_tokens)
                text = citation.text if citation.url is None else f"{citation.text} ({citation.url})"
                body_parts.append(
                    f"<li{_style_attribute(citation_style)}>[{html.escape(citation.label)}] {html.escape(text)}</li>",
                )
            body_parts.append("</ol>")

        header_html = self._header_footer_html(
            document.header,
            stylesheet.for_element(StyleElement.HEADER, token_names=document.header.style_tokens) if document.header else None,
        )
        footer_html = self._header_footer_html(
            document.footer,
            stylesheet.for_element(StyleElement.FOOTER, token_names=document.footer.style_tokens) if document.footer else None,
        )
        accent = stylesheet.accent_color_for(StyleElement.CALLOUT) or DEFAULT_ACCENT_COLOR
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
      .callout {{ border-left: 4px solid {accent}; background: {DEFAULT_SURFACE_COLOR}; padding: 0.75rem; margin: 1rem 0; }}
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

    def _render_block(self, block, style: ResolvedStyle, accent_color: str | None) -> str:
        attribute = _style_attribute(style)
        if isinstance(block, ParagraphBlock):
            return f"<p{attribute}>{html.escape(block.text)}</p>"
        if isinstance(block, ListBlock):
            tag = "ol" if block.ordered else "ul"
            items = "".join(f"<li>{html.escape(item)}</li>" for item in block.items)
            return f"<{tag}{attribute}>{items}</{tag}>"
        if isinstance(block, TableBlock):
            header_html = ""
            if block.headers:
                header_attribute = f" style=\"background-color: {accent_color}\"" if accent_color else ""
                header_cells = "".join(f"<th{header_attribute}>{html.escape(cell)}</th>" for cell in block.headers)
                header_html = f"<thead><tr>{header_cells}</tr></thead>"
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in block.rows
            )
            caption_html = f"<caption>{html.escape(block.caption)}</caption>" if block.caption else ""
            return f"<table{attribute}>{caption_html}{header_html}<tbody>{rows_html}</tbody></table>"
        if isinstance(block, CalloutBlock):
            title = f"<strong>{html.escape(block.title)}</strong><br>" if block.title else ""
            return (
                f"<div class='callout'{_style_attribute(style, border_color=accent_color)}>"
                f"{title}{html.escape(block.body)}</div>"
            )
        if isinstance(block, ImageBlock):
            if block.path is not None:
                image_uri = Path(block.path).expanduser().resolve(strict=False).as_uri()
                image_html = f"<img src='{html.escape(image_uri)}' alt='{html.escape(block.alt_text)}' style='max-width: 100%;'>"
            else:
                image_html = f"<div>[Image] {html.escape(block.alt_text)}</div>"
            caption = f"<div class='image-caption'{attribute}>{html.escape(block.caption)}</div>" if block.caption else ""
            return f"{image_html}{caption}"
        raise DocumentGenerationError("Encountered an unknown document block during PDF rendering.")

    def _header_footer_html(self, content, style: ResolvedStyle | None) -> str:
        if content is None:
            return ""
        values = [value for value in (content.left, content.center, content.right) if value]
        if not values:
            return ""
        joined = " | ".join(html.escape(value) for value in values)
        attribute = _style_attribute(style) if style is not None else ""
        return f"<div class='header-footer'{attribute}>{joined}</div>"

    def _render_with_reportlab(self, document: DocumentIR, stylesheet: DocumentStyleSheet) -> bytes:
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

        def styled(base, style: ResolvedStyle, name: str):
            """Derive a paragraph style from a resolved style, ignoring font family."""

            overrides: dict[str, object] = {}
            if style.text_color is not None:
                overrides["textColor"] = colors.HexColor(style.text_color)
            if style.background_color is not None:
                overrides["backColor"] = colors.HexColor(style.background_color)
            if style.font_size_pt is not None:
                overrides["fontSize"] = style.font_size_pt
                overrides["leading"] = style.font_size_pt * 1.2
            if not overrides:
                return base
            return ParagraphStyle(name, parent=base, **overrides)

        def marked(text: str, style: ResolvedStyle) -> str:
            escaped = html.escape(text)
            if style.bold:
                escaped = f"<b>{escaped}</b>"
            if style.italic:
                escaped = f"<i>{escaped}</i>"
            return escaped

        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        heading_style = styles["Heading1"]
        body_style = styles["BodyText"]
        callout_accent_color = stylesheet.accent_color_for(StyleElement.CALLOUT)
        callout_style = ParagraphStyle(
            "Callout",
            parent=body_style,
            backColor=DEFAULT_SURFACE_COLOR,
            borderColor=callout_accent_color or DEFAULT_ACCENT_COLOR,
            borderWidth=1,
            borderPadding=6,
            spaceAfter=10,
            leftIndent=6,
        )

        document_heading_style = stylesheet.for_element(StyleElement.HEADING)
        story = [
            Paragraph(marked(document.title, document_heading_style), styled(title_style, document_heading_style, "DocumentTitle")),
            Spacer(1, 0.2 * inch),
        ]
        for section_index, section in enumerate(document.sections):
            section_heading_style = stylesheet.for_element(StyleElement.HEADING, token_names=section.style_tokens)
            if section.heading:
                story.append(
                    Paragraph(
                        marked(section.heading, section_heading_style),
                        styled(heading_style, section_heading_style, f"Heading{section_index}"),
                    ),
                )
            for block_index, block in enumerate(section.blocks):
                token_names = [*section.style_tokens, *block.style_tokens]
                block_element = element_for_block(block)
                resolved = stylesheet.for_element(
                    block_element,
                    token_names=token_names,
                )
                block_accent_color = stylesheet.accent_color_for(block_element, token_names=token_names)
                block_name = f"Block{section_index}_{block_index}"
                if isinstance(block, ParagraphBlock):
                    story.append(Paragraph(marked(block.text, resolved), styled(body_style, resolved, block_name)))
                    story.append(Spacer(1, 0.12 * inch))
                elif isinstance(block, ListBlock):
                    item_style = styled(body_style, resolved, block_name)
                    items = [ListItem(Paragraph(marked(item, resolved), item_style)) for item in block.items]
                    story.append(ListFlowable(items, bulletType="1" if block.ordered else "bullet"))
                    story.append(Spacer(1, 0.12 * inch))
                elif isinstance(block, TableBlock):
                    table_data = [block.headers] if block.headers else []
                    table_data.extend(block.rows)
                    table = Table(table_data, hAlign="LEFT")
                    header_fill = (
                        colors.HexColor(block_accent_color)
                        if block_accent_color
                        else colors.HexColor("#E5E7EB")
                    )
                    commands: list[tuple] = [
                        ("BACKGROUND", (0, 0), (-1, 0), header_fill) if block.headers else ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                        ("PADDING", (0, 0), (-1, -1), 6),
                    ]
                    if resolved.text_color is not None:
                        commands.append(("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(resolved.text_color)))
                    if resolved.font_size_pt is not None:
                        commands.append(("FONTSIZE", (0, 0), (-1, -1), resolved.font_size_pt))
                    table.setStyle(TableStyle(commands))
                    story.append(table)
                    if block.caption:
                        story.append(Paragraph(marked(block.caption, resolved), styled(body_style, resolved, f"{block_name}_caption")))
                    story.append(Spacer(1, 0.12 * inch))
                elif isinstance(block, CalloutBlock):
                    title = f"<b>{html.escape(block.title)}</b><br/>" if block.title else ""
                    resolved_callout_style = callout_style
                    if block_accent_color is not None and block_accent_color != callout_accent_color:
                        resolved_callout_style = ParagraphStyle(
                            f"{block_name}_callout",
                            parent=callout_style,
                            borderColor=colors.HexColor(block_accent_color),
                        )
                    story.append(
                        Paragraph(
                            f"{title}{marked(block.body, resolved)}",
                            styled(resolved_callout_style, resolved, block_name),
                        ),
                    )
                    story.append(Spacer(1, 0.12 * inch))
                elif isinstance(block, ImageBlock):
                    if block.path is not None:
                        story.append(Image(str(Path(block.path).expanduser()), width=4 * inch, preserveAspectRatio=True, hAlign="LEFT"))
                    else:
                        story.append(Paragraph(f"[Image] {html.escape(block.alt_text)}", body_style))
                    if block.caption:
                        story.append(Paragraph(marked(block.caption, resolved), styled(body_style, resolved, f"{block_name}_caption")))
                    story.append(Spacer(1, 0.12 * inch))

        if document.citations:
            story.append(Paragraph(html.escape(document.references_title), heading_style))
            for citation_index, citation in enumerate(document.citations):
                resolved = stylesheet.for_element(StyleElement.CITATION, token_names=citation.style_tokens)
                text = citation.text if citation.url is None else f"{citation.text} ({citation.url})"
                story.append(
                    Paragraph(
                        f"[{html.escape(citation.label)}] {marked(text, resolved)}",
                        styled(body_style, resolved, f"Citation{citation_index}"),
                    ),
                )

        buffer = BytesIO()
        pdf = SimpleDocTemplate(buffer, pagesize=A4, title=document.title)

        def draw_header_footer(canvas, _doc) -> None:
            canvas.saveState()
            header_values = document.header and [value for value in (document.header.left, document.header.center, document.header.right) if value] or []
            footer_values = document.footer and [value for value in (document.footer.left, document.footer.center, document.footer.right) if value] or []
            canvas.setFont("Helvetica", 9)
            if header_values:
                header_style = stylesheet.for_element(StyleElement.HEADER, token_names=document.header.style_tokens)
                if header_style.text_color is not None:
                    canvas.setFillColor(colors.HexColor(header_style.text_color))
                canvas.drawString(pdf.leftMargin, A4[1] - 30, " | ".join(header_values))
                canvas.setFillColor(colors.black)
            if footer_values:
                footer_style = stylesheet.for_element(StyleElement.FOOTER, token_names=document.footer.style_tokens)
                if footer_style.text_color is not None:
                    canvas.setFillColor(colors.HexColor(footer_style.text_color))
                canvas.drawString(pdf.leftMargin, 20, " | ".join(footer_values))
            canvas.restoreState()

        pdf.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
        return buffer.getvalue()


def _style_attribute(style: ResolvedStyle, *, border_color: str | None = None) -> str:
    declarations = style_css(style)
    if border_color is not None:
        declarations = "; ".join(value for value in (declarations, f"border-left-color: {border_color}") if value)
    if not declarations:
        return ""
    return f' style="{declarations}"'
