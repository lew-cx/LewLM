from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from lewlm.documents.ingest.models import DocumentSourceType
from lewlm.documents.ingest.ocr import OcrExtractionResult
from lewlm.documents.ingest.service import DocumentIngestService
from lewlm.documents.ir.models import ImageBlock, ListBlock, ParagraphBlock, TableBlock


def test_document_ingest_service_extracts_supported_file_types(temp_settings, sample_ingest_sources) -> None:
    service = DocumentIngestService(workspace_root=temp_settings.temp_dir)

    csv_result = service.ingest([sample_ingest_sources["csv"]], allowed_file_roots=(temp_settings.data_dir,))
    xlsx_result = service.ingest([sample_ingest_sources["xlsx"]], allowed_file_roots=(temp_settings.data_dir,))
    docx_result = service.ingest([sample_ingest_sources["docx"]], allowed_file_roots=(temp_settings.data_dir,))
    text_result = service.ingest([sample_ingest_sources["text"]], allowed_file_roots=(temp_settings.data_dir,))
    markdown_result = service.ingest([sample_ingest_sources["markdown"]], allowed_file_roots=(temp_settings.data_dir,))
    pdf_result = service.ingest([sample_ingest_sources["pdf"]], allowed_file_roots=(temp_settings.data_dir,))
    image_result = service.ingest([sample_ingest_sources["image_bundle"]], allowed_file_roots=(temp_settings.data_dir,))

    csv_table = csv_result.document.sections[0].blocks[0]
    assert csv_result.sources[0].source_type == DocumentSourceType.CSV
    assert isinstance(csv_table, TableBlock)
    assert csv_table.headers == ["Category", "Amount"]
    assert xlsx_result.sources[0].source_type == DocumentSourceType.XLSX
    assert any(
        isinstance(block, TableBlock) and any(row and row[0] == "Hosting" for row in block.rows)
        for section in xlsx_result.document.sections
        for block in section.blocks
    )

    assert docx_result.sources[0].source_type == DocumentSourceType.DOCX
    assert any(
        isinstance(block, ParagraphBlock) and "Operations remained on track" in block.text
        for section in docx_result.document.sections
        for block in section.blocks
    )

    assert text_result.sources[0].source_type == DocumentSourceType.TEXT
    assert any(
        isinstance(block, ParagraphBlock) and "Escalate Linux host audit follow-up next." in block.text
        for section in text_result.document.sections
        for block in section.blocks
    )

    assert markdown_result.sources[0].source_type == DocumentSourceType.MARKDOWN
    assert any(
        isinstance(block, ListBlock) and block.items == ["Confirm Linux validation host booking", "Refresh milestone tracker"]
        for section in markdown_result.document.sections
        for block in section.blocks
    )
    assert markdown_result.sources[0].metadata["list_count"] == 2

    assert pdf_result.sources[0].source_type == DocumentSourceType.PDF
    assert any(
        isinstance(block, ParagraphBlock) and "Operations remained on track" in block.text
        for section in pdf_result.document.sections
        for block in section.blocks
    )

    assert image_result.sources[0].source_type == DocumentSourceType.IMAGE_BUNDLE
    assert any(isinstance(block, ImageBlock) for block in image_result.document.sections[0].blocks)
    assert csv_result.chunks
    assert pdf_result.document.metadata["chunk_count"] == len(pdf_result.chunks)
    assert image_result.document.metadata["ocr_available"] in {True, False}
    assert csv_result.sources[0].media_type == "text/csv"
    assert csv_result.document.sections[0].metadata["source_id"] == csv_result.sources[0].source_id
    assert csv_result.document.sections[0].metadata["section_id"].startswith(f"{csv_result.sources[0].source_id}-sec-")
    assert csv_result.chunks[0].source_id == csv_result.sources[0].source_id
    assert csv_result.chunks[0].chunk_id.startswith(f"{csv_result.chunks[0].section_id}-chunk-")
    assert pdf_result.chunks[0].section_label.endswith("Page 1")
    assert pdf_result.chunks[0].metadata["page_number"] == 1


def test_document_ingest_service_returns_stable_source_and_chunk_packaging(
    temp_settings,
    sample_ingest_sources,
) -> None:
    service = DocumentIngestService(workspace_root=temp_settings.temp_dir)

    first = service.ingest([sample_ingest_sources["markdown"]], allowed_file_roots=(temp_settings.data_dir,))
    second = service.ingest([sample_ingest_sources["markdown"]], allowed_file_roots=(temp_settings.data_dir,))

    assert first.sources[0].source_id == second.sources[0].source_id
    assert first.sources[0].source_label == "sample.md"
    assert first.sources[0].source_name == "sample.md"
    assert first.sources[0].media_type == "text/markdown"
    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]
    assert first.document.sections[0].metadata["section_label"] == "sample.md / Quarterly operations summary"
    assert first.document.sections[1].metadata["section_label"] == "sample.md / Action Items"
    assert first.chunks[0].source_label == "sample.md"
    assert first.chunks[0].section_id == first.document.sections[0].metadata["section_id"]
    assert first.chunks[0].metadata["source_media_type"] == "text/markdown"


def test_document_ingest_service_uses_ocr_for_scanned_pdf_pages(temp_settings, tmp_path: Path, monkeypatch) -> None:
    source_image = tmp_path / "scanned-page.png"
    scanned_pdf = tmp_path / "scanned.pdf"

    image = Image.new("RGB", (240, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 40), "Scanned milestone note", fill="black")
    image.save(source_image)

    pdf_canvas = canvas.Canvas(str(scanned_pdf))
    pdf_canvas.drawImage(ImageReader(str(source_image)), 72, 640, width=240, height=100)
    pdf_canvas.save()

    monkeypatch.setattr(
        "lewlm.documents.ingest.service.perform_ocr_on_image_bytes",
        lambda _raw: OcrExtractionResult(text="Scanned OCR text", backend_name="test-ocr"),
    )

    service = DocumentIngestService(workspace_root=temp_settings.temp_dir, sandbox_enabled=False)
    result = service.ingest([scanned_pdf], allowed_file_roots=(tmp_path,))

    assert result.sources[0].source_type == DocumentSourceType.PDF
    assert result.sources[0].metadata["ocr_used"] is True
    assert any(
        isinstance(block, ParagraphBlock) and "Scanned OCR text" in block.text
        for section in result.document.sections
        for block in section.blocks
    )
    assert any(
        isinstance(block, ImageBlock) and block.mime_type is not None
        for section in result.document.sections
        for block in section.blocks
    )


def test_document_ingest_service_resolves_relative_paths_from_base_dir_without_explicit_roots(
    temp_settings,
    sample_ingest_sources,
) -> None:
    service = DocumentIngestService(workspace_root=temp_settings.temp_dir)

    result = service.ingest(
        [sample_ingest_sources["csv"].name],
        base_dir=sample_ingest_sources["csv"].parent,
    )

    assert result.sources[0].source_type == DocumentSourceType.CSV
    assert result.sources[0].path == str(sample_ingest_sources["csv"].resolve(strict=False))
