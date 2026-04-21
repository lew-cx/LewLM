"""Catalog of built-in deterministic document skills."""

from __future__ import annotations

from lewlm.core.errors import PackUnavailableError, SkillNotFoundError
from lewlm.documents.ir.models import DocumentOutputFormat
from lewlm.documents.skills.models import BuiltInSkillDescriptor, DocumentSkillName


_ALL_OUTPUT_FORMATS = list(DocumentOutputFormat)


class DocumentSkillCatalogService:
    """Expose the built-in document skill catalog."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        disabled_reason: str | None = None,
    ) -> None:
        self._enabled = enabled
        self._disabled_reason = disabled_reason or "Document skills are disabled for this LewLM process."
        self._skills = {
            descriptor.name: descriptor
            for descriptor in (
                BuiltInSkillDescriptor(
                    name=DocumentSkillName.CONTRACT_TEXT_REPLACEMENT,
                    description="Replace placeholders inside deterministic contract text and render the result as a document artifact.",
                    supported_input_hints=["template_text", "replacements"],
                    supported_output_formats=list(_ALL_OUTPUT_FORMATS),
                    example_path="examples/contract-transform.json",
                    tags=["contracts", "templating", "documents"],
                ),
                BuiltInSkillDescriptor(
                    name=DocumentSkillName.RECEIPT_EXTRACTION,
                    description="Normalize receipt line items and totals into a structured document artifact for exports and downstream review.",
                    supported_input_hints=["vendor", "items", "totals"],
                    supported_output_formats=list(_ALL_OUTPUT_FORMATS),
                    example_path="examples/receipt-transform.json",
                    tags=["receipts", "extraction", "documents"],
                ),
                BuiltInSkillDescriptor(
                    name=DocumentSkillName.BRANDED_DOCUMENT_TEMPLATE,
                    description="Generate a branded document from structured settings, content sections, and optional local logo or hero images.",
                    supported_input_hints=["settings", "summary", "key_points", "sections"],
                    supported_output_formats=list(_ALL_OUTPUT_FORMATS),
                    example_path="examples/branded-document-template.json",
                    tags=["templates", "branding", "documents", "images"],
                ),
                BuiltInSkillDescriptor(
                    name=DocumentSkillName.FILE_TEMPLATE,
                    description="Render a local JSON DocumentIR template with replacement values and emit a deterministic output artifact.",
                    supported_input_hints=["template_path", "replacements"],
                    supported_output_formats=list(_ALL_OUTPUT_FORMATS),
                    example_path="examples/file-template-transform.json",
                    tags=["templates", "documents", "reusable"],
                ),
                BuiltInSkillDescriptor(
                    name=DocumentSkillName.DOCUMENT_COMPARISON,
                    description="Compare two text bodies into a structured summary with shared and unique sections for each side.",
                    supported_input_hints=["left_text", "right_text", "left_title", "right_title"],
                    supported_output_formats=list(_ALL_OUTPUT_FORMATS),
                    example_path="examples/document-compare-transform.json",
                    tags=["comparison", "summaries", "documents"],
                ),
                BuiltInSkillDescriptor(
                    name=DocumentSkillName.OCR_ASSISTED_EXTRACTION,
                    description="Turn OCR-like scanned-document text into a deterministic field table plus review notes for export or operator review.",
                    supported_input_hints=["ocr_text", "expected_fields", "source_title"],
                    supported_output_formats=list(_ALL_OUTPUT_FORMATS),
                    example_path="examples/ocr-assisted-extraction.json",
                    tags=["ocr", "extraction", "scanned-documents"],
                ),
                BuiltInSkillDescriptor(
                    name=DocumentSkillName.MEETING_TRANSCRIPT_NOTES,
                    description="Turn a meeting transcript into deterministic notes, extracted decisions, and action-item exports.",
                    supported_input_hints=["transcript_text", "participants", "meeting_date"],
                    supported_output_formats=list(_ALL_OUTPUT_FORMATS),
                    example_path="examples/meeting-transcript-notes.json",
                    tags=["meetings", "transcripts", "notes", "action-items"],
                ),
                BuiltInSkillDescriptor(
                    name=DocumentSkillName.LONG_DOCUMENT_MEMO,
                    description="Turn long-form source text into a structured memo with highlights, open questions, and a compact outline.",
                    supported_input_hints=["source_title", "source_text"],
                    supported_output_formats=list(_ALL_OUTPUT_FORMATS),
                    example_path="examples/long-document-memo.json",
                    tags=["summaries", "memo", "documents"],
                ),
                BuiltInSkillDescriptor(
                    name=DocumentSkillName.SPEECH_TRANSCRIPT_CLEANUP,
                    description="Normalize transcript turns into cleaned speaker-attributed text that can be exported or reviewed locally.",
                    supported_input_hints=["transcript_text", "language"],
                    supported_output_formats=list(_ALL_OUTPUT_FORMATS),
                    example_path="examples/speech-transcript-cleanup.json",
                    tags=["speech", "transcripts", "cleanup"],
                ),
            )
        }

    def list_skills(self) -> list[BuiltInSkillDescriptor]:
        if not self._enabled:
            return []
        return sorted(self._skills.values(), key=lambda descriptor: descriptor.name)

    def get_skill(self, skill_name: str) -> BuiltInSkillDescriptor:
        if not self._enabled:
            raise PackUnavailableError(
                "Built-in skills are unavailable because the `documents` feature pack is disabled.",
                details={"pack": "documents", "reason": self._disabled_reason},
            )
        try:
            return self._skills[skill_name]
        except KeyError as exc:
            raise SkillNotFoundError(
                "Built-in skill was not found.",
                details={"skill": skill_name},
            ) from exc
