"""Deterministic document skill implementations."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from lewlm.core.errors import DocumentValidationError, PackUnavailableError
from lewlm.documents.ir.models import (
    CalloutBlock,
    DocumentIR,
    DocumentSection,
    HeaderFooterContent,
    ImageBlock,
    ListBlock,
    ParagraphBlock,
    TableBlock,
)
from lewlm.documents.service import DocumentGenerationService
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.documents.skills.models import (
    BrandedDocumentTemplateInput,
    BrandedDocumentTemplateRequest,
    ContractTextReplacementInput,
    ContractTextReplacementRequest,
    DocumentComparisonInput,
    DocumentComparisonRequest,
    DocumentTransformRequest,
    FileTemplateTransformRequest,
    LongDocumentMemoInput,
    LongDocumentMemoRequest,
    MeetingTranscriptNotesInput,
    MeetingTranscriptNotesRequest,
    OCRAssistedExtractionField,
    OCRAssistedExtractionInput,
    OCRAssistedExtractionRequest,
    ReceiptExtractionInput,
    ReceiptExtractionRequest,
    SpeechTranscriptCleanupInput,
    SpeechTranscriptCleanupRequest,
)
from lewlm.security.files import read_scoped_text_file, validate_scoped_image_file


PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}|\[\[\s*([a-zA-Z0-9_]+)\s*\]\]")
TRANSCRIPT_SPEAKER_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9 ._'/-]{0,63})\s*:\s*(.+?)\s*$")
TRANSCRIPT_ACTION_PATTERN = re.compile(r"^\s*(?:ACTION|TODO)\s*:\s*(.+?)\s*$", re.IGNORECASE)
OCR_FIELD_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9 ._/#()%-]{0,63}?)\s*(?::| - )\s*(.+?)\s*$")
DECISION_MARKERS = ("decision:", "decided", "agreed", "approve", "approved", "confirmed", "will proceed")


class DocumentTransformService:
    """Run built-in document skills and render their output artifacts."""

    def __init__(
        self,
        *,
        generation_service: DocumentGenerationService | None = None,
        event_bus: EventBus | None = None,
        enabled: bool = True,
        disabled_reason: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.disabled_reason = disabled_reason or "Document transform is disabled for this LewLM process."
        self.generation_service = generation_service or DocumentGenerationService(
            event_bus=event_bus,
            enabled=enabled,
            disabled_reason=self.disabled_reason,
        )
        self.event_bus = event_bus

    def transform(
        self,
        request: DocumentTransformRequest,
        *,
        allowed_file_roots: Sequence[Path | str] | None = None,
        base_dir: Path | str | None = None,
        request_id: str | None = None,
    ):
        self._ensure_enabled()
        resolved_request_id = request_id or str(uuid4())
        self._publish(
            EventType.DOCUMENT_TRANSFORM_STARTED,
            {
                "request_id": resolved_request_id,
                "skill": request.skill,
                "output_format": request.output_format.value,
            },
        )
        generation_base_dir = base_dir
        try:
            if isinstance(request, ContractTextReplacementRequest):
                document = self._build_contract_document(request.input)
            elif isinstance(request, ReceiptExtractionRequest):
                document = self._build_receipt_document(request.input)
            elif isinstance(request, BrandedDocumentTemplateRequest):
                document = self._build_branded_document_template(
                    request.input,
                    allowed_file_roots=allowed_file_roots,
                    base_dir=base_dir,
                )
            elif isinstance(request, OCRAssistedExtractionRequest):
                document = self._build_ocr_assisted_extraction(request.input)
            elif isinstance(request, DocumentComparisonRequest):
                document = self._build_document_comparison(request.input)
            elif isinstance(request, MeetingTranscriptNotesRequest):
                document = self._build_meeting_transcript_notes(request.input)
            elif isinstance(request, LongDocumentMemoRequest):
                document = self._build_long_document_memo(request.input)
            elif isinstance(request, SpeechTranscriptCleanupRequest):
                document = self._build_speech_transcript_cleanup(request.input)
            elif isinstance(request, FileTemplateTransformRequest):
                document, generation_base_dir = self._build_template_document(
                    request,
                    allowed_file_roots=allowed_file_roots,
                    base_dir=base_dir,
                )
            else:
                raise DocumentValidationError("Unsupported transform request.")
            self._publish(
                EventType.OPERATION_PROGRESS,
                {
                    "request_id": resolved_request_id,
                    "operation": "document.transform",
                    "stage": "document_built",
                    "completed_steps": 1,
                    "total_steps": 2,
                    "progress": 0.5,
                    "skill": request.skill,
                },
            )
            artifact = self.generation_service.generate(
                document,
                output_format=request.output_format,
                file_name=request.file_name,
                allowed_file_roots=allowed_file_roots,
                base_dir=generation_base_dir,
                request_id=resolved_request_id,
            )
            self._publish(
                EventType.DOCUMENT_TRANSFORM_COMPLETED,
                {
                    "request_id": resolved_request_id,
                    "skill": request.skill,
                    "output_format": request.output_format.value,
                    "file_name": artifact.file_name,
                    "size_bytes": artifact.size_bytes,
                },
            )
            return artifact
        except Exception as exc:
            self._publish(
                EventType.DOCUMENT_TRANSFORM_FAILED,
                {
                    "request_id": resolved_request_id,
                    "skill": request.skill,
                    "output_format": request.output_format.value,
                    "error": str(exc),
                },
            )
            raise

    def _ensure_enabled(self) -> None:
        if self.enabled:
            return
        raise PackUnavailableError(
            "Document transforms are unavailable because the `documents` feature pack is disabled.",
            details={"pack": "documents", "reason": self.disabled_reason},
        )

    def _build_contract_document(self, payload: ContractTextReplacementInput) -> DocumentIR:
        def replace_placeholder(match: re.Match[str]) -> str:
            key = next(group for group in match.groups() if group is not None)
            try:
                return payload.replacements[key]
            except KeyError as exc:
                raise DocumentValidationError(
                    "Contract replacement is missing a required placeholder value.",
                    details={"placeholder": key},
                ) from exc

        rendered_text = PLACEHOLDER_PATTERN.sub(replace_placeholder, payload.template_text)
        unresolved = [match.group(0) for match in PLACEHOLDER_PATTERN.finditer(rendered_text)]
        if unresolved:
            raise DocumentValidationError(
                "Contract replacement left unresolved placeholders in the template.",
                details={"placeholders": unresolved},
            )
        paragraphs = [ParagraphBlock(text=segment.strip()) for segment in re.split(r"\n\s*\n", rendered_text) if segment.strip()]
        if not paragraphs:
            raise DocumentValidationError("Contract replacement produced an empty document.")
        return DocumentIR(
            title=payload.title,
            metadata={"skill": "contract_text_replacement"},
            sections=[DocumentSection(heading="Contract", blocks=paragraphs)],
        )

    def _build_receipt_document(self, payload: ReceiptExtractionInput) -> DocumentIR:
        if not payload.items:
            raise DocumentValidationError("Receipt extraction requires at least one line item.")
        summary_lines = [f"Vendor: {payload.vendor}"]
        if payload.receipt_number:
            summary_lines.append(f"Receipt Number: {payload.receipt_number}")
        if payload.purchased_at:
            summary_lines.append(f"Purchased At: {payload.purchased_at}")
        if payload.currency:
            summary_lines.append(f"Currency: {payload.currency}")

        totals = [value for value in (payload.subtotal, payload.tax, payload.total) if value is not None]
        total_body = " | ".join(
            filter(
                None,
                [
                    f"Subtotal: {payload.subtotal}" if payload.subtotal else None,
                    f"Tax: {payload.tax}" if payload.tax else None,
                    f"Total: {payload.total}" if payload.total else None,
                ],
            ),
        )

        return DocumentIR(
            title=payload.title,
            metadata={"skill": "receipt_extraction", "vendor": payload.vendor},
            sections=[
                DocumentSection(
                    heading="Receipt Summary",
                    blocks=[ParagraphBlock(text=line) for line in summary_lines]
                    + ([CalloutBlock(kind="info", title="Totals", body=total_body)] if totals else []),
                ),
                DocumentSection(
                    heading="Line Items",
                    blocks=[
                        TableBlock(
                            headers=["Description", "Quantity", "Unit Price", "Line Total"],
                            rows=[
                                [
                                    item.description,
                                    item.quantity,
                                    item.unit_price or "",
                                    item.total or "",
                                ]
                                for item in payload.items
                            ],
                            caption="Extracted receipt line items",
                        ),
                    ],
                ),
            ],
        )

    def _build_branded_document_template(
        self,
        payload: BrandedDocumentTemplateInput,
        *,
        allowed_file_roots: Sequence[Path | str] | None,
        base_dir: Path | str | None,
    ) -> DocumentIR:
        summary_text = payload.summary.strip()
        if not summary_text:
            raise DocumentValidationError("Branded document template requires a non-empty summary.")

        rendered_sections: list[DocumentSection] = []
        for section in payload.sections:
            blocks: list[ParagraphBlock | ListBlock | CalloutBlock] = []
            paragraphs = [value.strip() for value in section.paragraphs if value.strip()]
            bullets = [value.strip() for value in section.bullets if value.strip()]
            if paragraphs:
                blocks.extend(ParagraphBlock(text=paragraph) for paragraph in paragraphs)
            if bullets:
                blocks.append(ListBlock(items=bullets))
            if section.callout_body and section.callout_body.strip():
                blocks.append(
                    CalloutBlock(
                        kind="note",
                        title=section.callout_title.strip() if section.callout_title and section.callout_title.strip() else None,
                        body=section.callout_body.strip(),
                    ),
                )
            if not blocks:
                raise DocumentValidationError(
                    "Branded document sections must include paragraph, bullet, or callout content.",
                    details={"heading": section.heading},
                )
            rendered_sections.append(DocumentSection(heading=section.heading, blocks=blocks))

        key_points = [item.strip() for item in payload.key_points if item.strip()]
        if not rendered_sections and not key_points:
            raise DocumentValidationError(
                "Branded document template requires at least one content section or key point.",
            )

        settings = payload.settings
        logo_path = self._resolve_branded_image_path(
            settings.logo_path,
            allowed_file_roots=allowed_file_roots,
            base_dir=base_dir,
            label="logo",
        )
        hero_image_path = self._resolve_branded_image_path(
            settings.hero_image_path,
            allowed_file_roots=allowed_file_roots,
            base_dir=base_dir,
            label="hero image",
        )

        overview_blocks: list[ParagraphBlock | ListBlock | CalloutBlock | TableBlock | ImageBlock] = []
        if logo_path is not None:
            overview_blocks.append(
                ImageBlock(
                    alt_text=f"{settings.organization_name} logo",
                    caption=f"{settings.organization_name} logo",
                    role="logo",
                    path=logo_path,
                ),
            )
        if settings.subtitle:
            overview_blocks.append(ParagraphBlock(text=settings.subtitle))
        overview_blocks.append(CalloutBlock(kind="info", title="Summary", body=summary_text))
        if hero_image_path is not None:
            overview_blocks.append(
                ImageBlock(
                    alt_text=f"{payload.title} hero image",
                    caption=f"{payload.title} visual",
                    role="image",
                    path=hero_image_path,
                ),
            )
        if key_points:
            overview_blocks.append(ListBlock(items=key_points))

        settings_rows = [
            [label, value]
            for label, value in (
                ("Organization", settings.organization_name),
                ("Audience", settings.audience),
                ("Issued On", settings.issued_on),
                ("Contact", settings.contact_line),
            )
            if value
        ]
        if settings_rows:
            overview_blocks.append(
                TableBlock(
                    headers=["Setting", "Value"],
                    rows=settings_rows,
                    caption="Document settings",
                ),
            )

        return DocumentIR(
            title=payload.title,
            metadata={
                "skill": "branded_document_template",
                "organization_name": settings.organization_name,
                "section_count": len(rendered_sections),
                "key_point_count": len(key_points),
                "has_logo": logo_path is not None,
                "has_hero_image": hero_image_path is not None,
            },
            header=HeaderFooterContent(
                left=settings.organization_name,
                center=settings.header_text,
                right=settings.issued_on,
            ),
            footer=HeaderFooterContent(
                left=settings.contact_line,
                center=settings.footer_text,
                right=settings.audience,
            ),
            sections=[
                DocumentSection(
                    heading="Overview",
                    blocks=overview_blocks,
                ),
                *rendered_sections,
            ],
        )

    def _build_document_comparison(self, payload: DocumentComparisonInput) -> DocumentIR:
        left_segments = self._normalize_text_segments(payload.left_text, label="left_text")
        right_segments = self._normalize_text_segments(payload.right_text, label="right_text")

        left_map = {segment: segment for segment in left_segments}
        right_map = {segment: segment for segment in right_segments}

        shared_segments = [left_map[key] for key in left_map if key in right_map]
        left_only_segments = [left_map[key] for key in left_map if key not in right_map]
        right_only_segments = [right_map[key] for key in right_map if key not in left_map]

        shared_count = len(shared_segments)
        left_only_count = len(left_only_segments)
        right_only_count = len(right_only_segments)
        total_unique = len(set(left_map) | set(right_map))
        overlap_percent = 100.0 if total_unique == 0 else round((shared_count / total_unique) * 100, 1)

        if left_only_count == 0 and right_only_count == 0:
            overview_callout = CalloutBlock(
                kind="success",
                title="Documents Match",
                body="The compared documents are identical after whitespace-normalized paragraph comparison.",
            )
        else:
            overview_callout = CalloutBlock(
                kind="warning",
                title="Differences Detected",
                body=(
                    f"Found {shared_count} shared segment(s), "
                    f"{left_only_count} segment(s) only in {payload.left_title}, "
                    f"and {right_only_count} segment(s) only in {payload.right_title}."
                ),
            )

        return DocumentIR(
            title=payload.title,
            metadata={
                "skill": "document_comparison",
                "left_title": payload.left_title,
                "right_title": payload.right_title,
                "shared_segment_count": shared_count,
                "left_only_segment_count": left_only_count,
                "right_only_segment_count": right_only_count,
            },
            sections=[
                DocumentSection(
                    heading="Overview",
                    blocks=[
                        overview_callout,
                        TableBlock(
                            headers=["Metric", "Value"],
                            rows=[
                                ["Left document", payload.left_title],
                                ["Right document", payload.right_title],
                                ["Shared segments", str(shared_count)],
                                [f"Only in {payload.left_title}", str(left_only_count)],
                                [f"Only in {payload.right_title}", str(right_only_count)],
                                ["Overlap", f"{overlap_percent}%"],
                            ],
                            caption="Comparison summary",
                        ),
                    ],
                ),
                DocumentSection(
                    heading="Shared Segments",
                    blocks=self._comparison_block(
                        items=shared_segments,
                        empty_message="No shared segments were detected.",
                    ),
                ),
                DocumentSection(
                    heading=f"Only in {payload.left_title}",
                    blocks=self._comparison_block(
                        items=left_only_segments,
                        empty_message=f"No unique segments were detected in {payload.left_title}.",
                    ),
                ),
                DocumentSection(
                    heading=f"Only in {payload.right_title}",
                    blocks=self._comparison_block(
                        items=right_only_segments,
                        empty_message=f"No unique segments were detected in {payload.right_title}.",
                    ),
                ),
            ],
        )

    def _build_ocr_assisted_extraction(self, payload: OCRAssistedExtractionInput) -> DocumentIR:
        parsed_fields, review_lines = self._parse_ocr_lines(payload.ocr_text)
        if not parsed_fields and not payload.expected_fields:
            raise DocumentValidationError(
                "OCR-assisted extraction requires field-like OCR text or at least one expected field target.",
            )

        rows: list[list[str]] = []
        matched_indexes: set[int] = set()
        if payload.expected_fields:
            for target in payload.expected_fields:
                matched_index = self._match_ocr_target(target, parsed_fields, matched_indexes)
                if matched_index is None:
                    rows.append([target.field, "", "missing"])
                    continue
                matched_indexes.add(matched_index)
                rows.append([target.field, parsed_fields[matched_index]["value"], "extracted"])
            for index, field in enumerate(parsed_fields):
                if index not in matched_indexes:
                    rows.append([field["label"], field["value"], "additional"])
        else:
            rows = [[field["label"], field["value"], "extracted"] for field in parsed_fields]

        if not rows:
            raise DocumentValidationError("OCR-assisted extraction did not yield any exportable rows.")

        extracted_count = sum(1 for row in rows if row[2] == "extracted")
        missing_count = sum(1 for row in rows if row[2] == "missing")
        additional_count = sum(1 for row in rows if row[2] == "additional")
        overview_blocks: list[CalloutBlock | ParagraphBlock] = [
            CalloutBlock(
                kind="info",
                title="Extraction Summary",
                body=(
                    f"Prepared {len(rows)} field row(s) from OCR text with {extracted_count} extracted value(s), "
                    f"{missing_count} missing target(s), and {additional_count} additional detected field(s)."
                ),
            ),
            ParagraphBlock(text=f"Source: {payload.source_title}"),
        ]
        if payload.document_type:
            overview_blocks.append(ParagraphBlock(text=f"Document Type: {payload.document_type}"))

        return DocumentIR(
            title=payload.title,
            metadata={
                "skill": "ocr_assisted_extraction",
                "source_title": payload.source_title,
                "document_type": payload.document_type,
                "field_row_count": len(rows),
                "review_line_count": len(review_lines),
                "expected_field_count": len(payload.expected_fields),
            },
            sections=[
                DocumentSection(
                    heading="Extracted Fields",
                    blocks=[
                        TableBlock(
                            headers=["Field", "Value", "Status"],
                            rows=rows,
                            caption="Structured fields extracted from OCR text",
                        ),
                    ],
                ),
                DocumentSection(
                    heading="Overview",
                    blocks=overview_blocks,
                ),
                DocumentSection(
                    heading="Review Notes",
                    blocks=(
                        [ListBlock(items=review_lines)]
                        if review_lines
                        else [ParagraphBlock(text="No unmapped OCR lines were detected.")]
                    ),
                ),
            ],
        )

    def _build_meeting_transcript_notes(self, payload: MeetingTranscriptNotesInput) -> DocumentIR:
        transcript_entries, explicit_action_rows = self._parse_meeting_transcript(payload.transcript_text)
        if not transcript_entries and not explicit_action_rows:
            raise DocumentValidationError("Meeting transcript notes require non-empty transcript content.")

        participants = list(dict.fromkeys(payload.participants + self._extract_participants(transcript_entries)))
        highlight_items = [
            self._format_transcript_entry(entry["speaker"], entry["text"])
            for entry in transcript_entries[:6]
        ]
        decision_items = [
            self._format_transcript_entry(entry["speaker"], entry["text"])
            for entry in transcript_entries
            if self._contains_decision_marker(entry["text"])
        ]

        action_rows = explicit_action_rows or self._derive_action_rows(transcript_entries)
        if not action_rows:
            action_rows = [["Unassigned", "No explicit action items were detected in the transcript.", ""]]

        overview_blocks: list[CalloutBlock | ParagraphBlock] = [
            CalloutBlock(
                kind="info",
                title="Meeting Summary",
                body=(
                    f"Captured {len(highlight_items) or len(transcript_entries)} discussion highlight(s), "
                    f"{len(decision_items)} decision(s), and {len(action_rows)} action item(s)."
                ),
            ),
        ]
        if participants:
            overview_blocks.append(ParagraphBlock(text="Participants: " + ", ".join(participants)))
        if payload.meeting_date:
            overview_blocks.append(ParagraphBlock(text=f"Meeting Date: {payload.meeting_date}"))

        return DocumentIR(
            title=payload.title,
            metadata={
                "skill": "meeting_transcript_notes",
                "participant_count": len(participants),
                "highlight_count": len(highlight_items),
                "decision_count": len(decision_items),
                "action_item_count": len(action_rows),
            },
            sections=[
                DocumentSection(
                    heading="Action Items",
                    blocks=[
                        TableBlock(
                            headers=["Owner", "Action", "Due"],
                            rows=action_rows,
                            caption="Action items extracted from the meeting transcript",
                        ),
                    ],
                ),
                DocumentSection(
                    heading="Overview",
                    blocks=overview_blocks,
                ),
                DocumentSection(
                    heading="Discussion Highlights",
                    blocks=(
                        [ListBlock(items=highlight_items, ordered=True)]
                        if highlight_items
                        else [ParagraphBlock(text="No discussion highlights were detected.")]
                    ),
                ),
                DocumentSection(
                    heading="Decisions",
                    blocks=(
                        [ListBlock(items=list(dict.fromkeys(decision_items)))]
                        if decision_items
                        else [ParagraphBlock(text="No explicit decisions were detected.")]
                    ),
                ),
            ],
        )

    def _build_long_document_memo(self, payload: LongDocumentMemoInput) -> DocumentIR:
        segments = self._normalize_text_segments(payload.source_text, label="source_text")
        highlight_items = [self._truncate_segment(segment) for segment in segments[:5]]
        question_items = [self._truncate_segment(segment) for segment in segments if "?" in segment]
        summary_text = " ".join(highlight_items[:2]) if highlight_items else segments[0]
        outline_rows = [
            [str(index), self._truncate_segment(segment, limit=160)]
            for index, segment in enumerate(segments[:8], start=1)
        ]

        return DocumentIR(
            title=payload.title,
            metadata={
                "skill": "long_document_memo",
                "source_title": payload.source_title,
                "segment_count": len(segments),
                "highlight_count": len(highlight_items),
                "question_count": len(question_items),
            },
            sections=[
                DocumentSection(
                    heading="Memo Summary",
                    blocks=[
                        CalloutBlock(
                            kind="info",
                            title="Executive Summary",
                            body=(
                                f"Prepared a structured memo from {payload.source_title} using {len(segments)} normalized "
                                f"section(s). {summary_text}"
                            ),
                        ),
                        TableBlock(
                            headers=["Metric", "Value"],
                            rows=[
                                ["Source document", payload.source_title],
                                ["Sections analyzed", str(len(segments))],
                                ["Highlights captured", str(len(highlight_items))],
                                ["Open questions", str(len(question_items))],
                            ],
                            caption="Memo summary",
                        ),
                    ],
                ),
                DocumentSection(
                    heading="Key Highlights",
                    blocks=(
                        [ListBlock(items=highlight_items, ordered=True)]
                        if highlight_items
                        else [ParagraphBlock(text="No highlights were detected.")]
                    ),
                ),
                DocumentSection(
                    heading="Open Questions",
                    blocks=(
                        [ListBlock(items=list(dict.fromkeys(question_items)))]
                        if question_items
                        else [ParagraphBlock(text="No open questions were detected.")]
                    ),
                ),
                DocumentSection(
                    heading="Source Outline",
                    blocks=[
                        TableBlock(
                            headers=["Section", "Excerpt"],
                            rows=outline_rows,
                            caption="Source outline",
                        ),
                    ],
                ),
            ],
        )

    def _build_speech_transcript_cleanup(self, payload: SpeechTranscriptCleanupInput) -> DocumentIR:
        transcript_entries = self._parse_cleanup_transcript(payload.transcript_text)
        if not transcript_entries:
            raise DocumentValidationError("Speech transcript cleanup requires non-empty transcript content.")

        participants = [
            speaker
            for speaker in dict.fromkeys(entry["speaker"] for entry in transcript_entries if entry["speaker"] != "Unknown")
        ]
        turn_rows = [[entry["speaker"], entry["text"]] for entry in transcript_entries]
        unknown_turns = sum(1 for entry in transcript_entries if entry["speaker"] == "Unknown")

        overview_blocks: list[CalloutBlock | ParagraphBlock] = [
            CalloutBlock(
                kind="info",
                title="Cleanup Summary",
                body=(
                    f"Normalized {len(transcript_entries)} transcript turn(s) with "
                    f"{len(participants)} named speaker(s) and {unknown_turns} unattributed turn(s)."
                ),
            ),
        ]
        if participants:
            overview_blocks.append(ParagraphBlock(text="Speakers: " + ", ".join(participants)))
        if payload.language:
            overview_blocks.append(ParagraphBlock(text=f"Language: {payload.language}"))

        return DocumentIR(
            title=payload.title,
            metadata={
                "skill": "speech_transcript_cleanup",
                "speaker_count": len(participants),
                "turn_count": len(transcript_entries),
                "unknown_turn_count": unknown_turns,
            },
            sections=[
                DocumentSection(
                    heading="Cleaned Transcript",
                    blocks=[
                        TableBlock(
                            headers=["Speaker", "Utterance"],
                            rows=turn_rows,
                            caption="Normalized transcript turns",
                        ),
                    ],
                ),
                DocumentSection(
                    heading="Overview",
                    blocks=overview_blocks,
                ),
            ],
        )

    def _build_template_document(
        self,
        request: FileTemplateTransformRequest,
        *,
        allowed_file_roots: Sequence[Path | str] | None,
        base_dir: Path | str | None,
    ) -> tuple[DocumentIR, Path]:
        if allowed_file_roots is None:
            template_path = Path(request.template_path).expanduser().resolve(strict=False)
            if not template_path.exists() or not template_path.is_file():
                raise DocumentValidationError(
                    "File template transform requires an existing JSON template file.",
                    details={"template_path": str(template_path)},
                )
            template_text = template_path.read_text(encoding="utf-8")
        else:
            template_path, template_text = read_scoped_text_file(
                request.template_path,
                allowed_roots=allowed_file_roots,
                purpose="Document template",
                media_type="application/json",
                base_dir=base_dir,
            )
        try:
            template_payload = json.loads(template_text)
        except json.JSONDecodeError as exc:
            raise DocumentValidationError(
                "File template transform requires a valid JSON template file.",
                details={"template_path": str(template_path)},
            ) from exc
        replaced_payload = self._replace_placeholders(template_payload, request.input.replacements)
        try:
            document = DocumentIR.model_validate(replaced_payload)
        except ValidationError as exc:
            raise DocumentValidationError(
                "File template transform did not resolve to a valid document template.",
                details={"template_path": str(template_path)},
            ) from exc
        if request.input.title:
            document = document.model_copy(update={"title": request.input.title})
        unresolved = sorted(set(self._find_unresolved_placeholders(document.model_dump(mode="python"))))
        if unresolved:
            raise DocumentValidationError(
                "File template transform left unresolved placeholders in the template.",
                details={"template_path": str(template_path), "placeholders": unresolved},
            )
        return document, template_path.parent

    def _replace_placeholders(self, payload: Any, replacements: dict[str, str]) -> Any:
        if isinstance(payload, dict):
            return {key: self._replace_placeholders(value, replacements) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self._replace_placeholders(item, replacements) for item in payload]
        if isinstance(payload, str):
            return self._render_template_string(payload, replacements)
        return payload

    def _render_template_string(self, value: str, replacements: dict[str, str]) -> str:
        def replace_placeholder(match: re.Match[str]) -> str:
            key = next(group for group in match.groups() if group is not None)
            try:
                return replacements[key]
            except KeyError as exc:
                raise DocumentValidationError(
                    "File template transform is missing a required placeholder value.",
                    details={"placeholder": key},
                ) from exc

        return PLACEHOLDER_PATTERN.sub(replace_placeholder, value)

    def _find_unresolved_placeholders(self, payload: Any) -> list[str]:
        if isinstance(payload, dict):
            unresolved: list[str] = []
            for value in payload.values():
                unresolved.extend(self._find_unresolved_placeholders(value))
            return unresolved
        if isinstance(payload, list):
            unresolved = []
            for item in payload:
                unresolved.extend(self._find_unresolved_placeholders(item))
            return unresolved
        if isinstance(payload, str):
            return [match.group(0) for match in PLACEHOLDER_PATTERN.finditer(payload)]
        return []

    def _parse_meeting_transcript(self, transcript_text: str) -> tuple[list[dict[str, str]], list[list[str]]]:
        transcript_entries: list[dict[str, str]] = []
        action_rows: list[list[str]] = []
        for raw_line in transcript_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            action_match = TRANSCRIPT_ACTION_PATTERN.match(line)
            if action_match is not None:
                action_rows.append(self._parse_action_row(action_match.group(1).strip()))
                continue
            speaker_match = TRANSCRIPT_SPEAKER_PATTERN.match(line)
            if speaker_match is not None:
                transcript_entries.append(
                    {
                        "speaker": speaker_match.group(1).strip(),
                        "text": speaker_match.group(2).strip(),
                    },
                )
                continue
            transcript_entries.append({"speaker": "", "text": line})
        return transcript_entries, action_rows

    def _parse_cleanup_transcript(self, transcript_text: str) -> list[dict[str, str]]:
        transcript_entries: list[dict[str, str]] = []
        for raw_line in transcript_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            speaker_match = TRANSCRIPT_SPEAKER_PATTERN.match(line)
            if speaker_match is not None:
                speaker = self._normalize_speaker_label(speaker_match.group(1))
                text = self._normalize_transcript_sentence(speaker_match.group(2))
                transcript_entries.append({"speaker": speaker, "text": text})
                continue
            transcript_entries.append({"speaker": "Unknown", "text": self._normalize_transcript_sentence(line)})
        return transcript_entries

    def _parse_action_row(self, payload: str) -> list[str]:
        if "|" in payload:
            parts = [part.strip() for part in payload.split("|", maxsplit=2)]
            if len(parts) == 2:
                return [parts[0] or "Unassigned", parts[1], ""]
            if len(parts) == 3:
                return [parts[0] or "Unassigned", parts[1], parts[2]]
        return ["Unassigned", payload, ""]

    def _extract_participants(self, transcript_entries: list[dict[str, str]]) -> list[str]:
        return [
            speaker
            for speaker in dict.fromkeys(entry["speaker"] for entry in transcript_entries if entry["speaker"])
        ]

    def _parse_ocr_lines(self, ocr_text: str) -> tuple[list[dict[str, str]], list[str]]:
        parsed_fields: list[dict[str, str]] = []
        review_lines: list[str] = []
        for raw_line in ocr_text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            field_match = OCR_FIELD_PATTERN.match(line)
            if field_match is None:
                review_lines.append(line)
                continue
            label = re.sub(r"\s+", " ", field_match.group(1)).strip()
            value = re.sub(r"\s+", " ", field_match.group(2)).strip()
            if not value:
                review_lines.append(line)
                continue
            parsed_fields.append({"label": label, "value": value})
        return parsed_fields, review_lines

    def _match_ocr_target(
        self,
        target: OCRAssistedExtractionField,
        parsed_fields: list[dict[str, str]],
        used_indexes: set[int],
    ) -> int | None:
        normalized_targets = {
            self._normalize_extraction_key(candidate)
            for candidate in [target.field, *target.aliases]
            if candidate.strip()
        }
        for index, parsed_field in enumerate(parsed_fields):
            if index in used_indexes:
                continue
            normalized_label = self._normalize_extraction_key(parsed_field["label"])
            if any(
                candidate == normalized_label
                or candidate in normalized_label
                or normalized_label in candidate
                for candidate in normalized_targets
            ):
                return index
        return None

    def _contains_decision_marker(self, text: str) -> bool:
        normalized = text.casefold()
        return any(marker in normalized for marker in DECISION_MARKERS)

    def _derive_action_rows(self, transcript_entries: list[dict[str, str]]) -> list[list[str]]:
        derived_rows: list[list[str]] = []
        for entry in transcript_entries:
            normalized = entry["text"].casefold()
            if " will " not in f" {normalized} " and "follow up" not in normalized:
                continue
            derived_rows.append([entry["speaker"] or "Unassigned", entry["text"], ""])
        return derived_rows

    def _format_transcript_entry(self, speaker: str, text: str) -> str:
        if speaker:
            return f"{speaker}: {text}"
        return text

    def _normalize_text_segments(self, text: str, *, label: str) -> list[str]:
        segments = [
            re.sub(r"\s+", " ", segment).strip()
            for segment in re.split(r"\n\s*\n", text)
            if segment.strip()
        ]
        if not segments:
            raise DocumentValidationError(
                "Document comparison requires non-empty text inputs for both documents.",
                details={"field": label},
            )
        return list(dict.fromkeys(segments))

    def _truncate_segment(self, text: str, *, limit: int = 120) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _normalize_transcript_sentence(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        normalized = re.sub(r"\s+([,.!?;:])", r"\1", normalized)
        if not normalized:
            return normalized
        normalized = normalized[0].upper() + normalized[1:]
        if normalized[-1] not in ".!?":
            normalized += "."
        return normalized

    def _normalize_speaker_label(self, speaker: str) -> str:
        normalized = re.sub(r"\s+", " ", speaker).strip()
        if not normalized:
            return "Unknown"
        return normalized.title()

    def _resolve_branded_image_path(
        self,
        value: str | None,
        *,
        allowed_file_roots: Sequence[Path | str] | None,
        base_dir: Path | str | None,
        label: str,
    ) -> str | None:
        if value is None:
            return None
        candidate = value.strip()
        if not candidate:
            return None
        purpose = f"Branded document {label}"
        if allowed_file_roots is not None:
            return str(
                validate_scoped_image_file(
                    candidate,
                    allowed_roots=allowed_file_roots,
                    purpose=purpose,
                    base_dir=base_dir,
                ),
            )

        image_path = Path(candidate).expanduser()
        if not image_path.is_absolute():
            if base_dir is not None:
                image_path = Path(base_dir).expanduser().resolve(strict=False) / image_path
            else:
                image_path = Path.cwd() / image_path
        resolved = image_path.resolve(strict=False)
        if not resolved.exists() or not resolved.is_file():
            raise DocumentValidationError(
                "Branded document image file does not exist.",
                details={"path": str(resolved), "label": label},
            )
        return str(
            validate_scoped_image_file(
                resolved,
                allowed_roots=(resolved.parent,),
                purpose=purpose,
            ),
        )

    def _normalize_extraction_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    def _comparison_block(self, *, items: list[str], empty_message: str) -> list[ListBlock | ParagraphBlock]:
        if items:
            return [ListBlock(items=items)]
        return [ParagraphBlock(text=empty_message)]

    def _publish(self, event_type: EventType, payload: dict[str, object]) -> None:
        if self.event_bus is None:
            return
        self.event_bus.publish_threadsafe(
            StreamEvent(type=event_type, scope=EventScope.REQUEST, payload=payload),
        )
