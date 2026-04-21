"""Deterministic document skill request models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from lewlm.documents.ir.models import DocumentOutputFormat


class DocumentSkillName(str):
    CONTRACT_TEXT_REPLACEMENT = "contract_text_replacement"
    RECEIPT_EXTRACTION = "receipt_extraction"
    BRANDED_DOCUMENT_TEMPLATE = "branded_document_template"
    FILE_TEMPLATE = "file_template"
    DOCUMENT_COMPARISON = "document_comparison"
    OCR_ASSISTED_EXTRACTION = "ocr_assisted_extraction"
    MEETING_TRANSCRIPT_NOTES = "meeting_transcript_notes"
    LONG_DOCUMENT_MEMO = "long_document_memo"
    SPEECH_TRANSCRIPT_CLEANUP = "speech_transcript_cleanup"


class BuiltInSkillDescriptor(BaseModel):
    name: str
    version: str = "1.0.0"
    category: Literal["document_transform"] = "document_transform"
    description: str
    tool_name: str = "documents.transform"
    required_authorization: str = "document_transform"
    supported_input_hints: list[str] = Field(default_factory=list)
    supported_output_formats: list[DocumentOutputFormat] = Field(default_factory=list)
    example_path: str | None = None
    tags: list[str] = Field(default_factory=list)


class ContractTextReplacementInput(BaseModel):
    title: str
    template_text: str
    replacements: dict[str, str] = Field(default_factory=dict)


class ReceiptLineItem(BaseModel):
    description: str
    quantity: str = "1"
    unit_price: str | None = None
    total: str | None = None


class ReceiptExtractionInput(BaseModel):
    title: str = "Receipt Extraction"
    vendor: str
    receipt_number: str | None = None
    purchased_at: str | None = None
    currency: str | None = None
    items: list[ReceiptLineItem] = Field(default_factory=list)
    subtotal: str | None = None
    tax: str | None = None
    total: str | None = None


class AuthorizedToolRequest(BaseModel):
    authorized_actions: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None


class ContractTextReplacementRequest(AuthorizedToolRequest):
    skill: Literal["contract_text_replacement"] = "contract_text_replacement"
    output_format: DocumentOutputFormat
    file_name: str | None = None
    input: ContractTextReplacementInput


class ReceiptExtractionRequest(AuthorizedToolRequest):
    skill: Literal["receipt_extraction"] = "receipt_extraction"
    output_format: DocumentOutputFormat
    file_name: str | None = None
    input: ReceiptExtractionInput


class OCRAssistedExtractionField(BaseModel):
    field: str
    aliases: list[str] = Field(default_factory=list)
    required: bool = False


class OCRAssistedExtractionInput(BaseModel):
    title: str = "OCR Extraction"
    source_title: str = "Scanned Document"
    document_type: str | None = None
    ocr_text: str
    expected_fields: list[OCRAssistedExtractionField] = Field(default_factory=list)


class OCRAssistedExtractionRequest(AuthorizedToolRequest):
    skill: Literal["ocr_assisted_extraction"] = "ocr_assisted_extraction"
    output_format: DocumentOutputFormat
    file_name: str | None = None
    input: OCRAssistedExtractionInput


class BrandedDocumentSettings(BaseModel):
    organization_name: str
    subtitle: str | None = None
    audience: str | None = None
    issued_on: str | None = None
    contact_line: str | None = None
    header_text: str | None = None
    footer_text: str | None = None
    logo_path: str | None = None
    hero_image_path: str | None = None


class BrandedDocumentSectionInput(BaseModel):
    heading: str
    paragraphs: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    callout_title: str | None = None
    callout_body: str | None = None


class BrandedDocumentTemplateInput(BaseModel):
    title: str = "Branded Brief"
    settings: BrandedDocumentSettings
    summary: str
    key_points: list[str] = Field(default_factory=list)
    sections: list[BrandedDocumentSectionInput] = Field(default_factory=list)


class BrandedDocumentTemplateRequest(AuthorizedToolRequest):
    skill: Literal["branded_document_template"] = "branded_document_template"
    output_format: DocumentOutputFormat
    file_name: str | None = None
    input: BrandedDocumentTemplateInput


class FileTemplateTransformInput(BaseModel):
    replacements: dict[str, str] = Field(default_factory=dict)
    title: str | None = None


class FileTemplateTransformRequest(AuthorizedToolRequest):
    skill: Literal["file_template"] = "file_template"
    output_format: DocumentOutputFormat
    file_name: str | None = None
    template_path: str
    input: FileTemplateTransformInput = Field(default_factory=FileTemplateTransformInput)


class DocumentComparisonInput(BaseModel):
    title: str = "Document Comparison"
    left_title: str = "Document A"
    left_text: str
    right_title: str = "Document B"
    right_text: str


class DocumentComparisonRequest(AuthorizedToolRequest):
    skill: Literal["document_comparison"] = "document_comparison"
    output_format: DocumentOutputFormat
    file_name: str | None = None
    input: DocumentComparisonInput


class MeetingTranscriptNotesInput(BaseModel):
    title: str = "Meeting Notes"
    transcript_text: str
    participants: list[str] = Field(default_factory=list)
    meeting_date: str | None = None


class MeetingTranscriptNotesRequest(AuthorizedToolRequest):
    skill: Literal["meeting_transcript_notes"] = "meeting_transcript_notes"
    output_format: DocumentOutputFormat
    file_name: str | None = None
    input: MeetingTranscriptNotesInput


class LongDocumentMemoInput(BaseModel):
    title: str = "Structured Memo"
    source_title: str = "Source Document"
    source_text: str


class LongDocumentMemoRequest(AuthorizedToolRequest):
    skill: Literal["long_document_memo"] = "long_document_memo"
    output_format: DocumentOutputFormat
    file_name: str | None = None
    input: LongDocumentMemoInput


class SpeechTranscriptCleanupInput(BaseModel):
    title: str = "Transcript Cleanup"
    transcript_text: str
    language: str | None = None


class SpeechTranscriptCleanupRequest(AuthorizedToolRequest):
    skill: Literal["speech_transcript_cleanup"] = "speech_transcript_cleanup"
    output_format: DocumentOutputFormat
    file_name: str | None = None
    input: SpeechTranscriptCleanupInput


DocumentTransformRequest = Annotated[
    ContractTextReplacementRequest
    | ReceiptExtractionRequest
    | OCRAssistedExtractionRequest
    | BrandedDocumentTemplateRequest
    | FileTemplateTransformRequest
    | DocumentComparisonRequest
    | MeetingTranscriptNotesRequest
    | LongDocumentMemoRequest
    | SpeechTranscriptCleanupRequest,
    Field(discriminator="skill"),
]

DOCUMENT_TRANSFORM_REQUEST_ADAPTER = TypeAdapter(DocumentTransformRequest)


def parse_document_transform_request(payload: str | bytes) -> DocumentTransformRequest:
    """Parse a JSON payload into a typed transform request."""

    return DOCUMENT_TRANSFORM_REQUEST_ADAPTER.validate_json(payload)
