"""Helpers for normalizing chat and responses inputs into runtime messages."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from lewlm.api.schemas.chat import (
    ChatMessage,
    InputAudioPart,
    InputFilePart,
    InputImagePart,
    InputTextPart,
    MessageContentPart,
    ResponseInputMessage,
)
from lewlm.core.bootstrap import LewLMServices
from lewlm.core.contracts import GenerateAttachment, GenerateMessage
from lewlm.core.errors import ConfigurationError
from lewlm.documents.ingest.models import DocumentSourceType
from lewlm.documents.ir.models import CalloutBlock, DocumentIR, ImageBlock, ListBlock, ParagraphBlock, TableBlock
from lewlm.security.files import read_scoped_text_file, resolve_scoped_path, validate_audio_bytes


TEXT_SUFFIXES = {".md", ".text", ".txt"}
AttachmentPathPart = InputImagePart | InputFilePart | InputAudioPart


async def normalize_chat_messages(
    messages: list[ChatMessage],
    services: LewLMServices,
    *,
    uploaded_files: Mapping[str, Path] | None = None,
    file_access_roots: tuple[Path, ...] | None = None,
) -> list[GenerateMessage]:
    return [
        await _normalize_message(
            role=message.role,
            content=message.content,
            services=services,
            uploaded_files=uploaded_files,
            file_access_roots=file_access_roots,
        )
        for message in messages
    ]


async def normalize_response_input(
    input_value: str | list[ResponseInputMessage],
    services: LewLMServices,
    *,
    uploaded_files: Mapping[str, Path] | None = None,
    file_access_roots: tuple[Path, ...] | None = None,
) -> list[GenerateMessage]:
    if isinstance(input_value, str):
        return [GenerateMessage(role="user", content=input_value)]
    return [
        await _normalize_message(
            role=item.role,
            content=item.content,
            services=services,
            uploaded_files=uploaded_files,
            file_access_roots=file_access_roots,
        )
        for item in input_value
    ]


async def _normalize_message(
    *,
    role: str,
    content: str | list[MessageContentPart],
    services: LewLMServices,
    uploaded_files: Mapping[str, Path] | None = None,
    file_access_roots: tuple[Path, ...] | None = None,
) -> GenerateMessage:
    if isinstance(content, str):
        return GenerateMessage(role=role, content=content)

    segments: list[str] = []
    attachments: list[GenerateAttachment] = []
    for part in content:
        if isinstance(part, InputTextPart):
            if part.text.strip():
                segments.append(part.text.strip())
            continue
        if isinstance(part, InputAudioPart):
            attachment = await _normalize_audio_attachment(
                part,
                services,
                uploaded_files=uploaded_files,
                file_access_roots=file_access_roots,
            )
        else:
            attachment = _normalize_path_attachment(
                part,
                services,
                uploaded_files=uploaded_files,
                file_access_roots=file_access_roots,
            )
        attachments.append(attachment)
        segments.append(_attachment_prompt_block(attachment))

    return GenerateMessage(
        role=role,
        content="\n\n".join(segment for segment in segments if segment.strip()),
        attachments=attachments,
    )


def _normalize_path_attachment(
    part: InputImagePart | InputFilePart,
    services: LewLMServices,
    *,
    uploaded_files: Mapping[str, Path] | None = None,
    file_access_roots: tuple[Path, ...] | None = None,
) -> GenerateAttachment:
    resolved_path, allowed_roots = _resolve_attachment_path(
        part,
        services,
        uploaded_files=uploaded_files,
        file_access_roots=file_access_roots,
        purpose="Chat attachment",
        expect="any",
    )
    raw_bytes = resolved_path.read_bytes() if resolved_path.is_file() else None
    suffix = resolved_path.suffix.casefold()
    cache_key = None
    if raw_bytes is not None:
        cache_key = services.multimodal_feature_cache.cache_key_for_path_attachment(raw_bytes=raw_bytes, suffix=suffix)
        cached_attachment = services.multimodal_feature_cache.get_attachment(
            cache_key=cache_key,
            name=resolved_path.name,
            source_path=str(resolved_path),
        )
        if cached_attachment is not None:
            return cached_attachment
    if resolved_path.is_file() and suffix in TEXT_SUFFIXES:
        _, text = read_scoped_text_file(
            resolved_path,
            allowed_roots=allowed_roots,
            purpose="Text attachment",
            media_type="text/plain",
        )
        attachment = GenerateAttachment(
            attachment_type="text",
            name=resolved_path.name,
            source_path=str(resolved_path),
            media_type="text/plain",
            extracted_text=_truncate_text(text),
            metadata={"source_kind": "text_file"},
        )
        if cache_key is not None and raw_bytes is not None:
            services.multimodal_feature_cache.put_attachment(
                cache_key=cache_key,
                attachment=attachment,
                cache_metadata={"source_kind": "text_file", "source_suffix": suffix, "input_bytes": len(raw_bytes)},
            )
        return attachment
    if resolved_path.is_file() and suffix == ".json":
        _, text = read_scoped_text_file(
            resolved_path,
            allowed_roots=allowed_roots,
            purpose="JSON attachment",
            media_type="application/json",
        )
        attachment = GenerateAttachment(
            attachment_type="text",
            name=resolved_path.name,
            source_path=str(resolved_path),
            media_type="application/json",
            extracted_text=_truncate_text(text),
            metadata={"source_kind": "json_file"},
        )
        if cache_key is not None and raw_bytes is not None:
            services.multimodal_feature_cache.put_attachment(
                cache_key=cache_key,
                attachment=attachment,
                cache_metadata={"source_kind": "json_file", "source_suffix": suffix, "input_bytes": len(raw_bytes)},
            )
        return attachment

    ingest_result = services.document_ingest_service.ingest(
        [resolved_path],
        allowed_file_roots=allowed_roots,
    )
    source = ingest_result.sources[0]
    attachment_type = (
        "image"
        if source.source_type in {DocumentSourceType.IMAGE, DocumentSourceType.IMAGE_BUNDLE}
        else "document"
    )
    attachment = GenerateAttachment(
        attachment_type=attachment_type,
        name=resolved_path.name,
        source_path=str(resolved_path),
        media_type=_source_media_type(source.source_type),
        extracted_text=_document_to_prompt_text(ingest_result.document),
        metadata={
            "source_type": source.source_type.value,
            "source_metadata": source.metadata,
        },
    )
    if cache_key is not None and raw_bytes is not None:
        services.multimodal_feature_cache.put_attachment(
            cache_key=cache_key,
            attachment=attachment,
            cache_metadata={
                "source_kind": attachment_type,
                "source_suffix": suffix,
                "input_bytes": len(raw_bytes),
                "source_type": source.source_type.value,
            },
        )
    return attachment


async def _normalize_audio_attachment(
    part: InputAudioPart,
    services: LewLMServices,
    *,
    uploaded_files: Mapping[str, Path] | None = None,
    file_access_roots: tuple[Path, ...] | None = None,
) -> GenerateAttachment:
    resolved_path, _ = _resolve_attachment_path(
        part,
        services,
        uploaded_files=uploaded_files,
        file_access_roots=file_access_roots,
        purpose="Audio attachment",
        expect="file",
    )
    raw = resolved_path.read_bytes()
    media_type = validate_audio_bytes(raw, purpose="Audio attachment", file_name=resolved_path.name)
    cache_key = services.multimodal_feature_cache.cache_key_for_audio_attachment(
        raw_bytes=raw,
        file_name=resolved_path.name,
        suffix=resolved_path.suffix.casefold(),
        language=part.language,
        prompt=part.prompt,
    )
    cached_attachment = services.multimodal_feature_cache.get_attachment(
        cache_key=cache_key,
        name=resolved_path.name,
        source_path=str(resolved_path),
    )
    if cached_attachment is not None:
        return cached_attachment
    execution = await services.multimodal_orchestrator.transcribe_audio(
        model_id=None,
        audio_bytes=raw,
        file_name=resolved_path.name,
        language=part.language,
        prompt=part.prompt,
    )
    attachment = GenerateAttachment(
        attachment_type="audio",
        name=resolved_path.name,
        source_path=str(resolved_path),
        media_type=media_type,
        extracted_text=_truncate_text(execution.response.text),
        metadata={
            "language": execution.response.language,
            "duration_seconds": execution.response.duration_seconds,
            "transcription_model_id": execution.response.model_id,
        },
    )
    services.multimodal_feature_cache.put_attachment(
        cache_key=cache_key,
        attachment=attachment,
        cache_metadata={
            "source_kind": "audio",
            "source_suffix": resolved_path.suffix.casefold(),
            "input_bytes": len(raw),
            "language": part.language,
            "prompt": part.prompt,
        },
    )
    return attachment


def _resolve_attachment_path(
    part: AttachmentPathPart,
    services: LewLMServices,
    *,
    uploaded_files: Mapping[str, Path] | None,
    file_access_roots: tuple[Path, ...] | None,
    purpose: str,
    expect: str,
) -> tuple[Path, tuple[Path, ...]]:
    if part.upload_name:
        if uploaded_files is None or part.upload_name not in uploaded_files:
            raise ConfigurationError(
                "The request referenced a multipart upload that was not provided.",
                details={"upload_name": part.upload_name, "purpose": purpose},
            )
        resolved_path = uploaded_files[part.upload_name].resolve(strict=False)
        return resolved_path, (resolved_path.parent,)
    if part.path is None:
        raise ConfigurationError(
            "Attachment content parts require either `path` or `upload_name`.",
            details={"purpose": purpose},
        )
    resolved_path = resolve_scoped_path(
        Path(part.path),
        allowed_roots=_allowed_file_roots(services, file_access_roots),
        purpose=purpose,
        expect=expect,
    )
    return resolved_path, _allowed_file_roots(services, file_access_roots)


def _allowed_file_roots(
    services: LewLMServices,
    file_access_roots: tuple[Path, ...] | None,
) -> tuple[Path, ...]:
    if file_access_roots is not None:
        return file_access_roots
    return tuple(services.settings.file_access_roots)


def _attachment_prompt_block(attachment: GenerateAttachment) -> str:
    header = f"[Attached {attachment.attachment_type}: {attachment.name}]"
    if not attachment.extracted_text:
        return header
    return f"{header}\n{attachment.extracted_text}"


def _document_to_prompt_text(document: DocumentIR) -> str:
    lines = [f"Document title: {document.title}"]
    for section in document.sections:
        if section.heading:
            lines.append(f"Section: {section.heading}")
        for block in section.blocks:
            if isinstance(block, ParagraphBlock):
                lines.append(block.text)
            elif isinstance(block, ListBlock):
                lines.extend(f"- {item}" for item in block.items[:8])
                if len(block.items) > 8:
                    lines.append(f"- ... {len(block.items) - 8} more item(s)")
            elif isinstance(block, CalloutBlock):
                title = f"{block.kind.upper()}: {block.title}" if block.title else block.kind.upper()
                lines.append(f"{title} - {block.body}")
            elif isinstance(block, TableBlock):
                caption = block.caption or "Table"
                lines.append(caption)
                if block.headers:
                    lines.append("Headers: " + ", ".join(block.headers))
                for row in block.rows[:5]:
                    lines.append("Row: " + " | ".join(row))
                if len(block.rows) > 5:
                    lines.append(f"... {len(block.rows) - 5} more row(s)")
            elif isinstance(block, ImageBlock):
                caption = f" ({block.caption})" if block.caption else ""
                lines.append(f"Image: {block.alt_text}{caption}")
    if document.citations:
        lines.append("Citations:")
        lines.extend(f"- {citation.label}: {citation.text}" for citation in document.citations[:5])
    return _truncate_text("\n".join(line for line in lines if line.strip()))


def _truncate_text(text: str, *, max_characters: int = 4000) -> str:
    stripped = text.strip()
    if len(stripped) <= max_characters:
        return stripped
    return stripped[: max_characters - 16].rstrip() + "\n...[truncated]"


def _source_media_type(source_type: DocumentSourceType) -> str:
    if source_type == DocumentSourceType.CSV:
        return "text/csv"
    if source_type == DocumentSourceType.XLSX:
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if source_type == DocumentSourceType.DOCX:
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if source_type == DocumentSourceType.PDF:
        return "application/pdf"
    return "image/*"
