"""Helpers for app-facing citation context and response packaging."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

from lewlm.documents.ingest.models import DocumentChunk, IngestedDocumentSource

_CITATION_PREFIX = "[[cite:"
_CITATION_SUFFIX = "]]"
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"[ \t]+([,.;:!?])")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_MULTILINE_GAP_RE = re.compile(r"\n{3,}")


class CitationContextPackage(BaseModel):
    """Reusable source and chunk packages supplied by a host application."""

    sources: list[IngestedDocumentSource] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_identifiers(self) -> "CitationContextPackage":
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("citation_context.sources must not contain duplicate source_id values.")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("citation_context.chunks must not contain duplicate chunk_id values.")
        return self

    def has_entries(self) -> bool:
        return bool(self.sources or self.chunks)


class GeneratedCitationReference(BaseModel):
    """Stable machine-readable citation reference resolved from generated output."""

    reference_id: str = Field(description="Stable citation token emitted by the model and resolved by LewLM.")
    source_id: str = Field(description="Stable source identifier aligned with document ingest output.")
    chunk_id: str | None = Field(default=None, description="Stable chunk identifier when the citation points to one chunk.")
    section_id: str | None = Field(
        default=None,
        description="Stable section identifier when LewLM can resolve the citation to a known section.",
    )
    source_label: str = Field(description="Readable source label aligned with document ingest packaging.")
    section_label: str | None = Field(
        default=None,
        description="Readable section label aligned with document ingest packaging when LewLM can resolve one.",
    )


def render_citation_context_message(citation_context: CitationContextPackage | None) -> str | None:
    """Render one system message that teaches the model how to emit stable citations."""

    if citation_context is None or not citation_context.has_entries():
        return None

    lines = [
        "Citation context:",
        "LewLM received reusable source and chunk packages from the caller.",
        "When a statement relies on that grounded context, append citation markers in the exact form [[cite:<id>]].",
        "Prefer chunk_id markers when a chunk is available. Use source_id only for broader source-level support.",
        "Use only ids listed below. Do not invent ids.",
        "LewLM strips valid markers from the final text and returns them separately as structured citations.",
    ]

    if citation_context.sources:
        lines.extend(("", "Available source ids:"))
        for source in citation_context.sources:
            lines.append(
                f"- source_id={source.source_id} | source_label={source.source_label}"
                + (f" | media_type={source.media_type}" if source.media_type else "")
            )

    if citation_context.chunks:
        lines.extend(("", "Grounding chunk records:"))
        for chunk in citation_context.chunks:
            lines.extend(
                (
                    f"- chunk_id={chunk.chunk_id} | source_id={chunk.source_id} | section_id={chunk.section_id}",
                    f"  source_label={chunk.source_label}",
                    f"  section_label={chunk.section_label}",
                    f"  text={chunk.text.strip()}",
                ),
            )
    else:
        lines.extend(
            (
                "",
                "No chunk text was supplied. Use source_id markers only when the conversation already includes grounded content from those sources.",
            ),
        )

    return "\n".join(lines)


def resolve_generated_citations(
    text: str,
    citation_context: CitationContextPackage | None,
) -> tuple[str, list[GeneratedCitationReference]]:
    """Strip valid LewLM citation markers and resolve them into stable references."""

    processor = CitationStreamProcessor(citation_context)
    visible = processor.consume(text)
    tail, citations = processor.finalize()
    cleaned = cleanup_citation_text(f"{visible}{tail}")
    return cleaned, citations


def cleanup_citation_text(text: str) -> str:
    """Normalize visible text after citation markers are stripped."""

    cleaned = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    cleaned = _MULTILINE_GAP_RE.sub("\n\n", cleaned)
    return cleaned.strip()


class CitationStreamProcessor:
    """Incrementally strip citation markers while preserving visible text order."""

    def __init__(self, citation_context: CitationContextPackage | None) -> None:
        self._resolver = _CitationResolver(citation_context)
        self._buffer = ""
        self._citations: list[GeneratedCitationReference] = []
        self._seen_reference_ids: set[str] = set()

    def consume(self, text: str) -> str:
        if not text or self._resolver.is_empty:
            return text

        self._buffer += text
        emitted: list[str] = []
        while True:
            marker_start = self._buffer.find(_CITATION_PREFIX)
            if marker_start < 0:
                partial_prefix_length = _trailing_prefix_length(self._buffer, _CITATION_PREFIX)
                safe_length = len(self._buffer) - partial_prefix_length
                if safe_length > 0:
                    emitted.append(self._buffer[:safe_length])
                    self._buffer = self._buffer[safe_length:]
                break
            if marker_start > 0:
                emitted.append(self._buffer[:marker_start])
                self._buffer = self._buffer[marker_start:]
            marker_end = self._buffer.find(_CITATION_SUFFIX)
            if marker_end < 0:
                break
            marker = self._buffer[: marker_end + len(_CITATION_SUFFIX)]
            replacement, references = self._resolver.transform_marker(marker)
            if replacement:
                emitted.append(replacement)
            self._buffer = self._buffer[marker_end + len(_CITATION_SUFFIX) :]
            self._record_references(references)
        return "".join(emitted)

    def finalize(self) -> tuple[str, list[GeneratedCitationReference]]:
        tail = self._buffer
        self._buffer = ""
        return tail, list(self._citations)

    def _record_references(self, references: list[GeneratedCitationReference]) -> None:
        for reference in references:
            if reference.reference_id in self._seen_reference_ids:
                continue
            self._seen_reference_ids.add(reference.reference_id)
            self._citations.append(reference)


class _CitationResolver:
    def __init__(self, citation_context: CitationContextPackage | None) -> None:
        self._by_chunk_id: dict[str, GeneratedCitationReference] = {}
        self._by_section_id: dict[str, GeneratedCitationReference] = {}
        self._by_source_id: dict[str, GeneratedCitationReference] = {}
        if citation_context is None or not citation_context.has_entries():
            return

        for chunk in citation_context.chunks:
            self._by_chunk_id[chunk.chunk_id] = GeneratedCitationReference(
                reference_id=chunk.chunk_id,
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
                section_id=chunk.section_id,
                source_label=chunk.source_label,
                section_label=chunk.section_label,
            )
            self._by_section_id.setdefault(
                chunk.section_id,
                GeneratedCitationReference(
                    reference_id=chunk.section_id,
                    source_id=chunk.source_id,
                    chunk_id=None,
                    section_id=chunk.section_id,
                    source_label=chunk.source_label,
                    section_label=chunk.section_label,
                ),
            )
            self._by_source_id.setdefault(
                chunk.source_id,
                GeneratedCitationReference(
                    reference_id=chunk.source_id,
                    source_id=chunk.source_id,
                    chunk_id=None,
                    section_id=None,
                    source_label=chunk.source_label,
                    section_label=None,
                ),
            )

        for source in citation_context.sources:
            self._by_source_id[source.source_id] = GeneratedCitationReference(
                reference_id=source.source_id,
                source_id=source.source_id,
                chunk_id=None,
                section_id=None,
                source_label=source.source_label,
                section_label=None,
            )

    @property
    def is_empty(self) -> bool:
        return not self._by_chunk_id and not self._by_section_id and not self._by_source_id

    def transform_marker(self, marker: str) -> tuple[str, list[GeneratedCitationReference]]:
        payload = marker.removeprefix(_CITATION_PREFIX).removesuffix(_CITATION_SUFFIX)
        tokens = _split_marker_tokens(payload)
        if not tokens:
            return marker, []

        resolved: list[GeneratedCitationReference] = []
        unknown: list[str] = []
        for token in tokens:
            reference = self.resolve_token(token)
            if reference is None:
                unknown.append(token)
                continue
            resolved.append(reference)

        if not resolved:
            return marker, []
        if unknown:
            return f"[[cite:{', '.join(unknown)}]]", resolved
        return "", resolved

    def resolve_token(self, token: str) -> GeneratedCitationReference | None:
        return (
            self._by_chunk_id.get(token)
            or self._by_section_id.get(token)
            or self._by_source_id.get(token)
        )


def _split_marker_tokens(payload: str) -> list[str]:
    return [token for token in re.split(r"[\s,]+", payload.strip()) if token]


def _trailing_prefix_length(text: str, prefix: str) -> int:
    max_length = min(len(text), len(prefix) - 1)
    for length in range(max_length, 0, -1):
        if prefix.startswith(text[-length:]):
            return length
    return 0
