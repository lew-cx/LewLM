"""Helpers for policy-aware handling of model-emitted reasoning."""

from __future__ import annotations

from dataclasses import dataclass

from lewlm.core.contracts import ReasoningOutput, ReasoningVisibility


_REASONING_TAGS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<reasoning>", "</reasoning>"),
)


@dataclass(slots=True)
class StreamReasoningDelta:
    """Visible content or exposed reasoning emitted from a streamed chunk."""

    content: str | None = None
    reasoning: str | None = None


class ReasoningStreamProcessor:
    """Strip or expose explicit model-emitted reasoning according to policy."""

    def __init__(self, visibility: ReasoningVisibility) -> None:
        self.visibility = visibility
        self._buffer = ""
        self._active_close_tag: str | None = None
        self._output_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._reasoning_detected = False

    @property
    def output_text(self) -> str:
        return "".join(self._output_parts)

    def consume(self, delta: str) -> list[StreamReasoningDelta]:
        self._buffer += delta
        emitted: list[StreamReasoningDelta] = []
        while self._buffer:
            if self._active_close_tag is None:
                tag_match = _find_open_tag(self._buffer)
                if tag_match is None:
                    safe_suffix = _safe_suffix_length(self._buffer, tuple(open_tag for open_tag, _ in _REASONING_TAGS))
                    content = self._buffer[:-safe_suffix] if safe_suffix else self._buffer
                    self._buffer = self._buffer[-safe_suffix:] if safe_suffix else ""
                    if content:
                        self._output_parts.append(content)
                        emitted.append(StreamReasoningDelta(content=content))
                    break
                tag_index, open_tag, close_tag = tag_match
                if tag_index > 0:
                    content = self._buffer[:tag_index]
                    self._output_parts.append(content)
                    emitted.append(StreamReasoningDelta(content=content))
                self._reasoning_detected = True
                self._buffer = self._buffer[tag_index + len(open_tag) :]
                self._active_close_tag = close_tag
                continue

            close_index = self._buffer.casefold().find(self._active_close_tag.casefold())
            if close_index == -1:
                safe_suffix = _safe_suffix_length(self._buffer, (self._active_close_tag,))
                reasoning_text = self._buffer[:-safe_suffix] if safe_suffix else self._buffer
                self._buffer = self._buffer[-safe_suffix:] if safe_suffix else ""
                self._record_reasoning(reasoning_text, emitted)
                break

            reasoning_text = self._buffer[:close_index]
            self._record_reasoning(reasoning_text, emitted)
            self._buffer = self._buffer[close_index + len(self._active_close_tag) :]
            self._active_close_tag = None
        return emitted

    def finalize(self) -> ReasoningOutput:
        if self._buffer:
            if self._active_close_tag is None:
                self._output_parts.append(self._buffer)
            else:
                self._record_reasoning(self._buffer, [])
            self._buffer = ""
            self._active_close_tag = None
        return build_reasoning_output(
            visibility=self.visibility,
            reasoning_text="".join(self._reasoning_parts),
            reasoning_detected=self._reasoning_detected,
        )

    def _record_reasoning(self, reasoning_text: str, emitted: list[StreamReasoningDelta]) -> None:
        if not reasoning_text:
            return
        self._reasoning_parts.append(reasoning_text)
        if self.visibility == ReasoningVisibility.RAW_MODEL_EMITTED:
            emitted.append(StreamReasoningDelta(reasoning=reasoning_text))


def apply_reasoning_visibility(
    output_text: str,
    visibility: ReasoningVisibility,
    *,
    existing_reasoning: ReasoningOutput | None = None,
) -> tuple[str, ReasoningOutput]:
    """Return visible output text plus structured reasoning metadata."""

    if existing_reasoning is not None and existing_reasoning.available:
        return output_text, _apply_visibility_to_existing_reasoning(existing_reasoning, visibility)

    processor = ReasoningStreamProcessor(visibility)
    processor.consume(output_text)
    reasoning = processor.finalize()
    return processor.output_text, reasoning


def build_reasoning_output(
    *,
    visibility: ReasoningVisibility,
    reasoning_text: str,
    reasoning_detected: bool,
) -> ReasoningOutput:
    normalized = " ".join(reasoning_text.split())
    if not reasoning_detected and not normalized:
        return ReasoningOutput(visibility=visibility, available=False)
    if visibility == ReasoningVisibility.HIDDEN:
        return ReasoningOutput(visibility=visibility, available=True)
    if visibility == ReasoningVisibility.SUMMARIZED:
        return ReasoningOutput(
            visibility=visibility,
            available=True,
            summary=summarize_reasoning_text(normalized),
        )
    return ReasoningOutput(
        visibility=visibility,
        available=True,
        content=normalized,
    )


def summarize_reasoning_text(reasoning_text: str, *, max_length: int = 160) -> str | None:
    """Produce a deterministic short summary of model-emitted reasoning."""

    normalized = " ".join(reasoning_text.split())
    if not normalized:
        return None
    if len(normalized) <= max_length:
        return normalized
    truncated = normalized[:max_length].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return f"{truncated}..."


def reasoning_exposed(reasoning: ReasoningOutput | None) -> bool:
    return bool(reasoning and reasoning.available and (reasoning.content or reasoning.summary))


def reasoning_available(reasoning: ReasoningOutput | None) -> bool:
    return bool(reasoning and reasoning.available)


def _apply_visibility_to_existing_reasoning(
    existing_reasoning: ReasoningOutput,
    visibility: ReasoningVisibility,
) -> ReasoningOutput:
    if not existing_reasoning.available:
        return ReasoningOutput(visibility=visibility, available=False)
    if visibility == ReasoningVisibility.HIDDEN:
        return ReasoningOutput(visibility=visibility, available=True)
    if visibility == ReasoningVisibility.SUMMARIZED:
        summary = existing_reasoning.summary or summarize_reasoning_text(existing_reasoning.content or "")
        return ReasoningOutput(visibility=visibility, available=True, summary=summary)
    return ReasoningOutput(
        visibility=visibility,
        available=True,
        content=existing_reasoning.content or existing_reasoning.summary,
    )


def _find_open_tag(buffer: str) -> tuple[int, str, str] | None:
    normalized = buffer.casefold()
    match: tuple[int, str, str] | None = None
    for open_tag, close_tag in _REASONING_TAGS:
        index = normalized.find(open_tag.casefold())
        if index == -1:
            continue
        if match is None or index < match[0]:
            match = (index, open_tag, close_tag)
    return match


def _safe_suffix_length(buffer: str, tags: tuple[str, ...]) -> int:
    normalized = buffer.casefold()
    max_length = 0
    for tag in tags:
        tag_prefix = tag.casefold()
        upper_bound = min(len(normalized), len(tag_prefix) - 1)
        for size in range(upper_bound, 0, -1):
            if normalized.endswith(tag_prefix[:size]):
                max_length = max(max_length, size)
                break
    return max_length
