from __future__ import annotations

from lewlm.core.contracts import ReasoningVisibility
from lewlm.core.reasoning import ReasoningStreamProcessor, apply_reasoning_visibility


def test_apply_reasoning_visibility_strips_model_emitted_reasoning_when_hidden() -> None:
    output_text, reasoning = apply_reasoning_visibility(
        "<think>Inspect the prompt before replying.</think>Echo: hidden output",
        ReasoningVisibility.HIDDEN,
    )

    assert output_text == "Echo: hidden output"
    assert reasoning.visibility == ReasoningVisibility.HIDDEN
    assert reasoning.available is True
    assert reasoning.content is None
    assert reasoning.summary is None


def test_apply_reasoning_visibility_returns_summary_when_requested() -> None:
    output_text, reasoning = apply_reasoning_visibility(
        "<reasoning>Inspect the prompt before replying so the answer stays deterministic.</reasoning>Echo: summarized output",
        ReasoningVisibility.SUMMARIZED,
    )

    assert output_text == "Echo: summarized output"
    assert reasoning.visibility == ReasoningVisibility.SUMMARIZED
    assert reasoning.available is True
    assert reasoning.summary == "Inspect the prompt before replying so the answer stays deterministic."
    assert reasoning.content is None


def test_reasoning_stream_processor_handles_split_tags_and_raw_reasoning() -> None:
    processor = ReasoningStreamProcessor(ReasoningVisibility.RAW_MODEL_EMITTED)

    emitted = []
    emitted.extend(processor.consume("<thi"))
    emitted.extend(processor.consume("nk>Inspect the prompt "))
    emitted.extend(processor.consume("before replying.</thi"))
    emitted.extend(processor.consume("nk>Echo"))
    emitted.extend(processor.consume(": stream output"))
    reasoning = processor.finalize()

    assert [item.reasoning for item in emitted if item.reasoning] == ["Inspect the prompt ", "before replying."]
    assert [item.content for item in emitted if item.content] == ["Echo", ": stream output"]
    assert processor.output_text == "Echo: stream output"
    assert reasoning.visibility == ReasoningVisibility.RAW_MODEL_EMITTED
    assert reasoning.available is True
    assert reasoning.content == "Inspect the prompt before replying."
