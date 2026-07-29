"""A client that stops reading must deterministically close the source stream."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lewlm.api.routes.chat import (
    _chat_completion_stream,
    _close_abandoned_stream,
    _response_stream,
)
from lewlm.core.chat import ChatStreamDelta


class _RecordingStream:
    """An async iterator that records whether it was closed before finishing."""

    def __init__(self, items: list[str], *, on_close) -> None:
        self._items = list(items)
        self._on_close = on_close
        self.completed = False
        self.closed = False

    def __aiter__(self) -> "_RecordingStream":
        return self

    async def __anext__(self) -> ChatStreamDelta:
        if not self._items:
            self.completed = True
            raise StopAsyncIteration
        return ChatStreamDelta(content=self._items.pop(0))

    async def aclose(self) -> None:
        self.closed = True
        self._on_close(self.completed)


def _session(stream) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-1",
        created_at=1,
        model_id="model-1",
        stream_items=stream,
        stream=None,
        reasoning=None,
        reasoning_visibility=None,
        citations=[],
        metadata=None,
        structured_output=None,
        tool_calls=None,
        serving_profile=None,
        usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        usage_measured=True,
    )


async def test_abandoning_a_chat_stream_closes_the_source() -> None:
    cancellations: list[bool] = []
    stream = _RecordingStream(["a", "b", "c", "d"], on_close=cancellations.append)

    generator = _chat_completion_stream(_session(stream))
    await generator.__anext__()  # read one frame, then walk away
    await generator.aclose()

    assert stream.closed is True
    # `completed=False` is what tells the serving core the consumer is gone.
    assert cancellations == [False]


async def test_abandoning_a_responses_stream_closes_the_source() -> None:
    cancellations: list[bool] = []
    stream = _RecordingStream(["a", "b", "c"], on_close=cancellations.append)

    generator = _response_stream(_session(stream))
    await generator.__anext__()
    await generator.aclose()

    assert stream.closed is True
    assert cancellations == [False]


async def test_a_fully_read_stream_is_not_closed_as_abandoned() -> None:
    cancellations: list[bool] = []
    stream = _RecordingStream(["a", "b"], on_close=cancellations.append)

    frames = [frame async for frame in _chat_completion_stream(_session(stream))]

    assert frames[-1] == "data: [DONE]\n\n"
    # The stream ended on its own, so no abandonment close is issued.
    assert cancellations == []


async def test_the_final_chunk_of_a_completed_stream_carries_usage() -> None:
    stream = _RecordingStream(["a"], on_close=lambda completed: None)
    frames = [frame async for frame in _chat_completion_stream(_session(stream))]

    final = json.loads(frames[-2][len("data: ") :])
    assert final["usage"]["total_tokens"] == 5
    assert final["usage"]["measured"] is True


async def test_close_helper_is_a_no_op_for_a_completed_stream() -> None:
    closed: list[bool] = []

    class _Stream:
        async def aclose(self) -> None:
            closed.append(True)

    await _close_abandoned_stream(_Stream(), completed=True, request_id="req-1")
    assert closed == []

    await _close_abandoned_stream(_Stream(), completed=False, request_id="req-1")
    assert closed == [True]


async def test_close_helper_tolerates_a_missing_or_failing_closer() -> None:
    await _close_abandoned_stream(None, completed=False, request_id="req-1")
    await _close_abandoned_stream(object(), completed=False, request_id="req-1")

    class _Failing:
        async def aclose(self) -> None:
            raise RuntimeError("backend already gone")

    # Cleanup failure must not mask the original reason the stream ended.
    await _close_abandoned_stream(_Failing(), completed=False, request_id="req-1")


async def test_close_helper_propagates_cancellation() -> None:
    class _Cancelling:
        async def aclose(self) -> None:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _close_abandoned_stream(_Cancelling(), completed=False, request_id="req-1")
