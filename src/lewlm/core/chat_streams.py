"""Shared async stream helpers for chat orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol, TypeVar


class _ContentDelta(Protocol):
    content: str | None


_DeltaT = TypeVar("_DeltaT")
_ContentDeltaT = TypeVar("_ContentDeltaT", bound=_ContentDelta)


async def _empty_stream() -> AsyncIterator[str]:
    if False:
        yield ""


async def _empty_item_stream(*, item_factory: Callable[[], _DeltaT]) -> AsyncIterator[_DeltaT]:
    if False:
        yield item_factory()


async def _queued_item_stream(
    queue: asyncio.Queue[object],
    *,
    stream_end: object,
    item_type: type[_DeltaT],
    on_close: Callable[[bool], Awaitable[None]] | None = None,
) -> AsyncIterator[_DeltaT]:
    completed = False
    try:
        while True:
            item = await queue.get()
            if item is stream_end:
                completed = True
                return
            if isinstance(item, Exception):
                raise item
            if isinstance(item, item_type):
                yield item
    finally:
        if on_close is not None:
            await on_close(completed)


async def _stream_items_with_structured_output(
    stream_items: AsyncIterator[_ContentDeltaT],
    *,
    on_completed: Callable[[str], None],
) -> AsyncIterator[_ContentDeltaT]:
    deltas: list[str] = []
    async for item in stream_items:
        if item.content is not None:
            deltas.append(item.content)
        yield item
    on_completed("".join(deltas))


async def _content_stream(stream_items: AsyncIterator[_ContentDelta]) -> AsyncIterator[str]:
    async for item in stream_items:
        if item.content:
            yield item.content
