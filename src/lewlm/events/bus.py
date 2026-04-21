"""In-process async event bus."""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from lewlm.events.schema import StreamEvent


@dataclass(slots=True)
class EventSubscription:
    """Handle used to consume events from the event bus."""

    subscription_id: str
    queue: asyncio.Queue[StreamEvent]
    _close: Callable[[str], None]

    async def get(self) -> StreamEvent:
        return await self.queue.get()

    def close(self) -> None:
        self._close(self.subscription_id)


class EventBus:
    """Simple pub/sub dispatcher backed by asyncio queues."""

    def __init__(self) -> None:
        self._subscribers: dict[str, asyncio.Queue[StreamEvent]] = {}
        self._loop: AbstractEventLoop | None = None

    def attach_loop(self, loop: AbstractEventLoop) -> None:
        self._loop = None if loop.is_closed() else loop

    def subscribe(self) -> EventSubscription:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_running_loop()
        subscription_id = str(uuid4())
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._subscribers[subscription_id] = queue
        return EventSubscription(subscription_id=subscription_id, queue=queue, _close=self.unsubscribe)

    def unsubscribe(self, subscription_id: str) -> None:
        self._subscribers.pop(subscription_id, None)

    async def publish(self, event: StreamEvent) -> None:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_running_loop()
        for queue in tuple(self._subscribers.values()):
            await queue.put(event)

    def publish_threadsafe(self, event: StreamEvent) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(self._publish_nowait, event)
        except RuntimeError:
            if self._loop.is_closed():
                self._loop = None
                return
            raise

    def _publish_nowait(self, event: StreamEvent) -> None:
        for queue in tuple(self._subscribers.values()):
            queue.put_nowait(event)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
