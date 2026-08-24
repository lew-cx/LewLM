"""In-process async event bus."""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4

from lewlm.events.filters import EventFilter
from lewlm.events.schema import StreamEvent


@dataclass(slots=True)
class EventSubscription:
    """Handle used to consume events from the event bus."""

    subscription_id: str
    queue: asyncio.Queue[StreamEvent]
    _close: Callable[[str], None]
    event_filter: EventFilter = field(default_factory=EventFilter)

    async def get(self) -> StreamEvent:
        return await self.queue.get()

    def close(self) -> None:
        self._close(self.subscription_id)


class EventBus:
    """Simple pub/sub dispatcher backed by asyncio queues.

    A subscriber may narrow what it receives. The filter is applied here rather
    than by the consumer, so an event a subscriber did not ask for never reaches
    its queue and never has to be serialized for it.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, _Subscriber] = {}
        self._loop: AbstractEventLoop | None = None

    def attach_loop(self, loop: AbstractEventLoop) -> None:
        self._loop = None if loop.is_closed() else loop

    def subscribe(self, event_filter: EventFilter | None = None) -> EventSubscription:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_running_loop()
        subscription_id = str(uuid4())
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        resolved_filter = event_filter or EventFilter()
        self._subscribers[subscription_id] = _Subscriber(queue=queue, event_filter=resolved_filter)
        return EventSubscription(
            subscription_id=subscription_id,
            queue=queue,
            _close=self.unsubscribe,
            event_filter=resolved_filter,
        )

    def unsubscribe(self, subscription_id: str) -> None:
        self._subscribers.pop(subscription_id, None)

    async def publish(self, event: StreamEvent) -> None:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_running_loop()
        for subscriber in self._matching_subscribers(event):
            await subscriber.queue.put(event)

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
        for subscriber in self._matching_subscribers(event):
            subscriber.queue.put_nowait(event)

    def _matching_subscribers(self, event: StreamEvent) -> tuple["_Subscriber", ...]:
        return tuple(
            subscriber
            for subscriber in tuple(self._subscribers.values())
            if subscriber.event_filter.is_empty or subscriber.event_filter.matches(event)
        )

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


@dataclass(slots=True)
class _Subscriber:
    queue: asyncio.Queue[StreamEvent]
    event_filter: EventFilter
