"""In-process async event bus."""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from collections import deque
from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4

from lewlm.core.errors import InvalidRequestError
from lewlm.events.filters import EventFilter
from lewlm.events.schema import EventScope, EventType, StreamEvent

#: Events kept for replay when the setting does not say otherwise. A streaming
#: generation publishes one `token.delta` per token, so this covers a
#: reconnect measured in seconds, not a client that was away for an hour.
DEFAULT_REPLAY_BUFFER_SIZE = 4096


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

    Every published event is stamped with a cursor, `<epoch>:<sequence>`, and
    the last `replay_buffer_size` of them are retained so a subscriber that
    reconnects can resume from the cursor it last saw. The epoch is new for
    every bus, so a cursor from a previous server lifetime is recognized as
    foreign instead of being matched against an unrelated sequence.
    """

    def __init__(self, *, replay_buffer_size: int = DEFAULT_REPLAY_BUFFER_SIZE) -> None:
        self._subscribers: dict[str, _Subscriber] = {}
        self._loop: AbstractEventLoop | None = None
        self._epoch = uuid4().hex[:8]
        self._sequence = 0
        self._replay_buffer_size = max(0, int(replay_buffer_size))
        self._buffer: deque[tuple[int, StreamEvent]] = deque(maxlen=self._replay_buffer_size)

    def attach_loop(self, loop: AbstractEventLoop) -> None:
        self._loop = None if loop.is_closed() else loop

    @property
    def replay_buffer_size(self) -> int:
        return self._replay_buffer_size

    @property
    def latest_cursor(self) -> str | None:
        """Cursor of the most recently published event, or None before the first."""

        return self._cursor(self._sequence) if self._sequence else None

    def subscribe(self, event_filter: EventFilter | None = None, *, after: str | None = None) -> EventSubscription:
        """Open a subscription, optionally resuming from the cursor `after`.

        A resumed subscription's queue starts with one `events.resumed` marker
        and then every retained event newer than `after` that the filter
        admits, in order, before anything live. The snapshot and the
        registration happen in one step on the loop, so an event published
        while the client reconnects is either replayed or delivered live,
        never both and never neither.
        """

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_running_loop()
        subscription_id = str(uuid4())
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        resolved_filter = event_filter or EventFilter()
        if after is not None:
            self._replay_into(queue, resolved_filter, after)
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
        stamped = self._stamp(event)
        for subscriber in self._matching_subscribers(stamped):
            await subscriber.queue.put(stamped)

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
        stamped = self._stamp(event)
        for subscriber in self._matching_subscribers(stamped):
            subscriber.queue.put_nowait(stamped)

    def _stamp(self, event: StreamEvent) -> StreamEvent:
        """Assign the next cursor and retain the event for replay.

        Both publish paths run on the loop thread, so the sequence is
        monotonic without a lock. The caller's object is left alone: the
        stamped copy is what subscribers and the buffer see.
        """

        self._sequence += 1
        stamped = event.model_copy(update={"cursor": self._cursor(self._sequence)})
        # A zero-length deque keeps nothing, which is what a disabled buffer means.
        self._buffer.append((self._sequence, stamped))
        return stamped

    def _cursor(self, sequence: int) -> str:
        return f"{self._epoch}:{sequence}"

    def _replay_into(self, queue: asyncio.Queue[StreamEvent], event_filter: EventFilter, after: str) -> None:
        epoch, sequence = _parse_cursor(after)
        replayed: list[StreamEvent] = []
        lost: int | None
        if epoch != self._epoch:
            # Another server lifetime: nothing since that cursor is knowable.
            lost = None
        else:
            if sequence > self._sequence:
                raise InvalidRequestError(
                    "Event cursor is ahead of this stream.",
                    details={"field": "after", "cursor": after, "latest": self.latest_cursor},
                )
            oldest_retained = self._buffer[0][0] if self._buffer else self._sequence + 1
            # Everything published after the cursor but before the oldest
            # retained event is gone. Counted across every type: which of them
            # the filter would have admitted is unknowable.
            lost = max(0, oldest_retained - sequence - 1)
            replayed = [
                event
                for retained_sequence, event in self._buffer
                if retained_sequence > sequence and (event_filter.is_empty or event_filter.matches(event))
            ]
        # The marker goes first: it is a report about what follows.
        queue.put_nowait(
            StreamEvent(
                type=EventType.EVENTS_RESUMED,
                scope=EventScope.SYSTEM,
                payload={
                    "after": after,
                    "replayed": len(replayed),
                    "lost": lost,
                    "latest": self.latest_cursor,
                    "retained": len(self._buffer),
                    "replay_buffer_size": self._replay_buffer_size,
                },
            ),
        )
        for event in replayed:
            queue.put_nowait(event)

    def _matching_subscribers(self, event: StreamEvent) -> tuple["_Subscriber", ...]:
        return tuple(
            subscriber
            for subscriber in tuple(self._subscribers.values())
            if subscriber.event_filter.is_empty or subscriber.event_filter.matches(event)
        )

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


def _parse_cursor(value: str) -> tuple[str, int]:
    """Split `<epoch>:<sequence>`, refusing anything that is not one."""

    epoch, separator, sequence_text = value.strip().partition(":")
    if not separator or not epoch or not sequence_text.isdigit():
        raise InvalidRequestError(
            "Event cursor is not one this server issued.",
            details={"field": "after", "cursor": value, "expected": "<epoch>:<sequence>"},
        )
    return epoch, int(sequence_text)


@dataclass(slots=True)
class _Subscriber:
    queue: asyncio.Queue[StreamEvent]
    event_filter: EventFilter
