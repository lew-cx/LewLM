from __future__ import annotations

from lewlm.events.bus import EventBus
from lewlm.events.filters import EventFilter
from lewlm.events.schema import EventScope, EventType, StreamEvent


async def test_event_bus_delivers_published_events() -> None:
    bus = EventBus()
    subscription = bus.subscribe()
    event = StreamEvent(
        type=EventType.SYSTEM_READY,
        scope=EventScope.SYSTEM,
        payload={"ready": True},
    )

    await bus.publish(event)
    delivered = await subscription.get()

    assert delivered.type == EventType.SYSTEM_READY
    assert delivered.payload["ready"] is True
    subscription.close()
    assert bus.subscriber_count == 0


async def test_filtered_subscriber_never_queues_the_events_it_excluded() -> None:
    """The point of filtering at the bus is that an excluded event costs nothing.

    A subscriber that filtered client-side would still have every event queued
    and serialized for it, which is what the queue depth here asserts against.
    """

    bus = EventBus()
    tokens_only = bus.subscribe(EventFilter(types=frozenset({EventType.TOKEN_DELTA})))
    everything = bus.subscribe()

    await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={"text": "a"}))
    await bus.publish(StreamEvent(type=EventType.SYSTEM_READY, payload={"ready": True}))
    await bus.publish(StreamEvent(type=EventType.REQUEST_COMPLETED, payload={"request_id": "req-1"}))

    assert tokens_only.queue.qsize() == 1
    assert everything.queue.qsize() == 3
    assert (await tokens_only.get()).type == EventType.TOKEN_DELTA

    tokens_only.close()
    everything.close()


async def test_request_filter_isolates_one_request_from_the_rest() -> None:
    bus = EventBus()
    subscription = bus.subscribe(EventFilter(request_ids=frozenset({"req-1"})))

    await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={"request_id": "req-1", "text": "mine"}))
    await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={"request_id": "req-2", "text": "theirs"}))
    # A host-wide event belongs to no request, so it is not this request's.
    await bus.publish(StreamEvent(type=EventType.SYSTEM_READY, payload={"ready": True}))

    assert subscription.queue.qsize() == 1
    delivered = await subscription.get()
    assert delivered.payload["text"] == "mine"
    subscription.close()


async def test_filter_dimensions_combine() -> None:
    bus = EventBus()
    subscription = bus.subscribe(
        EventFilter(types=frozenset({EventType.TOKEN_DELTA}), model_ids=frozenset({"model-a"})),
    )

    await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={"model_id": "model-b"}))
    await bus.publish(StreamEvent(type=EventType.MODEL_LOADED, payload={"model_id": "model-a"}))
    await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={"model_id": "model-a", "text": "hit"}))

    assert subscription.queue.qsize() == 1
    assert (await subscription.get()).payload["text"] == "hit"
    subscription.close()


async def test_values_within_one_dimension_are_alternatives() -> None:
    bus = EventBus()
    subscription = bus.subscribe(
        EventFilter(types=frozenset({EventType.TOKEN_DELTA, EventType.REQUEST_COMPLETED})),
    )

    await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={}))
    await bus.publish(StreamEvent(type=EventType.REQUEST_COMPLETED, payload={"request_id": "req-1"}))
    await bus.publish(StreamEvent(type=EventType.SYSTEM_READY, payload={}))

    assert subscription.queue.qsize() == 2
    subscription.close()


async def test_threadsafe_publish_honours_the_same_filter() -> None:
    """Streaming runtimes publish token deltas from a worker thread."""

    import asyncio

    bus = EventBus()
    bus.attach_loop(asyncio.get_running_loop())
    subscription = bus.subscribe(EventFilter(types=frozenset({EventType.TOKEN_DELTA})))

    bus.publish_threadsafe(StreamEvent(type=EventType.TOKEN_DELTA, payload={"text": "a"}))
    bus.publish_threadsafe(StreamEvent(type=EventType.SYSTEM_READY, payload={}))
    await asyncio.sleep(0)

    assert subscription.queue.qsize() == 1
    subscription.close()


async def test_empty_filter_admits_everything() -> None:
    bus = EventBus()
    subscription = bus.subscribe(EventFilter())

    await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={}))
    await bus.publish(StreamEvent(type=EventType.SYSTEM_READY, payload={}))

    assert subscription.queue.qsize() == 2
    assert subscription.event_filter.is_empty is True
    subscription.close()


async def test_scope_filter_selects_by_event_scope() -> None:
    bus = EventBus()
    subscription = bus.subscribe(EventFilter(scopes=frozenset({EventScope.REQUEST})))

    await bus.publish(StreamEvent(type=EventType.SYSTEM_READY, scope=EventScope.SYSTEM))
    await bus.publish(
        StreamEvent(type=EventType.REQUEST_COMPLETED, scope=EventScope.REQUEST, payload={"request_id": "req-1"}),
    )

    assert subscription.queue.qsize() == 1
    assert (await subscription.get()).scope == EventScope.REQUEST
    subscription.close()


# --- every event has a cursor and a reconnect can resume from it (G13) --------


def _drain(subscription) -> list[StreamEvent]:
    return [subscription.queue.get_nowait() for _ in range(subscription.queue.qsize())]


async def test_every_published_event_carries_a_monotonic_cursor_and_the_caller_object_is_untouched() -> None:
    bus = EventBus()
    subscription = bus.subscribe()
    original = StreamEvent(type=EventType.SYSTEM_READY)

    await bus.publish(original)
    await bus.publish(StreamEvent(type=EventType.REQUEST_ACCEPTED, payload={"request_id": "req-1"}))
    first, second = _drain(subscription)

    assert original.cursor is None, "the bus stamps a copy, not the publisher's object"
    assert first.cursor is not None and second.cursor is not None
    epoch, _, first_sequence = first.cursor.partition(":")
    assert epoch and int(first_sequence) == 1
    assert second.cursor == f"{epoch}:2"
    assert bus.latest_cursor == second.cursor
    # The frame's `id:` is the cursor, so an EventSource can send it back.
    assert first.to_event_stream().startswith(f"id: {first.cursor}\nevent: system.ready\n")


async def test_a_resumed_subscription_replays_exactly_what_it_missed_then_goes_live() -> None:
    bus = EventBus()
    watcher = bus.subscribe()
    for index in range(1, 6):
        await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={"n": index}))
    seen = _drain(watcher)
    watcher.close()  # the client drops after event 3

    resumed = bus.subscribe(after=seen[2].cursor)
    await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={"n": 6}))
    frames = _drain(resumed)

    marker, *events = frames
    assert marker.type == EventType.EVENTS_RESUMED and marker.cursor is None
    assert marker.payload["after"] == seen[2].cursor
    assert marker.payload["replayed"] == 2 and marker.payload["lost"] == 0
    assert marker.payload["latest"] == seen[4].cursor, "latest is as of the resume, before the live event"
    assert [event.payload["n"] for event in events] == [4, 5, 6], "replayed in order, then live, no gap, no repeat"
    assert [event.cursor for event in events[:2]] == [seen[3].cursor, seen[4].cursor], "replayed events keep their cursors"


async def test_replay_honours_the_subscriber_filter() -> None:
    bus = EventBus()
    watcher = bus.subscribe()
    await bus.publish(StreamEvent(type=EventType.REQUEST_ACCEPTED, payload={"request_id": "req-1"}))
    await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={"request_id": "req-1"}))
    await bus.publish(StreamEvent(type=EventType.REQUEST_COMPLETED, payload={"request_id": "req-1"}))
    first, *_ = _drain(watcher)

    resumed = bus.subscribe(EventFilter(types=frozenset({EventType.REQUEST_COMPLETED})), after=first.cursor)
    marker, *events = _drain(resumed)

    assert marker.payload["replayed"] == 1
    assert [event.type for event in events] == [EventType.REQUEST_COMPLETED]


async def test_a_cursor_older_than_the_buffer_reports_how_much_was_lost() -> None:
    bus = EventBus(replay_buffer_size=3)
    watcher = bus.subscribe()
    for index in range(1, 8):
        await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={"n": index}))
    seen = _drain(watcher)

    resumed = bus.subscribe(after=seen[0].cursor)
    marker, *events = _drain(resumed)

    # Events 2..4 were published after the cursor but are no longer retained.
    assert marker.payload["lost"] == 3 and marker.payload["replayed"] == 3
    assert marker.payload["retained"] == 3 and marker.payload["replay_buffer_size"] == 3
    assert [event.payload["n"] for event in events] == [5, 6, 7]


async def test_a_cursor_from_another_server_lifetime_is_not_matched_against_this_one() -> None:
    previous = EventBus()
    previous_watcher = previous.subscribe()
    await previous.publish(StreamEvent(type=EventType.SYSTEM_READY))
    stale_cursor = _drain(previous_watcher)[0].cursor

    bus = EventBus()
    for _ in range(3):
        await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA))
    resumed = bus.subscribe(after=stale_cursor)
    marker, *events = _drain(resumed)

    # Sequence 1 exists here too, but it is not the same stream: nothing is
    # replayed and the loss is unknowable rather than zero.
    assert marker.payload["lost"] is None and marker.payload["replayed"] == 0
    assert events == []


async def test_a_disabled_buffer_still_stamps_cursors_but_replays_nothing() -> None:
    bus = EventBus(replay_buffer_size=0)
    watcher = bus.subscribe()
    await bus.publish(StreamEvent(type=EventType.SYSTEM_READY))
    await bus.publish(StreamEvent(type=EventType.SYSTEM_READY))
    first, second = _drain(watcher)

    resumed = bus.subscribe(after=first.cursor)
    marker, *events = _drain(resumed)

    assert first.cursor is not None and second.cursor is not None
    assert marker.payload["lost"] == 1 and marker.payload["replayed"] == 0 and events == []


async def test_malformed_and_future_cursors_are_refused() -> None:
    import pytest

    from lewlm.core.errors import InvalidRequestError

    bus = EventBus()
    await bus.publish(StreamEvent(type=EventType.SYSTEM_READY))

    with pytest.raises(InvalidRequestError) as malformed:
        bus.subscribe(after="not-a-cursor")
    assert malformed.value.details["field"] == "after"

    with pytest.raises(InvalidRequestError) as ahead:
        bus.subscribe(after=f"{bus.latest_cursor.partition(':')[0]}:99")
    assert ahead.value.details["latest"] == bus.latest_cursor
    assert bus.subscriber_count == 0, "a refused resume leaves no subscription behind"


async def test_exclude_types_drops_the_flood_and_keeps_the_rest() -> None:
    bus = EventBus()
    no_tokens = bus.subscribe(EventFilter(excluded_types=frozenset({EventType.TOKEN_DELTA, EventType.REASONING_DELTA})))

    await bus.publish(StreamEvent(type=EventType.REQUEST_ACCEPTED, payload={"request_id": "req-1"}))
    await bus.publish(StreamEvent(type=EventType.TOKEN_DELTA, payload={"request_id": "req-1"}))
    await bus.publish(StreamEvent(type=EventType.REASONING_DELTA, payload={"request_id": "req-1"}))
    await bus.publish(StreamEvent(type=EventType.REQUEST_COMPLETED, payload={"request_id": "req-1"}))

    assert [event.type for event in _drain(no_tokens)] == [EventType.REQUEST_ACCEPTED, EventType.REQUEST_COMPLETED]
