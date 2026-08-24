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
