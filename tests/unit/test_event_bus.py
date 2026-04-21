from __future__ import annotations

from lewlm.events.bus import EventBus
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
