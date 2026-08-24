"""Server-side narrowing of the event stream.

A subscriber that wants one request's tokens should not have to receive every
other request's tokens to find them. Without a filter the bus fans every event
out to every queue and each client discards what it did not ask for, which
costs the most exactly when the server is busiest — a single streaming
generation emits one `token.delta` per token, to every listener.

The filter is applied at the bus, before an event is enqueued, so an unwanted
event is never queued, never serialized and never sent. That makes it a
backpressure control rather than a convenience: a filtered subscriber's queue
stays shallow while an unfiltered one grows.

Every dimension is optional. Values inside one dimension are alternatives (any
may match); dimensions are combined (all must match). An empty filter admits
everything, which is what an unfiltered subscriber gets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lewlm.events.schema import EventScope, EventType, StreamEvent


@dataclass(frozen=True, slots=True)
class EventFilter:
    """Which events a subscriber wants to receive."""

    types: frozenset[EventType] = field(default_factory=frozenset)
    scopes: frozenset[EventScope] = field(default_factory=frozenset)
    request_ids: frozenset[str] = field(default_factory=frozenset)
    model_ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_empty(self) -> bool:
        """True when the filter constrains nothing and can be skipped entirely."""

        return not (self.types or self.scopes or self.request_ids or self.model_ids)

    def matches(self, event: StreamEvent) -> bool:
        if self.types and event.type not in self.types:
            return False
        if self.scopes and event.scope not in self.scopes:
            return False
        # An event carrying no `request_id` cannot belong to a requested one, so
        # asking for a request excludes host-wide events rather than passing them
        # through as unattributed.
        if self.request_ids and event.request_id not in self.request_ids:
            return False
        if self.model_ids and event.model_id not in self.model_ids:
            return False
        return True
