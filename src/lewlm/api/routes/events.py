"""SSE and WebSocket event streams."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Annotated

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from lewlm.api.dependencies import get_services
from lewlm.core.errors import InvalidRequestError, LewLMError
from lewlm.events.filters import EventFilter
from lewlm.events.schema import EventScope, EventType


router = APIRouter(tags=["events"])
#: Published on the query parameters so the schema names every value a filter
#: accepts, rather than leaving a caller to discover them by trial.
_EVENT_TYPE_SCHEMA = {"items": {"type": "string", "enum": [item.value for item in EventType]}}
_EVENT_SCOPE_SCHEMA = {"items": {"type": "string", "enum": [item.value for item in EventScope]}}
_FILTER_DESCRIPTION = "Repeatable or comma-separated. Values here are alternatives; separate filters combine."
#: RFC 6455 close codes for the two ways a handshake can be refused.
_WEBSOCKET_CLOSE_CODES = {401: 1008, 403: 1008, 429: 1013}
_EVENT_STREAM_EXAMPLE = (
    'event: request.completed\n'
    'data: {"event_id":"evt-001","type":"request.completed","scope":"request",'
    '"created_at":"2026-04-17T17:46:33Z","payload":{"request_id":"req-chat-001","path":"/v1/chat/completions"}}\n\n'
)


@router.get(
    "/v1/events",
    responses={
        200: {
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": (
                            "Server-sent event frames containing LewLM event envelopes and "
                            "keep-alive comments."
                        ),
                    },
                    "example": _EVENT_STREAM_EXAMPLE,
                },
            },
        },
    },
)
async def stream_events(
    request: Request,
    types: Annotated[
        list[str],
        Query(description=f"Event types to deliver. {_FILTER_DESCRIPTION}", json_schema_extra=_EVENT_TYPE_SCHEMA),
    ] = (),
    scope: Annotated[
        list[str],
        Query(description=f"Event scopes to deliver. {_FILTER_DESCRIPTION}", json_schema_extra=_EVENT_SCOPE_SCHEMA),
    ] = (),
    request_id: Annotated[
        list[str],
        Query(description=f"Deliver only events belonging to these requests. {_FILTER_DESCRIPTION}"),
    ] = (),
    model_id: Annotated[
        list[str],
        Query(description=f"Deliver only events about these models. {_FILTER_DESCRIPTION}"),
    ] = (),
) -> StreamingResponse:
    """Stream runtime and request lifecycle events over SSE.

    Each parameter narrows the stream and may be repeated or comma-separated.
    Values within one parameter are alternatives; the parameters combine, so
    `?types=token.delta&request_id=req-1` is one request's tokens and nothing
    else. Filtering happens before an event is queued for this connection, so an
    excluded event costs the connection nothing.
    """

    services = get_services(request)
    subscription = services.event_bus.subscribe(
        _event_filter_from_query(types=types, scope=scope, request_id=request_id, model_id=model_id),
    )

    async def iterator() -> AsyncIterator[str]:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(subscription.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield event.to_event_stream()
        finally:
            subscription.close()

    return StreamingResponse(iterator(), media_type="text/event-stream")


@router.websocket("/v1/events")
async def websocket_events(websocket: WebSocket) -> None:
    """Stream runtime and request lifecycle events over WebSocket.

    The HTTP middleware that carries `RequestGuard` does not run for WebSocket
    routes, so the handshake is guarded here before anything is accepted.
    """

    guard = getattr(websocket.app.state, "request_guard", None)
    if guard is not None:
        try:
            guard.enforce_websocket(websocket)
        except LewLMError as exc:
            # Refuse before accepting: the client sees a failed handshake with
            # the reason, not an open socket that goes quiet.
            await websocket.close(code=_WEBSOCKET_CLOSE_CODES.get(exc.status_code, 1008), reason=exc.code)
            return

    try:
        event_filter = _event_filter_from_query(
            types=websocket.query_params.getlist("types"),
            scope=websocket.query_params.getlist("scope"),
            request_id=websocket.query_params.getlist("request_id"),
            model_id=websocket.query_params.getlist("model_id"),
        )
    except InvalidRequestError as exc:
        # Refuse before accepting, for the same reason the guard does: a client
        # that mistyped a filter should see a failed handshake, not an open
        # socket that silently delivers everything or nothing.
        await websocket.close(code=1008, reason=exc.code)
        return

    await websocket.accept()
    services = websocket.app.state.services
    subscription = services.event_bus.subscribe(event_filter)
    try:
        while True:
            event = await subscription.get()
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    finally:
        subscription.close()


def _event_filter_from_query(
    *,
    types: Iterable[str] | None,
    scope: Iterable[str] | None,
    request_id: Iterable[str] | None,
    model_id: Iterable[str] | None,
) -> EventFilter:
    """Build a filter from query parameters, refusing values that name nothing.

    An unrecognized event type is refused rather than ignored: silently dropping
    it would hand back an empty stream that looks exactly like a quiet server.
    """

    invalid_fields: list[dict[str, str]] = []
    event_types = frozenset(
        _resolve_enum_values(EventType, _split_values(types), field="types", invalid_fields=invalid_fields),
    )
    scopes = frozenset(
        _resolve_enum_values(EventScope, _split_values(scope), field="scope", invalid_fields=invalid_fields),
    )
    if invalid_fields:
        raise InvalidRequestError(
            "Event stream filter names a value that does not exist.",
            details={"fields": invalid_fields},
        )
    return EventFilter(
        types=event_types,
        scopes=scopes,
        request_ids=frozenset(_split_values(request_id)),
        model_ids=frozenset(_split_values(model_id)),
    )


def _split_values(values: Iterable[str] | None) -> list[str]:
    """Accept both repeated parameters and comma-separated lists."""

    if not values:
        return []
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def _resolve_enum_values(
    enum_type: type[EventType] | type[EventScope],
    values: list[str],
    *,
    field: str,
    invalid_fields: list[dict[str, str]],
) -> list[EventType | EventScope]:
    resolved: list[EventType | EventScope] = []
    for value in values:
        try:
            resolved.append(enum_type(value))
        except ValueError:
            invalid_fields.append(
                {
                    "field": field,
                    "message": f"`{value}` is not a known {field.rstrip('s')} value.",
                    "type": "enum",
                },
            )
    return resolved
