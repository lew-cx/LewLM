"""SSE and WebSocket event streams."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from lewlm.api.dependencies import get_services
from lewlm.core.errors import LewLMError


router = APIRouter(tags=["events"])
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
async def stream_events(request: Request) -> StreamingResponse:
    """Stream runtime and request lifecycle events over SSE."""

    services = get_services(request)
    subscription = services.event_bus.subscribe()

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

    await websocket.accept()
    services = websocket.app.state.services
    subscription = services.event_bus.subscribe()
    try:
        while True:
            event = await subscription.get()
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    finally:
        subscription.close()
