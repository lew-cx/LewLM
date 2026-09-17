"""Bounded asynchronous HTTP transport for loopback engine bridges."""

from __future__ import annotations

import codecs
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
import json
import os
from typing import Any

import httpx

from lewlm.config.endpoints import ExternalEndpoint, server_root
from lewlm.core.errors import RuntimeUnavailableError


@dataclass(frozen=True, slots=True)
class ServerSentEvent:
    """One decoded SSE event.

    ``data`` follows the SSE rule for repeated ``data:`` fields: their values
    are joined with a newline after incremental UTF-8 and line decoding.
    """

    data: str
    event: str | None = None
    event_id: str | None = None


class _SSEDecoder:
    """Incrementally decode arbitrary byte fragments into SSE events."""

    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")("strict")
        self._buffer = ""
        self._data: list[str] = []
        self._event: str | None = None
        self._event_id: str | None = None

    def feed(self, chunk: bytes) -> list[ServerSentEvent]:
        try:
            self._buffer += self._utf8.decode(chunk)
        except UnicodeDecodeError as exc:
            raise RuntimeUnavailableError(
                "External accelerator returned invalid UTF-8 in its event stream.",
                details={"error_kind": "malformed_stream"},
            ) from exc
        return self._consume_lines(final=False)

    def finish(self) -> list[ServerSentEvent]:
        try:
            self._buffer += self._utf8.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise RuntimeUnavailableError(
                "External accelerator ended with an incomplete UTF-8 sequence.",
                details={"error_kind": "malformed_stream"},
            ) from exc
        events = self._consume_lines(final=True)
        if self._buffer:
            events.extend(self._consume_line(self._buffer))
            self._buffer = ""
        event = self._dispatch()
        if event is not None:
            events.append(event)
        return events

    def _consume_lines(self, *, final: bool) -> list[ServerSentEvent]:
        events: list[ServerSentEvent] = []
        cursor = 0
        index = 0
        length = len(self._buffer)
        while index < length:
            character = self._buffer[index]
            if character not in {"\r", "\n"}:
                index += 1
                continue
            if character == "\r" and index + 1 >= length and not final:
                break
            line = self._buffer[cursor:index]
            if character == "\r" and index + 1 < length and self._buffer[index + 1] == "\n":
                index += 1
            index += 1
            cursor = index
            events.extend(self._consume_line(line))
        self._buffer = self._buffer[cursor:]
        return events

    def _consume_line(self, line: str) -> list[ServerSentEvent]:
        if not line:
            event = self._dispatch()
            return [event] if event is not None else []
        if line.startswith(":"):
            return []
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value
        elif field == "id" and "\x00" not in value:
            self._event_id = value
        return []

    def _dispatch(self) -> ServerSentEvent | None:
        if not self._data:
            self._event = None
            return None
        event = ServerSentEvent(data="\n".join(self._data), event=self._event, event_id=self._event_id)
        self._data = []
        self._event = None
        return event


class AsyncBridgeTransport:
    """Reusable, bounded and proxy-independent client for one local endpoint."""

    def __init__(
        self,
        *,
        endpoint: ExternalEndpoint,
        runtime_name: str,
        max_connections: int = 32,
        max_keepalive_connections: int = 16,
    ) -> None:
        self.endpoint = endpoint
        self.runtime_name = runtime_name
        self._secret_values: tuple[str, ...] = ()
        self._missing_api_key_env: str | None = None
        self._client = httpx.AsyncClient(
            base_url=server_root(endpoint.base_url),
            timeout=httpx.Timeout(
                connect=endpoint.connect_timeout_seconds,
                read=endpoint.read_timeout_seconds,
                write=endpoint.read_timeout_seconds,
                pool=endpoint.pool_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
            trust_env=False,
            follow_redirects=False,
            headers=self._authorization_headers(),
        )
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._client.aclose()

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        files: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._send(method, path, payload=payload, files=files, data=data)
        try:
            parsed = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise self._error(
                "External accelerator returned malformed JSON.",
                path=path,
                error_kind="malformed_response",
            ) from exc
        if not isinstance(parsed, dict):
            raise self._error(
                "External accelerator returned an unexpected JSON payload.",
                path=path,
                error_kind="malformed_response",
                payload_type=type(parsed).__name__,
            )
        return parsed

    async def request_bytes(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        accept: str = "application/octet-stream",
    ) -> tuple[bytes, str]:
        response = await self._send(method, path, payload=payload, headers={"Accept": accept})
        media_type = response.headers.get("content-type", "application/octet-stream").partition(";")[0].strip()
        return response.content, media_type

    async def stream_sse(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any],
    ) -> AsyncIterator[ServerSentEvent]:
        self._ensure_credentials()
        try:
            async with self._client.stream(method, path, json=dict(payload), headers={"Accept": "text/event-stream"}) as response:
                self._raise_for_response(response, path=path)
                decoder = _SSEDecoder()
                async for chunk in response.aiter_bytes():
                    for event in decoder.feed(chunk):
                        yield event
                for event in decoder.finish():
                    yield event
        except RuntimeUnavailableError:
            raise
        except httpx.TimeoutException as exc:
            raise self._error(
                "External accelerator request timed out.",
                path=path,
                error_kind="timeout",
                reason=str(exc),
            ) from exc
        except httpx.ConnectError as exc:
            raise self._error(
                "Could not connect to the configured external accelerator endpoint.",
                path=path,
                error_kind="unavailable",
                reason=str(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise self._error(
                "External accelerator stream failed.",
                path=path,
                error_kind="unavailable",
                reason=str(exc),
            ) from exc

    async def _send(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        files: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        self._ensure_credentials()
        try:
            response = await self._client.request(
                method,
                path,
                json=dict(payload) if payload is not None else None,
                files=files,
                data=data,
                headers=headers,
            )
            self._raise_for_response(response, path=path)
            return response
        except RuntimeUnavailableError:
            raise
        except httpx.TimeoutException as exc:
            raise self._error(
                "External accelerator request timed out.",
                path=path,
                error_kind="timeout",
                reason=str(exc),
            ) from exc
        except httpx.ConnectError as exc:
            raise self._error(
                "Could not connect to the configured external accelerator endpoint.",
                path=path,
                error_kind="unavailable",
                reason=str(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise self._error(
                "External accelerator request failed.",
                path=path,
                error_kind="unavailable",
                reason=str(exc),
            ) from exc

    def _raise_for_response(self, response: httpx.Response, *, path: str) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        category = (
            "authentication" if status in {401, 403}
            else "redirect" if 300 <= status < 400
            else "rate_limited" if status == 429
            else "model_not_found" if status == 404
            else "invalid_request" if 400 <= status < 500
            else "unavailable"
        )
        body = response.text[:4096]
        raise self._error(
            f"External accelerator request failed with HTTP {status}.",
            path=path,
            error_kind=category,
            status_code=status,
            body=body,
        )

    def _authorization_headers(self) -> dict[str, str]:
        variable = self.endpoint.api_key_env
        if variable is None:
            return {}
        value = os.environ.get(variable)
        if not value:
            self._missing_api_key_env = variable
            return {}
        self._secret_values = (value,)
        return {"Authorization": f"Bearer {value}"}

    def _ensure_credentials(self) -> None:
        variable = self.endpoint.api_key_env
        if variable is None:
            return
        value = os.environ.get(variable)
        if value:
            self._missing_api_key_env = None
            self._secret_values = (value,)
            self._client.headers["Authorization"] = f"Bearer {value}"
            return
        self._missing_api_key_env = variable
        raise RuntimeUnavailableError(
            f"External endpoint `{self.endpoint.endpoint_id}` requires the environment variable `{variable}`.",
            details={
                "runtime": self.runtime_name,
                "endpoint_id": self.endpoint.endpoint_id,
                "error_kind": "authentication",
                "api_key_env": variable,
            },
        )

    def _error(self, message: str, *, path: str, error_kind: str, **details: Any) -> RuntimeUnavailableError:
        return RuntimeUnavailableError(
            message,
            details={
                "runtime": self.runtime_name,
                "endpoint_id": self.endpoint.endpoint_id,
                "path": path,
                "error_kind": error_kind,
                **{
                    key: self._redact(value)
                    for key, value in details.items()
                },
            },
        )

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            for secret in self._secret_values:
                value = value.replace(secret, "[REDACTED]")
        return value


__all__ = ["AsyncBridgeTransport", "ServerSentEvent"]
