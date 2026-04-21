"""In-flight request coalescing for deterministic runtime work."""

from __future__ import annotations

import asyncio
from threading import Lock
from typing import Generic, TypeVar


ResponseT = TypeVar("ResponseT")


class InFlightRequestCoalescer(Generic[ResponseT]):
    """Ensure concurrent identical requests share one owner execution."""

    def __init__(self) -> None:
        self._futures: dict[str, asyncio.Future[ResponseT]] = {}
        self._lock = Lock()

    def claim(self, key: str) -> tuple[bool, asyncio.Future[ResponseT]]:
        loop = asyncio.get_running_loop()
        with self._lock:
            future = self._futures.get(key)
            if future is not None:
                return False, future
            future = loop.create_future()
            self._futures[key] = future
            return True, future

    def resolve(self, key: str, response: ResponseT) -> None:
        with self._lock:
            future = self._futures.pop(key, None)
        if future is not None and not future.done():
            future.set_result(response)

    def reject(self, key: str, exc: Exception) -> None:
        with self._lock:
            future = self._futures.pop(key, None)
        if future is not None and not future.done():
            future.set_exception(exc)
