"""Shared API dependency helpers."""

from __future__ import annotations

from fastapi import Request

from lewlm.core.bootstrap import LewLMServices


def get_services(request: Request) -> LewLMServices:
    """Fetch the initialized service container from the application state."""

    return request.app.state.services  # type: ignore[no-any-return]
