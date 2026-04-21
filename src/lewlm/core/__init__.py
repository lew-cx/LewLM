"""Core contracts and application services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lewlm.core.bootstrap import LewLMServices, bootstrap_services

__all__ = ["LewLMServices", "bootstrap_services"]


def __getattr__(name: str) -> Any:
    if name in {"LewLMServices", "bootstrap_services"}:
        from lewlm.core.bootstrap import LewLMServices, bootstrap_services

        return {"LewLMServices": LewLMServices, "bootstrap_services": bootstrap_services}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
