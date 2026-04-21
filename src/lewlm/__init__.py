"""LewLM package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lewlm._version import __version__

if TYPE_CHECKING:
    from lewlm.api.app import create_app
    from lewlm.app_helpers import LewLMAppClient
    from lewlm.config.settings import LewLMSettings
    from lewlm.core.bootstrap import LewLMServices, bootstrap_services
    from lewlm.library import LewLM

__all__ = [
    "LewLM",
    "LewLMAppClient",
    "LewLMServices",
    "LewLMSettings",
    "__version__",
    "bootstrap_services",
    "create_app",
]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from lewlm.api.app import create_app

        return create_app
    if name == "LewLMSettings":
        from lewlm.config.settings import LewLMSettings

        return LewLMSettings
    if name in {"LewLMServices", "bootstrap_services"}:
        from lewlm.core.bootstrap import LewLMServices, bootstrap_services

        return {"LewLMServices": LewLMServices, "bootstrap_services": bootstrap_services}[name]
    if name == "LewLM":
        from lewlm.library import LewLM

        return LewLM
    if name == "LewLMAppClient":
        from lewlm.app_helpers import LewLMAppClient

        return LewLMAppClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
