"""FastAPI application factory for LewLM."""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from lewlm._version import __version__
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import LewLMServices, bootstrap_services
from lewlm.core.errors import LewLMError
from lewlm.security.http import RequestGuard

from .routes.chat import router as chat_router
from .routes.cluster import router as cluster_router
from .routes.documents import router as documents_router
from .routes.events import router as events_router
from .routes.health import router as health_router
from .routes.history import router as history_router
from .routes.lewlm import router as lewlm_router
from .routes.models import router as models_router
from .routes.multimodal import router as multimodal_router
from .routes.operations import router as operations_router
from .routes.skills import router as skills_router
from .routes.tools import router as tools_router


def create_app(
    settings: LewLMSettings | None = None,
    *,
    services: LewLMServices | None = None,
) -> FastAPI:
    """Create a configured FastAPI application."""

    resolved_settings = services.settings if services is not None else (settings or LewLMSettings())
    owns_services = services is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services or bootstrap_services(resolved_settings)
        app.state.services.event_bus.attach_loop(asyncio.get_running_loop())
        try:
            yield
        finally:
            if owns_services:
                await app.state.services.aclose()

    app = FastAPI(
        title="LewLM",
        version=resolved_settings.version if resolved_settings is not None else __version__,
        openapi_url="/v1/openapi.json",
        lifespan=lifespan,
    )
    request_guard = RequestGuard(resolved_settings)

    @app.middleware("http")
    async def request_guard_middleware(request: Request, call_next):
        try:
            await request_guard.enforce(request)
        except LewLMError as exc:
            _audit_request_failure(request, exc)
            return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})
        return await call_next(request)

    app.include_router(chat_router)
    app.include_router(cluster_router)
    app.include_router(documents_router)
    app.include_router(events_router)
    app.include_router(health_router)
    app.include_router(history_router)
    app.include_router(lewlm_router)
    app.include_router(models_router)
    app.include_router(multimodal_router)
    app.include_router(operations_router)
    app.include_router(skills_router)
    app.include_router(tools_router)

    @app.exception_handler(LewLMError)
    async def handle_lewlm_error(request: Request, exc: LewLMError) -> JSONResponse:
        _audit_request_failure(request, exc)
        return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})

    return app


def _audit_request_failure(request: Request, exc: LewLMError) -> None:
    services = getattr(request.app.state, "services", None)
    if services is None:
        return
    services.audit_logger.record(
        action="http_request",
        outcome="failed",
        actor="api",
        details={
            "path": request.url.path,
            "method": request.method,
            "error_code": exc.code,
            "status_code": exc.status_code,
        },
    )
