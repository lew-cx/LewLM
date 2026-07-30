"""FastAPI application factory for LewLM."""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import logging
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from lewlm._version import __version__
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import LewLMServices, bootstrap_services
from lewlm.core.errors import FRAMEWORK_ERROR_CODES, InternalError, InvalidRequestError, LewLMError
from lewlm.security.http import RequestGuard
from lewlm.api.openapi import normalize_openapi_schema, register_named_schemas
from lewlm.api.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ResponseChunk,
    ResponseCreateRequest,
    ResponseCreateResponse,
)
from lewlm.api.schemas.multimodal import (
    AudioTranscriptionCreateRequest,
    AudioTranscriptionMultipartRequest,
)
from lewlm.events.schema import StreamEvent
from lewlm.runtime.request_context import (
    current_correlation_id,
    normalize_correlation_id,
    reset_application_context,
    set_application_context,
)

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


#: Models the document must name even though no route binds them as a body or
#: `response_model`: the streaming chunks, which only ever appear inside SSE
#: frames, and the request bodies declared through `openapi_extra` — including
#: both shapes the audio-transcription route accepts, which it hand-parses.
#: Without this they exist only as anonymous inline schemas, so a code generator
#: has no named type to emit for the streaming half of the API.
PUBLISHED_SCHEMA_MODELS: tuple[type, ...] = (
    AudioTranscriptionCreateRequest,
    AudioTranscriptionMultipartRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ResponseCreateRequest,
    ResponseCreateResponse,
    ResponseChunk,
    StreamEvent,
)


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
    # WebSocket routes never reach the HTTP middleware, so they read the guard
    # off app state and enforce it on the handshake themselves.
    app.state.request_guard = request_guard

    if resolved_settings.cors_enabled:
        # Opt-in only, and never with a wildcard origin plus credentials —
        # settings validation refuses that combination.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_allow_origins),
            allow_credentials=resolved_settings.cors_allow_credentials,
            allow_methods=list(resolved_settings.cors_allow_methods),
            allow_headers=list(resolved_settings.cors_allow_headers),
            expose_headers=list(resolved_settings.cors_expose_headers),
            max_age=resolved_settings.cors_max_age_seconds,
        )

    @app.middleware("http")
    async def request_guard_middleware(request: Request, call_next):
        correlation_id = normalize_correlation_id(request.headers.get("x-lewlm-correlation-id"))
        # A caller-supplied request ID is echoed as-is; otherwise LewLM mints one
        # so every response is traceable in logs even without a caller ID.
        request_id = normalize_correlation_id(request.headers.get("x-request-id")) or str(uuid4())
        context_tokens = set_application_context(
            application_id=request.headers.get("x-lewlm-application-id"),
            client_instance_id=request.headers.get("x-lewlm-client-instance-id"),
            correlation_id=correlation_id,
        )
        request.state.request_id = request_id
        try:
            await request_guard.enforce(request)
        except LewLMError as exc:
            _audit_request_failure(request, exc)
            response = JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})
        else:
            response = await call_next(request)
        finally:
            reset_application_context(context_tokens)
        response.headers["x-request-id"] = request_id
        if correlation_id is not None:
            # Echoing the header lets a caller correlate even a response it
            # cannot parse, such as a proxy-generated error.
            response.headers["x-lewlm-correlation-id"] = correlation_id
        return response

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
        return _error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Return FastAPI's body-binding failures in LewLM's envelope."""

        error = InvalidRequestError(
            "Request payload failed validation.",
            details={"fields": _validation_field_details(exc.errors())},
        )
        _audit_request_failure(request, error)
        return _error_response(request, error)

    @app.exception_handler(ValidationError)
    async def handle_model_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        """Routes that parse bodies themselves raise pydantic errors directly.

        Without this they escape as an untyped 500, which is indistinguishable
        from a server fault for a caller that merely sent a bad field.
        """

        error = InvalidRequestError(
            "Request payload failed validation.",
            details={"fields": _validation_field_details(exc.errors())},
        )
        _audit_request_failure(request, error)
        return _error_response(request, error)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Normalize framework HTTP errors (404, 405, …) onto the same envelope."""

        error = LewLMError(
            str(exc.detail),
            code=_HTTP_ERROR_CODES.get(int(exc.status_code), "http_error"),
            status_code=int(exc.status_code),
        )
        return _error_response(request, error, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Never let an unhandled failure reach a caller as a bare 500.

        The message stays generic; the exception type is surfaced so a host app
        can report something actionable without LewLM leaking internals.
        """

        error = InternalError(
            "LewLM failed to complete the request.",
            details={"cause_type": type(exc).__name__},
        )
        logging.getLogger("lewlm.api").exception(
            "Unhandled error serving %s %s", request.method, request.url.path,
        )
        _audit_request_failure(request, error)
        return _error_response(request, error)

    base_openapi = app.openapi

    def openapi() -> dict:
        """Publish a document whose references resolve as served."""

        if app.openapi_schema is None:
            app.openapi_schema = normalize_openapi_schema(
                register_named_schemas(base_openapi(), PUBLISHED_SCHEMA_MODELS),
            )
        return app.openapi_schema

    app.openapi = openapi
    return app


#: Framework status codes mapped onto stable LewLM error codes. Declared beside
#: the exception classes so the published catalog and the runtime agree.
_HTTP_ERROR_CODES = FRAMEWORK_ERROR_CODES

#: Cap on reported field errors so a pathological body cannot inflate a response.
_MAX_REPORTED_VALIDATION_FIELDS = 20


def _validation_field_details(errors: list[dict]) -> list[dict[str, str]]:
    """Reduce validation errors to a stable, caller-usable field list."""

    details: list[dict[str, str]] = []
    for item in errors[:_MAX_REPORTED_VALIDATION_FIELDS]:
        location = [str(part) for part in item.get("loc", ()) if part != "body"]
        details.append(
            {
                "field": ".".join(location) or "<root>",
                "message": str(item.get("msg", "Invalid value.")),
                "type": str(item.get("type", "value_error")),
            },
        )
    return details


def _error_response(request: Request, exc: LewLMError, *, headers: dict | None = None) -> JSONResponse:
    """Render one error envelope, preserving request correlation headers."""

    response = JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.to_dict()},
        headers=headers,
    )
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        response.headers["x-request-id"] = request_id
    correlation_id = current_correlation_id()
    if correlation_id:
        response.headers["x-lewlm-correlation-id"] = correlation_id
    return response


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
