"""HTTP request guards for local LewLM deployments."""

from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
import time

from fastapi import Request

from lewlm.config.settings import LewLMSettings
from lewlm.core.errors import AuthenticationError, RateLimitError, RequestTooLargeError, UnsupportedMediaTypeError


EXEMPT_API_KEY_PATHS = {
    "/v1/health",
    "/v1/openapi.json",
    "/v1/cluster/workers/enroll",
    "/v1/cluster/workers/heartbeat",
    "/v1/cluster/worker/pipeline-stage",
}
EXEMPT_RATE_LIMIT_PATHS = {
    "/v1/health",
    "/v1/openapi.json",
    "/v1/cluster/workers/heartbeat",
    "/v1/cluster/worker/pipeline-stage",
}
ALLOWED_BODY_MEDIA_TYPES = {
    "/v1/cluster/tokens": {"application/json"},
    "/v1/cluster/workers/enroll": {"application/json"},
    "/v1/cluster/workers/heartbeat": {"application/json"},
    "/v1/cluster/worker/pipeline-stage": {"application/json"},
    "/v1/cluster/plans": {"application/json"},
    "/v1/audio/speech": {"application/json"},
    "/v1/audio/transcriptions": {"application/json", "multipart/form-data"},
    "/v1/chat/completions": {"application/json", "multipart/form-data"},
    "/v1/documents/generate": {"application/json"},
    "/v1/documents/ingest": {"application/json"},
    "/v1/documents/transform": {"application/json"},
    "/v1/embeddings": {"application/json"},
    "/v1/models/convert": {"application/json"},
    "/v1/models/scan": {"application/json"},
    "/v1/rerank": {"application/json"},
    "/v1/responses": {"application/json", "multipart/form-data"},
}


class RequestRateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max(0, max_requests)
        self.window_seconds = max(1, window_seconds)
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def enforce(self, request: Request) -> None:
        if self.max_requests <= 0 or request.url.path in EXEMPT_RATE_LIMIT_PATHS:
            return
        key = _rate_limit_key(request)
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
                raise RateLimitError(
                    "Request rate limit exceeded.",
                    details={
                        "key": key,
                        "rate_limit_requests": self.max_requests,
                        "rate_limit_window_seconds": self.window_seconds,
                        "retry_after_seconds": retry_after,
                    },
                )
            bucket.append(now)


class RequestGuard:
    """Apply request guards backed by shared in-memory state."""

    def __init__(self, settings: LewLMSettings) -> None:
        self.settings = settings
        self.rate_limiter = RequestRateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )

    async def enforce(self, request: Request) -> None:
        await _enforce_request_size(request, self.settings)
        _enforce_content_type(request)
        self.rate_limiter.enforce(request)
        _enforce_api_key(request, self.settings)


async def _enforce_request_size(request: Request, settings: LewLMSettings) -> None:
    content_length_header = request.headers.get("content-length")
    if content_length_header is None:
        if request.method not in {"POST", "PUT", "PATCH"}:
            return
        body = await request.body()
        content_length = len(body)
    else:
        try:
            content_length = int(content_length_header)
        except ValueError:
            return
    if content_length > settings.request_max_bytes:
        raise RequestTooLargeError(
            "Request body exceeds the configured maximum size.",
            details={"content_length": content_length, "request_max_bytes": settings.request_max_bytes},
        )


def _enforce_content_type(request: Request) -> None:
    if request.method not in {"POST", "PUT", "PATCH"}:
        return
    allowed_media_types = ALLOWED_BODY_MEDIA_TYPES.get(request.url.path)
    if allowed_media_types is None:
        return
    content_type = request.headers.get("content-type", "")
    media_type = content_type.partition(";")[0].strip().casefold()
    if media_type not in allowed_media_types:
        raise UnsupportedMediaTypeError(
            "This endpoint requires a supported request content type.",
            details={
                "path": request.url.path,
                "content_type": content_type or None,
                "allowed_content_types": sorted(allowed_media_types),
            },
        )


def _enforce_api_key(request: Request, settings: LewLMSettings) -> None:
    if not settings.api_key_required or request.url.path in EXEMPT_API_KEY_PATHS:
        return
    provided_key = request.headers.get("x-api-key")
    authorization = request.headers.get("authorization", "")
    if provided_key is None and authorization.lower().startswith("bearer "):
        provided_key = authorization[7:].strip()
    valid_keys = {secret.get_secret_value() for secret in settings.api_keys}
    if provided_key not in valid_keys:
        raise AuthenticationError("A valid API key is required for this request.")


def _rate_limit_key(request: Request) -> str:
    provided_key = request.headers.get("x-api-key")
    authorization = request.headers.get("authorization", "")
    if provided_key is None and authorization.lower().startswith("bearer "):
        provided_key = authorization[7:].strip()
    if provided_key:
        return f"api:{provided_key}"
    host = request.client.host if request.client is not None else "anonymous"
    return f"client:{host}"
