"""Structured LewLM exceptions."""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any


class LewLMError(Exception):
    """Base class for structured LewLM errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "lewlm_error",
        status_code: int = HTTPStatus.BAD_REQUEST,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = int(status_code)
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }


class ConfigurationError(LewLMError):
    """Raised when application settings are invalid."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="configuration_error",
            status_code=HTTPStatus.BAD_REQUEST,
            details=details,
        )


class InvalidRequestError(LewLMError):
    """Raised when a request body or parameter fails validation.

    Distinct from `ConfigurationError`, which is about server settings. This
    carries per-field detail so a host application can point a user at the
    offending field instead of re-deriving it from a message string.
    """

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="invalid_request",
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            details=details,
        )


class InternalError(LewLMError):
    """Raised when an unexpected failure would otherwise escape as a bare 500.

    LewLM's contract is that every non-success response carries the same
    envelope. An unhandled exception reaching the transport as plain text
    breaks that for exactly the callers least able to diagnose it.
    """

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="internal_error",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            details=details,
        )


class StorageError(LewLMError):
    """Raised when persistence or metadata access fails."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="storage_error",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            details=details,
        )


class ModelScanError(LewLMError):
    """Raised when model discovery cannot complete."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="model_scan_error",
            status_code=HTTPStatus.BAD_REQUEST,
            details=details,
        )


class ModelNotFoundError(LewLMError):
    """Raised when a referenced model is not present in the registry."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="model_not_found",
            status_code=HTTPStatus.NOT_FOUND,
            details=details,
        )


class RoutingError(LewLMError):
    """Raised when the router cannot choose a suitable model/runtime pair."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="routing_error",
            status_code=HTTPStatus.BAD_REQUEST,
            details=details,
        )


class RuntimeUnavailableError(LewLMError):
    """Raised when a runtime backend is unavailable on the current system."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="runtime_unavailable",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            details=details,
        )


class ModelLoadError(LewLMError):
    """Raised when a backend cannot load a model on the current host.

    Distinct from `RuntimeUnavailableError`: the runtime itself is installed and
    usable, but this particular model failed to load — an unsupported
    architecture or an incompatible load option, for example. Host applications
    rely on the `model_load_failed` code to explain the failure to a user.
    """

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="model_load_failed",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            details=details,
        )


class BackendContractError(LewLMError):
    """Raised when a backend returns a result LewLM cannot safely trust.

    Distinct from `ModelLoadError`: the model ran, but its response violated the
    contract LewLM depends on — a rerank index outside the candidate range or a
    non-finite score, for example. Silently repairing such a response would
    corrupt a ranking in a way the caller can never observe, so it fails loudly.
    Bridge-backed runtimes are third-party servers LewLM does not control, which
    is exactly where this is expected to fire.
    """

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="backend_contract_violation",
            status_code=HTTPStatus.BAD_GATEWAY,
            details=details,
        )


class ModelLifecycleConflictError(LewLMError):
    """Raised when a model lifecycle action would interrupt active use."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="model_lifecycle_conflict",
            status_code=HTTPStatus.CONFLICT,
            details=details,
        )


class PackUnavailableError(LewLMError):
    """Raised when a disabled or missing pack blocks a requested surface."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="pack_unavailable",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            details=details,
        )


class UnsupportedCapabilityError(LewLMError):
    """Raised when the selected model or runtime lacks a requested capability."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="unsupported_capability",
            status_code=HTTPStatus.BAD_REQUEST,
            details=details,
        )


class IdempotencyConflictError(LewLMError):
    """Raised when an idempotency key is reused for a different request payload."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="idempotency_conflict",
            status_code=HTTPStatus.CONFLICT,
            details=details,
        )


class DocumentValidationError(LewLMError):
    """Raised when a document IR payload is invalid or incomplete."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="document_validation_error",
            status_code=HTTPStatus.BAD_REQUEST,
            details=details,
        )


class DocumentGenerationError(LewLMError):
    """Raised when an output document artifact cannot be rendered."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="document_generation_error",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            details=details,
        )


class AuthenticationError(LewLMError):
    """Raised when an API request is missing or has invalid credentials."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="authentication_error",
            status_code=HTTPStatus.UNAUTHORIZED,
            details=details,
        )


class RequestTooLargeError(LewLMError):
    """Raised when a request body exceeds configured limits."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="request_too_large",
            status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            details=details,
        )


class RateLimitError(LewLMError):
    """Raised when a client exceeds the configured request rate."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="rate_limit_error",
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
            details=details,
        )


class BackpressureError(LewLMError):
    """Raised when runtime request admission control rejects or times out a request."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="backpressure_error",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            details=details,
        )


class UnsupportedMediaTypeError(LewLMError):
    """Raised when a request or file payload uses an unsupported media type."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="unsupported_media_type",
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            details=details,
        )


class FileAccessError(LewLMError):
    """Raised when a file path falls outside the allowed local scope."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="file_access_error",
            status_code=HTTPStatus.FORBIDDEN,
            details=details,
        )


class PrivacyModeError(LewLMError):
    """Raised when a persistent feature is blocked by privacy mode."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="privacy_mode_enabled",
            status_code=HTTPStatus.FORBIDDEN,
            details=details,
        )


class ToolAuthorizationError(LewLMError):
    """Raised when an operation is not explicitly authorized."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="tool_authorization_error",
            status_code=HTTPStatus.FORBIDDEN,
            details=details,
        )


class SessionNotFoundError(LewLMError):
    """Raised when a requested session does not exist."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="session_not_found",
            status_code=HTTPStatus.NOT_FOUND,
            details=details,
        )


class SkillNotFoundError(LewLMError):
    """Raised when a requested built-in skill does not exist."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="skill_not_found",
            status_code=HTTPStatus.NOT_FOUND,
            details=details,
        )


class ToolNotFoundError(LewLMError):
    """Raised when a requested local tool does not exist."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="tool_not_found",
            status_code=HTTPStatus.NOT_FOUND,
            details=details,
        )


class SandboxExecutionError(LewLMError):
    """Raised when a sandboxed worker fails or times out."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="sandbox_execution_error",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            details=details,
        )


class JobNotFoundError(LewLMError):
    """Raised when a background job cannot be found."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="job_not_found",
            status_code=HTTPStatus.NOT_FOUND,
            details=details,
        )


class ConversionError(LewLMError):
    """Raised when a model conversion job fails."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="conversion_error",
            status_code=HTTPStatus.BAD_REQUEST,
            details=details,
        )


class NotImplementedLewLMError(LewLMError):
    """Raised when a CLI or API feature exists but is not yet implemented."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="not_implemented",
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            details=details,
        )


_ERROR_CLASS_BY_CODE: dict[str, type[LewLMError]] = {
    "configuration_error": ConfigurationError,
    "invalid_request": InvalidRequestError,
    "internal_error": InternalError,
    "model_load_failed": ModelLoadError,
    "backend_contract_violation": BackendContractError,
    "model_lifecycle_conflict": ModelLifecycleConflictError,
    "storage_error": StorageError,
    "model_scan_error": ModelScanError,
    "model_not_found": ModelNotFoundError,
    "routing_error": RoutingError,
    "runtime_unavailable": RuntimeUnavailableError,
    "pack_unavailable": PackUnavailableError,
    "unsupported_capability": UnsupportedCapabilityError,
    "idempotency_conflict": IdempotencyConflictError,
    "document_validation_error": DocumentValidationError,
    "document_generation_error": DocumentGenerationError,
    "authentication_error": AuthenticationError,
    "request_too_large": RequestTooLargeError,
    "rate_limit_error": RateLimitError,
    "backpressure_error": BackpressureError,
    "unsupported_media_type": UnsupportedMediaTypeError,
    "file_access_error": FileAccessError,
    "privacy_mode_enabled": PrivacyModeError,
    "tool_authorization_error": ToolAuthorizationError,
    "session_not_found": SessionNotFoundError,
    "skill_not_found": SkillNotFoundError,
    "tool_not_found": ToolNotFoundError,
    "sandbox_execution_error": SandboxExecutionError,
    "job_not_found": JobNotFoundError,
    "conversion_error": ConversionError,
    "not_implemented": NotImplementedLewLMError,
}


#: Codes emitted for framework-level failures that have no exception class of
#: their own — an unrouted path, a wrong method. They are declared here so the
#: published catalog covers every code a caller can actually receive, and
#: `lewlm.api.app` maps status codes through this table rather than its own.
FRAMEWORK_ERROR_CODES: dict[int, str] = {
    401: "authentication_required",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "request_too_large",
    415: "unsupported_media_type",
    429: "rate_limited",
}

_FRAMEWORK_ERROR_DESCRIPTIONS: dict[str, str] = {
    "authentication_required": "The endpoint requires an API key and none was accepted.",
    "forbidden": "The credential is valid but not authorized for this endpoint.",
    "not_found": "No route or resource matched the request path.",
    "method_not_allowed": "The route exists but does not accept this HTTP method.",
    "rate_limited": "The configured request rate for this client was exceeded.",
    "http_error": "A framework-level HTTP failure with no more specific LewLM code.",
    "lewlm_error": "An unclassified LewLM failure. Treat the message as the only detail.",
    "response_too_large": (
        "A response exceeded the client's configured size limit and was refused rather than buffered."
    ),
}

#: Codes where the identical request may succeed later without being changed.
#: Everything else needs the caller to change something first, so retrying is
#: at best wasted work and at worst a hot loop against a failing host.
_RETRYABLE_ERROR_CODES = frozenset(
    {
        "rate_limit_error",
        "rate_limited",
        "backpressure_error",
        "runtime_unavailable",
        "model_lifecycle_conflict",
        "storage_error",
    },
)


def error_code_catalog() -> list[dict[str, Any]]:
    """Return every error code the API can emit, with status and retryability.

    Derived from the exception classes themselves — status codes from their
    constructors, descriptions from their docstrings — so the catalog cannot
    drift from the behaviour, and a host app never has to scrape this module.
    """

    entries: dict[str, dict[str, Any]] = {}
    for error_type in _error_classes():
        probe = error_type("catalog probe")
        entries[probe.code] = {
            "code": probe.code,
            "http_status": probe.status_code,
            "retryable": probe.code in _RETRYABLE_ERROR_CODES,
            "description": _summarize_docstring(error_type),
        }
    status_by_framework_code = {code: status for status, code in FRAMEWORK_ERROR_CODES.items()}
    for code, description in _FRAMEWORK_ERROR_DESCRIPTIONS.items():
        if code in entries:
            continue
        entries[code] = {
            "code": code,
            "http_status": status_by_framework_code.get(code, _DEFAULT_FRAMEWORK_STATUS.get(code, 400)),
            "retryable": code in _RETRYABLE_ERROR_CODES,
            "description": description,
        }
    return [entries[code] for code in sorted(entries)]


#: Statuses for catalog codes that are not raised through `FRAMEWORK_ERROR_CODES`.
_DEFAULT_FRAMEWORK_STATUS = {
    "http_error": int(HTTPStatus.INTERNAL_SERVER_ERROR),
    "lewlm_error": int(HTTPStatus.BAD_REQUEST),
    "response_too_large": int(HTTPStatus.INSUFFICIENT_STORAGE),
}


def _error_classes() -> list[type[LewLMError]]:
    """Every concrete error class declared in this module, base class included."""

    return [LewLMError, *sorted(_subclasses(LewLMError), key=lambda item: item.__name__)]


def _subclasses(root: type[LewLMError]) -> set[type[LewLMError]]:
    found: set[type[LewLMError]] = set()
    for subclass in root.__subclasses__():
        if subclass.__module__ != __name__:
            continue
        found.add(subclass)
        found |= _subclasses(subclass)
    return found


def _summarize_docstring(error_type: type[LewLMError]) -> str:
    doc = (error_type.__doc__ or "").strip()
    if not doc:
        return f"Raised as `{error_type.__name__}`."
    return " ".join(doc.split("\n\n", 1)[0].split())


def error_from_dict(payload: Mapping[str, Any]) -> LewLMError:
    """Rehydrate a structured LewLM error payload into an exception."""

    message = str(payload.get("message") or "LewLM error")
    code = str(payload.get("code") or "lewlm_error")
    details_raw = payload.get("details")
    details = dict(details_raw) if isinstance(details_raw, Mapping) else {}
    error_type = _ERROR_CLASS_BY_CODE.get(code)
    if error_type is not None:
        return error_type(message, details=details)
    status_code = payload.get("status_code")
    resolved_status_code = int(status_code) if isinstance(status_code, int) else int(HTTPStatus.BAD_REQUEST)
    return LewLMError(message, code=code, status_code=resolved_status_code, details=details)
