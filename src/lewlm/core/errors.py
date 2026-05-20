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
