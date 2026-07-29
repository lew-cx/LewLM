"""Untrusted application identity propagated for operational observability."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass


application_id_var: ContextVar[str | None] = ContextVar("lewlm_application_id", default=None)
client_instance_id_var: ContextVar[str | None] = ContextVar("lewlm_client_instance_id", default=None)
#: Caller-supplied correlation identifier. LewLM never generates one — an absent
#: value stays absent so a caller can tell its own ID from a LewLM request ID.
correlation_id_var: ContextVar[str | None] = ContextVar("lewlm_correlation_id", default=None)

#: Bound so a hostile or buggy caller cannot push unbounded strings into every
#: log line, event payload, and metadata envelope.
MAX_CORRELATION_ID_CHARACTERS = 128


def normalize_correlation_id(value: str | None) -> str | None:
    """Trim and bound an untrusted correlation identifier."""

    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    return trimmed[:MAX_CORRELATION_ID_CHARACTERS]


@dataclass(frozen=True, slots=True)
class ApplicationContextTokens:
    application_id: Token[str | None]
    client_instance_id: Token[str | None]
    correlation_id: Token[str | None]


def set_application_context(
    *,
    application_id: str | None,
    client_instance_id: str | None,
    correlation_id: str | None = None,
) -> ApplicationContextTokens:
    return ApplicationContextTokens(
        application_id=application_id_var.set(application_id),
        client_instance_id=client_instance_id_var.set(client_instance_id),
        correlation_id=correlation_id_var.set(normalize_correlation_id(correlation_id)),
    )


def reset_application_context(tokens: ApplicationContextTokens) -> None:
    application_id_var.reset(tokens.application_id)
    client_instance_id_var.reset(tokens.client_instance_id)
    correlation_id_var.reset(tokens.correlation_id)


def current_correlation_id() -> str | None:
    """The correlation ID for the request being served, if the caller sent one."""

    return correlation_id_var.get()


def apply_body_correlation_id(correlation_id: str | None) -> None:
    """Adopt a correlation ID supplied in a request body.

    Middleware applies the header before the body is parsed, so a request that
    carries the ID in its payload instead needs this. An explicit header wins,
    being the transport-level identity. No reset token is returned: the value
    stays set for the rest of the request — including a streaming response that
    outlives the route function — and `reset_application_context` restores the
    pre-request value when the middleware unwinds.
    """

    normalized = normalize_correlation_id(correlation_id)
    if normalized is None or correlation_id_var.get() is not None:
        return
    correlation_id_var.set(normalized)


@contextmanager
def correlation_scope(correlation_id: str | None) -> Iterator[None]:
    """Apply a correlation ID for the duration of a block, then restore it."""

    normalized = normalize_correlation_id(correlation_id)
    if normalized is None:
        yield
        return
    token = correlation_id_var.set(normalized)
    try:
        yield
    finally:
        correlation_id_var.reset(token)
