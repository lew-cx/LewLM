"""Externally addressable cancellation handles for in-flight requests.

A caller that owns a request only while it is awaiting its own HTTP response can
cancel it by dropping the connection. An orchestrator in a *different* process
cannot: it has no task, no socket, and no handle. This module gives every served
request a stable, caller-supplied identity (`x-request-id`) that a third party
can cancel by name.

Cancellation is cooperative and best-effort, and the contract says so: LewLM
stops the request at its next cancellation checkpoint — admission control, a
tool-execution entry, a per-source ingestion step. Work already committed to a
single bounded backend call runs to completion, and no claim is made that a
model call is interrupted mid-flight.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import hmac
import re
import secrets
import threading

from pydantic import BaseModel

from lewlm.core.contracts import utc_now
from lewlm.core.errors import InvalidRequestError, RequestCancelledError, RequestHandleConflictError, ToolAuthorizationError

#: How long an unmatched cancellation intent is honoured. A cancel can legitimately
#: arrive before the request it targets — the orchestrator and the worker are
#: separate processes — so an unknown handle is remembered rather than discarded.
DEFAULT_CANCELLATION_INTENT_TTL_SECONDS = 300.0

#: Bound on remembered handles, applied separately to pending intents and to
#: terminal records, so a caller cannot grow this registry without limit.
DEFAULT_MAX_TRACKED_REQUESTS = 512

# Handles travel in both an HTTP header and a URL path. Keeping their alphabet
# path-safe avoids multiple spellings of one identity (or query/fragment
# truncation) between the request that registers it and the call that cancels it.
MAX_REQUEST_HANDLE_CHARACTERS = 128
_REQUEST_HANDLE_PATTERN = re.compile(rf"[A-Za-z0-9._~-]{{1,{MAX_REQUEST_HANDLE_CHARACTERS}}}\Z")


def validate_request_handle(request_id: str) -> str:
    """Return one canonical, URL-safe request handle or raise a typed 422."""

    if not isinstance(request_id, str) or _REQUEST_HANDLE_PATTERN.fullmatch(request_id) is None:
        raise InvalidRequestError(
            "Request handles must be 1-128 URL-safe characters.",
            details={
                "field": "x-request-id",
                "allowed_characters": "A-Z a-z 0-9 . _ ~ -",
                "max_characters": MAX_REQUEST_HANDLE_CHARACTERS,
            },
        )
    return request_id


def _resolve_cancellation_waiter(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


class RequestCancellationState(str, Enum):
    """What LewLM knows about one request handle."""

    #: An active request with this handle was signalled; it has not stopped yet.
    CANCELLING = "cancelling"
    #: No request with this handle is active or remembered. The intent is held
    #: until `expires_at`, and a request that arrives with it stops immediately.
    PENDING = "pending"
    #: The request observed the signal and stopped at a checkpoint.
    CANCELLED = "cancelled"
    #: The request reached a terminal state without observing the signal.
    COMPLETED = "completed"


class RequestCancellationRecord(BaseModel):
    """Acknowledgement returned for a cancellation request.

    The same handle can be cancelled repeatedly; each call returns the current
    state rather than failing, so this doubles as the way to observe whether the
    target actually stopped.
    """

    request_id: str
    state: RequestCancellationState
    runtime_instance_id: str
    application_id: str | None = None
    #: When cancellation was first requested for this handle.
    requested_at: datetime | None = None
    #: Set only while the state is `pending`: after this, the intent is dropped.
    expires_at: datetime | None = None
    #: When the tracked request reached a terminal state.
    completed_at: datetime | None = None


@dataclass(slots=True)
class CancellationToken:
    """The in-flight side of one handle, shared by every stage of the request."""

    request_id: str
    application_id: str | None = None
    credential_fingerprint: str | None = None
    requested_at: datetime | None = None
    _cancelled: threading.Event = field(default_factory=threading.Event)
    _observed: threading.Event = field(default_factory=threading.Event)
    _waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = field(default_factory=list)
    _waiters_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def observed(self) -> bool:
        """Whether a checkpoint actually acted on the cancellation signal."""

        return self._observed.is_set()

    def cancel(self) -> datetime:
        """Signal the request. Repeated calls keep the first requested time."""

        if self.requested_at is None:
            self.requested_at = utc_now()
        self._cancelled.set()
        with self._waiters_lock:
            waiters = tuple(self._waiters)
            self._waiters.clear()
        for loop, future in waiters:
            loop.call_soon_threadsafe(_resolve_cancellation_waiter, future)
        return self.requested_at

    async def wait_cancelled(self) -> None:
        """Wait without polling until this token is signalled."""

        if self.cancelled:
            return
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        entry = (loop, future)
        with self._waiters_lock:
            if self.cancelled:
                future.set_result(None)
            else:
                self._waiters.append(entry)
        try:
            await future
        finally:
            with self._waiters_lock:
                if entry in self._waiters:
                    self._waiters.remove(entry)

    def raise_if_cancelled(self, *, stage: str | None = None) -> None:
        """Stop the request here if its handle has been cancelled."""

        if not self.cancelled:
            return
        self._observed.set()
        details: dict[str, str] = {"request_id": self.request_id}
        if stage is not None:
            details["stage"] = stage
        raise RequestCancelledError(
            "The request was cancelled through its cancellation handle.",
            details=details,
        )

    def observe_if_cancelled(self) -> bool:
        """Record a non-raising streaming checkpoint and return its state."""

        if not self.cancelled:
            return False
        self._observed.set()
        return True


#: The handle of the request being served on this task/thread, if it has one.
#: Set once per request and read by every checkpoint, so a checkpoint never has
#: to be threaded through a call signature.
cancellation_token_var: ContextVar[CancellationToken | None] = ContextVar(
    "lewlm_cancellation_token",
    default=None,
)


def current_cancellation_token() -> CancellationToken | None:
    return cancellation_token_var.get()


def request_cancelled() -> bool:
    """Whether the request being served has been cancelled, without raising.

    For streaming paths, where stopping means closing the stream cleanly rather
    than raising through a response that has already begun.
    """

    token = cancellation_token_var.get()
    return token is not None and token.observe_if_cancelled()


def raise_if_request_cancelled(*, stage: str | None = None) -> None:
    """Cancellation checkpoint. A no-op for requests without a handle."""

    token = cancellation_token_var.get()
    if token is not None:
        token.raise_if_cancelled(stage=stage)


@dataclass(slots=True)
class _PendingIntent:
    application_id: str | None
    credential_fingerprint: str | None
    requested_at: datetime
    expires_at: datetime
    trusted_owner_override: bool = False


class RequestCancellationRegistry:
    """Process-local index of cancellable request handles.

    One registry belongs to one runtime service container, so a handle is only
    addressable on the LewLM process that is serving it. A deployment behind a
    load balancer must therefore cancel against the instance that accepted the
    request; `runtime_instance_id` on every record identifies it.
    """

    def __init__(
        self,
        *,
        runtime_instance_id: str,
        intent_ttl_seconds: float = DEFAULT_CANCELLATION_INTENT_TTL_SECONDS,
        max_tracked_requests: int = DEFAULT_MAX_TRACKED_REQUESTS,
    ) -> None:
        self.runtime_instance_id = runtime_instance_id
        self.intent_ttl_seconds = max(1.0, float(intent_ttl_seconds))
        self.max_tracked_requests = max(1, int(max_tracked_requests))
        # Guarded by a plain lock rather than an asyncio one: every operation is
        # short and non-blocking, and checkpoints run on worker threads as well
        # as on the event loop.
        self._lock = threading.Lock()
        self._credential_salt = secrets.token_bytes(32)
        self._active: dict[str, CancellationToken] = {}
        self._pending: OrderedDict[str, _PendingIntent] = OrderedDict()
        self._terminal: OrderedDict[str, RequestCancellationRecord] = OrderedDict()
        self._terminal_owners: dict[str, tuple[str | None, str | None]] = {}

    @property
    def active_request_count(self) -> int:
        with self._lock:
            return len(self._active)

    @contextmanager
    def track(
        self,
        request_id: str,
        *,
        application_id: str | None = None,
        credential: str | None = None,
    ) -> Iterator[CancellationToken]:
        """Make one request addressable by its handle for the duration of the block.

        A handle that was already cancelled yields a token that is cancelled from
        the start, so the first checkpoint stops the request before any work.
        """

        request_id = validate_request_handle(request_id)
        credential_fingerprint = self._credential_fingerprint(credential)
        token = CancellationToken(
            request_id=request_id,
            application_id=application_id,
            credential_fingerprint=credential_fingerprint,
        )
        with self._lock:
            self._expire_intents_locked()
            if request_id in self._active:
                raise RequestHandleConflictError(
                    "The request handle is already active.",
                    details={"request_id": request_id},
                )
            # A caller may reuse a trace identifier after its earlier request is
            # terminal. The new live request becomes the meaning of the handle;
            # only concurrent reuse is ambiguous and therefore rejected.
            if request_id in self._terminal:
                del self._terminal[request_id]
                self._terminal_owners.pop(request_id, None)
            intent = self._pending.get(request_id)
            if intent is not None:
                if not intent.trusted_owner_override and not self._owners_match(
                    intent.application_id,
                    intent.credential_fingerprint,
                    application_id,
                    credential_fingerprint,
                ):
                    raise RequestHandleConflictError(
                        "The request handle is reserved by another cancellation owner.",
                        details={"request_id": request_id},
                    )
                del self._pending[request_id]
                token.requested_at = intent.requested_at
                token.cancel()
            self._active[request_id] = token
        reset_token = cancellation_token_var.set(token)
        try:
            yield token
        finally:
            cancellation_token_var.reset(reset_token)
            self._release(token)

    def cancel(
        self,
        request_id: str,
        *,
        application_id: str | None = None,
        credential: str | None = None,
        trusted_caller: bool = False,
    ) -> RequestCancellationRecord:
        """Cancel a handle, whether or not its request has arrived. Idempotent.

        In an authenticated deployment, a handle may only be cancelled by the
        API credential that issued it. Application IDs remain observability
        metadata and never grant authority. In an open deployment, possession
        of the high-entropy handle is the only available capability boundary.
        `trusted_caller` waives the credential check for an in-process host
        embedding LewLM: it already owns the process.
        """

        request_id = validate_request_handle(request_id)
        credential_fingerprint = self._credential_fingerprint(credential)
        with self._lock:
            self._expire_intents_locked()
            token = self._active.get(request_id)
            if token is not None:
                self._require_owner(
                    token.application_id,
                    token.credential_fingerprint,
                    application_id,
                    credential_fingerprint,
                    request_id=request_id,
                    trusted_caller=trusted_caller,
                )
                requested_at = token.cancel()
                return RequestCancellationRecord(
                    request_id=request_id,
                    state=RequestCancellationState.CANCELLING,
                    runtime_instance_id=self.runtime_instance_id,
                    application_id=token.application_id,
                    requested_at=requested_at,
                )
            terminal = self._terminal.get(request_id)
            if terminal is not None:
                self._require_owner(
                    terminal.application_id,
                    self._terminal_owners.get(request_id, (terminal.application_id, None))[1],
                    application_id,
                    credential_fingerprint,
                    request_id=request_id,
                    trusted_caller=trusted_caller,
                )
                if terminal.requested_at is None:
                    terminal = terminal.model_copy(update={"requested_at": utc_now()})
                    self._terminal[request_id] = terminal
                return terminal.model_copy(deep=True)
            existing = self._pending.get(request_id)
            if existing is not None:
                self._require_owner(
                    existing.application_id,
                    existing.credential_fingerprint,
                    application_id,
                    credential_fingerprint,
                    request_id=request_id,
                    trusted_caller=trusted_caller,
                )
                requested_at = existing.requested_at
                expires_at = existing.expires_at
                # Idempotent repeat calls must not rewrite ownership. In
                # particular, an embedded trusted caller has no API credential;
                # replacing the fingerprint with None would lock out the
                # authenticated request when it eventually arrives.
                pending_application_id = existing.application_id
                pending_credential_fingerprint = existing.credential_fingerprint
                pending_trusted_owner_override = existing.trusted_owner_override
            else:
                requested_at = utc_now()
                expires_at = requested_at + timedelta(seconds=self.intent_ttl_seconds)
                pending_application_id = application_id
                pending_credential_fingerprint = credential_fingerprint
                # An in-process host owns the runtime but has no HTTP credential
                # to bind here. Let the eventual request supply its authenticated
                # owner, then retain that owner on the terminal record.
                pending_trusted_owner_override = trusted_caller
            self._pending[request_id] = _PendingIntent(
                application_id=pending_application_id,
                credential_fingerprint=pending_credential_fingerprint,
                requested_at=requested_at,
                expires_at=expires_at,
                trusted_owner_override=pending_trusted_owner_override,
            )
            self._pending.move_to_end(request_id)
            self._evict_locked(self._pending)
            return RequestCancellationRecord(
                request_id=request_id,
                state=RequestCancellationState.PENDING,
                runtime_instance_id=self.runtime_instance_id,
                application_id=pending_application_id,
                requested_at=requested_at,
                expires_at=expires_at,
            )

    def _release(self, token: CancellationToken) -> None:
        completed_at = utc_now()
        with self._lock:
            if self._active.get(token.request_id) is token:
                del self._active[token.request_id]
            self._terminal[token.request_id] = RequestCancellationRecord(
                request_id=token.request_id,
                state=(
                    RequestCancellationState.CANCELLED
                    if token.observed
                    else RequestCancellationState.COMPLETED
                ),
                runtime_instance_id=self.runtime_instance_id,
                application_id=token.application_id,
                requested_at=token.requested_at,
                completed_at=completed_at,
            )
            self._terminal_owners[token.request_id] = (
                token.application_id,
                token.credential_fingerprint,
            )
            self._terminal.move_to_end(token.request_id)
            self._evict_terminal_locked()

    def _require_owner(
        self,
        owner: str | None,
        owner_credential_fingerprint: str | None,
        caller: str | None,
        caller_credential_fingerprint: str | None,
        *,
        request_id: str,
        trusted_caller: bool = False,
    ) -> None:
        if trusted_caller or self._owners_match(
            owner,
            owner_credential_fingerprint,
            caller,
            caller_credential_fingerprint,
        ):
            return
        raise ToolAuthorizationError(
            "The request handle belongs to a different authenticated caller.",
            details={"request_id": request_id},
        )

    def _credential_fingerprint(self, credential: str | None) -> str | None:
        if credential is None:
            return None
        return hmac.new(self._credential_salt, credential.encode("utf-8"), hashlib.sha256).hexdigest()

    @staticmethod
    def _owners_match(
        owner_application_id: str | None,
        owner_credential_fingerprint: str | None,
        caller_application_id: str | None,
        caller_credential_fingerprint: str | None,
    ) -> bool:
        # Application IDs are caller-supplied observability metadata, not an
        # authentication primitive. Keep the parameters explicit to make that
        # distinction visible at every call site while comparing only secrets.
        del owner_application_id, caller_application_id
        if owner_credential_fingerprint is None or caller_credential_fingerprint is None:
            return owner_credential_fingerprint is caller_credential_fingerprint
        return hmac.compare_digest(owner_credential_fingerprint, caller_credential_fingerprint)

    def _expire_intents_locked(self) -> None:
        if not self._pending:
            return
        now = utc_now()
        for request_id, intent in tuple(self._pending.items()):
            if intent.expires_at <= now:
                del self._pending[request_id]

    def _evict_locked(self, entries: OrderedDict) -> None:
        while len(entries) > self.max_tracked_requests:
            entries.popitem(last=False)

    def _evict_terminal_locked(self) -> None:
        while len(self._terminal) > self.max_tracked_requests:
            request_id, _ = self._terminal.popitem(last=False)
            self._terminal_owners.pop(request_id, None)
