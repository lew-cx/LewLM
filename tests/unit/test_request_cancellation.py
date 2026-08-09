"""Coverage for externally addressable request cancellation handles."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from lewlm.core.contracts import utc_now
from lewlm.core.errors import InvalidRequestError, RequestCancelledError, RequestHandleConflictError, ToolAuthorizationError
from lewlm.runtime import cancellation
from lewlm.runtime.cancellation import (
    RequestCancellationRegistry,
    RequestCancellationState,
    raise_if_request_cancelled,
    request_cancelled,
)
from lewlm.runtime.scheduler import RuntimeRequestScheduler


def _registry(**overrides) -> RequestCancellationRegistry:
    return RequestCancellationRegistry(runtime_instance_id="runtime-1", **overrides)


# --- registry ----------------------------------------------------------------


def test_cancelling_an_active_handle_signals_its_checkpoints() -> None:
    registry = _registry()

    with registry.track("req-1", application_id="docktizo"):
        record = registry.cancel("req-1", application_id="docktizo")

        assert record.state is RequestCancellationState.CANCELLING
        assert record.runtime_instance_id == "runtime-1"
        assert record.requested_at is not None
        assert request_cancelled() is True
        with pytest.raises(RequestCancelledError) as exc_info:
            raise_if_request_cancelled(stage="runtime_admission")

    assert exc_info.value.status_code == 499
    assert exc_info.value.code == "request_cancelled"
    assert exc_info.value.details == {"request_id": "req-1", "stage": "runtime_admission"}


def test_a_handle_cancelled_before_it_arrives_stops_the_request_on_arrival() -> None:
    registry = _registry()

    pending = registry.cancel("req-2", application_id="docktizo")
    with registry.track("req-2", application_id="docktizo") as token:
        # The worker process may only issue the request after the orchestrator
        # has already given up on it.
        assert token.cancelled is True
        with pytest.raises(RequestCancelledError):
            raise_if_request_cancelled()

    assert pending.state is RequestCancellationState.PENDING
    assert pending.expires_at is not None
    assert pending.expires_at > pending.requested_at
    assert registry.cancel("req-2", application_id="docktizo").state is RequestCancellationState.CANCELLED


def test_a_request_that_finished_first_reports_completed_not_cancelled() -> None:
    registry = _registry()

    with registry.track("req-3", application_id="docktizo"):
        pass

    record = registry.cancel("req-3", application_id="docktizo")

    assert record.state is RequestCancellationState.COMPLETED
    assert record.completed_at is not None
    # Idempotent: asking again reports the same terminal outcome.
    assert registry.cancel("req-3", application_id="docktizo").state is RequestCancellationState.COMPLETED


def test_repeated_cancellation_keeps_the_first_requested_time() -> None:
    registry = _registry()

    with registry.track("req-4", application_id="docktizo"):
        first = registry.cancel("req-4", application_id="docktizo")
        second = registry.cancel("req-4", application_id="docktizo")

    assert second.requested_at == first.requested_at


def test_a_signal_not_observed_before_completion_reports_completed() -> None:
    registry = _registry()

    with registry.track("req-too-late", application_id="docktizo"):
        registry.cancel("req-too-late", application_id="docktizo")
        # Simulates cancellation landing after the final checkpoint: the signal
        # was delivered, but it did not stop the request.

    record = registry.cancel("req-too-late", application_id="docktizo")

    assert record.state is RequestCancellationState.COMPLETED
    assert record.requested_at is not None
    assert record.completed_at is not None


def test_an_expired_intent_no_longer_cancels_a_later_request(monkeypatch) -> None:
    registry = _registry(intent_ttl_seconds=1)
    registry.cancel("req-5", application_id="docktizo")

    monkeypatch.setattr(cancellation, "utc_now", lambda: utc_now() + timedelta(seconds=5))
    with registry.track("req-5", application_id="docktizo") as token:
        assert token.cancelled is False


def test_application_metadata_is_not_used_as_authorization() -> None:
    registry = _registry()

    with registry.track("req-6", application_id="docktizo") as token:
        record = registry.cancel("req-6", application_id="another-label")

    assert record.state is RequestCancellationState.CANCELLING
    assert token.cancelled is True


def test_a_pending_intent_is_matched_by_credential_not_application_metadata() -> None:
    registry = _registry()
    registry.cancel("req-7", application_id="docktizo", credential="owner-key")

    with pytest.raises(RequestHandleConflictError) as exc_info:
        with registry.track("req-7", application_id="docktizo", credential="intruder-key"):
            pass

    assert exc_info.value.status_code == 409
    # The original intent is still present and stops its owner's request.
    with registry.track("req-7", application_id="renamed-app", credential="owner-key") as token:
        assert token.cancelled is True


def test_a_concurrent_duplicate_handle_does_not_replace_the_original_request() -> None:
    registry = _registry()

    with registry.track("req-duplicate", application_id="docktizo") as original:
        with pytest.raises(RequestHandleConflictError) as exc_info:
            with registry.track("req-duplicate", application_id="docktizo"):
                pass
        registry.cancel("req-duplicate", application_id="docktizo")

        assert exc_info.value.status_code == 409
        assert original.cancelled is True
        assert registry.active_request_count == 1


def test_application_metadata_cannot_cross_a_credential_boundary() -> None:
    registry = _registry()

    with registry.track("req-secret", application_id="docktizo", credential="owner-key") as token:
        with pytest.raises(ToolAuthorizationError):
            registry.cancel(
                "req-secret",
                application_id="docktizo",
                credential="intruder-key",
            )
        registry.cancel("req-secret", application_id="docktizo", credential="owner-key")

    assert token.cancelled is True
    assert "owner-key" not in repr(registry._terminal_owners)


def test_a_trusted_repeat_preserves_a_pending_intents_authenticated_owner() -> None:
    registry = _registry()
    registry.cancel("req-pending-owner", credential="owner-key")

    registry.cancel("req-pending-owner", trusted_caller=True)

    with registry.track("req-pending-owner", credential="owner-key") as token:
        assert token.cancelled is True


def test_a_trusted_prearrival_intent_binds_to_the_authenticated_request_on_arrival() -> None:
    registry = _registry()
    pending = registry.cancel("req-trusted-future", trusted_caller=True)

    with registry.track("req-trusted-future", credential="owner-key") as token:
        assert token.cancelled is True
        assert request_cancelled() is True

    assert pending.state is RequestCancellationState.PENDING
    with pytest.raises(ToolAuthorizationError):
        registry.cancel("req-trusted-future", credential="intruder-key")
    assert (
        registry.cancel("req-trusted-future", credential="owner-key").state
        is RequestCancellationState.CANCELLED
    )


@pytest.mark.parametrize("request_id", ["", " leading", "trailing ", "has/slash", "has?query", "x" * 129])
def test_request_handles_have_one_bounded_url_safe_spelling(request_id: str) -> None:
    registry = _registry()

    with pytest.raises(InvalidRequestError):
        registry.cancel(request_id)


def test_an_in_process_host_may_cancel_a_handle_it_does_not_own() -> None:
    registry = _registry()

    with registry.track("req-8", application_id="docktizo") as token:
        record = registry.cancel("req-8", trusted_caller=True)

    assert record.state is RequestCancellationState.CANCELLING
    assert token.cancelled is True


def test_remembered_handles_stay_bounded() -> None:
    registry = _registry(max_tracked_requests=2)

    for index in range(5):
        registry.cancel(f"pending-{index}")
        with registry.track(f"done-{index}"):
            pass

    assert len(registry._pending) == 2
    assert len(registry._terminal) == 2
    # The oldest handles are the ones dropped.
    assert registry.cancel("done-0").state is RequestCancellationState.PENDING


def test_concurrent_requests_are_tracked_independently() -> None:
    registry = _registry()

    with registry.track("req-a", application_id="docktizo") as first:
        with registry.track("req-b", application_id="docktizo") as second:
            registry.cancel("req-b", application_id="docktizo")
            assert registry.active_request_count == 2
        assert first.cancelled is False
        assert second.cancelled is True


def test_a_request_without_a_handle_never_hits_a_checkpoint() -> None:
    assert request_cancelled() is False
    raise_if_request_cancelled(stage="runtime_admission")


# --- admission control -------------------------------------------------------


async def test_a_request_cancelled_before_admission_never_reaches_the_runtime() -> None:
    registry = _registry()
    scheduler = RuntimeRequestScheduler(max_concurrent_requests=1, queue_limit=4, queue_timeout_seconds=5)

    with registry.track("req-admission", application_id="docktizo"):
        registry.cancel("req-admission", application_id="docktizo")
        with pytest.raises(RequestCancelledError):
            await scheduler.acquire()

    assert scheduler.snapshot()["active_requests"] == 0


async def test_a_queued_request_cancelled_while_waiting_hands_its_slot_back() -> None:
    registry = _registry()
    scheduler = RuntimeRequestScheduler(max_concurrent_requests=1, queue_limit=4, queue_timeout_seconds=5)
    holder = await scheduler.acquire()

    async def queued() -> None:
        with registry.track("req-queued", application_id="docktizo"):
            with pytest.raises(RequestCancelledError):
                await scheduler.acquire()

    waiter = asyncio.create_task(queued())
    while scheduler.snapshot()["queued_requests"] == 0:
        await asyncio.sleep(0)
    registry.cancel("req-queued", application_id="docktizo")

    # Cancellation wakes the queued request itself; it does not wait for the
    # holder to finish and grant it a slot first.
    await asyncio.wait_for(waiter, timeout=0.5)
    snapshot = scheduler.snapshot()
    assert snapshot["queued_requests"] == 0
    assert snapshot["active_requests"] == 1

    holder.release()

    assert scheduler.snapshot()["active_requests"] == 0
