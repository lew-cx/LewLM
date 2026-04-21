"""Runtime request admission control, queue shaping, and backpressure tracking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import Lock
import time
from typing import Any, Generic, Literal, TypeVar

from lewlm.core.errors import BackpressureError

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


@dataclass(slots=True)
class RuntimeRequestAdmission:
    scheduler: "RuntimeRequestScheduler"
    was_queued: bool
    wait_seconds: float
    scheduling_lane: Literal["decode", "prefill"] = "decode"
    decode_priority_applied: bool = False
    prefill_isolated: bool = False
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self.scheduler.release(
            scheduling_lane=self.scheduling_lane,
            prefill_isolated=self.prefill_isolated,
        )
        self._released = True


@dataclass(slots=True)
class _GrantedRuntimeRequest:
    decode_priority_applied: bool


@dataclass(slots=True)
class _QueuedRuntimeRequest:
    future: asyncio.Future[_GrantedRuntimeRequest]
    enqueued_at: float
    scheduling_lane: Literal["decode", "prefill"]
    prefill_isolated: bool


class RuntimeRequestScheduler:
    """Control concurrent runtime work with priority-aware in-memory queueing."""

    def __init__(
        self,
        *,
        max_concurrent_requests: int,
        queue_limit: int,
        queue_timeout_seconds: int,
        decode_priority_enabled: bool = False,
        long_prefill_token_threshold: int = 0,
        prefill_isolation_enabled: bool = False,
        prefill_isolation_max_concurrent_requests: int = 1,
        prefill_isolation_decode_reserve: int = 1,
    ) -> None:
        self.max_concurrent_requests = max(0, max_concurrent_requests)
        self.queue_limit = max(0, queue_limit)
        self.queue_timeout_seconds = max(1, queue_timeout_seconds)
        self.decode_priority_enabled = decode_priority_enabled
        self.long_prefill_token_threshold = max(0, long_prefill_token_threshold)
        self.prefill_isolation_enabled = prefill_isolation_enabled
        self.prefill_isolation_max_concurrent_requests = max(1, prefill_isolation_max_concurrent_requests)
        self.prefill_isolation_decode_reserve = max(0, prefill_isolation_decode_reserve)
        self._lock = Lock()
        self._waiters: list[_QueuedRuntimeRequest] = []
        self._active_requests = 0
        self._queued_requests = 0
        self._active_decode_requests = 0
        self._active_prefill_requests = 0
        self._queued_decode_requests = 0
        self._queued_prefill_requests = 0
        self._peak_active_requests = 0
        self._max_observed_queue_depth = 0
        self._max_observed_decode_queue_depth = 0
        self._max_observed_prefill_queue_depth = 0
        self._total_queued_requests = 0
        self._rejected_requests = 0
        self._timed_out_requests = 0
        self._total_queue_wait_seconds = 0.0
        self._max_queue_wait_seconds = 0.0
        self._decode_priority_request_count = 0
        self._prefill_request_count = 0
        self._prioritized_decode_grants = 0
        self._isolated_prefill_requests = 0
        self._frontier_total_batches = 0
        self._frontier_total_requests = 0
        self._frontier_batched_requests = 0
        self._frontier_coalesced_requests = 0
        self._frontier_total_queue_delay_seconds = 0.0
        self._frontier_max_queue_delay_seconds = 0.0
        self._frontier_total_batch_size = 0
        self._frontier_total_batch_utilization = 0.0
        self._frontier_window_milliseconds = 0
        self._frontier_max_batch_size = 0

    async def acquire(
        self,
        *,
        prefill_heavy: bool = False,
        decode_priority: bool = False,
        prefill_isolation: bool = False,
    ) -> RuntimeRequestAdmission:
        scheduling_lane: Literal["decode", "prefill"] = "prefill" if prefill_heavy else "decode"
        decode_priority_requested = self.decode_priority_enabled and scheduling_lane == "decode" and decode_priority
        prefill_isolated = self._prefill_isolation_active(
            scheduling_lane=scheduling_lane,
            prefill_isolation=prefill_isolation,
        )
        with self._lock:
            if decode_priority_requested:
                self._decode_priority_request_count += 1
            if scheduling_lane == "prefill":
                self._prefill_request_count += 1
                if prefill_isolated:
                    self._isolated_prefill_requests += 1
        if self.max_concurrent_requests <= 0:
            with self._lock:
                self._grant_slot_locked(
                    scheduling_lane=scheduling_lane,
                    prefill_isolated=prefill_isolated,
                )
            return RuntimeRequestAdmission(
                scheduler=self,
                was_queued=False,
                wait_seconds=0.0,
                scheduling_lane=scheduling_lane,
                prefill_isolated=prefill_isolated,
            )
        immediate_grant: _GrantedRuntimeRequest | None = None
        waiter: _QueuedRuntimeRequest | None = None
        with self._lock:
            if self._can_grant_locked(
                scheduling_lane=scheduling_lane,
                prefill_isolated=prefill_isolated,
            ) and not self._higher_priority_waiter_exists_locked(scheduling_lane):
                self._grant_slot_locked(
                    scheduling_lane=scheduling_lane,
                    prefill_isolated=prefill_isolated,
                )
                immediate_grant = _GrantedRuntimeRequest(decode_priority_applied=False)
            else:
                if self._queued_requests >= self.queue_limit:
                    self._rejected_requests += 1
                    raise BackpressureError(
                        "Runtime request queue is full.",
                        details={
                            "max_concurrent_runtime_requests": self.max_concurrent_requests,
                            "runtime_request_queue_limit": self.queue_limit,
                            "runtime_request_queue_timeout_seconds": self.queue_timeout_seconds,
                        },
                    )
                loop = asyncio.get_running_loop()
                waiter = _QueuedRuntimeRequest(
                    future=loop.create_future(),
                    enqueued_at=time.perf_counter(),
                    scheduling_lane=scheduling_lane,
                    prefill_isolated=prefill_isolated,
                )
                self._waiters.append(waiter)
                self._queued_requests += 1
                self._total_queued_requests += 1
                self._max_observed_queue_depth = max(self._max_observed_queue_depth, self._queued_requests)
                if scheduling_lane == "decode":
                    self._queued_decode_requests += 1
                    self._max_observed_decode_queue_depth = max(
                        self._max_observed_decode_queue_depth,
                        self._queued_decode_requests,
                    )
                else:
                    self._queued_prefill_requests += 1
                    self._max_observed_prefill_queue_depth = max(
                        self._max_observed_prefill_queue_depth,
                        self._queued_prefill_requests,
                    )
        if immediate_grant is not None:
            return RuntimeRequestAdmission(
                scheduler=self,
                was_queued=False,
                wait_seconds=0.0,
                scheduling_lane=scheduling_lane,
                decode_priority_applied=immediate_grant.decode_priority_applied,
                prefill_isolated=prefill_isolated,
            )
        if waiter is None:
            raise AssertionError("Runtime scheduler waiter was not created.")
        try:
            granted = await asyncio.wait_for(waiter.future, timeout=self.queue_timeout_seconds)
        except asyncio.CancelledError:
            with self._lock:
                self._remove_waiter_locked(waiter)
            raise
        except asyncio.TimeoutError as exc:
            with self._lock:
                self._remove_waiter_locked(waiter)
                self._timed_out_requests += 1
            raise BackpressureError(
                "Runtime request queue wait timed out.",
                details={
                    "max_concurrent_runtime_requests": self.max_concurrent_requests,
                    "runtime_request_queue_limit": self.queue_limit,
                    "runtime_request_queue_timeout_seconds": self.queue_timeout_seconds,
                },
            ) from exc

        wait_seconds = time.perf_counter() - waiter.enqueued_at
        with self._lock:
            self._total_queue_wait_seconds += max(wait_seconds, 0.0)
            self._max_queue_wait_seconds = max(self._max_queue_wait_seconds, wait_seconds)
        return RuntimeRequestAdmission(
            scheduler=self,
            was_queued=True,
            wait_seconds=round(wait_seconds, 4),
            scheduling_lane=scheduling_lane,
            decode_priority_applied=granted.decode_priority_applied,
            prefill_isolated=prefill_isolated,
        )

    def release(
        self,
        *,
        scheduling_lane: Literal["decode", "prefill"] = "decode",
        prefill_isolated: bool = False,
    ) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            if scheduling_lane == "prefill":
                self._active_prefill_requests = max(0, self._active_prefill_requests - 1)
            else:
                self._active_decode_requests = max(0, self._active_decode_requests - 1)
            self._dispatch_waiters_locked()

    def record_continuous_batch(
        self,
        *,
        batch_size: int,
        max_batch_size: int,
        queue_delay_seconds: list[float],
        batch_window_milliseconds: int,
    ) -> None:
        normalized_batch_size = max(batch_size, 0)
        normalized_max_batch_size = max(max_batch_size, 1)
        normalized_delays = [max(delay, 0.0) for delay in queue_delay_seconds]
        with self._lock:
            self._frontier_total_batches += 1
            self._frontier_total_requests += normalized_batch_size
            self._frontier_total_batch_size += normalized_batch_size
            self._frontier_total_batch_utilization += normalized_batch_size / normalized_max_batch_size
            self._frontier_batched_requests += normalized_batch_size if normalized_batch_size > 1 else 0
            self._frontier_coalesced_requests += max(normalized_batch_size - 1, 0)
            self._frontier_total_queue_delay_seconds += sum(normalized_delays)
            self._frontier_max_queue_delay_seconds = max(
                self._frontier_max_queue_delay_seconds,
                max(normalized_delays, default=0.0),
            )
            self._frontier_window_milliseconds = max(self._frontier_window_milliseconds, batch_window_milliseconds)
            self._frontier_max_batch_size = max(self._frontier_max_batch_size, normalized_max_batch_size)

    def record_frontier_batch(
        self,
        *,
        batch_size: int,
        max_batch_size: int,
        queue_delay_seconds: list[float],
        batch_window_milliseconds: int,
    ) -> None:
        self.record_continuous_batch(
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            queue_delay_seconds=queue_delay_seconds,
            batch_window_milliseconds=batch_window_milliseconds,
        )

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            average_queue_wait_seconds = (
                round(self._total_queue_wait_seconds / self._total_queued_requests, 4)
                if self._total_queued_requests
                else 0.0
            )
            frontier_average_batch_size = (
                round(self._frontier_total_batch_size / self._frontier_total_batches, 4)
                if self._frontier_total_batches
                else 0.0
            )
            frontier_average_batch_utilization = (
                round(self._frontier_total_batch_utilization / self._frontier_total_batches, 4)
                if self._frontier_total_batches
                else 0.0
            )
            frontier_average_queue_delay_seconds = (
                round(self._frontier_total_queue_delay_seconds / self._frontier_total_requests, 4)
                if self._frontier_total_requests
                else 0.0
            )
            return {
                "max_concurrent_requests": self.max_concurrent_requests,
                "queue_limit": self.queue_limit,
                "queue_timeout_seconds": self.queue_timeout_seconds,
                "decode_priority_enabled": self.decode_priority_enabled,
                "long_prefill_token_threshold": self.long_prefill_token_threshold,
                "prefill_isolation_enabled": self.prefill_isolation_enabled,
                "prefill_isolation_max_concurrent_requests": self.prefill_isolation_max_concurrent_requests,
                "prefill_isolation_decode_reserve": self.prefill_isolation_decode_reserve,
                "active_requests": self._active_requests,
                "queued_requests": self._queued_requests,
                "active_decode_requests": self._active_decode_requests,
                "active_prefill_requests": self._active_prefill_requests,
                "queued_decode_requests": self._queued_decode_requests,
                "queued_prefill_requests": self._queued_prefill_requests,
                "peak_active_requests": self._peak_active_requests,
                "max_observed_queue_depth": self._max_observed_queue_depth,
                "max_observed_decode_queue_depth": self._max_observed_decode_queue_depth,
                "max_observed_prefill_queue_depth": self._max_observed_prefill_queue_depth,
                "total_queued_requests": self._total_queued_requests,
                "rejected_requests": self._rejected_requests,
                "timed_out_requests": self._timed_out_requests,
                "total_queue_wait_seconds": round(self._total_queue_wait_seconds, 4),
                "average_queue_wait_seconds": average_queue_wait_seconds,
                "max_queue_wait_seconds": round(self._max_queue_wait_seconds, 4),
                "decode_priority_requests": self._decode_priority_request_count,
                "prefill_heavy_requests": self._prefill_request_count,
                "prioritized_decode_grants": self._prioritized_decode_grants,
                "isolated_prefill_requests": self._isolated_prefill_requests,
                "native_window_milliseconds": self._frontier_window_milliseconds,
                "native_max_batch_size": self._frontier_max_batch_size,
                "native_total_batches": self._frontier_total_batches,
                "native_total_requests": self._frontier_total_requests,
                "native_batched_requests": self._frontier_batched_requests,
                "native_coalesced_requests": self._frontier_coalesced_requests,
                "native_total_queue_delay_seconds": round(self._frontier_total_queue_delay_seconds, 4),
                "native_average_queue_delay_seconds": frontier_average_queue_delay_seconds,
                "native_max_queue_delay_seconds": round(self._frontier_max_queue_delay_seconds, 4),
                "native_average_batch_size": frontier_average_batch_size,
                "native_average_batch_utilization": frontier_average_batch_utilization,
                "frontier_window_milliseconds": self._frontier_window_milliseconds,
                "frontier_max_batch_size": self._frontier_max_batch_size,
                "frontier_total_batches": self._frontier_total_batches,
                "frontier_total_requests": self._frontier_total_requests,
                "frontier_batched_requests": self._frontier_batched_requests,
                "frontier_coalesced_requests": self._frontier_coalesced_requests,
                "frontier_total_queue_delay_seconds": round(self._frontier_total_queue_delay_seconds, 4),
                "frontier_average_queue_delay_seconds": frontier_average_queue_delay_seconds,
                "frontier_max_queue_delay_seconds": round(self._frontier_max_queue_delay_seconds, 4),
                "frontier_average_batch_size": frontier_average_batch_size,
                "frontier_average_batch_utilization": frontier_average_batch_utilization,
            }

    def _prefill_isolation_active(
        self,
        *,
        scheduling_lane: Literal["decode", "prefill"],
        prefill_isolation: bool,
    ) -> bool:
        if scheduling_lane != "prefill" or not prefill_isolation or not self.prefill_isolation_enabled:
            return False
        if self.max_concurrent_requests <= 1:
            return False
        available_slots = self.max_concurrent_requests - min(
            self.prefill_isolation_decode_reserve,
            max(self.max_concurrent_requests - 1, 0),
        )
        return available_slots > 0

    def _grant_slot_locked(
        self,
        *,
        scheduling_lane: Literal["decode", "prefill"],
        prefill_isolated: bool,
    ) -> None:
        self._active_requests += 1
        self._peak_active_requests = max(self._peak_active_requests, self._active_requests)
        if scheduling_lane == "prefill":
            self._active_prefill_requests += 1
        else:
            self._active_decode_requests += 1

    def _can_grant_locked(
        self,
        *,
        scheduling_lane: Literal["decode", "prefill"],
        prefill_isolated: bool,
    ) -> bool:
        if self._active_requests >= self.max_concurrent_requests:
            return False
        if scheduling_lane != "prefill" or not prefill_isolated:
            return True
        available_prefill_slots = max(
            1,
            min(
                self.prefill_isolation_max_concurrent_requests,
                self.max_concurrent_requests - min(
                    self.prefill_isolation_decode_reserve,
                    max(self.max_concurrent_requests - 1, 0),
                ),
            ),
        )
        max_active_requests_for_prefill = max(
            1,
            self.max_concurrent_requests - min(
                self.prefill_isolation_decode_reserve,
                max(self.max_concurrent_requests - 1, 0),
            ),
        )
        return (
            self._active_prefill_requests < available_prefill_slots
            and self._active_requests < max_active_requests_for_prefill
        )

    def _higher_priority_waiter_exists_locked(
        self,
        scheduling_lane: Literal["decode", "prefill"],
    ) -> bool:
        if not self.decode_priority_enabled or scheduling_lane != "prefill":
            return False
        return any(waiter.scheduling_lane == "decode" for waiter in self._waiters)

    def _remove_waiter_locked(self, waiter: _QueuedRuntimeRequest) -> None:
        try:
            self._waiters.remove(waiter)
        except ValueError:
            return
        self._queued_requests = max(0, self._queued_requests - 1)
        if waiter.scheduling_lane == "decode":
            self._queued_decode_requests = max(0, self._queued_decode_requests - 1)
        else:
            self._queued_prefill_requests = max(0, self._queued_prefill_requests - 1)

    def _dispatch_waiters_locked(self) -> None:
        if self.max_concurrent_requests <= 0:
            return
        while self._waiters and self._active_requests < self.max_concurrent_requests:
            next_waiter_index, decode_priority_applied = self._next_waiter_index_locked()
            if next_waiter_index is None:
                return
            waiter = self._waiters.pop(next_waiter_index)
            self._queued_requests = max(0, self._queued_requests - 1)
            if waiter.scheduling_lane == "decode":
                self._queued_decode_requests = max(0, self._queued_decode_requests - 1)
            else:
                self._queued_prefill_requests = max(0, self._queued_prefill_requests - 1)
            if decode_priority_applied:
                self._prioritized_decode_grants += 1
            self._grant_slot_locked(
                scheduling_lane=waiter.scheduling_lane,
                prefill_isolated=waiter.prefill_isolated,
            )
            if not waiter.future.done():
                waiter.future.set_result(
                    _GrantedRuntimeRequest(
                        decode_priority_applied=decode_priority_applied,
                    ),
                )

    def _next_waiter_index_locked(self) -> tuple[int | None, bool]:
        first_eligible_index: int | None = None
        first_decode_index: int | None = None
        for index, waiter in enumerate(self._waiters):
            if not self._can_grant_locked(
                scheduling_lane=waiter.scheduling_lane,
                prefill_isolated=waiter.prefill_isolated,
            ):
                continue
            if first_eligible_index is None:
                first_eligible_index = index
            if waiter.scheduling_lane == "decode":
                first_decode_index = index
                break
        if first_eligible_index is None:
            return None, False
        if (
            self.decode_priority_enabled
            and first_decode_index is not None
            and first_decode_index != first_eligible_index
        ):
            return first_decode_index, True
        return first_eligible_index, False


@dataclass(slots=True)
class FrontierBatchMetrics:
    batch_size: int
    batch_position: int
    batch_utilization: float
    queue_delay_seconds: float
    batch_window_milliseconds: int
    max_batch_size: int


@dataclass(slots=True)
class FrontierBatchResult(Generic[ResultT]):
    value: ResultT
    metrics: FrontierBatchMetrics


@dataclass(slots=True)
class _PendingFrontierBatchItem(Generic[PayloadT, ResultT]):
    payload: PayloadT
    enqueued_at: float
    future: asyncio.Future[FrontierBatchResult[ResultT]]


@dataclass(slots=True)
class _FrontierBatchState(Generic[PayloadT, ResultT]):
    queue: list[_PendingFrontierBatchItem[PayloadT, ResultT]]
    task: asyncio.Task[None] | None
    wake_event: asyncio.Event


class FrontierBatchScheduler(Generic[PayloadT, ResultT]):
    """Group same-model requests behind a short backend-native continuous-batching window."""

    def __init__(
        self,
        *,
        runtime_request_scheduler: RuntimeRequestScheduler,
        batch_window_milliseconds: int,
        max_batch_size: int,
    ) -> None:
        self.runtime_request_scheduler = runtime_request_scheduler
        self.batch_window_milliseconds = max(1, batch_window_milliseconds)
        self.max_batch_size = max(1, max_batch_size)
        self._states: dict[str, _FrontierBatchState[PayloadT, ResultT]] = {}
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        *,
        key: str,
        payload: PayloadT,
        execute_batch: Any,
    ) -> FrontierBatchResult[ResultT]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[FrontierBatchResult[ResultT]] = loop.create_future()
        async with self._lock:
            state = self._states.get(key)
            if state is None:
                state = _FrontierBatchState(queue=[], task=None, wake_event=asyncio.Event())
                self._states[key] = state
            state.queue.append(
                _PendingFrontierBatchItem(
                    payload=payload,
                    enqueued_at=time.perf_counter(),
                    future=future,
                ),
            )
            if state.task is None or state.task.done():
                state.task = asyncio.create_task(self._drain(key=key, state=state, execute_batch=execute_batch))
            if len(state.queue) >= self.max_batch_size:
                state.wake_event.set()
        return await future

    async def _drain(self, *, key: str, state: _FrontierBatchState[PayloadT, ResultT], execute_batch: Any) -> None:
        try:
            while True:
                await self._wait_for_window(state)
                batch_items = await self._take_batch(key=key, state=state)
                if not batch_items:
                    return
                dispatched_at = time.perf_counter()
                queue_delays = [round(dispatched_at - item.enqueued_at, 4) for item in batch_items]
                batch_size = len(batch_items)
                try:
                    metrics = [
                        FrontierBatchMetrics(
                            batch_size=batch_size,
                            batch_position=batch_position,
                            batch_utilization=round(batch_size / self.max_batch_size, 4),
                            queue_delay_seconds=queue_delay_seconds,
                            batch_window_milliseconds=self.batch_window_milliseconds,
                            max_batch_size=self.max_batch_size,
                        )
                        for batch_position, queue_delay_seconds in enumerate(queue_delays)
                    ]
                    values = await execute_batch(
                        [
                            (item.payload, item_metrics)
                            for item, item_metrics in zip(batch_items, metrics, strict=True)
                        ],
                    )
                    if len(values) != batch_size:
                        raise ValueError(f"Expected {batch_size} batched results, received {len(values)}.")
                except Exception as exc:
                    for item in batch_items:
                        if not item.future.done():
                            item.future.set_exception(exc)
                    continue
                self.runtime_request_scheduler.record_continuous_batch(
                    batch_size=batch_size,
                    max_batch_size=self.max_batch_size,
                    queue_delay_seconds=queue_delays,
                    batch_window_milliseconds=self.batch_window_milliseconds,
                )
                for item, value, item_metrics in zip(
                    batch_items,
                    values,
                    metrics,
                    strict=True,
                ):
                    if item.future.done():
                        continue
                    item.future.set_result(
                        FrontierBatchResult(
                            value=value,
                            metrics=item_metrics,
                        ),
                    )
        finally:
            async with self._lock:
                existing = self._states.get(key)
                if existing is state:
                    if state.queue:
                        state.task = asyncio.create_task(self._drain(key=key, state=state, execute_batch=execute_batch))
                    else:
                        self._states.pop(key, None)

    async def _wait_for_window(self, state: _FrontierBatchState[PayloadT, ResultT]) -> None:
        async with self._lock:
            if not state.queue:
                return
            state.wake_event.clear()
            if len(state.queue) >= self.max_batch_size:
                return
            wake_event = state.wake_event
            effective_window_milliseconds = self._effective_batch_window_milliseconds(queue_depth=len(state.queue))
        try:
            await asyncio.wait_for(wake_event.wait(), timeout=effective_window_milliseconds / 1000)
        except asyncio.TimeoutError:
            return

    def _effective_batch_window_milliseconds(self, *, queue_depth: int) -> int:
        if queue_depth <= 1:
            runtime_snapshot = self.runtime_request_scheduler.snapshot()
            if runtime_snapshot["active_requests"] == 0 and runtime_snapshot["queued_requests"] == 0:
                return max(1, self.batch_window_milliseconds // 2)
        return self.batch_window_milliseconds
    async def _take_batch(
        self,
        *,
        key: str,
        state: _FrontierBatchState[PayloadT, ResultT],
    ) -> list[_PendingFrontierBatchItem[PayloadT, ResultT]]:
        async with self._lock:
            current = self._states.get(key)
            if current is not state or not state.queue:
                return []
            batch_items = state.queue[: self.max_batch_size]
            del state.queue[: self.max_batch_size]
            if len(state.queue) >= self.max_batch_size:
                state.wake_event.set()
            return batch_items


ContinuousBatchMetrics = FrontierBatchMetrics
ContinuousBatchResult = FrontierBatchResult
ContinuousBatchScheduler = FrontierBatchScheduler
