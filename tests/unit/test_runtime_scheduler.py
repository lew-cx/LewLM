from __future__ import annotations

import asyncio

import pytest

from lewlm.core.errors import BackpressureError
from lewlm.runtime.scheduler import FrontierBatchScheduler, RuntimeRequestScheduler


async def test_runtime_request_scheduler_tracks_queue_depth() -> None:
    scheduler = RuntimeRequestScheduler(
        max_concurrent_requests=1,
        queue_limit=1,
        queue_timeout_seconds=1,
    )
    first = await scheduler.acquire()

    async def acquire_second():
        return await scheduler.acquire()

    second_task = asyncio.create_task(acquire_second())
    await asyncio.sleep(0.05)
    queued_snapshot = scheduler.snapshot()
    assert queued_snapshot["active_requests"] == 1
    assert queued_snapshot["queued_requests"] == 1
    assert queued_snapshot["total_queued_requests"] == 1

    first.release()
    second = await second_task
    assert second.was_queued is True
    assert second.wait_seconds >= 0.0
    second.release()

    final_snapshot = scheduler.snapshot()
    assert final_snapshot["active_requests"] == 0
    assert final_snapshot["queued_requests"] == 0
    assert final_snapshot["max_observed_queue_depth"] == 1


async def test_runtime_request_scheduler_rejects_when_queue_is_full() -> None:
    scheduler = RuntimeRequestScheduler(
        max_concurrent_requests=1,
        queue_limit=0,
        queue_timeout_seconds=1,
    )
    first = await scheduler.acquire()

    with pytest.raises(BackpressureError):
        await scheduler.acquire()

    first.release()
    snapshot = scheduler.snapshot()
    assert snapshot["rejected_requests"] == 1


async def test_runtime_request_scheduler_prioritizes_decode_over_prefill() -> None:
    scheduler = RuntimeRequestScheduler(
        max_concurrent_requests=1,
        queue_limit=2,
        queue_timeout_seconds=1,
        decode_priority_enabled=True,
        long_prefill_token_threshold=32,
    )
    first = await scheduler.acquire(prefill_heavy=True, prefill_isolation=False)
    prefill_task = asyncio.create_task(scheduler.acquire(prefill_heavy=True, prefill_isolation=False))
    await asyncio.sleep(0.05)
    decode_task = asyncio.create_task(
        scheduler.acquire(prefill_heavy=False, decode_priority=True, prefill_isolation=False),
    )
    await asyncio.sleep(0.05)

    first.release()
    decode = await decode_task
    snapshot = scheduler.snapshot()

    assert decode.scheduling_lane == "decode"
    assert decode.decode_priority_applied is True
    assert snapshot["prioritized_decode_grants"] == 1
    assert snapshot["decode_priority_requests"] == 1
    assert snapshot["prefill_heavy_requests"] == 2

    decode.release()
    prefill = await prefill_task
    assert prefill.scheduling_lane == "prefill"
    prefill.release()


async def test_runtime_request_scheduler_isolates_prefill_capacity() -> None:
    scheduler = RuntimeRequestScheduler(
        max_concurrent_requests=2,
        queue_limit=2,
        queue_timeout_seconds=1,
        decode_priority_enabled=True,
        long_prefill_token_threshold=32,
        prefill_isolation_enabled=True,
        prefill_isolation_max_concurrent_requests=1,
        prefill_isolation_decode_reserve=1,
    )
    first_prefill = await scheduler.acquire(prefill_heavy=True, prefill_isolation=True)
    second_task = asyncio.create_task(scheduler.acquire(prefill_heavy=True, prefill_isolation=True))
    await asyncio.sleep(0.05)
    queued_snapshot = scheduler.snapshot()

    assert first_prefill.prefill_isolated is True
    assert queued_snapshot["active_prefill_requests"] == 1
    assert queued_snapshot["queued_prefill_requests"] == 1
    assert queued_snapshot["isolated_prefill_requests"] == 2

    first_prefill.release()
    second_prefill = await second_task
    second_prefill.release()


async def test_frontier_batch_scheduler_records_batch_metrics() -> None:
    scheduler = RuntimeRequestScheduler(
        max_concurrent_requests=2,
        queue_limit=2,
        queue_timeout_seconds=1,
    )
    batch_scheduler: FrontierBatchScheduler[int, int] = FrontierBatchScheduler(
        runtime_request_scheduler=scheduler,
        batch_window_milliseconds=25,
        max_batch_size=2,
    )

    async def execute_batch(batch_payloads):
        return [payload for payload, _ in batch_payloads]

    first_task = asyncio.create_task(
        batch_scheduler.enqueue(key="chat", payload=1, execute_batch=execute_batch),
    )
    second_task = asyncio.create_task(
        batch_scheduler.enqueue(key="chat", payload=2, execute_batch=execute_batch),
    )

    first_result, second_result = await asyncio.gather(first_task, second_task)
    snapshot = scheduler.snapshot()

    assert [first_result.value, second_result.value] == [1, 2]
    assert first_result.metrics.batch_size == 2
    assert second_result.metrics.batch_size == 2
    assert snapshot["native_total_batches"] == 1
    assert snapshot["native_total_requests"] == 2
    assert snapshot["native_batched_requests"] == 2
    assert snapshot["native_coalesced_requests"] == 1
    assert snapshot["native_average_batch_size"] == 2.0
    assert snapshot["native_average_batch_utilization"] == 1.0
    assert snapshot["native_window_milliseconds"] == 25
    assert snapshot["frontier_total_batches"] == 1
    assert snapshot["frontier_total_requests"] == 2
    assert snapshot["frontier_batched_requests"] == 2
    assert snapshot["frontier_coalesced_requests"] == 1
    assert snapshot["frontier_average_batch_size"] == 2.0
    assert snapshot["frontier_average_batch_utilization"] == 1.0
    assert snapshot["frontier_window_milliseconds"] == 25


async def test_frontier_batch_scheduler_halves_idle_single_request_window() -> None:
    scheduler = RuntimeRequestScheduler(
        max_concurrent_requests=1,
        queue_limit=1,
        queue_timeout_seconds=1,
    )
    batch_scheduler: FrontierBatchScheduler[int, int] = FrontierBatchScheduler(
        runtime_request_scheduler=scheduler,
        batch_window_milliseconds=25,
        max_batch_size=4,
    )

    assert batch_scheduler._effective_batch_window_milliseconds(queue_depth=1) == 12
    assert batch_scheduler._effective_batch_window_milliseconds(queue_depth=2) == 25

    admission = await scheduler.acquire()
    try:
        assert batch_scheduler._effective_batch_window_milliseconds(queue_depth=1) == 25
    finally:
        admission.release()
