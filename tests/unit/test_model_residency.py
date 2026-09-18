"""Concurrency and lifecycle coverage for the shared-runtime residency baseline."""

from __future__ import annotations

import asyncio

import pytest

from lewlm.core.contracts import (
    ConversionStatus,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.core.errors import ModelLifecycleConflictError, ModelLoadError, RuntimeUnavailableError
from lewlm.events.bus import EventBus
from lewlm.runtime.base import ManagedRuntime
from lewlm.runtime.residency import ModelResidencyManager, ModelResidencyState
from lewlm.runtime.scheduler import RuntimeRequestScheduler


class ControlledRuntime(ManagedRuntime):
    name = "controlled"
    affinity = RuntimeAffinity.LLAMACPP
    supported_formats = (ModelFormat.GGUF,)
    supported_modalities = (ModelModality.TEXT,)

    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self.unload_calls = 0
        self.load_started = asyncio.Event()
        self.allow_load = asyncio.Event()
        self.allow_load.set()
        self.unload_started = asyncio.Event()
        self.allow_unload = asyncio.Event()
        self.allow_unload.set()
        self.fail_load = False
        self.active_loads = 0
        self.peak_active_loads = 0

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, None

    async def _load_model(self, manifest: ModelManifest) -> None:
        self.load_calls += 1
        self.active_loads += 1
        self.peak_active_loads = max(self.peak_active_loads, self.active_loads)
        self.load_started.set()
        try:
            await self.allow_load.wait()
            if self.fail_load:
                raise RuntimeError("synthetic load failure")
        finally:
            self.active_loads -= 1

    async def _unload_model(self, model_id: str) -> None:
        self.unload_calls += 1
        self.unload_started.set()
        await self.allow_unload.wait()


def manifest(model_id: str = "model-a") -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        display_name=model_id,
        architecture_family="test",
        modality=(ModelModality.TEXT,),
        source_path=f"/tmp/{model_id}",
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint=f"fingerprint-{model_id}",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )


def manager(*, max_loads: int = 2, event_bus: EventBus | None = None) -> ModelResidencyManager:
    return ModelResidencyManager(
        runtime_instance_id="runtime-test",
        model_load_scheduler=RuntimeRequestScheduler(
            max_concurrent_requests=max_loads,
            queue_limit=8,
            queue_timeout_seconds=5,
        ),
        event_bus=event_bus or EventBus(),
    )


async def test_same_model_load_is_single_flight() -> None:
    runtime = ControlledRuntime()
    runtime.allow_load.clear()
    residency = manager()
    selected = manifest()

    first = asyncio.create_task(residency.ensure_loaded(runtime, selected))
    await runtime.load_started.wait()
    second = asyncio.create_task(residency.ensure_loaded(runtime, selected))
    await asyncio.sleep(0)
    assert runtime.load_calls == 1

    runtime.allow_load.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.current_state == ModelResidencyState.READY
    assert second_result.current_state == ModelResidencyState.READY
    assert second_result.joined_existing_operation is True
    assert (await residency.list_residencies())[0].joined_load_waiter_count == 1


async def test_cancelled_waiter_does_not_cancel_shared_load() -> None:
    runtime = ControlledRuntime()
    runtime.allow_load.clear()
    residency = manager()
    selected = manifest()

    owner = asyncio.create_task(residency.ensure_loaded(runtime, selected))
    await runtime.load_started.wait()
    waiter = asyncio.create_task(residency.ensure_loaded(runtime, selected))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    runtime.allow_load.set()
    result = await owner
    assert result.current_state == ModelResidencyState.READY
    assert runtime.load_calls == 1


async def test_different_models_follow_global_parallel_load_policy() -> None:
    runtime = ControlledRuntime()
    runtime.allow_load.clear()
    residency = manager(max_loads=2)

    first = asyncio.create_task(residency.ensure_loaded(runtime, manifest("model-a")))
    second = asyncio.create_task(residency.ensure_loaded(runtime, manifest("model-b")))
    for _ in range(20):
        if runtime.load_calls == 2:
            break
        await asyncio.sleep(0)

    assert runtime.load_calls == 2
    assert runtime.peak_active_loads == 2
    runtime.allow_load.set()
    await asyncio.gather(first, second)


async def test_failed_load_can_be_retried() -> None:
    runtime = ControlledRuntime()
    runtime.fail_load = True
    residency = manager()
    selected = manifest()

    first, second = await asyncio.gather(
        residency.ensure_loaded(runtime, selected),
        residency.ensure_loaded(runtime, selected),
        return_exceptions=True,
    )
    # Backend load failures reach every waiter as the typed lifecycle error.
    assert isinstance(first, ModelLoadError)
    assert isinstance(second, ModelLoadError)
    assert first.code == "model_load_failed"
    assert (await residency.list_residencies())[0].state == ModelResidencyState.FAILED

    runtime.fail_load = False
    result = await residency.ensure_loaded(runtime, selected)
    assert result.current_state == ModelResidencyState.READY
    assert runtime.load_calls == 2


async def test_usage_lease_cleans_up_and_blocks_unload() -> None:
    runtime = ControlledRuntime()
    residency = manager()
    selected = manifest()

    async with residency.acquire(runtime, selected):
        snapshot = await residency.get_residency(selected.model_id)
        assert snapshot is not None
        assert snapshot.active_usage_count == 1
        with pytest.raises(ModelLifecycleConflictError) as exc_info:
            await residency.unload(runtime, selected)
        assert exc_info.value.status_code == 409

    snapshot = await residency.get_residency(selected.model_id)
    assert snapshot is not None
    assert snapshot.active_usage_count == 0
    result = await residency.unload(runtime, selected)
    assert result.backend_operation_performed is True
    assert runtime.unload_calls == 1


async def test_concurrent_unloads_join_one_backend_operation() -> None:
    runtime = ControlledRuntime()
    runtime.allow_unload.clear()
    residency = manager()
    selected = manifest()
    await residency.ensure_loaded(runtime, selected)

    first = asyncio.create_task(residency.unload(runtime, selected, drain=True, timeout_seconds=2))
    await runtime.unload_started.wait()
    second = asyncio.create_task(residency.unload(runtime, selected, drain=True, timeout_seconds=2))
    await asyncio.sleep(0)

    assert runtime.unload_calls == 1
    runtime.allow_unload.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.backend_operation_performed is True
    assert second_result.backend_operation_performed is False
    assert second_result.joined_existing_operation is True
    assert runtime.unload_calls == 1


async def test_drain_rejects_non_positive_timeout() -> None:
    runtime = ControlledRuntime()
    residency = manager()
    selected = manifest()
    await residency.ensure_loaded(runtime, selected)

    with pytest.raises(ValueError, match="greater than zero"):
        await residency.unload(runtime, selected, drain=True, timeout_seconds=0)

    assert runtime.unload_calls == 0
    assert (await residency.list_residencies())[0].state == ModelResidencyState.READY


async def test_usage_lease_cleans_up_after_capability_failure() -> None:
    runtime = ControlledRuntime()
    residency = manager()
    selected = manifest()

    with pytest.raises(RuntimeError, match="capability failed"):
        async with residency.acquire(runtime, selected):
            raise RuntimeError("capability failed")

    snapshot = await residency.get_residency(selected.model_id)
    assert snapshot is not None
    assert snapshot.active_usage_count == 0


async def test_drain_stops_new_leases_and_waits_for_existing_usage() -> None:
    runtime = ControlledRuntime()
    residency = manager()
    selected = manifest()
    lease = residency.acquire(runtime, selected)
    await lease.__aenter__()

    drain = asyncio.create_task(residency.unload(runtime, selected, drain=True, timeout_seconds=2))
    await asyncio.sleep(0)
    snapshot = await residency.get_residency(selected.model_id)
    assert snapshot is not None
    assert snapshot.state == ModelResidencyState.DRAINING
    with pytest.raises(ModelLifecycleConflictError):
        async with residency.acquire(runtime, selected):
            pass

    await lease.__aexit__(None, None, None)
    result = await drain
    assert result.backend_operation_performed is True
    assert await residency.get_residency(selected.model_id) is None


async def test_shutdown_stops_new_acquisitions_and_unloads_once_after_usage_finishes() -> None:
    runtime = ControlledRuntime()
    residency = manager()
    selected = manifest()
    lease = residency.acquire(runtime, selected)
    await lease.__aenter__()

    shutdown = asyncio.create_task(residency.shutdown())
    for _ in range(20):
        snapshot = await residency.get_residency(selected.model_id)
        if snapshot is not None and snapshot.state == ModelResidencyState.DRAINING:
            break
        await asyncio.sleep(0)

    with pytest.raises(RuntimeUnavailableError):
        async with residency.acquire(runtime, selected):
            pass

    await lease.__aexit__(None, None, None)
    await shutdown
    await residency.shutdown()

    assert runtime.unload_calls == 1
    assert await residency.list_residencies() == []


async def test_residency_events_distinguish_joined_load_usage_and_drain() -> None:
    runtime = ControlledRuntime()
    runtime.allow_load.clear()
    event_bus = EventBus()
    subscription = event_bus.subscribe()
    residency = manager(event_bus=event_bus)
    selected = manifest()

    owner = asyncio.create_task(
        residency.ensure_loaded(runtime, selected, request_id="owner", application_id="rag-chat"),
    )
    requested = await subscription.get()
    waiter = asyncio.create_task(
        residency.ensure_loaded(runtime, selected, request_id="waiter", application_id="document-generator"),
    )
    joined = await subscription.get()
    runtime.allow_load.set()
    await asyncio.gather(owner, waiter)

    async with residency.acquire(
        runtime,
        selected,
        request_id="usage",
        application_id="document-generator",
    ):
        acquired = await subscription.get()
    released = await subscription.get()
    await residency.unload(
        runtime,
        selected,
        drain=True,
        timeout_seconds=1,
        request_id="drain",
        application_id="operator",
    )
    drain_events = [await subscription.get() for _ in range(4)]
    subscription.close()

    assert requested.type.value == "model.load.requested"
    assert joined.type.value == "model.load.joined"
    assert requested.payload["operation_id"] == joined.payload["operation_id"]
    assert requested.payload["application_id"] == "rag-chat"
    assert joined.payload["application_id"] == "document-generator"
    assert [acquired.type.value, released.type.value] == [
        "model.usage.acquired",
        "model.usage.released",
    ]
    assert acquired.payload["active_usage_count"] == 1
    assert released.payload["active_usage_count"] == 0
    assert [event.type.value for event in drain_events] == [
        "model.drain.requested",
        "model.draining",
        "model.unloading",
        "model.unloaded",
    ]
    assert len({event.payload["operation_id"] for event in drain_events}) == 1


async def test_a_waiter_about_to_lease_is_counted_until_its_lease_is_taken() -> None:
    """The `balanced` policy unloads a runtime's *other* idle models after each
    request. A model whose load just finished has no active lease for an
    instant before its waiter takes one; that waiter counts as pending until
    `acquire` converts it into usage under the same lock, so the model never
    looks idle in between."""

    runtime = ControlledRuntime()
    runtime.allow_load.clear()
    residency = manager()
    selected = manifest("model-b")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def use() -> None:
        async with residency.acquire(runtime, selected):
            entered.set()
            await release.wait()

    waiter = asyncio.create_task(use())
    await runtime.load_started.wait()
    snapshot = await residency.get_residency("model-b")
    assert snapshot is not None and snapshot.state == ModelResidencyState.LOADING
    assert snapshot.pending_lease_count == 1, "the caller waiting on the load is visible"
    with pytest.raises(ModelLifecycleConflictError) as excinfo:
        await residency.unload(runtime, selected)
    assert excinfo.value.details["pending_lease_count"] == 1

    runtime.allow_load.set()
    await entered.wait()
    snapshot = await residency.get_residency("model-b")
    assert snapshot is not None and snapshot.state == ModelResidencyState.READY
    assert snapshot.active_usage_count == 1 and snapshot.pending_lease_count == 0, "pending became the lease, never idle in between"
    with pytest.raises(ModelLifecycleConflictError):
        await residency.unload(runtime, selected)

    release.set()
    await waiter
    snapshot = await residency.get_residency("model-b")
    assert snapshot is not None and snapshot.active_usage_count == 0 and snapshot.pending_lease_count == 0
    await residency.unload(runtime, selected)
    assert runtime.unload_calls == 1
