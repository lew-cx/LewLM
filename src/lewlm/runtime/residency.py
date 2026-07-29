"""Process-local model residency, single-flight loading, and usage leases."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import AsyncIterator
from uuid import uuid4

from pydantic import BaseModel, Field

from lewlm.core.contracts import ModelManifest, RuntimeContract, utc_now
from lewlm.core.errors import ModelLifecycleConflictError, RuntimeUnavailableError
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.runtime.scheduler import RuntimeRequestScheduler
from lewlm.runtime.request_context import application_id_var, client_instance_id_var
from lewlm.telemetry.runtime_metrics import RuntimeMetricsRecorder


class ModelResidencyState(str, Enum):
    """Observable states for one process-local model residency."""

    LOADING = "loading"
    READY = "ready"
    DRAINING = "draining"
    UNLOADING = "unloading"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ModelResidencyKey:
    """Identity of a resident model within one runtime process."""

    runtime_name: str
    model_id: str


class ModelResidencySnapshot(BaseModel):
    """Safe public view of one resident model record."""

    model_id: str
    runtime: str
    state: ModelResidencyState
    load_started_at: datetime | None = None
    loaded_at: datetime | None = None
    last_used_at: datetime | None = None
    active_usage_count: int = 0
    pending_unload: bool = False
    failure: str | None = None
    load_attempt_count: int = 0
    joined_load_waiter_count: int = 0
    estimated_memory_mb: int | None = None


class ModelLifecycleResult(BaseModel):
    """Result of a warm, drain, or unload lifecycle operation."""

    operation_id: str = Field(default_factory=lambda: str(uuid4()))
    runtime_instance_id: str
    model_id: str
    runtime: str
    previous_state: ModelResidencyState | None = None
    current_state: ModelResidencyState | None = None
    active_usage_count: int = 0
    backend_operation_performed: bool = False
    joined_existing_operation: bool = False
    reason: str


@dataclass(slots=True)
class _ResidencyRecord:
    key: ModelResidencyKey
    manifest: ModelManifest
    runtime: RuntimeContract
    state: ModelResidencyState
    load_started_at: datetime | None = None
    loaded_at: datetime | None = None
    last_used_at: datetime | None = None
    active_usage_count: int = 0
    pending_unload: bool = False
    failure: str | None = None
    load_attempt_count: int = 0
    joined_load_waiter_count: int = 0
    load_operation_id: str = field(default_factory=lambda: str(uuid4()))
    load_task: asyncio.Task[None] | None = None
    idle_event: asyncio.Event = field(default_factory=asyncio.Event)
    lifecycle_event: asyncio.Event = field(default_factory=asyncio.Event)


class ModelResidencyManager:
    """Own process-local live-model coordination above backend runtimes.

    The manifest registry remains the source of persisted model identity. This
    manager coordinates only live objects in this service container.
    """

    def __init__(
        self,
        *,
        runtime_instance_id: str,
        model_load_scheduler: RuntimeRequestScheduler,
        event_bus: EventBus,
        runtime_metrics_recorder: RuntimeMetricsRecorder | None = None,
    ) -> None:
        self.runtime_instance_id = runtime_instance_id
        self.model_load_scheduler = model_load_scheduler
        self.event_bus = event_bus
        self.runtime_metrics_recorder = runtime_metrics_recorder
        self._records: dict[ModelResidencyKey, _ResidencyRecord] = {}
        self._lock = asyncio.Lock()
        self._closing = False

    @staticmethod
    def key_for(runtime: RuntimeContract, manifest: ModelManifest) -> ModelResidencyKey:
        return ModelResidencyKey(runtime_name=runtime.name, model_id=manifest.model_id)

    async def ensure_loaded(
        self,
        runtime: RuntimeContract,
        manifest: ModelManifest,
        *,
        request_id: str | None = None,
        application_id: str | None = None,
        capability: str | None = None,
    ) -> ModelLifecycleResult:
        """Load a residency once, joining an existing same-key load when present."""

        key = self.key_for(runtime, manifest)
        joined = False
        backend_operation = False
        async with self._lock:
            if self._closing:
                raise RuntimeUnavailableError("The LewLM runtime is shutting down.")
            record = self._records.get(key)
            if (
                record is not None
                and record.state == ModelResidencyState.READY
                and runtime.is_model_loaded(manifest.model_id)
            ):
                record.last_used_at = utc_now()
                return self._result(
                    record,
                    previous_state=ModelResidencyState.READY,
                    backend_operation=False,
                    joined=False,
                    reason="Model residency is already ready.",
                )
            if record is not None and record.state == ModelResidencyState.READY:
                # A legacy policy, probe, or backend-specific path may have
                # unloaded directly. Reconcile instead of returning stale
                # process state.
                record.state = ModelResidencyState.FAILED
                record.failure = "Backend no longer reports the model as loaded."
            if record is not None and record.state in {ModelResidencyState.DRAINING, ModelResidencyState.UNLOADING}:
                raise ModelLifecycleConflictError(
                    "The model is draining or unloading and cannot accept new use.",
                    details=self._conflict_details(record),
                )
            if record is not None and record.state == ModelResidencyState.LOADING and record.load_task is not None:
                task = record.load_task
                record.joined_load_waiter_count += 1
                joined = True
            else:
                previous_state = record.state if record is not None else None
                if runtime.is_model_loaded(manifest.model_id):
                    now = utc_now()
                    record = record or _ResidencyRecord(
                        key=key,
                        manifest=manifest,
                        runtime=runtime,
                        state=ModelResidencyState.READY,
                    )
                    record.state = ModelResidencyState.READY
                    record.loaded_at = record.loaded_at or now
                    record.last_used_at = now
                    record.failure = None
                    record.idle_event.set()
                    self._records[key] = record
                    return self._result(
                        record,
                        previous_state=previous_state,
                        backend_operation=False,
                        joined=False,
                        reason="Backend already held the model; residency tracking was adopted.",
                    )
                now = utc_now()
                record = record or _ResidencyRecord(
                    key=key,
                    manifest=manifest,
                    runtime=runtime,
                    state=ModelResidencyState.LOADING,
                )
                record.manifest = manifest
                record.runtime = runtime
                record.state = ModelResidencyState.LOADING
                record.load_started_at = now
                record.loaded_at = None
                record.failure = None
                record.pending_unload = False
                record.load_attempt_count += 1
                record.load_operation_id = str(uuid4())
                record.idle_event.clear()
                task = asyncio.create_task(
                    self._perform_load(
                        record,
                        request_id=request_id,
                        application_id=application_id,
                        capability=capability,
                    ),
                    name=f"lewlm-load-{runtime.name}-{manifest.model_id}",
                )
                task.add_done_callback(self._consume_unobserved_task_exception)
                record.load_task = task
                self._records[key] = record
                backend_operation = True
        if joined:
            await self._publish(
                EventType.MODEL_LOAD_JOINED,
                record,
                request_id=request_id,
                application_id=application_id,
                operation_id=record.load_operation_id,
                capability=capability,
            )
        # Shield the shared operation so cancellation of one HTTP waiter cannot
        # cancel a load needed by other applications.
        await asyncio.shield(task)
        async with self._lock:
            completed = self._records.get(key)
            if completed is None or completed.state != ModelResidencyState.READY:
                failure = completed.failure if completed is not None else "Residency disappeared during loading."
                raise RuntimeUnavailableError(
                    "The model could not be made resident.",
                    details={"model_id": manifest.model_id, "runtime": runtime.name, "failure": failure},
                )
            return self._result(
                completed,
                previous_state=ModelResidencyState.LOADING,
                backend_operation=backend_operation,
                joined=joined,
                reason="Joined the in-flight model load." if joined else "Model loaded and is ready.",
            )

    @asynccontextmanager
    async def acquire(
        self,
        runtime: RuntimeContract,
        manifest: ModelManifest,
        *,
        request_id: str | None = None,
        application_id: str | None = None,
        capability: str | None = None,
    ) -> AsyncIterator[ModelResidencySnapshot]:
        """Hold a model usage lease for the complete backend operation."""

        wait_started_at = time.perf_counter()
        load_result = await self.ensure_loaded(
            runtime,
            manifest,
            request_id=request_id,
            application_id=application_id,
            capability=capability,
        )
        key = self.key_for(runtime, manifest)
        async with self._lock:
            record = self._records.get(key)
            if record is None or record.state != ModelResidencyState.READY:
                raise ModelLifecycleConflictError(
                    "The model stopped accepting usage before a lease could be acquired.",
                    details={"model_id": manifest.model_id, "runtime": runtime.name},
                )
            record.active_usage_count += 1
            record.last_used_at = utc_now()
            record.idle_event.clear()
            snapshot = self._snapshot(record)
        await self._publish(
            EventType.MODEL_USAGE_ACQUIRED,
            record,
            request_id=request_id,
            application_id=application_id,
            operation_id=record.load_operation_id,
            capability=capability,
        )
        resolved_application_id = application_id or application_id_var.get()
        if self.runtime_metrics_recorder is not None:
            self.runtime_metrics_recorder.record_lease_acquired(
                application_id=resolved_application_id,
                model_id=manifest.model_id,
                capability=capability,
                residency_wait_seconds=time.perf_counter() - wait_started_at,
                contended=load_result.joined_existing_operation,
            )
        try:
            yield snapshot
        finally:
            async with self._lock:
                current = self._records.get(key)
                if current is not None:
                    current.active_usage_count = max(0, current.active_usage_count - 1)
                    current.last_used_at = utc_now()
                    if current.active_usage_count == 0:
                        current.idle_event.set()
                    record = current
            await self._publish(
                EventType.MODEL_USAGE_RELEASED,
                record,
                request_id=request_id,
                application_id=application_id,
                operation_id=record.load_operation_id,
                capability=capability,
            )
            if self.runtime_metrics_recorder is not None:
                self.runtime_metrics_recorder.record_lease_released(
                    application_id=resolved_application_id,
                )

    async def warm(
        self,
        runtime: RuntimeContract,
        manifest: ModelManifest,
        *,
        request_id: str | None = None,
        application_id: str | None = None,
    ) -> ModelLifecycleResult:
        result = await self.ensure_loaded(
            runtime,
            manifest,
            request_id=request_id,
            application_id=application_id,
            capability="model_lifecycle",
        )
        async with self.acquire(
            runtime,
            manifest,
            request_id=request_id,
            application_id=application_id,
            capability="model_lifecycle",
        ):
            await runtime.warm_model(manifest.model_id)
        return result.model_copy(update={"reason": "Model is loaded and its backend warm hook completed."})

    async def unload(
        self,
        runtime: RuntimeContract,
        manifest: ModelManifest,
        *,
        drain: bool = False,
        timeout_seconds: float | None = None,
        request_id: str | None = None,
        application_id: str | None = None,
    ) -> ModelLifecycleResult:
        """Safely unload a model, optionally draining existing usage first."""

        if drain and timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero when draining.")
        key = self.key_for(runtime, manifest)
        operation_id = str(uuid4())
        joined_existing_operation = False
        while True:
            blocked_record: _ResidencyRecord | None = None
            idle_event: asyncio.Event | None = None
            join_event: asyncio.Event | None = None
            async with self._lock:
                record = self._records.get(key)
                if record is None and runtime.is_model_loaded(manifest.model_id):
                    now = utc_now()
                    record = _ResidencyRecord(
                        key=key,
                        manifest=manifest,
                        runtime=runtime,
                        state=ModelResidencyState.READY,
                        loaded_at=now,
                        last_used_at=now,
                    )
                    record.idle_event.set()
                    self._records[key] = record
                if record is None:
                    return ModelLifecycleResult(
                        operation_id=operation_id,
                        runtime_instance_id=self.runtime_instance_id,
                        model_id=manifest.model_id,
                        runtime=runtime.name,
                        joined_existing_operation=joined_existing_operation,
                        reason=(
                            "Joined the in-flight unload; the model is now unloaded."
                            if joined_existing_operation
                            else "Model was already unloaded."
                        ),
                    )
                previous_state = record.state
                if record.state == ModelResidencyState.LOADING:
                    raise ModelLifecycleConflictError(
                        "The model is still loading and cannot be unloaded yet.",
                        details=self._conflict_details(record),
                    )
                if record.state in {ModelResidencyState.DRAINING, ModelResidencyState.UNLOADING}:
                    join_event = record.lifecycle_event
                elif record.active_usage_count > 0 and not drain:
                    blocked_record = record
                elif drain:
                    record.state = ModelResidencyState.DRAINING
                    record.pending_unload = True
                    record.lifecycle_event.clear()
                    idle_event = record.idle_event
                else:
                    record.state = ModelResidencyState.UNLOADING
                    record.lifecycle_event.clear()
            if join_event is None:
                break
            # A concurrent lifecycle caller joins the existing transition
            # without being able to cancel its backend operation.
            await asyncio.shield(join_event.wait())
            joined_existing_operation = True
        if blocked_record is not None:
            await self._publish(
                EventType.MODEL_UNLOAD_BLOCKED,
                blocked_record,
                request_id=request_id,
                application_id=application_id,
                operation_id=operation_id,
                capability="model_lifecycle",
            )
            raise ModelLifecycleConflictError(
                "The model is actively in use; request drain or retry after usage completes.",
                details=self._conflict_details(blocked_record),
            )
        if idle_event is not None:
            await self._publish(
                EventType.MODEL_DRAIN_REQUESTED,
                record,
                request_id=request_id,
                application_id=application_id,
                operation_id=operation_id,
                capability="model_lifecycle",
            )
            await self._publish(
                EventType.MODEL_DRAINING,
                record,
                request_id=request_id,
                application_id=application_id,
                operation_id=operation_id,
                capability="model_lifecycle",
            )
            try:
                if timeout_seconds is None:
                    await idle_event.wait()
                else:
                    await asyncio.wait_for(idle_event.wait(), timeout=max(0.0, timeout_seconds))
            except asyncio.CancelledError:
                async with self._lock:
                    current = self._records.get(key)
                    if current is not None and current.state == ModelResidencyState.DRAINING:
                        current.state = ModelResidencyState.READY
                        current.pending_unload = False
                        current.lifecycle_event.set()
                raise
            except TimeoutError as exc:
                async with self._lock:
                    current = self._records.get(key)
                    if current is not None and current.state == ModelResidencyState.DRAINING:
                        current.state = ModelResidencyState.READY
                        current.pending_unload = False
                        current.lifecycle_event.set()
                raise ModelLifecycleConflictError(
                    "Timed out waiting for active model usage to drain.",
                    details=self._conflict_details(record),
                ) from exc
            async with self._lock:
                current = self._records.get(key)
                if current is None:
                    return ModelLifecycleResult(
                        operation_id=operation_id,
                        runtime_instance_id=self.runtime_instance_id,
                        model_id=manifest.model_id,
                        runtime=runtime.name,
                        previous_state=previous_state,
                        reason="Model was unloaded while the drain request was waiting.",
                    )
                current.state = ModelResidencyState.UNLOADING
                record = current
        await self._publish(
            EventType.MODEL_UNLOADING,
            record,
            request_id=request_id,
            application_id=application_id,
            operation_id=operation_id,
            capability="model_lifecycle",
        )
        try:
            await runtime.unload_model(manifest.model_id)
        except asyncio.CancelledError:
            async with self._lock:
                current = self._records.get(key)
                if current is not None:
                    if runtime.is_model_loaded(manifest.model_id):
                        current.state = ModelResidencyState.READY
                        current.pending_unload = False
                        current.lifecycle_event.set()
                    else:
                        current.lifecycle_event.set()
                        self._records.pop(key, None)
            raise
        except Exception as exc:
            async with self._lock:
                current = self._records.get(key)
                if current is not None:
                    current.state = ModelResidencyState.FAILED
                    current.failure = self._safe_error(exc)
                    current.pending_unload = False
                    current.lifecycle_event.set()
            await self._publish(
                EventType.MODEL_UNLOAD_FAILED,
                record,
                request_id=request_id,
                application_id=application_id,
                operation_id=operation_id,
                capability="model_lifecycle",
            )
            raise
        async with self._lock:
            current = self._records.pop(key, None)
            if current is not None:
                current.lifecycle_event.set()
        await self._publish(
            EventType.MODEL_UNLOADED,
            record,
            request_id=request_id,
            application_id=application_id,
            operation_id=operation_id,
            capability="model_lifecycle",
        )
        return ModelLifecycleResult(
            operation_id=operation_id,
            runtime_instance_id=self.runtime_instance_id,
            model_id=manifest.model_id,
            runtime=runtime.name,
            previous_state=previous_state,
            active_usage_count=0,
            backend_operation_performed=True,
            joined_existing_operation=joined_existing_operation,
            reason="Model drained and unloaded." if drain else "Model unloaded.",
        )

    async def list_residencies(self) -> list[ModelResidencySnapshot]:
        async with self._lock:
            return sorted(
                (self._snapshot(record) for record in self._records.values()),
                key=lambda item: (item.runtime, item.model_id),
            )

    async def get_residency(self, model_id: str, *, runtime_name: str | None = None) -> ModelResidencySnapshot | None:
        async with self._lock:
            candidates = [
                record
                for key, record in self._records.items()
                if key.model_id == model_id and (runtime_name is None or key.runtime_name == runtime_name)
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda item: item.key.runtime_name)
            return self._snapshot(candidates[0])

    async def shutdown(self) -> None:
        """Stop new leases, finish shared loads, and unload tracked models once."""

        async with self._lock:
            self._closing = True
            tasks = [record.load_task for record in self._records.values() if record.load_task is not None]
        if tasks:
            await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)
        for snapshot in await self.list_residencies():
            key = ModelResidencyKey(snapshot.runtime, snapshot.model_id)
            async with self._lock:
                record = self._records.get(key)
            if record is None:
                continue
            try:
                await self.unload(record.runtime, record.manifest, drain=True)
            except Exception:
                # Shutdown is best-effort across heterogeneous backend adapters.
                continue

    async def _perform_load(
        self,
        record: _ResidencyRecord,
        *,
        request_id: str | None,
        application_id: str | None,
        capability: str | None = None,
    ) -> None:
        admission = None
        try:
            await self._publish(
                EventType.MODEL_LOAD_REQUESTED,
                record,
                request_id=request_id,
                application_id=application_id,
                operation_id=record.load_operation_id,
                capability=capability,
            )
            admission = await self.model_load_scheduler.acquire()
            await record.runtime.load_model(record.manifest)
        except BaseException as exc:
            async with self._lock:
                current = self._records.get(record.key)
                if current is not None:
                    current.state = ModelResidencyState.FAILED
                    current.failure = self._safe_error(exc)
                    current.load_task = None
                    current.idle_event.set()
            await self._publish(
                EventType.MODEL_LOAD_FAILED,
                record,
                request_id=request_id,
                application_id=application_id,
                operation_id=record.load_operation_id,
                capability=capability,
            )
            raise
        finally:
            if admission is not None:
                admission.release()
        async with self._lock:
            current = self._records.get(record.key)
            if current is not None:
                now = utc_now()
                current.state = ModelResidencyState.READY
                current.loaded_at = now
                current.last_used_at = now
                current.failure = None
                current.load_task = None
                current.idle_event.set()
                record = current

    async def _publish(
        self,
        event_type: EventType,
        record: _ResidencyRecord,
        *,
        request_id: str | None,
        application_id: str | None,
        operation_id: str | None = None,
        capability: str | None = None,
    ) -> None:
        application_id = application_id or application_id_var.get()
        client_instance_id = client_instance_id_var.get()
        payload: dict[str, object] = {
            "runtime_instance_id": self.runtime_instance_id,
            "model_id": record.key.model_id,
            "runtime": record.key.runtime_name,
            "state": record.state.value,
            "active_usage_count": record.active_usage_count,
        }
        if request_id:
            payload["request_id"] = request_id
        if operation_id:
            payload["operation_id"] = operation_id
        if capability:
            payload["capability"] = capability
            if capability == "model_lifecycle":
                payload["operation"] = "model.lifecycle"
        if application_id:
            payload["application_id"] = application_id
        if client_instance_id:
            payload["client_instance_id"] = client_instance_id
        await self.event_bus.publish(
            StreamEvent(type=event_type, scope=EventScope.REQUEST if request_id else EventScope.SYSTEM, payload=payload),
        )

    def _result(
        self,
        record: _ResidencyRecord,
        *,
        previous_state: ModelResidencyState | None,
        backend_operation: bool,
        joined: bool,
        reason: str,
    ) -> ModelLifecycleResult:
        return ModelLifecycleResult(
            operation_id=record.load_operation_id,
            runtime_instance_id=self.runtime_instance_id,
            model_id=record.key.model_id,
            runtime=record.key.runtime_name,
            previous_state=previous_state,
            current_state=record.state,
            active_usage_count=record.active_usage_count,
            backend_operation_performed=backend_operation,
            joined_existing_operation=joined,
            reason=reason,
        )

    @staticmethod
    def _snapshot(record: _ResidencyRecord) -> ModelResidencySnapshot:
        return ModelResidencySnapshot(
            model_id=record.key.model_id,
            runtime=record.key.runtime_name,
            state=record.state,
            load_started_at=record.load_started_at,
            loaded_at=record.loaded_at,
            last_used_at=record.last_used_at,
            active_usage_count=record.active_usage_count,
            pending_unload=record.pending_unload,
            failure=record.failure,
            load_attempt_count=record.load_attempt_count,
            joined_load_waiter_count=record.joined_load_waiter_count,
            estimated_memory_mb=record.manifest.estimated_memory_mb,
        )

    @staticmethod
    def _conflict_details(record: _ResidencyRecord) -> dict[str, object]:
        return {
            "model_id": record.key.model_id,
            "runtime": record.key.runtime_name,
            "state": record.state.value,
            "active_usage_count": record.active_usage_count,
        }

    @staticmethod
    def _safe_error(exc: BaseException) -> str:
        message = str(exc).strip()
        return f"{type(exc).__name__}: {message}" if message else type(exc).__name__

    @staticmethod
    def _consume_unobserved_task_exception(task: asyncio.Task[None]) -> None:
        """Retrieve a detached load failure after all waiters were cancelled."""

        if task.cancelled():
            return
        task.exception()
