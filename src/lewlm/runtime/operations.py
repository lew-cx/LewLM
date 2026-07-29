"""Observable asynchronous operations for shared-runtime lifecycle work."""

from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from lewlm.core.contracts import utc_now
from lewlm.core.errors import JobNotFoundError, LewLMError, ModelLifecycleConflictError, RuntimeUnavailableError
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.routing.service import ModelRouter
from lewlm.runtime.residency import ModelLifecycleResult
from lewlm.security.audit import AuditLogger


class LifecycleOperationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LifecycleOperationKind(str, Enum):
    DRAIN_MODEL = "drain_model"


class LifecycleOperationError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class LifecycleOperationRecord(BaseModel):
    operation_id: str
    operation: LifecycleOperationKind
    status: LifecycleOperationStatus
    runtime_instance_id: str
    model_id: str
    runtime: str | None = None
    application_id: str | None = None
    client_instance_id: str | None = None
    idempotency_key: str | None = None
    idempotent_replay: bool = False
    timeout_seconds: float
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: ModelLifecycleResult | None = None
    error: LifecycleOperationError | None = None


class LifecycleOperationManager:
    """Own background drain tasks for one runtime service container."""

    def __init__(
        self,
        *,
        runtime_instance_id: str,
        model_router: ModelRouter,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ) -> None:
        self.runtime_instance_id = runtime_instance_id
        self.model_router = model_router
        self.event_bus = event_bus
        self.audit_logger = audit_logger
        self._records: dict[str, LifecycleOperationRecord] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._idempotency: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()
        self._closing = False

    async def submit_drain(
        self,
        model_id: str,
        *,
        timeout_seconds: float,
        application_id: str | None,
        client_instance_id: str | None,
        idempotency_key: str | None = None,
    ) -> LifecycleOperationRecord:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        # Resolve eagerly so unknown or unsupported models fail before a 202.
        _, runtime, _ = self.model_router.route_lifecycle(model_id)
        identity = application_id or "anonymous"
        async with self._lock:
            if self._closing:
                raise RuntimeUnavailableError("The lifecycle operation manager is shutting down.")
            if idempotency_key:
                existing_id = self._idempotency.get((identity, idempotency_key))
                if existing_id is not None:
                    existing = self._records[existing_id]
                    if (
                        existing.model_id != model_id
                        or existing.operation != LifecycleOperationKind.DRAIN_MODEL
                        or existing.timeout_seconds != timeout_seconds
                    ):
                        raise ModelLifecycleConflictError(
                            "The idempotency key is already bound to a different lifecycle operation.",
                            details={
                                "idempotency_key": idempotency_key,
                                "operation_id": existing_id,
                                "existing_model_id": existing.model_id,
                                "existing_timeout_seconds": existing.timeout_seconds,
                                "requested_model_id": model_id,
                                "requested_timeout_seconds": timeout_seconds,
                            },
                        )
                    return existing.model_copy(update={"idempotent_replay": True})
            operation_id = str(uuid4())
            record = LifecycleOperationRecord(
                operation_id=operation_id,
                operation=LifecycleOperationKind.DRAIN_MODEL,
                status=LifecycleOperationStatus.PENDING,
                runtime_instance_id=self.runtime_instance_id,
                model_id=model_id,
                runtime=runtime.name,
                application_id=application_id,
                client_instance_id=client_instance_id,
                idempotency_key=idempotency_key,
                timeout_seconds=timeout_seconds,
            )
            self._records[operation_id] = record
            if idempotency_key:
                self._idempotency[(identity, idempotency_key)] = operation_id
            task = asyncio.create_task(self._run_drain(operation_id), name=f"lewlm-drain-{model_id}-{operation_id}")
            task.add_done_callback(self._consume_task_result)
            self._tasks[operation_id] = task
            snapshot = record.model_copy(deep=True)
        await self._publish(snapshot)
        return snapshot

    async def get(self, operation_id: str) -> LifecycleOperationRecord:
        async with self._lock:
            record = self._records.get(operation_id)
            if record is None:
                raise JobNotFoundError(
                    "Lifecycle operation was not found.",
                    details={"operation_id": operation_id},
                )
            return record.model_copy(deep=True)

    async def submit_drain_and_wait(
        self,
        model_id: str,
        *,
        timeout_seconds: float,
        application_id: str | None,
        client_instance_id: str | None,
        idempotency_key: str | None = None,
    ) -> LifecycleOperationRecord:
        """Run a recorded drain to a terminal state for synchronous embedding."""

        record = await self.submit_drain(
            model_id,
            timeout_seconds=timeout_seconds,
            application_id=application_id,
            client_instance_id=client_instance_id,
            idempotency_key=idempotency_key,
        )
        async with self._lock:
            task = self._tasks.get(record.operation_id)
        if task is not None:
            await asyncio.shield(task)
        return await self.get(record.operation_id)

    async def cancel(self, operation_id: str) -> LifecycleOperationRecord:
        async with self._lock:
            record = self._records.get(operation_id)
            if record is None:
                raise JobNotFoundError(
                    "Lifecycle operation was not found.",
                    details={"operation_id": operation_id},
                )
            task = self._tasks.get(operation_id)
            if task is None or record.status in {
                LifecycleOperationStatus.SUCCEEDED,
                LifecycleOperationStatus.FAILED,
                LifecycleOperationStatus.CANCELLED,
            }:
                return record.model_copy(deep=True)
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await self._finish_cancelled_if_active(operation_id)
        return await self.get(operation_id)

    async def shutdown(self) -> None:
        async with self._lock:
            self._closing = True
            tasks = list(self._tasks.values())
            for task in tasks:
                if not task.done():
                    task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for operation_id in tuple(self._tasks):
            await self._finish_cancelled_if_active(operation_id)

    async def _finish_cancelled_if_active(self, operation_id: str) -> None:
        async with self._lock:
            record = self._records.get(operation_id)
            active = record is not None and record.status in {
                LifecycleOperationStatus.PENDING,
                LifecycleOperationStatus.RUNNING,
            }
        if active:
            await self._finish(operation_id, status=LifecycleOperationStatus.CANCELLED)

    async def _run_drain(self, operation_id: str) -> None:
        async with self._lock:
            record = self._records[operation_id]
            record.status = LifecycleOperationStatus.RUNNING
            record.started_at = utc_now()
            snapshot = record.model_copy(deep=True)
        await self._publish(snapshot)
        self._audit(snapshot, "running")
        try:
            _, result = await self.model_router.unload_model_lifecycle(
                record.model_id,
                drain=True,
                timeout_seconds=record.timeout_seconds,
                request_id=operation_id,
                application_id=record.application_id,
            )
        except asyncio.CancelledError:
            await self._finish(operation_id, status=LifecycleOperationStatus.CANCELLED)
            raise
        except Exception as exc:
            error = self._error_from_exception(exc)
            await self._finish(operation_id, status=LifecycleOperationStatus.FAILED, error=error)
        else:
            await self._finish(operation_id, status=LifecycleOperationStatus.SUCCEEDED, result=result)

    async def _finish(
        self,
        operation_id: str,
        *,
        status: LifecycleOperationStatus,
        result: ModelLifecycleResult | None = None,
        error: LifecycleOperationError | None = None,
    ) -> None:
        async with self._lock:
            record = self._records[operation_id]
            record.status = status
            record.completed_at = utc_now()
            record.result = result
            record.error = error
            self._tasks.pop(operation_id, None)
            snapshot = record.model_copy(deep=True)
        await self._publish(snapshot)
        self._audit(snapshot, status.value)

    async def _publish(self, record: LifecycleOperationRecord) -> None:
        payload = {
            "operation": "model.drain",
            "operation_id": record.operation_id,
            "status": record.status.value,
            "runtime_instance_id": record.runtime_instance_id,
            "model_id": record.model_id,
            "runtime": record.runtime,
            "application_id": record.application_id,
            "client_instance_id": record.client_instance_id,
        }
        if record.error is not None:
            payload["error"] = record.error.model_dump(mode="json")
        await self.event_bus.publish(
            StreamEvent(type=EventType.OPERATION_PROGRESS, scope=EventScope.SYSTEM, payload=payload),
        )

    def _audit(self, record: LifecycleOperationRecord, outcome: str) -> None:
        self.audit_logger.record(
            action="lifecycle.drain_operation",
            outcome=outcome,
            actor=record.application_id or "api",
            details={
                "operation_id": record.operation_id,
                "model_id": record.model_id,
                "runtime": record.runtime,
                "application_id": record.application_id,
                "client_instance_id": record.client_instance_id,
                "status": record.status.value,
                "error": record.error.model_dump(mode="json") if record.error is not None else None,
            },
        )

    @staticmethod
    def _error_from_exception(exc: Exception) -> LifecycleOperationError:
        if isinstance(exc, LewLMError):
            return LifecycleOperationError(code=exc.code, message=str(exc), details=dict(exc.details))
        return LifecycleOperationError(code="lifecycle_operation_failed", message=str(exc) or type(exc).__name__)

    @staticmethod
    def _consume_task_result(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        task.exception()
