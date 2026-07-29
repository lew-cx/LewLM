from __future__ import annotations

from types import SimpleNamespace

from lewlm.events.bus import EventBus
from lewlm.runtime.operations import LifecycleOperationManager, LifecycleOperationStatus
from lewlm.security.audit import AuditLogger


class _DeferredLifecycleRouter:
    def __init__(self) -> None:
        self.drain_started = False

    def route_lifecycle(self, model_id: str):
        return None, SimpleNamespace(name="deferred-runtime"), None

    async def unload_model_lifecycle(self, model_id: str, **kwargs):
        self.drain_started = True
        raise AssertionError("A pre-start cancelled operation must not reach the backend.")


async def test_immediate_operation_cancellation_reaches_terminal_state(temp_settings) -> None:
    router = _DeferredLifecycleRouter()
    manager = LifecycleOperationManager(
        runtime_instance_id="runtime-test",
        model_router=router,  # type: ignore[arg-type]
        event_bus=EventBus(),
        audit_logger=AuditLogger(temp_settings),
    )

    created = await manager.submit_drain(
        "model-a",
        timeout_seconds=2,
        application_id="operator",
        client_instance_id="operator-1",
    )
    cancelled = await manager.cancel(created.operation_id)

    assert cancelled.status == LifecycleOperationStatus.CANCELLED
    assert cancelled.completed_at is not None
    assert router.drain_started is False
    assert (await manager.get(created.operation_id)).status == LifecycleOperationStatus.CANCELLED
