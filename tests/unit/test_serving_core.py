from __future__ import annotations

from lewlm.core.contracts import CapabilityName
from lewlm.core.serving_core import (
    ServingCore,
    ServingPhase,
    ServingQueueType,
    ServingRuntimeAdapterKind,
    describe_serving_runtime_adapter,
)
from lewlm.runtime.scheduler import FrontierBatchMetrics, RuntimeRequestAdmission, RuntimeRequestScheduler


class _RequestScopedRuntime:
    name = "request_scoped_runtime"

    def supports_continuous_batching(self, capability: CapabilityName) -> bool:
        return False

    def supports_chunked_prefill(self, capability: CapabilityName) -> bool:
        return False

    def supports_prefill_isolation(self, capability: CapabilityName) -> bool:
        return False


class _BackendBatchRuntime(_RequestScopedRuntime):
    name = "backend_batch_runtime"

    def supports_continuous_batching(self, capability: CapabilityName) -> bool:
        return capability in {CapabilityName.CHAT, CapabilityName.STREAMING}

    def supports_chunked_prefill(self, capability: CapabilityName) -> bool:
        return capability == CapabilityName.CHAT

    def supports_prefill_isolation(self, capability: CapabilityName) -> bool:
        return capability == CapabilityName.CHAT


class _LewLMOwnedBatchRuntime(_BackendBatchRuntime):
    name = "lewlm_owned_batch_runtime"

    def continuous_batching_ownership(self, capability: CapabilityName) -> str:
        return "lewlm_owned" if capability in {CapabilityName.CHAT, CapabilityName.STREAMING} else "unsupported"


def test_describe_serving_runtime_adapter_reports_backend_batch_support() -> None:
    adapter = describe_serving_runtime_adapter(
        runtime=_BackendBatchRuntime(),
        capability=CapabilityName.CHAT,
    )

    assert adapter.kind == ServingRuntimeAdapterKind.BACKEND_NATIVE_BATCH
    assert adapter.backend_batching is True
    assert adapter.chunked_prefill is True
    assert adapter.prefill_isolation is True


def test_describe_serving_runtime_adapter_reports_lewlm_owned_batch_support() -> None:
    adapter = describe_serving_runtime_adapter(
        runtime=_LewLMOwnedBatchRuntime(),
        capability=CapabilityName.STREAMING,
    )

    assert adapter.kind == ServingRuntimeAdapterKind.LEWLM_OWNED_BATCH
    assert adapter.backend_batching is False
    assert "LewLM owns persistent continuous-batch admission" in adapter.notes[0]


def test_serving_core_tracks_sequence_lifecycle_and_cancellation() -> None:
    scheduler = RuntimeRequestScheduler(
        max_concurrent_requests=1,
        queue_limit=1,
        queue_timeout_seconds=1,
    )
    core = ServingCore(recent_sequence_limit=2)
    adapter = describe_serving_runtime_adapter(
        runtime=_BackendBatchRuntime(),
        capability=CapabilityName.STREAMING,
    )
    core.register_sequence(
        request_id="req-1",
        requested_model_id="requested-model",
        model_id="resolved-model",
        runtime_name="backend_batch_runtime",
        capability=CapabilityName.STREAMING,
        runtime_adapter=adapter,
        streaming=True,
        queue_lane="prefill",
        prefill_heavy=True,
        decode_priority_requested=False,
        prefill_isolation_requested=True,
        prompt_token_estimate=256,
        chunk_count=2,
    )

    core.record_batch_metrics(
        request_id="req-1",
        batch_metrics=FrontierBatchMetrics(
            batch_size=2,
            batch_position=1,
            batch_utilization=1.0,
            queue_delay_seconds=0.012,
            batch_window_milliseconds=25,
            max_batch_size=2,
        ),
    )
    core.record_queue(request_id="req-1", queue_type=ServingQueueType.BATCH_WINDOW, wait_seconds=0.012)
    core.record_queue(request_id="req-1", queue_type=ServingQueueType.MODEL_LOAD, wait_seconds=0.005)
    core.admit_sequence(
        request_id="req-1",
        admission=RuntimeRequestAdmission(
            scheduler=scheduler,
            was_queued=True,
            wait_seconds=0.008,
            scheduling_lane="prefill",
            prefill_isolated=True,
        ),
    )
    core.transition_phase(request_id="req-1", phase=ServingPhase.MODEL_LOADING, detail="model_loading_started")
    core.transition_phase(request_id="req-1", phase=ServingPhase.PREFILL, detail="prefill_started")
    core.transition_phase(request_id="req-1", phase=ServingPhase.DECODE, detail="decode_started")
    core.request_cancellation(request_id="req-1", reason="stream_consumer_closed")
    core.complete_sequence(request_id="req-1")

    snapshot = core.snapshot()

    assert snapshot.active_sequence_count == 0
    assert snapshot.total_sequences_started == 1
    assert snapshot.total_sequences_completed == 1
    assert snapshot.total_cancellation_requests == 1
    assert snapshot.recent_sequence_count == 1
    sequence = snapshot.recent_sequences[0]
    assert sequence.phase == ServingPhase.COMPLETED
    assert sequence.streaming is True
    assert sequence.cancellation_requested is True
    assert sequence.cancellation_reason == "stream_consumer_closed"
    assert sequence.batch.batched is True
    assert sequence.batch.backend_owned is True
    assert sequence.batch.queue_delay_milliseconds == 12
    assert sequence.queue_residency_milliseconds == 17
    assert [item.queue_type.value for item in sequence.queue_residencies] == ["batch_window", "model_load"]
    assert [item.phase.value for item in sequence.phase_history] == [
        "created",
        "queued",
        "queued",
        "admitted",
        "model_loading",
        "prefill",
        "decode",
        "completed",
    ]
