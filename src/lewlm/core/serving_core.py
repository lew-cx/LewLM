"""Stable serving-core state tracking for chat and streaming requests."""

from __future__ import annotations

from collections import Counter
from enum import Enum
from threading import Lock
from typing import Literal

from pydantic import BaseModel, Field

from lewlm.core.contracts import CapabilityName, RuntimeContract, utc_now
from lewlm.runtime.scheduler import FrontierBatchMetrics, RuntimeRequestAdmission


class ServingPhase(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    ADMITTED = "admitted"
    MODEL_LOADING = "model_loading"
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETED = "completed"
    FAILED = "failed"


class ServingQueueType(str, Enum):
    BATCH_WINDOW = "batch_window"
    RUNTIME_REQUEST = "runtime_request"
    MODEL_LOAD = "model_load"


class ServingRuntimeAdapterKind(str, Enum):
    REQUEST_SCOPED = "request_scoped"
    LEWLM_OWNED_BATCH = "lewlm_owned_batch"
    BACKEND_NATIVE_BATCH = "backend_native_batch"


class StreamingOwner(str, Enum):
    NONE = "none"
    LEWLM = "lewlm"


class ServingRuntimeAdapter(BaseModel):
    runtime_name: str
    capability: str
    kind: ServingRuntimeAdapterKind
    backend_batching: bool = False
    chunked_prefill: bool = False
    prefill_isolation: bool = False
    notes: list[str] = Field(default_factory=list)


class ServingPhaseTransition(BaseModel):
    phase: ServingPhase
    detail: str | None = None
    entered_at: str = Field(default_factory=lambda: utc_now().isoformat())


class ServingQueueResidency(BaseModel):
    queue_type: ServingQueueType
    wait_milliseconds: int


class ServingAdmissionState(BaseModel):
    queue_lane: Literal["decode", "prefill"] | None = None
    prefill_heavy: bool = False
    decode_priority_requested: bool = False
    decode_priority_active: bool = False
    prefill_isolation_requested: bool = False
    prefill_isolation_active: bool = False
    prompt_token_estimate: int | None = None
    chunk_count: int | None = None


class ServingBatchState(BaseModel):
    batched: bool = False
    backend_owned: bool = False
    batch_size: int = 1
    batch_position: int = 0
    batch_window_milliseconds: int = 0
    queue_delay_milliseconds: int = 0


class ServingSequenceSnapshot(BaseModel):
    request_id: str
    requested_model_id: str | None = None
    model_id: str
    runtime_name: str
    capability: str
    streaming: bool = False
    streaming_owner: StreamingOwner = StreamingOwner.NONE
    runtime_adapter: ServingRuntimeAdapter
    phase: ServingPhase = ServingPhase.CREATED
    active: bool = True
    cancellation_requested: bool = False
    cancellation_reason: str | None = None
    failure: str | None = None
    queue_residency_milliseconds: int = 0
    queue_residencies: list[ServingQueueResidency] = Field(default_factory=list)
    admission: ServingAdmissionState = Field(default_factory=ServingAdmissionState)
    batch: ServingBatchState = Field(default_factory=ServingBatchState)
    phase_history: list[ServingPhaseTransition] = Field(default_factory=list)


class ServingCoreSnapshot(BaseModel):
    version: Literal["v1"] = "v1"
    total_sequences_started: int = 0
    total_sequences_completed: int = 0
    total_sequences_failed: int = 0
    total_cancellation_requests: int = 0
    active_sequence_count: int = 0
    active_stream_count: int = 0
    recent_sequence_count: int = 0
    active_phase_counts: dict[str, int] = Field(default_factory=dict)
    active_sequences: list[ServingSequenceSnapshot] = Field(default_factory=list)
    recent_sequences: list[ServingSequenceSnapshot] = Field(default_factory=list)


ServingRuntimeAdapter.model_rebuild()
ServingPhaseTransition.model_rebuild()
ServingQueueResidency.model_rebuild()
ServingAdmissionState.model_rebuild()
ServingBatchState.model_rebuild()
ServingSequenceSnapshot.model_rebuild()
ServingCoreSnapshot.model_rebuild()


def describe_serving_runtime_adapter(
    *,
    runtime: RuntimeContract,
    capability: CapabilityName,
) -> ServingRuntimeAdapter:
    ownership = continuous_batching_ownership(runtime=runtime, capability=capability)
    backend_batching = ownership == "backend_native"
    chunked_prefill = capability in {CapabilityName.CHAT, CapabilityName.STREAMING} and runtime.supports_chunked_prefill(
        capability,
    )
    prefill_isolation = (
        capability in {CapabilityName.CHAT, CapabilityName.STREAMING} and runtime.supports_prefill_isolation(capability)
    )
    kind = (
        ServingRuntimeAdapterKind.LEWLM_OWNED_BATCH
        if ownership == "lewlm_owned"
        else ServingRuntimeAdapterKind.BACKEND_NATIVE_BATCH
        if backend_batching
        else ServingRuntimeAdapterKind.REQUEST_SCOPED
    )
    notes = [
        (
            "LewLM owns persistent continuous-batch admission on this runtime path while delegating token generation to the runtime backend."
            if ownership == "lewlm_owned"
            else "LewLM tracks serving state while the runtime retains backend-native batch execution details."
            if backend_batching
            else "LewLM tracks request-scoped serving state for this runtime path."
        ),
    ]
    if chunked_prefill:
        notes.append("Runtime advertises chunked prefill support.")
    if prefill_isolation:
        notes.append("Runtime advertises prefill-isolation support.")
    return ServingRuntimeAdapter(
        runtime_name=runtime.name,
        capability=capability.value,
        kind=kind,
        backend_batching=backend_batching,
        chunked_prefill=chunked_prefill,
        prefill_isolation=prefill_isolation,
        notes=notes,
    )


def continuous_batching_ownership(
    *,
    runtime: RuntimeContract,
    capability: CapabilityName,
) -> str:
    ownership_resolver = getattr(runtime, "continuous_batching_ownership", None)
    if callable(ownership_resolver):
        ownership = ownership_resolver(capability)
        if ownership in {"lewlm_owned", "backend_native", "unsupported"}:
            return ownership
    return (
        "backend_native"
        if capability in {CapabilityName.CHAT, CapabilityName.STREAMING}
        and runtime.supports_continuous_batching(capability)
        else "unsupported"
    )


class ServingCore:
    """Track serving lifecycle independently of any one backend batch API."""

    def __init__(self, *, recent_sequence_limit: int = 20) -> None:
        self.recent_sequence_limit = max(1, recent_sequence_limit)
        self._lock = Lock()
        self._active: dict[str, ServingSequenceSnapshot] = {}
        self._recent: list[ServingSequenceSnapshot] = []
        self._total_sequences_started = 0
        self._total_sequences_completed = 0
        self._total_sequences_failed = 0
        self._total_cancellation_requests = 0

    def register_sequence(
        self,
        *,
        request_id: str,
        requested_model_id: str | None,
        model_id: str,
        runtime_name: str,
        capability: CapabilityName,
        runtime_adapter: ServingRuntimeAdapter,
        streaming: bool,
        queue_lane: Literal["decode", "prefill"],
        prefill_heavy: bool,
        decode_priority_requested: bool,
        prefill_isolation_requested: bool,
        prompt_token_estimate: int,
        chunk_count: int,
    ) -> None:
        sequence = ServingSequenceSnapshot(
            request_id=request_id,
            requested_model_id=requested_model_id,
            model_id=model_id,
            runtime_name=runtime_name,
            capability=capability.value,
            streaming=streaming,
            streaming_owner=StreamingOwner.LEWLM if streaming else StreamingOwner.NONE,
            runtime_adapter=runtime_adapter,
            admission=ServingAdmissionState(
                queue_lane=queue_lane,
                prefill_heavy=prefill_heavy,
                decode_priority_requested=decode_priority_requested,
                prefill_isolation_requested=prefill_isolation_requested,
                prompt_token_estimate=prompt_token_estimate,
                chunk_count=chunk_count,
            ),
            phase_history=[
                ServingPhaseTransition(phase=ServingPhase.CREATED, detail="request_registered"),
            ],
        )
        with self._lock:
            self._total_sequences_started += 1
            self._active[request_id] = sequence

    def record_batch_metrics(self, *, request_id: str, batch_metrics: FrontierBatchMetrics | None) -> None:
        if batch_metrics is None:
            return
        with self._lock:
            sequence = self._lookup_sequence_locked(request_id)
            if sequence is None:
                return
            sequence.batch = ServingBatchState(
                batched=batch_metrics.batch_size > 1,
                backend_owned=sequence.runtime_adapter.backend_batching,
                batch_size=batch_metrics.batch_size,
                batch_position=batch_metrics.batch_position,
                batch_window_milliseconds=batch_metrics.batch_window_milliseconds,
                queue_delay_milliseconds=max(int(round(batch_metrics.queue_delay_seconds * 1000)), 0),
            )

    def record_queue(
        self,
        *,
        request_id: str,
        queue_type: ServingQueueType,
        wait_seconds: float,
    ) -> None:
        wait_milliseconds = max(int(round(max(wait_seconds, 0.0) * 1000)), 0)
        with self._lock:
            sequence = self._lookup_sequence_locked(request_id)
            if sequence is None:
                return
            sequence.queue_residencies.append(
                ServingQueueResidency(
                    queue_type=queue_type,
                    wait_milliseconds=wait_milliseconds,
                ),
            )
            sequence.queue_residency_milliseconds += wait_milliseconds
            self._transition_locked(sequence, ServingPhase.QUEUED, detail=queue_type.value)

    def admit_sequence(self, *, request_id: str, admission: RuntimeRequestAdmission) -> None:
        with self._lock:
            sequence = self._lookup_sequence_locked(request_id)
            if sequence is None:
                return
            sequence.admission.queue_lane = admission.scheduling_lane
            sequence.admission.decode_priority_active = admission.decode_priority_applied
            sequence.admission.prefill_isolation_active = admission.prefill_isolated
            self._transition_locked(sequence, ServingPhase.ADMITTED, detail=admission.scheduling_lane)

    def transition_phase(
        self,
        *,
        request_id: str,
        phase: ServingPhase,
        detail: str | None = None,
    ) -> None:
        with self._lock:
            sequence = self._lookup_sequence_locked(request_id)
            if sequence is None:
                return
            self._transition_locked(sequence, phase, detail=detail)

    def request_cancellation(self, *, request_id: str, reason: str) -> None:
        with self._lock:
            sequence = self._lookup_sequence_locked(request_id)
            if sequence is None or sequence.cancellation_requested:
                return
            sequence.cancellation_requested = True
            sequence.cancellation_reason = reason
            self._total_cancellation_requests += 1

    def complete_sequence(self, *, request_id: str) -> None:
        with self._lock:
            sequence = self._lookup_sequence_locked(request_id)
            if sequence is None:
                return
            self._transition_locked(sequence, ServingPhase.COMPLETED, detail="request_completed")
            sequence.active = False
            self._total_sequences_completed += 1
            self._finalize_locked(request_id, sequence)

    def fail_sequence(self, *, request_id: str, error: str) -> None:
        with self._lock:
            sequence = self._lookup_sequence_locked(request_id)
            if sequence is None:
                return
            sequence.failure = error
            self._transition_locked(sequence, ServingPhase.FAILED, detail="request_failed")
            sequence.active = False
            self._total_sequences_failed += 1
            self._finalize_locked(request_id, sequence)

    def sequence_metadata(self, request_id: str) -> dict[str, object] | None:
        with self._lock:
            sequence = self._lookup_sequence_locked(request_id)
            if sequence is None:
                return None
            return sequence.model_dump(mode="json")

    def snapshot(self) -> ServingCoreSnapshot:
        with self._lock:
            active_sequences = [sequence.model_copy(deep=True) for sequence in self._active.values()]
            recent_sequences = [sequence.model_copy(deep=True) for sequence in self._recent]
            active_phase_counts = Counter(sequence.phase.value for sequence in active_sequences)
            return ServingCoreSnapshot(
                total_sequences_started=self._total_sequences_started,
                total_sequences_completed=self._total_sequences_completed,
                total_sequences_failed=self._total_sequences_failed,
                total_cancellation_requests=self._total_cancellation_requests,
                active_sequence_count=len(active_sequences),
                active_stream_count=sum(1 for sequence in active_sequences if sequence.streaming),
                recent_sequence_count=len(recent_sequences),
                active_phase_counts=dict(active_phase_counts),
                active_sequences=active_sequences,
                recent_sequences=recent_sequences,
            )

    def _lookup_sequence_locked(self, request_id: str) -> ServingSequenceSnapshot | None:
        sequence = self._active.get(request_id)
        if sequence is not None:
            return sequence
        return next((item for item in self._recent if item.request_id == request_id), None)

    @staticmethod
    def _transition_locked(
        sequence: ServingSequenceSnapshot,
        phase: ServingPhase,
        *,
        detail: str | None,
    ) -> None:
        last_transition = sequence.phase_history[-1] if sequence.phase_history else None
        if last_transition is not None and last_transition.phase == phase and last_transition.detail == detail:
            sequence.phase = phase
            return
        sequence.phase = phase
        sequence.phase_history.append(ServingPhaseTransition(phase=phase, detail=detail))

    def _finalize_locked(self, request_id: str, sequence: ServingSequenceSnapshot) -> None:
        self._active.pop(request_id, None)
        self._recent.append(sequence.model_copy(deep=True))
        if len(self._recent) > self.recent_sequence_limit:
            del self._recent[: len(self._recent) - self.recent_sequence_limit]
