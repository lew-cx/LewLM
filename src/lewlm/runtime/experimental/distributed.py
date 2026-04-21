"""Experimental multi-host coordinator/worker runtime helpers."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import timedelta
from typing import Any, Protocol
from urllib.parse import urljoin
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    CapabilityName,
    GenerateMessage,
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    ModelModality,
    RuntimeAffinity,
    RuntimeEstimate,
    utc_now,
)
from lewlm.core.errors import AuthenticationError, ConfigurationError, RuntimeUnavailableError
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.runtime.base import ManagedTextRuntime
from lewlm.runtime.experimental.distributed_utils import (
    aggregate_execution_metrics as _aggregate_execution_metrics,
    coerce_float as _coerce_float,
    coerce_fraction as _coerce_fraction,
    coerce_int as _coerce_int,
    dominant_bottleneck as _dominant_bottleneck,
    parse_timestamp as _parse_timestamp,
    selected_worker_profiles as _selected_worker_profiles,
    stage_assignment_profile as _stage_assignment_profile,
    stage_execution_profile as _stage_execution_profile,
    urlsafe_b64decode as _urlsafe_b64decode,
    weighted_layer_spans as _weighted_layer_spans,
    worker_profile as _worker_profile,
)
from lewlm.security.audit import AuditLogger
from lewlm.storage.metadata import MetadataStore

_CLUSTER_WORKERS_KEY = "cluster.workers"
_CLUSTER_PLANS_KEY = "cluster.plans"
_CLUSTER_WORKER_SESSION_KEY = "cluster.worker_session"
_DISTRIBUTED_BOUNDARY_NOTE = (
    "Experimental pipeline-parallel proof path only; tensor-parallel execution and backend-native shard kernels "
    "remain follow-up work."
)


class ClusterTransport(Protocol):
    """Transport used by the coordinator to contact workers or peer coordinators."""

    async def request_json(
        self,
        *,
        method: str,
        base_url: str,
        path: str,
        payload: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Issue a JSON request and return the decoded JSON payload."""


class HttpClusterTransport:
    """Simple JSON transport implemented with the Python standard library."""

    async def request_json(
        self,
        *,
        method: str,
        base_url: str,
        path: str,
        payload: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._request_json,
            method=method,
            base_url=base_url,
            path=path,
            payload=payload,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )

    def _request_json(
        self,
        *,
        method: str,
        base_url: str,
        path: str,
        payload: dict[str, Any] | None,
        headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        request_headers = {"accept": "application/json", **dict(headers or {})}
        data = None
        if payload is not None:
            request_headers.setdefault("content-type", "application/json")
            data = json.dumps(payload).encode("utf-8")
        url = urljoin(f"{base_url.rstrip('/')}/", path.lstrip("/"))
        request = urllib.request.Request(
            url,
            method=method.upper(),
            headers=request_headers,
            data=data,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds or 30.0) as response:
                response_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            response_text = exc.read().decode("utf-8")
            message = response_text
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
                error_payload = parsed["error"]
                raise RuntimeUnavailableError(
                    str(error_payload.get("message") or "Cluster peer request failed."),
                    details={"status_code": exc.code, "peer_error": error_payload},
                ) from exc
            raise RuntimeUnavailableError(
                "Cluster peer request failed.",
                details={"status_code": exc.code, "body": message},
            ) from exc
        except OSError as exc:
            raise RuntimeUnavailableError(
                "Cluster peer is unreachable.",
                details={"base_url": base_url, "path": path, "reason": str(exc)},
            ) from exc
        if not response_text:
            return {}
        try:
            payload_obj = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeUnavailableError(
                "Cluster peer returned invalid JSON.",
                details={"base_url": base_url, "path": path},
            ) from exc
        if not isinstance(payload_obj, dict):
            raise RuntimeUnavailableError(
                "Cluster peer returned an unexpected response shape.",
                details={"base_url": base_url, "path": path},
            )
        return payload_obj


class WorkerEnrollmentToken(BaseModel):
    cluster_name: str
    issued_at: str
    expires_at: str
    worker_name: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    nonce: str


class ClusterWorkerRecord(BaseModel):
    worker_id: str
    worker_name: str
    endpoint: str
    capabilities: list[str] = Field(default_factory=list)
    status: str
    enrolled_at: str
    last_heartbeat_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_token: str | None = None
    last_error: str | None = None


class ClusterWorkerSession(BaseModel):
    worker_id: str
    worker_name: str
    coordinator_url: str
    endpoint: str
    session_token: str
    enrolled_at: str


class ClusterWorkerProfile(BaseModel):
    worker_id: str
    worker_name: str
    endpoint: str
    relative_weight: float = 1.0
    selection_score: float = 1.0
    network_latency_ms: float = 0.0
    network_bandwidth_gbps: float = 0.0
    max_batch_tokens: int = 0
    prefetch_tokens: int = 0
    overlap_ratio: float = 0.0


class ClusterStageAssignment(BaseModel):
    stage_index: int
    stage_count: int
    stage_name: str
    worker_id: str
    worker_name: str
    endpoint: str
    start_layer: int
    end_layer: int
    relative_weight: float = 1.0
    selection_score: float = 1.0
    target_batch_tokens: int = 0
    prefetch_tokens: int = 0
    network_latency_ms: float = 0.0
    network_bandwidth_gbps: float = 0.0
    overlap_ratio: float = 0.0
    expected_compute_seconds: float = 0.0
    expected_network_seconds: float = 0.0
    expected_queue_seconds: float = 0.0
    overlap_credit_seconds: float = 0.0
    expected_stage_seconds: float = 0.0
    expected_utilization: float = 0.0


class DistributedExecutionPlan(BaseModel):
    model_id: str
    runtime_affinity: str
    stage_count: int
    required_workers: int
    total_layers: int
    recovery_count: int = 0
    assignments: list[ClusterStageAssignment] = Field(default_factory=list)
    worker_profiles: list[ClusterWorkerProfile] = Field(default_factory=list)
    scheduling: dict[str, int | float | str | bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    last_execution_at: str | None = None


class ClusterStatus(BaseModel):
    role: str
    enabled: bool
    cluster_name: str
    node_name: str
    coordinator_url: str | None = None
    public_base_url: str | None = None
    ready_worker_count: int = 0
    stale_worker_count: int = 0
    plan_count: int = 0
    worker_heartbeat_timeout_seconds: int
    worker_session: ClusterWorkerSession | None = None
    workers: list[ClusterWorkerRecord] = Field(default_factory=list)
    latest_execution_metrics: dict[str, int | float | str | bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ClusterIssueTokenResponse(BaseModel):
    cluster_name: str
    token: str
    expires_at: str
    capabilities: list[str] = Field(default_factory=list)
    worker_name: str | None = None


class ClusterEnrollWorkerRequest(BaseModel):
    token: str
    worker_name: str | None = None
    endpoint: str
    capabilities: list[str] = Field(default_factory=lambda: [CapabilityName.CHAT.value])
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClusterEnrollWorkerResponse(BaseModel):
    worker: ClusterWorkerRecord
    coordinator_url: str
    session_token: str


class ClusterHeartbeatRequest(BaseModel):
    worker_id: str
    session_token: str


class ClusterStageRequest(BaseModel):
    worker_id: str
    model_id: str
    stage: ClusterStageAssignment
    prompt: str
    pipeline: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class ClusterStageResponse(BaseModel):
    trace: list[dict[str, Any]] = Field(default_factory=list)
    output_text: str | None = None
    metrics: dict[str, int | float | str | bool] = Field(default_factory=dict)


class DistributedClusterService:
    """Coordinator/worker registry plus experimental distributed execution helpers."""

    def __init__(
        self,
        *,
        settings: LewLMSettings,
        metadata_store: MetadataStore,
        event_bus: EventBus,
        audit_logger: AuditLogger,
        transport: ClusterTransport | None = None,
    ) -> None:
        self.settings = settings
        self.metadata_store = metadata_store
        self.event_bus = event_bus
        self.audit_logger = audit_logger
        self.transport: ClusterTransport = transport or HttpClusterTransport()
        self._last_execution_metrics: dict[str, int | float | str | bool] = {}

    def set_transport(self, transport: ClusterTransport) -> None:
        self.transport = transport

    def status(self) -> ClusterStatus:
        workers = self._refreshed_workers()
        public_workers = [public_worker_record(worker) for worker in workers]
        ready_worker_count = sum(1 for worker in workers if worker.status == "ready")
        stale_worker_count = sum(1 for worker in workers if worker.status == "stale")
        plans = self._stored_plans()
        worker_session = self.worker_session()
        notes = [_DISTRIBUTED_BOUNDARY_NOTE]
        if self.settings.cluster_role == "standalone":
            notes.append("Cluster coordination is disabled until `cluster_role` is set to `coordinator` or `worker`.")
        elif self.settings.cluster_role == "coordinator" and self.settings.cluster_enrollment_secret is None:
            notes.append("Set `LEWLM_CLUSTER_ENROLLMENT_SECRET` before issuing worker enrollment tokens.")
        elif self.settings.cluster_role == "coordinator" and ready_worker_count < 2:
            notes.append("Enroll at least two ready workers before routing distributed experimental models.")
        return ClusterStatus(
            role=self.settings.cluster_role,
            enabled=self.settings.cluster_role != "standalone",
            cluster_name=self.settings.cluster_name,
            node_name=self.settings.cluster_node_name,
            coordinator_url=self.settings.cluster_coordinator_url,
            public_base_url=self.public_base_url(),
            ready_worker_count=ready_worker_count,
            stale_worker_count=stale_worker_count,
            plan_count=len(plans),
            worker_heartbeat_timeout_seconds=self.settings.cluster_worker_heartbeat_timeout_seconds,
            worker_session=public_worker_session(worker_session) if worker_session is not None else None,
            workers=public_workers,
            latest_execution_metrics=dict(self._last_execution_metrics),
            notes=notes,
        )

    def public_base_url(self) -> str:
        configured = self.settings.cluster_public_base_url
        if configured is not None and configured.strip():
            return configured.rstrip("/")
        return f"http://{self.settings.host}:{self.settings.port}"

    def runtime_availability(self) -> tuple[bool, str | None]:
        status = self.status()
        if self.settings.cluster_role != "coordinator":
            return False, "Distributed coordinator execution is only available when `cluster_role=coordinator`."
        if self.settings.cluster_enrollment_secret is None:
            return False, "Set `LEWLM_CLUSTER_ENROLLMENT_SECRET` before using distributed experimental runtimes."
        if status.ready_worker_count < 2:
            return False, "At least two ready workers must be enrolled for distributed experimental execution."
        return True, None

    def issue_enrollment_token(
        self,
        *,
        worker_name: str | None = None,
        capabilities: Sequence[str] = (CapabilityName.CHAT.value,),
        ttl_seconds: int | None = None,
    ) -> ClusterIssueTokenResponse:
        self._require_coordinator()
        issued_at = utc_now()
        expires_at = issued_at + timedelta(seconds=ttl_seconds or self.settings.cluster_token_ttl_seconds)
        claims = WorkerEnrollmentToken(
            cluster_name=self.settings.cluster_name,
            issued_at=issued_at.isoformat(),
            expires_at=expires_at.isoformat(),
            worker_name=worker_name,
            capabilities=sorted({item for item in capabilities if item}),
            nonce=uuid4().hex,
        )
        token = self._encode_token(claims)
        self.audit_logger.record(
            action="cluster_issue_token",
            outcome="completed",
            actor="service",
            details={"worker_name": worker_name, "expires_at": expires_at.isoformat()},
        )
        self._emit_event(
            EventType.CLUSTER_TOKEN_ISSUED,
            {"worker_name": worker_name, "expires_at": expires_at.isoformat()},
        )
        return ClusterIssueTokenResponse(
            cluster_name=self.settings.cluster_name,
            token=token,
            expires_at=expires_at.isoformat(),
            capabilities=claims.capabilities,
            worker_name=worker_name,
        )

    def enroll_worker(self, request: ClusterEnrollWorkerRequest) -> ClusterEnrollWorkerResponse:
        self._require_coordinator()
        claims = self._decode_token(request.token)
        if claims.cluster_name != self.settings.cluster_name:
            raise AuthenticationError(
                "Worker enrollment token was issued for a different cluster.",
                details={"expected_cluster": self.settings.cluster_name, "actual_cluster": claims.cluster_name},
            )
        worker_name = request.worker_name or claims.worker_name or request.endpoint.rstrip("/").rsplit("/", maxsplit=1)[-1]
        worker = ClusterWorkerRecord(
            worker_id=uuid4().hex,
            worker_name=worker_name,
            endpoint=request.endpoint.rstrip("/"),
            capabilities=sorted({*claims.capabilities, *request.capabilities}),
            status="ready",
            enrolled_at=utc_now().isoformat(),
            last_heartbeat_at=utc_now().isoformat(),
            session_token=secrets.token_urlsafe(24),
            metadata=dict(request.metadata),
        )
        workers = [
            existing
            for existing in self._stored_workers()
            if existing.endpoint != worker.endpoint and existing.worker_name != worker.worker_name
        ]
        workers.append(worker)
        self._store_workers(workers)
        self.audit_logger.record(
            action="cluster_worker_enroll",
            outcome="completed",
            actor="service",
            details={"worker_id": worker.worker_id, "worker_name": worker.worker_name, "endpoint": worker.endpoint},
        )
        self._emit_event(
            EventType.CLUSTER_WORKER_ENROLLED,
            {"worker_id": worker.worker_id, "worker_name": worker.worker_name, "endpoint": worker.endpoint},
        )
        return ClusterEnrollWorkerResponse(
            worker=public_worker_record(worker),
            coordinator_url=self.public_base_url(),
            session_token=worker.session_token or "",
        )

    def complete_worker_enrollment(self, response: ClusterEnrollWorkerResponse) -> ClusterWorkerSession:
        session = ClusterWorkerSession(
            worker_id=response.worker.worker_id,
            worker_name=response.worker.worker_name,
            coordinator_url=response.coordinator_url.rstrip("/"),
            endpoint=self.public_base_url(),
            session_token=response.session_token,
            enrolled_at=response.worker.enrolled_at,
        )
        self.metadata_store.set_value(_CLUSTER_WORKER_SESSION_KEY, session.model_dump(mode="json"))
        self.audit_logger.record(
            action="cluster_worker_session",
            outcome="completed",
            actor="service",
            details={"worker_id": session.worker_id, "worker_name": session.worker_name},
        )
        return session

    def worker_session(self) -> ClusterWorkerSession | None:
        payload = self.metadata_store.get_value(_CLUSTER_WORKER_SESSION_KEY)
        if not isinstance(payload, dict):
            return None
        try:
            return ClusterWorkerSession.model_validate(payload)
        except Exception:
            return None

    def record_worker_heartbeat(self, request: ClusterHeartbeatRequest) -> ClusterWorkerRecord:
        self._require_coordinator()
        workers = self._stored_workers()
        updated_worker: ClusterWorkerRecord | None = None
        for index, worker in enumerate(workers):
            if worker.worker_id != request.worker_id:
                continue
            if worker.session_token != request.session_token:
                raise AuthenticationError("Worker heartbeat token is invalid.", details={"worker_id": request.worker_id})
            updated_worker = worker.model_copy(
                update={"last_heartbeat_at": utc_now().isoformat(), "status": "ready", "last_error": None},
            )
            workers[index] = updated_worker
            break
        if updated_worker is None:
            raise RuntimeUnavailableError("Worker is not enrolled in this coordinator.", details={"worker_id": request.worker_id})
        self._store_workers(workers)
        self._emit_event(EventType.CLUSTER_WORKER_HEARTBEAT, {"worker_id": updated_worker.worker_id})
        return public_worker_record(updated_worker)

    def plan_manifest(self, manifest: ModelManifest) -> DistributedExecutionPlan:
        self._require_coordinator()
        pipeline = manifest_distributed_pipeline(manifest)
        if not pipeline:
            raise RuntimeUnavailableError(
                "Selected model does not declare experimental distributed pipeline metadata.",
                details={"model_id": manifest.model_id},
            )
        ready_workers = [worker for worker in self._refreshed_workers() if worker.status == "ready"]
        required_workers = max(2, int(pipeline.get("required_workers", 2)))
        if len(ready_workers) < required_workers:
            raise RuntimeUnavailableError(
                "Not enough ready workers are enrolled for the requested distributed model.",
                details={
                    "model_id": manifest.model_id,
                    "required_workers": required_workers,
                    "ready_worker_count": len(ready_workers),
                },
            )
        total_layers = max(required_workers, int(pipeline.get("layer_count", required_workers * 16)))
        stage_names = [str(item) for item in pipeline.get("stage_names", []) if isinstance(item, str) and item.strip()]
        if len(stage_names) < required_workers:
            stage_names.extend(f"stage-{index + 1}" for index in range(len(stage_names), required_workers))
        worker_profiles = _selected_worker_profiles(
            workers=ready_workers,
            required_workers=required_workers,
            pipeline=pipeline,
            worker_profile_factory=lambda **kwargs: _worker_profile(profile_type=ClusterWorkerProfile, **kwargs),
        )
        heterogeneity_ratio = round(
            max((profile.relative_weight for profile in worker_profiles), default=1.0)
            / max(min((profile.relative_weight for profile in worker_profiles), default=1.0), 0.1),
            4,
        )
        layer_spans = _weighted_layer_spans(
            total_layers=total_layers,
            weights=[profile.relative_weight for profile in worker_profiles],
        )
        assignments: list[ClusterStageAssignment] = []
        next_start = 0
        for index, profile in enumerate(worker_profiles):
            layer_span = layer_spans[index]
            next_end = min(total_layers, next_start + layer_span)
            assignment_profile = _stage_assignment_profile(
                layer_span=layer_span,
                profile=profile,
                pipeline=pipeline,
                heterogeneity_ratio=heterogeneity_ratio,
            )
            assignments.append(
                ClusterStageAssignment(
                    stage_index=index,
                    stage_count=required_workers,
                    stage_name=stage_names[index],
                    worker_id=profile.worker_id,
                    worker_name=profile.worker_name,
                    endpoint=profile.endpoint,
                    start_layer=next_start,
                    end_layer=max(next_end, next_start + 1),
                    relative_weight=profile.relative_weight,
                    selection_score=profile.selection_score,
                    target_batch_tokens=int(assignment_profile["target_batch_tokens"]),
                    prefetch_tokens=int(assignment_profile["prefetch_tokens"]),
                    network_latency_ms=float(assignment_profile["network_latency_ms"]),
                    network_bandwidth_gbps=float(assignment_profile["network_bandwidth_gbps"]),
                    overlap_ratio=float(assignment_profile["overlap_ratio"]),
                    expected_compute_seconds=float(assignment_profile["expected_compute_seconds"]),
                    expected_network_seconds=float(assignment_profile["expected_network_seconds"]),
                    expected_queue_seconds=float(assignment_profile["expected_queue_seconds"]),
                    overlap_credit_seconds=float(assignment_profile["overlap_credit_seconds"]),
                    expected_stage_seconds=float(assignment_profile["expected_stage_seconds"]),
                    expected_utilization=float(assignment_profile["expected_utilization"]),
                ),
            )
            next_start = next_end
        average_network_latency_ms = round(
            sum(assignment.network_latency_ms for assignment in assignments) / len(assignments),
            4,
        )
        effective_batch_tokens = round(
            sum(assignment.target_batch_tokens for assignment in assignments) / len(assignments),
        )
        average_prefetch_tokens = round(
            sum(assignment.prefetch_tokens for assignment in assignments) / len(assignments),
        )
        plan = DistributedExecutionPlan(
            model_id=manifest.model_id,
            runtime_affinity=RuntimeAffinity.DISTRIBUTED_EXPERIMENTAL.value,
            stage_count=required_workers,
            required_workers=required_workers,
            total_layers=total_layers,
            assignments=assignments,
            worker_profiles=worker_profiles,
            scheduling={
                "selection_mode": "heterogeneous_weighted_latency_aware",
                "heterogeneity_ratio": heterogeneity_ratio,
                "weighted_capacity": round(sum(profile.relative_weight for profile in worker_profiles), 4),
                "effective_batch_tokens": effective_batch_tokens,
                "average_prefetch_tokens": average_prefetch_tokens,
                "average_network_latency_ms": average_network_latency_ms,
                "overlap_enabled": any(profile.overlap_ratio > 0.0 for profile in worker_profiles),
                "prefetch_enabled": average_prefetch_tokens > 0,
            },
            notes=[
                _DISTRIBUTED_BOUNDARY_NOTE,
                "Stage plans weight layer spans and batch targets by worker-relative throughput and observed network cost hints.",
            ],
            created_at=utc_now().isoformat(),
            updated_at=utc_now().isoformat(),
        )
        plans = self._stored_plans()
        plans[manifest.model_id] = plan
        self._store_plans(plans)
        self._emit_event(
            EventType.CLUSTER_PLAN_UPDATED,
            {"model_id": manifest.model_id, "stage_count": required_workers, "worker_count": required_workers},
        )
        self.audit_logger.record(
            action="cluster_plan_model",
            outcome="completed",
            actor="service",
            details={"model_id": manifest.model_id, "stage_count": required_workers},
        )
        return plan

    def plan_for_model(self, model_id: str) -> DistributedExecutionPlan | None:
        return self._stored_plans().get(model_id)

    async def generate(self, manifest: ModelManifest, request: GenerateRequest) -> GenerateResponse:
        plan = self.plan_manifest(manifest)
        pipeline = manifest_distributed_pipeline(manifest)
        prompt = render_generate_request_prompt(request)
        trace: list[dict[str, Any]] = []
        recovery_count = 0
        stage_latencies: list[float] = []
        stage_metrics: list[dict[str, Any]] = []
        assignments = list(plan.assignments)
        for stage_index, assignment in enumerate(list(assignments)):
            try:
                stage_response = await self._execute_stage_request(
                    manifest=manifest,
                    assignment=assignment,
                    prompt=prompt,
                    pipeline=pipeline,
                    trace=trace,
                )
            except RuntimeUnavailableError as exc:
                replacement = self._recover_assignment(assignments, stage_index, failed_worker_id=assignment.worker_id)
                if replacement is None:
                    raise RuntimeUnavailableError(
                        "Distributed pipeline stage failed and no replacement worker was available.",
                        details={"model_id": manifest.model_id, "stage_index": stage_index, "reason": str(exc)},
                    ) from exc
                assignments[stage_index] = replacement
                recovery_count += 1
                stage_response = await self._execute_stage_request(
                    manifest=manifest,
                    assignment=replacement,
                    prompt=prompt,
                    pipeline=pipeline,
                    trace=trace,
                )
            trace = list(stage_response.trace)
            stage_latency = stage_response.metrics.get("stage_elapsed_seconds")
            if isinstance(stage_latency, (int, float)):
                stage_latencies.append(float(stage_latency))
            stage_metrics.append(
                {
                    **dict(stage_response.metrics),
                    "worker_id": assignment.worker_id,
                    "worker_name": assignment.worker_name,
                    "stage_name": assignment.stage_name,
                    "layer_span": assignment.end_layer - assignment.start_layer,
                    "layer_range": f"{assignment.start_layer}:{assignment.end_layer}",
                },
            )
        output_text = stage_response.output_text or f"Distributed proof response: {prompt}"
        completion_tokens = len(output_text.split())
        prompt_tokens = len(prompt.split())
        execution_metrics = _aggregate_execution_metrics(
            stage_metrics=stage_metrics,
            assignments=assignments,
            completion_tokens=completion_tokens,
            distributed_boundary_note=_DISTRIBUTED_BOUNDARY_NOTE,
        )
        updated_plan = plan.model_copy(
            update={
                "assignments": assignments,
                "recovery_count": recovery_count,
                "updated_at": utc_now().isoformat(),
                "last_execution_at": utc_now().isoformat(),
            },
        )
        plans = self._stored_plans()
        plans[manifest.model_id] = updated_plan
        self._store_plans(plans)
        execution_notes = execution_metrics.pop("notes", [])
        metadata_payload = {
            "model_id": manifest.model_id,
            "stage_count": updated_plan.stage_count,
            "worker_count": len(updated_plan.assignments),
            "recovery_count": recovery_count,
            "assignments": [assignment.model_dump(mode="json") for assignment in updated_plan.assignments],
            "worker_profiles": [profile.model_dump(mode="json") for profile in updated_plan.worker_profiles],
            "scheduling": dict(updated_plan.scheduling),
            "stage_metrics": stage_metrics,
            "notes": [*list(updated_plan.notes), *[note for note in execution_notes if isinstance(note, str) and note]],
            **execution_metrics,
        }
        request.metadata["distributed_pipeline"] = metadata_payload
        self._last_execution_metrics = {
            key: value
            for key, value in metadata_payload.items()
            if key
            in {
                "model_id",
                "stage_count",
                "worker_count",
                "recovery_count",
                "average_stage_elapsed_seconds",
                "pipeline_latency_seconds",
                "critical_path_seconds",
                "throughput_tokens_per_second",
                "completion_tokens_per_second",
                "average_stage_utilization",
                "average_prefetch_tokens",
                "effective_batch_tokens",
                "average_network_latency_ms",
                "heterogeneity_ratio",
                "speedup_vs_single_host_percent",
                "network_share_percent",
                "compute_share_percent",
                "scheduling_share_percent",
                "pipeline_overlap_efficiency_percent",
                "bottleneck",
            }
        }
        self._emit_event(
            EventType.CLUSTER_PIPELINE_COMPLETED,
            {
                "model_id": manifest.model_id,
                "stage_count": updated_plan.stage_count,
                "recovery_count": recovery_count,
            },
        )
        return GenerateResponse(
            model_id=request.model_id,
            output_text=output_text,
            finish_reason="stop",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "distributed_stage_count": updated_plan.stage_count,
                "distributed_worker_count": len(updated_plan.assignments),
                "distributed_recovery_count": recovery_count,
            },
        )

    async def _execute_stage_request(
        self,
        *,
        manifest: ModelManifest,
        assignment: ClusterStageAssignment,
        prompt: str,
        pipeline: dict[str, Any],
        trace: list[dict[str, Any]],
    ) -> ClusterStageResponse:
        worker = self._worker_by_id(assignment.worker_id)
        if worker.status != "ready":
            raise RuntimeUnavailableError(
                "Assigned worker is not ready for pipeline execution.",
                details={"worker_id": assignment.worker_id, "status": worker.status},
            )
        request_payload = ClusterStageRequest(
            worker_id=worker.worker_id,
            model_id=manifest.model_id,
            stage=assignment,
            prompt=prompt,
            pipeline=pipeline,
            trace=trace,
        )
        started_at = time.perf_counter()
        payload = await self.transport.request_json(
            method="POST",
            base_url=worker.endpoint,
            path="/v1/cluster/worker/pipeline-stage",
            payload=request_payload.model_dump(mode="json"),
            headers={"authorization": f"Bearer {worker.session_token}"},
            timeout_seconds=float(self.settings.cluster_stage_timeout_seconds),
        )
        stage_response = ClusterStageResponse.model_validate(payload)
        elapsed_seconds = round(time.perf_counter() - started_at, 4)
        stage_response.metrics.setdefault("stage_elapsed_seconds", elapsed_seconds)
        return stage_response

    def _recover_assignment(
        self,
        assignments: list[ClusterStageAssignment],
        stage_index: int,
        *,
        failed_worker_id: str,
    ) -> ClusterStageAssignment | None:
        workers = self._refreshed_workers()
        in_use_worker_ids = {assignment.worker_id for assignment in assignments}
        replacement_worker = next(
            (
                worker
                for worker in workers
                if worker.status == "ready"
                and worker.worker_id not in in_use_worker_ids
                and worker.worker_id != failed_worker_id
            ),
            None,
        )
        self._mark_worker_state(failed_worker_id, status="unavailable", last_error="stage execution failed")
        if replacement_worker is None:
            return None
        current_assignment = assignments[stage_index]
        self._emit_event(
            EventType.CLUSTER_WORKER_RECOVERED,
            {
                "stage_index": stage_index,
                "failed_worker_id": failed_worker_id,
                "replacement_worker_id": replacement_worker.worker_id,
            },
        )
        return current_assignment.model_copy(
            update={
                "worker_id": replacement_worker.worker_id,
                "worker_name": replacement_worker.worker_name,
                "endpoint": replacement_worker.endpoint,
            },
        )

    def execute_stage(self, request: ClusterStageRequest, *, authorization: str | None) -> ClusterStageResponse:
        session = self.worker_session()
        if self.settings.cluster_role != "worker" or session is None:
            raise RuntimeUnavailableError(
                "This node is not configured as an enrolled cluster worker.",
                details={"role": self.settings.cluster_role},
            )
        expected_authorization = f"Bearer {session.session_token}"
        if authorization != expected_authorization:
            raise AuthenticationError("Worker stage authorization is invalid.", details={"worker_id": request.worker_id})
        if request.worker_id != session.worker_id:
            raise AuthenticationError(
                "Worker stage request targets a different enrolled worker.",
                details={"expected_worker_id": session.worker_id, "actual_worker_id": request.worker_id},
            )
        trace = list(request.trace)
        stage_profile = _stage_execution_profile(request.stage)
        service_seconds = max(float(stage_profile["expected_stage_seconds"]), 0.0)
        if service_seconds:
            time.sleep(service_seconds)
        trace.append(
            {
                "stage_index": request.stage.stage_index,
                "stage_name": request.stage.stage_name,
                "worker_id": session.worker_id,
                "worker_name": session.worker_name,
                "layer_range": f"{request.stage.start_layer}:{request.stage.end_layer}",
                "target_batch_tokens": stage_profile["target_batch_tokens"],
                "prefetch_tokens": stage_profile["prefetch_tokens"],
            },
        )
        output_text = None
        if request.stage.stage_index >= request.stage.stage_count - 1:
            output_text = format_distributed_output(
                model_id=request.model_id,
                prompt=request.prompt,
                pipeline=request.pipeline,
                trace=trace,
            )
        self._emit_event(
            EventType.CLUSTER_PIPELINE_STAGE_COMPLETED,
            {
                "model_id": request.model_id,
                "stage_index": request.stage.stage_index,
                "worker_id": session.worker_id,
            },
        )
        return ClusterStageResponse(
            trace=trace,
            output_text=output_text,
            metrics={
                "stage_index": request.stage.stage_index,
                "stage_count": request.stage.stage_count,
                "worker_name": session.worker_name,
                "layer_span": request.stage.end_layer - request.stage.start_layer,
                "relative_weight": stage_profile["relative_weight"],
                "selection_score": stage_profile["selection_score"],
                "target_batch_tokens": stage_profile["target_batch_tokens"],
                "prefetch_tokens": stage_profile["prefetch_tokens"],
                "network_latency_ms": stage_profile["network_latency_ms"],
                "network_bandwidth_gbps": stage_profile["network_bandwidth_gbps"],
                "overlap_ratio": stage_profile["overlap_ratio"],
                "compute_seconds": stage_profile["expected_compute_seconds"],
                "network_seconds": stage_profile["expected_network_seconds"],
                "scheduling_seconds": stage_profile["expected_queue_seconds"],
                "overlap_credit_seconds": stage_profile["overlap_credit_seconds"],
                "expected_stage_seconds": stage_profile["expected_stage_seconds"],
                "utilization": stage_profile["expected_utilization"],
                "bottleneck": _dominant_bottleneck(
                    compute_seconds=float(stage_profile["expected_compute_seconds"]),
                    network_seconds=float(stage_profile["expected_network_seconds"]),
                    scheduling_seconds=float(stage_profile["expected_queue_seconds"]),
                ),
            },
        )

    def performance_feature_snapshot(self) -> dict[str, Any]:
        status = self.status()
        return {
            "distributed_pipeline": {
                "supported": self.runtime_availability()[0],
                "active": bool(self._last_execution_metrics),
                "reason": (
                    "LewLM can coordinate an experimental multi-host pipeline-parallel proof executor."
                    if self.runtime_availability()[0]
                    else self.runtime_availability()[1]
                ),
                "metrics": {
                    "ready_worker_count": status.ready_worker_count,
                    "stale_worker_count": status.stale_worker_count,
                    "plan_count": status.plan_count,
                    **self._last_execution_metrics,
                },
                "notes": [_DISTRIBUTED_BOUNDARY_NOTE],
            },
        }

    def estimate_resources(self, manifest: ModelManifest) -> RuntimeEstimate:
        pipeline = manifest_distributed_pipeline(manifest)
        if not pipeline:
            return RuntimeEstimate(
                estimated_memory_mb=manifest.estimated_memory_mb,
                notes=["No distributed pipeline metadata was declared for this manifest."],
            )
        required_workers = max(2, int(pipeline.get("required_workers", 2)))
        return RuntimeEstimate(
            estimated_memory_mb=manifest.estimated_memory_mb,
            notes=[
                f"Experimental distributed plan requires at least {required_workers} worker hosts.",
                _DISTRIBUTED_BOUNDARY_NOTE,
            ],
        )

    def _stored_workers(self) -> list[ClusterWorkerRecord]:
        payload = self.metadata_store.get_value(_CLUSTER_WORKERS_KEY)
        if not isinstance(payload, list):
            return []
        workers: list[ClusterWorkerRecord] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                workers.append(ClusterWorkerRecord.model_validate(item))
            except Exception:
                continue
        return workers

    def _store_workers(self, workers: Sequence[ClusterWorkerRecord]) -> None:
        self.metadata_store.set_value(
            _CLUSTER_WORKERS_KEY,
            [worker.model_dump(mode="json") for worker in workers],
        )

    def _stored_plans(self) -> dict[str, DistributedExecutionPlan]:
        payload = self.metadata_store.get_value(_CLUSTER_PLANS_KEY)
        if not isinstance(payload, dict):
            return {}
        plans: dict[str, DistributedExecutionPlan] = {}
        for model_id, item in payload.items():
            if not isinstance(model_id, str) or not isinstance(item, dict):
                continue
            try:
                plans[model_id] = DistributedExecutionPlan.model_validate(item)
            except Exception:
                continue
        return plans

    def _store_plans(self, plans: Mapping[str, DistributedExecutionPlan]) -> None:
        self.metadata_store.set_value(
            _CLUSTER_PLANS_KEY,
            {model_id: plan.model_dump(mode="json") for model_id, plan in plans.items()},
        )

    def _refreshed_workers(self) -> list[ClusterWorkerRecord]:
        workers = self._stored_workers()
        refreshed = [self._refresh_worker_state(worker) for worker in workers]
        self._store_workers(refreshed)
        return refreshed

    def _refresh_worker_state(self, worker: ClusterWorkerRecord) -> ClusterWorkerRecord:
        timeout = max(1, self.settings.cluster_worker_heartbeat_timeout_seconds)
        last_heartbeat_at = _parse_timestamp(worker.last_heartbeat_at)
        if last_heartbeat_at is None:
            return worker.model_copy(update={"status": "stale"})
        age_seconds = max((utc_now() - last_heartbeat_at).total_seconds(), 0.0)
        if age_seconds > timeout and worker.status == "ready":
            return worker.model_copy(update={"status": "stale"})
        return worker

    def _worker_by_id(self, worker_id: str) -> ClusterWorkerRecord:
        worker = next((item for item in self._refreshed_workers() if item.worker_id == worker_id), None)
        if worker is None:
            raise RuntimeUnavailableError("Assigned worker is no longer registered.", details={"worker_id": worker_id})
        return worker

    def _mark_worker_state(self, worker_id: str, *, status: str, last_error: str | None = None) -> None:
        workers = self._stored_workers()
        updated = False
        for index, worker in enumerate(workers):
            if worker.worker_id != worker_id:
                continue
            workers[index] = worker.model_copy(update={"status": status, "last_error": last_error})
            updated = True
            break
        if updated:
            self._store_workers(workers)

    def _require_coordinator(self) -> None:
        if self.settings.cluster_role != "coordinator":
            raise ConfigurationError("This operation requires `cluster_role=coordinator`.")
        if self.settings.cluster_enrollment_secret is None:
            raise ConfigurationError("Set `LEWLM_CLUSTER_ENROLLMENT_SECRET` before using coordinator enrollment workflows.")

    def _cluster_secret(self) -> str:
        secret: SecretStr | None = self.settings.cluster_enrollment_secret
        if secret is None:
            raise ConfigurationError("Set `LEWLM_CLUSTER_ENROLLMENT_SECRET` before using cluster enrollment.")
        return secret.get_secret_value()

    def _encode_token(self, claims: WorkerEnrollmentToken) -> str:
        payload = json.dumps(claims.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")
        signature = hmac.new(self._cluster_secret().encode("utf-8"), payload, hashlib.sha256).digest()
        return (
            base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")
            + "."
            + base64.urlsafe_b64encode(signature).decode("utf-8").rstrip("=")
        )

    def _decode_token(self, token: str) -> WorkerEnrollmentToken:
        try:
            encoded_payload, encoded_signature = token.split(".", maxsplit=1)
        except ValueError as exc:
            raise AuthenticationError("Worker enrollment token is malformed.") from exc
        payload = _urlsafe_b64decode(encoded_payload)
        signature = _urlsafe_b64decode(encoded_signature)
        expected_signature = hmac.new(self._cluster_secret().encode("utf-8"), payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected_signature):
            raise AuthenticationError("Worker enrollment token signature is invalid.")
        try:
            claims = WorkerEnrollmentToken.model_validate_json(payload)
        except Exception as exc:
            raise AuthenticationError("Worker enrollment token payload is invalid.") from exc
        expires_at = _parse_timestamp(claims.expires_at)
        if expires_at is None or expires_at <= utc_now():
            raise AuthenticationError("Worker enrollment token has expired.", details={"expires_at": claims.expires_at})
        return claims

    def _emit_event(self, event_type: EventType, payload: dict[str, object]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.event_bus.publish(StreamEvent(type=event_type, scope=EventScope.SYSTEM, payload=payload)))


class DistributedExperimentalRuntime(ManagedTextRuntime):
    """Experimental runtime that coordinates proof-only distributed execution."""

    name = "distributed_experimental"
    affinity = RuntimeAffinity.DISTRIBUTED_EXPERIMENTAL
    supported_formats = (ModelFormat.GGUF, ModelFormat.MLX, ModelFormat.HUGGINGFACE, ModelFormat.UNKNOWN)
    supported_modalities = (ModelModality.TEXT, ModelModality.MULTIMODAL)
    supported_capabilities = frozenset({CapabilityName.CHAT, CapabilityName.STREAMING})
    platform_guidance = _DISTRIBUTED_BOUNDARY_NOTE

    def __init__(self, *, settings: LewLMSettings, cluster_service: DistributedClusterService) -> None:
        super().__init__()
        self.settings = settings
        self.cluster_service = cluster_service

    def _check_environment(self) -> tuple[bool, str | None]:
        return self.cluster_service.runtime_availability()

    def supports_manifest(self, manifest: ModelManifest) -> bool:
        return bool(manifest_distributed_pipeline(manifest))

    async def _load_model(self, manifest: ModelManifest) -> None:
        self.cluster_service.plan_manifest(manifest)

    async def _unload_model(self, model_id: str) -> None:
        return None

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        manifest = self._loaded_manifests[request.model_id]
        return await self.cluster_service.generate(manifest, request)

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        response = await self._generate(request)
        chunks = response.output_text.split(" ")
        for index, chunk in enumerate(chunks):
            suffix = "" if index == len(chunks) - 1 else " "
            yield f"{chunk}{suffix}"
            await asyncio.sleep(0)

    def _tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens: Sequence[int]) -> str:
        return bytes(tokens).decode("utf-8")

    def performance_feature_snapshot(self) -> dict[str, Any]:
        return self.cluster_service.performance_feature_snapshot()

    def estimate_resources(self, manifest: ModelManifest) -> RuntimeEstimate:
        return self.cluster_service.estimate_resources(manifest)


def manifest_distributed_pipeline(manifest: ModelManifest) -> dict[str, Any]:
    pipeline = manifest.metadata.get("distributed_pipeline")
    return pipeline if isinstance(pipeline, dict) else {}


def distributed_pipeline_measurements(request_metadata: dict[str, Any]) -> dict[str, int | float | bool]:
    payload = request_metadata.get("distributed_pipeline")
    if not isinstance(payload, dict):
        return {}
    values: dict[str, int | float | bool | None] = {
        "distributed_pipeline_requests": 1,
        "distributed_stage_count": _coerce_int(payload.get("stage_count")),
        "distributed_worker_count": _coerce_int(payload.get("worker_count")),
        "distributed_recovery_count": _coerce_int(payload.get("recovery_count")),
        "distributed_pipeline_latency_seconds": _coerce_float(payload.get("pipeline_latency_seconds")),
        "distributed_critical_path_seconds": _coerce_float(payload.get("critical_path_seconds")),
        "distributed_average_stage_elapsed_seconds": _coerce_float(payload.get("average_stage_elapsed_seconds")),
        "distributed_average_stage_utilization": _coerce_float(payload.get("average_stage_utilization")),
        "distributed_throughput_tokens_per_second": _coerce_float(payload.get("throughput_tokens_per_second")),
        "distributed_completion_tokens_per_second": _coerce_float(payload.get("completion_tokens_per_second")),
        "distributed_effective_batch_tokens": _coerce_int(payload.get("effective_batch_tokens")),
        "distributed_average_prefetch_tokens": _coerce_int(payload.get("average_prefetch_tokens")),
        "distributed_average_network_latency_ms": _coerce_float(payload.get("average_network_latency_ms")),
        "distributed_network_share_percent": _coerce_float(payload.get("network_share_percent")),
        "distributed_compute_share_percent": _coerce_float(payload.get("compute_share_percent")),
        "distributed_scheduling_share_percent": _coerce_float(payload.get("scheduling_share_percent")),
        "distributed_pipeline_overlap_efficiency_percent": _coerce_float(
            payload.get("pipeline_overlap_efficiency_percent"),
        ),
        "distributed_speedup_vs_single_host_percent": _coerce_float(payload.get("speedup_vs_single_host_percent")),
        "distributed_heterogeneity_ratio": _coerce_float(payload.get("heterogeneity_ratio")),
        "distributed_prefetch_enabled": bool(payload.get("prefetch_enabled", False)),
        "distributed_overlap_enabled": bool(payload.get("overlap_enabled", False)),
    }
    return {
        key: value
        for key, value in values.items()
        if value is not None
    }


def render_generate_request_prompt(request: GenerateRequest) -> str:
    rendered_lines: list[str] = []
    for message in request.messages:
        attachments = ", ".join(attachment.name or attachment.attachment_type for attachment in message.attachments)
        attachment_suffix = f" [{attachments}]" if attachments else ""
        rendered_lines.append(f"{message.role}: {message.content}{attachment_suffix}")
    return "\n".join(rendered_lines) if rendered_lines else "user:"


def format_distributed_output(
    *,
    model_id: str,
    prompt: str,
    pipeline: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
) -> str:
    worker_trace = " -> ".join(str(item.get("worker_name", "worker")) for item in trace)
    layer_trace = ", ".join(str(item.get("layer_range", "?")) for item in trace)
    template = pipeline.get("response_template")
    if isinstance(template, str) and template:
        try:
            return template.format(
                model_id=model_id,
                prompt=prompt,
                worker_trace=worker_trace,
                layer_trace=layer_trace,
                stage_count=len(trace),
            )
        except KeyError:
            pass
    prompt_excerpt = prompt.splitlines()[-1] if prompt else ""
    return (
        f"Distributed experimental response for `{model_id}`\n"
        f"Prompt: {prompt_excerpt}\n"
        f"Workers: {worker_trace}\n"
        f"Layers: {layer_trace}"
    )


def public_worker_record(worker: ClusterWorkerRecord) -> ClusterWorkerRecord:
    return worker.model_copy(update={"session_token": None})


def public_worker_session(session: ClusterWorkerSession) -> ClusterWorkerSession:
    return session.model_copy(update={"session_token": "<redacted>"})
