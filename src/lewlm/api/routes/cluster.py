"""Experimental cluster coordination routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field

from lewlm.api.dependencies import get_services
from lewlm.runtime.experimental import (
    ClusterEnrollWorkerRequest,
    ClusterEnrollWorkerResponse,
    ClusterHeartbeatRequest,
    ClusterIssueTokenResponse,
    ClusterStageRequest,
    ClusterStageResponse,
    ClusterStatus,
    DistributedExecutionPlan,
)


router = APIRouter(tags=["cluster"])


class ClusterTokenIssueRequest(BaseModel):
    worker_name: str | None = None
    capabilities: list[str] = Field(default_factory=lambda: ["chat"])
    ttl_seconds: int | None = None


class ClusterPlanRequest(BaseModel):
    model_id: str


@router.get("/v1/cluster/status", response_model=ClusterStatus)
def cluster_status(request: Request) -> ClusterStatus:
    """Return cluster coordinator or worker state."""

    services = get_services(request)
    return services.cluster_service.status()


@router.post("/v1/cluster/tokens", response_model=ClusterIssueTokenResponse)
def issue_cluster_token(payload: ClusterTokenIssueRequest, request: Request) -> ClusterIssueTokenResponse:
    """Issue a signed worker enrollment token on a coordinator node."""

    services = get_services(request)
    return services.cluster_service.issue_enrollment_token(
        worker_name=payload.worker_name,
        capabilities=payload.capabilities,
        ttl_seconds=payload.ttl_seconds,
    )


@router.post("/v1/cluster/workers/enroll", response_model=ClusterEnrollWorkerResponse)
def enroll_cluster_worker(payload: ClusterEnrollWorkerRequest, request: Request) -> ClusterEnrollWorkerResponse:
    """Enroll a worker on the coordinator and return its session token."""

    services = get_services(request)
    return services.cluster_service.enroll_worker(payload)


@router.post("/v1/cluster/workers/heartbeat", response_model=dict[str, Any])
def cluster_worker_heartbeat(payload: ClusterHeartbeatRequest, request: Request) -> dict[str, Any]:
    """Refresh a worker heartbeat on the coordinator."""

    services = get_services(request)
    worker = services.cluster_service.record_worker_heartbeat(payload)
    return {"worker": worker.model_dump(mode="json"), "status": services.cluster_service.status().model_dump(mode="json")}


@router.post("/v1/cluster/plans", response_model=DistributedExecutionPlan)
def cluster_plan(payload: ClusterPlanRequest, request: Request) -> DistributedExecutionPlan:
    """Create or refresh a deterministic distributed plan for a model."""

    services = get_services(request)
    manifest = services.model_registry.get_manifest(payload.model_id)
    return services.cluster_service.plan_manifest(manifest)


@router.post("/v1/cluster/worker/pipeline-stage", response_model=ClusterStageResponse)
def cluster_pipeline_stage(
    payload: ClusterStageRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> ClusterStageResponse:
    """Execute one worker stage of the distributed proof pipeline."""

    services = get_services(request)
    return services.cluster_service.execute_stage(payload, authorization=authorization)
