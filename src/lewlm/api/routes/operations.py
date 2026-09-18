"""Operational routes for jobs, cache, and runtime stats."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from lewlm.api.dependencies import get_services
from lewlm.conversion.models import JobRecord
from lewlm.core.contracts import CapabilityName
from lewlm.runtime.experimental import ClusterStatus
from lewlm.runtime.cancellation import RequestCancellationRecord
from lewlm.runtime.identity import EngineStartupPhase, ModelWarmth, RuntimeInfo, StartupPhases
from lewlm.runtime.residency import ModelResidencySnapshot
from lewlm.runtime.operations import LifecycleOperationRecord
from lewlm.security.authorization import LifecycleCapability, request_api_credential
from lewlm.telemetry.stats import (
    CacheStats,
    RuntimeStats,
    ServingProfileInventory,
    ServingProfileRecommendation,
)


router = APIRouter(tags=["operations"])


class AutotuneRequest(BaseModel):
    model_id: str | None = None
    prompt: str = Field(default="Benchmark ping")
    capability: str = Field(default=CapabilityName.CHAT.value)
    workload_class: str | None = None
    preset: str | None = Field(
        default=None,
        description="Serving-profile preset to measure: `interactive` (latency-first) or `throughput`; defaults to the configured preset.",
    )


@router.get("/v1/runtime", response_model=RuntimeInfo)
async def runtime_info(request: Request) -> RuntimeInfo:
    """Return stable identity for this model-owning service container."""

    services = get_services(request)
    residencies = await services.model_residency_manager.list_residencies()
    scheduler = services.runtime_request_scheduler.snapshot()
    return RuntimeInfo(
        **services.runtime_instance.model_dump(),
        loaded_model_count=sum(1 for item in residencies if item.state.value == "ready"),
        active_request_count=int(scheduler["active_requests"]),
        startup=_startup_phases(services, residencies),
    )


def _startup_phases(services, residencies: list[ModelResidencySnapshot]) -> StartupPhases:
    """Three phases from cached state only: no engine probe, no model load."""

    instance = services.runtime_instance
    ready_seconds = None
    if instance.ready_at is not None:
        ready_seconds = round(max((instance.ready_at - instance.started_at).total_seconds(), 0.0), 3)
    engines: list[EngineStartupPhase] = []
    for endpoint_id, runtime in sorted(services.runtime_catalog.endpoint_runtimes().items()):
        snapshot_method = getattr(runtime, "endpoint_snapshot", None)
        if not callable(snapshot_method):
            continue
        snapshot = snapshot_method()
        engines.append(
            EngineStartupPhase(
                endpoint_id=endpoint_id,
                profile=str(snapshot.get("profile")),
                enabled=bool(snapshot.get("enabled", True)),
                state=str(snapshot.get("inventory_state", "unknown")),
                inventory_age_seconds=snapshot.get("inventory_age_seconds"),
                advertised_model_count=len(snapshot.get("advertised_model_ids") or ()),
                first_advertised_at=snapshot.get("first_advertised_at"),
            ),
        )

    def warmth(item: ModelResidencySnapshot) -> ModelWarmth:
        load_seconds = None
        if item.loaded_at is not None and item.load_started_at is not None:
            load_seconds = round(max((item.loaded_at - item.load_started_at).total_seconds(), 0.0), 3)
        return ModelWarmth(model_id=item.model_id, runtime=item.runtime, state=item.state.value, loaded_at=item.loaded_at, load_seconds=load_seconds)

    return StartupPhases(
        lewlm_ready_at=instance.ready_at,
        lewlm_ready_seconds=ready_seconds,
        engines=engines,
        warm_models=[warmth(item) for item in residencies if item.state.value == "ready"],
        loading_models=[warmth(item) for item in residencies if item.state.value == "loading"],
    )


@router.get("/v1/runtime/residencies", response_model=list[ModelResidencySnapshot])
async def runtime_residencies(request: Request) -> list[ModelResidencySnapshot]:
    """List process-local live model residency state."""

    services = get_services(request)
    _authorize_lifecycle(request, services, LifecycleCapability.INSPECT_RESIDENCY)
    return await services.model_residency_manager.list_residencies()


@router.get("/v1/model-lifecycle/operations/{operation_id}", response_model=LifecycleOperationRecord)
async def get_lifecycle_operation(operation_id: str, request: Request) -> LifecycleOperationRecord:
    """Poll an asynchronous lifecycle operation."""

    services = get_services(request)
    _authorize_lifecycle(request, services, LifecycleCapability.INSPECT_RESIDENCY)
    return await services.lifecycle_operation_manager.get(operation_id)


@router.delete("/v1/model-lifecycle/operations/{operation_id}", response_model=LifecycleOperationRecord)
async def cancel_lifecycle_operation(operation_id: str, request: Request) -> LifecycleOperationRecord:
    """Cancel a pending or running lifecycle operation."""

    services = get_services(request)
    _authorize_lifecycle(request, services, LifecycleCapability.DRAIN_MODEL)
    return await services.lifecycle_operation_manager.cancel(operation_id)


@router.post("/v1/requests/{request_id}/cancel", response_model=RequestCancellationRecord)
async def cancel_request(request_id: str, request: Request) -> RequestCancellationRecord:
    """Cancel an in-flight request by the `x-request-id` handle it was sent with.

    Idempotent, and safe to call before the target request arrives: an unknown
    handle records an intent that stops a matching request on arrival. Repeat the
    call to observe whether the request actually stopped — `cancelling` means the
    signal was delivered, `cancelled` that a checkpoint acted on it, `completed`
    that the request finished first.

    Cancellation is best-effort and process-local: only the LewLM instance named
    by `runtime_instance_id` can act on the handle, and work already committed to
    one bounded backend call runs to completion.
    """

    services = get_services(request)
    return services.request_cancellation_registry.cancel(
        request_id,
        application_id=request.headers.get("x-lewlm-application-id"),
        credential=(
            request_api_credential(request.headers)
            if services.settings.api_key_required
            else None
        ),
    )


@router.get("/v1/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str, request: Request) -> JobRecord:
    """Return the status of a background job."""

    services = get_services(request)
    return services.conversion_service.get_job(job_id)


@router.get("/v1/cache/stats", response_model=CacheStats)
def cache_stats(request: Request) -> CacheStats:
    """Return managed cache statistics."""

    services = get_services(request)
    return services.telemetry_service.cache_stats()


@router.get("/v1/runtime/stats", response_model=RuntimeStats)
async def runtime_stats(request: Request) -> RuntimeStats:
    """Return runtime availability and residency statistics."""

    services = get_services(request)
    return await services.telemetry_service.runtime_stats()


@router.get("/v1/cluster/stats", response_model=ClusterStatus)
def cluster_stats(request: Request) -> ClusterStatus:
    """Return experimental cluster status."""

    services = get_services(request)
    return services.cluster_service.status()


@router.get("/v1/serving-profiles", response_model=ServingProfileInventory)
def list_serving_profiles(
    request: Request,
    model: str | None = None,
    capability: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> ServingProfileInventory:
    """List serving profiles stored on this host, newest first.

    `limit` bounds the stored profiles read before `model` and `capability`
    narrow them, so it is a scan window rather than a page size.
    """

    services = get_services(request)
    return services.telemetry_service.list_serving_profiles(
        model_id=model,
        capability=capability,
        limit=limit,
    )


@router.post("/v1/benchmarks/autotune", response_model=ServingProfileRecommendation)
async def autotune(payload: AutotuneRequest, request: Request) -> ServingProfileRecommendation:
    """Benchmark serving-profile candidates and persist the recommended profile."""

    services = get_services(request)
    _authorize_lifecycle(request, services, LifecycleCapability.RUN_DISRUPTIVE_DIAGNOSTICS)
    return await services.telemetry_service.autotune(
        model_id=payload.model_id,
        prompt=payload.prompt,
        capability=payload.capability,
        workload_class=payload.workload_class,
        preset=payload.preset,
    )


def _authorize_lifecycle(request: Request, services, capability: LifecycleCapability) -> None:
    application_id = request.headers.get("x-lewlm-application-id")
    services.tool_authorizer.require_lifecycle(
        capability,
        credential=request_api_credential(request.headers),
        authorizations=[
            item.strip()
            for item in request.headers.get("x-lewlm-authorized-actions", "").split(",")
            if item.strip()
        ],
        actor=application_id or "api",
        application_id=application_id,
        client_instance_id=request.headers.get("x-lewlm-client-instance-id"),
        details={"path": request.url.path},
    )
