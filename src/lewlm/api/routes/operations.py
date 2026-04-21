"""Operational routes for jobs, cache, and runtime stats."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from lewlm.api.dependencies import get_services
from lewlm.conversion.models import JobRecord
from lewlm.core.contracts import CapabilityName
from lewlm.runtime.experimental import ClusterStatus
from lewlm.telemetry.stats import CacheStats, RuntimeStats, ServingProfileRecommendation


router = APIRouter(tags=["operations"])


class AutotuneRequest(BaseModel):
    model_id: str | None = None
    prompt: str = Field(default="Benchmark ping")
    capability: str = Field(default=CapabilityName.CHAT.value)
    workload_class: str | None = None


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


@router.post("/v1/benchmarks/autotune", response_model=ServingProfileRecommendation)
async def autotune(payload: AutotuneRequest, request: Request) -> ServingProfileRecommendation:
    """Benchmark serving-profile candidates and persist the recommended profile."""

    services = get_services(request)
    return await services.telemetry_service.autotune(
        model_id=payload.model_id,
        prompt=payload.prompt,
        capability=payload.capability,
        workload_class=payload.workload_class,
    )
