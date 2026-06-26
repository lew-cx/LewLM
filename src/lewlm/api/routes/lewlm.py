"""LewLM middleware-specific routes."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from lewlm.api.dependencies import get_services
from lewlm.conversion.models import (
    ConversionJobRequest,
    ConversionPolicy,
    ConversionTargetPlanningReport,
    JobRecord,
)
from lewlm.core.contracts import (
    CapabilityEvidence,
    CapabilityName,
    LewLMMiddlewareCapabilitiesReport,
    ModelArtifactLineageReport,
)
from lewlm.core.errors import ConfigurationError
from lewlm.core.middleware import (
    build_middleware_capabilities_report,
    build_model_artifact_lineage_report,
)
from lewlm.core.probes import run_model_smoke_probe
from lewlm.security.authorization import ToolAction


router = APIRouter(tags=["lewlm"])


class LewLMProbeRequest(BaseModel):
    """Live middleware probe request.

    Routing probes are non-generating and remain the default. Load and generation probes
    are opt-in smoke tests for one explicit model.
    """

    model_id: str | None = None
    capability: CapabilityName | None = None
    mode: Literal["routing", "load", "generate"] = "routing"
    prompt: str = "LewLM runtime probe"
    max_tokens: int = 1


class LewLMProbeResponse(BaseModel):
    model_id: str | None = None
    capability: CapabilityName | None = None
    mode: Literal["routing", "load", "generate"] = "routing"
    evidence: list[CapabilityEvidence] = Field(default_factory=list)
    persisted: bool = False
    reason: str
    generated_text: str | None = None


class LewLMBenchmarkRequest(BaseModel):
    model_id: str | None = None
    all_models: bool = False
    prompt: str = "Benchmark ping"
    capability: str = CapabilityName.CHAT.value
    warmup_run_count: int = 1
    workload_class: str | None = None
    include_scenarios: bool = False


class LewLMConversionPlanRequest(BaseModel):
    model_id: str
    policy: ConversionPolicy = ConversionPolicy.BALANCED
    custom_bits: int | None = None


@router.get("/v1/lewlm/capabilities", response_model=LewLMMiddlewareCapabilitiesReport)
def lewlm_capabilities(request: Request) -> LewLMMiddlewareCapabilitiesReport:
    """Return LewLM's host-level middleware capability evidence report."""

    services = get_services(request)
    return build_middleware_capabilities_report(services)


@router.post("/v1/lewlm/probes", response_model=LewLMProbeResponse)
async def probe_lewlm_capabilities(payload: LewLMProbeRequest, request: Request) -> LewLMProbeResponse:
    """Run a routing probe or an explicit runtime smoke probe."""

    services = get_services(request)
    if payload.mode != "routing":
        if payload.model_id is None:
            raise ConfigurationError("`model_id` is required for load or generate probes.")
        outcome = await run_model_smoke_probe(
            services,
            model_id=payload.model_id,
            capability=payload.capability or CapabilityName.CHAT,
            mode=payload.mode,
            prompt=payload.prompt,
            max_tokens=payload.max_tokens,
        )
        return LewLMProbeResponse(
            model_id=outcome.model_id,
            capability=outcome.capability,
            mode=outcome.mode,
            evidence=outcome.evidence,
            persisted=outcome.persisted,
            reason=outcome.reason,
            generated_text=outcome.generated_text,
        )
    if payload.model_id is None:
        report = build_middleware_capabilities_report(services)
        evidence = report.capability_evidence
        if payload.capability is not None:
            evidence = [item for item in evidence if item.capability == payload.capability]
        return LewLMProbeResponse(
            capability=payload.capability,
            mode=payload.mode,
            evidence=evidence,
            reason="Host-level readiness probe completed without loading or generating from a model.",
        )
    report = services.model_router.model_capability_report(payload.model_id)
    evidence = report.capability_evidence
    if payload.capability is not None:
        evidence = [item for item in evidence if item.capability == payload.capability]
    return LewLMProbeResponse(
        model_id=report.model_id,
        capability=payload.capability,
        mode=payload.mode,
        evidence=evidence,
        reason="Model-level routing probe completed without loading or generating from the model.",
    )


@router.post("/v1/lewlm/conversions/plan", response_model=ConversionTargetPlanningReport)
def plan_lewlm_conversion(payload: LewLMConversionPlanRequest, request: Request) -> ConversionTargetPlanningReport:
    """Return read-only conversion target options without queueing a job."""

    services = get_services(request)
    return services.conversion_service.plan_targets(
        payload.model_id,
        policy=payload.policy,
        custom_bits=payload.custom_bits,
    )


@router.post("/v1/lewlm/conversions", response_model=JobRecord)
def create_lewlm_conversion(payload: ConversionJobRequest, request: Request) -> JobRecord:
    """Queue or resolve a model conversion through the LewLM middleware namespace."""

    services = get_services(request)
    services.tool_authorizer.require(
        ToolAction.MODEL_CONVERSION,
        authorizations=payload.authorized_actions,
        actor="api",
        details={"model_id": payload.model_id, "policy": payload.policy.value},
    )
    return services.conversion_service.submit(payload)


@router.get("/v1/lewlm/conversions/{job_id}", response_model=JobRecord)
def get_lewlm_conversion(job_id: str, request: Request) -> JobRecord:
    """Return a conversion job by id through the LewLM namespace."""

    services = get_services(request)
    return services.conversion_service.get_job(job_id)


@router.post("/v1/lewlm/benchmarks", response_model=dict[str, Any])
async def create_lewlm_benchmark(payload: LewLMBenchmarkRequest, request: Request) -> dict[str, Any]:
    """Run a benchmark through the LewLM middleware namespace."""

    services = get_services(request)
    if payload.all_models:
        result = await services.telemetry_service.benchmark_suite_lightweight(
            prompt=payload.prompt,
            model_ids=None,
            capability=payload.capability,
            warmup_run_count=payload.warmup_run_count,
            workload_class=payload.workload_class,
        )
        return result.model_dump(mode="json")
    benchmark = (
        services.telemetry_service.benchmark
        if payload.include_scenarios
        else services.telemetry_service.benchmark_lightweight
    )
    result = await benchmark(
        model_id=payload.model_id,
        prompt=payload.prompt,
        capability=payload.capability,
        warmup_run_count=payload.warmup_run_count,
        workload_class=payload.workload_class,
    )
    return result.model_dump(mode="json")


@router.get("/v1/lewlm/models/{model_id}/artifacts", response_model=ModelArtifactLineageReport)
def lewlm_model_artifacts(model_id: str, request: Request) -> ModelArtifactLineageReport:
    """Return lineage, conversion, benchmark, and capability evidence for a model."""

    services = get_services(request)
    return build_model_artifact_lineage_report(services, model_id)
