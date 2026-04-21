"""Model registry routes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from lewlm.api.dependencies import get_services
from lewlm.core.contracts import ModelCapabilityReport, ModelInventory, ModelScanSummary
from lewlm.conversion.models import ConversionJobRequest, JobRecord
from lewlm.security.authorization import ToolAction
from lewlm.security.files import resolve_scoped_path


router = APIRouter(tags=["models"])


class ModelScanRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)


class ModelLifecycleResponse(BaseModel):
    status: Literal["warmed", "unloaded"]
    model_id: str
    runtime: str
    reason: str


@router.get("/v1/models", response_model=ModelInventory)
def list_models(request: Request) -> ModelInventory:
    """List models currently stored in the local registry."""

    services = get_services(request)
    return services.model_registry.inventory()


@router.get("/v1/models/{model_id}/capabilities", response_model=ModelCapabilityReport)
def model_capabilities(model_id: str, request: Request) -> ModelCapabilityReport:
    """Describe model/runtime capability support for the current host."""

    services = get_services(request)
    return services.model_router.model_capability_report(model_id)


@router.post("/v1/models/scan", response_model=ModelScanSummary)
def scan_models(payload: ModelScanRequest, request: Request) -> ModelScanSummary:
    """Scan configured or requested roots and update the local registry."""

    services = get_services(request)
    roots = (
        [
            resolve_scoped_path(
                Path(path),
                allowed_roots=services.settings.models_dir,
                purpose="Model scan root",
                expect="dir",
            )
            for path in payload.paths
        ]
        if payload.paths
        else None
    )
    return services.model_registry.scan(roots=roots)


@router.post("/v1/models/convert", response_model=JobRecord)
def convert_model(payload: ConversionJobRequest, request: Request) -> JobRecord:
    """Queue or resolve a conversion job for a registered model."""

    services = get_services(request)
    services.tool_authorizer.require(
        ToolAction.MODEL_CONVERSION,
        authorizations=payload.authorized_actions,
        actor="api",
        details={"model_id": payload.model_id, "policy": payload.policy.value},
    )
    return services.conversion_service.submit(payload)


@router.post("/v1/models/{model_id}/warm", response_model=ModelLifecycleResponse)
async def warm_model(model_id: str, request: Request) -> ModelLifecycleResponse:
    """Warm a registered model in its selected runtime."""

    services = get_services(request)
    decision = await services.model_router.warm_model(model_id)
    return ModelLifecycleResponse(
        status="warmed",
        model_id=decision.model_id,
        runtime=decision.runtime_name,
        reason=decision.reason,
    )


@router.post("/v1/models/{model_id}/unload", response_model=ModelLifecycleResponse)
async def unload_model(model_id: str, request: Request) -> ModelLifecycleResponse:
    """Unload a registered model from its selected runtime."""

    services = get_services(request)
    decision = await services.model_router.unload_model(model_id)
    return ModelLifecycleResponse(
        status="unloaded",
        model_id=decision.model_id,
        runtime=decision.runtime_name,
        reason=decision.reason,
    )
