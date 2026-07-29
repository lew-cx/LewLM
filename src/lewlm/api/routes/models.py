"""Model registry routes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from lewlm.api.dependencies import get_services
from lewlm.core.contracts import (
    ModelCapabilityAvailability,
    ModelCapabilityReport,
    ModelInventory,
    ModelManifest,
    ModelScanSummary,
)
from lewlm.conversion.models import ConversionJobRequest, JobRecord
from lewlm.security.authorization import LifecycleCapability, ToolAction, request_api_credential
from lewlm.security.files import resolve_scoped_path
from lewlm.runtime.residency import ModelResidencySnapshot, ModelResidencyState
from lewlm.runtime.operations import LifecycleOperationRecord


router = APIRouter(tags=["models"])


class ModelScanRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)


class ModelDetail(BaseModel):
    """One registered model and its serving readiness on this host."""

    model: ModelManifest
    capability_availability: ModelCapabilityAvailability


class ModelLifecycleResponse(BaseModel):
    status: Literal["warmed", "drained", "unloaded"]
    model_id: str
    runtime: str
    reason: str
    runtime_instance_id: str
    operation_id: str
    previous_state: ModelResidencyState | None = None
    current_state: ModelResidencyState | None = None
    active_usage_count: int = 0
    backend_operation_performed: bool = False
    joined_existing_operation: bool = False


class AsyncDrainRequest(BaseModel):
    timeout_seconds: float | None = Field(default=None, gt=0)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


@router.get("/v1/models", response_model=ModelInventory)
def list_models(request: Request) -> ModelInventory:
    """List models currently stored in the local registry.

    Each item is annotated with the capabilities it can actually serve on this
    host, so a caller can choose a usable model without one request per model.
    """

    services = get_services(request)
    return services.model_router.annotate_inventory(services.model_registry.inventory())


@router.get("/v1/models/{model_id}", response_model=ModelDetail)
def get_model(model_id: str, request: Request) -> ModelDetail:
    """Return one registered model with the readiness annotation from the list.

    Without it a caller has to fetch the whole inventory and filter client-side
    just to render one model.
    """

    services = get_services(request)
    manifest = services.model_registry.get_manifest(model_id)
    return ModelDetail(
        model=manifest,
        capability_availability=services.model_router.model_capability_availability(manifest),
    )


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
    application_id = request.headers.get("x-lewlm-application-id")
    request_id = request.headers.get("x-request-id") or str(uuid4())
    services.tool_authorizer.require_lifecycle(
        LifecycleCapability.LOAD_MODEL,
        credential=request_api_credential(request.headers),
        authorizations=_authorized_actions(request),
        actor=application_id or "api",
        application_id=application_id,
        client_instance_id=request.headers.get("x-lewlm-client-instance-id"),
        details={"model_id": model_id, "request_id": request_id},
    )
    decision, lifecycle = await services.model_router.warm_model_lifecycle(
        model_id,
        request_id=request_id,
        application_id=application_id,
    )
    return ModelLifecycleResponse(
        status="warmed",
        model_id=decision.model_id,
        runtime=decision.runtime_name,
        reason=lifecycle.reason,
        **lifecycle.model_dump(exclude={"model_id", "runtime", "reason"}),
    )


@router.post("/v1/models/{model_id}/unload", response_model=ModelLifecycleResponse)
async def unload_model(model_id: str, request: Request) -> ModelLifecycleResponse:
    """Unload a registered model from its selected runtime."""

    services = get_services(request)
    application_id = request.headers.get("x-lewlm-application-id")
    request_id = request.headers.get("x-request-id") or str(uuid4())
    services.tool_authorizer.require_lifecycle(
        LifecycleCapability.UNLOAD_MODEL,
        credential=request_api_credential(request.headers),
        authorizations=_authorized_actions(request),
        actor=application_id or "api",
        application_id=application_id,
        client_instance_id=request.headers.get("x-lewlm-client-instance-id"),
        details={"model_id": model_id, "request_id": request_id},
    )
    decision, lifecycle = await services.model_router.unload_model_lifecycle(
        model_id,
        request_id=request_id,
        application_id=application_id,
    )
    return ModelLifecycleResponse(
        status="unloaded",
        model_id=decision.model_id,
        runtime=decision.runtime_name,
        reason=lifecycle.reason,
        **lifecycle.model_dump(exclude={"model_id", "runtime", "reason"}),
    )


@router.post("/v1/models/{model_id}/drain", response_model=ModelLifecycleResponse)
async def drain_model(model_id: str, request: Request) -> ModelLifecycleResponse:
    """Stop new leases, wait for current usage, and unload a model."""

    services = get_services(request)
    application_id = request.headers.get("x-lewlm-application-id")
    request_id = request.headers.get("x-request-id") or str(uuid4())
    services.tool_authorizer.require_lifecycle(
        LifecycleCapability.DRAIN_MODEL,
        credential=request_api_credential(request.headers),
        authorizations=_authorized_actions(request),
        actor=application_id or "api",
        application_id=application_id,
        client_instance_id=request.headers.get("x-lewlm-client-instance-id"),
        details={"model_id": model_id, "request_id": request_id},
    )
    decision, lifecycle = await services.model_router.unload_model_lifecycle(
        model_id,
        drain=True,
        timeout_seconds=float(services.settings.model_drain_timeout_seconds),
        request_id=request_id,
        application_id=application_id,
    )
    return ModelLifecycleResponse(
        status="drained",
        model_id=decision.model_id,
        runtime=decision.runtime_name,
        reason=lifecycle.reason,
        **lifecycle.model_dump(exclude={"model_id", "runtime", "reason"}),
    )


@router.post(
    "/v1/models/{model_id}/drain-operations",
    response_model=LifecycleOperationRecord,
    status_code=202,
)
async def create_drain_operation(
    model_id: str,
    payload: AsyncDrainRequest,
    request: Request,
) -> LifecycleOperationRecord:
    """Start an observable drain without holding the HTTP request open."""

    services = get_services(request)
    application_id = request.headers.get("x-lewlm-application-id")
    client_instance_id = request.headers.get("x-lewlm-client-instance-id")
    services.tool_authorizer.require_lifecycle(
        LifecycleCapability.DRAIN_MODEL,
        credential=request_api_credential(request.headers),
        authorizations=_authorized_actions(request),
        actor=application_id or "api",
        application_id=application_id,
        client_instance_id=client_instance_id,
        details={"model_id": model_id, "idempotency_key": payload.idempotency_key},
    )
    return await services.lifecycle_operation_manager.submit_drain(
        model_id,
        timeout_seconds=(
            payload.timeout_seconds
            if payload.timeout_seconds is not None
            else float(services.settings.model_drain_timeout_seconds)
        ),
        application_id=application_id,
        client_instance_id=client_instance_id,
        idempotency_key=payload.idempotency_key,
    )


@router.get("/v1/models/{model_id}/residency", response_model=ModelResidencySnapshot | None)
async def model_residency(model_id: str, request: Request, runtime: str | None = None) -> ModelResidencySnapshot | None:
    """Return live residency state for one registered model."""

    services = get_services(request)
    application_id = request.headers.get("x-lewlm-application-id")
    services.tool_authorizer.require_lifecycle(
        LifecycleCapability.INSPECT_RESIDENCY,
        credential=request_api_credential(request.headers),
        authorizations=_authorized_actions(request),
        actor=application_id or "api",
        application_id=application_id,
        client_instance_id=request.headers.get("x-lewlm-client-instance-id"),
        details={"model_id": model_id, "runtime": runtime},
    )
    services.model_registry.get_manifest(model_id)
    return await services.model_residency_manager.get_residency(model_id, runtime_name=runtime)


def _authorized_actions(request: Request) -> list[str]:
    return [
        item.strip()
        for item in request.headers.get("x-lewlm-authorized-actions", "").split(",")
        if item.strip()
    ]
