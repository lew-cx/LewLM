"""Health and diagnostics routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from lewlm.api.dependencies import get_services
from lewlm.api.schemas.health import ConfigurationHealth, HealthResponse, StorageHealth
from lewlm.core.contracts import utc_now
from lewlm.install_profiles import summarize_install_profiles


router = APIRouter(tags=["health"])


@router.get("/v1/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return service and storage health for the local LewLM instance."""

    services = get_services(request)
    storage_snapshot = services.metadata_store.snapshot()
    inventory = services.model_registry.inventory()
    settings = services.settings
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.version,
        time=utc_now(),
        install_profiles=summarize_install_profiles(),
        readiness=services.model_router.capability_readiness_summary(),
        storage=StorageHealth(
            healthy=True,
            database_path=storage_snapshot["database_path"],
            schema_version=storage_snapshot["schema_version"],
            model_count=inventory.count,
        ),
        configuration=ConfigurationHealth(
            data_dir=str(settings.data_dir),
            models_dir=[str(path) for path in settings.models_dir],
            runtime_packs=services.pack_registry.runtime_pack_reports(),
            feature_packs=services.pack_registry.feature_pack_reports(),
            privacy_mode=settings.privacy_mode,
            telemetry_enabled=settings.telemetry_enabled,
            allow_outbound_network=settings.allow_outbound_network,
            audit_log_enabled=settings.audit_log_enabled,
            persistence_encryption_enabled=settings.persistence_encryption_enabled,
            tool_authorization_required=settings.tool_authorization_required,
            parser_sandbox_enabled=settings.parser_sandbox_enabled,
            tool_sandbox_enabled=settings.tool_sandbox_enabled,
            conversion_sandbox_enabled=settings.conversion_sandbox_enabled,
        ),
        cluster=services.cluster_service.status().model_dump(mode="json"),
    )
