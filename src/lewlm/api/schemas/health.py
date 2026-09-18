"""Health API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from lewlm.core.contracts import ServiceReadinessSummary
from lewlm.install_profiles import InstallProfileSummary
from lewlm.pack_registry import PackReport


class StorageHealth(BaseModel):
    healthy: bool
    database_path: str
    schema_version: int
    model_count: int


class ConfigurationHealth(BaseModel):
    data_dir: str
    models_dir: list[str]
    runtime_packs: list[PackReport] = []
    feature_packs: list[PackReport] = []
    privacy_mode: bool
    telemetry_enabled: bool
    allow_outbound_network: bool
    audit_log_enabled: bool
    persistence_encryption_enabled: bool
    tool_authorization_required: bool
    parser_sandbox_enabled: bool
    tool_sandbox_enabled: bool
    conversion_sandbox_enabled: bool


class EngineHealth(BaseModel):
    """One configured external engine, from LewLM's cached inventory — never a live probe.

    `status: ok` on the health response means *this service*; an engine may be
    unreachable at the same time, and a model may be cold. Read `engines` here,
    and `startup.warm_models` on `GET /v1/runtime`, instead of guessing from
    the HTTP status.
    """

    endpoint_id: str
    profile: str
    enabled: bool = True
    state: str = "unknown"
    inventory_age_seconds: float | None = None
    advertised_model_count: int = 0
    inventory_error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    runtime_instance_id: str | None = None
    started_at: datetime | None = None
    process_id: int | None = None
    hostname: str | None = None
    time: datetime
    install_profiles: InstallProfileSummary
    readiness: ServiceReadinessSummary
    storage: StorageHealth
    configuration: ConfigurationHealth
    cluster: dict[str, Any] | None = None
    engines: list[EngineHealth] = Field(default_factory=list)
