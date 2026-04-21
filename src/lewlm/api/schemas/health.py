"""Health API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

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


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    time: datetime
    install_profiles: InstallProfileSummary
    readiness: ServiceReadinessSummary
    storage: StorageHealth
    configuration: ConfigurationHealth
    cluster: dict[str, Any] | None = None
