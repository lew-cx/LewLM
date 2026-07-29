"""Explicit authorization gates for tool-like operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hmac
from typing import Any

from lewlm.config.settings import LewLMSettings
from lewlm.core.errors import ToolAuthorizationError
from lewlm.security.audit import AuditLogger


class ToolAction(str, Enum):
    DOCUMENT_GENERATE = "document_generate"
    DOCUMENT_INGEST = "document_ingest"
    DOCUMENT_TRANSFORM = "document_transform"
    MODEL_CONVERSION = "model_conversion"
    MODEL_WARM = "model_warm"
    MODEL_DRAIN = "model_drain"
    MODEL_UNLOAD = "model_unload"
    MODEL_RESIDENCY_INSPECT = "model_residency_inspect"
    RUNTIME_DIAGNOSTICS = "runtime_diagnostics"


class LifecycleRole(str, Enum):
    """Trusted lifecycle roles assigned by scoped API credentials."""

    OPERATOR = "operator"
    ADMINISTRATOR = "administrator"


class LifecycleCapability(str, Enum):
    """Explicit permissions for shared-runtime lifecycle surfaces."""

    LOAD_MODEL = "load_model"
    DRAIN_MODEL = "drain_model"
    UNLOAD_MODEL = "unload_model"
    FORCE_UNLOAD_MODEL = "force_unload_model"
    INSPECT_RESIDENCY = "inspect_residency"
    RUN_DISRUPTIVE_DIAGNOSTICS = "run_disruptive_diagnostics"


_CAPABILITY_ROLE = {
    LifecycleCapability.LOAD_MODEL: LifecycleRole.OPERATOR,
    LifecycleCapability.DRAIN_MODEL: LifecycleRole.OPERATOR,
    LifecycleCapability.INSPECT_RESIDENCY: LifecycleRole.OPERATOR,
    LifecycleCapability.UNLOAD_MODEL: LifecycleRole.ADMINISTRATOR,
    LifecycleCapability.FORCE_UNLOAD_MODEL: LifecycleRole.ADMINISTRATOR,
    LifecycleCapability.RUN_DISRUPTIVE_DIAGNOSTICS: LifecycleRole.ADMINISTRATOR,
}
_CAPABILITY_LEGACY_ACTION = {
    LifecycleCapability.LOAD_MODEL: ToolAction.MODEL_WARM,
    LifecycleCapability.DRAIN_MODEL: ToolAction.MODEL_DRAIN,
    LifecycleCapability.UNLOAD_MODEL: ToolAction.MODEL_UNLOAD,
    LifecycleCapability.FORCE_UNLOAD_MODEL: ToolAction.MODEL_UNLOAD,
    LifecycleCapability.INSPECT_RESIDENCY: ToolAction.MODEL_RESIDENCY_INSPECT,
    LifecycleCapability.RUN_DISRUPTIVE_DIAGNOSTICS: ToolAction.RUNTIME_DIAGNOSTICS,
}


@dataclass(frozen=True, slots=True)
class LifecycleAuthorizationDecision:
    capability: LifecycleCapability
    required_role: LifecycleRole
    granted_role: LifecycleRole | None
    mode: str


class ToolAuthorizer:
    """Require explicit action authorization when the policy is enabled."""

    def __init__(self, *, settings: LewLMSettings, audit_logger: AuditLogger) -> None:
        self.settings = settings
        self.audit_logger = audit_logger

    def require(
        self,
        action: ToolAction,
        *,
        authorizations: list[str] | tuple[str, ...] | None,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.settings.tool_authorization_required:
            return
        allowed_actions = {
            value.strip().casefold()
            for value in (authorizations or ())
            if isinstance(value, str) and value.strip()
        }
        action_name = action.value
        audit_details = {
            "required_action": action_name,
            "authorized_actions": sorted(allowed_actions),
            **(details or {}),
        }
        if action_name not in allowed_actions:
            self.audit_logger.record(
                action=action_name,
                outcome="denied",
                actor=actor,
                details=audit_details,
            )
            raise ToolAuthorizationError(
                "This operation requires explicit authorization.",
                details=audit_details,
            )
        self.audit_logger.record(
            action=action_name,
            outcome="authorized",
            actor=actor,
            details=audit_details,
        )

    def require_lifecycle(
        self,
        capability: LifecycleCapability,
        *,
        credential: str | None,
        authorizations: list[str] | tuple[str, ...] | None,
        actor: str,
        application_id: str | None = None,
        client_instance_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> LifecycleAuthorizationDecision:
        """Authorize one lifecycle capability through scoped credentials.

        Scoped operator/admin keys are authoritative when configured. The old
        explicit-action header remains a compatibility path only for deployments
        that have not configured scoped lifecycle credentials.
        """

        required_role = _CAPABILITY_ROLE[capability]
        scoped_credentials_configured = bool(
            self.settings.lifecycle_operator_api_keys
            or self.settings.lifecycle_administrator_api_keys
        )
        granted_role = self._role_for_credential(credential) if scoped_credentials_configured else None
        mode = "scoped_api_key" if scoped_credentials_configured else "open"
        allowed = False
        if scoped_credentials_configured:
            allowed = granted_role == LifecycleRole.ADMINISTRATOR or granted_role == required_role
        elif self.settings.tool_authorization_required:
            mode = "legacy_explicit_action"
            declared = {
                value.strip().casefold()
                for value in (authorizations or ())
                if isinstance(value, str) and value.strip()
            }
            if "lifecycle_administrator" in declared:
                granted_role = LifecycleRole.ADMINISTRATOR
            elif "lifecycle_operator" in declared:
                granted_role = LifecycleRole.OPERATOR
            legacy_action = _CAPABILITY_LEGACY_ACTION[capability].value
            allowed = (
                legacy_action in declared
                or granted_role == LifecycleRole.ADMINISTRATOR
                or (granted_role == LifecycleRole.OPERATOR and required_role == LifecycleRole.OPERATOR)
            )
        else:
            allowed = True

        audit_details = {
            "lifecycle_capability": capability.value,
            "required_role": required_role.value,
            "granted_role": granted_role.value if granted_role is not None else None,
            "authorization_mode": mode,
            "application_id": application_id,
            "client_instance_id": client_instance_id,
            **(details or {}),
        }
        outcome = "authorized" if allowed else "denied"
        self.audit_logger.record(
            action=_CAPABILITY_LEGACY_ACTION[capability].value,
            outcome=outcome,
            actor=actor,
            details=audit_details,
        )
        if not allowed:
            raise ToolAuthorizationError(
                f"This lifecycle operation requires the {required_role.value} role.",
                details=audit_details,
            )
        return LifecycleAuthorizationDecision(
            capability=capability,
            required_role=required_role,
            granted_role=granted_role,
            mode=mode,
        )

    def _role_for_credential(self, credential: str | None) -> LifecycleRole | None:
        if credential is None:
            return None
        administrator_keys = (
            secret.get_secret_value()
            for secret in self.settings.lifecycle_administrator_api_keys
        )
        if any(hmac.compare_digest(credential, candidate) for candidate in administrator_keys):
            return LifecycleRole.ADMINISTRATOR
        operator_keys = (
            secret.get_secret_value()
            for secret in self.settings.lifecycle_operator_api_keys
        )
        if any(hmac.compare_digest(credential, candidate) for candidate in operator_keys):
            return LifecycleRole.OPERATOR
        return None


def request_api_credential(headers: Any) -> str | None:
    """Extract the API credential without treating application identity as auth."""

    provided = headers.get("x-api-key")
    authorization = headers.get("authorization", "")
    if provided is None and isinstance(authorization, str) and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    return provided
