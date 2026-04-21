"""Explicit authorization gates for tool-like operations."""

from __future__ import annotations

from enum import Enum
from typing import Any

from lewlm.config.settings import LewLMSettings
from lewlm.core.errors import ToolAuthorizationError
from lewlm.security.audit import AuditLogger


class ToolAction(str, Enum):
    DOCUMENT_GENERATE = "document_generate"
    DOCUMENT_INGEST = "document_ingest"
    DOCUMENT_TRANSFORM = "document_transform"
    MODEL_CONVERSION = "model_conversion"


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

