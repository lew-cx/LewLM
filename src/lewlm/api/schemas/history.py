"""Session persistence API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from lewlm.core.contracts import GenerateMessage
from lewlm.history.models import SessionContextPolicy, SessionExportBundle, SessionRecord


class SessionCreateRequest(BaseModel):
    title: str | None = None
    context_policy: SessionContextPolicy = "full_history"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionUpdateRequest(BaseModel):
    """Partial update for a session. Omitted fields are left unchanged."""

    title: str | None = None
    context_policy: SessionContextPolicy | None = None
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Metadata to apply. Merged into existing metadata unless `replace_metadata` is true.",
    )
    replace_metadata: bool = Field(
        default=False,
        description="Replace stored metadata outright instead of merging.",
    )

    @model_validator(mode="after")
    def _require_a_change(self) -> "SessionUpdateRequest":
        if self.title is None and self.context_policy is None and self.metadata is None:
            raise ValueError("Session updates require at least one of `title`, `context_policy`, or `metadata`.")
        return self


class SessionImportRequest(BaseModel):
    bundle: SessionExportBundle
    title: str | None = None


class SessionListResponse(BaseModel):
    count: int
    items: list[SessionRecord]


class SessionMessagesResponse(BaseModel):
    session_id: str
    count: int
    messages: list[GenerateMessage]


class SessionDeleteResponse(BaseModel):
    status: Literal["deleted"] = "deleted"
    session_id: str
