"""Session persistence API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from lewlm.core.contracts import GenerateMessage
from lewlm.history.models import SessionContextPolicy, SessionExportBundle, SessionRecord


class SessionCreateRequest(BaseModel):
    title: str | None = None
    context_policy: SessionContextPolicy = "full_history"
    metadata: dict[str, Any] = Field(default_factory=dict)


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
