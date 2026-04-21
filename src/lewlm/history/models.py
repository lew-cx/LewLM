"""Session persistence models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from lewlm.core.contracts import GenerateMessage, utc_now


SESSION_CONTEXT_POLICIES = ("full_history", "last_turn", "summary_and_last_turn")
SessionContextPolicy = Literal["full_history", "last_turn", "summary_and_last_turn"]
SessionRequestKind = Literal["chat.completions", "responses", "cli.chat"]


class SessionRecord(BaseModel):
    """Stored chat session metadata."""

    session_id: str
    title: str | None = None
    context_policy: SessionContextPolicy = "full_history"
    metadata: dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0
    turn_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SessionTurnRecord(BaseModel):
    """Single persisted turn inside a session."""

    turn_id: str
    session_id: str
    request_kind: SessionRequestKind
    input_messages: list[GenerateMessage] = Field(default_factory=list)
    response_message: GenerateMessage
    requested_model_id: str | None = None
    model_id: str
    max_tokens: int = 512
    temperature: float = 0.7
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class SessionDetail(SessionRecord):
    """Stored session metadata plus its turns."""

    turns: list[SessionTurnRecord] = Field(default_factory=list)


class SessionExportBundle(BaseModel):
    """Portable export bundle for a persisted session."""

    version: int = 1
    session: SessionRecord
    turns: list[SessionTurnRecord] = Field(default_factory=list)


def flatten_session_turns(turns: list[SessionTurnRecord]) -> list[GenerateMessage]:
    """Flatten stored turns into chronological chat messages."""

    messages: list[GenerateMessage] = []
    for turn in sorted(turns, key=lambda item: (item.created_at, item.turn_id)):
        messages.extend(turn.input_messages)
        messages.append(turn.response_message)
    return messages
