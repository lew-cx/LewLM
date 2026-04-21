"""Session persistence service."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import GenerateMessage
from lewlm.core.errors import PrivacyModeError, SessionNotFoundError
from lewlm.history.models import (
    SessionContextPolicy,
    SessionDetail,
    SessionExportBundle,
    SessionRecord,
    SessionRequestKind,
    SessionTurnRecord,
    flatten_session_turns,
)
from lewlm.storage.metadata import MetadataStore


class SessionHistoryService:
    """Persist, retrieve, and export local chat sessions."""

    def __init__(self, *, metadata_store: MetadataStore, settings: LewLMSettings) -> None:
        self.metadata_store = metadata_store
        self.settings = settings

    def create_session(
        self,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        context_policy: SessionContextPolicy = "full_history",
    ) -> SessionRecord:
        self._require_persistence_enabled()
        session = SessionRecord(
            session_id=str(uuid4()),
            title=title,
            context_policy=context_policy,
            metadata=dict(metadata or {}),
        )
        self.metadata_store.upsert_session(session)
        return session

    def list_sessions(self) -> list[SessionRecord]:
        self._require_persistence_enabled()
        return self.metadata_store.list_sessions()

    def get_session(self, session_id: str) -> SessionRecord:
        self._require_persistence_enabled()
        return self._require_session(session_id)

    def get_session_detail(self, session_id: str) -> SessionDetail:
        self._require_persistence_enabled()
        session = self._require_session(session_id)
        return SessionDetail.model_validate(
            {
                **session.model_dump(mode="python"),
                "turns": self.metadata_store.list_session_turns(session_id),
            },
        )

    def list_messages(self, session_id: str) -> list[GenerateMessage]:
        detail = self.get_session_detail(session_id)
        return flatten_session_turns(detail.turns)

    def build_conversation_messages(
        self,
        *,
        session_id: str,
        new_messages: list[GenerateMessage],
    ) -> list[GenerateMessage]:
        detail = self.get_session_detail(session_id)
        return [*self._build_context_messages(detail), *new_messages]

    def record_turn(
        self,
        *,
        session_id: str,
        request_kind: SessionRequestKind,
        input_messages: list[GenerateMessage],
        response_message: GenerateMessage,
        requested_model_id: str | None,
        model_id: str,
        max_tokens: int,
        temperature: float,
        finish_reason: str,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionDetail:
        self._require_persistence_enabled()
        self._require_session(session_id)
        turn = SessionTurnRecord(
            turn_id=str(uuid4()),
            session_id=session_id,
            request_kind=request_kind,
            input_messages=list(input_messages),
            response_message=response_message,
            requested_model_id=requested_model_id,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            finish_reason=finish_reason,
            usage=dict(usage or {}),
            metadata=dict(metadata or {}),
        )
        self.metadata_store.append_session_turn(turn)
        return self.get_session_detail(session_id)

    def delete_session(self, session_id: str) -> SessionRecord:
        self._require_persistence_enabled()
        session = self._require_session(session_id)
        self.metadata_store.delete_session(session_id)
        return session

    def export_session(self, session_id: str) -> SessionExportBundle:
        detail = self.get_session_detail(session_id)
        return SessionExportBundle(
            session=SessionRecord.model_validate(detail.model_dump(mode="python", exclude={"turns"})),
            turns=detail.turns,
        )

    def import_session(
        self,
        bundle: SessionExportBundle,
        *,
        title: str | None = None,
    ) -> SessionDetail:
        self._require_persistence_enabled()
        imported_metadata = dict(bundle.session.metadata)
        imported_metadata["imported_from_session_id"] = bundle.session.session_id
        session = SessionRecord(
            session_id=str(uuid4()),
            title=title or bundle.session.title,
            context_policy=bundle.session.context_policy,
            metadata=imported_metadata,
            created_at=bundle.session.created_at,
            updated_at=bundle.session.updated_at,
        )
        self.metadata_store.upsert_session(session)
        for turn in sorted(bundle.turns, key=lambda item: (item.created_at, item.turn_id)):
            self.metadata_store.append_session_turn(
                turn.model_copy(
                    update={
                        "turn_id": str(uuid4()),
                        "session_id": session.session_id,
                        "metadata": {
                            **turn.metadata,
                            "imported_from_turn_id": turn.turn_id,
                        },
                    },
                ),
            )
        return self.get_session_detail(session.session_id)

    def _require_persistence_enabled(self) -> None:
        if self.settings.privacy_mode:
            raise PrivacyModeError("Session persistence is disabled while privacy mode is enabled.")

    def _require_session(self, session_id: str) -> SessionRecord:
        session = self.metadata_store.get_session(session_id)
        if session is None:
            raise SessionNotFoundError("Session was not found.", details={"session_id": session_id})
        return session

    def _build_context_messages(self, detail: SessionDetail) -> list[GenerateMessage]:
        ordered_turns = sorted(detail.turns, key=lambda item: (item.created_at, item.turn_id))
        if detail.context_policy == "full_history":
            return flatten_session_turns(ordered_turns)
        if detail.context_policy == "last_turn":
            return flatten_session_turns(ordered_turns[-1:])
        if detail.context_policy == "summary_and_last_turn":
            return self._summary_and_last_turn_messages(detail, ordered_turns)
        return flatten_session_turns(ordered_turns)

    def _summary_and_last_turn_messages(
        self,
        detail: SessionDetail,
        turns: list[SessionTurnRecord],
    ) -> list[GenerateMessage]:
        if len(turns) <= 1:
            return flatten_session_turns(turns)
        compacted_turns = turns[:-1]
        recent_turns = turns[-1:]
        summary_lines = [
            f"Session summary for {detail.title or detail.session_id}.",
            f"Compacted {len(compacted_turns)} earlier turn(s) into this note; keep the latest turn verbatim below.",
        ]
        for index, turn in enumerate(compacted_turns, start=1):
            user_excerpt = self._summarize_messages(turn.input_messages)
            assistant_excerpt = self._truncate_text(turn.response_message.content, limit=120)
            summary_lines.append(
                (
                    f"{index}. {turn.request_kind} via {turn.model_id} "
                    f"(max_tokens={turn.max_tokens}, temperature={turn.temperature:g}) | "
                    f"user={user_excerpt} | assistant={assistant_excerpt}"
                )
            )
        return [
            GenerateMessage(role="system", content="\n".join(summary_lines)),
            *flatten_session_turns(recent_turns),
        ]

    def _summarize_messages(self, messages: list[GenerateMessage]) -> str:
        if not messages:
            return "(no input)"
        return " || ".join(self._summarize_message(message) for message in messages)

    def _summarize_message(self, message: GenerateMessage) -> str:
        normalized_text = re.sub(r"\s+", " ", message.content).strip()
        excerpt = self._truncate_text(normalized_text or "(empty)", limit=80)
        if not message.attachments:
            return f"{message.role}: {excerpt}"
        attachment_names = ", ".join(
            self._truncate_text(attachment.name, limit=24)
            for attachment in message.attachments[:3]
        )
        if len(message.attachments) > 3:
            attachment_names = f"{attachment_names}, +{len(message.attachments) - 3} more"
        return f"{message.role}: {excerpt} [attachments: {attachment_names}]"

    def _truncate_text(self, value: str, *, limit: int) -> str:
        if len(value) <= limit:
            return value
        return f"{value[: limit - 3]}..."
