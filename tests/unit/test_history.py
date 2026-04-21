from __future__ import annotations

import pytest

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import GenerateMessage
from lewlm.core.errors import PrivacyModeError
from lewlm.history.models import SessionExportBundle
from lewlm.history.service import SessionHistoryService
from lewlm.storage.metadata import MetadataStore


def _build_service(settings: LewLMSettings) -> SessionHistoryService:
    settings.prepare_directories()
    store = MetadataStore(settings.database_path)
    store.initialize()
    return SessionHistoryService(metadata_store=store, settings=settings)


def test_session_history_service_roundtrips_sessions_and_turns(session_enabled_settings: LewLMSettings) -> None:
    service = _build_service(session_enabled_settings)
    session = service.create_session(title="Milestone 9")

    detail = service.record_turn(
        session_id=session.session_id,
        request_kind="chat.completions",
        input_messages=[GenerateMessage(role="user", content="Draft the update")],
        response_message=GenerateMessage(role="assistant", content="Echo: Draft the update"),
        requested_model_id="requested-model",
        model_id="resolved-model",
        max_tokens=128,
        temperature=0.2,
        finish_reason="stop",
        usage={"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5},
    )

    assert detail.turn_count == 1
    assert detail.message_count == 2
    assert service.list_messages(session.session_id)[1].content == "Echo: Draft the update"

    bundle = service.export_session(session.session_id)
    assert isinstance(bundle, SessionExportBundle)
    imported = service.import_session(bundle, title="Imported milestone 9")
    assert imported.session_id != session.session_id
    assert imported.turn_count == 1
    assert imported.turns[0].response_message.content == "Echo: Draft the update"

    service.delete_session(session.session_id)
    assert len(service.list_sessions()) == 1


def test_session_history_service_applies_context_compaction_policies(session_enabled_settings: LewLMSettings) -> None:
    service = _build_service(session_enabled_settings)
    last_turn_session = service.create_session(title="Last turn", context_policy="last_turn")
    summary_session = service.create_session(title="Summary", context_policy="summary_and_last_turn")

    turns = [
        ("Capture the first note", "Echo: Capture the first note"),
        ("Capture the second note", "Echo: Capture the second note"),
    ]
    for prompt, response in turns:
        for session in (last_turn_session, summary_session):
            service.record_turn(
                session_id=session.session_id,
                request_kind="chat.completions",
                input_messages=[GenerateMessage(role="user", content=prompt)],
                response_message=GenerateMessage(role="assistant", content=response),
                requested_model_id="requested-model",
                model_id="resolved-model",
                max_tokens=128,
                temperature=0.2,
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5},
            )

    last_turn_messages = service.build_conversation_messages(
        session_id=last_turn_session.session_id,
        new_messages=[GenerateMessage(role="user", content="Third note")],
    )
    assert [message.role for message in last_turn_messages] == ["user", "assistant", "user"]
    assert last_turn_messages[0].content == "Capture the second note"

    summary_messages = service.build_conversation_messages(
        session_id=summary_session.session_id,
        new_messages=[GenerateMessage(role="user", content="Third note")],
    )
    assert [message.role for message in summary_messages] == ["system", "user", "assistant", "user"]
    assert "Compacted 1 earlier turn(s)" in summary_messages[0].content
    assert "Capture the first note" in summary_messages[0].content
    assert summary_messages[1].content == "Capture the second note"


def test_session_history_service_respects_privacy_mode(temp_settings: LewLMSettings) -> None:
    service = _build_service(temp_settings)
    with pytest.raises(PrivacyModeError):
        service.create_session(title="Blocked")
