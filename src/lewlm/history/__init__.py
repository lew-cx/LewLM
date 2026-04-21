"""Session persistence exports."""

from __future__ import annotations

from .models import SESSION_CONTEXT_POLICIES, SessionDetail, SessionExportBundle, SessionRecord, SessionTurnRecord

__all__ = [
    "SESSION_CONTEXT_POLICIES",
    "SessionDetail",
    "SessionExportBundle",
    "SessionHistoryService",
    "SessionRecord",
    "SessionTurnRecord",
]


def __getattr__(name: str):
    if name == "SessionHistoryService":
        from .service import SessionHistoryService

        return SessionHistoryService
    raise AttributeError(name)
