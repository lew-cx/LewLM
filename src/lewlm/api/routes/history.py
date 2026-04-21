"""Session persistence routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from lewlm.api.dependencies import get_services
from lewlm.api.schemas.history import (
    SessionCreateRequest,
    SessionDeleteResponse,
    SessionImportRequest,
    SessionListResponse,
    SessionMessagesResponse,
)
from lewlm.history.models import SessionDetail, SessionExportBundle, SessionRecord


router = APIRouter(tags=["history"])


@router.post("/v1/sessions", response_model=SessionRecord)
def create_session(payload: SessionCreateRequest, request: Request) -> SessionRecord:
    """Create a new persisted local session."""

    services = get_services(request)
    session = services.session_history_service.create_session(
        title=payload.title,
        metadata=payload.metadata,
        context_policy=payload.context_policy,
    )
    services.audit_logger.record(
        action="session_create",
        outcome="success",
        actor="api",
        details={"session_id": session.session_id},
    )
    return session


@router.get("/v1/sessions", response_model=SessionListResponse)
def list_sessions(request: Request) -> SessionListResponse:
    """List persisted local sessions."""

    services = get_services(request)
    sessions = services.session_history_service.list_sessions()
    return SessionListResponse(count=len(sessions), items=sessions)


@router.get("/v1/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, request: Request) -> SessionDetail:
    """Return a persisted session and its turn history."""

    services = get_services(request)
    return services.session_history_service.get_session_detail(session_id)


@router.get("/v1/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
def get_session_messages(session_id: str, request: Request) -> SessionMessagesResponse:
    """Return flattened chat messages for a persisted session."""

    services = get_services(request)
    messages = services.session_history_service.list_messages(session_id)
    return SessionMessagesResponse(session_id=session_id, count=len(messages), messages=messages)


@router.get("/v1/sessions/{session_id}/export", response_model=SessionExportBundle)
def export_session(session_id: str, request: Request) -> SessionExportBundle:
    """Export a persisted session as a portable bundle."""

    services = get_services(request)
    bundle = services.session_history_service.export_session(session_id)
    services.audit_logger.record(
        action="session_export",
        outcome="success",
        actor="api",
        details={"session_id": session_id, "turn_count": len(bundle.turns)},
    )
    return bundle


@router.post("/v1/sessions/import", response_model=SessionDetail)
def import_session(payload: SessionImportRequest, request: Request) -> SessionDetail:
    """Import a portable session bundle into local persistence."""

    services = get_services(request)
    session = services.session_history_service.import_session(payload.bundle, title=payload.title)
    services.audit_logger.record(
        action="session_import",
        outcome="success",
        actor="api",
        details={"session_id": session.session_id, "turn_count": len(session.turns)},
    )
    return session


@router.delete("/v1/sessions/{session_id}", response_model=SessionDeleteResponse)
def delete_session(session_id: str, request: Request) -> SessionDeleteResponse:
    """Delete a persisted session and its turns."""

    services = get_services(request)
    services.session_history_service.delete_session(session_id)
    services.audit_logger.record(
        action="session_delete",
        outcome="success",
        actor="api",
        details={"session_id": session_id},
    )
    return SessionDeleteResponse(session_id=session_id)
