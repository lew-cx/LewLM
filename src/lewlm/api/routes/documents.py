"""Document generation routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from lewlm.api.dependencies import get_services
from lewlm.api.schemas.documents import (
    DocumentGenerateRequest,
    DocumentGenerateResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentTransformResponse,
)
from lewlm.core.execution_metadata import ExecutionMetadata, build_tool_execution_metadata
from lewlm.documents.skills.models import DocumentTransformRequest
from lewlm.tools.models import (
    DocumentGenerateToolRequest,
    DocumentIngestToolRequest,
    DocumentTransformToolRequest,
    GenerateDocumentToolInput,
    IngestDocumentToolInput,
    ToolExecutionEnvelope,
)


router = APIRouter(tags=["documents"])


@router.post("/v1/documents/generate", response_model=DocumentGenerateResponse)
def generate_document(payload: DocumentGenerateRequest, request: Request) -> DocumentGenerateResponse:
    """Render a deterministic document artifact from the structured IR."""

    services = get_services(request)
    envelope = services.tool_execution_service.execute(
        DocumentGenerateToolRequest(
            input=GenerateDocumentToolInput(
                output_format=payload.output_format,
                document=payload.document,
                file_name=payload.file_name,
                authorized_actions=payload.authorized_actions,
                idempotency_key=payload.idempotency_key,
            ),
        ),
        actor="api",
        allowed_file_roots=services.settings.file_access_roots,
        emit_tool_events=False,
    )
    return DocumentGenerateResponse(
        request_id=envelope.request_id,
        idempotency_key=envelope.idempotency_key,
        idempotent_replay=envelope.idempotent_replay,
        file_name=str(envelope.result["file_name"]),
        output_format=envelope.result["output_format"],
        media_type=str(envelope.result["media_type"]),
        size_bytes=int(envelope.result["size_bytes"]),
        content_base64=str(envelope.result["content_base64"]),
        metadata=_document_metadata(envelope),
    )


@router.post("/v1/documents/ingest", response_model=DocumentIngestResponse)
def ingest_document(payload: DocumentIngestRequest, request: Request) -> DocumentIngestResponse:
    """Extract a structured document representation from local files."""

    services = get_services(request)
    envelope = services.tool_execution_service.execute(
        DocumentIngestToolRequest(
            input=IngestDocumentToolInput(
                paths=payload.paths,
                title=payload.title,
                authorized_actions=payload.authorized_actions,
                idempotency_key=payload.idempotency_key,
            ),
        ),
        actor="api",
        allowed_file_roots=services.settings.file_access_roots,
        emit_tool_events=False,
    )
    return DocumentIngestResponse.model_validate(
        {
            **envelope.result,
            "request_id": envelope.request_id,
            "idempotency_key": envelope.idempotency_key,
            "idempotent_replay": envelope.idempotent_replay,
            "metadata": _document_metadata(envelope).model_dump(mode="json"),
        },
    )


@router.post("/v1/documents/transform", response_model=DocumentTransformResponse)
def transform_document(payload: DocumentTransformRequest, request: Request) -> DocumentTransformResponse:
    """Run a built-in document skill and render its output artifact."""

    services = get_services(request)
    envelope = services.tool_execution_service.execute(
        DocumentTransformToolRequest(input=payload),
        actor="api",
        allowed_file_roots=services.settings.file_access_roots,
        emit_tool_events=False,
    )
    return DocumentTransformResponse(
        request_id=envelope.request_id,
        idempotency_key=envelope.idempotency_key,
        idempotent_replay=envelope.idempotent_replay,
        skill=payload.skill,
        file_name=str(envelope.result["file_name"]),
        output_format=envelope.result["output_format"],
        media_type=str(envelope.result["media_type"]),
        size_bytes=int(envelope.result["size_bytes"]),
        content_base64=str(envelope.result["content_base64"]),
        metadata=_document_metadata(envelope),
    )


def _document_metadata(envelope: ToolExecutionEnvelope) -> ExecutionMetadata:
    return build_tool_execution_metadata(
        request_id=envelope.request_id,
        created=int(envelope.trace.started_at.timestamp()),
        tool_name=envelope.tool,
        duration_milliseconds=envelope.trace.duration_ms,
        idempotency_key=envelope.idempotency_key,
        idempotent_replay=envelope.idempotent_replay,
    )
