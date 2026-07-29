"""Document generation routes."""

from __future__ import annotations

from typing import Any

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
from lewlm.core.provenance import ComponentProvenance, renderer_provenance
from lewlm.documents.skills.models import DocumentTransformRequest
from lewlm.runtime.request_context import apply_body_correlation_id
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

    apply_body_correlation_id(payload.correlation_id)
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
        metadata=_document_metadata(
            envelope,
            correlation_id=payload.correlation_id,
            components=[renderer_provenance(str(envelope.result["output_format"])).model_dump(mode="json")],
        ),
    )


@router.post("/v1/documents/ingest", response_model=DocumentIngestResponse)
def ingest_document(payload: DocumentIngestRequest, request: Request) -> DocumentIngestResponse:
    """Extract a structured document representation from local files or uploaded bytes."""

    apply_body_correlation_id(payload.correlation_id)
    services = get_services(request)
    envelope = services.tool_execution_service.execute(
        DocumentIngestToolRequest(
            input=IngestDocumentToolInput(
                paths=payload.paths,
                sources=payload.sources,
                title=payload.title,
                authorized_actions=payload.authorized_actions,
                idempotency_key=payload.idempotency_key,
            ),
        ),
        actor="api",
        allowed_file_roots=services.settings.file_access_roots,
        emit_tool_events=False,
    )
    result = envelope.result
    return DocumentIngestResponse.model_validate(
        {
            **result,
            "request_id": envelope.request_id,
            "idempotency_key": envelope.idempotency_key,
            "idempotent_replay": envelope.idempotent_replay,
            "metadata": _document_metadata(
                envelope,
                correlation_id=payload.correlation_id,
                components=result.get("components"),
            ).model_dump(mode="json"),
        },
    )


@router.post("/v1/documents/transform", response_model=DocumentTransformResponse)
def transform_document(payload: DocumentTransformRequest, request: Request) -> DocumentTransformResponse:
    """Run a built-in document skill and render its output artifact."""

    apply_body_correlation_id(payload.correlation_id)
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
        metadata=_document_metadata(
            envelope,
            correlation_id=payload.correlation_id,
            components=[renderer_provenance(str(envelope.result["output_format"])).model_dump(mode="json")],
        ),
    )


def _document_metadata(
    envelope: ToolExecutionEnvelope,
    *,
    correlation_id: str | None = None,
    components: list[Any] | None = None,
) -> ExecutionMetadata:
    return build_tool_execution_metadata(
        request_id=envelope.request_id,
        created=int(envelope.trace.started_at.timestamp()),
        tool_name=envelope.tool,
        duration_milliseconds=envelope.trace.duration_ms,
        idempotency_key=envelope.idempotency_key,
        idempotent_replay=envelope.idempotent_replay,
        correlation_id=correlation_id,
        components=[ComponentProvenance.model_validate(item) for item in (components or ())],
    )
