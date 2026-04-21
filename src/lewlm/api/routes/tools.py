"""Local tool catalog and execution routes."""

from __future__ import annotations

from fastapi import APIRouter, Body, Request

from lewlm.api.dependencies import get_services
from lewlm.api.schemas.tools import ToolListResponse
from lewlm.tools.models import LocalToolDescriptor, ToolExecutionEnvelope, ToolExecutionRequest


router = APIRouter(tags=["tools"])
_TOOL_LIST_EXAMPLE = {
    "count": 1,
    "items": [
        {
            "name": "documents.generate",
            "version": "1.0.0",
            "description": "Render a structured DocumentIR payload into a deterministic output artifact.",
            "execution_mode": "local",
            "required_authorization": "document_generate",
            "result_type": "artifact",
            "input_schema": {
                "type": "object",
                "properties": {
                    "output_format": {"type": "string"},
                    "document": {"type": "object"},
                    "file_name": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["output_format", "document"],
            },
            "tags": ["documents", "generation", "local"],
            "aliases": ["document_generate"],
        }
    ],
}
_TOOL_DESCRIPTOR_EXAMPLE = _TOOL_LIST_EXAMPLE["items"][0]
_TOOL_EXECUTE_REQUEST_EXAMPLE = {
    "tool": "documents.generate",
    "input": {
        "output_format": "markdown",
        "file_name": "starter-proof.md",
        "document": {
            "title": "Starter proof",
            "sections": [
                {
                    "heading": "Summary",
                    "blocks": [
                        {
                            "type": "paragraph",
                            "text": "LewLM can render deterministic local document artifacts through the shared tool contract.",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "metadata": {},
        },
        "authorized_actions": ["document_generate"],
    },
}
_TOOL_EXECUTE_RESPONSE_EXAMPLE = {
    "request_id": "tool-001",
    "tool": "documents.generate",
    "idempotency_key": None,
    "idempotent_replay": False,
    "trace": {
        "tool": "documents.generate",
        "version": "1.0.0",
        "execution_mode": "local",
        "actor": "api",
        "required_authorization": "document_generate",
        "started_at": "2026-04-18T22:00:00Z",
        "completed_at": "2026-04-18T22:00:00Z",
        "duration_ms": 14,
        "summary": "Generated one deterministic markdown artifact.",
        "details": {"file_name": "starter-proof.md"},
    },
    "result": {
        "artifact": {
            "file_name": "starter-proof.md",
            "media_type": "text/markdown",
            "output_format": "markdown",
        }
    },
}


@router.get(
    "/v1/tools",
    response_model=ToolListResponse,
    responses={200: {"content": {"application/json": {"example": _TOOL_LIST_EXAMPLE}}}},
)
def list_tools(request: Request) -> ToolListResponse:
    """List registered local tools."""

    services = get_services(request)
    tools = services.tool_catalog_service.list_tools()
    return ToolListResponse(count=len(tools), items=tools)


@router.get(
    "/v1/tools/{tool_name}",
    response_model=LocalToolDescriptor,
    responses={200: {"content": {"application/json": {"example": _TOOL_DESCRIPTOR_EXAMPLE}}}},
)
def get_tool(tool_name: str, request: Request) -> LocalToolDescriptor:
    """Return a registered local tool descriptor."""

    services = get_services(request)
    return services.tool_catalog_service.get_tool(tool_name)


@router.post(
    "/v1/tools/execute",
    response_model=ToolExecutionEnvelope,
    responses={200: {"content": {"application/json": {"example": _TOOL_EXECUTE_RESPONSE_EXAMPLE}}}},
)
def execute_tool(
    request: Request,
    payload: ToolExecutionRequest = Body(openapi_examples={"default": {"value": _TOOL_EXECUTE_REQUEST_EXAMPLE}}),
) -> ToolExecutionEnvelope:
    """Execute a registered local tool and return its trace plus result."""

    services = get_services(request)
    return services.tool_execution_service.execute(
        payload,
        actor="api",
        allowed_file_roots=services.settings.file_access_roots,
    )
