"""Typed local tool catalog and execution models."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from lewlm.core.contracts import utc_now
from lewlm.documents.ingest.models import DocumentIngestResult
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.documents.skills.models import DocumentTransformRequest
from lewlm.tools.descriptors import LocalToolDescriptor


class AuthorizedToolInput(BaseModel):
    authorized_actions: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None


class GenerateDocumentToolInput(AuthorizedToolInput):
    output_format: DocumentOutputFormat
    document: DocumentIR
    file_name: str | None = None


class IngestDocumentToolInput(AuthorizedToolInput):
    paths: list[str] = Field(default_factory=list)
    title: str | None = None


class DocumentGenerateToolRequest(BaseModel):
    tool: Literal["documents.generate"] = "documents.generate"
    input: GenerateDocumentToolInput


class DocumentIngestToolRequest(BaseModel):
    tool: Literal["documents.ingest"] = "documents.ingest"
    input: IngestDocumentToolInput


class DocumentTransformToolRequest(BaseModel):
    tool: Literal["documents.transform"] = "documents.transform"
    input: DocumentTransformRequest


ToolExecutionRequest = Annotated[
    DocumentGenerateToolRequest | DocumentIngestToolRequest | DocumentTransformToolRequest,
    Field(discriminator="tool"),
]

TOOL_EXECUTION_REQUEST_ADAPTER = TypeAdapter(ToolExecutionRequest)


class ToolExecutionTrace(BaseModel):
    tool: str
    version: str
    execution_mode: Literal["local"] = "local"
    actor: Literal["api", "cli"]
    required_authorization: str
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)
    duration_ms: int = 0
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionEnvelope(BaseModel):
    request_id: str
    tool: str
    idempotency_key: str | None = None
    idempotent_replay: bool = False
    trace: ToolExecutionTrace
    result: dict[str, Any] = Field(default_factory=dict)


def parse_tool_execution_request(payload: str | bytes) -> ToolExecutionRequest:
    """Parse a JSON payload into a typed local tool execution request."""

    return TOOL_EXECUTION_REQUEST_ADAPTER.validate_json(payload)
