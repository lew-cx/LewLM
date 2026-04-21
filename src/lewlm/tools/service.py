"""Local tool execution service."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import IdempotentOperationRecord, utc_now
from lewlm.core.errors import IdempotencyConflictError, SandboxExecutionError, error_from_dict
from lewlm.documents.ingest.service import DocumentIngestService
from lewlm.documents.service import DocumentGenerationService
from lewlm.documents.skills.service import DocumentTransformService
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.security.audit import AuditLogger
from lewlm.security.authorization import ToolAction, ToolAuthorizer
from lewlm.security.sandbox import run_in_subprocess
from lewlm.storage.metadata import MetadataStore
from lewlm.tools.catalog import ToolCatalogService
from lewlm.tools.models import (
    TOOL_EXECUTION_REQUEST_ADAPTER,
    DocumentGenerateToolRequest,
    DocumentIngestToolRequest,
    DocumentTransformToolRequest,
    ToolExecutionEnvelope,
    ToolExecutionRequest,
    ToolExecutionTrace,
)


@dataclass(slots=True)
class _ToolExecutionOutcome:
    result: dict[str, Any]
    summary: str
    details: dict[str, object]
    sandboxed: bool


class _RecordedEventBus:
    """Minimal event collector for sandboxed tool execution."""

    def __init__(self) -> None:
        self.events: list[StreamEvent] = []

    def publish_threadsafe(self, event: StreamEvent) -> None:
        self.events.append(event)


def _execute_tool_request(
    request: ToolExecutionRequest,
    *,
    document_generation_service: DocumentGenerationService,
    document_ingest_service: DocumentIngestService,
    document_transform_service: DocumentTransformService,
    allowed_file_roots: Sequence[Path | str] | None,
    base_dir: Path | str | None,
    request_id: str,
) -> tuple[dict[str, Any], str, dict[str, object]]:
    if isinstance(request, DocumentGenerateToolRequest):
        artifact = document_generation_service.generate(
            request.input.document,
            output_format=request.input.output_format,
            file_name=request.input.file_name,
            allowed_file_roots=allowed_file_roots,
            base_dir=base_dir,
            request_id=request_id,
        )
        return (
            {
                "file_name": artifact.file_name,
                "output_format": artifact.output_format.value,
                "media_type": artifact.media_type,
                "size_bytes": artifact.size_bytes,
                "content_base64": base64.b64encode(artifact.content).decode("ascii"),
            },
            f"Generated {artifact.file_name} as {artifact.output_format.value}.",
            {"size_bytes": artifact.size_bytes},
        )
    if isinstance(request, DocumentIngestToolRequest):
        result = document_ingest_service.ingest(
            request.input.paths,
            title=request.input.title,
            allowed_file_roots=allowed_file_roots,
            base_dir=base_dir,
            request_id=request_id,
        )
        return (
            result.model_dump(mode="json"),
            f"Ingested {len(result.sources)} source(s) into DocumentIR.",
            {"source_count": len(result.sources)},
        )
    artifact = document_transform_service.transform(
        request.input,
        allowed_file_roots=allowed_file_roots,
        base_dir=base_dir,
        request_id=request_id,
    )
    return (
        {
            "skill": request.input.skill,
            "file_name": artifact.file_name,
            "output_format": artifact.output_format.value,
            "media_type": artifact.media_type,
            "size_bytes": artifact.size_bytes,
            "content_base64": base64.b64encode(artifact.content).decode("ascii"),
        },
        f"Executed {request.input.skill} into {artifact.file_name}.",
        {"skill": request.input.skill, "size_bytes": artifact.size_bytes},
    )


def _execute_tool_request_in_worker(
    request_payload: dict[str, Any],
    parser_workspace_root: str,
    parser_sandbox_enabled: bool,
    parser_sandbox_timeout_seconds: int,
    parser_sandbox_clear_environment: bool,
    allowed_file_roots: tuple[str, ...] | None,
    base_dir: str | None,
    request_id: str,
) -> dict[str, Any]:
    event_bus = _RecordedEventBus()
    document_generation_service = DocumentGenerationService(event_bus=event_bus)
    document_ingest_service = DocumentIngestService(
        workspace_root=Path(parser_workspace_root),
        sandbox_enabled=parser_sandbox_enabled,
        sandbox_timeout_seconds=parser_sandbox_timeout_seconds,
        sandbox_clear_environment=parser_sandbox_clear_environment,
        event_bus=event_bus,
    )
    document_transform_service = DocumentTransformService(
        generation_service=document_generation_service,
        event_bus=event_bus,
    )
    request = TOOL_EXECUTION_REQUEST_ADAPTER.validate_python(request_payload)
    resolved_roots = tuple(Path(root) for root in allowed_file_roots) if allowed_file_roots is not None else None
    resolved_base_dir = Path(base_dir) if base_dir is not None else None
    try:
        result, summary, details = _execute_tool_request(
            request,
            document_generation_service=document_generation_service,
            document_ingest_service=document_ingest_service,
            document_transform_service=document_transform_service,
            allowed_file_roots=resolved_roots,
            base_dir=resolved_base_dir,
            request_id=request_id,
        )
        return {
            "ok": True,
            "events": [event.model_dump(mode="json") for event in event_bus.events],
            "result": result,
            "summary": summary,
            "details": details,
        }
    except Exception as exc:  # noqa: BLE001 - converted to structured parent-side handling
        error_payload: dict[str, Any]
        if hasattr(exc, "to_dict") and hasattr(exc, "status_code"):
            error_payload = {**exc.to_dict(), "status_code": exc.status_code}
        else:
            error_payload = {
                "message": str(exc),
                "error_type": type(exc).__name__,
            }
        return {
            "ok": False,
            "events": [event.model_dump(mode="json") for event in event_bus.events],
            "error": error_payload,
        }


class ToolExecutionService:
    """Execute registered local tools with explicit authorization and traces."""

    def __init__(
        self,
        *,
        settings: LewLMSettings,
        tool_catalog: ToolCatalogService,
        document_generation_service: DocumentGenerationService,
        document_ingest_service: DocumentIngestService,
        document_transform_service: DocumentTransformService,
        tool_authorizer: ToolAuthorizer,
        audit_logger: AuditLogger,
        event_bus: EventBus | None = None,
        metadata_store: MetadataStore | None = None,
    ) -> None:
        self.settings = settings
        self.tool_catalog = tool_catalog
        self.document_generation_service = document_generation_service
        self.document_ingest_service = document_ingest_service
        self.document_transform_service = document_transform_service
        self.tool_authorizer = tool_authorizer
        self.audit_logger = audit_logger
        self.event_bus = event_bus
        self.metadata_store = metadata_store

    def execute(
        self,
        request: ToolExecutionRequest,
        *,
        actor: str,
        allowed_file_roots: Sequence[Path | str] | None = None,
        base_dir: Path | str | None = None,
        request_id: str | None = None,
        emit_tool_events: bool = True,
    ) -> ToolExecutionEnvelope:
        descriptor = self.tool_catalog.get_tool(request.tool)
        started_at = utc_now()
        started_perf = time.perf_counter()
        resolved_request_id = request_id or str(uuid4())
        normalized_actor = "api" if actor == "api" else "cli"
        idempotency_key = self._idempotency_key(request)
        authorization_details = self._request_audit_details(descriptor.name, request)
        operation_started = False
        sandboxed = False

        try:
            self._authorize(
                descriptor.required_authorization,
                authorizations=request.input.authorized_actions,
                actor=actor,
                details=authorization_details,
            )
            replayed_response = self._lookup_idempotent_response(
                descriptor.name,
                request,
                idempotency_key=idempotency_key,
            )
            if replayed_response is not None:
                self.audit_logger.record(
                    action=descriptor.required_authorization,
                    outcome="idempotent_replay",
                    actor=actor,
                    details={"tool": descriptor.name, "idempotency_key": idempotency_key},
                )
                return replayed_response
            self._publish(
                EventType.TOOL_PENDING,
                request_id=resolved_request_id,
                descriptor=descriptor,
                actor=normalized_actor,
                emit_tool_events=emit_tool_events,
            )
            self._publish(
                EventType.TOOL_STARTED,
                request_id=resolved_request_id,
                descriptor=descriptor,
                actor=normalized_actor,
                emit_tool_events=emit_tool_events,
            )
            operation_started = True
            sandboxed = self.settings.tool_sandbox_enabled
            outcome = self._execute_operation(
                request,
                allowed_file_roots=allowed_file_roots,
                base_dir=base_dir,
                request_id=resolved_request_id,
            )
            sandboxed = outcome.sandboxed
            self.audit_logger.record(
                action=descriptor.required_authorization,
                outcome="success",
                actor=actor,
                details=self._success_audit_details(descriptor.name, request, outcome.result, sandboxed=sandboxed),
            )
            response = ToolExecutionEnvelope(
                request_id=resolved_request_id,
                tool=descriptor.name,
                idempotency_key=idempotency_key,
                trace=self._trace(
                    descriptor=descriptor,
                    actor=actor,
                    started_at=started_at,
                    started_perf=started_perf,
                    summary=outcome.summary,
                    details={**outcome.details, "sandboxed": sandboxed},
                ),
                result=outcome.result,
            )
        except Exception as exc:
            if operation_started:
                self.audit_logger.record(
                    action=descriptor.required_authorization,
                    outcome="failed",
                    actor=actor,
                    details={
                        **authorization_details,
                        "sandboxed": sandboxed,
                        "error": str(exc),
                    },
                )
            self._publish(
                EventType.TOOL_FAILED,
                request_id=resolved_request_id,
                descriptor=descriptor,
                actor=normalized_actor,
                emit_tool_events=emit_tool_events,
                error=str(exc),
            )
            raise

        self._store_idempotent_response(
            descriptor.name,
            request,
            idempotency_key=idempotency_key,
            response=response,
        )
        self._publish(
            EventType.TOOL_FINISHED,
            request_id=resolved_request_id,
            descriptor=descriptor,
            actor=normalized_actor,
            emit_tool_events=emit_tool_events,
            duration_ms=response.trace.duration_ms,
            summary=response.trace.summary,
        )
        return response

    def _authorize(
        self,
        action_name: str,
        *,
        authorizations: list[str],
        actor: str,
        details: dict[str, object],
    ) -> None:
        self.tool_authorizer.require(
            ToolAction(action_name),
            authorizations=authorizations,
            actor=actor,
            details=details,
        )

    def _trace(
        self,
        *,
        descriptor,
        actor: str,
        started_at,
        started_perf: float,
        summary: str,
        details: dict[str, object],
    ) -> ToolExecutionTrace:
        completed_at = utc_now()
        duration_ms = int(round((time.perf_counter() - started_perf) * 1000))
        return ToolExecutionTrace(
            tool=descriptor.name,
            version=descriptor.version,
            actor="api" if actor == "api" else "cli",
            required_authorization=descriptor.required_authorization,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=max(duration_ms, 0),
            summary=summary,
            details=details,
        )

    def _execute_operation(
        self,
        request: ToolExecutionRequest,
        *,
        allowed_file_roots: Sequence[Path | str] | None,
        base_dir: Path | str | None,
        request_id: str,
    ) -> _ToolExecutionOutcome:
        if not self.settings.tool_sandbox_enabled:
            result, summary, details = _execute_tool_request(
                request,
                document_generation_service=self.document_generation_service,
                document_ingest_service=self.document_ingest_service,
                document_transform_service=self.document_transform_service,
                allowed_file_roots=allowed_file_roots,
                base_dir=base_dir,
                request_id=request_id,
            )
            return _ToolExecutionOutcome(
                result=result,
                summary=summary,
                details=details,
                sandboxed=False,
            )
        payload = run_in_subprocess(
            _execute_tool_request_in_worker,
            request.model_dump(mode="json"),
            str(self.settings.parser_sandbox_dir),
            self.settings.parser_sandbox_enabled,
            self.settings.parser_sandbox_timeout_seconds,
            self.settings.parser_sandbox_clear_environment,
            tuple(str(Path(root)) for root in allowed_file_roots) if allowed_file_roots is not None else None,
            str(base_dir) if base_dir is not None else None,
            request_id,
            operation=f"Tool execution for {request.tool}",
            timeout_seconds=self.settings.tool_sandbox_timeout_seconds,
            enabled=True,
            clear_environment=self.settings.tool_sandbox_clear_environment,
            workspace_root=self.settings.tool_sandbox_dir,
        )
        self._replay_sandbox_events(payload["events"])
        if payload["ok"]:
            return _ToolExecutionOutcome(
                result=payload["result"],
                summary=payload["summary"],
                details=payload["details"],
                sandboxed=True,
            )
        error_payload = payload["error"]
        if "code" in error_payload:
            raise error_from_dict(error_payload)
        raise SandboxExecutionError(
            "Tool execution failed in the sandbox worker.",
            details={
                "tool": request.tool,
                "error": error_payload.get("message"),
                "error_type": error_payload.get("error_type"),
            },
        )

    def _request_audit_details(
        self,
        tool_name: str,
        request: ToolExecutionRequest,
    ) -> dict[str, object]:
        if isinstance(request, DocumentGenerateToolRequest):
            return {
                "tool": tool_name,
                "output_format": request.input.output_format.value,
            }
        if isinstance(request, DocumentIngestToolRequest):
            return {
                "tool": tool_name,
                "path_count": len(request.input.paths),
            }
        return {
            "tool": tool_name,
            "skill": request.input.skill,
            "output_format": request.input.output_format.value,
        }

    def _success_audit_details(
        self,
        tool_name: str,
        request: ToolExecutionRequest,
        result: dict[str, Any],
        *,
        sandboxed: bool,
    ) -> dict[str, object]:
        details = self._request_audit_details(tool_name, request)
        if isinstance(request, DocumentGenerateToolRequest):
            details["file_name"] = result["file_name"]
        elif isinstance(request, DocumentIngestToolRequest):
            details["source_types"] = [
                source["source_type"]
                for source in result.get("sources", [])
                if isinstance(source, dict) and "source_type" in source
            ]
        else:
            details["file_name"] = result["file_name"]
        details["sandboxed"] = sandboxed
        return details

    def _replay_sandbox_events(self, events: list[dict[str, Any]]) -> None:
        if self.event_bus is None:
            return
        for event_payload in events:
            self.event_bus.publish_threadsafe(StreamEvent.model_validate(event_payload))

    def _publish(
        self,
        event_type: EventType,
        *,
        request_id: str,
        descriptor,
        actor: str,
        emit_tool_events: bool = True,
        **payload: object,
    ) -> None:
        if self.event_bus is None or not emit_tool_events:
            return
        self.event_bus.publish_threadsafe(
            StreamEvent(
                type=event_type,
                scope=EventScope.REQUEST,
                payload={
                    "request_id": request_id,
                    "tool": descriptor.name,
                    "tool_version": descriptor.version,
                    "actor": actor,
                    **payload,
                },
            ),
        )

    @staticmethod
    def _idempotency_key(request: ToolExecutionRequest) -> str | None:
        if isinstance(request, DocumentTransformToolRequest):
            return request.input.idempotency_key
        return request.input.idempotency_key

    def _lookup_idempotent_response(
        self,
        operation_name: str,
        request: ToolExecutionRequest,
        *,
        idempotency_key: str | None,
    ) -> ToolExecutionEnvelope | None:
        if self.metadata_store is None or not idempotency_key:
            return None
        record = self.metadata_store.get_idempotent_operation_result(operation_name, idempotency_key)
        if record is None:
            return None
        request_hash = self._request_hash(operation_name, request)
        if record.request_hash != request_hash:
            raise IdempotencyConflictError(
                "The supplied idempotency key has already been used for a different request payload.",
                details={
                    "operation_name": operation_name,
                    "idempotency_key": idempotency_key,
                    "fallback_guidance": [
                        "Reuse the same request body when retrying an idempotent operation.",
                        "Choose a new idempotency key for a materially different request.",
                    ],
                },
            )
        envelope = ToolExecutionEnvelope.model_validate(record.response_payload)
        return envelope.model_copy(update={"idempotent_replay": True})

    def _store_idempotent_response(
        self,
        operation_name: str,
        request: ToolExecutionRequest,
        *,
        idempotency_key: str | None,
        response: ToolExecutionEnvelope,
    ) -> None:
        if self.metadata_store is None or not idempotency_key:
            return
        self.metadata_store.upsert_idempotent_operation_result(
            IdempotentOperationRecord(
                operation_name=operation_name,
                idempotency_key=idempotency_key,
                request_hash=self._request_hash(operation_name, request),
                response_payload=response.model_dump(mode="json"),
            ),
        )

    def _request_hash(self, operation_name: str, request: ToolExecutionRequest) -> str:
        payload = request.model_dump(mode="json")
        if isinstance(payload.get("input"), dict):
            authorizations = payload["input"].get("authorized_actions")
            if isinstance(authorizations, list):
                payload["input"]["authorized_actions"] = sorted({str(item) for item in authorizations})
        serialized = json.dumps(
            {"operation_name": operation_name, **payload},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
