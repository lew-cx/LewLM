"""High-level document generation service."""

from __future__ import annotations

import re
from pathlib import Path
from collections.abc import Sequence
from uuid import uuid4

from lewlm.core.errors import PackUnavailableError
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.documents.render.service import DocumentRendererRegistry, GeneratedDocumentArtifact
from lewlm.documents.validators.ir import DocumentIRValidator
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.security.files import scope_document_paths


class DocumentGenerationService:
    """Validate document IR and render deterministic artifacts."""

    def __init__(
        self,
        *,
        validator: DocumentIRValidator | None = None,
        renderer_registry: DocumentRendererRegistry | None = None,
        event_bus: EventBus | None = None,
        enabled: bool = True,
        disabled_reason: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.disabled_reason = disabled_reason or "Document generation is disabled for this LewLM process."
        self.validator = (validator or DocumentIRValidator()) if enabled else None
        self.renderer_registry = (
            renderer_registry or DocumentRendererRegistry(self.validator)
            if enabled
            else None
        )
        self.event_bus = event_bus

    def generate(
        self,
        document: DocumentIR,
        *,
        output_format: DocumentOutputFormat,
        file_name: str | None = None,
        allowed_file_roots: Sequence[Path | str] | None = None,
        base_dir: Path | str | None = None,
        request_id: str | None = None,
    ) -> GeneratedDocumentArtifact:
        self._ensure_enabled()
        resolved_request_id = request_id or str(uuid4())
        self._publish(
            EventType.DOCUMENT_RENDER_STARTED,
            {
                "request_id": resolved_request_id,
                "title": document.title,
                "output_format": output_format.value,
            },
        )
        try:
            validated_document = self.validator.validate(document)
            self._publish_progress(
                request_id=resolved_request_id,
                operation="document.render",
                stage="validated",
                completed_steps=1,
                total_steps=2,
            )
            if allowed_file_roots is not None:
                validated_document = scope_document_paths(
                    validated_document,
                    allowed_roots=allowed_file_roots,
                    base_dir=base_dir,
                )
            resolved_file_name = file_name or self._default_file_name(validated_document.title, output_format)
            artifact = self.renderer_registry.render(
                validated_document,
                output_format=output_format,
                file_name=resolved_file_name,
            )
            self._publish_progress(
                request_id=resolved_request_id,
                operation="document.render",
                stage="rendered",
                completed_steps=2,
                total_steps=2,
                size_bytes=artifact.size_bytes,
            )
            self._publish(
                EventType.DOCUMENT_RENDER_COMPLETED,
                {
                    "request_id": resolved_request_id,
                    "title": validated_document.title,
                    "output_format": output_format.value,
                    "file_name": artifact.file_name,
                    "size_bytes": artifact.size_bytes,
                },
            )
            return artifact
        except Exception as exc:
            self._publish(
                EventType.DOCUMENT_RENDER_FAILED,
                {
                    "request_id": resolved_request_id,
                    "title": document.title,
                    "output_format": output_format.value,
                    "error": str(exc),
                },
            )
            raise

    def _ensure_enabled(self) -> None:
        if self.enabled:
            return
        raise PackUnavailableError(
            "Document generation is unavailable because the `documents` feature pack is disabled.",
            details={"pack": "documents", "reason": self.disabled_reason},
        )

    def _default_file_name(self, title: str, output_format: DocumentOutputFormat) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-") or "document"
        return f"{slug}{output_format.default_extension}"

    def _publish(self, event_type: EventType, payload: dict[str, object]) -> None:
        if self.event_bus is None:
            return
        self.event_bus.publish_threadsafe(
            StreamEvent(type=event_type, scope=EventScope.REQUEST, payload=payload),
        )

    def _publish_progress(
        self,
        *,
        request_id: str,
        operation: str,
        stage: str,
        completed_steps: int,
        total_steps: int,
        **payload: object,
    ) -> None:
        progress = round(completed_steps / total_steps, 4) if total_steps else 0.0
        self._publish(
            EventType.OPERATION_PROGRESS,
            {
                "request_id": request_id,
                "operation": operation,
                "stage": stage,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "progress": progress,
                **payload,
            },
        )
