"""Event schemas used by LewLM services."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from lewlm.core.contracts import utc_now


class EventScope(str, Enum):
    SYSTEM = "system"
    REQUEST = "request"
    JOB = "job"


class EventType(str, Enum):
    SYSTEM_READY = "system.ready"
    OPERATION_PROGRESS = "operation.progress"
    REQUEST_ACCEPTED = "request.accepted"
    REQUEST_QUEUED = "request.queued"
    REQUEST_FAILED = "request.failed"
    REQUEST_COMPLETED = "request.completed"
    PREFILL_STARTED = "prefill.started"
    MODEL_SCAN_STARTED = "model.scan.started"
    MODEL_SCAN_COMPLETED = "model.scan.completed"
    MODEL_SCAN_FAILED = "model.scan.failed"
    MODEL_LOADING = "model.loading"
    MODEL_LOADED = "model.loaded"
    AUDIO_CHUNK = "audio.chunk"
    AUDIO_TRANSCRIPTION_STARTED = "audio.transcription.started"
    AUDIO_TRANSCRIPTION_COMPLETED = "audio.transcription.completed"
    AUDIO_TRANSCRIPTION_FAILED = "audio.transcription.failed"
    AUDIO_SPEECH_STARTED = "audio.speech.started"
    AUDIO_SPEECH_COMPLETED = "audio.speech.completed"
    AUDIO_SPEECH_FAILED = "audio.speech.failed"
    DOCUMENT_PARSE_STARTED = "document.parse.started"
    DOCUMENT_PARSE_COMPLETED = "document.parse.completed"
    DOCUMENT_PARSE_FAILED = "document.parse.failed"
    DOCUMENT_RENDER_STARTED = "document.render.started"
    DOCUMENT_RENDER_COMPLETED = "document.render.completed"
    DOCUMENT_RENDER_FAILED = "document.render.failed"
    DOCUMENT_TRANSFORM_STARTED = "document.transform.started"
    DOCUMENT_TRANSFORM_COMPLETED = "document.transform.completed"
    DOCUMENT_TRANSFORM_FAILED = "document.transform.failed"
    CLUSTER_TOKEN_ISSUED = "cluster.token.issued"
    CLUSTER_WORKER_ENROLLED = "cluster.worker.enrolled"
    CLUSTER_WORKER_HEARTBEAT = "cluster.worker.heartbeat"
    CLUSTER_PLAN_UPDATED = "cluster.plan.updated"
    CLUSTER_PIPELINE_STAGE_COMPLETED = "cluster.pipeline.stage.completed"
    CLUSTER_PIPELINE_COMPLETED = "cluster.pipeline.completed"
    CLUSTER_WORKER_RECOVERED = "cluster.worker.recovered"
    AUTOTUNE_COMPLETED = "autotune.completed"
    TOKEN_DELTA = "token.delta"
    REASONING_DELTA = "reasoning.delta"
    SPECULATION_STARTED = "speculation.started"
    SPECULATION_ACCEPTED = "speculation.accepted"
    TOOL_PENDING = "tool.pending"
    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"
    TOOL_FAILED = "tool.failed"


class StreamEvent(BaseModel):
    """An event emitted by LewLM subsystems."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    type: EventType
    scope: EventScope = EventScope.SYSTEM
    created_at: datetime = Field(default_factory=utc_now)
    payload: dict[str, object] = Field(default_factory=dict)
    request_id: str | None = None
    model_id: str | None = None
    runtime: str | None = None
    capability: str | None = None
    operation: str | None = None
    stage: str | None = None
    status: str | None = None

    @model_validator(mode="after")
    def _populate_common_fields(self) -> "StreamEvent":
        payload = dict(self.payload)

        request_id = self.request_id or _payload_string(payload, "request_id")
        model_id = self.model_id or _payload_string(payload, "model_id")
        runtime = self.runtime or _payload_string(payload, "runtime")
        capability = (
            self.capability
            or _payload_string(payload, "capability")
            or _infer_event_capability(self.type, payload)
        )
        operation = (
            self.operation
            or _payload_string(payload, "operation")
            or _infer_event_operation(self.type, payload, capability=capability)
        )
        stage = self.stage or _payload_string(payload, "stage")
        status = self.status or _payload_string(payload, "status") or _infer_event_status(self.type)

        if request_id is not None:
            payload.setdefault("request_id", request_id)
        if model_id is not None:
            payload.setdefault("model_id", model_id)
        if runtime is not None:
            payload.setdefault("runtime", runtime)
        if capability is not None:
            payload.setdefault("capability", capability)
        if operation is not None:
            payload.setdefault("operation", operation)
        if stage is not None:
            payload.setdefault("stage", stage)
        if status is not None:
            payload.setdefault("status", status)

        self.payload = payload
        self.request_id = request_id
        self.model_id = model_id
        self.runtime = runtime
        self.capability = capability
        self.operation = operation
        self.stage = stage
        self.status = status
        return self

    def to_event_stream(self) -> str:
        return f"event: {self.type.value}\ndata: {self.model_dump_json()}\n\n"


def _payload_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _infer_event_capability(event_type: EventType, payload: dict[str, object]) -> str | None:
    operation = _payload_string(payload, "operation")
    if operation is not None:
        return _capability_from_operation(operation)
    if event_type in {
        EventType.PREFILL_STARTED,
        EventType.TOKEN_DELTA,
        EventType.REASONING_DELTA,
        EventType.SPECULATION_STARTED,
        EventType.SPECULATION_ACCEPTED,
    }:
        return "chat"
    if event_type in {
        EventType.AUDIO_CHUNK,
        EventType.AUDIO_TRANSCRIPTION_STARTED,
        EventType.AUDIO_TRANSCRIPTION_COMPLETED,
        EventType.AUDIO_TRANSCRIPTION_FAILED,
    }:
        return "audio_transcription"
    if event_type in {
        EventType.AUDIO_SPEECH_STARTED,
        EventType.AUDIO_SPEECH_COMPLETED,
        EventType.AUDIO_SPEECH_FAILED,
    }:
        return "audio_speech"
    if event_type in {
        EventType.DOCUMENT_PARSE_STARTED,
        EventType.DOCUMENT_PARSE_COMPLETED,
        EventType.DOCUMENT_PARSE_FAILED,
        EventType.DOCUMENT_RENDER_STARTED,
        EventType.DOCUMENT_RENDER_COMPLETED,
        EventType.DOCUMENT_RENDER_FAILED,
        EventType.DOCUMENT_TRANSFORM_STARTED,
        EventType.DOCUMENT_TRANSFORM_COMPLETED,
        EventType.DOCUMENT_TRANSFORM_FAILED,
    }:
        return "documents"
    if event_type in {
        EventType.TOOL_PENDING,
        EventType.TOOL_STARTED,
        EventType.TOOL_FINISHED,
        EventType.TOOL_FAILED,
    }:
        return "tool"
    return None


def _infer_event_operation(
    event_type: EventType,
    payload: dict[str, object],
    *,
    capability: str | None,
) -> str | None:
    if event_type in {
        EventType.DOCUMENT_PARSE_STARTED,
        EventType.DOCUMENT_PARSE_COMPLETED,
        EventType.DOCUMENT_PARSE_FAILED,
    }:
        return "document.parse"
    if event_type in {
        EventType.DOCUMENT_RENDER_STARTED,
        EventType.DOCUMENT_RENDER_COMPLETED,
        EventType.DOCUMENT_RENDER_FAILED,
    }:
        return "document.render"
    if event_type in {
        EventType.DOCUMENT_TRANSFORM_STARTED,
        EventType.DOCUMENT_TRANSFORM_COMPLETED,
        EventType.DOCUMENT_TRANSFORM_FAILED,
    }:
        return "document.transform"
    if event_type in {
        EventType.AUDIO_CHUNK,
        EventType.AUDIO_TRANSCRIPTION_STARTED,
        EventType.AUDIO_TRANSCRIPTION_COMPLETED,
        EventType.AUDIO_TRANSCRIPTION_FAILED,
    }:
        return "audio.transcription"
    if event_type in {
        EventType.AUDIO_SPEECH_STARTED,
        EventType.AUDIO_SPEECH_COMPLETED,
        EventType.AUDIO_SPEECH_FAILED,
    }:
        return "audio.speech"
    if event_type in {
        EventType.PREFILL_STARTED,
        EventType.TOKEN_DELTA,
        EventType.REASONING_DELTA,
        EventType.SPECULATION_STARTED,
        EventType.SPECULATION_ACCEPTED,
    }:
        return "text.generation"
    if event_type in {
        EventType.TOOL_PENDING,
        EventType.TOOL_STARTED,
        EventType.TOOL_FINISHED,
        EventType.TOOL_FAILED,
    }:
        return "tool.execute"
    return _operation_from_capability(capability)


def _infer_event_status(event_type: EventType) -> str | None:
    if event_type == EventType.SYSTEM_READY:
        return "ready"
    if event_type == EventType.OPERATION_PROGRESS:
        return "in_progress"
    if event_type == EventType.REQUEST_ACCEPTED:
        return "accepted"
    if event_type == EventType.REQUEST_QUEUED:
        return "queued"
    if event_type == EventType.REQUEST_FAILED:
        return "failed"
    if event_type == EventType.REQUEST_COMPLETED:
        return "completed"
    if event_type == EventType.PREFILL_STARTED:
        return "prefill_started"
    if event_type == EventType.SPECULATION_STARTED:
        return "started"
    if event_type == EventType.SPECULATION_ACCEPTED:
        return "completed"
    if event_type in {EventType.MODEL_SCAN_STARTED, EventType.MODEL_LOADING}:
        return "started"
    if event_type in {
        EventType.MODEL_SCAN_COMPLETED,
        EventType.MODEL_LOADED,
        EventType.AUDIO_TRANSCRIPTION_COMPLETED,
        EventType.AUDIO_SPEECH_COMPLETED,
        EventType.DOCUMENT_PARSE_COMPLETED,
        EventType.DOCUMENT_RENDER_COMPLETED,
        EventType.DOCUMENT_TRANSFORM_COMPLETED,
        EventType.AUTOTUNE_COMPLETED,
        EventType.TOOL_FINISHED,
    }:
        return "completed"
    if event_type in {
        EventType.MODEL_SCAN_FAILED,
        EventType.AUDIO_TRANSCRIPTION_FAILED,
        EventType.AUDIO_SPEECH_FAILED,
        EventType.DOCUMENT_PARSE_FAILED,
        EventType.DOCUMENT_RENDER_FAILED,
        EventType.DOCUMENT_TRANSFORM_FAILED,
        EventType.TOOL_FAILED,
    }:
        return "failed"
    if event_type in {
        EventType.AUDIO_TRANSCRIPTION_STARTED,
        EventType.AUDIO_SPEECH_STARTED,
        EventType.DOCUMENT_PARSE_STARTED,
        EventType.DOCUMENT_RENDER_STARTED,
        EventType.DOCUMENT_TRANSFORM_STARTED,
        EventType.TOOL_STARTED,
        EventType.TOOL_PENDING,
    }:
        return "started"
    if event_type in {
        EventType.TOKEN_DELTA,
        EventType.REASONING_DELTA,
        EventType.SPECULATION_STARTED,
        EventType.SPECULATION_ACCEPTED,
        EventType.AUDIO_CHUNK,
    }:
        return "streaming"
    return None


def _capability_from_operation(operation: str) -> str | None:
    if operation == "text.generation":
        return "chat"
    if operation in {"embeddings", "rerank"}:
        return operation
    if operation == "audio.transcription":
        return "audio_transcription"
    if operation == "audio.speech":
        return "audio_speech"
    if operation.startswith("document."):
        return "documents"
    if operation == "tool.execute":
        return "tool"
    return None


def _operation_from_capability(capability: str | None) -> str | None:
    if capability in {"chat", "streaming"}:
        return "text.generation"
    if capability in {"embeddings", "rerank"}:
        return capability
    if capability == "audio_transcription":
        return "audio.transcription"
    if capability == "audio_speech":
        return "audio.speech"
    if capability == "documents":
        return "document.operation"
    if capability == "tool":
        return "tool.execute"
    return None
