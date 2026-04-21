"""Shared execution metadata surfaced on public execution APIs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from lewlm.core.contracts import RequestModality, RoutingDecision, RoutingModalityPath, RuntimeAffinity


def milliseconds_from_seconds(value: float) -> int:
    """Convert a timing value in seconds to a rounded non-negative millisecond count."""

    return max(int(round(max(value, 0.0) * 1000)), 0)


class ExecutionModelMetadata(BaseModel):
    requested_model_id: str | None = None
    resolved_model_id: str | None = None
    runtime_name: str | None = None
    runtime_affinity: RuntimeAffinity | None = None


class ExecutionRoutingMetadata(BaseModel):
    kind: Literal["model_router", "tool_execution"]
    reason: str | None = None
    request_modality: RequestModality | None = None
    modality_path: RoutingModalityPath | None = None
    modality_path_reason: str | None = None
    alternatives: list[str] = Field(default_factory=list)


class ExecutionTimingMetadata(BaseModel):
    queue_milliseconds: int = 0
    load_milliseconds: int = 0
    execute_milliseconds: int = 0
    total_milliseconds: int = 0


class ExecutionServingMetadata(BaseModel):
    capability: str | None = None
    phase: str | None = None
    streaming: bool = False
    streaming_owner: str | None = None
    runtime_adapter_kind: str | None = None
    cancellation_requested: bool = False
    queue_residency_milliseconds: int = 0
    queue_count: int = 0
    batched: bool = False
    batch_size: int = 1


class ExecutionMetadata(BaseModel):
    version: Literal["v1"] = "v1"
    request_id: str
    created: int
    result_origin: Literal["runtime", "cache_hit", "coalesced", "tool_execution", "idempotent_replay"] = "runtime"
    model: ExecutionModelMetadata = Field(default_factory=ExecutionModelMetadata)
    routing: ExecutionRoutingMetadata
    timing: ExecutionTimingMetadata = Field(default_factory=ExecutionTimingMetadata)
    serving: ExecutionServingMetadata | None = None
    idempotency_key: str | None = None
    idempotent_replay: bool = False


def build_routed_execution_metadata(
    *,
    request_id: str,
    created: int,
    requested_model_id: str | None,
    routing: RoutingDecision,
    queue_milliseconds: int = 0,
    load_milliseconds: int = 0,
    execute_milliseconds: int = 0,
    serving: ExecutionServingMetadata | None = None,
    result_origin: Literal["runtime", "cache_hit", "coalesced"] = "runtime",
) -> ExecutionMetadata:
    total_milliseconds = max(queue_milliseconds + load_milliseconds + execute_milliseconds, 0)
    return ExecutionMetadata(
        request_id=request_id,
        created=created,
        result_origin=result_origin,
        model=ExecutionModelMetadata(
            requested_model_id=requested_model_id,
            resolved_model_id=routing.model_id,
            runtime_name=routing.runtime_name,
            runtime_affinity=routing.runtime_affinity,
        ),
        routing=ExecutionRoutingMetadata(
            kind="model_router",
            reason=routing.reason,
            request_modality=routing.request_modality,
            modality_path=routing.modality_path,
            modality_path_reason=routing.modality_path_reason,
            alternatives=list(routing.alternatives),
        ),
        timing=ExecutionTimingMetadata(
            queue_milliseconds=max(queue_milliseconds, 0),
            load_milliseconds=max(load_milliseconds, 0),
            execute_milliseconds=max(execute_milliseconds, 0),
            total_milliseconds=total_milliseconds,
        ),
        serving=serving,
    )


def build_tool_execution_metadata(
    *,
    request_id: str,
    created: int,
    tool_name: str,
    duration_milliseconds: int,
    idempotency_key: str | None = None,
    idempotent_replay: bool = False,
    runtime_name: str = "tool_execution_service",
) -> ExecutionMetadata:
    result_origin: Literal["tool_execution", "idempotent_replay"] = (
        "idempotent_replay" if idempotent_replay else "tool_execution"
    )
    duration_milliseconds = max(duration_milliseconds, 0)
    return ExecutionMetadata(
        request_id=request_id,
        created=created,
        result_origin=result_origin,
        model=ExecutionModelMetadata(runtime_name=runtime_name),
        routing=ExecutionRoutingMetadata(kind="tool_execution", reason=tool_name),
        timing=ExecutionTimingMetadata(
            queue_milliseconds=0,
            load_milliseconds=0,
            execute_milliseconds=duration_milliseconds,
            total_milliseconds=duration_milliseconds,
        ),
        idempotency_key=idempotency_key,
        idempotent_replay=idempotent_replay,
    )
