"""Pure helper utilities for distributed experimental runtime flows."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any


def parse_timestamp(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")


def coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def coerce_fraction(value: Any, *, default: float) -> float:
    coerced = coerce_float(value)
    if coerced is None:
        return default
    return min(max(coerced, 0.0), 0.9)


def selected_worker_profiles(
    *,
    workers: Sequence[object],
    required_workers: int,
    pipeline: Mapping[str, Any],
    worker_profile_factory,
) -> list[object]:
    ranked = sorted(
        (worker_profile_factory(worker=worker, pipeline=pipeline) for worker in workers),
        key=lambda profile: (-profile.selection_score, profile.network_latency_ms, profile.worker_name),
    )
    selected = ranked[:required_workers]
    return sorted(
        selected,
        key=lambda profile: (profile.network_latency_ms / max(profile.relative_weight, 0.1), -profile.relative_weight),
    )


def worker_profile(
    *,
    worker: object,
    pipeline: Mapping[str, Any],
    profile_type,
) -> object:
    metadata = worker.metadata if isinstance(worker.metadata, dict) else {}
    relative_weight = coerce_float(metadata.get("relative_weight")) or coerce_float(metadata.get("compute_weight")) or 1.0
    relative_weight = max(relative_weight, 0.1)
    network_latency_ms = coerce_float(metadata.get("network_latency_ms")) or coerce_float(pipeline.get("network_latency_ms")) or 3.0
    network_bandwidth_gbps = coerce_float(metadata.get("network_bandwidth_gbps")) or coerce_float(
        pipeline.get("network_bandwidth_gbps"),
    ) or 10.0
    max_batch_tokens = max(
        64,
        coerce_int(metadata.get("max_batch_tokens"))
        or coerce_int(metadata.get("batch_tokens"))
        or coerce_int(pipeline.get("batch_tokens"))
        or 256,
    )
    prefetch_tokens = max(
        0,
        coerce_int(metadata.get("prefetch_tokens"))
        or coerce_int(pipeline.get("prefetch_tokens"))
        or round(max_batch_tokens * (coerce_fraction(pipeline.get("prefetch_ratio"), default=0.25))),
    )
    overlap_ratio = coerce_fraction(
        metadata.get("overlap_ratio", pipeline.get("overlap_ratio")),
        default=0.2,
    )
    selection_score = round(relative_weight / (1.0 + (network_latency_ms / 8.0)), 4)
    return profile_type(
        worker_id=worker.worker_id,
        worker_name=worker.worker_name,
        endpoint=worker.endpoint,
        relative_weight=round(relative_weight, 4),
        selection_score=selection_score,
        network_latency_ms=round(network_latency_ms, 4),
        network_bandwidth_gbps=round(max(network_bandwidth_gbps, 0.1), 4),
        max_batch_tokens=max_batch_tokens,
        prefetch_tokens=min(prefetch_tokens, max_batch_tokens),
        overlap_ratio=round(overlap_ratio, 4),
    )


def stage_assignment_profile(
    *,
    layer_span: int,
    profile: object,
    pipeline: Mapping[str, Any],
    heterogeneity_ratio: float,
) -> dict[str, int | float]:
    base_batch_tokens = max(64, coerce_int(pipeline.get("batch_tokens")) or 256)
    average_weight = max(coerce_float(pipeline.get("average_weight_hint")) or 1.0, 0.1)
    relative_batch_multiplier = max(profile.relative_weight / average_weight, 0.75)
    target_batch_tokens = min(
        profile.max_batch_tokens,
        max(64, round(base_batch_tokens * min(relative_batch_multiplier, 1.5))),
    )
    prefetch_tokens = min(
        profile.max_batch_tokens,
        max(profile.prefetch_tokens, round(target_batch_tokens * coerce_fraction(pipeline.get("prefetch_ratio"), default=0.25))),
    )
    base_compute_ms_per_layer = coerce_float(pipeline.get("base_compute_ms_per_layer")) or 1.35
    batch_pressure = max(target_batch_tokens / base_batch_tokens, 0.75)
    compute_seconds = round((layer_span * base_compute_ms_per_layer * batch_pressure) / (1000.0 * profile.relative_weight), 4)
    network_seconds = round(
        (profile.network_latency_ms / 1000.0)
        + (prefetch_tokens / max(profile.network_bandwidth_gbps * 32000.0, 1.0)),
        4,
    )
    queue_seconds = round(
        (coerce_float(pipeline.get("queue_delay_ms")) or 0.8) / 1000.0
        + max(0.0, heterogeneity_ratio - profile.relative_weight) * 0.0015,
        4,
    )
    overlap_credit_seconds = round(
        min(compute_seconds, network_seconds + queue_seconds) * profile.overlap_ratio,
        4,
    )
    expected_stage_seconds = round(
        max(compute_seconds + network_seconds + queue_seconds - overlap_credit_seconds, 0.001),
        4,
    )
    expected_utilization = round(min(target_batch_tokens / profile.max_batch_tokens, 1.0), 4)
    return {
        "target_batch_tokens": target_batch_tokens,
        "prefetch_tokens": prefetch_tokens,
        "network_latency_ms": profile.network_latency_ms,
        "network_bandwidth_gbps": profile.network_bandwidth_gbps,
        "overlap_ratio": profile.overlap_ratio,
        "expected_compute_seconds": compute_seconds,
        "expected_network_seconds": network_seconds,
        "expected_queue_seconds": queue_seconds,
        "overlap_credit_seconds": overlap_credit_seconds,
        "expected_stage_seconds": expected_stage_seconds,
        "expected_utilization": expected_utilization,
    }


def stage_execution_profile(stage: object) -> dict[str, int | float]:
    return {
        "relative_weight": stage.relative_weight,
        "selection_score": stage.selection_score,
        "target_batch_tokens": stage.target_batch_tokens,
        "prefetch_tokens": stage.prefetch_tokens,
        "network_latency_ms": stage.network_latency_ms,
        "network_bandwidth_gbps": stage.network_bandwidth_gbps,
        "overlap_ratio": stage.overlap_ratio,
        "expected_compute_seconds": stage.expected_compute_seconds,
        "expected_network_seconds": stage.expected_network_seconds,
        "expected_queue_seconds": stage.expected_queue_seconds,
        "overlap_credit_seconds": stage.overlap_credit_seconds,
        "expected_stage_seconds": stage.expected_stage_seconds,
        "expected_utilization": stage.expected_utilization,
    }


def aggregate_execution_metrics(
    *,
    stage_metrics: Sequence[Mapping[str, Any]],
    assignments: Sequence[object],
    completion_tokens: int,
    distributed_boundary_note: str,
) -> dict[str, int | float | str | bool]:
    total_compute_seconds = round(sum(coerce_float(item.get("compute_seconds")) or 0.0 for item in stage_metrics), 4)
    total_network_seconds = round(sum(coerce_float(item.get("network_seconds")) or 0.0 for item in stage_metrics), 4)
    total_scheduling_seconds = round(sum(coerce_float(item.get("scheduling_seconds")) or 0.0 for item in stage_metrics), 4)
    total_stage_elapsed_seconds = round(sum(coerce_float(item.get("stage_elapsed_seconds")) or 0.0 for item in stage_metrics), 4)
    average_stage_elapsed_seconds = round(total_stage_elapsed_seconds / len(stage_metrics), 4) if stage_metrics else 0.0
    stage_utilizations = [coerce_float(item.get("utilization")) for item in stage_metrics]
    average_stage_utilization = round(
        sum(value for value in stage_utilizations if value is not None) / len(stage_metrics),
        4,
    ) if stage_metrics else 0.0
    pipeline_overlap_credit_seconds = 0.0
    for current, nxt in zip(stage_metrics, stage_metrics[1:], strict=False):
        current_stage = coerce_float(current.get("expected_stage_seconds")) or coerce_float(current.get("stage_elapsed_seconds")) or 0.0
        next_stage = coerce_float(nxt.get("expected_stage_seconds")) or coerce_float(nxt.get("stage_elapsed_seconds")) or 0.0
        overlap_ratio = min(
            coerce_fraction(current.get("overlap_ratio"), default=0.0),
            coerce_fraction(nxt.get("overlap_ratio"), default=0.0),
        )
        pipeline_overlap_credit_seconds += min(current_stage, next_stage) * overlap_ratio * 0.5
    pipeline_overlap_credit_seconds = round(pipeline_overlap_credit_seconds, 4)
    critical_path_seconds = round(
        max(
            max((coerce_float(item.get("stage_elapsed_seconds")) or 0.0 for item in stage_metrics), default=0.0),
            total_stage_elapsed_seconds - pipeline_overlap_credit_seconds,
        ),
        4,
    )
    total_serial_seconds = total_compute_seconds + total_network_seconds + total_scheduling_seconds
    throughput_tokens_per_second = round(completion_tokens / critical_path_seconds, 4) if completion_tokens > 0 and critical_path_seconds > 0 else 0.0
    completion_tokens_per_second = (
        round(completion_tokens / total_stage_elapsed_seconds, 4)
        if completion_tokens > 0 and total_stage_elapsed_seconds > 0
        else 0.0
    )
    compute_units = sum(
        (coerce_float(item.get("compute_seconds")) or 0.0) * max(coerce_float(item.get("relative_weight")) or 1.0, 0.1)
        for item in stage_metrics
    )
    strongest_weight = max((coerce_float(item.get("relative_weight")) or 1.0 for item in stage_metrics), default=1.0)
    estimated_single_host_seconds = round(compute_units / max(strongest_weight, 0.1), 4)
    speedup_vs_single_host_percent = (
        round(((estimated_single_host_seconds - critical_path_seconds) / estimated_single_host_seconds) * 100.0, 4)
        if estimated_single_host_seconds > 0
        else 0.0
    )
    denominator = total_serial_seconds or 1.0
    compute_share_percent = round((total_compute_seconds / denominator) * 100.0, 4)
    network_share_percent = round((total_network_seconds / denominator) * 100.0, 4)
    scheduling_share_percent = round((total_scheduling_seconds / denominator) * 100.0, 4)
    heterogeneity_ratio = round(
        max((assignment.relative_weight for assignment in assignments), default=1.0)
        / max(min((assignment.relative_weight for assignment in assignments), default=1.0), 0.1),
        4,
    )
    average_network_latency_ms = round(
        sum(assignment.network_latency_ms for assignment in assignments) / len(assignments),
        4,
    ) if assignments else 0.0
    effective_batch_tokens = round(
        sum(assignment.target_batch_tokens for assignment in assignments) / len(assignments),
    ) if assignments else 0
    average_prefetch_tokens = round(
        sum(assignment.prefetch_tokens for assignment in assignments) / len(assignments),
    ) if assignments else 0
    pipeline_overlap_efficiency_percent = (
        round((pipeline_overlap_credit_seconds / total_stage_elapsed_seconds) * 100.0, 4)
        if total_stage_elapsed_seconds > 0
        else 0.0
    )
    bottleneck = dominant_bottleneck(
        compute_seconds=total_compute_seconds,
        network_seconds=total_network_seconds,
        scheduling_seconds=total_scheduling_seconds,
    )
    notes = [distributed_boundary_note]
    if bottleneck == "network":
        notes.append("Network transfer or RTT dominates the current distributed run; add lower-latency links or smaller stage payloads.")
    elif bottleneck == "scheduling":
        notes.append("Queueing and stage imbalance dominate the current distributed run; rebalance worker weights or reduce micro-batch pressure.")
    elif bottleneck == "model_execution":
        notes.append("Model-execution time dominates; the current host mix is compute-bound rather than transport-bound.")
    if speedup_vs_single_host_percent > 0:
        notes.append("The weighted multi-host critical path beats the strongest single-host estimate for this workload.")
    elif speedup_vs_single_host_percent < 0:
        notes.append("Distributed overhead outweighs weighted multi-host compute gain for this workload.")
    return {
        "pipeline_latency_seconds": total_stage_elapsed_seconds,
        "critical_path_seconds": critical_path_seconds,
        "average_stage_elapsed_seconds": average_stage_elapsed_seconds,
        "average_stage_utilization": average_stage_utilization,
        "throughput_tokens_per_second": throughput_tokens_per_second,
        "completion_tokens_per_second": completion_tokens_per_second,
        "estimated_single_host_seconds": estimated_single_host_seconds,
        "speedup_vs_single_host_percent": speedup_vs_single_host_percent,
        "compute_share_percent": compute_share_percent,
        "network_share_percent": network_share_percent,
        "scheduling_share_percent": scheduling_share_percent,
        "pipeline_overlap_efficiency_percent": pipeline_overlap_efficiency_percent,
        "heterogeneity_ratio": heterogeneity_ratio,
        "effective_batch_tokens": effective_batch_tokens,
        "average_prefetch_tokens": average_prefetch_tokens,
        "average_network_latency_ms": average_network_latency_ms,
        "prefetch_enabled": average_prefetch_tokens > 0,
        "overlap_enabled": pipeline_overlap_efficiency_percent > 0,
        "bottleneck": bottleneck,
        "notes": notes,
    }


def dominant_bottleneck(*, compute_seconds: float, network_seconds: float, scheduling_seconds: float) -> str:
    totals = {
        "model_execution": compute_seconds,
        "network": network_seconds,
        "scheduling": scheduling_seconds,
    }
    total = sum(totals.values())
    if total <= 0:
        return "balanced"
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) <= 0.002:
        return "balanced"
    if (ranked[0][1] / total) < 0.45:
        return "balanced"
    return ranked[0][0]


def weighted_layer_spans(*, total_layers: int, weights: Sequence[float]) -> list[int]:
    if not weights:
        return []
    normalized_weights = [max(weight, 0.1) for weight in weights]
    total_weight = sum(normalized_weights)
    base_allocations = [max(1, int(total_layers * (weight / total_weight))) for weight in normalized_weights]
    while sum(base_allocations) > total_layers:
        index = max(range(len(base_allocations)), key=lambda item: base_allocations[item])
        if base_allocations[index] <= 1:
            break
        base_allocations[index] -= 1
    fractions = [
        (index, (total_layers * (weight / total_weight)) - int(total_layers * (weight / total_weight)))
        for index, weight in enumerate(normalized_weights)
    ]
    for index, _ in sorted(fractions, key=lambda item: item[1], reverse=True):
        if sum(base_allocations) >= total_layers:
            break
        base_allocations[index] += 1
    if sum(base_allocations) < total_layers:
        base_allocations[-1] += total_layers - sum(base_allocations)
    return base_allocations
