"""In-process runtime request metrics collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Mapping

from lewlm.core.contracts import utc_now


@dataclass(slots=True)
class _ModelMetricEntry:
    model_id: str
    runtime: str
    capability_counts: dict[str, int] = field(default_factory=dict)
    request_count: int = 0
    failure_count: int = 0
    total_load_seconds: float = 0.0
    total_execution_seconds: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    throughput_sample_total: float = 0.0
    throughput_sample_count: int = 0
    last_request_at: datetime | None = None
    last_error_at: datetime | None = None


@dataclass(slots=True)
class _CapabilityMetricEntry:
    capability: str
    request_count: int = 0
    failure_count: int = 0
    total_load_seconds: float = 0.0
    total_execution_seconds: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    throughput_sample_total: float = 0.0
    throughput_sample_count: int = 0
    metric_totals: dict[str, float] = field(default_factory=dict)
    last_request_at: datetime | None = None
    last_error_at: datetime | None = None


class RuntimeMetricsRecorder:
    """Track request counts, timings, and failure rates per model/runtime."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _ModelMetricEntry] = {}
        self._capability_entries: dict[str, _CapabilityMetricEntry] = {}
        self._lock = Lock()

    def record_success(
        self,
        *,
        model_id: str,
        runtime: str,
        capability: str,
        load_seconds: float,
        execution_seconds: float,
        usage: Mapping[str, Any] | None = None,
        measurements: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            entry = self._entry_for(model_id=model_id, runtime=runtime)
            capability_entry = self._capability_entry_for(capability=capability)
            timestamp = utc_now()
            entry.request_count += 1
            entry.total_load_seconds += max(load_seconds, 0.0)
            entry.total_execution_seconds += max(execution_seconds, 0.0)
            entry.last_request_at = timestamp
            entry.capability_counts[capability] = entry.capability_counts.get(capability, 0) + 1
            prompt_tokens = _coerce_int(usage.get("prompt_tokens")) if usage is not None else 0
            completion_tokens = _coerce_int(usage.get("completion_tokens")) if usage is not None else 0
            entry.total_prompt_tokens += prompt_tokens
            entry.total_completion_tokens += completion_tokens
            capability_entry.request_count += 1
            capability_entry.total_load_seconds += max(load_seconds, 0.0)
            capability_entry.total_execution_seconds += max(execution_seconds, 0.0)
            capability_entry.last_request_at = timestamp
            capability_entry.total_prompt_tokens += prompt_tokens
            capability_entry.total_completion_tokens += completion_tokens
            if completion_tokens > 0 and execution_seconds > 0:
                entry.throughput_sample_total += completion_tokens / execution_seconds
                entry.throughput_sample_count += 1
                capability_entry.throughput_sample_total += completion_tokens / execution_seconds
                capability_entry.throughput_sample_count += 1
            _merge_measurements(capability_entry.metric_totals, measurements)

    def record_failure(
        self,
        *,
        model_id: str,
        runtime: str,
        capability: str,
        load_seconds: float,
        execution_seconds: float,
        usage: Mapping[str, Any] | None = None,
        measurements: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            entry = self._entry_for(model_id=model_id, runtime=runtime)
            capability_entry = self._capability_entry_for(capability=capability)
            timestamp = utc_now()
            entry.request_count += 1
            entry.failure_count += 1
            entry.total_load_seconds += max(load_seconds, 0.0)
            entry.total_execution_seconds += max(execution_seconds, 0.0)
            entry.last_request_at = timestamp
            entry.last_error_at = timestamp
            entry.capability_counts[capability] = entry.capability_counts.get(capability, 0) + 1
            prompt_tokens = _coerce_int(usage.get("prompt_tokens")) if usage is not None else 0
            completion_tokens = _coerce_int(usage.get("completion_tokens")) if usage is not None else 0
            entry.total_prompt_tokens += prompt_tokens
            entry.total_completion_tokens += completion_tokens
            capability_entry.request_count += 1
            capability_entry.failure_count += 1
            capability_entry.total_load_seconds += max(load_seconds, 0.0)
            capability_entry.total_execution_seconds += max(execution_seconds, 0.0)
            capability_entry.last_request_at = timestamp
            capability_entry.last_error_at = timestamp
            capability_entry.total_prompt_tokens += prompt_tokens
            capability_entry.total_completion_tokens += completion_tokens
            _merge_measurements(capability_entry.metric_totals, measurements)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            entries = sorted(self._entries.values(), key=lambda item: (item.runtime, item.model_id))
            models = [self._snapshot_entry(entry) for entry in entries]
            capability_entries = sorted(self._capability_entries.values(), key=lambda item: item.capability)
            capabilities = [self._snapshot_capability_entry(entry) for entry in capability_entries]
        total_requests = sum(item["request_count"] for item in models)
        total_failures = sum(item["failure_count"] for item in models)
        success_count = total_requests - total_failures
        total_prompt_tokens = sum(item["total_prompt_tokens"] for item in models)
        total_completion_tokens = sum(item["total_completion_tokens"] for item in models)
        total_load_seconds = sum(item["_total_load_seconds"] for item in models)
        total_execution_seconds = sum(item["_total_execution_seconds"] for item in models)
        throughput_values = [item["_average_completion_tokens_per_second"] for item in models if item["_average_completion_tokens_per_second"] is not None]
        return {
            "total_requests": total_requests,
            "success_count": success_count,
            "failure_count": total_failures,
            "success_rate": round(success_count / total_requests, 4) if total_requests else 1.0,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "average_load_seconds": round(total_load_seconds / total_requests, 4) if total_requests else None,
            "average_execution_seconds": round(total_execution_seconds / total_requests, 4) if total_requests else None,
            "average_completion_tokens_per_second": (
                round(sum(throughput_values) / len(throughput_values), 4) if throughput_values else None
            ),
            "models": [self._public_snapshot(item) for item in models],
            "capabilities": [self._public_snapshot(item) for item in capabilities],
        }

    def _entry_for(self, *, model_id: str, runtime: str) -> _ModelMetricEntry:
        key = (model_id, runtime)
        entry = self._entries.get(key)
        if entry is None:
            entry = _ModelMetricEntry(model_id=model_id, runtime=runtime)
            self._entries[key] = entry
        return entry

    def _capability_entry_for(self, *, capability: str) -> _CapabilityMetricEntry:
        entry = self._capability_entries.get(capability)
        if entry is None:
            entry = _CapabilityMetricEntry(capability=capability)
            self._capability_entries[capability] = entry
        return entry

    @staticmethod
    def _snapshot_entry(entry: _ModelMetricEntry) -> dict[str, Any]:
        success_count = entry.request_count - entry.failure_count
        average_throughput = (
            round(entry.throughput_sample_total / entry.throughput_sample_count, 4)
            if entry.throughput_sample_count
            else None
        )
        return {
            "model_id": entry.model_id,
            "runtime": entry.runtime,
            "request_count": entry.request_count,
            "success_count": success_count,
            "failure_count": entry.failure_count,
            "success_rate": round(success_count / entry.request_count, 4) if entry.request_count else 1.0,
            "capability_counts": dict(sorted(entry.capability_counts.items())),
            "last_request_at": entry.last_request_at,
            "last_error_at": entry.last_error_at,
            "total_prompt_tokens": entry.total_prompt_tokens,
            "total_completion_tokens": entry.total_completion_tokens,
            "average_load_seconds": round(entry.total_load_seconds / entry.request_count, 4) if entry.request_count else None,
            "average_execution_seconds": (
                round(entry.total_execution_seconds / entry.request_count, 4) if entry.request_count else None
            ),
            "average_completion_tokens_per_second": average_throughput,
            "_average_completion_tokens_per_second": average_throughput,
            "_total_load_seconds": entry.total_load_seconds,
            "_total_execution_seconds": entry.total_execution_seconds,
        }

    @staticmethod
    def _snapshot_capability_entry(entry: _CapabilityMetricEntry) -> dict[str, Any]:
        success_count = entry.request_count - entry.failure_count
        average_throughput = (
            round(entry.throughput_sample_total / entry.throughput_sample_count, 4)
            if entry.throughput_sample_count
            else None
        )
        metric_totals = {
            key: _normalize_numeric(value)
            for key, value in sorted(entry.metric_totals.items())
        }
        metric_averages = {
            key: _normalize_numeric(round(value / entry.request_count, 4))
            for key, value in sorted(entry.metric_totals.items())
            if entry.request_count
        }
        return {
            "capability": entry.capability,
            "request_count": entry.request_count,
            "success_count": success_count,
            "failure_count": entry.failure_count,
            "success_rate": round(success_count / entry.request_count, 4) if entry.request_count else 1.0,
            "last_request_at": entry.last_request_at,
            "last_error_at": entry.last_error_at,
            "total_prompt_tokens": entry.total_prompt_tokens,
            "total_completion_tokens": entry.total_completion_tokens,
            "average_load_seconds": round(entry.total_load_seconds / entry.request_count, 4) if entry.request_count else None,
            "average_execution_seconds": (
                round(entry.total_execution_seconds / entry.request_count, 4) if entry.request_count else None
            ),
            "average_completion_tokens_per_second": average_throughput,
            "metric_totals": metric_totals,
            "metric_averages": metric_averages,
            "_average_completion_tokens_per_second": average_throughput,
        }

    @staticmethod
    def _public_snapshot(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in item.items()
            if not key.startswith("_")
        }


def _coerce_int(value: Any) -> int:
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


def _coerce_float(value: Any) -> float | None:
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


def _merge_measurements(target: dict[str, float], measurements: Mapping[str, Any] | None) -> None:
    if measurements is None:
        return
    for key, value in measurements.items():
        numeric_value = _coerce_float(value)
        if numeric_value is None:
            continue
        target[key] = target.get(key, 0.0) + numeric_value


def _normalize_numeric(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return round(value, 4)
