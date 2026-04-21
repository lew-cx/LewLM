"""Telemetry package exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["BenchmarkResult", "CacheStats", "RuntimeStats", "TelemetryService"]

if TYPE_CHECKING:
    from lewlm.telemetry.stats import BenchmarkResult, CacheStats, RuntimeStats, TelemetryService


def __getattr__(name: str) -> Any:
    if name in __all__:
        from lewlm.telemetry.stats import BenchmarkResult, CacheStats, RuntimeStats, TelemetryService

        return {
            "BenchmarkResult": BenchmarkResult,
            "CacheStats": CacheStats,
            "RuntimeStats": RuntimeStats,
            "TelemetryService": TelemetryService,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
