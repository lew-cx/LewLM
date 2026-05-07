"""Shared helpers for LewLM's documented runtime support strategy."""

from __future__ import annotations

from lewlm.core.contracts import RuntimeAffinity

FIRST_CLASS_LOCAL_RUNTIME_AFFINITIES = frozenset(
    {
        RuntimeAffinity.MLX_TEXT.value,
        RuntimeAffinity.LLAMACPP.value,
    },
)
FIRST_CLASS_NON_APPLE_RUNTIME_AFFINITY = RuntimeAffinity.LLAMACPP.value


def runtime_affinity_value(value: object) -> str | None:
    """Normalize a runtime-affinity enum or string to its string value."""

    if isinstance(value, RuntimeAffinity):
        return value.value
    if isinstance(value, str) and value:
        return value
    return None


def is_first_class_local_runtime_affinity(value: object) -> bool:
    return runtime_affinity_value(value) in FIRST_CLASS_LOCAL_RUNTIME_AFFINITIES


def is_first_class_non_apple_runtime_affinity(value: object) -> bool:
    return runtime_affinity_value(value) == FIRST_CLASS_NON_APPLE_RUNTIME_AFFINITY


def external_adapter_demotion_reason(*, baseline_runtime_affinity: object) -> str | None:
    if not is_first_class_local_runtime_affinity(baseline_runtime_affinity):
        return None
    return "external accelerators remain a measured bridge path and do not replace the first-class local runtime default on this host"
