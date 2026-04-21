"""Helpers for frontier architecture detection and bounded-memory planning."""

from __future__ import annotations

from math import ceil
from typing import Any

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import ArchitectureSubtype, ModelManifest, ModelModality

_SSM_MARKERS = (
    "mamba",
    "gateddeltanet",
    "gated_delta_net",
    "selective_scan",
    "state_space",
)
_MOE_MARKERS = (
    "mixtral",
    "moe",
    "mixture_of_experts",
    "deepseekmoe",
    "deepseek_moe",
    "jamba",
)
_ATTENTION_KEYS = (
    "attention_config",
    "attn_config",
    "num_attention_heads",
    "num_key_value_heads",
    "rope_theta",
)
_EXPERT_COUNT_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_experts",
    "moe_num_experts",
    "expert_count",
)
_ACTIVE_EXPERT_KEYS = (
    "experts_per_token",
    "num_experts_per_tok",
    "num_experts_per_token",
    "top_k_experts",
    "moe_top_k",
)
_ROUTING_KEYS = ("routing_type", "router_type", "routing_algorithm", "router_algorithm")
_STATE_SIZE_KEYS = ("state_size", "d_state", "ssm_state_size")


def frontier_architecture_values() -> set[str]:
    return {
        ArchitectureSubtype.SSM_MAMBA.value,
        ArchitectureSubtype.HYBRID_SSM.value,
        ArchitectureSubtype.MOE.value,
        ArchitectureSubtype.HYBRID_MOE.value,
    }


def infer_architecture_subtype(*, name: str, config_data: dict[str, Any]) -> ArchitectureSubtype:
    searchable = " ".join(_identifier_values(name=name, payload=config_data))
    has_ssm = any(marker in searchable for marker in _SSM_MARKERS) or _first_int(config_data, _STATE_SIZE_KEYS) is not None
    expert_count = _first_int(config_data, _EXPERT_COUNT_KEYS)
    has_moe = expert_count is not None or any(marker in searchable for marker in _MOE_MARKERS)
    has_attention = any(_nested_value(config_data, key) is not None for key in _ATTENTION_KEYS)
    if has_moe and has_ssm:
        return ArchitectureSubtype.HYBRID_MOE
    if has_moe and has_attention:
        return ArchitectureSubtype.HYBRID_MOE
    if has_moe:
        return ArchitectureSubtype.MOE
    if has_ssm and has_attention:
        return ArchitectureSubtype.HYBRID_SSM
    if has_ssm:
        return ArchitectureSubtype.SSM_MAMBA
    if searchable:
        return ArchitectureSubtype.TRANSFORMER
    return ArchitectureSubtype.UNKNOWN


def extract_architecture_metadata(
    *,
    name: str,
    config_data: dict[str, Any],
    architecture_subtype: ArchitectureSubtype,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "frontier_architecture": architecture_subtype.value in frontier_architecture_values(),
        "cache_state_handling": _cache_state_handling(architecture_subtype),
    }
    state_size = _first_int(config_data, _STATE_SIZE_KEYS)
    if state_size is not None:
        metadata["state_size"] = state_size
    expert_count = _first_int(config_data, _EXPERT_COUNT_KEYS)
    if expert_count is not None:
        metadata["expert_count"] = expert_count
    active_expert_count = _first_int(config_data, _ACTIVE_EXPERT_KEYS)
    if active_expert_count is not None:
        metadata["active_expert_count"] = active_expert_count
    routing_type = _first_str(config_data, _ROUTING_KEYS)
    if routing_type is not None:
        metadata["expert_routing_type"] = routing_type
    elif active_expert_count is not None:
        metadata["expert_routing_type"] = f"top-{active_expert_count}"
    if architecture_subtype in {ArchitectureSubtype.MOE, ArchitectureSubtype.HYBRID_MOE} and expert_count is None:
        metadata.update(_expert_count_from_name(name))
    return {key: value for key, value in metadata.items() if value is not None}


def is_frontier_architecture(manifest: ModelManifest) -> bool:
    return manifest.architecture_subtype.value in frontier_architecture_values()


def build_frontier_serving_plan(
    *,
    manifest: ModelManifest,
    settings: LewLMSettings,
) -> dict[str, int | float | str | bool] | None:
    if ModelModality.TEXT not in manifest.modality or not is_frontier_architecture(manifest):
        return None
    metadata = manifest.metadata
    subtype = manifest.architecture_subtype
    full_memory_mb = manifest.estimated_memory_mb
    cache_state_handling = str(metadata.get("cache_state_handling") or _cache_state_handling(subtype))
    plan: dict[str, int | float | str | bool] = {
        "architecture_subtype": subtype.value,
        "cache_state_handling": cache_state_handling,
        "planning_only": True,
    }
    if full_memory_mb is not None:
        plan["full_estimated_memory_mb"] = full_memory_mb
    if subtype in {ArchitectureSubtype.SSM_MAMBA, ArchitectureSubtype.HYBRID_SSM}:
        state_size = _coerce_int(metadata.get("state_size"))
        if state_size is not None:
            plan["state_size"] = state_size
            plan["estimated_state_cache_kb_per_token"] = round((state_size * 2) / 1024, 4)
        plan["bounded_memory_mode"] = "state_cache_planning"
        plan["quality_tradeoff_note"] = (
            "State-cache handling is detected and reported, but LewLM still depends on the concrete backend for real selective-scan execution."
        )
        plan["performance_tradeoff_note"] = (
            "Hybrid SSM models may trade lower KV-style cache growth for backend-specific scan overhead and compatibility constraints."
        )
        return plan
    expert_count = _coerce_int(metadata.get("expert_count"))
    active_expert_count = max(_coerce_int(metadata.get("active_expert_count")) or 2, 1)
    bounded_mode = settings.moe_bounded_memory_mode
    plan["bounded_memory_mode"] = bounded_mode
    if expert_count is not None:
        plan["expert_count"] = expert_count
        plan["active_expert_count"] = active_expert_count
    if expert_count is None or full_memory_mb is None:
        plan["quality_tradeoff_note"] = (
            "MoE routing metadata was detected, but LewLM does not yet have enough bundle metadata to estimate a bounded-memory serving envelope."
        )
        plan["performance_tradeoff_note"] = (
            "Benchmark artifacts can still record expert counts and routing hints even when memory estimates are incomplete."
        )
        return plan
    resident_expert_count = (
        expert_count
        if bounded_mode == "off"
        else min(max(settings.moe_resident_expert_count, active_expert_count), expert_count)
    )
    expert_memory_fraction = 0.65 if subtype == ArchitectureSubtype.MOE else 0.55
    dense_memory_mb = ceil(full_memory_mb * (1.0 - expert_memory_fraction))
    total_expert_memory_mb = max(full_memory_mb - dense_memory_mb, 0)
    per_expert_memory_mb = max(1, ceil(total_expert_memory_mb / max(expert_count, 1)))
    planned_memory_mb = (
        full_memory_mb
        if bounded_mode == "off"
        else dense_memory_mb + per_expert_memory_mb * resident_expert_count
    )
    memory_savings_mb = max(full_memory_mb - planned_memory_mb, 0)
    plan.update(
        {
            "resident_expert_count": resident_expert_count,
            "streamed_expert_count": max(expert_count - resident_expert_count, 0),
            "planned_memory_mb": planned_memory_mb,
            "memory_savings_mb": memory_savings_mb,
            "estimated_swap_mb_per_request": per_expert_memory_mb * active_expert_count if bounded_mode != "off" else 0,
            "estimated_expert_memory_mb": total_expert_memory_mb,
            "estimated_dense_memory_mb": dense_memory_mb,
        },
    )
    if bounded_mode == "off":
        plan["quality_tradeoff_note"] = (
            "All experts are assumed resident; this preserves baseline quality while forgoing bounded-memory savings."
        )
        plan["performance_tradeoff_note"] = (
            "Full-resident MoE serving avoids swap overhead but keeps peak memory at the full-model estimate."
        )
    elif bounded_mode == "partial_load":
        plan["quality_tradeoff_note"] = (
            "Partial expert residency preserves the same routed experts per token but may introduce expert miss latency when non-resident experts are needed."
        )
        plan["performance_tradeoff_note"] = (
            "Partial-load mode lowers planned peak memory at the cost of extra expert materialization work on cache misses."
        )
    else:
        plan["quality_tradeoff_note"] = (
            "Expert-streaming mode keeps only a bounded expert window resident and assumes SSD-backed reload for the remainder."
        )
        plan["performance_tradeoff_note"] = (
            "Expert-streaming mode maximizes memory savings but typically pays the highest per-request swap and warmup penalty."
        )
    return plan


def frontier_plan_summary(plan: dict[str, int | float | str | bool] | None) -> str | None:
    if not plan:
        return None
    subtype = str(plan.get("architecture_subtype") or ArchitectureSubtype.UNKNOWN.value)
    expert_count = _coerce_int(plan.get("expert_count"))
    if expert_count is not None:
        bounded_mode = str(plan.get("bounded_memory_mode") or "off")
        if bounded_mode != "off":
            return (
                f"frontier `{subtype}` plan {bounded_mode} "
                f"{plan.get('planned_memory_mb', plan.get('full_estimated_memory_mb', 'unknown'))}/"
                f"{plan.get('full_estimated_memory_mb', 'unknown')} MB with "
                f"{plan.get('resident_expert_count', 'unknown')}/{expert_count} resident experts"
            )
        return f"frontier `{subtype}` profile detected with {expert_count} experts"
    return f"frontier `{subtype}` cache handling `{plan.get('cache_state_handling', 'unknown')}`"


def frontier_plan_notes(plan: dict[str, int | float | str | bool] | None) -> list[str]:
    if not plan:
        return []
    notes = [
        "LewLM currently treats this frontier architecture plan as explicit metadata and bounded-memory planning rather than proof of backend-native execution.",
    ]
    quality = plan.get("quality_tradeoff_note")
    if isinstance(quality, str) and quality:
        notes.append(quality)
    performance = plan.get("performance_tradeoff_note")
    if isinstance(performance, str) and performance:
        notes.append(performance)
    return notes


def frontier_architecture_measurements(request_metadata: dict[str, Any]) -> dict[str, int | float | str | bool]:
    plan = request_metadata.get("frontier_architecture")
    if not isinstance(plan, dict):
        return {}
    subtype = str(plan.get("architecture_subtype") or ArchitectureSubtype.UNKNOWN.value)
    is_ssm = subtype in {ArchitectureSubtype.SSM_MAMBA.value, ArchitectureSubtype.HYBRID_SSM.value}
    is_moe = subtype in {ArchitectureSubtype.MOE.value, ArchitectureSubtype.HYBRID_MOE.value}
    bounded_mode = str(plan.get("bounded_memory_mode") or "off")
    values: dict[str, int | float | str | bool | None] = {
        "frontier_architecture_detected": subtype != ArchitectureSubtype.UNKNOWN.value,
        "frontier_ssm_requests": 1 if is_ssm else 0,
        "frontier_moe_requests": 1 if is_moe else 0,
        "frontier_bounded_memory_requests": 1 if is_moe and bounded_mode != "off" else 0,
        "frontier_full_estimated_memory_mb": _coerce_int(plan.get("full_estimated_memory_mb")),
        "frontier_planned_memory_mb": _coerce_int(plan.get("planned_memory_mb")),
        "frontier_memory_savings_mb": _coerce_int(plan.get("memory_savings_mb")),
        "frontier_expert_count": _coerce_int(plan.get("expert_count")),
        "frontier_resident_expert_count": _coerce_int(plan.get("resident_expert_count")),
        "frontier_streamed_expert_count": _coerce_int(plan.get("streamed_expert_count")),
        "frontier_requested_expert_count": _coerce_int(plan.get("requested_expert_count")),
        "frontier_expert_swap_count": _coerce_int(plan.get("expert_swap_count")),
        "frontier_expert_swap_mb": _coerce_int(plan.get("expert_swap_mb")),
        "frontier_estimated_swap_mb_per_request": _coerce_int(plan.get("estimated_swap_mb_per_request")),
        "frontier_state_size": _coerce_int(plan.get("state_size")),
        "frontier_state_cache_requests": _coerce_int(plan.get("state_cache_requests")),
        "frontier_state_cache_hits": _coerce_int(plan.get("state_cache_hits")),
        "frontier_state_cache_misses": _coerce_int(plan.get("state_cache_misses")),
        "frontier_state_cache_entry_count": _coerce_int(plan.get("state_cache_entry_count")),
        "frontier_state_cache_bytes": _coerce_int(plan.get("state_cache_bytes")),
        "frontier_effective_loaded_memory_mb": _coerce_int(plan.get("effective_loaded_memory_mb")),
        "frontier_planning_only": bool(plan.get("planning_only", True)),
    }
    return {
        key: value
        for key, value in values.items()
        if value is not None and not (isinstance(value, str) and value == "")
    }


def _cache_state_handling(architecture_subtype: ArchitectureSubtype) -> str:
    if architecture_subtype == ArchitectureSubtype.SSM_MAMBA:
        return "selective_scan_state"
    if architecture_subtype == ArchitectureSubtype.HYBRID_SSM:
        return "hybrid_attention_state"
    if architecture_subtype in {ArchitectureSubtype.MOE, ArchitectureSubtype.HYBRID_MOE}:
        return "expert_resident_window"
    return "standard_transformer_kv"


def _identifier_values(*, name: str, payload: Any) -> list[str]:
    identifiers = [name.casefold()]
    if isinstance(payload, dict):
        for key, value in payload.items():
            identifiers.append(str(key).casefold())
            identifiers.extend(_identifier_values(name=str(value), payload=value))
        return identifiers
    if isinstance(payload, list):
        for item in payload:
            identifiers.extend(_identifier_values(name=str(item), payload=item))
        return identifiers
    identifiers.append(str(payload).casefold())
    return identifiers


def _expert_count_from_name(name: str) -> dict[str, int] | dict[str, str]:
    lowered = name.casefold()
    if "8x7b" in lowered or "8x22b" in lowered:
        return {"expert_count": 8}
    return {}


def _nested_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            nested = _nested_value(value, key)
            if nested is not None:
                return nested
    if isinstance(payload, list):
        for item in payload:
            nested = _nested_value(item, key)
            if nested is not None:
                return nested
    return None


def _first_int(payload: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        coerced = _coerce_int(_nested_value(payload, key))
        if coerced is not None:
            return coerced
    return None


def _first_str(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _nested_value(payload, key)
        if isinstance(value, str) and value:
            return value
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
