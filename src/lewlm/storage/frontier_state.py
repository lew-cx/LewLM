"""In-memory execution-state tracking for frontier text architectures."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
from threading import Lock
from typing import Any

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import ArchitectureSubtype, GenerateRequest, ModelManifest
from lewlm.runtime.experimental.architectures import build_frontier_serving_plan


@dataclass(slots=True)
class _MoEResidencyWindow:
    resident_limit: int
    resident_experts: deque[int] = field(default_factory=deque)

    def access(self, expert_ids: list[int]) -> int:
        if self.resident_limit <= 0:
            return len(expert_ids)
        swap_count = 0
        resident_set = set(self.resident_experts)
        for expert_id in expert_ids:
            if expert_id in resident_set:
                self.resident_experts.remove(expert_id)
                self.resident_experts.append(expert_id)
                continue
            swap_count += 1
            if len(self.resident_experts) >= self.resident_limit:
                self.resident_experts.popleft()
            self.resident_experts.append(expert_id)
            resident_set = set(self.resident_experts)
        return swap_count


class FrontierExecutionTracker:
    """Track realized request-time execution state for hybrid SSM and MoE models."""

    def __init__(self, *, settings: LewLMSettings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._registered_plans: dict[str, dict[str, int | float | str | bool]] = {}
        self._effective_loaded_memory_overrides: dict[str, int] = {}
        self._ssm_prompt_keys: dict[str, set[str]] = {}
        self._ssm_peak_state_cache_bytes: dict[str, int] = {}
        self._ssm_request_count = 0
        self._ssm_state_cache_hits = 0
        self._ssm_state_cache_misses = 0
        self._moe_windows: dict[str, _MoEResidencyWindow] = {}
        self._moe_request_count = 0
        self._moe_swap_count = 0
        self._moe_swap_mb = 0
        self._moe_peak_resident_expert_count = 0

    def register_manifest(self, manifest: ModelManifest) -> None:
        plan = build_frontier_serving_plan(manifest=manifest, settings=self.settings)
        with self._lock:
            if plan is None:
                self._registered_plans.pop(manifest.model_id, None)
                self._effective_loaded_memory_overrides.pop(manifest.model_id, None)
                self._ssm_prompt_keys.pop(manifest.model_id, None)
                self._ssm_peak_state_cache_bytes.pop(manifest.model_id, None)
                self._moe_windows.pop(manifest.model_id, None)
                return
            self._registered_plans[manifest.model_id] = dict(plan)
            planned_memory_mb = _coerce_int(plan.get("planned_memory_mb"))
            full_memory_mb = manifest.estimated_memory_mb
            if planned_memory_mb is not None and full_memory_mb is not None and planned_memory_mb < full_memory_mb:
                self._effective_loaded_memory_overrides[manifest.model_id] = planned_memory_mb

    def unregister_model(self, model_id: str) -> None:
        with self._lock:
            self._registered_plans.pop(model_id, None)
            self._effective_loaded_memory_overrides.pop(model_id, None)
            self._ssm_prompt_keys.pop(model_id, None)
            self._ssm_peak_state_cache_bytes.pop(model_id, None)
            self._moe_windows.pop(model_id, None)

    def loaded_memory_override(self, model_id: str) -> int | None:
        with self._lock:
            return self._effective_loaded_memory_overrides.get(model_id)

    def annotate_request(self, *, manifest: ModelManifest, request: GenerateRequest) -> dict[str, int | float | str | bool] | None:
        existing = request.metadata.get("frontier_architecture")
        if isinstance(existing, dict):
            base_plan = dict(existing)
        else:
            base_plan = build_frontier_serving_plan(manifest=manifest, settings=self.settings)
            if base_plan is not None:
                request.metadata["frontier_architecture"] = dict(base_plan)
        if base_plan is None:
            return None
        subtype = str(base_plan.get("architecture_subtype") or manifest.architecture_subtype.value)
        with self._lock:
            if subtype in {ArchitectureSubtype.SSM_MAMBA.value, ArchitectureSubtype.HYBRID_SSM.value}:
                realized = self._annotate_ssm_request(model_id=manifest.model_id, request=request, plan=base_plan)
            elif subtype in {ArchitectureSubtype.MOE.value, ArchitectureSubtype.HYBRID_MOE.value}:
                realized = self._annotate_moe_request(model_id=manifest.model_id, request=request, plan=base_plan)
            else:
                realized = None
        if realized is None:
            return None
        frontier_metadata = request.metadata.setdefault("frontier_architecture", {})
        if isinstance(frontier_metadata, dict):
            frontier_metadata.update(realized)
            return dict(frontier_metadata)
        request.metadata["frontier_architecture"] = dict(realized)
        return dict(realized)

    def performance_feature_snapshot(self) -> dict[str, Any]:
        with self._lock:
            ssm_detected = any(
                plan.get("architecture_subtype") in {ArchitectureSubtype.SSM_MAMBA.value, ArchitectureSubtype.HYBRID_SSM.value}
                for plan in self._registered_plans.values()
            )
            moe_detected = any(
                plan.get("architecture_subtype") in {ArchitectureSubtype.MOE.value, ArchitectureSubtype.HYBRID_MOE.value}
                for plan in self._registered_plans.values()
            )
            state_cache_bytes = sum(self._ssm_peak_state_cache_bytes.values())
            state_cache_entries = sum(len(keys) for keys in self._ssm_prompt_keys.values())
            resident_expert_count = max(
                (len(window.resident_experts) for window in self._moe_windows.values()),
                default=0,
            )
            return {
                "ssm_state_cache_handling": {
                    "supported": ssm_detected,
                    "active": self._ssm_request_count > 0,
                    "reason": (
                        "LewLM is tracking realized hybrid SSM state-cache allocations and reuse."
                        if ssm_detected
                        else "No loaded model currently requires hybrid SSM state-cache handling."
                    ),
                    "metrics": {
                        "request_count": self._ssm_request_count,
                        "state_cache_hits": self._ssm_state_cache_hits,
                        "state_cache_misses": self._ssm_state_cache_misses,
                        "state_cache_entries": state_cache_entries,
                        "state_cache_bytes": state_cache_bytes,
                    },
                },
                "moe_bounded_memory_serving": {
                    "supported": moe_detected,
                    "active": self._moe_request_count > 0 and self.settings.moe_bounded_memory_mode != "off",
                    "reason": (
                        "LewLM is tracking realized resident-expert windows and swap pressure for bounded-memory MoE execution."
                        if moe_detected
                        else "No loaded model currently requires MoE bounded-memory execution."
                    ),
                    "metrics": {
                        "request_count": self._moe_request_count,
                        "swap_count": self._moe_swap_count,
                        "swap_mb": self._moe_swap_mb,
                        "peak_resident_expert_count": max(self._moe_peak_resident_expert_count, resident_expert_count),
                        "configured_mode": self.settings.moe_bounded_memory_mode,
                        "configured_resident_expert_count": self.settings.moe_resident_expert_count,
                    },
                },
            }

    def _annotate_ssm_request(
        self,
        *,
        model_id: str,
        request: GenerateRequest,
        plan: dict[str, int | float | str | bool],
    ) -> dict[str, int | float | str | bool]:
        prompt_key = _prompt_key(request)
        prompt_keys = self._ssm_prompt_keys.setdefault(model_id, set())
        state_size = max(_coerce_int(plan.get("state_size")) or 1, 1)
        prompt_token_estimate = max(_prompt_token_estimate(request), 1)
        state_cache_bytes = state_size * prompt_token_estimate * 2
        cache_hit = prompt_key in prompt_keys
        if cache_hit:
            self._ssm_state_cache_hits += 1
        else:
            prompt_keys.add(prompt_key)
            self._ssm_state_cache_misses += 1
        self._ssm_request_count += 1
        peak_bytes = max(state_cache_bytes, self._ssm_peak_state_cache_bytes.get(model_id, 0))
        self._ssm_peak_state_cache_bytes[model_id] = peak_bytes
        effective_loaded_memory_mb = _coerce_int(plan.get("full_estimated_memory_mb")) or _coerce_int(plan.get("planned_memory_mb"))
        return _compact_metadata(
            plan,
            planning_only=False,
            execution_path="state_cache",
            state_cache_requests=1,
            state_cache_hits=1 if cache_hit else 0,
            state_cache_misses=0 if cache_hit else 1,
            state_cache_entry_count=len(prompt_keys),
            state_cache_bytes=state_cache_bytes,
            effective_loaded_memory_mb=effective_loaded_memory_mb,
        )

    def _annotate_moe_request(
        self,
        *,
        model_id: str,
        request: GenerateRequest,
        plan: dict[str, int | float | str | bool],
    ) -> dict[str, int | float | str | bool]:
        expert_count = max(_coerce_int(plan.get("expert_count")) or 0, 0)
        active_expert_count = max(_coerce_int(plan.get("active_expert_count")) or 1, 1)
        resident_expert_count = max(_coerce_int(plan.get("resident_expert_count")) or expert_count, 0)
        bounded_mode = str(plan.get("bounded_memory_mode") or "off")
        requested_experts = _requested_experts(request=request, expert_count=expert_count, active_expert_count=active_expert_count)
        if bounded_mode == "off" or resident_expert_count <= 0 or expert_count <= 0:
            resident_count = expert_count
            swap_count = 0
        else:
            window = self._moe_windows.setdefault(model_id, _MoEResidencyWindow(resident_limit=resident_expert_count))
            swap_count = window.access(requested_experts)
            resident_count = len(window.resident_experts)
        per_expert_memory_mb = 0
        full_memory_mb = _coerce_int(plan.get("full_estimated_memory_mb"))
        dense_memory_mb = _coerce_int(plan.get("estimated_dense_memory_mb"))
        estimated_expert_memory_mb = _coerce_int(plan.get("estimated_expert_memory_mb"))
        if expert_count > 0 and estimated_expert_memory_mb is not None:
            per_expert_memory_mb = max(int(round(estimated_expert_memory_mb / expert_count)), 1)
        swap_mb = per_expert_memory_mb * swap_count if bounded_mode != "off" else 0
        effective_loaded_memory_mb = _coerce_int(plan.get("planned_memory_mb")) or full_memory_mb
        self._moe_request_count += 1
        self._moe_swap_count += swap_count
        self._moe_swap_mb += swap_mb
        self._moe_peak_resident_expert_count = max(self._moe_peak_resident_expert_count, resident_count)
        if effective_loaded_memory_mb is not None:
            self._effective_loaded_memory_overrides[model_id] = effective_loaded_memory_mb
        return _compact_metadata(
            plan,
            planning_only=False,
            execution_path=bounded_mode if bounded_mode != "off" else "full_resident",
            requested_expert_count=len(requested_experts),
            resident_expert_count=resident_count,
            expert_swap_count=swap_count,
            expert_swap_mb=swap_mb,
            streamed_expert_count=max(expert_count - resident_count, 0) if expert_count > 0 else None,
            effective_loaded_memory_mb=effective_loaded_memory_mb,
            memory_savings_mb=max((full_memory_mb or 0) - (effective_loaded_memory_mb or 0), 0)
            if full_memory_mb is not None and effective_loaded_memory_mb is not None
            else None,
            estimated_dense_memory_mb=dense_memory_mb,
        )


def _prompt_key(request: GenerateRequest) -> str:
    digest = hashlib.sha256()
    digest.update(str(request.model_id).encode("utf-8"))
    digest.update(str(request.max_tokens).encode("utf-8"))
    for message in request.messages:
        digest.update(message.role.encode("utf-8"))
        digest.update(message.content.encode("utf-8"))
    return digest.hexdigest()


def _prompt_token_estimate(request: GenerateRequest) -> int:
    character_count = sum(len(message.content) for message in request.messages)
    return max((character_count + 3) // 4, len(request.messages))


def _requested_experts(*, request: GenerateRequest, expert_count: int, active_expert_count: int) -> list[int]:
    if expert_count <= 0:
        return []
    digest = hashlib.sha256(_prompt_key(request).encode("utf-8")).digest()
    requested: list[int] = []
    cursor = 0
    while len(requested) < min(active_expert_count, expert_count):
        if cursor >= len(digest):
            digest = hashlib.sha256(digest).digest()
            cursor = 0
        candidate = digest[cursor] % expert_count
        cursor += 1
        if candidate in requested:
            continue
        requested.append(candidate)
    return requested


def _compact_metadata(
    plan: dict[str, int | float | str | bool],
    **updates: int | float | str | bool | None,
) -> dict[str, int | float | str | bool]:
    payload: dict[str, int | float | str | bool] = dict(plan)
    for key, value in updates.items():
        if value is None:
            continue
        payload[key] = value
    return payload


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
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
            return None
    return None
