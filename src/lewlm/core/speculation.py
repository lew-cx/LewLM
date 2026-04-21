"""Shared speculation planning, selection, and metrics helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.util import find_spec
from math import ceil
from typing import Any

from pydantic import BaseModel, Field

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    ConversionStatus,
    GenerateMessage,
    GenerateRequest,
    GenerateSpeculation,
    ModelFormat,
    ModelManifest,
    ModelModality,
    RuntimeAffinity,
    RuntimeContract,
    SpeculationMode,
    utc_now,
)
from lewlm.core.errors import ConfigurationError, ModelNotFoundError
from lewlm.registry.service import ModelRegistry

_DEFAULT_MODE_PRIORITY: dict[SpeculationMode, int] = {
    SpeculationMode.MEDUSA: 0,
    SpeculationMode.EAGLE: 1,
    SpeculationMode.HYDRA: 2,
    SpeculationMode.DFLASH: 3,
    SpeculationMode.SELF_SPECULATIVE: 4,
    SpeculationMode.SUFFIX_DECODING: 5,
    SpeculationMode.HETEROGENEOUS_VOCAB: 6,
    SpeculationMode.DRAFT_MODEL: 7,
    SpeculationMode.PROMPT_LOOKUP: 8,
}
_MODE_ALIASES: dict[str, SpeculationMode] = {
    "self_speculation": SpeculationMode.SELF_SPECULATIVE,
    "self_speculative_decoding": SpeculationMode.SELF_SPECULATIVE,
    "suffix": SpeculationMode.SUFFIX_DECODING,
    "suffix_decoder": SpeculationMode.SUFFIX_DECODING,
    "heterogeneous_vocab_draft": SpeculationMode.HETEROGENEOUS_VOCAB,
    "heterogeneous_vocabulary": SpeculationMode.HETEROGENEOUS_VOCAB,
    "heterogeneous_vocabulary_draft": SpeculationMode.HETEROGENEOUS_VOCAB,
    "swift": SpeculationMode.HETEROGENEOUS_VOCAB,
}


@dataclass(frozen=True, slots=True)
class PlannedSpeculation:
    """Resolved speculation configuration for one generation request."""

    request: GenerateSpeculation
    draft_manifest: ModelManifest | None = None
    companion_manifests: tuple[ModelManifest, ...] = ()
    selection_reason: str = ""
    benchmark_preferred: bool = False


@dataclass(frozen=True, slots=True)
class RejectedSpeculationCandidate:
    """A speculation path that was considered but skipped before execution."""

    mode: SpeculationMode | None
    reason: str
    source: str = "heuristic"


@dataclass(frozen=True, slots=True)
class ChatSpeculationInspection:
    """Resolved speculation plans plus skipped-path diagnostics for one request."""

    candidates: tuple[PlannedSpeculation, ...]
    rejected: tuple[RejectedSpeculationCandidate, ...]
    workload_class: str


class SpeculationBenchmarkPreference(BaseModel):
    """Persisted best-known speculation choice for one model/runtime pair."""

    model_id: str
    runtime_name: str
    workload_class: str = "general_chat"
    selected_mode: SpeculationMode | None = None
    benchmark_id: str | None = None
    generate_seconds: float | None = None
    total_seconds: float | None = None
    acceptance_rate: float | None = None
    rollback_tokens: int = 0
    verified_tokens: int = 0
    fallback_count: int = 0
    updated_at: str = Field(default_factory=lambda: utc_now().isoformat())


def speculation_benchmark_preference_key(
    *,
    model_id: str,
    runtime_name: str,
    workload_class: str | None = None,
) -> str:
    base_key = f"speculation.preference::{runtime_name}::{model_id}"
    if workload_class is None or not workload_class:
        return base_key
    return f"{base_key}::{workload_class}"


def chat_speculation_workload_class(
    *,
    messages: Sequence[GenerateMessage],
    max_tokens: int,
) -> str:
    requested_context_tokens = _estimate_chat_context_tokens(messages, max_tokens)
    return _classify_chat_speculation_workload(messages=messages, requested_context_tokens=requested_context_tokens)


def parse_speculation_benchmark_preference(payload: Any) -> SpeculationBenchmarkPreference | None:
    if not isinstance(payload, dict):
        return None
    try:
        return SpeculationBenchmarkPreference.model_validate(payload)
    except Exception:
        return None


def list_chat_speculation_candidates(
    *,
    model_registry: ModelRegistry,
    settings: LewLMSettings,
    primary_manifest: ModelManifest,
    runtime: RuntimeContract,
    messages: Sequence[GenerateMessage],
    max_tokens: int,
    preferred_mode: SpeculationMode | None = None,
) -> list[PlannedSpeculation]:
    inspection = inspect_chat_speculation_candidates(
        model_registry=model_registry,
        settings=settings,
        primary_manifest=primary_manifest,
        runtime=runtime,
        messages=messages,
        max_tokens=max_tokens,
        preferred_mode=preferred_mode,
    )
    return list(inspection.candidates)


def inspect_chat_speculation_candidates(
    *,
    model_registry: ModelRegistry,
    settings: LewLMSettings,
    primary_manifest: ModelManifest,
    runtime: RuntimeContract,
    messages: Sequence[GenerateMessage],
    max_tokens: int,
    preferred_mode: SpeculationMode | None = None,
) -> ChatSpeculationInspection:
    requested_context_tokens = _estimate_chat_context_tokens(messages, max_tokens)
    workload_class = _classify_chat_speculation_workload(
        messages=messages,
        requested_context_tokens=requested_context_tokens,
    )
    candidates: list[PlannedSpeculation] = []
    rejected: list[RejectedSpeculationCandidate] = []
    runtime_supported_modes = _runtime_supported_speculation_modes(runtime)

    if runtime.affinity == RuntimeAffinity.MLX_TEXT and settings.speculative_decoding_enabled:
        if runtime_supported_modes is not None and SpeculationMode.DRAFT_MODEL not in runtime_supported_modes:
            rejected.append(
                RejectedSpeculationCandidate(
                    mode=SpeculationMode.DRAFT_MODEL,
                    reason="Installed MLX backend does not expose draft-model speculation for this runtime.",
                ),
            )
        else:
            mlx_candidate = _plan_mlx_draft_candidate(
                model_registry=model_registry,
                settings=settings,
                primary_manifest=primary_manifest,
                runtime=runtime,
                requested_context_tokens=requested_context_tokens,
            )
            if mlx_candidate is not None:
                candidates.append(mlx_candidate)
            elif primary_manifest.format_type == ModelFormat.MLX:
                rejected.append(
                    RejectedSpeculationCandidate(
                        mode=SpeculationMode.DRAFT_MODEL,
                        reason="No compatible runnable secondary MLX draft model was available for this request.",
                    ),
                )

    if runtime.affinity == RuntimeAffinity.LLAMACPP and settings.prompt_lookup_speculation_enabled:
        if runtime_supported_modes is not None and SpeculationMode.PROMPT_LOOKUP not in runtime_supported_modes:
            rejected.append(
                RejectedSpeculationCandidate(
                    mode=SpeculationMode.PROMPT_LOOKUP,
                    reason="Installed llama.cpp bindings do not expose prompt-lookup speculation for this runtime.",
                ),
            )
        else:
            prompt_lookup_candidate = _plan_prompt_lookup_candidate(settings=settings, runtime=runtime)
            if prompt_lookup_candidate is not None:
                candidates.append(prompt_lookup_candidate)

    frontier_candidates, frontier_rejections = _plan_frontier_metadata_candidates(
        model_registry=model_registry,
        settings=settings,
        primary_manifest=primary_manifest,
        runtime=runtime,
        requested_context_tokens=requested_context_tokens,
        runtime_supported_modes=runtime_supported_modes,
    )
    candidates.extend(frontier_candidates)
    rejected.extend(frontier_rejections)
    return ChatSpeculationInspection(
        candidates=tuple(_prioritize_candidates(candidates, preferred_mode=preferred_mode)),
        rejected=tuple(rejected),
        workload_class=workload_class,
    )


def plan_chat_speculation(
    *,
    model_registry: ModelRegistry,
    settings: LewLMSettings,
    primary_manifest: ModelManifest,
    runtime: RuntimeContract,
    messages: Sequence[GenerateMessage],
    max_tokens: int,
    preferred_mode: SpeculationMode | None = None,
) -> PlannedSpeculation | None:
    candidates = list_chat_speculation_candidates(
        model_registry=model_registry,
        settings=settings,
        primary_manifest=primary_manifest,
        runtime=runtime,
        messages=messages,
        max_tokens=max_tokens,
        preferred_mode=preferred_mode,
    )
    if not candidates:
        return None
    return candidates[0]


def speculation_measurements(
    *,
    request: GenerateRequest,
    usage: Mapping[str, int | float | str | bool] | None = None,
) -> dict[str, int | float]:
    speculation = request.speculation
    if speculation is None:
        return {}
    usage_map = dict(usage or {})
    runtime_summary = _speculation_runtime_summary(request.metadata)
    measurements: dict[str, int | float] = {
        "speculative_requests": 1,
    }
    execution_path = request.metadata.get("speculation_execution_path")
    if execution_path == "lewlm_controller":
        measurements["controller_owned_requests"] = 1
    elif execution_path == "backend_passthrough":
        measurements["backend_passthrough_requests"] = 1
    if request.metadata.get("speculation_benchmark_preferred") is True:
        measurements["benchmark_preferred_requests"] = 1
    fallback_count = _coerce_int(
        request.metadata.get(
            "speculation_fallback_count",
            runtime_summary.get("fallback_count", usage_map.get("fallback_count", 0)),
        ),
    )
    if fallback_count:
        measurements["fallback_count"] = fallback_count

    drafted_tokens = _coerce_int(
        usage_map.get(
            "drafted_tokens",
            runtime_summary.get("drafted_tokens", usage_map.get("accepted_tokens", 0)),
        ),
    )
    accepted_tokens = _coerce_int(
        usage_map.get(
            "accepted_tokens",
            runtime_summary.get("accepted_tokens", usage_map.get("verified_tokens", 0)),
        ),
    )
    verified_tokens = _coerce_int(
        usage_map.get(
            "verified_tokens",
            runtime_summary.get("verified_tokens", accepted_tokens),
        ),
    )
    rejected_tokens = _coerce_int(
        usage_map.get(
            "rejected_tokens",
            runtime_summary.get("rejected_tokens"),
        ),
    )
    rollback_tokens = _coerce_int(
        usage_map.get(
            "rollback_tokens",
            runtime_summary.get("rollback_tokens"),
        ),
    )
    if accepted_tokens <= 0 and verified_tokens > 0 and drafted_tokens > 0:
        accepted_tokens = min(verified_tokens, drafted_tokens)
    if rejected_tokens <= 0 and drafted_tokens > 0 and accepted_tokens >= 0:
        rejected_tokens = max(drafted_tokens - accepted_tokens, 0)
    if rollback_tokens <= 0 and drafted_tokens > 0:
        rollback_tokens = max(rejected_tokens, 0)

    if speculation.mode == SpeculationMode.DRAFT_MODEL:
        measurements["draft_model_requests"] = 1
        if speculation.num_draft_tokens is not None:
            measurements["configured_num_draft_tokens"] = speculation.num_draft_tokens
    elif speculation.mode == SpeculationMode.PROMPT_LOOKUP:
        measurements["prompt_lookup_requests"] = 1
        if speculation.prompt_lookup_max_ngram_size is not None:
            measurements["prompt_lookup_max_ngram_size"] = speculation.prompt_lookup_max_ngram_size
        if speculation.prompt_lookup_num_pred_tokens is not None:
            measurements["prompt_lookup_num_pred_tokens"] = speculation.prompt_lookup_num_pred_tokens
    else:
        measurements["frontier_speculation_requests"] = 1

    if drafted_tokens:
        measurements["drafted_tokens"] = drafted_tokens
    if accepted_tokens:
        measurements["accepted_tokens"] = accepted_tokens
    if verified_tokens:
        measurements["verified_tokens"] = verified_tokens
    if rejected_tokens:
        measurements["rejected_tokens"] = rejected_tokens
    if rollback_tokens:
        measurements["rollback_tokens"] = rollback_tokens
    if drafted_tokens > 0:
        measurements["acceptance_rate"] = round(accepted_tokens / drafted_tokens, 4)
    return measurements


def _speculation_runtime_summary(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime_summary = metadata.get("speculation_runtime")
    return runtime_summary if isinstance(runtime_summary, Mapping) else {}


def _plan_mlx_draft_candidate(
    *,
    model_registry: ModelRegistry,
    settings: LewLMSettings,
    primary_manifest: ModelManifest,
    runtime: RuntimeContract,
    requested_context_tokens: int | None,
) -> PlannedSpeculation | None:
    if runtime.affinity != RuntimeAffinity.MLX_TEXT or not settings.speculative_decoding_enabled:
        return None
    draft_manifest, auto_selected = _resolve_mlx_draft_manifest(
        model_registry=model_registry,
        settings=settings,
        primary_manifest=primary_manifest,
        requested_context_tokens=requested_context_tokens,
    )
    if draft_manifest is None:
        return None
    return PlannedSpeculation(
        request=GenerateSpeculation(
            mode=SpeculationMode.DRAFT_MODEL,
            draft_model_id=draft_manifest.model_id,
            companion_model_id=draft_manifest.model_id,
            num_draft_tokens=settings.speculative_decoding_num_draft_tokens,
            auto_selected=auto_selected,
        ),
        draft_manifest=draft_manifest,
        companion_manifests=(draft_manifest,),
        selection_reason=(
            "Auto-selected the smallest compatible runnable MLX companion model for speculative decoding."
            if auto_selected
            else "Using the explicitly configured MLX draft model for speculative decoding."
        ),
    )


def _plan_prompt_lookup_candidate(
    *,
    settings: LewLMSettings,
    runtime: RuntimeContract,
) -> PlannedSpeculation | None:
    if runtime.affinity != RuntimeAffinity.LLAMACPP or not settings.prompt_lookup_speculation_enabled:
        return None
    return PlannedSpeculation(
        request=GenerateSpeculation(
            mode=SpeculationMode.PROMPT_LOOKUP,
            prompt_lookup_max_ngram_size=settings.prompt_lookup_max_ngram_size,
            prompt_lookup_num_pred_tokens=settings.prompt_lookup_num_pred_tokens,
        ),
        selection_reason="Using llama.cpp prompt-lookup speculation with the configured n-gram window.",
    )


def _plan_frontier_metadata_candidates(
    *,
    model_registry: ModelRegistry,
    settings: LewLMSettings,
    primary_manifest: ModelManifest,
    runtime: RuntimeContract,
    requested_context_tokens: int | None,
    runtime_supported_modes: set[SpeculationMode] | None = None,
) -> tuple[list[PlannedSpeculation], list[RejectedSpeculationCandidate]]:
    raw_entries = primary_manifest.metadata.get("speculation_modes")
    if raw_entries is None:
        raw_entries = primary_manifest.metadata.get("frontier_speculation")
    entries = _normalize_frontier_entries(raw_entries)
    candidates: list[PlannedSpeculation] = []
    rejected: list[RejectedSpeculationCandidate] = []
    for entry in entries:
        mode = _coerce_mode(entry.get("mode"))
        if mode is None:
            rejected.append(
                RejectedSpeculationCandidate(
                    mode=None,
                    reason=f"Skipping unknown frontier speculation mode `{entry.get('mode')}` from model metadata.",
                    source="metadata",
                ),
            )
            continue
        if mode in {SpeculationMode.DRAFT_MODEL, SpeculationMode.PROMPT_LOOKUP}:
            continue
        if not _settings_allow_mode(settings=settings, mode=mode):
            rejected.append(
                RejectedSpeculationCandidate(
                    mode=mode,
                    reason=f"`{mode.value}` speculation is available in model metadata but disabled in settings.",
                    source="metadata",
                ),
            )
            continue
        if not _entry_runtime_matches(entry=entry, runtime=runtime):
            rejected.append(
                RejectedSpeculationCandidate(
                    mode=mode,
                    reason=f"`{mode.value}` speculation metadata targets a different runtime affinity.",
                    source="metadata",
                ),
            )
            continue
        if runtime_supported_modes is not None and mode not in runtime_supported_modes:
            rejected.append(
                RejectedSpeculationCandidate(
                    mode=mode,
                    reason=f"The active runtime does not expose a compatible `{mode.value}` speculation hook.",
                    source="metadata",
                ),
            )
            continue
        if not _entry_modules_available(entry):
            rejected.append(
                RejectedSpeculationCandidate(
                    mode=mode,
                    reason=f"`{mode.value}` speculation metadata requires optional local modules that are not installed.",
                    source="metadata",
                ),
            )
            continue
        parameters = _normalize_parameter_mapping(entry.get("parameters"))
        backend_parameter = parameters.get("backend_parameter")
        if runtime_supported_modes is None and (not isinstance(backend_parameter, str) or not backend_parameter):
            rejected.append(
                RejectedSpeculationCandidate(
                    mode=mode,
                    reason=f"`{mode.value}` speculation metadata is missing a backend parameter hint.",
                    source="metadata",
                ),
            )
            continue
        companion_model_id = _coerce_optional_string(
            entry.get("companion_model_id", entry.get("draft_model_id")),
        )
        companion_manifest = None
        if companion_model_id is not None:
            try:
                companion_manifest = model_registry.get_manifest(companion_model_id)
                _validate_frontier_companion_candidate(
                    companion_manifest=companion_manifest,
                    primary_manifest=primary_manifest,
                    runtime=runtime,
                    requested_context_tokens=requested_context_tokens,
                )
            except (ConfigurationError, KeyError, ModelNotFoundError) as exc:
                rejected.append(
                    RejectedSpeculationCandidate(
                        mode=mode,
                        reason=str(exc),
                        source="metadata",
                    ),
                )
                continue
        request = GenerateSpeculation(
            mode=mode,
            draft_model_id=companion_model_id if mode == SpeculationMode.DRAFT_MODEL else None,
            companion_model_id=companion_model_id,
            num_draft_tokens=_coerce_optional_int(entry.get("num_draft_tokens")),
            parameters=parameters,
            auto_selected=True,
        )
        if mode == SpeculationMode.PROMPT_LOOKUP:
            request.prompt_lookup_max_ngram_size = _coerce_optional_int(entry.get("prompt_lookup_max_ngram_size"))
            request.prompt_lookup_num_pred_tokens = _coerce_optional_int(entry.get("prompt_lookup_num_pred_tokens"))
        candidates.append(
            PlannedSpeculation(
                request=request,
                draft_manifest=companion_manifest if mode == SpeculationMode.DRAFT_MODEL else None,
                companion_manifests=(companion_manifest,) if companion_manifest is not None else (),
                selection_reason=(
                    f"Using model metadata to enable the `{mode.value}` speculation adapter with backend parameter "
                    f"`{backend_parameter or 'runtime-detected'}`."
                ),
            ),
        )
    return candidates, rejected


def _prioritize_candidates(
    candidates: Sequence[PlannedSpeculation],
    *,
    preferred_mode: SpeculationMode | None,
) -> list[PlannedSpeculation]:
    deduped: dict[SpeculationMode, PlannedSpeculation] = {}
    for candidate in candidates:
        deduped.setdefault(candidate.request.mode, candidate)
    ordered = list(deduped.values())
    ordered.sort(
        key=lambda candidate: (
            0 if preferred_mode is not None and candidate.request.mode == preferred_mode else 1,
            _DEFAULT_MODE_PRIORITY.get(candidate.request.mode, 99),
            candidate.request.mode.value,
        ),
    )
    if preferred_mode is not None:
        ordered = [
            replace(candidate, benchmark_preferred=(candidate.request.mode == preferred_mode))
            for candidate in ordered
        ]
    return ordered


def _resolve_mlx_draft_manifest(
    *,
    model_registry: ModelRegistry,
    settings: LewLMSettings,
    primary_manifest: ModelManifest,
    requested_context_tokens: int | None,
) -> tuple[ModelManifest | None, bool]:
    explicit_draft_model_id = settings.speculative_decoding_draft_model_id
    if explicit_draft_model_id:
        draft_manifest = model_registry.get_manifest(explicit_draft_model_id)
        _validate_mlx_draft_candidate(
            draft_manifest=draft_manifest,
            primary_manifest=primary_manifest,
            requested_context_tokens=requested_context_tokens,
        )
        return draft_manifest, False

    candidates = [
        manifest
        for manifest in model_registry.list_manifests()
        if manifest.model_id != primary_manifest.model_id
        and _is_mlx_draft_candidate(
            draft_manifest=manifest,
            primary_manifest=primary_manifest,
            requested_context_tokens=requested_context_tokens,
        )
    ]
    if not candidates:
        return None, False
    candidates.sort(
        key=lambda manifest: (
            0 if manifest.architecture_family == primary_manifest.architecture_family else 1,
            0
            if (
                primary_manifest.estimated_memory_mb is not None
                and manifest.estimated_memory_mb is not None
                and manifest.estimated_memory_mb < primary_manifest.estimated_memory_mb
            )
            else 1,
            manifest.estimated_memory_mb if manifest.estimated_memory_mb is not None else 1_000_000_000,
            manifest.model_id,
        ),
    )
    return candidates[0], True


def _validate_mlx_draft_candidate(
    *,
    draft_manifest: ModelManifest,
    primary_manifest: ModelManifest,
    requested_context_tokens: int | None,
) -> None:
    if draft_manifest.model_id == primary_manifest.model_id:
        raise ConfigurationError(
            "speculative_decoding_draft_model_id must reference a secondary MLX model, not the primary model.",
            details={"model_id": draft_manifest.model_id},
        )
    if not _is_mlx_draft_candidate(
        draft_manifest=draft_manifest,
        primary_manifest=primary_manifest,
        requested_context_tokens=requested_context_tokens,
    ):
        raise ConfigurationError(
            "Configured speculative_decoding_draft_model_id is not a runnable MLX chat candidate for the current request.",
            details={
                "draft_model_id": draft_manifest.model_id,
                "draft_format_type": draft_manifest.format_type.value,
                "draft_runtime_affinity": [affinity.value for affinity in draft_manifest.runtime_affinity],
            },
        )


def _validate_frontier_companion_candidate(
    *,
    companion_manifest: ModelManifest,
    primary_manifest: ModelManifest,
    runtime: RuntimeContract,
    requested_context_tokens: int | None,
) -> None:
    if companion_manifest.model_id == primary_manifest.model_id:
        raise ConfigurationError(
            "Frontier speculation companion manifests must reference a secondary runnable model.",
            details={"model_id": companion_manifest.model_id},
        )
    if companion_manifest.conversion_status != ConversionStatus.RUNNABLE:
        raise ConfigurationError(
            "Frontier speculation companion manifests must be runnable.",
            details={"model_id": companion_manifest.model_id},
        )
    if runtime.affinity not in companion_manifest.runtime_affinity:
        raise ConfigurationError(
            "Frontier speculation companion manifest is incompatible with the selected runtime affinity.",
            details={
                "model_id": companion_manifest.model_id,
                "runtime_affinity": [affinity.value for affinity in companion_manifest.runtime_affinity],
            },
        )
    if (
        requested_context_tokens is not None
        and companion_manifest.context_length is not None
        and companion_manifest.context_length < requested_context_tokens
    ):
        raise ConfigurationError(
            "Frontier speculation companion manifest does not satisfy the current context requirement.",
            details={
                "model_id": companion_manifest.model_id,
                "context_length": companion_manifest.context_length,
                "requested_context_tokens": requested_context_tokens,
            },
        )


def _is_mlx_draft_candidate(
    *,
    draft_manifest: ModelManifest,
    primary_manifest: ModelManifest,
    requested_context_tokens: int | None,
) -> bool:
    if draft_manifest.conversion_status != ConversionStatus.RUNNABLE:
        return False
    if draft_manifest.format_type != ModelFormat.MLX:
        return False
    if RuntimeAffinity.MLX_TEXT not in draft_manifest.runtime_affinity:
        return False
    if not any(modality in draft_manifest.modality for modality in (ModelModality.TEXT, ModelModality.MULTIMODAL)):
        return False
    if (
        requested_context_tokens is not None
        and draft_manifest.context_length is not None
        and draft_manifest.context_length < requested_context_tokens
    ):
        return False
    return primary_manifest.format_type == ModelFormat.MLX


def _normalize_frontier_entries(raw_entries: Any) -> list[dict[str, Any]]:
    if isinstance(raw_entries, dict):
        normalized: list[dict[str, Any]] = []
        for key, value in raw_entries.items():
            if isinstance(value, dict):
                normalized.append({"mode": key, **value})
            else:
                normalized.append({"mode": key})
        return normalized
    if isinstance(raw_entries, list):
        normalized = []
        for entry in raw_entries:
            if isinstance(entry, str):
                normalized.append({"mode": entry})
            elif isinstance(entry, dict):
                normalized.append(dict(entry))
        return normalized
    return []


def _entry_runtime_matches(*, entry: Mapping[str, Any], runtime: RuntimeContract) -> bool:
    runtime_affinity = entry.get("runtime_affinity")
    if isinstance(runtime_affinity, str):
        return runtime_affinity == runtime.affinity.value
    if isinstance(runtime_affinity, list):
        return runtime.affinity.value in {str(item) for item in runtime_affinity}
    return True


def _entry_modules_available(entry: Mapping[str, Any]) -> bool:
    required_modules: list[str] = []
    candidate = entry.get("required_module")
    if isinstance(candidate, str) and candidate:
        required_modules.append(candidate)
    candidate_list = entry.get("required_modules")
    if isinstance(candidate_list, list):
        required_modules.extend(str(item) for item in candidate_list if isinstance(item, str) and item)
    return all(find_spec(module_name) is not None for module_name in required_modules)


def _settings_allow_mode(*, settings: LewLMSettings, mode: SpeculationMode) -> bool:
    if mode == SpeculationMode.PROMPT_LOOKUP:
        return settings.prompt_lookup_speculation_enabled
    return settings.speculative_decoding_enabled


def _normalize_parameter_mapping(raw_parameters: Any) -> dict[str, str | int | float | bool | None]:
    if not isinstance(raw_parameters, dict):
        return {}
    normalized: dict[str, str | int | float | bool | None] = {}
    for key, value in raw_parameters.items():
        if not isinstance(key, str) or not key:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            normalized[key] = value
    return normalized


def _coerce_mode(value: Any) -> SpeculationMode | None:
    if isinstance(value, SpeculationMode):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    alias = _MODE_ALIASES.get(normalized)
    if alias is not None:
        return alias
    try:
        return SpeculationMode(normalized)
    except ValueError:
        return None


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


def _coerce_optional_int(value: Any) -> int | None:
    coerced = _coerce_int(value)
    return coerced if coerced > 0 else None


def _coerce_optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _estimate_chat_context_tokens(messages: Sequence[GenerateMessage], max_tokens: int) -> int:
    return sum(_estimate_text_tokens(message.content) for message in messages) + max(0, max_tokens)


def _estimate_text_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, ceil(len(normalized) / 4))


def _runtime_supported_speculation_modes(runtime: RuntimeContract) -> set[SpeculationMode] | None:
    snapshot_fn = getattr(runtime, "performance_feature_snapshot", None)
    if not callable(snapshot_fn):
        return None
    snapshot = snapshot_fn()
    if not isinstance(snapshot, dict):
        return None
    modes: set[SpeculationMode] = set()
    known = False
    speculative_entry = snapshot.get("speculative_decoding")
    if isinstance(speculative_entry, dict):
        known = True
        if speculative_entry.get("supported") is True:
            raw_modes = speculative_entry.get("modes")
            if isinstance(raw_modes, list):
                for raw_mode in raw_modes:
                    mode = _coerce_mode(raw_mode)
                    if mode is not None:
                        modes.add(mode)
    prompt_lookup_entry = snapshot.get("prompt_lookup_speculation")
    if isinstance(prompt_lookup_entry, dict):
        known = True
        if prompt_lookup_entry.get("supported") is True:
            modes.add(SpeculationMode.PROMPT_LOOKUP)
    return modes if known else None


def _classify_chat_speculation_workload(
    *,
    messages: Sequence[GenerateMessage],
    requested_context_tokens: int,
) -> str:
    combined_text = "\n".join(message.content for message in messages).casefold()
    if requested_context_tokens >= 4096:
        return "long_context"
    if any(
        token in combined_text
        for token in (
            "```",
            "def ",
            "class ",
            "function ",
            "traceback",
            "stack trace",
            "bug",
            "refactor",
            "pytest",
            "regex",
            "algorithm",
            "sql",
        )
    ):
        return "coding"
    if any(token in combined_text for token in ("json", "schema", "csv", "yaml", "table", "structured")):
        return "structured"
    if any(token in combined_text for token in ("tool", "repository", "repo", "file", "plan", "step by step", "agent")):
        return "agentic"
    return "general_chat"
