"""Performance-core acceptance evidence helpers for release manifests."""

from __future__ import annotations

from typing import Any

PERFORMANCE_CORE_PILLAR_GUIDANCE: dict[str, list[str]] = {
    "serving_core": [
        "Run a chat or streaming request on a host-verified runtime so LewLM emits serving-core sequence, queue, and phase metadata.",
    ],
    "continuous_batching": [
        "Benchmark a chat-capable model until the `continuous_batching` scenario is recorded, then persist a serving profile for that host/model/runtime path.",
    ],
    "prefix_reuse": [
        "Benchmark repeated prompts until `repeated_prefix` and `warm_chat_cache` evidence is recorded for the target host.",
    ],
    "tiered_kv_cache": [
        "Benchmark a first-class text path that reports `paged_kv_cache`, then persist a serving profile so the KV default is benchmark-backed on the target host.",
    ],
    "speculation": [
        "Benchmark a model that exposes speculation until `speculation_selection` is recorded, then persist the selected serving profile.",
    ],
    "constrained_decoding": [
        "Record a decode-time constrained-output benchmark artifact or serving profile once LewLM owns that path on the target host.",
    ],
    "measured_registry_defaults": [
        "Persist serving profiles and benchmark-backed optimization defaults so runtime and optimization choices are adopted from measured host evidence rather than declarations alone.",
    ],
}

_CONSTRAINED_DECODING_FEATURE_NAMES = (
    "constrained_decoding",
    "json_schema_constrained_decoding",
    "grammar_constrained_decoding",
)
_CONSTRAINED_DECODING_SETTING_KEYS = (
    "constrained_decoding",
    "constraint_mode",
    "structured_output_mode",
)


def build_performance_core_acceptance_summary(
    *,
    runtime_stats: dict[str, Any],
    benchmark_artifacts: list[dict[str, Any]],
    serving_profiles: list[dict[str, Any]],
    optimization_defaults: dict[str, Any] | None,
    capability_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize release-manifest evidence for Milestone 81 performance-core proof."""

    sanitized_artifacts = [artifact for artifact in benchmark_artifacts if isinstance(artifact, dict)]
    sanitized_profiles = [profile for profile in serving_profiles if isinstance(profile, dict)]
    sanitized_capability_reports = [report for report in (capability_reports or []) if isinstance(report, dict)]
    artifact_by_id = {
        str(artifact.get("artifact_id")): artifact
        for artifact in sanitized_artifacts
        if artifact.get("artifact_id")
    }
    feature_map = _runtime_feature_map(runtime_stats)
    optimization_defaults = optimization_defaults if isinstance(optimization_defaults, dict) else {}

    pillars = {
        "serving_core": _serving_core_entry(runtime_stats=runtime_stats, feature_map=feature_map),
        "continuous_batching": _benchmark_feature_entry(
            pillar="continuous_batching",
            feature_map=feature_map,
            benchmark_artifacts=sanitized_artifacts,
            serving_profiles=sanitized_profiles,
            artifact_by_id=artifact_by_id,
            feature_names=("continuous_batching",),
            scenario_names=("continuous_batching",),
            runtime_feature_name="continuous_batching",
            missing_reason="No benchmark artifact or persisted serving profile currently proves continuous batching on this host.",
        ),
        "prefix_reuse": _benchmark_feature_entry(
            pillar="prefix_reuse",
            feature_map=feature_map,
            benchmark_artifacts=sanitized_artifacts,
            serving_profiles=sanitized_profiles,
            artifact_by_id=artifact_by_id,
            feature_names=("prefix_cache", "persistent_multi_context_cache"),
            scenario_names=("repeated_prefix", "warm_chat_cache"),
            runtime_feature_name="prefix_cache",
            extra_runtime_feature_names=("persistent_multi_context_cache",),
            missing_reason="No benchmark artifact or persisted serving profile currently proves repeated-prefix or warm-cache reuse on this host.",
        ),
        "tiered_kv_cache": _tiered_kv_cache_entry(
            feature_map=feature_map,
            benchmark_artifacts=sanitized_artifacts,
            serving_profiles=sanitized_profiles,
            artifact_by_id=artifact_by_id,
            optimization_defaults=optimization_defaults,
        ),
        "speculation": _speculation_entry(
            feature_map=feature_map,
            benchmark_artifacts=sanitized_artifacts,
            serving_profiles=sanitized_profiles,
            artifact_by_id=artifact_by_id,
            optimization_defaults=optimization_defaults,
        ),
        "constrained_decoding": _constrained_decoding_entry(
            benchmark_artifacts=sanitized_artifacts,
            serving_profiles=sanitized_profiles,
            artifact_by_id=artifact_by_id,
            capability_reports=sanitized_capability_reports,
        ),
        "measured_registry_defaults": _measured_registry_defaults_entry(
            optimization_defaults=optimization_defaults,
            serving_profiles=sanitized_profiles,
        ),
    }

    covered_pillars = sorted(name for name, entry in pillars.items() if entry.get("status") == "covered")
    supported_unproved_pillars = sorted(
        name for name, entry in pillars.items() if entry.get("status") == "supported_unproved"
    )
    missing_pillars = sorted(name for name, entry in pillars.items() if entry.get("status") == "missing")

    return {
        "format": "lewlm-performance-core-acceptance-v1",
        "host_platform": runtime_stats.get("platform", {}),
        "benchmark_artifact_count": len(sanitized_artifacts),
        "recommended_profile_count": len(sanitized_profiles),
        "covered_pillar_count": len(covered_pillars),
        "supported_unproved_pillar_count": len(supported_unproved_pillars),
        "missing_pillar_count": len(missing_pillars),
        "covered_pillars": covered_pillars,
        "supported_unproved_pillars": supported_unproved_pillars,
        "missing_pillars": missing_pillars,
        "complete": not supported_unproved_pillars and not missing_pillars,
        "pillars": pillars,
        "recommended_profiles": [_profile_summary(profile) for profile in sanitized_profiles[:20]],
        "benchmark_artifacts": [_artifact_summary(artifact) for artifact in sanitized_artifacts[:20]],
    }


def _serving_core_entry(
    *,
    runtime_stats: dict[str, Any],
    feature_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    feature = feature_map.get("serving_core")
    serving_core = runtime_stats.get("serving_core")
    if isinstance(feature, dict) and bool(feature.get("supported")) and isinstance(serving_core, dict):
        return {
            "status": "covered",
            "reason": "Runtime stats expose LewLM serving-core sequence, queue, and scheduler diagnostics on this host.",
            "benchmark_backed": False,
            "runtime_names": list(feature.get("runtime_names", [])),
            "feature_names": ["serving_core"],
            "profile_ids": [],
            "artifact_ids": [],
            "artifact_paths": [],
            "metrics": {
                "recent_sequence_count": int(serving_core.get("recent_sequence_count", 0) or 0),
                "total_sequences_started": int(serving_core.get("total_sequences_started", 0) or 0),
                "total_sequences_completed": int(serving_core.get("total_sequences_completed", 0) or 0),
                "scheduler_peak_active_requests": int(
                    ((runtime_stats.get("request_scheduler") or {}).get("peak_active_requests", 0) or 0),
                ),
                "load_scheduler_peak_active_requests": int(
                    ((runtime_stats.get("load_scheduler") or {}).get("peak_active_requests", 0) or 0),
                ),
            },
            "fallback_guidance": [],
        }
    if isinstance(feature, dict) and bool(feature.get("supported")):
        return {
            "status": "supported_unproved",
            "reason": "A runtime reports serving-core support, but the release manifest does not yet include a serving-core runtime snapshot for this host.",
            "benchmark_backed": False,
            "runtime_names": list(feature.get("runtime_names", [])),
            "feature_names": ["serving_core"],
            "profile_ids": [],
            "artifact_ids": [],
            "artifact_paths": [],
            "metrics": {},
            "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE["serving_core"],
        }
    return {
        "status": "missing",
        "reason": "No chat-capable runtime currently reports serving-core support in the release manifest.",
        "benchmark_backed": False,
        "runtime_names": [],
        "feature_names": ["serving_core"],
        "profile_ids": [],
        "artifact_ids": [],
        "artifact_paths": [],
        "metrics": {},
        "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE["serving_core"],
    }


def _benchmark_feature_entry(
    *,
    pillar: str,
    feature_map: dict[str, dict[str, Any]],
    benchmark_artifacts: list[dict[str, Any]],
    serving_profiles: list[dict[str, Any]],
    artifact_by_id: dict[str, dict[str, Any]],
    feature_names: tuple[str, ...],
    scenario_names: tuple[str, ...],
    runtime_feature_name: str,
    extra_runtime_feature_names: tuple[str, ...] = (),
    missing_reason: str,
) -> dict[str, Any]:
    matched_artifacts = _matching_artifacts(
        benchmark_artifacts=benchmark_artifacts,
        feature_names=feature_names,
        scenario_names=scenario_names,
    )
    matched_profiles = _matching_profiles(
        serving_profiles=serving_profiles,
        artifact_by_id=artifact_by_id,
        feature_names=feature_names,
        scenario_names=scenario_names,
    )
    runtime_names = sorted(
        {
            runtime_name
            for feature_name in (runtime_feature_name, *extra_runtime_feature_names)
            for runtime_name in _feature_runtime_names(feature_map.get(feature_name))
        },
    )
    metrics = {
        "artifact_count": len(matched_artifacts),
        "profile_count": len(matched_profiles),
    }
    if matched_artifacts or matched_profiles:
        return {
            "status": "covered",
            "reason": (
                f"Benchmark artifacts and/or persisted serving profiles record `{pillar}` evidence on this host."
            ),
            "benchmark_backed": True,
            "runtime_names": runtime_names,
            "feature_names": list(feature_names),
            "profile_ids": [str(profile.get("profile_id")) for profile in matched_profiles if profile.get("profile_id")],
            "artifact_ids": [str(artifact.get("artifact_id")) for artifact in matched_artifacts if artifact.get("artifact_id")],
            "artifact_paths": [
                str(artifact.get("artifact_path"))
                for artifact in matched_artifacts
                if artifact.get("artifact_path")
            ],
            "metrics": metrics,
            "fallback_guidance": [],
        }
    runtime_feature_supported = any(
        bool((feature_map.get(feature_name) or {}).get("supported"))
        for feature_name in (runtime_feature_name, *extra_runtime_feature_names)
    )
    if runtime_feature_supported:
        return {
            "status": "supported_unproved",
            "reason": (
                f"Runtime health reports `{pillar}` support, but no benchmark-backed artifact or persisted serving profile is recorded yet."
            ),
            "benchmark_backed": False,
            "runtime_names": runtime_names,
            "feature_names": list(feature_names),
            "profile_ids": [],
            "artifact_ids": [],
            "artifact_paths": [],
            "metrics": metrics,
            "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE[pillar],
        }
    return {
        "status": "missing",
        "reason": missing_reason,
        "benchmark_backed": False,
        "runtime_names": runtime_names,
        "feature_names": list(feature_names),
        "profile_ids": [],
        "artifact_ids": [],
        "artifact_paths": [],
        "metrics": metrics,
        "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE[pillar],
    }


def _tiered_kv_cache_entry(
    *,
    feature_map: dict[str, dict[str, Any]],
    benchmark_artifacts: list[dict[str, Any]],
    serving_profiles: list[dict[str, Any]],
    artifact_by_id: dict[str, dict[str, Any]],
    optimization_defaults: dict[str, Any],
) -> dict[str, Any]:
    matched_artifacts = _matching_artifacts(
        benchmark_artifacts=benchmark_artifacts,
        feature_names=("paged_kv_cache", "kv_cache_quantization"),
        scenario_names=(),
    )
    matched_profiles = _matching_profiles(
        serving_profiles=serving_profiles,
        artifact_by_id=artifact_by_id,
        feature_names=("paged_kv_cache", "kv_cache_quantization"),
        scenario_names=(),
    )
    benchmark_backed_classes = _benchmark_backed_classes(optimization_defaults)
    resolved_classes = _resolved_classes(optimization_defaults)
    runtime_names = sorted(
        set(_feature_runtime_names(feature_map.get("paged_kv_cache")))
        | set(_feature_runtime_names(feature_map.get("kv_cache_quantization")))
    )
    metrics = {
        "artifact_count": len(matched_artifacts),
        "profile_count": len(matched_profiles),
        "tiered_kv_cache_benchmark_backed": "tiered_kv_cache" in benchmark_backed_classes,
    }
    if matched_artifacts and "tiered_kv_cache" in resolved_classes:
        return {
            "status": "covered",
            "reason": "Benchmark evidence and optimization-default resolution both confirm LewLM-owned tiered KV behavior on this host.",
            "benchmark_backed": "tiered_kv_cache" in benchmark_backed_classes,
            "runtime_names": runtime_names,
            "feature_names": ["paged_kv_cache", "kv_cache_quantization"],
            "profile_ids": [str(profile.get("profile_id")) for profile in matched_profiles if profile.get("profile_id")],
            "artifact_ids": [str(artifact.get("artifact_id")) for artifact in matched_artifacts if artifact.get("artifact_id")],
            "artifact_paths": [
                str(artifact.get("artifact_path"))
                for artifact in matched_artifacts
                if artifact.get("artifact_path")
            ],
            "metrics": metrics,
            "fallback_guidance": [],
        }
    runtime_feature_supported = any(
        bool((feature_map.get(feature_name) or {}).get("supported"))
        for feature_name in ("paged_kv_cache", "kv_cache_quantization")
    )
    if runtime_feature_supported or "tiered_kv_cache" in resolved_classes:
        return {
            "status": "supported_unproved",
            "reason": "Tiered KV support is present in runtime health or defaults, but the release manifest does not yet contain full benchmark-backed proof for this host.",
            "benchmark_backed": "tiered_kv_cache" in benchmark_backed_classes,
            "runtime_names": runtime_names,
            "feature_names": ["paged_kv_cache", "kv_cache_quantization"],
            "profile_ids": [str(profile.get("profile_id")) for profile in matched_profiles if profile.get("profile_id")],
            "artifact_ids": [str(artifact.get("artifact_id")) for artifact in matched_artifacts if artifact.get("artifact_id")],
            "artifact_paths": [
                str(artifact.get("artifact_path"))
                for artifact in matched_artifacts
                if artifact.get("artifact_path")
            ],
            "metrics": metrics,
            "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE["tiered_kv_cache"],
        }
    return {
        "status": "missing",
        "reason": "No benchmark artifact, persisted serving profile, or resolved optimization default currently proves tiered KV ownership on this host.",
        "benchmark_backed": False,
        "runtime_names": runtime_names,
        "feature_names": ["paged_kv_cache", "kv_cache_quantization"],
        "profile_ids": [],
        "artifact_ids": [],
        "artifact_paths": [],
        "metrics": metrics,
        "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE["tiered_kv_cache"],
    }


def _speculation_entry(
    *,
    feature_map: dict[str, dict[str, Any]],
    benchmark_artifacts: list[dict[str, Any]],
    serving_profiles: list[dict[str, Any]],
    artifact_by_id: dict[str, dict[str, Any]],
    optimization_defaults: dict[str, Any],
) -> dict[str, Any]:
    matched_artifacts = _matching_artifacts(
        benchmark_artifacts=benchmark_artifacts,
        feature_names=("speculative_decoding", "prompt_lookup_speculation"),
        scenario_names=("speculation_selection",),
    )
    matched_profiles = [
        profile
        for profile in serving_profiles
        if profile.get("selected_speculation_mode")
        or _profile_matches(
            profile=profile,
            artifact_by_id=artifact_by_id,
            feature_names=("speculative_decoding", "prompt_lookup_speculation"),
            scenario_names=("speculation_selection",),
        )
    ]
    benchmark_backed_classes = _benchmark_backed_classes(optimization_defaults)
    resolved_classes = _resolved_classes(optimization_defaults)
    runtime_names = sorted(
        set(_feature_runtime_names(feature_map.get("speculative_decoding")))
        | set(_feature_runtime_names(feature_map.get("prompt_lookup_speculation")))
    )
    metrics = {
        "artifact_count": len(matched_artifacts),
        "profile_count": len(matched_profiles),
        "speculation_benchmark_backed": "speculation" in benchmark_backed_classes,
    }
    if matched_artifacts or matched_profiles:
        return {
            "status": "covered",
            "reason": "Benchmark artifacts and/or serving profiles record speculative execution evidence on this host.",
            "benchmark_backed": True,
            "runtime_names": runtime_names,
            "feature_names": ["speculative_decoding", "prompt_lookup_speculation"],
            "profile_ids": [str(profile.get("profile_id")) for profile in matched_profiles if profile.get("profile_id")],
            "artifact_ids": [str(artifact.get("artifact_id")) for artifact in matched_artifacts if artifact.get("artifact_id")],
            "artifact_paths": [
                str(artifact.get("artifact_path"))
                for artifact in matched_artifacts
                if artifact.get("artifact_path")
            ],
            "metrics": metrics,
            "fallback_guidance": [],
        }
    runtime_feature_supported = any(
        bool((feature_map.get(feature_name) or {}).get("supported"))
        for feature_name in ("speculative_decoding", "prompt_lookup_speculation")
    )
    if runtime_feature_supported or "speculation" in resolved_classes:
        return {
            "status": "supported_unproved",
            "reason": "Speculation support is present in runtime health or defaults, but benchmark-backed proof is not yet recorded in the release manifest.",
            "benchmark_backed": "speculation" in benchmark_backed_classes,
            "runtime_names": runtime_names,
            "feature_names": ["speculative_decoding", "prompt_lookup_speculation"],
            "profile_ids": [],
            "artifact_ids": [],
            "artifact_paths": [],
            "metrics": metrics,
            "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE["speculation"],
        }
    return {
        "status": "missing",
        "reason": "No benchmark artifact or persisted serving profile currently proves speculation on this host.",
        "benchmark_backed": False,
        "runtime_names": runtime_names,
        "feature_names": ["speculative_decoding", "prompt_lookup_speculation"],
        "profile_ids": [],
        "artifact_ids": [],
        "artifact_paths": [],
        "metrics": metrics,
        "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE["speculation"],
    }


def _constrained_decoding_entry(
    *,
    benchmark_artifacts: list[dict[str, Any]],
    serving_profiles: list[dict[str, Any]],
    artifact_by_id: dict[str, dict[str, Any]],
    capability_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    matched_artifacts = _matching_artifacts(
        benchmark_artifacts=benchmark_artifacts,
        feature_names=_CONSTRAINED_DECODING_FEATURE_NAMES,
        scenario_names=("constrained_decoding",),
    )
    matched_profiles = [
        profile
        for profile in serving_profiles
        if _profile_has_constrained_decoding(profile)
        or _profile_matches(
            profile=profile,
            artifact_by_id=artifact_by_id,
            feature_names=_CONSTRAINED_DECODING_FEATURE_NAMES,
            scenario_names=("constrained_decoding",),
        )
    ]
    measured_entries = _measured_capability_entries(
        capability_reports=capability_reports,
        category_name="constrained_decoding",
    )
    measured_statuses = sorted(
        {
            str(entry.get("status"))
            for entry in measured_entries
            if entry.get("status")
        },
    )
    runtime_names = sorted(
        {
            str(runtime_name)
            for entry in measured_entries
            for runtime_name in entry.get("runtime_names", [])
            if runtime_name
        },
    )
    metrics = {
        "artifact_count": len(matched_artifacts),
        "profile_count": len(matched_profiles),
        "measured_report_count": len(measured_entries),
    }
    if matched_artifacts or matched_profiles:
        return {
            "status": "covered",
            "reason": "Release-manifest artifacts record decode-time constrained-output enforcement on this host.",
            "benchmark_backed": True,
            "runtime_names": runtime_names,
            "feature_names": list(_CONSTRAINED_DECODING_FEATURE_NAMES),
            "profile_ids": [str(profile.get("profile_id")) for profile in matched_profiles if profile.get("profile_id")],
            "artifact_ids": [str(artifact.get("artifact_id")) for artifact in matched_artifacts if artifact.get("artifact_id")],
            "artifact_paths": [
                str(artifact.get("artifact_path"))
                for artifact in matched_artifacts
                if artifact.get("artifact_path")
            ],
            "metrics": metrics,
            "fallback_guidance": [],
        }
    if any(status in {"supported", "degraded", "fallback", "mixed"} for status in measured_statuses):
        return {
            "status": "supported_unproved",
            "reason": (
                "Model capability reports record host-side constrained-decoding evidence, but no benchmark artifact "
                "or persisted serving profile currently proves the path in the release manifest."
            ),
            "benchmark_backed": False,
            "runtime_names": runtime_names,
            "feature_names": list(_CONSTRAINED_DECODING_FEATURE_NAMES),
            "profile_ids": [],
            "artifact_ids": [],
            "artifact_paths": [],
            "metrics": metrics | {"measured_statuses": ",".join(measured_statuses)},
            "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE["constrained_decoding"],
        }
    if "rejected" in measured_statuses:
        return {
            "status": "missing",
            "reason": "Host capability reports still reject decode-time constrained decoding for the routed path on this host.",
            "benchmark_backed": False,
            "runtime_names": runtime_names,
            "feature_names": list(_CONSTRAINED_DECODING_FEATURE_NAMES),
            "profile_ids": [],
            "artifact_ids": [],
            "artifact_paths": [],
            "metrics": metrics | {"measured_statuses": ",".join(measured_statuses)},
            "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE["constrained_decoding"],
        }
    return {
        "status": "missing",
        "reason": "No benchmark artifact or persisted serving profile currently records decode-time constrained output enforcement on this host.",
        "benchmark_backed": False,
        "runtime_names": runtime_names,
        "feature_names": list(_CONSTRAINED_DECODING_FEATURE_NAMES),
        "profile_ids": [],
        "artifact_ids": [],
        "artifact_paths": [],
        "metrics": metrics | {"measured_statuses": ",".join(measured_statuses)},
        "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE["constrained_decoding"],
    }


def _measured_registry_defaults_entry(
    *,
    optimization_defaults: dict[str, Any],
    serving_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    resolved_classes = _resolved_classes(optimization_defaults)
    benchmark_backed_classes = _benchmark_backed_classes(optimization_defaults)
    complete = bool(optimization_defaults.get("complete", False))
    selected_profiles = [
        profile
        for profile in serving_profiles
        if str(profile.get("capability") or "") == "chat"
    ]
    metrics = {
        "resolved_class_count": len(resolved_classes),
        "benchmark_backed_class_count": len(benchmark_backed_classes),
        "selected_profile_count": len(selected_profiles),
        "resolved_model_count": int(optimization_defaults.get("resolved_model_count", 0) or 0),
        "model_count": int(optimization_defaults.get("model_count", 0) or 0),
    }
    if complete and benchmark_backed_classes and selected_profiles:
        return {
            "status": "covered",
            "reason": "Optimization defaults are complete and benchmark-backed classes are flowing into persisted serving-profile decisions on this host.",
            "benchmark_backed": True,
            "runtime_names": [],
            "feature_names": [],
            "profile_ids": [str(profile.get("profile_id")) for profile in selected_profiles if profile.get("profile_id")],
            "artifact_ids": [],
            "artifact_paths": [],
            "metrics": metrics | {
                "resolved_classes": ",".join(resolved_classes),
                "benchmark_backed_classes": ",".join(benchmark_backed_classes),
            },
            "fallback_guidance": [],
        }
    if optimization_defaults:
        return {
            "status": "supported_unproved",
            "reason": "Optimization-default reporting exists, but the release manifest does not yet prove complete benchmark-backed default adoption on this host.",
            "benchmark_backed": bool(benchmark_backed_classes),
            "runtime_names": [],
            "feature_names": [],
            "profile_ids": [str(profile.get("profile_id")) for profile in selected_profiles if profile.get("profile_id")],
            "artifact_ids": [],
            "artifact_paths": [],
            "metrics": metrics | {
                "resolved_classes": ",".join(resolved_classes),
                "benchmark_backed_classes": ",".join(benchmark_backed_classes),
            },
            "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE["measured_registry_defaults"],
        }
    return {
        "status": "missing",
        "reason": "No optimization-default summary is available in the release manifest for this host.",
        "benchmark_backed": False,
        "runtime_names": [],
        "feature_names": [],
        "profile_ids": [],
        "artifact_ids": [],
        "artifact_paths": [],
        "metrics": metrics,
        "fallback_guidance": PERFORMANCE_CORE_PILLAR_GUIDANCE["measured_registry_defaults"],
    }


def _runtime_feature_map(runtime_stats: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = runtime_stats.get("performance_features")
    if not isinstance(features, list):
        return {}
    return {
        str(item.get("feature")): item
        for item in features
        if isinstance(item, dict) and item.get("feature")
    }


def _feature_runtime_names(feature: dict[str, Any] | None) -> list[str]:
    if not isinstance(feature, dict):
        return []
    runtime_names = feature.get("runtime_names")
    if not isinstance(runtime_names, list):
        return []
    return [str(item) for item in runtime_names if item]


def _matching_artifacts(
    *,
    benchmark_artifacts: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    scenario_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        artifact
        for artifact in benchmark_artifacts
        if _artifact_matches(artifact, feature_names=feature_names, scenario_names=scenario_names)
    ]


def _matching_profiles(
    *,
    serving_profiles: list[dict[str, Any]],
    artifact_by_id: dict[str, dict[str, Any]],
    feature_names: tuple[str, ...],
    scenario_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        profile
        for profile in serving_profiles
        if _profile_matches(
            profile=profile,
            artifact_by_id=artifact_by_id,
            feature_names=feature_names,
            scenario_names=scenario_names,
        )
    ]


def _artifact_matches(
    artifact: dict[str, Any],
    *,
    feature_names: tuple[str, ...],
    scenario_names: tuple[str, ...],
) -> bool:
    return any(_artifact_has_feature(artifact, feature_name) for feature_name in feature_names) or any(
        _artifact_has_scenario(artifact, scenario_name) for scenario_name in scenario_names
    )


def _profile_matches(
    *,
    profile: dict[str, Any],
    artifact_by_id: dict[str, dict[str, Any]],
    feature_names: tuple[str, ...],
    scenario_names: tuple[str, ...],
) -> bool:
    artifact = _artifact_for_profile(profile, artifact_by_id)
    return isinstance(artifact, dict) and _artifact_matches(
        artifact,
        feature_names=feature_names,
        scenario_names=scenario_names,
    )


def _artifact_for_profile(
    profile: dict[str, Any],
    artifact_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    artifact_ref = profile.get("artifact")
    if not isinstance(artifact_ref, dict):
        return None
    artifact_id = artifact_ref.get("artifact_id")
    if not artifact_id:
        return None
    return artifact_by_id.get(str(artifact_id))


def _artifact_has_feature(artifact: dict[str, Any], feature_name: str) -> bool:
    for feature in artifact.get("performance_features", []):
        if not isinstance(feature, dict):
            continue
        if feature.get("feature") != feature_name:
            continue
        if feature.get("supported") is True or feature.get("active") is True:
            return True
    return False


def _artifact_has_scenario(artifact: dict[str, Any], scenario_name: str) -> bool:
    for scenario in artifact.get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        if scenario.get("scenario") != scenario_name:
            continue
        if scenario.get("status") in {"observed", "selected"}:
            return True
    return False


def _profile_has_constrained_decoding(profile: dict[str, Any]) -> bool:
    effective_settings = profile.get("effective_settings")
    if not isinstance(effective_settings, dict):
        return False
    return any(bool(effective_settings.get(key)) for key in _CONSTRAINED_DECODING_SETTING_KEYS)


def _measured_capability_entries(
    *,
    capability_reports: list[dict[str, Any]],
    category_name: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for report in capability_reports:
        measured_capabilities = report.get("measured_capabilities")
        if not isinstance(measured_capabilities, list):
            continue
        for item in measured_capabilities:
            if not isinstance(item, dict):
                continue
            if item.get("category") != category_name:
                continue
            entries.append(item)
    return entries


def _resolved_classes(optimization_defaults: dict[str, Any]) -> list[str]:
    resolved = optimization_defaults.get("resolved_classes")
    if not isinstance(resolved, list):
        return []
    return sorted(str(item) for item in resolved if item)


def _benchmark_backed_classes(optimization_defaults: dict[str, Any]) -> list[str]:
    benchmark_backed = optimization_defaults.get("benchmark_backed_classes")
    if not isinstance(benchmark_backed, list):
        return []
    return sorted(str(item) for item in benchmark_backed if item)


def _profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    artifact = profile.get("artifact") if isinstance(profile.get("artifact"), dict) else {}
    return {
        "profile_id": profile.get("profile_id"),
        "model_id": profile.get("model_id"),
        "capability": profile.get("capability"),
        "runtime": profile.get("runtime"),
        "workload_class": profile.get("workload_class"),
        "artifact_id": artifact.get("artifact_id"),
        "artifact_path": artifact.get("artifact_path"),
        "selected_speculation_mode": profile.get("selected_speculation_mode"),
    }


def _artifact_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    feature_names = sorted(
        str(feature.get("feature"))
        for feature in artifact.get("performance_features", [])
        if isinstance(feature, dict) and feature.get("feature")
    )
    scenario_names = sorted(
        str(scenario.get("scenario"))
        for scenario in artifact.get("scenarios", [])
        if isinstance(scenario, dict) and scenario.get("scenario")
    )
    return {
        "artifact_id": artifact.get("artifact_id"),
        "artifact_path": artifact.get("artifact_path"),
        "model_id": artifact.get("model_id"),
        "runtime": artifact.get("runtime"),
        "feature_names": feature_names,
        "scenario_names": scenario_names,
    }
