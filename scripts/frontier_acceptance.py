"""Frontier benchmark/profile evidence helpers for release manifests."""

from __future__ import annotations

from typing import Any

FRONTIER_FAMILY_GUIDANCE: dict[str, list[str]] = {
    "dense_text": [
        "Run `lewlm benchmark --model <text-model>` on the primary host, then persist a serving recommendation with `lewlm autotune --model <text-model>`.",
    ],
    "vlm": [
        "Benchmark a vision-capable chat model and persist a serving profile for it on the host you intend to validate.",
    ],
    "repeated_multimodal": [
        "Use a vision-capable chat model with repeated image or attachment prompts until the `multimodal_reuse` benchmark scenario is recorded, then persist a serving profile for that model.",
    ],
    "mixed_precision_conversion": [
        "Convert a model with `--profile activation_aware`, `--profile mixed_precision`, `--profile hybrid_fp8`, or another advanced quantization profile, benchmark it, then autotune the converted model on the target host.",
    ],
    "speculative_family": [
        "Benchmark a model that advertises speculation support until `speculation_selection` is recorded, then persist a serving profile for that model on the target host.",
    ],
    "frontier_architecture": [
        "Benchmark a detected SSM or MoE-family model so `frontier_architecture_modes` evidence is captured, then persist a serving profile for that model.",
    ],
    "distributed_multi_host": [
        "Enroll at least two ready workers, benchmark a model with `distributed_pipeline.json` via `lewlm cluster benchmark`, and persist a serving profile for that distributed-capable model when the host should claim multi-host proof coverage.",
    ],
}


def build_frontier_acceptance_summary(
    *,
    capability_reports: list[dict[str, Any]],
    benchmark_artifacts: list[dict[str, Any]],
    serving_profiles: list[dict[str, Any]],
    host_platform: dict[str, Any],
) -> dict[str, Any]:
    report_by_model_id = {
        str(report.get("model_id")): report
        for report in capability_reports
        if isinstance(report, dict) and report.get("model_id")
    }
    artifact_by_id = {
        str(artifact.get("artifact_id")): artifact
        for artifact in benchmark_artifacts
        if isinstance(artifact, dict) and artifact.get("artifact_id")
    }
    families = {
        family: _family_entry(
            family=family,
            capability_reports=capability_reports,
            serving_profiles=serving_profiles,
            report_by_model_id=report_by_model_id,
            artifact_by_id=artifact_by_id,
        )
        for family in FRONTIER_FAMILY_GUIDANCE
    }
    covered_families = sorted(
        family
        for family, entry in families.items()
        if entry.get("status") == "covered"
    )
    return {
        "format": "lewlm-frontier-acceptance-v1",
        "host_platform": host_platform,
        "benchmark_artifact_count": len(benchmark_artifacts),
        "recommended_profile_count": len(serving_profiles),
        "covered_family_count": len(covered_families),
        "covered_families": covered_families,
        "families": families,
        "recommended_profiles": [_profile_summary(profile) for profile in serving_profiles[:20]],
        "benchmark_artifacts": [_artifact_summary(artifact) for artifact in benchmark_artifacts[:20]],
    }


def _family_entry(
    *,
    family: str,
    capability_reports: list[dict[str, Any]],
    serving_profiles: list[dict[str, Any]],
    report_by_model_id: dict[str, dict[str, Any]],
    artifact_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for profile in serving_profiles:
        if not isinstance(profile, dict):
            continue
        model_id = str(profile.get("model_id") or "")
        report = report_by_model_id.get(model_id)
        if report is None:
            continue
        artifact_payload = _artifact_for_profile(profile, artifact_by_id)
        if _profile_matches_family(
            family=family,
            profile=profile,
            report=report,
            artifact_payload=artifact_payload,
        ):
            artifact_ref = profile.get("artifact") if isinstance(profile.get("artifact"), dict) else {}
            return {
                "status": "covered",
                "reason": (
                    f"Benchmark-backed serving profile `{profile.get('profile_id')}` covers `{family}` "
                    f"for model `{model_id}` on this host."
                ),
                "model_id": model_id,
                "runtime": profile.get("runtime"),
                "profile_id": profile.get("profile_id"),
                "artifact_id": artifact_ref.get("artifact_id"),
                "artifact_path": artifact_ref.get("artifact_path"),
                "selected_speculation_mode": profile.get("selected_speculation_mode"),
                "quantization_profile": profile.get("quantization_profile"),
                "effective_settings": profile.get("effective_settings", {}),
                "fallback_guidance": [],
            }
    candidate_models = sorted(
        {
            str(report.get("model_id"))
            for report in capability_reports
            if isinstance(report, dict) and _report_matches_family(family=family, report=report)
        },
    )
    reason = f"No benchmark-backed serving profile is currently recorded for `{family}` on this host."
    if candidate_models:
        reason += " Candidate model(s): " + ", ".join(candidate_models[:5]) + "."
    return {
        "status": "missing",
        "reason": reason,
        "model_id": None,
        "runtime": None,
        "profile_id": None,
        "artifact_id": None,
        "artifact_path": None,
        "selected_speculation_mode": None,
        "quantization_profile": None,
        "effective_settings": {},
        "fallback_guidance": FRONTIER_FAMILY_GUIDANCE[family],
    }


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


def _profile_matches_family(
    *,
    family: str,
    profile: dict[str, Any],
    report: dict[str, Any],
    artifact_payload: dict[str, Any] | None,
) -> bool:
    if str(profile.get("capability")) != "chat":
        return False
    if family == "dense_text":
        return _chat_capable(report) and not _is_multimodal_report(report)
    if family == "vlm":
        return _chat_capable(report) and _is_multimodal_report(report)
    if family == "repeated_multimodal":
        return _chat_capable(report) and _is_multimodal_report(report) and _artifact_has_scenario(
            artifact_payload,
            "multimodal_reuse",
        )
    if family == "mixed_precision_conversion":
        return _chat_capable(report) and _is_mixed_precision_report(report)
    if family == "speculative_family":
        return _chat_capable(report) and (
            bool(profile.get("selected_speculation_mode"))
            or _artifact_has_scenario(artifact_payload, "speculation_selection")
        )
    if family == "frontier_architecture":
        return _chat_capable(report) and (
            _frontier_architecture_subtype(report) is not None
            or _artifact_has_scenario(artifact_payload, "frontier_architecture_modes")
        )
    if family == "distributed_multi_host":
        return str(profile.get("runtime")) == "distributed_experimental"
    return False


def _report_matches_family(*, family: str, report: dict[str, Any]) -> bool:
    if family == "dense_text":
        return _chat_capable(report) and not _is_multimodal_report(report)
    if family == "vlm":
        return _chat_capable(report) and _is_multimodal_report(report)
    if family == "repeated_multimodal":
        return _chat_capable(report) and _is_multimodal_report(report)
    if family == "mixed_precision_conversion":
        return _chat_capable(report) and _is_mixed_precision_report(report)
    if family == "speculative_family":
        return _chat_capable(report)
    if family == "frontier_architecture":
        return _chat_capable(report) and _frontier_architecture_subtype(report) is not None
    if family == "distributed_multi_host":
        return any(
            isinstance(candidate, dict) and candidate.get("runtime_affinity") == "distributed_experimental"
            for candidate in report.get("runtime_candidates", [])
        )
    return False


def _chat_capable(report: dict[str, Any]) -> bool:
    return any(
        isinstance(capability, dict)
        and capability.get("capability") == "chat"
        and capability.get("supported") is True
        for capability in report.get("capabilities", [])
    )


def _is_multimodal_report(report: dict[str, Any]) -> bool:
    modalities = {str(item) for item in report.get("modality", [])}
    return bool({"vision", "multimodal"} & modalities)


def _is_mixed_precision_report(report: dict[str, Any]) -> bool:
    profile = report.get("quantization_profile")
    if not isinstance(profile, dict):
        return False
    strategy = str(profile.get("strategy") or "")
    return strategy in {
        "activation_aware",
        "mixed_precision",
        "hybrid_fp8",
        "external_adaptive",
    }


def _frontier_architecture_subtype(report: dict[str, Any]) -> str | None:
    subtype = str(report.get("architecture_subtype") or "")
    if subtype in {"ssm_mamba", "hybrid_ssm", "moe", "hybrid_moe"}:
        return subtype
    return None


def _artifact_has_scenario(artifact_payload: dict[str, Any] | None, scenario_name: str) -> bool:
    if not isinstance(artifact_payload, dict):
        return False
    for scenario in artifact_payload.get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        if scenario.get("scenario") == scenario_name and scenario.get("status") in {"observed", "selected"}:
            return True
    return False


def _profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    artifact = profile.get("artifact") if isinstance(profile.get("artifact"), dict) else {}
    return {
        "profile_id": profile.get("profile_id"),
        "model_id": profile.get("model_id"),
        "capability": profile.get("capability"),
        "runtime": profile.get("runtime"),
        "artifact_id": artifact.get("artifact_id"),
        "artifact_path": artifact.get("artifact_path"),
        "recommended_at": profile.get("recommended_at"),
        "selected_speculation_mode": profile.get("selected_speculation_mode"),
        "quantization_profile": profile.get("quantization_profile"),
    }


def _artifact_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    result = artifact.get("result") if isinstance(artifact.get("result"), dict) else {}
    return {
        "artifact_id": artifact.get("artifact_id"),
        "artifact_path": artifact.get("artifact_path"),
        "capability": artifact.get("capability"),
        "model_ids": artifact.get("model_ids", []),
        "primary_model_id": result.get("model_id"),
        "runtime": result.get("runtime"),
        "created_at": artifact.get("created_at"),
        "scenario_names": [
            str(scenario.get("scenario"))
            for scenario in artifact.get("scenarios", [])
            if isinstance(scenario, dict) and scenario.get("scenario")
        ],
    }
