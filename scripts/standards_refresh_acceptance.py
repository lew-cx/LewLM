#!/usr/bin/env python3
"""Build a machine-readable completion summary for the 2026 standards-refresh branch."""

from __future__ import annotations

from typing import Any


REQUIRED_STANDARDS_REFRESH_MILESTONES = tuple(range(121, 133))


def build_standards_refresh_acceptance_summary(
    *,
    host_platform: dict[str, Any] | None,
    install_profiles: dict[str, Any] | None,
    dependency_audit: dict[str, Any] | None,
) -> dict[str, object]:
    host_platform = host_platform if isinstance(host_platform, dict) else {}
    install_profiles = install_profiles if isinstance(install_profiles, dict) else {}
    dependency_audit = dependency_audit if isinstance(dependency_audit, dict) else {}

    system = str(host_platform.get("system", ""))
    machine = str(host_platform.get("machine", ""))
    host_target = f"{system}:{machine}" if system and machine else "unknown"
    dependency_gates = dependency_audit.get("compatibility_gates", {})
    gate_entries = dependency_gates.get("gates", {}) if isinstance(dependency_gates, dict) else {}

    return {
        "format": "lewlm-standards-refresh-acceptance-v1",
        "scope": "2026_standards_refresh",
        "host_target": host_target,
        "completion_mode": "explicit_support_state_reporting",
        "required_milestones": list(REQUIRED_STANDARDS_REFRESH_MILESTONES),
        "completed_milestones": list(REQUIRED_STANDARDS_REFRESH_MILESTONES),
        "milestones": [
            _milestone(
                milestone=121,
                title="Bridge Probe Matrix for Current Local Servers",
                machine_readable_surfaces=[
                    "install_profiles.profiles[]",
                    "registered_models[].runtime_candidates[]",
                    "runtime_stats.performance_features[]",
                    "runtime_stats.target_platforms[]",
                ],
                required_manifest_sections=["install_profiles", "registered_models", "runtime_stats"],
                validation_surfaces=[
                    "tests/unit/test_external_adapter_runtime.py",
                    "tests/unit/test_install_profiles.py",
                ],
                notes=[
                    "Loopback bridge reporting stays explicit about packaged-versus-bridge ownership.",
                    "Profile coverage includes generic OpenAI-compatible servers plus Apple-local, vLLM, SGLang, Ollama-compatible, and llama.cpp-server-compatible aliases when the loopback endpoint satisfies the same local contract.",
                ],
            ),
            _milestone(
                milestone=122,
                title="GGUF and llama.cpp 2026 Packaged Path Refresh",
                machine_readable_surfaces=[
                    "install_profiles.profiles[]",
                    "registered_models[].runtime_candidates[]",
                    "runtime_stats.optimization_defaults",
                    "benchmark_artifacts[]",
                ],
                required_manifest_sections=["install_profiles", "registered_models", "runtime_stats", "benchmark_artifacts"],
                validation_surfaces=[
                    "tests/unit/test_llamacpp_runtime.py",
                    "tests/unit/test_llamacpp_structured_output.py",
                    "tests/integration/test_operations.py",
                ],
                notes=[
                    "GGUF remains the first-class packaged non-Apple runtime family.",
                    "Platform-specific readiness that LewLM does not own remains explicit through fallback, unsupported, or unverified reporting instead of being promoted to packaged parity claims.",
                ],
            ),
            _milestone(
                milestone=123,
                title="MLX 0.31+ Apple Path Refresh",
                machine_readable_surfaces=[
                    "install_profiles.profiles[]",
                    "registered_models[].runtime_candidates[]",
                    "runtime_stats.performance_features[]",
                    "runtime_stats.target_platforms[]",
                ],
                required_manifest_sections=["install_profiles", "registered_models", "runtime_stats"],
                validation_surfaces=[
                    "tests/unit/test_mlx_text_runtime.py",
                    "tests/unit/test_mlx_vision_runtime.py",
                    "tests/unit/test_mlx_audio_runtime.py",
                ],
                notes=[
                    "Apple Silicon remains the strongest LewLM-owned optimization path.",
                    "MLX package-baseline drift is tracked through dependency compatibility gates instead of being hidden behind a blanket ready claim.",
                ],
            ),
            _milestone(
                milestone=124,
                title="Strict Structured Output, Tool Calls, and Reasoning Tags",
                machine_readable_surfaces=[
                    "registered_models[].capabilities[]",
                    "registered_models[].runtime_support_strategy",
                    "runtime_stats.performance_features[]",
                    "runtime_stats.standards_acceptance_contract",
                ],
                required_manifest_sections=["registered_models", "runtime_stats"],
                validation_surfaces=[
                    "tests/unit/test_reasoning.py",
                    "tests/unit/test_llamacpp_structured_output.py",
                    "tests/integration/test_api.py",
                ],
                notes=[
                    "Structured-output and reasoning reporting preserves enforced, fallback, partial, unsupported, and unverified boundaries.",
                    "LewLM keeps the middleware contract stable without turning tool execution into a workflow engine.",
                ],
            ),
            _milestone(
                milestone=125,
                title="KV Offload, Hybrid Memory, and Long-Context Evidence",
                machine_readable_surfaces=[
                    "runtime_stats.performance_features[]",
                    "runtime_stats.serving_core",
                    "serving_profiles[]",
                    "benchmark_artifacts[]",
                ],
                required_manifest_sections=["runtime_stats", "serving_profiles", "benchmark_artifacts"],
                validation_surfaces=[
                    "tests/unit/test_llamacpp_runtime.py",
                    "tests/unit/test_mlx_text_runtime.py",
                    "tests/integration/test_operations.py",
                ],
                notes=[
                    "Long-context memory features stay explicit about LewLM-owned, backend-native, partial, fallback, unsupported, and unverified states.",
                    "Release proof treats missing host evidence as unverified instead of silently upgrading the claim.",
                ],
            ),
            _milestone(
                milestone=126,
                title="Speculative Decoding 2026 Refresh",
                machine_readable_surfaces=[
                    "runtime_stats.performance_features[]",
                    "runtime_stats.optimization_defaults",
                    "serving_profiles[]",
                    "benchmark_artifacts[]",
                ],
                required_manifest_sections=["runtime_stats", "serving_profiles", "benchmark_artifacts"],
                validation_surfaces=[
                    "tests/unit/test_speculation.py",
                    "tests/unit/test_mlx_text_runtime.py",
                    "tests/integration/test_speculation_benchmarks.py",
                ],
                notes=[
                    "Speculation remains benchmark-backed where LewLM can measure it and explicitly unverified or unsupported where the runtime boundary stays external.",
                ],
            ),
            _milestone(
                milestone=127,
                title="Multimodal and Document Optional Module Refresh",
                machine_readable_surfaces=[
                    "install_profiles.profiles[]",
                    "registered_models[].capabilities[]",
                    "runtime_stats.target_platforms[]",
                ],
                required_manifest_sections=["install_profiles", "registered_models", "runtime_stats"],
                validation_surfaces=[
                    "tests/unit/test_mlx_vision_runtime.py",
                    "tests/unit/test_mlx_audio_runtime.py",
                    "tests/unit/test_documents.py",
                ],
                notes=[
                    "Documents and multimodal paths remain optional, not mandatory core runtime surfaces.",
                    "Transformer-style OCR readiness stays explicit instead of being implied by the presence of generic OCR tooling.",
                ],
            ),
            _milestone(
                milestone=128,
                title="Semantic Text 2026 Refresh",
                machine_readable_surfaces=[
                    "install_profiles.recommended_feature_paths[]",
                    "registered_models[].capabilities[]",
                    "runtime_stats.target_platforms[]",
                ],
                required_manifest_sections=["install_profiles", "registered_models", "runtime_stats"],
                validation_surfaces=[
                    "tests/unit/test_multimodal.py",
                    "tests/unit/test_external_adapter_runtime.py",
                    "tests/integration/test_operations.py",
                ],
                notes=[
                    "Packaged semantic GGUF defaults and bridge-backed semantic endpoints stay distinct in the operator contract.",
                    "Long-context embedding remains an explicit standards term instead of an implied parity claim.",
                ],
            ),
            _milestone(
                milestone=129,
                title="Agent-Surface Compatibility Without Agent-Framework Drift",
                machine_readable_surfaces=[
                    "configuration",
                    "runtime_stats.standards_acceptance_contract",
                    "install_profiles.notes[]",
                ],
                required_manifest_sections=["configuration", "runtime_stats", "install_profiles"],
                validation_surfaces=[
                    "tests/unit/test_tools.py",
                    "tests/integration/test_integration_bundle.py",
                ],
                notes=[
                    "Sandbox, locality, and deterministic-tooling boundaries stay explicit so LewLM remains a backend contract under the host application.",
                ],
            ),
            _milestone(
                milestone=130,
                title="Dependency Baseline and Optional Pack Gates",
                machine_readable_surfaces=[
                    "dependency_audit.compatibility_gates",
                    "install_profiles.profiles[]",
                    "runtime_stats.standards_acceptance_contract",
                ],
                required_manifest_sections=["dependency_audit.compatibility_gates", "install_profiles", "runtime_stats"],
                validation_surfaces=[
                    "tests/unit/test_dependency_audit.py",
                    "tests/unit/test_install_profiles.py",
                    "tests/unit/test_release_candidate_validation.py",
                ],
                notes=[
                    "Dependency gates classify baseline expectations as required, optional, bridge_owned, unsupported, or watchlisted instead of flattening them into one ready claim.",
                ],
            ),
            _milestone(
                milestone=131,
                title="Real-Host Standards Validation Matrix",
                machine_readable_surfaces=[
                    "runtime_stats.target_platforms[]",
                    "dependency_audit",
                    "standards_refresh_acceptance",
                ],
                required_manifest_sections=["runtime_stats", "dependency_audit", "standards_refresh_acceptance"],
                validation_surfaces=[
                    "tests/unit/test_capture_host_validation.py",
                    "tests/unit/test_release_candidate_validation.py",
                    "tests/unit/test_release_bundle_capture.py",
                ],
                notes=[
                    "Release validation now treats every missing standards-refresh proof as a visible gap instead of an implicit pass.",
                ],
            ),
            _milestone(
                milestone=132,
                title="2026 Standards Release Prove-Out",
                machine_readable_surfaces=[
                    "standards_refresh_acceptance",
                    "install_profiles",
                    "dependency_audit.compatibility_gates",
                    "runtime_stats.standards_acceptance_contract",
                ],
                required_manifest_sections=["standards_refresh_acceptance", "install_profiles", "dependency_audit.compatibility_gates", "runtime_stats"],
                validation_surfaces=[
                    "tests/unit/test_release_manifest.py",
                    "tests/unit/test_release_candidate_validation.py",
                ],
                notes=[
                    "The release manifest includes a branch-level operator summary so packaged, bridge-backed, optional, experimental, unsupported, and unverified states can be reported from one machine-readable artifact.",
                ],
            ),
        ],
        "operator_summary": {
            "current": _recommended_feature_lines(install_profiles),
            "bridge_backed": [
                "OpenAI-compatible loopback bridges remain explicit bridge paths for chat, streaming, semantic, vision, and audio endpoints when the local server exposes those routes.",
                "Bridge wins do not replace LewLM's packaged GGUF default on non-Apple hosts or MLX default on Apple Silicon hosts.",
            ],
            "optional": [
                "Document tooling stays additive through the documents extra.",
                "llama.cpp bindings remain optional instead of inflating the core install.",
                "Agent-surface metadata stays available without turning LewLM into an agent framework.",
            ],
            "experimental": [
                "Frontier and distributed runtimes remain diagnostic or experimental surfaces.",
            ],
            "unsupported_or_unverified": _unsupported_or_unverified_lines(gate_entries),
        },
        "notes": [
            "Standards-refresh completion means LewLM now reports the May 2026 contract, boundaries, and evidence consistently across release artifacts and public docs.",
            "Completion does not imply universal backend parity; unsupported, fallback, partial, and unverified states remain part of the public contract by design.",
        ],
    }


def _milestone(
    *,
    milestone: int,
    title: str,
    machine_readable_surfaces: list[str],
    required_manifest_sections: list[str],
    validation_surfaces: list[str],
    notes: list[str],
) -> dict[str, object]:
    return {
        "milestone": milestone,
        "title": title,
        "status": "completed",
        "machine_readable_surfaces": machine_readable_surfaces,
        "required_manifest_sections": required_manifest_sections,
        "validation_surfaces": validation_surfaces,
        "notes": notes,
    }


def _recommended_feature_lines(install_profiles: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for recommendation in install_profiles.get("recommended_feature_paths", []):
        if not isinstance(recommendation, dict):
            continue
        feature_class = recommendation.get("feature_class")
        summary = recommendation.get("summary")
        if feature_class and summary:
            lines.append(f"{feature_class}: {summary}")
    if lines:
        return lines
    return [
        "chat: use the packaged host-default runtime path when available, and keep bridge fallback explicit when LewLM is fronting a loopback server.",
    ]


def _unsupported_or_unverified_lines(gate_entries: dict[str, Any]) -> list[str]:
    lines = [
        "Non-Apple packaged vision and audio remain unsupported today; those paths stay bridge-backed.",
        "Strict tool parsing, streaming tool calls, and deeper backend-native reasoning metadata remain explicit fallback, partial, unsupported, or unverified claims when the selected runtime cannot prove stronger behavior.",
    ]
    watchlisted = sorted(
        gate_name
        for gate_name, entry in gate_entries.items()
        if isinstance(entry, dict) and entry.get("classification") == "watchlisted"
    )
    if watchlisted:
        lines.append(
            "Watchlisted dependency gates remain explicit until host proof or stronger package baselines exist: "
            + ", ".join(watchlisted)
            + ".",
        )
    return lines