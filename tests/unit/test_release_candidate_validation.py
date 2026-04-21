from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_release_validation_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "validate_release_candidate.py"
    spec = spec_from_file_location("lewlm_release_candidate_validation", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _release_manifest(
    *,
    system: str,
    machine: str,
    git_commit: str,
    spec_digest: str,
    audit_status: str = "passed",
    verified_model_count: int = 2,
    covered_families: list[str] | None = None,
    resolved_optimization_classes: list[str] | None = None,
    covered_performance_core_pillars: list[str] | None = None,
) -> dict[str, object]:
    registered_models = []
    for index in range(2):
        verified = index < verified_model_count
        registered_models.append(
            {
                "model_id": f"{system.lower()}-model-{index}",
                "validation_key": f"{system.lower()}-key-{index}",
                "target_platforms": [
                    {
                        "system": system,
                        "machine": machine,
                        "supported": verified,
                        "readiness_state": "verified" if verified else "declared",
                        "verification_method": "host_probe" if verified else "runtime_contract",
                    },
                ],
            },
        )
    return {
        "format": "lewlm-release-manifest-v1",
        "generated_at": "2026-04-15T00:00:00+00:00",
        "git_commit": git_commit,
        "registered_model_count": 2,
        "registered_models": registered_models,
        "platform": {
            "system": system,
            "machine": machine,
            "release": "validated-host",
        },
        "dependency_audit": {
            "format": "lewlm-dependency-audit-v1",
            "dependency_spec": {"digest": spec_digest},
            "resolved_environment": {"package_count": 10, "package_digest": f"{system.lower()}-digest"},
            "consistency_check": {
                "tool": "pip check",
                "status": audit_status,
                "exit_code": 0 if audit_status == "passed" else 1,
                "issues": [] if audit_status == "passed" else ["dependency mismatch"],
            },
        },
        "runtime_stats": {
            "platform": {
                "system": system,
                "machine": machine,
                "release": "validated-host",
                "python_version": "3.14.3",
            },
            "target_platforms": [
                {
                    "system": system,
                    "machine": machine,
                    "readiness_state": "verified",
                    "verification_method": "host_probe",
                },
            ],
        },
        "frontier_acceptance": {
            "format": "lewlm-frontier-acceptance-v1",
            "covered_families": covered_families or [],
            "recommended_profile_count": len(covered_families or []),
            "families": {
                family: {"status": "covered"}
                for family in (covered_families or [])
            },
        },
        "optimization_defaults": {
            "format": "lewlm-optimization-defaults-v1",
            "complete": True,
            "optimization_classes": [
                "runtime_selection",
                "continuous_batching",
                "tiered_kv_cache",
                "speculation",
                "kernel_acceleration",
                "precision_profile",
                "frontier_execution",
                "multimodal_default_selection",
            ],
            "resolved_classes": resolved_optimization_classes or [],
            "benchmark_backed_classes": resolved_optimization_classes or [],
            "model_count": 2,
            "resolved_model_count": 2,
            "models": [],
        },
        "performance_core_acceptance": {
            "format": "lewlm-performance-core-acceptance-v1",
            "complete": False,
            "covered_pillars": covered_performance_core_pillars or [],
            "supported_unproved_pillars": [],
            "missing_pillars": [],
            "pillars": {
                pillar: {"status": "covered"}
                for pillar in (covered_performance_core_pillars or [])
            },
        },
    }


def test_release_candidate_validation_passes_for_consistent_hosts(tmp_path) -> None:
    module = _load_release_validation_module()
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "linux-release-manifest.json").write_text(
        json.dumps(_release_manifest(system="Linux", machine="x86_64", git_commit="abc123", spec_digest="spec-1")),
        encoding="utf-8",
    )
    (output_dir / "windows-release-manifest.json").write_text(
        json.dumps(_release_manifest(system="Windows", machine="AMD64", git_commit="abc123", spec_digest="spec-1")),
        encoding="utf-8",
    )
    (output_dir / "sbom.json").write_text(json.dumps({"format": "lewlm-sbom-v1"}), encoding="utf-8")
    (output_dir / "dependency-audit.json").write_text(
        json.dumps({"format": "lewlm-dependency-audit-v1"}),
        encoding="utf-8",
    )

    payload = module.build_release_candidate_validation(
        [output_dir],
        required_systems=["Linux", "Windows"],
        required_targets=["Linux:x86_64", "Windows:AMD64"],
        minimum_verified_models=1,
    )

    assert payload["overall_status"] == "passed"
    assert payload["manifest_count"] == 2
    assert payload["systems_present"] == ["Linux", "Windows"]
    assert payload["targets_present"] == ["Linux:x86_64", "Windows:AMD64"]
    assert payload["checks"]["git_commit_consistent"]["passed"] is True
    assert payload["checks"]["dependency_spec_consistent"]["passed"] is True
    assert payload["checks"]["required_systems_verified"]["passed"] is True
    assert payload["checks"]["required_targets_verified"]["passed"] is True
    assert payload["checks"]["minimum_verified_models_per_target"]["passed"] is True
    assert payload["checks"]["required_frontier_families_verified"]["passed"] is True
    assert payload["verified_model_coverage"]["Windows:AMD64"]["verified_model_count"] == 2
    assert {item["reason"] for item in payload["skipped_files"]} == {"not_release_manifest"}


def test_release_candidate_validation_detects_drift_missing_target_and_insufficient_models(tmp_path) -> None:
    module = _load_release_validation_module()
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    (manifests_dir / "linux-release-manifest.json").write_text(
        json.dumps(_release_manifest(system="Linux", machine="x86_64", git_commit="abc123", spec_digest="spec-1")),
        encoding="utf-8",
    )
    (manifests_dir / "windows-release-manifest.json").write_text(
        json.dumps(
            _release_manifest(
                system="Windows",
                machine="AMD64",
                git_commit="def456",
                spec_digest="spec-1",
                audit_status="failed",
                verified_model_count=1,
            ),
        ),
        encoding="utf-8",
    )

    payload = module.build_release_candidate_validation(
        [manifests_dir],
        required_systems=["Linux", "Windows", "Darwin"],
        required_targets=["Linux:x86_64", "Windows:AMD64", "Darwin:arm64"],
        minimum_verified_models=2,
    )

    assert payload["overall_status"] == "failed"
    assert payload["checks"]["git_commit_consistent"]["passed"] is False
    assert payload["checks"]["dependency_audit_passed"]["passed"] is False
    assert payload["checks"]["required_systems_covered"]["passed"] is False
    assert payload["checks"]["required_systems_verified"]["passed"] is False
    assert payload["checks"]["required_targets_covered"]["passed"] is False
    assert payload["checks"]["minimum_verified_models_per_target"]["passed"] is False
    assert "Darwin" in payload["checks"]["required_systems_covered"]["reason"]
    assert "Darwin:arm64" in payload["checks"]["required_targets_covered"]["reason"]
    assert "Windows:AMD64" in payload["checks"]["minimum_verified_models_per_target"]["reason"]


def test_release_candidate_validation_can_require_frontier_families(tmp_path) -> None:
    module = _load_release_validation_module()
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    (manifests_dir / "darwin-release-manifest.json").write_text(
        json.dumps(
            _release_manifest(
                system="Darwin",
                machine="arm64",
                git_commit="abc123",
                spec_digest="spec-1",
                covered_families=["dense_text", "speculative_family"],
            ),
        ),
        encoding="utf-8",
    )

    payload = module.build_release_candidate_validation(
        [manifests_dir],
        required_targets=["Darwin:arm64"],
        required_frontier_families=["dense_text", "speculative_family"],
    )

    assert payload["overall_status"] == "passed"
    assert payload["checks"]["required_frontier_families_verified"]["passed"] is True
    assert payload["frontier_family_coverage"]["Darwin:arm64"]["covered_families"] == [
        "dense_text",
        "speculative_family",
    ]


def test_release_candidate_validation_can_require_optimization_classes(tmp_path) -> None:
    module = _load_release_validation_module()
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    (manifests_dir / "darwin-release-manifest.json").write_text(
        json.dumps(
            _release_manifest(
                system="Darwin",
                machine="arm64",
                git_commit="abc123",
                spec_digest="spec-1",
                resolved_optimization_classes=[
                    "runtime_selection",
                    "continuous_batching",
                    "kernel_acceleration",
                    "multimodal_default_selection",
                ],
            ),
        ),
        encoding="utf-8",
    )

    payload = module.build_release_candidate_validation(
        [manifests_dir],
        required_targets=["Darwin:arm64"],
        required_optimization_classes=["runtime_selection", "continuous_batching", "multimodal_default_selection"],
    )

    assert payload["overall_status"] == "passed"
    assert payload["checks"]["required_optimization_classes_resolved"]["passed"] is True
    assert payload["optimization_class_coverage"]["Darwin:arm64"]["resolved_classes"] == [
        "continuous_batching",
        "kernel_acceleration",
        "multimodal_default_selection",
        "runtime_selection",
    ]


def test_release_candidate_validation_can_require_performance_core_pillars(tmp_path) -> None:
    module = _load_release_validation_module()
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    (manifests_dir / "darwin-release-manifest.json").write_text(
        json.dumps(
            _release_manifest(
                system="Darwin",
                machine="arm64",
                git_commit="abc123",
                spec_digest="spec-1",
                covered_performance_core_pillars=[
                    "serving_core",
                    "continuous_batching",
                    "measured_registry_defaults",
                ],
            ),
        ),
        encoding="utf-8",
    )

    payload = module.build_release_candidate_validation(
        [manifests_dir],
        required_targets=["Darwin:arm64"],
        required_performance_core_pillars=["serving_core", "measured_registry_defaults"],
    )

    assert payload["overall_status"] == "passed"
    assert payload["checks"]["required_performance_core_pillars_verified"]["passed"] is True
    assert payload["performance_core_pillar_coverage"]["Darwin:arm64"]["covered_pillars"] == [
        "continuous_batching",
        "measured_registry_defaults",
        "serving_core",
    ]


def test_release_candidate_validation_reports_missing_frontier_family(tmp_path) -> None:
    module = _load_release_validation_module()
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    (manifests_dir / "darwin-release-manifest.json").write_text(
        json.dumps(
            _release_manifest(
                system="Darwin",
                machine="arm64",
                git_commit="abc123",
                spec_digest="spec-1",
                covered_families=["dense_text"],
            ),
        ),
        encoding="utf-8",
    )

    payload = module.build_release_candidate_validation(
        [manifests_dir],
        required_targets=["Darwin:arm64"],
        required_frontier_families=["dense_text", "vlm"],
    )

    assert payload["overall_status"] == "failed"
    assert payload["checks"]["required_frontier_families_verified"]["passed"] is False
    assert "Darwin:arm64:vlm" in payload["checks"]["required_frontier_families_verified"]["reason"]


def test_release_candidate_validation_reports_missing_optimization_class(tmp_path) -> None:
    module = _load_release_validation_module()
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    (manifests_dir / "darwin-release-manifest.json").write_text(
        json.dumps(
            _release_manifest(
                system="Darwin",
                machine="arm64",
                git_commit="abc123",
                spec_digest="spec-1",
                resolved_optimization_classes=["runtime_selection"],
            ),
        ),
        encoding="utf-8",
    )

    payload = module.build_release_candidate_validation(
        [manifests_dir],
        required_targets=["Darwin:arm64"],
        required_optimization_classes=["runtime_selection", "kernel_acceleration"],
    )

    assert payload["overall_status"] == "failed"
    assert payload["checks"]["required_optimization_classes_resolved"]["passed"] is False
    assert payload["checks"]["required_optimization_classes_resolved"]["failed_pairs"] == [
        "Darwin:arm64:kernel_acceleration",
    ]
