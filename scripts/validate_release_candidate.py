#!/usr/bin/env python3
"""Validate a LewLM release candidate across one or more host release manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

RELEASE_MANIFEST_FORMAT = "lewlm-release-manifest-v1"


def build_release_candidate_validation(
    paths: list[str | Path],
    *,
    required_systems: list[str] | tuple[str, ...] = (),
    required_targets: list[str] | tuple[str, ...] = (),
    minimum_verified_models: int = 0,
    required_frontier_families: list[str] | tuple[str, ...] = (),
    required_optimization_classes: list[str] | tuple[str, ...] = (),
    required_performance_core_pillars: list[str] | tuple[str, ...] = (),
) -> dict[str, object]:
    manifest_records, skipped_files = _load_release_manifests(paths)
    manifests = [record["payload"] for record in manifest_records]
    required = [system for system in required_systems if system]
    required_target_labels = [_normalize_target_label(target) for target in required_targets if target]
    required_frontier = [family for family in required_frontier_families if family]
    required_optimizations = [item for item in required_optimization_classes if item]
    required_performance_core = [item for item in required_performance_core_pillars if item]
    systems_present = sorted({str(manifest.get("platform", {}).get("system", "")) for manifest in manifests if manifest.get("platform")})
    git_commits = sorted({commit for commit in (manifest.get("git_commit") for manifest in manifests) if commit})
    dependency_spec_digests = sorted(
        {
            digest
            for digest in (
                _dependency_spec_digest(manifest)
                for manifest in manifests
            )
            if digest
        },
    )

    manifest_summaries = [_manifest_summary(record["path"], record["payload"]) for record in manifest_records]
    targets_present = sorted({summary["platform"]["target"] for summary in manifest_summaries})
    verified_model_coverage = _verified_model_coverage(manifest_summaries)
    frontier_family_coverage = _frontier_family_coverage(manifest_summaries)
    optimization_class_coverage = _optimization_class_coverage(manifest_summaries)
    performance_core_pillar_coverage = _performance_core_pillar_coverage(manifest_summaries)
    dependency_failures = [
        summary["path"]
        for summary in manifest_summaries
        if summary["dependency_audit"]["consistency_check_status"] != "passed"
    ]
    missing_dependency_audit = [
        summary["path"]
        for summary in manifest_summaries
        if summary["dependency_audit"]["present"] is False
    ]
    host_verification_failures = [
        summary["path"]
        for summary in manifest_summaries
        if summary["host_platform_validation"]["passed"] is False
    ]
    missing_required_systems = [system for system in required if system not in systems_present]
    unverified_required_systems = [
        system
        for system in required
        if not any(
            summary["platform"]["system"] == system and summary["host_platform_validation"]["passed"]
            for summary in manifest_summaries
        )
    ]
    missing_required_targets = [target for target in required_target_labels if target not in targets_present]
    unverified_required_targets = [
        target
        for target in required_target_labels
        if not any(
            summary["platform"]["target"] == target and summary["host_platform_validation"]["passed"]
            for summary in manifest_summaries
        )
    ]
    targets_to_enforce = required_target_labels or [
        summary["platform"]["target"]
        for summary in manifest_summaries
        if summary["host_platform_validation"]["passed"]
    ]
    missing_required_frontier_pairs = sorted(
        f"{target}:{family}"
        for target in dict.fromkeys(targets_to_enforce)
        for family in required_frontier
        if family not in frontier_family_coverage.get(target, {}).get("covered_families", [])
    )
    missing_required_optimization_pairs = sorted(
        f"{target}:{optimization_class}"
        for target in dict.fromkeys(targets_to_enforce)
        for optimization_class in required_optimizations
        if optimization_class not in optimization_class_coverage.get(target, {}).get("resolved_classes", [])
    )
    missing_required_performance_core_pairs = sorted(
        f"{target}:{pillar}"
        for target in dict.fromkeys(targets_to_enforce)
        for pillar in required_performance_core
        if pillar not in performance_core_pillar_coverage.get(target, {}).get("covered_pillars", [])
    )
    insufficient_verified_model_targets = sorted(
        target
        for target in dict.fromkeys(targets_to_enforce)
        if verified_model_coverage.get(target, {}).get("verified_model_count", 0) < minimum_verified_models
    )

    checks = {
        "manifests_loaded": {
            "passed": bool(manifest_summaries),
            "reason": (
                f"Loaded {len(manifest_summaries)} release manifest(s)."
                if manifest_summaries
                else "No LewLM release manifests were found in the provided paths."
            ),
        },
        "git_commit_consistent": {
            "passed": len(git_commits) <= 1,
            "reason": (
                "All manifests reference the same git commit."
                if len(git_commits) <= 1
                else f"Found multiple git commits: {', '.join(git_commits)}."
            ),
            "values": git_commits,
        },
        "dependency_spec_consistent": {
            "passed": not missing_dependency_audit and len(dependency_spec_digests) <= 1,
            "reason": (
                "All manifests include the same dependency specification digest."
                if not missing_dependency_audit and len(dependency_spec_digests) <= 1
                else (
                    f"Dependency audit missing for {len(missing_dependency_audit)} manifest(s)."
                    if missing_dependency_audit
                    else f"Found multiple dependency spec digests: {', '.join(dependency_spec_digests)}."
                )
            ),
            "values": dependency_spec_digests,
        },
        "dependency_audit_passed": {
            "passed": not dependency_failures and not missing_dependency_audit,
            "reason": (
                "All manifests passed dependency audit checks."
                if not dependency_failures and not missing_dependency_audit
                else (
                    f"Dependency audit missing for {len(missing_dependency_audit)} manifest(s)."
                    if missing_dependency_audit
                    else f"Dependency audit failed for {len(dependency_failures)} manifest(s)."
                )
            ),
            "failed_paths": dependency_failures,
        },
        "required_systems_covered": {
            "passed": not missing_required_systems,
            "reason": (
                "All required systems are represented."
                if not missing_required_systems
                else f"Missing required system manifests: {', '.join(missing_required_systems)}."
            ),
            "required_systems": required,
        },
        "required_systems_verified": {
            "passed": not unverified_required_systems,
            "reason": (
                "All required systems include a host-verified manifest."
                if not unverified_required_systems
                else f"Required systems without host-verified manifests: {', '.join(unverified_required_systems)}."
            ),
            "required_systems": required,
        },
        "required_targets_covered": {
            "passed": not missing_required_targets,
            "reason": (
                "All required targets are represented."
                if not missing_required_targets
                else f"Missing required target manifests: {', '.join(missing_required_targets)}."
            ),
            "required_targets": required_target_labels,
        },
        "required_targets_verified": {
            "passed": not unverified_required_targets,
            "reason": (
                "All required targets include a host-verified manifest."
                if not unverified_required_targets
                else f"Required targets without host-verified manifests: {', '.join(unverified_required_targets)}."
            ),
            "required_targets": required_target_labels,
        },
        "host_platform_verified": {
            "passed": not host_verification_failures,
            "reason": (
                "All manifests include a host-verified target-platform row."
                if not host_verification_failures
                else f"{len(host_verification_failures)} manifest(s) lack a verified host target row."
            ),
            "failed_paths": host_verification_failures,
        },
        "minimum_verified_models_per_target": {
            "passed": (
                minimum_verified_models <= 0
                or (bool(targets_to_enforce) and not insufficient_verified_model_targets)
            ),
            "reason": (
                "Minimum verified-model threshold is disabled."
                if minimum_verified_models <= 0
                else (
                    f"All enforced targets meet the minimum of {minimum_verified_models} verified model(s)."
                    if targets_to_enforce and not insufficient_verified_model_targets
                    else (
                        "No host-verified targets were available for verified-model coverage checks."
                        if not targets_to_enforce
                        else (
                            f"Targets below the minimum of {minimum_verified_models} verified model(s): "
                            f"{', '.join(insufficient_verified_model_targets)}."
                        )
                    )
                )
            ),
            "minimum_verified_models": minimum_verified_models,
            "failed_targets": insufficient_verified_model_targets,
        },
        "required_frontier_families_verified": {
            "passed": (
                not required_frontier
                or (bool(targets_to_enforce) and not missing_required_frontier_pairs)
            ),
            "reason": (
                "Frontier-family validation is disabled."
                if not required_frontier
                else (
                    "All enforced targets include host-backed frontier evidence for the required families."
                    if targets_to_enforce and not missing_required_frontier_pairs
                    else (
                        "No host-verified targets were available for frontier-family coverage checks."
                        if not targets_to_enforce
                        else "Missing required frontier evidence: " + ", ".join(missing_required_frontier_pairs) + "."
                    )
                )
            ),
            "required_frontier_families": required_frontier,
            "failed_pairs": missing_required_frontier_pairs,
        },
        "required_optimization_classes_resolved": {
            "passed": (
                not required_optimizations
                or (bool(targets_to_enforce) and not missing_required_optimization_pairs)
            ),
            "reason": (
                "Optimization-class validation is disabled."
                if not required_optimizations
                else (
                    "All enforced targets include resolved optimization-default decisions for the required classes."
                    if targets_to_enforce and not missing_required_optimization_pairs
                    else (
                        "No host-verified targets were available for optimization-class coverage checks."
                        if not targets_to_enforce
                        else "Missing required optimization-default coverage: "
                        + ", ".join(missing_required_optimization_pairs)
                        + "."
                    )
                )
            ),
            "required_optimization_classes": required_optimizations,
            "failed_pairs": missing_required_optimization_pairs,
        },
        "required_performance_core_pillars_verified": {
            "passed": (
                not required_performance_core
                or (bool(targets_to_enforce) and not missing_required_performance_core_pairs)
            ),
            "reason": (
                "Performance-core pillar validation is disabled."
                if not required_performance_core
                else (
                    "All enforced targets provide covered performance-core pillars for the requested proof set."
                    if targets_to_enforce and not missing_required_performance_core_pairs
                    else (
                        "No host-verified targets were available for performance-core pillar checks."
                        if not targets_to_enforce
                        else "Missing required performance-core coverage: "
                        + ", ".join(missing_required_performance_core_pairs)
                        + "."
                    )
                )
            ),
            "required_performance_core_pillars": required_performance_core,
            "failed_pairs": missing_required_performance_core_pairs,
        },
    }
    overall_passed = all(check["passed"] for check in checks.values())

    return {
        "format": "lewlm-release-candidate-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "required_systems": required,
        "required_targets": required_target_labels,
        "minimum_verified_models": minimum_verified_models,
        "required_frontier_families": required_frontier,
        "required_optimization_classes": required_optimizations,
        "required_performance_core_pillars": required_performance_core,
        "manifest_count": len(manifest_summaries),
        "systems_present": systems_present,
        "targets_present": targets_present,
        "git_commits": git_commits,
        "dependency_spec_digests": dependency_spec_digests,
        "verified_model_coverage": verified_model_coverage,
        "frontier_family_coverage": frontier_family_coverage,
        "optimization_class_coverage": optimization_class_coverage,
        "performance_core_pillar_coverage": performance_core_pillar_coverage,
        "manifests": manifest_summaries,
        "skipped_files": skipped_files,
        "checks": checks,
        "overall_status": "passed" if overall_passed else "failed",
    }


def _load_release_manifests(paths: list[str | Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    manifest_records: list[dict[str, Any]] = []
    skipped_files: list[dict[str, str]] = []
    for candidate in _candidate_paths(paths):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped_files.append({"path": str(candidate), "reason": "invalid_json"})
            continue
        if payload.get("format") != RELEASE_MANIFEST_FORMAT:
            skipped_files.append({"path": str(candidate), "reason": "not_release_manifest"})
            continue
        manifest_records.append({"path": candidate, "payload": payload})
    return manifest_records, skipped_files


def _candidate_paths(paths: list[str | Path]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve(strict=False)
        if not path.exists():
            continue
        if path.is_dir():
            discovered = sorted(item for item in path.rglob("*.json") if item.is_file())
        else:
            discovered = [path]
        for candidate in discovered:
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _manifest_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    platform = payload.get("platform", {})
    dependency_audit = payload.get("dependency_audit", {})
    consistency_check = dependency_audit.get("consistency_check", {})
    system = str(platform.get("system", ""))
    machine = str(platform.get("machine", ""))
    frontier_acceptance = payload.get("frontier_acceptance", {})
    optimization_defaults = payload.get("optimization_defaults", {})
    performance_core_acceptance = payload.get("performance_core_acceptance", {})
    return {
        "path": str(path),
        "platform": {
            "system": system,
            "machine": machine,
            "target": _target_label(system, machine),
            "release": str(platform.get("release", "")),
        },
        "git_commit": payload.get("git_commit"),
        "registered_model_count": int(payload.get("registered_model_count", 0) or 0),
        "dependency_audit": {
            "present": bool(dependency_audit),
            "consistency_check_status": consistency_check.get("status"),
            "dependency_spec_digest": _dependency_spec_digest(payload),
            "resolved_package_digest": dependency_audit.get("resolved_environment", {}).get("package_digest"),
        },
        "host_platform_validation": _host_platform_validation(payload),
        "frontier_acceptance": {
            "present": isinstance(frontier_acceptance, dict) and bool(frontier_acceptance),
            "covered_families": _covered_frontier_families(frontier_acceptance),
            "recommended_profile_count": int(frontier_acceptance.get("recommended_profile_count", 0) or 0)
            if isinstance(frontier_acceptance, dict)
            else 0,
        },
        "optimization_defaults": {
            "present": isinstance(optimization_defaults, dict) and bool(optimization_defaults),
            "complete": bool(optimization_defaults.get("complete", False)) if isinstance(optimization_defaults, dict) else False,
            "resolved_classes": _resolved_optimization_classes(optimization_defaults),
            "benchmark_backed_classes": _benchmark_backed_optimization_classes(optimization_defaults),
            "model_count": int(optimization_defaults.get("model_count", 0) or 0)
            if isinstance(optimization_defaults, dict)
            else 0,
            "resolved_model_count": int(optimization_defaults.get("resolved_model_count", 0) or 0)
            if isinstance(optimization_defaults, dict)
            else 0,
        },
        "performance_core_acceptance": {
            "present": isinstance(performance_core_acceptance, dict) and bool(performance_core_acceptance),
            "complete": (
                bool(performance_core_acceptance.get("complete", False))
                if isinstance(performance_core_acceptance, dict)
                else False
            ),
            "covered_pillars": _covered_performance_core_pillars(performance_core_acceptance),
            "supported_unproved_pillars": _supported_unproved_performance_core_pillars(performance_core_acceptance),
            "missing_pillars": _missing_performance_core_pillars(performance_core_acceptance),
        },
    }


def _dependency_spec_digest(payload: dict[str, Any]) -> str | None:
    dependency_audit = payload.get("dependency_audit", {})
    dependency_spec = dependency_audit.get("dependency_spec", {})
    digest = dependency_spec.get("digest")
    return str(digest) if digest else None


def _host_platform_validation(payload: dict[str, Any]) -> dict[str, Any]:
    platform = payload.get("platform", {})
    system = str(platform.get("system", ""))
    machine = str(platform.get("machine", ""))
    target_platforms = payload.get("runtime_stats", {}).get("target_platforms", [])
    match = next(
        (
            target
            for target in target_platforms
            if str(target.get("system", "")) == system and str(target.get("machine", "")) == machine
        ),
        None,
    )
    if match is None:
        return {
            "passed": False,
            "reason": f"Missing target-platform row for host platform {system} {machine}.",
            "readiness_state": None,
            "verification_method": None,
        }
    readiness_state = match.get("readiness_state")
    verification_method = match.get("verification_method")
    verified_models = _verified_models_for_target(payload, system=system, machine=machine)
    passed = readiness_state == "verified" and verification_method == "host_probe"
    return {
        "passed": passed,
        "reason": (
            f"Host platform {system} {machine} is verified by host probe."
            if passed
            else (
                f"Host platform {system} {machine} is recorded as {readiness_state}/{verification_method}."
            )
        ),
        "readiness_state": readiness_state,
        "verification_method": verification_method,
        "verified_model_count": len(verified_models),
        "verified_model_ids": sorted(model["model_id"] for model in verified_models if model["model_id"]),
        "verified_validation_keys": sorted(
            {
                model["validation_key"]
                for model in verified_models
                if model["validation_key"]
            },
        ),
    }


def _verified_models_for_target(payload: dict[str, Any], *, system: str, machine: str) -> list[dict[str, str | None]]:
    verified_models: list[dict[str, str | None]] = []
    for report in payload.get("registered_models", []):
        if not isinstance(report, dict):
            continue
        target = next(
            (
                item
                for item in report.get("target_platforms", [])
                if str(item.get("system", "")) == system and str(item.get("machine", "")) == machine
            ),
            None,
        )
        if target is None:
            continue
        if target.get("readiness_state") != "verified" or target.get("verification_method") != "host_probe":
            continue
        if target.get("supported") is False:
            continue
        verified_models.append(
            {
                "model_id": str(report.get("model_id")) if report.get("model_id") else None,
                "validation_key": str(report.get("validation_key")) if report.get("validation_key") else None,
            },
        )
    return verified_models


def _verified_model_coverage(manifest_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for summary in manifest_summaries:
        target = summary["platform"]["target"]
        existing = coverage.get(target)
        current_count = summary["host_platform_validation"]["verified_model_count"]
        if existing is None or current_count > existing["verified_model_count"]:
            coverage[target] = {
                "verified_model_count": current_count,
                "verified_model_ids": summary["host_platform_validation"]["verified_model_ids"],
                "verified_validation_keys": summary["host_platform_validation"]["verified_validation_keys"],
            }
    return coverage


def _frontier_family_coverage(manifest_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for summary in manifest_summaries:
        if summary["host_platform_validation"]["passed"] is False:
            continue
        target = summary["platform"]["target"]
        covered_families = list(summary["frontier_acceptance"]["covered_families"])
        existing = coverage.get(target)
        if existing is None or len(covered_families) > len(existing["covered_families"]):
            coverage[target] = {
                "covered_families": covered_families,
                "recommended_profile_count": summary["frontier_acceptance"]["recommended_profile_count"],
                "path": summary["path"],
            }
    return coverage


def _covered_frontier_families(frontier_acceptance: Any) -> list[str]:
    if not isinstance(frontier_acceptance, dict):
        return []
    covered = frontier_acceptance.get("covered_families")
    if isinstance(covered, list):
        return sorted(str(item) for item in covered if item)
    families = frontier_acceptance.get("families", {})
    if not isinstance(families, dict):
        return []
    return sorted(
        str(family)
        for family, entry in families.items()
        if isinstance(entry, dict) and entry.get("status") == "covered"
    )


def _optimization_class_coverage(manifest_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for summary in manifest_summaries:
        if summary["host_platform_validation"]["passed"] is False:
            continue
        target = summary["platform"]["target"]
        resolved_classes = list(summary["optimization_defaults"]["resolved_classes"])
        existing = coverage.get(target)
        if existing is None or len(resolved_classes) > len(existing["resolved_classes"]):
            coverage[target] = {
                "resolved_classes": resolved_classes,
                "benchmark_backed_classes": list(summary["optimization_defaults"]["benchmark_backed_classes"]),
                "complete": bool(summary["optimization_defaults"]["complete"]),
                "model_count": summary["optimization_defaults"]["model_count"],
                "resolved_model_count": summary["optimization_defaults"]["resolved_model_count"],
                "path": summary["path"],
            }
    return coverage


def _performance_core_pillar_coverage(manifest_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for summary in manifest_summaries:
        if summary["host_platform_validation"]["passed"] is False:
            continue
        target = summary["platform"]["target"]
        covered_pillars = list(summary["performance_core_acceptance"]["covered_pillars"])
        existing = coverage.get(target)
        if existing is None or len(covered_pillars) > len(existing["covered_pillars"]):
            coverage[target] = {
                "covered_pillars": covered_pillars,
                "supported_unproved_pillars": list(summary["performance_core_acceptance"]["supported_unproved_pillars"]),
                "missing_pillars": list(summary["performance_core_acceptance"]["missing_pillars"]),
                "complete": bool(summary["performance_core_acceptance"]["complete"]),
                "path": summary["path"],
            }
    return coverage


def _resolved_optimization_classes(optimization_defaults: Any) -> list[str]:
    if not isinstance(optimization_defaults, dict):
        return []
    resolved = optimization_defaults.get("resolved_classes")
    if isinstance(resolved, list):
        return sorted(str(item) for item in resolved if item)
    return []


def _benchmark_backed_optimization_classes(optimization_defaults: Any) -> list[str]:
    if not isinstance(optimization_defaults, dict):
        return []
    benchmark_backed = optimization_defaults.get("benchmark_backed_classes")
    if isinstance(benchmark_backed, list):
        return sorted(str(item) for item in benchmark_backed if item)
    return []


def _covered_performance_core_pillars(performance_core_acceptance: Any) -> list[str]:
    if not isinstance(performance_core_acceptance, dict):
        return []
    covered = performance_core_acceptance.get("covered_pillars")
    if isinstance(covered, list):
        return sorted(str(item) for item in covered if item)
    pillars = performance_core_acceptance.get("pillars", {})
    if not isinstance(pillars, dict):
        return []
    return sorted(
        str(pillar)
        for pillar, entry in pillars.items()
        if isinstance(entry, dict) and entry.get("status") == "covered"
    )


def _supported_unproved_performance_core_pillars(performance_core_acceptance: Any) -> list[str]:
    if not isinstance(performance_core_acceptance, dict):
        return []
    supported_unproved = performance_core_acceptance.get("supported_unproved_pillars")
    if isinstance(supported_unproved, list):
        return sorted(str(item) for item in supported_unproved if item)
    pillars = performance_core_acceptance.get("pillars", {})
    if not isinstance(pillars, dict):
        return []
    return sorted(
        str(pillar)
        for pillar, entry in pillars.items()
        if isinstance(entry, dict) and entry.get("status") == "supported_unproved"
    )


def _missing_performance_core_pillars(performance_core_acceptance: Any) -> list[str]:
    if not isinstance(performance_core_acceptance, dict):
        return []
    missing = performance_core_acceptance.get("missing_pillars")
    if isinstance(missing, list):
        return sorted(str(item) for item in missing if item)
    pillars = performance_core_acceptance.get("pillars", {})
    if not isinstance(pillars, dict):
        return []
    return sorted(
        str(pillar)
        for pillar, entry in pillars.items()
        if isinstance(entry, dict) and entry.get("status") == "missing"
    )


def _target_label(system: str, machine: str) -> str:
    return f"{system}:{machine}"


def _normalize_target_label(target: str) -> str:
    system, separator, machine = str(target).partition(":")
    if not separator or not system or not machine:
        raise ValueError(f"Invalid target '{target}'. Expected format SYSTEM:MACHINE.")
    return _target_label(system, machine)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Release manifest files or directories to validate.")
    parser.add_argument(
        "--require-system",
        action="append",
        default=[],
        dest="required_systems",
        help="Require at least one host-verified manifest for the named platform system.",
    )
    parser.add_argument(
        "--require-target",
        action="append",
        default=[],
        dest="required_targets",
        help="Require at least one host-verified manifest for the exact SYSTEM:MACHINE target pair.",
    )
    parser.add_argument(
        "--minimum-verified-models",
        type=int,
        default=0,
        help="Require each enforced target to include at least this many host-verified registered models.",
    )
    parser.add_argument(
        "--require-frontier-family",
        action="append",
        default=[],
        dest="required_frontier_families",
        help="Require host-backed evidence for the named frontier family on each enforced target.",
    )
    parser.add_argument(
        "--require-optimization-class",
        action="append",
        default=[],
        dest="required_optimization_classes",
        help="Require resolved optimization-default coverage for the named optimization class on each enforced target.",
    )
    parser.add_argument(
        "--require-performance-core-pillar",
        action="append",
        default=[],
        dest="required_performance_core_pillars",
        help="Require covered performance-core proof for the named pillar on each enforced target.",
    )
    args = parser.parse_args(argv)
    try:
        payload = build_release_candidate_validation(
            args.paths,
            required_systems=args.required_systems,
            required_targets=args.required_targets,
            minimum_verified_models=args.minimum_verified_models,
            required_frontier_families=args.required_frontier_families,
            required_optimization_classes=args.required_optimization_classes,
            required_performance_core_pillars=args.required_performance_core_pillars,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
