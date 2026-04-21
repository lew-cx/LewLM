#!/usr/bin/env python3
"""Generate a reproducible LewLM release manifest for the current host."""

from __future__ import annotations

import asyncio
import json
import platform
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from frontier_acceptance import build_frontier_acceptance_summary
from performance_core_acceptance import build_performance_core_acceptance_summary
from generate_dependency_audit import build_dependency_audit
from generate_sbom import build_sbom


def build_release_manifest() -> dict[str, object]:
    settings = LewLMSettings()
    services = bootstrap_services(settings)
    runtime_stats = asyncio.run(services.telemetry_service.runtime_stats())
    inventory = services.model_registry.inventory()
    pip_freeze = _pip_freeze()
    capability_reports = [
        services.model_router.model_capability_report(manifest.model_id).model_dump(mode="json")
        for manifest in inventory.items
    ]
    benchmark_artifacts = services.metadata_store.list_benchmark_artifacts(limit=50)
    serving_profiles = services.metadata_store.list_serving_profiles(limit=50)
    runtime_preferences = services.metadata_store.list_runtime_preferences(limit=50)
    return {
        "format": "lewlm-release-manifest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git_commit": _git_commit(),
        "pip_freeze": pip_freeze,
        "dependency_audit": build_dependency_audit(resolved_packages=pip_freeze),
        "configuration": settings.redacted_snapshot(),
        "registered_model_count": inventory.count,
        "registered_models": capability_reports,
        "benchmark_artifacts": benchmark_artifacts,
        "serving_profiles": serving_profiles,
        "runtime_preferences": runtime_preferences,
        "optimization_defaults": runtime_stats.optimization_defaults.model_dump(mode="json")
        if runtime_stats.optimization_defaults is not None
        else None,
        "frontier_acceptance": build_frontier_acceptance_summary(
            capability_reports=capability_reports,
            benchmark_artifacts=benchmark_artifacts,
            serving_profiles=serving_profiles,
            host_platform=runtime_stats.platform.model_dump(mode="json"),
        ),
        "performance_core_acceptance": build_performance_core_acceptance_summary(
            runtime_stats=runtime_stats.model_dump(mode="json"),
            benchmark_artifacts=benchmark_artifacts,
            serving_profiles=serving_profiles,
            optimization_defaults=(
                runtime_stats.optimization_defaults.model_dump(mode="json")
                if runtime_stats.optimization_defaults is not None
                else None
            ),
            capability_reports=capability_reports,
        ),
        "runtime_stats": runtime_stats.model_dump(mode="json"),
        "sbom": build_sbom(),
    }


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    return commit or None


def _pip_freeze() -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


if __name__ == "__main__":
    print(json.dumps(build_release_manifest(), indent=2))
