#!/usr/bin/env python3
"""Generate dependency consistency and reproducibility evidence for the current environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"


def build_dependency_audit(*, resolved_packages: list[str] | None = None) -> dict[str, object]:
    dependency_spec = _dependency_spec()
    normalized_packages = sorted(line.strip() for line in (resolved_packages or _pip_freeze()) if line.strip())
    return {
        "format": "lewlm-dependency-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "project_file": {
            "path": str(PYPROJECT_PATH.relative_to(ROOT_DIR)),
            "sha256": _file_sha256(PYPROJECT_PATH),
        },
        "dependency_spec": dependency_spec,
        "compatibility_gates": _dependency_compatibility_gates(dependency_spec),
        "resolved_environment": {
            "package_count": len(normalized_packages),
            "package_digest": _sha256_lines(normalized_packages),
        },
        "consistency_check": _pip_check(),
    }


def _dependency_spec() -> dict[str, object]:
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    dependencies = list(project.get("dependencies", []))
    optional_groups = {
        name: list(requirements)
        for name, requirements in project.get("optional-dependencies", {}).items()
    }
    canonical_payload = {
        "dependencies": dependencies,
        "optional_groups": optional_groups,
    }
    return {
        **canonical_payload,
        "digest": _sha256_json(canonical_payload),
    }


def _dependency_compatibility_gates(dependency_spec: dict[str, object]) -> dict[str, object]:
    optional_groups = dependency_spec.get("optional_groups", {})
    if not isinstance(optional_groups, dict):
        optional_groups = {}
    mlx_requirements = _optional_group_requirements(optional_groups, "mlx")
    llamacpp_requirements = _optional_group_requirements(optional_groups, "llamacpp")
    documents_requirements = _optional_group_requirements(optional_groups, "documents")

    return {
        "format": "lewlm-dependency-compatibility-gates-v1",
        "classifications": ["required", "optional", "bridge_owned", "unsupported", "watchlisted"],
        "gates": {
            "transformers_v5_ready": {
                "classification": "watchlisted",
                "summary": "Transformers v5 is not a core LewLM dependency; compatibility stays watchlisted until a packaged or explicitly validated path depends on it.",
                "requirements": [],
            },
            "cuda13_ready": {
                "classification": "watchlisted",
                "summary": "CUDA 13 readiness depends on optional packaged or bridge-owned runtimes and remains watchlisted instead of implied by the core install.",
                "requirements": [],
            },
            "pytorch211_ready": {
                "classification": "watchlisted",
                "summary": "PyTorch 2.11 is not a base-package requirement and remains a watchlisted compatibility gate.",
                "requirements": [],
            },
            "cxx20_ready": {
                "classification": "watchlisted",
                "summary": "C++20-sensitive runtime expectations remain watchlisted because LewLM does not bundle platform toolchains or binaries directly.",
                "requirements": llamacpp_requirements,
            },
            "mlx_031_plus": {
                "classification": "watchlisted",
                "summary": "The MLX packaged path is Apple-first, but the current Python extra still declares older minimums than the 0.31+ watch target, so the gate stays explicit and watchlisted.",
                "requirements": mlx_requirements,
            },
            "llama_cpp_python_bindings": {
                "classification": "optional",
                "summary": "llama.cpp Python bindings remain an optional packaged runtime dependency rather than a core install requirement.",
                "requirements": llamacpp_requirements,
            },
            "document_tooling": {
                "classification": "optional",
                "summary": "Document parsing and OCR-side Python packages stay isolated in the optional documents extra.",
                "requirements": documents_requirements,
            },
            "optional_bridge_clients": {
                "classification": "bridge_owned",
                "summary": "Loopback bridge compatibility depends on the external local server contract rather than a mandatory LewLM runtime dependency set.",
                "requirements": [],
            },
        },
        "notes": [
            "Compatibility gates classify 2026 dependency expectations without forcing heavyweight runtime packages into the core install.",
            "Watchlisted gates remain visible until host proof, package-baseline updates, or stronger packaged evidence exists.",
        ],
    }


def _optional_group_requirements(optional_groups: dict[str, object], group_name: str) -> list[str]:
    requirements = optional_groups.get(group_name, [])
    if not isinstance(requirements, list):
        return []
    return [str(requirement) for requirement in requirements if requirement]


def _pip_check() -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    issues = [
        line.strip()
        for line in (completed.stdout.splitlines() + completed.stderr.splitlines())
        if line.strip()
    ]
    return {
        "tool": "pip check",
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "issues": issues,
    }


def _pip_freeze() -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _sha256_json(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    print(json.dumps(build_dependency_audit(), indent=2))
