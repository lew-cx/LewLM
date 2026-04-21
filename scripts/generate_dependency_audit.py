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
