"""Inject concise LewLM workspace context at the start of agent sessions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run_git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip()


def project_summary(root: Path) -> str:
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return "Python project context unavailable."

    name = "unknown"
    version = "unknown"
    requires_python = "unknown"
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("name = "):
            name = stripped.split("=", 1)[1].strip().strip('"')
        elif stripped.startswith("version = "):
            version = stripped.split("=", 1)[1].strip().strip('"')
        elif stripped.startswith("requires-python = "):
            requires_python = stripped.split("=", 1)[1].strip().strip('"')

    return f"{name} {version}, Python {requires_python}"


def main() -> int:
    root = Path.cwd()
    branch = run_git(["branch", "--show-current"]) or "unknown branch"
    changed = [line for line in run_git(["status", "--short"]).splitlines() if line]
    dirty_text = "clean working tree" if not changed else f"{len(changed)} changed file(s)"

    message = (
        f"LewLM workspace context: {project_summary(root)} on {branch} with {dirty_text}. "
        "Preserve local-first behavior, avoid committing model weights/secrets/machine paths, "
        "prefer targeted `python -m pytest <path> -q`, and treat long_running tests as opt-in."
    )

    print(json.dumps({"continue": True, "systemMessage": message}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
