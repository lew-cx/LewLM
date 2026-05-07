"""Validate workspace AI customization files after edit-style tool calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


EDIT_INDICATORS = (
    "apply_patch",
    "create_or_update",
    "delete_file",
    "edit",
    "push_files",
    "write",
)


def read_hook_payload() -> dict[str, Any]:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def should_validate(payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, sort_keys=True).lower()
    return any(indicator in text for indicator in EDIT_INDICATORS)


def parse_frontmatter(path: Path) -> tuple[dict[str, str], str | None]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, "missing opening frontmatter marker"

    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        return {}, "missing closing frontmatter marker"

    metadata: dict[str, str] = {}
    for line in lines[1:closing_index]:
        if not line.strip() or line.lstrip().startswith("#") or line.startswith((" ", "\t")):
            continue
        if ":" not in line:
            return metadata, f"invalid frontmatter line: {line}"
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, None


def validate_json_hooks(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted((root / ".github" / "hooks").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: invalid JSON ({exc})")
            continue
        if not isinstance(data.get("hooks"), dict):
            errors.append(f"{path}: missing top-level hooks object")
    return errors


def validate_markdown_customizations(root: Path) -> list[str]:
    errors: list[str] = []
    github = root / ".github"

    for path in sorted(github.glob("agents/*.agent.md")):
        metadata, error = parse_frontmatter(path)
        if error:
            errors.append(f"{path}: {error}")
        elif not metadata.get("description"):
            errors.append(f"{path}: missing description")

    for path in sorted(github.glob("instructions/*.instructions.md")):
        metadata, error = parse_frontmatter(path)
        if error:
            errors.append(f"{path}: {error}")
        elif not metadata.get("description"):
            errors.append(f"{path}: missing description")

    for path in sorted(github.glob("skills/*/SKILL.md")):
        metadata, error = parse_frontmatter(path)
        folder_name = path.parent.name
        if error:
            errors.append(f"{path}: {error}")
            continue
        if metadata.get("name") != folder_name:
            errors.append(f"{path}: name must match folder '{folder_name}'")
        if not metadata.get("description"):
            errors.append(f"{path}: missing description")

    return errors


def block(errors: list[str]) -> None:
    summary = "AI customization validation failed:\n" + "\n".join(f"- {error}" for error in errors)
    print(json.dumps({"continue": False, "stopReason": summary, "decision": "block"}))


def main() -> int:
    payload = read_hook_payload()
    if payload and not should_validate(payload):
        return 0

    root = Path.cwd()
    errors = [*validate_json_hooks(root), *validate_markdown_customizations(root)]
    if errors:
        block(errors)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
