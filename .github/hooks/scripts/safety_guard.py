"""Block clearly destructive tool commands before an agent runs them."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from typing import Any


COMMAND_KEYS = {
    "args",
    "arguments",
    "cmd",
    "command",
    "input",
    "script",
    "shell_command",
}

DENY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
        "Refusing to run git reset --hard without explicit user approval.",
    ),
    (
        re.compile(r"\bgit\s+clean\s+-[^\n]*[fdx][^\n]*[fdx]", re.IGNORECASE),
        "Refusing to run git clean with force/delete flags without explicit user approval.",
    ),
    (
        re.compile(r"\brm\s+-rf\s+(?:/|\*|\.|~|\$HOME|%USERPROFILE%)\b", re.IGNORECASE),
        "Refusing a broad recursive delete command.",
    ),
    (
        re.compile(
            r"\bRemove-Item\b(?=[^\n]*\b-Recurse\b)(?=[^\n]*\b-Force\b)"
            r"(?=[^\n]*(?:\.git|\\\*|/\*|\$HOME|%USERPROFILE%|~))",
            re.IGNORECASE,
        ),
        "Refusing a broad forced recursive PowerShell delete command.",
    ),
    (
        re.compile(
            r"\b(?:curl|iwr|Invoke-WebRequest|wget)\b[^\n]*\|\s*"
            r"(?:sh|bash|pwsh|powershell|iex|Invoke-Expression)\b",
            re.IGNORECASE,
        ),
        "Refusing to pipe downloaded content directly into a shell.",
    ),
)


def iter_command_strings(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key).lower()
            if isinstance(nested, str) and key_text in COMMAND_KEYS:
                yield nested
            elif isinstance(nested, dict | list):
                yield from iter_command_strings(nested)
    elif isinstance(value, list):
        for item in value:
            yield from iter_command_strings(item)


def deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "continue": False,
                "stopReason": reason,
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
            }
        )
    )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    for command in iter_command_strings(payload):
        for pattern, reason in DENY_PATTERNS:
            if pattern.search(command):
                deny(reason)
                return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
