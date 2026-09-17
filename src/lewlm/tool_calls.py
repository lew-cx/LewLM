"""Strict, schema-validated extraction of model-emitted tool calls.

Middleware-owned (``lewlm_owned``) implementation of the Milestone 120
``strict_tool_parser`` and ``parallel_tool_calls`` vocabulary terms. Tool
calls are only reported when a candidate parses strictly and validates
against the declared tool's ``input_schema``. Malformed candidates surface
as explicit issues — LewLM never silently repairs model output, and partial
success is labeled ``partial`` instead of being flattened into success.

Accepted candidate shapes (inside a ```` ```json ```` fenced block or as the
entire output text):

- ``{"tool_call": {"name": ..., "arguments": {...}}}``
- ``{"tool_calls": [{"name": ..., "arguments": {...}}, ...]}``
- ``{"name": ..., "arguments": {...}}``

``"input"`` is accepted as an alias for ``"arguments"`` because prompt tool
declarations advertise ``input_schema``.

The accepted keys live in :mod:`lewlm.tool_call_contract`, which the prompt
compiler also uses to tell the model what to emit, so the two halves cannot
drift apart.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from lewlm.prompting import PromptToolDefinition
from lewlm.structured_output import validate_value_against_json_schema
from lewlm.tool_call_contract import (
    ARGUMENT_KEYS,
    TOOL_CALL_BATCH_KEY,
    TOOL_CALL_MARKER_KEYS,
    TOOL_CALL_NAME_KEY,
    TOOL_CALL_SINGLE_KEY,
    UNRECOGNIZED_SHAPE_MESSAGE,
    tool_call_invocation_instructions,
)

STRICT_TOOL_PARSER_NAME = "lewlm_strict_tool_parser"

_JSON_FENCE_PATTERN = re.compile(r"```(?P<language>[A-Za-z0-9_-]*)[ \t]*\r?\n(?P<body>.*?)```", re.DOTALL)

_ARGUMENT_KEYS = ARGUMENT_KEYS


class ParsedToolCall(BaseModel):
    """One strictly parsed and schema-validated tool call."""

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallParseIssue(BaseModel):
    """One explicit reason a tool-call candidate was not accepted."""

    code: Literal[
        "invalid_json",
        "unrecognized_shape",
        "missing_name",
        "unknown_tool",
        "arguments_not_object",
        "schema_violation",
    ]
    message: str
    candidate_index: int


class ToolCallParseResult(BaseModel):
    """Strict parse outcome for one model output text."""

    status: Literal["no_tool_calls", "parsed", "partial", "failed"]
    parser: str = STRICT_TOOL_PARSER_NAME
    tool_calls: list[ParsedToolCall] = Field(default_factory=list)
    issues: list[ToolCallParseIssue] = Field(default_factory=list)
    remaining_text: str = ""
    parallel: bool = False


def parse_tool_calls(
    output_text: str,
    *,
    tools: Sequence[PromptToolDefinition],
) -> ToolCallParseResult:
    """Extract tool calls from model output with strict validation.

    Returns ``no_tool_calls`` when no candidate is found, ``parsed`` when every
    candidate validated, ``partial`` when only some did, and ``failed`` when
    candidates existed but none validated.
    """

    tool_index = {tool.name: tool for tool in tools}
    candidates, remaining_text = _extract_candidates(output_text)
    if not candidates:
        return ToolCallParseResult(status="no_tool_calls", remaining_text=output_text)

    tool_calls: list[ParsedToolCall] = []
    issues: list[ToolCallParseIssue] = []
    for candidate_index, candidate_text in enumerate(candidates):
        try:
            payload = json.loads(candidate_text)
        except json.JSONDecodeError as exc:
            issues.append(
                ToolCallParseIssue(
                    code="invalid_json",
                    message=f"Candidate is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno}).",
                    candidate_index=candidate_index,
                ),
            )
            continue
        call_payloads, shape_issue = _normalize_candidate_shape(payload)
        if shape_issue is not None:
            issues.append(
                ToolCallParseIssue(
                    code="unrecognized_shape",
                    message=shape_issue,
                    candidate_index=candidate_index,
                ),
            )
            continue
        for call_payload in call_payloads:
            parsed_call, call_issue = _validate_call_payload(
                call_payload,
                tool_index=tool_index,
                candidate_index=candidate_index,
                call_number=len(tool_calls) + 1,
            )
            if call_issue is not None:
                issues.append(call_issue)
            if parsed_call is not None:
                tool_calls.append(parsed_call)

    if tool_calls and not issues:
        status: Literal["parsed", "partial", "failed"] = "parsed"
    elif tool_calls:
        status = "partial"
    else:
        status = "failed"
    return ToolCallParseResult(
        status=status,
        tool_calls=tool_calls,
        issues=issues,
        remaining_text=remaining_text,
        parallel=len(tool_calls) > 1,
    )


def _extract_candidates(output_text: str) -> tuple[list[str], str]:
    """Return candidate JSON texts and the output text without consumed spans."""

    candidates: list[str] = []
    consumed_spans: list[tuple[int, int]] = []
    for match in _JSON_FENCE_PATTERN.finditer(output_text):
        language = match.group("language").casefold()
        body = match.group("body").strip()
        if language and language != "json":
            continue
        if language != "json" and not _looks_like_tool_call_json(body):
            continue
        candidates.append(body)
        consumed_spans.append(match.span())
    if not candidates:
        stripped = output_text.strip()
        if _looks_like_tool_call_json(stripped):
            return [stripped], ""
    remaining = _remove_spans(output_text, consumed_spans).strip()
    return candidates, remaining


def _looks_like_tool_call_json(text: str) -> bool:
    if not (text.startswith("{") and text.endswith("}")):
        return False
    return any(f'"{marker}"' in text for marker in TOOL_CALL_MARKER_KEYS)


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _normalize_candidate_shape(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(payload, dict):
        return [], UNRECOGNIZED_SHAPE_MESSAGE
    if TOOL_CALL_BATCH_KEY in payload:
        calls = payload[TOOL_CALL_BATCH_KEY]
        if not isinstance(calls, list) or not calls:
            return [], f"`{TOOL_CALL_BATCH_KEY}` must be a non-empty array of tool-call objects."
        non_objects = [item for item in calls if not isinstance(item, dict)]
        if non_objects:
            return [], f"`{TOOL_CALL_BATCH_KEY}` entries must all be objects."
        return list(calls), None
    if TOOL_CALL_SINGLE_KEY in payload:
        call = payload[TOOL_CALL_SINGLE_KEY]
        if not isinstance(call, dict):
            return [], f"`{TOOL_CALL_SINGLE_KEY}` must be an object."
        return [call], None
    if TOOL_CALL_NAME_KEY in payload:
        return [payload], None
    return [], UNRECOGNIZED_SHAPE_MESSAGE


def _validate_call_payload(
    call_payload: dict[str, Any],
    *,
    tool_index: dict[str, PromptToolDefinition],
    candidate_index: int,
    call_number: int,
) -> tuple[ParsedToolCall | None, ToolCallParseIssue | None]:
    name = call_payload.get(TOOL_CALL_NAME_KEY)
    if not isinstance(name, str) or not name.strip():
        return None, ToolCallParseIssue(
            code="missing_name",
            message=f"Tool call is missing a non-empty string `{TOOL_CALL_NAME_KEY}`.",
            candidate_index=candidate_index,
        )
    tool = tool_index.get(name)
    if tool is None:
        declared = ", ".join(sorted(tool_index)) or "none"
        return None, ToolCallParseIssue(
            code="unknown_tool",
            message=f"Tool `{name}` is not declared for this request (declared tools: {declared}).",
            candidate_index=candidate_index,
        )
    arguments: Any = {}
    for key in _ARGUMENT_KEYS:
        if key in call_payload:
            arguments = call_payload[key]
            break
    if not isinstance(arguments, dict):
        return None, ToolCallParseIssue(
            code="arguments_not_object",
            message=f"Tool `{name}` arguments must be a JSON object.",
            candidate_index=candidate_index,
        )
    if tool.input_schema:
        schema_issues = validate_value_against_json_schema(schema=tool.input_schema, value=arguments)
        if schema_issues:
            details = "; ".join(
                f"{'/'.join(str(part) for part in issue.path) or '<root>'}: {issue.message}"
                for issue in schema_issues
            )
            return None, ToolCallParseIssue(
                code="schema_violation",
                message=f"Tool `{name}` arguments failed input-schema validation: {details}",
                candidate_index=candidate_index,
            )
    call_id = call_payload.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{call_number}"
    return (
        ParsedToolCall(call_id=call_id, name=name, arguments=arguments),
        None,
    )


__all__ = [
    "STRICT_TOOL_PARSER_NAME",
    "ParsedToolCall",
    "ToolCallParseIssue",
    "ToolCallParseResult",
    "parse_tool_calls",
]
