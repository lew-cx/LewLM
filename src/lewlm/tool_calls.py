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

Gemma 4 checkpoints ignore that contract and emit their trained-in call syntax,
``<|tool_call>call:NAME{key: value, ...}<tool_call|>``, with bare keys and
``<|"|>`` string delimiters. That form is read as well, strictly: the argument
object must parse completely, and it is then validated like any other call.

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

_GEMMA_NATIVE_CALL_PATTERN = re.compile(
    r"<\|tool_call>\s*call:(?P<name>[^\s{<]+)\s*(?P<arguments>\{.*?\})\s*<tool_call\|>",
    re.DOTALL,
)
_GEMMA_STRING_DELIMITER = '<|"|>'
_GEMMA_BARE_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_GEMMA_SCALAR_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null")
_GEMMA_SCALAR_LITERALS: dict[str, Any] = {"true": True, "false": False, "null": None}


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
    native_candidates, text_without_native = _extract_gemma_native_candidates(output_text)
    json_candidates, remaining_text = _extract_candidates(text_without_native)
    decoded_candidates = native_candidates + [_decode_json_candidate(text) for text in json_candidates]
    if not decoded_candidates:
        return ToolCallParseResult(status="no_tool_calls", remaining_text=output_text)

    tool_calls: list[ParsedToolCall] = []
    issues: list[ToolCallParseIssue] = []
    for candidate_index, (payload, decode_error) in enumerate(decoded_candidates):
        if decode_error is not None:
            issues.append(
                ToolCallParseIssue(
                    code="invalid_json",
                    message=decode_error,
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


def _decode_json_candidate(candidate_text: str) -> tuple[Any, str | None]:
    try:
        return json.loads(candidate_text), None
    except json.JSONDecodeError as exc:
        return None, f"Candidate is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."


def _extract_gemma_native_candidates(output_text: str) -> tuple[list[tuple[Any, str | None]], str]:
    """Decode Gemma 4 ``<|tool_call>call:NAME{...}<tool_call|>`` spans into call payloads."""

    candidates: list[tuple[Any, str | None]] = []
    consumed_spans: list[tuple[int, int]] = []
    for match in _GEMMA_NATIVE_CALL_PATTERN.finditer(output_text):
        consumed_spans.append(match.span())
        name = match.group("name")
        try:
            arguments = _parse_gemma_native_arguments(match.group("arguments"))
        except ValueError as exc:
            candidates.append((None, f"Gemma native tool call `{name}` has unparseable arguments: {exc}"))
            continue
        candidates.append(({TOOL_CALL_NAME_KEY: name, _ARGUMENT_KEYS[0]: arguments}, None))
    return candidates, _remove_spans(output_text, consumed_spans)


def _parse_gemma_native_arguments(text: str) -> Any:
    value, position = _parse_gemma_native_value(text, 0)
    position = _skip_whitespace(text, position)
    if position != len(text):
        raise ValueError(f"unexpected trailing text at offset {position}.")
    return value


def _parse_gemma_native_value(text: str, position: int) -> tuple[Any, int]:
    position = _skip_whitespace(text, position)
    if text.startswith(_GEMMA_STRING_DELIMITER, position):
        start = position + len(_GEMMA_STRING_DELIMITER)
        end = text.find(_GEMMA_STRING_DELIMITER, start)
        if end < 0:
            raise ValueError(f"unterminated string at offset {position}.")
        return text[start:end], end + len(_GEMMA_STRING_DELIMITER)
    if text.startswith('"', position):
        try:
            return json.JSONDecoder().raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid quoted string at offset {position}: {exc.msg}.") from exc
    if text.startswith("{", position):
        return _parse_gemma_native_container(text, position + 1, closing="}", keyed=True)
    if text.startswith("[", position):
        return _parse_gemma_native_container(text, position + 1, closing="]", keyed=False)
    scalar = _GEMMA_SCALAR_PATTERN.match(text, position)
    if scalar is None:
        raise ValueError(f"unexpected value at offset {position}.")
    literal = scalar.group(0)
    if literal in _GEMMA_SCALAR_LITERALS:
        return _GEMMA_SCALAR_LITERALS[literal], scalar.end()
    return json.loads(literal), scalar.end()


def _parse_gemma_native_container(text: str, position: int, *, closing: str, keyed: bool) -> tuple[Any, int]:
    items: dict[str, Any] | list[Any] = {} if keyed else []
    position = _skip_whitespace(text, position)
    if text.startswith(closing, position):
        return items, position + 1
    while True:
        if keyed:
            key, position = _parse_gemma_native_key(text, position)
            position = _skip_whitespace(text, position)
            if not text.startswith(":", position):
                raise ValueError(f"expected `:` after key `{key}` at offset {position}.")
            value, position = _parse_gemma_native_value(text, position + 1)
            items[key] = value  # type: ignore[index]
        else:
            value, position = _parse_gemma_native_value(text, position)
            items.append(value)  # type: ignore[union-attr]
        position = _skip_whitespace(text, position)
        if text.startswith(",", position):
            position = _skip_whitespace(text, position + 1)
            continue
        if text.startswith(closing, position):
            return items, position + 1
        raise ValueError(f"expected `,` or `{closing}` at offset {position}.")


def _parse_gemma_native_key(text: str, position: int) -> tuple[str, int]:
    position = _skip_whitespace(text, position)
    if text.startswith(_GEMMA_STRING_DELIMITER, position) or text.startswith('"', position):
        key, position = _parse_gemma_native_value(text, position)
        return str(key), position
    bare = _GEMMA_BARE_KEY_PATTERN.match(text, position)
    if bare is None:
        raise ValueError(f"expected a key at offset {position}.")
    return bare.group(0), bare.end()


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


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
