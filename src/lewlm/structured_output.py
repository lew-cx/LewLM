"""Shared structured-output request, runtime status, and result models."""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal
from urllib.parse import unquote

from pydantic import BaseModel, ConfigDict, Field


_REF_KEY = "$ref"

#: A `$ref` chain that never consumes any of the value is a self-referential
#: schema, not a deep one. Bound it rather than recursing until the stack fails.
_MAX_REF_DEPTH = 64


class TextResponseFormat(BaseModel):
    """Explicit plain-text response format."""

    type: Literal["text"] = "text"


class JSONSchemaResponseFormat(BaseModel):
    """JSON-schema structured-output contract."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["json_schema"] = "json_schema"
    schema_payload: dict[str, Any] = Field(alias="schema", serialization_alias="schema")
    name: str | None = None
    strict: bool = True


class GrammarResponseFormat(BaseModel):
    """Grammar-based structured-output contract."""

    type: Literal["grammar"] = "grammar"
    grammar: str
    syntax: str = "ebnf"
    name: str | None = None
    strict: bool = True


StructuredOutputRequest = Annotated[
    TextResponseFormat | JSONSchemaResponseFormat | GrammarResponseFormat,
    Field(discriminator="type"),
]


class StructuredOutputIssue(BaseModel):
    """Single structured-output validation issue."""

    code: str
    message: str
    path: list[str | int] = Field(default_factory=list)


class StructuredOutputValidation(BaseModel):
    """Post-generation validation metadata for a structured-output request."""

    state: Literal["not_requested", "valid", "invalid", "unavailable"] = "not_requested"
    validator: Literal["none", "json_parse_only", "full_json_schema", "grammar"] = "none"
    message: str | None = None
    issues: list[StructuredOutputIssue] = Field(default_factory=list)


class StructuredOutputRuntimeStatus(BaseModel):
    """Runtime-side enforcement status recorded during generation."""

    runtime: str | None = None
    mode: Literal["text", "json_schema", "grammar"] = "text"
    enforcement: Literal["prompt_guided", "decode_time"] = "prompt_guided"
    decoder_enforced: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None


class StructuredOutputResult(BaseModel):
    """Public structured-output status attached to generation responses."""

    requested: bool = False
    contract: StructuredOutputRequest | None = None
    enforcement: Literal["none", "prompt_guided", "decode_time"] = "none"
    decoder_enforced: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    parsed_output: Any | None = None
    validation: StructuredOutputValidation = Field(default_factory=StructuredOutputValidation)


_PROMPT_GUIDED_FALLBACK_REASON = (
    "LewLM recorded the structured-output contract, but this path still relies on "
    "prompt-guided fallback rather than decode-time constrained decoding."
)


def build_structured_output_request(
    *,
    format: Literal["text", "json_schema", "grammar"],
    schema: dict[str, Any] | None = None,
    grammar: str | None = None,
    syntax: str | None = None,
    name: str | None = None,
    strict: bool | None = None,
) -> StructuredOutputRequest | None:
    """Build a concrete structured-output request from a normalized contract."""

    if format == "text":
        return None
    if format == "json_schema":
        return JSONSchemaResponseFormat(
            schema=dict(schema or {}),
            name=name,
            strict=True if strict is None else strict,
        )
    return GrammarResponseFormat(
        grammar=grammar or "",
        syntax=syntax or "ebnf",
        name=name,
        strict=True if strict is None else strict,
    )


def analyze_structured_output(
    *,
    format: Literal["text", "json_schema", "grammar"],
    output_text: str,
    schema: dict[str, Any] | None = None,
    grammar: str | None = None,
    syntax: str | None = None,
    name: str | None = None,
    strict: bool | None = None,
    runtime_status: StructuredOutputRuntimeStatus | dict[str, Any] | None = None,
) -> StructuredOutputResult | None:
    """Build structured-output compatibility and validation metadata."""

    contract = build_structured_output_request(
        format=format,
        schema=schema,
        grammar=grammar,
        syntax=syntax,
        name=name,
        strict=strict,
    )
    if contract is None:
        return None

    status = _runtime_status(runtime_status, format=format)
    if format == "json_schema":
        parsed_output, validation = _analyze_json_schema_output(
            output_text,
            schema=contract.schema_payload,
            decoder_enforced=status.decoder_enforced,
        )
        return StructuredOutputResult(
            requested=True,
            contract=contract,
            enforcement=status.enforcement,
            decoder_enforced=status.decoder_enforced,
            fallback_used=status.fallback_used,
            fallback_reason=status.fallback_reason,
            parsed_output=parsed_output,
            validation=validation,
        )

    validation = (
        StructuredOutputValidation(
            state="valid",
            validator="grammar",
            message="LewLM enforced the requested grammar at decode time on the selected runtime path.",
        )
        if status.decoder_enforced
        else StructuredOutputValidation(
            state="unavailable",
            validator="grammar",
            message=(
                "LewLM recorded the requested grammar contract, but this path does not yet "
                "provide decode-time or post-generation grammar validation."
            ),
        )
    )
    return StructuredOutputResult(
        requested=True,
        contract=contract,
        enforcement=status.enforcement,
        decoder_enforced=status.decoder_enforced,
        fallback_used=status.fallback_used,
        fallback_reason=status.fallback_reason,
        validation=validation,
    )


def _runtime_status(
    payload: StructuredOutputRuntimeStatus | dict[str, Any] | None,
    *,
    format: Literal["text", "json_schema", "grammar"],
) -> StructuredOutputRuntimeStatus:
    if isinstance(payload, StructuredOutputRuntimeStatus):
        return payload
    if isinstance(payload, dict):
        return StructuredOutputRuntimeStatus.model_validate(payload)
    return StructuredOutputRuntimeStatus(
        mode=format,
        enforcement="prompt_guided",
        decoder_enforced=False,
        fallback_used=True,
        fallback_reason=_PROMPT_GUIDED_FALLBACK_REASON,
    )


def _analyze_json_schema_output(
    output_text: str,
    *,
    schema: dict[str, Any],
    decoder_enforced: bool,
) -> tuple[Any | None, StructuredOutputValidation]:
    stripped = output_text.strip()
    if not stripped:
        return None, StructuredOutputValidation(
            state="invalid",
            validator="json_parse_only",
            message="Model output was empty for a requested json_schema contract.",
            issues=[
                StructuredOutputIssue(
                    code="invalid_json",
                    message="Expected a JSON value, but the model returned an empty response.",
                ),
            ],
        )
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, StructuredOutputValidation(
            state="invalid",
            validator="json_parse_only",
            message="Model output was not valid JSON for the requested json_schema contract.",
            issues=[
                StructuredOutputIssue(
                    code="invalid_json",
                    message=f"{exc.msg} at line {exc.lineno}, column {exc.colno}.",
                ),
            ],
        )
    issues = _validate_json_schema(schema=schema, value=parsed)
    if issues:
        return parsed, StructuredOutputValidation(
            state="invalid",
            validator="full_json_schema",
            message="Parsed JSON did not conform to the requested JSON schema.",
            issues=issues,
        )
    return parsed, StructuredOutputValidation(
        state="valid",
        validator="full_json_schema",
        message=(
            "LewLM enforced the requested JSON schema at decode time on the selected runtime path."
            if decoder_enforced
            else "LewLM validated the parsed JSON against the requested JSON schema after generation."
        ),
    )


def _validate_json_schema(
    *,
    schema: dict[str, Any],
    value: Any,
    path: list[str | int] | None = None,
) -> list[StructuredOutputIssue]:
    """Validate `value` against `schema`, treating `schema` as its own `$ref` root."""

    return _validate_node(
        schema=schema,
        value=value,
        path=list(path or []),
        root=schema,
        ref_depth=0,
    )


def _validate_node(
    *,
    schema: Any,
    value: Any,
    path: list[str | int],
    root: Any,
    ref_depth: int,
) -> list[StructuredOutputIssue]:
    # JSON Schema permits a bare boolean anywhere a schema is expected.
    if schema is True:
        return []
    if schema is False:
        return [
            StructuredOutputIssue(
                code="schema_forbids_value",
                message="The schema forbids any value at this location.",
                path=path,
            ),
        ]
    if not isinstance(schema, dict):
        return []

    issues: list[StructuredOutputIssue] = []
    issues.extend(_validate_reference(schema=schema, value=value, path=path, root=root, ref_depth=ref_depth))

    if "const" in schema and not _json_equal(value, schema["const"]):
        issues.append(
            StructuredOutputIssue(
                code="const_mismatch",
                message=f"Expected constant value {schema['const']!r}.",
                path=path,
            ),
        )
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and not any(_json_equal(value, candidate) for candidate in enum_values):
        issues.append(
            StructuredOutputIssue(
                code="enum_mismatch",
                message=f"Expected one of {enum_values!r}.",
                path=path,
            ),
        )

    expected_types = _expected_json_types(schema.get("type"))
    if expected_types and not any(_matches_json_type(value, expected_type) for expected_type in expected_types):
        issues.append(
            StructuredOutputIssue(
                code="type_mismatch",
                message=f"Expected {_describe_types(expected_types)}, but received {_json_type_name(value)}.",
                path=path,
            ),
        )
        # Keyword checks below are all type-specific, so continuing here would
        # only pile cascading noise on top of the one issue that matters.
        return issues

    issues.extend(_validate_composition(schema=schema, value=value, path=path, root=root, ref_depth=ref_depth))

    if isinstance(value, dict):
        issues.extend(_validate_object(schema=schema, value=value, path=path, root=root))
    elif isinstance(value, list):
        issues.extend(_validate_array(schema=schema, value=value, path=path, root=root))
    elif isinstance(value, str):
        issues.extend(_validate_string(schema=schema, value=value, path=path))
    elif _json_type_name(value) in {"integer", "number"}:
        issues.extend(_validate_number(schema=schema, value=value, path=path))
    return issues


def _validate_reference(
    *,
    schema: dict[str, Any],
    value: Any,
    path: list[str | int],
    root: Any,
    ref_depth: int,
) -> list[StructuredOutputIssue]:
    """Validate against a `$ref` target, if the node carries one.

    Sibling keywords still apply — the caller evaluates them — which matches
    2020-12 and is what Pydantic emits for an annotated model reference.
    """

    reference = schema.get(_REF_KEY)
    if not isinstance(reference, str):
        return []
    if ref_depth >= _MAX_REF_DEPTH:
        return [
            StructuredOutputIssue(
                code="schema_recursion",
                message=(
                    f"Stopped resolving `$ref` after {_MAX_REF_DEPTH} hops without consuming any of the "
                    "value; the schema references itself unconditionally."
                ),
                path=path,
            ),
        ]
    target, resolved = _resolve_reference(reference, root=root)
    if not resolved:
        # Silently passing here is how an unvalidatable schema turns into a
        # false `valid`, which is the failure this validator exists to avoid.
        return [
            StructuredOutputIssue(
                code="unresolvable_ref",
                message=(
                    f"Cannot resolve `$ref` `{reference}`; LewLM validates against the submitted schema "
                    "document only and does not fetch external references."
                ),
                path=path,
            ),
        ]
    return _validate_node(schema=target, value=value, path=path, root=root, ref_depth=ref_depth + 1)


def _validate_composition(
    *,
    schema: dict[str, Any],
    value: Any,
    path: list[str | int],
    root: Any,
    ref_depth: int,
) -> list[StructuredOutputIssue]:
    issues: list[StructuredOutputIssue] = []

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for subschema in all_of:
            issues.extend(
                _validate_node(schema=subschema, value=value, path=path, root=root, ref_depth=ref_depth),
            )

    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        branch_issues = [
            _validate_node(schema=subschema, value=value, path=path, root=root, ref_depth=ref_depth)
            for subschema in any_of
        ]
        if all(branch for branch in branch_issues):
            issues.append(_branch_mismatch("any_of_mismatch", branch_issues, path=path, requirement="at least one"))

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        branch_issues = [
            _validate_node(schema=subschema, value=value, path=path, root=root, ref_depth=ref_depth)
            for subschema in one_of
        ]
        matched = [index for index, branch in enumerate(branch_issues) if not branch]
        if not matched:
            issues.append(_branch_mismatch("one_of_mismatch", branch_issues, path=path, requirement="exactly one"))
        elif len(matched) > 1:
            issues.append(
                StructuredOutputIssue(
                    code="one_of_ambiguous",
                    message=(
                        f"Expected exactly one `oneOf` branch to match, but branches "
                        f"{', '.join(str(index) for index in matched)} all matched."
                    ),
                    path=path,
                ),
            )

    if "not" in schema:
        negated = _validate_node(schema=schema["not"], value=value, path=path, root=root, ref_depth=ref_depth)
        if not negated:
            issues.append(
                StructuredOutputIssue(
                    code="not_mismatch",
                    message="Value matched a schema declared under `not`.",
                    path=path,
                ),
            )
    return issues


def _branch_mismatch(
    code: str,
    branch_issues: list[list[StructuredOutputIssue]],
    *,
    path: list[str | int],
    requirement: str,
) -> StructuredOutputIssue:
    """Summarize why every branch of a composition keyword rejected the value."""

    if len(branch_issues) == 1:
        detail = _describe_branch_issues(branch_issues[0], relative_to=path)
    else:
        # The branch that came closest is the one worth reporting in full.
        closest = min(range(len(branch_issues)), key=lambda index: len(branch_issues[index]))
        detail = (
            f"{len(branch_issues)} branches were tried; the closest (branch {closest}) reported: "
            + _describe_branch_issues(branch_issues[closest], relative_to=path)
        )
    return StructuredOutputIssue(
        code=code,
        message=f"Expected {requirement} branch to match. {detail}",
        path=path,
    )


def _describe_branch_issues(
    issues: list[StructuredOutputIssue],
    *,
    relative_to: list[str | int],
) -> str:
    """Render branch issues, keeping the sub-path that the summary would lose."""

    rendered: list[str] = []
    for issue in issues:
        suffix = issue.path[len(relative_to) :]
        location = _render_path(suffix)
        rendered.append(f"{issue.message} (at `{location}`)" if location else issue.message)
    return "; ".join(rendered)


def _render_path(segments: list[str | int]) -> str:
    parts: list[str] = []
    for segment in segments:
        parts.append(f"[{segment}]" if isinstance(segment, int) else f".{segment}")
    return "".join(parts).lstrip(".")


def _validate_object(
    *,
    schema: dict[str, Any],
    value: dict[str, Any],
    path: list[str | int],
    root: Any,
) -> list[StructuredOutputIssue]:
    issues: list[StructuredOutputIssue] = []

    required = schema.get("required")
    if isinstance(required, list):
        for key in required:
            if isinstance(key, str) and key not in value:
                issues.append(
                    StructuredOutputIssue(
                        code="required_property",
                        message=f"Missing required property `{key}`.",
                        path=[*path, key],
                    ),
                )

    dependent_required = schema.get("dependentRequired")
    if isinstance(dependent_required, dict):
        for trigger, dependents in dependent_required.items():
            if trigger not in value or not isinstance(dependents, list):
                continue
            for key in dependents:
                if isinstance(key, str) and key not in value:
                    issues.append(
                        StructuredOutputIssue(
                            code="dependent_required_property",
                            message=f"Property `{key}` is required when `{trigger}` is present.",
                            path=[*path, key],
                        ),
                    )

    min_properties = schema.get("minProperties")
    if isinstance(min_properties, int) and not isinstance(min_properties, bool) and len(value) < min_properties:
        issues.append(
            StructuredOutputIssue(
                code="min_properties",
                message=f"Expected at least {min_properties} propert(y/ies).",
                path=path,
            ),
        )
    max_properties = schema.get("maxProperties")
    if isinstance(max_properties, int) and not isinstance(max_properties, bool) and len(value) > max_properties:
        issues.append(
            StructuredOutputIssue(
                code="max_properties",
                message=f"Expected at most {max_properties} propert(y/ies).",
                path=path,
            ),
        )

    property_map = schema.get("properties")
    property_map = property_map if isinstance(property_map, dict) else {}
    for key, property_schema in property_map.items():
        if key in value:
            issues.extend(
                _validate_node(
                    schema=property_schema,
                    value=value[key],
                    path=[*path, key],
                    root=root,
                    ref_depth=0,
                ),
            )

    pattern_properties = schema.get("patternProperties")
    pattern_properties = pattern_properties if isinstance(pattern_properties, dict) else {}
    matched_by_pattern: set[str] = set()
    for pattern, property_schema in pattern_properties.items():
        compiled = _compile_pattern(pattern)
        if compiled is None:
            continue
        for key in value:
            if compiled.search(key) is None:
                continue
            matched_by_pattern.add(key)
            issues.extend(
                _validate_node(
                    schema=property_schema,
                    value=value[key],
                    path=[*path, key],
                    root=root,
                    ref_depth=0,
                ),
            )

    property_names = schema.get("propertyNames")
    if property_names is not None:
        for key in value:
            issues.extend(
                _validate_node(
                    schema=property_names,
                    value=key,
                    path=[*path, key],
                    root=root,
                    ref_depth=0,
                ),
            )

    # `additionalProperties` is scoped to the `properties` and `patternProperties`
    # declared on this same schema object, per spec — a sibling `allOf` branch
    # does not widen it.
    additional_properties = schema.get("additionalProperties", True)
    if additional_properties is not True:
        extra_keys = [key for key in value if key not in property_map and key not in matched_by_pattern]
        if additional_properties is False:
            issues.extend(
                StructuredOutputIssue(
                    code="additional_property",
                    message=f"Unexpected property `{key}`.",
                    path=[*path, key],
                )
                for key in extra_keys
            )
        else:
            for key in extra_keys:
                issues.extend(
                    _validate_node(
                        schema=additional_properties,
                        value=value[key],
                        path=[*path, key],
                        root=root,
                        ref_depth=0,
                    ),
                )
    return issues


def _validate_array(
    *,
    schema: dict[str, Any],
    value: list[Any],
    path: list[str | int],
    root: Any,
) -> list[StructuredOutputIssue]:
    issues: list[StructuredOutputIssue] = []

    min_items = schema.get("minItems")
    if isinstance(min_items, int) and not isinstance(min_items, bool) and len(value) < min_items:
        issues.append(
            StructuredOutputIssue(
                code="min_items",
                message=f"Expected at least {min_items} item(s).",
                path=path,
            ),
        )
    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and not isinstance(max_items, bool) and len(value) > max_items:
        issues.append(
            StructuredOutputIssue(
                code="max_items",
                message=f"Expected at most {max_items} item(s).",
                path=path,
            ),
        )
    if schema.get("uniqueItems") is True:
        seen: list[Any] = []
        for index, item in enumerate(value):
            if any(_json_equal(item, previous) for previous in seen):
                issues.append(
                    StructuredOutputIssue(
                        code="unique_items",
                        message="Expected every item to be unique.",
                        path=[*path, index],
                    ),
                )
            else:
                seen.append(item)

    prefix_items = schema.get("prefixItems")
    prefix_count = 0
    if isinstance(prefix_items, list):
        prefix_count = len(prefix_items)
        for index, subschema in enumerate(prefix_items):
            if index >= len(value):
                break
            issues.extend(
                _validate_node(
                    schema=subschema,
                    value=value[index],
                    path=[*path, index],
                    root=root,
                    ref_depth=0,
                ),
            )

    items_schema = schema.get("items")
    if items_schema is not None:
        for index in range(prefix_count, len(value)):
            issues.extend(
                _validate_node(
                    schema=items_schema,
                    value=value[index],
                    path=[*path, index],
                    root=root,
                    ref_depth=0,
                ),
            )

    contains = schema.get("contains")
    if contains is not None:
        match_count = sum(
            1
            for index, item in enumerate(value)
            if not _validate_node(schema=contains, value=item, path=[*path, index], root=root, ref_depth=0)
        )
        min_contains = schema.get("minContains")
        min_contains = min_contains if isinstance(min_contains, int) and not isinstance(min_contains, bool) else 1
        if match_count < min_contains:
            issues.append(
                StructuredOutputIssue(
                    code="contains",
                    message=f"Expected at least {min_contains} item(s) matching `contains`, found {match_count}.",
                    path=path,
                ),
            )
        max_contains = schema.get("maxContains")
        if isinstance(max_contains, int) and not isinstance(max_contains, bool) and match_count > max_contains:
            issues.append(
                StructuredOutputIssue(
                    code="max_contains",
                    message=f"Expected at most {max_contains} item(s) matching `contains`, found {match_count}.",
                    path=path,
                ),
            )
    return issues


def _validate_string(
    *,
    schema: dict[str, Any],
    value: str,
    path: list[str | int],
) -> list[StructuredOutputIssue]:
    issues: list[StructuredOutputIssue] = []

    min_length = schema.get("minLength")
    if isinstance(min_length, int) and not isinstance(min_length, bool) and len(value) < min_length:
        issues.append(
            StructuredOutputIssue(
                code="min_length",
                message=f"Expected string length >= {min_length}.",
                path=path,
            ),
        )
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and not isinstance(max_length, bool) and len(value) > max_length:
        issues.append(
            StructuredOutputIssue(
                code="max_length",
                message=f"Expected string length <= {max_length}.",
                path=path,
            ),
        )
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        compiled = _compile_pattern(pattern)
        if compiled is not None and compiled.search(value) is None:
            issues.append(
                StructuredOutputIssue(
                    code="pattern_mismatch",
                    message=f"Expected a string matching `{pattern}`.",
                    path=path,
                ),
            )
    return issues


def _validate_number(
    *,
    schema: dict[str, Any],
    value: int | float,
    path: list[str | int],
) -> list[StructuredOutputIssue]:
    issues: list[StructuredOutputIssue] = []

    minimum = schema.get("minimum")
    if _is_number(minimum) and value < minimum:
        issues.append(
            StructuredOutputIssue(
                code="minimum",
                message=f"Expected value >= {minimum}.",
                path=path,
            ),
        )
    maximum = schema.get("maximum")
    if _is_number(maximum) and value > maximum:
        issues.append(
            StructuredOutputIssue(
                code="maximum",
                message=f"Expected value <= {maximum}.",
                path=path,
            ),
        )
    exclusive_minimum = schema.get("exclusiveMinimum")
    if _is_number(exclusive_minimum) and value <= exclusive_minimum:
        issues.append(
            StructuredOutputIssue(
                code="exclusive_minimum",
                message=f"Expected value > {exclusive_minimum}.",
                path=path,
            ),
        )
    exclusive_maximum = schema.get("exclusiveMaximum")
    if _is_number(exclusive_maximum) and value >= exclusive_maximum:
        issues.append(
            StructuredOutputIssue(
                code="exclusive_maximum",
                message=f"Expected value < {exclusive_maximum}.",
                path=path,
            ),
        )
    multiple_of = schema.get("multipleOf")
    if _is_number(multiple_of) and multiple_of > 0:
        quotient = value / multiple_of
        if abs(quotient - round(quotient)) > 1e-9:
            issues.append(
                StructuredOutputIssue(
                    code="multiple_of",
                    message=f"Expected a multiple of {multiple_of}.",
                    path=path,
                ),
            )
    return issues


def _resolve_reference(reference: str, *, root: Any) -> tuple[Any, bool]:
    """Resolve a local JSON pointer against `root`.

    Returns `(target, resolved)`. External and plain-name (`$anchor`) references
    are reported as unresolved rather than assumed valid.
    """

    if not reference.startswith("#"):
        return None, False
    pointer = reference[1:]
    if not pointer:
        return root, True
    if not pointer.startswith("/"):
        return None, False

    node = root
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        token = unquote(token)
        if isinstance(node, dict):
            if token not in node:
                return None, False
            node = node[token]
        elif isinstance(node, list):
            try:
                index = int(token)
            except ValueError:
                return None, False
            if index < 0 or index >= len(node):
                return None, False
            node = node[index]
        else:
            return None, False
    return node, True


def _compile_pattern(pattern: str) -> re.Pattern[str] | None:
    """Compile a schema pattern, ignoring ECMA-only syntax Python cannot parse."""

    try:
        return re.compile(pattern)
    except re.error:
        return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_equal(left: Any, right: Any) -> bool:
    """Compare two JSON values without Python's `True == 1` conflation."""

    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (list, dict)) or isinstance(right, (list, dict)):
        return False
    return left == right


def _expected_json_types(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _json_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def _describe_types(expected_types: list[str]) -> str:
    if len(expected_types) == 1:
        return expected_types[0]
    return " or ".join(expected_types)


def validate_value_against_json_schema(
    *,
    schema: dict[str, Any],
    value: Any,
) -> list[StructuredOutputIssue]:
    """Validate a parsed JSON value against a JSON schema and return issues.

    `schema` is both the contract and the `$ref` resolution root, so a schema
    generated by `BaseModel.model_json_schema()` — which references its nested
    models through `#/$defs/...` — validates without being pre-flattened.
    """

    return _validate_json_schema(schema=schema, value=value)


__all__ = [
    "GrammarResponseFormat",
    "JSONSchemaResponseFormat",
    "StructuredOutputRuntimeStatus",
    "StructuredOutputIssue",
    "StructuredOutputRequest",
    "StructuredOutputResult",
    "StructuredOutputValidation",
    "TextResponseFormat",
    "analyze_structured_output",
    "build_structured_output_request",
    "validate_value_against_json_schema",
]
