"""Shared structured-output request, runtime status, and result models."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    current_path = list(path or [])
    issues: list[StructuredOutputIssue] = []

    const = schema.get("const")
    if "const" in schema and value != const:
        issues.append(
            StructuredOutputIssue(
                code="const_mismatch",
                message=f"Expected constant value {const!r}.",
                path=current_path,
            ),
        )
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        issues.append(
            StructuredOutputIssue(
                code="enum_mismatch",
                message=f"Expected one of {enum_values!r}.",
                path=current_path,
            ),
        )

    expected_types = _expected_json_types(schema.get("type"))
    if expected_types and not any(_matches_json_type(value, expected_type) for expected_type in expected_types):
        issues.append(
            StructuredOutputIssue(
                code="type_mismatch",
                message=f"Expected {_describe_types(expected_types)}, but received {_json_type_name(value)}.",
                path=current_path,
            ),
        )
        return issues

    if isinstance(value, dict):
        properties = schema.get("properties")
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    issues.append(
                        StructuredOutputIssue(
                            code="required_property",
                            message=f"Missing required property `{key}`.",
                            path=[*current_path, key],
                        ),
                    )
        property_map = properties if isinstance(properties, dict) else {}
        for key, property_schema in property_map.items():
            if key in value and isinstance(property_schema, dict):
                issues.extend(
                    _validate_json_schema(
                        schema=property_schema,
                        value=value[key],
                        path=[*current_path, key],
                    ),
                )
        additional_properties = schema.get("additionalProperties", True)
        extra_keys = [key for key in value if key not in property_map]
        if additional_properties is False:
            issues.extend(
                StructuredOutputIssue(
                    code="additional_property",
                    message=f"Unexpected property `{key}`.",
                    path=[*current_path, key],
                )
                for key in extra_keys
            )
        elif isinstance(additional_properties, dict):
            for key in extra_keys:
                issues.extend(
                    _validate_json_schema(
                        schema=additional_properties,
                        value=value[key],
                        path=[*current_path, key],
                    ),
                )
        return issues

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            issues.append(
                StructuredOutputIssue(
                    code="min_items",
                    message=f"Expected at least {min_items} item(s).",
                    path=current_path,
                ),
            )
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            issues.append(
                StructuredOutputIssue(
                    code="max_items",
                    message=f"Expected at most {max_items} item(s).",
                    path=current_path,
                ),
            )
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(value):
                issues.extend(
                    _validate_json_schema(
                        schema=items_schema,
                        value=item,
                        path=[*current_path, index],
                    ),
                )
        return issues

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            issues.append(
                StructuredOutputIssue(
                    code="min_length",
                    message=f"Expected string length >= {min_length}.",
                    path=current_path,
                ),
            )
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(value) > max_length:
            issues.append(
                StructuredOutputIssue(
                    code="max_length",
                    message=f"Expected string length <= {max_length}.",
                    path=current_path,
                ),
            )
        return issues

    if _json_type_name(value) in {"integer", "number"}:
        minimum = schema.get("minimum")
        if isinstance(minimum, int | float) and value < minimum:
            issues.append(
                StructuredOutputIssue(
                    code="minimum",
                    message=f"Expected value >= {minimum}.",
                    path=current_path,
                ),
            )
        maximum = schema.get("maximum")
        if isinstance(maximum, int | float) and value > maximum:
            issues.append(
                StructuredOutputIssue(
                    code="maximum",
                    message=f"Expected value <= {maximum}.",
                    path=current_path,
                ),
            )
    return issues


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
]
