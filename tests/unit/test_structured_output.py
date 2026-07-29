from __future__ import annotations

from pydantic import BaseModel

from lewlm.structured_output import (
    StructuredOutputRuntimeStatus,
    analyze_structured_output,
    build_structured_output_request,
    validate_value_against_json_schema,
)


def test_build_structured_output_request_returns_json_schema_contract() -> None:
    contract = build_structured_output_request(
        format="json_schema",
        name="status",
        schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    )

    assert contract is not None
    assert contract.type == "json_schema"
    assert contract.schema_payload["required"] == ["summary"]


def test_analyze_structured_output_validates_prompt_guided_json_against_schema() -> None:
    result = analyze_structured_output(
        format="json_schema",
        output_text='{"summary":"ok","extra":true}',
        schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
        runtime_status=StructuredOutputRuntimeStatus(
            runtime="mlx_text",
            mode="json_schema",
            enforcement="prompt_guided",
            decoder_enforced=False,
            fallback_used=True,
            fallback_reason="mlx_text does not support constrained decoding.",
        ),
    )

    assert result is not None
    assert result.enforcement == "prompt_guided"
    assert result.fallback_used is True
    assert result.validation.state == "invalid"
    assert result.validation.validator == "full_json_schema"
    assert result.validation.issues[0].code == "additional_property"


def test_analyze_structured_output_marks_decode_time_grammar_enforcement_valid() -> None:
    result = analyze_structured_output(
        format="grammar",
        output_text="ok",
        grammar='root ::= "ok"',
        runtime_status=StructuredOutputRuntimeStatus(
            runtime="llamacpp",
            mode="grammar",
            enforcement="decode_time",
            decoder_enforced=True,
            fallback_used=False,
        ),
    )

    assert result is not None
    assert result.enforcement == "decode_time"
    assert result.decoder_enforced is True
    assert result.fallback_used is False
    assert result.validation.state == "valid"
    assert result.validation.validator == "grammar"


def test_validator_resolves_pydantic_refs_and_defs_instead_of_passing_anything() -> None:
    """A `$ref` to a nested model used to validate nothing, so bad output read `valid`."""

    class Author(BaseModel):
        name: str
        email: str | None = None

    class Doc(BaseModel):
        title: str
        author: Author

    schema = Doc.model_json_schema()

    assert validate_value_against_json_schema(
        schema=schema,
        value={"title": "t", "author": {"name": "a", "email": None}},
    ) == []

    issues = validate_value_against_json_schema(
        schema=schema,
        value={"title": "t", "author": {"email": "a@example.com"}},
    )

    assert [issue.code for issue in issues] == ["required_property"]
    assert issues[0].path == ["author", "name"]


def test_validator_reports_the_closest_any_of_branch_for_optional_models() -> None:
    class Author(BaseModel):
        name: str

    class Doc(BaseModel):
        reviewer: Author | None = None

    issues = validate_value_against_json_schema(
        schema=Doc.model_json_schema(),
        value={"reviewer": {"name": 5}},
    )

    assert [issue.code for issue in issues] == ["any_of_mismatch"]
    assert issues[0].path == ["reviewer"]
    # The sub-path is what makes the summary actionable.
    assert "at `name`" in issues[0].message


def test_validator_handles_one_of_all_of_and_not() -> None:
    one_of = {"oneOf": [{"type": "integer"}, {"type": "string"}]}
    assert validate_value_against_json_schema(schema=one_of, value=1) == []
    assert [issue.code for issue in validate_value_against_json_schema(schema=one_of, value=[])] == [
        "one_of_mismatch",
    ]

    ambiguous = {"oneOf": [{"type": "integer"}, {"minimum": 0}]}
    assert [issue.code for issue in validate_value_against_json_schema(schema=ambiguous, value=1)] == [
        "one_of_ambiguous",
    ]

    all_of = {"allOf": [{"type": "string"}, {"minLength": 3}]}
    assert validate_value_against_json_schema(schema=all_of, value="abc") == []
    assert [issue.code for issue in validate_value_against_json_schema(schema=all_of, value="ab")] == ["min_length"]

    negated = {"not": {"type": "string"}}
    assert validate_value_against_json_schema(schema=negated, value=1) == []
    assert [issue.code for issue in validate_value_against_json_schema(schema=negated, value="x")] == ["not_mismatch"]


def test_validator_walks_recursive_refs_without_recursing_forever() -> None:
    class Node(BaseModel):
        name: str
        children: list["Node"] = []

    Node.model_rebuild()
    schema = Node.model_json_schema()

    assert validate_value_against_json_schema(
        schema=schema,
        value={"name": "a", "children": [{"name": "b", "children": []}]},
    ) == []

    issues = validate_value_against_json_schema(
        schema=schema,
        value={"name": "a", "children": [{"name": 1, "children": []}]},
    )
    assert issues[0].path == ["children", 0, "name"]

    # A `$ref` cycle that never consumes the value is bounded rather than fatal.
    cyclic = {"$ref": "#/$defs/Loop", "$defs": {"Loop": {"allOf": [{"$ref": "#/$defs/Loop"}]}}}
    assert [issue.code for issue in validate_value_against_json_schema(schema=cyclic, value=1)] == [
        "schema_recursion",
    ]


def test_validator_reports_an_unresolvable_reference_rather_than_passing_it() -> None:
    issues = validate_value_against_json_schema(
        schema={"$ref": "https://example.com/person.json"},
        value={"anything": True},
    )

    assert [issue.code for issue in issues] == ["unresolvable_ref"]

    missing = validate_value_against_json_schema(schema={"$ref": "#/$defs/Absent"}, value=1)
    assert [issue.code for issue in missing] == ["unresolvable_ref"]


def test_validator_applies_the_remaining_json_schema_assertions() -> None:
    assert [
        issue.code
        for issue in validate_value_against_json_schema(schema={"type": "string", "pattern": "^a"}, value="b")
    ] == ["pattern_mismatch"]
    assert [
        issue.code
        for issue in validate_value_against_json_schema(schema={"type": "array", "uniqueItems": True}, value=[1, 1])
    ] == ["unique_items"]
    assert [
        issue.code
        for issue in validate_value_against_json_schema(schema={"exclusiveMinimum": 0}, value=0)
    ] == ["exclusive_minimum"]
    assert [
        issue.code for issue in validate_value_against_json_schema(schema={"multipleOf": 5}, value=7)
    ] == ["multiple_of"]
    assert [
        issue.code
        for issue in validate_value_against_json_schema(
            schema={"type": "array", "prefixItems": [{"type": "string"}], "items": {"type": "integer"}},
            value=["a", 1, "b"],
        )
    ] == ["type_mismatch"]
    assert [
        issue.code
        for issue in validate_value_against_json_schema(
            schema={"type": "object", "patternProperties": {"^x": {"type": "integer"}}, "additionalProperties": False},
            value={"xa": 1, "y": 2},
        )
    ] == ["additional_property"]
    assert [
        issue.code
        for issue in validate_value_against_json_schema(
            schema={"dependentRequired": {"card": ["expiry"]}},
            value={"card": "1234"},
        )
    ] == ["dependent_required_property"]


def test_validator_does_not_conflate_booleans_with_numbers() -> None:
    assert [issue.code for issue in validate_value_against_json_schema(schema={"const": True}, value=1)] == [
        "const_mismatch",
    ]
    assert [issue.code for issue in validate_value_against_json_schema(schema={"enum": [True]}, value=1)] == [
        "enum_mismatch",
    ]
    assert validate_value_against_json_schema(schema={"const": True}, value=True) == []
