from __future__ import annotations

from lewlm.structured_output import StructuredOutputRuntimeStatus, analyze_structured_output, build_structured_output_request


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
