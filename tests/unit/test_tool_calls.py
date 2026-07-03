from __future__ import annotations

from lewlm.prompting import PromptToolDefinition
from lewlm.tool_calls import parse_tool_calls

_WEATHER_TOOL = PromptToolDefinition(
    name="get_weather",
    description="Look up the local weather.",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
)
_SEARCH_TOOL = PromptToolDefinition(
    name="search_notes",
    description="Search local notes.",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


def test_parses_single_fenced_tool_call_and_strips_it_from_text() -> None:
    output = (
        "I'll check the weather.\n"
        "```json\n"
        '{"tool_call": {"name": "get_weather", "arguments": {"city": "Oslo"}}}\n'
        "```\n"
        "Done."
    )

    result = parse_tool_calls(output, tools=[_WEATHER_TOOL])

    assert result.status == "parsed"
    assert result.parallel is False
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.call_id == "call_1"
    assert call.name == "get_weather"
    assert call.arguments == {"city": "Oslo"}
    assert "```" not in result.remaining_text
    assert "I'll check the weather." in result.remaining_text
    assert "Done." in result.remaining_text


def test_parses_parallel_tool_calls_array() -> None:
    output = (
        "```json\n"
        '{"tool_calls": ['
        '{"name": "get_weather", "arguments": {"city": "Oslo"}},'
        '{"name": "search_notes", "arguments": {"query": "trip"}}'
        "]}\n"
        "```"
    )

    result = parse_tool_calls(output, tools=[_WEATHER_TOOL, _SEARCH_TOOL])

    assert result.status == "parsed"
    assert result.parallel is True
    assert [call.name for call in result.tool_calls] == ["get_weather", "search_notes"]
    assert [call.call_id for call in result.tool_calls] == ["call_1", "call_2"]


def test_parses_bare_whole_text_tool_call() -> None:
    output = '{"name": "get_weather", "arguments": {"city": "Oslo", "unit": "celsius"}}'

    result = parse_tool_calls(output, tools=[_WEATHER_TOOL])

    assert result.status == "parsed"
    assert result.tool_calls[0].arguments["unit"] == "celsius"
    assert result.remaining_text == ""


def test_accepts_input_alias_for_arguments() -> None:
    output = '{"tool_call": {"name": "search_notes", "input": {"query": "receipts"}}}'

    result = parse_tool_calls(output, tools=[_SEARCH_TOOL])

    assert result.status == "parsed"
    assert result.tool_calls[0].arguments == {"query": "receipts"}


def test_rejects_unknown_tool_with_explicit_issue() -> None:
    output = '{"tool_call": {"name": "delete_everything", "arguments": {}}}'

    result = parse_tool_calls(output, tools=[_WEATHER_TOOL])

    assert result.status == "failed"
    assert result.tool_calls == []
    assert result.issues[0].code == "unknown_tool"
    assert "delete_everything" in result.issues[0].message
    assert "get_weather" in result.issues[0].message


def test_rejects_schema_violations_without_silent_repair() -> None:
    output = '{"tool_call": {"name": "get_weather", "arguments": {"unit": "kelvin"}}}'

    result = parse_tool_calls(output, tools=[_WEATHER_TOOL])

    assert result.status == "failed"
    assert result.issues[0].code == "schema_violation"
    assert "get_weather" in result.issues[0].message


def test_reports_malformed_json_in_json_fence() -> None:
    output = "```json\n{\"tool_call\": {\"name\": \"get_weather\",}\n```"

    result = parse_tool_calls(output, tools=[_WEATHER_TOOL])

    assert result.status == "failed"
    assert result.issues[0].code == "invalid_json"


def test_mixed_valid_and_invalid_candidates_report_partial() -> None:
    output = (
        "```json\n"
        '{"tool_call": {"name": "get_weather", "arguments": {"city": "Oslo"}}}\n'
        "```\n"
        "```json\n"
        '{"tool_call": {"name": "unknown_tool", "arguments": {}}}\n'
        "```"
    )

    result = parse_tool_calls(output, tools=[_WEATHER_TOOL])

    assert result.status == "partial"
    assert len(result.tool_calls) == 1
    assert result.issues[0].code == "unknown_tool"


def test_plain_text_reports_no_tool_calls() -> None:
    output = "The weather in Oslo is usually mild in summer."

    result = parse_tool_calls(output, tools=[_WEATHER_TOOL])

    assert result.status == "no_tool_calls"
    assert result.tool_calls == []
    assert result.issues == []
    assert result.remaining_text == output


def test_non_json_code_fence_is_not_treated_as_candidate() -> None:
    output = "```python\nprint('hello')\n```"

    result = parse_tool_calls(output, tools=[_WEATHER_TOOL])

    assert result.status == "no_tool_calls"


def test_unrecognized_json_shape_reports_explicit_issue() -> None:
    output = '```json\n{"tool_calls": "not-an-array", "name": "x"}\n```'

    result = parse_tool_calls(output, tools=[_WEATHER_TOOL])

    assert result.status == "failed"
    assert result.issues[0].code == "unrecognized_shape"


def test_arguments_must_be_object() -> None:
    output = '{"tool_call": {"name": "get_weather", "arguments": "Oslo"}}'

    result = parse_tool_calls(output, tools=[_WEATHER_TOOL])

    assert result.status == "failed"
    assert result.issues[0].code == "arguments_not_object"
