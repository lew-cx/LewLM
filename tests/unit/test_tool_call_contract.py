"""The prompt's tool-call instructions and the parser must not drift apart."""

from __future__ import annotations

import json
import re

import pytest

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import GenerateMessage
from lewlm.prompting import (
    PromptCompilationRequest,
    PromptCompiler,
    PromptMCPToolDefinition,
    PromptToolDefinition,
)
from lewlm.tool_call_contract import (
    ARGUMENT_KEYS,
    TOOL_CALL_BATCH_KEY,
    TOOL_CALL_MARKER_KEYS,
    TOOL_CALL_NAME_KEY,
    TOOL_CALL_SINGLE_KEY,
    tool_call_invocation_instructions,
)
from lewlm.tool_calls import parse_tool_calls

_TOOL = PromptToolDefinition(
    name="get_weather",
    description="Look up current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def _compile(request: PromptCompilationRequest) -> str:
    result = PromptCompiler(LewLMSettings()).compile(
        messages=[GenerateMessage(role="user", content="What's the weather in Oslo?")],
        request=request,
    )
    return "\n\n".join(message.content for message in result.messages if message.role == "system")


# --- the prompt states the contract ------------------------------------------


def test_declaring_a_tool_also_states_the_invocation_shape() -> None:
    prompt = _compile(PromptCompilationRequest(tools=[_TOOL]))

    assert "get_weather" in prompt
    # Every accepted marker key must be named in the prompt.
    for marker in TOOL_CALL_MARKER_KEYS:
        assert f'"{marker}"' in prompt, f"prompt never mentions the `{marker}` shape"
    assert ARGUMENT_KEYS[0] in prompt
    assert ARGUMENT_KEYS[1] in prompt


def test_the_prompt_warns_against_the_arguments_only_shape() -> None:
    # This is the exact failure mode: the model emits bare arguments and the
    # strict parser reports `no_tool_calls`, which reads as a refusal.
    prompt = _compile(PromptCompilationRequest(tools=[_TOOL]))
    assert "only arguments is not a tool call" in prompt


def test_mcp_tool_listings_also_get_the_invocation_shape() -> None:
    prompt = _compile(
        PromptCompilationRequest(
            mcp_tools=[
                PromptMCPToolDefinition(
                    name="mcp_lookup",
                    server="local",
                    description="Look something up.",
                    input_schema={"type": "object", "properties": {}},
                ),
            ],
        ),
    )
    assert "mcp_lookup" in prompt
    assert f'"{TOOL_CALL_NAME_KEY}"' in prompt


def test_the_contract_is_emitted_once_when_both_tool_kinds_are_declared() -> None:
    prompt = _compile(
        PromptCompilationRequest(
            tools=[_TOOL],
            mcp_tools=[
                PromptMCPToolDefinition(
                    name="mcp_lookup",
                    server="local",
                    input_schema={"type": "object", "properties": {}},
                ),
            ],
        ),
    )
    assert prompt.count("To call a tool, reply with a JSON object") == 1


def test_no_tool_instructions_when_no_tools_are_declared() -> None:
    prompt = _compile(PromptCompilationRequest(system_prompt="Be brief."))
    assert "To call a tool" not in prompt


def test_the_caller_system_prompt_slot_is_left_alone() -> None:
    # Host apps must not have to spend their own system-prompt slot on this.
    prompt = _compile(PromptCompilationRequest(tools=[_TOOL], system_prompt="Answer in Norwegian."))
    assert "Answer in Norwegian." in prompt
    assert "To call a tool, reply with a JSON object" in prompt


# --- what the prompt advertises is exactly what the parser accepts -----------


def _advertised_json_shapes() -> list[str]:
    """Pull the literal JSON shapes out of the generated instruction text."""

    shapes = []
    for line in tool_call_invocation_instructions().splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            # Trim any trailing prose after the JSON example.
            depth = 0
            for index, character in enumerate(stripped):
                depth += character == "{"
                depth -= character == "}"
                if depth == 0:
                    shapes.append(stripped[: index + 1])
                    break
    return shapes


def test_the_instructions_actually_contain_json_examples() -> None:
    shapes = _advertised_json_shapes()
    assert len(shapes) == 3, shapes


@pytest.mark.parametrize("shape", _advertised_json_shapes())
def test_every_advertised_shape_is_accepted_by_the_parser(shape: str) -> None:
    # Substitute the placeholders for a real call, keeping the structure.
    concrete = shape.replace('"<tool name>"', '"get_weather"').replace("{...}", '{"city": "Oslo"}')
    # The advertised text must be valid JSON once filled in.
    json.loads(concrete)

    result = parse_tool_calls(concrete, tools=[_TOOL])
    assert result.status == "parsed", f"{shape} -> {result.status}: {result.issues}"
    assert [call.name for call in result.tool_calls] == ["get_weather"]
    assert result.tool_calls[0].arguments == {"city": "Oslo"}


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("bare", '{"name": "get_weather", "arguments": {"city": "Oslo"}}'),
        ("single", '{"tool_call": {"name": "get_weather", "arguments": {"city": "Oslo"}}}'),
        ("batch", '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Oslo"}}]}'),
        ("fenced", '```json\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n```'),
        ("input alias", '{"name": "get_weather", "input": {"city": "Oslo"}}'),
    ],
)
def test_documented_shapes_parse(label: str, text: str) -> None:
    result = parse_tool_calls(text, tools=[_TOOL])
    assert result.status == "parsed", label


def test_the_arguments_only_shape_still_does_not_parse() -> None:
    # The parser is not loosened; the prompt is what changed.
    result = parse_tool_calls('{"city": "Oslo"}', tools=[_TOOL])
    assert result.status == "no_tool_calls"


def test_fenced_block_instruction_matches_the_parser_fence_language() -> None:
    instructions = tool_call_invocation_instructions()
    assert "```json" in instructions
    fenced = '```json\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n```'
    assert parse_tool_calls(fenced, tools=[_TOOL]).status == "parsed"


# --- the constants are genuinely shared --------------------------------------


def test_parser_and_prompt_read_the_same_constants() -> None:
    from lewlm import tool_calls

    assert tool_calls.TOOL_CALL_MARKER_KEYS is TOOL_CALL_MARKER_KEYS
    assert tool_calls.ARGUMENT_KEYS is ARGUMENT_KEYS
    assert tool_calls._ARGUMENT_KEYS is ARGUMENT_KEYS


def test_renaming_a_key_would_change_both_halves_together() -> None:
    # A structural guarantee: the instruction text is generated from the same
    # constants the parser branches on, so a rename cannot update only one side.
    instructions = tool_call_invocation_instructions()
    assert TOOL_CALL_SINGLE_KEY in instructions
    assert TOOL_CALL_BATCH_KEY in instructions
    assert TOOL_CALL_NAME_KEY in instructions
    for marker in TOOL_CALL_MARKER_KEYS:
        assert re.search(rf'"{re.escape(marker)}"', instructions)
