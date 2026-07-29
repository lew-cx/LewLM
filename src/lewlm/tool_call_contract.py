"""The single source of truth for LewLM's tool-call wire shape.

Two halves have to agree: the prompt tells the model what to emit, and
`lewlm.tool_calls` decides what to accept. When only one half knows the
contract, tool calling fails *quietly* — the model emits bare arguments, no
candidate matches, and the result is reported as `no_tool_calls`, which is
indistinguishable from the model choosing not to call a tool.

This module is a leaf with no LewLM imports, so both the prompt compiler and
the parser can depend on it without a cycle, and neither can drift.
"""

from __future__ import annotations

#: Keys that identify a tool-call candidate.
TOOL_CALL_SINGLE_KEY = "tool_call"
TOOL_CALL_BATCH_KEY = "tool_calls"
TOOL_CALL_NAME_KEY = "name"

#: Accepted argument keys, most-preferred first. `input` is accepted because
#: prompt tool declarations advertise `input_schema`.
ARGUMENT_KEYS = ("arguments", "input")

#: Presence of any of these makes a JSON object worth parsing as a tool call.
TOOL_CALL_MARKER_KEYS = (TOOL_CALL_SINGLE_KEY, TOOL_CALL_BATCH_KEY, TOOL_CALL_NAME_KEY)

UNRECOGNIZED_SHAPE_MESSAGE = (
    f"Candidate JSON must be an object with a `{TOOL_CALL_SINGLE_KEY}`, "
    f"`{TOOL_CALL_BATCH_KEY}`, or `{TOOL_CALL_NAME_KEY}` shape."
)


def tool_call_example(*, tool_name: str = "<tool name>") -> str:
    """One accepted call object, rendered from the shared keys."""

    return f'{{"{TOOL_CALL_NAME_KEY}": "{tool_name}", "{ARGUMENT_KEYS[0]}": {{...}}}}'


def tool_call_invocation_instructions() -> str:
    """Describe, for the model, the exact output shape the parser accepts.

    Generated from the same constants the parser uses, so the prompt cannot
    advertise a shape that `parse_tool_calls` would reject.
    """

    primary, alias = ARGUMENT_KEYS[0], ARGUMENT_KEYS[1]
    call_shape = tool_call_example()
    return "\n".join(
        [
            "To call a tool, reply with a JSON object in one of these shapes and nothing else:",
            f"  {call_shape}",
            f'  {{"{TOOL_CALL_SINGLE_KEY}": {call_shape}}}',
            f'  {{"{TOOL_CALL_BATCH_KEY}": [{call_shape}]}}  (one entry per call, to request several at once)',
            f"The `{TOOL_CALL_NAME_KEY}` field is required and must match a declared tool exactly. "
            "A reply containing only arguments is not a tool call and will not be executed.",
            f"`{primary}` must be a JSON object satisfying that tool's input_schema "
            f"(`{alias}` is accepted as an alias).",
            "Emit the object on its own or inside a ```json fenced block.",
            "If no tool applies, answer normally and do not emit a tool-call object.",
        ],
    )
