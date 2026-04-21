from __future__ import annotations

from lewlm.core.contracts import GenerateMessage, GenerateRequest
from lewlm.runtime.llamacpp.runtime import LlamaCppRuntime
from lewlm.structured_output import GrammarResponseFormat, JSONSchemaResponseFormat


class _DummyLlamaClient:
    def create_chat_completion(
        self,
        *,
        messages,
        max_tokens,
        temperature,
        stream,
        grammar=None,
    ):
        return {}


def test_llamacpp_builds_decode_time_json_schema_grammar() -> None:
    runtime = LlamaCppRuntime()
    request = GenerateRequest(
        model_id="test-model",
        messages=[GenerateMessage(role="user", content="Return status")],
        structured_output=JSONSchemaResponseFormat(
            name="status",
            schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        ),
    )

    options = runtime._structured_output_options(request=request, client=_DummyLlamaClient())

    assert "grammar" in options
    assert request.metadata["structured_output_runtime"]["enforcement"] == "decode_time"
    assert request.metadata["structured_output_runtime"]["fallback_used"] is False


def test_llamacpp_reports_fallback_for_unsupported_grammar_syntax() -> None:
    runtime = LlamaCppRuntime()
    request = GenerateRequest(
        model_id="test-model",
        messages=[GenerateMessage(role="user", content="Return status")],
        structured_output=GrammarResponseFormat(
            grammar='root ::= "ok"',
            syntax="regex",
            name="status",
        ),
    )

    options = runtime._structured_output_options(request=request, client=_DummyLlamaClient())

    assert options == {}
    assert request.metadata["structured_output_runtime"]["enforcement"] == "prompt_guided"
    assert request.metadata["structured_output_runtime"]["fallback_used"] is True
    assert "expects `ebnf`/`gbnf`" in request.metadata["structured_output_runtime"]["fallback_reason"]
