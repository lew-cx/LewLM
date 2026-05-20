from __future__ import annotations

from types import SimpleNamespace

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


def test_llamacpp_builds_decode_time_json_schema_grammar(monkeypatch) -> None:
    fake_grammar = SimpleNamespace(
        from_json_schema=lambda schema, verbose=False: ("json", schema, verbose),
        from_string=lambda grammar, verbose=False: ("grammar", grammar, verbose),
    )
    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.runtime.import_module",
        lambda name: SimpleNamespace(LlamaGrammar=fake_grammar),
    )

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


def test_llamacpp_reports_fallback_when_chat_completion_lacks_grammar_parameter() -> None:
    class NoGrammarClient:
        def create_chat_completion(
            self,
            *,
            messages,
            max_tokens,
            temperature,
            stream,
        ):
            return {}

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

    options = runtime._structured_output_options(request=request, client=NoGrammarClient())

    assert options == {}
    assert request.metadata["structured_output_runtime"]["enforcement"] == "prompt_guided"
    assert request.metadata["structured_output_runtime"]["fallback_used"] is True
    assert "`grammar` parameter" in request.metadata["structured_output_runtime"]["fallback_reason"]


def test_llamacpp_falls_back_when_json_schema_grammar_factory_rejects_schema(monkeypatch) -> None:
    def reject_schema(schema, verbose=False):
        raise ValueError("unsupported schema")

    fake_grammar = SimpleNamespace(
        from_json_schema=reject_schema,
        from_string=lambda grammar, verbose=False: ("grammar", grammar, verbose),
    )
    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.runtime.import_module",
        lambda name: SimpleNamespace(LlamaGrammar=fake_grammar),
    )

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

    assert options == {}
    assert request.metadata["structured_output_runtime"]["enforcement"] == "prompt_guided"
    assert request.metadata["structured_output_runtime"]["fallback_used"] is True
    assert "unsupported schema" in request.metadata["structured_output_runtime"]["fallback_reason"]
