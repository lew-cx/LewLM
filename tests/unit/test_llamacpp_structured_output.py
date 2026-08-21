from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lewlm.core.errors import InvalidRequestError
from lewlm.core.contracts import GenerateMessage, GenerateRequest
from lewlm.runtime.llamacpp.runtime import LlamaCppRuntime
from lewlm.structured_output import GrammarResponseFormat, JSONSchemaResponseFormat


class _DummyLlamaClient:
    #: A loaded llama.cpp client exposes the vocabulary LewLM parses a grammar
    #: against before handing it to the decoder.
    _model = SimpleNamespace(vocab=object())

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


def _fake_llama_cpp(
    *,
    to_gbnf=None,
    from_json_schema=None,
    accepts_grammar: bool = True,
) -> SimpleNamespace:
    """A stand-in for `llama_cpp` shaped like the real bindings.

    `to_gbnf` compiles a schema to grammar text the way `json_schema_to_gbnf`
    does; `accepts_grammar` decides whether the native parser stand-in accepts
    the result, which is the difference between a usable grammar and the null
    sampler that kills the process.
    """

    def default_to_gbnf(payload: str) -> str:
        properties = json.loads(payload).get("properties", {})
        bounds = "".join(f' "{name}"' for name in properties)
        return f'root ::= "{{"{bounds} "}}"\n'

    grammar_class = SimpleNamespace(
        from_string=lambda grammar, verbose=False: ("grammar", grammar, verbose),
    )
    if from_json_schema is not None:
        grammar_class.from_json_schema = from_json_schema
    else:
        grammar_class.from_json_schema = lambda payload, verbose=False: SimpleNamespace(
            _grammar=(to_gbnf or default_to_gbnf)(payload),
        )
    return SimpleNamespace(
        LlamaGrammar=grammar_class,
        llama_grammar=SimpleNamespace(json_schema_to_gbnf=to_gbnf or default_to_gbnf),
        llama_sampler_init_grammar=lambda vocab, text, root: ("sampler" if accepts_grammar else None),
        llama_sampler_free=lambda sampler: None,
    )


def test_llamacpp_builds_decode_time_json_schema_grammar(monkeypatch) -> None:
    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.runtime.import_module",
        lambda name: _fake_llama_cpp(),
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
    def reject_schema(payload: str) -> str:
        raise ValueError("unsupported schema")

    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.runtime.import_module",
        lambda name: _fake_llama_cpp(to_gbnf=reject_schema),
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


def _status_request(schema: dict) -> GenerateRequest:
    return GenerateRequest(
        model_id="test-model",
        messages=[GenerateMessage(role="user", content="Return status")],
        structured_output=JSONSchemaResponseFormat(name="status", schema=schema),
    )


def test_llamacpp_keeps_oversized_string_bounds_out_of_the_grammar(monkeypatch) -> None:
    """G30: a `maxLength` past the parser's ceiling must not reach the decoder.

    llama.cpp compiles a bounded string into one rule per permitted character
    and refuses past its complexity ceiling, and that refusal kills the process.
    The bound is dropped from the grammar, reported, and left to post-generation
    validation; the structure is still enforced at decode time.
    """

    seen: dict[str, str] = {}

    def record(payload: str) -> str:
        seen["payload"] = payload
        return 'root ::= "{" "summary" "}"\n'

    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.runtime.import_module",
        lambda name: _fake_llama_cpp(to_gbnf=record),
    )

    request = _status_request(
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "maxLength": 5000},
                "title": {"type": "string", "maxLength": 64},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    )

    options = LlamaCppRuntime()._structured_output_options(request=request, client=_DummyLlamaClient())

    compiled_schema = json.loads(seen["payload"])
    assert "maxLength" not in compiled_schema["properties"]["summary"]
    assert compiled_schema["properties"]["title"]["maxLength"] == 64
    assert "grammar" in options
    status = request.metadata["structured_output_runtime"]
    assert status["enforcement"] == "decode_time"
    assert status["grammar_relaxations"] == ["properties.summary.maxLength (5000)"]
    # The bound itself is still enforced, after generation rather than during it.
    assert request.structured_output.schema_payload["properties"]["summary"]["maxLength"] == 5000


def test_llamacpp_rejects_a_json_schema_grammar_the_parser_refuses(monkeypatch) -> None:
    """G30: a refused grammar is a request error, never a dead server."""

    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.runtime.import_module",
        lambda name: _fake_llama_cpp(
            to_gbnf=lambda payload: 'summary ::= "\\"" ' + "(char " * 4000 + ")?" * 4000 + '\nroot ::= summary\n',
            accepts_grammar=False,
        ),
    )

    request = _status_request(
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    )

    with pytest.raises(InvalidRequestError) as excinfo:
        LlamaCppRuntime()._structured_output_options(request=request, client=_DummyLlamaClient())

    assert excinfo.value.status_code == 422
    assert excinfo.value.details["response_format"] == "json_schema"
    assert excinfo.value.details["grammar_rule"] == "summary"


def test_llamacpp_rejects_a_caller_supplied_grammar_the_parser_refuses(monkeypatch) -> None:
    """G30: the same holds for a grammar a caller writes by hand."""

    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.runtime.import_module",
        lambda name: _fake_llama_cpp(accepts_grammar=False),
    )

    request = GenerateRequest(
        model_id="test-model",
        messages=[GenerateMessage(role="user", content="Return status")],
        structured_output=GrammarResponseFormat(
            grammar='root ::= item{0,100000}\nitem ::= "a"\n',
            syntax="gbnf",
            name="status",
        ),
    )

    with pytest.raises(InvalidRequestError) as excinfo:
        LlamaCppRuntime()._structured_output_options(request=request, client=_DummyLlamaClient())

    assert excinfo.value.details["response_format"] == "grammar"
    assert excinfo.value.details["grammar_rule"] == "root"


def test_llamacpp_refuses_a_bad_contract_before_a_response_starts(monkeypatch) -> None:
    """G30: a streaming request has nowhere to put an error once it has begun.

    `validate_structured_output` runs while the caller can still be answered, so
    a contract the decoder cannot be constrained to is a 422 rather than a
    stream that stops after zero bytes.
    """

    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.runtime.import_module",
        lambda name: _fake_llama_cpp(accepts_grammar=False),
    )
    runtime = LlamaCppRuntime()
    runtime._clients["test-model"] = _DummyLlamaClient()

    with pytest.raises(InvalidRequestError) as excinfo:
        runtime.validate_structured_output(
            GrammarResponseFormat(grammar='root ::= item{0,100000}\nitem ::= "a"\n', syntax="gbnf"),
            model_id="test-model",
        )

    assert excinfo.value.details["grammar_rule"] == "root"


def test_llamacpp_accepts_a_workable_contract_before_a_response_starts(monkeypatch) -> None:
    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.runtime.import_module",
        lambda name: _fake_llama_cpp(),
    )
    runtime = LlamaCppRuntime()
    runtime._clients["test-model"] = _DummyLlamaClient()

    assert (
        runtime.validate_structured_output(
            JSONSchemaResponseFormat(
                name="status",
                schema={
                    "type": "object",
                    "properties": {"summary": {"type": "string", "maxLength": 5000}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            ),
            model_id="test-model",
        )
        is None
    )
