"""G30: no caller-supplied schema or grammar may reach an unparsed decoder."""

from __future__ import annotations


import pytest

from lewlm.runtime.llamacpp import grammar as grammar_support


def test_relax_schema_bounds_drops_only_the_oversized_bounds() -> None:
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "maxLength": 5000},
            "title": {"type": "string", "maxLength": 64, "minLength": 2},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 9000},
        },
    }

    relaxed, bounds = grammar_support.relax_schema_bounds(schema)

    assert "maxLength" not in relaxed["properties"]["summary"]
    assert relaxed["properties"]["title"] == {"type": "string", "maxLength": 64, "minLength": 2}
    assert "maxItems" not in relaxed["properties"]["tags"]
    assert [bound.describe() for bound in bounds] == [
        "properties.summary.maxLength (5000)",
        "properties.tags.maxItems (9000)",
    ]
    # The caller's schema is untouched; it still drives post-generation validation.
    assert schema["properties"]["summary"]["maxLength"] == 5000


def test_relax_schema_bounds_leaves_a_property_named_like_a_keyword_alone() -> None:
    schema = {"type": "object", "properties": {"maxLength": {"type": "integer"}}}

    relaxed, bounds = grammar_support.relax_schema_bounds(schema)

    assert relaxed == schema
    assert bounds == ()


def test_worst_repetition_reads_both_spellings_of_a_bounded_repeat() -> None:
    nested = 'summary ::= "\\"" ' + "(char " * 30 + ")?" * 30 + "\n"
    explicit = "title ::= char{0,700}\n"

    assert grammar_support.worst_repetition(nested + explicit) == ("title", 700)
    assert grammar_support.worst_repetition('root ::= "ok"\n') == (None, 0)


def test_preflight_refuses_a_grammar_past_the_parser_ceiling_without_a_vocab() -> None:
    text = "root ::= char{0,50000}\n"

    with pytest.raises(grammar_support.GrammarUnsupportedError) as excinfo:
        grammar_support.preflight_grammar(text, llama_cpp=object(), vocab=None)

    assert excinfo.value.details["grammar_rule"] == "root"
    assert excinfo.value.details["repetitions"] == 50000


def test_preflight_reports_a_null_sampler_as_a_refusal() -> None:
    class _Bindings:
        @staticmethod
        def llama_sampler_init_grammar(vocab, text, root):
            return None

        @staticmethod
        def llama_sampler_free(sampler):  # pragma: no cover - never reached
            raise AssertionError("a refused grammar has nothing to free")

    with pytest.raises(grammar_support.GrammarUnsupportedError):
        grammar_support.preflight_grammar('root ::= "ok"\n', llama_cpp=_Bindings, vocab=object())


def test_preflight_frees_the_sampler_it_parsed_with() -> None:
    freed: list[object] = []

    class _Bindings:
        @staticmethod
        def llama_sampler_init_grammar(vocab, text, root):
            return "sampler"

        @staticmethod
        def llama_sampler_free(sampler):
            freed.append(sampler)

    grammar_support.preflight_grammar('root ::= "ok"\n', llama_cpp=_Bindings, vocab=object())

    assert freed == ["sampler"]


def test_installed_bindings_compile_a_generator_written_schema() -> None:
    """The schema shape that killed the server: bounded strings from a code generator."""

    llama_cpp = pytest.importorskip("llama_cpp")
    schema = {
        "type": "object",
        "properties": {f"field_{index}": {"type": "string", "maxLength": 5000} for index in range(7)},
        "required": [f"field_{index}" for index in range(7)],
        "additionalProperties": False,
    }

    compiled = grammar_support.compile_json_schema_grammar(schema, llama_cpp=llama_cpp)

    assert [bound.keyword for bound in compiled.relaxed_bounds] == ["maxLength"] * 7
    _, repetitions = grammar_support.worst_repetition(compiled.text)
    assert repetitions <= grammar_support.PARSER_REPETITION_LIMIT
