"""GBNF compilation and preflight for the llama.cpp runtime.

llama.cpp expands a bounded repetition into one grammar rule per permitted
item, so a `"maxLength": 2000` string costs on the order of 2000 rules and the
parser refuses it past its own complexity ceiling::

    parse: error parsing grammar: number of rules that are going to be repeated
    multiplied by the new repetition exceeds sane defaults, please reduce the
    number of repetitions or rule complexity

That refusal is not survivable through the bindings. `llama_sampler_init_grammar`
returns a null sampler, the sampling chain accepts it anyway, and the next
sample dereferences it and takes the process down — so a single caller-supplied
`response_format` ends the server for every application on the host. A bound of
this size is not exotic: every code generator that emits JSON Schema from typed
models writes `maxLength` for bounded strings.

Two things here make that unreachable:

* :func:`relax_schema_bounds` keeps the bounds LewLM compiles into a grammar
  inside the parser's ceiling. A larger bound is dropped from the *grammar*
  only — `lewlm.structured_output` still enforces it against the generated text
  after the fact — and is reported so a caller can see which constraint was not
  enforced at decode time.
* :func:`preflight_grammar` parses the finished grammar with llama.cpp's own
  parser and frees it again, before any of it can reach a sampling chain. What
  the parser refuses becomes an `invalid_request` naming the offending rule
  instead of a dead process.
"""

from __future__ import annotations

import ctypes
import json
import re
from dataclasses import dataclass
from typing import Any


#: Largest bound LewLM will compile into a grammar. Measured against the
#: packaged llama.cpp: a bounded JSON string rule parses at 1000 repetitions and
#: is refused by 1020. The real ceiling is `rule size x repetitions`, so a bound
#: on a more complex item rule fails sooner than a bound on a plain character —
#: hence a ceiling at half of what a plain character survives.
MAX_GRAMMAR_REPETITION = 512

#: Where llama.cpp itself stops. Used only for the static estimate, which backs
#: up the native preflight when there is no vocabulary to parse against.
PARSER_REPETITION_LIMIT = 1023

#: Schema keywords whose value becomes a repetition count in the compiled
#: grammar. `minLength`/`minItems` are as costly as their maxima: the converter
#: emits the item rule verbatim that many times.
_BOUNDED_KEYWORDS = ("maxLength", "minLength", "maxItems", "minItems")

_RULE_NAME = re.compile(r"^\s*([A-Za-z0-9\-_]+)\s*::=")
_OPTIONAL_RUN = re.compile(r"(?:\)\s*\?\s*)+")
_REPETITION = re.compile(r"\{\s*(\d+)?\s*(?:,\s*(\d+)?\s*)?\}")


@dataclass(frozen=True)
class RelaxedBound:
    """A schema bound LewLM left out of the grammar to stay parseable."""

    path: str
    keyword: str
    value: int

    def describe(self) -> str:
        location = f"{self.path}.{self.keyword}" if self.path else self.keyword
        return f"{location} ({self.value})"


@dataclass(frozen=True)
class CompiledGrammar:
    """A grammar that llama.cpp's parser has already accepted."""

    text: str
    relaxed_bounds: tuple[RelaxedBound, ...] = ()


class GrammarError(Exception):
    """Base class for grammar-compilation failures."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class GrammarUnsupportedError(GrammarError):
    """Raised when llama.cpp will not parse the grammar a request asks for.

    This is a caller error — the request describes an output the decoder cannot
    be constrained to — and the runtime turns it into `invalid_request`.
    """


class GrammarUnverifiableError(GrammarError):
    """Raised when LewLM cannot get a grammar into a form it can check first.

    Not a caller error: the installed bindings are the limitation. The runtime
    falls back to prompt-guided output rather than send a decoder something it
    has not parsed.
    """


def relax_schema_bounds(
    schema: Any,
    *,
    limit: int = MAX_GRAMMAR_REPETITION,
) -> tuple[Any, tuple[RelaxedBound, ...]]:
    """Return `schema` with oversized repetition bounds removed.

    The returned copy is for grammar compilation only. The caller keeps the
    original for post-generation validation, so a dropped bound is still
    enforced — just after the tokens rather than during them.
    """

    found: list[RelaxedBound] = []
    return _relax(schema, path=(), limit=limit, found=found), tuple(found)


def _relax(node: Any, *, path: tuple[str, ...], limit: int, found: list[RelaxedBound]) -> Any:
    if isinstance(node, list):
        return [_relax(item, path=(*path, str(index)), limit=limit, found=found) for index, item in enumerate(node)]
    if not isinstance(node, dict):
        return node
    relaxed: dict[str, Any] = {}
    for key, value in node.items():
        if key in _BOUNDED_KEYWORDS and _is_count(value) and int(value) > limit:
            found.append(RelaxedBound(path=".".join(path), keyword=str(key), value=int(value)))
            continue
        relaxed[key] = _relax(value, path=(*path, str(key)), limit=limit, found=found)
    return relaxed


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def compile_json_schema_grammar(
    schema: dict[str, Any],
    *,
    llama_cpp: Any,
    vocab: Any = None,
    limit: int = MAX_GRAMMAR_REPETITION,
) -> CompiledGrammar:
    """Compile `schema` to a GBNF grammar llama.cpp has agreed to parse.

    Raises `GrammarUnsupportedError` if the grammar is refused, and whatever the
    installed converter raises (`TypeError`/`ValueError`) if the schema uses a
    feature it cannot express — the latter is a fallback to prompt-guided
    output, not a request error.
    """

    relaxed_schema, relaxed_bounds = relax_schema_bounds(schema, limit=limit)
    text = json_schema_grammar_text(relaxed_schema, llama_cpp=llama_cpp)
    preflight_grammar(text, llama_cpp=llama_cpp, vocab=vocab)
    return CompiledGrammar(text=text, relaxed_bounds=relaxed_bounds)


def json_schema_grammar_text(schema: dict[str, Any], *, llama_cpp: Any) -> str:
    """Convert a JSON schema to GBNF text using the installed bindings."""

    payload = json.dumps(schema, sort_keys=True)
    grammar_module = getattr(llama_cpp, "llama_grammar", None)
    converter = getattr(grammar_module, "json_schema_to_gbnf", None)
    if callable(converter):
        return str(converter(payload))

    grammar_class = getattr(llama_cpp, "LlamaGrammar", None)
    factory = getattr(grammar_class, "from_json_schema", None)
    if not callable(factory):
        raise GrammarUnverifiableError(
            "Installed llama.cpp bindings do not expose JSON-schema grammar compilation for decode-time "
            "constrained decoding.",
        )
    compiled = factory(payload, verbose=False)
    text = getattr(compiled, "_grammar", None)
    if not isinstance(text, str):
        raise GrammarUnverifiableError(
            "Installed llama.cpp bindings compiled the JSON schema to a grammar LewLM cannot inspect, "
            "so it cannot be checked before it reaches the decoder.",
        )
    return text


def preflight_grammar(text: str, *, llama_cpp: Any, vocab: Any = None) -> None:
    """Parse `text` with llama.cpp's parser and free it again.

    The parse is the whole point: it is the only way to know that a grammar is
    acceptable, and it is safe here because a refusal is reported as a null
    sampler that is never added to a chain. Without a vocabulary to parse
    against — no model loaded, or bindings that predate the sampler API — the
    static estimate below stands in.
    """

    init = getattr(llama_cpp, "llama_sampler_init_grammar", None)
    free = getattr(llama_cpp, "llama_sampler_free", None)
    if vocab is None or not callable(init) or not callable(free):
        _preflight_statically(text)
        return
    try:
        sampler = init(vocab, text.encode("utf-8"), b"root")
    except Exception:  # pragma: no cover - binding mismatch, not a caller error
        _preflight_statically(text)
        return
    if _is_null(sampler):
        rule, repetitions = worst_repetition(text)
        raise GrammarUnsupportedError(
            "llama.cpp refused to parse the grammar this request compiles to.",
            details=_refusal_details(rule=rule, repetitions=repetitions),
        )
    free(sampler)


def _preflight_statically(text: str) -> None:
    rule, repetitions = worst_repetition(text)
    if repetitions > PARSER_REPETITION_LIMIT:
        raise GrammarUnsupportedError(
            "The grammar this request compiles to exceeds llama.cpp's grammar-parser complexity ceiling.",
            details=_refusal_details(rule=rule, repetitions=repetitions),
        )


def _refusal_details(*, rule: str | None, repetitions: int) -> dict[str, Any]:
    details: dict[str, Any] = {"parser_repetition_limit": PARSER_REPETITION_LIMIT}
    if rule is not None:
        details["grammar_rule"] = rule
    if repetitions:
        details["repetitions"] = repetitions
    return details


def worst_repetition(text: str) -> tuple[str | None, int]:
    """Return the most-repeated rule in a GBNF grammar and its repetition count.

    Both spellings count: the nested-optional expansion the installed converter
    emits for a bounded string, and an explicit `{m,n}` repetition operator.
    """

    worst_rule: str | None = None
    worst_count = 0
    for name, body in _rules(text):
        count = _repetition_cost(body)
        if count > worst_count:
            worst_rule, worst_count = name, count
    return worst_rule, worst_count


def _rules(text: str) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = _RULE_NAME.match(line)
        if match:
            rules.append((match.group(1), line[match.end() :]))
        elif rules:
            name, body = rules[-1]
            rules[-1] = (name, f"{body}\n{line}")
    return rules


def _repetition_cost(body: str) -> int:
    nested = max((run.group(0).count("?") for run in _OPTIONAL_RUN.finditer(body)), default=0)
    explicit = 0
    for match in _REPETITION.finditer(body):
        explicit = max(explicit, *(int(group) for group in match.groups() if group is not None), 0)
    return max(nested, explicit)


def _is_null(pointer: Any) -> bool:
    if pointer is None:
        return True
    try:
        return not ctypes.cast(pointer, ctypes.c_void_p).value
    except (TypeError, ctypes.ArgumentError):  # pragma: no cover - non-pointer stand-in
        return False


def vocab_pointer(client: Any) -> Any:
    """Return the vocabulary of a loaded llama.cpp client, if it exposes one."""

    return getattr(getattr(client, "_model", None), "vocab", None)
