from __future__ import annotations

import struct
from pathlib import Path

from lewlm.registry.gguf_header import GGUFHeaderFacts, read_gguf_header_facts


TYPE_UINT32 = 4
TYPE_FLOAT32 = 6
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_UINT64 = 10


def _string(value: str) -> bytes:
    payload = value.encode("utf-8")
    return struct.pack("<Q", len(payload)) + payload


def _value(value_type: int, value: object) -> bytes:
    if value_type == TYPE_STRING:
        return _string(str(value))
    if value_type == TYPE_UINT32:
        return struct.pack("<I", int(value))
    if value_type == TYPE_UINT64:
        return struct.pack("<Q", int(value))
    if value_type == TYPE_FLOAT32:
        return struct.pack("<f", float(value))
    if value_type == TYPE_ARRAY:
        element_type, items = value  # type: ignore[misc]
        payload = struct.pack("<I", element_type) + struct.pack("<Q", len(items))
        return payload + b"".join(_value(element_type, item) for item in items)
    raise AssertionError(f"unsupported test value type {value_type}")


def _gguf_bytes(
    pairs: list[tuple[str, int, object]],
    *,
    magic: bytes = b"GGUF",
    version: int = 3,
    tensor_count: int = 0,
    declared_kv_count: int | None = None,
) -> bytes:
    header = magic + struct.pack("<I", version) + struct.pack("<Q", tensor_count)
    header += struct.pack("<Q", len(pairs) if declared_kv_count is None else declared_kv_count)
    for key, value_type, value in pairs:
        header += _string(key) + struct.pack("<I", value_type) + _value(value_type, value)
    return header


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def test_reads_architecture_and_context_length(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "model.gguf",
        _gguf_bytes(
            [
                ("general.architecture", TYPE_STRING, "gemma4"),
                ("general.name", TYPE_STRING, "Gemma 4 E4B"),
                ("gemma4.block_count", TYPE_UINT32, 42),
                ("gemma4.context_length", TYPE_UINT32, 131_072),
            ],
        ),
    )

    assert read_gguf_header_facts(path) == GGUFHeaderFacts(architecture="gemma4", context_length=131_072)


def test_stops_reading_once_both_facts_are_known(tmp_path: Path) -> None:
    """The tokenizer vocabulary is the bulk of a GGUF header and is never needed.

    Truncating the file immediately after the context length stands in for a
    multi-megabyte vocabulary: a reader that walked to the end would fail here.
    """

    complete = _gguf_bytes(
        [
            ("general.architecture", TYPE_STRING, "qwen3"),
            ("qwen3.context_length", TYPE_UINT32, 32_768),
            ("tokenizer.ggml.tokens", TYPE_ARRAY, (TYPE_STRING, ["a", "b", "c"])),
        ],
    )
    truncated = complete[: complete.index(b"tokenizer.ggml.tokens") - 8]
    path = _write(tmp_path / "truncated-after-context.gguf", truncated)

    assert read_gguf_header_facts(path) == GGUFHeaderFacts(architecture="qwen3", context_length=32_768)


def test_skips_arrays_that_precede_the_wanted_keys(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "arrays-first.gguf",
        _gguf_bytes(
            [
                ("tokenizer.ggml.tokens", TYPE_ARRAY, (TYPE_STRING, ["alpha", "beta", "gamma"])),
                ("tokenizer.ggml.scores", TYPE_ARRAY, (TYPE_FLOAT32, [0.5, 1.5, 2.5])),
                ("general.architecture", TYPE_STRING, "llama"),
                ("llama.context_length", TYPE_UINT64, 8_192),
            ],
        ),
    )

    assert read_gguf_header_facts(path) == GGUFHeaderFacts(architecture="llama", context_length=8_192)


def test_unambiguous_context_length_survives_a_missing_architecture(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "no-architecture.gguf",
        _gguf_bytes([("mystery.context_length", TYPE_UINT32, 4_096)]),
    )

    assert read_gguf_header_facts(path) == GGUFHeaderFacts(architecture=None, context_length=4_096)


def test_competing_context_lengths_without_an_architecture_are_declined(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "ambiguous.gguf",
        _gguf_bytes(
            [
                ("one.context_length", TYPE_UINT32, 4_096),
                ("two.context_length", TYPE_UINT32, 8_192),
            ],
        ),
    )

    assert read_gguf_header_facts(path).context_length is None


def test_architecture_selects_among_competing_context_lengths(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "adapter.gguf",
        _gguf_bytes(
            [
                ("clip.context_length", TYPE_UINT32, 77),
                ("general.architecture", TYPE_STRING, "gemma4"),
                ("gemma4.context_length", TYPE_UINT32, 131_072),
            ],
        ),
    )

    assert read_gguf_header_facts(path).context_length == 131_072


def test_zero_context_length_is_not_a_context_length(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "zero.gguf",
        _gguf_bytes(
            [
                ("general.architecture", TYPE_STRING, "llama"),
                ("llama.context_length", TYPE_UINT32, 0),
            ],
        ),
    )

    assert read_gguf_header_facts(path).context_length is None


def test_non_gguf_bytes_yield_no_facts(tmp_path: Path) -> None:
    assert read_gguf_header_facts(_write(tmp_path / "stub.gguf", b"gguf")) == GGUFHeaderFacts()
    assert read_gguf_header_facts(_write(tmp_path / "empty.gguf", b"")) == GGUFHeaderFacts()
    assert read_gguf_header_facts(tmp_path / "absent.gguf") == GGUFHeaderFacts()


def test_unsupported_version_is_declined_rather_than_guessed(tmp_path: Path) -> None:
    payload = _gguf_bytes(
        [
            ("general.architecture", TYPE_STRING, "llama"),
            ("llama.context_length", TYPE_UINT32, 8_192),
        ],
        version=1,
    )

    assert read_gguf_header_facts(_write(tmp_path / "v1.gguf", payload)) == GGUFHeaderFacts()


def test_truncated_header_yields_no_facts(tmp_path: Path) -> None:
    payload = _gguf_bytes(
        [
            ("general.architecture", TYPE_STRING, "llama"),
            ("llama.context_length", TYPE_UINT32, 8_192),
        ],
    )

    assert read_gguf_header_facts(_write(tmp_path / "cut.gguf", payload[:20])) == GGUFHeaderFacts()


def test_declared_lengths_larger_than_the_file_are_refused(tmp_path: Path) -> None:
    """A misread length must not become an allocation."""

    overlong_key = struct.pack("<Q", 1 << 40)
    payload = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1) + overlong_key
    assert read_gguf_header_facts(_write(tmp_path / "overlong.gguf", payload)) == GGUFHeaderFacts()

    implausible_kv_count = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1 << 40)
    assert read_gguf_header_facts(_write(tmp_path / "many.gguf", implausible_kv_count)) == GGUFHeaderFacts()


def test_unknown_value_type_stops_the_walk_without_raising(tmp_path: Path) -> None:
    payload = _gguf_bytes([("general.architecture", TYPE_STRING, "llama")])
    payload += _string("llama.rope.scaling") + struct.pack("<I", 99)
    payload = payload[:44] + struct.pack("<Q", 2) + payload[52:]

    assert read_gguf_header_facts(_write(tmp_path / "unknown-type.gguf", payload)) == GGUFHeaderFacts()
