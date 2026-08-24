"""Read a GGUF file's metadata header without loading the model.

A GGUF file opens with a fixed preamble followed by its metadata key/value
block, so the facts discovery wants — the architecture the file was built for
and the context window it was trained with — sit in the first bytes of the file
and need no runtime, no tensor read and no optional dependency to recover.
Recording them is what keeps `context_length` off `null` for a bundle that has
no `config.json` to read it from.

The reader is deliberately total. Discovery walks whatever a model root happens
to contain, and a `.gguf` suffix is not a promise about the bytes underneath it,
so a truncated file, an unrecognized version, an unknown value type or a length
that does not fit inside the file yields an empty result rather than an
exception. Every declared length is checked against the file size before it is
used to read or seek, so a corrupt header cannot make the reader allocate for it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


GGUF_MAGIC = b"GGUF"
# v2 widened every length prefix from 32 to 64 bits and v3 changed only tensor
# payloads, so the two share one header parser. v1 is a different layout and is
# declined rather than guessed at.
SUPPORTED_GGUF_VERSIONS = (2, 3)

_ARCHITECTURE_KEY = "general.architecture"
_CONTEXT_LENGTH_SUFFIX = ".context_length"

_TYPE_UINT8 = 0
_TYPE_INT8 = 1
_TYPE_UINT16 = 2
_TYPE_INT16 = 3
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10
_TYPE_INT64 = 11
_TYPE_FLOAT64 = 12

# Scalar value types, mapped to their `struct` format. `STRING` and `ARRAY`
# carry their own length prefixes and are handled separately.
_SCALAR_FORMATS = {
    _TYPE_UINT8: "<B",
    _TYPE_INT8: "<b",
    _TYPE_UINT16: "<H",
    _TYPE_INT16: "<h",
    _TYPE_UINT32: "<I",
    _TYPE_INT32: "<i",
    _TYPE_FLOAT32: "<f",
    _TYPE_BOOL: "<?",
    _TYPE_UINT64: "<Q",
    _TYPE_INT64: "<q",
    _TYPE_FLOAT64: "<d",
}
_INTEGER_TYPES = frozenset(
    {
        _TYPE_UINT8,
        _TYPE_INT8,
        _TYPE_UINT16,
        _TYPE_INT16,
        _TYPE_UINT32,
        _TYPE_INT32,
        _TYPE_UINT64,
        _TYPE_INT64,
    },
)

# A key is an identifier, not a payload. Anything longer than this is a sign the
# length prefix was not a length.
_MAX_KEY_BYTES = 1024


class _MalformedHeader(Exception):
    """Raised internally when the bytes stop describing a GGUF header."""


@dataclass(frozen=True)
class GGUFHeaderFacts:
    """What discovery recovers from a GGUF metadata header."""

    architecture: str | None = None
    context_length: int | None = None


def read_gguf_header_facts(path: Path) -> GGUFHeaderFacts:
    """Return the architecture and context length declared by a GGUF file.

    Missing or unreadable values come back as `None`; nothing raises.
    """

    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            return _read_facts(handle, file_size=file_size)
    except (OSError, _MalformedHeader, struct.error):
        return GGUFHeaderFacts()


def _read_facts(handle: BinaryIO, *, file_size: int) -> GGUFHeaderFacts:
    if _read_exact(handle, 4) != GGUF_MAGIC:
        return GGUFHeaderFacts()
    version = _read_scalar(handle, "<I")
    if version not in SUPPORTED_GGUF_VERSIONS:
        return GGUFHeaderFacts()
    _read_scalar(handle, "<Q")  # tensor count; the tensor block is never visited
    kv_count = _read_scalar(handle, "<Q")
    # Every pair costs at least a length prefix, a type tag and a value, so a
    # count that could not fit in the file is a misread rather than a big model.
    if kv_count * 16 > file_size:
        return GGUFHeaderFacts()

    architecture: str | None = None
    # `<arch>.context_length` cannot be matched until the architecture is known,
    # and key order is not guaranteed, so every context length seen is kept until
    # one of them turns out to be the architecture's own.
    context_lengths: dict[str, int] = {}
    for _ in range(kv_count):
        key = _read_string(handle, file_size=file_size, max_bytes=_MAX_KEY_BYTES)
        value_type = _read_scalar(handle, "<I")
        if key == _ARCHITECTURE_KEY and value_type == _TYPE_STRING:
            architecture = _read_string(handle, file_size=file_size, max_bytes=_MAX_KEY_BYTES)
        elif key.endswith(_CONTEXT_LENGTH_SUFFIX) and value_type in _INTEGER_TYPES:
            context_lengths[key] = _read_scalar(handle, _SCALAR_FORMATS[value_type])
        else:
            _skip_value(handle, value_type, file_size=file_size)
        # The tokenizer vocabulary is the bulk of a GGUF header and always
        # follows the architecture block, so leaving as soon as both facts are
        # in hand keeps discovery reading kilobytes instead of megabytes.
        if architecture is not None and f"{architecture}{_CONTEXT_LENGTH_SUFFIX}" in context_lengths:
            break

    context_length = context_lengths.get(f"{architecture}{_CONTEXT_LENGTH_SUFFIX}") if architecture else None
    if context_length is None and len(context_lengths) == 1:
        # A file that names no architecture, or names one that does not own the
        # only context length present, still leaves that value unambiguous.
        context_length = next(iter(context_lengths.values()))
    return GGUFHeaderFacts(
        architecture=architecture or None,
        context_length=context_length if context_length and context_length > 0 else None,
    )


def _read_exact(handle: BinaryIO, count: int) -> bytes:
    payload = handle.read(count)
    if len(payload) != count:
        raise _MalformedHeader("GGUF header ended early.")
    return payload


def _read_scalar(handle: BinaryIO, fmt: str) -> int | float | bool:
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))[0]


def _read_length(handle: BinaryIO, *, file_size: int, max_bytes: int | None = None) -> int:
    length = _read_scalar(handle, "<Q")
    ceiling = file_size if max_bytes is None else min(file_size, max_bytes)
    if not isinstance(length, int) or length < 0 or length > ceiling:
        raise _MalformedHeader("GGUF header declares a length the file cannot hold.")
    return length


def _read_string(handle: BinaryIO, *, file_size: int, max_bytes: int | None = None) -> str:
    length = _read_length(handle, file_size=file_size, max_bytes=max_bytes)
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _skip_value(handle: BinaryIO, value_type: int, *, file_size: int) -> None:
    if value_type == _TYPE_STRING:
        _seek_forward(handle, _read_length(handle, file_size=file_size), file_size=file_size)
        return
    if value_type == _TYPE_ARRAY:
        element_type = _read_scalar(handle, "<I")
        count = _read_length(handle, file_size=file_size)
        if element_type == _TYPE_STRING:
            for _ in range(count):
                _seek_forward(handle, _read_length(handle, file_size=file_size), file_size=file_size)
            return
        if element_type == _TYPE_ARRAY:
            # Nested arrays carry no fixed stride, so each element is walked.
            for _ in range(count):
                _skip_value(handle, _TYPE_ARRAY, file_size=file_size)
            return
        fmt = _SCALAR_FORMATS.get(element_type)
        if fmt is None:
            raise _MalformedHeader(f"GGUF array declares unknown element type {element_type}.")
        _seek_forward(handle, count * struct.calcsize(fmt), file_size=file_size)
        return
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise _MalformedHeader(f"GGUF header declares unknown value type {value_type}.")
    _seek_forward(handle, struct.calcsize(fmt), file_size=file_size)


def _seek_forward(handle: BinaryIO, count: int, *, file_size: int) -> None:
    if count < 0:
        raise _MalformedHeader("GGUF header declares a negative span.")
    position = handle.seek(count, 1)
    if position > file_size:
        raise _MalformedHeader("GGUF header skips past the end of the file.")
