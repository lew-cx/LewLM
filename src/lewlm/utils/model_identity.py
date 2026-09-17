"""Stable model identity helpers used across local and imported metadata.

Also home to the URI-backed source namespace. A manifest whose ``source_path``
is a URI (``ollama://<tag>`` or ``external://<endpoint_id>/<upstream_id>``)
has no file behind it: it names a model an operator-managed server advertises.
Every code path that would open ``source_path`` as a file -- hashing,
conversion, local weight loading -- checks :func:`is_uri_source` first.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import quote, unquote

from lewlm.core.contracts import ModelFormat, ModelManifest, ModelModality

OLLAMA_SOURCE_PREFIX = "ollama://"
EXTERNAL_SOURCE_SCHEME = "external"
EXTERNAL_SOURCE_PREFIX = f"{EXTERNAL_SOURCE_SCHEME}://"
_URI_SOURCE_PREFIXES = (OLLAMA_SOURCE_PREFIX, EXTERNAL_SOURCE_PREFIX)


def is_external_source(source_path: str) -> bool:
    """Whether ``source_path`` names a model advertised by a named external endpoint."""

    return source_path.startswith(EXTERNAL_SOURCE_PREFIX)


def is_uri_source(source_path: str) -> bool:
    """Whether ``source_path`` is any server-advertised URI rather than a local file."""

    return source_path.startswith(_URI_SOURCE_PREFIXES)


def external_source_path(endpoint_id: str, upstream_model_id: str) -> str:
    """Build the ``external://<endpoint_id>/<encoded-upstream-id>`` source URI.

    The upstream id is percent-encoded in full (``/`` included) so an id such
    as ``org/model`` stays one path segment and round-trips exactly.
    """

    return f"{EXTERNAL_SOURCE_PREFIX}{endpoint_id}/{quote(upstream_model_id, safe='')}"


def parse_external_source(source_path: str) -> tuple[str, str] | None:
    """Return ``(endpoint_id, upstream_model_id)`` for an ``external://`` URI, else ``None``."""

    if not is_external_source(source_path):
        return None
    remainder = source_path[len(EXTERNAL_SOURCE_PREFIX):]
    endpoint_id, separator, encoded = remainder.partition("/")
    if not endpoint_id or not separator or not encoded:
        return None
    return endpoint_id, unquote(encoded)


def external_model_id(endpoint_id: str, upstream_model_id: str) -> str:
    """Stable, URL-safe LewLM model id for an endpoint-advertised model.

    The slug keeps ids readable; the digest keeps two upstream ids that slug
    identically (``Org/Model-1`` and ``org-model_1``) distinct, and keeps the
    same upstream name on two endpoints distinct. Nothing here claims the
    artifact is the same as any other endpoint's -- the endpoint is part of
    the identity on purpose.
    """

    digest = hashlib.sha256(f"{endpoint_id}\x00{upstream_model_id}".encode("utf-8")).hexdigest()
    return f"{_slug(upstream_model_id) or 'model'}-{_slug(endpoint_id) or 'endpoint'}-{digest[:8]}"


def build_model_validation_key(
    *,
    display_name: str,
    format_type: ModelFormat | str,
    architecture_family: str,
    quantization: str | None,
    modality: tuple[ModelModality, ...] | list[ModelModality] | tuple[str, ...] | list[str],
) -> str:
    """Build a cross-host model identity key from stable metadata."""

    slug = _slug(display_name) or "model"
    format_value = format_type.value if isinstance(format_type, ModelFormat) else str(format_type)
    architecture_value = _slug(architecture_family) or "unknown"
    quantization_value = _slug(quantization or "na") or "na"
    modality_values = sorted(
        item.value if isinstance(item, ModelModality) else str(item)
        for item in modality
    )
    modality_value = _slug("-".join(modality_values)) or "unknown"
    return f"{slug}:{format_value}:{architecture_value}:{quantization_value}:{modality_value}"


def build_manifest_validation_key(manifest: ModelManifest) -> str:
    """Build the stable validation key for a discovered manifest."""

    return build_model_validation_key(
        display_name=manifest.display_name,
        format_type=manifest.format_type,
        architecture_family=manifest.architecture_family,
        quantization=manifest.quantization,
        modality=manifest.modality,
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
