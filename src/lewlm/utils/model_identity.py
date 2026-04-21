"""Stable model identity helpers used across local and imported metadata."""

from __future__ import annotations

import re

from lewlm.core.contracts import ModelFormat, ModelManifest, ModelModality


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
