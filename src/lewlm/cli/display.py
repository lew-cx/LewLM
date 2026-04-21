"""CLI display and formatting helpers."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from lewlm.core.contracts import (
    ConversionStatus,
    ModelArtifactLayer,
    ModelArtifactRole,
    ModelInventory,
    ModelManifest,
    ModelModality,
)


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_ARTIFACT_ROLE_LABELS = {
    ModelArtifactRole.STANDALONE: "standalone",
    ModelArtifactRole.SOURCE_BUNDLE: "source bundle",
    ModelArtifactRole.MULTIMODAL_RUNNABLE: "multimodal runnable",
    ModelArtifactRole.TEXT_RUNNABLE: "text runnable",
}


@dataclass(slots=True)
class InventoryDisplayGroup:
    title: str
    source_path: str
    source_lineage: ModelArtifactLayer | None = None
    source_manifest: ModelManifest | None = None
    runnable_variants: list[ModelManifest] = field(default_factory=list)
    auxiliary_variants: list[ModelManifest] = field(default_factory=list)


def print_benchmark_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    alignments: tuple[str, ...] | None = None,
) -> None:
    resolved_alignments = alignments or ("left",) * len(headers)
    if len(resolved_alignments) != len(headers):
        raise ValueError("Benchmark table alignments must match the number of headers.")
    widths = [display_width(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], display_width(value))
    header_line = "  ".join(
        style(pad_display_text(header, widths[index], resolved_alignments[index]), "1")
        for index, header in enumerate(headers)
    )
    print(header_line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(pad_display_text(value, widths[index], resolved_alignments[index]) for index, value in enumerate(row)))


def inventory_display_lines(inventory: ModelInventory) -> list[str]:
    if not inventory.items:
        return ["No models registered."]
    groups = _build_inventory_display_groups(inventory.items)
    lines: list[str] = []
    for index, group in enumerate(groups):
        preferred_manifest = _preferred_manifest(group)
        advertises_converted = _advertises_converted(group)
        lines.append(_display_title(group, preferred_manifest=preferred_manifest, advertises_converted=advertises_converted))
        if preferred_manifest is not None:
            lines.append(f"  use: {preferred_manifest.model_id}")
        summary_manifest = preferred_manifest if advertises_converted else group.source_manifest or group.source_lineage
        if summary_manifest is not None:
            lines.append(f"  {'model' if advertises_converted else 'source'}: {_manifest_summary(summary_manifest)}")
        displayed_variants = _displayed_runnable_variants(
            group,
            preferred_manifest=preferred_manifest,
            advertises_converted=advertises_converted,
        )
        if displayed_variants:
            lines.append("  other runnable variants:" if advertises_converted else "  runnable variants:")
            for manifest in displayed_variants:
                lines.append(f"    - {_variant_summary(manifest)}")
        if group.auxiliary_variants:
            lines.append("  other artifacts:")
            for manifest in sorted(group.auxiliary_variants, key=_variant_sort_key):
                lines.append(f"    - {_variant_summary(manifest)}")
        path_lines = _path_lines(
            group,
            preferred_manifest=preferred_manifest,
            advertises_converted=advertises_converted,
        )
        lines.extend(path_lines)
        if index < len(groups) - 1:
            lines.append("")
    return lines


def display_width(value: str) -> int:
    return len(_ANSI_ESCAPE_RE.sub("", value))


def pad_display_text(value: str, width: int, alignment: str) -> str:
    padding = max(width - display_width(value), 0)
    if alignment == "right":
        return (" " * padding) + value
    return value + (" " * padding)


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}s"


def format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} tok/s"


def format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value > 0:
        return style(f"+{value:.4f}s", "32")
    if value < 0:
        return style(f"{value:.4f}s", "31")
    return f"{value:.4f}s"


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value > 0:
        return style(f"+{value:.2f}%", "32")
    if value < 0:
        return style(f"{value:.2f}%", "31")
    return f"{value:.2f}%"


def style(text: str, *codes: str) -> str:
    if not supports_color():
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    stream = getattr(sys, "stdout", None)
    return bool(stream is not None and hasattr(stream, "isatty") and stream.isatty())


def coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _build_inventory_display_groups(items: list[ModelManifest]) -> list[InventoryDisplayGroup]:
    grouped_source_model_ids = {
        source_model_id: f"source-model:{source_model_id}"
        for manifest in items
        if isinstance((source_model_id := manifest.metadata.get("source_model_id")), str) and source_model_id
    }
    grouped_source_paths = {
        source_lineage.source_path: f"source-path:{source_lineage.source_path}"
        for manifest in items
        if (source_lineage := _source_lineage(manifest)) is not None
    }
    grouped: dict[str, InventoryDisplayGroup] = {}
    for manifest in items:
        source_lineage = _source_lineage(manifest)
        group_key = _group_key(
            manifest,
            source_lineage=source_lineage,
            grouped_source_model_ids=grouped_source_model_ids,
            grouped_source_paths=grouped_source_paths,
        )
        source_path = source_lineage.source_path if source_lineage is not None else manifest.source_path
        group = grouped.get(group_key)
        if group is None:
            group = InventoryDisplayGroup(
                title=source_lineage.display_name if source_lineage is not None else manifest.display_name,
                source_path=source_path,
                source_lineage=source_lineage,
            )
            grouped[group_key] = group
        elif group.source_lineage is None and source_lineage is not None:
            group.source_lineage = source_lineage
        if _is_converted_runnable_manifest(manifest):
            group.runnable_variants.append(manifest)
            continue
        if manifest.artifact_role in {ModelArtifactRole.STANDALONE, ModelArtifactRole.SOURCE_BUNDLE}:
            if group.source_manifest is None or manifest.artifact_role == ModelArtifactRole.STANDALONE:
                group.source_manifest = manifest
            continue
        if manifest.artifact_role in {ModelArtifactRole.MULTIMODAL_RUNNABLE, ModelArtifactRole.TEXT_RUNNABLE}:
            group.runnable_variants.append(manifest)
            continue
        group.auxiliary_variants.append(manifest)
    return sorted(grouped.values(), key=lambda group: (group.title.casefold(), group.source_path))


def _source_lineage(manifest: ModelManifest) -> ModelArtifactLayer | None:
    for layer in manifest.artifact_lineage:
        if layer.role == ModelArtifactRole.SOURCE_BUNDLE:
            return layer
    return None


def _preferred_manifest(group: InventoryDisplayGroup) -> ModelManifest | None:
    if group.runnable_variants:
        source_modalities = (
            group.source_manifest.modality
            if group.source_manifest is not None
            else group.source_lineage.modality
            if group.source_lineage is not None
            else ()
        )
        if ModelModality.VISION in source_modalities or ModelModality.MULTIMODAL in source_modalities:
            for manifest in group.runnable_variants:
                if manifest.artifact_role == ModelArtifactRole.MULTIMODAL_RUNNABLE:
                    return manifest
        for manifest in group.runnable_variants:
            if manifest.artifact_role == ModelArtifactRole.TEXT_RUNNABLE:
                return manifest
        return group.runnable_variants[0]
    return group.source_manifest


def _advertises_converted(group: InventoryDisplayGroup) -> bool:
    return bool(group.runnable_variants)


def _display_title(
    group: InventoryDisplayGroup,
    *,
    preferred_manifest: ModelManifest | None,
    advertises_converted: bool,
) -> str:
    if advertises_converted and preferred_manifest is not None:
        return preferred_manifest.display_name
    return group.title


def _displayed_runnable_variants(
    group: InventoryDisplayGroup,
    *,
    preferred_manifest: ModelManifest | None,
    advertises_converted: bool,
) -> list[ModelManifest]:
    variants = sorted(group.runnable_variants, key=_variant_sort_key)
    if not advertises_converted or preferred_manifest is None:
        return variants
    return [manifest for manifest in variants if manifest.model_id != preferred_manifest.model_id]


def _manifest_summary(manifest: ModelManifest | ModelArtifactLayer | None) -> str:
    if manifest is None:
        return "unknown"
    modalities = ",".join(modality.value for modality in manifest.modality) or "unknown"
    affinities = ",".join(affinity.value for affinity in manifest.runtime_affinity) or "n/a"
    format_value = manifest.format_type.value
    return f"{format_value} [{modalities}] -> {affinities}"


def _variant_summary(manifest: ModelManifest) -> str:
    label = _ARTIFACT_ROLE_LABELS.get(manifest.artifact_role, manifest.artifact_role.value.replace("_", " "))
    return f"{label}: {_manifest_summary(manifest)} (use: {manifest.model_id})"


def _variant_sort_key(manifest: ModelManifest) -> tuple[int, str, str]:
    role_rank = {
        ModelArtifactRole.MULTIMODAL_RUNNABLE: 0,
        ModelArtifactRole.TEXT_RUNNABLE: 1,
    }.get(manifest.artifact_role, 2)
    return (role_rank, manifest.display_name.casefold(), manifest.model_id)


def _path_lines(
    group: InventoryDisplayGroup,
    *,
    preferred_manifest: ModelManifest | None,
    advertises_converted: bool,
) -> list[str]:
    if advertises_converted and preferred_manifest is not None:
        lines = [f"  path: {preferred_manifest.source_path}"]
        source_path = _source_manifest_path(group)
        if source_path is not None and source_path != preferred_manifest.source_path:
            lines.append(f"  source path: {source_path}")
        for manifest in sorted(group.runnable_variants, key=_variant_sort_key):
            if manifest.model_id == preferred_manifest.model_id or manifest.source_path == preferred_manifest.source_path:
                continue
            lines.append(f"  artifact path: {manifest.source_path}")
        return lines
    source_path = _source_manifest_path(group) or group.source_path
    if source_path is None:
        return []
    artifact_paths = [
        manifest.source_path
        for manifest in [*group.runnable_variants, *group.auxiliary_variants]
        if manifest.source_path != source_path
    ]
    if not artifact_paths:
        return [f"  path: {source_path}"]
    lines = [f"  source path: {source_path}"]
    for artifact_path in artifact_paths:
        lines.append(f"  artifact path: {artifact_path}")
    return lines


def _source_manifest_path(group: InventoryDisplayGroup) -> str | None:
    if group.source_manifest is not None:
        return group.source_manifest.source_path
    if group.source_lineage is not None:
        return group.source_lineage.source_path
    return None


def _group_key(
    manifest: ModelManifest,
    *,
    source_lineage: ModelArtifactLayer | None,
    grouped_source_model_ids: dict[str, str],
    grouped_source_paths: dict[str, str],
) -> str:
    source_model_id = manifest.metadata.get("source_model_id")
    if isinstance(source_model_id, str) and source_model_id:
        return grouped_source_model_ids.get(source_model_id, f"source-model:{source_model_id}")
    if manifest.model_id in grouped_source_model_ids:
        return grouped_source_model_ids[manifest.model_id]
    if source_lineage is not None:
        return f"source-path:{source_lineage.source_path}"
    if manifest.source_path in grouped_source_paths:
        return grouped_source_paths[manifest.source_path]
    return f"path:{manifest.source_path}"


def _is_converted_runnable_manifest(manifest: ModelManifest) -> bool:
    if manifest.conversion_status != ConversionStatus.RUNNABLE:
        return False
    return bool(manifest.metadata.get("converted_output")) or manifest.artifact_role in {
        ModelArtifactRole.MULTIMODAL_RUNNABLE,
        ModelArtifactRole.TEXT_RUNNABLE,
    }
