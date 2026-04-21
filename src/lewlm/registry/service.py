"""Model registry orchestration service."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import ModelArtifactLayer, ModelArtifactRole, ModelInventory, ModelManifest, ModelModality, ModelScanSummary
from lewlm.core.errors import ModelNotFoundError, ModelScanError
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.registry.discovery import discover_models
from lewlm.security.audit import AuditLogger
from lewlm.storage.metadata import MetadataStore


@dataclass(slots=True)
class _AdvertisedInventoryGroup:
    title: str
    source_path: str
    source_lineage: ModelArtifactLayer | None = None
    source_manifest: ModelManifest | None = None
    runnable_variants: list[ModelManifest] = field(default_factory=list)
    auxiliary_variants: list[ModelManifest] = field(default_factory=list)


class ModelRegistry:
    """Discover, persist, and query local model manifests."""

    def __init__(
        self,
        *,
        settings: LewLMSettings,
        metadata_store: MetadataStore,
        event_bus: EventBus,
        audit_logger: AuditLogger,
    ) -> None:
        self.settings = settings
        self.metadata_store = metadata_store
        self.event_bus = event_bus
        self.audit_logger = audit_logger

    def inventory(self) -> ModelInventory:
        manifests = self._advertised_manifests(self.list_manifests())
        return ModelInventory(count=len(manifests), items=manifests)

    def get_manifest(self, model_id: str) -> ModelManifest:
        manifest = self.metadata_store.get_model_manifest(model_id)
        if manifest is None:
            manifest = self._resolve_manifest_selector(model_id)
        if manifest is None:
            raise ModelNotFoundError("Requested model was not found in the local registry.", details={"model_id": model_id})
        return manifest

    def list_manifests(self) -> list[ModelManifest]:
        return self.metadata_store.list_model_manifests()

    def scan(self, roots: list[Path] | tuple[Path, ...] | None = None) -> ModelScanSummary:
        resolved_roots = self._resolve_roots(roots)
        self._emit_event(
            EventType.MODEL_SCAN_STARTED,
            {
                "roots": [str(root) for root in resolved_roots],
            },
        )
        existing_by_source = dict(self.metadata_store.list_model_manifest_records())
        manifests = discover_models(resolved_roots)
        discovered_sources = {manifest.source_path for manifest in manifests}

        new_count = 0
        updated_count = 0
        unchanged_count = 0
        for manifest in manifests:
            existing_fingerprint = existing_by_source.get(manifest.source_path)
            if existing_fingerprint is None:
                new_count += 1
            elif existing_fingerprint == manifest.fingerprint:
                unchanged_count += 1
            else:
                updated_count += 1

        stale_sources = [
            source_path
            for source_path in existing_by_source
            if self._is_under_roots(Path(source_path), resolved_roots) and source_path not in discovered_sources
        ]
        self.metadata_store.replace_model_manifests(manifests, stale_source_paths=stale_sources)
        summary = ModelScanSummary(
            roots_scanned=tuple(str(root) for root in resolved_roots),
            discovered_count=len(manifests),
            new_count=new_count,
            updated_count=updated_count,
            unchanged_count=unchanged_count,
            removed_count=len(stale_sources),
            manifests=manifests,
        )
        self.metadata_store.set_value("last_model_scan", summary.model_dump(mode="json"))
        self._emit_event(
            EventType.MODEL_SCAN_COMPLETED,
            {
                "roots": list(summary.roots_scanned),
                "discovered_count": summary.discovered_count,
                "new_count": summary.new_count,
                "updated_count": summary.updated_count,
                "unchanged_count": summary.unchanged_count,
                "removed_count": summary.removed_count,
            },
        )
        self.audit_logger.record(
            action="model_scan",
            outcome="completed",
            actor="service",
            details={
                "roots": list(summary.roots_scanned),
                "discovered_count": summary.discovered_count,
                "new_count": summary.new_count,
                "updated_count": summary.updated_count,
                "removed_count": summary.removed_count,
            },
        )
        return summary

    def _resolve_roots(self, roots: list[Path] | tuple[Path, ...] | None) -> tuple[Path, ...]:
        requested_roots = tuple(Path(root).expanduser().resolve(strict=False) for root in (roots or self.settings.models_dir))
        for root in requested_roots:
            if not root.exists():
                raise ModelScanError("Model root does not exist.", details={"path": str(root)})
            if not root.is_dir():
                raise ModelScanError("Model root is not a directory.", details={"path": str(root)})
        return requested_roots

    def _resolve_manifest_selector(self, selector: str) -> ModelManifest | None:
        normalized_selector = selector.strip()
        if not normalized_selector:
            return None
        manifests = self.list_manifests()
        selector_casefold = normalized_selector.casefold()
        for resolver in (
            lambda: self._resolve_unique(
                manifests,
                predicate=lambda manifest: manifest.display_name.casefold() == selector_casefold,
            ),
            lambda: self._resolve_unique(
                manifests,
                predicate=lambda manifest: Path(manifest.source_path).name.casefold() == selector_casefold,
            ),
            lambda: self._resolve_unique(
                manifests,
                predicate=lambda manifest: _slugify_manifest_selector(manifest.display_name) == selector_casefold,
            ),
            lambda: self._resolve_unique(
                manifests,
                predicate=lambda manifest: manifest.model_id.casefold().startswith(selector_casefold),
            ),
        ):
            match = resolver()
            if match is not None:
                return match
        return None

    @staticmethod
    def _resolve_unique(
        manifests: list[ModelManifest],
        *,
        predicate,
    ) -> ModelManifest | None:
        matches = [manifest for manifest in manifests if predicate(manifest)]
        if len(matches) == 1:
            return matches[0]
        return None

    def _advertised_manifests(self, manifests: list[ModelManifest]) -> list[ModelManifest]:
        if not manifests:
            return []
        grouped_source_model_ids = {
            source_model_id: f"source-model:{source_model_id}"
            for manifest in manifests
            if isinstance((source_model_id := manifest.metadata.get("source_model_id")), str) and source_model_id
        }
        grouped_source_paths = {
            source_lineage.source_path: f"source-path:{source_lineage.source_path}"
            for manifest in manifests
            if (source_lineage := self._source_lineage(manifest)) is not None
        }
        grouped: dict[str, _AdvertisedInventoryGroup] = {}
        for manifest in manifests:
            source_lineage = self._source_lineage(manifest)
            group_key = self._group_key(
                manifest,
                source_lineage=source_lineage,
                grouped_source_model_ids=grouped_source_model_ids,
                grouped_source_paths=grouped_source_paths,
            )
            source_path = source_lineage.source_path if source_lineage is not None else manifest.source_path
            group = grouped.get(group_key)
            if group is None:
                group = _AdvertisedInventoryGroup(
                    title=source_lineage.display_name if source_lineage is not None else manifest.display_name,
                    source_path=source_path,
                    source_lineage=source_lineage,
                )
                grouped[group_key] = group
            elif group.source_lineage is None and source_lineage is not None:
                group.source_lineage = source_lineage
            if self._is_converted_runnable_manifest(manifest):
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
        return [
            preferred_manifest
            for group in sorted(grouped.values(), key=lambda item: (item.title.casefold(), item.source_path))
            if (preferred_manifest := self._preferred_manifest(group)) is not None
        ]

    @staticmethod
    def _source_lineage(manifest: ModelManifest) -> ModelArtifactLayer | None:
        for layer in manifest.artifact_lineage:
            if layer.role == ModelArtifactRole.SOURCE_BUNDLE:
                return layer
        return None

    @classmethod
    def _group_key(
        cls,
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

    @staticmethod
    def _preferred_manifest(group: _AdvertisedInventoryGroup) -> ModelManifest | None:
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
        if group.source_manifest is not None:
            return group.source_manifest
        if group.auxiliary_variants:
            return sorted(group.auxiliary_variants, key=lambda item: (item.display_name.casefold(), item.model_id))[0]
        return None

    @staticmethod
    def _is_converted_runnable_manifest(manifest: ModelManifest) -> bool:
        return bool(manifest.metadata.get("converted_output")) or manifest.artifact_role in {
            ModelArtifactRole.MULTIMODAL_RUNNABLE,
            ModelArtifactRole.TEXT_RUNNABLE,
        }

    def _emit_event(self, event_type: EventType, payload: dict[str, object]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self.event_bus.publish(
                StreamEvent(type=event_type, scope=EventScope.SYSTEM, payload=payload),
            ),
        )

    @staticmethod
    def _is_under_roots(source_path: Path, roots: tuple[Path, ...]) -> bool:
        for root in roots:
            try:
                source_path.relative_to(root)
                return True
            except ValueError:
                continue
        return False


def _slugify_manifest_selector(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
