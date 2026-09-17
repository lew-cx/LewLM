"""Model registry orchestration service."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import ModelArtifactLayer, ModelArtifactRole, ModelInventory, ModelManifest, ModelModality, ModelScanSummary
from lewlm.core.errors import ModelNotFoundError, ModelScanError
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.registry.discovery import discover_models
from lewlm.registry.external_inventory import ExternalInventoryResult, discover_external_models
from lewlm.registry.ollama_inventory import OllamaInventoryResult, discover_ollama_models, is_ollama_source
from lewlm.utils.model_identity import is_external_source, is_uri_source, parse_external_source
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
        # Named endpoint runtimes are built after the registry (they need the
        # catalog); bootstrap binds them so `scan` can read `/v1/models` through
        # the same runtime that will serve the requests. Unbound means no
        # external inventory, never an error.
        self._endpoint_runtime_provider: Callable[[], Mapping[str, Any]] | None = None

    def bind_endpoint_runtimes(self, provider: Callable[[], Mapping[str, Any]] | None) -> None:
        self._endpoint_runtime_provider = provider

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
        existing_by_source = {
            manifest.source_path: manifest for manifest in self.metadata_store.list_model_manifests()
        }
        manifests = discover_models(resolved_roots)
        ollama_result = self._discover_ollama_models()
        if ollama_result is not None:
            manifests.extend(ollama_result.manifests)
        external_results = self._discover_external_models()
        for external_result in external_results:
            manifests.extend(external_result.manifests)
        discovered_sources = {manifest.source_path for manifest in manifests}

        new_count = 0
        updated_count = 0
        unchanged_count = 0
        for manifest in manifests:
            existing_manifest = existing_by_source.get(manifest.source_path)
            if existing_manifest is None:
                new_count += 1
            elif _manifest_change_signature(existing_manifest) == _manifest_change_signature(manifest):
                unchanged_count += 1
            else:
                # A scan rewrites every manifest it discovers, so anything the
                # rewrite would change counts as an update — not just a changed
                # fingerprint. Otherwise a scan reports no change while the
                # registry moves under the caller.
                updated_count += 1

        stale_sources = [
            source_path
            for source_path in existing_by_source
            if not is_uri_source(source_path)
            and self._is_under_roots(Path(source_path), resolved_roots)
            and source_path not in discovered_sources
        ]
        stale_sources.extend(
            self._stale_ollama_sources(existing_by_source, discovered_sources, ollama_result),
        )
        stale_sources.extend(
            self._stale_external_sources(existing_by_source, discovered_sources, external_results),
        )
        self.metadata_store.replace_model_manifests(manifests, stale_source_paths=stale_sources)
        scanned_roots = [str(root) for root in resolved_roots]
        if ollama_result is not None:
            scanned_roots.append(ollama_result.endpoint)
        scanned_roots.extend(f"{result.endpoint} ({result.endpoint_id})" for result in external_results)
        summary = ModelScanSummary(
            roots_scanned=tuple(scanned_roots),
            discovered_count=len(manifests),
            new_count=new_count,
            updated_count=updated_count,
            unchanged_count=unchanged_count,
            removed_count=len(stale_sources),
            manifests=manifests,
            notes=[
                *self._ollama_scan_notes(ollama_result),
                *self._external_scan_notes(external_results, existing_by_source),
            ],
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

    def _discover_ollama_models(self) -> OllamaInventoryResult | None:
        """Read the operator's Ollama daemon, or `None` when discovery is off.

        Off is the default, and while it is off LewLM never contacts the daemon.
        """

        if not self.settings.ollama_discovery_enabled:
            return None
        return discover_ollama_models(self.settings)

    def _discover_external_models(self) -> list[ExternalInventoryResult]:
        """Read every enabled named endpoint, or nothing when none are bound."""

        if self._endpoint_runtime_provider is None:
            return []
        runtimes = self._endpoint_runtime_provider()
        if not runtimes:
            return []
        return discover_external_models(
            runtimes,
            max_concurrency=self.settings.external_inventory_concurrency,
            refresh=True,
        )

    def _enabled_external_endpoint_ids(self) -> set[str]:
        return {
            endpoint.endpoint_id
            for endpoint in self.settings.resolved_external_endpoints()
            if endpoint.enabled and endpoint.profile != "ollama_local"
        }

    @staticmethod
    def _external_scan_notes(
        results: list[ExternalInventoryResult],
        existing_by_source: dict[str, ModelManifest],
    ) -> list[str]:
        notes: list[str] = []
        for result in results:
            if result.succeeded:
                notes.append(
                    f"Endpoint `{result.endpoint_id}` ({result.profile}) advertised "
                    f"{len(result.manifests)} model(s) at {result.endpoint}.",
                )
                continue
            retained = sum(
                1 for source in existing_by_source
                if (parsed := parse_external_source(source)) is not None and parsed[0] == result.endpoint_id
            )
            # Unreachable is not deleted: the registered models stay, flagged stale.
            notes.append(
                f"Endpoint `{result.endpoint_id}` could not be read ({result.error}); "
                f"keeping {retained} previously registered model(s) as stale.",
            )
        return notes

    def _stale_external_sources(
        self,
        existing_by_source: dict[str, ModelManifest],
        discovered_sources: set[str],
        results: list[ExternalInventoryResult],
    ) -> list[str]:
        """Which registered `external://` models this scan should forget.

        An endpoint that is no longer configured or enabled retires its
        namespace, so disabling an engine actually removes its advertised
        models instead of stranding unroutable rows. A successful read retires
        exactly what it no longer advertises. A failed read retires nothing.
        A scan that could not read endpoints at all (no runtimes bound)
        retires nothing either.
        """

        registered = [source for source in existing_by_source if is_external_source(source)]
        if not registered:
            return []
        if self._endpoint_runtime_provider is None:
            return []
        enabled_ids = self._enabled_external_endpoint_ids()
        results_by_id = {result.endpoint_id: result for result in results}
        stale: list[str] = []
        for source in registered:
            parsed = parse_external_source(source)
            if parsed is None:
                stale.append(source)
                continue
            endpoint_id = parsed[0]
            if endpoint_id not in enabled_ids:
                stale.append(source)
                continue
            result = results_by_id.get(endpoint_id)
            if result is None or not result.succeeded:
                continue
            if source not in discovered_sources:
                stale.append(source)
        return stale

    @staticmethod
    def _ollama_scan_notes(ollama_result: OllamaInventoryResult | None) -> list[str]:
        if ollama_result is None:
            return []
        if not ollama_result.succeeded:
            # A daemon LewLM does not manage can be down for entirely ordinary
            # reasons, so a failed read degrades the scan instead of failing it —
            # but it must not do so silently.
            return [f"Ollama discovery could not read {ollama_result.endpoint}: {ollama_result.error}"]
        notes = [
            f"Ollama discovery read {len(ollama_result.manifests)} model(s) from {ollama_result.endpoint}.",
        ]
        if ollama_result.binding_note:
            notes.append(ollama_result.binding_note)
        if ollama_result.skipped_cloud:
            notes.append(
                f"Skipped {len(ollama_result.skipped_cloud)} cloud-backed Ollama model(s) that execute "
                f"off-host: {', '.join(ollama_result.skipped_cloud)}. "
                "Set LEWLM_OLLAMA_CLOUD_ENABLED=true to include them.",
            )
        return notes

    @staticmethod
    def _stale_ollama_sources(
        existing_by_source: dict[str, ModelManifest],
        discovered_sources: set[str],
        ollama_result: OllamaInventoryResult | None,
    ) -> list[str]:
        """Which registered Ollama models this scan should forget.

        Discovery turned off retires the whole namespace, so clearing the flag
        actually removes the models rather than stranding them. A failed read
        retires nothing: an unreachable daemon is not evidence that the
        operator's models are gone.
        """

        registered = [source for source in existing_by_source if is_ollama_source(source)]
        if ollama_result is None:
            return registered
        if not ollama_result.succeeded:
            return []
        return [source for source in registered if source not in discovered_sources]

    def _resolve_roots(self, roots: list[Path] | tuple[Path, ...] | None) -> tuple[Path, ...]:
        requested_roots = tuple(Path(root).expanduser().resolve(strict=False) for root in (roots or self.settings.models_dir))
        if roots is None:
            requested_roots = (*requested_roots, *self._conversion_artifact_scan_roots())
        unique_roots = tuple(dict.fromkeys(requested_roots))
        requested_roots = tuple(
            root
            for root in unique_roots
            if not any(root != other and root.is_relative_to(other) for other in unique_roots)
        )
        for root in requested_roots:
            if not root.exists():
                raise ModelScanError("Model root does not exist.", details={"path": str(root)})
            if not root.is_dir():
                raise ModelScanError("Model root is not a directory.", details={"path": str(root)})
        return requested_roots

    def _conversion_artifact_scan_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        for artifact in self.metadata_store.list_conversion_artifacts():
            output_path = Path(artifact.output_path).expanduser().resolve(strict=False)
            if output_path.exists() and output_path.is_dir():
                roots.append(output_path)
        return tuple(roots)

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


def _manifest_change_signature(manifest: ModelManifest) -> str:
    """Serialize the parts of a manifest a rescan is expected to preserve.

    Discovery timestamps move on every scan, so they are excluded; everything
    else is compared, because everything else is what a caller reads.
    """

    payload = manifest.model_dump(mode="json", exclude={"discovered_at"})
    validation = payload.get("last_validation_result")
    if isinstance(validation, dict):
        validation.pop("checked_at", None)
    return json.dumps(payload, sort_keys=True, default=str)


def _slugify_manifest_selector(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
