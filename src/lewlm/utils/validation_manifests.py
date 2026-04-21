"""External release-manifest loading and validation-evidence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lewlm.core.contracts import ModelManifest, ModelTargetPlatformReport
from lewlm.utils.model_identity import build_manifest_validation_key, build_model_validation_key


class ValidationManifestPlatform(BaseModel):
    system: str
    machine: str
    release: str | None = None


class ValidationManifestModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_id: str
    display_name: str
    architecture_family: str = "unknown"
    format_type: str
    modality: list[str] = Field(default_factory=list)
    quantization: str | None = None
    validation_key: str | None = None
    target_platforms: list[ModelTargetPlatformReport] = Field(default_factory=list)

    def resolved_validation_key(self) -> str:
        return self.validation_key or build_model_validation_key(
            display_name=self.display_name,
            format_type=self.format_type,
            architecture_family=self.architecture_family,
            quantization=self.quantization,
            modality=self.modality,
        )


class ValidationManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    format: str
    generated_at: str | None = None
    git_commit: str | None = None
    platform: ValidationManifestPlatform
    registered_models: list[ValidationManifestModel] = Field(default_factory=list)
    source_path: str

    def host_label(self) -> str:
        suffix = self.git_commit[:7] if self.git_commit else Path(self.source_path).name
        return f"{self.platform.system} {self.platform.machine} ({suffix})"


def load_validation_manifests(paths: tuple[Path, ...] | list[Path] | tuple[str, ...] | list[str]) -> list[ValidationManifest]:
    """Load valid LewLM release-manifest files from configured paths."""

    manifests: list[ValidationManifest] = []
    for path in _iter_manifest_files(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("format") != "lewlm-release-manifest-v1":
            continue
        manifests.append(ValidationManifest.model_validate({**payload, "source_path": str(path)}))
    return manifests


def apply_external_validation_to_model_targets(
    target_reports: list[ModelTargetPlatformReport],
    *,
    manifest: ModelManifest,
    validation_manifests: list[ValidationManifest],
) -> list[ModelTargetPlatformReport]:
    """Upgrade per-model target reports when external host validation evidence exists."""

    validation_key = build_manifest_validation_key(manifest)
    updated_reports: list[ModelTargetPlatformReport] = []
    for report in target_reports:
        verified_hosts = verified_hosts_for_model_target(
            validation_manifests,
            validation_key=validation_key,
            system=report.system,
            machine=report.machine,
        )
        updated = report.model_copy(deep=True)
        updated.validation_manifest_count = len(verified_hosts)
        updated.verified_hosts = verified_hosts
        if verified_hosts:
            if updated.supported and updated.readiness_state != "verified":
                updated.readiness_state = "verified_external"
                updated.verification_method = "external_release_manifest"
                updated.reason = f"Validated on external host manifest(s): {', '.join(verified_hosts[:3])}."
            else:
                updated.notes.append(
                    f"External host validation evidence is available from {len(verified_hosts)} manifest(s).",
                )
        updated_reports.append(updated)
    return updated_reports


def apply_external_validation_to_target_matrix(
    target_payloads: list[dict[str, Any]],
    *,
    local_manifests: list[ModelManifest],
    validation_manifests: list[ValidationManifest],
) -> list[dict[str, Any]]:
    """Upgrade target-platform matrix rows with external host validation evidence."""

    validation_keys = {
        manifest.model_id: build_manifest_validation_key(manifest)
        for manifest in local_manifests
    }
    local_by_id = {manifest.model_id: manifest for manifest in local_manifests}
    updated_rows: list[dict[str, Any]] = []
    for payload in target_payloads:
        system = str(payload["system"])
        machine = str(payload["machine"])
        verified_hosts = verified_hosts_for_target(validation_manifests, system=system, machine=machine)
        compatible_models = [
            model_id
            for model_id in payload.get("compatible_models", [])
            if model_id in local_by_id
        ]
        verified_models = [
            model_id
            for model_id in compatible_models
            if verified_hosts_for_model_target(
                validation_manifests,
                validation_key=validation_keys[model_id],
                system=system,
                machine=machine,
            )
        ]
        updated = dict(payload)
        updated["validation_manifest_count"] = len(verified_hosts)
        updated["verified_hosts"] = verified_hosts
        updated["verified_model_count"] = len(verified_models)
        updated["verified_models"] = verified_models
        notes = list(updated.get("notes", []))
        if verified_models:
            notes.append(
                f"Validated {len(verified_models)} compatible model(s) on {len(verified_hosts)} external host manifest(s).",
            )
            if updated.get("readiness_state") != "verified":
                updated["readiness_state"] = "verified_external"
                updated["verification_method"] = "external_release_manifest"
        updated["notes"] = sorted(dict.fromkeys(notes))
        updated_rows.append(updated)
    return updated_rows


def verified_hosts_for_target(
    validation_manifests: list[ValidationManifest],
    *,
    system: str,
    machine: str,
) -> list[str]:
    """Return external hosts that match the requested target platform."""

    hosts = {
        manifest.host_label()
        for manifest in validation_manifests
        if _matches_target(manifest.platform.system, manifest.platform.machine, system=system, machine=machine)
    }
    return sorted(hosts)


def verified_hosts_for_model_target(
    validation_manifests: list[ValidationManifest],
    *,
    validation_key: str,
    system: str,
    machine: str,
) -> list[str]:
    """Return host labels that verified a specific model for a target platform."""

    hosts: set[str] = set()
    for manifest in validation_manifests:
        if not _matches_target(manifest.platform.system, manifest.platform.machine, system=system, machine=machine):
            continue
        for report in manifest.registered_models:
            if report.resolved_validation_key() != validation_key:
                continue
            target_report = next(
                (
                    target
                    for target in report.target_platforms
                    if _matches_target(target.system, target.machine, system=system, machine=machine)
                ),
                None,
            )
            if target_report is not None and target_report.supported:
                hosts.add(manifest.host_label())
    return sorted(hosts)


def _iter_manifest_files(
    paths: tuple[Path, ...] | list[Path] | tuple[str, ...] | list[str],
) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve(strict=False)
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(candidate for candidate in path.glob("*.json") if candidate.is_file()))
    return files


def _matches_target(current_system: str, current_machine: str, *, system: str, machine: str) -> bool:
    return current_system.casefold() == system.casefold() and current_machine.casefold() == machine.casefold()
