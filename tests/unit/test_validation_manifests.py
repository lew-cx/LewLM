from __future__ import annotations

import json
from pathlib import Path

from conftest import write_external_validation_manifest
from lewlm.core.contracts import (
    ConversionStatus,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelTargetPlatformReport,
    ModelValidationResult,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.utils.model_identity import build_manifest_validation_key
from lewlm.utils.validation_manifests import (
    apply_external_validation_to_model_targets,
    apply_external_validation_to_target_matrix,
    load_validation_manifests,
)


def _manifest(*, model_id: str = "gguf-model") -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        display_name="GGUF Model",
        architecture_family="llama",
        modality=(ModelModality.TEXT,),
        source_path="X:\\models\\gguf-model.gguf",
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
        estimated_memory_mb=512,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint=f"{model_id}-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )


def _capability_report(manifest: ModelManifest) -> dict[str, object]:
    return {
        "model_id": manifest.model_id,
        "display_name": manifest.display_name,
        "architecture_family": manifest.architecture_family,
        "format_type": manifest.format_type.value,
        "modality": [modality.value for modality in manifest.modality],
        "validation_key": build_manifest_validation_key(manifest),
    }


def test_load_validation_manifests_ignores_non_release_payloads(tmp_path: Path) -> None:
    valid_manifest = tmp_path / "valid.json"
    write_external_validation_manifest(
        valid_manifest,
        capability_report=_capability_report(_manifest()),
        system="Linux",
        machine="x86_64",
    )
    (tmp_path / "invalid.json").write_text(json.dumps({"format": "different"}), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")

    manifests = load_validation_manifests([tmp_path])

    assert len(manifests) == 1
    assert manifests[0].platform.system == "Linux"
    assert manifests[0].source_path == str(valid_manifest)


def test_apply_external_validation_to_model_targets_upgrades_declared_support(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "linux.json"
    write_external_validation_manifest(
        manifest_path,
        capability_report=_capability_report(manifest),
        system="Linux",
        machine="x86_64",
    )
    validation_manifests = load_validation_manifests([manifest_path])
    target_reports = [
        ModelTargetPlatformReport(
            system="Linux",
            machine="x86_64",
            supported=True,
            readiness_state="declared",
            verification_method="runtime_contract",
            runtime_affinities=[RuntimeAffinity.LLAMACPP],
            reason="Declared compatible via runtime contract for llamacpp.",
        ),
        ModelTargetPlatformReport(
            system="Windows",
            machine="AMD64",
            supported=True,
            readiness_state="declared",
            verification_method="runtime_contract",
            runtime_affinities=[RuntimeAffinity.LLAMACPP],
            reason="Declared compatible via runtime contract for llamacpp.",
        ),
    ]

    updated = apply_external_validation_to_model_targets(
        target_reports,
        manifest=manifest,
        validation_manifests=validation_manifests,
    )
    linux_target, windows_target = updated

    assert linux_target.readiness_state == "verified_external"
    assert linux_target.verification_method == "external_release_manifest"
    assert linux_target.validation_manifest_count == 1
    assert linux_target.verified_hosts
    assert windows_target.readiness_state == "declared"
    assert windows_target.validation_manifest_count == 0


def test_apply_external_validation_to_target_matrix_marks_verified_models(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "windows.json"
    write_external_validation_manifest(
        manifest_path,
        capability_report=_capability_report(manifest),
        system="Windows",
        machine="AMD64",
    )
    validation_manifests = load_validation_manifests([manifest_path])
    target_rows = [
        {
            "system": "Windows",
            "machine": "AMD64",
            "supported_runtime_count": 1,
            "unsupported_runtime_count": 0,
            "compatible_model_count": 1,
            "incompatible_model_count": 0,
            "blocked_model_count": 0,
            "fallback_model_count": 0,
            "compatible_models": [manifest.model_id],
            "incompatible_models": [],
            "blocked_models": [],
            "fallback_models": [],
            "readiness_state": "declared",
            "verification_method": "runtime_contract",
            "notes": [],
            "runtimes": [],
        },
    ]

    updated = apply_external_validation_to_target_matrix(
        target_rows,
        local_manifests=[manifest],
        validation_manifests=validation_manifests,
    )
    windows_target = updated[0]

    assert windows_target["readiness_state"] == "verified_external"
    assert windows_target["verification_method"] == "external_release_manifest"
    assert windows_target["validation_manifest_count"] == 1
    assert windows_target["verified_model_count"] == 1
    assert windows_target["verified_models"] == [manifest.model_id]
    assert windows_target["verified_hosts"]
