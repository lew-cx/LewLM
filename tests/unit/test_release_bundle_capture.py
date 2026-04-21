from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load_release_bundle_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "capture_release_bundle.py"
    spec = spec_from_file_location("lewlm_release_bundle", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_release_bundle_writes_artifacts_and_indexes_inputs(tmp_path, monkeypatch) -> None:
    module = _load_release_bundle_module()
    captured: dict[str, object] = {}
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_manifest = external_dir / "linux-release-manifest.json"
    external_manifest.write_text(
        json.dumps({"format": "lewlm-release-manifest-v1", "platform": {"system": "Linux", "machine": "x86_64"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "build_sbom", lambda: {"format": "lewlm-sbom-v1"})
    monkeypatch.setattr(
        module,
        "build_dependency_audit",
        lambda: {"format": "lewlm-dependency-audit-v1"},
    )
    monkeypatch.setattr(
        module,
        "build_release_manifest",
        lambda: {"format": "lewlm-release-manifest-v1", "platform": {"system": "Darwin", "machine": "arm64"}},
    )

    def fake_validator(paths, **kwargs):
        captured["paths"] = [str(path) for path in paths]
        captured["kwargs"] = kwargs
        return {
            "format": "lewlm-release-candidate-validation-v1",
            "overall_status": "passed",
            "checks": {"manifests_loaded": {"passed": True}},
        }

    monkeypatch.setattr(module, "build_release_candidate_validation", fake_validator)

    payload = module.capture_release_bundle(
        tmp_path / "out",
        validation_manifest_paths=[external_dir],
        required_targets=["Darwin:arm64"],
        minimum_verified_models=1,
        required_frontier_families=["dense_text", "speculative_family"],
        required_optimization_classes=["runtime_selection", "continuous_batching"],
    )

    assert payload["format"] == "lewlm-release-bundle-v1"
    assert payload["validation"]["overall_status"] == "passed"
    artifact_paths = {item["path"] for item in payload["artifacts"]}
    assert artifact_paths == {
        "sbom.json",
        "dependency-audit.json",
        "release-manifest.json",
        "release-candidate-validation.json",
    }

    index_path = Path(payload["index_path"])
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["format"] == "lewlm-release-artifact-index-v1"
    assert {item["path"] for item in index_payload["artifacts"]} == artifact_paths
    assert len(index_payload["artifacts"][0]["sha256"]) == 64
    assert index_payload["validation_inputs"][0]["source_path"] == str(external_manifest.resolve())
    assert index_payload["validation_inputs"][0]["bundled_path"] == "validation-manifests/linux-x86_64-release-manifest.json"
    bundled_input = Path(tmp_path / "out" / index_payload["validation_inputs"][0]["bundled_path"])
    assert bundled_input.exists()
    assert captured["kwargs"] == {
        "required_systems": (),
        "required_targets": ["Darwin:arm64"],
        "minimum_verified_models": 1,
        "required_frontier_families": ["dense_text", "speculative_family"],
        "required_optimization_classes": ["runtime_selection", "continuous_batching"],
    }
    assert str((tmp_path / "out").resolve()) in captured["paths"]
    assert str(bundled_input.resolve()) in captured["paths"]


def test_capture_release_bundle_rejects_missing_validation_inputs(tmp_path) -> None:
    module = _load_release_bundle_module()

    with pytest.raises(FileNotFoundError):
        module.capture_release_bundle(
            tmp_path / "out",
            validation_manifest_paths=[tmp_path / "missing"],
        )
