from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lewlm.utils.backend_compatibility import BackendCompatibilityManifest, compatibility_schema, load_backend_compatibility

ROOT = Path(__file__).resolve().parents[2]


def test_committed_backend_candidates_validate_and_schema_is_current():
    manifest = load_backend_compatibility(ROOT / "examples/backends/compatibility.json")
    assert {r.profile for r in manifest.recipes} == {"omlx", "exllamav3_tabby", "vllm_local", "sglang_local"}
    assert json.loads((ROOT / "examples/backends/compatibility.schema.json").read_text()) == compatibility_schema()


def test_a_source_pin_cannot_be_promoted_without_environment_and_evidence():
    payload = load_backend_compatibility(ROOT / "examples/backends/compatibility.json").model_dump()
    payload["recipes"][0] = {k: v for k, v in payload["recipes"][0].items() if k not in {"reason", "next_step"}}
    payload["recipes"][0]["status"] = "validated"
    with pytest.raises(ValidationError, match="environment"):
        BackendCompatibilityManifest.model_validate(payload)


@pytest.mark.parametrize("digest", ["latest", "sha256:bad", "x" * 64])
def test_model_artifacts_require_real_sha256_shape(digest):
    payload = load_backend_compatibility(ROOT / "examples/backends/compatibility.json").model_dump()
    payload["recipes"][0]["candidate"]["model"]["files_sha256"] = {"model.safetensors": digest}
    with pytest.raises(ValidationError):
        BackendCompatibilityManifest.model_validate(payload)


def test_duplicate_profiles_are_rejected():
    payload = load_backend_compatibility(ROOT / "examples/backends/compatibility.json").model_dump()
    payload["recipes"].append(payload["recipes"][0])
    with pytest.raises(ValidationError, match="unique"):
        BackendCompatibilityManifest.model_validate(payload)


@pytest.mark.parametrize("installation", [{"kind": "image", "image": "engine:latest"}, {"kind": "lock", "path": "requirements.txt"}])
def test_validated_recipe_requires_an_immutable_installation(installation):
    payload = load_backend_compatibility(ROOT / "examples/backends/compatibility.json").model_dump()
    recipe = payload["recipes"][0]
    recipe.pop("reason")
    recipe.pop("next_step")
    recipe.update(status="validated", evidence_path="proof.json", environment={
        "python": "3.12.10", "system": "Darwin", "machine": "arm64",
        "accelerator": "Metal", "driver": "macOS", "toolkit": "Metal",
        "engine_versions": {"omlx": "test-version"}, "installation": installation,
    })
    with pytest.raises(ValidationError, match="installation"):
        BackendCompatibilityManifest.model_validate(payload)
    recipe["environment"]["installation"] = {"kind": "lock", "path": "requirements.txt", "sha256": "a" * 64}
    assert BackendCompatibilityManifest.model_validate(payload).recipes[0].status == "validated"
