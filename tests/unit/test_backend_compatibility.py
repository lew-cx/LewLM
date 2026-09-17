from __future__ import annotations

import hashlib
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


def _deferred_index(payload: dict) -> int:
    return next(index for index, recipe in enumerate(payload["recipes"]) if recipe["status"] == "deferred")


def test_a_source_pin_cannot_be_promoted_without_environment_and_evidence():
    payload = load_backend_compatibility(ROOT / "examples/backends/compatibility.json").model_dump()
    index = _deferred_index(payload)
    payload["recipes"][index] = {k: v for k, v in payload["recipes"][index].items() if k not in {"reason", "next_step"}}
    payload["recipes"][index]["status"] = "validated"
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
    index = _deferred_index(payload)
    recipe = payload["recipes"][index]
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
    assert BackendCompatibilityManifest.model_validate(payload).recipes[index].status == "validated"


def test_validated_recipes_point_at_real_locks_and_evidence():
    """A validated label must be backed by the committed lock and evidence it names."""

    manifest = load_backend_compatibility(ROOT / "examples/backends/compatibility.json")
    validated = [recipe for recipe in manifest.recipes if recipe.status == "validated"]
    assert {recipe.profile for recipe in validated} == {"omlx"}
    for recipe in validated:
        installation = recipe.environment.installation
        assert installation.kind == "lock"
        lock_path = ROOT / installation.path
        assert lock_path.is_file(), installation.path
        assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == installation.sha256
        evidence = ROOT / recipe.evidence_path
        assert (evidence / "acceptance.json").is_file()
        acceptance = json.loads((evidence / "acceptance.json").read_text())
        assert acceptance["summary"]["failed"] == 0
        assert recipe.notes, "a validated recipe states exactly what passed"
