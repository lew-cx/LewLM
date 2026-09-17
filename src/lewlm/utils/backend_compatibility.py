"""Reproducible backend validation inputs; never an engine installer or detector."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, Field(min_length=1)]


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelInput(EvidenceModel):
    repository: NonEmpty
    revision: Revision
    tokenizer_revision: Revision
    license: NonEmpty
    format: NonEmpty
    quantization: NonEmpty
    files_sha256: dict[NonEmpty, Digest] = Field(min_length=1)
    context_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)


class CandidatePins(EvidenceModel):
    # A source pin does not establish a working dependency or hardware stack.
    sources: dict[NonEmpty, Revision] = Field(min_length=1)
    model: ModelInput


class EnvironmentPins(EvidenceModel):
    python: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    system: Literal["Darwin", "Linux", "Windows"]
    machine: NonEmpty
    accelerator: NonEmpty
    driver: NonEmpty
    toolkit: NonEmpty
    engine_versions: dict[NonEmpty, NonEmpty] = Field(min_length=1)
    installation: Annotated["LockedInstall | ContainerInstall", Field(discriminator="kind")]


class LockedInstall(EvidenceModel):
    kind: Literal["lock"]
    path: NonEmpty
    sha256: Digest


class ContainerInstall(EvidenceModel):
    kind: Literal["image"]
    image: Annotated[str, Field(pattern=r"^.+@sha256:[0-9a-f]{64}$")]


class DeferredRecipe(EvidenceModel):
    profile: NonEmpty
    status: Literal["deferred"]
    candidate: CandidatePins
    reason: NonEmpty
    next_step: NonEmpty


class ValidatedRecipe(EvidenceModel):
    profile: NonEmpty
    status: Literal["validated"]
    candidate: CandidatePins
    environment: EnvironmentPins
    evidence_path: NonEmpty
    # Exactly which cases passed, and which were inconclusive or not probed.
    # A validated recipe is a passing configuration, never a blanket claim.
    notes: list[NonEmpty] = Field(default_factory=list)


class FailedRecipe(EvidenceModel):
    profile: NonEmpty
    status: Literal["failed"]
    candidate: CandidatePins
    reason: NonEmpty
    evidence_path: NonEmpty


class BackendCompatibilityManifest(EvidenceModel):
    format: Literal["lewlm-backend-compatibility-v1"]
    recipes: list[Annotated[DeferredRecipe | ValidatedRecipe | FailedRecipe, Field(discriminator="status")]]

    @model_validator(mode="after")
    def unique_profiles(self) -> BackendCompatibilityManifest:
        profiles = [recipe.profile for recipe in self.recipes]
        if len(profiles) != len(set(profiles)):
            raise ValueError("Backend profiles must be unique.")
        return self


def load_backend_compatibility(path: str | Path) -> BackendCompatibilityManifest:
    return BackendCompatibilityManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def compatibility_schema() -> dict:
    """The same status-discriminated contract for non-Python consumers."""
    return BackendCompatibilityManifest.model_json_schema()
