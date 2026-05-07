"""Runtime-pack and feature-pack selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from typing import Any

from pydantic import BaseModel, Field

from lewlm.core.contracts import RuntimeAffinity
from lewlm.core.errors import PackUnavailableError


def canonicalize_pack_name(name: str) -> str:
    """Normalize a user-provided pack name to LewLM's canonical form."""

    normalized = name.strip().casefold().replace("-", "_")
    aliases = {
        "llama_cpp": "llamacpp",
    }
    return aliases.get(normalized, normalized)


class PackKind(str, Enum):
    """High-level pack category."""

    RUNTIME = "runtime"
    FEATURE = "feature"


class PackStatus(str, Enum):
    """Operator-facing pack state."""

    ACTIVE = "active"
    DISABLED = "disabled"
    MISSING_DEPENDENCY = "missing_dependency"


class PackReport(BaseModel):
    """Machine-readable pack status."""

    name: str
    kind: PackKind
    description: str
    active: bool
    status: PackStatus
    reason: str
    optional_dependency_group: str | None = None
    missing_modules: list[str] = Field(default_factory=list)
    runtime_affinities: list[RuntimeAffinity] = Field(default_factory=list)
    surfaces: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PackDefinition:
    """Static metadata for one built-in LewLM pack."""

    name: str
    kind: PackKind
    description: str
    optional_dependency_group: str | None = None
    required_modules: tuple[str, ...] = ()
    runtime_affinities: tuple[RuntimeAffinity, ...] = ()
    surfaces: tuple[str, ...] = ()


_RUNTIME_PACK_DEFINITIONS: tuple[PackDefinition, ...] = (
    PackDefinition(
        name="mlx",
        kind=PackKind.RUNTIME,
        description="Apple MLX runtime pack for text, vision, and audio workloads on Apple Silicon macOS.",
        optional_dependency_group="mlx",
        required_modules=("mlx_lm", "mlx_vlm", "mlx_audio"),
        runtime_affinities=(
            RuntimeAffinity.MLX_TEXT,
            RuntimeAffinity.MLX_VISION,
            RuntimeAffinity.MLX_AUDIO,
        ),
    ),
    PackDefinition(
        name="llamacpp",
        kind=PackKind.RUNTIME,
        description="Cross-platform GGUF runtime pack through llama-cpp-python.",
        optional_dependency_group="llamacpp",
        required_modules=("llama_cpp",),
        runtime_affinities=(RuntimeAffinity.LLAMACPP,),
    ),
    PackDefinition(
        name="external_accelerator",
        kind=PackKind.RUNTIME,
        description="Bridge runtime pack for loopback-only OpenAI-compatible local servers that LewLM fronts instead of importing in-process.",
        runtime_affinities=(RuntimeAffinity.EXTERNAL_ACCELERATOR,),
    ),
    PackDefinition(
        name="experimental",
        kind=PackKind.RUNTIME,
        description="Planning-only experimental runtime surfaces.",
        runtime_affinities=(RuntimeAffinity.EXPERIMENTAL,),
    ),
    PackDefinition(
        name="distributed_experimental",
        kind=PackKind.RUNTIME,
        description="Proof-oriented distributed runtime surfaces.",
        runtime_affinities=(RuntimeAffinity.DISTRIBUTED_EXPERIMENTAL,),
    ),
)

_FEATURE_PACK_DEFINITIONS: tuple[PackDefinition, ...] = (
    PackDefinition(
        name="documents",
        kind=PackKind.FEATURE,
        description="Additive documents pack for ingest, generation, transform, tool, and skill surfaces.",
        optional_dependency_group="documents",
        surfaces=(
            "documents_api",
            "document_cli",
            "tool_catalog",
            "skill_catalog",
        ),
    ),
)

_PACK_DEFINITIONS = {
    definition.name: definition
    for definition in (*_RUNTIME_PACK_DEFINITIONS, *_FEATURE_PACK_DEFINITIONS)
}
_PACK_BY_AFFINITY = {
    affinity: definition.name
    for definition in _RUNTIME_PACK_DEFINITIONS
    for affinity in definition.runtime_affinities
}

KNOWN_RUNTIME_PACKS = frozenset(definition.name for definition in _RUNTIME_PACK_DEFINITIONS)
KNOWN_FEATURE_PACKS = frozenset(definition.name for definition in _FEATURE_PACK_DEFINITIONS)


class PackRegistry:
    """Resolve which built-in packs LewLM should activate for a process."""

    def __init__(
        self,
        *,
        runtime_packs: tuple[str, ...] = (),
        disabled_runtime_packs: tuple[str, ...] = (),
        feature_packs: tuple[str, ...] = (),
        disabled_feature_packs: tuple[str, ...] = (),
    ) -> None:
        self._runtime_packs = tuple(runtime_packs)
        self._disabled_runtime_packs = frozenset(disabled_runtime_packs)
        self._feature_packs = tuple(feature_packs)
        self._disabled_feature_packs = frozenset(disabled_feature_packs)

    @classmethod
    def from_settings(cls, settings: Any) -> "PackRegistry":
        """Build a pack registry from a LewLMSettings-like object."""

        return cls(
            runtime_packs=tuple(getattr(settings, "runtime_packs", ()) or ()),
            disabled_runtime_packs=tuple(getattr(settings, "disabled_runtime_packs", ()) or ()),
            feature_packs=tuple(getattr(settings, "feature_packs", ()) or ()),
            disabled_feature_packs=tuple(getattr(settings, "disabled_feature_packs", ()) or ()),
        )

    def runtime_pack_reports(self) -> list[PackReport]:
        return [self.report(definition.name) for definition in _RUNTIME_PACK_DEFINITIONS]

    def feature_pack_reports(self) -> list[PackReport]:
        return [self.report(definition.name) for definition in _FEATURE_PACK_DEFINITIONS]

    def report(self, name: str) -> PackReport:
        definition = _PACK_DEFINITIONS[canonicalize_pack_name(name)]
        enabled = self._pack_enabled(definition)
        if not enabled:
            return PackReport(
                name=definition.name,
                kind=definition.kind,
                description=definition.description,
                active=False,
                status=PackStatus.DISABLED,
                reason=self._disabled_reason(definition),
                optional_dependency_group=definition.optional_dependency_group,
                runtime_affinities=list(definition.runtime_affinities),
                surfaces=list(definition.surfaces),
            )
        missing_modules = [module_name for module_name in definition.required_modules if find_spec(module_name) is None]
        if missing_modules:
            dependency_group = (
                f"`{definition.optional_dependency_group}`"
                if definition.optional_dependency_group is not None
                else "the required dependency group"
            )
            return PackReport(
                name=definition.name,
                kind=definition.kind,
                description=definition.description,
                active=False,
                status=PackStatus.MISSING_DEPENDENCY,
                reason=(
                    f"Pack selected, but {dependency_group} is incomplete on this host: "
                    f"{', '.join(missing_modules)}."
                ),
                optional_dependency_group=definition.optional_dependency_group,
                missing_modules=missing_modules,
                runtime_affinities=list(definition.runtime_affinities),
                surfaces=list(definition.surfaces),
            )
        active_reason = (
            "Pack is active."
            if definition.kind == PackKind.FEATURE
            else "Pack is active; per-runtime readiness is reported separately."
        )
        return PackReport(
            name=definition.name,
            kind=definition.kind,
            description=definition.description,
            active=True,
            status=PackStatus.ACTIVE,
            reason=active_reason,
            optional_dependency_group=definition.optional_dependency_group,
            runtime_affinities=list(definition.runtime_affinities),
            surfaces=list(definition.surfaces),
        )

    def runtime_affinity_load_enabled(self, affinity: RuntimeAffinity) -> bool:
        pack_name = _PACK_BY_AFFINITY.get(affinity)
        if pack_name is None:
            return True
        return self._pack_enabled(_PACK_DEFINITIONS[pack_name])

    def runtime_affinity_absence_reason(self, affinity: RuntimeAffinity) -> str | None:
        pack_name = _PACK_BY_AFFINITY.get(affinity)
        if pack_name is None:
            return None
        report = self.report(pack_name)
        if report.status != PackStatus.DISABLED:
            return None
        return f"Runtime pack `{pack_name}` is disabled."

    def feature_enabled(self, name: str) -> bool:
        return self.report(name).status == PackStatus.ACTIVE

    def require_feature_pack(self, name: str, *, surface: str) -> None:
        report = self.report(name)
        if report.status == PackStatus.ACTIVE:
            return
        raise PackUnavailableError(
            f"LewLM cannot use `{surface}` because feature pack `{report.name}` is unavailable.",
            details={
                "pack": report.name,
                "pack_kind": report.kind.value,
                "pack_status": report.status.value,
                "reason": report.reason,
                "surface": surface,
            },
        )

    def _pack_enabled(self, definition: PackDefinition) -> bool:
        if definition.kind == PackKind.RUNTIME:
            allowlist = self._runtime_packs
            denylist = self._disabled_runtime_packs
        else:
            allowlist = self._feature_packs
            denylist = self._disabled_feature_packs
        if allowlist and definition.name not in allowlist:
            return False
        return definition.name not in denylist

    @staticmethod
    def _disabled_reason(definition: PackDefinition) -> str:
        if definition.kind == PackKind.RUNTIME:
            return (
                f"Runtime pack `{definition.name}` is disabled by configuration. "
                "Adjust `runtime_packs` or `disabled_runtime_packs` to load it."
            )
        return (
            f"Feature pack `{definition.name}` is disabled by configuration. "
            "Adjust `feature_packs` or `disabled_feature_packs` to load it."
        )
