"""Runtime catalog and backend selection helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
import platform
from typing import Literal

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    CapabilityName,
    ConversionStatus,
    HostPlatformSnapshot,
    ModelManifest,
    ModelModality,
    ModelTargetPlatformReport,
    RequestModality,
    RuntimeAffinity,
    RuntimeCandidateReport,
    RuntimeContract,
    RuntimeReadinessState,
)
from lewlm.core.errors import RoutingError
from lewlm.pack_registry import PackRegistry
from lewlm.runtime.llamacpp.runtime import LlamaCppRuntime
from lewlm.runtime.experimental import DistributedClusterService, DistributedExperimentalRuntime, FrontierExperimentalRuntime
from lewlm.runtime.adapters import LocalOpenAICompatibleAdapterRuntime
from lewlm.runtime.mlx_audio.runtime import MLXAudioRuntime
from lewlm.runtime.mlx_text.runtime import MLXTextRuntime
from lewlm.runtime.mlx_vision.runtime import MLXVisionRuntime
from lewlm.storage.block_cache import MultimodalEncoderCache


class RuntimeCatalog:
    """Registry of runtime backends keyed by runtime affinity."""

    def __init__(
        self,
        runtimes: Mapping[RuntimeAffinity, RuntimeContract],
        *,
        pack_registry: PackRegistry | None = None,
    ) -> None:
        self._runtimes = dict(runtimes)
        self._pack_registry = pack_registry

    @property
    def pack_registry(self) -> PackRegistry | None:
        return self._pack_registry

    def get_runtime(self, affinity: RuntimeAffinity) -> RuntimeContract | None:
        return self._runtimes.get(affinity)

    def find_runtime_by_name(self, runtime_name: str) -> RuntimeContract | None:
        for runtime in self._runtimes.values():
            if runtime.name == runtime_name:
                return runtime
        return None

    @staticmethod
    def host_platform_snapshot() -> HostPlatformSnapshot:
        return HostPlatformSnapshot(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            python_version=platform.python_version(),
        )

    def select_runtime(
        self,
        manifest: ModelManifest,
        *,
        capability: CapabilityName,
        request_modality: RequestModality | None = None,
    ) -> RuntimeContract:
        compatible_runtimes, alternatives = self.compatible_runtimes(
            manifest,
            capability=capability,
            request_modality=request_modality,
        )
        if compatible_runtimes:
            return compatible_runtimes[0]
        raise RoutingError(
            "No compatible runtime is currently available for the selected model.",
            details={
                "model_id": manifest.model_id,
                "requested_capability": capability.value,
                "alternatives": alternatives,
            },
        )

    def compatible_runtimes(
        self,
        manifest: ModelManifest,
        *,
        capability: CapabilityName,
        request_modality: RequestModality | None = None,
    ) -> tuple[list[RuntimeContract], list[str]]:
        alternatives: list[str] = []
        compatible: list[RuntimeContract] = []
        for affinity in self._candidate_affinities(manifest, request_modality=request_modality):
            runtime = self.get_runtime(affinity)
            if runtime is None:
                alternatives.append(f"{affinity.value}: {self._runtime_absence_reason(affinity)}")
                continue
            if not runtime.is_available():
                alternatives.append(f"{affinity.value}: {runtime.availability_reason() or 'runtime unavailable'}")
                continue
            if not runtime.supports_manifest(manifest):
                alternatives.append(f"{affinity.value}: manifest format unsupported")
                continue
            if not runtime.supports_capability(capability):
                reason = runtime.availability_reason() or "capability unavailable"
                alternatives.append(f"{affinity.value}: {reason}")
                continue
            compatible.append(runtime)
        return compatible, alternatives

    async def warm_model(self, manifest: ModelManifest) -> RuntimeContract:
        runtime = self.select_runtime(manifest, capability=CapabilityName.CHAT)
        await runtime.load_model(manifest)
        await runtime.warm_model(manifest.model_id)
        return runtime

    async def unload_model(self, manifest: ModelManifest) -> RuntimeContract:
        runtime = self.select_runtime(manifest, capability=CapabilityName.CHAT)
        await runtime.unload_model(manifest.model_id)
        return runtime

    def describe_manifest_runtimes(self, manifest: ModelManifest) -> list[RuntimeCandidateReport]:
        reports: list[RuntimeCandidateReport] = []
        for affinity in self._candidate_affinities(manifest):
            runtime = self.get_runtime(affinity)
            if runtime is None:
                reports.append(
                    RuntimeCandidateReport(
                        runtime_name=affinity.value,
                        runtime_affinity=affinity,
                        readiness_state=RuntimeReadinessState.UNREGISTERED,
                        registered=False,
                        available=False,
                        availability_reason=self._runtime_absence_reason(affinity),
                        host_platform_supported=False,
                        supported_systems=[],
                        supported_machines=[],
                        supports_manifest=False,
                    ),
                )
                continue
            candidate_report = getattr(runtime, "candidate_report", None)
            if callable(candidate_report):
                reports.append(candidate_report(manifest))
                continue
            reports.append(
                RuntimeCandidateReport(
                    runtime_name=runtime.name,
                    runtime_affinity=runtime.affinity,
                    readiness_state=_runtime_candidate_readiness_state(runtime, manifest),
                    registered=True,
                    available=runtime.is_available(),
                    availability_reason=runtime.availability_reason(),
                    host_platform_supported=runtime.supports_host_platform(),
                    supported_systems=list(runtime.supported_systems),
                    supported_machines=list(runtime.supported_machines),
                    supports_manifest=runtime.supports_manifest(manifest),
                ),
            )
        return reports

    def target_platform_matrix(self, manifests: list[ModelManifest]) -> list[dict[str, object]]:
        reports: list[dict[str, object]] = []
        host_platform = self.host_platform_snapshot()
        for system, machine in self._target_platforms():
            runtime_reports: list[dict[str, object]] = []
            for runtime in self._runtimes.values():
                runtime_reports.append(
                    runtime.target_platform_status(
                        system,
                        machine,
                        host_platform=host_platform,
                    ),
                )
            compatible_models: list[str] = []
            incompatible_models: list[str] = []
            blocked_models: list[str] = []
            fallback_models: list[str] = []
            notes: set[str] = set()
            for manifest in manifests:
                if manifest.conversion_status != ConversionStatus.RUNNABLE:
                    fallback_note = None
                    if manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION:
                        fallback_note = self._fallback_guidance_for_manifest(manifest, system=system, machine=machine)
                    if fallback_note is not None:
                        fallback_models.append(manifest.model_id)
                        notes.add(fallback_note)
                        notes.add(
                            "Some discovered bundles still require conversion or GGUF export before target-platform readiness can be verified.",
                        )
                        continue
                    blocked_models.append(manifest.model_id)
                    if manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION:
                        notes.add(
                            "Some discovered bundles still require conversion before target-platform readiness can be verified.",
                        )
                    continue
                if self._manifest_supports_target(
                    manifest,
                    system=system,
                    machine=machine,
                    host_platform=host_platform,
                ):
                    compatible_models.append(manifest.model_id)
                    continue
                fallback_note = self._fallback_guidance_for_manifest(manifest, system=system, machine=machine)
                if fallback_note is not None:
                    fallback_models.append(manifest.model_id)
                    notes.add(fallback_note)
                    continue
                incompatible_models.append(manifest.model_id)
            reports.append(
                {
                    "system": system,
                    "machine": machine,
                    "supported_runtime_count": sum(1 for runtime in runtime_reports if runtime["supported"]),
                    "unsupported_runtime_count": sum(1 for runtime in runtime_reports if not runtime["supported"]),
                    "compatible_model_count": len(compatible_models),
                    "incompatible_model_count": len(incompatible_models),
                    "blocked_model_count": len(blocked_models),
                    "fallback_model_count": len(fallback_models),
                    "compatible_models": compatible_models,
                    "incompatible_models": incompatible_models,
                    "blocked_models": blocked_models,
                    "fallback_models": fallback_models,
                    "readiness_state": self._target_readiness_state(
                        system=system,
                        machine=machine,
                        host_platform=host_platform,
                        compatible_model_count=len(compatible_models),
                        fallback_model_count=len(fallback_models),
                    ),
                    "verification_method": (
                        "host_probe"
                        if self._matches_host_platform(host_platform, system=system, machine=machine)
                        else "runtime_contract"
                    ),
                    "notes": sorted(notes),
                    "runtimes": runtime_reports,
                },
            )
        return reports

    async def health_snapshot(self) -> list[dict[str, object]]:
        return [await runtime.health_check() for runtime in self._runtimes.values()]

    def performance_snapshot(self) -> list[dict[str, object]]:
        return [
            {
                "name": runtime.name,
                "available": runtime.is_available(),
                "supported_capabilities": sorted(
                    capability.value
                    for capability in CapabilityName
                    if runtime.supports_capability(capability)
                ),
                "performance_features": runtime.performance_feature_snapshot(),
            }
            for runtime in self._runtimes.values()
        ]

    async def unload_all_models(self) -> None:
        for runtime in self._runtimes.values():
            for loaded_manifest in runtime.loaded_manifests():
                await runtime.unload_model(loaded_manifest.model_id)

    def describe_manifest_targets(self, manifest: ModelManifest) -> list[ModelTargetPlatformReport]:
        host_platform = self.host_platform_snapshot()
        reports: list[ModelTargetPlatformReport] = []
        for system, machine in self._target_platforms():
            if manifest.conversion_status != ConversionStatus.RUNNABLE:
                fallback_reason = None
                if manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION:
                    fallback_reason = self._fallback_guidance_for_manifest(manifest, system=system, machine=machine)
                if fallback_reason is not None:
                    install_hints: list[str] = []
                    fallback_runtime = self.get_runtime(RuntimeAffinity.LLAMACPP)
                    fallback_hint = getattr(fallback_runtime, "platform_guidance", None)
                    if isinstance(fallback_hint, str) and fallback_hint:
                        install_hints.append(fallback_hint)
                    reports.append(
                        ModelTargetPlatformReport(
                            system=system,
                            machine=machine,
                            supported=False,
                            readiness_state="fallback_guided",
                            verification_method="runtime_contract",
                            reason=fallback_reason,
                            fallback_available=True,
                            fallback_reason=fallback_reason,
                            install_hints=install_hints,
                            notes=[
                                "Conversion or additional preparation is still required for this model bundle.",
                            ],
                        ),
                    )
                    continue
                reports.append(
                    ModelTargetPlatformReport(
                        system=system,
                        machine=machine,
                        supported=False,
                        readiness_state="blocked",
                        verification_method="none",
                        reason=(
                            f"Model is `{manifest.conversion_status.value}` and must become runnable "
                            "before target-platform readiness can be verified."
                        ),
                        notes=[
                            "Conversion or additional preparation is still required for this model bundle.",
                        ],
                    ),
                )
                continue
            matching_affinities: list[RuntimeAffinity] = []
            install_hints: list[str] = []
            reasons: list[str] = []
            for affinity in manifest.runtime_affinity:
                runtime = self.get_runtime(affinity)
                if runtime is None or not runtime.supports_manifest(manifest):
                    continue
                status = runtime.target_platform_status(system, machine, host_platform=host_platform)
                if status["supported"]:
                    matching_affinities.append(affinity)
                elif status["reason"]:
                    reasons.append(f"{runtime.name}: {status['reason']}")
                hint = status.get("install_hint")
                if isinstance(hint, str) and hint and hint not in install_hints:
                    install_hints.append(hint)
            fallback_reason = None
            if not matching_affinities:
                fallback_reason = self._fallback_guidance_for_manifest(manifest, system=system, machine=machine)
            host_target = self._matches_host_platform(host_platform, system=system, machine=machine)
            if matching_affinities:
                runtime_names = ", ".join(affinity.value for affinity in matching_affinities)
                reports.append(
                    ModelTargetPlatformReport(
                        system=system,
                        machine=machine,
                        supported=True,
                        readiness_state="verified" if host_target else "declared",
                        verification_method="host_probe" if host_target else "runtime_contract",
                        runtime_affinities=matching_affinities,
                        reason=(
                            f"Verified on the current host via {runtime_names}."
                            if host_target
                            else f"Declared compatible via runtime contract for {runtime_names}."
                        ),
                        install_hints=install_hints,
                    ),
                )
                continue
            reports.append(
                ModelTargetPlatformReport(
                    system=system,
                    machine=machine,
                    supported=False,
                    readiness_state="fallback_guided" if fallback_reason is not None else "blocked",
                    verification_method="runtime_contract" if fallback_reason is not None else "none",
                    reason=fallback_reason or "No compatible runtime/backend path is currently available for this target.",
                    fallback_available=fallback_reason is not None,
                    fallback_reason=fallback_reason,
                    install_hints=install_hints,
                    notes=reasons[:4],
                ),
            )
        return reports
    @staticmethod
    def host_total_memory_mb() -> int | None:
        if hasattr(os, "sysconf"):
            page_size_name = "SC_PAGE_SIZE"
            page_count_name = "SC_PHYS_PAGES"
            if page_size_name in os.sysconf_names and page_count_name in os.sysconf_names:
                try:
                    page_size = int(os.sysconf(page_size_name))
                    page_count = int(os.sysconf(page_count_name))
                except (OSError, ValueError):
                    return None
                total_bytes = page_size * page_count
                if total_bytes > 0:
                    return max(1, total_bytes // (1024 * 1024))
        return None

    async def prepare_runtime_for_request(
        self,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        *,
        policy: Literal["keep_warm", "balanced", "aggressive_unload"],
    ) -> None:
        if policy == "keep_warm":
            await runtime.warm_model(manifest.model_id)

    async def finalize_runtime_for_request(
        self,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        *,
        policy: Literal["keep_warm", "balanced", "aggressive_unload"],
    ) -> None:
        if policy == "aggressive_unload":
            await runtime.unload_model(manifest.model_id)
            return
        if policy != "balanced":
            return
        for loaded_manifest in runtime.loaded_manifests():
            if loaded_manifest.model_id == manifest.model_id:
                continue
            await runtime.unload_model(loaded_manifest.model_id)

    def _manifest_supports_target(
        self,
        manifest: ModelManifest,
        *,
        system: str,
        machine: str,
        host_platform: HostPlatformSnapshot,
    ) -> bool:
        host_target = self._matches_host_platform(host_platform, system=system, machine=machine)
        for affinity in self._candidate_affinities(manifest):
            runtime = self.get_runtime(affinity)
            if runtime is None:
                continue
            if not runtime.supports_manifest(manifest):
                continue
            if host_target:
                if runtime.supports_target_platform(system, machine) and runtime.is_available():
                    return True
                continue
            if runtime.supports_target_platform(system, machine):
                return True
        return False

    def _fallback_guidance_for_manifest(self, manifest: ModelManifest, *, system: str, machine: str) -> str | None:
        if set(manifest.modality) != {ModelModality.TEXT}:
            return None
        fallback_runtime = self.get_runtime(RuntimeAffinity.LLAMACPP)
        if fallback_runtime is None or not fallback_runtime.supports_target_platform(system, machine):
            return None
        if manifest.format_type.value == "mlx":
            return (
                f"Pure text MLX models can use the {fallback_runtime.name} path on {system} {machine} "
                "after preparing a GGUF build for that target host."
            )
        if (
            manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION
            and manifest.format_type.value in {"huggingface", "adapter_bundle"}
            and system in {"Linux", "Windows"}
        ):
            bundle_label = "adapter bundles" if manifest.format_type.value == "adapter_bundle" else "Hugging Face bundles"
            return (
                f"Text {bundle_label} can target {fallback_runtime.name} on {system} {machine} "
                "after exporting or preparing a GGUF build for that host."
            )
        return None

    @staticmethod
    def _target_readiness_state(
        *,
        system: str,
        machine: str,
        host_platform: HostPlatformSnapshot,
        compatible_model_count: int,
        fallback_model_count: int,
    ) -> str:
        if RuntimeCatalog._matches_host_platform(host_platform, system=system, machine=machine):
            return "verified" if compatible_model_count > 0 else "blocked"
        if compatible_model_count > 0:
            return "declared"
        if fallback_model_count > 0:
            return "fallback_guided"
        return "blocked"

    @staticmethod
    def _matches_host_platform(
        host_platform: HostPlatformSnapshot,
        *,
        system: str,
        machine: str,
    ) -> bool:
        return (
            host_platform.system.casefold() == system.casefold()
            and host_platform.machine.casefold() == machine.casefold()
        )

    @staticmethod
    def _target_platforms() -> tuple[tuple[str, str], ...]:
        return (
            ("Darwin", "arm64"),
            ("Linux", "x86_64"),
            ("Linux", "arm64"),
            ("Windows", "AMD64"),
        )

    def _candidate_affinities(
        self,
        manifest: ModelManifest,
        *,
        request_modality: RequestModality | None = None,
    ) -> tuple[RuntimeAffinity, ...]:
        affinities: list[RuntimeAffinity] = []
        if request_modality == RequestModality.TEXT_ONLY:
            affinities.extend(manifest.text_only_runtime_affinity)
        affinities.extend(manifest.runtime_affinity)
        external_runtime = self.get_runtime(RuntimeAffinity.EXTERNAL_ACCELERATOR)
        if (
            external_runtime is not None
            and RuntimeAffinity.EXTERNAL_ACCELERATOR not in affinities
            and self._structurally_supports_manifest(external_runtime, manifest)
        ):
            affinities.append(RuntimeAffinity.EXTERNAL_ACCELERATOR)
        deduped: list[RuntimeAffinity] = []
        for affinity in affinities:
            if affinity not in deduped:
                deduped.append(affinity)
        return tuple(deduped)

    @staticmethod
    def _structurally_supports_manifest(runtime: RuntimeContract, manifest: ModelManifest) -> bool:
        return (
            manifest.format_type in runtime.supported_formats
            and any(modality in runtime.supported_modalities for modality in manifest.modality)
        )

    def _runtime_absence_reason(self, affinity: RuntimeAffinity) -> str:
        if self._pack_registry is None:
            return "No runtime registered for this affinity."
        return self._pack_registry.runtime_affinity_absence_reason(affinity) or "No runtime registered for this affinity."


def _runtime_candidate_readiness_state(
    runtime: RuntimeContract,
    manifest: ModelManifest,
) -> RuntimeReadinessState:
    if not runtime.supports_host_platform():
        return RuntimeReadinessState.HOST_UNSUPPORTED
    if not runtime.is_available():
        return RuntimeReadinessState.RUNTIME_UNAVAILABLE
    if not runtime.supports_manifest(manifest):
        return RuntimeReadinessState.MANIFEST_UNSUPPORTED
    return RuntimeReadinessState.READY


def build_default_runtime_catalog(
    settings: LewLMSettings,
    *,
    multimodal_encoder_cache: MultimodalEncoderCache | None = None,
    cluster_service: DistributedClusterService,
    pack_registry: PackRegistry | None = None,
    runtime_overrides: Mapping[RuntimeAffinity, RuntimeContract] | None = None,
) -> RuntimeCatalog:
    """Build the default runtime catalog, optionally overriding specific runtimes."""

    resolved_pack_registry = pack_registry or PackRegistry.from_settings(settings)
    runtimes: dict[RuntimeAffinity, RuntimeContract] = {}
    runtime_builders: dict[RuntimeAffinity, Callable[[], RuntimeContract]] = {
        RuntimeAffinity.DISTRIBUTED_EXPERIMENTAL: lambda: DistributedExperimentalRuntime(
            settings=settings,
            cluster_service=cluster_service,
        ),
        RuntimeAffinity.EXTERNAL_ACCELERATOR: lambda: LocalOpenAICompatibleAdapterRuntime(settings=settings),
        RuntimeAffinity.MLX_TEXT: lambda: MLXTextRuntime(settings=settings),
        RuntimeAffinity.MLX_AUDIO: lambda: MLXAudioRuntime(multimodal_encoder_cache=multimodal_encoder_cache),
        RuntimeAffinity.MLX_VISION: lambda: MLXVisionRuntime(
            settings=settings,
            multimodal_encoder_cache=multimodal_encoder_cache,
        ),
        RuntimeAffinity.LLAMACPP: lambda: LlamaCppRuntime(),
    }
    for affinity, builder in runtime_builders.items():
        if not resolved_pack_registry.runtime_affinity_load_enabled(affinity):
            continue
        runtimes[affinity] = builder()
    if (
        (runtime_overrides is None or RuntimeAffinity.EXPERIMENTAL not in runtime_overrides)
        and resolved_pack_registry.runtime_affinity_load_enabled(RuntimeAffinity.EXPERIMENTAL)
    ):
        try:
            runtimes[RuntimeAffinity.EXPERIMENTAL] = FrontierExperimentalRuntime(settings=settings)
        except TypeError:
            pass
    if runtime_overrides:
        for affinity, runtime in runtime_overrides.items():
            if not resolved_pack_registry.runtime_affinity_load_enabled(affinity):
                continue
            if hasattr(runtime, "_multimodal_encoder_cache"):
                setattr(runtime, "_multimodal_encoder_cache", multimodal_encoder_cache)
            runtimes[affinity] = runtime
    return RuntimeCatalog(runtimes, pack_registry=resolved_pack_registry)
