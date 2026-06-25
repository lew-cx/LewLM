"""Runtime catalog and backend selection helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import ctypes
import os
from pathlib import Path
import platform
from typing import Literal

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    build_portable_performance_core_evidence,
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
    runtime_support_path_for_affinity,
)
from lewlm.core.errors import RoutingError
from lewlm.pack_registry import PackRegistry
from lewlm.runtime.llamacpp.runtime import LlamaCppRuntime
from lewlm.runtime.experimental import DistributedClusterService, DistributedExperimentalRuntime, FrontierExperimentalRuntime
from lewlm.runtime.adapters import LocalOpenAICompatibleAdapterRuntime
from lewlm.runtime.mlx_audio.runtime import MLXAudioRuntime
from lewlm.runtime.mlx_text.runtime import MLXTextRuntime
from lewlm.runtime.mlx_vision.runtime import MLXVisionRuntime
from lewlm.runtime.onnx_genai.runtime import ONNXGenAIRuntime
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
        total_memory_mb, total_memory_source, total_memory_reason = RuntimeCatalog.host_total_memory_snapshot()
        return HostPlatformSnapshot(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            python_version=platform.python_version(),
            total_memory_mb=total_memory_mb,
            total_memory_source=total_memory_source,
            total_memory_reason=total_memory_reason,
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
        required_capabilities = _required_runtime_capabilities(
            capability=capability,
            request_modality=request_modality,
        )
        for affinity in self._candidate_affinities(manifest, request_modality=request_modality):
            runtime = self.get_runtime(affinity)
            if runtime is None:
                alternatives.append(f"{affinity.value}: {self._runtime_absence_reason(affinity)}")
                continue
            candidate_report = getattr(runtime, "candidate_report", None)
            if callable(candidate_report):
                report = candidate_report(manifest)
                if not report.available:
                    alternatives.append(
                        f"{affinity.value}: {report.availability_reason or 'runtime unavailable'}",
                    )
                    continue
                if not report.supports_manifest:
                    alternatives.append(
                        f"{affinity.value}: {report.availability_reason or 'manifest unsupported'}",
                    )
                    continue
            else:
                if not runtime.is_available():
                    alternatives.append(f"{affinity.value}: {runtime.availability_reason() or 'runtime unavailable'}")
                    continue
                if not runtime.supports_manifest(manifest):
                    alternatives.append(f"{affinity.value}: manifest unsupported")
                    continue
            supports_manifest_capability = getattr(runtime, "supports_manifest_capability", None)
            manifest_capability_reason = getattr(runtime, "manifest_capability_reason", None)
            missing_required_capability = False
            for required_capability in required_capabilities:
                if callable(supports_manifest_capability):
                    if supports_manifest_capability(manifest, required_capability):
                        continue
                    reason = None
                    if callable(manifest_capability_reason):
                        reason = manifest_capability_reason(manifest, required_capability)
                    alternatives.append(
                        f"{affinity.value}: {reason or runtime.availability_reason() or 'capability unavailable'}",
                    )
                    missing_required_capability = True
                    break
                if runtime.supports_capability(required_capability):
                    continue
                reason = runtime.availability_reason() or "capability unavailable"
                alternatives.append(f"{affinity.value}: {reason}")
                missing_required_capability = True
                break
            if missing_required_capability:
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
                        support_path=runtime_support_path_for_affinity(affinity) or "packaged",
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
                    support_path=runtime_support_path_for_affinity(runtime.affinity) or "packaged",
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
                "performance_features": (performance_features := runtime.performance_feature_snapshot()),
                "performance_core_evidence": [
                    record.model_dump(mode="json")
                    for record in build_portable_performance_core_evidence(
                        performance_features=performance_features,
                        runtime_names=[runtime.name],
                    )
                ],
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
                    install_hints = self._fallback_install_hints_for_manifest(
                        manifest,
                        system=system,
                        machine=machine,
                    )
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
                if fallback_reason is not None:
                    install_hints.extend(
                        hint
                        for hint in self._fallback_install_hints_for_manifest(
                            manifest,
                            system=system,
                            machine=machine,
                        )
                        if hint not in install_hints
                    )
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
        total_memory_mb, _, _ = RuntimeCatalog.host_total_memory_snapshot()
        return total_memory_mb

    @staticmethod
    def host_total_memory_snapshot() -> tuple[int | None, str | None, str | None]:
        system = platform.system()
        if system == "Windows":
            total_memory_mb, reason = RuntimeCatalog._windows_total_memory_mb()
            return total_memory_mb, ("windows_globalmemorystatusex" if total_memory_mb is not None else None), reason
        total_memory_mb, reason = RuntimeCatalog._posix_total_memory_mb()
        if total_memory_mb is not None:
            return total_memory_mb, "posix_sysconf", None
        if system == "Linux":
            total_memory_mb, linux_reason = RuntimeCatalog._linux_proc_meminfo_total_memory_mb()
            if total_memory_mb is not None:
                return total_memory_mb, "linux_proc_meminfo", None
            return None, None, linux_reason or reason
        return None, None, reason

    @staticmethod
    def _posix_total_memory_mb() -> tuple[int | None, str | None]:
        sysconf_names = getattr(os, "sysconf_names", {})
        if not hasattr(os, "sysconf") or not sysconf_names:
            return None, "POSIX sysconf physical-memory probes are unavailable on this host."
        page_size_name = "SC_PAGE_SIZE"
        page_count_name = "SC_PHYS_PAGES"
        if page_size_name not in sysconf_names or page_count_name not in sysconf_names:
            return None, "POSIX sysconf did not expose physical-memory probe names on this host."
        try:
            page_size = int(os.sysconf(page_size_name))
            page_count = int(os.sysconf(page_count_name))
        except (OSError, ValueError):
            return None, "POSIX sysconf did not return usable physical-memory values."
        total_bytes = page_size * page_count
        total_memory_mb = RuntimeCatalog._bytes_to_mb(total_bytes)
        if total_memory_mb is None:
            return None, "POSIX sysconf returned a non-positive physical-memory total."
        return total_memory_mb, None

    @staticmethod
    def _linux_proc_meminfo_total_memory_mb() -> tuple[int | None, str | None]:
        meminfo_path = Path("/proc/meminfo")
        try:
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                if not line.startswith("MemTotal:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    break
                total_kib = int(parts[1])
                if total_kib > 0:
                    return max(1, total_kib // 1024), None
                break
        except (OSError, UnicodeDecodeError, ValueError):
            return None, "Linux /proc/meminfo could not be read for total-memory detection."
        return None, "Linux /proc/meminfo did not expose a usable MemTotal value."

    @staticmethod
    def _windows_total_memory_mb() -> tuple[int | None, str | None]:
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        except (AttributeError, OSError):
            return None, "Windows GlobalMemoryStatusEx is unavailable on this host."

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = _MemoryStatusEx()
        memory_status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        result = kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
        if not result:
            get_last_error = getattr(ctypes, "get_last_error", None)
            error_code = get_last_error() if callable(get_last_error) else 0
            detail = f" (WinError {error_code})" if error_code else ""
            return None, f"Windows GlobalMemoryStatusEx failed{detail}."
        total_memory_mb = RuntimeCatalog._bytes_to_mb(memory_status.ullTotalPhys)
        if total_memory_mb is None:
            return None, "Windows GlobalMemoryStatusEx returned a non-positive physical-memory total."
        return total_memory_mb, None

    @staticmethod
    def _bytes_to_mb(total_bytes: int) -> int | None:
        if total_bytes <= 0:
            return None
        return max(1, total_bytes // (1024 * 1024))

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
        if set(manifest.modality) == {ModelModality.TEXT}:
            fallback_runtime = self._text_fallback_runtime(system=system, machine=machine)
            if fallback_runtime is None:
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
        bridge_runtime = self._vision_bridge_runtime(manifest, system=system, machine=machine)
        if bridge_runtime is None:
            return None
        return (
            f"Image-conditioned chat on {system} {machine} currently stays bridge-backed via {bridge_runtime.name}; "
            "configure a compatible loopback-only local server that accepts OpenAI-style image content blocks on "
            "`/v1/chat/completions`."
        )

    def _fallback_install_hints_for_manifest(
        self,
        manifest: ModelManifest,
        *,
        system: str,
        machine: str,
    ) -> list[str]:
        runtime = None
        if set(manifest.modality) == {ModelModality.TEXT}:
            runtime = self._text_fallback_runtime(system=system, machine=machine)
        elif ModelModality.VISION in manifest.modality:
            runtime = self._vision_bridge_runtime(manifest, system=system, machine=machine)
        hint = getattr(runtime, "platform_guidance", None) if runtime is not None else None
        return [hint] if isinstance(hint, str) and hint else []

    def _text_fallback_runtime(self, *, system: str, machine: str) -> RuntimeContract | None:
        fallback_runtime = self.get_runtime(RuntimeAffinity.LLAMACPP)
        if fallback_runtime is None or not fallback_runtime.supports_target_platform(system, machine):
            return None
        return fallback_runtime

    def _vision_bridge_runtime(
        self,
        manifest: ModelManifest,
        *,
        system: str,
        machine: str,
    ) -> RuntimeContract | None:
        if system not in {"Linux", "Windows"} or ModelModality.VISION not in manifest.modality:
            return None
        bridge_runtime = self.get_runtime(RuntimeAffinity.EXTERNAL_ACCELERATOR)
        if bridge_runtime is None or not bridge_runtime.supports_target_platform(system, machine):
            return None
        if not self._structurally_supports_manifest(bridge_runtime, manifest):
            return None
        return bridge_runtime

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


def _required_runtime_capabilities(
    *,
    capability: CapabilityName,
    request_modality: RequestModality | None,
) -> tuple[CapabilityName, ...]:
    required = [capability]
    if capability in {CapabilityName.CHAT, CapabilityName.STREAMING} and request_modality in {
        RequestModality.IMAGE_CONDITIONED,
        RequestModality.FRAME_BUNDLE_VIDEO,
    }:
        required.append(CapabilityName.VISION)
    return tuple(required)


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
        RuntimeAffinity.ONNX_GENAI: lambda: ONNXGenAIRuntime(),
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
