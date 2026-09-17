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
from lewlm.core.errors import ModelLifecycleConflictError, RoutingError
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
        backend_feature_probes_enabled: bool = False,
        endpoint_runtimes: Mapping[str, RuntimeContract] | None = None,
    ) -> None:
        self._runtimes = dict(runtimes)
        self._endpoint_runtimes = dict(endpoint_runtimes or {})
        self._pack_registry = pack_registry
        self._backend_feature_probes_enabled = backend_feature_probes_enabled
        self.model_residency_manager = None

    @property
    def pack_registry(self) -> PackRegistry | None:
        return self._pack_registry

    def get_runtime(self, affinity: RuntimeAffinity) -> RuntimeContract | None:
        return self._runtimes.get(affinity)

    def get_endpoint_runtime(self, endpoint_id: str) -> RuntimeContract | None:
        return self._endpoint_runtimes.get(endpoint_id)

    def endpoint_runtimes(self) -> dict[str, RuntimeContract]:
        """Named endpoint runtimes keyed by endpoint id (a copy)."""

        return dict(self._endpoint_runtimes)

    def all_runtimes(self) -> tuple[RuntimeContract, ...]:
        return tuple({id(runtime): runtime for runtime in
                      (*self._runtimes.values(), *self._endpoint_runtimes.values())}.values())

    def _manifest_runtime(self, affinity: RuntimeAffinity, manifest: ModelManifest) -> RuntimeContract | None:
        if affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR:
            endpoint_id = manifest.metadata.get("external_endpoint_id")
            if isinstance(endpoint_id, str):
                return self.get_endpoint_runtime(endpoint_id)
            # Migrate existing Ollama manifests without changing their public IDs.
            ollama_endpoint = manifest.metadata.get("ollama_endpoint")
            if isinstance(ollama_endpoint, str) and self._endpoint_runtimes:
                from lewlm.config.endpoints import server_root
                for runtime in self._endpoint_runtimes.values():
                    endpoint = getattr(runtime, "endpoint", None)
                    if endpoint is not None and endpoint.profile == "ollama_local" and server_root(endpoint.base_url) == server_root(ollama_endpoint):
                        return runtime
                return None
        return self.get_runtime(affinity)

    def find_runtime_by_name(self, runtime_name: str) -> RuntimeContract | None:
        for runtime in self.all_runtimes():
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
        capability: CapabilityName | None,
        request_modality: RequestModality | None = None,
    ) -> tuple[list[RuntimeContract], list[str]]:
        alternatives: list[str] = []
        compatible: list[RuntimeContract] = []
        required_capabilities = _required_runtime_capabilities(
            capability=capability,
            request_modality=request_modality,
        )
        for affinity in self._candidate_affinities(manifest, request_modality=request_modality):
            runtime = self._manifest_runtime(affinity, manifest)
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

    def select_lifecycle_runtime(
        self,
        manifest: ModelManifest,
        *,
        runtime_name: str | None = None,
    ) -> RuntimeContract:
        """Select an available manifest-compatible runtime without requiring an inference capability."""

        compatible, alternatives = self.compatible_runtimes(manifest, capability=None)
        if runtime_name is not None:
            compatible = [runtime for runtime in compatible if runtime.name == runtime_name]
        if compatible:
            loaded = [runtime for runtime in compatible if runtime.is_model_loaded(manifest.model_id)]
            return loaded[0] if loaded else compatible[0]
        raise RoutingError(
            "No compatible runtime is currently available for the selected model lifecycle operation.",
            details={
                "model_id": manifest.model_id,
                "requested_runtime": runtime_name,
                "alternatives": alternatives,
            },
        )

    async def warm_model(self, manifest: ModelManifest) -> RuntimeContract:
        runtime = self.select_lifecycle_runtime(manifest)
        await runtime.load_model(manifest)
        await runtime.warm_model(manifest.model_id)
        return runtime

    async def unload_model(self, manifest: ModelManifest) -> RuntimeContract:
        runtime = self.select_lifecycle_runtime(manifest)
        await runtime.unload_model(manifest.model_id)
        return runtime

    def describe_manifest_runtimes(self, manifest: ModelManifest) -> list[RuntimeCandidateReport]:
        reports: list[RuntimeCandidateReport] = []
        for affinity in self._candidate_affinities(manifest):
            runtime = self._manifest_runtime(affinity, manifest)
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
            for runtime in self.all_runtimes():
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
        snapshots: list[dict[str, object]] = []
        for runtime in self.all_runtimes():
            if self._should_probe_runtime(runtime):
                snapshots.append(await runtime.health_check())
                continue
            lightweight_health_check = getattr(runtime, "lightweight_health_check", None)
            snapshots.append(
                await lightweight_health_check()
                if callable(lightweight_health_check)
                else await runtime.health_check()
            )
        return snapshots

    def performance_snapshot(self) -> list[dict[str, object]]:
        return [
            {
                "name": runtime.name,
                "available": runtime.is_available(),
                "supported_capabilities": self._supported_capability_names(runtime),
                "performance_features": (
                    performance_features := self.performance_features_for(runtime)
                ),
                "performance_core_evidence": [
                    record.model_dump(mode="json")
                    for record in build_portable_performance_core_evidence(
                        performance_features=performance_features,
                        runtime_names=[runtime.name],
                    )
                ],
            }
            for runtime in self.all_runtimes()
        ]

    def _supported_capability_names(self, runtime: RuntimeContract) -> list[str]:
        if isinstance(runtime, LocalOpenAICompatibleAdapterRuntime):
            return [capability.value for capability in runtime.cached_supported_capabilities()]
        if self._should_probe_runtime(runtime):
            capabilities = (
                capability
                for capability in CapabilityName
                if runtime.supports_capability(capability)
            )
        else:
            capabilities = getattr(runtime, "supported_capabilities", ())
        return sorted(capability.value for capability in capabilities)

    def performance_features_for(self, runtime: RuntimeContract) -> dict[str, object]:
        """Return feature details without importing an idle native MLX backend."""

        if isinstance(runtime, LocalOpenAICompatibleAdapterRuntime):
            return runtime.performance_feature_snapshot()
        return runtime.performance_feature_snapshot() if self._should_probe_runtime(runtime) else {}

    def _should_probe_runtime(self, runtime: RuntimeContract) -> bool:
        if isinstance(runtime, LocalOpenAICompatibleAdapterRuntime):
            return False
        module_name = type(runtime).__module__
        imports_native_mlx = module_name.startswith(
            (
                "lewlm.runtime.mlx_text.",
                "lewlm.runtime.mlx_vision.",
                "lewlm.runtime.mlx_audio.",
            ),
        )
        return (
            self._backend_feature_probes_enabled
            or bool(runtime.loaded_model_ids)
            or not imports_native_mlx
        )

    async def unload_all_models(self) -> None:
        for runtime in self.all_runtimes():
            for loaded_manifest in runtime.loaded_manifests():
                await runtime.unload_model(loaded_manifest.model_id)

    async def aclose(self) -> None:
        """Close runtime-owned connection pools and other async resources."""

        for runtime in self.all_runtimes():
            closer = getattr(runtime, "aclose", None)
            if callable(closer):
                await closer()

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
                runtime = self._manifest_runtime(affinity, manifest)
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
            if self.model_residency_manager is not None:
                await self.model_residency_manager.unload(runtime, manifest)
            else:
                await runtime.unload_model(manifest.model_id)
            return
        if policy != "balanced":
            return
        for loaded_manifest in runtime.loaded_manifests():
            if loaded_manifest.model_id == manifest.model_id:
                continue
            if self.model_residency_manager is None:
                await runtime.unload_model(loaded_manifest.model_id)
                continue
            try:
                await self.model_residency_manager.unload(runtime, loaded_manifest)
            except ModelLifecycleConflictError:
                # Balanced cleanup is opportunistic and must never interrupt a
                # model leased by another application request.
                continue

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
            runtime = self._manifest_runtime(affinity, manifest)
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
        bridge_runtime = self._manifest_runtime(RuntimeAffinity.EXTERNAL_ACCELERATOR, manifest)
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
        if manifest.metadata.get("external_endpoint_id") is not None:
            return (RuntimeAffinity.EXTERNAL_ACCELERATOR,)
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
        if affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR and self._endpoint_runtimes:
            return "No matching endpoint/default; bind the manifest with external_endpoint_id when multiple endpoints are configured."
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
    capability: CapabilityName | None,
    request_modality: RequestModality | None,
) -> tuple[CapabilityName, ...]:
    if capability is None:
        return ()
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
        RuntimeAffinity.MLX_AUDIO: lambda: MLXAudioRuntime(
            settings=settings,
            multimodal_encoder_cache=multimodal_encoder_cache,
        ),
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
        if affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR and settings.external_endpoints is not None:
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
    endpoint_runtimes = {}
    external_override = runtime_overrides.get(RuntimeAffinity.EXTERNAL_ACCELERATOR) if runtime_overrides else None
    if (settings.external_endpoints is not None
            and resolved_pack_registry.runtime_affinity_load_enabled(RuntimeAffinity.EXTERNAL_ACCELERATOR)
            and external_override is None):
        runtimes.pop(RuntimeAffinity.EXTERNAL_ACCELERATOR, None)
        endpoint_runtimes = {
            endpoint.endpoint_id: LocalOpenAICompatibleAdapterRuntime(settings=settings, endpoint=endpoint)
            for endpoint in settings.external_endpoints
        }
        # A single endpoint remains a compatible affinity default. Multiple
        # endpoints require a manifest binding; registration order is not policy.
        if len(endpoint_runtimes) == 1:
            runtimes[RuntimeAffinity.EXTERNAL_ACCELERATOR] = next(iter(endpoint_runtimes.values()))
    elif settings.external_endpoints is not None and external_override is not None:
        if len(settings.external_endpoints) != 1:
            raise ValueError("One affinity-level external runtime override cannot represent multiple named endpoints.")
        endpoint_runtimes[settings.external_endpoints[0].endpoint_id] = external_override
        runtimes[RuntimeAffinity.EXTERNAL_ACCELERATOR] = external_override
    elif settings.external_endpoints is None and RuntimeAffinity.EXTERNAL_ACCELERATOR in runtimes:
        endpoint_runtimes["legacy-default"] = runtimes[RuntimeAffinity.EXTERNAL_ACCELERATOR]
    return RuntimeCatalog(
        runtimes,
        pack_registry=resolved_pack_registry,
        backend_feature_probes_enabled=settings.backend_feature_probes_enabled,
        endpoint_runtimes=endpoint_runtimes,
    )
