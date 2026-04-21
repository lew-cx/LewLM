"""Shared helpers for runtime backend implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
import platform
from typing import Any

from lewlm.core.contracts import (
    AudioSpeechRequest,
    AudioSpeechResponse,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    CapabilityName,
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    HostPlatformSnapshot,
    ModelFormat,
    ModelManifest,
    ModelModality,
    RerankRequest,
    RerankResponse,
    RuntimeCandidateReport,
    RuntimeAffinity,
    RuntimeEstimate,
    RuntimeReadinessState,
    utc_now,
)
from lewlm.core.errors import NotImplementedLewLMError, RuntimeUnavailableError, UnsupportedCapabilityError


class ManagedRuntime(ABC):
    """Common lifecycle and validation logic for LewLM runtimes."""

    name: str
    affinity: RuntimeAffinity
    supported_formats: tuple[ModelFormat, ...]
    supported_modalities: tuple[ModelModality, ...] = ()
    supported_capabilities: frozenset[CapabilityName] = frozenset()
    supported_systems: tuple[str, ...] = ()
    supported_machines: tuple[str, ...] = ()
    platform_guidance: str | None = None

    def __init__(self) -> None:
        self._loaded_manifests: dict[str, ModelManifest] = {}
        self._loaded_at: dict[str, datetime] = {}
        self._last_used_at: dict[str, datetime] = {}
        self._total_load_count = 0
        self._total_unload_count = 0
        self._total_warm_count = 0
        self._total_model_switch_count = 0
        self._peak_loaded_model_count = 0
        self._peak_estimated_memory_mb = 0

    @property
    def loaded_model_ids(self) -> tuple[str, ...]:
        return tuple(self._loaded_manifests)

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded_manifests

    def loaded_model_count(self) -> int:
        return len(self._loaded_manifests)

    def loaded_manifests(self) -> tuple[ModelManifest, ...]:
        return tuple(self._loaded_manifests.values())

    def last_used_at(self, model_id: str) -> datetime | None:
        return self._last_used_at.get(model_id)

    def is_available(self) -> bool:
        if not self.supports_host_platform():
            return False
        available, _ = self._check_environment()
        return available

    def availability_reason(self) -> str | None:
        platform_reason = self.host_platform_reason()
        if platform_reason is not None:
            return platform_reason
        _, reason = self._check_environment()
        return reason

    def supports_host_platform(self) -> bool:
        return self.supports_target_platform(platform.system(), platform.machine())

    def host_platform_reason(self) -> str | None:
        return self.target_platform_reason(platform.system(), platform.machine())

    def supports_target_platform(self, system: str, machine: str) -> bool:
        return self.target_platform_reason(system, machine) is None

    def target_platform_reason(self, system: str, machine: str) -> str | None:
        if self.supported_systems and not _matches_platform_value(system, self.supported_systems):
            supported_systems = ", ".join(self.supported_systems)
            return f"Supported on {supported_systems}; target system is {system}."
        if self.supported_machines and not _matches_platform_value(machine, self.supported_machines):
            supported_machines = ", ".join(self.supported_machines)
            return f"Supported on {supported_machines}; target machine is {machine}."
        return None

    def supports_manifest(self, manifest: ModelManifest) -> bool:
        return (
            manifest.format_type in self.supported_formats
            and any(modality in self.supported_modalities for modality in manifest.modality)
        )

    async def load_model(self, manifest: ModelManifest) -> None:
        self._ensure_available()
        self._ensure_supported_manifest(manifest)
        if manifest.model_id in self._loaded_manifests:
            self._touch_model(manifest.model_id)
            return
        if self._loaded_manifests:
            self._total_model_switch_count += 1
        await self._load_model(manifest)
        loaded_at = utc_now()
        self._loaded_manifests[manifest.model_id] = manifest
        self._loaded_at[manifest.model_id] = loaded_at
        self._last_used_at[manifest.model_id] = loaded_at
        self._total_load_count += 1
        self._update_peak_loaded_state()

    async def unload_model(self, model_id: str) -> None:
        if model_id not in self._loaded_manifests:
            return
        await self._unload_model(model_id)
        self._loaded_manifests.pop(model_id, None)
        self._loaded_at.pop(model_id, None)
        self._last_used_at.pop(model_id, None)
        self._total_unload_count += 1

    async def warm_model(self, model_id: str) -> None:
        self._ensure_loaded(model_id)
        await self._warm_model(model_id)
        self._touch_model(model_id)
        self._total_warm_count += 1

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self._unsupported_capability(CapabilityName.CHAT)

    async def stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        self._unsupported_capability(CapabilityName.STREAMING)
        if False:
            yield ""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._unsupported_capability(CapabilityName.EMBEDDINGS)

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        self._unsupported_capability(CapabilityName.RERANK)

    async def transcribe_audio(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse:
        self._unsupported_capability(CapabilityName.AUDIO_TRANSCRIPTION)

    async def synthesize_speech(self, request: AudioSpeechRequest) -> AudioSpeechResponse:
        self._unsupported_capability(CapabilityName.AUDIO_SPEECH)

    def tokenize(self, text: str) -> list[int]:
        self._unsupported_capability(CapabilityName.CHAT)

    def detokenize(self, tokens: Sequence[int]) -> str:
        self._unsupported_capability(CapabilityName.CHAT)

    def estimate_resources(self, manifest: ModelManifest) -> RuntimeEstimate:
        return RuntimeEstimate(
            estimated_memory_mb=manifest.estimated_memory_mb,
            notes=[f"Estimated for {self.name} using registry metadata only."],
        )

    def supports_capability(self, capability: CapabilityName) -> bool:
        return self.is_available() and capability in self.supported_capabilities

    def candidate_report(self, manifest: ModelManifest | None = None) -> RuntimeCandidateReport:
        return RuntimeCandidateReport(
            runtime_name=self.name,
            runtime_affinity=self.affinity,
            readiness_state=self._candidate_readiness_state(manifest),
            registered=True,
            available=self.is_available(),
            availability_reason=self.availability_reason(),
            host_platform_supported=self.supports_host_platform(),
            supported_systems=list(self.supported_systems),
            supported_machines=list(self.supported_machines),
            supports_manifest=self.supports_manifest(manifest) if manifest is not None else True,
        )

    def target_platform_status(
        self,
        system: str,
        machine: str,
        *,
        host_platform: HostPlatformSnapshot | None = None,
    ) -> dict[str, Any]:
        host_target = _matches_host_platform(host_platform, system=system, machine=machine)
        if not self.supports_target_platform(system, machine):
            return {
                "runtime_name": self.name,
                "runtime_affinity": self.affinity.value,
                "supported": False,
                "reason": self.target_platform_reason(system, machine),
                "readiness_state": "unsupported",
                "verification_method": "none",
                "install_hint": self.platform_guidance,
            }
        if host_target:
            available = self.is_available()
            return {
                "runtime_name": self.name,
                "runtime_affinity": self.affinity.value,
                "supported": available,
                "reason": None if available else self.availability_reason(),
                "readiness_state": "verified" if available else "host_unavailable",
                "verification_method": "host_probe",
                "install_hint": self.platform_guidance,
            }
        return {
            "runtime_name": self.name,
            "runtime_affinity": self.affinity.value,
            "supported": True,
            "reason": None,
            "readiness_state": "declared",
            "verification_method": "runtime_contract",
            "install_hint": self.platform_guidance,
        }

    async def health_check(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "affinity": self.affinity.value,
            "readiness_state": self._host_readiness_state().value,
            "available": self.is_available(),
            "availability_reason": self.availability_reason(),
            "host_platform_supported": self.supports_host_platform(),
            "host_platform_reason": self.host_platform_reason(),
            "supported_systems": list(self.supported_systems),
            "supported_machines": list(self.supported_machines),
            "loaded_model_ids": list(self.loaded_model_ids),
            "loaded_model_count": self.loaded_model_count(),
            "estimated_loaded_memory_mb": self._estimated_loaded_memory_mb(),
            "peak_loaded_model_count": self._peak_loaded_model_count,
            "peak_estimated_memory_mb": self._peak_estimated_memory_mb,
            "total_load_count": self._total_load_count,
            "total_unload_count": self._total_unload_count,
            "total_warm_count": self._total_warm_count,
            "total_model_switch_count": self._total_model_switch_count,
            "loaded_models": self._loaded_model_snapshot(),
            "supported_capabilities": sorted(
                capability.value
                for capability in CapabilityName
                if self.supports_capability(capability)
            ),
            "performance_features": self.performance_feature_snapshot(),
        }

    def performance_feature_snapshot(self) -> dict[str, Any]:
        return {}

    def _host_readiness_state(self) -> RuntimeReadinessState:
        if not self.supports_host_platform():
            return RuntimeReadinessState.HOST_UNSUPPORTED
        if not self.is_available():
            return RuntimeReadinessState.RUNTIME_UNAVAILABLE
        return RuntimeReadinessState.READY

    def _candidate_readiness_state(self, manifest: ModelManifest | None) -> RuntimeReadinessState:
        if not self.supports_host_platform():
            return RuntimeReadinessState.HOST_UNSUPPORTED
        if not self.is_available():
            return RuntimeReadinessState.RUNTIME_UNAVAILABLE
        if manifest is not None and not self.supports_manifest(manifest):
            return RuntimeReadinessState.MANIFEST_UNSUPPORTED
        return RuntimeReadinessState.READY

    def _ensure_available(self) -> None:
        if not self.is_available():
            raise RuntimeUnavailableError(
                f"{self.name} is unavailable on this system.",
                details={"runtime": self.name, "reason": self.availability_reason()},
            )

    def _ensure_supported_manifest(self, manifest: ModelManifest) -> None:
        if self.supports_manifest(manifest):
            return
        raise UnsupportedCapabilityError(
            f"{self.name} cannot load the selected model.",
            details={
                "runtime": self.name,
                "model_id": manifest.model_id,
                "format_type": manifest.format_type.value,
                "runtime_affinity": [affinity.value for affinity in manifest.runtime_affinity],
            },
        )

    def _ensure_loaded(self, model_id: str) -> None:
        if model_id not in self._loaded_manifests:
            raise RuntimeUnavailableError(
                "Model is not loaded in the selected runtime.",
                details={"runtime": self.name, "model_id": model_id},
            )

    def _touch_model(self, model_id: str) -> None:
        if model_id in self._loaded_manifests:
            self._last_used_at[model_id] = utc_now()

    def _loaded_model_snapshot(self) -> list[dict[str, object]]:
        snapshot: list[dict[str, object]] = []
        now = utc_now()
        for manifest in self._loaded_manifests.values():
            loaded_at = self._loaded_at.get(manifest.model_id)
            last_used_at = self._last_used_at.get(manifest.model_id)
            snapshot.append(
                {
                    "model_id": manifest.model_id,
                    "display_name": manifest.display_name,
                    "estimated_memory_mb": self._loaded_manifest_memory_mb(manifest),
                    "loaded_at": loaded_at.isoformat() if loaded_at is not None else None,
                    "last_used_at": last_used_at.isoformat() if last_used_at is not None else None,
                    "residency_seconds": (
                        round(max((now - loaded_at).total_seconds(), 0.0), 4)
                        if loaded_at is not None
                        else None
                    ),
                },
            )
        return snapshot

    def _estimated_loaded_memory_mb(self) -> int:
        return sum(
            self._loaded_manifest_memory_mb(manifest) or 0
            for manifest in self._loaded_manifests.values()
        )

    def _update_peak_loaded_state(self) -> None:
        self._peak_loaded_model_count = max(self._peak_loaded_model_count, self.loaded_model_count())
        self._peak_estimated_memory_mb = max(self._peak_estimated_memory_mb, self._estimated_loaded_memory_mb())

    def _loaded_manifest_memory_mb(self, manifest: ModelManifest) -> int | None:
        return manifest.estimated_memory_mb

    def _unsupported_capability(self, capability: CapabilityName) -> None:
        raise UnsupportedCapabilityError(
            f"{self.name} does not support `{capability.value}`.",
            details={"runtime": self.name, "capability": capability.value},
        )

    @abstractmethod
    def _check_environment(self) -> tuple[bool, str | None]:
        """Return runtime availability and an optional reason."""

    @abstractmethod
    async def _load_model(self, manifest: ModelManifest) -> None:
        """Load a manifest into the backend runtime."""

    @abstractmethod
    async def _unload_model(self, model_id: str) -> None:
        """Unload a model from the backend runtime."""

    async def _warm_model(self, model_id: str) -> None:
        """Warm an already loaded model."""


class ManagedTextRuntime(ManagedRuntime):
    """Common lifecycle and validation logic for text-generation runtimes."""

    supported_modalities: tuple[ModelModality, ...] = (ModelModality.TEXT, ModelModality.MULTIMODAL)
    supported_capabilities: frozenset[CapabilityName] = frozenset(
        {CapabilityName.CHAT, CapabilityName.STREAMING},
    )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        return await self._generate(request)

    async def stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        async for delta in self._stream_generate(request):
            yield delta

    def supports_continuous_batching(self, capability: CapabilityName) -> bool:
        return False

    def supports_chunked_prefill(self, capability: CapabilityName) -> bool:
        return False

    def supports_prefill_isolation(self, capability: CapabilityName) -> bool:
        return False

    async def generate_batch(self, requests: Sequence[GenerateRequest]) -> list[GenerateResponse]:
        raise NotImplementedLewLMError(
            f"{self.name} does not implement batched chat generation.",
            details={"runtime": self.name, "capability": CapabilityName.CHAT.value},
        )

    async def stream_generate_batch(self, requests: Sequence[GenerateRequest]) -> AsyncIterator[tuple[int, str]]:
        raise NotImplementedLewLMError(
            f"{self.name} does not implement batched streaming generation.",
            details={"runtime": self.name, "capability": CapabilityName.STREAMING.value},
        )

    def tokenize(self, text: str) -> list[int]:
        self._ensure_available()
        return self._tokenize(text)

    def detokenize(self, tokens: Sequence[int]) -> str:
        self._ensure_available()
        return self._detokenize(tokens)

    @abstractmethod
    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        """Produce a full text response."""

    @abstractmethod
    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        """Yield text deltas as they are generated."""

    @abstractmethod
    def _tokenize(self, text: str) -> list[int]:
        """Convert text to token IDs."""

    @abstractmethod
    def _detokenize(self, tokens: Sequence[int]) -> str:
        """Convert token IDs back to text."""


class ManagedAudioRuntime(ManagedRuntime):
    """Common lifecycle and validation logic for audio runtimes."""

    supported_modalities: tuple[ModelModality, ...] = (ModelModality.AUDIO,)
    supported_capabilities: frozenset[CapabilityName] = frozenset(
        {CapabilityName.AUDIO_TRANSCRIPTION, CapabilityName.AUDIO_SPEECH},
    )

    async def transcribe_audio(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        return await self._transcribe_audio(request)

    async def synthesize_speech(self, request: AudioSpeechRequest) -> AudioSpeechResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        return await self._synthesize_speech(request)

    @abstractmethod
    async def _transcribe_audio(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse:
        """Transcribe audio input into text."""

    @abstractmethod
    async def _synthesize_speech(self, request: AudioSpeechRequest) -> AudioSpeechResponse:
        """Synthesize speech audio from text."""


def _matches_platform_value(current_value: str, supported_values: tuple[str, ...]) -> bool:
    normalized_current = current_value.casefold()
    return any(normalized_current == value.casefold() for value in supported_values)


def _matches_host_platform(
    host_platform: HostPlatformSnapshot | None,
    *,
    system: str,
    machine: str,
) -> bool:
    if host_platform is None:
        return False
    return host_platform.system.casefold() == system.casefold() and host_platform.machine.casefold() == machine.casefold()
