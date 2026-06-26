"""Local-only adapter runtime for OpenAI-compatible accelerator servers."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from secrets import token_hex
import threading
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import wave

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    AudioSpeechRequest,
    AudioSpeechResponse,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioTranscriptionSegment,
    CapabilityName,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    ModelModality,
    PerformanceFeatureOwnership,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RuntimeAffinity,
    RuntimeCandidateReport,
    RuntimeReadinessState,
    normalize_performance_feature_ownership,
    normalize_runtime_performance_feature_report,
    runtime_performance_feature_report,
)
from lewlm.core.errors import RuntimeUnavailableError
from lewlm.runtime.base import ManagedTextRuntime
from lewlm.structured_output import StructuredOutputRequest, StructuredOutputRuntimeStatus

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_SUPPORTED_SYSTEMS = ("Darwin", "Linux", "Windows")
_SEMANTIC_ENDPOINTS = {
    CapabilityName.VISION: "/v1/chat/completions",
    CapabilityName.AUDIO_TRANSCRIPTION: "/v1/audio/transcriptions",
    CapabilityName.AUDIO_SPEECH: "/v1/audio/speech",
    CapabilityName.EMBEDDINGS: "/v1/embeddings",
    CapabilityName.RERANK: "/v1/rerank",
}
_IMAGE_SUFFIX_MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_IMAGE_SUFFIXES = frozenset(_IMAGE_SUFFIX_MEDIA_TYPES)
_VISION_PROBE_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlAbI4AAAAASUVORK5CYII="
)
_PERFORMANCE_FEATURE_ORDER = (
    "continuous_batching",
    "prefix_cache",
    "paged_kv_cache",
    "kv_cache_quantization",
    "prefill_optimization",
    "speculative_decoding",
    "constrained_decoding",
)
_FEATURE_LABELS = {
    "continuous_batching": "continuous batching",
    "prefix_cache": "prefix cache reuse",
    "paged_kv_cache": "paged KV cache",
    "kv_cache_quantization": "KV cache quantization",
    "prefill_optimization": "prefill optimization",
    "speculative_decoding": "speculative decoding",
    "constrained_decoding": "constrained decoding",
}
_PROFILE_FEATURES: dict[str, dict[str, tuple[PerformanceFeatureOwnership, str]]] = {
    "openai_compatible": {
        "continuous_batching": (
            PerformanceFeatureOwnership.PARTIAL,
            "Local scheduler overlap can be preserved, but batching visibility depends on the upstream server.",
        ),
        "prefix_cache": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Generic OpenAI-compatible endpoints do not expose prompt-prefix cache state or reuse counters.",
        ),
        "paged_kv_cache": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Paged KV cache behavior is not surfaced through the generic compatibility layer.",
        ),
        "kv_cache_quantization": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Per-request KV cache quantization controls are not surfaced through the compatibility layer.",
        ),
        "prefill_optimization": (
            PerformanceFeatureOwnership.PARTIAL,
            "Fast prefill may still happen inside the external engine, but request-level tuning knobs are not preserved.",
        ),
        "speculative_decoding": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Speculative decoding controls are not mapped through the local compatibility contract.",
        ),
        "constrained_decoding": (
            PerformanceFeatureOwnership.PARTIAL,
            "Structured-output requests can survive through prompt-guided fallback, but decoder-level constrained decoding is not preserved across the compatibility layer.",
        ),
    },
    "vmlx": {
        "continuous_batching": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "The external scheduler can preserve continuous batching for compatible local OpenAI-style requests.",
        ),
        "prefix_cache": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "vMLX-class servers preserve prompt reuse internally for repeated compatible prefixes.",
        ),
        "paged_kv_cache": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "Paged KV state remains available inside the external accelerator runtime.",
        ),
        "kv_cache_quantization": (
            PerformanceFeatureOwnership.PARTIAL,
            "KV cache quantization may remain active in the external engine, but LewLM cannot tune it per request.",
        ),
        "prefill_optimization": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "Prefill acceleration remains active for compatible requests on the external server.",
        ),
        "speculative_decoding": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Speculative decoding controls are not part of the adapter contract yet.",
        ),
        "constrained_decoding": (
            PerformanceFeatureOwnership.PARTIAL,
            "The adapter can preserve the structured-output contract, but decode-time token constraints remain owned by the upstream server and are not portable through LewLM.",
        ),
    },
    "omlx": {
        "continuous_batching": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "The external server can keep local request batching active for compatible workloads.",
        ),
        "prefix_cache": (
            PerformanceFeatureOwnership.PARTIAL,
            "Prefix reuse may stay active, but the adapter cannot surface detailed hit accounting.",
        ),
        "paged_kv_cache": (
            PerformanceFeatureOwnership.PARTIAL,
            "KV residency stays external, but LewLM cannot expose allocator-level paging details.",
        ),
        "kv_cache_quantization": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "KV cache quantization settings are not mapped into the adapter path.",
        ),
        "prefill_optimization": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "Prefill acceleration remains available for compatible requests.",
        ),
        "speculative_decoding": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Speculative decoding controls are not preserved through the compatibility layer.",
        ),
        "constrained_decoding": (
            PerformanceFeatureOwnership.PARTIAL,
            "Structured-output fallback remains available, but the adapter cannot claim portable decode-time constrained decoding for the upstream runtime.",
        ),
    },
    "vllm_mlx": {
        "continuous_batching": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "The external runtime preserves batched scheduling for local compatible requests.",
        ),
        "prefix_cache": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "Automatic prefix reuse remains available inside the external runtime.",
        ),
        "paged_kv_cache": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "Paged KV cache residency remains active on the external accelerator path.",
        ),
        "kv_cache_quantization": (
            PerformanceFeatureOwnership.PARTIAL,
            "Quantized KV residency may remain active, but LewLM cannot inspect or tune the policy directly.",
        ),
        "prefill_optimization": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "Prefill acceleration remains active for local compatible requests.",
        ),
        "speculative_decoding": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Speculative decoding remains outside the adapter contract.",
        ),
        "constrained_decoding": (
            PerformanceFeatureOwnership.PARTIAL,
            "Structured-output fallback remains available, but decode-time token constraints are not preserved through the adapter contract.",
        ),
    },
    "vllm_local": {
        "continuous_batching": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "vLLM-class local servers preserve continuous batching for compatible loopback chat workloads.",
        ),
        "prefix_cache": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "Automatic prefix caching can stay active inside the local vLLM server.",
        ),
        "paged_kv_cache": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "Paged KV residency remains managed inside the local vLLM runtime.",
        ),
        "kv_cache_quantization": (
            PerformanceFeatureOwnership.PARTIAL,
            "Quantized KV residency may remain active in the external server, but LewLM cannot inspect or tune it per request.",
        ),
        "prefill_optimization": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "Chunked and optimized prefill can remain active inside the local vLLM server.",
        ),
        "speculative_decoding": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Speculative decoding may exist upstream, but the current adapter contract does not preserve those controls.",
        ),
        "constrained_decoding": (
            PerformanceFeatureOwnership.PARTIAL,
            "Structured-output fallback remains available, but decode-time token constraints are not preserved through the adapter contract.",
        ),
    },
    "sglang_local": {
        "continuous_batching": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "SGLang-class local servers preserve batched scheduling for compatible loopback chat workloads.",
        ),
        "prefix_cache": (
            PerformanceFeatureOwnership.PARTIAL,
            "Prefix reuse can remain active upstream, but detailed cache hit accounting is not exposed through the adapter.",
        ),
        "paged_kv_cache": (
            PerformanceFeatureOwnership.PARTIAL,
            "Paged KV residency may remain active upstream, but allocator-level residency details are not surfaced through the adapter path.",
        ),
        "kv_cache_quantization": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "KV cache quantization controls are not preserved through the current OpenAI-compatible adapter contract.",
        ),
        "prefill_optimization": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "Prefill acceleration can remain active inside the local server for compatible requests.",
        ),
        "speculative_decoding": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Speculative decoding remains outside the current adapter contract even when the local server supports it internally.",
        ),
        "constrained_decoding": (
            PerformanceFeatureOwnership.PARTIAL,
            "Structured-output fallback remains available, but decoder-level constrained decoding does not cross the adapter boundary as a portable LewLM contract.",
        ),
    },
    "tensorrt_llm_server": {
        "continuous_batching": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "TensorRT-LLM-class local servers can preserve backend-native batching behind the OpenAI-compatible loopback boundary.",
        ),
        "prefix_cache": (
            PerformanceFeatureOwnership.PARTIAL,
            "KV reuse may remain active upstream, but LewLM cannot inspect TensorRT-LLM cache residency through the adapter.",
        ),
        "paged_kv_cache": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "Paged KV-style residency can remain backend-native inside the TensorRT-LLM server.",
        ),
        "kv_cache_quantization": (
            PerformanceFeatureOwnership.PARTIAL,
            "Quantized or compressed KV behavior may remain active upstream, but LewLM cannot tune it per request through the bridge.",
        ),
        "prefill_optimization": (
            PerformanceFeatureOwnership.BACKEND_NATIVE,
            "TensorRT-LLM prefill optimizations can remain active inside the local server for compatible requests.",
        ),
        "speculative_decoding": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Speculative decoding controls are not preserved through the current OpenAI-compatible adapter contract.",
        ),
        "constrained_decoding": (
            PerformanceFeatureOwnership.PARTIAL,
            "Structured-output fallback remains available, but portable decode-time constrained decoding is not claimed across this bridge.",
        ),
    },
    "openvino_model_server": {
        "continuous_batching": (
            PerformanceFeatureOwnership.PARTIAL,
            "OpenVINO Model Server may batch requests upstream, but LewLM cannot own or inspect that scheduler through the adapter.",
        ),
        "prefix_cache": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Prefix-cache behavior is not part of the portable OpenVINO Model Server bridge contract today.",
        ),
        "paged_kv_cache": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Paged KV residency is not claimed through the OpenVINO Model Server bridge profile.",
        ),
        "kv_cache_quantization": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "KV cache quantization controls are not preserved through the current OpenAI-compatible adapter contract.",
        ),
        "prefill_optimization": (
            PerformanceFeatureOwnership.PARTIAL,
            "CPU, GPU, or NPU graph/runtime optimizations may remain active upstream, but LewLM reports them as bridge-owned.",
        ),
        "speculative_decoding": (
            PerformanceFeatureOwnership.UNSUPPORTED,
            "Speculative decoding remains outside the OpenVINO bridge contract.",
        ),
        "constrained_decoding": (
            PerformanceFeatureOwnership.PARTIAL,
            "Structured-output fallback remains available, but decoder-level constraints are not claimed through this bridge.",
        ),
    },
}
_PROFILE_ALIASES = {
    "ollama_local": "openai_compatible",
    "llamacpp_server": "openai_compatible",
}


def summarize_feature_preservation(
    *,
    native_features: dict[str, Any],
    external_features: dict[str, Any],
) -> dict[str, Any]:
    preserved: list[str] = []
    degraded: list[str] = []
    rejected: list[str] = []
    details: dict[str, dict[str, Any]] = {}
    for feature_name in _PERFORMANCE_FEATURE_ORDER:
        native_entry = _feature_entry(native_features.get(feature_name))
        external_entry = _feature_entry(external_features.get(feature_name))
        if not native_entry["supported"]:
            continue
        native_rank = _feature_coverage_rank(native_entry["ownership"])
        external_rank = _feature_coverage_rank(external_entry["ownership"])
        status = "rejected"
        if external_entry["supported"] and external_rank >= native_rank:
            status = "preserved"
            preserved.append(feature_name)
        elif external_entry["supported"]:
            status = "degraded"
            degraded.append(feature_name)
        else:
            rejected.append(feature_name)
        details[feature_name] = {
            "feature": feature_name,
            "label": _FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ")),
            "status": status,
            "native": native_entry,
            "external": external_entry,
        }
    return {
        "preserved": preserved,
        "degraded": degraded,
        "rejected": rejected,
        "details": details,
    }


class LocalOpenAICompatibleAdapterRuntime(ManagedTextRuntime):
    """Route compatible local requests to a loopback-only OpenAI-style local server."""

    name = "local_external_adapter"
    affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR
    supported_formats = (ModelFormat.MLX, ModelFormat.GGUF, ModelFormat.AUDIO_FOLDER)
    supported_modalities = (
        ModelModality.TEXT,
        ModelModality.VISION,
        ModelModality.AUDIO,
        ModelModality.EMBEDDING,
        ModelModality.RERANK,
        ModelModality.MULTIMODAL,
    )
    supported_capabilities = frozenset(
        {
            CapabilityName.CHAT,
            CapabilityName.STREAMING,
            CapabilityName.VISION,
            CapabilityName.AUDIO_TRANSCRIPTION,
            CapabilityName.AUDIO_SPEECH,
            CapabilityName.EMBEDDINGS,
            CapabilityName.RERANK,
        },
    )
    supported_systems = _SUPPORTED_SYSTEMS
    platform_guidance = (
        "Enable LEWLM_EXTERNAL_ACCELERATOR_ENABLED and point "
        "LEWLM_EXTERNAL_ACCELERATOR_BASE_URL at a loopback-only local OpenAI-compatible server on this host."
    )

    def __init__(self, *, settings: LewLMSettings) -> None:
        super().__init__()
        self._settings = settings
        self._discovered_model_ids: tuple[str, ...] | None = None
        self._discovered_model_records: tuple[dict[str, Any], ...] | None = None
        self._discovery_error: str | None = None
        self._capability_support_cache: dict[CapabilityName, bool] = {}
        self._capability_reason_cache: dict[CapabilityName, str | None] = {}
        self._model_capability_support_cache: dict[tuple[str, CapabilityName], bool] = {}
        self._model_capability_reason_cache: dict[tuple[str, CapabilityName], str | None] = {}

    def supports_manifest(self, manifest: ModelManifest) -> bool:
        if not super().supports_manifest(manifest):
            return False
        if not self.is_available():
            return False
        try:
            return self._resolve_remote_model_id(manifest) is not None
        except RuntimeUnavailableError:
            return False

    def candidate_report(self, manifest: ModelManifest | None = None) -> RuntimeCandidateReport:
        report = super().candidate_report(manifest)
        if manifest is None or not report.available:
            return report
        try:
            remote_model_id = self._resolve_remote_model_id(manifest)
        except RuntimeUnavailableError as exc:
            return report.model_copy(
                update={
                    "available": False,
                    "readiness_state": RuntimeReadinessState.RUNTIME_UNAVAILABLE,
                    "supports_manifest": False,
                    "availability_reason": str(exc),
                },
            )
        if remote_model_id is not None:
            return report
        available_models = list(self._available_remote_models())
        return report.model_copy(
            update={
                "supports_manifest": False,
                "availability_reason": (
                    "The configured external accelerator endpoint did not advertise a compatible local model id. "
                    f"Available ids: {available_models or ['none discovered']}."
                ),
            },
        )

    def performance_feature_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for feature_name, (ownership, reason) in _profile_feature_map(self._settings).items():
            snapshot[feature_name] = runtime_performance_feature_report(
                ownership=ownership,
                active=ownership != PerformanceFeatureOwnership.UNSUPPORTED,
                reason=reason,
                metrics={
                    "adapter_profile": self._settings.external_accelerator_profile,
                    "contract": "openai_compatible_local",
                    **(
                        {
                            "decoder_enforced": False,
                            "fallback_used": True,
                            "enforcement": "prompt_guided",
                        }
                        if feature_name == "constrained_decoding"
                        else {}
                    ),
                },
                modes=(["prompt_guided"] if feature_name == "constrained_decoding" else []),
                notes=(
                    [
                        "LewLM reports only the portable contract preserved across the loopback adapter boundary; deeper scheduler or cache internals remain owned by the upstream server."
                    ]
                    if ownership != PerformanceFeatureOwnership.UNSUPPORTED
                    else []
                ),
            )
        return snapshot

    def structured_output_runtime_status(
        self,
        contract: StructuredOutputRequest | None,
    ) -> StructuredOutputRuntimeStatus | None:
        if contract is None or contract.type == "text":
            return None
        _, reason = _profile_feature_map(self._settings)["constrained_decoding"]
        return StructuredOutputRuntimeStatus(
            runtime=self.name,
            mode=contract.type,
            enforcement="prompt_guided",
            decoder_enforced=False,
            fallback_used=True,
            fallback_reason=f"{reason} This remains a loopback adapter boundary rather than packaged decode-time parity.",
        )

    def supports_capability(self, capability: CapabilityName) -> bool:
        if capability in {CapabilityName.CHAT, CapabilityName.STREAMING}:
            return super().supports_capability(capability)
        if capability not in self.supported_capabilities or not self.is_available():
            return False
        if capability in self._capability_support_cache:
            return self._capability_support_cache[capability]
        try:
            for remote_model_id in self._available_remote_models():
                supported, reason = self._probe_remote_model_capability(remote_model_id, capability)
                if supported:
                    self._capability_support_cache[capability] = True
                    self._capability_reason_cache[capability] = None
                    return True
                if reason:
                    self._capability_reason_cache[capability] = reason
        except RuntimeUnavailableError as exc:
            self._capability_support_cache[capability] = False
            self._capability_reason_cache[capability] = str(exc)
            return False
        self._capability_support_cache[capability] = False
        if capability not in self._capability_reason_cache:
            self._capability_reason_cache[capability] = (
                "The configured external accelerator did not advertise any local models."
                if not self._available_remote_models()
                else f"No advertised local model accepted `{capability.value}` through the adapter contract."
            )
        return False

    def _record_structured_output_runtime(self, request: GenerateRequest) -> None:
        status = self.structured_output_runtime_status(request.structured_output)
        if status is None:
            return
        request.metadata["structured_output_runtime"] = status.model_dump(mode="json")

    def supports_manifest_capability(self, manifest: ModelManifest, capability: CapabilityName) -> bool:
        if not self.supports_manifest(manifest):
            return False
        if not _manifest_supports_external_capability(manifest, capability):
            return False
        if capability in {CapabilityName.CHAT, CapabilityName.STREAMING}:
            return True
        remote_model_id = self._resolve_remote_model_id(manifest)
        if remote_model_id is None:
            return False
        supported, _ = self._probe_remote_model_capability(remote_model_id, capability)
        return supported

    def manifest_capability_reason(self, manifest: ModelManifest, capability: CapabilityName) -> str | None:
        if not self.supports_manifest(manifest):
            return (
                "The configured external accelerator endpoint did not advertise a compatible local model id. "
                f"Available ids: {list(self._available_remote_models()) or ['none discovered']}."
            )
        if not _manifest_supports_external_capability(manifest, capability):
            required_modalities = ", ".join(
                modality.value
                for modality in _external_capability_modalities(capability)
            )
            return (
                f"The external accelerator bridge only supports `{capability.value}` for manifests that include "
                f"{required_modalities}."
            )
        if capability in {CapabilityName.CHAT, CapabilityName.STREAMING}:
            return None
        remote_model_id = self._resolve_remote_model_id(manifest)
        if remote_model_id is None:
            return (
                "The configured external accelerator endpoint did not advertise a compatible local model id. "
                f"Available ids: {list(self._available_remote_models()) or ['none discovered']}."
            )
        _, reason = self._probe_remote_model_capability(remote_model_id, capability)
        return reason

    async def _load_model(self, manifest: ModelManifest) -> None:
        remote_model_id = self._resolve_remote_model_id(manifest)
        if remote_model_id is None:
            raise RuntimeUnavailableError(
                "The configured external accelerator does not advertise a compatible local model.",
                details={
                    "runtime": self.name,
                    "model_id": manifest.model_id,
                    "advertised_model_ids": list(self._available_remote_models()),
                },
            )

    async def _unload_model(self, model_id: str) -> None:
        return None

    async def _warm_model(self, model_id: str) -> None:
        manifest = self._loaded_manifests[model_id]
        request = GenerateRequest(
            model_id=model_id,
            messages=[{"role": "user", "content": "Warm the local accelerator path."}],
            max_tokens=1,
        )
        await self._generate_with_manifest(manifest, request)

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        manifest = self._loaded_manifests[request.model_id]
        self._record_structured_output_runtime(request)
        return await self._generate_with_manifest(manifest, request)

    async def _stream_generate(self, request: GenerateRequest):
        manifest = self._loaded_manifests[request.model_id]
        self._record_structured_output_runtime(request)
        remote_model_id = self._require_remote_model_id(manifest)
        payload = self._chat_payload(remote_model_id=remote_model_id, request=request, stream=True)
        queue: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()

        def _worker() -> None:
            try:
                for delta in self._stream_chat_completion(payload):
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
            except Exception as exc:  # pragma: no cover - surfaced through queue
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        threading.Thread(target=_worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield cast(str, item)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        manifest = self._loaded_manifests[request.model_id]
        remote_model_id = self._require_remote_model_id(manifest)
        supported, reason = self._probe_remote_model_capability(remote_model_id, CapabilityName.EMBEDDINGS)
        if not supported:
            raise RuntimeUnavailableError(
                reason or "The configured external accelerator could not satisfy `embeddings`.",
                details={
                    "runtime": self.name,
                    "model_id": request.model_id,
                    "remote_model_id": remote_model_id,
                    "capability": CapabilityName.EMBEDDINGS.value,
                },
            )
        payload = await asyncio.to_thread(
            self._request_json,
            "POST",
            _SEMANTIC_ENDPOINTS[CapabilityName.EMBEDDINGS],
            {"model": remote_model_id, "input": request.inputs},
        )
        data_payload = payload.get("data", payload.get("embeddings", payload.get("vectors", [])))
        usage_payload = payload.get("usage", {})
        vectors = _normalize_embedding_payload(data_payload)
        if len(vectors) != len(request.inputs):
            raise RuntimeUnavailableError(
                _semantic_invalid_payload_reason(
                    capability=CapabilityName.EMBEDDINGS,
                    remote_model_id=remote_model_id,
                ),
                details={
                    "runtime": self.name,
                    "model_id": request.model_id,
                    "remote_model_id": remote_model_id,
                    "capability": CapabilityName.EMBEDDINGS.value,
                    "vector_count": len(vectors),
                    "input_count": len(request.inputs),
                },
            )
        usage = _normalize_usage(usage_payload)
        prompt_tokens = usage.get("prompt_tokens", sum(max(1, len(text.split())) for text in request.inputs))
        return EmbeddingResponse(
            model_id=request.model_id,
            data=[EmbeddingVector(index=index, embedding=vector) for index, vector in enumerate(vectors)],
            usage={
                "prompt_tokens": prompt_tokens,
                "total_tokens": usage.get("total_tokens", prompt_tokens),
            },
        )

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        manifest = self._loaded_manifests[request.model_id]
        remote_model_id = self._require_remote_model_id(manifest)
        supported, reason = self._probe_remote_model_capability(remote_model_id, CapabilityName.RERANK)
        if not supported:
            raise RuntimeUnavailableError(
                reason or "The configured external accelerator could not satisfy `rerank`.",
                details={
                    "runtime": self.name,
                    "model_id": request.model_id,
                    "remote_model_id": remote_model_id,
                    "capability": CapabilityName.RERANK.value,
                },
            )
        payload = await asyncio.to_thread(
            self._request_json,
            "POST",
            _SEMANTIC_ENDPOINTS[CapabilityName.RERANK],
            {
                "model": remote_model_id,
                "query": request.query,
                "documents": request.documents,
                "top_n": request.top_n,
            },
        )
        results_payload = payload.get("results", payload.get("data", payload.get("scores", [])))
        results = _normalize_rerank_payload(results_payload, request)
        if request.top_n is not None:
            results = results[: request.top_n]
        return RerankResponse(model_id=request.model_id, results=results)

    async def transcribe_audio(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        manifest = self._loaded_manifests[request.model_id]
        remote_model_id = self._require_remote_model_id(manifest)
        supported, reason = self._probe_remote_model_capability(remote_model_id, CapabilityName.AUDIO_TRANSCRIPTION)
        if not supported:
            raise RuntimeUnavailableError(
                reason or "The configured external accelerator could not satisfy `audio_transcription`.",
                details=_bridge_capability_error_details(
                    runtime_name=self.name,
                    model_id=request.model_id,
                    remote_model_id=remote_model_id,
                    capability=CapabilityName.AUDIO_TRANSCRIPTION,
                ),
            )
        payload = await asyncio.to_thread(
            self._request_multipart_json,
            "POST",
            _SEMANTIC_ENDPOINTS[CapabilityName.AUDIO_TRANSCRIPTION],
            {
                "model": remote_model_id,
                "language": request.language,
                "prompt": request.prompt,
            },
            {
                "file": (
                    request.file_name,
                    request.audio_bytes,
                    _audio_media_type_for_bytes(request.audio_bytes),
                ),
            },
        )
        return _normalize_audio_transcription_response(payload, request)

    async def synthesize_speech(self, request: AudioSpeechRequest) -> AudioSpeechResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        manifest = self._loaded_manifests[request.model_id]
        remote_model_id = self._require_remote_model_id(manifest)
        supported, reason = self._probe_remote_model_capability(remote_model_id, CapabilityName.AUDIO_SPEECH)
        if not supported:
            raise RuntimeUnavailableError(
                reason or "The configured external accelerator could not satisfy `audio_speech`.",
                details=_bridge_capability_error_details(
                    runtime_name=self.name,
                    model_id=request.model_id,
                    remote_model_id=remote_model_id,
                    capability=CapabilityName.AUDIO_SPEECH,
                ),
            )
        audio_bytes, media_type = await asyncio.to_thread(
            self._request_bytes,
            "POST",
            _SEMANTIC_ENDPOINTS[CapabilityName.AUDIO_SPEECH],
            {
                "model": remote_model_id,
                "input": request.input_text,
                "voice": request.voice or "alloy",
                "response_format": request.audio_format,
            },
        )
        return AudioSpeechResponse(
            model_id=request.model_id,
            audio_bytes=audio_bytes,
            media_type=media_type,
            voice=request.voice or "alloy",
            duration_seconds=_duration_seconds_from_audio_bytes(audio_bytes, media_type=media_type),
        )

    def _tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")

    def _check_environment(self) -> tuple[bool, str | None]:
        if not self._settings.external_accelerator_enabled:
            return False, "External accelerator adapter is disabled."
        base_url = self._settings.external_accelerator_base_url
        if base_url is None:
            return False, "Set LEWLM_EXTERNAL_ACCELERATOR_BASE_URL to a local loopback endpoint."
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            return False, "External accelerator base URL must use http or https."
        host = parsed.hostname
        if host not in _LOOPBACK_HOSTS:
            return False, "External accelerator base URL must target a loopback-only local host."
        return True, None

    async def _generate_with_manifest(self, manifest: ModelManifest, request: GenerateRequest) -> GenerateResponse:
        remote_model_id = self._require_remote_model_id(manifest)
        payload = self._chat_payload(remote_model_id=remote_model_id, request=request, stream=False)
        response_payload = await asyncio.to_thread(self._request_json, "POST", "/v1/chat/completions", payload)
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeUnavailableError(
                "External accelerator returned an invalid chat completion payload.",
                details={"runtime": self.name, "response_keys": sorted(response_payload)},
            )
        message = choices[0].get("message", {})
        output_text = _normalize_content_text(message.get("content"))
        usage = response_payload.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        return GenerateResponse(
            model_id=request.model_id,
            output_text=output_text,
            finish_reason=str(choices[0].get("finish_reason", "stop")),
            usage={
                key: int(value)
                for key, value in usage.items()
                if isinstance(key, str) and isinstance(value, int | float)
            },
        )

    def _require_remote_model_id(self, manifest: ModelManifest) -> str:
        remote_model_id = self._resolve_remote_model_id(manifest)
        if remote_model_id is None:
            raise RuntimeUnavailableError(
                "External accelerator adapter could not match the manifest to a local advertised model.",
                details={
                    "runtime": self.name,
                    "model_id": manifest.model_id,
                    "advertised_model_ids": list(self._available_remote_models()),
                },
            )
        return remote_model_id

    def _resolve_remote_model_id(self, manifest: ModelManifest) -> str | None:
        available_ids = {model_id.casefold(): model_id for model_id in self._available_remote_models()}
        for candidate in _remote_model_candidates(manifest):
            resolved = available_ids.get(candidate.casefold())
            if resolved is not None:
                return resolved
        record_candidates: dict[str, str] = {}
        for record in self._available_remote_model_records():
            model_id = record.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            for candidate in _remote_record_candidates(record):
                record_candidates.setdefault(candidate.casefold(), model_id)
        for candidate in _remote_model_candidates(manifest):
            resolved = record_candidates.get(candidate.casefold())
            if resolved is not None:
                return resolved
        return None

    def _available_remote_models(self) -> tuple[str, ...]:
        if self._discovered_model_ids is not None:
            return self._discovered_model_ids
        payload = self._request_json("GET", "/v1/models", None)
        data = payload.get("data")
        model_ids: list[str] = []
        discovered_records: list[dict[str, Any]] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                discovered_records.append(item)
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id:
                    model_ids.append(model_id)
        self._discovered_model_ids = tuple(model_ids)
        self._discovered_model_records = tuple(discovered_records)
        self._discovery_error = None
        return self._discovered_model_ids

    def _available_remote_model_records(self) -> tuple[dict[str, Any], ...]:
        if self._discovered_model_records is None:
            self._available_remote_models()
        return self._discovered_model_records or ()

    def _probe_remote_model_capability(
        self,
        remote_model_id: str,
        capability: CapabilityName,
    ) -> tuple[bool, str | None]:
        cache_key = (remote_model_id, capability)
        if cache_key in self._model_capability_support_cache:
            return (
                self._model_capability_support_cache[cache_key],
                self._model_capability_reason_cache.get(cache_key),
            )
        try:
            if capability == CapabilityName.AUDIO_TRANSCRIPTION:
                payload = self._request_multipart_json(
                    "POST",
                    _SEMANTIC_ENDPOINTS[capability],
                    {"model": remote_model_id, "language": "en", "prompt": "LewLM audio probe"},
                    {
                        "file": (
                            "probe.wav",
                            _probe_audio_bytes(),
                            "audio/wav",
                        ),
                    },
                )
                supported = _semantic_probe_payload_is_usable(capability=capability, payload=payload)
                reason = None
            elif capability == CapabilityName.AUDIO_SPEECH:
                audio_bytes, media_type = self._request_bytes(
                    "POST",
                    _SEMANTIC_ENDPOINTS[capability],
                    {
                        "model": remote_model_id,
                        "input": "LewLM audio probe",
                        "voice": "alloy",
                        "response_format": "wav",
                    },
                )
                supported = bool(audio_bytes) and media_type.startswith("audio/")
                reason = None
            else:
                payload = self._request_json(
                    "POST",
                    _SEMANTIC_ENDPOINTS[capability],
                    _semantic_probe_payload(remote_model_id=remote_model_id, capability=capability),
                )
                supported = _semantic_probe_payload_is_usable(capability=capability, payload=payload)
                reason = None
        except RuntimeUnavailableError as exc:
            reason = _semantic_probe_failure_reason(
                error=exc,
                capability=capability,
                remote_model_id=remote_model_id,
            )
            self._model_capability_support_cache[cache_key] = False
            self._model_capability_reason_cache[cache_key] = reason
            return False, reason
        if not supported:
            reason = _semantic_invalid_payload_reason(
                capability=capability,
                remote_model_id=remote_model_id,
            )
        self._model_capability_support_cache[cache_key] = supported
        self._model_capability_reason_cache[cache_key] = reason
        return supported, reason

    def _chat_payload(self, *, remote_model_id: str, request: GenerateRequest, stream: bool) -> dict[str, Any]:
        return {
            "model": remote_model_id,
            "messages": [_message_payload(message) for message in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }

    def _stream_chat_completion(self, payload: dict[str, Any]):
        with self._request("POST", "/v1/chat/completions", payload=payload) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeUnavailableError(
                        "External accelerator returned malformed streaming JSON.",
                        details={"runtime": self.name, "line": data},
                    ) from exc
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta", {})
                if not isinstance(delta, dict):
                    continue
                content = _normalize_content_text(delta.get("content"))
                if content:
                    yield content

    def _request_multipart_json(
        self,
        method: str,
        path: str,
        fields: dict[str, Any],
        files: dict[str, tuple[str, bytes, str]],
    ) -> dict[str, Any]:
        boundary, data = _multipart_form_data(fields=fields, files=files)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        with self._request(method, path, body=data, headers=headers) as response:
            body = response.read().decode("utf-8", errors="ignore")
        if not body:
            return {}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeUnavailableError(
                "External accelerator returned malformed JSON.",
                details={"runtime": self.name, "path": path},
            ) from exc
        if isinstance(parsed, dict):
            return parsed
        raise RuntimeUnavailableError(
            "External accelerator returned an unexpected JSON payload.",
            details={"runtime": self.name, "path": path, "payload_type": type(parsed).__name__},
        )

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        with self._request(method, path, payload=payload) as response:
            body = response.read().decode("utf-8", errors="ignore")
        if not body:
            return {}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeUnavailableError(
                "External accelerator returned malformed JSON.",
                details={"runtime": self.name, "path": path},
            ) from exc
        if isinstance(parsed, dict):
            return parsed
        raise RuntimeUnavailableError(
            "External accelerator returned an unexpected JSON payload.",
            details={"runtime": self.name, "path": path, "payload_type": type(parsed).__name__},
        )

    def _request_bytes(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[bytes, str]:
        with self._request(
            method,
            path,
            payload=payload,
            accept="audio/*,application/octet-stream",
        ) as response:
            media_type = _response_media_type(response)
            return response.read(), media_type

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        accept: str = "application/json",
    ):
        available, reason = self._check_environment()
        if not available:
            raise RuntimeUnavailableError(
                reason or "External accelerator adapter is unavailable.",
                details={"runtime": self.name},
            )
        base_url = self._settings.external_accelerator_base_url or ""
        if payload is not None and body is not None:
            raise ValueError("payload and body cannot both be provided to the external accelerator request helper.")
        request_headers = {"Accept": accept, **(headers or {})}
        data: bytes | None = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        elif body is not None:
            data = body
        request_url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
        request = Request(
            request_url,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            return urlopen(request, timeout=self._settings.external_accelerator_timeout_seconds)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeUnavailableError(
                f"External accelerator request failed with HTTP {exc.code}.",
                details={"runtime": self.name, "path": path, "body": body, "status_code": exc.code},
            ) from exc
        except URLError as exc:
            self._discovered_model_ids = ()
            self._discovered_model_records = ()
            self._discovery_error = str(exc.reason)
            if isinstance(exc.reason, ConnectionRefusedError):
                raise RuntimeUnavailableError(
                    f"The configured external accelerator endpoint `{request_url}` refused the connection.",
                    details={"runtime": self.name, "path": path, "reason": str(exc.reason)},
                ) from exc
            raise RuntimeUnavailableError(
                "Could not reach the configured external accelerator endpoint.",
                details={"runtime": self.name, "path": path, "reason": str(exc.reason)},
            ) from exc


def _profile_feature_map(settings: LewLMSettings) -> dict[str, tuple[PerformanceFeatureOwnership, str]]:
    profile = _PROFILE_ALIASES.get(settings.external_accelerator_profile, settings.external_accelerator_profile)
    return _PROFILE_FEATURES.get(profile, _PROFILE_FEATURES["openai_compatible"])


def _remote_model_candidates(manifest: ModelManifest) -> tuple[str, ...]:
    candidates: list[str] = []
    explicit_model_id = manifest.metadata.get("external_adapter_model_id")
    if isinstance(explicit_model_id, str) and explicit_model_id:
        candidates.append(explicit_model_id)
    explicit_model_ids = manifest.metadata.get("external_adapter_model_ids")
    if isinstance(explicit_model_ids, list):
        candidates.extend(
            item
            for item in explicit_model_ids
            if isinstance(item, str) and item
        )
    source_model_id = manifest.metadata.get("source_model_id")
    if isinstance(source_model_id, str) and source_model_id:
        candidates.append(source_model_id)
    source_display_name = manifest.metadata.get("source_display_name")
    if isinstance(source_display_name, str) and source_display_name:
        candidates.append(source_display_name)
    candidates.extend((manifest.model_id, manifest.display_name, *_portable_path_name_candidates(manifest.source_path)))
    for layer in manifest.artifact_lineage:
        candidates.extend((layer.display_name, *_portable_path_name_candidates(layer.source_path)))
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def _remote_record_candidates(record: dict[str, Any]) -> tuple[str, ...]:
    candidates: list[str] = []
    model_id = record.get("id")
    if isinstance(model_id, str) and model_id:
        candidates.append(model_id)
    root = record.get("root")
    if isinstance(root, str) and root:
        candidates.extend(_portable_path_name_candidates(root))
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        aliases = metadata.get("aliases")
        if isinstance(aliases, list):
            candidates.extend(item for item in aliases if isinstance(item, str) and item)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def _portable_path_name_candidates(raw_path: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for pure_path in (PurePosixPath(raw_path), PureWindowsPath(raw_path)):
        for item in (pure_path.name, pure_path.stem):
            if item and item not in candidates:
                candidates.append(item)
    return tuple(candidates)


def _feature_entry(payload: Any) -> dict[str, Any]:
    normalized = normalize_runtime_performance_feature_report(payload if isinstance(payload, dict) else None)
    return {
        "supported": bool(normalized.get("supported")),
        "support_level": str(normalized.get("support_level", "unsupported")),
        "ownership": str(normalized.get("ownership", PerformanceFeatureOwnership.UNSUPPORTED.value)),
        "reason": normalized.get("reason"),
        "metrics": normalized.get("metrics", {}),
    }


def _feature_coverage_rank(ownership: str) -> int:
    normalized = normalize_performance_feature_ownership(ownership=ownership)
    if normalized in {
        PerformanceFeatureOwnership.LEWLM_OWNED,
        PerformanceFeatureOwnership.BACKEND_NATIVE,
    }:
        return 2
    if normalized == PerformanceFeatureOwnership.PARTIAL:
        return 1
    return 0


def _semantic_probe_payload(*, remote_model_id: str, capability: CapabilityName) -> dict[str, Any]:
    if capability == CapabilityName.VISION:
        return {
            "model": remote_model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "LewLM vision capability probe"},
                        {"type": "image_url", "image_url": {"url": _VISION_PROBE_IMAGE_URL}},
                    ],
                },
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
    if capability == CapabilityName.EMBEDDINGS:
        return {"model": remote_model_id, "input": ["LewLM semantic capability probe"]}
    if capability == CapabilityName.RERANK:
        return {
            "model": remote_model_id,
            "query": "LewLM semantic capability probe",
            "documents": ["LewLM semantic capability probe"],
            "top_n": 1,
        }
    raise ValueError(f"Unsupported semantic capability probe: {capability.value}")


def _semantic_probe_payload_is_usable(*, capability: CapabilityName, payload: dict[str, Any]) -> bool:
    if capability == CapabilityName.VISION:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return False
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            return False
        content = _normalize_content_text(message.get("content"))
        return bool(content)
    if capability == CapabilityName.AUDIO_TRANSCRIPTION:
        text = payload.get("text")
        return isinstance(text, str) and bool(text.strip())
    if capability == CapabilityName.EMBEDDINGS:
        return isinstance(payload.get("data", payload.get("embeddings", payload.get("vectors"))), list)
    if capability == CapabilityName.RERANK:
        results = payload.get("results", payload.get("data", payload.get("scores")))
        return isinstance(results, list)
    return False


def _semantic_probe_failure_reason(
    *,
    error: RuntimeUnavailableError,
    capability: CapabilityName,
    remote_model_id: str,
) -> str:
    path = _SEMANTIC_ENDPOINTS[capability]
    body = error.details.get("body") if isinstance(error.details, dict) else None
    response_detail = f" Upstream response: {body}" if isinstance(body, str) and body else ""
    if capability == CapabilityName.VISION:
        return _vision_probe_failure_reason(
            error=error,
            remote_model_id=remote_model_id,
            path=path,
            response_detail=response_detail,
        )
    if capability in {CapabilityName.AUDIO_TRANSCRIPTION, CapabilityName.AUDIO_SPEECH}:
        return _audio_probe_failure_reason(
            capability=capability,
            error=error,
            remote_model_id=remote_model_id,
            path=path,
            response_detail=response_detail,
        )
    return (
        f"The configured external accelerator could not satisfy `{capability.value}` for local model "
        f"`{remote_model_id}` via `{path}`. {_bridge_endpoint_sentence(capability)}{response_detail}"
    )


def _vision_probe_failure_reason(
    *,
    error: RuntimeUnavailableError,
    remote_model_id: str,
    path: str,
    response_detail: str,
) -> str:
    details = error.details if isinstance(error.details, dict) else {}
    status_code = details.get("status_code")
    body = details.get("body")
    normalized_body = body.casefold() if isinstance(body, str) else ""
    if status_code == 404:
        return (
            f"The configured external accelerator did not expose `{path}` for local model `{remote_model_id}`. "
            "LewLM's bridge-only vision path requires a compatible loopback server that accepts OpenAI-style image "
            f"content blocks on that endpoint.{response_detail}"
        )
    if status_code in {400, 415, 422} and any(
        token in normalized_body
        for token in ("image", "image_url", "vision", "multimodal", "content block", "content blocks")
    ):
        return (
            f"The configured external accelerator reached `{path}`, but local model `{remote_model_id}` rejected "
            "OpenAI-style image content blocks. LewLM's bridge-only vision path requires a compatible server/model "
            f"pair that accepts `image_url` parts on that route.{response_detail}"
        )
    return (
        f"The configured external accelerator could not satisfy `vision` for local model `{remote_model_id}` via "
        f"`{path}`.{response_detail}"
    )


def _audio_probe_failure_reason(
    *,
    capability: CapabilityName,
    error: RuntimeUnavailableError,
    remote_model_id: str,
    path: str,
    response_detail: str,
) -> str:
    details = error.details if isinstance(error.details, dict) else {}
    status_code = details.get("status_code")
    body = details.get("body")
    normalized_body = body.casefold() if isinstance(body, str) else ""
    audio_label = "speech synthesis" if capability == CapabilityName.AUDIO_SPEECH else "audio transcription"
    if status_code == 404:
        return (
            f"The configured external accelerator did not expose `{path}` for local model `{remote_model_id}`. "
            f"LewLM's bridge-only non-Apple {audio_label} path requires a compatible loopback server that implements "
            f"that endpoint.{response_detail}"
        )
    if status_code in {400, 415, 422} and any(
        token in normalized_body
        for token in ("audio", "multipart", "file", "speech", "voice", "wav", "transcription", "tts", "stt")
    ):
        return (
            f"The configured external accelerator reached `{path}`, but local model `{remote_model_id}` rejected "
            f"the bridge-backed {audio_label} probe. LewLM expects a compatible server/model pair on that loopback "
            f"endpoint.{response_detail}"
        )
    return (
        f"The configured external accelerator could not satisfy `{capability.value}` for local model "
        f"`{remote_model_id}` via `{path}`. {_bridge_endpoint_sentence(capability)}{response_detail}"
    )


def _semantic_invalid_payload_reason(*, capability: CapabilityName, remote_model_id: str) -> str:
    path = _SEMANTIC_ENDPOINTS[capability]
    return (
        f"The configured external accelerator returned an invalid `{capability.value}` payload for local model "
        f"`{remote_model_id}` via `{path}`. {_bridge_endpoint_sentence(capability)}"
    )


def _bridge_endpoint_sentence(capability: CapabilityName) -> str:
    path = _SEMANTIC_ENDPOINTS[capability]
    if capability == CapabilityName.AUDIO_TRANSCRIPTION:
        return (
            f"LewLM expects a compatible loopback `{path}` endpoint for this bridge-backed audio-transcription path. "
            "This remains the intentionally narrower non-Apple audio parity boundary."
        )
    if capability == CapabilityName.AUDIO_SPEECH:
        return (
            f"LewLM expects a compatible loopback `{path}` endpoint for this bridge-backed speech path. "
            "This remains the intentionally narrower non-Apple audio parity boundary."
        )
    if capability == CapabilityName.EMBEDDINGS:
        return f"LewLM expects a compatible loopback `{path}` endpoint for adapter-backed embeddings."
    if capability == CapabilityName.RERANK:
        return f"LewLM expects a compatible loopback `{path}` endpoint or equivalent extension for adapter-backed rerank."
    return f"LewLM expects a compatible loopback `{path}` endpoint for this bridge-backed capability."


def _bridge_capability_error_details(
    *,
    runtime_name: str,
    model_id: str,
    remote_model_id: str,
    capability: CapabilityName,
) -> dict[str, Any]:
    path = _SEMANTIC_ENDPOINTS[capability]
    details: dict[str, Any] = {
        "runtime": runtime_name,
        "model_id": model_id,
        "remote_model_id": remote_model_id,
        "capability": capability.value,
        "support_path": "bridge",
        "expected_endpoint": path,
        "fallback_guidance": list(_bridge_capability_guidance(capability)),
    }
    if capability in {CapabilityName.AUDIO_TRANSCRIPTION, CapabilityName.AUDIO_SPEECH}:
        details["bridge_only"] = True
        details["parity_contract"] = "bridge_only_audio"
    return details


def _bridge_capability_guidance(capability: CapabilityName) -> tuple[str, ...]:
    path = _SEMANTIC_ENDPOINTS[capability]
    if capability == CapabilityName.AUDIO_TRANSCRIPTION:
        return (
            f"Expose a compatible local `{path}` endpoint on the loopback server for bridge-backed transcription.",
            "LewLM keeps non-Apple audio parity bridge-backed here and does not bundle the upstream STT server.",
        )
    if capability == CapabilityName.AUDIO_SPEECH:
        return (
            f"Expose a compatible local `{path}` endpoint on the loopback server for bridge-backed speech synthesis.",
            "LewLM keeps non-Apple audio parity bridge-backed here and does not bundle the upstream TTS server.",
        )
    return (f"Expose a compatible local `{path}` endpoint on the loopback server.",)


def _normalize_usage(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, int | float):
            normalized[key] = int(value)
    return normalized


def _normalize_embedding_payload(payload: Any) -> list[list[float]]:
    if isinstance(payload, list):
        if payload and all(isinstance(item, (int, float)) for item in payload):
            return [[float(value) for value in payload]]
        vectors: list[list[float]] = []
        for item in payload:
            vector_payload = item
            if isinstance(item, dict):
                vector_payload = item.get("embedding", item.get("vector", item.get("values", [])))
            if isinstance(vector_payload, list) and all(isinstance(value, (int, float)) for value in vector_payload):
                vectors.append([float(value) for value in vector_payload])
        return vectors
    return []


def _normalize_rerank_payload(payload: Any, request: RerankRequest) -> list[RerankResult]:
    if not isinstance(payload, list):
        return []
    if payload and all(isinstance(item, (int, float)) for item in payload):
        return [
            RerankResult(index=index, relevance_score=float(score), document=request.documents[index])
            for index, score in enumerate(payload)
        ]
    normalized: list[RerankResult] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        item_index = item.get("index", index)
        if not isinstance(item_index, int):
            item_index = index
        document = item.get("document")
        if not isinstance(document, str) and 0 <= item_index < len(request.documents):
            document = request.documents[item_index]
        score = item.get("relevance_score", item.get("score", 0.0))
        if not isinstance(score, int | float):
            score = 0.0
        normalized.append(
            RerankResult(
                index=item_index,
                relevance_score=float(score),
                document=document,
            ),
        )
    normalized.sort(key=lambda item: (-item.relevance_score, item.index))
    return normalized


def _external_capability_modalities(capability: CapabilityName) -> tuple[ModelModality, ...]:
    if capability in {CapabilityName.CHAT, CapabilityName.STREAMING}:
        return (ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL)
    if capability == CapabilityName.VISION:
        return (ModelModality.VISION, ModelModality.MULTIMODAL)
    if capability in {CapabilityName.AUDIO_TRANSCRIPTION, CapabilityName.AUDIO_SPEECH}:
        return (ModelModality.AUDIO,)
    if capability == CapabilityName.EMBEDDINGS:
        return (ModelModality.EMBEDDING,)
    if capability == CapabilityName.RERANK:
        return (ModelModality.RERANK,)
    return ()


def _manifest_supports_external_capability(manifest: ModelManifest, capability: CapabilityName) -> bool:
    required_modalities = _external_capability_modalities(capability)
    if not required_modalities:
        return False
    return any(modality in manifest.modality for modality in required_modalities)


def _message_payload(message: Any) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if isinstance(message.content, str) and message.content:
        parts.append({"type": "text", "text": message.content})
    attachments = getattr(message, "attachments", [])
    if isinstance(attachments, list):
        for attachment in attachments:
            if getattr(attachment, "attachment_type", None) != "image":
                continue
            parts.extend(_image_message_parts(attachment))
    return {
        "role": getattr(message, "role", "user"),
        "content": parts if parts else getattr(message, "content", ""),
    }


def _image_message_parts(attachment: Any) -> list[dict[str, Any]]:
    source_path = getattr(attachment, "source_path", None)
    if not isinstance(source_path, str) or not source_path:
        raise RuntimeUnavailableError(
            "Image attachments on the external accelerator bridge require a readable local `source_path`.",
            details={"attachment_name": getattr(attachment, "name", None)},
        )
    source = Path(source_path).expanduser().resolve(strict=False)
    if not source.exists():
        raise RuntimeUnavailableError(
            "Image attachment path does not exist for the external accelerator bridge.",
            details={"source_path": str(source)},
        )
    candidate_paths = _expanded_image_paths(source)
    if not candidate_paths:
        raise RuntimeUnavailableError(
            "The external accelerator bridge could not find any local image files at the attachment path.",
            details={"source_path": str(source)},
        )
    parts: list[dict[str, Any]] = []
    default_media_type = getattr(attachment, "media_type", None)
    detail = _image_detail_value(attachment)
    for candidate in candidate_paths:
        image_path = Path(candidate)
        if not image_path.exists() or not image_path.is_file():
            continue
        media_type = _image_media_type(image_path, default_media_type=default_media_type)
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        image_url_payload: dict[str, Any] = {"url": f"data:{media_type};base64,{encoded}"}
        if detail is not None:
            image_url_payload["detail"] = detail
        parts.append(
            {
                "type": "image_url",
                "image_url": image_url_payload,
            },
        )
    if not parts:
        raise RuntimeUnavailableError(
            "The external accelerator bridge could not encode any local image files from the attachment path.",
            details={"source_path": str(source)},
        )
    return parts


def _expanded_image_paths(source_path: Path) -> list[Path]:
    if not source_path.exists():
        return []
    if source_path.is_dir():
        return sorted(
            candidate
            for candidate in source_path.iterdir()
            if candidate.is_file() and candidate.suffix.casefold() in _IMAGE_SUFFIXES
        )
    return [source_path]


def _image_media_type(path: Path, *, default_media_type: str | None) -> str:
    if isinstance(default_media_type, str) and default_media_type.startswith("image/"):
        return default_media_type
    return _IMAGE_SUFFIX_MEDIA_TYPES.get(path.suffix.casefold(), "image/png")


def _image_detail_value(attachment: Any) -> str | None:
    detail = getattr(attachment, "detail", None)
    if detail is None:
        metadata = getattr(attachment, "metadata", None)
        if isinstance(metadata, dict):
            detail = metadata.get("detail")
    if not isinstance(detail, str):
        return None
    normalized = detail.casefold()
    return normalized if normalized in {"auto", "low", "high"} else None


def _normalize_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    text_parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).casefold()
        if item_type in {"text", "output_text"}:
            text = item.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
                continue
            if isinstance(text, dict):
                value = text.get("value")
                if isinstance(value, str) and value:
                    text_parts.append(value)
    return "\n".join(text_parts)


def _multipart_form_data(
    *,
    fields: dict[str, Any],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[str, bytes]:
    boundary = f"----lewlm{token_hex(8)}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        if value is None:
            continue
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ],
        )
    for name, (file_name, file_bytes, media_type) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{file_name}"\r\n'
                    f"Content-Type: {media_type}\r\n\r\n"
                ).encode("utf-8"),
                file_bytes,
                b"\r\n",
            ],
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return boundary, b"".join(chunks)


def _response_media_type(response: Any) -> str:
    headers = getattr(response, "headers", None)
    if headers is not None:
        get_content_type = getattr(headers, "get_content_type", None)
        if callable(get_content_type):
            return str(get_content_type())
        content_type = headers.get("Content-Type")
        if isinstance(content_type, str) and content_type:
            return content_type.split(";", 1)[0].strip()
    return "application/octet-stream"


def _probe_audio_bytes() -> bytes:
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x00" * 160)
        return buffer.getvalue()


def _audio_media_type_for_bytes(audio_bytes: bytes) -> str:
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return "audio/wav"
    if audio_bytes.startswith(b"ID3"):
        return "audio/mpeg"
    if audio_bytes.startswith(b"fLaC"):
        return "audio/flac"
    if audio_bytes.startswith(b"OggS"):
        return "audio/ogg"
    return "application/octet-stream"


def _normalize_audio_transcription_response(
    payload: dict[str, Any],
    request: AudioTranscriptionRequest,
) -> AudioTranscriptionResponse:
    segments_payload = payload.get("segments", [])
    segments: list[AudioTranscriptionSegment] = []
    if isinstance(segments_payload, list):
        for item in segments_payload:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text:
                continue
            start_seconds = item.get("start", item.get("start_seconds"))
            end_seconds = item.get("end", item.get("end_seconds"))
            segments.append(
                AudioTranscriptionSegment(
                    start_seconds=float(start_seconds) if isinstance(start_seconds, (int, float)) else None,
                    end_seconds=float(end_seconds) if isinstance(end_seconds, (int, float)) else None,
                    text=text,
                ),
            )
    return AudioTranscriptionResponse(
        model_id=request.model_id,
        text=str(payload.get("text", "")),
        language=payload.get("language") if isinstance(payload.get("language"), str) else request.language,
        duration_seconds=(
            float(payload.get("duration"))
            if isinstance(payload.get("duration"), (int, float))
            else _duration_seconds_from_audio_bytes(request.audio_bytes, media_type=_audio_media_type_for_bytes(request.audio_bytes))
        ),
        segments=segments,
    )


def _duration_seconds_from_audio_bytes(audio_bytes: bytes, *, media_type: str) -> float | None:
    if media_type != "audio/wav" or not audio_bytes:
        return None
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as handle:
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
    except wave.Error:
        return None
    if frame_rate <= 0:
        return None
    return round(frame_count / frame_rate, 4)
