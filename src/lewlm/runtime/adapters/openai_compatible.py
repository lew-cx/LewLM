"""Local-only adapter runtime for OpenAI-compatible accelerator servers."""

from __future__ import annotations

import asyncio
import base64
from contextlib import aclosing
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from secrets import token_hex
from datetime import datetime
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import wave

from lewlm.config.settings import LewLMSettings
from lewlm.config.endpoints import ExternalEndpoint, server_root
from lewlm.core.contracts import (
    utc_now,
    BridgeProfile,
    RuntimeProvider,
    AudioSpeechFormat,
    AudioSpeechFormatSupport,
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
    RuntimeStreamEvent,
    RuntimeToolCallDelta,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelToolCallingSupport,
    PerformanceFeatureOwnership,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RuntimeAffinity,
    RuntimeCandidateReport,
    RuntimeReadinessState,
    build_portable_performance_core_evidence,
    normalize_performance_feature_ownership,
    normalize_runtime_performance_feature_report,
    runtime_performance_feature_report,
)
from lewlm.core.errors import RuntimeUnavailableError
from lewlm.runtime.base import ManagedTextRuntime
from lewlm.runtime.adapters.http_transport import STREAM_OPENED, AsyncBridgeTransport
from lewlm.runtime.sampling import attach_sampling_report, resolve_sampling_controls
from lewlm.structured_output import StructuredOutputRequest, StructuredOutputRuntimeStatus

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# Discovery and explicit capability probes use synchronous I/O. Apply the
# same endpoint boundary as the async generation transport, including when
# the process has ambient proxy settings.
urlopen = build_opener(ProxyHandler({}), _RejectRedirects()).open
# How long a failed `/v1/models` read stays cached before it is retried. Long
# enough that a down server does not cost a connection attempt per lookup,
# short enough that a server started after LewLM becomes usable on its own.
_DISCOVERY_FAILURE_RETRY_SECONDS = 5.0
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
_PROFILE_FEATURES["exllamav3_tabby"] = {
    "continuous_batching": (
        PerformanceFeatureOwnership.BACKEND_NATIVE,
        "TabbyAPI schedules batched ExLlamaV3 generation inside the server; LewLM keeps an admission cap and forwards requests.",
    ),
    "prefix_cache": (
        PerformanceFeatureOwnership.PARTIAL,
        "ExLlamaV3's paged cache may reuse prefixes upstream; hit counters are not exposed through the bridge.",
    ),
    "paged_kv_cache": (
        PerformanceFeatureOwnership.BACKEND_NATIVE,
        "Paged KV cache residency is owned by ExLlamaV3 inside TabbyAPI.",
    ),
    "kv_cache_quantization": (
        PerformanceFeatureOwnership.BACKEND_NATIVE,
        "TabbyAPI's `cache_mode` (FP16, Q8/Q6/Q4, or k_bits,v_bits) is configured server-side; LewLM cannot tune it per request.",
    ),
    "prefill_optimization": (
        PerformanceFeatureOwnership.BACKEND_NATIVE,
        "Chunked prefill (`chunk_size`) is a server setting inside TabbyAPI.",
    ),
    "speculative_decoding": (
        PerformanceFeatureOwnership.UNSUPPORTED,
        "TabbyAPI draft models and n-gram drafting stay outside the adapter contract; configure them in TabbyAPI if wanted.",
    ),
    "constrained_decoding": (
        PerformanceFeatureOwnership.PARTIAL,
        "json_schema is forwarded as a native response_format; whether TabbyAPI enforces it for the loaded model is validated after generation, not assumed.",
    ),
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

    #: Streams begin with `RuntimeStreamEvent(opened=True)` once the engine
    #: has accepted the request (see `STREAM_OPENED`).
    announces_stream_open = True

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

    def __init__(self, *, settings: LewLMSettings, endpoint: ExternalEndpoint | None = None) -> None:
        super().__init__()
        if endpoint is None and settings.external_endpoints is not None:
            raise ValueError("An explicit endpoint is required when using external_endpoints.")
        self.endpoint = endpoint or settings.resolved_external_endpoints()[0]
        if endpoint is not None:
            self.name = f"local_external_adapter:{endpoint.endpoint_id}"
            self._settings = settings.model_copy(update={
                "external_accelerator_enabled": endpoint.enabled,
                "external_accelerator_profile": endpoint.profile,
                "external_accelerator_base_url": server_root(endpoint.base_url),
                "external_accelerator_timeout_seconds": endpoint.read_timeout_seconds,
            })
        else:
            self._settings = settings
        self._discovered_model_ids: tuple[str, ...] | None = None
        self._discovered_model_records: tuple[dict[str, Any], ...] | None = None
        self._discovery_error: str | None = None
        self._discovery_failed_at: float | None = None
        # When the last successful read happened; drives the TTL. A failed
        # refresh after a success keeps the last-known list and marks it stale
        # rather than pretending the upstream models were deleted.
        self._discovered_at: float | None = None
        # Wall-clock time of the first successful read in this process: the
        # "engine ready" phase as LewLM observed it.
        self._first_advertised_at: datetime | None = None
        self._inventory_stale = False
        self._force_refresh = False
        self._model_capability_support_cache: dict[tuple[str, CapabilityName], bool] = {}
        self._model_capability_reason_cache: dict[tuple[str, CapabilityName], str | None] = {}
        self._transport = AsyncBridgeTransport(
            endpoint=self.endpoint,
            runtime_name=self.name,
            # A refused request is the freshest evidence there is: health and
            # availability report the endpoint down at once instead of waiting
            # for the inventory TTL, and the next successful read restores it.
            on_unreachable=self._record_discovery_failure,
        )

    async def aclose(self) -> None:
        """Close the reusable bridge connection pool."""

        await self._transport.aclose()

    @property
    def cache_namespace(self) -> str:
        return self.endpoint.cache_namespace

    def endpoint_snapshot(self) -> dict[str, Any]:
        """Cached evidence only. Health must not make generation/discovery calls."""
        return {
            "endpoint_id": self.endpoint.endpoint_id,
            "profile": self.endpoint.profile,
            "enabled": self.endpoint.enabled,
            "inventory_state": self.inventory_state,
            "inventory_error": self._discovery_error,
            "inventory_age_seconds": (
                round(monotonic() - self._discovered_at, 3) if self._discovered_at is not None else None
            ),
            "inventory_ttl_seconds": self._inventory_ttl_seconds(),
            "advertised_model_ids": list(self._discovered_model_ids or ()),
            "capability_evidence": [
                {"upstream_model_id": model_id, "capability": capability.value,
                 "state": "generate_passed" if supported else "probe_failed"}
                for (model_id, capability), supported in self._model_capability_support_cache.items()
            ],
            "first_advertised_at": self._first_advertised_at.isoformat() if self._first_advertised_at is not None else None,
            "upstream_residency": "unknown",
            "upstream_cancellation": "unknown",
        }

    @property
    def inventory_state(self) -> str:
        """``unknown`` (never read), ``advertised``, ``stale`` (last read failed,
        earlier list retained), or ``failed`` (never succeeded)."""

        if self._discovered_model_ids is None:
            return "unknown"
        if self._discovery_error is not None:
            return "stale" if self._inventory_stale else "failed"
        return "advertised"

    def advertised_model_records(self, *, refresh: bool = False) -> tuple[dict[str, Any], ...]:
        """The upstream ``/v1/models`` records, honoring the TTL.

        ``refresh=True`` is the explicit-refresh path (``lewlm scan``): it
        drops the cached list first so the read is live. Raises
        :class:`RuntimeUnavailableError` when the endpoint cannot be read; the
        last-known records stay cached and :attr:`inventory_state` says
        ``stale``.
        """

        if refresh:
            self._force_refresh = True
        self._available_remote_models()
        return self._discovered_model_records or ()

    def _inventory_ttl_seconds(self) -> float:
        return float(getattr(self._settings, "external_inventory_ttl_seconds", 0.0) or 0.0)

    def _discovery_cache_expired(self) -> bool:
        """Whether a *successful* list has outlived the configured TTL.

        Failure retry timing is `_discovery_cache_is_stale_failure`'s job, and
        a list with no timestamp (set directly by a test) never expires.
        """

        if self._discovery_error is not None or self._discovered_at is None:
            return False
        ttl = self._inventory_ttl_seconds()
        if ttl <= 0:
            return False
        age = monotonic() - self._discovered_at
        return age < 0 or age >= ttl

    def cached_supported_capabilities(self) -> tuple[CapabilityName, ...]:
        return tuple(sorted({capability for (_, capability), supported in
                             self._model_capability_support_cache.items() if supported}, key=lambda c: c.value))

    def bridge_profile(self) -> BridgeProfile:
        profile = self.endpoint.profile
        provider = (RuntimeProvider.VLLM if profile in {"vllm_local", "vllm_mlx"} else
                    RuntimeProvider.SGLANG if profile == "sglang_local" else
                    RuntimeProvider.EXLLAMAV3 if profile == "exllamav3_tabby" else
                    RuntimeProvider.OLLAMA if profile == "ollama_local" else
                    RuntimeProvider.LLAMACPP_SERVER if profile == "llamacpp_server" else
                    RuntimeProvider.OPENAI_COMPATIBLE)
        return BridgeProfile(profile_id=profile, endpoint_id=self.endpoint.endpoint_id, provider=provider,
                             base_url=server_root(self.endpoint.base_url),
                             supported_capabilities=list(self.cached_supported_capabilities()),
                             notes=["Configured bridge; upstream behavior requires model-specific validation."])

    async def lightweight_health_check(self) -> dict[str, Any]:
        # Passive: no discovery, no capability probes. The performance-feature
        # snapshot is a static profile map (ownership + `active=False` until
        # observed), so attaching it costs no network call and keeps the
        # runtime-stats aggregate honest about which ownership modes exist.
        result = await super().lightweight_health_check()
        performance_features = self.performance_feature_snapshot()
        result["performance_features"] = performance_features
        result["performance_core_evidence"] = [
            record.model_dump(mode="json")
            for record in build_portable_performance_core_evidence(
                performance_features=performance_features,
                runtime_names=[self.name],
            )
        ]
        result["endpoint"] = self.endpoint_snapshot()
        return result

    def continuous_batching_ownership(self, capability: CapabilityName) -> str:
        """Who batches requests for this endpoint: the engine, or nobody LewLM can see.

        ``backend_native`` when the profile's feature map says the server
        schedules its own batches (vLLM, SGLang, TabbyAPI, oMLX, ...). LewLM
        never opens a microbatch window in front of such a server: it keeps
        the bounded admission cap and per-request cancellation, and hands each
        request straight to the shared transport so two concurrent requests
        reach the engine concurrently. ``supports_continuous_batching`` stays
        False for the same reason: there is no LewLM-side batch API here.
        """

        if capability not in {CapabilityName.CHAT, CapabilityName.STREAMING}:
            return "unsupported"
        ownership, _ = _profile_feature_map(self._settings)["continuous_batching"]
        return "backend_native" if ownership is PerformanceFeatureOwnership.BACKEND_NATIVE else "unsupported"

    def supports_manifest(self, manifest: ModelManifest) -> bool:
        binding = manifest.metadata.get("external_endpoint_id")
        if binding is not None and binding != self.endpoint.endpoint_id:
            return False
        if binding is not None:
            # The endpoint owns the artifact; its weight format (often unknown
            # from `/v1/models`) is not a reason to refuse serving it. Modality
            # still has to be one the bridge contract can carry.
            if not any(modality in self.supported_modalities for modality in manifest.modality):
                return False
        elif not super().supports_manifest(manifest):
            return False
        if not self.is_available():
            return False
        try:
            return self._resolve_remote_model_id(manifest) is not None
        except RuntimeUnavailableError:
            return False

    def candidate_report(self, manifest: ModelManifest | None = None) -> RuntimeCandidateReport:
        report = super().candidate_report(manifest)
        report.metadata["endpoint"] = self.endpoint_snapshot()
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
        if self.inventory_state == "failed":
            # Nothing has ever been read from this endpoint and the last attempt
            # failed: that is an unavailable runtime, not a missing model.
            return report.model_copy(
                update={
                    "available": False,
                    "readiness_state": RuntimeReadinessState.RUNTIME_UNAVAILABLE,
                    "supports_manifest": False,
                    "availability_reason": self._discovery_error,
                },
            )
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
                active=False,
                reason=reason,
                metrics={
                    "adapter_profile": self._settings.external_accelerator_profile,
                    "contract": "openai_compatible_local",
                    "endpoint_id": self.endpoint.endpoint_id,
                    "evidence_state": "unverified",
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

    def tool_calling_support(self) -> ModelToolCallingSupport:
        # `_chat_payload` forwards declared tools as OpenAI `tools` on every
        # profile and strips the prompt's tool scaffolding when it does.
        return ModelToolCallingSupport(
            runtime_name=self.name,
            support="native",
            parallel=None,
            reason=(
                f"`{self.name}` forwards the declared tools to the `{self.endpoint.profile}` server, which emits "
                "structured calls; LewLM validates each against the tool's input_schema. Whether one reply can "
                "carry several calls is the server's choice."
            ),
        )

    def structured_output_runtime_status(
        self,
        contract: StructuredOutputRequest | None,
    ) -> StructuredOutputRuntimeStatus | None:
        if contract is None or contract.type == "text":
            return None
        # This is the same verdict `_chat_payload` records, so the capability
        # prediction and the generation outcome cannot drift apart.
        if contract.type == "json_schema" or self._forwards_grammar_natively():
            return self._upstream_native_status(contract.type)
        _, reason = _profile_feature_map(self._settings)["constrained_decoding"]
        return StructuredOutputRuntimeStatus(
            runtime=self.name,
            mode=contract.type,
            enforcement="prompt_guided",
            decoder_enforced=False,
            enforcement_evidence="prompt",
            fallback_used=True,
            fallback_reason=f"{reason} This remains a loopback adapter boundary rather than packaged decode-time parity.",
        )

    def _forwards_grammar_natively(self) -> bool:
        return self.endpoint.profile in {"vllm_local", "sglang_local", "vllm_mlx"}

    def _upstream_native_status(self, mode: str) -> StructuredOutputRuntimeStatus:
        """The contract is forwarded to the server as-is; the server's decoder enforces it.

        `decode_time` names the path (no prompt scaffolding), `decoder_enforced`
        stays False because LewLM never observes the upstream decoder, and the
        output is validated after generation. Consumers that need a guarantee
        read `validation`, not this flag.
        """

        return StructuredOutputRuntimeStatus(
            runtime=self.name,
            mode=mode,  # type: ignore[arg-type]
            enforcement="decode_time",
            decoder_enforced=False,
            enforcement_evidence="upstream_native",
            fallback_used=False,
        )

    # Readiness, health, and model listing ask the three methods below on every
    # refresh. They never send a request: doing so made `GET /v1/health` issue
    # a generation per advertised model, which on Ollama loads every model
    # before health can answer. A capability the endpoint's model list
    # advertises is a routing candidate until a request, or an explicit
    # `probe_manifest_capability`, observes otherwise; that observation is
    # cached and reported here from then on, with its reason.

    def supports_capability(self, capability: CapabilityName) -> bool:
        if capability in {CapabilityName.CHAT, CapabilityName.STREAMING}:
            return super().supports_capability(capability)
        if capability not in self.supported_capabilities or not self.is_available():
            return False
        observed = {
            model_id: supported
            for (model_id, observed_capability), supported in self._model_capability_support_cache.items()
            if observed_capability == capability
        }
        if any(observed.values()):
            return True
        remote_model_ids = list(self._available_remote_models())
        if not remote_model_ids:
            return False
        # Every advertised model was exercised and refused: that is a refusal,
        # not an unknown.
        return not all(model_id in observed for model_id in remote_model_ids)

    def probe_manifest_capability(
        self,
        manifest: ModelManifest,
        capability: CapabilityName,
    ) -> tuple[bool, str | None]:
        """Exercise `capability` against the endpoint for `manifest`, once, and remember the outcome.

        The explicit operation: it sends one request (a one-image chat, an
        embedding, a rerank, a transcription, or a synthesis) and may make the
        engine load the model. Readiness paths never call it.
        """

        if not self.supports_manifest(manifest):
            return False, self.manifest_capability_reason(manifest, capability)
        if not _manifest_supports_external_capability(manifest, capability):
            return False, self.manifest_capability_reason(manifest, capability)
        if capability in {CapabilityName.CHAT, CapabilityName.STREAMING}:
            return True, None
        remote_model_id = self._resolve_remote_model_id(manifest)
        if remote_model_id is None:
            return False, self.manifest_capability_reason(manifest, capability)
        return self._probe_remote_model_capability(remote_model_id, capability)

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
        observed = self._model_capability_support_cache.get((remote_model_id, capability))
        return True if observed is None else observed

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
        if (remote_model_id, capability) not in self._model_capability_support_cache:
            return None
        return self._model_capability_reason_cache.get((remote_model_id, capability))

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
        # Nothing to free here: the external server owns the weights. Only
        # LewLM's lease bookkeeping changes, and the lifecycle result says so.
        return None

    def lifecycle_note(self, operation: str) -> str | None:
        if operation == "unload":
            return (
                "This released LewLM's bridge lease only; the external server owns model residency and LewLM "
                "did not free its memory (upstream residency: unknown)."
            )
        if operation == "warm":
            return "Warm sent a one-token probe to the external server; residency and eviction remain the server's."
        return None

    def lifecycle_backend_operation_performed(self, operation: str) -> bool:
        return operation != "unload"

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
        tool_calls: dict[int, dict[str, str | None]] = {}
        async for event in self._stream_generate_events(request):
            if event.content:
                yield event.content
            if event.tool_call is not None:
                _accumulate_tool_call_delta(tool_calls, event.tool_call)
        tool_text = _accumulated_tool_calls_as_text(tool_calls)
        if tool_text:
            yield tool_text

    async def _stream_generate_events(self, request: GenerateRequest):
        manifest = self._loaded_manifests[request.model_id]
        self._record_structured_output_runtime(request)
        remote_model_id = self._require_remote_model_id(manifest)
        payload = self._chat_payload(remote_model_id=remote_model_id, request=request, stream=True)
        async for event in self._stream_chat_completion(payload):
            yield event

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
        payload = await self._request_json_async(
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
        payload = await self._request_json_async(
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
        payload = await self._request_multipart_json_async(
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
        audio_bytes, media_type = await self._request_bytes_async(
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
        if self.endpoint.api_key_env is not None and not os.environ.get(self.endpoint.api_key_env):
            return False, f"Set {self.endpoint.api_key_env} for external endpoint `{self.endpoint.endpoint_id}`."
        return True, None

    async def _generate_with_manifest(self, manifest: ModelManifest, request: GenerateRequest) -> GenerateResponse:
        remote_model_id = self._require_remote_model_id(manifest)
        payload = self._chat_payload(remote_model_id=remote_model_id, request=request, stream=False)
        response_payload = await self._request_json_async("POST", "/v1/chat/completions", payload)
        choices = response_payload.get("choices")
        if (not isinstance(choices, list) or not choices
                or not isinstance(choices[0], dict)
                or not isinstance(choices[0].get("message"), dict)):
            raise RuntimeUnavailableError(
                "External accelerator returned an invalid chat completion payload.",
                details={"runtime": self.name, "response_keys": sorted(response_payload)},
            )
        message = choices[0].get("message", {})
        output_text = _normalize_content_text(message.get("content"))
        tool_calls_text = _native_tool_calls_as_text(message.get("tool_calls"))
        if not output_text:
            output_text = tool_calls_text
        return GenerateResponse(
            model_id=request.model_id,
            output_text=output_text,
            finish_reason=str(choices[0].get("finish_reason", "stop")),
            usage=_normalize_usage(response_payload.get("usage")),
            native_tool_calls=json.loads(tool_calls_text)["tool_calls"] if tool_calls_text else [],
        )

    def speech_formats(self, manifest: ModelManifest) -> AudioSpeechFormatSupport:
        """The upstream decides: `format` is forwarded as `response_format`.

        WAV is verified once the speech probe has returned audio for this
        model (read from the probe cache; never probes here). Nothing else is
        probed and the list is open, so a format not listed is forwarded, not
        refused.
        """

        remote_model_id = self._resolve_remote_model_id(manifest)
        verified = bool(
            remote_model_id is not None
            and self._model_capability_support_cache.get((remote_model_id, CapabilityName.AUDIO_SPEECH), False),
        )
        return AudioSpeechFormatSupport(
            formats=[AudioSpeechFormat(format="wav", media_type="audio/wav", verified=verified)],
            exhaustive=False,
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

    def invalidate_discovery_cache(self) -> None:
        """Forget the advertised model list so the next lookup re-reads `/v1/models`.

        The external server owns its own inventory, so a model pulled or removed
        after this process started is invisible until the cache is dropped.
        """

        self._discovered_model_ids = None
        self._discovered_model_records = None
        self._discovery_failed_at = None
        self._discovery_error = None
        self._discovered_at = None
        self._inventory_stale = False
        self._force_refresh = False

    def _available_remote_models(self) -> tuple[str, ...]:
        if (
            self._discovered_model_ids is not None
            and not self._force_refresh
            and not self._discovery_cache_is_stale_failure()
            and not self._discovery_cache_expired()
        ):
            if self._discovery_error is not None and self._inventory_stale:
                # The last read failed and the retry window has not passed. The
                # stale list stays for the registry (an unreachable engine is not
                # evidence its models were deleted), but it is not a routing
                # candidate: routing must see the outage before submitting, so
                # an explicit fallback alias can apply and the caller gets a 503
                # naming the endpoint rather than a transport failure later. A
                # never-successful read keeps returning its empty list instead.
                raise RuntimeUnavailableError(
                    self._discovery_error,
                    details={"runtime": self.name, "endpoint_id": self.endpoint.endpoint_id, "inventory_state": self.inventory_state},
                )
            return self._discovered_model_ids
        self._force_refresh = False
        try:
            payload = self._request_json("GET", "/v1/models", None)
        except RuntimeUnavailableError as exc:
            # `_request` already recorded transport failures; HTTP errors and
            # malformed bodies land here without a record, so record them too.
            if self._discovery_error is None or self._discovery_failed_at is None:
                self._record_discovery_failure(str(exc))
            raise
        data = payload.get("data")
        if not isinstance(data, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"].strip()
            for item in data
        ):
            reason = "External accelerator returned an invalid model inventory."
            self._record_discovery_failure(reason)
            raise RuntimeUnavailableError(
                reason,
                details={"runtime": self.name, "error_kind": "malformed_response"},
            )
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
        self._discovery_failed_at = None
        self._discovery_error = None
        self._discovered_at = monotonic()
        self._inventory_stale = False
        if self._first_advertised_at is None:
            self._first_advertised_at = utc_now()
        return self._discovered_model_ids

    def _discovery_cache_is_stale_failure(self) -> bool:
        """Whether the cached empty list came from a transport failure worth retrying.

        A successful discovery is cached for the life of the process; a failed one
        is not, or a server that starts after LewLM would stay unusable forever.
        """

        if self._discovery_failed_at is None:
            return False
        elapsed = monotonic() - self._discovery_failed_at
        # Defensive for tests and rare platform clock resets: a negative age is
        # not a sound basis for keeping a failed network result indefinitely.
        return elapsed < 0 or elapsed >= _DISCOVERY_FAILURE_RETRY_SECONDS

    def _record_discovery_failure(self, reason: str) -> None:
        # An unreachable endpoint is not evidence that its models were deleted:
        # a list that was read successfully before is kept and marked stale.
        # With no earlier success, `()` rather than `None` so callers that report
        # advertised ids while handling the error see an empty list instead of
        # re-entering discovery.
        if self._discovered_model_ids and self._discovery_error is None:
            self._inventory_stale = True
        elif not self._discovered_model_ids:
            self._discovered_model_ids = ()
            self._discovered_model_records = ()
            self._inventory_stale = False
        self._discovery_failed_at = monotonic()
        self._discovery_error = reason

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
        sampling, report = resolve_sampling_controls(
            request.sampling,
            runtime_name=self.name,
            family=(
                "external_bridge_sglang"
                if self.endpoint.profile == "sglang_local"
                else "external_bridge_tabby"
                if self.endpoint.profile == "exllamav3_tabby"
                else "external_bridge_ollama"
                if self.endpoint.profile == "ollama_local"
                else "external_bridge_extended"
                if self.endpoint.profile in {"vllm_local", "vllm_mlx"}
                else "external_bridge"
            ),
        )
        attach_sampling_report(request.metadata, report)
        tools = _bridge_tools(request.metadata.get("bridge_tools"))
        structured_output = _bridge_structured_output(request.structured_output)
        messages = [_message_payload(message) for message in request.messages]
        if tools:
            messages = [message for message in messages if not _is_prompt_tool_scaffolding(message)]
        if structured_output is not None:
            messages = [message for message in messages if not _is_prompt_structured_output_scaffolding(message)]
        payload: dict[str, Any] = {
            "model": remote_model_id,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
            **sampling,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if "seed" in sampling and self.endpoint.profile == "vllm_local":
            # vLLM's prefix cache makes a seeded reply depend on cache state: a
            # hit re-evaluates only the prompt's tail, and the shifted logits can
            # change what the seed samples. A fresh salt forces the whole prompt
            # to be evaluated, so the reported determinism holds; unseeded
            # requests keep full prefix reuse.
            payload["cache_salt"] = token_hex(32)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = request.metadata.get("tool_choice", "auto")
        if structured_output is not None:
            payload["response_format"] = structured_output
            request.metadata["structured_output_bridge"] = {
                "forwarded": True,
                "type": request.structured_output.type if request.structured_output is not None else "text",
            }
            request.metadata["structured_output_runtime"] = self._upstream_native_status(
                request.structured_output.type if request.structured_output is not None else "text",
            ).model_dump(mode="json")
        elif (
            request.structured_output is not None
            and request.structured_output.type == "grammar"
            and self.endpoint.profile in {"vllm_local", "sglang_local", "vllm_mlx"}
        ):
            payload["structured_outputs"] = {"grammar": request.structured_output.grammar}
            messages = [message for message in messages if not _is_prompt_structured_output_scaffolding(message)]
            payload["messages"] = messages
            request.metadata["structured_output_bridge"] = {
                "forwarded": True,
                "type": "grammar",
            }
            request.metadata["structured_output_runtime"] = self._upstream_native_status("grammar").model_dump(mode="json")
        return payload

    async def _stream_chat_completion(self, payload: dict[str, Any]):
        saw_done = False
        async with aclosing(self._transport.stream_sse(
            "POST",
            "/v1/chat/completions",
            payload=payload,
            announce_open=True,
        )) as stream:
            async for message in stream:
                if message is STREAM_OPENED:
                    yield RuntimeStreamEvent(opened=True)
                    continue
                data = message.data.strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeUnavailableError(
                        "External accelerator returned malformed streaming JSON.",
                        details={"runtime": self.name, "error_kind": "malformed_stream"},
                    ) from exc
                if not isinstance(event, dict):
                    raise RuntimeUnavailableError(
                        "External accelerator returned an unexpected streaming payload.",
                        details={"runtime": self.name, "payload_type": type(event).__name__, "error_kind": "malformed_stream"},
                    )
                if "error" in event or message.event == "error":
                    raise RuntimeUnavailableError(
                        "External accelerator reported a streaming error.",
                        details={"runtime": self.name, "error_kind": "upstream_error"},
                    )
                usage = _normalize_usage(event.get("usage"))
                if usage:
                    yield RuntimeStreamEvent(usage=usage)
                choices = event.get("choices")
                if choices == []:
                    # OpenAI-style usage-only terminal events intentionally carry
                    # an empty choices array.
                    continue
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise RuntimeUnavailableError(
                        "External accelerator returned invalid streaming choices.",
                        details={"runtime": self.name, "error_kind": "malformed_stream"},
                    )
                delta = choices[0].get("delta", {})
                if not isinstance(delta, dict):
                    raise RuntimeUnavailableError(
                        "External accelerator returned an invalid streaming delta.",
                        details={"runtime": self.name, "error_kind": "malformed_stream"},
                    )
                content = _normalize_content_text(delta.get("content"))
                reasoning = _normalize_content_text(delta.get("reasoning_content", delta.get("reasoning")))
                if content or reasoning:
                    yield RuntimeStreamEvent(content=content or None, reasoning=reasoning or None)
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        normalized = _runtime_tool_call_delta(tool_call)
                        if normalized is not None:
                            yield RuntimeStreamEvent(tool_call=normalized)
                finish_reason = choices[0].get("finish_reason")
                if isinstance(finish_reason, str) and finish_reason:
                    yield RuntimeStreamEvent(finish_reason=finish_reason)
        if not saw_done:
            raise RuntimeUnavailableError(
                "External accelerator event stream ended before the `[DONE]` marker.",
                details={"runtime": self.name, "path": "/v1/chat/completions", "error_kind": "premature_eof"},
            )

    async def _request_json_async(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # Preserve instance-level test/probe injection while production calls
        # use the shared async pool.
        if "_request_json" in self.__dict__:
            return await asyncio.to_thread(self._request_json, method, path, payload)
        return await self._transport.request_json(method, path, payload=payload)

    async def _request_multipart_json_async(
        self,
        method: str,
        path: str,
        fields: dict[str, Any],
        files: dict[str, tuple[str, bytes, str]],
    ) -> dict[str, Any]:
        if "_request_multipart_json" in self.__dict__:
            return await asyncio.to_thread(self._request_multipart_json, method, path, fields, files)
        filtered_fields = {key: value for key, value in fields.items() if value is not None}
        return await self._transport.request_json(method, path, data=filtered_fields, files=files)

    async def _request_bytes_async(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[bytes, str]:
        if "_request_bytes" in self.__dict__:
            return await asyncio.to_thread(self._request_bytes, method, path, payload)
        return await self._transport.request_bytes(method, path, payload=payload, accept="audio/*,application/octet-stream")

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
        base_url = server_root(self.endpoint.base_url)
        if payload is not None and body is not None:
            raise ValueError("payload and body cannot both be provided to the external accelerator request helper.")
        request_headers = {"Accept": accept, **(headers or {})}
        if self.endpoint.api_key_env is not None:
            api_key = os.environ.get(self.endpoint.api_key_env)
            if api_key:
                request_headers["Authorization"] = f"Bearer {api_key}"
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
            exc.close()
            raise RuntimeUnavailableError(
                f"External accelerator request failed with HTTP {exc.code}.",
                details={"runtime": self.name, "path": path, "status_code": exc.code},
            ) from exc
        except URLError as exc:
            self._record_discovery_failure(str(exc.reason))
            if isinstance(exc.reason, ConnectionRefusedError):
                raise RuntimeUnavailableError(
                    f"The configured external accelerator endpoint `{request_url}` refused the connection.",
                    details={"runtime": self.name, "path": path, "reason": str(exc.reason)},
                ) from exc
            if isinstance(exc.reason, TimeoutError):
                raise RuntimeUnavailableError(
                    self._timeout_message(request_url),
                    details={"runtime": self.name, "path": path, "reason": str(exc.reason)},
                ) from exc
            raise RuntimeUnavailableError(
                "Could not reach the configured external accelerator endpoint.",
                details={"runtime": self.name, "path": path, "reason": str(exc.reason)},
            ) from exc
        except TimeoutError as exc:
            # A read that stalls past the deadline surfaces as a bare socket
            # timeout rather than a URLError, so it needs its own arm or it
            # escapes the adapter as an untyped traceback.
            self._record_discovery_failure(str(exc) or "timed out")
            raise RuntimeUnavailableError(
                self._timeout_message(request_url),
                details={"runtime": self.name, "path": path, "reason": str(exc) or "timed out"},
            ) from exc
        except OSError as exc:
            # Anything else the transport can raise (a reset connection, a DNS
            # failure on a hostname alias) belongs to the adapter contract too.
            self._record_discovery_failure(str(exc))
            raise RuntimeUnavailableError(
                "Could not reach the configured external accelerator endpoint.",
                details={"runtime": self.name, "path": path, "reason": str(exc)},
            ) from exc

    def _timeout_message(self, request_url: str) -> str:
        return (
            f"The configured external accelerator endpoint `{request_url}` did not respond within "
            f"{self._settings.external_accelerator_timeout_seconds}s. Raise "
            "LEWLM_EXTERNAL_ACCELERATOR_TIMEOUT_SECONDS when the server loads models on first request."
        )

    def _redact_backend_secret(self, value: str) -> str:
        if self.endpoint.api_key_env is None:
            return value
        secret = os.environ.get(self.endpoint.api_key_env)
        return value.replace(secret, "[REDACTED]") if secret else value


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
    """Flatten an OpenAI-style usage object to integer counters.

    ``prompt_tokens_details.cached_tokens`` is the one nested field kept, as
    ``cached_tokens``: it is the only prefix-cache hit counter a server can
    expose through this contract (SGLang ``--enable-cache-report``, vLLM
    ``--enable-prompt-tokens-details``). Absent means unknown, never zero.
    """

    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, int | float) and not isinstance(value, bool):
            normalized[key] = int(value)
    details = payload.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int | float) and not isinstance(cached, bool):
            normalized["cached_tokens"] = int(cached)
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
    payload: dict[str, Any] = {
        "role": getattr(message, "role", "user"),
        "content": parts if parts else getattr(message, "content", ""),
    }
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = tool_calls
        if not parts:
            # OpenAI's shape for a turn that only called tools.
            payload["content"] = None
    return payload


def _bridge_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    tools: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        function: dict[str, Any] = {
            "name": name,
            "parameters": item.get("input_schema") if isinstance(item.get("input_schema"), dict) else {},
        }
        if isinstance(item.get("description"), str):
            function["description"] = item["description"]
        tools.append({"type": "function", "function": function})
    return tools


def _bridge_structured_output(contract: StructuredOutputRequest | None) -> dict[str, Any] | None:
    if contract is None or contract.type != "json_schema":
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": contract.name or "lewlm_response",
            "schema": contract.schema_payload,
            "strict": contract.strict,
        },
    }


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _is_prompt_tool_scaffolding(message: dict[str, Any]) -> bool:
    if message.get("role") != "system":
        return False
    text = _message_text(message)
    return text.startswith((
        "Declared tools:\n",
        "Local MCP tool listings:\n",
        "To call a tool, reply with a JSON object in one of these shapes and nothing else:",
    ))


def _is_prompt_structured_output_scaffolding(message: dict[str, Any]) -> bool:
    return message.get("role") == "system" and _message_text(message).startswith("Structured output contract:\n")


def _native_tool_calls_as_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    calls: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        call: dict[str, Any] = {"name": function["name"], "arguments": arguments}
        if isinstance(item.get("id"), str) and item["id"]:
            call["id"] = item["id"]
        calls.append(call)
    if not calls:
        return ""
    return json.dumps({"tool_calls": calls}, separators=(",", ":"))


def _runtime_tool_call_delta(value: Any) -> RuntimeToolCallDelta | None:
    if not isinstance(value, dict):
        return None
    function = value.get("function")
    if not isinstance(function, dict):
        function = {}
    index = value.get("index", 0)
    if not isinstance(index, int) or isinstance(index, bool):
        index = 0
    call_id = value.get("id")
    name = function.get("name")
    arguments = function.get("arguments")
    return RuntimeToolCallDelta(
        index=index,
        call_id=call_id if isinstance(call_id, str) else None,
        name=name if isinstance(name, str) else None,
        arguments=arguments if isinstance(arguments, str) else None,
    )


def _accumulate_tool_call_delta(
    accumulator: dict[int, dict[str, str | None]],
    delta: RuntimeToolCallDelta,
) -> None:
    current = accumulator.setdefault(delta.index, {"id": None, "name": None, "arguments": ""})
    if delta.call_id:
        current["id"] = delta.call_id
    if delta.name:
        current["name"] = (current.get("name") or "") + delta.name
    if delta.arguments:
        current["arguments"] = (current.get("arguments") or "") + delta.arguments


def _accumulated_tool_calls_as_text(accumulator: dict[int, dict[str, str | None]]) -> str:
    calls: list[dict[str, Any]] = []
    for index in sorted(accumulator):
        item = accumulator[index]
        name = item.get("name")
        if not name:
            continue
        arguments_text = item.get("arguments") or "{}"
        try:
            arguments: Any = json.loads(arguments_text)
        except json.JSONDecodeError:
            arguments = arguments_text
        call: dict[str, Any] = {"name": name, "arguments": arguments}
        if item.get("id"):
            call["id"] = item["id"]
        calls.append(call)
    return json.dumps({"tool_calls": calls}, separators=(",", ":")) if calls else ""


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
