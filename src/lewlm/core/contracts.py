"""Typed contracts shared across LewLM subsystems."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from lewlm.structured_output import StructuredOutputRequest, StructuredOutputRuntimeStatus

from lewlm.core.citations import GeneratedCitationReference


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class ModelModality(str, Enum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    MULTIMODAL = "multimodal"


class ModelFormat(str, Enum):
    GGUF = "gguf"
    MLX = "mlx"
    HUGGINGFACE = "huggingface"
    AUDIO_FOLDER = "audio_folder"
    ADAPTER_BUNDLE = "adapter_bundle"
    UNKNOWN = "unknown"


class RuntimeAffinity(str, Enum):
    MLX_TEXT = "mlx_text"
    MLX_VISION = "mlx_vision"
    MLX_AUDIO = "mlx_audio"
    LLAMACPP = "llamacpp"
    EXTERNAL_ACCELERATOR = "external_accelerator"
    CONVERSION = "conversion"
    EXPERIMENTAL = "experimental"
    DISTRIBUTED_EXPERIMENTAL = "distributed_experimental"


class ConversionStatus(str, Enum):
    RUNNABLE = "runnable"
    REQUIRES_CONVERSION = "requires_conversion"
    NOT_SUPPORTED = "not_supported"
    UNKNOWN = "unknown"


class ModelArtifactRole(str, Enum):
    STANDALONE = "standalone"
    SOURCE_BUNDLE = "source_bundle"
    MULTIMODAL_RUNNABLE = "multimodal_runnable"
    TEXT_RUNNABLE = "text_runnable"


class ValidationState(str, Enum):
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


class ArchitectureSubtype(str, Enum):
    TRANSFORMER = "transformer"
    SSM_MAMBA = "ssm_mamba"
    HYBRID_SSM = "hybrid_ssm"
    MOE = "moe"
    HYBRID_MOE = "hybrid_moe"
    UNKNOWN = "unknown"


class QuantizationPrecision(str, Enum):
    INT2 = "int2"
    INT3 = "int3"
    INT4 = "int4"
    INT6 = "int6"
    INT8 = "int8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


class QuantizationStrategy(str, Enum):
    WEIGHT_ONLY = "weight_only"
    ACTIVATION_AWARE = "activation_aware"
    MIXED_PRECISION = "mixed_precision"
    HYBRID_FP8 = "hybrid_fp8"
    EXTERNAL_ADAPTIVE = "external_adaptive"


class LayerQuantizationOverride(BaseModel):
    """Per-layer mixed-precision override recorded on converted artifacts."""

    layer_pattern: str
    weight_precision: QuantizationPrecision | None = None
    activation_precision: QuantizationPrecision | None = None
    compute_precision: QuantizationPrecision | None = None
    note: str | None = None


class ExternalQuantizerReference(BaseModel):
    """Explicit external quantizer selected by the operator."""

    name: str
    profile: str | None = None
    module: str | None = None
    required_packages: list[str] = Field(default_factory=list)


class QuantizationProfile(BaseModel):
    """Structured quantization profile preserved across conversion and discovery."""

    name: str | None = None
    strategy: QuantizationStrategy = QuantizationStrategy.WEIGHT_ONLY
    weight_precision: QuantizationPrecision | None = None
    activation_precision: QuantizationPrecision | None = None
    kv_cache_precision: QuantizationPrecision | None = None
    compute_precision: QuantizationPrecision | None = None
    calibration_samples: int | None = None
    group_size: int | None = None
    layer_overrides: list[LayerQuantizationOverride] = Field(default_factory=list)
    external_quantizer: ExternalQuantizerReference | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def quantization_profile_label(profile: QuantizationProfile | None) -> str | None:
    """Return a compact operator-facing label for a quantization profile."""

    if profile is None:
        return None
    if profile.strategy == QuantizationStrategy.WEIGHT_ONLY:
        if profile.weight_precision is not None:
            return profile.weight_precision.value
        return profile.name or "weight_only"
    if profile.strategy == QuantizationStrategy.ACTIVATION_AWARE:
        if profile.weight_precision is not None and profile.activation_precision is not None:
            return f"{profile.weight_precision.value}-{profile.activation_precision.value}-aaq"
        return profile.name or "activation_aware"
    if profile.strategy == QuantizationStrategy.HYBRID_FP8:
        if profile.weight_precision is not None and profile.compute_precision is not None:
            return f"{profile.weight_precision.value}-{profile.compute_precision.value}-hybrid"
        return profile.name or "hybrid_fp8"
    if profile.strategy == QuantizationStrategy.MIXED_PRECISION:
        if profile.layer_overrides:
            return f"mixed-{len(profile.layer_overrides)}-layer"
        return profile.name or "mixed_precision"
    if profile.external_quantizer is not None:
        if profile.external_quantizer.profile:
            return f"{profile.external_quantizer.name}:{profile.external_quantizer.profile}"
        return profile.external_quantizer.name
    return profile.name or profile.strategy.value


class CapabilityName(str, Enum):
    CHAT = "chat"
    STREAMING = "streaming"
    VISION = "vision"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_SPEECH = "audio_speech"
    EMBEDDINGS = "embeddings"
    RERANK = "rerank"
    CONVERSION = "conversion"


class CapabilityReadinessState(str, Enum):
    READY = "ready"
    NO_MODELS = "no_models"
    CONVERSION_REQUIRED = "conversion_required"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    BLOCKED = "blocked"


class RuntimeReadinessState(str, Enum):
    READY = "ready"
    UNREGISTERED = "unregistered"
    HOST_UNSUPPORTED = "host_unsupported"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    MANIFEST_UNSUPPORTED = "manifest_unsupported"


class ServiceReadinessState(str, Enum):
    READY = "ready"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class ReasoningVisibility(str, Enum):
    HIDDEN = "hidden"
    SUMMARIZED = "summarized"
    RAW_MODEL_EMITTED = "raw_model_emitted"


class ReasoningOutput(BaseModel):
    """Structured reasoning metadata exposed according to policy."""

    visibility: ReasoningVisibility
    available: bool = False
    content: str | None = None
    summary: str | None = None


class ModelValidationResult(BaseModel):
    """Latest validation status associated with a discovered model."""

    status: ValidationState
    message: str
    checked_at: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class ModelArtifactLayer(BaseModel):
    """A single layer in a converted model artifact lineage."""

    artifact_key: str
    role: ModelArtifactRole
    display_name: str
    format_type: ModelFormat
    source_path: str
    modality: tuple[ModelModality, ...] = ()
    runtime_affinity: tuple[RuntimeAffinity, ...] = ()
    tokenizer_path: str | None = None
    processor_path: str | None = None
    quantization: str | None = None
    quantization_profile: QuantizationProfile | None = None
    derived_from: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelManifest(BaseModel):
    """Normalized record stored in the local model registry."""

    model_id: str
    display_name: str
    architecture_family: str
    architecture_subtype: ArchitectureSubtype = ArchitectureSubtype.UNKNOWN
    modality: tuple[ModelModality, ...]
    source_path: str
    format_type: ModelFormat
    quantization: str | None = None
    quantization_profile: QuantizationProfile | None = None
    tokenizer_path: str | None = None
    processor_path: str | None = None
    runtime_affinity: tuple[RuntimeAffinity, ...]
    text_only_runtime_affinity: tuple[RuntimeAffinity, ...] = Field(default_factory=tuple)
    text_only_runtime_source: str | None = None
    text_only_runtime_reason: str | None = None
    artifact_key: str | None = None
    artifact_role: ModelArtifactRole = ModelArtifactRole.STANDALONE
    artifact_family_id: str | None = None
    artifact_lineage: list[ModelArtifactLayer] = Field(default_factory=list)
    required_extra_files: list[str] = Field(default_factory=list)
    estimated_memory_mb: int | None = None
    context_length: int | None = None
    conversion_status: ConversionStatus
    fingerprint: str
    last_validation_result: ModelValidationResult
    metadata: dict[str, Any] = Field(default_factory=dict)
    discovered_at: datetime = Field(default_factory=utc_now)


class ModelInventory(BaseModel):
    """API response envelope for listing discovered models."""

    count: int
    items: list[ModelManifest]


class ModelScanSummary(BaseModel):
    """Result payload returned after scanning model directories."""

    roots_scanned: tuple[str, ...]
    discovered_count: int
    new_count: int
    updated_count: int
    unchanged_count: int
    removed_count: int
    manifests: list[ModelManifest]
    scanned_at: datetime = Field(default_factory=utc_now)


class HostPlatformSnapshot(BaseModel):
    """Basic information about the current host platform."""

    system: str
    release: str
    machine: str
    python_version: str
    total_memory_mb: int | None = None
    total_memory_source: str | None = None
    total_memory_reason: str | None = None


class RuntimeCandidateReport(BaseModel):
    """Availability and compatibility details for a candidate runtime."""

    runtime_name: str
    runtime_affinity: RuntimeAffinity
    readiness_state: RuntimeReadinessState
    registered: bool
    available: bool
    availability_reason: str | None = None
    host_platform_supported: bool
    supported_systems: list[str] = Field(default_factory=list)
    supported_machines: list[str] = Field(default_factory=list)
    support_path: "RuntimeSupportPath" = "packaged"
    supports_manifest: bool


class RequestModality(str, Enum):
    TEXT_ONLY = "text_only"
    IMAGE_CONDITIONED = "image_conditioned"
    FRAME_BUNDLE_VIDEO = "frame_bundle_video"
    AUDIO_CONDITIONED = "audio_conditioned"


class RoutingModalityPath(str, Enum):
    TEXT_DEFAULT = "text_default"
    TEXT_FAST_PATH = "text_fast_path"
    MULTIMODAL_DEFAULT = "multimodal_default"


class RuntimeSupportPath(str, Enum):
    PACKAGED = "packaged"
    BRIDGE = "bridge"


class RoutingDecision(BaseModel):
    """Explainable routing result for a generation request."""

    model_id: str
    runtime_name: str
    runtime_affinity: RuntimeAffinity
    support_path: RuntimeSupportPath | None = None
    reason: str
    request_modality: RequestModality | None = None
    modality_path: RoutingModalityPath | None = None
    modality_path_reason: str | None = None
    alternatives: list[str] = Field(default_factory=list)


class IdempotentOperationRecord(BaseModel):
    """Persisted response envelope reused by idempotent document/tool operations."""

    operation_name: str
    idempotency_key: str
    request_hash: str
    response_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class CapabilityReport(BaseModel):
    """Runtime support for a specific capability."""

    capability: CapabilityName
    supported: bool
    reason: str | None = None


class ModelCapabilityStatus(BaseModel):
    """Capability outcome for a specific model on the current host."""

    capability: CapabilityName
    supported: bool
    readiness_state: CapabilityReadinessState = CapabilityReadinessState.BLOCKED
    runtime_name: str | None = None
    runtime_affinity: RuntimeAffinity | None = None
    support_path: RuntimeSupportPath | None = None
    reason: str
    alternatives: list[str] = Field(default_factory=list)
    estimated_memory_mb: int | None = None
    notes: list[str] = Field(default_factory=list)


class MeasuredCapabilityCategory(str, Enum):
    BATCHING = "batching"
    CACHE_REUSE = "cache_reuse"
    SPECULATION = "speculation"
    CONSTRAINED_DECODING = "constrained_decoding"
    COMPILE_KERNELS = "compile_kernels"
    ADAPTER_PRESERVATION = "adapter_preservation"


class MeasuredCapabilityStatus(str, Enum):
    SUPPORTED = "supported"
    DEGRADED = "degraded"
    FALLBACK = "fallback"
    REJECTED = "rejected"
    NOT_APPLICABLE = "not_applicable"
    UNMEASURED = "unmeasured"
    MIXED = "mixed"


class MeasuredCapabilityEvidenceSource(str, Enum):
    BENCHMARK_FEATURE = "benchmark_feature"
    BENCHMARK_SCENARIO = "benchmark_scenario"
    CODE_PROBE = "code_probe"
    EXTERNAL_ADAPTER_COMPARISON = "external_adapter_comparison"


class MeasuredCapabilityProbeRecord(BaseModel):
    """Persisted probe or benchmark evidence for one measured capability category."""

    probe_key: str
    category: MeasuredCapabilityCategory
    probe_name: str
    status: MeasuredCapabilityStatus
    source: MeasuredCapabilityEvidenceSource
    reason: str
    runtime_name: str | None = None
    runtime_affinity: RuntimeAffinity | None = None
    model_id: str | None = None
    workload_class: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=utc_now)


class MeasuredCapabilitySummary(BaseModel):
    """Summarized measured evidence for one capability category."""

    category: MeasuredCapabilityCategory
    status: MeasuredCapabilityStatus = MeasuredCapabilityStatus.UNMEASURED
    reason: str
    record_count: int = 0
    latest_recorded_at: datetime | None = None
    runtime_names: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    probes: list[MeasuredCapabilityProbeRecord] = Field(default_factory=list)


class PerformanceFeatureOwnership(str, Enum):
    """Truthful ownership mode for one performance-core feature surface."""

    LEWLM_OWNED = "lewlm_owned"
    BACKEND_NATIVE = "backend_native"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class PerformanceCoreEvidenceFamily(str, Enum):
    CONTINUOUS_BATCHING = "continuous_batching"
    TIERED_KV = "tiered_kv"
    PREFIX_REUSE = "prefix_reuse"
    PREFILL_ISOLATION = "prefill_isolation"
    SPECULATION = "speculation"
    CONSTRAINED_DECODING = "constrained_decoding"
    KERNEL_ACCELERATION = "kernel_acceleration"


class PerformanceCoreEvidenceMode(str, Enum):
    LEWLM_OWNED = "lewlm_owned"
    BACKEND_NATIVE = "backend_native"
    FALLBACK = "fallback"
    UNSUPPORTED = "unsupported"


class PerformanceCoreEvidenceSource(str, Enum):
    RUNTIME_FEATURE = "runtime_feature"
    BENCHMARK_FEATURE = "benchmark_feature"
    BENCHMARK_SCENARIO = "benchmark_scenario"
    MEASURED_CAPABILITY = "measured_capability"
    RUNTIME_SUPPORT_STRATEGY = "runtime_support_strategy"


class PerformanceCoreEvidenceRecord(BaseModel):
    """Portable performance-core evidence summary shared across reporting surfaces."""

    family: PerformanceCoreEvidenceFamily
    mode: PerformanceCoreEvidenceMode = PerformanceCoreEvidenceMode.UNSUPPORTED
    reason: str
    runtime_names: list[str] = Field(default_factory=list)
    feature_names: list[str] = Field(default_factory=list)
    measured_categories: list[MeasuredCapabilityCategory] = Field(default_factory=list)
    sources: list[PerformanceCoreEvidenceSource] = Field(default_factory=list)
    benchmark_backed: bool = False
    notes: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class RuntimePerformanceFeatureReport(BaseModel):
    """Portable runtime-facing report for one performance feature."""

    supported: bool
    active: bool = False
    ownership: PerformanceFeatureOwnership = PerformanceFeatureOwnership.UNSUPPORTED
    reason: str | None = None
    metrics: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    modes: list[str] = Field(default_factory=list)


def normalize_performance_feature_ownership(
    *,
    ownership: PerformanceFeatureOwnership | str | None = None,
    supported: bool | None = None,
    support_level: str | None = None,
) -> PerformanceFeatureOwnership:
    """Normalize legacy or partial feature payloads onto one ownership enum."""

    if isinstance(ownership, PerformanceFeatureOwnership):
        return ownership
    if isinstance(ownership, str):
        normalized = ownership.casefold()
        for candidate in PerformanceFeatureOwnership:
            if candidate.value == normalized:
                return candidate
    if isinstance(support_level, str):
        normalized_support_level = support_level.casefold()
        if normalized_support_level == PerformanceFeatureOwnership.PARTIAL.value:
            return PerformanceFeatureOwnership.PARTIAL
        if normalized_support_level == PerformanceFeatureOwnership.UNSUPPORTED.value:
            return PerformanceFeatureOwnership.UNSUPPORTED
        if normalized_support_level == "supported":
            return (
                PerformanceFeatureOwnership.BACKEND_NATIVE
                if supported is not False
                else PerformanceFeatureOwnership.UNSUPPORTED
            )
    return (
        PerformanceFeatureOwnership.BACKEND_NATIVE
        if supported
        else PerformanceFeatureOwnership.UNSUPPORTED
    )


def performance_feature_support_level(ownership: PerformanceFeatureOwnership | str) -> str:
    """Return the legacy support-level label derived from ownership."""

    normalized = normalize_performance_feature_ownership(ownership=ownership)
    if normalized in {
        PerformanceFeatureOwnership.LEWLM_OWNED,
        PerformanceFeatureOwnership.BACKEND_NATIVE,
    }:
        return "supported"
    return normalized.value


def normalize_performance_core_evidence_mode(
    *,
    mode: PerformanceCoreEvidenceMode | str | None = None,
    ownership: PerformanceFeatureOwnership | str | None = None,
    supported: bool | None = None,
    support_level: str | None = None,
) -> PerformanceCoreEvidenceMode:
    """Normalize portable evidence modes from either explicit modes or feature ownership."""

    if isinstance(mode, PerformanceCoreEvidenceMode):
        return mode
    if isinstance(mode, str):
        normalized_mode = mode.casefold()
        if normalized_mode == PerformanceFeatureOwnership.PARTIAL.value:
            return PerformanceCoreEvidenceMode.FALLBACK
        for candidate in PerformanceCoreEvidenceMode:
            if candidate.value == normalized_mode:
                return candidate
    normalized_ownership = normalize_performance_feature_ownership(
        ownership=ownership,
        supported=supported,
        support_level=support_level,
    )
    if normalized_ownership == PerformanceFeatureOwnership.LEWLM_OWNED:
        return PerformanceCoreEvidenceMode.LEWLM_OWNED
    if normalized_ownership == PerformanceFeatureOwnership.BACKEND_NATIVE:
        return PerformanceCoreEvidenceMode.BACKEND_NATIVE
    if normalized_ownership == PerformanceFeatureOwnership.PARTIAL:
        return PerformanceCoreEvidenceMode.FALLBACK
    return PerformanceCoreEvidenceMode.UNSUPPORTED


def performance_core_evidence_mode_from_measured_status(
    status: MeasuredCapabilityStatus | str | None,
) -> PerformanceCoreEvidenceMode:
    """Map measured capability outcomes onto the portable evidence vocabulary."""

    if isinstance(status, MeasuredCapabilityStatus):
        normalized_status = status
    elif isinstance(status, str):
        try:
            normalized_status = MeasuredCapabilityStatus(status)
        except ValueError:
            return PerformanceCoreEvidenceMode.UNSUPPORTED
    else:
        return PerformanceCoreEvidenceMode.UNSUPPORTED
    if normalized_status == MeasuredCapabilityStatus.SUPPORTED:
        return PerformanceCoreEvidenceMode.BACKEND_NATIVE
    if normalized_status in {
        MeasuredCapabilityStatus.DEGRADED,
        MeasuredCapabilityStatus.FALLBACK,
        MeasuredCapabilityStatus.MIXED,
    }:
        return PerformanceCoreEvidenceMode.FALLBACK
    return PerformanceCoreEvidenceMode.UNSUPPORTED


def runtime_performance_feature_report(
    *,
    ownership: PerformanceFeatureOwnership | str,
    reason: str | None,
    active: bool = False,
    metrics: Mapping[str, int | float | str | bool | None] | None = None,
    notes: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a portable runtime feature payload while retaining legacy fields."""

    normalized_ownership = normalize_performance_feature_ownership(ownership=ownership)
    payload = RuntimePerformanceFeatureReport(
        supported=normalized_ownership != PerformanceFeatureOwnership.UNSUPPORTED,
        active=active,
        ownership=normalized_ownership,
        reason=reason,
        metrics=dict(metrics or {}),
        notes=list(notes or []),
        modes=[mode for mode in modes or () if isinstance(mode, str) and mode],
    ).model_dump(mode="json")
    payload["support_level"] = performance_feature_support_level(normalized_ownership)
    return payload


def normalize_runtime_performance_feature_report(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize existing runtime feature payloads onto the portable contract."""

    if payload is None:
        return runtime_performance_feature_report(
            ownership=PerformanceFeatureOwnership.UNSUPPORTED,
            reason=None,
        )
    normalized_ownership = normalize_performance_feature_ownership(
        ownership=payload.get("ownership"),
        supported=bool(payload.get("supported")),
        support_level=(str(payload.get("support_level")) if payload.get("support_level") is not None else None),
    )
    if not bool(payload.get("supported")) and normalized_ownership != PerformanceFeatureOwnership.UNSUPPORTED:
        normalized_ownership = PerformanceFeatureOwnership.UNSUPPORTED
    metrics = payload.get("metrics")
    notes = payload.get("notes")
    modes = payload.get("modes")
    normalized = runtime_performance_feature_report(
        ownership=normalized_ownership,
        reason=(str(payload.get("reason")) if payload.get("reason") is not None else None),
        active=bool(payload.get("active")),
        metrics=metrics if isinstance(metrics, Mapping) else None,
        notes=notes if isinstance(notes, Sequence) and not isinstance(notes, str) else None,
        modes=modes if isinstance(modes, Sequence) and not isinstance(modes, str) else None,
    )
    for key, value in payload.items():
        if key not in normalized:
            normalized[str(key)] = value
    return normalized


_PORTABLE_PERFORMANCE_CORE_COMPONENTS: dict[PerformanceCoreEvidenceFamily, tuple[str, ...]] = {
    PerformanceCoreEvidenceFamily.CONTINUOUS_BATCHING: ("continuous_batching",),
    PerformanceCoreEvidenceFamily.TIERED_KV: ("paged_kv_cache", "kv_cache_quantization"),
    PerformanceCoreEvidenceFamily.PREFIX_REUSE: ("prefix_cache", "persistent_multi_context_cache"),
    PerformanceCoreEvidenceFamily.PREFILL_ISOLATION: (
        "prefill_isolation",
        "chunked_prefill",
        "prefill_optimization",
    ),
    PerformanceCoreEvidenceFamily.SPECULATION: (
        "speculative_decoding",
        "prompt_lookup_speculation",
    ),
    PerformanceCoreEvidenceFamily.CONSTRAINED_DECODING: ("constrained_decoding",),
    PerformanceCoreEvidenceFamily.KERNEL_ACCELERATION: (
        "graph_compilation",
        "attention_kernel_acceleration",
    ),
}

_PORTABLE_PERFORMANCE_CORE_UNSUPPORTED_REASONS: dict[PerformanceCoreEvidenceFamily, str] = {
    PerformanceCoreEvidenceFamily.CONTINUOUS_BATCHING: (
        "No runtime currently reports portable continuous-batching evidence on this host."
    ),
    PerformanceCoreEvidenceFamily.TIERED_KV: (
        "No runtime currently reports portable tiered-KV residency evidence on this host."
    ),
    PerformanceCoreEvidenceFamily.PREFIX_REUSE: (
        "No runtime currently reports portable prefix-reuse evidence on this host."
    ),
    PerformanceCoreEvidenceFamily.PREFILL_ISOLATION: (
        "No runtime currently reports portable prefill-isolation evidence on this host."
    ),
    PerformanceCoreEvidenceFamily.SPECULATION: (
        "No runtime currently reports portable speculation evidence on this host."
    ),
    PerformanceCoreEvidenceFamily.CONSTRAINED_DECODING: (
        "No runtime currently reports portable constrained-decoding evidence on this host."
    ),
    PerformanceCoreEvidenceFamily.KERNEL_ACCELERATION: (
        "No runtime currently reports portable kernel-acceleration evidence on this host."
    ),
}


def build_portable_performance_core_evidence(
    *,
    performance_features: Mapping[str, Any] | None,
    runtime_names: Sequence[str] | None = None,
    benchmark_backed: bool = False,
    source: PerformanceCoreEvidenceSource = PerformanceCoreEvidenceSource.RUNTIME_FEATURE,
) -> list[PerformanceCoreEvidenceRecord]:
    """Project a runtime feature map onto the portable performance-core families."""

    feature_map = {
        str(name): payload
        for name, payload in (performance_features or {}).items()
        if isinstance(name, str)
    }
    runtime_name_list = sorted({name for name in runtime_names or () if isinstance(name, str) and name})
    return [
        _portable_performance_core_record(
            family=family,
            feature_map=feature_map,
            runtime_names=runtime_name_list,
            benchmark_backed=benchmark_backed,
            source=source,
        )
        for family in PerformanceCoreEvidenceFamily
    ]


def _portable_performance_core_record(
    *,
    family: PerformanceCoreEvidenceFamily,
    feature_map: Mapping[str, Any],
    runtime_names: list[str],
    benchmark_backed: bool,
    source: PerformanceCoreEvidenceSource,
) -> PerformanceCoreEvidenceRecord:
    component_names = _PORTABLE_PERFORMANCE_CORE_COMPONENTS[family]
    components = [
        (
            component_name,
            normalize_runtime_performance_feature_report(
                payload if isinstance(payload := feature_map.get(component_name), Mapping) else None,
            ),
        )
        for component_name in component_names
    ]
    component_modes = {
        component_name: _portable_feature_mode(payload)
        for component_name, payload in components
    }
    supported_components = [component_name for component_name, payload in components if bool(payload.get("supported"))]
    mode = _portable_performance_core_mode(
        family=family,
        components=components,
        component_modes=component_modes,
    )
    notes = _portable_performance_core_notes(
        family=family,
        mode=mode,
        components=components,
    )
    return PerformanceCoreEvidenceRecord(
        family=family,
        mode=mode,
        reason=_portable_performance_core_reason(
            family=family,
            mode=mode,
            components=components,
            supported_components=supported_components,
        ),
        runtime_names=runtime_names,
        feature_names=list(component_names),
        sources=[source],
        benchmark_backed=benchmark_backed,
        notes=notes,
        metrics={
            "supported_component_count": len(supported_components),
            "component_modes": ",".join(
                f"{component_name}:{component_modes[component_name].value}"
                for component_name in component_names
            ),
        },
    )


def _portable_feature_mode(payload: Mapping[str, Any]) -> PerformanceCoreEvidenceMode:
    return normalize_performance_core_evidence_mode(
        ownership=(str(payload.get("ownership")) if payload.get("ownership") is not None else None),
        supported=bool(payload.get("supported")),
        support_level=(str(payload.get("support_level")) if payload.get("support_level") is not None else None),
    )


def _portable_performance_core_mode(
    *,
    family: PerformanceCoreEvidenceFamily,
    components: Sequence[tuple[str, Mapping[str, Any]]],
    component_modes: Mapping[str, PerformanceCoreEvidenceMode],
) -> PerformanceCoreEvidenceMode:
    supported_components = {
        component_name
        for component_name, payload in components
        if bool(payload.get("supported"))
    }
    if family == PerformanceCoreEvidenceFamily.TIERED_KV:
        if "paged_kv_cache" in supported_components:
            return component_modes["paged_kv_cache"]
        if "kv_cache_quantization" in supported_components:
            return PerformanceCoreEvidenceMode.FALLBACK
        return PerformanceCoreEvidenceMode.UNSUPPORTED
    if family == PerformanceCoreEvidenceFamily.PREFIX_REUSE:
        if "prefix_cache" in supported_components:
            return component_modes["prefix_cache"]
        if "persistent_multi_context_cache" in supported_components:
            return component_modes["persistent_multi_context_cache"]
        return PerformanceCoreEvidenceMode.UNSUPPORTED
    if family == PerformanceCoreEvidenceFamily.PREFILL_ISOLATION:
        if "prefill_isolation" in supported_components:
            return component_modes["prefill_isolation"]
        if {"chunked_prefill", "prefill_optimization"} & supported_components:
            return PerformanceCoreEvidenceMode.FALLBACK
        return PerformanceCoreEvidenceMode.UNSUPPORTED
    if family == PerformanceCoreEvidenceFamily.SPECULATION:
        if "speculative_decoding" in supported_components:
            return component_modes["speculative_decoding"]
        if "prompt_lookup_speculation" in supported_components:
            return component_modes["prompt_lookup_speculation"]
        return PerformanceCoreEvidenceMode.UNSUPPORTED
    if family == PerformanceCoreEvidenceFamily.KERNEL_ACCELERATION:
        fallback_requests = sum(
            _coerce_metric_int((payload.get("metrics") or {}).get("compile_fallback_requests"))
            + _coerce_metric_int((payload.get("metrics") or {}).get("kernel_fallback_requests"))
            for _, payload in components
        )
        if fallback_requests > 0 and supported_components:
            return PerformanceCoreEvidenceMode.FALLBACK
    for candidate in (
        PerformanceCoreEvidenceMode.LEWLM_OWNED,
        PerformanceCoreEvidenceMode.BACKEND_NATIVE,
        PerformanceCoreEvidenceMode.FALLBACK,
    ):
        if any(component_modes[name] == candidate for name in supported_components):
            return candidate
    return PerformanceCoreEvidenceMode.UNSUPPORTED


def _portable_performance_core_reason(
    *,
    family: PerformanceCoreEvidenceFamily,
    mode: PerformanceCoreEvidenceMode,
    components: Sequence[tuple[str, Mapping[str, Any]]],
    supported_components: Sequence[str],
) -> str:
    if mode == PerformanceCoreEvidenceMode.UNSUPPORTED:
        return _PORTABLE_PERFORMANCE_CORE_UNSUPPORTED_REASONS[family]
    if family == PerformanceCoreEvidenceFamily.TIERED_KV and "paged_kv_cache" not in supported_components:
        return (
            "A runtime reports KV-cache quantization details, but LewLM does not yet have matching paged-KV "
            "residency evidence for the same path."
        )
    if family == PerformanceCoreEvidenceFamily.PREFILL_ISOLATION and "prefill_isolation" not in supported_components:
        return (
            "A runtime reports prefill acceleration hooks, but LewLM does not see the combined scheduler and chunking "
            "signals needed to claim portable prefill isolation."
        )
    reasons = [
        str(payload.get("reason"))
        for component_name, payload in components
        if component_name in supported_components and isinstance(payload.get("reason"), str) and payload.get("reason")
    ]
    if reasons:
        return " ".join(dict.fromkeys(reasons))
    return _PORTABLE_PERFORMANCE_CORE_UNSUPPORTED_REASONS[family]


def _portable_performance_core_notes(
    *,
    family: PerformanceCoreEvidenceFamily,
    mode: PerformanceCoreEvidenceMode,
    components: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[str]:
    notes: list[str] = []
    seen: set[str] = set()
    for component_name, payload in components:
        for note in payload.get("notes", []):
            if not isinstance(note, str) or not note or note in seen:
                continue
            seen.add(note)
            notes.append(note)
    if family == PerformanceCoreEvidenceFamily.PREFILL_ISOLATION and mode == PerformanceCoreEvidenceMode.FALLBACK:
        fallback_note = (
            "Prefill optimization is present, but LewLM keeps prefill isolation in fallback state until the runtime "
            "exposes truthful isolation hooks."
        )
        if fallback_note not in seen:
            notes.append(fallback_note)
    return notes


def _coerce_metric_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class HostCapabilityReadiness(BaseModel):
    """Host-level readiness summary for one execution capability."""

    capability: CapabilityName
    ready: bool
    readiness_state: CapabilityReadinessState
    reason: str
    available_runtime_names: list[str] = Field(default_factory=list)
    available_support_paths: list[RuntimeSupportPath] = Field(default_factory=list)
    packaged_runtime_names: list[str] = Field(default_factory=list)
    bridge_runtime_names: list[str] = Field(default_factory=list)
    bridge_only: bool = False
    candidate_model_count: int = 0
    runnable_model_count: int = 0
    ready_model_ids: list[str] = Field(default_factory=list)
    blocked_model_ids: list[str] = Field(default_factory=list)
    conversion_required_model_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def runtime_support_path_for_affinity(
    affinity: RuntimeAffinity | str | None,
) -> RuntimeSupportPath | None:
    """Map a runtime affinity onto LewLM's packaged-versus-bridge path label."""

    if affinity is None:
        return None
    normalized_affinity = affinity.value if isinstance(affinity, RuntimeAffinity) else str(affinity)
    if normalized_affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR.value:
        return RuntimeSupportPath.BRIDGE
    return RuntimeSupportPath.PACKAGED


class ServiceReadinessSummary(BaseModel):
    """Machine-readable readiness summary for host-app consumers."""

    status: ServiceReadinessState
    host_platform: HostPlatformSnapshot
    discovered_model_count: int = 0
    runnable_model_count: int = 0
    capability_count: int = 0
    ready_capability_count: int = 0
    capabilities: list[HostCapabilityReadiness] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModelTargetPlatformReport(BaseModel):
    """Per-model readiness summary for a specific target platform."""

    system: str
    machine: str
    supported: bool
    readiness_state: str
    verification_method: str
    runtime_affinities: list[RuntimeAffinity] = Field(default_factory=list)
    reason: str
    fallback_available: bool = False
    fallback_reason: str | None = None
    install_hints: list[str] = Field(default_factory=list)
    validation_manifest_count: int = 0
    verified_hosts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModelCapabilityReport(BaseModel):
    """Per-model capability summary for the current host platform."""

    model_id: str
    display_name: str
    architecture_family: str
    format_type: ModelFormat
    modality: tuple[ModelModality, ...]
    quantization: str | None = None
    quantization_profile: QuantizationProfile | None = None
    validation_key: str
    conversion_status: ConversionStatus
    host_platform: HostPlatformSnapshot
    runtime_candidates: list[RuntimeCandidateReport] = Field(default_factory=list)
    target_platforms: list[ModelTargetPlatformReport] = Field(default_factory=list)
    capabilities: list[ModelCapabilityStatus] = Field(default_factory=list)
    measured_capabilities: list[MeasuredCapabilitySummary] = Field(default_factory=list)
    performance_core_evidence: list[PerformanceCoreEvidenceRecord] = Field(default_factory=list)


class RuntimeEstimate(BaseModel):
    """Estimated resources required to serve a model."""

    estimated_memory_mb: int | None = None
    notes: list[str] = Field(default_factory=list)


class GenerateAttachment(BaseModel):
    """Normalized attachment metadata associated with a message."""

    attachment_type: str
    name: str
    source_path: str | None = None
    media_type: str | None = None
    detail: Literal["auto", "low", "high"] | None = None
    extracted_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateMessage(BaseModel):
    """Normalized request message for text generation."""

    role: str
    content: str
    attachments: list[GenerateAttachment] = Field(default_factory=list)


class SpeculationMode(str, Enum):
    """Guide-level speculation modes for chat generation."""

    DRAFT_MODEL = "draft_model"
    PROMPT_LOOKUP = "prompt_lookup"
    MEDUSA = "medusa"
    EAGLE = "eagle"
    HYDRA = "hydra"
    DFLASH = "dflash"
    SELF_SPECULATIVE = "self_speculative"
    SUFFIX_DECODING = "suffix_decoding"
    HETEROGENEOUS_VOCAB = "heterogeneous_vocab"


class GenerateSpeculation(BaseModel):
    """Optional runtime-specific speculation configuration for one request."""

    mode: SpeculationMode
    draft_model_id: str | None = None
    companion_model_id: str | None = None
    num_draft_tokens: int | None = None
    prompt_lookup_max_ngram_size: int | None = None
    prompt_lookup_num_pred_tokens: int | None = None
    parameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    auto_selected: bool = False


class GenerateRequest(BaseModel):
    """Runtime-agnostic generation request."""

    model_id: str
    messages: list[GenerateMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    reasoning_visibility: ReasoningVisibility = ReasoningVisibility.HIDDEN
    speculation: GenerateSpeculation | None = None
    structured_output: StructuredOutputRequest | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    """Runtime-agnostic generation response."""

    model_id: str
    output_text: str
    finish_reason: str
    usage: dict[str, int] = Field(default_factory=dict)
    reasoning: ReasoningOutput | None = None
    citations: list[GeneratedCitationReference] = Field(default_factory=list)


class EmbeddingRequest(BaseModel):
    """Runtime-agnostic embedding request."""

    model_id: str
    inputs: list[str]
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingVector(BaseModel):
    """Single embedding vector result."""

    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """Runtime-agnostic embedding response."""

    model_id: str
    data: list[EmbeddingVector]
    usage: dict[str, int] = Field(default_factory=dict)


class RerankRequest(BaseModel):
    """Runtime-agnostic rerank request."""

    model_id: str
    query: str
    documents: list[str]
    top_n: int | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankResult(BaseModel):
    """Single rerank result."""

    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(BaseModel):
    """Runtime-agnostic rerank response."""

    model_id: str
    results: list[RerankResult]


class AudioTranscriptionRequest(BaseModel):
    """Runtime-agnostic audio transcription request."""

    model_id: str
    audio_bytes: bytes
    file_name: str
    language: str | None = None
    prompt: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioTranscriptionSegment(BaseModel):
    """Single transcription segment."""

    start_seconds: float | None = None
    end_seconds: float | None = None
    text: str


class AudioTranscriptionResponse(BaseModel):
    """Runtime-agnostic audio transcription response."""

    model_id: str
    text: str
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[AudioTranscriptionSegment] = Field(default_factory=list)


class AudioSpeechRequest(BaseModel):
    """Runtime-agnostic audio synthesis request."""

    model_id: str
    input_text: str
    voice: str | None = None
    audio_format: str = "wav"
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioSpeechResponse(BaseModel):
    """Runtime-agnostic audio synthesis response."""

    model_id: str
    audio_bytes: bytes
    media_type: str
    voice: str | None = None
    duration_seconds: float | None = None


@runtime_checkable
class RuntimeContract(Protocol):
    """Protocol implemented by runtime backends."""

    name: str
    affinity: RuntimeAffinity
    supported_systems: tuple[str, ...]
    supported_machines: tuple[str, ...]

    @property
    def loaded_model_ids(self) -> tuple[str, ...]: ...

    def is_available(self) -> bool: ...

    def availability_reason(self) -> str | None: ...

    def supports_host_platform(self) -> bool: ...

    def host_platform_reason(self) -> str | None: ...

    def supports_target_platform(self, system: str, machine: str) -> bool: ...

    def target_platform_reason(self, system: str, machine: str) -> str | None: ...

    def supports_manifest(self, manifest: ModelManifest) -> bool: ...

    def supports_manifest_capability(self, manifest: ModelManifest, capability: CapabilityName) -> bool: ...

    def manifest_capability_reason(self, manifest: ModelManifest, capability: CapabilityName) -> str | None: ...

    def is_model_loaded(self, model_id: str) -> bool: ...

    def loaded_model_count(self) -> int: ...

    def loaded_manifests(self) -> tuple[ModelManifest, ...]: ...

    def last_used_at(self, model_id: str) -> datetime | None: ...

    async def load_model(self, manifest: ModelManifest) -> None: ...

    async def unload_model(self, model_id: str) -> None: ...

    async def warm_model(self, model_id: str) -> None: ...

    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...

    def stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]: ...

    def structured_output_runtime_status(
        self,
        contract: StructuredOutputRequest | None,
    ) -> StructuredOutputRuntimeStatus | None: ...

    def supports_continuous_batching(self, capability: CapabilityName) -> bool: ...

    def supports_chunked_prefill(self, capability: CapabilityName) -> bool: ...

    def supports_prefill_isolation(self, capability: CapabilityName) -> bool: ...

    async def generate_batch(self, requests: Sequence[GenerateRequest]) -> list[GenerateResponse]: ...

    def stream_generate_batch(self, requests: Sequence[GenerateRequest]) -> AsyncIterator[tuple[int, str]]: ...

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...

    async def rerank(self, request: RerankRequest) -> RerankResponse: ...

    async def transcribe_audio(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse: ...

    async def synthesize_speech(self, request: AudioSpeechRequest) -> AudioSpeechResponse: ...

    def tokenize(self, text: str) -> list[int]: ...

    def detokenize(self, tokens: Sequence[int]) -> str: ...

    def estimate_resources(self, manifest: ModelManifest) -> RuntimeEstimate: ...

    def supports_capability(self, capability: CapabilityName) -> bool: ...

    def performance_feature_snapshot(self) -> dict[str, Any]: ...

    async def health_check(self) -> dict[str, Any]: ...
