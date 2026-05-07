"""Typed contracts shared across LewLM subsystems."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from lewlm.structured_output import StructuredOutputRequest

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


class RoutingDecision(BaseModel):
    """Explainable routing result for a generation request."""

    model_id: str
    runtime_name: str
    runtime_affinity: RuntimeAffinity
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


class HostCapabilityReadiness(BaseModel):
    """Host-level readiness summary for one execution capability."""

    capability: CapabilityName
    ready: bool
    readiness_state: CapabilityReadinessState
    reason: str
    available_runtime_names: list[str] = Field(default_factory=list)
    candidate_model_count: int = 0
    runnable_model_count: int = 0
    ready_model_ids: list[str] = Field(default_factory=list)
    blocked_model_ids: list[str] = Field(default_factory=list)
    conversion_required_model_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


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
