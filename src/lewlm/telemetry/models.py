"""Telemetry data models and shared constants."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from lewlm.core.contracts import CapabilityName, HostPlatformSnapshot, MeasuredCapabilitySummary, ServiceReadinessSummary
from lewlm.core.serving_core import ServingCoreSnapshot
from lewlm.pack_registry import PackReport
from lewlm.serving_profiles import ServingProfileApplication


class PerformanceFeatureName(str, Enum):
    SERVING_CORE = "serving_core"
    CONTINUOUS_BATCHING = "continuous_batching"
    DISTRIBUTED_PIPELINE = "distributed_pipeline"
    PREFIX_CACHE = "prefix_cache"
    PERSISTENT_MULTI_CONTEXT_CACHE = "persistent_multi_context_cache"
    HYBRID_SSM_ROUTING = "hybrid_ssm_routing"
    SSM_STATE_CACHE_HANDLING = "ssm_state_cache_handling"
    MOE_BOUNDED_MEMORY_SERVING = "moe_bounded_memory_serving"
    GRAPH_COMPILATION = "graph_compilation"
    ATTENTION_KERNEL_ACCELERATION = "attention_kernel_acceleration"
    PAGED_KV_CACHE = "paged_kv_cache"
    KV_CACHE_QUANTIZATION = "kv_cache_quantization"
    DISK_BACKED_CACHE = "disk_backed_cache"
    BLOCK_DISK_CACHE = "block_disk_cache"
    SPECULATIVE_DECODING = "speculative_decoding"
    PROMPT_LOOKUP_SPECULATION = "prompt_lookup_speculation"
    KEEP_WARM_MODEL_RESIDENCY = "keep_warm_model_residency"
    AGGRESSIVE_UNLOAD_MODE = "aggressive_unload_mode"
    BALANCED_RESIDENCY_MODE = "balanced_residency_mode"
    REQUEST_SCHEDULING_AND_BACKPRESSURE = "request_scheduling_and_backpressure"
    DECODE_PRIORITY_SCHEDULING = "decode_priority_scheduling"
    MODEL_LOAD_ADMISSION_CONTROL = "model_load_admission_control"
    PREFILL_OPTIMIZATION = "prefill_optimization"
    CHUNKED_PREFILL = "chunked_prefill"
    PREFILL_ISOLATION = "prefill_isolation"
    MULTIMODAL_FEATURE_CACHING = "multimodal_feature_caching"
    MULTIMODAL_ENCODER_CACHING = "multimodal_encoder_caching"


class PerformanceFeatureStatus(BaseModel):
    feature: PerformanceFeatureName
    supported: bool
    active: bool = False
    supported_capabilities: list[str] = Field(default_factory=list)
    runtime_names: list[str] = Field(default_factory=list)
    reason: str
    metrics: dict[str, int | float | str | bool] = Field(default_factory=dict)
    fallback_guidance: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CacheStats(BaseModel):
    cache_dir: str
    artifact_count: int
    file_count: int
    total_size_bytes: int
    cache_hits: int = 0
    cache_misses: int = 0
    conversion_cache_hits: int = 0
    conversion_cache_misses: int = 0
    runtime_response_count: int = 0
    runtime_response_bytes: int = 0
    runtime_cache_hits: int = 0
    runtime_cache_misses: int = 0
    block_cache_count: int = 0
    block_cache_bytes: int = 0
    block_cache_hits: int = 0
    block_cache_misses: int = 0
    multimodal_feature_count: int = 0
    multimodal_feature_bytes: int = 0
    multimodal_feature_cache_hits: int = 0
    multimodal_feature_cache_misses: int = 0
    multimodal_encoder_count: int = 0
    multimodal_encoder_bytes: int = 0
    multimodal_encoder_cache_hits: int = 0
    multimodal_encoder_cache_misses: int = 0
    multimodal_encoder_cache_invalidations: int = 0
    multimodal_encoder_resident_count: int = 0
    multimodal_encoder_resident_bytes: int = 0
    performance_features: list[PerformanceFeatureStatus] = Field(default_factory=list)


class ModelRuntimeMetrics(BaseModel):
    model_id: str
    runtime: str
    request_count: int
    success_count: int
    failure_count: int
    success_rate: float
    capability_counts: dict[str, int] = Field(default_factory=dict)
    last_request_at: datetime | None = None
    last_error_at: datetime | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    average_load_seconds: float | None = None
    average_execution_seconds: float | None = None
    average_completion_tokens_per_second: float | None = None


class CapabilityRuntimeMetrics(BaseModel):
    capability: str
    request_count: int
    success_count: int
    failure_count: int
    success_rate: float
    last_request_at: datetime | None = None
    last_error_at: datetime | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    average_load_seconds: float | None = None
    average_execution_seconds: float | None = None
    average_completion_tokens_per_second: float | None = None
    metric_totals: dict[str, int | float] = Field(default_factory=dict)
    metric_averages: dict[str, int | float] = Field(default_factory=dict)


class RuntimeRequestMetrics(BaseModel):
    total_requests: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 1.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    average_load_seconds: float | None = None
    average_execution_seconds: float | None = None
    average_completion_tokens_per_second: float | None = None
    models: list[ModelRuntimeMetrics] = Field(default_factory=list)
    capabilities: list[CapabilityRuntimeMetrics] = Field(default_factory=list)


class RuntimeSchedulerStats(BaseModel):
    max_concurrent_requests: int
    queue_limit: int
    queue_timeout_seconds: int
    decode_priority_enabled: bool = False
    long_prefill_token_threshold: int = 0
    prefill_isolation_enabled: bool = False
    prefill_isolation_max_concurrent_requests: int = 0
    prefill_isolation_decode_reserve: int = 0
    active_requests: int
    queued_requests: int
    active_decode_requests: int = 0
    active_prefill_requests: int = 0
    queued_decode_requests: int = 0
    queued_prefill_requests: int = 0
    peak_active_requests: int
    max_observed_queue_depth: int
    max_observed_decode_queue_depth: int = 0
    max_observed_prefill_queue_depth: int = 0
    total_queued_requests: int
    rejected_requests: int
    timed_out_requests: int
    total_queue_wait_seconds: float = 0.0
    average_queue_wait_seconds: float = 0.0
    max_queue_wait_seconds: float = 0.0
    decode_priority_requests: int = 0
    prefill_heavy_requests: int = 0
    prioritized_decode_grants: int = 0
    isolated_prefill_requests: int = 0
    native_window_milliseconds: int = 0
    native_max_batch_size: int = 0
    native_total_batches: int = 0
    native_total_requests: int = 0
    native_batched_requests: int = 0
    native_coalesced_requests: int = 0
    native_total_queue_delay_seconds: float = 0.0
    native_average_queue_delay_seconds: float = 0.0
    native_max_queue_delay_seconds: float = 0.0
    native_average_batch_size: float = 0.0
    native_average_batch_utilization: float = 0.0
    frontier_window_milliseconds: int = 0
    frontier_max_batch_size: int = 0
    frontier_total_batches: int = 0
    frontier_total_requests: int = 0
    frontier_batched_requests: int = 0
    frontier_coalesced_requests: int = 0
    frontier_total_queue_delay_seconds: float = 0.0
    frontier_average_queue_delay_seconds: float = 0.0
    frontier_max_queue_delay_seconds: float = 0.0
    frontier_average_batch_size: float = 0.0
    frontier_average_batch_utilization: float = 0.0


class BenchmarkRecord(BaseModel):
    benchmark_id: str
    model_id: str
    runtime: str
    capability: str = CapabilityName.CHAT.value
    workload_class: str | None = None
    reason: str
    prompt: str
    output_text: str
    load_seconds: float
    generate_seconds: float
    total_seconds: float
    usage: dict[str, int] = Field(default_factory=dict)
    measurements: dict[str, int | float] = Field(default_factory=dict)
    phase_breakdown: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
    optimization_attribution: dict[str, Any] = Field(default_factory=dict)
    completion_tokens_per_second: float | None = None
    created_at: datetime
    performance_features: list[PerformanceFeatureStatus] = Field(default_factory=list)
    serving_profile: ServingProfileApplication | None = None


class ModelBenchmarkSummary(BaseModel):
    model_id: str
    run_count: int
    average_total_seconds: float
    fastest_total_seconds: float
    last_run_at: datetime
    capability_counts: dict[str, int] = Field(default_factory=dict)


class BenchmarkSummary(BaseModel):
    total_runs: int = 0
    last_run_at: datetime | None = None
    average_total_seconds: float | None = None
    capability_counts: dict[str, int] = Field(default_factory=dict)
    recent_runs: list[BenchmarkRecord] = Field(default_factory=list)
    models: list[ModelBenchmarkSummary] = Field(default_factory=list)
    artifact_summary: "BenchmarkArtifactSummary" = Field(default_factory=lambda: BenchmarkArtifactSummary())


class MeasuredCapabilityRegistrySummary(BaseModel):
    host_platform: HostPlatformSnapshot
    total_records: int = 0
    latest_recorded_at: datetime | None = None
    categories: list[MeasuredCapabilitySummary] = Field(default_factory=list)


class BenchmarkScenarioSample(BaseModel):
    model_id: str | None = None
    runtime: str | None = None
    metrics: dict[str, int | float | str | bool] = Field(default_factory=dict)


class BenchmarkScenarioReport(BaseModel):
    scenario: str
    capability: str
    feature: PerformanceFeatureName | None = None
    status: str
    reason: str
    metrics: dict[str, int | float | str | bool] = Field(default_factory=dict)
    samples: list[BenchmarkScenarioSample] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class BenchmarkRegressionFailure(BaseModel):
    scope: str
    metric: str
    current: int | float | None = None
    baseline: int | float | None = None
    allowed: int | float | None = None
    message: str


class BenchmarkRegressionSummary(BaseModel):
    status: str
    compared_to_artifact_id: str | None = None
    compared_to_artifact_path: str | None = None
    failure_count: int = 0
    failures: list[BenchmarkRegressionFailure] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class BenchmarkArtifactReference(BaseModel):
    artifact_id: str
    artifact_path: str
    workload_signature: str
    created_at: datetime
    capability: str
    benchmark_count: int
    model_count: int
    regression_status: str
    compared_to_artifact_id: str | None = None


class BenchmarkArtifactSummary(BaseModel):
    total_artifacts: int = 0
    latest_artifact: BenchmarkArtifactReference | None = None
    recent_artifacts: list[BenchmarkArtifactReference] = Field(default_factory=list)


class TargetPlatformRuntimeStatus(BaseModel):
    runtime_name: str
    runtime_affinity: str
    supported: bool
    reason: str | None = None
    readiness_state: str
    verification_method: str
    install_hint: str | None = None


class TargetPlatformValidation(BaseModel):
    system: str
    machine: str
    supported_runtime_count: int
    unsupported_runtime_count: int
    compatible_model_count: int
    incompatible_model_count: int
    blocked_model_count: int
    fallback_model_count: int = 0
    compatible_models: list[str] = Field(default_factory=list)
    incompatible_models: list[str] = Field(default_factory=list)
    blocked_models: list[str] = Field(default_factory=list)
    fallback_models: list[str] = Field(default_factory=list)
    readiness_state: str
    verification_method: str
    validation_manifest_count: int = 0
    verified_model_count: int = 0
    verified_models: list[str] = Field(default_factory=list)
    verified_hosts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    runtimes: list[TargetPlatformRuntimeStatus] = Field(default_factory=list)


OPTIMIZATION_CLASS_NAMES = (
    "runtime_selection",
    "continuous_batching",
    "tiered_kv_cache",
    "speculation",
    "kernel_acceleration",
    "precision_profile",
    "frontier_execution",
    "multimodal_default_selection",
)

FRONTIER_ARCHITECTURE_SUBTYPES = {"ssm_mamba", "hybrid_ssm", "moe", "hybrid_moe"}


class OptimizationDefaultDecision(BaseModel):
    status: str
    reason: str
    benchmark_backed: bool = False
    source: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class WorkloadOptimizationDefault(BaseModel):
    workload_class: str
    runtime: str | None = None
    runtime_affinity: str | None = None
    request_modality: str | None = None
    modality_path: str | None = None
    profile_id: str | None = None
    profile_status: str = "unavailable"
    benchmark_backed: bool = False
    reason: str
    recommendation_reason: str | None = None


class ModelOptimizationDefaults(BaseModel):
    model_id: str
    display_name: str
    capability: str
    runtime: str | None = None
    runtime_affinity: str | None = None
    profile_id: str | None = None
    default_workload_class: str | None = None
    workload_defaults: list[WorkloadOptimizationDefault] = Field(default_factory=list)
    resolved: bool = False
    resolved_class_count: int = 0
    unresolved_classes: list[str] = Field(default_factory=list)
    decisions: dict[str, OptimizationDefaultDecision] = Field(default_factory=dict)


class OptimizationDefaultsSummary(BaseModel):
    format: str = "lewlm-optimization-defaults-v1"
    host_platform: HostPlatformSnapshot
    capability: str = CapabilityName.CHAT.value
    optimization_classes: list[str] = Field(default_factory=lambda: list(OPTIMIZATION_CLASS_NAMES))
    model_count: int = 0
    resolved_model_count: int = 0
    unresolved_model_count: int = 0
    resolved_classes: list[str] = Field(default_factory=list)
    benchmark_backed_classes: list[str] = Field(default_factory=list)
    complete: bool = False
    notes: list[str] = Field(default_factory=list)
    models: list[ModelOptimizationDefaults] = Field(default_factory=list)


class RuntimeStats(BaseModel):
    platform: HostPlatformSnapshot
    readiness: ServiceReadinessSummary
    runtime_policy: str
    request_max_bytes: int
    api_key_required: bool
    active_sessions: int
    queue_depth: int
    active_jobs: int
    current_loaded_models: list[str] = Field(default_factory=list)
    runtime_packs: list[PackReport] = Field(default_factory=list)
    feature_packs: list[PackReport] = Field(default_factory=list)
    runtimes: list[dict[str, Any]] = Field(default_factory=list)
    serving_core: ServingCoreSnapshot
    request_scheduler: RuntimeSchedulerStats
    load_scheduler: RuntimeSchedulerStats
    request_metrics: RuntimeRequestMetrics
    benchmark_summary: BenchmarkSummary
    measured_capability_registry: MeasuredCapabilityRegistrySummary | None = None
    validation_manifest_count: int = 0
    target_platforms: list[TargetPlatformValidation] = Field(default_factory=list)
    cluster: dict[str, Any] | None = None
    performance_features: list[PerformanceFeatureStatus] = Field(default_factory=list)
    optimization_defaults: OptimizationDefaultsSummary | None = None


RuntimeStats.model_rebuild()


class BenchmarkResult(BenchmarkRecord):
    scenarios: list[BenchmarkScenarioReport] = Field(default_factory=list)
    regression: BenchmarkRegressionSummary | None = None
    artifact: BenchmarkArtifactReference | None = None


class BenchmarkSuiteResult(BaseModel):
    capability: str
    prompt: str
    workload_class: str | None = None
    repeat_count: int = 1
    benchmark_count: int
    model_count: int
    total_load_seconds: float
    total_generate_seconds: float
    total_elapsed_seconds: float
    average_total_seconds: float | None = None
    models: list[ModelBenchmarkSummary] = Field(default_factory=list)
    results: list[BenchmarkResult] = Field(default_factory=list)
    performance_features: list[PerformanceFeatureStatus] = Field(default_factory=list)
    scenarios: list[BenchmarkScenarioReport] = Field(default_factory=list)
    regression: BenchmarkRegressionSummary | None = None
    artifact: BenchmarkArtifactReference | None = None


class AutotuneCandidateSummary(BaseModel):
    name: str
    benchmark_id: str
    runtime: str
    settings_overrides: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
    total_seconds: float
    load_seconds: float
    generate_seconds: float
    completion_tokens_per_second: float | None = None
    continuous_batching_throughput: float | None = None
    warm_cache_ttft_ratio: float | None = None
    selected_speculation_mode: str | None = None
    quantization_profile: str | None = None
    active_kernel_path: str | None = None
    active_cache_features: list[str] = Field(default_factory=list)
    artifact: BenchmarkArtifactReference | None = None
    notes: list[str] = Field(default_factory=list)


class ServingProfileRecommendation(BaseModel):
    profile_id: str
    model_id: str
    capability: str
    workload_class: str
    runtime: str
    host_platform: HostPlatformSnapshot
    prompt: str
    recommended_at: datetime
    selection_objective: str = "latency_first"
    reason: str
    settings_overrides: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
    effective_settings: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
    metrics: dict[str, int | float | str | bool] = Field(default_factory=dict)
    quantization_profile: str | None = None
    selected_speculation_mode: str | None = None
    active_kernel_path: str | None = None
    active_cache_features: list[str] = Field(default_factory=list)
    artifact: BenchmarkArtifactReference | None = None
    notes: list[str] = Field(default_factory=list)
    candidate_summaries: list[AutotuneCandidateSummary] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AutotuneCandidateSpec:
    name: str
    settings_overrides: dict[str, int | float | str | bool | None]
    notes: tuple[str, ...] = ()
