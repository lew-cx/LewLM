"""Cache, runtime, and benchmark statistics."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4
import wave

from pydantic import BaseModel, Field

from lewlm.config.settings import LewLMSettings
from lewlm.conversion.service import ConversionService
from lewlm.core.contracts import (
    AudioTranscriptionRequest,
    CapabilityName,
    ConversionStatus,
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateAttachment,
    GenerateMessage,
    GenerateRequest,
    GenerateResponse,
    HostPlatformSnapshot,
    MeasuredCapabilityCategory,
    MeasuredCapabilityEvidenceSource,
    MeasuredCapabilityProbeRecord,
    MeasuredCapabilityStatus,
    MeasuredCapabilitySummary,
    ModelManifest,
    ModelModality,
    RerankRequest,
    RerankResponse,
    RoutingDecision,
    RuntimeContract,
    ServiceReadinessSummary,
    SpeculationMode,
    quantization_profile_label,
    utc_now,
)
from lewlm.core.errors import ConfigurationError, RoutingError
from lewlm.core.speculation import (
    SpeculationBenchmarkPreference,
    inspect_chat_speculation_candidates,
    parse_speculation_benchmark_preference,
    speculation_benchmark_preference_key,
    speculation_measurements,
)
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.routing.measured_preferences import assess_runtime_preference
from lewlm.runtime.catalog import RuntimeCatalog
from lewlm.runtime.experimental import (
    build_frontier_serving_plan,
    distributed_pipeline_measurements,
    frontier_architecture_measurements,
    frontier_plan_notes,
)
from lewlm.runtime.metal import mlx_acceleration_measurements
from lewlm.runtime.response_cache import RuntimeResponseCache
from lewlm.runtime.scheduler import RuntimeRequestScheduler
from lewlm.pack_registry import PackReport
from lewlm.serving_profiles import (
    SERVING_PROFILE_SETTING_KEYS,
    ServingProfileApplication,
    default_serving_profile_workload_class,
    normalize_serving_profile_workload_class,
    resolve_serving_profile_application,
    serving_profile_supports_workload,
    serving_profile_effective_settings,
    supported_serving_profile_workload_classes,
    serving_profile_workload_class,
    serving_profile_requires_materialization,
)
from lewlm.storage.metadata import MetadataStore
from lewlm.storage import BlockDiskCache
from lewlm.structured_output import JSONSchemaResponseFormat, analyze_structured_output
from lewlm.telemetry.models import (
    FRONTIER_ARCHITECTURE_SUBTYPES as _FRONTIER_ARCHITECTURE_SUBTYPES,
    OPTIMIZATION_CLASS_NAMES,
    AutotuneCandidateSpec as _AutotuneCandidateSpec,
    AutotuneCandidateSummary,
    BenchmarkArtifactReference,
    BenchmarkArtifactSummary,
    BenchmarkRecord,
    BenchmarkRegressionFailure,
    BenchmarkRegressionSummary,
    BenchmarkResult,
    BenchmarkScenarioReport,
    BenchmarkScenarioSample,
    BenchmarkSummary,
    BenchmarkSuiteResult,
    CacheStats,
    CapabilityRuntimeMetrics,
    MeasuredCapabilityRegistrySummary,
    ModelBenchmarkSummary,
    ModelOptimizationDefaults,
    ModelRuntimeMetrics,
    OptimizationDefaultDecision,
    OptimizationDefaultsSummary,
    PerformanceFeatureName,
    PerformanceFeatureStatus,
    RuntimeRequestMetrics,
    RuntimeSchedulerStats,
    RuntimeStats,
    ServingProfileRecommendation,
    TargetPlatformValidation,
    WorkloadOptimizationDefault,
)
from lewlm.telemetry.probes import summarize_measured_capabilities
from lewlm.telemetry.runtime_metrics import RuntimeMetricsRecorder
from lewlm.utils.validation_manifests import (
    apply_external_validation_to_target_matrix,
    load_validation_manifests,
)

if TYPE_CHECKING:
    from lewlm.core.bootstrap import LewLMServices
    from lewlm.routing.service import ModelRouter
    from lewlm.core.chat import ChatOrchestrator
    from lewlm.core.multimodal import MultimodalOrchestrator
from lewlm.core.serving_core import ServingCoreSnapshot


BenchmarkResponseT = TypeVar("BenchmarkResponseT", GenerateResponse, EmbeddingResponse, RerankResponse)

_DETERMINISTIC_CACHE_CAPABILITY_NAMES = frozenset(
    {
        CapabilityName.EMBEDDINGS.value,
        CapabilityName.RERANK.value,
        CapabilityName.AUDIO_TRANSCRIPTION.value,
        CapabilityName.AUDIO_SPEECH.value,
    },
)

_MEASURED_CACHE_REUSE_SCENARIOS = frozenset(
    {
        "repeated_prefix",
        "warm_chat_cache",
        "multimodal_encoder_reuse",
        "multimodal_reuse",
    },
)

_BENCHMARK_CONSTRAINED_DECODING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "const": "ok"},
        "request_kind": {"type": "string", "const": "benchmark_probe"},
    },
    "required": ["status", "request_kind"],
    "additionalProperties": False,
}

class TelemetryService:
    """Expose cache stats, runtime stats, and lightweight benchmarks."""

    def __init__(
        self,
        *,
        settings: LewLMSettings,
        metadata_store: MetadataStore,
        event_bus: EventBus,
        runtime_catalog: RuntimeCatalog,
        model_router: ModelRouter,
        conversion_service: ConversionService,
        runtime_request_scheduler: RuntimeRequestScheduler,
        model_load_scheduler: RuntimeRequestScheduler,
        runtime_metrics_recorder: RuntimeMetricsRecorder,
        block_disk_cache: BlockDiskCache,
        runtime_response_cache: RuntimeResponseCache,
        chat_orchestrator: ChatOrchestrator,
        multimodal_orchestrator: MultimodalOrchestrator,
        cluster_service: Any | None = None,
        service_factory: Callable[[LewLMSettings], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.metadata_store = metadata_store
        self.event_bus = event_bus
        self.runtime_catalog = runtime_catalog
        self.model_router = model_router
        self.conversion_service = conversion_service
        self.runtime_request_scheduler = runtime_request_scheduler
        self.model_load_scheduler = model_load_scheduler
        self.runtime_metrics_recorder = runtime_metrics_recorder
        self.block_disk_cache = block_disk_cache
        self.runtime_response_cache = runtime_response_cache
        self.chat_orchestrator = chat_orchestrator
        self.multimodal_orchestrator = multimodal_orchestrator
        self.cluster_service = cluster_service
        self.service_factory = service_factory

    def cache_stats(self) -> CacheStats:
        conversion_stats = self.conversion_service.cache_stats()
        runtime_cache_stats = self.runtime_response_cache.cache_stats()
        block_cache_stats = self.block_disk_cache.stats()
        runtime_snapshots = self.runtime_catalog.performance_snapshot()
        base_stats = CacheStats.model_validate(
            {
                **conversion_stats,
                "file_count": int(conversion_stats.get("file_count", 0)) + block_cache_stats["block_cache_count"],
                "total_size_bytes": int(conversion_stats.get("total_size_bytes", 0)) + block_cache_stats["block_cache_bytes"],
                "cache_hits": (
                    int(conversion_stats.get("cache_hits", 0))
                    + runtime_cache_stats["runtime_cache_hits"]
                    + block_cache_stats["block_cache_hits"]
                ),
                "cache_misses": (
                    int(conversion_stats.get("cache_misses", 0))
                    + runtime_cache_stats["runtime_cache_misses"]
                    + block_cache_stats["block_cache_misses"]
                ),
                "conversion_cache_hits": int(conversion_stats.get("cache_hits", 0)),
                "conversion_cache_misses": int(conversion_stats.get("cache_misses", 0)),
                **runtime_cache_stats,
                **block_cache_stats,
            },
        )
        return base_stats.model_copy(
            update={
                "performance_features": self._cache_performance_features(
                    cache_stats=base_stats,
                    runtime_snapshots=runtime_snapshots,
                ),
            },
        )

    async def runtime_stats(self) -> RuntimeStats:
        runtime_health = await self.runtime_catalog.health_snapshot()
        loaded_models = sorted(
            {
                model_id
                for runtime in runtime_health
                for model_id in runtime.get("loaded_model_ids", [])
                if isinstance(model_id, str)
            },
        )
        manifests = self.model_router.model_registry.list_manifests()
        validation_manifests = load_validation_manifests(self.settings.validation_manifest_paths)
        request_scheduler = RuntimeSchedulerStats.model_validate(self.runtime_request_scheduler.snapshot())
        load_scheduler = RuntimeSchedulerStats.model_validate(self.model_load_scheduler.snapshot())
        request_metrics = RuntimeRequestMetrics.model_validate(self.runtime_metrics_recorder.snapshot())
        cache_stats = self.cache_stats()
        optimization_defaults = self.optimization_defaults(manifests=manifests)
        measured_registry = self.measured_capability_registry(manifests=manifests)
        return RuntimeStats(
            platform=self.runtime_catalog.host_platform_snapshot(),
            readiness=self.model_router.capability_readiness_summary(),
            runtime_policy=self.settings.runtime_policy,
            request_max_bytes=self.settings.request_max_bytes,
            api_key_required=self.settings.api_key_required,
            active_sessions=self.event_bus.subscriber_count,
            queue_depth=self.conversion_service.queue_depth(),
            active_jobs=self.conversion_service.active_job_count(),
            current_loaded_models=loaded_models,
            runtime_packs=(
                self.runtime_catalog.pack_registry.runtime_pack_reports()
                if self.runtime_catalog.pack_registry is not None
                else []
            ),
            feature_packs=(
                self.runtime_catalog.pack_registry.feature_pack_reports()
                if self.runtime_catalog.pack_registry is not None
                else []
            ),
            runtimes=runtime_health,
            serving_core=self.chat_orchestrator.serving_core.snapshot(),
            request_scheduler=request_scheduler,
            load_scheduler=load_scheduler,
            request_metrics=request_metrics,
            benchmark_summary=self.benchmark_summary(),
            measured_capability_registry=measured_registry,
            validation_manifest_count=len(validation_manifests),
            target_platforms=[
                TargetPlatformValidation.model_validate(payload)
                for payload in apply_external_validation_to_target_matrix(
                    self.runtime_catalog.target_platform_matrix(manifests),
                    local_manifests=manifests,
                    validation_manifests=validation_manifests,
                )
            ],
            cluster=(
                self.cluster_service.status().model_dump(mode="json")
                if self.cluster_service is not None
                else None
            ),
            performance_features=self._performance_features(
                runtime_health=runtime_health,
                request_scheduler=request_scheduler,
                load_scheduler=load_scheduler,
                request_metrics=request_metrics,
                cache_stats=cache_stats,
            ),
            optimization_defaults=optimization_defaults,
        )

    def optimization_defaults(
        self,
        *,
        manifests: Sequence[ModelManifest] | None = None,
    ) -> OptimizationDefaultsSummary:
        candidate_manifests = list(manifests) if manifests is not None else self.model_router.model_registry.list_manifests()
        models = [
            item
            for manifest in candidate_manifests
            if (item := self._model_optimization_defaults(manifest)) is not None
        ]
        resolved_classes = [
            class_name
            for class_name in OPTIMIZATION_CLASS_NAMES
            if models
            and all(model.decisions[class_name].status != "missing" for model in models)
        ]
        benchmark_backed_classes = [
            class_name
            for class_name in OPTIMIZATION_CLASS_NAMES
            if any(model.decisions[class_name].benchmark_backed for model in models)
        ]
        resolved_model_count = sum(1 for model in models if model.resolved)
        notes: list[str] = []
        if not models:
            notes.append("No runnable chat-capable models are currently available for optimization-default inspection.")
        elif resolved_model_count == len(models):
            notes.append("Each runnable chat model now exposes an adopted, rejected, deferred, or not-applicable decision for every tracked optimization class.")
        else:
            notes.append("Some runnable chat models still have unresolved optimization classes.")
        multimodal_default_count = sum(1 for model in models if len(model.workload_defaults) > 1)
        if multimodal_default_count:
            notes.append(
                f"{multimodal_default_count} runnable model(s) now also expose workload-specific multimodal default paths."
            )
        return OptimizationDefaultsSummary(
            host_platform=self.runtime_catalog.host_platform_snapshot(),
            model_count=len(models),
            resolved_model_count=resolved_model_count,
            unresolved_model_count=max(len(models) - resolved_model_count, 0),
            resolved_classes=resolved_classes,
            benchmark_backed_classes=benchmark_backed_classes,
            complete=bool(models) and resolved_model_count == len(models),
            notes=notes,
            models=models,
        )

    def _model_optimization_defaults(self, manifest: ModelManifest) -> ModelOptimizationDefaults | None:
        if manifest.conversion_status != ConversionStatus.RUNNABLE:
            return None
        host_platform = self.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
        workload_class = default_serving_profile_workload_class(manifest=manifest)
        workload_defaults = self._workload_optimization_defaults(
            manifest=manifest,
            host_platform=host_platform,
        )
        default_workload = next(
            (
                item
                for item in workload_defaults
                if item.workload_class == workload_class and item.runtime is not None
            ),
            next((item for item in workload_defaults if item.runtime is not None), None),
        )
        if default_workload is None or default_workload.runtime is None:
            return None
        workload_class = default_workload.workload_class
        _, runtime, _ = self.model_router.route_chat(
            manifest.model_id,
            messages=self._benchmark_workload_messages(
                prompt="Optimization default probe",
                workload_class=default_workload.workload_class,
            ),
            max_tokens=32,
        )
        runtime_features = runtime.performance_feature_snapshot()
        compatible_runtimes, compatibility_alternatives = self.runtime_catalog.compatible_runtimes(
            manifest,
            capability=CapabilityName.CHAT,
        )
        profile_payload = self.metadata_store.get_serving_profile(
            model_id=manifest.model_id,
            capability=CapabilityName.CHAT.value,
            host_platform=host_platform,
            runtime_name=runtime.name,
            workload_class=workload_class,
        )
        serving_profile = resolve_serving_profile_application(
            settings=self.settings,
            metadata_store=self.metadata_store,
            host_platform=host_platform,
            runtime=runtime,
            model_id=manifest.model_id,
            request_capability=CapabilityName.CHAT,
            apply_serving_profile=True,
            workload_class=workload_class,
        )
        profile_artifact = self._profile_artifact_payload(profile_payload)
        runtime_preference = self.metadata_store.get_runtime_preference(
            model_id=manifest.model_id,
            capability=CapabilityName.CHAT.value,
            host_platform=host_platform,
        )
        speculation_preference = self._speculation_preference(model_id=manifest.model_id, runtime_name=runtime.name)
        decisions = {
            "runtime_selection": self._runtime_selection_decision(
                manifest=manifest,
                runtime=runtime,
                compatible_runtimes=compatible_runtimes,
                compatibility_alternatives=compatibility_alternatives,
                runtime_preference=runtime_preference,
            ),
            "continuous_batching": self._continuous_batching_decision(
                runtime=runtime,
                serving_profile=serving_profile,
                profile_payload=profile_payload,
            ),
            "tiered_kv_cache": self._tiered_kv_cache_decision(
                runtime=runtime,
                runtime_features=runtime_features,
                serving_profile=serving_profile,
                profile_payload=profile_payload,
            ),
            "speculation": self._speculation_decision(
                runtime=runtime,
                runtime_features=runtime_features,
                profile_payload=profile_payload,
                speculation_preference=speculation_preference,
            ),
            "kernel_acceleration": self._kernel_acceleration_decision(
                runtime=runtime,
                runtime_features=runtime_features,
                serving_profile=serving_profile,
                profile_payload=profile_payload,
            ),
            "precision_profile": self._precision_profile_decision(
                manifest=manifest,
                serving_profile=serving_profile,
                profile_payload=profile_payload,
            ),
            "frontier_execution": self._frontier_execution_decision(
                manifest=manifest,
                runtime_features=runtime_features,
                profile_artifact=profile_artifact,
            ),
            "multimodal_default_selection": self._multimodal_default_selection_decision(
                manifest=manifest,
                workload_defaults=workload_defaults,
            ),
        }
        unresolved_classes = [
            class_name
            for class_name, decision in decisions.items()
            if decision.status == "missing"
        ]
        return ModelOptimizationDefaults(
            model_id=manifest.model_id,
            display_name=manifest.display_name,
            capability=CapabilityName.CHAT.value,
            runtime=runtime.name,
            runtime_affinity=runtime.affinity.value,
            profile_id=_string_or_none((profile_payload or {}).get("profile_id")) if isinstance(profile_payload, dict) else None,
            default_workload_class=workload_class,
            workload_defaults=workload_defaults,
            resolved=not unresolved_classes,
            resolved_class_count=len(decisions) - len(unresolved_classes),
            unresolved_classes=unresolved_classes,
            decisions=decisions,
        )

    def _workload_optimization_defaults(
        self,
        *,
        manifest: ModelManifest,
        host_platform: dict[str, Any],
    ) -> list[WorkloadOptimizationDefault]:
        return [
            self._workload_optimization_default(
                manifest=manifest,
                host_platform=host_platform,
                workload_class=workload_class,
            )
            for workload_class in supported_serving_profile_workload_classes(manifest=manifest)
        ]

    def _workload_optimization_default(
        self,
        *,
        manifest: ModelManifest,
        host_platform: dict[str, Any],
        workload_class: str,
    ) -> WorkloadOptimizationDefault:
        messages = self._benchmark_workload_messages(prompt="Optimization default probe", workload_class=workload_class)
        try:
            _, runtime, routing = self.model_router.route_chat(
                manifest.model_id,
                messages=messages,
                max_tokens=32,
            )
        except RoutingError as exc:
            return WorkloadOptimizationDefault(
                workload_class=workload_class,
                reason=str(exc),
            )
        serving_profile = resolve_serving_profile_application(
            settings=self.settings,
            metadata_store=self.metadata_store,
            host_platform=host_platform,
            runtime=runtime,
            model_id=manifest.model_id,
            request_capability=CapabilityName.CHAT,
            apply_serving_profile=True,
            workload_class=workload_class,
        )
        return WorkloadOptimizationDefault(
            workload_class=workload_class,
            runtime=runtime.name,
            runtime_affinity=runtime.affinity.value,
            request_modality=(
                routing.request_modality.value
                if routing.request_modality is not None
                else None
            ),
            modality_path=(
                routing.modality_path.value
                if routing.modality_path is not None
                else None
            ),
            profile_id=serving_profile.profile_id,
            profile_status=serving_profile.status,
            benchmark_backed=serving_profile.status == "selected",
            reason=(
                serving_profile.reason
                if serving_profile.status == "selected"
                else f"{routing.reason} {serving_profile.reason}".strip()
            ),
            recommendation_reason=serving_profile.recommendation_reason,
        )

    def _runtime_selection_decision(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        compatible_runtimes: Sequence[RuntimeContract],
        compatibility_alternatives: Sequence[str],
        runtime_preference: dict[str, Any] | None,
    ) -> OptimizationDefaultDecision:
        assessment = assess_runtime_preference(runtime_preference)
        if assessment is not None:
            selected_runtime = assessment.selected_runtime_name or runtime.name
            effective_runtime = assessment.effective_runtime_name or assessment.effective_runtime_affinity or runtime.name
            baseline_runtime = assessment.baseline_runtime_name
            metric = assessment.primary_metric
            selected_metric_value = assessment.selected_metric_value
            baseline_metric_value = assessment.baseline_metric_value
            comparison_detail = ""
            if metric is not None and selected_metric_value is not None and baseline_metric_value is not None:
                comparison_detail = f" ({metric}: {selected_metric_value} vs {baseline_metric_value})"
            if not assessment.adopted:
                return OptimizationDefaultDecision(
                    status="rejected",
                    reason=(
                        f"Measured routing evidence did not adopt `{selected_runtime}` for `{manifest.display_name}`. "
                        f"LewLM keeps `{effective_runtime}` because {assessment.downgrade_reason}."
                    ),
                    benchmark_backed=True,
                    source=assessment.source or "runtime_preference",
                    metrics=_compact_metrics(
                        selected_runtime_name=selected_runtime,
                        selected_runtime_affinity=assessment.selected_runtime_affinity,
                        effective_runtime_name=assessment.effective_runtime_name,
                        effective_runtime_affinity=assessment.effective_runtime_affinity,
                        baseline_runtime_name=baseline_runtime,
                        baseline_runtime_affinity=assessment.baseline_runtime_affinity,
                        primary_metric=metric,
                        selected_metric_value=selected_metric_value,
                        baseline_metric_value=baseline_metric_value,
                        degraded_features=list(assessment.degraded_features),
                        rejected_features=list(assessment.rejected_features),
                    ),
                    notes=list(assessment.notes),
                )
            return OptimizationDefaultDecision(
                status="adopted",
                reason=(
                    f"Benchmarks selected `{selected_runtime}` as the default runtime for `{manifest.display_name}`"
                    + (
                        f" over `{baseline_runtime}`{comparison_detail}."
                        if baseline_runtime is not None
                        else f"{comparison_detail or '.'}"
                    )
                ),
                benchmark_backed=True,
                source=assessment.source or "runtime_preference",
                metrics=_compact_metrics(
                    selected_runtime_name=selected_runtime,
                    selected_runtime_affinity=assessment.selected_runtime_affinity,
                    baseline_runtime_name=baseline_runtime,
                    baseline_runtime_affinity=assessment.baseline_runtime_affinity,
                    primary_metric=metric,
                    selected_metric_value=selected_metric_value,
                    baseline_metric_value=baseline_metric_value,
                ),
                notes=list(assessment.notes),
            )
        if len(compatible_runtimes) <= 1:
            return OptimizationDefaultDecision(
                status="rejected",
                reason=(
                    f"No benchmark-worthy alternate runtime is currently available for `{manifest.display_name}` on this host, "
                    f"so LewLM keeps `{runtime.name}` as the safe default."
                ),
                source="runtime_catalog",
                metrics=_compact_metrics(
                    selected_runtime_name=runtime.name,
                    compatible_runtime_count=len(compatible_runtimes),
                    alternatives=list(compatibility_alternatives),
                ),
            )
        return OptimizationDefaultDecision(
            status="deferred",
            reason=(
                f"Multiple compatible runtimes exist for `{manifest.display_name}`, but no compare-runtime benchmark "
                "preference is persisted for this host/model pair yet."
            ),
            source="runtime_catalog",
            metrics=_compact_metrics(
                selected_runtime_name=runtime.name,
                compatible_runtime_names=[candidate.name for candidate in compatible_runtimes],
                alternatives=list(compatibility_alternatives),
            ),
        )

    def _continuous_batching_decision(
        self,
        *,
        runtime: RuntimeContract,
        serving_profile: ServingProfileApplication,
        profile_payload: dict[str, Any] | None,
    ) -> OptimizationDefaultDecision:
        batching_supported = runtime.supports_continuous_batching(CapabilityName.CHAT)
        if serving_profile.status == "runtime_mismatch":
            return OptimizationDefaultDecision(
                status="deferred",
                reason="The persisted serving profile targets a different runtime, so batching defaults should be re-benchmarked for the current route.",
                source="serving_profile",
                metrics=_compact_metrics(runtime=runtime.name, profile_runtime=_string_or_none((profile_payload or {}).get("runtime"))),
            )
        if not batching_supported:
            return OptimizationDefaultDecision(
                status="rejected",
                reason=f"Runtime `{runtime.name}` does not advertise continuous batching for chat requests on this host.",
                source="runtime_features",
            )
        if profile_payload is None:
            return OptimizationDefaultDecision(
                status="deferred",
                reason="Continuous batching is supported, but no benchmark-backed serving profile has been recorded for this model on this host.",
                source="serving_profile",
            )
        effective_settings = serving_profile.effective_settings or {}
        batch_window_ms = _coerce_int(effective_settings.get("continuous_batch_window_milliseconds"))
        max_batch_size = _coerce_int(effective_settings.get("continuous_batch_max_batch_size"))
        candidate_names = self._profile_candidate_names(profile_payload)
        if max_batch_size > 1 and batch_window_ms > 1:
            return OptimizationDefaultDecision(
                status="adopted",
                reason="Autotune kept continuous batching enabled as the default path for this model/runtime pair.",
                benchmark_backed=True,
                source="serving_profile",
                metrics=_compact_metrics(
                    continuous_batch_window_milliseconds=batch_window_ms,
                    continuous_batch_max_batch_size=max_batch_size,
                    candidate_names=sorted(candidate_names),
                ),
            )
        return OptimizationDefaultDecision(
            status="rejected",
            reason="Autotune compared batching candidates and kept the effectively single-request path as the safe default.",
            benchmark_backed=True,
            source="serving_profile",
            metrics=_compact_metrics(
                continuous_batch_window_milliseconds=batch_window_ms,
                continuous_batch_max_batch_size=max_batch_size,
                candidate_names=sorted(candidate_names),
            ),
        )

    def _tiered_kv_cache_decision(
        self,
        *,
        runtime: RuntimeContract,
        runtime_features: dict[str, object],
        serving_profile: ServingProfileApplication,
        profile_payload: dict[str, Any] | None,
    ) -> OptimizationDefaultDecision:
        paged_kv_supported = self._feature_supported(runtime_features, PerformanceFeatureName.PAGED_KV_CACHE)
        persistent_cache_supported = self._feature_supported(runtime_features, PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE)
        kv_quantization_supported = self._feature_supported(runtime_features, PerformanceFeatureName.KV_CACHE_QUANTIZATION)
        if serving_profile.status == "runtime_mismatch":
            return OptimizationDefaultDecision(
                status="deferred",
                reason="The persisted serving profile targets a different runtime, so KV-cache defaults should be re-benchmarked for the current route.",
                source="serving_profile",
            )
        if not paged_kv_supported and not persistent_cache_supported:
            return OptimizationDefaultDecision(
                status="rejected",
                reason=f"Runtime `{runtime.name}` does not advertise tiered KV-cache support on this host.",
                source="runtime_features",
            )
        if profile_payload is None:
            return OptimizationDefaultDecision(
                status="deferred",
                reason="Tiered KV-cache support is available, but no benchmark-backed serving profile has been recorded for this model on this host.",
                source="serving_profile",
            )
        effective_settings = serving_profile.effective_settings or {}
        active_cache_features = [
            item
            for item in _string_list(profile_payload.get("active_cache_features"))
            if item in {"prefix_cache", "persistent_multi_context_cache"}
        ]
        return OptimizationDefaultDecision(
            status="adopted",
            reason="The persisted serving profile keeps LewLM's tiered KV-cache defaults active for this model/runtime pair.",
            benchmark_backed=True,
            source="serving_profile",
            metrics=_compact_metrics(
                paged_kv_cache_supported=paged_kv_supported,
                persistent_multi_context_cache_supported=persistent_cache_supported,
                kv_cache_quantization_supported=kv_quantization_supported,
                kv_cache_page_size=_coerce_int(effective_settings.get("kv_cache_page_size")),
                kv_cache_max_pages=_coerce_int(effective_settings.get("kv_cache_max_pages")),
                kv_cache_quantization_bits=_coerce_int(effective_settings.get("kv_cache_quantization_bits")),
                active_cache_features=active_cache_features,
            ),
        )

    def _speculation_decision(
        self,
        *,
        runtime: RuntimeContract,
        runtime_features: dict[str, object],
        profile_payload: dict[str, Any] | None,
        speculation_preference: SpeculationBenchmarkPreference | None,
    ) -> OptimizationDefaultDecision:
        speculation_supported = self._feature_supported(runtime_features, PerformanceFeatureName.SPECULATIVE_DECODING) or self._feature_supported(
            runtime_features,
            PerformanceFeatureName.PROMPT_LOOKUP_SPECULATION,
        )
        if speculation_preference is not None:
            if speculation_preference.selected_mode is not None:
                return OptimizationDefaultDecision(
                    status="adopted",
                    reason=f"Benchmarks selected `{speculation_preference.selected_mode.value}` as the default speculation mode for this runtime.",
                    benchmark_backed=True,
                    source="speculation_preference",
                    metrics=_compact_metrics(
                        selected_mode=speculation_preference.selected_mode.value,
                        total_seconds=speculation_preference.total_seconds,
                        acceptance_rate=speculation_preference.acceptance_rate,
                        rollback_tokens=speculation_preference.rollback_tokens,
                        verified_tokens=speculation_preference.verified_tokens,
                        fallback_count=speculation_preference.fallback_count,
                    ),
                )
            return OptimizationDefaultDecision(
                status="rejected",
                reason="Benchmarks kept the non-speculative baseline path as the correctness-preserving default for this runtime.",
                benchmark_backed=True,
                source="speculation_preference",
                metrics=_compact_metrics(
                    selected_mode="disabled",
                    total_seconds=speculation_preference.total_seconds,
                    rollback_tokens=speculation_preference.rollback_tokens,
                    verified_tokens=speculation_preference.verified_tokens,
                    fallback_count=speculation_preference.fallback_count,
                ),
            )
        selected_mode = _string_or_none((profile_payload or {}).get("selected_speculation_mode"))
        if selected_mode is not None:
            return OptimizationDefaultDecision(
                status="adopted",
                reason=f"The persisted benchmark profile selected `{selected_mode}` as the default speculation mode.",
                benchmark_backed=True,
                source="serving_profile",
                metrics={"selected_mode": selected_mode},
            )
        if profile_payload is not None:
            return OptimizationDefaultDecision(
                status="rejected",
                reason="The persisted benchmark profile kept the non-speculative baseline path as the safe default.",
                benchmark_backed=True,
                source="serving_profile",
                metrics={"selected_mode": "disabled"},
            )
        if not speculation_supported:
            return OptimizationDefaultDecision(
                status="rejected",
                reason=f"Runtime `{runtime.name}` does not advertise a compatible speculation path for this model on this host.",
                source="runtime_features",
            )
        return OptimizationDefaultDecision(
            status="deferred",
            reason="Speculation support is available, but no benchmark-backed default selection is persisted for this model/runtime pair yet.",
            source="speculation_preference",
        )

    def _kernel_acceleration_decision(
        self,
        *,
        runtime: RuntimeContract,
        runtime_features: dict[str, object],
        serving_profile: ServingProfileApplication,
        profile_payload: dict[str, Any] | None,
    ) -> OptimizationDefaultDecision:
        graph_supported = self._feature_supported(runtime_features, PerformanceFeatureName.GRAPH_COMPILATION)
        kernel_supported = self._feature_supported(runtime_features, PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION)
        if serving_profile.status == "runtime_mismatch":
            return OptimizationDefaultDecision(
                status="deferred",
                reason="The persisted serving profile targets a different runtime, so kernel-acceleration defaults should be re-benchmarked for the current route.",
                source="serving_profile",
            )
        if not graph_supported and not kernel_supported:
            return OptimizationDefaultDecision(
                status="rejected",
                reason=f"Runtime `{runtime.name}` does not advertise graph-compilation or attention-kernel acceleration support on this host.",
                source="runtime_features",
            )
        if profile_payload is None:
            return OptimizationDefaultDecision(
                status="deferred",
                reason="Kernel acceleration is available, but no benchmark-backed serving profile has been recorded for this model on this host.",
                source="serving_profile",
            )
        effective_settings = serving_profile.effective_settings or {}
        kernel_mode = _string_or_none(effective_settings.get("mlx_attention_kernel_mode")) or "stock"
        active_kernel_path = _string_or_none(profile_payload.get("active_kernel_path")) or "stock"
        graph_compile_enabled = bool(effective_settings.get("mlx_graph_compile_enabled", False))
        if active_kernel_path != "stock" or graph_compile_enabled or kernel_mode != "stock":
            return OptimizationDefaultDecision(
                status="adopted",
                reason="The persisted benchmark profile selected an accelerated kernel or graph-compiled path as the default.",
                benchmark_backed=True,
                source="serving_profile",
                metrics=_compact_metrics(
                    active_kernel_path=active_kernel_path,
                    mlx_graph_compile_enabled=graph_compile_enabled,
                    mlx_attention_kernel_mode=kernel_mode,
                ),
            )
        return OptimizationDefaultDecision(
            status="rejected",
            reason="Benchmarks compared the available acceleration options and kept the stock kernel path as the safe default.",
            benchmark_backed=True,
            source="serving_profile",
            metrics=_compact_metrics(
                active_kernel_path=active_kernel_path,
                mlx_graph_compile_enabled=graph_compile_enabled,
                mlx_attention_kernel_mode=kernel_mode,
            ),
        )

    def _precision_profile_decision(
        self,
        *,
        manifest: ModelManifest,
        serving_profile: ServingProfileApplication,
        profile_payload: dict[str, Any] | None,
    ) -> OptimizationDefaultDecision:
        quantization_label = _string_or_none((profile_payload or {}).get("quantization_profile")) or quantization_profile_label(
            manifest.quantization_profile,
        )
        if profile_payload is not None and serving_profile.status == "runtime_mismatch":
            return OptimizationDefaultDecision(
                status="deferred",
                reason="A precision-profile benchmark exists, but the persisted serving profile targets a different runtime than the current default route.",
                source="serving_profile",
                metrics=_compact_metrics(quantization_profile=quantization_label, artifact_quantization=manifest.quantization),
            )
        if profile_payload is not None and quantization_label is not None:
            return OptimizationDefaultDecision(
                status="adopted",
                reason=f"The persisted benchmark profile keeps `{quantization_label}` as the default precision profile for this host/model pair.",
                benchmark_backed=True,
                source="serving_profile",
                metrics=_compact_metrics(
                    quantization_profile=quantization_label,
                    artifact_quantization=manifest.quantization,
                ),
            )
        if profile_payload is not None:
            return OptimizationDefaultDecision(
                status="rejected",
                reason="The benchmarked default keeps the discovered artifact precision as-is without selecting an alternate precision profile.",
                benchmark_backed=True,
                source="serving_profile",
                metrics=_compact_metrics(
                    quantization_profile=quantization_label,
                    artifact_quantization=manifest.quantization,
                ),
            )
        if manifest.quantization_profile is not None:
            return OptimizationDefaultDecision(
                status="deferred",
                reason="This artifact carries an explicit precision profile, but no host-backed serving default has been persisted for it yet.",
                source="manifest",
                metrics=_compact_metrics(
                    quantization_profile=quantization_profile_label(manifest.quantization_profile),
                    artifact_quantization=manifest.quantization,
                ),
            )
        return OptimizationDefaultDecision(
            status="rejected",
            reason="No alternate precision-profile benchmark is recorded for this model on this host, so LewLM keeps the discovered artifact precision as the default.",
            source="manifest",
            metrics=_compact_metrics(
                quantization_profile=quantization_label,
                artifact_quantization=manifest.quantization,
            ),
        )

    def _frontier_execution_decision(
        self,
        *,
        manifest: ModelManifest,
        runtime_features: dict[str, object],
        profile_artifact: dict[str, Any] | None,
    ) -> OptimizationDefaultDecision:
        subtype = manifest.architecture_subtype.value
        if subtype not in _FRONTIER_ARCHITECTURE_SUBTYPES:
            return OptimizationDefaultDecision(
                status="not_applicable",
                reason="This model does not use a frontier SSM or MoE execution path.",
                source="manifest",
            )
        scenario = self._artifact_scenario(profile_artifact, "frontier_architecture_modes")
        scenario_metrics = scenario.get("metrics", {}) if isinstance(scenario, dict) else {}
        scenario_sample_metrics = self._artifact_scenario_sample_metrics(profile_artifact, "frontier_architecture_modes")
        planning_only = bool(
            scenario_metrics.get("planning_only")
            if isinstance(scenario_metrics, dict)
            else False,
        )
        if scenario_sample_metrics:
            planning_only = bool(scenario_sample_metrics.get("planning_only", planning_only))
        if isinstance(scenario, dict):
            if not planning_only:
                return OptimizationDefaultDecision(
                    status="adopted",
                    reason="Benchmark artifacts recorded realized frontier execution metrics for this model, so LewLM can keep the frontier path visible as the default.",
                    benchmark_backed=True,
                    source="benchmark_artifact",
                    metrics=_compact_metrics(
                        architecture_subtype=subtype,
                        execution_path=scenario_sample_metrics.get("execution_path") if scenario_sample_metrics else None,
                        effective_loaded_memory_mb=scenario_sample_metrics.get("effective_loaded_memory_mb") if scenario_sample_metrics else None,
                        resident_expert_count=scenario_sample_metrics.get("resident_expert_count") if scenario_sample_metrics else None,
                        state_cache_bytes=scenario_sample_metrics.get("state_cache_bytes") if scenario_sample_metrics else None,
                    ),
                )
            return OptimizationDefaultDecision(
                status="deferred",
                reason="Benchmark artifacts only captured frontier planning metadata for this model; keep a non-claiming default until realized execution metrics land.",
                benchmark_backed=True,
                source="benchmark_artifact",
                metrics=_compact_metrics(
                    architecture_subtype=subtype,
                    planning_only=True,
                    execution_path=scenario_sample_metrics.get("execution_path") if scenario_sample_metrics else None,
                ),
            )
        plan = build_frontier_serving_plan(manifest=manifest, settings=self.settings)
        if not isinstance(plan, dict):
            return OptimizationDefaultDecision(
                status="missing",
                reason="No frontier execution summary could be derived for this frontier-class model.",
                source="frontier_plan",
            )
        if bool(plan.get("planning_only", True)):
            return OptimizationDefaultDecision(
                status="deferred",
                reason="This model is frontier-class, but the current host only exposes planning metadata rather than realized execution metrics.",
                source="frontier_plan",
                metrics=_compact_metrics(
                    architecture_subtype=subtype,
                    feature=(
                        PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING.value
                        if subtype in {"moe", "hybrid_moe"}
                        else PerformanceFeatureName.HYBRID_SSM_ROUTING.value
                    ),
                    runtime_feature_supported=self._feature_supported(
                        runtime_features,
                        PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING
                        if subtype in {"moe", "hybrid_moe"}
                        else PerformanceFeatureName.HYBRID_SSM_ROUTING,
                    ),
                    planning_only=True,
                ),
            )
        return OptimizationDefaultDecision(
            status="adopted",
            reason="The current host exposes a realized frontier execution plan for this model.",
            source="frontier_plan",
            metrics=_compact_metrics(
                architecture_subtype=subtype,
                execution_path=plan.get("execution_path"),
                planning_only=bool(plan.get("planning_only", False)),
            ),
        )

    def _multimodal_default_selection_decision(
        self,
        *,
        manifest: ModelManifest,
        workload_defaults: Sequence[WorkloadOptimizationDefault],
    ) -> OptimizationDefaultDecision:
        if len(workload_defaults) <= 1:
            return OptimizationDefaultDecision(
                status="not_applicable",
                reason="This model does not expose multiple multimodal workload-specific default paths.",
                source="manifest",
            )
        benchmark_backed_workloads = [
            item.workload_class
            for item in workload_defaults
            if item.benchmark_backed
        ]
        uncovered_workloads = [
            item.workload_class
            for item in workload_defaults
            if not item.benchmark_backed
        ]
        metrics = _compact_metrics(
            supported_workload_count=len(workload_defaults),
            benchmark_backed_workload_count=len(benchmark_backed_workloads),
            benchmark_backed_workloads=benchmark_backed_workloads,
            uncovered_workloads=uncovered_workloads,
            routed_workloads=[
                item.workload_class
                for item in workload_defaults
                if item.runtime is not None
            ],
        )
        if not uncovered_workloads:
            return OptimizationDefaultDecision(
                status="adopted",
                reason=(
                    f"Benchmark-backed serving defaults now cover every supported multimodal workload class for "
                    f"`{manifest.display_name}` on this host."
                ),
                benchmark_backed=True,
                source="serving_profile",
                metrics=metrics,
                notes=[
                    ", ".join(
                        f"{item.workload_class}->{item.runtime or 'unroutable'}"
                        for item in workload_defaults
                    )
                ],
            )
        return OptimizationDefaultDecision(
            status="deferred",
            reason=(
                f"Multimodal workload routing defaults are available for `{manifest.display_name}`, but "
                f"benchmark-backed host selections are still missing for: {', '.join(uncovered_workloads)}."
            ),
            source="serving_profile",
            metrics=metrics,
        )

    def _profile_artifact_payload(self, profile_payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(profile_payload, dict):
            return None
        artifact = profile_payload.get("artifact")
        if not isinstance(artifact, dict):
            return None
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            return None
        return self.metadata_store.get_benchmark_artifact(artifact_id)

    def _speculation_preference(self, *, model_id: str, runtime_name: str) -> SpeculationBenchmarkPreference | None:
        payload = self.metadata_store.get_value(
            speculation_benchmark_preference_key(model_id=model_id, runtime_name=runtime_name),
        )
        return parse_speculation_benchmark_preference(payload)

    @staticmethod
    def _profile_candidate_names(profile_payload: dict[str, Any] | None) -> set[str]:
        if not isinstance(profile_payload, dict):
            return set()
        candidate_summaries = profile_payload.get("candidate_summaries")
        if not isinstance(candidate_summaries, list):
            return set()
        return {
            str(item.get("name"))
            for item in candidate_summaries
            if isinstance(item, dict) and item.get("name")
        }

    @staticmethod
    def _artifact_scenario(
        artifact_payload: dict[str, Any] | None,
        scenario_name: str,
    ) -> dict[str, Any] | None:
        if not isinstance(artifact_payload, dict):
            return None
        for scenario in artifact_payload.get("scenarios", []):
            if not isinstance(scenario, dict):
                continue
            if scenario.get("scenario") == scenario_name:
                return scenario
        return None

    def _artifact_scenario_sample_metrics(
        self,
        artifact_payload: dict[str, Any] | None,
        scenario_name: str,
    ) -> dict[str, Any]:
        scenario = self._artifact_scenario(artifact_payload, scenario_name)
        if scenario is None:
            return {}
        samples = scenario.get("samples")
        if not isinstance(samples, list):
            return {}
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            metrics = sample.get("metrics")
            if isinstance(metrics, dict):
                return metrics
        return {}

    async def benchmark(
        self,
        *,
        model_id: str | None,
        prompt: str,
        capability: str = CapabilityName.CHAT.value,
        warmup_run_count: int = 1,
        apply_serving_profile: bool = True,
        workload_class: str | None = None,
    ) -> BenchmarkResult:
        return await self._benchmark_with_artifact(
            model_id=model_id,
            prompt=prompt,
            capability=capability,
            warmup_run_count=warmup_run_count,
            apply_serving_profile=apply_serving_profile,
            workload_class=workload_class,
            include_scenarios=True,
        )

    async def benchmark_lightweight(
        self,
        *,
        model_id: str | None,
        prompt: str,
        capability: str = CapabilityName.CHAT.value,
        warmup_run_count: int = 1,
        apply_serving_profile: bool = True,
        workload_class: str | None = None,
    ) -> BenchmarkResult:
        return await self._benchmark_with_artifact(
            model_id=model_id,
            prompt=prompt,
            capability=capability,
            warmup_run_count=warmup_run_count,
            apply_serving_profile=apply_serving_profile,
            workload_class=workload_class,
            include_scenarios=False,
        )

    async def _benchmark_with_artifact(
        self,
        *,
        model_id: str | None,
        prompt: str,
        capability: str,
        warmup_run_count: int,
        apply_serving_profile: bool,
        workload_class: str | None,
        include_scenarios: bool,
    ) -> BenchmarkResult:
        result = await self._benchmark_model(
            model_id=model_id,
            prompt=prompt,
            capability=capability,
            warmup_run_count=warmup_run_count,
            apply_serving_profile=apply_serving_profile,
            workload_class=workload_class,
        )
        scenarios = (
            await self._benchmark_scenarios(
                capability=capability,
                prompt=prompt,
                model_ids=[result.model_id],
            )
            if include_scenarios
            else []
        )
        artifact, regression = await self._persist_benchmark_artifact(
            source="telemetry.benchmark",
            capability=capability,
            prompt=prompt,
            workload_class=result.workload_class,
            repeat_count=1,
            model_ids=[result.model_id],
            result_payload=result.model_dump(mode="json"),
            scenarios=scenarios,
            benchmark_count=1,
            model_count=1,
        )
        merged_scenarios = self._merge_benchmark_result_scenarios(
            existing=result.scenarios,
            additional=scenarios,
        )
        updated_result = result.model_copy(update={"scenarios": merged_scenarios, "regression": regression, "artifact": artifact})
        self.metadata_store.append_benchmark_record(updated_result.model_dump(mode="json"))
        self._persist_benchmark_probe_records(updated_result)
        return updated_result

    async def autotune(
        self,
        *,
        model_id: str | None,
        prompt: str,
        capability: str = CapabilityName.CHAT.value,
        workload_class: str | None = None,
    ) -> ServingProfileRecommendation:
        if capability != CapabilityName.CHAT.value:
            raise ConfigurationError("Autotuning currently supports chat-capable models only.")
        messages = self._benchmark_workload_messages(prompt=prompt, workload_class=workload_class)
        manifest, runtime, _ = self.model_router.route_chat(
            model_id,
            messages=messages,
            max_tokens=128,
        )
        normalized_workload_class = serving_profile_workload_class(
            messages=messages,
            manifest=manifest,
            workload_class_hint=workload_class,
        )
        self._ensure_manifest_supports_workload(manifest=manifest, workload_class=normalized_workload_class)
        candidate_specs = self._autotune_candidate_specs(runtime=runtime)
        candidate_summaries: list[AutotuneCandidateSummary] = []
        for candidate in candidate_specs:
            benchmark_result = await self._run_autotune_candidate(
                candidate=candidate,
                model_id=manifest.model_id,
                prompt=prompt,
                workload_class=normalized_workload_class,
            )
            candidate_summaries.append(
                self._autotune_candidate_summary(
                    benchmark_result=benchmark_result,
                    manifest=manifest,
                    candidate=candidate,
                ),
            )
        selected_candidate = min(candidate_summaries, key=self._autotune_candidate_sort_key)
        effective_settings = self._serving_profile_effective_settings(selected_candidate.settings_overrides)
        recommended_at = utc_now()
        recommendation = ServingProfileRecommendation(
            profile_id=uuid4().hex,
            model_id=manifest.model_id,
            capability=capability,
            workload_class=normalized_workload_class,
            runtime=selected_candidate.runtime,
            host_platform=self.runtime_catalog.host_platform_snapshot(),
            prompt=prompt,
            recommended_at=recommended_at,
            reason=(
                f"Selected `{selected_candidate.name}` because it produced the lowest measured chat latency for the "
                f"`{normalized_workload_class}` workload after benchmarking the available serving-profile candidates "
                "for this host."
            ),
            settings_overrides=selected_candidate.settings_overrides,
            effective_settings=effective_settings,
            metrics=_compact_metrics(
                total_seconds=selected_candidate.total_seconds,
                load_seconds=selected_candidate.load_seconds,
                generate_seconds=selected_candidate.generate_seconds,
                completion_tokens_per_second=selected_candidate.completion_tokens_per_second,
                continuous_batching_throughput=selected_candidate.continuous_batching_throughput,
                warm_cache_ttft_ratio=selected_candidate.warm_cache_ttft_ratio,
            ),
            quantization_profile=selected_candidate.quantization_profile,
            selected_speculation_mode=selected_candidate.selected_speculation_mode,
            active_kernel_path=selected_candidate.active_kernel_path,
            active_cache_features=selected_candidate.active_cache_features,
            artifact=selected_candidate.artifact,
            notes=(
                [
                    "Profile recommendations are benchmark-backed and stored per host/model/runtime/workload tuple.",
                    "The current selection objective prefers the lowest measured end-to-end latency and uses throughput as a tie-breaker.",
                ]
                + selected_candidate.notes
            ),
            candidate_summaries=candidate_summaries,
        )
        self.metadata_store.upsert_serving_profile(
            model_id=manifest.model_id,
            capability=capability,
            host_platform=recommendation.host_platform.model_dump(mode="json"),
            runtime_name=selected_candidate.runtime,
            workload_class=normalized_workload_class,
            payload=recommendation.model_dump(mode="json"),
        )
        await self.event_bus.publish(
            StreamEvent(
                type=EventType.AUTOTUNE_COMPLETED,
                scope=EventScope.SYSTEM,
                payload={
                    "model_id": recommendation.model_id,
                    "capability": recommendation.capability,
                    "workload_class": recommendation.workload_class,
                    "runtime": recommendation.runtime,
                    "profile_id": recommendation.profile_id,
                    "artifact_id": recommendation.artifact.artifact_id if recommendation.artifact is not None else None,
                },
            ),
        )
        return recommendation

    async def benchmark_suite(
        self,
        *,
        prompt: str,
        model_ids: list[str] | None = None,
        capability: str = CapabilityName.CHAT.value,
        repeat_count: int = 1,
        warmup_run_count: int = 1,
        apply_serving_profile: bool = True,
        workload_class: str | None = None,
    ) -> BenchmarkSuiteResult:
        return await self._benchmark_suite_with_artifact(
            prompt=prompt,
            model_ids=model_ids,
            capability=capability,
            repeat_count=repeat_count,
            warmup_run_count=warmup_run_count,
            apply_serving_profile=apply_serving_profile,
            workload_class=workload_class,
            include_scenarios=True,
        )

    async def benchmark_suite_lightweight(
        self,
        *,
        prompt: str,
        model_ids: list[str] | None = None,
        capability: str = CapabilityName.CHAT.value,
        repeat_count: int = 1,
        warmup_run_count: int = 1,
        apply_serving_profile: bool = True,
        workload_class: str | None = None,
    ) -> BenchmarkSuiteResult:
        return await self._benchmark_suite_with_artifact(
            prompt=prompt,
            model_ids=model_ids,
            capability=capability,
            repeat_count=repeat_count,
            warmup_run_count=warmup_run_count,
            apply_serving_profile=apply_serving_profile,
            workload_class=workload_class,
            include_scenarios=False,
        )

    async def _benchmark_suite_with_artifact(
        self,
        *,
        prompt: str,
        model_ids: list[str] | None,
        capability: str,
        repeat_count: int,
        warmup_run_count: int,
        apply_serving_profile: bool,
        workload_class: str | None,
        include_scenarios: bool,
    ) -> BenchmarkSuiteResult:
        if repeat_count < 1:
            raise ConfigurationError("Benchmark suite repeat_count must be at least 1.")
        candidate_model_ids = model_ids or self._benchmark_candidate_model_ids(capability=capability)
        if not candidate_model_ids:
            raise ConfigurationError(f"No runnable {capability} models are available for benchmarking.")
        total_started_at = time.perf_counter()
        results = [
            await self._benchmark_model(
                model_id=candidate_model_id,
                prompt=prompt,
                capability=capability,
                warmup_run_count=warmup_run_count,
                apply_serving_profile=apply_serving_profile,
                workload_class=workload_class,
            )
            for _ in range(repeat_count)
            for candidate_model_id in candidate_model_ids
        ]
        total_elapsed_seconds = time.perf_counter() - total_started_at
        total_load_seconds = sum(result.load_seconds for result in results)
        total_generate_seconds = sum(result.generate_seconds for result in results)
        total_seconds = [result.total_seconds for result in results]
        runtime_health = await self.runtime_catalog.health_snapshot()
        request_scheduler = RuntimeSchedulerStats.model_validate(self.runtime_request_scheduler.snapshot())
        load_scheduler = RuntimeSchedulerStats.model_validate(self.model_load_scheduler.snapshot())
        request_metrics = RuntimeRequestMetrics.model_validate(self.runtime_metrics_recorder.snapshot())
        cache_stats = self.cache_stats()
        suite = BenchmarkSuiteResult(
            capability=capability,
            prompt=prompt,
            workload_class=normalize_serving_profile_workload_class(workload_class),
            repeat_count=repeat_count,
            benchmark_count=len(results),
            model_count=len({result.model_id for result in results}),
            total_load_seconds=round(total_load_seconds, 4),
            total_generate_seconds=round(total_generate_seconds, 4),
            total_elapsed_seconds=round(total_elapsed_seconds, 4),
            average_total_seconds=round(fmean(total_seconds), 4) if total_seconds else None,
            models=self._model_benchmark_summaries(results),
            results=results,
            performance_features=self._performance_features(
                runtime_health=runtime_health,
                request_scheduler=request_scheduler,
                load_scheduler=load_scheduler,
                request_metrics=request_metrics,
                cache_stats=cache_stats,
                capability_focus=capability,
            ),
        )
        suite_model_ids = sorted({result.model_id for result in results})
        scenarios = (
            await self._benchmark_scenarios(
                capability=capability,
                prompt=prompt,
                model_ids=suite_model_ids,
            )
            if include_scenarios and (capability != CapabilityName.CHAT.value or len(suite_model_ids) <= 1)
            else []
        )
        artifact, regression = await self._persist_benchmark_artifact(
            source="telemetry.benchmark_suite",
            capability=capability,
            prompt=prompt,
            workload_class=suite.workload_class,
            repeat_count=repeat_count,
            model_ids=suite_model_ids,
            result_payload=suite.model_dump(mode="json"),
            scenarios=scenarios,
            benchmark_count=suite.benchmark_count,
            model_count=suite.model_count,
        )
        updated_suite = suite.model_copy(update={"scenarios": scenarios, "regression": regression, "artifact": artifact})
        if scenarios and results:
            self._persist_benchmark_probe_records(
                results[0].model_copy(update={"scenarios": scenarios, "regression": regression, "artifact": artifact}),
            )
        return updated_suite

    @staticmethod
    def _merge_benchmark_result_scenarios(
        *,
        existing: Sequence[BenchmarkScenarioReport],
        additional: Sequence[BenchmarkScenarioReport],
    ) -> list[BenchmarkScenarioReport]:
        merged_scenarios = list(existing)
        seen_scenarios = {item.scenario for item in merged_scenarios}
        for scenario in additional:
            if scenario.scenario in seen_scenarios:
                continue
            merged_scenarios.append(scenario)
        return merged_scenarios

    async def _benchmark_model(
        self,
        *,
        model_id: str | None,
        prompt: str,
        capability: str,
        warmup_run_count: int = 1,
        apply_serving_profile: bool = True,
        workload_class: str | None = None,
        serving_profile: ServingProfileApplication | None = None,
    ) -> BenchmarkResult:
        if capability == CapabilityName.CHAT.value:
            resolved_serving_profile = serving_profile or self._resolve_benchmark_serving_profile(
                model_id=model_id,
                prompt=prompt,
                apply_serving_profile=apply_serving_profile,
                workload_class=workload_class,
            )
            if (
                apply_serving_profile
                and serving_profile_requires_materialization(profile=resolved_serving_profile, settings=self.settings)
                and self.service_factory is not None
            ):
                candidate_services = self.service_factory(
                    self.settings.with_updates(**resolved_serving_profile.accepted_settings),
                )
                try:
                    self._sync_benchmark_child_manifests(candidate_services)
                    return await candidate_services.telemetry_service._benchmark_model(
                        model_id=model_id,
                        prompt=prompt,
                        capability=capability,
                        warmup_run_count=warmup_run_count,
                        apply_serving_profile=False,
                        workload_class=workload_class,
                        serving_profile=resolved_serving_profile,
                    )
                finally:
                    await candidate_services.aclose()
            return await self._benchmark_chat_model(
                model_id=model_id,
                prompt=prompt,
                warmup_run_count=warmup_run_count,
                workload_class=workload_class,
                serving_profile=resolved_serving_profile,
            )
        if capability == CapabilityName.EMBEDDINGS.value:
            return await self._benchmark_embeddings_model(
                model_id=model_id,
                prompt=prompt,
                warmup_run_count=warmup_run_count,
            )
        if capability == CapabilityName.RERANK.value:
            return await self._benchmark_rerank_model(
                model_id=model_id,
                prompt=prompt,
                warmup_run_count=warmup_run_count,
            )
        if capability == CapabilityName.AUDIO_TRANSCRIPTION.value:
            return await self._benchmark_audio_transcription_model(
                model_id=model_id,
                prompt=prompt,
                warmup_run_count=warmup_run_count,
            )
        raise ConfigurationError(f"Benchmarking does not support capability `{capability}`.")

    async def _run_autotune_candidate(
        self,
        *,
        candidate: _AutotuneCandidateSpec,
        model_id: str,
        prompt: str,
        workload_class: str | None,
    ) -> BenchmarkResult:
        candidate_prompt = f"{prompt}\n[autotune={hashlib.sha256(candidate.name.encode('utf-8')).hexdigest()[:8]}]"
        if self.service_factory is None:
            return await self.benchmark(
                model_id=model_id,
                prompt=candidate_prompt,
                capability=CapabilityName.CHAT.value,
                apply_serving_profile=False,
                workload_class=workload_class,
            )
        candidate_data_dir = self.settings.data_dir / "autotune-candidates" / hashlib.sha256(
            candidate.name.encode("utf-8"),
        ).hexdigest()[:12]
        candidate_settings = self.settings.with_updates(
            data_dir=candidate_data_dir,
            models_dir=self.settings.models_dir,
            **candidate.settings_overrides,
        )
        candidate_services = self.service_factory(candidate_settings)
        try:
            self._sync_benchmark_child_manifests(
                candidate_services,
                model_dirs=candidate_settings.models_dir,
            )
            return await candidate_services.telemetry_service.benchmark(
                model_id=model_id,
                prompt=candidate_prompt,
                capability=CapabilityName.CHAT.value,
                apply_serving_profile=False,
                workload_class=workload_class,
            )
        finally:
            await candidate_services.aclose()

    def _autotune_candidate_specs(self, *, runtime: RuntimeContract) -> list[_AutotuneCandidateSpec]:
        runtime_features = runtime.performance_feature_snapshot()
        specs: list[_AutotuneCandidateSpec] = [
            _AutotuneCandidateSpec(
                name="baseline",
                settings_overrides={},
                notes=("Benchmark the current serving configuration first.",),
            ),
        ]
        for runtime_policy in ("keep_warm", "balanced", "aggressive_unload"):
            specs.append(
                _AutotuneCandidateSpec(
                    name=f"residency_{runtime_policy}",
                    settings_overrides={"runtime_policy": runtime_policy},
                    notes=(f"Evaluate `{runtime_policy}` model residency behavior.",),
                ),
            )
        specs.extend(
            (
                _AutotuneCandidateSpec(
                    name="batching_disabled",
                    settings_overrides={
                        "continuous_batch_window_milliseconds": 1,
                        "continuous_batch_max_batch_size": 1,
                    },
                    notes=("Disable frontier batching to establish a single-request baseline.",),
                ),
                _AutotuneCandidateSpec(
                    name="batching_balanced",
                    settings_overrides={
                        "continuous_batch_window_milliseconds": max(8, self.settings.continuous_batch_window_milliseconds),
                        "continuous_batch_max_batch_size": max(4, self.settings.continuous_batch_max_batch_size),
                    },
                    notes=("Use a short continuous-batching window aimed at chat serving bursts.",),
                ),
            ),
        )
        if self._feature_supported(runtime_features, PerformanceFeatureName.GRAPH_COMPILATION):
            specs.append(
                _AutotuneCandidateSpec(
                    name="graph_compile_toggled",
                    settings_overrides={"mlx_graph_compile_enabled": not self.settings.mlx_graph_compile_enabled},
                    notes=("Toggle MLX graph compilation and compare the compiled path against the current setting.",),
                ),
            )
        if self._feature_supported(runtime_features, PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION):
            for kernel_mode in ("stock", "flash_attention", "custom_sdpa"):
                specs.append(
                    _AutotuneCandidateSpec(
                        name=f"attention_{kernel_mode}",
                        settings_overrides={"mlx_attention_kernel_mode": kernel_mode},
                        notes=(f"Benchmark the `{kernel_mode}` attention kernel path.",),
                    ),
                )
        if self._feature_supported(runtime_features, PerformanceFeatureName.KV_CACHE_QUANTIZATION):
            for quantization_bits in (None, 4, 8):
                specs.append(
                    _AutotuneCandidateSpec(
                        name=f"kv_quantization_{'off' if quantization_bits is None else quantization_bits}",
                        settings_overrides={"kv_cache_quantization_bits": quantization_bits},
                        notes=("Compare the configured KV-cache quantization level.",),
                    ),
                )
        if self._feature_supported(runtime_features, PerformanceFeatureName.PREFILL_OPTIMIZATION):
            for batch_size in sorted({256, 512, 1024, self.settings.prefill_token_batch_size}):
                specs.append(
                    _AutotuneCandidateSpec(
                        name=f"prefill_{batch_size}",
                        settings_overrides={"prefill_token_batch_size": batch_size},
                        notes=("Sweep the prompt prefill batch size used during long-prefix ingestion.",),
                    ),
                )
        unique_specs: list[_AutotuneCandidateSpec] = []
        seen_signatures: set[str] = set()
        for spec in specs:
            signature = json.dumps(spec.settings_overrides, sort_keys=True)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_specs.append(spec)
        return unique_specs

    def _autotune_candidate_summary(
        self,
        *,
        benchmark_result: BenchmarkResult,
        manifest: ModelManifest,
        candidate: _AutotuneCandidateSpec,
    ) -> AutotuneCandidateSummary:
        return AutotuneCandidateSummary(
            name=candidate.name,
            benchmark_id=benchmark_result.benchmark_id,
            runtime=benchmark_result.runtime,
            settings_overrides=candidate.settings_overrides,
            total_seconds=benchmark_result.total_seconds,
            load_seconds=benchmark_result.load_seconds,
            generate_seconds=benchmark_result.generate_seconds,
            completion_tokens_per_second=benchmark_result.completion_tokens_per_second,
            continuous_batching_throughput=self._scenario_metric_value(
                benchmark_result.scenarios,
                scenario_name="continuous_batching",
                metric_name="throughput_requests_per_second",
            ),
            warm_cache_ttft_ratio=self._scenario_metric_value(
                benchmark_result.scenarios,
                scenario_name="warm_chat_cache",
                metric_name="warm_over_cold_ttft_ratio",
            ),
            selected_speculation_mode=self._selected_speculation_mode_from_scenarios(benchmark_result.scenarios),
            quantization_profile=quantization_profile_label(manifest.quantization_profile) or manifest.quantization,
            active_kernel_path=self._kernel_path_from_performance_features(benchmark_result.performance_features),
            active_cache_features=self._active_cache_features(benchmark_result.performance_features),
            artifact=benchmark_result.artifact,
            notes=list(candidate.notes),
        )

    def _autotune_candidate_sort_key(self, candidate: AutotuneCandidateSummary) -> tuple[float, float, float, float]:
        total_seconds = candidate.total_seconds if candidate.total_seconds > 0 else float("inf")
        throughput = candidate.continuous_batching_throughput or 0.0
        token_throughput = candidate.completion_tokens_per_second or 0.0
        warm_cache_ratio = candidate.warm_cache_ttft_ratio if candidate.warm_cache_ttft_ratio is not None else float("inf")
        return (total_seconds, -throughput, -token_throughput, warm_cache_ratio)

    def _serving_profile_effective_settings(
        self,
        settings_overrides: dict[str, int | float | str | bool | None],
    ) -> dict[str, int | float | str | bool | None]:
        return serving_profile_effective_settings(self.settings.with_updates(**settings_overrides))

    def _feature_supported(self, runtime_features: dict[str, object], feature: PerformanceFeatureName) -> bool:
        payload = runtime_features.get(feature.value)
        return isinstance(payload, dict) and bool(payload.get("supported"))

    def _scenario_metric_value(
        self,
        scenarios: list[BenchmarkScenarioReport],
        *,
        scenario_name: str,
        metric_name: str,
    ) -> float | None:
        for scenario in scenarios:
            if scenario.scenario != scenario_name:
                continue
            value = scenario.metrics.get(metric_name)
            if isinstance(value, int | float):
                return float(value)
        return None

    def _selected_speculation_mode_from_scenarios(self, scenarios: list[BenchmarkScenarioReport]) -> str | None:
        for scenario in scenarios:
            if scenario.scenario != "speculation_selection":
                continue
            selected_mode = scenario.metrics.get("selected_mode")
            if isinstance(selected_mode, str) and selected_mode:
                return selected_mode
        return None

    def _kernel_path_from_performance_features(
        self,
        performance_features: list[PerformanceFeatureStatus],
    ) -> str | None:
        for feature in performance_features:
            if feature.feature != PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION:
                continue
            kernel_path = feature.metrics.get("last_kernel_path")
            if isinstance(kernel_path, str) and kernel_path:
                return kernel_path
        return None

    def _active_cache_features(
        self,
        performance_features: list[PerformanceFeatureStatus],
    ) -> list[str]:
        cache_features = {
            PerformanceFeatureName.PREFIX_CACHE,
            PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE,
            PerformanceFeatureName.PAGED_KV_CACHE,
            PerformanceFeatureName.KV_CACHE_QUANTIZATION,
            PerformanceFeatureName.DISK_BACKED_CACHE,
            PerformanceFeatureName.BLOCK_DISK_CACHE,
        }
        return [
            feature.feature.value
            for feature in performance_features
            if feature.feature in cache_features and (feature.active or feature.supported)
        ]

    def _benchmark_workload_messages(
        self,
        *,
        prompt: str,
        workload_class: str | None,
    ) -> list[GenerateMessage]:
        normalized_workload_class = normalize_serving_profile_workload_class(workload_class)
        if normalized_workload_class is None or normalized_workload_class in {"text_only", "text_only_multimodal"}:
            return [GenerateMessage(role="user", content=prompt)]
        assets = _ensure_benchmark_multimodal_assets(self.settings.benchmarks_dir)
        attachment_metadata = {"serving_profile_workload_class": normalized_workload_class}
        if normalized_workload_class == "single_image":
            attachments = [
                GenerateAttachment(
                    attachment_type="image",
                    name=assets["image"].name,
                    source_path=str(assets["image"]),
                    media_type="image/png",
                    metadata=attachment_metadata,
                ),
            ]
        elif normalized_workload_class == "repeated_image":
            attachments = [
                GenerateAttachment(
                    attachment_type="image",
                    name=assets["image"].name,
                    source_path=str(assets["image"]),
                    media_type="image/png",
                    metadata={**attachment_metadata, "repeated_image": True},
                ),
                GenerateAttachment(
                    attachment_type="image",
                    name=assets["image"].name,
                    source_path=str(assets["image"]),
                    media_type="image/png",
                    metadata={**attachment_metadata, "repeated_image": True},
                ),
            ]
        elif normalized_workload_class == "frame_bundle_video":
            attachments = [
                GenerateAttachment(
                    attachment_type="image",
                    name=assets["frame_bundle"].name,
                    source_path=str(assets["frame_bundle"]),
                    metadata={**attachment_metadata, "source_kind": "frame_bundle", "frame_count": 2},
                ),
            ]
        elif normalized_workload_class == "audio_conditioned":
            attachments = [
                GenerateAttachment(
                    attachment_type="audio",
                    name=assets["audio"].name,
                    source_path=str(assets["audio"]),
                    media_type="audio/wav",
                    metadata=attachment_metadata,
                ),
            ]
        else:
            return [GenerateMessage(role="user", content=prompt)]
        return [GenerateMessage(role="user", content=prompt, attachments=attachments)]

    def _ensure_manifest_supports_workload(
        self,
        *,
        manifest: ModelManifest,
        workload_class: str | None,
    ) -> None:
        normalized_workload_class = normalize_serving_profile_workload_class(workload_class)
        if normalized_workload_class is None or serving_profile_supports_workload(
            manifest=manifest,
            workload_class=normalized_workload_class,
        ):
            return
        modalities = ", ".join(modality.value for modality in manifest.modality)
        raise ConfigurationError(
            f"Model `{manifest.model_id}` does not support the `{normalized_workload_class}` benchmark workload.",
            details={"model_id": manifest.model_id, "workload_class": normalized_workload_class, "modality": modalities},
        )

    def _resolve_benchmark_serving_profile(
        self,
        *,
        model_id: str | None,
        prompt: str,
        apply_serving_profile: bool,
        workload_class: str | None,
    ) -> ServingProfileApplication:
        messages = self._benchmark_workload_messages(prompt=prompt, workload_class=workload_class)
        manifest, runtime, _ = self.model_router.route_chat(
            model_id,
            messages=messages,
            max_tokens=128,
        )
        normalized_workload_class = serving_profile_workload_class(
            messages=messages,
            manifest=manifest,
            workload_class_hint=workload_class,
        )
        self._ensure_manifest_supports_workload(manifest=manifest, workload_class=normalized_workload_class)
        return resolve_serving_profile_application(
            settings=self.settings,
            metadata_store=self.metadata_store,
            host_platform=self.runtime_catalog.host_platform_snapshot().model_dump(mode="json"),
            runtime=runtime,
            model_id=manifest.model_id,
            request_capability=CapabilityName.CHAT,
            apply_serving_profile=apply_serving_profile,
            workload_class=normalized_workload_class,
        )

    async def _benchmark_chat_model(
        self,
        *,
        model_id: str | None,
        prompt: str,
        warmup_run_count: int = 1,
        workload_class: str | None = None,
        serving_profile: ServingProfileApplication | None = None,
    ) -> BenchmarkResult:
        messages = self._benchmark_workload_messages(prompt=prompt, workload_class=workload_class)
        manifest, runtime, routing = self.model_router.route_chat(
            model_id,
            messages=messages,
            max_tokens=128,
        )
        normalized_workload_class = serving_profile_workload_class(
            messages=messages,
            manifest=manifest,
            workload_class_hint=workload_class,
        )
        self._ensure_manifest_supports_workload(manifest=manifest, workload_class=normalized_workload_class)
        probe = await self._isolated_chat_benchmark_probe(
            model_id=manifest.model_id,
            prompt=prompt,
            warmup_run_count=warmup_run_count,
            workload_class=normalized_workload_class,
        )
        baseline_run = probe["baseline_run"]
        candidate_runs = probe["candidate_runs"]
        rejected_candidates = probe["rejected_candidates"]
        safe_candidates = probe["safe_candidates"]
        workload_class = str(probe["workload_class"])
        speculation_workload_class = str(probe["speculation_workload_class"])
        selected_probe_run = probe["selected_run"]
        selected_request: GenerateRequest = selected_probe_run["request"].model_copy(deep=True)
        companion_manifests = tuple(
            self.model_router.model_registry.get_manifest(companion_manifest.model_id)
            for companion_manifest in selected_probe_run["companion_manifests"]
        )
        selected_response, selected_load_seconds, selected_generate_seconds, selected_total_seconds = await self._execute_benchmark_request(
            manifest=manifest,
            runtime=runtime,
            companion_manifests=companion_manifests,
            invoke=lambda request=selected_request: runtime.generate(request),
        )
        selected_usage = {key: int(value) for key, value in selected_response.usage.items()}
        selected_mode = selected_request.speculation.mode if selected_request.speculation is not None else None
        benchmark_id = uuid4().hex
        self._store_speculation_benchmark_preference(
            manifest=manifest,
            runtime=runtime,
            benchmark_id=benchmark_id,
            workload_class=speculation_workload_class,
            selected_mode=selected_mode,
            selected_run=selected_probe_run,
            candidate_runs=candidate_runs,
        )
        frontier_request_measurements = frontier_architecture_measurements(selected_request.metadata)
        frontier_scenario = self._frontier_request_scenario(manifest=manifest, request=selected_request)
        distributed_request_measurements = distributed_pipeline_measurements(selected_request.metadata)
        distributed_scenario = self._distributed_pipeline_scenario(
            manifest=manifest,
            runtime=runtime,
            request=selected_request,
        )
        constrained_decoding_scenario = await self._constrained_decoding_scenario(
            manifest=manifest,
            runtime=runtime,
            messages=selected_request.messages,
        )
        warm_phase_breakdown = probe["warm_phase_breakdown"]
        completion_tokens = selected_usage.get("completion_tokens", 0)
        completion_tokens_per_second = (
            round(completion_tokens / selected_generate_seconds, 4)
            if completion_tokens > 0 and selected_generate_seconds > 0
            else None
        )
        performance_features = await self._benchmark_performance_features(
            runtime=runtime,
            capability=CapabilityName.CHAT.value,
        )
        performance_features = self._apply_benchmark_request_feature_activity(
            performance_features=performance_features,
            request=selected_request,
            usage=selected_usage,
        )
        measurements = {
            "message_count": len(messages),
            "prompt_characters": len(prompt),
            "output_characters": len(selected_response.output_text),
            "selected_speculation_requests": 1 if selected_mode is not None else 0,
            "speculation_candidate_count": len(candidate_runs),
            "safe_speculation_candidate_count": len(safe_candidates),
            "speculation_skipped_candidate_count": len(rejected_candidates),
            "fallback_count": sum(int(run["fallback_count"]) for run in candidate_runs),
            **frontier_request_measurements,
            **distributed_request_measurements,
            **mlx_acceleration_measurements(request=selected_request),
            **speculation_measurements(request=selected_request, usage=selected_usage),
        }
        scenario = self._speculation_selection_scenario(
            manifest=manifest,
            runtime=runtime,
            baseline_run=baseline_run,
            candidate_runs=candidate_runs,
            rejected_candidates=rejected_candidates,
            selected_mode=selected_mode,
            workload_class=speculation_workload_class,
        )
        record = BenchmarkResult(
            benchmark_id=benchmark_id,
            model_id=manifest.model_id,
            runtime=runtime.name,
            capability=CapabilityName.CHAT.value,
            workload_class=workload_class,
            reason=routing.reason,
            prompt=prompt,
            output_text=selected_response.output_text,
            load_seconds=round(float(selected_load_seconds), 4),
            generate_seconds=round(float(selected_generate_seconds), 4),
            total_seconds=round(float(selected_total_seconds), 4),
            usage=selected_usage,
            measurements=measurements,
            phase_breakdown=warm_phase_breakdown,
            optimization_attribution=self._chat_optimization_attribution(
                manifest=manifest,
                routing=routing,
                performance_features=performance_features,
                measurements=measurements,
                phase_breakdown=warm_phase_breakdown,
                selected_mode=selected_mode,
                serving_profile=serving_profile,
            ),
            completion_tokens_per_second=completion_tokens_per_second,
            created_at=utc_now(),
            performance_features=performance_features,
            serving_profile=serving_profile,
            scenarios=[
                scenario,
                constrained_decoding_scenario,
                *([distributed_scenario] if distributed_scenario is not None else []),
                *([frontier_scenario] if frontier_scenario is not None else []),
            ],
        )
        self.metadata_store.append_benchmark_record(record.model_dump(mode="json"))
        self._persist_benchmark_probe_records(record)
        return record

    async def _isolated_chat_benchmark_probe(
        self,
        *,
        model_id: str,
        prompt: str,
        warmup_run_count: int,
        workload_class: str | None,
    ) -> dict[str, Any]:
        if self.service_factory is None:
            return await self._benchmark_chat_probe(
                model_id=model_id,
                prompt=prompt,
                warmup_run_count=warmup_run_count,
                workload_class=workload_class,
            )
        probe_settings = self.settings.with_updates(
            data_dir=self.settings.data_dir / "benchmark-probes" / uuid4().hex,
            models_dir=self.settings.models_dir,
        )
        probe_services = self.service_factory(probe_settings)
        try:
            self._sync_benchmark_child_manifests(
                probe_services,
                model_dirs=probe_settings.models_dir,
            )
            return await probe_services.telemetry_service._benchmark_chat_probe(
                model_id=model_id,
                prompt=prompt,
                warmup_run_count=warmup_run_count,
                workload_class=workload_class,
            )
        finally:
            await probe_services.aclose()

    def _sync_benchmark_child_manifests(
        self,
        candidate_services,
        *,
        model_dirs: Sequence[Path] | None = None,
    ) -> None:
        current_manifests = self.model_router.model_registry.list_manifests()
        if current_manifests:
            candidate_services.metadata_store.replace_model_manifests(
                current_manifests,
                stale_source_paths=(),
            )
            return
        candidate_services.model_registry.scan(list(model_dirs or candidate_services.settings.models_dir))

    async def _benchmark_chat_probe(
        self,
        *,
        model_id: str,
        prompt: str,
        warmup_run_count: int,
        workload_class: str | None,
    ) -> dict[str, Any]:
        messages = self._benchmark_workload_messages(prompt=prompt, workload_class=workload_class)
        manifest, runtime, _ = self.model_router.route_chat(
            model_id,
            messages=messages,
            max_tokens=128,
        )
        resolved_workload_class = serving_profile_workload_class(
            messages=messages,
            manifest=manifest,
            workload_class_hint=workload_class,
        )
        self._ensure_manifest_supports_workload(manifest=manifest, workload_class=resolved_workload_class)
        inspection = inspect_chat_speculation_candidates(
            model_registry=self.model_router.model_registry,
            settings=self.settings,
            primary_manifest=manifest,
            runtime=runtime,
            messages=messages,
            max_tokens=128,
        )
        candidate_plans = list(inspection.candidates)
        rejected_candidates = list(inspection.rejected)
        speculation_workload_class = inspection.workload_class
        baseline_request = GenerateRequest(
            model_id=manifest.model_id,
            messages=messages,
            max_tokens=128,
            temperature=0.0,
            metadata={
                "speculation_selection_source": "benchmark_baseline",
                "speculation_workload_class": speculation_workload_class,
                "serving_profile_workload_class": resolved_workload_class,
                **self._benchmark_frontier_request_metadata(manifest),
            },
        )
        baseline_response, baseline_load_seconds, baseline_generate_seconds, baseline_total_seconds = await self._execute_benchmark_request(
            manifest=manifest,
            runtime=runtime,
            invoke=lambda: runtime.generate(baseline_request),
        )
        baseline_usage = {key: int(value) for key, value in baseline_response.usage.items()}
        baseline_run = {
            "mode": None,
            "request": baseline_request,
            "response": baseline_response,
            "companion_manifests": (),
            "usage": baseline_usage,
            "load_seconds": baseline_load_seconds,
            "generate_seconds": baseline_generate_seconds,
            "total_seconds": baseline_total_seconds,
            "fallback_count": 0,
            "correctness_match": True,
            "selection_reason": "Baseline non-speculative benchmark run.",
        }
        candidate_runs: list[dict[str, Any]] = []
        for planned_speculation in candidate_plans:
            request = GenerateRequest(
                model_id=manifest.model_id,
                messages=messages,
                max_tokens=128,
                temperature=0.0,
                speculation=planned_speculation.request,
                metadata={
                        "speculation_selection_source": (
                            "benchmark_preference" if planned_speculation.benchmark_preferred else "benchmark_candidate"
                        ),
                        "speculation_benchmark_preferred": planned_speculation.benchmark_preferred,
                        "speculation_selection_reason": planned_speculation.selection_reason,
                        "speculation_workload_class": speculation_workload_class,
                        "serving_profile_workload_class": resolved_workload_class,
                        **self._benchmark_frontier_request_metadata(manifest),
                    },
                )
            try:
                response, load_seconds, generate_seconds, total_seconds = await self._execute_benchmark_request(
                    manifest=manifest,
                    runtime=runtime,
                    companion_manifests=planned_speculation.companion_manifests,
                    invoke=lambda request=request: runtime.generate(request),
                )
                usage = {key: int(value) for key, value in response.usage.items()}
                correctness_match = response.output_text == baseline_response.output_text
                candidate_runs.append(
                    {
                        "mode": planned_speculation.request.mode,
                        "request": request,
                        "response": response,
                        "companion_manifests": planned_speculation.companion_manifests,
                        "usage": usage,
                        "load_seconds": load_seconds,
                        "generate_seconds": generate_seconds,
                        "total_seconds": total_seconds,
                        "fallback_count": 0 if correctness_match else 1,
                        "correctness_match": correctness_match,
                        "selection_reason": planned_speculation.selection_reason,
                    },
                )
            except Exception as exc:
                candidate_runs.append(
                    {
                        "mode": planned_speculation.request.mode,
                        "request": request,
                        "response": None,
                        "companion_manifests": planned_speculation.companion_manifests,
                        "usage": {},
                        "load_seconds": 0.0,
                        "generate_seconds": 0.0,
                        "total_seconds": 0.0,
                        "fallback_count": 1,
                        "correctness_match": False,
                        "selection_reason": f"{planned_speculation.selection_reason} Benchmark candidate failed: {exc}",
                        "error": str(exc),
                    },
                )
        safe_candidates = [
            run
            for run in candidate_runs
            if run["response"] is not None and run["correctness_match"] is True
        ]
        selected_run = min(
            [baseline_run, *safe_candidates],
            key=lambda run: (
                float(run["total_seconds"]),
                float(run["generate_seconds"]),
                0 if run["mode"] is not None else 1,
            ),
        )
        warm_phase_breakdown = await self._chat_phase_breakdown(
            manifest=manifest,
            runtime=runtime,
            selected_run=selected_run,
            warmup_run_count=warmup_run_count,
        )
        return {
            "baseline_run": baseline_run,
            "candidate_runs": candidate_runs,
            "rejected_candidates": rejected_candidates,
            "safe_candidates": safe_candidates,
            "selected_run": selected_run,
            "warm_phase_breakdown": warm_phase_breakdown,
            "workload_class": resolved_workload_class,
            "speculation_workload_class": speculation_workload_class,
        }

    def _benchmark_frontier_request_metadata(self, manifest: ModelManifest) -> dict[str, object]:
        frontier_plan = build_frontier_serving_plan(manifest=manifest, settings=self.settings)
        if frontier_plan is None:
            return {}
        return {"frontier_architecture": dict(frontier_plan)}

    def _frontier_request_scenario(
        self,
        *,
        manifest: ModelManifest,
        request: GenerateRequest,
    ) -> BenchmarkScenarioReport | None:
        frontier_plan = request.metadata.get("frontier_architecture")
        if not isinstance(frontier_plan, dict):
            return None
        subtype = str(frontier_plan.get("architecture_subtype") or manifest.architecture_subtype.value)
        if subtype not in {"ssm_mamba", "hybrid_ssm", "moe", "hybrid_moe"}:
            return None
        is_moe = subtype in {"moe", "hybrid_moe"}
        planning_only = bool(frontier_plan.get("planning_only", True))
        sample_metrics = _compact_metrics(
            architecture_subtype=subtype,
            planning_only=planning_only,
            execution_path=frontier_plan.get("execution_path"),
            bounded_memory_mode=frontier_plan.get("bounded_memory_mode"),
            cache_state_handling=frontier_plan.get("cache_state_handling"),
            full_estimated_memory_mb=frontier_plan.get("full_estimated_memory_mb"),
            planned_memory_mb=frontier_plan.get("planned_memory_mb"),
            effective_loaded_memory_mb=frontier_plan.get("effective_loaded_memory_mb"),
            memory_savings_mb=frontier_plan.get("memory_savings_mb"),
            expert_count=frontier_plan.get("expert_count"),
            requested_expert_count=frontier_plan.get("requested_expert_count"),
            resident_expert_count=frontier_plan.get("resident_expert_count"),
            streamed_expert_count=frontier_plan.get("streamed_expert_count"),
            expert_swap_count=frontier_plan.get("expert_swap_count"),
            expert_swap_mb=frontier_plan.get("expert_swap_mb"),
            estimated_swap_mb_per_request=frontier_plan.get("estimated_swap_mb_per_request"),
            state_size=frontier_plan.get("state_size"),
            state_cache_hits=frontier_plan.get("state_cache_hits"),
            state_cache_misses=frontier_plan.get("state_cache_misses"),
            state_cache_entry_count=frontier_plan.get("state_cache_entry_count"),
            state_cache_bytes=frontier_plan.get("state_cache_bytes"),
        )
        return BenchmarkScenarioReport(
            scenario="frontier_architecture_modes",
            capability=CapabilityName.CHAT.value,
            feature=(
                PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING
                if is_moe
                else PerformanceFeatureName.HYBRID_SSM_ROUTING
            ),
            status="observed",
            reason=(
                "Measured realized bounded-memory MoE execution for the benchmark request."
                if is_moe and not planning_only
                else (
                    "Measured realized hybrid SSM state-cache execution for the benchmark request."
                    if not is_moe and not planning_only
                    else "Only planning metadata was available for the benchmark request."
                )
            ),
            metrics=_compact_metrics(
                sample_count=1,
                prompt_characters=sum(len(message.content) for message in request.messages),
                hybrid_ssm_model_count=0 if is_moe else 1,
                moe_model_count=1 if is_moe else 0,
                planning_only=planning_only,
                configured_moe_bounded_memory_mode=self.settings.moe_bounded_memory_mode if is_moe else None,
                configured_moe_resident_expert_count=self.settings.moe_resident_expert_count if is_moe else None,
            ),
            samples=[BenchmarkScenarioSample(model_id=manifest.model_id, metrics=sample_metrics)],
            notes=(
                []
                if not planning_only
                else ["This run exposed frontier architecture planning metadata without realized execution-state metrics."]
            ),
        )

    def _distributed_pipeline_scenario(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        request: GenerateRequest,
    ) -> BenchmarkScenarioReport | None:
        payload = request.metadata.get("distributed_pipeline")
        if not isinstance(payload, dict):
            return None
        stage_metrics = payload.get("stage_metrics")
        if not isinstance(stage_metrics, list) or not stage_metrics:
            return None
        samples: list[BenchmarkScenarioSample] = []
        for stage_metric in stage_metrics:
            if not isinstance(stage_metric, dict):
                continue
            samples.append(
                BenchmarkScenarioSample(
                    model_id=manifest.model_id,
                    runtime=runtime.name,
                    metrics=_compact_metrics(
                        stage_index=_coerce_int(stage_metric.get("stage_index")),
                        stage_name=str(stage_metric.get("stage_name") or f"stage-{len(samples)}"),
                        worker_name=str(stage_metric.get("worker_name") or "worker"),
                        layer_span=_coerce_int(stage_metric.get("layer_span")),
                        stage_elapsed_seconds=_coerce_float(stage_metric.get("stage_elapsed_seconds")),
                        compute_seconds=_coerce_float(stage_metric.get("compute_seconds")),
                        network_seconds=_coerce_float(stage_metric.get("network_seconds")),
                        scheduling_seconds=_coerce_float(stage_metric.get("scheduling_seconds")),
                        overlap_credit_seconds=_coerce_float(stage_metric.get("overlap_credit_seconds")),
                        utilization=_coerce_float(stage_metric.get("utilization")),
                        target_batch_tokens=_coerce_int(stage_metric.get("target_batch_tokens")),
                        prefetch_tokens=_coerce_int(stage_metric.get("prefetch_tokens")),
                        bottleneck=str(stage_metric.get("bottleneck") or "balanced"),
                    ),
                ),
            )
        return BenchmarkScenarioReport(
            scenario="distributed_pipeline_scaling",
            capability=CapabilityName.CHAT.value,
            feature=PerformanceFeatureName.DISTRIBUTED_PIPELINE,
            status="observed",
            reason=(
                "Benchmarked a multi-host distributed proof run and recorded per-stage compute, network, prefetch, "
                "and overlap metrics so operators can separate transport or scheduling limits from model execution."
            ),
            metrics=_compact_metrics(
                model_id=manifest.model_id,
                runtime=runtime.name,
                stage_count=_coerce_int(payload.get("stage_count")),
                worker_count=_coerce_int(payload.get("worker_count")),
                recovery_count=_coerce_int(payload.get("recovery_count")),
                pipeline_latency_seconds=_coerce_float(payload.get("pipeline_latency_seconds")),
                critical_path_seconds=_coerce_float(payload.get("critical_path_seconds")),
                throughput_tokens_per_second=_coerce_float(payload.get("throughput_tokens_per_second")),
                completion_tokens_per_second=_coerce_float(payload.get("completion_tokens_per_second")),
                average_stage_elapsed_seconds=_coerce_float(payload.get("average_stage_elapsed_seconds")),
                average_stage_utilization=_coerce_float(payload.get("average_stage_utilization")),
                effective_batch_tokens=_coerce_int(payload.get("effective_batch_tokens")),
                average_prefetch_tokens=_coerce_int(payload.get("average_prefetch_tokens")),
                average_network_latency_ms=_coerce_float(payload.get("average_network_latency_ms")),
                compute_share_percent=_coerce_float(payload.get("compute_share_percent")),
                network_share_percent=_coerce_float(payload.get("network_share_percent")),
                scheduling_share_percent=_coerce_float(payload.get("scheduling_share_percent")),
                pipeline_overlap_efficiency_percent=_coerce_float(payload.get("pipeline_overlap_efficiency_percent")),
                speedup_vs_single_host_percent=_coerce_float(payload.get("speedup_vs_single_host_percent")),
                heterogeneity_ratio=_coerce_float(payload.get("heterogeneity_ratio")),
                bottleneck=str(payload.get("bottleneck") or "balanced"),
                prefetch_enabled=bool(payload.get("prefetch_enabled", False)),
                overlap_enabled=bool(payload.get("overlap_enabled", False)),
            ),
            samples=samples,
            notes=[
                *[
                    note
                    for note in payload.get("notes", [])
                    if isinstance(note, str) and note
                ],
                "Stage samples reflect the weighted plan chosen for this run rather than a theoretical all-homogeneous cluster.",
            ],
        )

    def _speculation_selection_scenario(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        baseline_run: dict[str, Any],
        candidate_runs: list[dict[str, Any]],
        rejected_candidates: Sequence[Any],
        selected_mode: SpeculationMode | None,
        workload_class: str,
    ) -> BenchmarkScenarioReport:
        if not candidate_runs:
            return BenchmarkScenarioReport(
                scenario="speculation_selection",
                capability=CapabilityName.CHAT.value,
                status="not_applicable",
                reason="No compatible speculation adapters were available for this benchmarked model/runtime pair.",
                metrics={
                    "selected_mode": "disabled",
                    "workload_class": workload_class,
                    "skipped_candidate_count": len(rejected_candidates),
                },
            )
        samples = [
            BenchmarkScenarioSample(
                model_id=manifest.model_id,
                runtime=runtime.name,
                metrics=_compact_metrics(
                    mode="disabled",
                    load_seconds=round(float(baseline_run["load_seconds"]), 4),
                    generate_seconds=round(float(baseline_run["generate_seconds"]), 4),
                    total_seconds=round(float(baseline_run["total_seconds"]), 4),
                    correctness_match=True,
                    fallback_count=0,
                    selected=selected_mode is None,
                    selection_status="selected" if selected_mode is None else "baseline",
                    outcome_reason=(
                        "The non-speculative baseline remained the fastest correctness-preserving path."
                        if selected_mode is None
                        else "The baseline matched the output but lost on latency to a speculative path."
                    ),
                ),
            ),
        ]
        for run in candidate_runs:
            mode = run["mode"]
            usage = run["usage"]
            if run["response"] is None:
                selection_status = "lost"
                outcome_reason = str(run.get("error") or "The benchmark candidate failed before producing output.")
            elif not run["correctness_match"]:
                selection_status = "lost"
                outcome_reason = "The candidate changed the output relative to the baseline and was rejected."
            elif mode == selected_mode:
                selection_status = "selected"
                outcome_reason = "Selected as the fastest correctness-preserving speculation mode for this workload."
            else:
                selection_status = "lost"
                outcome_reason = "The candidate matched the baseline output but was slower than the selected path."
            samples.append(
                BenchmarkScenarioSample(
                    model_id=manifest.model_id,
                    runtime=runtime.name,
                    metrics=_compact_metrics(
                        mode=mode.value if isinstance(mode, SpeculationMode) else "unknown",
                        load_seconds=round(float(run["load_seconds"]), 4),
                        generate_seconds=round(float(run["generate_seconds"]), 4),
                        total_seconds=round(float(run["total_seconds"]), 4),
                        acceptance_rate=self._acceptance_rate_from_usage(usage),
                        verified_tokens=_coerce_int(usage.get("verified_tokens")),
                        rollback_tokens=self._rollback_tokens_from_usage(usage),
                        fallback_count=int(run["fallback_count"]),
                        correctness_match=bool(run["correctness_match"]),
                        selected=mode == selected_mode,
                        selection_status=selection_status,
                        selection_reason=str(run["selection_reason"]),
                        outcome_reason=outcome_reason,
                        error=str(run["error"]) if run.get("error") is not None else None,
                    ),
                ),
            )
        for rejection in rejected_candidates:
            mode = getattr(rejection, "mode", None)
            samples.append(
                BenchmarkScenarioSample(
                    model_id=manifest.model_id,
                    runtime=runtime.name,
                    metrics=_compact_metrics(
                        mode=mode.value if isinstance(mode, SpeculationMode) else "unknown",
                        selected=False,
                        selection_status="skipped",
                        source=str(getattr(rejection, "source", "heuristic")),
                        outcome_reason=str(getattr(rejection, "reason", "")),
                    ),
                ),
            )
        safe_candidate_count = sum(
            1
            for run in candidate_runs
            if run["response"] is not None and run["correctness_match"] is True
        )
        selected_usage = baseline_run["usage"]
        if selected_mode is not None:
            for run in candidate_runs:
                if run["mode"] == selected_mode:
                    selected_usage = run["usage"]
                    break
        selected_total_seconds = min(
            [
                float(baseline_run["total_seconds"]),
                *[
                    float(run["total_seconds"])
                    for run in candidate_runs
                    if run["response"] is not None and run["correctness_match"] is True
                ],
            ],
        )
        return BenchmarkScenarioReport(
            scenario="speculation_selection",
            capability=CapabilityName.CHAT.value,
            status="observed",
            reason=(
                "Benchmarked the baseline path plus each safe speculation adapter and selected the fastest output-matching mode."
            ),
            metrics=_compact_metrics(
                baseline_total_seconds=round(float(baseline_run["total_seconds"]), 4),
                selected_total_seconds=round(selected_total_seconds, 4),
                candidate_count=len(candidate_runs),
                safe_candidate_count=safe_candidate_count,
                skipped_candidate_count=len(rejected_candidates),
                fallback_count=sum(int(run["fallback_count"]) for run in candidate_runs),
                selected_mode=selected_mode.value if selected_mode is not None else "disabled",
                workload_class=workload_class,
                selected_acceptance_rate=self._acceptance_rate_from_usage(selected_usage),
                selected_verified_tokens=_coerce_int(selected_usage.get("verified_tokens")),
                selected_rollback_tokens=self._rollback_tokens_from_usage(selected_usage),
            ),
            samples=samples,
            notes=[
                "Speculative candidates only qualify when their benchmark output matches the non-speculative baseline exactly.",
            ],
        )

    async def _constrained_decoding_scenario(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        messages: Sequence[GenerateMessage],
    ) -> BenchmarkScenarioReport:
        request = GenerateRequest(
            model_id=manifest.model_id,
            messages=[message.model_copy(deep=True) for message in messages],
            max_tokens=32,
            temperature=0.0,
            structured_output=JSONSchemaResponseFormat(
                schema=_BENCHMARK_CONSTRAINED_DECODING_SCHEMA,
                name="lewlm_benchmark_probe",
                strict=True,
            ),
        )
        try:
            response, load_seconds, generate_seconds, total_seconds = await self._execute_benchmark_request(
                manifest=manifest,
                runtime=runtime,
                invoke=lambda request=request: runtime.generate(request),
            )
        except Exception as exc:
            return BenchmarkScenarioReport(
                scenario="constrained_decoding",
                capability=CapabilityName.CHAT.value,
                status="unsupported",
                reason=f"Structured-output benchmark probe failed on the routed runtime: {exc}",
                metrics=_compact_metrics(
                    sample_count=1,
                    failure_type=type(exc).__name__,
                ),
                samples=[
                    BenchmarkScenarioSample(
                        model_id=manifest.model_id,
                        runtime=runtime.name,
                        metrics=_compact_metrics(failure_type=type(exc).__name__),
                    ),
                ],
                notes=[
                    "LewLM only upgrades measured constrained decoding after a benchmark-side structured-output probe completes.",
                ],
            )
        structured_result = analyze_structured_output(
            format="json_schema",
            output_text=response.output_text,
            schema=_BENCHMARK_CONSTRAINED_DECODING_SCHEMA,
            name="lewlm_benchmark_probe",
            strict=True,
            runtime_status=request.metadata.get("structured_output_runtime"),
        )
        validation = structured_result.validation if structured_result is not None else None
        decoder_enforced = bool(structured_result.decoder_enforced) if structured_result is not None else False
        fallback_used = bool(structured_result.fallback_used) if structured_result is not None else False
        enforcement = structured_result.enforcement if structured_result is not None else "none"
        validation_state = validation.state if validation is not None else "unavailable"
        if decoder_enforced and validation_state == "valid":
            reason = "Benchmark verified decode-time constrained decoding on the routed runtime."
        elif fallback_used:
            reason = "Benchmark observed prompt-guided structured-output fallback instead of decode-time constrained decoding."
        elif validation_state == "valid":
            reason = "Benchmark observed structured output, but the runtime did not report decode-time decoder enforcement."
        else:
            reason = "Benchmark could not verify decode-time constrained decoding from the routed runtime response."
        notes: list[str] = []
        if validation is not None and validation.message:
            notes.append(validation.message)
        if structured_result is not None and structured_result.fallback_reason:
            notes.append(structured_result.fallback_reason)
        return BenchmarkScenarioReport(
            scenario="constrained_decoding",
            capability=CapabilityName.CHAT.value,
            status="observed",
            reason=reason,
            metrics=_compact_metrics(
                sample_count=1,
                enforcement=enforcement,
                decoder_enforced=decoder_enforced,
                fallback_used=fallback_used,
                validation_state=validation_state,
                validation_issue_count=len(validation.issues) if validation is not None else 0,
                load_seconds=round(float(load_seconds), 4),
                generate_seconds=round(float(generate_seconds), 4),
                total_seconds=round(float(total_seconds), 4),
                output_characters=len(response.output_text),
            ),
            samples=[
                BenchmarkScenarioSample(
                    model_id=manifest.model_id,
                    runtime=runtime.name,
                    metrics=_compact_metrics(
                        enforcement=enforcement,
                        decoder_enforced=decoder_enforced,
                        fallback_used=fallback_used,
                        validation_state=validation_state,
                        validation_issue_count=len(validation.issues) if validation is not None else 0,
                    ),
                ),
            ],
            notes=notes,
        )

    def _store_speculation_benchmark_preference(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        benchmark_id: str,
        workload_class: str,
        selected_mode: SpeculationMode | None,
        selected_run: Mapping[str, Any],
        candidate_runs: Sequence[Mapping[str, Any]],
    ) -> None:
        preference = SpeculationBenchmarkPreference(
            model_id=manifest.model_id,
            runtime_name=runtime.name,
            workload_class=workload_class,
            selected_mode=selected_mode,
            benchmark_id=benchmark_id,
            generate_seconds=round(float(selected_run["generate_seconds"]), 4),
            total_seconds=round(float(selected_run["total_seconds"]), 4),
            acceptance_rate=self._acceptance_rate_from_usage(selected_run.get("usage", {})),
            rollback_tokens=self._rollback_tokens_from_usage(selected_run.get("usage", {})),
            verified_tokens=_coerce_int((selected_run.get("usage") or {}).get("verified_tokens")),
            fallback_count=sum(_coerce_int(run.get("fallback_count")) for run in candidate_runs),
        )
        self.metadata_store.set_value(
            speculation_benchmark_preference_key(
                model_id=manifest.model_id,
                runtime_name=runtime.name,
                workload_class=workload_class,
            ),
            preference.model_dump(mode="json"),
        )

    @staticmethod
    def _acceptance_rate_from_usage(usage: Mapping[str, Any]) -> float | None:
        drafted_tokens = _coerce_int(usage.get("drafted_tokens"))
        verified_tokens = _coerce_int(usage.get("verified_tokens"))
        if drafted_tokens <= 0:
            return None
        return round(verified_tokens / drafted_tokens, 4)

    @staticmethod
    def _rollback_tokens_from_usage(usage: Mapping[str, Any]) -> int:
        rollback_tokens = _coerce_int(usage.get("rollback_tokens"))
        if rollback_tokens > 0:
            return rollback_tokens
        drafted_tokens = _coerce_int(usage.get("drafted_tokens"))
        verified_tokens = _coerce_int(usage.get("verified_tokens"))
        if drafted_tokens <= 0:
            return 0
        return max(drafted_tokens - verified_tokens, 0)

    async def _chat_phase_breakdown(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        selected_run: Mapping[str, Any],
        warmup_run_count: int,
    ) -> dict[str, int | float | str | bool | None]:
        realized_warmup_count = max(warmup_run_count, 0)
        request = selected_run["request"]
        usage = selected_run.get("usage", {})
        cold_load_seconds = float(selected_run["load_seconds"])
        cold_generate_seconds = float(selected_run["generate_seconds"])
        cold_total_seconds = float(selected_run["total_seconds"])
        companion_manifests = tuple(selected_run.get("companion_manifests", ()))
        warmup_elapsed_values: list[float] = []
        for _ in range(realized_warmup_count):
            _, _, _, elapsed_seconds = await self._execute_benchmark_request(
                manifest=manifest,
                runtime=runtime,
                companion_manifests=companion_manifests,
                invoke=lambda request=request: runtime.generate(request),
            )
            if elapsed_seconds is not None:
                warmup_elapsed_values.append(elapsed_seconds)
        warm_response, _, _, warm_total_seconds = await self._execute_benchmark_request(
            manifest=manifest,
            runtime=runtime,
            companion_manifests=companion_manifests,
            invoke=lambda request=request: runtime.generate(request),
        )
        warm_stream = await self._execute_stream_benchmark_request(
            manifest=manifest,
            runtime=runtime,
            request=request,
            companion_manifests=companion_manifests,
        )
        ttft_seconds = _coerce_float(warm_stream.get("ttft_seconds"))
        streamed_total_seconds = _coerce_float(warm_stream.get("elapsed_seconds"))
        steady_state_decode_seconds = (
            round(max(streamed_total_seconds - ttft_seconds, 0.0), 4)
            if streamed_total_seconds is not None and ttft_seconds is not None
            else None
        )
        completion_tokens = _coerce_int(usage.get("completion_tokens"))
        steady_state_decode_tokens_per_second = (
            round(completion_tokens / steady_state_decode_seconds, 4)
            if completion_tokens > 0 and steady_state_decode_seconds not in (None, 0.0)
            else None
        )
        return _compact_metrics(
            cold_load_seconds=round(cold_load_seconds, 4),
            cold_generate_seconds=round(cold_generate_seconds, 4),
            cold_total_seconds=round(cold_total_seconds, 4),
            warm_load_seconds=0.0,
            warm_generate_seconds=warm_total_seconds,
            warm_total_seconds=warm_total_seconds,
            ttft_seconds=ttft_seconds,
            steady_state_decode_seconds=steady_state_decode_seconds,
            steady_state_decode_tokens_per_second=steady_state_decode_tokens_per_second,
            warmup_run_count=realized_warmup_count,
            warmup_average_total_seconds=(
                round(fmean(warmup_elapsed_values), 4) if warmup_elapsed_values else None
            ),
            streamed_total_seconds=streamed_total_seconds,
            streamed_output_matches_warm_completion=(
                warm_stream.get("output_text") == warm_response.output_text
                if warm_stream.get("output_text") is not None
                else None
            ),
        )

    def _chat_optimization_attribution(
        self,
        *,
        manifest: ModelManifest,
        routing: RoutingDecision,
        performance_features: Sequence[PerformanceFeatureStatus],
        measurements: Mapping[str, Any],
        phase_breakdown: Mapping[str, Any],
        selected_mode: SpeculationMode | None,
        serving_profile: ServingProfileApplication | None,
    ) -> dict[str, Any]:
        feature_map = {item.feature: item for item in performance_features}
        cache_features = [
            feature
            for feature in (
                PerformanceFeatureName.PREFIX_CACHE,
                PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE,
                PerformanceFeatureName.MULTIMODAL_FEATURE_CACHING,
                PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING,
                PerformanceFeatureName.DISK_BACKED_CACHE,
                PerformanceFeatureName.BLOCK_DISK_CACHE,
            )
            if feature in feature_map
        ]
        active_cache_features = [feature.value for feature in cache_features if feature_map[feature].active]
        supported_cache_features = [feature.value for feature in cache_features if feature_map[feature].supported]
        batching_feature = feature_map.get(PerformanceFeatureName.CONTINUOUS_BATCHING)
        graph_feature = feature_map.get(PerformanceFeatureName.GRAPH_COMPILATION)
        kernel_feature = feature_map.get(PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION)
        frontier_feature = (
            feature_map.get(PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING)
            if manifest.architecture_subtype.value in {"moe", "hybrid_moe"}
            else feature_map.get(PerformanceFeatureName.HYBRID_SSM_ROUTING)
            if manifest.architecture_subtype.value in {"ssm_mamba", "hybrid_ssm"}
            else None
        )
        host_platform = self.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
        workload_class = (
            serving_profile.workload_class
            if serving_profile is not None and serving_profile.workload_class
            else default_serving_profile_workload_class(manifest=manifest)
        )
        persisted_profile = self.metadata_store.get_serving_profile(
            model_id=manifest.model_id,
            capability=CapabilityName.CHAT.value,
            host_platform=host_platform,
            runtime_name=routing.runtime_name,
            workload_class=workload_class,
        )
        effective_serving_settings = (
            dict(serving_profile.effective_settings)
            if serving_profile is not None and serving_profile.effective_settings
            else self._serving_profile_effective_settings({})
        )
        quantization_label = quantization_profile_label(manifest.quantization_profile) or manifest.quantization
        return {
            "modality_routing": {
                "status": (
                    "active"
                    if routing.modality_path is not None and routing.modality_path.value != "text_default"
                    else "default"
                ),
                "detail": routing.modality_path_reason or routing.reason,
                "metrics": {
                    "request_modality": routing.request_modality.value if routing.request_modality is not None else None,
                    "modality_path": routing.modality_path.value if routing.modality_path is not None else None,
                    "runtime_affinity": routing.runtime_affinity.value,
                },
            },
            "cache_reuse": {
                "status": "active" if active_cache_features else "inactive",
                "detail": (
                    "Observed cache reuse in the measured run."
                    if active_cache_features
                    else "No cache-reuse feature reported active during the measured run."
                ),
                "metrics": {
                    "active_features": active_cache_features,
                    "supported_features": supported_cache_features,
                    "warm_total_seconds": _coerce_float(phase_breakdown.get("warm_total_seconds")),
                    "ttft_seconds": _coerce_float(phase_breakdown.get("ttft_seconds")),
                },
            },
            "batching": {
                "status": "active" if batching_feature is not None and batching_feature.active else "inactive",
                "detail": (
                    "LewLM's continuous-batching scheduler recorded active batched work during the measured run."
                    if batching_feature is not None and batching_feature.active
                    else "The measured run stayed on the single-request path without active continuous batching."
                ),
                "metrics": dict(batching_feature.metrics) if batching_feature is not None else {},
            },
            "speculation": {
                "status": "active" if selected_mode is not None else "inactive",
                "detail": (
                    f"Selected `{selected_mode.value}` as the fastest correctness-preserving speculation mode."
                    if selected_mode is not None
                    else "The measured run used the non-speculative baseline path."
                ),
                "metrics": {
                    "selected_mode": selected_mode.value if selected_mode is not None else "disabled",
                    "candidate_count": _coerce_int(measurements.get("speculation_candidate_count")),
                    "safe_candidate_count": _coerce_int(measurements.get("safe_speculation_candidate_count")),
                    "skipped_candidate_count": _coerce_int(measurements.get("speculation_skipped_candidate_count")),
                    "acceptance_rate": _coerce_float(measurements.get("acceptance_rate")),
                    "rollback_tokens": _coerce_int(measurements.get("rollback_tokens")),
                    "verified_tokens": _coerce_int(measurements.get("verified_tokens")),
                    "fallback_count": _coerce_int(measurements.get("fallback_count")),
                },
            },
            "kernel_acceleration": {
                "status": (
                    "active"
                    if (graph_feature is not None and graph_feature.active)
                    or (kernel_feature is not None and kernel_feature.active)
                    else "inactive"
                ),
                "detail": (
                    "The measured run exercised MLX graph compilation or an accelerated attention-kernel path."
                    if (graph_feature is not None and graph_feature.active)
                    or (kernel_feature is not None and kernel_feature.active)
                    else "The measured run stayed on the stock kernel path without active MLX acceleration."
                ),
                "metrics": {
                    "graph_compilation_active": bool(graph_feature.active) if graph_feature is not None else False,
                    "attention_kernel_active": bool(kernel_feature.active) if kernel_feature is not None else False,
                    "graph_compile_requests": _coerce_int(measurements.get("graph_compile_requests")),
                    "graph_compile_fallbacks": _coerce_int(measurements.get("graph_compile_fallbacks")),
                    "flash_attention_requests": _coerce_int(measurements.get("flash_attention_requests")),
                    "custom_sdpa_requests": _coerce_int(measurements.get("custom_sdpa_requests")),
                    "kernel_fallback_requests": _coerce_int(measurements.get("kernel_fallback_requests")),
                },
            },
            "frontier_architecture": {
                "status": (
                    "active"
                    if frontier_feature is not None and bool(measurements.get("frontier_architecture_detected"))
                    else "inactive"
                ),
                "detail": (
                    "The measured run used frontier-architecture execution-state tracking rather than planning-only reporting."
                    if bool(measurements.get("frontier_architecture_detected")) and not bool(measurements.get("frontier_planning_only"))
                    else (
                        "The measured run exposed frontier planning metadata without realized execution-state metrics."
                        if bool(measurements.get("frontier_architecture_detected"))
                        else "The measured run stayed on a standard non-frontier transformer path."
                    )
                ),
                "metrics": {
                    "feature": frontier_feature.feature.value if frontier_feature is not None else None,
                    "planning_only": bool(measurements.get("frontier_planning_only")),
                    "effective_loaded_memory_mb": _coerce_int(measurements.get("frontier_effective_loaded_memory_mb")),
                    "resident_expert_count": _coerce_int(measurements.get("frontier_resident_expert_count")),
                    "requested_expert_count": _coerce_int(measurements.get("frontier_requested_expert_count")),
                    "expert_swap_count": _coerce_int(measurements.get("frontier_expert_swap_count")),
                    "state_cache_bytes": _coerce_int(measurements.get("frontier_state_cache_bytes")),
                },
            },
            "quantization_profile": {
                "status": "active" if quantization_label is not None else "inactive",
                "detail": (
                    f"Benchmarked the artifact quantization profile `{quantization_label}`."
                    if quantization_label is not None
                    else "No explicit artifact quantization profile was recorded for the measured run."
                ),
                "metrics": {
                    "artifact_quantization": manifest.quantization,
                    "artifact_quantization_profile": quantization_label,
                    "kv_cache_quantization_active": bool(
                        feature_map.get(PerformanceFeatureName.KV_CACHE_QUANTIZATION) is not None
                        and feature_map[PerformanceFeatureName.KV_CACHE_QUANTIZATION].active
                    ),
                },
            },
            "serving_profile_defaults": {
                "status": "active",
                "detail": (
                    "Benchmarks applied the persisted serving-profile defaults selected for this host/model pair."
                    if serving_profile is not None and serving_profile.status == "selected"
                    else (
                        "Benchmarks used the current host defaults because no matching persisted serving profile was applied."
                    )
                ),
                "metrics": {
                    "persisted_profile_available": persisted_profile is not None,
                    "persisted_profile_applied": bool(
                        serving_profile is not None and serving_profile.status == "selected"
                    ),
                    "profile_status": serving_profile.status if serving_profile is not None else None,
                    "effective_settings": effective_serving_settings,
                },
            },
        }

    @staticmethod
    def _default_phase_breakdown(
        *,
        load_seconds: float,
        generate_seconds: float,
        total_seconds: float,
    ) -> dict[str, int | float | str | bool | None]:
        return _compact_metrics(
            cold_load_seconds=round(load_seconds, 4),
            cold_generate_seconds=round(generate_seconds, 4),
            cold_total_seconds=round(total_seconds, 4),
            warmup_run_count=0,
        )

    async def _benchmark_embeddings_model(
        self,
        *,
        model_id: str | None,
        prompt: str,
        warmup_run_count: int = 1,
    ) -> BenchmarkResult:
        inputs = [prompt, f"{prompt} secondary"]
        manifest, runtime, routing = self.model_router.route_embeddings(model_id, inputs=inputs)
        response, load_seconds, generate_seconds, total_seconds = await self._execute_benchmark_request(
            manifest=manifest,
            runtime=runtime,
            invoke=lambda: runtime.embed(
                EmbeddingRequest(
                    model_id=manifest.model_id,
                    inputs=inputs,
                ),
            ),
        )
        usage = {key: int(value) for key, value in response.usage.items()}
        vector_dimensions = len(response.data[0].embedding) if response.data else 0
        record = BenchmarkResult(
            benchmark_id=uuid4().hex,
            model_id=manifest.model_id,
            runtime=runtime.name,
            capability=CapabilityName.EMBEDDINGS.value,
            reason=routing.reason,
            prompt=prompt,
            output_text=f"{len(response.data)} vector(s)",
            load_seconds=round(load_seconds, 4),
            generate_seconds=round(generate_seconds, 4),
            total_seconds=round(total_seconds, 4),
            usage=usage,
            measurements={
                "input_count": len(inputs),
                "input_characters": sum(len(item) for item in inputs),
                "vector_count": len(response.data),
                "vector_dimensions": vector_dimensions,
            },
            phase_breakdown=self._default_phase_breakdown(
                load_seconds=load_seconds,
                generate_seconds=generate_seconds,
                total_seconds=total_seconds,
            ),
            created_at=utc_now(),
            performance_features=await self._benchmark_performance_features(
                runtime=runtime,
                capability=CapabilityName.EMBEDDINGS.value,
            ),
        )
        self.metadata_store.append_benchmark_record(record.model_dump(mode="json"))
        self._persist_benchmark_probe_records(record)
        return record

    async def _benchmark_rerank_model(
        self,
        *,
        model_id: str | None,
        prompt: str,
        warmup_run_count: int = 1,
    ) -> BenchmarkResult:
        documents = [
            prompt,
            f"{prompt} contrast",
            "unrelated control document",
        ]
        manifest, runtime, routing = self.model_router.route_rerank(
            model_id,
            query=prompt,
            documents=documents,
        )
        response, load_seconds, generate_seconds, total_seconds = await self._execute_benchmark_request(
            manifest=manifest,
            runtime=runtime,
            invoke=lambda: runtime.rerank(
                RerankRequest(
                    model_id=manifest.model_id,
                    query=prompt,
                    documents=documents,
                    top_n=len(documents),
                ),
            ),
        )
        top_document = response.results[0].document if response.results else ""
        record = BenchmarkResult(
            benchmark_id=uuid4().hex,
            model_id=manifest.model_id,
            runtime=runtime.name,
            capability=CapabilityName.RERANK.value,
            reason=routing.reason,
            prompt=prompt,
            output_text=top_document or f"{len(response.results)} result(s)",
            load_seconds=round(load_seconds, 4),
            generate_seconds=round(generate_seconds, 4),
            total_seconds=round(total_seconds, 4),
            measurements={
                "document_count": len(documents),
                "query_characters": len(prompt),
                "document_characters": sum(len(item) for item in documents),
                "result_count": len(response.results),
            },
            phase_breakdown=self._default_phase_breakdown(
                load_seconds=load_seconds,
                generate_seconds=generate_seconds,
                total_seconds=total_seconds,
            ),
            created_at=utc_now(),
            performance_features=await self._benchmark_performance_features(
                runtime=runtime,
                capability=CapabilityName.RERANK.value,
            ),
        )
        self.metadata_store.append_benchmark_record(record.model_dump(mode="json"))
        self._persist_benchmark_probe_records(record)
        return record

    async def _benchmark_audio_transcription_model(
        self,
        *,
        model_id: str | None,
        prompt: str,
        warmup_run_count: int = 1,
    ) -> BenchmarkResult:
        audio_bytes = _benchmark_audio_bytes(prompt)
        file_name = "benchmark-audio.wav"
        manifest, runtime, routing = self.model_router.route_audio_transcription(model_id)
        response, load_seconds, generate_seconds, total_seconds = await self._execute_benchmark_request(
            manifest=manifest,
            runtime=runtime,
            invoke=lambda: runtime.transcribe_audio(
                AudioTranscriptionRequest(
                    model_id=manifest.model_id,
                    audio_bytes=audio_bytes,
                    file_name=file_name,
                    language="en",
                    prompt=prompt,
                    metadata={"source_locator": f"audio:{file_name}"},
                ),
            ),
        )
        record = BenchmarkResult(
            benchmark_id=uuid4().hex,
            model_id=manifest.model_id,
            runtime=runtime.name,
            capability=CapabilityName.AUDIO_TRANSCRIPTION.value,
            reason=routing.reason,
            prompt=prompt,
            output_text=response.text,
            load_seconds=round(load_seconds, 4),
            generate_seconds=round(generate_seconds, 4),
            total_seconds=round(total_seconds, 4),
            measurements={
                "audio_input_bytes": len(audio_bytes),
                "prompt_characters": len(prompt),
                "segment_count": len(response.segments),
                "output_characters": len(response.text),
            },
            phase_breakdown=self._default_phase_breakdown(
                load_seconds=load_seconds,
                generate_seconds=generate_seconds,
                total_seconds=total_seconds,
            ),
            created_at=utc_now(),
            performance_features=await self._benchmark_performance_features(
                runtime=runtime,
                capability=CapabilityName.AUDIO_TRANSCRIPTION.value,
            ),
        )
        self.metadata_store.append_benchmark_record(record.model_dump(mode="json"))
        self._persist_benchmark_probe_records(record)
        return record

    async def _execute_benchmark_request(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        companion_manifests: tuple[ModelManifest, ...] = (),
        invoke: Callable[[], Awaitable[BenchmarkResponseT]],
    ) -> tuple[BenchmarkResponseT, float, float, float]:
        total_start = time.perf_counter()
        load_start = time.perf_counter()
        load_admission = None
        if not runtime.is_model_loaded(manifest.model_id) or any(
            not runtime.is_model_loaded(companion_manifest.model_id)
            for companion_manifest in companion_manifests
        ):
            load_admission = await self.model_load_scheduler.acquire()
        await runtime.load_model(manifest)
        for companion_manifest in companion_manifests:
            await runtime.load_model(companion_manifest)
        await self.runtime_catalog.prepare_runtime_for_request(
            manifest,
            runtime,
            policy=self.settings.runtime_policy,
        )
        load_seconds = time.perf_counter() - load_start
        try:
            generate_start = time.perf_counter()
            response = await invoke()
            generate_seconds = time.perf_counter() - generate_start
        finally:
            await self.runtime_catalog.finalize_runtime_for_request(
                manifest,
                runtime,
                policy=self.settings.runtime_policy,
            )
            if self.settings.runtime_policy == "aggressive_unload":
                for companion_manifest in companion_manifests:
                    await runtime.unload_model(companion_manifest.model_id)
            if load_admission is not None:
                load_admission.release()
        total_seconds = time.perf_counter() - total_start
        return response, load_seconds, generate_seconds, total_seconds

    async def _execute_stream_benchmark_request(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        request: GenerateRequest,
        companion_manifests: tuple[ModelManifest, ...] = (),
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        load_start = time.perf_counter()
        load_admission = None
        if not runtime.is_model_loaded(manifest.model_id) or any(
            not runtime.is_model_loaded(companion_manifest.model_id)
            for companion_manifest in companion_manifests
        ):
            load_admission = await self.model_load_scheduler.acquire()
        await runtime.load_model(manifest)
        for companion_manifest in companion_manifests:
            await runtime.load_model(companion_manifest)
        await self.runtime_catalog.prepare_runtime_for_request(
            manifest,
            runtime,
            policy=self.settings.runtime_policy,
        )
        load_seconds = time.perf_counter() - load_start
        ttft_seconds: float | None = None
        output_chunks: list[str] = []
        try:
            async for delta in runtime.stream_generate(request):
                if ttft_seconds is None:
                    ttft_seconds = round(time.perf_counter() - total_start, 4)
                output_chunks.append(delta)
        finally:
            await self.runtime_catalog.finalize_runtime_for_request(
                manifest,
                runtime,
                policy=self.settings.runtime_policy,
            )
            if self.settings.runtime_policy == "aggressive_unload":
                for companion_manifest in companion_manifests:
                    await runtime.unload_model(companion_manifest.model_id)
            if load_admission is not None:
                load_admission.release()
        return {
            "load_seconds": round(load_seconds, 4),
            "elapsed_seconds": round(time.perf_counter() - total_start, 4),
            "ttft_seconds": ttft_seconds if ttft_seconds is not None else 0.0,
            "output_text": "".join(output_chunks),
        }

    def _benchmark_candidate_model_ids(self, *, capability: str) -> list[str]:
        candidate_model_ids: list[str] = []
        for manifest in self.model_router.model_registry.list_manifests():
            report = self.model_router.model_capability_report(manifest.model_id)
            if any(
                item.capability.value == capability and item.supported
                for item in report.capabilities
            ):
                candidate_model_ids.append(manifest.model_id)
        return candidate_model_ids

    async def _benchmark_performance_features(
        self,
        *,
        runtime: RuntimeContract,
        capability: str,
    ) -> list[PerformanceFeatureStatus]:
        return self._performance_features(
            runtime_health=[await runtime.health_check()],
            request_scheduler=RuntimeSchedulerStats.model_validate(self.runtime_request_scheduler.snapshot()),
            load_scheduler=RuntimeSchedulerStats.model_validate(self.model_load_scheduler.snapshot()),
            request_metrics=RuntimeRequestMetrics.model_validate(self.runtime_metrics_recorder.snapshot()),
            cache_stats=self.cache_stats(),
            capability_focus=capability,
        )

    def _persist_benchmark_probe_records(self, result: BenchmarkResult) -> None:
        host_platform = self.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
        runtime = self.runtime_catalog.find_runtime_by_name(result.runtime)
        runtime_affinity = runtime.affinity.value if runtime is not None else None
        for feature in result.performance_features:
            category = self._measured_category_for_feature_name(feature.feature)
            if category is None:
                continue
            status = self._feature_probe_status(feature)
            if status is None:
                continue
            self.metadata_store.upsert_capability_probe_record(
                category=category.value,
                probe_name=feature.feature.value,
                host_platform=host_platform,
                status=status.value,
                source=MeasuredCapabilityEvidenceSource.BENCHMARK_FEATURE.value,
                reason=feature.reason,
                runtime_name=result.runtime,
                runtime_affinity=runtime_affinity,
                model_id=result.model_id,
                workload_class=result.workload_class,
                recorded_at=result.created_at.isoformat(),
                details={
                    "benchmark_id": result.benchmark_id,
                    "capability": result.capability,
                    "feature": feature.feature.value,
                    "metrics": feature.metrics,
                    "notes": feature.notes,
                },
            )
        for scenario in result.scenarios:
            category = self._measured_category_for_scenario(scenario.scenario)
            if category is None:
                continue
            status = self._scenario_probe_status(category=category, scenario=scenario)
            if status is None:
                continue
            model_id = result.model_id
            runtime_name = result.runtime
            for sample in scenario.samples:
                if sample.model_id is not None:
                    model_id = sample.model_id
                    break
            for sample in scenario.samples:
                if sample.runtime is not None:
                    runtime_name = sample.runtime
                    break
            sample_runtime = self.runtime_catalog.find_runtime_by_name(runtime_name)
            self.metadata_store.upsert_capability_probe_record(
                category=category.value,
                probe_name=scenario.scenario,
                host_platform=host_platform,
                status=status.value,
                source=MeasuredCapabilityEvidenceSource.BENCHMARK_SCENARIO.value,
                reason=scenario.reason,
                runtime_name=runtime_name,
                runtime_affinity=sample_runtime.affinity.value if sample_runtime is not None else runtime_affinity,
                model_id=model_id,
                workload_class=result.workload_class,
                recorded_at=result.created_at.isoformat(),
                details={
                    "benchmark_id": result.benchmark_id,
                    "capability": scenario.capability,
                    "scenario": scenario.scenario,
                    "metrics": scenario.metrics,
                    "notes": scenario.notes,
                },
            )

    @staticmethod
    def _measured_category_for_feature_name(
        feature_name: PerformanceFeatureName,
    ) -> MeasuredCapabilityCategory | None:
        if feature_name == PerformanceFeatureName.CONTINUOUS_BATCHING:
            return MeasuredCapabilityCategory.BATCHING
        if feature_name in {
            PerformanceFeatureName.PREFIX_CACHE,
            PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE,
        }:
            return MeasuredCapabilityCategory.CACHE_REUSE
        if feature_name in {
            PerformanceFeatureName.SPECULATIVE_DECODING,
            PerformanceFeatureName.PROMPT_LOOKUP_SPECULATION,
        }:
            return MeasuredCapabilityCategory.SPECULATION
        if feature_name in {
            PerformanceFeatureName.GRAPH_COMPILATION,
            PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION,
        }:
            return MeasuredCapabilityCategory.COMPILE_KERNELS
        return None

    @staticmethod
    def _measured_category_for_scenario(scenario_name: str) -> MeasuredCapabilityCategory | None:
        if scenario_name == "continuous_batching":
            return MeasuredCapabilityCategory.BATCHING
        if scenario_name in _MEASURED_CACHE_REUSE_SCENARIOS:
            return MeasuredCapabilityCategory.CACHE_REUSE
        if scenario_name == "speculation_selection":
            return MeasuredCapabilityCategory.SPECULATION
        if scenario_name == "constrained_decoding":
            return MeasuredCapabilityCategory.CONSTRAINED_DECODING
        if scenario_name == "mlx_acceleration_paths":
            return MeasuredCapabilityCategory.COMPILE_KERNELS
        return None

    @staticmethod
    def _feature_probe_status(
        feature: PerformanceFeatureStatus,
    ) -> MeasuredCapabilityStatus | None:
        if not feature.supported or not feature.active:
            return None
        if feature.feature in {
            PerformanceFeatureName.GRAPH_COMPILATION,
            PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION,
        }:
            fallback_count = (
                _coerce_int(feature.metrics.get("kernel_fallback_requests"))
                + _coerce_int(feature.metrics.get("compile_fallback_requests"))
            )
            if fallback_count > 0:
                return MeasuredCapabilityStatus.FALLBACK
        return MeasuredCapabilityStatus.SUPPORTED

    @staticmethod
    def _scenario_probe_status(
        *,
        category: MeasuredCapabilityCategory,
        scenario: BenchmarkScenarioReport,
    ) -> MeasuredCapabilityStatus | None:
        if scenario.status == "unsupported":
            return MeasuredCapabilityStatus.REJECTED
        if scenario.status == "not_applicable":
            return MeasuredCapabilityStatus.NOT_APPLICABLE
        if scenario.status != "observed":
            return None
        if category == MeasuredCapabilityCategory.BATCHING:
            if _coerce_int(scenario.metrics.get("runtime_native_batch_request_count")) > 0:
                return MeasuredCapabilityStatus.SUPPORTED
            if _coerce_int(scenario.metrics.get("runtime_stock_single_request_fallback_request_count")) > 0:
                return MeasuredCapabilityStatus.FALLBACK
            return MeasuredCapabilityStatus.REJECTED
        if category == MeasuredCapabilityCategory.CACHE_REUSE:
            if (
                _coerce_int(scenario.metrics.get("attributed_sample_count")) > 0
                or _coerce_int(scenario.metrics.get("cache_hit_sample_count")) > 0
            ):
                return MeasuredCapabilityStatus.SUPPORTED
            return MeasuredCapabilityStatus.REJECTED
        if category == MeasuredCapabilityCategory.SPECULATION:
            if scenario.metrics.get("selected_mode") == "disabled":
                return (
                    MeasuredCapabilityStatus.FALLBACK
                    if _coerce_int(scenario.metrics.get("candidate_count")) > 0
                    else MeasuredCapabilityStatus.REJECTED
                )
            return (
                MeasuredCapabilityStatus.FALLBACK
                if _coerce_int(scenario.metrics.get("fallback_count")) > 0
                else MeasuredCapabilityStatus.SUPPORTED
            )
        if category == MeasuredCapabilityCategory.CONSTRAINED_DECODING:
            if bool(scenario.metrics.get("decoder_enforced")):
                return MeasuredCapabilityStatus.SUPPORTED
            if bool(scenario.metrics.get("fallback_used")) or scenario.metrics.get("enforcement") == "prompt_guided":
                return MeasuredCapabilityStatus.FALLBACK
            if scenario.metrics.get("validation_state") == "valid":
                return MeasuredCapabilityStatus.FALLBACK
            return MeasuredCapabilityStatus.REJECTED
        if category == MeasuredCapabilityCategory.COMPILE_KERNELS:
            compiled_count = (
                _coerce_int(scenario.metrics.get("compiled_sample_count"))
                + _coerce_int(scenario.metrics.get("decode_compiled_sample_count"))
                + _coerce_int(scenario.metrics.get("prefill_compiled_sample_count"))
            )
            fallback_count = _coerce_int(scenario.metrics.get("fallback_sample_count"))
            if compiled_count > 0 and fallback_count > 0:
                return MeasuredCapabilityStatus.FALLBACK
            if compiled_count > 0 or str(scenario.metrics.get("kernel_paths", "")).strip():
                return MeasuredCapabilityStatus.SUPPORTED
            if fallback_count > 0:
                return MeasuredCapabilityStatus.FALLBACK
            return MeasuredCapabilityStatus.REJECTED
        return None

    def _apply_benchmark_request_feature_activity(
        self,
        *,
        performance_features: Sequence[PerformanceFeatureStatus],
        request: GenerateRequest,
        usage: Mapping[str, int],
    ) -> list[PerformanceFeatureStatus]:
        speculation = request.speculation
        if speculation is None:
            return list(performance_features)
        if speculation.mode == SpeculationMode.DRAFT_MODEL:
            return self._override_benchmark_feature_status(
                performance_features=performance_features,
                feature_name=PerformanceFeatureName.SPECULATIVE_DECODING,
                metric_overrides={
                    "request_count": 1,
                    "drafted_tokens": int(usage.get("drafted_tokens", 0)),
                    "verified_tokens": int(usage.get("verified_tokens", 0)),
                    "configured_num_draft_tokens": speculation.num_draft_tokens or 0,
                },
            )
        if speculation.mode == SpeculationMode.PROMPT_LOOKUP:
            return self._override_benchmark_feature_status(
                performance_features=performance_features,
                feature_name=PerformanceFeatureName.PROMPT_LOOKUP_SPECULATION,
                metric_overrides={
                    "request_count": int(usage.get("prompt_lookup_requests", 1)),
                    "configured_max_ngram_size": speculation.prompt_lookup_max_ngram_size or 0,
                    "configured_num_pred_tokens": speculation.prompt_lookup_num_pred_tokens or 0,
                },
            )
        return list(performance_features)

    def _override_benchmark_feature_status(
        self,
        *,
        performance_features: Sequence[PerformanceFeatureStatus],
        feature_name: PerformanceFeatureName,
        metric_overrides: Mapping[str, int | float | str | bool],
    ) -> list[PerformanceFeatureStatus]:
        updated_features: list[PerformanceFeatureStatus] = []
        for feature in performance_features:
            if feature.feature != feature_name:
                updated_features.append(feature)
                continue
            metrics = dict(feature.metrics)
            for key, value in metric_overrides.items():
                metrics[key] = value
            updated_features.append(
                feature.model_copy(
                    update={
                        "active": True,
                        "metrics": metrics,
                    },
                ),
            )
        return updated_features

    def benchmark_summary(self, *, recent_limit: int = 10) -> BenchmarkSummary:
        records = [
            BenchmarkRecord.model_validate(payload)
            for payload in self.metadata_store.list_benchmark_records(limit=recent_limit)
        ]
        if not records:
            return BenchmarkSummary(artifact_summary=self.benchmark_artifact_summary())
        average_total = round(fmean(record.total_seconds for record in records), 4)
        capability_counts: dict[str, int] = {}
        for record in records:
            capability_counts[record.capability] = capability_counts.get(record.capability, 0) + 1
        model_summaries = self._model_benchmark_summaries(records)
        return BenchmarkSummary(
            total_runs=self.metadata_store.benchmark_record_count(),
            last_run_at=max(record.created_at for record in records),
            average_total_seconds=average_total,
            capability_counts=capability_counts,
            recent_runs=records,
            models=model_summaries,
            artifact_summary=self.benchmark_artifact_summary(),
        )

    def benchmark_artifact_summary(self, *, recent_limit: int = 5) -> BenchmarkArtifactSummary:
        artifacts = [
            self._benchmark_artifact_reference(payload)
            for payload in self.metadata_store.list_benchmark_artifacts(limit=recent_limit)
        ]
        return BenchmarkArtifactSummary(
            total_artifacts=self.metadata_store.benchmark_artifact_count(),
            latest_artifact=artifacts[0] if artifacts else None,
            recent_artifacts=artifacts,
        )

    def measured_capability_registry(
        self,
        *,
        manifests: Sequence[ModelManifest] | None = None,
    ) -> MeasuredCapabilityRegistrySummary:
        candidate_manifests = list(manifests) if manifests is not None else self.model_router.model_registry.list_manifests()
        self._record_builtin_capability_probes(candidate_manifests)
        host_platform = self.runtime_catalog.host_platform_snapshot()
        host_payload = host_platform.model_dump(mode="json")
        records = [
            MeasuredCapabilityProbeRecord.model_validate(payload)
            for payload in self.metadata_store.list_capability_probe_records(
                host_platform=host_payload,
                limit=500,
            )
        ]
        latest_recorded_at = max((record.recorded_at for record in records), default=None)
        return MeasuredCapabilityRegistrySummary(
            host_platform=host_platform,
            total_records=self.metadata_store.capability_probe_record_count(host_platform=host_payload),
            latest_recorded_at=latest_recorded_at,
            categories=summarize_measured_capabilities(records),
        )

    def _record_builtin_capability_probes(self, manifests: Sequence[ModelManifest]) -> None:
        host_platform = self.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
        for manifest in manifests:
            if manifest.conversion_status != ConversionStatus.RUNNABLE:
                continue
            try:
                _, runtime, _ = self.model_router.route_chat(
                    manifest.model_id,
                    messages=[GenerateMessage(role="user", content="Measured constrained decoding probe")],
                    max_tokens=8,
                )
            except RoutingError:
                continue
            if self._has_constrained_decoding_benchmark_evidence(
                host_platform=host_platform,
                model_id=manifest.model_id,
                runtime_name=runtime.name,
            ):
                continue
            self.metadata_store.upsert_capability_probe_record(
                category=MeasuredCapabilityCategory.CONSTRAINED_DECODING.value,
                probe_name="prompt_guided_structured_output",
                host_platform=host_platform,
                status=MeasuredCapabilityStatus.REJECTED.value,
                source=MeasuredCapabilityEvidenceSource.CODE_PROBE.value,
                reason=(
                    "LewLM still records structured output as prompt-guided fallback on this host; "
                    "decode-time constrained decoding is not yet benchmark-verified for the routed runtime."
                ),
                runtime_name=runtime.name,
                runtime_affinity=runtime.affinity.value,
                model_id=manifest.model_id,
                details={
                    "enforcement": "prompt_guided_fallback",
                    "capability": CapabilityName.CHAT.value,
                },
            )

    def _has_constrained_decoding_benchmark_evidence(
        self,
        *,
        host_platform: dict[str, Any],
        model_id: str,
        runtime_name: str,
    ) -> bool:
        records = [
            MeasuredCapabilityProbeRecord.model_validate(payload)
            for payload in self.metadata_store.list_capability_probe_records(
                host_platform=host_platform,
                model_id=model_id,
                runtime_name=runtime_name,
                category=MeasuredCapabilityCategory.CONSTRAINED_DECODING.value,
            )
        ]
        return any(record.source == MeasuredCapabilityEvidenceSource.BENCHMARK_SCENARIO for record in records)

    @staticmethod
    def _benchmark_artifact_reference(payload: dict[str, Any]) -> BenchmarkArtifactReference:
        return BenchmarkArtifactReference(
            artifact_id=str(payload["artifact_id"]),
            artifact_path=str(payload["artifact_path"]),
            workload_signature=str(payload["workload_signature"]),
            created_at=payload["created_at"],
            capability=str(payload["capability"]),
            benchmark_count=_coerce_int(payload.get("benchmark_count")),
            model_count=_coerce_int(payload.get("model_count")),
            regression_status=str(payload.get("regression", {}).get("status", "unknown")),
            compared_to_artifact_id=(
                str(payload["regression"]["compared_to_artifact_id"])
                if payload.get("regression", {}).get("compared_to_artifact_id") is not None
                else None
            ),
        )

    async def _persist_benchmark_artifact(
        self,
        *,
        source: str,
        capability: str,
        prompt: str,
        workload_class: str | None,
        repeat_count: int,
        model_ids: list[str],
        result_payload: dict[str, Any],
        scenarios: list[BenchmarkScenarioReport],
        benchmark_count: int,
        model_count: int,
    ) -> tuple[BenchmarkArtifactReference, BenchmarkRegressionSummary]:
        created_at = utc_now()
        artifact_id = uuid4().hex
        workload_signature = self._workload_signature(
            capability=capability,
            prompt=prompt,
            workload_class=workload_class,
            repeat_count=repeat_count,
            model_ids=model_ids,
            benchmark_count=benchmark_count,
            result_payload=result_payload,
            scenarios=scenarios,
        )
        runtime_health = await self.runtime_catalog.health_snapshot()
        request_scheduler = RuntimeSchedulerStats.model_validate(self.runtime_request_scheduler.snapshot())
        load_scheduler = RuntimeSchedulerStats.model_validate(self.model_load_scheduler.snapshot())
        request_metrics = RuntimeRequestMetrics.model_validate(self.runtime_metrics_recorder.snapshot())
        cache_stats = self.cache_stats()
        performance_features = self._performance_features(
            runtime_health=runtime_health,
            request_scheduler=request_scheduler,
            load_scheduler=load_scheduler,
            request_metrics=request_metrics,
            cache_stats=cache_stats,
            capability_focus=capability,
        )
        artifact_path = self._benchmark_artifact_path(
            created_at=created_at,
            workload_signature=workload_signature,
            artifact_id=artifact_id,
        )
        artifact_payload: dict[str, Any] = {
            "artifact_id": artifact_id,
            "artifact_path": str(artifact_path),
            "workload_signature": workload_signature,
            "created_at": created_at.isoformat(),
            "source": source,
            "capability": capability,
            "workload_class": workload_class,
            "prompt": prompt,
            "repeat_count": repeat_count,
            "model_ids": sorted(model_ids),
            "benchmark_count": benchmark_count,
            "model_count": model_count,
            "result": result_payload,
            "scenarios": [scenario.model_dump(mode="json") for scenario in scenarios],
            "runtime_health": runtime_health,
            "request_scheduler": request_scheduler.model_dump(mode="json"),
            "load_scheduler": load_scheduler.model_dump(mode="json"),
            "request_metrics": request_metrics.model_dump(mode="json"),
            "cache_stats": cache_stats.model_dump(mode="json"),
            "performance_features": [item.model_dump(mode="json") for item in performance_features],
        }
        baseline_payload = self.metadata_store.latest_benchmark_artifact(workload_signature=workload_signature)
        regression = self._evaluate_regression(
            current_payload=artifact_payload,
            baseline_payload=baseline_payload,
        )
        artifact_payload["regression"] = regression.model_dump(mode="json")
        artifact_path.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True), encoding="utf-8")
        self.metadata_store.append_benchmark_artifact(artifact_payload)
        return self._benchmark_artifact_reference(artifact_payload), regression

    async def _benchmark_scenarios(
        self,
        *,
        capability: str,
        prompt: str,
        model_ids: list[str],
    ) -> list[BenchmarkScenarioReport]:
        scenarios = [
            await self._request_queue_scenario(
                capability=capability,
                prompt=prompt,
                model_ids=model_ids,
            ),
            await self._load_admission_scenario(
                capability=capability,
                prompt=prompt,
                model_ids=model_ids,
            ),
        ]
        if capability == CapabilityName.CHAT.value:
            scenarios.insert(
                0,
                await self._repeated_prefix_scenario(prompt=prompt, model_ids=model_ids),
            )
            scenarios.insert(
                1,
                await self._continuous_batching_scenario(prompt=prompt, model_ids=model_ids),
            )
            scenarios.insert(
                2,
                await self._mixed_prefill_queue_scenario(prompt=prompt, model_ids=model_ids),
            )
            scenarios.insert(
                3,
                await self._warm_chat_cache_scenario(prompt=prompt, model_ids=model_ids),
            )
            scenarios.insert(
                4,
                await self._mlx_acceleration_scenario(prompt=prompt, model_ids=model_ids),
            )
            scenarios.insert(
                5,
                await self._frontier_architecture_scenario(prompt=prompt, model_ids=model_ids),
            )
            scenarios.append(
                await self._multimodal_encoder_reuse_scenario(
                    capability=capability,
                    prompt=prompt,
                    model_ids=model_ids,
                ),
            )
        if capability == CapabilityName.AUDIO_TRANSCRIPTION.value:
            scenarios.append(
                await self._multimodal_encoder_reuse_scenario(
                    capability=capability,
                    prompt=prompt,
                    model_ids=model_ids,
                ),
            )
        if capability in {CapabilityName.EMBEDDINGS.value, CapabilityName.RERANK.value}:
            scenarios.append(
                await self._multimodal_reuse_scenario(
                    capability=capability,
                    prompt=prompt,
                    model_ids=model_ids,
                ),
            )
        return scenarios

    async def _frontier_architecture_scenario(
        self,
        *,
        prompt: str,
        model_ids: list[str],
    ) -> BenchmarkScenarioReport:
        if not model_ids:
            return BenchmarkScenarioReport(
                scenario="frontier_architecture_modes",
                capability=CapabilityName.CHAT.value,
                status="not_applicable",
                reason="No chat-capable benchmark model was selected for frontier architecture planning.",
                metrics=_compact_metrics(prompt_characters=len(prompt)),
            )
        manifests: list[ModelManifest] = []
        for model_id in model_ids:
            manifest = self.model_router.model_registry.get_manifest(model_id)
            if manifest.architecture_subtype.value in {"ssm_mamba", "hybrid_ssm", "moe", "hybrid_moe"}:
                manifests.append(manifest)
        samples: list[BenchmarkScenarioSample] = []
        notes: list[str] = []
        total_expert_count = 0
        for manifest in manifests:
            plan = build_frontier_serving_plan(manifest=manifest, settings=self.settings)
            if plan is None:
                continue
            total_expert_count += _coerce_int(plan.get("expert_count")) or 0
            samples.append(
                BenchmarkScenarioSample(
                    model_id=manifest.model_id,
                    metrics=_compact_metrics(
                        architecture_subtype=plan.get("architecture_subtype"),
                        bounded_memory_mode=plan.get("bounded_memory_mode"),
                        cache_state_handling=plan.get("cache_state_handling"),
                        full_estimated_memory_mb=plan.get("full_estimated_memory_mb"),
                        planned_memory_mb=plan.get("planned_memory_mb"),
                        memory_savings_mb=plan.get("memory_savings_mb"),
                        expert_count=plan.get("expert_count"),
                        resident_expert_count=plan.get("resident_expert_count"),
                        streamed_expert_count=plan.get("streamed_expert_count"),
                        estimated_swap_mb_per_request=plan.get("estimated_swap_mb_per_request"),
                    ),
                ),
            )
            for note in frontier_plan_notes(plan):
                if note not in notes:
                    notes.append(note)
        if not samples:
            return BenchmarkScenarioReport(
                scenario="frontier_architecture_modes",
                capability=CapabilityName.CHAT.value,
                status="not_applicable",
                reason="No selected benchmark model was classified as a hybrid SSM or MoE architecture.",
                metrics=_compact_metrics(prompt_characters=len(prompt), model_count=len(model_ids)),
            )
        hybrid_ssm_model_count = sum(
            1
            for sample in samples
            if sample.metrics.get("architecture_subtype") in {"ssm_mamba", "hybrid_ssm"}
        )
        moe_model_count = sum(
            1
            for sample in samples
            if sample.metrics.get("architecture_subtype") in {"moe", "hybrid_moe"}
        )
        return BenchmarkScenarioReport(
            scenario="frontier_architecture_modes",
            capability=CapabilityName.CHAT.value,
            feature=(
                PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING
                if moe_model_count > 0
                else PerformanceFeatureName.HYBRID_SSM_ROUTING
            ),
            status="observed",
            reason=(
                "Recorded architecture-aware serving plans for hybrid SSM and MoE models so benchmark artifacts disclose bounded-memory strategy, cache-state handling, and tradeoff notes even before backend-native fast paths land."
            ),
            metrics=_compact_metrics(
                prompt_characters=len(prompt),
                sample_count=len(samples),
                hybrid_ssm_model_count=hybrid_ssm_model_count,
                moe_model_count=moe_model_count,
                total_expert_count=total_expert_count,
                configured_moe_bounded_memory_mode=self.settings.moe_bounded_memory_mode,
                configured_moe_resident_expert_count=self.settings.moe_resident_expert_count,
            ),
            samples=samples,
            notes=notes,
        )

    async def _mlx_acceleration_scenario(
        self,
        *,
        prompt: str,
        model_ids: list[str],
    ) -> BenchmarkScenarioReport:
        return BenchmarkScenarioReport(
            scenario="mlx_acceleration",
            capability=CapabilityName.CHAT.value,
            status="not_applicable",
            reason=(
                "Kernel and compile acceleration benchmarking stays reserved for the dedicated MLX acceleration track."
            ),
            metrics=_compact_metrics(
                prompt_characters=len(prompt),
                model_count=len(model_ids),
            ),
        )

    async def _continuous_batching_scenario(
        self,
        *,
        prompt: str,
        model_ids: list[str],
    ) -> BenchmarkScenarioReport:
        if not model_ids:
            return BenchmarkScenarioReport(
                scenario="continuous_batching",
                capability=CapabilityName.CHAT.value,
                feature=PerformanceFeatureName.CONTINUOUS_BATCHING,
                status="not_applicable",
                reason="No chat-capable benchmark model was selected for the continuous-batching sample.",
            )
        selected_model_id = model_ids[0]
        manifest, runtime, _ = self.model_router.route_chat(
            selected_model_id,
            messages=[GenerateMessage(role="user", content=prompt)],
            max_tokens=48,
        )
        if not runtime.supports_continuous_batching(CapabilityName.CHAT):
            return BenchmarkScenarioReport(
                scenario="continuous_batching",
                capability=CapabilityName.CHAT.value,
                feature=PerformanceFeatureName.CONTINUOUS_BATCHING,
                status="unsupported",
                reason=(
                    f"Selected runtime `{runtime.name}` does not advertise a backend-native batched chat "
                    "generation entrypoint."
                ),
            )
        concurrency = max(2, min(self.settings.continuous_batch_max_batch_size, 4))
        candidate_services = None
        telemetry = self
        multimodal_source_path: Path | None = None
        if any(modality in {ModelModality.VISION, ModelModality.MULTIMODAL} for modality in manifest.modality):
            multimodal_source_path = _ensure_benchmark_multimodal_assets(self.settings.benchmarks_dir)["image"]
        if self.service_factory is not None and self.settings.speculative_decoding_enabled:
            candidate_services = self.service_factory(self.settings.with_updates(speculative_decoding_enabled=False))
            self._sync_benchmark_child_manifests(candidate_services)
            telemetry = candidate_services.telemetry_service
        try:
            before = telemetry.runtime_request_scheduler.snapshot()
            if multimodal_source_path is None:
                single_request = await telemetry._timed_orchestrated_chat_request(
                    model_id=manifest.model_id,
                    prompt=f"{prompt} [single-latency]",
                )
            else:
                single_request = await telemetry._timed_multimodal_chat_request(
                    model_id=manifest.model_id,
                    prompt=f"{prompt} [single-latency]",
                    source_path=multimodal_source_path,
                    services=candidate_services,
                    use_cli_normalization=True,
                )
            started_at = time.perf_counter()
            if multimodal_source_path is None:
                concurrent_runs = await asyncio.gather(
                    *[
                        telemetry._timed_orchestrated_chat_request(
                            model_id=manifest.model_id,
                            prompt=f"{prompt} [batch-{index}]",
                        )
                        for index in range(concurrency)
                    ],
                )
            else:
                concurrent_runs = await asyncio.gather(
                    *[
                        telemetry._timed_multimodal_chat_request(
                            model_id=manifest.model_id,
                            prompt=f"{prompt} [batch-{index}]",
                            source_path=multimodal_source_path,
                            services=candidate_services,
                            use_cli_normalization=True,
                        )
                        for index in range(concurrency)
                    ],
                )
            aggregate_elapsed_seconds = round(time.perf_counter() - started_at, 4)
            after = telemetry.runtime_request_scheduler.snapshot()
        finally:
            if candidate_services is not None:
                await candidate_services.aclose()
        throughput_requests_per_second = (
            round(len(concurrent_runs) / aggregate_elapsed_seconds, 4)
            if aggregate_elapsed_seconds > 0
            else None
        )
        batch_backend_names: set[str] = set()
        fallback_reasons: list[str] = []
        runtime_native_batch_request_count = 0
        runtime_stock_single_request_fallback_request_count = 0
        for item in concurrent_runs:
            request_metadata = item.get("request_metadata")
            if not isinstance(request_metadata, dict):
                continue
            native_batching = request_metadata.get("native_batching")
            if not isinstance(native_batching, dict):
                continue
            if native_batching.get("active") is True:
                runtime_native_batch_request_count += 1
                backend = native_batching.get("backend")
                if isinstance(backend, str) and backend:
                    batch_backend_names.add(backend)
            if native_batching.get("stock_single_request_path") is True:
                runtime_stock_single_request_fallback_request_count += 1
                reason = native_batching.get("fallback_reason")
                if isinstance(reason, str) and reason:
                    fallback_reasons.append(reason)
        notes: list[str] = []
        if multimodal_source_path is not None:
            notes.append(
                "Used a repeated benchmark image attachment so the continuous-batching sample stays multimodal instead of silently collapsing to a text-only burst.",
            )
        for reason in dict.fromkeys(fallback_reasons):
            notes.append(f"Stock single-request path reason: {reason}")
        if runtime_native_batch_request_count > 0:
            scenario_reason = (
                "Measured one single-request run plus a concurrent same-model burst so the artifact records both "
                "single-request latency and aggregate throughput, including multimodal attachment-bearing batches "
                "when the selected model supports them."
                if multimodal_source_path is not None
                else "Measured one single-request chat run plus a concurrent same-model chat burst so the artifact "
                "records both single-request latency and aggregate batched throughput."
            )
        elif runtime_stock_single_request_fallback_request_count > 0:
            scenario_reason = (
                "Measured one single-request run plus a concurrent same-model burst, but the runtime stayed on its "
                "stock single-request path for the concurrent phase."
            )
        else:
            scenario_reason = (
                "Measured one single-request run plus a concurrent same-model burst so the artifact records both "
                "single-request latency and aggregate throughput."
            )
        return BenchmarkScenarioReport(
            scenario="continuous_batching",
            capability=CapabilityName.CHAT.value,
            feature=PerformanceFeatureName.CONTINUOUS_BATCHING,
            status="observed",
            reason=scenario_reason,
            metrics=_compact_metrics(
                model_id=manifest.model_id,
                runtime=runtime.name,
                concurrency=concurrency,
                request_shape="single_image" if multimodal_source_path is not None else "text_only",
                single_request_elapsed_seconds=single_request["elapsed_seconds"],
                aggregate_elapsed_seconds=aggregate_elapsed_seconds,
                throughput_requests_per_second=throughput_requests_per_second,
                native_batch_count_delta=after["native_total_batches"] - before["native_total_batches"],
                native_batched_request_delta=after["native_batched_requests"] - before["native_batched_requests"],
                frontier_batch_count_delta=after["frontier_total_batches"] - before["frontier_total_batches"],
                frontier_batched_request_delta=(
                    after["frontier_batched_requests"] - before["frontier_batched_requests"]
                ),
                native_average_batch_size=after["native_average_batch_size"],
                native_average_batch_utilization=after["native_average_batch_utilization"],
                native_average_queue_delay_seconds=after["native_average_queue_delay_seconds"],
                average_batch_size=after["frontier_average_batch_size"],
                average_batch_utilization=after["frontier_average_batch_utilization"],
                average_queue_delay_seconds=after["frontier_average_queue_delay_seconds"],
                runtime_native_batch_request_count=runtime_native_batch_request_count,
                runtime_stock_single_request_fallback_request_count=(
                    runtime_stock_single_request_fallback_request_count
                ),
                runtime_native_batch_backend=",".join(sorted(batch_backend_names)) if batch_backend_names else None,
            ),
            samples=[
                BenchmarkScenarioSample(
                    model_id=str(item["model_id"]),
                    runtime=str(item["runtime"]),
                    metrics=_compact_metrics(
                        elapsed_seconds=item["elapsed_seconds"],
                        native_batch_active=(
                            isinstance(item.get("request_metadata"), dict)
                            and isinstance(item["request_metadata"].get("native_batching"), dict)
                            and item["request_metadata"]["native_batching"].get("active") is True
                        ),
                        stock_single_request_path=(
                            isinstance(item.get("request_metadata"), dict)
                            and isinstance(item["request_metadata"].get("native_batching"), dict)
                            and item["request_metadata"]["native_batching"].get("stock_single_request_path") is True
                        ),
                    ),
                )
                for item in concurrent_runs
            ],
            notes=notes,
        )

    async def _mixed_prefill_queue_scenario(
        self,
        *,
        prompt: str,
        model_ids: list[str],
    ) -> BenchmarkScenarioReport:
        if not model_ids:
            return BenchmarkScenarioReport(
                scenario="mixed_prefill_queue",
                capability=CapabilityName.CHAT.value,
                feature=PerformanceFeatureName.DECODE_PRIORITY_SCHEDULING,
                status="not_applicable",
                reason="No chat-capable benchmark model was selected for the mixed short/long workload sample.",
            )
        selected_model_id = model_ids[0]
        manifest, runtime, _ = self.model_router.route_chat(
            selected_model_id,
            messages=[GenerateMessage(role="user", content=prompt)],
            max_tokens=48,
        )
        if not runtime.supports_capability(CapabilityName.STREAMING):
            return BenchmarkScenarioReport(
                scenario="mixed_prefill_queue",
                capability=CapabilityName.CHAT.value,
                feature=PerformanceFeatureName.DECODE_PRIORITY_SCHEDULING,
                status="unsupported",
                reason=(
                    f"Selected runtime `{runtime.name}` does not expose streaming generation, so LewLM cannot capture mixed-workload TTFT under scheduler pressure."
                ),
            )

        long_prompt = self._mixed_prefill_prompt(prompt)
        short_request_count = max(3, min(self.settings.continuous_batch_max_batch_size, 5))

        await self._timed_orchestrated_chat_request(model_id=manifest.model_id, prompt=f"{prompt} [warmup]")
        baseline_short = await self._timed_orchestrated_stream_request(
            model_id=manifest.model_id,
            prompt=f"{prompt} [baseline-short]",
        )

        before = self.runtime_request_scheduler.snapshot()
        long_task = asyncio.create_task(
            self._timed_orchestrated_stream_request(
                model_id=manifest.model_id,
                prompt=f"{long_prompt}\n[long-prefill]",
                max_tokens=96,
            ),
        )
        await asyncio.sleep(0.05)
        short_runs = await asyncio.gather(
            *[
                self._timed_orchestrated_stream_request(
                    model_id=manifest.model_id,
                    prompt=f"{prompt} [interactive-{index}]",
                )
                for index in range(short_request_count)
            ],
        )
        long_run = await long_task
        after = self.runtime_request_scheduler.snapshot()

        short_ttft_values = [
            _coerce_float(run.get("ttft_seconds"))
            for run in short_runs
            if _coerce_float(run.get("ttft_seconds")) is not None
        ]
        short_inter_token_values = [
            _coerce_float(run.get("inter_token_seconds"))
            for run in short_runs
            if _coerce_float(run.get("inter_token_seconds")) is not None
        ]
        decode_priority_active_count = sum(
            1 for run in short_runs if self._request_scheduling_flag(run, "decode_priority_active")
        )
        chunked_prefill_active = self._request_scheduling_flag(long_run, "chunked_prefill_active")
        prefill_isolation_active = self._request_scheduling_flag(long_run, "prefill_isolation_active")
        scenario_notes: list[str] = []
        if not self.settings.decode_priority_scheduling_enabled:
            scenario_notes.append("Decode-priority scheduling is disabled in settings, so interactive requests remain FIFO within scheduler capacity.")
        if not runtime.supports_chunked_prefill(CapabilityName.CHAT):
            scenario_notes.append(f"Runtime `{runtime.name}` does not advertise chunked prefill, so long prompts still enter as one backend prefill phase.")
        if not self.settings.prefill_isolation_enabled:
            scenario_notes.append("Prefill isolation is disabled in settings; scheduler metrics still disclose whether long-prefill requests were detected.")
        elif not runtime.supports_prefill_isolation(CapabilityName.CHAT):
            scenario_notes.append(f"Runtime `{runtime.name}` does not advertise the hooks LewLM requires for truthful single-host prefill isolation.")

        return BenchmarkScenarioReport(
            scenario="mixed_prefill_queue",
            capability=CapabilityName.CHAT.value,
            feature=PerformanceFeatureName.DECODE_PRIORITY_SCHEDULING,
            status="observed",
            reason=(
                "Measured one long-prefill streaming request alongside several short interactive streams so benchmark artifacts disclose scheduler lane selection, decode-priority grants, and chunked-prefill activity under mixed load."
            ),
            metrics=_compact_metrics(
                model_id=manifest.model_id,
                runtime=runtime.name,
                short_request_count=short_request_count,
                baseline_short_ttft_seconds=_coerce_float(baseline_short.get("ttft_seconds")),
                mixed_short_ttft_p95_seconds=_percentile(short_ttft_values, 0.95),
                mixed_short_ttft_p99_seconds=_percentile(short_ttft_values, 0.99),
                mixed_short_average_inter_token_seconds=(
                    round(fmean(short_inter_token_values), 4) if short_inter_token_values else None
                ),
                mixed_short_max_inter_token_seconds=max(short_inter_token_values) if short_inter_token_values else None,
                long_elapsed_seconds=_coerce_float(long_run.get("elapsed_seconds")),
                long_ttft_seconds=_coerce_float(long_run.get("ttft_seconds")),
                long_prompt_token_estimate=self._request_scheduling_metric(long_run, "prompt_token_estimate"),
                long_prefill_chunk_count=self._request_scheduling_metric(long_run, "chunk_count"),
                long_queue_lane=self._request_scheduling_text(long_run, "queue_lane"),
                chunked_prefill_active=chunked_prefill_active,
                prefill_isolation_active=prefill_isolation_active,
                decode_priority_active_count=decode_priority_active_count,
                prioritized_decode_grants_delta=after["prioritized_decode_grants"] - before["prioritized_decode_grants"],
                isolated_prefill_requests_delta=after["isolated_prefill_requests"] - before["isolated_prefill_requests"],
                decode_priority_requests_delta=after["decode_priority_requests"] - before["decode_priority_requests"],
                prefill_heavy_requests_delta=after["prefill_heavy_requests"] - before["prefill_heavy_requests"],
                max_observed_decode_queue_depth_delta=(
                    after["max_observed_decode_queue_depth"] - before["max_observed_decode_queue_depth"]
                ),
                max_observed_prefill_queue_depth_delta=(
                    after["max_observed_prefill_queue_depth"] - before["max_observed_prefill_queue_depth"]
                ),
            ),
            samples=[
                BenchmarkScenarioSample(
                    model_id=str(long_run["model_id"]),
                    runtime=str(long_run["runtime"]),
                    metrics=_compact_metrics(
                        request_type="long_prefill",
                        elapsed_seconds=long_run["elapsed_seconds"],
                        ttft_seconds=long_run["ttft_seconds"],
                        inter_token_seconds=long_run["inter_token_seconds"],
                        prompt_token_estimate=self._request_scheduling_metric(long_run, "prompt_token_estimate"),
                        queue_lane=self._request_scheduling_text(long_run, "queue_lane"),
                        chunked_prefill_active=chunked_prefill_active,
                        prefill_isolation_active=prefill_isolation_active,
                    ),
                ),
                *[
                    BenchmarkScenarioSample(
                        model_id=str(run["model_id"]),
                        runtime=str(run["runtime"]),
                        metrics=_compact_metrics(
                            request_type="interactive_short",
                            elapsed_seconds=run["elapsed_seconds"],
                            ttft_seconds=run["ttft_seconds"],
                            inter_token_seconds=run["inter_token_seconds"],
                            delta_count=run["delta_count"],
                            decode_priority_active=self._request_scheduling_flag(run, "decode_priority_active"),
                            queue_lane=self._request_scheduling_text(run, "queue_lane"),
                        ),
                    )
                    for run in short_runs
                ],
            ],
            notes=scenario_notes,
        )

    async def _repeated_prefix_scenario(
        self,
        *,
        prompt: str,
        model_ids: list[str],
    ) -> BenchmarkScenarioReport:
        if not model_ids:
            return BenchmarkScenarioReport(
                scenario="repeated_prefix",
                capability=CapabilityName.CHAT.value,
                feature=PerformanceFeatureName.PREFIX_CACHE,
                status="not_applicable",
                reason="No chat-capable benchmark model was selected for the repeated-prefix sample.",
            )
        prefix = (
            "Shared benchmark prefix for prompt-reuse checks. "
            "Return only the requested token and do not add explanation. "
        ) * 24
        samples: list[BenchmarkScenarioSample] = []
        for index, model_id in enumerate(model_ids):
            first_run = await self._timed_chat_request(
                model_id=model_id,
                prompt=f"{prefix}\nVariant {index}: reply with ALPHA only.",
            )
            second_run = await self._timed_chat_request(
                model_id=model_id,
                prompt=f"{prefix}\nVariant {index}: reply with BETA only.",
            )
            samples.append(
                BenchmarkScenarioSample(
                    model_id=model_id,
                    runtime=first_run["runtime"],
                    metrics=_compact_metrics(
                        first_elapsed_seconds=first_run["elapsed_seconds"],
                        second_elapsed_seconds=second_run["elapsed_seconds"],
                        second_over_first_ratio=(
                            round(second_run["elapsed_seconds"] / first_run["elapsed_seconds"], 4)
                            if first_run["elapsed_seconds"] > 0
                            else None
                        ),
                        first_cached_tokens=self._request_prefix_metric(first_run, "cached_tokens"),
                        second_cached_tokens=self._request_prefix_metric(second_run, "cached_tokens"),
                        second_cached_pages=self._request_prefix_metric(second_run, "cached_pages"),
                        second_saved_prefill_tokens=self._request_prefix_metric(second_run, "saved_prefill_tokens"),
                        second_cache_hits=self._request_prefix_metric(second_run, "cache_hits"),
                        second_resident_page_hits=self._request_prefix_metric(second_run, "resident_page_hits"),
                        second_copy_on_write_reused_pages=self._request_prefix_metric(
                            second_run,
                            "copy_on_write_reused_pages",
                        ),
                        second_lookup_source=self._request_prefix_text(second_run, "lookup_source"),
                        second_prefilled_uncached_tokens=self._request_prefix_metric(second_run, "prefilled_uncached_tokens"),
                        effective_prefill_mode=self._request_control_text(second_run, "prefill_optimization", "effective"),
                        effective_prefill_token_batch_size=self._request_control_metric(
                            second_run,
                            "prefill_optimization",
                            "effective_prefill_token_batch_size",
                        ),
                        effective_paged_kv_mode=self._request_control_text(second_run, "paged_kv_cache", "effective"),
                        effective_quantized_kv_mode=self._request_control_text(
                            second_run,
                            "kv_cache_quantization",
                            "effective",
                        ),
                        shared_prefix_characters=len(prefix),
                    ),
                ),
            )
        first_values = [
            _coerce_float(sample.metrics.get("first_elapsed_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("first_elapsed_seconds")) is not None
        ]
        second_values = [
            _coerce_float(sample.metrics.get("second_elapsed_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("second_elapsed_seconds")) is not None
        ]
        ratio_values = [
            _coerce_float(sample.metrics.get("second_over_first_ratio"))
            for sample in samples
            if _coerce_float(sample.metrics.get("second_over_first_ratio")) is not None
        ]
        second_cached_tokens = [
            _coerce_int(sample.metrics.get("second_cached_tokens"))
            for sample in samples
            if _coerce_int(sample.metrics.get("second_cached_tokens")) > 0
        ]
        second_saved_prefill_tokens = [
            _coerce_int(sample.metrics.get("second_saved_prefill_tokens"))
            for sample in samples
            if _coerce_int(sample.metrics.get("second_saved_prefill_tokens")) > 0
        ]
        second_cached_pages = [
            _coerce_int(sample.metrics.get("second_cached_pages"))
            for sample in samples
            if _coerce_int(sample.metrics.get("second_cached_pages")) > 0
        ]
        copy_on_write_reused_pages = [
            _coerce_int(sample.metrics.get("second_copy_on_write_reused_pages"))
            for sample in samples
            if _coerce_int(sample.metrics.get("second_copy_on_write_reused_pages")) > 0
        ]
        cache_hit_sample_count = sum(1 for sample in samples if _coerce_int(sample.metrics.get("second_cache_hits")) > 0)
        attributed_sample_count = sum(
            1
            for sample in samples
            if _coerce_int(sample.metrics.get("second_cached_tokens")) > 0
            or _coerce_int(sample.metrics.get("second_saved_prefill_tokens")) > 0
        )
        return BenchmarkScenarioReport(
            scenario="repeated_prefix",
            capability=CapabilityName.CHAT.value,
            feature=PerformanceFeatureName.PREFIX_CACHE,
            status="observed",
            reason=(
                "Measured repeated shared-prefix chat latency and captured runtime-reported cached-token and prefill-reuse "
                "metadata so artifacts can attribute warm-path improvements to concrete cache activity."
            ),
            metrics=_compact_metrics(
                sample_count=len(samples),
                shared_prefix_characters=len(prefix),
                average_first_elapsed_seconds=round(fmean(first_values), 4) if first_values else None,
                average_second_elapsed_seconds=round(fmean(second_values), 4) if second_values else None,
                average_second_over_first_ratio=round(fmean(ratio_values), 4) if ratio_values else None,
                total_second_cached_tokens=sum(second_cached_tokens),
                total_second_cached_pages=sum(second_cached_pages),
                total_second_saved_prefill_tokens=sum(second_saved_prefill_tokens),
                total_copy_on_write_reused_pages=sum(copy_on_write_reused_pages),
                cache_hit_sample_count=cache_hit_sample_count,
                attributed_sample_count=attributed_sample_count,
            ),
            samples=samples,
            notes=[
                "Prefix-cache support remains runtime-dependent; this scenario records both timing and runtime-reported reuse metadata.",
                (
                    "No repeated-prefix sample reported cached tokens or saved prefill tokens, so any timing delta should be treated as non-attributed warm-state noise."
                    if attributed_sample_count == 0
                    else "Timing deltas are only treated as attributed when the runtime reports cached tokens or saved prefill work on the second request."
                ),
            ],
        )

    async def _warm_chat_cache_scenario(
        self,
        *,
        prompt: str,
        model_ids: list[str],
    ) -> BenchmarkScenarioReport:
        if not model_ids:
            return BenchmarkScenarioReport(
                scenario="warm_chat_cache",
                capability=CapabilityName.CHAT.value,
                feature=PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE,
                status="not_applicable",
                reason="No chat-capable benchmark model was selected for the warm-cache sample.",
            )
        eligible_model_ids: list[str] = []
        for model_id in model_ids:
            try:
                _, runtime, _ = self.model_router.route_chat(
                    model_id,
                    messages=[GenerateMessage(role="user", content=prompt)],
                    max_tokens=48,
                )
            except Exception:
                continue
            runtime_health = await runtime.health_check()
            performance_features = runtime_health.get("performance_features", {})
            if not isinstance(performance_features, dict):
                continue
            feature_payload = performance_features.get(PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE.value, {})
            if isinstance(feature_payload, dict) and bool(feature_payload.get("supported")):
                eligible_model_ids.append(model_id)
        if not eligible_model_ids:
            return BenchmarkScenarioReport(
                scenario="warm_chat_cache",
                capability=CapabilityName.CHAT.value,
                feature=PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE,
                status="not_applicable",
                reason="No selected chat runtime advertised restart-resilient multi-context cache support.",
            )
        prefix = (
            "Warm-cache benchmark prefix for restart-safe prompt reuse checks. "
            "Reply with the requested token only and do not add explanation. "
        ) * 24
        samples: list[BenchmarkScenarioSample] = []
        for index, model_id in enumerate(eligible_model_ids):
            cold_run = await self._timed_stream_chat_request(
                model_id=model_id,
                prompt=f"{prefix}\nVariant {index}: reply with COLD only.",
            )
            warm_run = await self._timed_stream_chat_request(
                model_id=model_id,
                prompt=f"{prefix}\nVariant {index}: reply with WARM only.",
            )
            samples.append(
                BenchmarkScenarioSample(
                    model_id=model_id,
                    runtime=cold_run["runtime"],
                    metrics=_compact_metrics(
                        cold_ttft_seconds=cold_run["ttft_seconds"],
                        warm_ttft_seconds=warm_run["ttft_seconds"],
                        warm_over_cold_ttft_ratio=(
                            round(warm_run["ttft_seconds"] / cold_run["ttft_seconds"], 4)
                            if cold_run["ttft_seconds"] > 0
                            else None
                        ),
                        cold_elapsed_seconds=cold_run["elapsed_seconds"],
                        warm_elapsed_seconds=warm_run["elapsed_seconds"],
                        warm_cached_tokens=self._request_prefix_metric(warm_run, "cached_tokens"),
                        warm_cached_pages=self._request_prefix_metric(warm_run, "cached_pages"),
                        warm_saved_prefill_tokens=self._request_prefix_metric(warm_run, "saved_prefill_tokens"),
                        warm_persistent_cache_hits=self._request_prefix_metric(warm_run, "persistent_cache_hits"),
                        warm_persistent_page_hits=self._request_prefix_metric(warm_run, "persistent_page_hits"),
                        warm_cache_restores=self._request_prefix_metric(warm_run, "cache_restores"),
                        warm_restored_pages=self._request_prefix_metric(warm_run, "restored_pages"),
                        warm_lookup_source=self._request_prefix_text(warm_run, "lookup_source"),
                        warm_prefilled_uncached_tokens=self._request_prefix_metric(warm_run, "prefilled_uncached_tokens"),
                        warm_prefill_mode=self._request_control_text(warm_run, "prefill_optimization", "effective"),
                        shared_prefix_characters=len(prefix),
                    ),
                ),
            )
        cold_ttft_values = [
            _coerce_float(sample.metrics.get("cold_ttft_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("cold_ttft_seconds")) is not None
        ]
        warm_ttft_values = [
            _coerce_float(sample.metrics.get("warm_ttft_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("warm_ttft_seconds")) is not None
        ]
        ttft_ratio_values = [
            _coerce_float(sample.metrics.get("warm_over_cold_ttft_ratio"))
            for sample in samples
            if _coerce_float(sample.metrics.get("warm_over_cold_ttft_ratio")) is not None
        ]
        cold_elapsed_values = [
            _coerce_float(sample.metrics.get("cold_elapsed_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("cold_elapsed_seconds")) is not None
        ]
        warm_elapsed_values = [
            _coerce_float(sample.metrics.get("warm_elapsed_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("warm_elapsed_seconds")) is not None
        ]
        warm_cached_tokens = [
            _coerce_int(sample.metrics.get("warm_cached_tokens"))
            for sample in samples
            if _coerce_int(sample.metrics.get("warm_cached_tokens")) > 0
        ]
        warm_saved_prefill_tokens = [
            _coerce_int(sample.metrics.get("warm_saved_prefill_tokens"))
            for sample in samples
            if _coerce_int(sample.metrics.get("warm_saved_prefill_tokens")) > 0
        ]
        warm_cached_pages = [
            _coerce_int(sample.metrics.get("warm_cached_pages"))
            for sample in samples
            if _coerce_int(sample.metrics.get("warm_cached_pages")) > 0
        ]
        persistent_cache_hits = [
            _coerce_int(sample.metrics.get("warm_persistent_cache_hits"))
            for sample in samples
            if _coerce_int(sample.metrics.get("warm_persistent_cache_hits")) > 0
        ]
        persistent_page_hits = [
            _coerce_int(sample.metrics.get("warm_persistent_page_hits"))
            for sample in samples
            if _coerce_int(sample.metrics.get("warm_persistent_page_hits")) > 0
        ]
        cache_restores = [
            _coerce_int(sample.metrics.get("warm_cache_restores"))
            for sample in samples
            if _coerce_int(sample.metrics.get("warm_cache_restores")) > 0
        ]
        restored_pages = [
            _coerce_int(sample.metrics.get("warm_restored_pages"))
            for sample in samples
            if _coerce_int(sample.metrics.get("warm_restored_pages")) > 0
        ]
        attributed_sample_count = sum(
            1
            for sample in samples
            if _coerce_int(sample.metrics.get("warm_cached_tokens")) > 0
            or _coerce_int(sample.metrics.get("warm_cache_restores")) > 0
            or _coerce_int(sample.metrics.get("warm_persistent_cache_hits")) > 0
        )
        return BenchmarkScenarioReport(
            scenario="warm_chat_cache",
            capability=CapabilityName.CHAT.value,
            feature=PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE,
            status="observed",
            reason=(
                "Measured cold versus warm streaming first-token latency on a shared chat prefix and captured runtime-reported "
                "cache restore metadata so persisted multi-context cache wins can be attributed in the artifact."
            ),
            metrics=_compact_metrics(
                sample_count=len(samples),
                shared_prefix_characters=len(prefix),
                average_cold_ttft_seconds=round(fmean(cold_ttft_values), 4) if cold_ttft_values else None,
                average_warm_ttft_seconds=round(fmean(warm_ttft_values), 4) if warm_ttft_values else None,
                average_warm_over_cold_ttft_ratio=round(fmean(ttft_ratio_values), 4) if ttft_ratio_values else None,
                average_cold_elapsed_seconds=round(fmean(cold_elapsed_values), 4) if cold_elapsed_values else None,
                average_warm_elapsed_seconds=round(fmean(warm_elapsed_values), 4) if warm_elapsed_values else None,
                total_warm_cached_tokens=sum(warm_cached_tokens),
                total_warm_cached_pages=sum(warm_cached_pages),
                total_warm_saved_prefill_tokens=sum(warm_saved_prefill_tokens),
                total_persistent_cache_hits=sum(persistent_cache_hits),
                total_persistent_page_hits=sum(persistent_page_hits),
                total_cache_restores=sum(cache_restores),
                total_restored_pages=sum(restored_pages),
                attributed_sample_count=attributed_sample_count,
            ),
            samples=samples,
            notes=[
                "This scenario uses streaming TTFT because end-to-end completion timing can hide whether prompt reuse improved the first token.",
                (
                    "No warm-cache sample reported cache restores or persistent cache hits, so TTFT improvement alone is not treated as attributed reuse evidence."
                    if attributed_sample_count == 0
                    else "TTFT deltas are only treated as attributed when the warm request reports restored cache state or cached prompt tokens."
                ),
            ],
        )

    async def _mlx_acceleration_scenario(
        self,
        *,
        prompt: str,
        model_ids: list[str],
    ) -> BenchmarkScenarioReport:
        if not model_ids:
            return BenchmarkScenarioReport(
                scenario="mlx_acceleration_paths",
                capability=CapabilityName.CHAT.value,
                feature=PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION,
                status="not_applicable",
                reason="No chat-capable benchmark model was selected for the MLX acceleration sample.",
            )
        samples: list[BenchmarkScenarioSample] = []
        for index, model_id in enumerate(model_ids):
            messages = [GenerateMessage(role="user", content=f"{prompt} [mlx-accel-{index}]")]
            manifest, runtime, _ = self.model_router.route_chat(
                model_id,
                messages=messages,
                max_tokens=48,
            )
            health = await runtime.health_check()
            feature_map = health.get("performance_features")
            if not isinstance(feature_map, dict):
                continue
            graph_feature = feature_map.get(PerformanceFeatureName.GRAPH_COMPILATION.value)
            attention_feature = feature_map.get(PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION.value)
            graph_supported = isinstance(graph_feature, dict) and bool(graph_feature.get("supported"))
            attention_metrics = attention_feature.get("metrics") if isinstance(attention_feature, dict) else {}
            preferred_mode = (
                attention_metrics.get("preferred_mode")
                if isinstance(attention_metrics, dict) and isinstance(attention_metrics.get("preferred_mode"), str)
                else None
            )
            attention_supported = isinstance(attention_feature, dict) and bool(attention_feature.get("supported"))
            if not graph_supported and not attention_supported:
                continue
            stock_sample = await self._timed_mlx_acceleration_request(
                model_id=manifest.model_id,
                prompt=f"{prompt} [stock-{index}]",
                acceleration_overrides={
                    "graph_compile_enabled": False,
                    "attention_kernel_mode": "stock",
                },
            )
            accelerated_sample = await self._timed_mlx_acceleration_request(
                model_id=manifest.model_id,
                prompt=f"{prompt} [accelerated-{index}]",
                acceleration_overrides={
                    "graph_compile_enabled": graph_supported,
                    "attention_kernel_mode": preferred_mode if attention_supported and preferred_mode is not None else "stock",
                },
            )
            accelerated_seconds = _coerce_float(accelerated_sample.get("generate_seconds"))
            stock_seconds = _coerce_float(stock_sample.get("generate_seconds"))
            acceleration_payload = accelerated_sample.get("acceleration", {})
            prefix_cache_payload = accelerated_sample.get("prefix_cache", {})
            fallback_reason = (
                self._acceleration_fallback_reason(acceleration_payload)
                if isinstance(acceleration_payload, dict)
                else None
            )
            compile_state = (
                self._acceleration_compile_state(acceleration_payload)
                if isinstance(acceleration_payload, dict)
                else "stock"
            )
            shortcut_prefill_tokens = (
                _coerce_int(prefix_cache_payload.get("saved_prefill_tokens"))
                + _coerce_int(prefix_cache_payload.get("prefilled_uncached_tokens"))
                if isinstance(prefix_cache_payload, dict)
                else 0
            )
            samples.append(
                BenchmarkScenarioSample(
                    model_id=manifest.model_id,
                    runtime=str(accelerated_sample["runtime"]),
                    metrics=_compact_metrics(
                        stock_generate_seconds=stock_seconds,
                        accelerated_generate_seconds=accelerated_seconds,
                        accelerated_over_stock_ratio=(
                            round(accelerated_seconds / stock_seconds, 4)
                            if stock_seconds is not None and accelerated_seconds is not None and stock_seconds > 0
                            else None
                        ),
                        time_saved_seconds=(
                            round(stock_seconds - accelerated_seconds, 4)
                            if stock_seconds is not None and accelerated_seconds is not None
                            else None
                        ),
                        compile_state=compile_state,
                        graph_compile_used=bool(acceleration_payload.get("effective_graph_compile")),
                        prefill_compile_used=(
                            self._acceleration_phase_compile_used(acceleration_payload, "prefill")
                            if isinstance(acceleration_payload, dict)
                            else False
                        ),
                        decode_compile_used=(
                            self._acceleration_phase_compile_used(acceleration_payload, "decode")
                            if isinstance(acceleration_payload, dict)
                            else bool(acceleration_payload.get("effective_graph_compile"))
                        ),
                        requested_kernel_mode=acceleration_payload.get("requested_kernel_mode"),
                        kernel_path=acceleration_payload.get("effective_kernel_path"),
                        prefill_kernel_path=(
                            self._acceleration_phase_kernel_path(acceleration_payload, "prefill")
                            if isinstance(acceleration_payload, dict)
                            else None
                        ),
                        decode_kernel_path=(
                            self._acceleration_phase_kernel_path(acceleration_payload, "decode")
                            if isinstance(acceleration_payload, dict)
                            else acceleration_payload.get("effective_kernel_path")
                        ),
                        decode_shortcut_active=shortcut_prefill_tokens > 0,
                        shortcut_prefill_tokens=shortcut_prefill_tokens,
                        acceleration_fallback=bool(acceleration_payload.get("acceleration_fallback")),
                        fallback_reason=fallback_reason,
                    ),
                ),
            )
        if not samples:
            return BenchmarkScenarioReport(
                scenario="mlx_acceleration_paths",
                capability=CapabilityName.CHAT.value,
                feature=PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION,
                status="not_applicable",
                reason="No routed MLX chat runtime advertised graph compilation or accelerated attention hooks for comparison.",
            )
        stock_values = [
            _coerce_float(sample.metrics.get("stock_generate_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("stock_generate_seconds")) is not None
        ]
        accelerated_values = [
            _coerce_float(sample.metrics.get("accelerated_generate_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("accelerated_generate_seconds")) is not None
        ]
        ratio_values = [
            _coerce_float(sample.metrics.get("accelerated_over_stock_ratio"))
            for sample in samples
            if _coerce_float(sample.metrics.get("accelerated_over_stock_ratio")) is not None
        ]
        saved_values = [
            _coerce_float(sample.metrics.get("time_saved_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("time_saved_seconds")) is not None
        ]
        kernel_paths = sorted(
            {
                str(kernel_path)
                for sample in samples
                if isinstance((kernel_path := sample.metrics.get("kernel_path")), str) and kernel_path
            },
        )
        fallback_reasons = sorted(
            {
                str(reason)
                for sample in samples
                if isinstance((reason := sample.metrics.get("fallback_reason")), str) and reason
            },
        )
        compiled_sample_count = sum(1 for sample in samples if bool(sample.metrics.get("graph_compile_used")))
        prefill_compiled_sample_count = sum(1 for sample in samples if bool(sample.metrics.get("prefill_compile_used")))
        decode_compiled_sample_count = sum(1 for sample in samples if bool(sample.metrics.get("decode_compile_used")))
        fallback_sample_count = sum(1 for sample in samples if bool(sample.metrics.get("acceleration_fallback")))
        compile_states = sorted(
            {
                str(compile_state)
                for sample in samples
                if isinstance((compile_state := sample.metrics.get("compile_state")), str) and compile_state
            },
        )
        decode_shortcut_sample_count = sum(1 for sample in samples if bool(sample.metrics.get("decode_shortcut_active")))
        total_shortcut_prefill_tokens = sum(_coerce_int(sample.metrics.get("shortcut_prefill_tokens")) for sample in samples)
        return BenchmarkScenarioReport(
            scenario="mlx_acceleration_paths",
            capability=CapabilityName.CHAT.value,
            feature=PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION,
            status="observed",
            reason=(
                "Compared warmed stock and accelerated MLX chat requests so graph-compilation and accelerated attention "
                "paths leave reproducible benchmark evidence."
            ),
            metrics=_compact_metrics(
                sample_count=len(samples),
                average_stock_generate_seconds=round(fmean(stock_values), 4) if stock_values else None,
                average_accelerated_generate_seconds=round(fmean(accelerated_values), 4) if accelerated_values else None,
                average_accelerated_over_stock_ratio=round(fmean(ratio_values), 4) if ratio_values else None,
                average_time_saved_seconds=round(fmean(saved_values), 4) if saved_values else None,
                compiled_sample_count=compiled_sample_count,
                prefill_compiled_sample_count=prefill_compiled_sample_count,
                decode_compiled_sample_count=decode_compiled_sample_count,
                fallback_sample_count=fallback_sample_count,
                compile_states=",".join(compile_states) if compile_states else None,
                decode_shortcut_sample_count=decode_shortcut_sample_count,
                total_shortcut_prefill_tokens=total_shortcut_prefill_tokens,
                kernel_paths=",".join(kernel_paths) if kernel_paths else None,
            ),
            samples=samples,
            notes=[
                "Each path is warmed once before the measured sample so graph-compilation setup cost does not dominate the steady-state comparison.",
                "Compile state reports whether MLX graph capture stayed on the prefill phase, decode phase, or both.",
                *(
                    [f"Observed fallback reason(s): {' | '.join(fallback_reasons)}"]
                    if fallback_reasons
                    else []
                ),
            ],
        )

    async def _request_queue_scenario(
        self,
        *,
        capability: str,
        prompt: str,
        model_ids: list[str],
    ) -> BenchmarkScenarioReport:
        if self.runtime_request_scheduler.max_concurrent_requests <= 0:
            return BenchmarkScenarioReport(
                scenario="request_queue_pressure",
                capability=capability,
                feature=PerformanceFeatureName.REQUEST_SCHEDULING_AND_BACKPRESSURE,
                status="unsupported",
                reason="Runtime request scheduling is disabled because `max_concurrent_runtime_requests` is 0.",
            )
        if not model_ids:
            return BenchmarkScenarioReport(
                scenario="request_queue_pressure",
                capability=capability,
                feature=PerformanceFeatureName.REQUEST_SCHEDULING_AND_BACKPRESSURE,
                status="not_applicable",
                reason="No benchmark model was available for the queue-pressure sample.",
            )
        selected_model_id = model_ids[0]
        concurrency = max(2, self.runtime_request_scheduler.max_concurrent_requests + 2)
        before = self.runtime_request_scheduler.snapshot()
        started_at = time.perf_counter()
        executions = await asyncio.gather(
            *[
                self._timed_capability_request(
                    capability=capability,
                    model_id=selected_model_id,
                    prompt=prompt,
                    variant=index,
                )
                for index in range(concurrency)
            ],
            return_exceptions=True,
        )
        elapsed_seconds = round(time.perf_counter() - started_at, 4)
        after = self.runtime_request_scheduler.snapshot()
        successful_runs = [item for item in executions if isinstance(item, dict)]
        failed_runs = [item for item in executions if isinstance(item, Exception)]
        queued_delta = after["total_queued_requests"] - before["total_queued_requests"]
        rejected_delta = after["rejected_requests"] - before["rejected_requests"]
        timed_out_delta = after["timed_out_requests"] - before["timed_out_requests"]
        queue_depth_delta = after["max_observed_queue_depth"] - before["max_observed_queue_depth"]
        return BenchmarkScenarioReport(
            scenario="request_queue_pressure",
            capability=capability,
            feature=PerformanceFeatureName.REQUEST_SCHEDULING_AND_BACKPRESSURE,
            status="observed",
            reason=(
                "Issued more concurrent requests than the configured runtime concurrency limit so queue/backpressure "
                "counters are captured in the benchmark artifact."
            ),
            metrics=_compact_metrics(
                model_id=selected_model_id,
                concurrency=concurrency,
                elapsed_seconds=elapsed_seconds,
                successful_requests=len(successful_runs),
                failed_requests=len(failed_runs),
                queued_requests_delta=queued_delta,
                rejected_requests_delta=rejected_delta,
                timed_out_requests_delta=timed_out_delta,
                max_observed_queue_depth_delta=queue_depth_delta,
            ),
            samples=[
                BenchmarkScenarioSample(
                    model_id=str(item["model_id"]),
                    runtime=str(item["runtime"]),
                    metrics=_compact_metrics(elapsed_seconds=item["elapsed_seconds"]),
                )
                for item in successful_runs
            ],
            notes=(
                []
                if queued_delta or rejected_delta or timed_out_delta or queue_depth_delta
                else ["The stress sample completed without observable queue growth on this host/runtime mix."]
            ),
        )

    async def _load_admission_scenario(
        self,
        *,
        capability: str,
        prompt: str,
        model_ids: list[str],
    ) -> BenchmarkScenarioReport:
        if self.model_load_scheduler.max_concurrent_requests <= 0:
            return BenchmarkScenarioReport(
                scenario="model_load_admission",
                capability=capability,
                feature=PerformanceFeatureName.MODEL_LOAD_ADMISSION_CONTROL,
                status="unsupported",
                reason="Cold-model load admission is disabled because `max_concurrent_model_loads` is 0.",
            )
        if len(model_ids) < 2:
            return BenchmarkScenarioReport(
                scenario="model_load_admission",
                capability=capability,
                feature=PerformanceFeatureName.MODEL_LOAD_ADMISSION_CONTROL,
                status="not_applicable",
                reason="At least two benchmark models are required to exercise cold-load admission concurrency.",
            )
        selected_model_ids = model_ids[: max(2, min(len(model_ids), self.model_load_scheduler.max_concurrent_requests + 1))]
        await self.runtime_catalog.unload_all_models()
        before = self.model_load_scheduler.snapshot()
        started_at = time.perf_counter()
        executions = await asyncio.gather(
            *[
                self._timed_capability_request(
                    capability=capability,
                    model_id=model_id,
                    prompt=prompt,
                    variant=index,
                )
                for index, model_id in enumerate(selected_model_ids)
            ],
            return_exceptions=True,
        )
        elapsed_seconds = round(time.perf_counter() - started_at, 4)
        after = self.model_load_scheduler.snapshot()
        successful_runs = [item for item in executions if isinstance(item, dict)]
        failed_runs = [item for item in executions if isinstance(item, Exception)]
        queued_delta = after["total_queued_requests"] - before["total_queued_requests"]
        rejected_delta = after["rejected_requests"] - before["rejected_requests"]
        timed_out_delta = after["timed_out_requests"] - before["timed_out_requests"]
        queue_depth_delta = after["max_observed_queue_depth"] - before["max_observed_queue_depth"]
        return BenchmarkScenarioReport(
            scenario="model_load_admission",
            capability=capability,
            feature=PerformanceFeatureName.MODEL_LOAD_ADMISSION_CONTROL,
            status="observed",
            reason=(
                "Forced concurrent cold loads across multiple benchmark models so the dedicated model-load scheduler "
                "can emit admission metrics."
            ),
            metrics=_compact_metrics(
                distinct_model_count=len(selected_model_ids),
                elapsed_seconds=elapsed_seconds,
                successful_requests=len(successful_runs),
                failed_requests=len(failed_runs),
                queued_requests_delta=queued_delta,
                rejected_requests_delta=rejected_delta,
                timed_out_requests_delta=timed_out_delta,
                max_observed_queue_depth_delta=queue_depth_delta,
            ),
            samples=[
                BenchmarkScenarioSample(
                    model_id=str(item["model_id"]),
                    runtime=str(item["runtime"]),
                    metrics=_compact_metrics(elapsed_seconds=item["elapsed_seconds"]),
                )
                for item in successful_runs
            ],
            notes=(
                []
                if queued_delta or rejected_delta or timed_out_delta or queue_depth_delta
                else ["The selected models loaded without observable admission queueing during this sample."]
            ),
        )

    async def _multimodal_encoder_reuse_scenario(
        self,
        *,
        capability: str,
        prompt: str,
        model_ids: list[str],
    ) -> BenchmarkScenarioReport:
        if capability == CapabilityName.CHAT.value:
            multimodal_model_ids = [
                model_id
                for model_id in model_ids
                if any(
                    modality in {ModelModality.VISION, ModelModality.MULTIMODAL}
                    for modality in self.model_router.model_registry.get_manifest(model_id).modality
                )
            ]
            if not multimodal_model_ids:
                return BenchmarkScenarioReport(
                    scenario="multimodal_encoder_reuse",
                    capability=capability,
                    feature=PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING,
                    status="not_applicable",
                    reason="Selected chat benchmark models do not advertise multimodal image support.",
                )
            assets = _ensure_benchmark_multimodal_assets(self.settings.benchmarks_dir)
            samples: list[BenchmarkScenarioSample] = []
            for model_id in multimodal_model_ids:
                for sample_name, source_path in (
                    ("image", assets["image"]),
                    ("frame_bundle", assets["frame_bundle"]),
                ):
                    if self.service_factory is None:
                        first_run = await self._timed_multimodal_chat_request(
                            model_id=model_id,
                            prompt=prompt,
                            source_path=source_path,
                            use_cli_normalization=True,
                        )
                        second_run = await self._timed_multimodal_chat_request(
                            model_id=model_id,
                            prompt=prompt,
                            source_path=source_path,
                            use_cli_normalization=True,
                        )
                        samples.append(
                            BenchmarkScenarioSample(
                                model_id=model_id,
                                runtime=str(first_run["runtime"]),
                                metrics=_compact_metrics(
                                    sample_type=sample_name,
                                    first_elapsed_seconds=first_run["elapsed_seconds"],
                                    second_elapsed_seconds=second_run["elapsed_seconds"],
                                    second_over_first_ratio=(
                                        round(second_run["elapsed_seconds"] / first_run["elapsed_seconds"], 4)
                                        if first_run["elapsed_seconds"] > 0
                                        else None
                                    ),
                                    first_feature_cache_hit_delta=first_run["multimodal_feature_cache_hit_delta"],
                                    first_feature_cache_miss_delta=first_run["multimodal_feature_cache_miss_delta"],
                                    second_feature_cache_hit_delta=second_run["multimodal_feature_cache_hit_delta"],
                                    second_feature_cache_miss_delta=second_run["multimodal_feature_cache_miss_delta"],
                                    first_encoder_cache_hit_delta=first_run["multimodal_encoder_cache_hit_delta"],
                                    first_encoder_cache_miss_delta=first_run["multimodal_encoder_cache_miss_delta"],
                                    second_encoder_cache_hit_delta=second_run["multimodal_encoder_cache_hit_delta"],
                                    second_encoder_cache_miss_delta=second_run["multimodal_encoder_cache_miss_delta"],
                                ),
                            )
                        )
                        continue
                    with TemporaryDirectory(
                        dir=self.settings.temp_dir,
                        prefix=f"lewlm-benchmark-multimodal-{sample_name}-",
                    ) as temp_dir:
                        candidate_settings = self.settings.with_updates(data_dir=Path(temp_dir))
                        candidate_services = self.service_factory(candidate_settings)
                        try:
                            self._sync_benchmark_child_manifests(candidate_services)
                            cold_run = await self._timed_multimodal_chat_request(
                                model_id=model_id,
                                prompt=_benchmark_prompt_variant(prompt, phase="cold", sample_type=sample_name),
                                source_path=source_path,
                                services=candidate_services,
                                use_cli_normalization=True,
                            )
                            candidate_services.multimodal_encoder_cache.drop_runtime_resident_features(model_id=model_id)
                            first_run = await self._timed_multimodal_chat_request(
                                model_id=model_id,
                                prompt=_benchmark_prompt_variant(prompt, phase="attachment_only", sample_type=sample_name),
                                source_path=source_path,
                                services=candidate_services,
                                use_cli_normalization=True,
                            )
                            second_run = await self._timed_multimodal_chat_request(
                                model_id=model_id,
                                prompt=_benchmark_prompt_variant(prompt, phase="encoder_reuse", sample_type=sample_name),
                                source_path=source_path,
                                services=candidate_services,
                                use_cli_normalization=True,
                            )
                        finally:
                            await candidate_services.aclose()
                    samples.append(
                        BenchmarkScenarioSample(
                            model_id=model_id,
                            runtime=str(second_run["runtime"]),
                            metrics=_compact_metrics(
                                sample_type=sample_name,
                                cold_elapsed_seconds=cold_run["elapsed_seconds"],
                                first_elapsed_seconds=first_run["elapsed_seconds"],
                                second_elapsed_seconds=second_run["elapsed_seconds"],
                                first_over_cold_ratio=(
                                    round(first_run["elapsed_seconds"] / cold_run["elapsed_seconds"], 4)
                                    if cold_run["elapsed_seconds"] > 0
                                    else None
                                ),
                                second_over_first_ratio=(
                                    round(second_run["elapsed_seconds"] / first_run["elapsed_seconds"], 4)
                                    if first_run["elapsed_seconds"] > 0
                                    else None
                                ),
                                second_over_cold_ratio=(
                                    round(second_run["elapsed_seconds"] / cold_run["elapsed_seconds"], 4)
                                    if cold_run["elapsed_seconds"] > 0
                                    else None
                                ),
                                cold_feature_cache_hit_delta=cold_run["multimodal_feature_cache_hit_delta"],
                                cold_feature_cache_miss_delta=cold_run["multimodal_feature_cache_miss_delta"],
                                first_feature_cache_hit_delta=first_run["multimodal_feature_cache_hit_delta"],
                                first_feature_cache_miss_delta=first_run["multimodal_feature_cache_miss_delta"],
                                second_feature_cache_hit_delta=second_run["multimodal_feature_cache_hit_delta"],
                                second_feature_cache_miss_delta=second_run["multimodal_feature_cache_miss_delta"],
                                cold_encoder_cache_hit_delta=cold_run["multimodal_encoder_cache_hit_delta"],
                                cold_encoder_cache_miss_delta=cold_run["multimodal_encoder_cache_miss_delta"],
                                first_encoder_cache_hit_delta=first_run["multimodal_encoder_cache_hit_delta"],
                                first_encoder_cache_miss_delta=first_run["multimodal_encoder_cache_miss_delta"],
                                second_encoder_cache_hit_delta=second_run["multimodal_encoder_cache_hit_delta"],
                                second_encoder_cache_miss_delta=second_run["multimodal_encoder_cache_miss_delta"],
                                encoder_advantage_seconds=round(
                                    first_run["elapsed_seconds"] - second_run["elapsed_seconds"],
                                    4,
                                ),
                            ),
                        )
                    )
            return self._multimodal_encoder_scenario_report(
                capability=capability,
                samples=samples,
                reason=(
                    "Benchmarked cold, attachment-cache-only, and encoder-reuse passes for repeated image and frame-bundle "
                    "requests so benchmark output can attribute savings beyond prompt-ready attachment caching alone."
                ),
                notes=(
                    [
                        "Per-run prompts intentionally vary by phase so repeated-asset measurements do not quietly inherit text prefix-cache wins.",
                    ]
                    if self.service_factory is not None
                    else [
                        "This host fell back to the in-process two-pass benchmark path because isolated benchmark services are unavailable.",
                    ]
                ),
            )
        if capability == CapabilityName.AUDIO_TRANSCRIPTION.value:
            if not model_ids:
                return BenchmarkScenarioReport(
                    scenario="multimodal_encoder_reuse",
                    capability=capability,
                    feature=PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING,
                    status="not_applicable",
                    reason="No audio-transcription benchmark model was selected for the encoder-cache sample.",
                )
            audio_bytes = _benchmark_audio_bytes(prompt, duration_seconds=2.5)
            samples = []
            for model_id in model_ids:
                if self.service_factory is None:
                    first_run = await self._timed_audio_transcription_request(
                        model_id=model_id,
                        prompt=prompt,
                        audio_bytes=audio_bytes,
                    )
                    second_run = await self._timed_audio_transcription_request(
                        model_id=model_id,
                        prompt=prompt,
                        audio_bytes=audio_bytes,
                    )
                    samples.append(
                        BenchmarkScenarioSample(
                            model_id=model_id,
                            runtime=str(first_run["runtime"]),
                            metrics=_compact_metrics(
                                sample_type="audio",
                                first_elapsed_seconds=first_run["elapsed_seconds"],
                                second_elapsed_seconds=second_run["elapsed_seconds"],
                                second_over_first_ratio=(
                                    round(second_run["elapsed_seconds"] / first_run["elapsed_seconds"], 4)
                                    if first_run["elapsed_seconds"] > 0
                                    else None
                                ),
                                chunk_count=first_run["chunk_count"],
                                first_encoder_cache_hit_delta=first_run["multimodal_encoder_cache_hit_delta"],
                                first_encoder_cache_miss_delta=first_run["multimodal_encoder_cache_miss_delta"],
                                second_encoder_cache_hit_delta=second_run["multimodal_encoder_cache_hit_delta"],
                                second_encoder_cache_miss_delta=second_run["multimodal_encoder_cache_miss_delta"],
                            ),
                        )
                    )
                    continue
                with TemporaryDirectory(
                    dir=self.settings.temp_dir,
                    prefix="lewlm-benchmark-audio-encoder-",
                ) as temp_dir:
                    candidate_settings = self.settings.with_updates(data_dir=Path(temp_dir))
                    candidate_services = self.service_factory(candidate_settings)
                    try:
                        self._sync_benchmark_child_manifests(candidate_services)
                        first_run = await self._timed_audio_transcription_request(
                            model_id=model_id,
                            prompt=_benchmark_prompt_variant(prompt, phase="cold", sample_type="audio"),
                            audio_bytes=audio_bytes,
                            services=candidate_services,
                        )
                        second_run = await self._timed_audio_transcription_request(
                            model_id=model_id,
                            prompt=_benchmark_prompt_variant(prompt, phase="encoder_reuse", sample_type="audio"),
                            audio_bytes=audio_bytes,
                            services=candidate_services,
                        )
                    finally:
                        await candidate_services.aclose()
                samples.append(
                    BenchmarkScenarioSample(
                        model_id=model_id,
                        runtime=str(second_run["runtime"]),
                        metrics=_compact_metrics(
                            sample_type="audio",
                            first_elapsed_seconds=first_run["elapsed_seconds"],
                            second_elapsed_seconds=second_run["elapsed_seconds"],
                            second_over_first_ratio=(
                                round(second_run["elapsed_seconds"] / first_run["elapsed_seconds"], 4)
                                if first_run["elapsed_seconds"] > 0
                                else None
                            ),
                            chunk_count=first_run["chunk_count"],
                            first_encoder_cache_hit_delta=first_run["multimodal_encoder_cache_hit_delta"],
                            first_encoder_cache_miss_delta=first_run["multimodal_encoder_cache_miss_delta"],
                            second_encoder_cache_hit_delta=second_run["multimodal_encoder_cache_hit_delta"],
                            second_encoder_cache_miss_delta=second_run["multimodal_encoder_cache_miss_delta"],
                            encoder_advantage_seconds=round(
                                first_run["elapsed_seconds"] - second_run["elapsed_seconds"],
                                4,
                            ),
                        ),
                    )
                )
            return self._multimodal_encoder_scenario_report(
                capability=capability,
                samples=samples,
                reason=(
                    "Benchmarked cold and encoder-reuse audio transcription passes with a chunked WAV payload so encoder hits "
                    "remain visible after realistic chunking instead of being masked by deterministic response caching."
                ),
                notes=(
                    [
                        "Per-run prompts intentionally vary while remaining prompt-present so runtime response caching stays out of the encoder-reuse measurement.",
                    ]
                    if self.service_factory is not None
                    else [
                        "This host fell back to the in-process two-pass benchmark path because isolated benchmark services are unavailable.",
                    ]
                ),
            )
        return BenchmarkScenarioReport(
            scenario="multimodal_encoder_reuse",
            capability=capability,
            feature=PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING,
            status="not_applicable",
            reason=f"Selected `{capability}` benchmarks do not use the multimodal encoder reuse sample.",
        )

    def _multimodal_encoder_scenario_report(
        self,
        *,
        capability: str,
        samples: list[BenchmarkScenarioSample],
        reason: str,
        notes: list[str] | None = None,
    ) -> BenchmarkScenarioReport:
        first_values = [
            _coerce_float(sample.metrics.get("first_elapsed_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("first_elapsed_seconds")) is not None
        ]
        second_values = [
            _coerce_float(sample.metrics.get("second_elapsed_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("second_elapsed_seconds")) is not None
        ]
        ratio_values = [
            _coerce_float(sample.metrics.get("second_over_first_ratio"))
            for sample in samples
            if _coerce_float(sample.metrics.get("second_over_first_ratio")) is not None
        ]
        cold_values = [
            _coerce_float(sample.metrics.get("cold_elapsed_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("cold_elapsed_seconds")) is not None
        ]
        cold_ratio_values = [
            _coerce_float(sample.metrics.get("first_over_cold_ratio"))
            for sample in samples
            if _coerce_float(sample.metrics.get("first_over_cold_ratio")) is not None
        ]
        full_ratio_values = [
            _coerce_float(sample.metrics.get("second_over_cold_ratio"))
            for sample in samples
            if _coerce_float(sample.metrics.get("second_over_cold_ratio")) is not None
        ]
        advantage_values = [
            _coerce_float(sample.metrics.get("encoder_advantage_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("encoder_advantage_seconds")) is not None
        ]
        chunk_counts = [
            value
            for sample in samples
            if isinstance((value := sample.metrics.get("chunk_count")), int) and not isinstance(value, bool)
        ]
        return BenchmarkScenarioReport(
            scenario="multimodal_encoder_reuse",
            capability=capability,
            feature=PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING,
            status="observed",
            reason=reason,
            metrics=_compact_metrics(
                sample_count=len(samples),
                multimodal_feature_cache_hit_delta=_sum_benchmark_sample_metric(
                    samples,
                    "cold_feature_cache_hit_delta",
                    "first_feature_cache_hit_delta",
                    "second_feature_cache_hit_delta",
                ),
                multimodal_feature_cache_miss_delta=_sum_benchmark_sample_metric(
                    samples,
                    "cold_feature_cache_miss_delta",
                    "first_feature_cache_miss_delta",
                    "second_feature_cache_miss_delta",
                ),
                multimodal_encoder_cache_hit_delta=_sum_benchmark_sample_metric(
                    samples,
                    "cold_encoder_cache_hit_delta",
                    "first_encoder_cache_hit_delta",
                    "second_encoder_cache_hit_delta",
                ),
                multimodal_encoder_cache_miss_delta=_sum_benchmark_sample_metric(
                    samples,
                    "cold_encoder_cache_miss_delta",
                    "first_encoder_cache_miss_delta",
                    "second_encoder_cache_miss_delta",
                ),
                average_cold_elapsed_seconds=round(fmean(cold_values), 4) if cold_values else None,
                average_first_elapsed_seconds=round(fmean(first_values), 4) if first_values else None,
                average_second_elapsed_seconds=round(fmean(second_values), 4) if second_values else None,
                average_first_over_cold_ratio=round(fmean(cold_ratio_values), 4) if cold_ratio_values else None,
                average_second_over_first_ratio=round(fmean(ratio_values), 4) if ratio_values else None,
                average_second_over_cold_ratio=round(fmean(full_ratio_values), 4) if full_ratio_values else None,
                average_encoder_advantage_seconds=round(fmean(advantage_values), 4) if advantage_values else None,
                average_chunk_count=round(fmean(chunk_counts), 2) if chunk_counts else None,
            ),
            samples=samples,
            notes=list(notes or []),
        )

    async def _multimodal_reuse_scenario(
        self,
        *,
        capability: str,
        prompt: str,
        model_ids: list[str],
    ) -> BenchmarkScenarioReport:
        if capability not in {CapabilityName.EMBEDDINGS.value, CapabilityName.RERANK.value}:
            return BenchmarkScenarioReport(
                scenario="multimodal_reuse",
                capability=capability,
                feature=PerformanceFeatureName.MULTIMODAL_FEATURE_CACHING,
                status="not_applicable",
                reason=f"Selected `{capability}` benchmarks do not use the multimodal reuse sample.",
            )
        if not model_ids:
            return BenchmarkScenarioReport(
                scenario="multimodal_reuse",
                capability=capability,
                feature=PerformanceFeatureName.MULTIMODAL_FEATURE_CACHING,
                status="not_applicable",
                reason="No deterministic-capability benchmark model was selected for the reuse sample.",
            )
        before_cache = self.runtime_response_cache.cache_stats()
        samples: list[BenchmarkScenarioSample] = []
        for model_id in model_ids:
            first_run = await self._timed_capability_request(
                capability=capability,
                model_id=model_id,
                prompt=prompt,
                variant=0,
                reuse_key="shared",
            )
            second_run = await self._timed_capability_request(
                capability=capability,
                model_id=model_id,
                prompt=prompt,
                variant=0,
                reuse_key="shared",
            )
            samples.append(
                BenchmarkScenarioSample(
                    model_id=model_id,
                    runtime=str(first_run["runtime"]),
                    metrics=_compact_metrics(
                        first_elapsed_seconds=first_run["elapsed_seconds"],
                        second_elapsed_seconds=second_run["elapsed_seconds"],
                        second_over_first_ratio=(
                            round(second_run["elapsed_seconds"] / first_run["elapsed_seconds"], 4)
                            if first_run["elapsed_seconds"] > 0
                            else None
                        ),
                    ),
                ),
            )
        after_cache = self.runtime_response_cache.cache_stats()
        first_values = [
            _coerce_float(sample.metrics.get("first_elapsed_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("first_elapsed_seconds")) is not None
        ]
        second_values = [
            _coerce_float(sample.metrics.get("second_elapsed_seconds"))
            for sample in samples
            if _coerce_float(sample.metrics.get("second_elapsed_seconds")) is not None
        ]
        ratio_values = [
            _coerce_float(sample.metrics.get("second_over_first_ratio"))
            for sample in samples
            if _coerce_float(sample.metrics.get("second_over_first_ratio")) is not None
        ]
        return BenchmarkScenarioReport(
            scenario="multimodal_reuse",
            capability=capability,
            feature=PerformanceFeatureName.MULTIMODAL_FEATURE_CACHING,
            status="observed",
            reason=(
                "Replayed the same deterministic multimodal workload twice so cache reuse and future feature-cache "
                "optimizations can be proven in the artifact stream."
            ),
            metrics=_compact_metrics(
                sample_count=len(samples),
                cache_hit_delta=after_cache["runtime_cache_hits"] - before_cache["runtime_cache_hits"],
                cache_miss_delta=after_cache["runtime_cache_misses"] - before_cache["runtime_cache_misses"],
                average_first_elapsed_seconds=round(fmean(first_values), 4) if first_values else None,
                average_second_elapsed_seconds=round(fmean(second_values), 4) if second_values else None,
                average_second_over_first_ratio=round(fmean(ratio_values), 4) if ratio_values else None,
            ),
            samples=samples,
            notes=["LewLM currently proves deterministic response reuse here even before runtime-native feature tensors are cached."],
        )

    async def _timed_chat_request(self, *, model_id: str, prompt: str) -> dict[str, Any]:
        messages = [GenerateMessage(role="user", content=prompt)]
        manifest, runtime, _ = self.model_router.route_chat(model_id, messages=messages, max_tokens=48)
        request = GenerateRequest(
            model_id=manifest.model_id,
            messages=messages,
            max_tokens=48,
            temperature=0.0,
        )
        response, _, _, elapsed_seconds = await self._execute_benchmark_request(
            manifest=manifest,
            runtime=runtime,
            invoke=lambda request=request: runtime.generate(request),
        )
        return {
            "elapsed_seconds": round(elapsed_seconds, 4),
            "runtime": runtime.name,
            "model_id": manifest.model_id,
            "output_text": response.output_text,
            "usage": dict(response.usage),
            "request_metadata": dict(request.metadata),
        }

    async def _timed_orchestrated_chat_request(
        self,
        *,
        model_id: str,
        prompt: str,
        max_tokens: int = 48,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        execution = await self.chat_orchestrator.complete(
            model_id=model_id,
            messages=[GenerateMessage(role="user", content=prompt)],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return {
            "elapsed_seconds": round(time.perf_counter() - started_at, 4),
            "runtime": execution.routing.runtime_name,
            "model_id": execution.routing.model_id,
            "output_text": execution.response.output_text,
            "usage": execution.response.usage,
            "request_metadata": dict(execution.request_metadata),
        }

    async def _timed_orchestrated_stream_request(
        self,
        *,
        model_id: str,
        prompt: str,
        max_tokens: int = 48,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        session = await self.chat_orchestrator.stream(
            model_id=model_id,
            messages=[GenerateMessage(role="user", content=prompt)],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        ttft_seconds: float | None = None
        output_chunks: list[str] = []
        delta_count = 0
        async for delta in session.stream:
            if ttft_seconds is None:
                ttft_seconds = round(time.perf_counter() - started_at, 4)
            output_chunks.append(delta)
            delta_count += 1
        elapsed_seconds = round(time.perf_counter() - started_at, 4)
        inter_token_seconds = (
            round(max(elapsed_seconds - ttft_seconds, 0.0) / max(delta_count - 1, 1), 4)
            if ttft_seconds is not None and delta_count > 0
            else None
        )
        return {
            "elapsed_seconds": elapsed_seconds,
            "ttft_seconds": ttft_seconds if ttft_seconds is not None else 0.0,
            "inter_token_seconds": inter_token_seconds,
            "delta_count": delta_count,
            "runtime": session.routing.runtime_name,
            "model_id": session.routing.model_id,
            "output_text": "".join(output_chunks),
            "request_metadata": dict(session.request_metadata or {}),
        }

    async def _timed_multimodal_chat_request(
        self,
        *,
        model_id: str,
        prompt: str,
        source_path: Path,
        services: "LewLMServices | None" = None,
        use_cli_normalization: bool = False,
    ) -> dict[str, Any]:
        resolved_services = services
        resolved_telemetry = resolved_services.telemetry_service if resolved_services is not None else self
        before_cache = resolved_telemetry.cache_stats()
        attachment = GenerateAttachment(
            attachment_type="image",
            name=source_path.name,
            source_path=str(source_path),
            metadata={"source_kind": "image_bundle" if source_path.is_dir() else "image"},
        )
        if use_cli_normalization and resolved_services is not None and source_path.is_file():
            raw_bytes = source_path.read_bytes()
            cache_key = resolved_services.multimodal_feature_cache.cache_key_for_path_attachment(
                raw_bytes=raw_bytes,
                suffix=source_path.suffix.casefold(),
            )
            cached_attachment = resolved_services.multimodal_feature_cache.get_attachment(
                cache_key=cache_key,
                name=source_path.name,
                source_path=str(source_path),
            )
            if cached_attachment is not None:
                attachment = cached_attachment
            else:
                resolved_services.multimodal_feature_cache.put_attachment(
                    cache_key=cache_key,
                    attachment=attachment,
                    cache_metadata={
                        "source_kind": "image",
                        "source_suffix": source_path.suffix.casefold(),
                        "input_bytes": len(raw_bytes),
                    },
                )
        messages = [
            GenerateMessage(
                role="user",
                content=prompt,
                attachments=[attachment],
            )
        ]
        started_at = time.perf_counter()
        execution = await (
            resolved_services.chat_orchestrator.complete
            if resolved_services is not None
            else self.chat_orchestrator.complete
        )(
            model_id=model_id,
            messages=messages,
            max_tokens=48,
            temperature=0.0,
        )
        after_cache = resolved_telemetry.cache_stats()
        return {
            "elapsed_seconds": round(time.perf_counter() - started_at, 4),
            "runtime": execution.routing.runtime_name,
            "model_id": execution.routing.model_id,
            "usage": execution.response.usage,
            "request_metadata": execution.request_metadata,
            "multimodal_feature_cache_hit_delta": (
                after_cache.multimodal_feature_cache_hits - before_cache.multimodal_feature_cache_hits
            ),
            "multimodal_feature_cache_miss_delta": (
                after_cache.multimodal_feature_cache_misses - before_cache.multimodal_feature_cache_misses
            ),
            "multimodal_encoder_cache_hit_delta": (
                after_cache.multimodal_encoder_cache_hits - before_cache.multimodal_encoder_cache_hits
            ),
            "multimodal_encoder_cache_miss_delta": (
                after_cache.multimodal_encoder_cache_misses - before_cache.multimodal_encoder_cache_misses
            ),
            "runtime_cache_hit_delta": after_cache.runtime_cache_hits - before_cache.runtime_cache_hits,
            "runtime_cache_miss_delta": after_cache.runtime_cache_misses - before_cache.runtime_cache_misses,
        }

    async def _timed_audio_transcription_request(
        self,
        *,
        model_id: str,
        prompt: str,
        audio_bytes: bytes,
        services: "LewLMServices | None" = None,
    ) -> dict[str, Any]:
        from lewlm.core.multimodal import _plan_audio_transcription_chunks

        resolved_services = services
        resolved_telemetry = resolved_services.telemetry_service if resolved_services is not None else self
        before_cache = resolved_telemetry.cache_stats()
        chunk_count = _plan_audio_transcription_chunks(audio_bytes).chunk_count
        started_at = time.perf_counter()
        execution = await (
            resolved_services.multimodal_orchestrator.transcribe_audio
            if resolved_services is not None
            else self.multimodal_orchestrator.transcribe_audio
        )(
            model_id=model_id,
            audio_bytes=audio_bytes,
            file_name="benchmark-audio.wav",
            language="en",
            prompt=prompt,
        )
        after_cache = resolved_telemetry.cache_stats()
        return {
            "elapsed_seconds": round(time.perf_counter() - started_at, 4),
            "runtime": execution.routing.runtime_name,
            "model_id": execution.routing.model_id,
            "chunk_count": chunk_count,
            "multimodal_encoder_cache_hit_delta": (
                after_cache.multimodal_encoder_cache_hits - before_cache.multimodal_encoder_cache_hits
            ),
            "multimodal_encoder_cache_miss_delta": (
                after_cache.multimodal_encoder_cache_misses - before_cache.multimodal_encoder_cache_misses
            ),
            "runtime_cache_hit_delta": after_cache.runtime_cache_hits - before_cache.runtime_cache_hits,
            "runtime_cache_miss_delta": after_cache.runtime_cache_misses - before_cache.runtime_cache_misses,
        }

    async def _timed_stream_chat_request(self, *, model_id: str, prompt: str) -> dict[str, Any]:
        messages = [GenerateMessage(role="user", content=prompt)]
        manifest, runtime, _ = self.model_router.route_chat(model_id, messages=messages, max_tokens=48)
        request = GenerateRequest(
            model_id=manifest.model_id,
            messages=messages,
            max_tokens=48,
            temperature=0.0,
        )
        stream_payload = await self._execute_stream_benchmark_request(
            manifest=manifest,
            runtime=runtime,
            request=request,
        )
        return {
            "elapsed_seconds": stream_payload["elapsed_seconds"],
            "ttft_seconds": stream_payload["ttft_seconds"],
            "runtime": runtime.name,
            "model_id": manifest.model_id,
            "output_text": str(stream_payload["output_text"]),
            "request_metadata": dict(request.metadata),
        }

    @staticmethod
    def _request_metadata_payload(run: dict[str, Any], key: str) -> dict[str, Any]:
        metadata = run.get("request_metadata")
        if not isinstance(metadata, dict):
            return {}
        payload = metadata.get(key)
        return payload if isinstance(payload, dict) else {}

    @classmethod
    def _request_prefix_metric(cls, run: dict[str, Any], key: str) -> int:
        return _coerce_int(cls._request_metadata_payload(run, "prefix_cache").get(key))

    @classmethod
    def _request_prefix_text(cls, run: dict[str, Any], key: str) -> str | None:
        value = cls._request_metadata_payload(run, "prefix_cache").get(key)
        return value if isinstance(value, str) and value else None

    @classmethod
    def _request_performance_control(cls, run: dict[str, Any], control_name: str) -> dict[str, Any]:
        metadata = run.get("request_metadata")
        if not isinstance(metadata, dict):
            return {}
        controls = metadata.get("performance_controls")
        if not isinstance(controls, dict):
            return {}
        for phase_name in ("generate", "load"):
            phase_payload = controls.get(phase_name)
            if not isinstance(phase_payload, dict):
                continue
            control_payload = phase_payload.get(control_name)
            if isinstance(control_payload, dict):
                return control_payload
        return {}

    @classmethod
    def _request_control_metric(cls, run: dict[str, Any], control_name: str, key: str) -> int:
        return _coerce_int(cls._request_performance_control(run, control_name).get(key))

    @classmethod
    def _request_control_text(cls, run: dict[str, Any], control_name: str, key: str) -> str | None:
        value = cls._request_performance_control(run, control_name).get(key)
        return value if isinstance(value, str) and value else None

    @classmethod
    def _request_scheduling_metric(cls, run: dict[str, Any], key: str) -> int:
        return _coerce_int(cls._request_metadata_payload(run, "scheduling").get(key))

    @classmethod
    def _request_scheduling_flag(cls, run: dict[str, Any], key: str) -> bool:
        value = cls._request_metadata_payload(run, "scheduling").get(key)
        return bool(value) if isinstance(value, (bool, int)) else False

    @classmethod
    def _request_scheduling_text(cls, run: dict[str, Any], key: str) -> str | None:
        value = cls._request_metadata_payload(run, "scheduling").get(key)
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _mixed_prefill_prompt(prompt: str) -> str:
        repeated_fragment = "Long prefill queue-shaping benchmark segment."
        repeat_count = max(256, 4 * max(1, len(prompt.split())))
        return f"{prompt}\n" + " ".join(repeated_fragment for _ in range(repeat_count))

    @classmethod
    def _request_acceleration_payload(cls, run: dict[str, Any]) -> dict[str, Any]:
        metadata = run.get("request_metadata")
        if not isinstance(metadata, dict):
            return {}
        payload = metadata.get("mlx_acceleration")
        return payload if isinstance(payload, dict) else {}

    @classmethod
    def _request_acceleration_phase_details(cls, run: dict[str, Any]) -> dict[str, dict[str, Any]]:
        payload = cls._request_acceleration_payload(run)
        phase_details = payload.get("phase_details")
        if not isinstance(phase_details, dict):
            return {}
        return {
            str(phase): dict(details)
            for phase, details in phase_details.items()
            if isinstance(phase, str) and isinstance(details, dict)
        }

    @classmethod
    def _acceleration_phase_payload(cls, acceleration_payload: dict[str, Any], phase_name: str) -> dict[str, Any]:
        phase_details = acceleration_payload.get("phase_details")
        if not isinstance(phase_details, dict):
            return {}
        phase_payload = phase_details.get(phase_name)
        return phase_payload if isinstance(phase_payload, dict) else {}

    @classmethod
    def _acceleration_compile_state(cls, acceleration_payload: dict[str, Any]) -> str:
        compile_state = acceleration_payload.get("compile_state")
        if isinstance(compile_state, str) and compile_state:
            return compile_state
        compiled_phases = [
            phase_name
            for phase_name in ("prefill", "decode", "stream")
            if bool(cls._acceleration_phase_payload(acceleration_payload, phase_name).get("effective_graph_compile"))
        ]
        if compiled_phases:
            return "+".join(compiled_phases)
        return "decode" if bool(acceleration_payload.get("effective_graph_compile")) else "stock"

    @classmethod
    def _acceleration_phase_compile_used(cls, acceleration_payload: dict[str, Any], phase_name: str) -> bool:
        phase_payload = cls._acceleration_phase_payload(acceleration_payload, phase_name)
        if phase_payload:
            return bool(phase_payload.get("effective_graph_compile"))
        if phase_name == "decode":
            return bool(acceleration_payload.get("effective_graph_compile"))
        return False

    @classmethod
    def _acceleration_phase_kernel_path(cls, acceleration_payload: dict[str, Any], phase_name: str) -> str | None:
        phase_payload = cls._acceleration_phase_payload(acceleration_payload, phase_name)
        if phase_payload:
            kernel_path = phase_payload.get("effective_kernel_path")
            return kernel_path if isinstance(kernel_path, str) and kernel_path else None
        if phase_name == "decode":
            kernel_path = acceleration_payload.get("effective_kernel_path")
            return kernel_path if isinstance(kernel_path, str) and kernel_path else None
        return None

    @classmethod
    def _acceleration_fallback_reason(cls, acceleration_payload: dict[str, Any]) -> str | None:
        reasons = [
            str(reason)
            for phase_name in ("prefill", "decode", "stream")
            if isinstance(
                (reason := cls._acceleration_phase_payload(acceleration_payload, phase_name).get("fallback_reason")),
                str,
            )
            and reason
        ]
        if reasons:
            return " | ".join(dict.fromkeys(reasons))
        reason = acceleration_payload.get("fallback_reason")
        return reason if isinstance(reason, str) and reason else None

    async def _timed_mlx_acceleration_request(
        self,
        *,
        model_id: str,
        prompt: str,
        acceleration_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        await self.runtime_catalog.unload_all_models()
        messages = [GenerateMessage(role="user", content=prompt)]
        manifest, runtime, _ = self.model_router.route_chat(
            model_id,
            messages=messages,
            max_tokens=48,
        )
        for suffix in ("warmup", "measured"):
            request = GenerateRequest(
                model_id=manifest.model_id,
                messages=[GenerateMessage(role="user", content=f"{prompt} [{suffix}]")],
                max_tokens=48,
                temperature=0.0,
                metadata={"mlx_acceleration": dict(acceleration_overrides)},
            )
            _, load_seconds, generate_seconds, total_seconds = await self._execute_benchmark_request(
                manifest=manifest,
                runtime=runtime,
                invoke=lambda request=request: runtime.generate(request),
            )
            if suffix == "measured":
                return {
                    "runtime": runtime.name,
                    "model_id": manifest.model_id,
                    "load_seconds": round(load_seconds, 4),
                    "generate_seconds": round(generate_seconds, 4),
                    "total_seconds": round(total_seconds, 4),
                    "acceleration": request.metadata.get("mlx_acceleration", {}),
                    "prefix_cache": request.metadata.get("prefix_cache", {}),
                }
        raise AssertionError("Measured MLX acceleration request did not execute.")

    async def _timed_capability_request(
        self,
        *,
        capability: str,
        model_id: str,
        prompt: str,
        variant: int,
        reuse_key: str | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        if capability == CapabilityName.CHAT.value:
            execution = await self.chat_orchestrator.complete(
                model_id=model_id,
                messages=[
                    GenerateMessage(
                        role="user",
                        content=f"{prompt} [{reuse_key or variant}]",
                    ),
                ],
                max_tokens=48,
                temperature=0.0,
            )
            runtime_name = execution.routing.runtime_name
            resolved_model_id = execution.routing.model_id
        elif capability == CapabilityName.AUDIO_TRANSCRIPTION.value:
            execution = await self.multimodal_orchestrator.transcribe_audio(
                model_id=model_id,
                audio_bytes=_benchmark_audio_bytes(prompt),
                file_name="benchmark-audio.wav",
                language="en",
                prompt=prompt,
            )
            runtime_name = execution.routing.runtime_name
            resolved_model_id = execution.routing.model_id
        elif capability == CapabilityName.EMBEDDINGS.value:
            suffix = reuse_key or f"variant-{variant}"
            execution = await self.multimodal_orchestrator.embed(
                model_id=model_id,
                inputs=[f"{prompt} {suffix}", f"{prompt} secondary {suffix}"],
            )
            runtime_name = execution.routing.runtime_name
            resolved_model_id = execution.routing.model_id
        elif capability == CapabilityName.RERANK.value:
            suffix = reuse_key or f"variant-{variant}"
            execution = await self.multimodal_orchestrator.rerank(
                model_id=model_id,
                query=f"{prompt} {suffix}",
                documents=[
                    f"{prompt} {suffix}",
                    f"contrast document {suffix}",
                    "unrelated control document",
                ],
                top_n=3,
            )
            runtime_name = execution.routing.runtime_name
            resolved_model_id = execution.routing.model_id
        else:
            raise ConfigurationError(f"Benchmark scenarios do not support capability `{capability}`.")
        return {
            "elapsed_seconds": round(time.perf_counter() - started_at, 4),
            "runtime": runtime_name,
            "model_id": resolved_model_id,
        }

    def _workload_signature(
        self,
        *,
        capability: str,
        prompt: str,
        workload_class: str | None,
        repeat_count: int,
        model_ids: list[str],
        benchmark_count: int,
        result_payload: dict[str, Any],
        scenarios: list[BenchmarkScenarioReport],
    ) -> str:
        signature_payload = {
            "capability": capability,
            "prompt": prompt,
            "workload_class": workload_class,
            "repeat_count": repeat_count,
            "model_ids": sorted(model_ids),
            "benchmark_count": benchmark_count,
            "serving_profile": self._workload_serving_profile_signature(result_payload),
            "scenarios": [scenario.scenario for scenario in scenarios],
        }
        digest = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        mode = "suite" if benchmark_count > 1 else "single"
        return f"{capability}-{mode}-{len(model_ids)}m-r{repeat_count}-{digest}"

    def _workload_serving_profile_signature(self, result_payload: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        serving_profile = result_payload.get("serving_profile")
        if isinstance(serving_profile, dict):
            return {
                "profile_id": serving_profile.get("profile_id"),
                "workload_class": serving_profile.get("workload_class"),
                "effective_settings": serving_profile.get("effective_settings"),
            }
        results = result_payload.get("results")
        if not isinstance(results, list):
            return {"effective_settings": self._serving_profile_effective_settings({})}
        signature_items = []
        for item in results:
            if not isinstance(item, dict):
                continue
            item_profile = item.get("serving_profile")
            if not isinstance(item_profile, dict):
                continue
            signature_items.append(
                {
                    "model_id": item.get("model_id"),
                    "profile_id": item_profile.get("profile_id"),
                    "workload_class": item_profile.get("workload_class"),
                    "effective_settings": item_profile.get("effective_settings"),
                },
            )
        return sorted(signature_items, key=lambda item: (str(item.get("model_id")), str(item.get("profile_id"))))

    def _benchmark_artifact_path(self, *, created_at: datetime, workload_signature: str, artifact_id: str) -> Path:
        safe_signature = "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in workload_signature
        ).strip("-")
        timestamp = created_at.strftime("%Y%m%d-%H%M%S-%f")
        return self.settings.benchmarks_dir / f"{timestamp}-{safe_signature}-{artifact_id[:8]}.json"

    def _evaluate_regression(
        self,
        *,
        current_payload: dict[str, Any],
        baseline_payload: dict[str, Any] | None,
    ) -> BenchmarkRegressionSummary:
        if baseline_payload is None:
            return BenchmarkRegressionSummary(
                status="no_baseline",
                notes=["No prior artifact with the same workload signature was available for comparison."],
            )
        failures: list[BenchmarkRegressionFailure] = []
        current_result = current_payload.get("result", {})
        baseline_result = baseline_payload.get("result", {})
        current_total = _coerce_float(current_result.get("average_total_seconds", current_result.get("total_seconds")))
        baseline_total = _coerce_float(baseline_result.get("average_total_seconds", baseline_result.get("total_seconds")))
        if current_total is not None and baseline_total is not None:
            allowed_total = round((baseline_total * 1.25) + 0.05, 4)
            if current_total > allowed_total:
                failures.append(
                    BenchmarkRegressionFailure(
                        scope="suite" if "average_total_seconds" in current_result else "single",
                        metric="total_seconds",
                        current=round(current_total, 4),
                        baseline=round(baseline_total, 4),
                        allowed=allowed_total,
                        message="Total benchmark time regressed beyond the default tolerance.",
                    ),
                )
        current_models = {
            str(item["model_id"]): item
            for item in current_result.get("models", [])
            if isinstance(item, dict) and item.get("model_id") is not None
        }
        baseline_models = {
            str(item["model_id"]): item
            for item in baseline_result.get("models", [])
            if isinstance(item, dict) and item.get("model_id") is not None
        }
        for model_id, current_model in current_models.items():
            baseline_model = baseline_models.get(model_id)
            if baseline_model is None:
                continue
            current_average = _coerce_float(current_model.get("average_total_seconds"))
            baseline_average = _coerce_float(baseline_model.get("average_total_seconds"))
            if current_average is None or baseline_average is None:
                continue
            allowed_average = round((baseline_average * 1.35) + 0.05, 4)
            if current_average > allowed_average:
                failures.append(
                    BenchmarkRegressionFailure(
                        scope=f"model:{model_id}",
                        metric="average_total_seconds",
                        current=round(current_average, 4),
                        baseline=round(baseline_average, 4),
                        allowed=allowed_average,
                        message="Per-model average latency regressed beyond the default tolerance.",
                    ),
                )
        current_scenarios = {
            str(item["scenario"]): item
            for item in current_payload.get("scenarios", [])
            if isinstance(item, dict) and item.get("scenario") is not None
        }
        baseline_scenarios = {
            str(item["scenario"]): item
            for item in baseline_payload.get("scenarios", [])
            if isinstance(item, dict) and item.get("scenario") is not None
        }
        multimodal_reuse = current_scenarios.get("multimodal_reuse")
        if multimodal_reuse is not None and multimodal_reuse.get("status") == "observed":
            cache_hit_delta = _coerce_int(multimodal_reuse.get("metrics", {}).get("cache_hit_delta"))
            if cache_hit_delta < 1:
                failures.append(
                    BenchmarkRegressionFailure(
                        scope="scenario:multimodal_reuse",
                        metric="cache_hit_delta",
                        current=cache_hit_delta,
                        baseline=None,
                        allowed=1,
                        message="Deterministic multimodal reuse did not produce a persisted cache hit on replay.",
                    ),
                )
        repeated_prefix = current_scenarios.get("repeated_prefix")
        baseline_repeated_prefix = baseline_scenarios.get("repeated_prefix")
        current_prefix_ratio = (
            _coerce_float(repeated_prefix.get("metrics", {}).get("average_second_over_first_ratio"))
            if repeated_prefix is not None
            else None
        )
        baseline_prefix_ratio = (
            _coerce_float(baseline_repeated_prefix.get("metrics", {}).get("average_second_over_first_ratio"))
            if baseline_repeated_prefix is not None
            else None
        )
        if current_prefix_ratio is not None and baseline_prefix_ratio is not None:
            allowed_ratio = round(max(baseline_prefix_ratio * 1.25, 1.05), 4)
            if current_prefix_ratio > allowed_ratio:
                failures.append(
                    BenchmarkRegressionFailure(
                        scope="scenario:repeated_prefix",
                        metric="average_second_over_first_ratio",
                        current=round(current_prefix_ratio, 4),
                        baseline=round(baseline_prefix_ratio, 4),
                        allowed=allowed_ratio,
                        message="Repeated-prefix timing regressed beyond the default tolerance.",
                    ),
                )
        for scenario_name in ("request_queue_pressure", "model_load_admission"):
            current_scenario = current_scenarios.get(scenario_name)
            baseline_scenario = baseline_scenarios.get(scenario_name)
            if current_scenario is None or baseline_scenario is None:
                continue
            current_rejected = _coerce_int(current_scenario.get("metrics", {}).get("rejected_requests_delta"))
            baseline_rejected = _coerce_int(baseline_scenario.get("metrics", {}).get("rejected_requests_delta"))
            if current_rejected > baseline_rejected + 1:
                failures.append(
                    BenchmarkRegressionFailure(
                        scope=f"scenario:{scenario_name}",
                        metric="rejected_requests_delta",
                        current=current_rejected,
                        baseline=baseline_rejected,
                        allowed=baseline_rejected + 1,
                        message="Scheduler rejection count increased beyond the default tolerance.",
                    ),
                )
            current_timeouts = _coerce_int(current_scenario.get("metrics", {}).get("timed_out_requests_delta"))
            baseline_timeouts = _coerce_int(baseline_scenario.get("metrics", {}).get("timed_out_requests_delta"))
            if current_timeouts > baseline_timeouts:
                failures.append(
                    BenchmarkRegressionFailure(
                        scope=f"scenario:{scenario_name}",
                        metric="timed_out_requests_delta",
                        current=current_timeouts,
                        baseline=baseline_timeouts,
                        allowed=baseline_timeouts,
                        message="Scheduler timeout count increased relative to the previous artifact.",
                    ),
                )
        return BenchmarkRegressionSummary(
            status="failed" if failures else "passed",
            compared_to_artifact_id=(
                str(baseline_payload.get("artifact_id")) if baseline_payload.get("artifact_id") is not None else None
            ),
            compared_to_artifact_path=(
                str(baseline_payload.get("artifact_path")) if baseline_payload.get("artifact_path") is not None else None
            ),
            failure_count=len(failures),
            failures=failures,
            notes=["Default regression tolerances compare against the latest artifact with the same workload signature."],
        )

    def _cache_performance_features(
        self,
        *,
        cache_stats: CacheStats,
        runtime_snapshots: list[dict[str, object]],
    ) -> list[PerformanceFeatureStatus]:
        prefix_cache_entries = self._runtime_feature_entries(
            runtime_snapshots,
            PerformanceFeatureName.PREFIX_CACHE.value,
        )
        persistent_multi_context_entries = self._runtime_feature_entries(
            runtime_snapshots,
            PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE.value,
        )
        multimodal_encoder_entries = self._runtime_feature_entries(
            runtime_snapshots,
            PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING.value,
        )
        paged_kv_entries = self._runtime_feature_entries(
            runtime_snapshots,
            PerformanceFeatureName.PAGED_KV_CACHE.value,
        )
        kv_quantization_entries = self._runtime_feature_entries(
            runtime_snapshots,
            PerformanceFeatureName.KV_CACHE_QUANTIZATION.value,
        )
        paged_kv_runtime_names = sorted(
            entry["runtime_name"]
            for entry in paged_kv_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        kv_quantization_runtime_names = sorted(
            entry["runtime_name"]
            for entry in kv_quantization_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        prefix_cache_runtime_names = sorted(
            entry["runtime_name"]
            for entry in prefix_cache_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        persistent_multi_context_runtime_names = sorted(
            entry["runtime_name"]
            for entry in persistent_multi_context_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        multimodal_encoder_runtime_names = sorted(
            entry["runtime_name"]
            for entry in multimodal_encoder_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        return [
            self._feature_status(
                feature=PerformanceFeatureName.DISK_BACKED_CACHE,
                supported=True,
                active=True,
                supported_capabilities=sorted(_DETERMINISTIC_CACHE_CAPABILITY_NAMES),
                reason=(
                    "LewLM persists deterministic embeddings, rerank, audio transcription, and audio speech "
                    "responses in the shared runtime response cache."
                ),
                metrics=_compact_metrics(
                    runtime_response_count=cache_stats.runtime_response_count,
                    runtime_response_bytes=cache_stats.runtime_response_bytes,
                    runtime_cache_hits=cache_stats.runtime_cache_hits,
                    runtime_cache_misses=cache_stats.runtime_cache_misses,
                ),
                notes=["Chat and streaming requests instead populate the block-level multimodal feature cache."],
            ),
            self._feature_status(
                feature=PerformanceFeatureName.PREFIX_CACHE,
                supported=bool(prefix_cache_runtime_names),
                active=self._runtime_feature_active(prefix_cache_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if prefix_cache_runtime_names
                    else []
                ),
                runtime_names=prefix_cache_runtime_names,
                reason=(
                    "LewLM can reuse runtime-local prompt-prefix caches on "
                    + ", ".join(f"`{name}`" for name in prefix_cache_runtime_names)
                    + "."
                    if prefix_cache_runtime_names
                    else "LewLM does not implement a runtime prefix-cache surface today."
                ),
                metrics=self._runtime_feature_metrics(
                    prefix_cache_entries,
                    sum_keys=(
                        "page_size_tokens",
                        "cache_entries",
                        "cache_size_bytes",
                        "cache_hits",
                        "cache_misses",
                        "cache_saves",
                        "saved_prefill_tokens",
                        "resident_page_count",
                        "resident_page_size_bytes",
                        "page_hits",
                        "resident_page_hits",
                        "page_saves",
                        "copy_on_write_reused_pages",
                    ),
                    passthrough_keys=("max_saved_prefill_tokens", "page_size_tokens"),
                ),
                notes=self._runtime_feature_notes(prefix_cache_entries),
                fallback_guidance=(
                    []
                    if prefix_cache_runtime_names
                    else self._performance_fallback_guidance(PerformanceFeatureName.PREFIX_CACHE)
                ),
            ),
            self._feature_status(
                feature=PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE,
                supported=bool(persistent_multi_context_runtime_names),
                active=self._runtime_feature_active(persistent_multi_context_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if persistent_multi_context_runtime_names
                    else []
                ),
                runtime_names=persistent_multi_context_runtime_names,
                reason=(
                    "LewLM can restore persisted multi-context prompt-prefix cache entries on "
                    + ", ".join(f"`{name}`" for name in persistent_multi_context_runtime_names)
                    + "."
                    if persistent_multi_context_runtime_names
                    else "LewLM does not expose restart-resilient chat cache persistence on the active runtimes."
                ),
                metrics=self._runtime_feature_metrics(
                    persistent_multi_context_entries,
                    sum_keys=(
                        "resident_cache_entries",
                        "persisted_cache_entries",
                        "persisted_cache_size_bytes",
                        "resident_cache_hits",
                        "persistent_cache_hits",
                        "resident_page_count",
                        "resident_page_size_bytes",
                        "persisted_page_count",
                        "persisted_page_size_bytes",
                        "persistent_page_hits",
                        "cache_restores",
                        "page_restores",
                        "cache_evictions",
                        "page_evictions",
                        "cached_tokens",
                    ),
                    passthrough_keys=("page_size_tokens",),
                ),
                notes=self._runtime_feature_notes(persistent_multi_context_entries),
                fallback_guidance=(
                    []
                    if persistent_multi_context_runtime_names
                    else self._performance_fallback_guidance(PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE)
                ),
            ),
            self._feature_status(
                feature=PerformanceFeatureName.PAGED_KV_CACHE,
                supported=bool(paged_kv_runtime_names),
                active=self._runtime_feature_active(paged_kv_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if paged_kv_runtime_names
                    else []
                ),
                runtime_names=paged_kv_runtime_names,
                reason=(
                    "LewLM can configure runtime-local paged KV cache controls on "
                    + ", ".join(f"`{name}`" for name in paged_kv_runtime_names)
                    + "."
                    if paged_kv_runtime_names
                    else "LewLM does not expose paged KV-cache storage on the active runtimes."
                ),
                metrics=self._runtime_feature_metrics(
                    paged_kv_entries,
                    sum_keys=("requests_using_paged_kv", "paged_prompt_tokens"),
                    passthrough_keys=("page_size_tokens", "max_pages"),
                ),
                fallback_guidance=(
                    []
                    if paged_kv_runtime_names
                    else self._performance_fallback_guidance(PerformanceFeatureName.PAGED_KV_CACHE)
                ),
                notes=self._runtime_feature_notes(paged_kv_entries),
            ),
            self._feature_status(
                feature=PerformanceFeatureName.KV_CACHE_QUANTIZATION,
                supported=bool(kv_quantization_runtime_names),
                active=self._runtime_feature_active(kv_quantization_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if kv_quantization_runtime_names
                    else []
                ),
                runtime_names=kv_quantization_runtime_names,
                reason=(
                    "LewLM can configure runtime-local KV-cache quantization on "
                    + ", ".join(f"`{name}`" for name in kv_quantization_runtime_names)
                    + "."
                    if kv_quantization_runtime_names
                    else "LewLM does not quantize runtime KV caches separately from model weights."
                ),
                metrics=self._runtime_feature_metrics(
                    kv_quantization_entries,
                    sum_keys=("requests_using_quantized_kv",),
                    passthrough_keys=("quantization_bits",),
                ),
                fallback_guidance=(
                    []
                    if kv_quantization_runtime_names
                    else self._performance_fallback_guidance(PerformanceFeatureName.KV_CACHE_QUANTIZATION)
                ),
                notes=self._runtime_feature_notes(kv_quantization_entries),
            ),
            self._feature_status(
                feature=PerformanceFeatureName.BLOCK_DISK_CACHE,
                supported=True,
                active=True,
                supported_capabilities=[CapabilityName.CHAT.value],
                reason=(
                    "LewLM persists reusable attachment feature blocks in the local cache directory and indexes "
                    "them through SQLite metadata."
                ),
                metrics=_compact_metrics(
                    block_cache_count=cache_stats.block_cache_count,
                    block_cache_bytes=cache_stats.block_cache_bytes,
                    block_cache_hits=cache_stats.block_cache_hits,
                    block_cache_misses=cache_stats.block_cache_misses,
                ),
                notes=["Each block stores a prompt-ready local artifact feature payload for later reuse."],
            ),
            self._feature_status(
                feature=PerformanceFeatureName.MULTIMODAL_FEATURE_CACHING,
                supported=True,
                active=True,
                supported_capabilities=[CapabilityName.CHAT.value],
                reason=(
                    "LewLM persists prompt-ready attachment feature payloads for local text, document, image, and "
                    "audio inputs before they are routed into chat or responses requests."
                ),
                metrics=_compact_metrics(
                    multimodal_feature_count=cache_stats.multimodal_feature_count,
                    multimodal_feature_bytes=cache_stats.multimodal_feature_bytes,
                    multimodal_feature_cache_hits=cache_stats.multimodal_feature_cache_hits,
                    multimodal_feature_cache_misses=cache_stats.multimodal_feature_cache_misses,
                ),
                notes=["Audio transcription and speech responses still reuse the deterministic runtime response cache."],
            ),
            self._feature_status(
                feature=PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING,
                supported=bool(multimodal_encoder_runtime_names),
                active=self._runtime_feature_active(multimodal_encoder_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value, CapabilityName.AUDIO_TRANSCRIPTION.value]
                    if multimodal_encoder_runtime_names
                    else []
                ),
                runtime_names=multimodal_encoder_runtime_names,
                reason=(
                    "LewLM can reuse runtime-native multimodal encoder features on "
                    + ", ".join(f"`{name}`" for name in multimodal_encoder_runtime_names)
                    + "."
                    if multimodal_encoder_runtime_names
                    else "LewLM does not detect a runtime-native multimodal encoder cache hook on the active runtimes."
                ),
                metrics={
                    **_compact_metrics(
                        multimodal_encoder_count=cache_stats.multimodal_encoder_count,
                        multimodal_encoder_bytes=cache_stats.multimodal_encoder_bytes,
                        multimodal_encoder_cache_hits=cache_stats.multimodal_encoder_cache_hits,
                        multimodal_encoder_cache_misses=cache_stats.multimodal_encoder_cache_misses,
                        multimodal_encoder_cache_invalidations=cache_stats.multimodal_encoder_cache_invalidations,
                        multimodal_encoder_resident_count=cache_stats.multimodal_encoder_resident_count,
                        multimodal_encoder_resident_bytes=cache_stats.multimodal_encoder_resident_bytes,
                    ),
                    **self._runtime_feature_metrics(
                        multimodal_encoder_entries,
                        sum_keys=(
                            "request_count",
                            "cache_hits",
                            "cache_misses",
                            "cached_image_inputs",
                            "cached_frame_inputs",
                            "cached_bundle_requests",
                        ),
                    ),
                },
                notes=self._runtime_feature_notes(multimodal_encoder_entries),
                fallback_guidance=(
                    []
                    if multimodal_encoder_runtime_names
                    else self._performance_fallback_guidance(PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING)
                ),
            ),
        ]

    def _performance_features(
        self,
        *,
        runtime_health: list[dict[str, Any]],
        request_scheduler: RuntimeSchedulerStats,
        load_scheduler: RuntimeSchedulerStats,
        request_metrics: RuntimeRequestMetrics,
        cache_stats: CacheStats,
        capability_focus: str | None = None,
    ) -> list[PerformanceFeatureStatus]:
        available_runtime_names = self._runtime_names_for_capabilities(runtime_health)
        embeddings_runtime_names = self._runtime_names_for_capabilities(
            runtime_health,
            required_capabilities={CapabilityName.EMBEDDINGS.value},
        )
        deterministic_cache_runtime_names = self._runtime_names_for_capabilities(
            runtime_health,
            required_capabilities=_DETERMINISTIC_CACHE_CAPABILITY_NAMES,
        )
        chat_runtime_names = self._runtime_names_for_capabilities(
            runtime_health,
            required_capabilities={CapabilityName.CHAT.value},
        )
        serving_core_snapshot = self.chat_orchestrator.serving_core.snapshot()
        serving_core_supported = bool(chat_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if serving_core_supported:
            serving_core_reason = (
                "LewLM now maintains a runtime-agnostic serving-core state model for chat and streaming requests "
                "and maps backend-native batch paths in as adapter details."
            )
            serving_core_notes = [
                "Use `runtime_stats` or lifecycle event payloads when you need active sequence state, queue residency, or recent finalization details.",
            ]
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            serving_core_reason = f"Selected `{capability_focus}` requests do not use the chat serving-core state tracker."
            serving_core_notes = []
        else:
            serving_core_reason = "No available chat-capable runtime is registered for serving-core state tracking."
            serving_core_notes = []
        audio_transcription_runtime_names = self._runtime_names_for_capabilities(
            runtime_health,
            required_capabilities={CapabilityName.AUDIO_TRANSCRIPTION.value},
        )
        embeddings_metrics = self._capability_metrics_entry(request_metrics, CapabilityName.EMBEDDINGS.value)
        continuous_batching_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.CONTINUOUS_BATCHING.value,
        )
        distributed_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.DISTRIBUTED_PIPELINE.value,
        )
        lewlm_owned_runtime_names = sorted(
            entry["runtime_name"]
            for entry in continuous_batching_entries
            if (
                entry.get("supported")
                and isinstance(entry.get("runtime_name"), str)
                and self._continuous_batching_ownership(entry) == "lewlm_owned"
            )
        )
        backend_native_runtime_names = sorted(
            entry["runtime_name"]
            for entry in continuous_batching_entries
            if (
                entry.get("supported")
                and isinstance(entry.get("runtime_name"), str)
                and self._continuous_batching_ownership(entry) == "backend_native"
            )
        )
        native_runtime_names = sorted({*lewlm_owned_runtime_names, *backend_native_runtime_names})
        distributed_runtime_names = sorted(
            entry["runtime_name"]
            for entry in distributed_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        cluster_status = self.cluster_service.status() if self.cluster_service is not None else None
        embeddings_batching_supported = bool(embeddings_runtime_names) and capability_focus in {
            None,
            CapabilityName.EMBEDDINGS.value,
        }
        native_batching_supported = bool(native_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        continuous_batching_supported = embeddings_batching_supported or native_batching_supported
        if lewlm_owned_runtime_names and backend_native_runtime_names:
            text_batching_mode = "mixed"
            text_batching_reason = (
                "LewLM owns the primary chat or streaming continuous-batching path on "
                + ", ".join(f"`{name}`" for name in lewlm_owned_runtime_names)
                + " while "
                + ", ".join(f"`{name}`" for name in backend_native_runtime_names)
                + " remains backend-native."
            )
        elif lewlm_owned_runtime_names:
            text_batching_mode = "lewlm_owned"
            text_batching_reason = (
                "LewLM owns the primary chat or streaming continuous-batching path on "
                + ", ".join(f"`{name}`" for name in lewlm_owned_runtime_names)
                + "."
            )
        elif backend_native_runtime_names:
            text_batching_mode = "backend_native"
            text_batching_reason = (
                "LewLM can currently route chat or streaming requests only through backend-native continuous batching on "
                + ", ".join(f"`{name}`" for name in backend_native_runtime_names)
                + "; a LewLM-owned primary text scheduler is not active on this host."
            )
        else:
            text_batching_mode = "unsupported"
            text_batching_reason = ""
        if native_batching_supported and embeddings_batching_supported:
            continuous_batching_reason = (
                "LewLM batches embeddings requests over a short model-local window and " + text_batching_reason.lower()
            )
            continuous_batching_notes = self._runtime_feature_notes(continuous_batching_entries)
        elif native_batching_supported:
            continuous_batching_reason = text_batching_reason
            continuous_batching_notes = self._runtime_feature_notes(continuous_batching_entries)
            if backend_native_runtime_names and not lewlm_owned_runtime_names:
                continuous_batching_notes = [
                    *continuous_batching_notes,
                    "Chat and streaming continuous batching remain runtime-dependent until a LewLM-owned primary text scheduler is active.",
                ]
        elif embeddings_batching_supported:
            continuous_batching_reason = (
                "LewLM batches embeddings requests by model over a short window before issuing one backend "
                "`embed(...)` call."
            )
            continuous_batching_notes = (
                []
                if capability_focus == CapabilityName.EMBEDDINGS.value
                else ["Chat and streaming continuous batching remain runtime-dependent on this host."]
            )
        elif capability_focus in {CapabilityName.CHAT.value, CapabilityName.STREAMING.value}:
            continuous_batching_reason = (
                f"Selected `{capability_focus}` requests do not have a LewLM-owned or backend-native continuous batching path."
            )
            continuous_batching_notes = self._runtime_feature_notes(continuous_batching_entries)
        elif capability_focus == CapabilityName.EMBEDDINGS.value:
            continuous_batching_reason = "No available embeddings runtime is registered on this host."
            continuous_batching_notes = []
        else:
            continuous_batching_reason = (
                "No active runtime currently advertises embeddings batching or a chat/streaming continuous batching path."
            )
            continuous_batching_notes = self._runtime_feature_notes(continuous_batching_entries)
        distributed_supported = bool(distributed_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if distributed_supported:
            distributed_reason = (
                "LewLM can coordinate an experimental multi-host pipeline-parallel proof executor on "
                + ", ".join(f"`{name}`" for name in distributed_runtime_names)
                + "."
            )
            distributed_notes = self._runtime_feature_notes(distributed_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            distributed_reason = f"Selected `{capability_focus}` requests do not use distributed pipeline execution."
            distributed_notes = []
        else:
            distributed_reason = (
                "LewLM does not currently have enough enrolled ready workers to route distributed experimental models."
            )
            distributed_notes = self._runtime_feature_notes(distributed_entries)

        disk_backed_cache_supported = bool(deterministic_cache_runtime_names) and capability_focus in {
            None,
            *sorted(_DETERMINISTIC_CACHE_CAPABILITY_NAMES),
        }
        if disk_backed_cache_supported:
            disk_backed_cache_reason = (
                "LewLM persists deterministic runtime responses in SQLite-backed cache storage."
            )
            disk_backed_cache_notes = (
                []
                if capability_focus in _DETERMINISTIC_CACHE_CAPABILITY_NAMES
                else ["Cached runtime responses currently cover embeddings, rerank, audio transcription, and speech."]
            )
        elif capability_focus is not None and capability_focus not in _DETERMINISTIC_CACHE_CAPABILITY_NAMES:
            disk_backed_cache_reason = (
                f"Selected `{capability_focus}` requests do not use the persisted runtime response cache."
            )
            disk_backed_cache_notes = (
                ["Deterministic cache reuse is limited to embeddings, rerank, audio transcription, and speech."]
                if deterministic_cache_runtime_names
                else []
            )
        else:
            disk_backed_cache_reason = (
                "No available deterministic runtime capability is registered for persisted runtime-response caching."
            )
            disk_backed_cache_notes = []
        disk_backed_cache_metrics = _compact_metrics(
            runtime_response_count=cache_stats.runtime_response_count,
            runtime_response_bytes=cache_stats.runtime_response_bytes,
            runtime_cache_hits=cache_stats.runtime_cache_hits,
            runtime_cache_misses=cache_stats.runtime_cache_misses,
        )
        if capability_focus in _DETERMINISTIC_CACHE_CAPABILITY_NAMES:
            focused_cache_metrics = self._capability_metrics_entry(request_metrics, capability_focus)
            if focused_cache_metrics is not None:
                disk_backed_cache_metrics = {
                    **disk_backed_cache_metrics,
                    **_compact_metrics(
                        capability_cache_hits=focused_cache_metrics.metric_totals.get("cache_hits"),
                        capability_cache_misses=focused_cache_metrics.metric_totals.get("cache_misses"),
                    ),
                }

        block_disk_cache_supported = bool(chat_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
        }
        if block_disk_cache_supported:
            block_disk_cache_reason = (
                "LewLM persists reusable attachment feature blocks on disk for chat and responses requests."
            )
            block_disk_cache_notes = (
                []
                if capability_focus == CapabilityName.CHAT.value
                else ["Block entries are populated only when requests include local or uploaded attachments."]
            )
        elif capability_focus is not None and capability_focus != CapabilityName.CHAT.value:
            block_disk_cache_reason = (
                f"Selected `{capability_focus}` requests do not populate LewLM's attachment block cache."
            )
            block_disk_cache_notes = (
                ["Block entries are populated by chat and responses requests with attachments."]
                if chat_runtime_names
                else []
            )
        else:
            block_disk_cache_reason = "No available chat-capable runtime is registered on this host."
            block_disk_cache_notes = []
        block_disk_cache_metrics = _compact_metrics(
            block_cache_count=cache_stats.block_cache_count,
            block_cache_bytes=cache_stats.block_cache_bytes,
            block_cache_hits=cache_stats.block_cache_hits,
            block_cache_misses=cache_stats.block_cache_misses,
        )

        multimodal_feature_cache_supported = bool(chat_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
        }
        if multimodal_feature_cache_supported:
            multimodal_feature_cache_reason = (
                "LewLM caches prompt-ready attachment feature payloads before forwarding multimodal chat work to the "
                "selected runtime."
            )
            multimodal_feature_cache_notes = (
                []
                if capability_focus == CapabilityName.CHAT.value
                else ["Feature entries are populated only when chat or responses requests include attachments."]
            )
        elif capability_focus is not None and capability_focus != CapabilityName.CHAT.value:
            multimodal_feature_cache_reason = (
                f"Selected `{capability_focus}` requests bypass LewLM's attachment feature cache."
            )
            multimodal_feature_cache_notes = (
                ["Feature entries are populated by chat and responses requests with attachments."]
                if chat_runtime_names
                else []
            )
        else:
            multimodal_feature_cache_reason = "No available chat-capable runtime is registered on this host."
            multimodal_feature_cache_notes = []
        multimodal_feature_cache_metrics = _compact_metrics(
            multimodal_feature_count=cache_stats.multimodal_feature_count,
            multimodal_feature_bytes=cache_stats.multimodal_feature_bytes,
            multimodal_feature_cache_hits=cache_stats.multimodal_feature_cache_hits,
            multimodal_feature_cache_misses=cache_stats.multimodal_feature_cache_misses,
        )
        multimodal_encoder_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING.value,
        )
        multimodal_encoder_runtime_names = sorted(
            entry["runtime_name"]
            for entry in multimodal_encoder_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        multimodal_encoder_cache_supported = bool(multimodal_encoder_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
            CapabilityName.AUDIO_TRANSCRIPTION.value,
        }
        if multimodal_encoder_cache_supported:
            multimodal_encoder_cache_reason = (
                "LewLM can reuse runtime-native image, frame-bundle, and audio encoder outputs on "
                + ", ".join(f"`{name}`" for name in multimodal_encoder_runtime_names)
                + "."
            )
            multimodal_encoder_cache_notes = self._runtime_feature_notes(multimodal_encoder_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
            CapabilityName.AUDIO_TRANSCRIPTION.value,
        }:
            multimodal_encoder_cache_reason = (
                f"Selected `{capability_focus}` requests do not use LewLM's runtime-native multimodal encoder cache."
            )
            multimodal_encoder_cache_notes = (
                ["Encoder-cache reuse currently applies to multimodal chat, streaming, and audio transcription."]
                if multimodal_encoder_runtime_names
                else []
            )
        else:
            multimodal_encoder_cache_reason = (
                "LewLM does not detect a compatible runtime-native multimodal encoder cache hook on the active runtimes."
            )
            multimodal_encoder_cache_notes = self._runtime_feature_notes(multimodal_encoder_entries)
        multimodal_encoder_cache_metrics = {
            **_compact_metrics(
                multimodal_encoder_count=cache_stats.multimodal_encoder_count,
                multimodal_encoder_bytes=cache_stats.multimodal_encoder_bytes,
                multimodal_encoder_cache_hits=cache_stats.multimodal_encoder_cache_hits,
                multimodal_encoder_cache_misses=cache_stats.multimodal_encoder_cache_misses,
                multimodal_encoder_cache_invalidations=cache_stats.multimodal_encoder_cache_invalidations,
                multimodal_encoder_resident_count=cache_stats.multimodal_encoder_resident_count,
                multimodal_encoder_resident_bytes=cache_stats.multimodal_encoder_resident_bytes,
            ),
            **self._runtime_feature_metrics(
                multimodal_encoder_entries,
                sum_keys=(
                    "request_count",
                    "cache_hits",
                    "cache_misses",
                    "cached_image_inputs",
                    "cached_frame_inputs",
                    "cached_bundle_requests",
                ),
            ),
        }
        speculative_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.SPECULATIVE_DECODING.value,
        )
        prompt_lookup_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.PROMPT_LOOKUP_SPECULATION.value,
        )
        prefix_cache_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.PREFIX_CACHE.value,
        )
        persistent_multi_context_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE.value,
        )
        speculative_runtime_names = sorted(
            entry["runtime_name"]
            for entry in speculative_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        prompt_lookup_runtime_names = sorted(
            entry["runtime_name"]
            for entry in prompt_lookup_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        prefix_cache_runtime_names = sorted(
            entry["runtime_name"]
            for entry in prefix_cache_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        persistent_multi_context_runtime_names = sorted(
            entry["runtime_name"]
            for entry in persistent_multi_context_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )

        request_scheduling_supported = request_scheduler.max_concurrent_requests > 0
        load_admission_supported = load_scheduler.max_concurrent_requests > 0
        decode_priority_supported = request_scheduling_supported
        if decode_priority_supported:
            decode_priority_reason = (
                "LewLM can reserve scheduler preference for decode-lane work so short interactive requests do not sit behind long-prefill admissions."
            )
            decode_priority_notes = (
                []
                if request_scheduler.decode_priority_enabled
                else ["Decode-priority scheduling support exists, but it is currently disabled in settings."]
            )
        else:
            decode_priority_reason = (
                "Runtime request admission is currently unbounded, so LewLM cannot apply decode-priority queue shaping."
            )
            decode_priority_notes = []
        speculative_supported = bool(speculative_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if speculative_supported:
            speculative_reason = (
                "LewLM can use draft-model speculative decoding on "
                + ", ".join(f"`{name}`" for name in speculative_runtime_names)
                + "."
            )
            speculative_notes = self._runtime_feature_notes(speculative_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            speculative_reason = f"Selected `{capability_focus}` requests do not use draft-model speculative decoding."
            speculative_notes = (
                ["Speculative decoding currently applies to chat and streaming generation."]
                if speculative_runtime_names
                else []
            )
        else:
            speculative_reason = "LewLM does not configure draft-model speculative decoding on the active runtimes."
            speculative_notes = []
        prompt_lookup_supported = bool(prompt_lookup_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if prompt_lookup_supported:
            prompt_lookup_reason = (
                "LewLM can use prompt-lookup or n-gram speculation on "
                + ", ".join(f"`{name}`" for name in prompt_lookup_runtime_names)
                + "."
            )
            prompt_lookup_notes = self._runtime_feature_notes(prompt_lookup_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            prompt_lookup_reason = f"Selected `{capability_focus}` requests do not use prompt-lookup speculation."
            prompt_lookup_notes = (
                ["Prompt-lookup speculation currently applies to chat and streaming generation."]
                if prompt_lookup_runtime_names
                else []
            )
        else:
            prompt_lookup_reason = "LewLM does not implement prompt-lookup or n-gram speculative decoding paths today."
            prompt_lookup_notes = []
        prefix_cache_supported = bool(prefix_cache_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if prefix_cache_supported:
            prefix_cache_reason = (
                "LewLM can reuse runtime-local prompt-prefix caches on "
                + ", ".join(f"`{name}`" for name in prefix_cache_runtime_names)
                + "."
            )
            prefix_cache_notes = self._runtime_feature_notes(prefix_cache_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            prefix_cache_reason = f"Selected `{capability_focus}` requests do not use runtime-local prompt-prefix caches."
            prefix_cache_notes = (
                ["Prefix-cache reuse currently applies to chat and streaming generation."]
                if prefix_cache_runtime_names
                else []
            )
        else:
            prefix_cache_reason = "LewLM does not implement runtime prefix-cache reuse on the active backends."
            prefix_cache_notes = []
        persistent_multi_context_supported = bool(persistent_multi_context_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if persistent_multi_context_supported:
            persistent_multi_context_reason = (
                "LewLM can restore persisted multi-context prompt-prefix cache entries on "
                + ", ".join(f"`{name}`" for name in persistent_multi_context_runtime_names)
                + "."
            )
            persistent_multi_context_notes = self._runtime_feature_notes(persistent_multi_context_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            persistent_multi_context_reason = (
                f"Selected `{capability_focus}` requests do not use LewLM's persisted multi-context chat cache."
            )
            persistent_multi_context_notes = (
                ["Persistent multi-context cache reuse currently applies to chat and streaming generation."]
                if persistent_multi_context_runtime_names
                else []
            )
        else:
            persistent_multi_context_reason = (
                "LewLM does not expose restart-resilient multi-context chat cache persistence on the active backends."
            )
            persistent_multi_context_notes = []
        graph_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.GRAPH_COMPILATION.value,
        )
        attention_kernel_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION.value,
        )
        graph_runtime_names = sorted(
            entry["runtime_name"]
            for entry in graph_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        attention_kernel_runtime_names = sorted(
            entry["runtime_name"]
            for entry in attention_kernel_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        graph_supported = bool(graph_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if graph_supported:
            graph_reason = (
                "LewLM can request MLX graph compilation on "
                + ", ".join(f"`{name}`" for name in graph_runtime_names)
                + "."
            )
            graph_notes = self._runtime_feature_notes(graph_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            graph_reason = f"Selected `{capability_focus}` requests do not use MLX graph compilation."
            graph_notes = (
                ["Graph compilation is currently limited to chat and streaming generation."]
                if graph_runtime_names
                else []
            )
        else:
            graph_reason = "LewLM does not detect a stable MLX graph-compilation hook on the active runtimes."
            graph_notes = self._runtime_feature_notes(graph_entries)
        attention_kernel_supported = bool(attention_kernel_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if attention_kernel_supported:
            attention_kernel_reason = (
                "LewLM can request accelerated attention kernels on "
                + ", ".join(f"`{name}`" for name in attention_kernel_runtime_names)
                + "."
            )
            attention_kernel_notes = self._runtime_feature_notes(attention_kernel_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            attention_kernel_reason = f"Selected `{capability_focus}` requests do not use accelerated attention kernels."
            attention_kernel_notes = (
                ["Accelerated attention hooks are currently limited to chat and streaming generation."]
                if attention_kernel_runtime_names
                else []
            )
        else:
            attention_kernel_reason = "LewLM does not detect an accelerated attention-kernel hook on the active runtimes."
            attention_kernel_notes = self._runtime_feature_notes(attention_kernel_entries)
        paged_kv_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.PAGED_KV_CACHE.value,
        )
        kv_quantization_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.KV_CACHE_QUANTIZATION.value,
        )
        prefill_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.PREFILL_OPTIMIZATION.value,
        )
        chunked_prefill_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.CHUNKED_PREFILL.value,
        )
        prefill_isolation_entries = self._runtime_feature_entries(
            runtime_health,
            PerformanceFeatureName.PREFILL_ISOLATION.value,
        )
        paged_kv_runtime_names = sorted(
            entry["runtime_name"]
            for entry in paged_kv_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        kv_quantization_runtime_names = sorted(
            entry["runtime_name"]
            for entry in kv_quantization_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        prefill_runtime_names = sorted(
            entry["runtime_name"]
            for entry in prefill_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        chunked_prefill_runtime_names = sorted(
            entry["runtime_name"]
            for entry in chunked_prefill_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        prefill_isolation_runtime_names = sorted(
            entry["runtime_name"]
            for entry in prefill_isolation_entries
            if entry.get("supported") and isinstance(entry.get("runtime_name"), str)
        )
        paged_kv_supported = bool(paged_kv_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if paged_kv_supported:
            paged_kv_reason = (
                "LewLM owns first-class paged-KV residency accounting on "
                + ", ".join(f"`{name}`" for name in paged_kv_runtime_names)
                + " and reports page reuse, eviction, and lane pressure without overclaiming backend parity elsewhere."
            )
            paged_kv_notes = self._runtime_feature_notes(paged_kv_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            paged_kv_reason = f"Selected `{capability_focus}` requests do not use first-class paged-KV residency accounting."
            paged_kv_notes = (
                ["Paged KV cache is currently limited to chat and streaming generation."]
                if paged_kv_runtime_names
                else []
            )
        else:
            paged_kv_reason = "LewLM does not expose first-class paged-KV residency accounting on the active runtimes."
            paged_kv_notes = self._runtime_feature_notes(paged_kv_entries)
        kv_quantization_supported = bool(kv_quantization_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if kv_quantization_supported:
            kv_quantization_reason = (
                "LewLM can configure runtime-local KV-cache quantization on "
                + ", ".join(f"`{name}`" for name in kv_quantization_runtime_names)
                + "."
            )
            kv_quantization_notes = self._runtime_feature_notes(kv_quantization_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            kv_quantization_reason = f"Selected `{capability_focus}` requests do not use runtime-local KV-cache quantization."
            kv_quantization_notes = (
                ["KV-cache quantization is currently limited to chat and streaming generation."]
                if kv_quantization_runtime_names
                else []
            )
        else:
            kv_quantization_reason = "LewLM does not quantize runtime KV caches separately from model weights."
            kv_quantization_notes = self._runtime_feature_notes(kv_quantization_entries)
        prefill_supported = bool(prefill_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if prefill_supported:
            prefill_reason = (
                "LewLM can apply runtime-local prefill optimization on "
                + ", ".join(f"`{name}`" for name in prefill_runtime_names)
                + "."
            )
            prefill_notes = self._runtime_feature_notes(prefill_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            prefill_reason = f"Selected `{capability_focus}` requests do not use runtime-local prefill optimization."
            prefill_notes = (
                ["Prefill optimization is currently limited to chat and streaming generation."]
                if prefill_runtime_names
                else []
            )
        else:
            prefill_reason = (
                "LewLM publishes prefill lifecycle events, but it does not yet enable runtime-specific prefill "
                "acceleration or cache reuse."
            )
            prefill_notes = self._runtime_feature_notes(prefill_entries)
        chunked_prefill_supported = bool(chunked_prefill_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if chunked_prefill_supported:
            chunked_prefill_reason = (
                "LewLM can split long prompt ingest into bounded prefill chunks on "
                + ", ".join(f"`{name}`" for name in chunked_prefill_runtime_names)
                + "."
            )
            chunked_prefill_notes = self._runtime_feature_notes(chunked_prefill_entries)
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            chunked_prefill_reason = f"Selected `{capability_focus}` requests do not use chunked prefill controls."
            chunked_prefill_notes = (
                ["Chunked prefill is currently limited to chat and streaming generation."]
                if chunked_prefill_runtime_names
                else []
            )
        else:
            chunked_prefill_reason = "LewLM does not detect chunked-prefill support on the active runtimes."
            chunked_prefill_notes = self._runtime_feature_notes(chunked_prefill_entries)
        prefill_isolation_supported = bool(prefill_isolation_runtime_names) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if prefill_isolation_supported:
            prefill_isolation_reason = (
                "LewLM can reserve decode headroom while long-prefill requests are active on "
                + ", ".join(f"`{name}`" for name in prefill_isolation_runtime_names)
                + "."
            )
            prefill_isolation_notes = self._runtime_feature_notes(prefill_isolation_entries)
            if not request_scheduler.prefill_isolation_enabled:
                prefill_isolation_notes = [
                    *prefill_isolation_notes,
                    "The runtime capability is present, but scheduler-side prefill isolation is currently disabled in settings.",
                ]
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            prefill_isolation_reason = f"Selected `{capability_focus}` requests do not use prefill isolation."
            prefill_isolation_notes = (
                ["Prefill isolation is currently limited to chat and streaming generation."]
                if prefill_isolation_runtime_names
                else []
            )
        else:
            prefill_isolation_reason = (
                "LewLM does not detect a runtime that exposes the combined chunked-prefill and continuous-batching hooks needed for truthful single-host prefill isolation."
            )
            prefill_isolation_notes = self._runtime_feature_notes(prefill_isolation_entries)
        frontier_manifests = [
            manifest
            for manifest in self.model_router.model_registry.list_manifests()
            if ModelModality.TEXT in manifest.modality
        ]
        frontier_plans = [
            plan
            for manifest in frontier_manifests
            if (plan := build_frontier_serving_plan(manifest=manifest, settings=self.settings)) is not None
        ]
        hybrid_ssm_plans = [
            plan
            for plan in frontier_plans
            if plan.get("architecture_subtype") in {"ssm_mamba", "hybrid_ssm"}
        ]
        moe_plans = [
            plan
            for plan in frontier_plans
            if plan.get("architecture_subtype") in {"moe", "hybrid_moe"}
        ]
        chat_metrics = self._capability_metrics_entry(request_metrics, CapabilityName.CHAT.value)
        streaming_metrics = self._capability_metrics_entry(request_metrics, CapabilityName.STREAMING.value)
        frontier_metric_entries = [entry for entry in (chat_metrics, streaming_metrics) if entry is not None]
        frontier_sample_metrics = (
            chat_metrics.metric_averages
            if chat_metrics is not None and chat_metrics.metric_averages
            else (streaming_metrics.metric_averages if streaming_metrics is not None else {})
        )
        hybrid_ssm_request_count = sum(
            _coerce_int(entry.metric_totals.get("frontier_ssm_requests")) for entry in frontier_metric_entries
        )
        moe_request_count = sum(
            _coerce_int(entry.metric_totals.get("frontier_moe_requests")) for entry in frontier_metric_entries
        )
        bounded_memory_request_count = sum(
            _coerce_int(entry.metric_totals.get("frontier_bounded_memory_requests")) for entry in frontier_metric_entries
        )
        hybrid_ssm_supported = bool(hybrid_ssm_plans) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if hybrid_ssm_supported:
            hybrid_ssm_reason = (
                (
                    f"LewLM recorded {hybrid_ssm_request_count} hybrid SSM/Mamba request(s) with realized execution-state metrics."
                    if hybrid_ssm_request_count > 0
                    else f"LewLM can detect {len(hybrid_ssm_plans)} hybrid SSM/Mamba model(s) and carry architecture-aware routing notes for chat or streaming requests."
                )
            )
            hybrid_ssm_notes = [
                "Detected models: "
                + ", ".join(
                    manifest.display_name
                    for manifest in frontier_manifests
                    if manifest.architecture_subtype.value in {"ssm_mamba", "hybrid_ssm"}
                )
            ]
            hybrid_ssm_notes.extend(frontier_plan_notes(hybrid_ssm_plans[0]))
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            hybrid_ssm_reason = f"Selected `{capability_focus}` requests do not use hybrid SSM routing notes."
            hybrid_ssm_notes = []
        else:
            hybrid_ssm_reason = "No discovered text model is currently classified as a hybrid SSM or Mamba-family architecture."
            hybrid_ssm_notes = []
        ssm_cache_supported = bool(hybrid_ssm_plans)
        if ssm_cache_supported:
            sample_ssm_plan = hybrid_ssm_plans[0]
            ssm_cache_reason = (
                (
                    "LewLM recorded realized hybrid SSM state-cache allocations and reuse metrics for measured requests."
                    if hybrid_ssm_request_count > 0
                    else "LewLM records specialized cache-state handling plans for detected hybrid SSM architectures so runtime stats and benchmarks can distinguish selective-scan state from standard transformer KV state."
                )
            )
            ssm_cache_notes = frontier_plan_notes(sample_ssm_plan)
        else:
            sample_ssm_plan = None
            ssm_cache_reason = "LewLM has not detected a hybrid SSM architecture that needs specialized cache-state handling."
            ssm_cache_notes = []
        moe_supported = bool(moe_plans) and capability_focus in {
            None,
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }
        if moe_supported:
            sample_moe_plan = max(
                moe_plans,
                key=lambda item: _coerce_int(item.get("full_estimated_memory_mb")),
            )
            moe_reason = (
                (
                    f"LewLM recorded {bounded_memory_request_count or moe_request_count} MoE request(s) with realized `{self.settings.moe_bounded_memory_mode}` bounded-memory execution metrics."
                    if moe_request_count > 0
                    else f"LewLM can detect {len(moe_plans)} MoE model(s) and build `{self.settings.moe_bounded_memory_mode}` bounded-memory serving plans with explicit expert-residency tradeoffs."
                )
            )
            moe_notes = [
                "Detected MoE models: "
                + ", ".join(
                    manifest.display_name
                    for manifest in frontier_manifests
                    if manifest.architecture_subtype.value in {"moe", "hybrid_moe"}
                )
            ]
            moe_notes.extend(frontier_plan_notes(sample_moe_plan))
        elif capability_focus is not None and capability_focus not in {
            CapabilityName.CHAT.value,
            CapabilityName.STREAMING.value,
        }:
            sample_moe_plan = None
            moe_reason = f"Selected `{capability_focus}` requests do not use MoE bounded-memory serving plans."
            moe_notes = []
        else:
            sample_moe_plan = None
            moe_reason = "No discovered text model is currently classified as a MoE architecture."
            moe_notes = []

        return [
            self._feature_status(
                feature=PerformanceFeatureName.HYBRID_SSM_ROUTING,
                supported=hybrid_ssm_supported,
                active=hybrid_ssm_request_count > 0,
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value] if hybrid_ssm_plans else []
                ),
                reason=hybrid_ssm_reason,
                metrics=_compact_metrics(
                    detected_model_count=len(hybrid_ssm_plans),
                    request_count=hybrid_ssm_request_count,
                    sample_state_size=(sample_ssm_plan or {}).get("state_size"),
                    sample_cache_state_handling=(sample_ssm_plan or {}).get("cache_state_handling"),
                    sample_state_cache_bytes=frontier_sample_metrics.get("frontier_state_cache_bytes"),
                    sample_state_cache_hits=frontier_sample_metrics.get("frontier_state_cache_hits"),
                    sample_state_cache_misses=frontier_sample_metrics.get("frontier_state_cache_misses"),
                    sample_estimated_state_cache_kb_per_token=(
                        (sample_ssm_plan or {}).get("estimated_state_cache_kb_per_token")
                    ),
                ),
                fallback_guidance=(
                    []
                    if hybrid_ssm_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.HYBRID_SSM_ROUTING)
                ),
                notes=hybrid_ssm_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.SSM_STATE_CACHE_HANDLING,
                supported=ssm_cache_supported,
                active=hybrid_ssm_request_count > 0,
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value] if hybrid_ssm_plans else []
                ),
                reason=ssm_cache_reason,
                metrics=_compact_metrics(
                    detected_model_count=len(hybrid_ssm_plans),
                    request_count=hybrid_ssm_request_count,
                    sample_cache_state_handling=(sample_ssm_plan or {}).get("cache_state_handling"),
                    sample_state_size=(sample_ssm_plan or {}).get("state_size"),
                    sample_full_estimated_memory_mb=(sample_ssm_plan or {}).get("full_estimated_memory_mb"),
                    sample_state_cache_bytes=frontier_sample_metrics.get("frontier_state_cache_bytes"),
                    sample_state_cache_hits=frontier_sample_metrics.get("frontier_state_cache_hits"),
                    sample_state_cache_misses=frontier_sample_metrics.get("frontier_state_cache_misses"),
                ),
                fallback_guidance=(
                    []
                    if ssm_cache_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.SSM_STATE_CACHE_HANDLING)
                ),
                notes=ssm_cache_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING,
                supported=moe_supported,
                active=bounded_memory_request_count > 0 or (bool(moe_plans) and self.settings.moe_bounded_memory_mode != "off"),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value] if moe_plans else []
                ),
                reason=moe_reason,
                metrics=_compact_metrics(
                    detected_model_count=len(moe_plans),
                    request_count=moe_request_count,
                    bounded_memory_request_count=bounded_memory_request_count,
                    configured_mode=self.settings.moe_bounded_memory_mode,
                    configured_resident_expert_count=self.settings.moe_resident_expert_count,
                    total_expert_count=sum(_coerce_int(plan.get("expert_count")) for plan in moe_plans),
                    sample_full_estimated_memory_mb=(sample_moe_plan or {}).get("full_estimated_memory_mb"),
                    sample_planned_memory_mb=(sample_moe_plan or {}).get("planned_memory_mb"),
                    sample_effective_loaded_memory_mb=frontier_sample_metrics.get("frontier_effective_loaded_memory_mb"),
                    sample_memory_savings_mb=(sample_moe_plan or {}).get("memory_savings_mb"),
                    sample_resident_expert_count=(sample_moe_plan or {}).get("resident_expert_count"),
                    sample_requested_expert_count=frontier_sample_metrics.get("frontier_requested_expert_count"),
                    sample_expert_swap_count=frontier_sample_metrics.get("frontier_expert_swap_count"),
                    sample_expert_swap_mb=frontier_sample_metrics.get("frontier_expert_swap_mb"),
                    sample_streamed_expert_count=(sample_moe_plan or {}).get("streamed_expert_count"),
                    sample_estimated_swap_mb_per_request=(sample_moe_plan or {}).get("estimated_swap_mb_per_request"),
                ),
                fallback_guidance=(
                    []
                    if moe_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING)
                ),
                notes=moe_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.DISTRIBUTED_PIPELINE,
                supported=distributed_supported,
                active=bool(cluster_status is not None and cluster_status.ready_worker_count >= 2),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if distributed_runtime_names
                    else []
                ),
                runtime_names=distributed_runtime_names,
                reason=distributed_reason,
                metrics={
                    **_compact_metrics(
                        ready_worker_count=(cluster_status.ready_worker_count if cluster_status is not None else None),
                        stale_worker_count=(cluster_status.stale_worker_count if cluster_status is not None else None),
                        plan_count=(cluster_status.plan_count if cluster_status is not None else None),
                    ),
                    **self._runtime_feature_metrics(
                        distributed_entries,
                        sum_keys=("ready_worker_count", "stale_worker_count", "plan_count", "recovery_count"),
                        passthrough_keys=(
                            "stage_count",
                            "worker_count",
                            "average_stage_elapsed_seconds",
                            "pipeline_latency_seconds",
                            "critical_path_seconds",
                            "throughput_tokens_per_second",
                            "completion_tokens_per_second",
                            "average_stage_utilization",
                            "average_prefetch_tokens",
                            "effective_batch_tokens",
                            "average_network_latency_ms",
                            "heterogeneity_ratio",
                            "speedup_vs_single_host_percent",
                            "network_share_percent",
                            "compute_share_percent",
                            "scheduling_share_percent",
                            "pipeline_overlap_efficiency_percent",
                            "bottleneck",
                        ),
                    ),
                },
                fallback_guidance=(
                    []
                    if distributed_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.DISTRIBUTED_PIPELINE)
                ),
                notes=distributed_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.SERVING_CORE,
                supported=serving_core_supported,
                active=serving_core_snapshot.total_sequences_started > 0,
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if serving_core_supported
                    else []
                ),
                runtime_names=chat_runtime_names,
                reason=serving_core_reason,
                metrics=_compact_metrics(
                    total_sequences_started=serving_core_snapshot.total_sequences_started,
                    total_sequences_completed=serving_core_snapshot.total_sequences_completed,
                    total_sequences_failed=serving_core_snapshot.total_sequences_failed,
                    total_cancellation_requests=serving_core_snapshot.total_cancellation_requests,
                    active_sequence_count=serving_core_snapshot.active_sequence_count,
                    active_stream_count=serving_core_snapshot.active_stream_count,
                    recent_sequence_count=serving_core_snapshot.recent_sequence_count,
                ),
                fallback_guidance=(
                    []
                    if serving_core_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.SERVING_CORE)
                ),
                notes=serving_core_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.CONTINUOUS_BATCHING,
                supported=continuous_batching_supported,
                active=continuous_batching_supported,
                supported_capabilities=sorted(
                    {
                        *(
                            [CapabilityName.EMBEDDINGS.value]
                            if embeddings_runtime_names
                            else []
                        ),
                        *(
                            [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                            if native_runtime_names
                            else []
                        ),
                    }
                ),
                runtime_names=sorted({*embeddings_runtime_names, *native_runtime_names}),
                reason=continuous_batching_reason,
                metrics=_compact_metrics(
                    chat_streaming_ownership_mode=text_batching_mode,
                    lewlm_owned_runtime_count=len(lewlm_owned_runtime_names),
                    backend_native_runtime_count=len(backend_native_runtime_names),
                    batched_requests=(
                        embeddings_metrics.metric_totals.get("batched_requests") if embeddings_metrics is not None else None
                    ),
                    coalesced_requests=(
                        embeddings_metrics.metric_totals.get("coalesced_requests")
                        if embeddings_metrics is not None
                        else None
                    ),
                    average_batch_size=(
                        embeddings_metrics.metric_averages.get("batch_size") if embeddings_metrics is not None else None
                    ),
                    native_total_batches=request_scheduler.native_total_batches,
                    native_batched_requests=request_scheduler.native_batched_requests,
                    native_coalesced_requests=request_scheduler.native_coalesced_requests,
                    native_average_batch_size=request_scheduler.native_average_batch_size,
                    native_average_batch_utilization=request_scheduler.native_average_batch_utilization,
                    native_average_queue_delay_seconds=request_scheduler.native_average_queue_delay_seconds,
                    native_max_queue_delay_seconds=request_scheduler.native_max_queue_delay_seconds,
                    frontier_total_batches=request_scheduler.frontier_total_batches,
                    frontier_batched_requests=request_scheduler.frontier_batched_requests,
                    frontier_coalesced_requests=request_scheduler.frontier_coalesced_requests,
                    frontier_average_batch_size=request_scheduler.frontier_average_batch_size,
                    frontier_average_batch_utilization=request_scheduler.frontier_average_batch_utilization,
                    frontier_average_queue_delay_seconds=request_scheduler.frontier_average_queue_delay_seconds,
                    frontier_max_queue_delay_seconds=request_scheduler.frontier_max_queue_delay_seconds,
                ),
                fallback_guidance=(
                    []
                    if continuous_batching_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.CONTINUOUS_BATCHING)
                ),
                notes=continuous_batching_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.PREFIX_CACHE,
                supported=prefix_cache_supported,
                active=self._runtime_feature_active(prefix_cache_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if prefix_cache_runtime_names
                    else []
                ),
                runtime_names=prefix_cache_runtime_names,
                reason=prefix_cache_reason,
                metrics=self._runtime_feature_metrics(
                    prefix_cache_entries,
                    sum_keys=(
                        "page_size_tokens",
                        "cache_entries",
                        "cache_size_bytes",
                        "cache_hits",
                        "cache_misses",
                        "cache_saves",
                        "saved_prefill_tokens",
                        "resident_page_count",
                        "resident_page_size_bytes",
                        "page_hits",
                        "resident_page_hits",
                        "page_saves",
                        "copy_on_write_reused_pages",
                    ),
                    passthrough_keys=("max_saved_prefill_tokens", "page_size_tokens"),
                ),
                fallback_guidance=(
                    []
                    if prefix_cache_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.PREFIX_CACHE)
                ),
                notes=prefix_cache_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE,
                supported=persistent_multi_context_supported,
                active=self._runtime_feature_active(persistent_multi_context_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if persistent_multi_context_runtime_names
                    else []
                ),
                runtime_names=persistent_multi_context_runtime_names,
                reason=persistent_multi_context_reason,
                metrics=self._runtime_feature_metrics(
                    persistent_multi_context_entries,
                    sum_keys=(
                        "resident_cache_entries",
                        "persisted_cache_entries",
                        "persisted_cache_size_bytes",
                        "resident_cache_hits",
                        "persistent_cache_hits",
                        "resident_page_count",
                        "resident_page_size_bytes",
                        "persisted_page_count",
                        "persisted_page_size_bytes",
                        "persistent_page_hits",
                        "cache_restores",
                        "page_restores",
                        "cache_evictions",
                        "page_evictions",
                        "cached_tokens",
                    ),
                    passthrough_keys=("page_size_tokens",),
                ),
                fallback_guidance=(
                    []
                    if persistent_multi_context_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE)
                ),
                notes=persistent_multi_context_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.GRAPH_COMPILATION,
                supported=graph_supported,
                active=self._runtime_feature_active(graph_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if graph_runtime_names
                    else []
                ),
                runtime_names=graph_runtime_names,
                reason=graph_reason,
                metrics=self._runtime_feature_metrics(
                    graph_entries,
                    sum_keys=("compile_attempts", "compiled_requests", "compile_fallback_requests", "compile_failures"),
                    passthrough_keys=("configured_enabled", "compiled_callable_count"),
                ),
                fallback_guidance=(
                    []
                    if graph_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.GRAPH_COMPILATION)
                ),
                notes=graph_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION,
                supported=attention_kernel_supported,
                active=self._runtime_feature_active(attention_kernel_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if attention_kernel_runtime_names
                    else []
                ),
                runtime_names=attention_kernel_runtime_names,
                reason=attention_kernel_reason,
                metrics=self._runtime_feature_metrics(
                    attention_kernel_entries,
                    sum_keys=(
                        "stock_requests",
                        "flash_attention_requests",
                        "custom_sdpa_requests",
                        "kernel_fallback_requests",
                    ),
                    passthrough_keys=("configured_mode", "preferred_mode", "supported_modes", "kernel_parameter", "last_kernel_path"),
                ),
                fallback_guidance=(
                    []
                    if attention_kernel_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION)
                ),
                notes=attention_kernel_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.PAGED_KV_CACHE,
                supported=paged_kv_supported,
                active=self._runtime_feature_active(paged_kv_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if paged_kv_runtime_names
                    else []
                ),
                runtime_names=paged_kv_runtime_names,
                reason=paged_kv_reason,
                metrics=self._runtime_feature_metrics(
                    paged_kv_entries,
                    sum_keys=(
                        "requests_using_paged_kv",
                        "paged_prompt_tokens",
                        "resident_pages",
                        "active_pages",
                        "active_decode_pages",
                        "active_prefill_pages",
                        "resident_decode_pages",
                        "resident_prefill_pages",
                        "decode_lane_reservations",
                        "prefill_lane_reservations",
                        "reused_pages",
                        "new_pages",
                        "evicted_pages",
                        "prefill_evicted_pages",
                        "decode_evicted_pages",
                        "decode_headroom_preservation_events",
                        "prefill_decode_tradeoff_events",
                        "overflow_events",
                        "overflow_pages",
                        "high_pressure_events",
                        "peak_resident_pages",
                        "peak_total_pages",
                    ),
                    passthrough_keys=(
                        "page_size_tokens",
                        "max_pages",
                        "native_control_supported",
                        "pressure_ratio",
                        "peak_pressure_ratio",
                        "pressure_level",
                    ),
                    preserve_zero_sum_keys=(
                        "requests_using_paged_kv",
                        "paged_prompt_tokens",
                        "resident_pages",
                        "active_pages",
                        "active_decode_pages",
                        "active_prefill_pages",
                        "resident_decode_pages",
                        "resident_prefill_pages",
                        "decode_lane_reservations",
                        "prefill_lane_reservations",
                        "reused_pages",
                        "new_pages",
                        "evicted_pages",
                        "prefill_evicted_pages",
                        "decode_evicted_pages",
                        "decode_headroom_preservation_events",
                        "prefill_decode_tradeoff_events",
                        "overflow_events",
                        "overflow_pages",
                        "high_pressure_events",
                        "peak_resident_pages",
                        "peak_total_pages",
                    ),
                ),
                fallback_guidance=(
                    []
                    if paged_kv_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.PAGED_KV_CACHE)
                ),
                notes=paged_kv_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.KV_CACHE_QUANTIZATION,
                supported=kv_quantization_supported,
                active=self._runtime_feature_active(kv_quantization_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if kv_quantization_runtime_names
                    else []
                ),
                runtime_names=kv_quantization_runtime_names,
                reason=kv_quantization_reason,
                metrics=self._runtime_feature_metrics(
                    kv_quantization_entries,
                    sum_keys=("requests_using_quantized_kv",),
                    passthrough_keys=("quantization_bits",),
                ),
                fallback_guidance=(
                    []
                    if kv_quantization_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.KV_CACHE_QUANTIZATION)
                ),
                notes=kv_quantization_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.DISK_BACKED_CACHE,
                supported=disk_backed_cache_supported,
                active=disk_backed_cache_supported,
                supported_capabilities=sorted(_DETERMINISTIC_CACHE_CAPABILITY_NAMES) if deterministic_cache_runtime_names else [],
                runtime_names=deterministic_cache_runtime_names,
                reason=disk_backed_cache_reason,
                metrics=disk_backed_cache_metrics,
                fallback_guidance=(
                    []
                    if disk_backed_cache_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.DISK_BACKED_CACHE)
                ),
                notes=disk_backed_cache_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.BLOCK_DISK_CACHE,
                supported=block_disk_cache_supported,
                active=block_disk_cache_supported,
                supported_capabilities=[CapabilityName.CHAT.value] if chat_runtime_names else [],
                runtime_names=chat_runtime_names,
                reason=block_disk_cache_reason,
                metrics=block_disk_cache_metrics,
                fallback_guidance=(
                    []
                    if block_disk_cache_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.BLOCK_DISK_CACHE)
                ),
                notes=block_disk_cache_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.SPECULATIVE_DECODING,
                supported=speculative_supported,
                active=self._runtime_feature_active(speculative_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if speculative_runtime_names
                    else []
                ),
                runtime_names=speculative_runtime_names,
                reason=speculative_reason,
                metrics=self._runtime_feature_metrics(
                    speculative_entries,
                    sum_keys=("request_count", "drafted_tokens", "verified_tokens", "rollback_tokens"),
                    passthrough_keys=("configured_num_draft_tokens",),
                ),
                fallback_guidance=(
                    []
                    if speculative_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.SPECULATIVE_DECODING)
                ),
                notes=speculative_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.PROMPT_LOOKUP_SPECULATION,
                supported=prompt_lookup_supported,
                active=self._runtime_feature_active(prompt_lookup_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if prompt_lookup_runtime_names
                    else []
                ),
                runtime_names=prompt_lookup_runtime_names,
                reason=prompt_lookup_reason,
                metrics=self._runtime_feature_metrics(
                    prompt_lookup_entries,
                    sum_keys=("request_count",),
                    passthrough_keys=("configured_max_ngram_size", "configured_num_pred_tokens"),
                ),
                fallback_guidance=(
                    []
                    if prompt_lookup_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.PROMPT_LOOKUP_SPECULATION)
                ),
                notes=prompt_lookup_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.KEEP_WARM_MODEL_RESIDENCY,
                supported=bool(available_runtime_names),
                active=bool(available_runtime_names) and self.settings.runtime_policy == "keep_warm",
                runtime_names=available_runtime_names,
                reason=(
                    "LewLM can keep models resident and call runtime warm hooks before requests when "
                    "`runtime_policy=keep_warm`."
                    if available_runtime_names
                    else "No available runtime is currently registered on this host."
                ),
                metrics=_compact_metrics(
                    loaded_model_count=self._sum_runtime_health_metric(runtime_health, "loaded_model_count"),
                    peak_loaded_model_count=self._sum_runtime_health_metric(runtime_health, "peak_loaded_model_count"),
                    total_warm_count=self._sum_runtime_health_metric(runtime_health, "total_warm_count"),
                ),
                fallback_guidance=(
                    []
                    if available_runtime_names
                    else self._performance_fallback_guidance(PerformanceFeatureName.KEEP_WARM_MODEL_RESIDENCY)
                ),
                notes=(
                    []
                    if self.settings.runtime_policy == "keep_warm"
                    else [f"Current runtime policy is `{self.settings.runtime_policy}`."]
                ),
            ),
            self._feature_status(
                feature=PerformanceFeatureName.AGGRESSIVE_UNLOAD_MODE,
                supported=bool(available_runtime_names),
                active=bool(available_runtime_names) and self.settings.runtime_policy == "aggressive_unload",
                runtime_names=available_runtime_names,
                reason=(
                    "LewLM can unload the just-used model after each request when `runtime_policy=aggressive_unload`."
                    if available_runtime_names
                    else "No available runtime is currently registered on this host."
                ),
                metrics=_compact_metrics(
                    loaded_model_count=self._sum_runtime_health_metric(runtime_health, "loaded_model_count"),
                    total_unload_count=self._sum_runtime_health_metric(runtime_health, "total_unload_count"),
                ),
                fallback_guidance=(
                    []
                    if available_runtime_names
                    else self._performance_fallback_guidance(PerformanceFeatureName.AGGRESSIVE_UNLOAD_MODE)
                ),
                notes=(
                    []
                    if self.settings.runtime_policy == "aggressive_unload"
                    else [f"Current runtime policy is `{self.settings.runtime_policy}`."]
                ),
            ),
            self._feature_status(
                feature=PerformanceFeatureName.BALANCED_RESIDENCY_MODE,
                supported=bool(available_runtime_names),
                active=bool(available_runtime_names) and self.settings.runtime_policy == "balanced",
                runtime_names=available_runtime_names,
                reason=(
                    "LewLM can unload other loaded models after a request completes when `runtime_policy=balanced`."
                    if available_runtime_names
                    else "No available runtime is currently registered on this host."
                ),
                metrics=_compact_metrics(
                    loaded_model_count=self._sum_runtime_health_metric(runtime_health, "loaded_model_count"),
                    peak_loaded_model_count=self._sum_runtime_health_metric(runtime_health, "peak_loaded_model_count"),
                    total_model_switch_count=self._sum_runtime_health_metric(runtime_health, "total_model_switch_count"),
                ),
                fallback_guidance=(
                    []
                    if available_runtime_names
                    else self._performance_fallback_guidance(PerformanceFeatureName.BALANCED_RESIDENCY_MODE)
                ),
                notes=(
                    []
                    if self.settings.runtime_policy == "balanced"
                    else [f"Current runtime policy is `{self.settings.runtime_policy}`."]
                ),
            ),
            self._feature_status(
                feature=PerformanceFeatureName.REQUEST_SCHEDULING_AND_BACKPRESSURE,
                supported=request_scheduling_supported,
                active=request_scheduling_supported,
                reason=(
                    "LewLM gates runtime work through a bounded in-memory scheduler that can queue or reject requests."
                    if request_scheduling_supported
                    else "Runtime request admission is currently unbounded because `max_concurrent_runtime_requests` is 0."
                ),
                metrics=_compact_metrics(
                    max_concurrent_requests=request_scheduler.max_concurrent_requests,
                    queue_limit=request_scheduler.queue_limit,
                    queue_timeout_seconds=request_scheduler.queue_timeout_seconds,
                    total_queued_requests=request_scheduler.total_queued_requests,
                    rejected_requests=request_scheduler.rejected_requests,
                    timed_out_requests=request_scheduler.timed_out_requests,
                    max_observed_queue_depth=request_scheduler.max_observed_queue_depth,
                    queued_decode_requests=request_scheduler.queued_decode_requests,
                    queued_prefill_requests=request_scheduler.queued_prefill_requests,
                    average_queue_wait_seconds=request_scheduler.average_queue_wait_seconds,
                    max_queue_wait_seconds=request_scheduler.max_queue_wait_seconds,
                ),
                fallback_guidance=(
                    []
                    if request_scheduling_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.REQUEST_SCHEDULING_AND_BACKPRESSURE)
                ),
                notes=(
                    []
                    if not request_scheduling_supported
                    else [
                        f"Decode lane queued={request_scheduler.queued_decode_requests}, prefill lane queued={request_scheduler.queued_prefill_requests}.",
                    ]
                ),
            ),
            self._feature_status(
                feature=PerformanceFeatureName.DECODE_PRIORITY_SCHEDULING,
                supported=decode_priority_supported,
                active=request_scheduling_supported and request_scheduler.decode_priority_enabled,
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if request_scheduling_supported
                    else []
                ),
                reason=decode_priority_reason,
                metrics=_compact_metrics(
                    decode_priority_enabled=request_scheduler.decode_priority_enabled,
                    long_prefill_token_threshold=request_scheduler.long_prefill_token_threshold,
                    decode_priority_requests=request_scheduler.decode_priority_requests,
                    prioritized_decode_grants=request_scheduler.prioritized_decode_grants,
                    active_decode_requests=request_scheduler.active_decode_requests,
                    queued_decode_requests=request_scheduler.queued_decode_requests,
                ),
                fallback_guidance=(
                    []
                    if decode_priority_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.DECODE_PRIORITY_SCHEDULING)
                ),
                notes=decode_priority_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.MODEL_LOAD_ADMISSION_CONTROL,
                supported=load_admission_supported,
                active=load_admission_supported,
                reason=(
                    "LewLM separates cold-model load concurrency from general request concurrency with a dedicated scheduler."
                    if load_admission_supported
                    else "Cold-model load admission is currently unbounded because `max_concurrent_model_loads` is 0."
                ),
                metrics=_compact_metrics(
                    max_concurrent_requests=load_scheduler.max_concurrent_requests,
                    queue_limit=load_scheduler.queue_limit,
                    queue_timeout_seconds=load_scheduler.queue_timeout_seconds,
                    total_queued_requests=load_scheduler.total_queued_requests,
                    rejected_requests=load_scheduler.rejected_requests,
                    timed_out_requests=load_scheduler.timed_out_requests,
                    max_observed_queue_depth=load_scheduler.max_observed_queue_depth,
                ),
                fallback_guidance=(
                    []
                    if load_admission_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.MODEL_LOAD_ADMISSION_CONTROL)
                ),
            ),
            self._feature_status(
                feature=PerformanceFeatureName.PREFILL_OPTIMIZATION,
                supported=prefill_supported,
                active=self._runtime_feature_active(prefill_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if prefill_runtime_names
                    else []
                ),
                runtime_names=prefill_runtime_names,
                reason=prefill_reason,
                metrics=self._runtime_feature_metrics(
                    prefill_entries,
                    sum_keys=("optimized_requests", "optimized_prompt_tokens", "prefill_batches_planned"),
                    passthrough_keys=("prefill_token_batch_size",),
                ),
                fallback_guidance=(
                    []
                    if prefill_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.PREFILL_OPTIMIZATION)
                ),
                notes=prefill_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.CHUNKED_PREFILL,
                supported=chunked_prefill_supported,
                active=self._runtime_feature_active(chunked_prefill_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if chunked_prefill_runtime_names
                    else []
                ),
                runtime_names=chunked_prefill_runtime_names,
                reason=chunked_prefill_reason,
                metrics=self._runtime_feature_metrics(
                    chunked_prefill_entries,
                    sum_keys=("chunked_requests", "chunked_prompt_tokens", "chunk_count"),
                    passthrough_keys=("prefill_token_batch_size",),
                ),
                fallback_guidance=(
                    []
                    if chunked_prefill_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.CHUNKED_PREFILL)
                ),
                notes=chunked_prefill_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.PREFILL_ISOLATION,
                supported=prefill_isolation_supported,
                active=request_scheduler.prefill_isolation_enabled and request_scheduler.isolated_prefill_requests > 0,
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value]
                    if prefill_isolation_runtime_names
                    else []
                ),
                runtime_names=prefill_isolation_runtime_names,
                reason=prefill_isolation_reason,
                metrics={
                    **self._runtime_feature_metrics(
                        prefill_isolation_entries,
                        sum_keys=("chunked_requests", "chunk_count"),
                        passthrough_keys=("prefill_token_batch_size",),
                    ),
                    **_compact_metrics(
                        prefill_isolation_enabled=request_scheduler.prefill_isolation_enabled,
                        prefill_isolation_max_concurrent_requests=request_scheduler.prefill_isolation_max_concurrent_requests,
                        prefill_isolation_decode_reserve=request_scheduler.prefill_isolation_decode_reserve,
                        isolated_prefill_requests=request_scheduler.isolated_prefill_requests,
                        active_prefill_requests=request_scheduler.active_prefill_requests,
                        queued_prefill_requests=request_scheduler.queued_prefill_requests,
                    ),
                },
                fallback_guidance=(
                    []
                    if prefill_isolation_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.PREFILL_ISOLATION)
                ),
                notes=prefill_isolation_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.MULTIMODAL_FEATURE_CACHING,
                supported=multimodal_feature_cache_supported,
                active=multimodal_feature_cache_supported,
                supported_capabilities=[CapabilityName.CHAT.value] if chat_runtime_names else [],
                runtime_names=chat_runtime_names,
                reason=multimodal_feature_cache_reason,
                metrics=multimodal_feature_cache_metrics,
                fallback_guidance=(
                    []
                    if multimodal_feature_cache_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.MULTIMODAL_FEATURE_CACHING)
                ),
                notes=multimodal_feature_cache_notes,
            ),
            self._feature_status(
                feature=PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING,
                supported=multimodal_encoder_cache_supported,
                active=self._runtime_feature_active(multimodal_encoder_entries),
                supported_capabilities=(
                    [CapabilityName.CHAT.value, CapabilityName.STREAMING.value, CapabilityName.AUDIO_TRANSCRIPTION.value]
                    if multimodal_encoder_runtime_names
                    else []
                ),
                runtime_names=multimodal_encoder_runtime_names,
                reason=multimodal_encoder_cache_reason,
                metrics=multimodal_encoder_cache_metrics,
                fallback_guidance=(
                    []
                    if multimodal_encoder_cache_supported
                    else self._performance_fallback_guidance(PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING)
                ),
                notes=multimodal_encoder_cache_notes,
            ),
        ]

    @staticmethod
    def _capability_metrics_entry(
        request_metrics: RuntimeRequestMetrics,
        capability: str,
    ) -> CapabilityRuntimeMetrics | None:
        return next((item for item in request_metrics.capabilities if item.capability == capability), None)

    @staticmethod
    def _runtime_names_for_capabilities(
        runtime_health: list[dict[str, Any]],
        required_capabilities: set[str] | frozenset[str] | None = None,
    ) -> list[str]:
        runtime_names: list[str] = []
        for runtime in runtime_health:
            if not runtime.get("available"):
                continue
            supported_capabilities = {
                str(value)
                for value in runtime.get("supported_capabilities", [])
                if isinstance(value, str)
            }
            if required_capabilities and not supported_capabilities.intersection(required_capabilities):
                continue
            name = runtime.get("name")
            if isinstance(name, str):
                runtime_names.append(name)
        return sorted(runtime_names)

    @staticmethod
    def _runtime_feature_entries(
        runtime_health: list[dict[str, Any]],
        feature_name: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for runtime in runtime_health:
            feature_map = runtime.get("performance_features")
            if not isinstance(feature_map, dict):
                continue
            feature = feature_map.get(feature_name)
            if not isinstance(feature, dict):
                continue
            entries.append({"runtime_name": runtime.get("name"), **feature})
        return entries

    @staticmethod
    def _runtime_feature_active(entries: list[dict[str, Any]]) -> bool:
        return any(bool(entry.get("active")) for entry in entries)

    @staticmethod
    def _runtime_feature_notes(entries: list[dict[str, Any]]) -> list[str]:
        notes: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            for note in entry.get("notes", []):
                if not isinstance(note, str) or not note or note in seen:
                    continue
                seen.add(note)
                notes.append(note)
        return notes

    @staticmethod
    def _continuous_batching_ownership(entry: dict[str, Any]) -> str:
        ownership = entry.get("ownership")
        if ownership in {"lewlm_owned", "backend_native", "unsupported"}:
            return str(ownership)
        return "backend_native" if bool(entry.get("supported")) else "unsupported"

    @staticmethod
    def _runtime_feature_metrics(
        entries: list[dict[str, Any]],
        *,
        sum_keys: tuple[str, ...],
        passthrough_keys: tuple[str, ...] = (),
        preserve_zero_sum_keys: tuple[str, ...] = (),
    ) -> dict[str, int | float | str | bool]:
        metrics: dict[str, int | float | str | bool] = {}
        for key in sum_keys:
            total = sum(_coerce_int((entry.get("metrics") or {}).get(key)) for entry in entries)
            if total or key in preserve_zero_sum_keys:
                metrics[key] = total
        for key in passthrough_keys:
            for entry in entries:
                feature_metrics = entry.get("metrics")
                if not isinstance(feature_metrics, dict):
                    continue
                value = feature_metrics.get(key)
                if value is None:
                    continue
                metrics[key] = value
                break
        return metrics

    @staticmethod
    def _sum_runtime_health_metric(runtime_health: list[dict[str, Any]], key: str) -> int:
        return sum(_coerce_int(runtime.get(key)) for runtime in runtime_health)

    @staticmethod
    def _feature_status(
        *,
        feature: PerformanceFeatureName,
        supported: bool,
        reason: str,
        active: bool = False,
        supported_capabilities: list[str] | None = None,
        runtime_names: list[str] | None = None,
        metrics: dict[str, int | float | str | bool] | None = None,
        fallback_guidance: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> PerformanceFeatureStatus:
        return PerformanceFeatureStatus(
            feature=feature,
            supported=supported,
            active=active,
            supported_capabilities=list(supported_capabilities or []),
            runtime_names=list(runtime_names or []),
            reason=reason,
            metrics=dict(metrics or {}),
            fallback_guidance=list(fallback_guidance or []),
            notes=list(notes or []),
        )

    @staticmethod
    def _performance_fallback_guidance(feature: PerformanceFeatureName) -> list[str]:
        if feature == PerformanceFeatureName.SERVING_CORE:
            return [
                "Register at least one chat-capable runtime before relying on serving-core sequence and queue diagnostics.",
                "Use request and event metadata on chat or streaming calls once a chat runtime is available.",
            ]
        if feature == PerformanceFeatureName.CONTINUOUS_BATCHING:
            return [
                "Choose a runtime that advertises `continuous_batching` for chat or streaming bursts on this host.",
                "Use one embeddings request with multiple inputs when you need the existing semantic batch path.",
            ]
        if feature == PerformanceFeatureName.DISTRIBUTED_PIPELINE:
            return [
                "Run one coordinator plus at least two enrolled workers with a shared `LEWLM_CLUSTER_ENROLLMENT_SECRET`.",
                "Add `distributed_pipeline.json` metadata to the target model bundle so LewLM can publish a deterministic stage plan.",
            ]
        if feature == PerformanceFeatureName.PREFIX_CACHE:
            return [
                "Use `keep_warm` residency to avoid repeated cold loads while prefix-cache reuse is unavailable.",
                "Prefer shorter prompts or smaller local models when repeated prompt prefixes dominate latency.",
            ]
        if feature == PerformanceFeatureName.PERSISTENT_MULTI_CONTEXT_CACHE:
            return [
                "Use session history plus `keep_warm` residency when restart-resilient chat cache persistence is unavailable.",
                "Inspect benchmark artifacts for the warm-chat-cache scenario to confirm whether the active runtime exposes TTFT wins.",
            ]
        if feature in {
            PerformanceFeatureName.HYBRID_SSM_ROUTING,
            PerformanceFeatureName.SSM_STATE_CACHE_HANDLING,
        }:
            return [
                "Use the capability report to confirm whether the detected SSM-family model can run on a concrete backend today.",
                "Benchmark artifacts can still disclose cache-state handling and tradeoff notes even when native SSM kernels are unavailable.",
            ]
        if feature == PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING:
            return [
                "Enable `moe_bounded_memory_mode` and tune `moe_resident_expert_count` when you want LewLM to publish bounded-memory MoE plans.",
                "Choose a smaller quantized MoE or dense fallback when the detected expert count still exceeds host memory limits.",
            ]
        if feature in {
            PerformanceFeatureName.GRAPH_COMPILATION,
            PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION,
        }:
            return [
                "Stay on the stock MLX path or choose a smaller local model when Apple-native acceleration hooks are unavailable.",
                "Use benchmark artifacts on an Apple Silicon host to confirm whether compiled or accelerated attention paths are actually active.",
            ]
        if feature in {
            PerformanceFeatureName.PAGED_KV_CACHE,
            PerformanceFeatureName.KV_CACHE_QUANTIZATION,
            PerformanceFeatureName.PREFILL_OPTIMIZATION,
            PerformanceFeatureName.CHUNKED_PREFILL,
        }:
            return [
                "Reduce prompt/context size or choose a smaller quantized model bundle on this host.",
                "Use LewLM residency policies to trade memory usage against repeated-load latency.",
            ]
        if feature == PerformanceFeatureName.PREFILL_ISOLATION:
            return [
                "Use a runtime that advertises both chunked prefill and continuous batching before enabling prefill isolation on this host.",
                "Keep decode-priority scheduling enabled so interactive requests can still bypass long-prefill work when isolation is inactive.",
            ]
        if feature == PerformanceFeatureName.DECODE_PRIORITY_SCHEDULING:
            return [
                "Enable bounded runtime admission and decode-priority scheduling in settings to protect short interactive requests from long prompt bursts.",
            ]
        if feature in {
            PerformanceFeatureName.BLOCK_DISK_CACHE,
            PerformanceFeatureName.MULTIMODAL_FEATURE_CACHING,
            PerformanceFeatureName.MULTIMODAL_ENCODER_CACHING,
        }:
            return [
                "Rely on the current persisted runtime response cache for deterministic semantic and audio requests.",
                "Inspect `GET /v1/cache/stats` or `lewlm cache --json` to confirm what the current cache layer stores.",
            ]
        if feature == PerformanceFeatureName.DISK_BACKED_CACHE:
            return [
                "Use deterministic embeddings, rerank, audio transcription, or speech requests to populate the persisted runtime response cache.",
            ]
        if feature in {
            PerformanceFeatureName.SPECULATIVE_DECODING,
            PerformanceFeatureName.PROMPT_LOOKUP_SPECULATION,
        }:
            return [
                "Use a faster local model or lower `max_tokens` when speculative decoding is unavailable.",
            ]
        if feature in {
            PerformanceFeatureName.KEEP_WARM_MODEL_RESIDENCY,
            PerformanceFeatureName.AGGRESSIVE_UNLOAD_MODE,
            PerformanceFeatureName.BALANCED_RESIDENCY_MODE,
        }:
            return ["Choose a runnable local runtime on the current host before relying on runtime residency policies."]
        if feature in {
            PerformanceFeatureName.REQUEST_SCHEDULING_AND_BACKPRESSURE,
            PerformanceFeatureName.MODEL_LOAD_ADMISSION_CONTROL,
        }:
            return ["Configure the runtime admission limits in settings to enable bounded queueing and explicit backpressure."]
        return []

    def _model_benchmark_summaries(self, records: list[BenchmarkRecord]) -> list[ModelBenchmarkSummary]:
        model_groups: dict[str, list[BenchmarkRecord]] = {}
        for record in records:
            model_groups.setdefault(record.model_id, []).append(record)
        return [
            ModelBenchmarkSummary(
                model_id=model_id,
                run_count=len(model_records),
                average_total_seconds=round(fmean(record.total_seconds for record in model_records), 4),
                fastest_total_seconds=round(min(record.total_seconds for record in model_records), 4),
                last_run_at=max(record.created_at for record in model_records),
                capability_counts={
                    capability: len([record for record in model_records if record.capability == capability])
                    for capability in sorted({record.capability for record in model_records})
                },
            )
            for model_id, model_records in sorted(model_groups.items())
        ]


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _compact_metrics(**values: int | float | str | bool | None) -> dict[str, int | float | str | bool]:
    return {
        key: value
        for key, value in values.items()
        if value is not None
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    normalized_quantile = min(max(quantile, 0.0), 1.0)
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * normalized_quantile))))
    return round(ordered[index], 4)


def _ensure_benchmark_multimodal_assets(benchmarks_dir: Path) -> dict[str, Path]:
    assets_dir = benchmarks_dir / "_multimodal_encoder_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_path = assets_dir / "sample-image.png"
    frame_bundle_dir = assets_dir / "sample-frames"
    audio_path = assets_dir / "sample-audio.wav"
    image_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP8zwACTGCSAQANHQEDgslx/wAAAABJRU5ErkJggg==")
    image_path.write_bytes(image_bytes)
    frame_bundle_dir.mkdir(exist_ok=True)
    frame_one = frame_bundle_dir / "frame-0001.png"
    frame_two = frame_bundle_dir / "frame-0002.png"
    frame_one.write_bytes(image_bytes)
    frame_two.write_bytes(image_bytes)
    audio_path.write_bytes(_benchmark_audio_bytes("LewLM benchmark audio", duration_seconds=2.5))
    return {"image": image_path, "frame_bundle": frame_bundle_dir, "audio": audio_path}


def _benchmark_prompt_variant(prompt: str, *, phase: str, sample_type: str) -> str:
    phase_label = phase.replace("_", " ")
    return f"{prompt}\n\n[benchmark asset={sample_type}; phase={phase_label}]"


def _sum_benchmark_sample_metric(samples: Sequence[BenchmarkScenarioSample], *metric_names: str) -> int:
    total = 0
    for sample in samples:
        for metric_name in metric_names:
            value = sample.metrics.get(metric_name)
            if isinstance(value, bool):
                total += int(value)
            elif isinstance(value, int):
                total += value
    return total


def _benchmark_audio_bytes(seed_text: str, *, duration_seconds: float = 1.0) -> bytes:
    payload = seed_text.encode("utf-8") or b"benchmark"
    target_bytes = max(int(16_000 * max(duration_seconds, 0.0)), len(payload))
    repeated_payload = (payload * ((target_bytes + len(payload) - 1) // len(payload)))[:target_bytes]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(1)
        wav_file.setframerate(16_000)
        wav_file.writeframes(repeated_payload)
    return buffer.getvalue()
