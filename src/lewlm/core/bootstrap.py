"""Service bootstrap helpers for API and CLI entrypoints."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from collections.abc import Mapping
import inspect

from lewlm.config.settings import LewLMSettings, get_settings
from lewlm.conversion.backend import ConversionBackend
from lewlm.conversion.service import ConversionService
from lewlm.core.chat import ChatOrchestrator
from lewlm.core.multimodal import MultimodalOrchestrator
from lewlm.documents.ingest.service import DocumentIngestService
from lewlm.documents.service import DocumentGenerationService
from lewlm.documents.skills.catalog import DocumentSkillCatalogService
from lewlm.documents.skills.service import DocumentTransformService
from lewlm.events.bus import EventBus
from lewlm.history.service import SessionHistoryService
from lewlm.pack_registry import PackRegistry
from lewlm.registry.service import ModelRegistry
from lewlm.routing.service import ModelRouter
from lewlm.security.audit import AuditLogger
from lewlm.security.authorization import ToolAuthorizer
from lewlm.security.persistence import PersistenceEncryptor
from lewlm.runtime.catalog import RuntimeCatalog, build_default_runtime_catalog
from lewlm.runtime.experimental import DistributedClusterService
from lewlm.runtime.request_coalescer import InFlightRequestCoalescer
from lewlm.runtime.response_cache import RuntimeResponseCache
from lewlm.runtime.scheduler import RuntimeRequestScheduler
from lewlm.storage import BlockDiskCache, MetadataStore, MultimodalEncoderCache, MultimodalFeatureCache
from lewlm.telemetry.stats import TelemetryService
from lewlm.telemetry.runtime_metrics import RuntimeMetricsRecorder
from lewlm.tools import ToolCatalogService, ToolExecutionService
from lewlm.utils.logging import configure_logging
from lewlm.core.contracts import RuntimeAffinity, RuntimeContract
from lewlm.prompting import PromptCompiler


@dataclass(slots=True)
class LewLMServices:
    """Application service container."""

    settings: LewLMSettings
    pack_registry: PackRegistry
    audit_logger: AuditLogger
    tool_authorizer: ToolAuthorizer
    event_bus: EventBus
    metadata_store: MetadataStore
    model_registry: ModelRegistry
    runtime_catalog: RuntimeCatalog
    model_router: ModelRouter
    prompt_compiler: PromptCompiler
    skill_catalog_service: DocumentSkillCatalogService
    tool_catalog_service: ToolCatalogService
    runtime_request_scheduler: RuntimeRequestScheduler
    model_load_scheduler: RuntimeRequestScheduler
    runtime_metrics_recorder: RuntimeMetricsRecorder
    block_disk_cache: BlockDiskCache
    multimodal_encoder_cache: MultimodalEncoderCache
    multimodal_feature_cache: MultimodalFeatureCache
    runtime_response_cache: RuntimeResponseCache
    runtime_request_coalescer: InFlightRequestCoalescer[object]
    chat_orchestrator: ChatOrchestrator
    multimodal_orchestrator: MultimodalOrchestrator
    session_history_service: SessionHistoryService
    document_generation_service: DocumentGenerationService
    document_ingest_service: DocumentIngestService
    document_transform_service: DocumentTransformService
    tool_execution_service: ToolExecutionService
    conversion_service: ConversionService
    telemetry_service: TelemetryService
    cluster_service: DistributedClusterService

    async def aclose(self) -> None:
        """Release long-lived resources owned by the service container."""

        await self.runtime_catalog.unload_all_models()
        self.conversion_service.close()

    def close(self) -> None:
        """Release long-lived resources owned by the service container."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
            return
        raise RuntimeError(
            "LewLMServices.close cannot run inside an active asyncio event loop. Use `await LewLMServices.aclose()` instead.",
        )


@dataclass(slots=True)
class _CoreFoundationServices:
    audit_logger: AuditLogger
    tool_authorizer: ToolAuthorizer
    event_bus: EventBus
    metadata_store: MetadataStore
    model_registry: ModelRegistry


@dataclass(slots=True)
class _PerformanceCoreServices:
    block_disk_cache: BlockDiskCache
    multimodal_encoder_cache: MultimodalEncoderCache
    multimodal_feature_cache: MultimodalFeatureCache
    runtime_response_cache: RuntimeResponseCache
    runtime_request_coalescer: InFlightRequestCoalescer[object]
    runtime_request_scheduler: RuntimeRequestScheduler
    model_load_scheduler: RuntimeRequestScheduler
    runtime_metrics_recorder: RuntimeMetricsRecorder


@dataclass(slots=True)
class _ExperimentalServices:
    cluster_service: DistributedClusterService


@dataclass(slots=True)
class _RuntimeCoreServices:
    runtime_catalog: RuntimeCatalog
    model_router: ModelRouter


@dataclass(slots=True)
class _OptionalModuleServices:
    skill_catalog_service: DocumentSkillCatalogService
    tool_catalog_service: ToolCatalogService
    document_generation_service: DocumentGenerationService
    document_ingest_service: DocumentIngestService
    document_transform_service: DocumentTransformService
    tool_execution_service: ToolExecutionService


def _clone_runtime_overrides(
    runtime_overrides: Mapping[RuntimeAffinity, RuntimeContract] | None,
    *,
    settings: LewLMSettings,
    reinstantiate: bool = False,
) -> Mapping[RuntimeAffinity, RuntimeContract] | None:
    """Resolve runtime overrides for a service container.

    The primary container binds the caller's exact runtime instances so the
    public ``LewLM(runtime_overrides=...)`` facade and the test suite can
    observe the runtime they injected. Settings-aware runtimes are re-built so
    they pick up the resolved settings. When ``reinstantiate`` is True (the
    on-demand ``service_factory`` path that spins up independent containers with
    alternate settings, e.g. autotune/benchmark candidates), stateless runtimes
    are also rebuilt so each container gets an isolated instance.
    """

    if runtime_overrides is None:
        return None
    cloned: dict[RuntimeAffinity, RuntimeContract] = {}
    for affinity, runtime in runtime_overrides.items():
        if not reinstantiate:
            # Primary container: use the caller's exact instance so the public
            # LewLM(runtime_overrides=...) facade and the test suite observe the
            # runtime they injected.
            cloned[affinity] = runtime
            continue
        # service_factory path: rebuild an isolated instance per container,
        # injecting the candidate settings when the runtime accepts them.
        runtime_type = type(runtime)
        try:
            accepts_settings = "settings" in inspect.signature(runtime_type).parameters
        except (TypeError, ValueError):
            accepts_settings = False
        try:
            cloned[affinity] = runtime_type(settings=settings) if accepts_settings else runtime_type()
            continue
        except (TypeError, ValueError):
            pass
        cloned[affinity] = runtime
    return cloned


def _build_core_foundation_services(settings: LewLMSettings) -> _CoreFoundationServices:
    """Build LewLM's core foundation layer."""

    encryptor = PersistenceEncryptor(settings) if settings.persistence_encryption_enabled else None
    audit_logger = AuditLogger(settings, encryptor=encryptor)
    tool_authorizer = ToolAuthorizer(settings=settings, audit_logger=audit_logger)
    metadata_store = MetadataStore(settings.database_path, encryptor=encryptor)
    metadata_store.initialize()
    event_bus = EventBus()
    model_registry = ModelRegistry(
        settings=settings,
        metadata_store=metadata_store,
        event_bus=event_bus,
        audit_logger=audit_logger,
    )
    return _CoreFoundationServices(
        audit_logger=audit_logger,
        tool_authorizer=tool_authorizer,
        event_bus=event_bus,
        metadata_store=metadata_store,
        model_registry=model_registry,
    )


def _build_performance_core_services(
    settings: LewLMSettings,
    *,
    metadata_store: MetadataStore,
) -> _PerformanceCoreServices:
    """Build the performance-core serving and cache layer."""

    block_disk_cache = BlockDiskCache(cache_root=settings.cache_dir, metadata_store=metadata_store)
    multimodal_encoder_cache = MultimodalEncoderCache(block_disk_cache=block_disk_cache)
    multimodal_feature_cache = MultimodalFeatureCache(block_disk_cache=block_disk_cache)
    runtime_response_cache = RuntimeResponseCache(metadata_store=metadata_store)
    runtime_request_scheduler = RuntimeRequestScheduler(
        max_concurrent_requests=settings.max_concurrent_runtime_requests,
        queue_limit=settings.runtime_request_queue_limit,
        queue_timeout_seconds=settings.runtime_request_queue_timeout_seconds,
        decode_priority_enabled=settings.decode_priority_scheduling_enabled,
        long_prefill_token_threshold=settings.long_prefill_token_threshold,
        prefill_isolation_enabled=settings.prefill_isolation_enabled,
        prefill_isolation_max_concurrent_requests=settings.prefill_isolation_max_concurrent_requests,
        prefill_isolation_decode_reserve=settings.prefill_isolation_decode_reserve,
    )
    model_load_scheduler = RuntimeRequestScheduler(
        max_concurrent_requests=settings.max_concurrent_model_loads,
        queue_limit=settings.runtime_request_queue_limit,
        queue_timeout_seconds=settings.runtime_request_queue_timeout_seconds,
    )
    return _PerformanceCoreServices(
        block_disk_cache=block_disk_cache,
        multimodal_encoder_cache=multimodal_encoder_cache,
        multimodal_feature_cache=multimodal_feature_cache,
        runtime_response_cache=runtime_response_cache,
        runtime_request_coalescer=InFlightRequestCoalescer(),
        runtime_request_scheduler=runtime_request_scheduler,
        model_load_scheduler=model_load_scheduler,
        runtime_metrics_recorder=RuntimeMetricsRecorder(),
    )


def _build_experimental_services(
    settings: LewLMSettings,
    *,
    metadata_store: MetadataStore,
    event_bus: EventBus,
    audit_logger: AuditLogger,
) -> _ExperimentalServices:
    """Build experimental services that stay outside the default runtime path."""

    return _ExperimentalServices(
        cluster_service=DistributedClusterService(
            settings=settings,
            metadata_store=metadata_store,
            event_bus=event_bus,
            audit_logger=audit_logger,
        ),
    )


def _build_runtime_core_services(
    settings: LewLMSettings,
    *,
    model_registry: ModelRegistry,
    multimodal_encoder_cache: MultimodalEncoderCache,
    cluster_service: DistributedClusterService,
    pack_registry: PackRegistry,
    runtime_overrides: Mapping[RuntimeAffinity, RuntimeContract] | None,
) -> _RuntimeCoreServices:
    """Build the runtime catalog and router that sit at LewLM's core."""

    runtime_catalog = build_default_runtime_catalog(
        settings,
        multimodal_encoder_cache=multimodal_encoder_cache,
        cluster_service=cluster_service,
        pack_registry=pack_registry,
        runtime_overrides=runtime_overrides,
    )
    model_router = ModelRouter(
        model_registry=model_registry,
        runtime_catalog=runtime_catalog,
        settings=settings,
    )
    return _RuntimeCoreServices(
        runtime_catalog=runtime_catalog,
        model_router=model_router,
    )


def _build_optional_module_services(
    settings: LewLMSettings,
    *,
    pack_registry: PackRegistry,
    audit_logger: AuditLogger,
    tool_authorizer: ToolAuthorizer,
    event_bus: EventBus,
    metadata_store: MetadataStore,
) -> _OptionalModuleServices:
    """Build optional document and local-tooling services."""

    documents_enabled = pack_registry.feature_enabled("documents")
    documents_reason = pack_registry.report("documents").reason
    skill_catalog_service = DocumentSkillCatalogService(
        enabled=documents_enabled,
        disabled_reason=documents_reason,
    )
    tool_catalog_service = ToolCatalogService(
        enabled=documents_enabled,
        disabled_reason=documents_reason,
    )
    document_generation_service = DocumentGenerationService(
        event_bus=event_bus,
        enabled=documents_enabled,
        disabled_reason=documents_reason,
    )
    document_ingest_service = DocumentIngestService(
        workspace_root=settings.temp_dir,
        sandbox_enabled=settings.parser_sandbox_enabled,
        sandbox_timeout_seconds=settings.parser_sandbox_timeout_seconds,
        sandbox_clear_environment=settings.parser_sandbox_clear_environment,
        event_bus=event_bus,
        enabled=documents_enabled,
        disabled_reason=documents_reason,
    )
    document_transform_service = DocumentTransformService(
        generation_service=document_generation_service,
        event_bus=event_bus,
        enabled=documents_enabled,
        disabled_reason=documents_reason,
    )
    tool_execution_service = ToolExecutionService(
        settings=settings,
        tool_catalog=tool_catalog_service,
        document_generation_service=document_generation_service,
        document_ingest_service=document_ingest_service,
        document_transform_service=document_transform_service,
        tool_authorizer=tool_authorizer,
        audit_logger=audit_logger,
        event_bus=event_bus,
        metadata_store=metadata_store,
    )
    return _OptionalModuleServices(
        skill_catalog_service=skill_catalog_service,
        tool_catalog_service=tool_catalog_service,
        document_generation_service=document_generation_service,
        document_ingest_service=document_ingest_service,
        document_transform_service=document_transform_service,
        tool_execution_service=tool_execution_service,
    )


def bootstrap_services(
    settings: LewLMSettings | None = None,
    *,
    runtime_overrides: Mapping[RuntimeAffinity, RuntimeContract] | None = None,
    conversion_backend: ConversionBackend | None = None,
) -> LewLMServices:
    """Create and initialize the service container."""

    resolved_settings = settings or get_settings()
    resolved_settings.prepare_directories()
    configure_logging(resolved_settings)
    pack_registry = PackRegistry.from_settings(resolved_settings)
    scoped_runtime_overrides = _clone_runtime_overrides(runtime_overrides, settings=resolved_settings)

    core_foundation = _build_core_foundation_services(resolved_settings)
    performance_core = _build_performance_core_services(
        resolved_settings,
        metadata_store=core_foundation.metadata_store,
    )
    experimental_services = _build_experimental_services(
        resolved_settings,
        metadata_store=core_foundation.metadata_store,
        event_bus=core_foundation.event_bus,
        audit_logger=core_foundation.audit_logger,
    )
    runtime_core = _build_runtime_core_services(
        resolved_settings,
        model_registry=core_foundation.model_registry,
        multimodal_encoder_cache=performance_core.multimodal_encoder_cache,
        cluster_service=experimental_services.cluster_service,
        pack_registry=pack_registry,
        runtime_overrides=scoped_runtime_overrides,
    )
    optional_modules = _build_optional_module_services(
        resolved_settings,
        pack_registry=pack_registry,
        audit_logger=core_foundation.audit_logger,
        tool_authorizer=core_foundation.tool_authorizer,
        event_bus=core_foundation.event_bus,
        metadata_store=core_foundation.metadata_store,
    )
    prompt_compiler = PromptCompiler(
        resolved_settings,
        tool_catalog=optional_modules.tool_catalog_service,
    )
    service_factory = lambda candidate_settings: bootstrap_services(  # noqa: E731 - local factory keeps cloned settings wiring close to use sites
        candidate_settings,
        runtime_overrides=_clone_runtime_overrides(
            scoped_runtime_overrides,
            settings=candidate_settings,
            reinstantiate=True,
        ),
        conversion_backend=conversion_backend,
    )
    chat_orchestrator = ChatOrchestrator(
        model_router=runtime_core.model_router,
        event_bus=core_foundation.event_bus,
        prompt_compiler=prompt_compiler,
        audit_logger=core_foundation.audit_logger,
        settings=resolved_settings,
        runtime_catalog=runtime_core.runtime_catalog,
        runtime_request_scheduler=performance_core.runtime_request_scheduler,
        model_load_scheduler=performance_core.model_load_scheduler,
        runtime_metrics_recorder=performance_core.runtime_metrics_recorder,
        metadata_store=core_foundation.metadata_store,
        service_factory=service_factory,
    )
    multimodal_orchestrator = MultimodalOrchestrator(
        model_router=runtime_core.model_router,
        event_bus=core_foundation.event_bus,
        runtime_request_scheduler=performance_core.runtime_request_scheduler,
        model_load_scheduler=performance_core.model_load_scheduler,
        runtime_metrics_recorder=performance_core.runtime_metrics_recorder,
        runtime_response_cache=performance_core.runtime_response_cache,
        runtime_request_coalescer=performance_core.runtime_request_coalescer,
    )
    session_history_service = SessionHistoryService(
        metadata_store=core_foundation.metadata_store,
        settings=resolved_settings,
    )
    conversion_service = ConversionService(
        settings=resolved_settings,
        model_registry=core_foundation.model_registry,
        metadata_store=core_foundation.metadata_store,
        event_bus=core_foundation.event_bus,
        backend=conversion_backend,
        audit_logger=core_foundation.audit_logger,
    )
    telemetry_service = TelemetryService(
        settings=resolved_settings,
        metadata_store=core_foundation.metadata_store,
        event_bus=core_foundation.event_bus,
        runtime_catalog=runtime_core.runtime_catalog,
        model_router=runtime_core.model_router,
        conversion_service=conversion_service,
        runtime_request_scheduler=performance_core.runtime_request_scheduler,
        model_load_scheduler=performance_core.model_load_scheduler,
        runtime_metrics_recorder=performance_core.runtime_metrics_recorder,
        block_disk_cache=performance_core.block_disk_cache,
        runtime_response_cache=performance_core.runtime_response_cache,
        chat_orchestrator=chat_orchestrator,
        multimodal_orchestrator=multimodal_orchestrator,
        cluster_service=experimental_services.cluster_service,
        service_factory=service_factory,
    )
    return LewLMServices(
        settings=resolved_settings,
        pack_registry=pack_registry,
        audit_logger=core_foundation.audit_logger,
        tool_authorizer=core_foundation.tool_authorizer,
        event_bus=core_foundation.event_bus,
        metadata_store=core_foundation.metadata_store,
        model_registry=core_foundation.model_registry,
        runtime_catalog=runtime_core.runtime_catalog,
        model_router=runtime_core.model_router,
        prompt_compiler=prompt_compiler,
        skill_catalog_service=optional_modules.skill_catalog_service,
        tool_catalog_service=optional_modules.tool_catalog_service,
        runtime_request_scheduler=performance_core.runtime_request_scheduler,
        model_load_scheduler=performance_core.model_load_scheduler,
        runtime_metrics_recorder=performance_core.runtime_metrics_recorder,
        block_disk_cache=performance_core.block_disk_cache,
        multimodal_encoder_cache=performance_core.multimodal_encoder_cache,
        multimodal_feature_cache=performance_core.multimodal_feature_cache,
        runtime_response_cache=performance_core.runtime_response_cache,
        runtime_request_coalescer=performance_core.runtime_request_coalescer,
        chat_orchestrator=chat_orchestrator,
        multimodal_orchestrator=multimodal_orchestrator,
        session_history_service=session_history_service,
        document_generation_service=optional_modules.document_generation_service,
        document_ingest_service=optional_modules.document_ingest_service,
        document_transform_service=optional_modules.document_transform_service,
        tool_execution_service=optional_modules.tool_execution_service,
        conversion_service=conversion_service,
        telemetry_service=telemetry_service,
        cluster_service=experimental_services.cluster_service,
    )
