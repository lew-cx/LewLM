"""Chat orchestration across routing, runtimes, events, and prompt compilation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from lewlm.config.settings import LewLMSettings
from lewlm.core.chat_metrics import (
    _chat_coerce_float,
    _chat_coerce_int,
    _chat_measurements,
    _citation_context_metadata,
    _continuous_batch_measurements,
    _request_cache_measurements,
    _request_scheduling_measurements,
    _structured_output_result,
)
from lewlm.core.chat_streams import (
    _content_stream,
    _empty_item_stream,
    _empty_stream,
    _queued_item_stream,
    _stream_items_with_structured_output,
)
from lewlm.core.citations import (
    CitationContextPackage,
    CitationStreamProcessor,
    GeneratedCitationReference,
    render_citation_context_message,
    resolve_generated_citations,
)
from lewlm.core.contracts import (
    CapabilityName,
    GenerateMessage,
    GenerateRequest,
    GenerateResponse,
    ModelManifest,
    ReasoningOutput,
    ReasoningVisibility,
    RoutingDecision,
    RuntimeContract,
    utc_now,
)
from lewlm.core.execution_metadata import (
    ExecutionMetadata,
    ExecutionServingMetadata,
    build_routed_execution_metadata,
    milliseconds_from_seconds,
)
from lewlm.core.serving_core import (
    ServingCore,
    ServingPhase,
    ServingQueueType,
    ServingRuntimeAdapter,
    continuous_batching_ownership,
    describe_serving_runtime_adapter,
)
from lewlm.core.reasoning import (
    ReasoningStreamProcessor,
    apply_reasoning_visibility,
    reasoning_available,
    reasoning_exposed,
)
from lewlm.core.speculation import (
    chat_speculation_workload_class,
    parse_speculation_benchmark_preference,
    plan_chat_speculation,
    speculation_benchmark_preference_key,
    speculation_measurements,
)
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.prompting import PromptCompilationRequest, PromptCompilationTrace, PromptCompiler, PromptOverrideRecord
from lewlm.routing.service import ModelRouter
from lewlm.runtime.catalog import RuntimeCatalog
from lewlm.runtime.scheduler import (
    FrontierBatchMetrics,
    FrontierBatchScheduler,
    RuntimeRequestAdmission,
    RuntimeRequestScheduler,
)
from lewlm.runtime.experimental import build_frontier_serving_plan, frontier_architecture_measurements
from lewlm.security.audit import AuditLogger
from lewlm.serving_profiles import (
    ServingProfileApplication,
    resolve_serving_profile_application,
    serving_profile_workload_class,
    serving_profile_requires_materialization,
)
from lewlm.storage import MetadataStore
from lewlm.structured_output import (
    StructuredOutputResult,
    build_structured_output_request,
)
from lewlm.telemetry.runtime_metrics import RuntimeMetricsRecorder


@dataclass(slots=True)
class ChatExecution:
    """Completed chat invocation and its routing metadata."""

    request_id: str
    created_at: int
    response: GenerateResponse
    routing: RoutingDecision
    prompt_trace: PromptCompilationTrace
    request_metadata: dict[str, object]
    metadata: ExecutionMetadata
    structured_output: StructuredOutputResult | None = None
    serving_profile: ServingProfileApplication | None = None


@dataclass(slots=True)
class ChatStreamSession:
    """Streaming chat invocation details and token iterator."""

    request_id: str
    created_at: int
    model_id: str
    routing: RoutingDecision
    prompt_trace: PromptCompilationTrace
    reasoning_visibility: ReasoningVisibility
    request: GenerateRequest
    stream: AsyncIterator[str]
    stream_items: AsyncIterator["ChatStreamDelta"] | None = None
    reasoning: ReasoningOutput | None = None
    citations: list[GeneratedCitationReference] = field(default_factory=list)
    request_metadata: dict[str, object] | None = None
    metadata: ExecutionMetadata | None = None
    structured_output: StructuredOutputResult | None = None
    serving_profile: ServingProfileApplication | None = None


@dataclass(slots=True)
class ChatStreamDelta:
    """Single streamed content or reasoning delta."""

    content: str | None = None
    reasoning: str | None = None


@dataclass(slots=True)
class _ChatInvocationContext:
    request_id: str
    created_at: int
    requested_model_id: str | None
    manifest: ModelManifest
    runtime: RuntimeContract
    routing: RoutingDecision
    prompt_trace: PromptCompilationTrace
    prompt_trace_requested: bool
    reasoning_visibility: ReasoningVisibility
    citation_context: CitationContextPackage | None
    request: GenerateRequest
    companion_manifests: tuple[ModelManifest, ...]
    prompt_token_estimate: int
    prefill_chunk_count_estimate: int
    prefill_heavy: bool
    decode_priority_requested: bool
    prefill_isolation_requested: bool
    serving_adapter: ServingRuntimeAdapter


_STREAM_END = object()


class ChatOrchestrator:
    """Execute chat requests while publishing lifecycle events."""

    def __init__(
        self,
        *,
        model_router: ModelRouter,
        event_bus: EventBus,
        prompt_compiler: PromptCompiler,
        audit_logger: AuditLogger,
        settings: LewLMSettings,
        runtime_catalog: RuntimeCatalog,
        runtime_request_scheduler: RuntimeRequestScheduler,
        model_load_scheduler: RuntimeRequestScheduler,
        runtime_metrics_recorder: RuntimeMetricsRecorder,
        metadata_store: MetadataStore | None = None,
        service_factory: Any | None = None,
    ) -> None:
        self.model_router = model_router
        self.event_bus = event_bus
        self.prompt_compiler = prompt_compiler
        self.audit_logger = audit_logger
        self.settings = settings
        self.runtime_catalog = runtime_catalog
        self.runtime_request_scheduler = runtime_request_scheduler
        self.model_load_scheduler = model_load_scheduler
        self.runtime_metrics_recorder = runtime_metrics_recorder
        self.metadata_store = metadata_store
        self.service_factory = service_factory
        self.serving_core = ServingCore()
        self._complete_batch_scheduler: FrontierBatchScheduler[_ChatInvocationContext, ChatExecution] = (
            FrontierBatchScheduler(
                runtime_request_scheduler=runtime_request_scheduler,
                batch_window_milliseconds=settings.continuous_batch_window_milliseconds,
                max_batch_size=settings.continuous_batch_max_batch_size,
            )
        )
        self._stream_batch_scheduler: FrontierBatchScheduler[_ChatInvocationContext, ChatStreamSession] = (
            FrontierBatchScheduler(
                runtime_request_scheduler=runtime_request_scheduler,
                batch_window_milliseconds=settings.continuous_batch_window_milliseconds,
                max_batch_size=settings.continuous_batch_max_batch_size,
            )
        )

    def _preferred_speculation_mode(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        messages: Sequence[GenerateMessage],
        max_tokens: int,
    ):
        if self.metadata_store is None:
            return None
        workload_class = chat_speculation_workload_class(messages=messages, max_tokens=max_tokens)
        preference = parse_speculation_benchmark_preference(
            self.metadata_store.get_value(
                speculation_benchmark_preference_key(
                    model_id=manifest.model_id,
                    runtime_name=runtime.name,
                    workload_class=workload_class,
                ),
            ),
        )
        if preference is None:
            preference = parse_speculation_benchmark_preference(
                self.metadata_store.get_value(
                    speculation_benchmark_preference_key(model_id=manifest.model_id, runtime_name=runtime.name),
                ),
            )
        return None if preference is None else preference.selected_mode

    @staticmethod
    def _speculation_request_metadata(planned_speculation) -> dict[str, object]:
        if planned_speculation is None:
            return {}
        return {
            "speculation_selection_source": (
                "benchmark_preference" if planned_speculation.benchmark_preferred else "heuristic"
            ),
            "speculation_benchmark_preferred": planned_speculation.benchmark_preferred,
            "speculation_selection_reason": planned_speculation.selection_reason,
        }

    async def complete(
        self,
        *,
        model_id: str | None,
        messages: list[GenerateMessage],
        citation_context: CitationContextPackage | None = None,
        max_tokens: int,
        temperature: float,
        reasoning_visibility: ReasoningVisibility = ReasoningVisibility.HIDDEN,
        prompt_request: PromptCompilationRequest | None = None,
        allowed_prompt_file_roots: Sequence[Path | str] | None = None,
        apply_serving_profile: bool = True,
        serving_profile: ServingProfileApplication | None = None,
    ) -> ChatExecution:
        routed_target: tuple[ModelManifest, RuntimeContract, RoutingDecision] | None = None
        structured_output_requested = prompt_request is not None and prompt_request.requests_structured_output()
        if self.metadata_store is not None and serving_profile is None:
            routed_target = self.model_router.route_chat(
                model_id,
                messages=messages,
                max_tokens=max_tokens,
                structured_output_requested=structured_output_requested,
            )
        resolved_serving_profile = serving_profile or self._resolve_serving_profile(
            model_id=model_id,
            messages=messages,
            max_tokens=max_tokens,
            capability=CapabilityName.CHAT,
            apply_serving_profile=apply_serving_profile,
            routed_target=routed_target,
        )
        if (
            apply_serving_profile
            and serving_profile_requires_materialization(profile=resolved_serving_profile, settings=self.settings)
            and self.service_factory is not None
        ):
            return await self._delegate_complete_with_serving_profile(
                model_id=model_id,
                messages=messages,
                citation_context=citation_context,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_visibility=reasoning_visibility,
                prompt_request=prompt_request,
                allowed_prompt_file_roots=allowed_prompt_file_roots,
                serving_profile=resolved_serving_profile,
            )
        context = self._prepare_context(
            model_id=model_id,
            messages=messages,
            citation_context=citation_context,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_visibility=reasoning_visibility,
            prompt_request=prompt_request,
            allowed_prompt_file_roots=allowed_prompt_file_roots,
            serving_profile=resolved_serving_profile,
            capability=CapabilityName.CHAT,
            routed_target=routed_target,
        )
        if self._supports_frontier_batching(context=context, capability=CapabilityName.CHAT):
            batch_result = await self._complete_batch_scheduler.enqueue(
                key=self._continuous_batch_key(context=context, capability=CapabilityName.CHAT),
                payload=context,
                execute_batch=self._execute_complete_batch,
            )
            return batch_result.value
        return await self._execute_complete_context(context)

    async def stream(
        self,
        *,
        model_id: str | None,
        messages: list[GenerateMessage],
        citation_context: CitationContextPackage | None = None,
        max_tokens: int,
        temperature: float,
        reasoning_visibility: ReasoningVisibility = ReasoningVisibility.HIDDEN,
        prompt_request: PromptCompilationRequest | None = None,
        allowed_prompt_file_roots: Sequence[Path | str] | None = None,
        apply_serving_profile: bool = True,
        serving_profile: ServingProfileApplication | None = None,
    ) -> ChatStreamSession:
        routed_target: tuple[ModelManifest, RuntimeContract, RoutingDecision] | None = None
        structured_output_requested = prompt_request is not None and prompt_request.requests_structured_output()
        if self.metadata_store is not None and serving_profile is None:
            routed_target = self.model_router.route_chat(
                model_id,
                messages=messages,
                max_tokens=max_tokens,
                structured_output_requested=structured_output_requested,
            )
        resolved_serving_profile = serving_profile or self._resolve_serving_profile(
            model_id=model_id,
            messages=messages,
            max_tokens=max_tokens,
            capability=CapabilityName.STREAMING,
            apply_serving_profile=apply_serving_profile,
            routed_target=routed_target,
        )
        if (
            apply_serving_profile
            and serving_profile_requires_materialization(profile=resolved_serving_profile, settings=self.settings)
            and self.service_factory is not None
        ):
            return await self._delegate_stream_with_serving_profile(
                model_id=model_id,
                messages=messages,
                citation_context=citation_context,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_visibility=reasoning_visibility,
                prompt_request=prompt_request,
                allowed_prompt_file_roots=allowed_prompt_file_roots,
                serving_profile=resolved_serving_profile,
            )
        context = self._prepare_context(
            model_id=model_id,
            messages=messages,
            citation_context=citation_context,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_visibility=reasoning_visibility,
            prompt_request=prompt_request,
            allowed_prompt_file_roots=allowed_prompt_file_roots,
            serving_profile=resolved_serving_profile,
            capability=CapabilityName.STREAMING,
            routed_target=routed_target,
        )
        if self._supports_frontier_batching(context=context, capability=CapabilityName.STREAMING):
            batch_result = await self._stream_batch_scheduler.enqueue(
                key=self._continuous_batch_key(context=context, capability=CapabilityName.STREAMING),
                payload=context,
                execute_batch=self._execute_stream_batch,
            )
            return batch_result.value
        return await self._execute_stream_context(context)

    def _prepare_context(
        self,
        *,
        model_id: str | None,
        messages: list[GenerateMessage],
        citation_context: CitationContextPackage | None,
        max_tokens: int,
        temperature: float,
        reasoning_visibility: ReasoningVisibility,
        prompt_request: PromptCompilationRequest | None,
        allowed_prompt_file_roots: Sequence[Path | str] | None,
        serving_profile: ServingProfileApplication | None,
        capability: CapabilityName,
        routed_target: tuple[ModelManifest, RuntimeContract, RoutingDecision] | None = None,
    ) -> _ChatInvocationContext:
        request_id = str(uuid4())
        created_at = int(utc_now().timestamp())
        structured_output_requested = prompt_request is not None and prompt_request.requests_structured_output()
        if routed_target is None:
            manifest, runtime, routing = self.model_router.route_chat(
                model_id,
                messages=messages,
                max_tokens=max_tokens,
                structured_output_requested=structured_output_requested,
            )
        else:
            manifest, runtime, routing = routed_target
        compilation = self.prompt_compiler.compile(
            messages=messages,
            request=prompt_request,
            requested_model_id=model_id,
            resolved_model_id=manifest.model_id,
            model_manifest=manifest,
            allowed_file_roots=allowed_prompt_file_roots,
        )
        compiled_messages, prompt_trace = self._apply_citation_context(
            compilation_messages=compilation.messages,
            prompt_trace=compilation.trace,
            citation_context=citation_context,
        )
        self._audit_prompt_overrides(
            request_id=request_id,
            requested_model_id=model_id,
            prompt_request=prompt_request,
            prompt_trace=prompt_trace,
        )
        preferred_speculation_mode = self._preferred_speculation_mode(
            manifest=manifest,
            runtime=runtime,
            messages=compiled_messages,
            max_tokens=max_tokens,
        )
        planned_speculation = plan_chat_speculation(
            model_registry=self.model_router.model_registry,
            settings=self.model_router.settings,
            primary_manifest=manifest,
            runtime=runtime,
            messages=compiled_messages,
            max_tokens=max_tokens,
            preferred_mode=preferred_speculation_mode,
        )
        structured_output_request = build_structured_output_request(
            format=prompt_trace.output_contract.format,
            schema=prompt_trace.output_contract.schema_payload,
            grammar=prompt_trace.output_contract.grammar,
            syntax=prompt_trace.output_contract.syntax,
            name=prompt_trace.output_contract.name,
            strict=prompt_trace.output_contract.strict,
        )
        structured_output_runtime = runtime.structured_output_runtime_status(structured_output_request)
        request = GenerateRequest(
            model_id=manifest.model_id,
            messages=compiled_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_visibility=reasoning_visibility,
            speculation=planned_speculation.request if planned_speculation is not None else None,
            structured_output=structured_output_request,
            request_id=request_id,
            metadata={
                **(
                    {"prompt_trace": prompt_trace.model_dump(mode="json", by_alias=True)}
                    if prompt_request is not None and prompt_request.include_trace
                    else {}
                ),
                **(
                    {"citation_context": _citation_context_metadata(citation_context)}
                    if citation_context is not None and citation_context.has_entries()
                    else {}
                ),
                **(
                    {"serving_profile": serving_profile.model_dump(mode="json")}
                    if serving_profile is not None
                    else {}
                ),
                **(
                    {"frontier_architecture": frontier_plan}
                    if (frontier_plan := build_frontier_serving_plan(manifest=manifest, settings=self.settings)) is not None
                    else {}
                ),
                "routing": routing.model_dump(mode="json"),
                **self._speculation_request_metadata(planned_speculation),
                **(
                    {
                        "structured_output_runtime": structured_output_runtime.model_dump(mode="json"),
                    }
                    if structured_output_runtime is not None
                    else {}
                ),
            },
        )
        total_prompt_tokens = self._estimate_prompt_tokens(
            manifest=manifest,
            runtime=runtime,
            messages=compiled_messages,
        )
        prefix_cache_preview = self._prefix_cache_scheduling_preview(
            runtime=runtime,
            model_id=manifest.model_id,
            messages=compiled_messages,
        )
        prompt_token_estimate = total_prompt_tokens
        if isinstance(prefix_cache_preview.get("effective_prefill_tokens"), int) and not isinstance(
            prefix_cache_preview.get("effective_prefill_tokens"),
            bool,
        ):
            prompt_token_estimate = max(int(prefix_cache_preview["effective_prefill_tokens"]), 0)
        cached_prefix_tokens = (
            max(int(prefix_cache_preview["cached_prefix_tokens"]), 0)
            if isinstance(prefix_cache_preview.get("cached_prefix_tokens"), int)
            and not isinstance(prefix_cache_preview.get("cached_prefix_tokens"), bool)
            else 0
        )
        prefill_chunk_count_estimate = max(
            1,
            math.ceil(prompt_token_estimate / self.settings.prefill_token_batch_size),
        )
        prefill_heavy = (
            prompt_token_estimate >= self.settings.long_prefill_token_threshold
            or prefill_chunk_count_estimate > 1
        )
        decode_priority_requested = self.settings.decode_priority_scheduling_enabled and not prefill_heavy
        prefill_isolation_requested = (
            self.settings.prefill_isolation_enabled
            and prefill_heavy
            and runtime.supports_prefill_isolation(capability)
        )
        request.metadata["scheduling"] = {
            "prompt_token_estimate": prompt_token_estimate,
            "total_prompt_tokens": total_prompt_tokens,
            "cached_prefix_tokens": cached_prefix_tokens,
            "prefill_chunk_count_estimate": prefill_chunk_count_estimate,
            "long_prefill_threshold_tokens": self.settings.long_prefill_token_threshold,
            "prefill_heavy": prefill_heavy,
            "queue_lane": "prefill" if prefill_heavy else "decode",
            "decode_priority_requested": decode_priority_requested,
            "decode_priority_active": False,
            "prefix_cache_candidate": cached_prefix_tokens > 0,
            "prefix_cache_lookup_source": prefix_cache_preview.get("lookup_source", "miss"),
            "cached_pages": prefix_cache_preview.get("cached_pages", 0),
            "prefill_isolation_requested": prefill_isolation_requested,
            "prefill_isolation_active": False,
            "chunked_prefill_requested": runtime.supports_chunked_prefill(capability) and prefill_chunk_count_estimate > 1,
            "chunked_prefill_active": False,
            "chunk_count": prefill_chunk_count_estimate,
        }
        serving_adapter = describe_serving_runtime_adapter(runtime=runtime, capability=capability)
        self.serving_core.register_sequence(
            request_id=request_id,
            requested_model_id=model_id,
            model_id=manifest.model_id,
            runtime_name=runtime.name,
            capability=capability,
            runtime_adapter=serving_adapter,
            streaming=capability == CapabilityName.STREAMING,
            queue_lane="prefill" if prefill_heavy else "decode",
            prefill_heavy=prefill_heavy,
            decode_priority_requested=decode_priority_requested,
            prefill_isolation_requested=prefill_isolation_requested,
            prompt_token_estimate=prompt_token_estimate,
            chunk_count=prefill_chunk_count_estimate,
        )
        self._sync_serving_request_metadata(request)
        return _ChatInvocationContext(
            request_id=request_id,
            created_at=created_at,
            requested_model_id=model_id,
            manifest=manifest,
            runtime=runtime,
            routing=routing,
            prompt_trace=prompt_trace,
            prompt_trace_requested=bool(prompt_request and prompt_request.include_trace),
            reasoning_visibility=reasoning_visibility,
            citation_context=citation_context if citation_context and citation_context.has_entries() else None,
            request=request,
            companion_manifests=planned_speculation.companion_manifests if planned_speculation is not None else (),
            prompt_token_estimate=prompt_token_estimate,
            prefill_chunk_count_estimate=prefill_chunk_count_estimate,
            prefill_heavy=prefill_heavy,
            decode_priority_requested=decode_priority_requested,
            prefill_isolation_requested=prefill_isolation_requested,
            serving_adapter=serving_adapter,
        )

    def _resolve_serving_profile(
        self,
        *,
        model_id: str | None,
        messages: list[GenerateMessage],
        max_tokens: int,
        capability: CapabilityName,
        apply_serving_profile: bool,
        routed_target: tuple[ModelManifest, RuntimeContract, RoutingDecision] | None = None,
    ) -> ServingProfileApplication | None:
        if self.metadata_store is None:
            return None
        if routed_target is None:
            manifest, runtime, _ = self.model_router.route_chat(
                model_id,
                messages=messages,
                max_tokens=max_tokens,
            )
        else:
            manifest, runtime, _ = routed_target
        workload_class = serving_profile_workload_class(messages=messages, manifest=manifest)
        return resolve_serving_profile_application(
            settings=self.settings,
            metadata_store=self.metadata_store,
            host_platform=self.runtime_catalog.host_platform_snapshot().model_dump(mode="json"),
            runtime=runtime,
            model_id=manifest.model_id,
            request_capability=capability,
            apply_serving_profile=apply_serving_profile,
            workload_class=workload_class,
        )

    async def _delegate_complete_with_serving_profile(
        self,
        *,
        model_id: str | None,
        messages: list[GenerateMessage],
        citation_context: CitationContextPackage | None,
        max_tokens: int,
        temperature: float,
        reasoning_visibility: ReasoningVisibility,
        prompt_request: PromptCompilationRequest | None,
        allowed_prompt_file_roots: Sequence[Path | str] | None,
        serving_profile: ServingProfileApplication,
    ) -> ChatExecution:
        candidate_services = self.service_factory(self.settings.with_updates(**serving_profile.accepted_settings))
        bridge_task = self._bridge_candidate_events(candidate_services)
        try:
            return await candidate_services.chat_orchestrator.complete(
                model_id=model_id,
                messages=messages,
                citation_context=citation_context,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_visibility=reasoning_visibility,
                prompt_request=prompt_request,
                allowed_prompt_file_roots=allowed_prompt_file_roots,
                apply_serving_profile=False,
                serving_profile=serving_profile,
            )
        finally:
            await self._close_candidate_services(candidate_services=candidate_services, bridge_task=bridge_task)

    async def _delegate_stream_with_serving_profile(
        self,
        *,
        model_id: str | None,
        messages: list[GenerateMessage],
        citation_context: CitationContextPackage | None,
        max_tokens: int,
        temperature: float,
        reasoning_visibility: ReasoningVisibility,
        prompt_request: PromptCompilationRequest | None,
        allowed_prompt_file_roots: Sequence[Path | str] | None,
        serving_profile: ServingProfileApplication,
    ) -> ChatStreamSession:
        candidate_services = self.service_factory(self.settings.with_updates(**serving_profile.accepted_settings))
        bridge_task = self._bridge_candidate_events(candidate_services)
        try:
            session = await candidate_services.chat_orchestrator.stream(
                model_id=model_id,
                messages=messages,
                citation_context=citation_context,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_visibility=reasoning_visibility,
                prompt_request=prompt_request,
                allowed_prompt_file_roots=allowed_prompt_file_roots,
                apply_serving_profile=False,
                serving_profile=serving_profile,
            )
        except Exception:
            await self._close_candidate_services(candidate_services=candidate_services, bridge_task=bridge_task)
            raise
        return self._wrap_stream_session(
            session=session,
            candidate_services=candidate_services,
            bridge_task=bridge_task,
        )

    def _wrap_stream_session(
        self,
        *,
        session: ChatStreamSession,
        candidate_services,
        bridge_task: asyncio.Task[None] | None,
    ) -> ChatStreamSession:
        cleaned_up = False
        cleanup_lock = asyncio.Lock()

        async def cleanup() -> None:
            nonlocal cleaned_up
            async with cleanup_lock:
                if cleaned_up:
                    return
                cleaned_up = True
                await self._close_candidate_services(candidate_services=candidate_services, bridge_task=bridge_task)

        async def wrap_content_stream(source: AsyncIterator[str]) -> AsyncIterator[str]:
            try:
                async for chunk in source:
                    yield chunk
            finally:
                await cleanup()

        async def wrap_item_stream(source: AsyncIterator[ChatStreamDelta]) -> AsyncIterator[ChatStreamDelta]:
            try:
                async for item in source:
                    yield item
            finally:
                await cleanup()

        session.stream = wrap_content_stream(session.stream)
        if session.stream_items is not None:
            session.stream_items = wrap_item_stream(session.stream_items)
        return session

    def _bridge_candidate_events(self, candidate_services) -> asyncio.Task[None] | None:
        if not hasattr(candidate_services, "event_bus") or candidate_services.event_bus is self.event_bus:
            return None
        subscription = candidate_services.event_bus.subscribe()

        async def forward_events() -> None:
            try:
                while True:
                    event = await subscription.get()
                    await self.event_bus.publish(event)
            finally:
                subscription.close()

        return asyncio.create_task(forward_events())

    def _apply_citation_context(
        self,
        *,
        compilation_messages: list[GenerateMessage],
        prompt_trace: PromptCompilationTrace,
        citation_context: CitationContextPackage | None,
    ) -> tuple[list[GenerateMessage], PromptCompilationTrace]:
        if citation_context is None or not citation_context.has_entries():
            return compilation_messages, prompt_trace

        rendered_message = render_citation_context_message(citation_context)
        if rendered_message is None:
            return compilation_messages, prompt_trace

        insertion_index = 0
        while insertion_index < len(compilation_messages) and compilation_messages[insertion_index].role == "system":
            insertion_index += 1
        compiled_messages = [
            *compilation_messages[:insertion_index],
            GenerateMessage(role="system", content=rendered_message),
            *compilation_messages[insertion_index:],
        ]
        overrides = list(prompt_trace.overrides)
        overrides.append(
            PromptOverrideRecord(
                source="citation_context",
                scope="grounding",
                summary=(
                    "Applied citation context "
                    f"with {len(citation_context.sources)} source(s) and {len(citation_context.chunks)} chunk(s)."
                ),
            ),
        )
        serialized_model_prompt = prompt_trace.serialized_model_prompt
        if serialized_model_prompt is not None:
            serialized_model_prompt = self.prompt_compiler.prompt_template_catalog.serialize_messages(
                compiled_messages,
                prompt_trace.model_prompt_template,
            )
        return compiled_messages, prompt_trace.model_copy(
            update={
                "serialized_model_prompt": serialized_model_prompt,
                "message_count": len(compiled_messages),
                "message_roles": [message.role for message in compiled_messages],
                "overrides": overrides,
            },
        )

    async def _close_candidate_services(self, *, candidate_services, bridge_task: asyncio.Task[None] | None) -> None:
        if bridge_task is not None:
            await asyncio.sleep(0)
            bridge_task.cancel()
            with suppress(asyncio.CancelledError):
                await bridge_task
        await candidate_services.aclose()

    def _supports_frontier_batching(
        self,
        *,
        context: _ChatInvocationContext,
        capability: CapabilityName,
    ) -> bool:
        if context is None:
            return False
        return (
            self.settings.continuous_batch_max_batch_size > 1
            and context.request.speculation is None
            and continuous_batching_ownership(runtime=context.runtime, capability=capability) != "lewlm_owned"
            and context.runtime.supports_continuous_batching(capability)
        )

    def _continuous_batch_key(
        self,
        *,
        context: _ChatInvocationContext,
        capability: CapabilityName,
    ) -> str:
        companion_key = ",".join(sorted(manifest.model_id for manifest in context.companion_manifests))
        return "|".join(
            (
                context.runtime.name,
                context.manifest.model_id,
                capability.value,
                str(context.request.max_tokens),
                f"{context.request.temperature:.4f}",
                companion_key,
            ),
        )

    def _estimate_prompt_tokens(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        messages: Sequence[GenerateMessage],
    ) -> int:
        prompt = "\n".join(f"{message.role}: {message.content}" for message in messages)
        if runtime.is_model_loaded(manifest.model_id):
            try:
                return max(1, len(runtime.tokenize(prompt)))
            except Exception:
                pass
        word_estimate = sum(max(1, len(message.content.split())) for message in messages)
        attachment_estimate = sum(len(message.attachments) * 32 for message in messages)
        byte_estimate = max(1, len(prompt.encode("utf-8")) // 4)
        return max(1, word_estimate + attachment_estimate, byte_estimate)

    @staticmethod
    def _prefix_cache_scheduling_preview(
        *,
        runtime: RuntimeContract,
        model_id: str,
        messages: Sequence[GenerateMessage],
    ) -> dict[str, object]:
        preview_method = getattr(runtime, "prefix_cache_admission_preview", None)
        if not callable(preview_method):
            return {}
        preview = preview_method(model_id=model_id, messages=messages)
        return preview if isinstance(preview, dict) else {}

    def _apply_scheduler_admission(
        self,
        *,
        request: GenerateRequest,
        admission: RuntimeRequestAdmission,
    ) -> dict[str, object]:
        scheduling = request.metadata.get("scheduling")
        if not isinstance(scheduling, dict):
            return {}
        scheduling["queue_lane"] = admission.scheduling_lane
        scheduling["decode_priority_active"] = admission.decode_priority_applied
        scheduling["prefill_isolation_active"] = admission.prefill_isolated
        scheduling["scheduler_wait_milliseconds"] = int(round(max(admission.wait_seconds, 0.0) * 1000))
        return {
            "queue_lane": scheduling.get("queue_lane"),
            "decode_priority_requested": scheduling.get("decode_priority_requested"),
            "decode_priority_active": scheduling.get("decode_priority_active"),
            "prefill_heavy": scheduling.get("prefill_heavy"),
            "total_prompt_tokens": scheduling.get("total_prompt_tokens"),
            "cached_prefix_tokens": scheduling.get("cached_prefix_tokens"),
            "prefix_cache_candidate": scheduling.get("prefix_cache_candidate"),
            "prefix_cache_lookup_source": scheduling.get("prefix_cache_lookup_source"),
            "prefill_isolation_requested": scheduling.get("prefill_isolation_requested"),
            "prefill_isolation_active": scheduling.get("prefill_isolation_active"),
            "prompt_token_estimate": scheduling.get("prompt_token_estimate"),
            "scheduler_wait_milliseconds": scheduling.get("scheduler_wait_milliseconds"),
            "chunked_prefill_requested": scheduling.get("chunked_prefill_requested"),
            "chunk_count": scheduling.get("chunk_count"),
        }

    def _sync_serving_request_metadata(self, request: GenerateRequest) -> None:
        request_id = request.request_id
        if not request_id:
            return
        if serving_payload := self.serving_core.sequence_metadata(request_id):
            request.metadata["serving"] = serving_payload

    def _record_serving_batch_metrics(
        self,
        *,
        request: GenerateRequest,
        batch_metrics: FrontierBatchMetrics | None,
    ) -> None:
        request_id = request.request_id
        if not request_id:
            return
        self.serving_core.record_batch_metrics(request_id=request_id, batch_metrics=batch_metrics)
        self._sync_serving_request_metadata(request)

    def _record_serving_queue(
        self,
        *,
        request: GenerateRequest,
        queue_type: ServingQueueType,
        wait_seconds: float,
    ) -> None:
        request_id = request.request_id
        if not request_id:
            return
        self.serving_core.record_queue(
            request_id=request_id,
            queue_type=queue_type,
            wait_seconds=wait_seconds,
        )
        self._sync_serving_request_metadata(request)

    def _record_serving_admission(
        self,
        *,
        request: GenerateRequest,
        admission: RuntimeRequestAdmission,
    ) -> None:
        request_id = request.request_id
        if not request_id:
            return
        self.serving_core.admit_sequence(request_id=request_id, admission=admission)
        self._sync_serving_request_metadata(request)

    def _record_serving_phase(
        self,
        *,
        request: GenerateRequest,
        phase: ServingPhase,
        detail: str | None,
    ) -> None:
        request_id = request.request_id
        if not request_id:
            return
        self.serving_core.transition_phase(request_id=request_id, phase=phase, detail=detail)
        self._sync_serving_request_metadata(request)

    @staticmethod
    def _request_scheduling_payload(request: GenerateRequest) -> dict[str, object]:
        scheduling = request.metadata.get("scheduling")
        if not isinstance(scheduling, dict):
            return {}
        return {
            "queue_lane": scheduling.get("queue_lane"),
            "decode_priority_requested": scheduling.get("decode_priority_requested"),
            "decode_priority_active": scheduling.get("decode_priority_active"),
            "prefill_heavy": scheduling.get("prefill_heavy"),
            "total_prompt_tokens": scheduling.get("total_prompt_tokens"),
            "cached_prefix_tokens": scheduling.get("cached_prefix_tokens"),
            "prefix_cache_candidate": scheduling.get("prefix_cache_candidate"),
            "prefix_cache_lookup_source": scheduling.get("prefix_cache_lookup_source"),
            "prefill_isolation_requested": scheduling.get("prefill_isolation_requested"),
            "prefill_isolation_active": scheduling.get("prefill_isolation_active"),
            "prompt_token_estimate": scheduling.get("prompt_token_estimate"),
            "scheduler_wait_milliseconds": scheduling.get("scheduler_wait_milliseconds"),
            "chunked_prefill_requested": scheduling.get("chunked_prefill_requested"),
            "chunked_prefill_active": scheduling.get("chunked_prefill_active"),
            "chunk_count": scheduling.get("chunk_count"),
        }

    @staticmethod
    def _finalize_request_scheduling_metadata(request: GenerateRequest) -> None:
        scheduling = request.metadata.get("scheduling")
        if not isinstance(scheduling, dict):
            return
        controls = request.metadata.get("performance_controls")
        chunked_control: dict[str, object] = {}
        if isinstance(controls, dict):
            for phase_name in ("generate", "load"):
                phase_payload = controls.get(phase_name)
                if not isinstance(phase_payload, dict):
                    continue
                candidate = phase_payload.get("chunked_prefill")
                if isinstance(candidate, dict):
                    chunked_control = candidate
                    break
        if chunked_control:
            scheduling["chunked_prefill_requested"] = bool(chunked_control.get("requested"))
            scheduling["chunked_prefill_active"] = str(chunked_control.get("effective")) == "enabled"
            chunk_count = chunked_control.get("chunk_count")
            if isinstance(chunk_count, int) and not isinstance(chunk_count, bool):
                scheduling["chunk_count"] = chunk_count

    @staticmethod
    def _batch_admission_policy(
        contexts: Sequence[_ChatInvocationContext],
    ) -> tuple[bool, bool, bool]:
        prefill_heavy = bool(contexts) and all(context.prefill_heavy for context in contexts)
        decode_priority = any(context.decode_priority_requested for context in contexts)
        prefill_isolation = prefill_heavy and all(context.prefill_isolation_requested for context in contexts)
        return prefill_heavy, decode_priority, prefill_isolation

    async def _execute_complete_context(
        self,
        context: _ChatInvocationContext,
        *,
        batch_metrics: FrontierBatchMetrics | None = None,
        load_seconds_scale: float = 1.0,
        execution_seconds_scale: float = 1.0,
    ) -> ChatExecution:
        try:
            admission = await self.runtime_request_scheduler.acquire(
                prefill_heavy=context.prefill_heavy,
                decode_priority=context.decode_priority_requested,
                prefill_isolation=context.prefill_isolation_requested,
            )
        except Exception as exc:
            self.serving_core.fail_sequence(request_id=context.request_id, error=str(exc))
            self._sync_serving_request_metadata(context.request)
            await self._publish(
                EventType.REQUEST_FAILED,
                {"request_id": context.request_id, "model_id": context.manifest.model_id, "error": str(exc)},
            )
            raise
        self._record_serving_batch_metrics(request=context.request, batch_metrics=batch_metrics)
        batch_payload = {
            **self._continuous_batch_payload(batch_metrics),
            **self._apply_scheduler_admission(request=context.request, admission=admission),
        }
        if batch_metrics is not None and batch_metrics.queue_delay_seconds > 0 and batch_metrics.batch_size > 1:
            self._record_serving_queue(
                request=context.request,
                queue_type=ServingQueueType.BATCH_WINDOW,
                wait_seconds=batch_metrics.queue_delay_seconds,
            )
            await self._publish(
                EventType.REQUEST_QUEUED,
                {
                    "request_id": context.request_id,
                    "model_id": context.manifest.model_id,
                    "runtime": context.runtime.name,
                    "wait_seconds": batch_metrics.queue_delay_seconds,
                    "queue_type": "continuous_batching",
                    **batch_payload,
                },
            )
        if admission.was_queued:
            self._record_serving_queue(
                request=context.request,
                queue_type=ServingQueueType.RUNTIME_REQUEST,
                wait_seconds=admission.wait_seconds,
            )
            await self._publish(
                EventType.REQUEST_QUEUED,
                {
                    "request_id": context.request_id,
                    "model_id": context.manifest.model_id,
                    "runtime": context.runtime.name,
                    "wait_seconds": admission.wait_seconds,
                    "queue_type": "runtime_request",
                    **batch_payload,
                },
            )
        self._record_serving_admission(request=context.request, admission=admission)
        await self._publish(
            EventType.REQUEST_ACCEPTED,
            {
                "request_id": context.request_id,
                "requested_model_id": context.requested_model_id,
                "model_id": context.manifest.model_id,
                **batch_payload,
            },
        )
        await self._publish_progress(
            request_id=context.request_id,
            operation="text.generation",
            stage="prompt_compiled",
            completed_steps=1,
            total_steps=4,
            reasoning_visibility=context.reasoning_visibility,
            model_id=context.manifest.model_id,
            message_count=len(context.request.messages),
            prompt_trace_requested=context.prompt_trace_requested,
            **batch_payload,
        )
        load_admission: RuntimeRequestAdmission | None = None
        try:
            load_started_at = time.perf_counter()
            load_admission = await self._acquire_model_load_admission(
                request=context.request,
                request_id=context.request_id,
                requested_model_id=context.requested_model_id,
                manifest=context.manifest,
                runtime=context.runtime,
            )
            self._record_serving_phase(
                request=context.request,
                phase=ServingPhase.MODEL_LOADING,
                detail="model_loading_started",
            )
            await self._publish(
                EventType.MODEL_LOADING,
                {"request_id": context.request_id, "model_id": context.manifest.model_id, "runtime": context.runtime.name},
            )
            await context.runtime.load_model(context.manifest)
            for companion_manifest in context.companion_manifests:
                await context.runtime.load_model(companion_manifest)
            await self.model_router.runtime_catalog.prepare_runtime_for_request(
                context.manifest,
                context.runtime,
                policy=self.model_router.settings.runtime_policy,
            )
            load_seconds = time.perf_counter() - load_started_at
            await self._publish(
                EventType.MODEL_LOADED,
                {"request_id": context.request_id, "model_id": context.manifest.model_id, "runtime": context.runtime.name},
            )
            await self._publish_progress(
                request_id=context.request_id,
                operation="text.generation",
                stage="model_ready",
                completed_steps=2,
                total_steps=4,
                reasoning_visibility=context.reasoning_visibility,
                model_id=context.manifest.model_id,
                runtime=context.runtime.name,
                **batch_payload,
            )
            self._record_serving_phase(
                request=context.request,
                phase=ServingPhase.PREFILL,
                detail="prefill_started",
            )
            await self._publish(
                EventType.PREFILL_STARTED,
                {"request_id": context.request_id, "model_id": context.manifest.model_id},
            )
            await self._publish_speculation_started(
                request=context.request,
                request_id=context.request_id,
                model_id=context.manifest.model_id,
                runtime=context.runtime.name,
            )
            self._record_serving_phase(
                request=context.request,
                phase=ServingPhase.DECODE,
                detail="decode_started",
            )
            await self._publish_progress(
                request_id=context.request_id,
                operation="text.generation",
                stage="response_generating",
                completed_steps=3,
                total_steps=4,
                reasoning_visibility=context.reasoning_visibility,
                model_id=context.manifest.model_id,
                runtime=context.runtime.name,
                **batch_payload,
            )
            generate_started_at = time.perf_counter()
            response = await context.runtime.generate(context.request)
            self._finalize_request_scheduling_metadata(context.request)
            execution_seconds = time.perf_counter() - generate_started_at
            response_metadata = self._set_execution_metadata(
                request=context.request,
                request_id=context.request_id,
                created_at=context.created_at,
                requested_model_id=context.requested_model_id,
                routing=context.routing,
                admission=admission,
                load_admission=load_admission,
                batch_metrics=batch_metrics,
                load_seconds=load_seconds,
                execute_seconds=execution_seconds,
            )
            output_text, reasoning = apply_reasoning_visibility(
                response.output_text,
                context.reasoning_visibility,
                existing_reasoning=response.reasoning,
            )
            output_text, citations = resolve_generated_citations(output_text, context.citation_context)
            response = response.model_copy(
                update={"output_text": output_text, "reasoning": reasoning, "citations": citations},
            )
            self.runtime_metrics_recorder.record_success(
                model_id=context.manifest.model_id,
                runtime=context.runtime.name,
                capability="chat",
                load_seconds=load_seconds * load_seconds_scale,
                execution_seconds=execution_seconds * execution_seconds_scale,
                usage=response.usage,
                measurements={
                    **_chat_measurements(context.request.messages, output_characters=len(response.output_text)),
                    **_request_cache_measurements(request=context.request),
                    **_request_scheduling_measurements(request=context.request),
                    **frontier_architecture_measurements(context.request.metadata),
                    **speculation_measurements(request=context.request, usage=response.usage),
                    **_continuous_batch_measurements(batch_metrics=batch_metrics),
                },
            )
            await self._publish_progress(
                request_id=context.request_id,
                operation="text.generation",
                stage="response_ready",
                completed_steps=4,
                total_steps=4,
                reasoning_visibility=context.reasoning_visibility,
                reasoning=response.reasoning,
                model_id=context.manifest.model_id,
                output_characters=len(response.output_text),
                finish_reason=response.finish_reason,
                **batch_payload,
            )
            await self._publish_speculation_result(
                request=context.request,
                request_id=context.request_id,
                model_id=context.manifest.model_id,
                runtime=context.runtime.name,
                usage=response.usage,
            )
            self.serving_core.complete_sequence(request_id=context.request_id)
            self._sync_serving_request_metadata(context.request)
            self._refresh_execution_serving_metadata(request=context.request, metadata=response_metadata)
            await self._publish(
                EventType.REQUEST_COMPLETED,
                {"request_id": context.request_id, "model_id": context.manifest.model_id, "finish_reason": response.finish_reason},
            )
            return ChatExecution(
                request_id=context.request_id,
                created_at=context.created_at,
                response=response,
                routing=context.routing,
                prompt_trace=context.prompt_trace,
                request_metadata=dict(context.request.metadata),
                metadata=response_metadata,
                structured_output=_structured_output_result(context.request, context.prompt_trace, response.output_text),
                serving_profile=self._serving_profile_from_metadata(context.request.metadata),
            )
        except Exception as exc:
            now = time.perf_counter()
            load_seconds = now - load_started_at if "load_started_at" in locals() else 0.0
            execution_seconds = now - generate_started_at if "generate_started_at" in locals() else 0.0
            if "generate_started_at" in locals():
                load_seconds = max(load_seconds - execution_seconds, 0.0)
            self._finalize_request_scheduling_metadata(context.request)
            self.runtime_metrics_recorder.record_failure(
                model_id=context.manifest.model_id,
                runtime=context.runtime.name,
                capability="chat",
                load_seconds=load_seconds * load_seconds_scale,
                execution_seconds=execution_seconds * execution_seconds_scale,
                measurements={
                    **_chat_measurements(context.request.messages),
                    **_request_cache_measurements(request=context.request),
                    **_request_scheduling_measurements(request=context.request),
                    **frontier_architecture_measurements(context.request.metadata),
                    **speculation_measurements(request=context.request),
                    **_continuous_batch_measurements(batch_metrics=batch_metrics),
                },
            )
            self.serving_core.fail_sequence(request_id=context.request_id, error=str(exc))
            self._sync_serving_request_metadata(context.request)
            await self._publish(
                EventType.REQUEST_FAILED,
                {"request_id": context.request_id, "model_id": context.manifest.model_id, "error": str(exc)},
            )
            raise
        finally:
            if load_admission is not None:
                load_admission.release()
            admission.release()
            await self._finalize_runtime_cleanup(
                manifest=context.manifest,
                runtime=context.runtime,
                companion_manifests=context.companion_manifests,
            )

    async def _execute_complete_batch(
        self,
        batch_payloads: list[tuple[_ChatInvocationContext, FrontierBatchMetrics]],
    ) -> list[ChatExecution]:
        if not batch_payloads:
            return []
        contexts = [context for context, _ in batch_payloads]
        manifest = contexts[0].manifest
        runtime = contexts[0].runtime
        prefill_heavy, decode_priority, prefill_isolation = self._batch_admission_policy(contexts)
        try:
            admission = await self.runtime_request_scheduler.acquire(
                prefill_heavy=prefill_heavy,
                decode_priority=decode_priority,
                prefill_isolation=prefill_isolation,
            )
        except Exception as exc:
            for context in contexts:
                self.serving_core.fail_sequence(request_id=context.request_id, error=str(exc))
                self._sync_serving_request_metadata(context.request)
                await self._publish(
                    EventType.REQUEST_FAILED,
                    {"request_id": context.request_id, "model_id": context.manifest.model_id, "error": str(exc)},
                )
            raise
        if admission.was_queued:
            for context, frontier_metrics in batch_payloads:
                self._record_serving_batch_metrics(request=context.request, batch_metrics=frontier_metrics)
                scheduler_payload = self._apply_scheduler_admission(request=context.request, admission=admission)
                self._record_serving_queue(
                    request=context.request,
                    queue_type=ServingQueueType.RUNTIME_REQUEST,
                    wait_seconds=admission.wait_seconds,
                )
                await self._publish(
                    EventType.REQUEST_QUEUED,
                    {
                        "request_id": context.request_id,
                        "model_id": context.manifest.model_id,
                        "runtime": runtime.name,
                        "wait_seconds": admission.wait_seconds,
                        "queue_type": "runtime_request",
                        **self._continuous_batch_payload(frontier_metrics),
                        **scheduler_payload,
                    },
                )
        for context, frontier_metrics in batch_payloads:
            self._record_serving_batch_metrics(request=context.request, batch_metrics=frontier_metrics)
            scheduler_payload = self._apply_scheduler_admission(request=context.request, admission=admission)
            if frontier_metrics.queue_delay_seconds > 0 and frontier_metrics.batch_size > 1:
                self._record_serving_queue(
                    request=context.request,
                    queue_type=ServingQueueType.BATCH_WINDOW,
                    wait_seconds=frontier_metrics.queue_delay_seconds,
                )
                await self._publish(
                    EventType.REQUEST_QUEUED,
                    {
                        "request_id": context.request_id,
                        "model_id": context.manifest.model_id,
                        "runtime": runtime.name,
                        "wait_seconds": frontier_metrics.queue_delay_seconds,
                        "queue_type": "continuous_batching",
                        **self._continuous_batch_payload(frontier_metrics),
                        **scheduler_payload,
                    },
                )
            self._record_serving_admission(request=context.request, admission=admission)
            await self._publish(
                EventType.REQUEST_ACCEPTED,
                {
                    "request_id": context.request_id,
                    "requested_model_id": context.requested_model_id,
                    "model_id": context.manifest.model_id,
                    **self._continuous_batch_payload(frontier_metrics),
                    **scheduler_payload,
                },
            )
            await self._publish_progress(
                request_id=context.request_id,
                operation="text.generation",
                stage="prompt_compiled",
                completed_steps=1,
                total_steps=4,
                reasoning_visibility=context.reasoning_visibility,
                model_id=context.manifest.model_id,
                message_count=len(context.request.messages),
                prompt_trace_requested=context.prompt_trace_requested,
                **self._continuous_batch_payload(frontier_metrics),
                **scheduler_payload,
            )
        load_admission: RuntimeRequestAdmission | None = None
        try:
            load_started_at = time.perf_counter()
            load_admission = await self._acquire_model_load_admission(
                request=contexts[0].request,
                request_id=contexts[0].request_id,
                requested_model_id=contexts[0].requested_model_id,
                manifest=manifest,
                runtime=runtime,
            )
            for context, frontier_metrics in batch_payloads:
                scheduler_payload = self._request_scheduling_payload(context.request)
                self._record_serving_phase(
                    request=context.request,
                    phase=ServingPhase.MODEL_LOADING,
                    detail="model_loading_started",
                )
                await self._publish(
                    EventType.MODEL_LOADING,
                    {
                        "request_id": context.request_id,
                        "model_id": context.manifest.model_id,
                        "runtime": runtime.name,
                        **self._continuous_batch_payload(frontier_metrics),
                        **scheduler_payload,
                    },
                )
            await runtime.load_model(manifest)
            for companion_manifest in self._unique_companion_manifests(contexts):
                await runtime.load_model(companion_manifest)
            await self.model_router.runtime_catalog.prepare_runtime_for_request(
                manifest,
                runtime,
                policy=self.model_router.settings.runtime_policy,
            )
            load_seconds = time.perf_counter() - load_started_at
            for context, frontier_metrics in batch_payloads:
                scheduler_payload = self._request_scheduling_payload(context.request)
                await self._publish(
                    EventType.MODEL_LOADED,
                    {
                        "request_id": context.request_id,
                        "model_id": context.manifest.model_id,
                        "runtime": runtime.name,
                        **self._continuous_batch_payload(frontier_metrics),
                        **scheduler_payload,
                    },
                )
                await self._publish_progress(
                    request_id=context.request_id,
                    operation="text.generation",
                    stage="model_ready",
                    completed_steps=2,
                    total_steps=4,
                    reasoning_visibility=context.reasoning_visibility,
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                    **self._continuous_batch_payload(frontier_metrics),
                    **scheduler_payload,
                )
                self._record_serving_phase(
                    request=context.request,
                    phase=ServingPhase.PREFILL,
                    detail="prefill_started",
                )
                await self._publish(
                    EventType.PREFILL_STARTED,
                    {"request_id": context.request_id, "model_id": context.manifest.model_id},
                )
                await self._publish_speculation_started(
                    request=context.request,
                    request_id=context.request_id,
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                )
                self._record_serving_phase(
                    request=context.request,
                    phase=ServingPhase.DECODE,
                    detail="decode_started",
                )
                await self._publish_progress(
                    request_id=context.request_id,
                    operation="text.generation",
                    stage="response_generating",
                    completed_steps=3,
                    total_steps=4,
                    reasoning_visibility=context.reasoning_visibility,
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                    **self._continuous_batch_payload(frontier_metrics),
                    **scheduler_payload,
                )
            generate_started_at = time.perf_counter()
            responses = await runtime.generate_batch([context.request for context in contexts])
            execution_seconds = time.perf_counter() - generate_started_at
            if len(responses) != len(contexts):
                raise ValueError(f"Expected {len(contexts)} batched responses, received {len(responses)}.")
            per_request_load_seconds = load_seconds / len(contexts)
            per_request_execution_seconds = execution_seconds / len(contexts)
            executions: list[ChatExecution] = []
            for context, frontier_metrics, response in zip(
                contexts,
                [metrics for _, metrics in batch_payloads],
                responses,
                strict=True,
            ):
                self._finalize_request_scheduling_metadata(context.request)
                response_metadata = self._set_execution_metadata(
                    request=context.request,
                    request_id=context.request_id,
                    created_at=context.created_at,
                    requested_model_id=context.requested_model_id,
                    routing=context.routing,
                    admission=admission,
                    load_admission=load_admission,
                    batch_metrics=frontier_metrics,
                    load_seconds=load_seconds,
                    execute_seconds=execution_seconds,
                )
                output_text, reasoning = apply_reasoning_visibility(
                    response.output_text,
                    context.reasoning_visibility,
                    existing_reasoning=response.reasoning,
                )
                output_text, citations = resolve_generated_citations(output_text, context.citation_context)
                response = response.model_copy(
                    update={"output_text": output_text, "reasoning": reasoning, "citations": citations},
                )
                self.runtime_metrics_recorder.record_success(
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                    capability="chat",
                    load_seconds=per_request_load_seconds,
                    execution_seconds=per_request_execution_seconds,
                    usage=response.usage,
                    measurements={
                        **_chat_measurements(context.request.messages, output_characters=len(response.output_text)),
                        **_request_cache_measurements(request=context.request),
                        **_request_scheduling_measurements(request=context.request),
                        **frontier_architecture_measurements(context.request.metadata),
                        **speculation_measurements(request=context.request, usage=response.usage),
                        **_continuous_batch_measurements(batch_metrics=frontier_metrics),
                    },
                )
                await self._publish_progress(
                    request_id=context.request_id,
                    operation="text.generation",
                    stage="response_ready",
                    completed_steps=4,
                    total_steps=4,
                    reasoning_visibility=context.reasoning_visibility,
                    reasoning=response.reasoning,
                    model_id=context.manifest.model_id,
                    output_characters=len(response.output_text),
                    finish_reason=response.finish_reason,
                    **self._continuous_batch_payload(frontier_metrics),
                )
                await self._publish_speculation_result(
                    request=context.request,
                    request_id=context.request_id,
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                    usage=response.usage,
                )
                self.serving_core.complete_sequence(request_id=context.request_id)
                self._sync_serving_request_metadata(context.request)
                self._refresh_execution_serving_metadata(request=context.request, metadata=response_metadata)
                await self._publish(
                    EventType.REQUEST_COMPLETED,
                    {"request_id": context.request_id, "model_id": context.manifest.model_id, "finish_reason": response.finish_reason},
                )
                executions.append(
                    ChatExecution(
                        request_id=context.request_id,
                        created_at=context.created_at,
                        response=response,
                        routing=context.routing,
                        prompt_trace=context.prompt_trace,
                        request_metadata=dict(context.request.metadata),
                        metadata=response_metadata,
                        structured_output=_structured_output_result(context.request, context.prompt_trace, response.output_text),
                        serving_profile=self._serving_profile_from_metadata(context.request.metadata),
                    ),
                )
            return executions
        except Exception as exc:
            now = time.perf_counter()
            load_seconds = now - load_started_at if "load_started_at" in locals() else 0.0
            execution_seconds = now - generate_started_at if "generate_started_at" in locals() else 0.0
            if "generate_started_at" in locals():
                load_seconds = max(load_seconds - execution_seconds, 0.0)
            per_request_load_seconds = load_seconds / len(contexts) if contexts else 0.0
            per_request_execution_seconds = execution_seconds / len(contexts) if contexts else 0.0
            for context, frontier_metrics in batch_payloads:
                self._finalize_request_scheduling_metadata(context.request)
                self.runtime_metrics_recorder.record_failure(
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                    capability="chat",
                    load_seconds=per_request_load_seconds,
                    execution_seconds=per_request_execution_seconds,
                    measurements={
                        **_chat_measurements(context.request.messages),
                        **_request_cache_measurements(request=context.request),
                        **_request_scheduling_measurements(request=context.request),
                        **frontier_architecture_measurements(context.request.metadata),
                        **speculation_measurements(request=context.request),
                        **_continuous_batch_measurements(batch_metrics=frontier_metrics),
                    },
                )
                self.serving_core.fail_sequence(request_id=context.request_id, error=str(exc))
                self._sync_serving_request_metadata(context.request)
                await self._publish(
                    EventType.REQUEST_FAILED,
                    {"request_id": context.request_id, "model_id": context.manifest.model_id, "error": str(exc)},
                )
            raise
        finally:
            if load_admission is not None:
                load_admission.release()
            admission.release()
            await self._finalize_runtime_cleanup(
                manifest=manifest,
                runtime=runtime,
                companion_manifests=self._unique_companion_manifests(contexts),
            )

    async def _execute_stream_context(
        self,
        context: _ChatInvocationContext,
        *,
        batch_metrics: FrontierBatchMetrics | None = None,
        load_seconds_scale: float = 1.0,
        execution_seconds_scale: float = 1.0,
    ) -> ChatStreamSession:
        try:
            admission = await self.runtime_request_scheduler.acquire(
                prefill_heavy=context.prefill_heavy,
                decode_priority=context.decode_priority_requested,
                prefill_isolation=context.prefill_isolation_requested,
            )
        except Exception as exc:
            self.serving_core.fail_sequence(request_id=context.request_id, error=str(exc))
            self._sync_serving_request_metadata(context.request)
            await self._publish(
                EventType.REQUEST_FAILED,
                {"request_id": context.request_id, "model_id": context.manifest.model_id, "error": str(exc)},
            )
            raise
        self._record_serving_batch_metrics(request=context.request, batch_metrics=batch_metrics)
        batch_payload = {
            **self._continuous_batch_payload(batch_metrics),
            **self._apply_scheduler_admission(request=context.request, admission=admission),
        }
        if batch_metrics is not None and batch_metrics.queue_delay_seconds > 0 and batch_metrics.batch_size > 1:
            self._record_serving_queue(
                request=context.request,
                queue_type=ServingQueueType.BATCH_WINDOW,
                wait_seconds=batch_metrics.queue_delay_seconds,
            )
            await self._publish(
                EventType.REQUEST_QUEUED,
                {
                    "request_id": context.request_id,
                    "model_id": context.manifest.model_id,
                    "runtime": context.runtime.name,
                    "wait_seconds": batch_metrics.queue_delay_seconds,
                    "queue_type": "continuous_batching",
                    **batch_payload,
                },
            )
        if admission.was_queued:
            self._record_serving_queue(
                request=context.request,
                queue_type=ServingQueueType.RUNTIME_REQUEST,
                wait_seconds=admission.wait_seconds,
            )
            await self._publish(
                EventType.REQUEST_QUEUED,
                {
                    "request_id": context.request_id,
                    "model_id": context.manifest.model_id,
                    "runtime": context.runtime.name,
                    "wait_seconds": admission.wait_seconds,
                    "queue_type": "runtime_request",
                    **batch_payload,
                },
            )
        self._record_serving_admission(request=context.request, admission=admission)
        await self._publish(
            EventType.REQUEST_ACCEPTED,
            {
                "request_id": context.request_id,
                "requested_model_id": context.requested_model_id,
                "model_id": context.manifest.model_id,
                **batch_payload,
            },
        )
        await self._publish_progress(
            request_id=context.request_id,
            operation="text.streaming",
            stage="prompt_compiled",
            completed_steps=1,
            total_steps=5,
            reasoning_visibility=context.reasoning_visibility,
            model_id=context.manifest.model_id,
            message_count=len(context.request.messages),
            prompt_trace_requested=context.prompt_trace_requested,
            **batch_payload,
        )
        load_admission: RuntimeRequestAdmission | None = None
        try:
            load_started_at = time.perf_counter()
            load_admission = await self._acquire_model_load_admission(
                request=context.request,
                request_id=context.request_id,
                requested_model_id=context.requested_model_id,
                manifest=context.manifest,
                runtime=context.runtime,
            )
            self._record_serving_phase(
                request=context.request,
                phase=ServingPhase.MODEL_LOADING,
                detail="model_loading_started",
            )
            await self._publish(
                EventType.MODEL_LOADING,
                {"request_id": context.request_id, "model_id": context.manifest.model_id, "runtime": context.runtime.name},
            )
            await context.runtime.load_model(context.manifest)
            for companion_manifest in context.companion_manifests:
                await context.runtime.load_model(companion_manifest)
            await self.model_router.runtime_catalog.prepare_runtime_for_request(
                context.manifest,
                context.runtime,
                policy=self.model_router.settings.runtime_policy,
            )
            load_seconds = time.perf_counter() - load_started_at
            await self._publish(
                EventType.MODEL_LOADED,
                {"request_id": context.request_id, "model_id": context.manifest.model_id, "runtime": context.runtime.name},
            )
            await self._publish_progress(
                request_id=context.request_id,
                operation="text.streaming",
                stage="model_ready",
                completed_steps=2,
                total_steps=5,
                reasoning_visibility=context.reasoning_visibility,
                model_id=context.manifest.model_id,
                runtime=context.runtime.name,
                **batch_payload,
            )
            self._record_serving_phase(
                request=context.request,
                phase=ServingPhase.PREFILL,
                detail="prefill_started",
            )
            await self._publish(
                EventType.PREFILL_STARTED,
                {"request_id": context.request_id, "model_id": context.manifest.model_id},
            )
            await self._publish_speculation_started(
                request=context.request,
                request_id=context.request_id,
                model_id=context.manifest.model_id,
                runtime=context.runtime.name,
            )
            self._record_serving_phase(
                request=context.request,
                phase=ServingPhase.DECODE,
                detail="decode_started",
            )
            await self._publish_progress(
                request_id=context.request_id,
                operation="text.streaming",
                stage="stream_open",
                completed_steps=3,
                total_steps=5,
                reasoning_visibility=context.reasoning_visibility,
                model_id=context.manifest.model_id,
                runtime=context.runtime.name,
                **batch_payload,
            )
        except Exception as exc:
            self.runtime_metrics_recorder.record_failure(
                model_id=context.manifest.model_id,
                runtime=context.runtime.name,
                capability="streaming",
                load_seconds=(time.perf_counter() - load_started_at) * load_seconds_scale,
                execution_seconds=0.0,
                measurements={
                    **_request_scheduling_measurements(request=context.request),
                    **_continuous_batch_measurements(batch_metrics=batch_metrics),
                },
            )
            self.serving_core.fail_sequence(request_id=context.request_id, error=str(exc))
            self._sync_serving_request_metadata(context.request)
            await self._publish(
                EventType.REQUEST_FAILED,
                {"request_id": context.request_id, "model_id": context.manifest.model_id, "error": str(exc)},
            )
            await self.model_router.runtime_catalog.finalize_runtime_for_request(
                context.manifest,
                context.runtime,
                policy=self.model_router.settings.runtime_policy,
            )
            admission.release()
            raise

        stream_queue: asyncio.Queue[object] = asyncio.Queue()
        stream_session = self._stream_session_from_queue(context=context, queue=stream_queue)

        async def iterator() -> None:
            generate_started_at = time.perf_counter()
            emitted_delta_count = 0
            emitted_characters = 0
            emitted_stream_progress = False
            reasoning_processor = ReasoningStreamProcessor(context.reasoning_visibility)
            citation_processor = CitationStreamProcessor(context.citation_context)
            try:
                async for raw_delta in context.runtime.stream_generate(context.request):
                    for delta in reasoning_processor.consume(raw_delta):
                        visible_content = citation_processor.consume(delta.content) if delta.content is not None else None
                        if visible_content == "":
                            visible_content = None
                        if not emitted_stream_progress and (visible_content or delta.reasoning):
                            emitted_stream_progress = True
                            await self._publish_progress(
                                request_id=context.request_id,
                                operation="text.streaming",
                                stage="first_delta_emitted",
                                completed_steps=4,
                                total_steps=5,
                                reasoning_visibility=context.reasoning_visibility,
                                model_id=context.manifest.model_id,
                                runtime=context.runtime.name,
                                **batch_payload,
                            )
                        if delta.reasoning:
                            await self._publish(
                                EventType.REASONING_DELTA,
                                {
                                    "request_id": context.request_id,
                                    "model_id": context.manifest.model_id,
                                    "runtime": context.runtime.name,
                                    "delta": delta.reasoning,
                                    "reasoning_visibility": context.reasoning_visibility.value,
                                },
                            )
                        if visible_content:
                            emitted_delta_count += 1
                            emitted_characters += len(visible_content)
                            await self._publish(
                                EventType.TOKEN_DELTA,
                                {
                                    "request_id": context.request_id,
                                    "model_id": context.manifest.model_id,
                                    "delta": visible_content,
                                },
                            )
                        await stream_queue.put(
                            ChatStreamDelta(content=visible_content, reasoning=delta.reasoning),
                        )
                stream_reasoning = reasoning_processor.finalize()
                trailing_content, citations = citation_processor.finalize()
                if trailing_content:
                    emitted_delta_count += 1
                    emitted_characters += len(trailing_content)
                    await self._publish(
                        EventType.TOKEN_DELTA,
                        {
                            "request_id": context.request_id,
                            "model_id": context.manifest.model_id,
                            "delta": trailing_content,
                        },
                    )
                    await stream_queue.put(ChatStreamDelta(content=trailing_content))
                self._finalize_request_scheduling_metadata(context.request)
                execution_seconds = time.perf_counter() - generate_started_at
                stream_metadata = self._set_execution_metadata(
                    request=context.request,
                    request_id=context.request_id,
                    created_at=context.created_at,
                    requested_model_id=context.requested_model_id,
                    routing=context.routing,
                    admission=admission,
                    load_admission=load_admission,
                    batch_metrics=batch_metrics,
                    load_seconds=load_seconds,
                    execute_seconds=execution_seconds,
                )
                self.runtime_metrics_recorder.record_success(
                    model_id=context.manifest.model_id,
                    runtime=context.runtime.name,
                    capability="streaming",
                    load_seconds=load_seconds * load_seconds_scale,
                    execution_seconds=execution_seconds * execution_seconds_scale,
                    measurements={
                        **_chat_measurements(
                            context.request.messages,
                            output_characters=emitted_characters,
                            delta_count=emitted_delta_count,
                        ),
                        **_request_cache_measurements(request=context.request),
                        **_request_scheduling_measurements(request=context.request),
                        **frontier_architecture_measurements(context.request.metadata),
                        **speculation_measurements(request=context.request),
                        **_continuous_batch_measurements(batch_metrics=batch_metrics),
                    },
                )
                await self._publish_progress(
                    request_id=context.request_id,
                    operation="text.streaming",
                    stage="stream_completed",
                    completed_steps=5,
                    total_steps=5,
                    reasoning_visibility=context.reasoning_visibility,
                    reasoning=stream_reasoning,
                    model_id=context.manifest.model_id,
                    output_characters=emitted_characters,
                    delta_count=emitted_delta_count,
                    **batch_payload,
                )
                self.serving_core.complete_sequence(request_id=context.request_id)
                self._sync_serving_request_metadata(context.request)
                self._refresh_execution_serving_metadata(request=context.request, metadata=stream_metadata)
                stream_session.reasoning = stream_reasoning
                stream_session.citations = citations
                if stream_session.request_metadata is not None:
                    stream_session.request_metadata.clear()
                    stream_session.request_metadata.update(context.request.metadata)
                stream_session.metadata = stream_metadata
                await self._publish_speculation_result(
                    request=context.request,
                    request_id=context.request_id,
                    model_id=context.manifest.model_id,
                    runtime=context.runtime.name,
                )
                await self._publish(
                    EventType.REQUEST_COMPLETED,
                    {"request_id": context.request_id, "model_id": context.manifest.model_id, "finish_reason": "stop"},
                )
                await stream_queue.put(_STREAM_END)
            except Exception as exc:
                self._finalize_request_scheduling_metadata(context.request)
                self.runtime_metrics_recorder.record_failure(
                    model_id=context.manifest.model_id,
                    runtime=context.runtime.name,
                    capability="streaming",
                    load_seconds=load_seconds * load_seconds_scale,
                    execution_seconds=(time.perf_counter() - generate_started_at) * execution_seconds_scale,
                    measurements={
                        **_chat_measurements(
                            context.request.messages,
                            output_characters=emitted_characters,
                            delta_count=emitted_delta_count,
                        ),
                        **_request_cache_measurements(request=context.request),
                        **_request_scheduling_measurements(request=context.request),
                        **frontier_architecture_measurements(context.request.metadata),
                        **speculation_measurements(request=context.request),
                        **_continuous_batch_measurements(batch_metrics=batch_metrics),
                    },
                )
                self.serving_core.fail_sequence(request_id=context.request_id, error=str(exc))
                self._sync_serving_request_metadata(context.request)
                await self._publish(
                    EventType.REQUEST_FAILED,
                    {"request_id": context.request_id, "model_id": context.manifest.model_id, "error": str(exc)},
                )
                await stream_queue.put(exc)
            finally:
                if load_admission is not None:
                    load_admission.release()
                admission.release()
                await self._finalize_runtime_cleanup(
                    manifest=context.manifest,
                    runtime=context.runtime,
                    companion_manifests=context.companion_manifests,
                )

        asyncio.create_task(iterator())
        return stream_session

    async def _execute_stream_batch(
        self,
        batch_payloads: list[tuple[_ChatInvocationContext, FrontierBatchMetrics]],
    ) -> list[ChatStreamSession]:
        if not batch_payloads:
            return []
        contexts = [context for context, _ in batch_payloads]
        manifest = contexts[0].manifest
        runtime = contexts[0].runtime
        prefill_heavy, decode_priority, prefill_isolation = self._batch_admission_policy(contexts)
        try:
            admission = await self.runtime_request_scheduler.acquire(
                prefill_heavy=prefill_heavy,
                decode_priority=decode_priority,
                prefill_isolation=prefill_isolation,
            )
        except Exception as exc:
            for context in contexts:
                self.serving_core.fail_sequence(request_id=context.request_id, error=str(exc))
                self._sync_serving_request_metadata(context.request)
                await self._publish(
                    EventType.REQUEST_FAILED,
                    {"request_id": context.request_id, "model_id": context.manifest.model_id, "error": str(exc)},
                )
            raise
        if admission.was_queued:
            for context, frontier_metrics in batch_payloads:
                self._record_serving_batch_metrics(request=context.request, batch_metrics=frontier_metrics)
                scheduler_payload = self._apply_scheduler_admission(request=context.request, admission=admission)
                self._record_serving_queue(
                    request=context.request,
                    queue_type=ServingQueueType.RUNTIME_REQUEST,
                    wait_seconds=admission.wait_seconds,
                )
                await self._publish(
                    EventType.REQUEST_QUEUED,
                    {
                        "request_id": context.request_id,
                        "model_id": context.manifest.model_id,
                        "runtime": runtime.name,
                        "wait_seconds": admission.wait_seconds,
                        "queue_type": "runtime_request",
                        **self._continuous_batch_payload(frontier_metrics),
                        **scheduler_payload,
                    },
                )
        for context, frontier_metrics in batch_payloads:
            self._record_serving_batch_metrics(request=context.request, batch_metrics=frontier_metrics)
            scheduler_payload = self._apply_scheduler_admission(request=context.request, admission=admission)
            if frontier_metrics.queue_delay_seconds > 0 and frontier_metrics.batch_size > 1:
                self._record_serving_queue(
                    request=context.request,
                    queue_type=ServingQueueType.BATCH_WINDOW,
                    wait_seconds=frontier_metrics.queue_delay_seconds,
                )
                await self._publish(
                    EventType.REQUEST_QUEUED,
                    {
                        "request_id": context.request_id,
                        "model_id": context.manifest.model_id,
                        "runtime": runtime.name,
                        "wait_seconds": frontier_metrics.queue_delay_seconds,
                        "queue_type": "continuous_batching",
                        **self._continuous_batch_payload(frontier_metrics),
                        **scheduler_payload,
                    },
                )
            self._record_serving_admission(request=context.request, admission=admission)
            await self._publish(
                EventType.REQUEST_ACCEPTED,
                {
                    "request_id": context.request_id,
                    "requested_model_id": context.requested_model_id,
                    "model_id": context.manifest.model_id,
                    **self._continuous_batch_payload(frontier_metrics),
                    **scheduler_payload,
                },
            )
            await self._publish_progress(
                request_id=context.request_id,
                operation="text.streaming",
                stage="prompt_compiled",
                completed_steps=1,
                total_steps=5,
                reasoning_visibility=context.reasoning_visibility,
                model_id=context.manifest.model_id,
                message_count=len(context.request.messages),
                prompt_trace_requested=context.prompt_trace_requested,
                **self._continuous_batch_payload(frontier_metrics),
                **scheduler_payload,
            )
        load_admission: RuntimeRequestAdmission | None = None
        try:
            load_started_at = time.perf_counter()
            load_admission = await self._acquire_model_load_admission(
                request=contexts[0].request,
                request_id=contexts[0].request_id,
                requested_model_id=contexts[0].requested_model_id,
                manifest=manifest,
                runtime=runtime,
            )
            for context, frontier_metrics in batch_payloads:
                self._record_serving_phase(
                    request=context.request,
                    phase=ServingPhase.MODEL_LOADING,
                    detail="model_loading_started",
                )
                await self._publish(
                    EventType.MODEL_LOADING,
                    {
                        "request_id": context.request_id,
                        "model_id": context.manifest.model_id,
                        "runtime": runtime.name,
                        **self._continuous_batch_payload(frontier_metrics),
                    },
                )
            await runtime.load_model(manifest)
            for companion_manifest in self._unique_companion_manifests(contexts):
                await runtime.load_model(companion_manifest)
            await self.model_router.runtime_catalog.prepare_runtime_for_request(
                manifest,
                runtime,
                policy=self.model_router.settings.runtime_policy,
            )
            load_seconds = time.perf_counter() - load_started_at
            queues: list[asyncio.Queue[object]] = [asyncio.Queue() for _ in contexts]
            sessions = [
                self._stream_session_from_queue(context=context, queue=queue)
                for context, queue in zip(contexts, queues, strict=True)
            ]
            for context, frontier_metrics in batch_payloads:
                await self._publish(
                    EventType.MODEL_LOADED,
                    {
                        "request_id": context.request_id,
                        "model_id": context.manifest.model_id,
                        "runtime": runtime.name,
                        **self._continuous_batch_payload(frontier_metrics),
                    },
                )
                await self._publish_progress(
                    request_id=context.request_id,
                    operation="text.streaming",
                    stage="model_ready",
                    completed_steps=2,
                    total_steps=5,
                    reasoning_visibility=context.reasoning_visibility,
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                    **self._continuous_batch_payload(frontier_metrics),
                )
                self._record_serving_phase(
                    request=context.request,
                    phase=ServingPhase.PREFILL,
                    detail="prefill_started",
                )
                await self._publish(
                    EventType.PREFILL_STARTED,
                    {"request_id": context.request_id, "model_id": context.manifest.model_id},
                )
                await self._publish_speculation_started(
                    request=context.request,
                    request_id=context.request_id,
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                )
                self._record_serving_phase(
                    request=context.request,
                    phase=ServingPhase.DECODE,
                    detail="decode_started",
                )
                await self._publish_progress(
                    request_id=context.request_id,
                    operation="text.streaming",
                    stage="stream_open",
                    completed_steps=3,
                    total_steps=5,
                    reasoning_visibility=context.reasoning_visibility,
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                    **self._continuous_batch_payload(frontier_metrics),
                )
            asyncio.create_task(
                self._run_batched_stream(
                    contexts=contexts,
                    frontier_metrics=[metrics for _, metrics in batch_payloads],
                    sessions=sessions,
                    queues=queues,
                    runtime=runtime,
                    manifest=manifest,
                    load_seconds=load_seconds,
                    load_admission=load_admission,
                    admission=admission,
                ),
            )
            return sessions
        except Exception as exc:
            await self.model_router.runtime_catalog.finalize_runtime_for_request(
                manifest,
                runtime,
                policy=self.model_router.settings.runtime_policy,
            )
            if load_admission is not None:
                load_admission.release()
            admission.release()
            for context, frontier_metrics in batch_payloads:
                self._finalize_request_scheduling_metadata(context.request)
                self.runtime_metrics_recorder.record_failure(
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                    capability="streaming",
                    load_seconds=(time.perf_counter() - load_started_at) / len(contexts),
                    execution_seconds=0.0,
                    measurements={
                        **_request_scheduling_measurements(request=context.request),
                        **_continuous_batch_measurements(batch_metrics=frontier_metrics),
                    },
                )
                self.serving_core.fail_sequence(request_id=context.request_id, error=str(exc))
                self._sync_serving_request_metadata(context.request)
                await self._publish(
                    EventType.REQUEST_FAILED,
                    {"request_id": context.request_id, "model_id": context.manifest.model_id, "error": str(exc)},
                )
            raise

    async def _run_batched_stream(
        self,
        *,
        contexts: list[_ChatInvocationContext],
        frontier_metrics: list[FrontierBatchMetrics],
        sessions: list[ChatStreamSession],
        queues: list[asyncio.Queue[object]],
        runtime: RuntimeContract,
        manifest: ModelManifest,
        load_seconds: float,
        load_admission: RuntimeRequestAdmission | None,
        admission: RuntimeRequestAdmission,
    ) -> None:
        generate_started_at = time.perf_counter()
        emitted_delta_count = [0 for _ in contexts]
        emitted_characters = [0 for _ in contexts]
        emitted_stream_progress = [False for _ in contexts]
        reasoning_processors = [ReasoningStreamProcessor(context.reasoning_visibility) for context in contexts]
        citation_processors = [CitationStreamProcessor(context.citation_context) for context in contexts]
        try:
            async for request_index, raw_delta in runtime.stream_generate_batch([context.request for context in contexts]):
                if request_index < 0 or request_index >= len(contexts):
                    continue
                context = contexts[request_index]
                queue = queues[request_index]
                for delta in reasoning_processors[request_index].consume(raw_delta):
                    visible_content = (
                        citation_processors[request_index].consume(delta.content)
                        if delta.content is not None
                        else None
                    )
                    if visible_content == "":
                        visible_content = None
                    if not emitted_stream_progress[request_index] and (visible_content or delta.reasoning):
                        emitted_stream_progress[request_index] = True
                        scheduler_payload = self._request_scheduling_payload(context.request)
                        await self._publish_progress(
                            request_id=context.request_id,
                            operation="text.streaming",
                            stage="first_delta_emitted",
                            completed_steps=4,
                            total_steps=5,
                            reasoning_visibility=context.reasoning_visibility,
                            model_id=context.manifest.model_id,
                            runtime=runtime.name,
                            **self._continuous_batch_payload(frontier_metrics[request_index]),
                            **scheduler_payload,
                        )
                    if delta.reasoning:
                        await self._publish(
                            EventType.REASONING_DELTA,
                            {
                                "request_id": context.request_id,
                                "model_id": context.manifest.model_id,
                                "runtime": runtime.name,
                                "delta": delta.reasoning,
                                "reasoning_visibility": context.reasoning_visibility.value,
                                },
                            )
                    if visible_content:
                        emitted_delta_count[request_index] += 1
                        emitted_characters[request_index] += len(visible_content)
                        await self._publish(
                            EventType.TOKEN_DELTA,
                            {
                                "request_id": context.request_id,
                                "model_id": context.manifest.model_id,
                                "delta": visible_content,
                            },
                        )
                    await queue.put(
                        ChatStreamDelta(content=visible_content, reasoning=delta.reasoning),
                    )
            execution_seconds = time.perf_counter() - generate_started_at
            per_request_load_seconds = load_seconds / len(contexts)
            per_request_execution_seconds = execution_seconds / len(contexts)
            for index, (context, session, queue, item_metrics) in enumerate(
                zip(contexts, sessions, queues, frontier_metrics, strict=True),
            ):
                stream_reasoning = reasoning_processors[index].finalize()
                trailing_content, citations = citation_processors[index].finalize()
                if trailing_content:
                    emitted_delta_count[index] += 1
                    emitted_characters[index] += len(trailing_content)
                    await self._publish(
                        EventType.TOKEN_DELTA,
                        {
                            "request_id": context.request_id,
                            "model_id": context.manifest.model_id,
                            "delta": trailing_content,
                        },
                    )
                    await queue.put(ChatStreamDelta(content=trailing_content))
                self._finalize_request_scheduling_metadata(context.request)
                session_metadata = self._set_execution_metadata(
                    request=context.request,
                    request_id=context.request_id,
                    created_at=context.created_at,
                    requested_model_id=context.requested_model_id,
                    routing=context.routing,
                    admission=admission,
                    load_admission=load_admission,
                    batch_metrics=item_metrics,
                    load_seconds=load_seconds,
                    execute_seconds=execution_seconds,
                )
                self.runtime_metrics_recorder.record_success(
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                    capability="streaming",
                    load_seconds=per_request_load_seconds,
                    execution_seconds=per_request_execution_seconds,
                    measurements={
                        **_chat_measurements(
                            context.request.messages,
                            output_characters=emitted_characters[index],
                            delta_count=emitted_delta_count[index],
                        ),
                        **_request_cache_measurements(request=context.request),
                        **_request_scheduling_measurements(request=context.request),
                        **frontier_architecture_measurements(context.request.metadata),
                        **speculation_measurements(request=context.request),
                        **_continuous_batch_measurements(batch_metrics=item_metrics),
                    },
                )
                await self._publish_progress(
                    request_id=context.request_id,
                    operation="text.streaming",
                    stage="stream_completed",
                    completed_steps=5,
                    total_steps=5,
                    reasoning_visibility=context.reasoning_visibility,
                    reasoning=stream_reasoning,
                    model_id=context.manifest.model_id,
                    output_characters=emitted_characters[index],
                    delta_count=emitted_delta_count[index],
                    **self._continuous_batch_payload(item_metrics),
                    **self._request_scheduling_payload(context.request),
                )
                self.serving_core.complete_sequence(request_id=context.request_id)
                self._sync_serving_request_metadata(context.request)
                self._refresh_execution_serving_metadata(request=context.request, metadata=session_metadata)
                session.reasoning = stream_reasoning
                session.citations = citations
                if session.request_metadata is not None:
                    session.request_metadata.clear()
                    session.request_metadata.update(context.request.metadata)
                session.metadata = session_metadata
                await self._publish_speculation_result(
                    request=context.request,
                    request_id=context.request_id,
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                )
                await self._publish(
                    EventType.REQUEST_COMPLETED,
                    {"request_id": context.request_id, "model_id": context.manifest.model_id, "finish_reason": "stop"},
                )
                await queue.put(_STREAM_END)
        except Exception as exc:
            execution_seconds = time.perf_counter() - generate_started_at
            per_request_load_seconds = load_seconds / len(contexts)
            per_request_execution_seconds = execution_seconds / len(contexts)
            for context, queue, item_metrics in zip(contexts, queues, frontier_metrics, strict=True):
                self._finalize_request_scheduling_metadata(context.request)
                self.runtime_metrics_recorder.record_failure(
                    model_id=context.manifest.model_id,
                    runtime=runtime.name,
                    capability="streaming",
                    load_seconds=per_request_load_seconds,
                    execution_seconds=per_request_execution_seconds,
                    measurements={
                        **_chat_measurements(context.request.messages),
                        **_request_cache_measurements(request=context.request),
                        **_request_scheduling_measurements(request=context.request),
                        **frontier_architecture_measurements(context.request.metadata),
                        **speculation_measurements(request=context.request),
                        **_continuous_batch_measurements(batch_metrics=item_metrics),
                    },
                )
                self.serving_core.fail_sequence(request_id=context.request_id, error=str(exc))
                self._sync_serving_request_metadata(context.request)
                await self._publish(
                    EventType.REQUEST_FAILED,
                    {"request_id": context.request_id, "model_id": context.manifest.model_id, "error": str(exc)},
                )
                await queue.put(exc)
        finally:
            if load_admission is not None:
                load_admission.release()
            admission.release()
            await self._finalize_runtime_cleanup(
                manifest=manifest,
                runtime=runtime,
                companion_manifests=self._unique_companion_manifests(contexts),
            )

    @staticmethod
    def _unique_companion_manifests(contexts: Sequence[_ChatInvocationContext]) -> tuple[ModelManifest, ...]:
        unique: dict[str, ModelManifest] = {}
        for context in contexts:
            for manifest in context.companion_manifests:
                unique[manifest.model_id] = manifest
        return tuple(unique.values())

    async def _finalize_runtime_cleanup(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        companion_manifests: Sequence[ModelManifest] = (),
    ) -> None:
        policy = self.model_router.settings.runtime_policy
        if policy == "aggressive_unload":
            scheduler_snapshot = self.runtime_request_scheduler.snapshot()
            if scheduler_snapshot["active_requests"] > 0 or scheduler_snapshot["queued_requests"] > 0:
                return
        await self.model_router.runtime_catalog.finalize_runtime_for_request(
            manifest,
            runtime,
            policy=policy,
        )
        if policy == "aggressive_unload":
            for companion_manifest in companion_manifests:
                await runtime.unload_model(companion_manifest.model_id)

    @staticmethod
    def _continuous_batch_payload(batch_metrics: FrontierBatchMetrics | None) -> dict[str, object]:
        if batch_metrics is None:
            return {}
        return {
            "batched": batch_metrics.batch_size > 1,
            "batch_size": batch_metrics.batch_size,
            "batch_position": batch_metrics.batch_position,
            "batch_utilization": batch_metrics.batch_utilization,
            "batch_window_milliseconds": batch_metrics.batch_window_milliseconds,
            "queue_delay_seconds": batch_metrics.queue_delay_seconds,
        }

    def _stream_session_from_queue(
        self,
        *,
        context: _ChatInvocationContext,
        queue: asyncio.Queue[object],
    ) -> ChatStreamSession:
        async def on_close(completed: bool) -> None:
            if completed:
                return
            self.serving_core.request_cancellation(
                request_id=context.request_id,
                reason="stream_consumer_closed",
            )
            self._sync_serving_request_metadata(context.request)

        stream_session = ChatStreamSession(
            request_id=context.request_id,
            created_at=context.created_at,
            model_id=context.manifest.model_id,
            routing=context.routing,
            prompt_trace=context.prompt_trace,
            reasoning_visibility=context.reasoning_visibility,
            request=context.request,
            stream=_empty_stream(),
            stream_items=_empty_item_stream(item_factory=ChatStreamDelta),
            citations=[],
            request_metadata=dict(context.request.metadata),
            serving_profile=ChatOrchestrator._serving_profile_from_metadata(context.request.metadata),
        )

        def set_structured_output(output_text: str) -> None:
            stream_session.structured_output = _structured_output_result(
                stream_session.request,
                stream_session.prompt_trace,
                output_text,
            )

        stream_session.stream_items = _stream_items_with_structured_output(
            _queued_item_stream(
                queue,
                stream_end=_STREAM_END,
                item_type=ChatStreamDelta,
                on_close=on_close,
            ),
            on_completed=set_structured_output,
        )
        stream_session.stream = _content_stream(stream_session.stream_items)
        return stream_session

    @staticmethod
    def _queue_milliseconds(
        *,
        admission: RuntimeRequestAdmission,
        load_admission: RuntimeRequestAdmission | None,
        batch_metrics: FrontierBatchMetrics | None,
    ) -> int:
        total_seconds = admission.wait_seconds
        if load_admission is not None:
            total_seconds += load_admission.wait_seconds
        if batch_metrics is not None:
            total_seconds += batch_metrics.queue_delay_seconds
        return milliseconds_from_seconds(total_seconds)

    def _set_execution_metadata(
        self,
        *,
        request: GenerateRequest,
        request_id: str,
        created_at: int,
        requested_model_id: str | None,
        routing: RoutingDecision,
        admission: RuntimeRequestAdmission,
        load_admission: RuntimeRequestAdmission | None,
        batch_metrics: FrontierBatchMetrics | None,
        load_seconds: float,
        execute_seconds: float,
    ) -> ExecutionMetadata:
        metadata = build_routed_execution_metadata(
            request_id=request_id,
            created=created_at,
            requested_model_id=requested_model_id,
            routing=routing,
            queue_milliseconds=self._queue_milliseconds(
                admission=admission,
                load_admission=load_admission,
                batch_metrics=batch_metrics,
            ),
            load_milliseconds=milliseconds_from_seconds(load_seconds),
            execute_milliseconds=milliseconds_from_seconds(execute_seconds),
            serving=self._execution_serving_metadata(request.metadata.get("serving")),
        )
        request.metadata["execution_metadata"] = metadata.model_dump(mode="json")
        return metadata

    def _refresh_execution_serving_metadata(
        self,
        *,
        request: GenerateRequest,
        metadata: ExecutionMetadata,
    ) -> None:
        metadata.serving = self._execution_serving_metadata(request.metadata.get("serving"))
        request.metadata["execution_metadata"] = metadata.model_dump(mode="json")

    @staticmethod
    def _execution_serving_metadata(payload: object) -> ExecutionServingMetadata | None:
        if not isinstance(payload, dict):
            return None
        batch = payload.get("batch")
        queue_residencies = payload.get("queue_residencies")
        runtime_adapter = payload.get("runtime_adapter")
        queue_count = len(queue_residencies) if isinstance(queue_residencies, list) else 0
        batch_size = 1
        batched = False
        if isinstance(batch, dict):
            batch_size = int(batch.get("batch_size", 1) or 1)
            batched = bool(batch.get("batched", False))
        runtime_adapter_kind = None
        if isinstance(runtime_adapter, dict):
            adapter_kind = runtime_adapter.get("kind")
            runtime_adapter_kind = str(adapter_kind) if adapter_kind is not None else None
        return ExecutionServingMetadata(
            capability=str(payload.get("capability")) if payload.get("capability") is not None else None,
            phase=str(payload.get("phase")) if payload.get("phase") is not None else None,
            streaming=bool(payload.get("streaming", False)),
            streaming_owner=(
                str(payload.get("streaming_owner")) if payload.get("streaming_owner") is not None else None
            ),
            runtime_adapter_kind=runtime_adapter_kind,
            cancellation_requested=bool(payload.get("cancellation_requested", False)),
            queue_residency_milliseconds=int(payload.get("queue_residency_milliseconds", 0) or 0),
            queue_count=queue_count,
            batched=batched,
            batch_size=batch_size,
        )

    @staticmethod
    def _serving_profile_from_metadata(metadata: dict[str, object]) -> ServingProfileApplication | None:
        serving_profile = metadata.get("serving_profile")
        if not isinstance(serving_profile, dict):
            return None
        return ServingProfileApplication.model_validate(serving_profile)

    async def _acquire_model_load_admission(
        self,
        *,
        request: GenerateRequest,
        request_id: str,
        requested_model_id: str | None,
        manifest: ModelManifest,
        runtime: RuntimeContract,
    ) -> RuntimeRequestAdmission | None:
        if runtime.is_model_loaded(manifest.model_id):
            return None
        admission = await self.model_load_scheduler.acquire()
        if admission.was_queued:
            self._record_serving_queue(
                request=request,
                queue_type=ServingQueueType.MODEL_LOAD,
                wait_seconds=admission.wait_seconds,
            )
            await self._publish(
                EventType.REQUEST_QUEUED,
                {
                    "request_id": request_id,
                    "requested_model_id": requested_model_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "wait_seconds": admission.wait_seconds,
                    "queue_type": "model_load",
                },
            )
        return admission

    async def _publish(self, event_type: EventType, payload: dict[str, object]) -> None:
        request_id = payload.get("request_id")
        if isinstance(request_id, str) and request_id:
            if serving_payload := self.serving_core.sequence_metadata(request_id):
                payload = {**payload, "serving": serving_payload}
        await self.event_bus.publish(
            StreamEvent(
                type=event_type,
                scope=EventScope.REQUEST,
                payload={
                    "capability": CapabilityName.CHAT.value,
                    "operation": "text.generation",
                    **payload,
                },
            ),
        )

    async def _publish_progress(
        self,
        *,
        request_id: str,
        operation: str,
        stage: str,
        completed_steps: int,
        total_steps: int,
        reasoning_visibility: ReasoningVisibility,
        reasoning: ReasoningOutput | None = None,
        **payload: object,
    ) -> None:
        progress = round(completed_steps / total_steps, 4) if total_steps else 0.0
        await self._publish(
            EventType.OPERATION_PROGRESS,
            {
                "request_id": request_id,
                "operation": operation,
                "stage": stage,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "progress": progress,
                "reasoning_visibility": reasoning_visibility.value,
                "reasoning_available": reasoning_available(reasoning),
                "reasoning_exposed": reasoning_exposed(reasoning),
                **payload,
            },
        )

    async def _publish_speculation_started(
        self,
        *,
        request: GenerateRequest,
        request_id: str,
        model_id: str,
        runtime: str,
    ) -> None:
        speculation = request.speculation
        if speculation is None:
            return
        await self._publish(
            EventType.SPECULATION_STARTED,
            {
                "request_id": request_id,
                "model_id": model_id,
                "runtime": runtime,
                "mode": speculation.mode.value,
                "draft_model_id": speculation.draft_model_id,
                "companion_model_id": speculation.companion_model_id,
                "num_draft_tokens": speculation.num_draft_tokens,
                "execution_path": request.metadata.get("speculation_execution_path", "pending"),
                "selection_source": request.metadata.get("speculation_selection_source"),
            },
        )

    async def _publish_speculation_result(
        self,
        *,
        request: GenerateRequest,
        request_id: str,
        model_id: str,
        runtime: str,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        speculation = request.speculation
        if speculation is None:
            return
        measurements = speculation_measurements(request=request, usage=usage)
        await self._publish(
            EventType.SPECULATION_ACCEPTED,
            {
                "request_id": request_id,
                "model_id": model_id,
                "runtime": runtime,
                "mode": speculation.mode.value,
                "execution_path": request.metadata.get("speculation_execution_path", "unknown"),
                "drafted_tokens": _chat_coerce_int(measurements.get("drafted_tokens")),
                "accepted_tokens": _chat_coerce_int(measurements.get("accepted_tokens")),
                "verified_tokens": _chat_coerce_int(measurements.get("verified_tokens")),
                "rejected_tokens": _chat_coerce_int(measurements.get("rejected_tokens")),
                "rollback_tokens": _chat_coerce_int(measurements.get("rollback_tokens")),
                "fallback_count": _chat_coerce_int(measurements.get("fallback_count")),
                "acceptance_rate": _chat_coerce_float(measurements.get("acceptance_rate")),
            },
        )

    def _audit_prompt_overrides(
        self,
        *,
        request_id: str,
        requested_model_id: str | None,
        prompt_request: PromptCompilationRequest | None,
        prompt_trace: PromptCompilationTrace,
    ) -> None:
        if prompt_request is None or not prompt_trace.overrides:
            return
        self.audit_logger.record(
            action="prompt_override",
            outcome="applied",
            actor=prompt_request.actor,
            details={
                "request_id": request_id,
                "requested_model_id": requested_model_id,
                "resolved_model_id": prompt_trace.resolved_model_id,
                "selected_template": prompt_trace.selected_template,
                "override_count": len(prompt_trace.overrides),
                "attachment_count": len(prompt_trace.attachment_plan),
                "tool_count": len(prompt_trace.tool_plan),
                "output_contract": prompt_trace.output_contract.model_dump(mode="json", by_alias=True),
                "overrides": [override.model_dump(mode="json") for override in prompt_trace.overrides],
            },
        )
