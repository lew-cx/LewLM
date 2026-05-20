"""First-pass MLX text and semantic runtime adapter."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    CapabilityName,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    GenerateMessage,
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
    SpeculationMode,
    runtime_performance_feature_report,
)
from lewlm.core.errors import ConfigurationError
from lewlm.runtime.base import ManagedTextRuntime
from lewlm.runtime.introspection import invoke_with_signature, resolve_backend_callable
from lewlm.runtime.metal import MLXAccelerationTracker
from lewlm.runtime.paged_kv import PagedKVReservation, PagedKVResidencyManager
from lewlm.runtime.prefix_cache import InMemoryTokenPrefixCache, longest_token_prefix
from lewlm.storage import FrontierExecutionTracker, PersistentPrefixCacheStore
from lewlm.structured_output import StructuredOutputRequest, StructuredOutputRuntimeStatus


_KV_CACHE_CONFIG_PARAMETERS = ("kv_cache_config", "cache_config")
_KV_CACHE_PAGE_SIZE_PARAMETERS = ("kv_cache_page_size", "cache_page_size", "paged_kv_cache_page_size")
_KV_CACHE_MAX_PAGES_PARAMETERS = ("kv_cache_max_pages", "max_kv_cache_pages", "paged_kv_cache_max_pages")
_KV_CACHE_QUANTIZATION_PARAMETERS = ("kv_cache_quantization_bits", "kv_cache_bits", "kv_bits")
_PREFILL_CONFIG_PARAMETERS = ("prefill_config",)
_PREFILL_BATCH_SIZE_PARAMETERS = (
    "prefill_token_batch_size",
    "prefill_chunk_size",
    "prefill_step_size",
    "prefill_batch_size",
)
_PROMPT_TOKEN_LIST_PARAMETERS = ("prompt_tokens", "input_ids", "tokens")
_PROMPT_TOKEN_COUNT_PARAMETERS = ("prompt_token_count", "input_token_count")
_DRAFT_MODEL_PARAMETERS = ("draft_model", "draft", "draft_client")
_NUM_DRAFT_TOKENS_PARAMETERS = ("num_draft_tokens", "draft_tokens")
_FRONTIER_SPECULATION_PARAMETER_ALIASES: dict[SpeculationMode, tuple[str, ...]] = {
    SpeculationMode.MEDUSA: ("medusa", "medusa_model", "medusa_head", "medusa_heads"),
    SpeculationMode.EAGLE: ("eagle", "eagle_model", "eagle_head", "eagle_decoder"),
    SpeculationMode.HYDRA: ("hydra", "hydra_model", "hydra_head", "hydra_heads"),
    SpeculationMode.DFLASH: ("dflash", "dflash_model", "dflash_head", "dflash_decoder"),
    SpeculationMode.SELF_SPECULATIVE: (
        "self_speculative",
        "self_speculation",
        "self_draft",
        "self_draft_model",
    ),
    SpeculationMode.SUFFIX_DECODING: (
        "suffix_decoding",
        "suffix_decoder",
        "suffix_model",
        "suffix_draft",
    ),
    SpeculationMode.HETEROGENEOUS_VOCAB: (
        "heterogeneous_vocab",
        "heterogeneous_draft",
        "heterogeneous_vocabulary",
        "swift",
        "swift_model",
    ),
}


_OWNED_STREAM_END = object()


@dataclass(slots=True)
class _QueuedGenerateRequest:
    request: GenerateRequest
    enqueued_at: float
    future: asyncio.Future[GenerateResponse]


@dataclass(slots=True)
class _ActiveGenerateRequest:
    queued: _QueuedGenerateRequest
    prompt_token_count: int
    kv_reservation: PagedKVReservation
    generated_tokens: list[int]


@dataclass(slots=True)
class _QueuedStreamRequest:
    request: GenerateRequest
    enqueued_at: float
    queue: asyncio.Queue[object]
    completion: asyncio.Future[None]
    cancelled: bool = False


@dataclass(slots=True)
class _ActiveStreamRequest:
    queued: _QueuedStreamRequest
    prompt_token_count: int
    kv_reservation: PagedKVReservation
    generated_tokens: list[int]
    emitted_text: str = ""


class _MLXTextGenerateController:
    def __init__(self, *, runtime: MLXTextRuntime, model_id: str, temperature: float) -> None:
        self.runtime = runtime
        self.model_id = model_id
        self.temperature = temperature
        self._loop = asyncio.get_running_loop()
        self._pending: deque[_QueuedGenerateRequest] = deque()
        self._active: dict[int, _ActiveGenerateRequest] = {}
        self._wake_event = asyncio.Event()
        self._closed = False
        self._task = self._loop.create_task(self._run())
        self._task.add_done_callback(self._consume_task_result)

    @staticmethod
    def _consume_task_result(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        task.exception()

    @property
    def closed(self) -> bool:
        return self._closed or self._loop.is_closed() or self._task.done()

    def close(self) -> None:
        self._closed = True
        self._wake_event.set()
        self._task.cancel()

    async def enqueue(self, request: GenerateRequest) -> GenerateResponse:
        future: asyncio.Future[GenerateResponse] = self._loop.create_future()
        self._pending.append(
            _QueuedGenerateRequest(
                request=request,
                enqueued_at=time.perf_counter(),
                future=future,
            ),
        )
        self._wake_event.set()
        return await future

    async def _run(self) -> None:
        generator = None
        try:
            module = import_module("mlx_lm")
            batch_generator_class = _resolve_mlx_batch_generator_class(module)
            if batch_generator_class is None:
                raise ConfigurationError(
                    "Installed MLX text package does not expose `BatchGenerator` for LewLM-owned continuous batching.",
                )
            model, tokenizer = self.runtime._client_components(self.model_id)
            generator = self.runtime._create_native_batch_generator(
                batch_generator_class=batch_generator_class,
                model=model,
                tokenizer=tokenizer,
                temperature=self.temperature,
            )
            while True:
                await self._wait_for_work()
                if self._closed:
                    return
                self._insert_pending(generator=generator)
                if not self._active:
                    continue
                responses = list(generator.next_generated() or [])
                if not responses:
                    await asyncio.sleep(0)
                    continue
                for response in responses:
                    active_request = self._active.get(int(response.uid))
                    if active_request is None:
                        continue
                    token_value = getattr(response, "token", None)
                    if isinstance(token_value, (int, float)):
                        active_request.generated_tokens.append(int(token_value))
                    finish_reason = getattr(response, "finish_reason", None)
                    if finish_reason is None:
                        continue
                    self._active.pop(int(response.uid), None)
                    if not active_request.queued.future.done():
                        active_request.queued.future.set_result(
                            GenerateResponse(
                                model_id=active_request.queued.request.model_id,
                                output_text=_decode_generated_tokens(
                                    tokenizer=tokenizer,
                                    tokens=active_request.generated_tokens,
                                ),
                                finish_reason=str(finish_reason),
                                usage={
                                    "prompt_tokens": active_request.prompt_token_count,
                                    "completion_tokens": len(active_request.generated_tokens),
                                    "total_tokens": active_request.prompt_token_count + len(active_request.generated_tokens),
                                },
                            ),
                        )
                    self.runtime._paged_kv_manager.release(active_request.kv_reservation)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for queued_request in self._pending:
                if not queued_request.future.done():
                    queued_request.future.set_exception(exc)
            self._pending.clear()
            for active_request in self._active.values():
                if not active_request.queued.future.done():
                    active_request.queued.future.set_exception(exc)
                self.runtime._paged_kv_manager.release(active_request.kv_reservation)
            self._active.clear()
            raise
        finally:
            if generator is not None:
                close = getattr(generator, "close", None)
                if callable(close):
                    close()

    async def _wait_for_work(self) -> None:
        while not self._closed and not self._pending and not self._active:
            self._wake_event.clear()
            await self._wake_event.wait()

    def _insert_pending(self, *, generator: Any) -> None:
        if len(self._active) >= self.runtime.settings.continuous_batch_max_batch_size:
            return
        queued_requests: list[_QueuedGenerateRequest] = []
        capacity = self.runtime.settings.continuous_batch_max_batch_size - len(self._active)
        while self._pending and len(queued_requests) < capacity:
            queued_request = self._pending.popleft()
            if queued_request.future.done():
                continue
            queued_requests.append(queued_request)
        if not queued_requests:
            return
        state = self.runtime._prepare_native_batch_state(
            requests=[queued_request.request for queued_request in queued_requests],
            model_id=self.model_id,
        )
        try:
            uid_order = list(
                generator.insert(
                    prompts=state["prompt_values"],
                    max_tokens=state["max_tokens"],
                    caches=state["prompt_caches"],
                ),
            )
        except Exception:
            self.runtime._release_paged_kv_reservations(state["kv_reservations"])
            raise
        if len(uid_order) != len(queued_requests):
            self.runtime._release_paged_kv_reservations(state["kv_reservations"])
            raise ConfigurationError(
                f"Expected {len(queued_requests)} MLX continuous-batch ids, received {len(uid_order)}.",
            )
        active_batch_size = len(self._active) + len(uid_order)
        self.runtime._native_batch_generate_calls += 1
        self.runtime._native_batch_request_count += len(queued_requests)
        self.runtime._native_batch_max_size = max(self.runtime._native_batch_max_size, active_batch_size)
        inserted_at = time.perf_counter()
        for queued_request, uid, prompt_token_count, feature_usage, kv_reservation in zip(
            queued_requests,
            uid_order,
            state["prompt_token_counts"],
            state["feature_usages"],
            state["kv_reservations"],
            strict=True,
        ):
            self.runtime._record_feature_usage(prompt_token_count=prompt_token_count, feature_usage=feature_usage)
            self.runtime._record_native_batching_metadata(
                request=queued_request.request,
                capability=CapabilityName.CHAT,
                backend="lewlm.mlx_text_continuous_batch_scheduler",
                batch_size=active_batch_size,
                ownership="lewlm_owned",
                backend_primitive="mlx_lm.BatchGenerator",
                queue_delay_seconds=max(inserted_at - queued_request.enqueued_at, 0.0),
                persistent_scheduler=True,
            )
            self._active[int(uid)] = _ActiveGenerateRequest(
                queued=queued_request,
                prompt_token_count=prompt_token_count,
                kv_reservation=kv_reservation,
                generated_tokens=[],
            )


class _MLXTextStreamController:
    def __init__(self, *, runtime: MLXTextRuntime, model_id: str, temperature: float) -> None:
        self.runtime = runtime
        self.model_id = model_id
        self.temperature = temperature
        self._loop = asyncio.get_running_loop()
        self._pending: deque[_QueuedStreamRequest] = deque()
        self._active: dict[int, _ActiveStreamRequest] = {}
        self._wake_event = asyncio.Event()
        self._closed = False
        self._task = self._loop.create_task(self._run())
        self._task.add_done_callback(self._consume_task_result)

    @staticmethod
    def _consume_task_result(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        task.exception()

    @property
    def closed(self) -> bool:
        return self._closed or self._loop.is_closed() or self._task.done()

    def close(self) -> None:
        self._closed = True
        self._wake_event.set()
        self._task.cancel()

    def enqueue(self, request: GenerateRequest) -> _QueuedStreamRequest:
        queued_request = _QueuedStreamRequest(
            request=request,
            enqueued_at=time.perf_counter(),
            queue=asyncio.Queue(),
            completion=self._loop.create_future(),
        )
        self._pending.append(queued_request)
        self._wake_event.set()
        return queued_request

    async def _run(self) -> None:
        generator = None
        try:
            module = import_module("mlx_lm")
            batch_generator_class = _resolve_mlx_batch_generator_class(module)
            if batch_generator_class is None:
                raise ConfigurationError(
                    "Installed MLX text package does not expose `BatchGenerator` for LewLM-owned streaming batching.",
                )
            model, tokenizer = self.runtime._client_components(self.model_id)
            generator = self.runtime._create_native_batch_generator(
                batch_generator_class=batch_generator_class,
                model=model,
                tokenizer=tokenizer,
                temperature=self.temperature,
            )
            while True:
                await self._wait_for_work()
                if self._closed:
                    return
                self._insert_pending(generator=generator)
                if not self._active:
                    continue
                responses = list(generator.next_generated() or [])
                if not responses:
                    await asyncio.sleep(0)
                    continue
                for response in responses:
                    active_request = self._active.get(int(response.uid))
                    if active_request is None:
                        continue
                    token_value = getattr(response, "token", None)
                    if isinstance(token_value, (int, float)):
                        active_request.generated_tokens.append(int(token_value))
                    rendered_text = _decode_generated_tokens(
                        tokenizer=tokenizer,
                        tokens=active_request.generated_tokens,
                    )
                    delta = rendered_text[len(active_request.emitted_text) :]
                    active_request.emitted_text = rendered_text
                    if delta and not active_request.queued.cancelled:
                        await active_request.queued.queue.put(delta)
                    if getattr(response, "finish_reason", None) is None:
                        continue
                    self._active.pop(int(response.uid), None)
                    if not active_request.queued.cancelled:
                        await active_request.queued.queue.put(_OWNED_STREAM_END)
                    if not active_request.queued.completion.done():
                        active_request.queued.completion.set_result(None)
                    self.runtime._paged_kv_manager.release(active_request.kv_reservation)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for queued_request in self._pending:
                if not queued_request.completion.done():
                    queued_request.completion.set_exception(exc)
                if not queued_request.cancelled:
                    await queued_request.queue.put(exc)
                    await queued_request.queue.put(_OWNED_STREAM_END)
            self._pending.clear()
            for active_request in self._active.values():
                if not active_request.queued.completion.done():
                    active_request.queued.completion.set_exception(exc)
                if not active_request.queued.cancelled:
                    await active_request.queued.queue.put(exc)
                    await active_request.queued.queue.put(_OWNED_STREAM_END)
                self.runtime._paged_kv_manager.release(active_request.kv_reservation)
            self._active.clear()
            raise
        finally:
            if generator is not None:
                close = getattr(generator, "close", None)
                if callable(close):
                    close()

    async def _wait_for_work(self) -> None:
        while not self._closed and not self._pending and not self._active:
            self._wake_event.clear()
            await self._wake_event.wait()

    def _insert_pending(self, *, generator: Any) -> None:
        if len(self._active) >= self.runtime.settings.continuous_batch_max_batch_size:
            return
        queued_requests: list[_QueuedStreamRequest] = []
        capacity = self.runtime.settings.continuous_batch_max_batch_size - len(self._active)
        while self._pending and len(queued_requests) < capacity:
            queued_request = self._pending.popleft()
            if queued_request.cancelled:
                continue
            queued_requests.append(queued_request)
        if not queued_requests:
            return
        state = self.runtime._prepare_native_batch_state(
            requests=[queued_request.request for queued_request in queued_requests],
            model_id=self.model_id,
        )
        try:
            uid_order = list(
                generator.insert(
                    prompts=state["prompt_values"],
                    max_tokens=state["max_tokens"],
                    caches=state["prompt_caches"],
                ),
            )
        except Exception:
            self.runtime._release_paged_kv_reservations(state["kv_reservations"])
            raise
        if len(uid_order) != len(queued_requests):
            self.runtime._release_paged_kv_reservations(state["kv_reservations"])
            raise ConfigurationError(
                f"Expected {len(queued_requests)} MLX continuous-batch stream ids, received {len(uid_order)}.",
            )
        active_batch_size = len(self._active) + len(uid_order)
        self.runtime._native_batch_stream_calls += 1
        self.runtime._native_batch_request_count += len(queued_requests)
        self.runtime._native_batch_max_size = max(self.runtime._native_batch_max_size, active_batch_size)
        inserted_at = time.perf_counter()
        for queued_request, uid, prompt_token_count, feature_usage, kv_reservation in zip(
            queued_requests,
            uid_order,
            state["prompt_token_counts"],
            state["feature_usages"],
            state["kv_reservations"],
            strict=True,
        ):
            self.runtime._record_feature_usage(prompt_token_count=prompt_token_count, feature_usage=feature_usage)
            self.runtime._record_native_batching_metadata(
                request=queued_request.request,
                capability=CapabilityName.STREAMING,
                backend="lewlm.mlx_text_continuous_batch_scheduler",
                batch_size=active_batch_size,
                ownership="lewlm_owned",
                backend_primitive="mlx_lm.BatchGenerator",
                queue_delay_seconds=max(inserted_at - queued_request.enqueued_at, 0.0),
                persistent_scheduler=True,
            )
            self._active[int(uid)] = _ActiveStreamRequest(
                queued=queued_request,
                prompt_token_count=prompt_token_count,
                kv_reservation=kv_reservation,
                generated_tokens=[],
            )


class MLXTextRuntime(ManagedTextRuntime):
    """Adapter for MLX-native text, embedding, and rerank models on Apple Silicon."""

    name = "mlx_text"
    affinity = RuntimeAffinity.MLX_TEXT
    supported_formats = (ModelFormat.MLX,)
    supported_modalities = (
        ModelModality.TEXT,
        ModelModality.EMBEDDING,
        ModelModality.RERANK,
        ModelModality.MULTIMODAL,
    )
    supported_capabilities = frozenset(
        {
            CapabilityName.CHAT,
            CapabilityName.STREAMING,
            CapabilityName.EMBEDDINGS,
            CapabilityName.RERANK,
        },
    )
    supported_systems = ("Darwin",)
    supported_machines = ("arm64", "aarch64")
    platform_guidance = "Install the `mlx` extra on Apple Silicon macOS to enable MLX-native text generation."

    def __init__(self, *, settings: LewLMSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or LewLMSettings()
        self._clients: dict[str, tuple[Any, Any | None]] = {}
        self._manifest_support_cache: dict[str, bool] = {}
        self._loaded_feature_usage: dict[str, dict[str, bool]] = {}
        self._loaded_performance_controls: dict[str, dict[str, dict[str, Any]]] = {}
        self._frontier_execution = FrontierExecutionTracker(settings=self.settings)
        self._paged_kv_manager = PagedKVResidencyManager(
            page_size_tokens=self.settings.kv_cache_page_size,
            max_pages=self.settings.kv_cache_max_pages,
        )
        self._prefix_cache = InMemoryTokenPrefixCache(
            page_size_tokens=self.settings.kv_cache_page_size,
            persistent_store=PersistentPrefixCacheStore(
                cache_root=self.settings.cache_dir,
                namespace=self.name,
                page_size_tokens=self.settings.kv_cache_page_size,
            ),
        )
        self._acceleration = MLXAccelerationTracker(
            settings=self.settings,
            runtime_name=self.name,
            import_module_fn=import_module,
        )
        self._paged_kv_request_count = 0
        self._paged_kv_prompt_tokens = 0
        self._quantized_kv_request_count = 0
        self._prefill_optimized_request_count = 0
        self._prefill_prompt_tokens = 0
        self._prefill_batch_count = 0
        self._chunked_prefill_request_count = 0
        self._chunked_prefill_prompt_tokens = 0
        self._chunked_prefill_chunk_count = 0
        self._speculative_request_count = 0
        self._controller_speculative_request_count = 0
        self._backend_passthrough_speculative_request_count = 0
        self._drafted_token_count = 0
        self._accepted_token_count = 0
        self._verified_token_count = 0
        self._rejected_token_count = 0
        self._rollback_token_count = 0
        self._speculation_fallback_count = 0
        self._native_batch_generate_calls = 0
        self._native_batch_stream_calls = 0
        self._native_batch_request_count = 0
        self._native_batch_max_size = 0
        self._owned_generate_controllers: dict[tuple[int, str, str, str], _MLXTextGenerateController] = {}
        self._owned_stream_controllers: dict[tuple[int, str, str, str], _MLXTextStreamController] = {}

    def supports_manifest(self, manifest: ModelManifest) -> bool:
        if not super().supports_manifest(manifest):
            return False
        cache_key = manifest.source_path
        if cache_key in self._manifest_support_cache:
            return self._manifest_support_cache[cache_key]
        supported = _mlx_lm_supports_manifest(manifest.source_path)
        self._manifest_support_cache[cache_key] = supported
        return supported

    def _check_environment(self) -> tuple[bool, str | None]:
        try:
            import_module("mlx_lm")
        except ImportError:
            return False, "mlx-lm is not installed"
        return True, None

    def performance_feature_snapshot(self) -> dict[str, Any]:
        if not self.is_available():
            return {}
        module = import_module("mlx_lm")
        load = resolve_backend_callable(module, ("load", "load_model", "load_pipeline"), required=False)
        generate = resolve_backend_callable(module, ("generate", "chat", "generate_text"), required=False)
        generate_stream = resolve_backend_callable(module, ("generate_stream", "stream_generate"), required=False)
        batch_generate = _resolve_mlx_batch_generate(module)
        batch_generator = _resolve_mlx_batch_generator_class(module)
        all_parameter_names = (
            _callable_parameter_names(load)
            | _callable_parameter_names(generate)
            | _callable_parameter_names(generate_stream)
        )
        generation_parameter_names = _callable_parameter_names(generate) | _callable_parameter_names(generate_stream)
        speculative_modes = self._supported_speculation_modes(generate_stream)
        speculative_supported = bool(speculative_modes)
        paged_kv_supported = bool(
            all_parameter_names
            & set(_KV_CACHE_CONFIG_PARAMETERS + _KV_CACHE_PAGE_SIZE_PARAMETERS + _KV_CACHE_MAX_PAGES_PARAMETERS),
        )
        kv_quantization_supported = bool(
            all_parameter_names
            & set(_KV_CACHE_CONFIG_PARAMETERS + _KV_CACHE_QUANTIZATION_PARAMETERS),
        )
        paged_kv_snapshot = self._paged_kv_manager.snapshot()
        prefill_supported = bool(
            generation_parameter_names
            & set(
                _PREFILL_CONFIG_PARAMETERS
                + _PREFILL_BATCH_SIZE_PARAMETERS
                + _PROMPT_TOKEN_LIST_PARAMETERS
                + _PROMPT_TOKEN_COUNT_PARAMETERS
            ),
        )
        prefix_cache_snapshot = (
            self._prefix_cache.snapshot()
            if self._supports_prefix_cache(module=module)
            else {
                "supported": False,
                "active": False,
                "page_size_tokens": self.settings.kv_cache_page_size,
                "cache_entries": 0,
                "cache_size_bytes": 0,
                "resident_cache_entries": 0,
                "resident_cache_hits": 0,
                "resident_page_count": 0,
                "resident_page_size_bytes": 0,
                "persisted_cache_entries": 0,
                "persisted_cache_size_bytes": 0,
                "persisted_page_count": 0,
                "persisted_page_size_bytes": 0,
                "persistent_cache_hits": 0,
                "page_hits": 0,
                "resident_page_hits": 0,
                "persistent_page_hits": 0,
                "page_saves": 0,
                "page_restores": 0,
                "cache_restores": 0,
                "page_evictions": 0,
                "cache_evictions": 0,
                "cache_invalidations": 0,
                "copy_on_write_reused_pages": 0,
                "cached_tokens": 0,
                "restart_resilient": False,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_saves": 0,
                "saved_prefill_tokens": 0,
                "max_saved_prefill_tokens": 0,
            }
        )
        continuous_batching_ownership = self.continuous_batching_ownership(CapabilityName.CHAT)
        feature_snapshot = {
            "continuous_batching": runtime_performance_feature_report(
                ownership=continuous_batching_ownership,
                active=self._native_batch_request_count > 0,
                reason=(
                    "LewLM owns the persistent continuous-batch scheduler on the primary MLX text path and drives MLX `BatchGenerator` as the decode primitive."
                    if continuous_batching_ownership == PerformanceFeatureOwnership.LEWLM_OWNED.value
                    else "Installed MLX text generation exposes backend-native continuous batching through `batch_generate`."
                    if continuous_batching_ownership == PerformanceFeatureOwnership.BACKEND_NATIVE.value
                    else "The current MLX text adapter does not detect an explicit batched chat or streaming entrypoint."
                ),
                notes=(
                    [
                        "LewLM keeps a persistent per-model scheduler alive for non-speculative chat and streaming requests, then inserts them into MLX `BatchGenerator` as capacity opens.",
                        "Other runtimes still report backend-native or unsupported continuous batching explicitly; LewLM does not claim ownership outside the primary MLX text path.",
                    ]
                    if continuous_batching_ownership == PerformanceFeatureOwnership.LEWLM_OWNED.value
                    else [
                        "LewLM routes same-model chat bursts through MLX's native `batch_generate` entrypoint when `BatchGenerator` is unavailable.",
                    ]
                    if continuous_batching_ownership == PerformanceFeatureOwnership.BACKEND_NATIVE.value
                    else [
                        "LewLM only activates backend-native continuous batching when the selected runtime exposes native batched generation hooks.",
                    ]
                ),
                metrics=_compact_runtime_metrics(
                    chat_batch_calls=self._native_batch_generate_calls,
                    stream_batch_calls=self._native_batch_stream_calls,
                    batched_requests=self._native_batch_request_count,
                    max_batch_size=self._native_batch_max_size,
                ),
            ),
            "prefix_cache": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.LEWLM_OWNED
                    if prefix_cache_snapshot["supported"]
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=bool(prefix_cache_snapshot["active"]),
                reason=(
                    "MLX text generation can reuse paged radix-style prompt-prefix KV caches by prefilling uncached suffix tokens only."
                    if prefix_cache_snapshot["supported"]
                    else "Installed MLX text entrypoints do not expose reusable prompt-cache primitives."
                ),
                notes=[
                    "LewLM keeps a hot resident page tier in RAM, reuses radix-style shared prefix pages with copy-on-write suffix extension, and reports per-page reuse directly.",
                ]
                if prefix_cache_snapshot["supported"]
                else [],
                metrics=_compact_runtime_metrics(
                    page_size_tokens=int(prefix_cache_snapshot["page_size_tokens"]),
                    cache_entries=int(prefix_cache_snapshot["cache_entries"]),
                    cache_hits=int(prefix_cache_snapshot["cache_hits"]),
                    cache_misses=int(prefix_cache_snapshot["cache_misses"]),
                    cache_saves=int(prefix_cache_snapshot["cache_saves"]),
                    resident_cache_entries=int(prefix_cache_snapshot["resident_cache_entries"]),
                    resident_cache_hits=int(prefix_cache_snapshot["resident_cache_hits"]),
                    resident_page_count=int(prefix_cache_snapshot["resident_page_count"]),
                    resident_page_size_bytes=int(prefix_cache_snapshot["resident_page_size_bytes"]),
                    page_hits=int(prefix_cache_snapshot["page_hits"]),
                    resident_page_hits=int(prefix_cache_snapshot["resident_page_hits"]),
                    page_saves=int(prefix_cache_snapshot["page_saves"]),
                    copy_on_write_reused_pages=int(prefix_cache_snapshot["copy_on_write_reused_pages"]),
                    persisted_cache_entries=int(prefix_cache_snapshot["persisted_cache_entries"]),
                    persisted_cache_size_bytes=int(prefix_cache_snapshot["persisted_cache_size_bytes"]),
                    persistent_cache_hits=int(prefix_cache_snapshot["persistent_cache_hits"]),
                    cache_restores=int(prefix_cache_snapshot["cache_restores"]),
                    cache_evictions=int(prefix_cache_snapshot["cache_evictions"]),
                    cache_invalidations=int(prefix_cache_snapshot["cache_invalidations"]),
                    cached_tokens=int(prefix_cache_snapshot["cached_tokens"]),
                    saved_prefill_tokens=int(prefix_cache_snapshot["saved_prefill_tokens"]),
                    max_saved_prefill_tokens=int(prefix_cache_snapshot["max_saved_prefill_tokens"]),
                ),
            ),
            "persistent_multi_context_cache": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.LEWLM_OWNED
                    if prefix_cache_snapshot["restart_resilient"]
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=bool(prefix_cache_snapshot["persisted_cache_entries"])
                or int(prefix_cache_snapshot["persistent_cache_hits"]) > 0,
                reason=(
                    "LewLM persists content-addressed cold-tier MLX prompt-prefix pages on disk and restores JSON-safe "
                    "prompt-cache payloads into the resident hot set across restarts."
                ),
                notes=[
                    "Resident entries use LRU eviction while persisted radix pages stay content-addressed under the local cache root.",
                    "When MLX prompt-cache payloads cannot be serialized safely, LewLM preserves cache-key metadata and reports the limitation explicitly.",
                ],
                metrics=_compact_runtime_metrics(
                    page_size_tokens=int(prefix_cache_snapshot["page_size_tokens"]),
                    resident_cache_entries=int(prefix_cache_snapshot["resident_cache_entries"]),
                    persisted_cache_entries=int(prefix_cache_snapshot["persisted_cache_entries"]),
                    persisted_cache_size_bytes=int(prefix_cache_snapshot["persisted_cache_size_bytes"]),
                    resident_cache_hits=int(prefix_cache_snapshot["resident_cache_hits"]),
                    persistent_cache_hits=int(prefix_cache_snapshot["persistent_cache_hits"]),
                    resident_page_count=int(prefix_cache_snapshot["resident_page_count"]),
                    resident_page_size_bytes=int(prefix_cache_snapshot["resident_page_size_bytes"]),
                    persisted_page_count=int(prefix_cache_snapshot["persisted_page_count"]),
                    persisted_page_size_bytes=int(prefix_cache_snapshot["persisted_page_size_bytes"]),
                    persistent_page_hits=int(prefix_cache_snapshot["persistent_page_hits"]),
                    cache_restores=int(prefix_cache_snapshot["cache_restores"]),
                    page_restores=int(prefix_cache_snapshot["page_restores"]),
                    cache_evictions=int(prefix_cache_snapshot["cache_evictions"]),
                    cache_invalidations=int(prefix_cache_snapshot["cache_invalidations"]),
                    page_evictions=int(prefix_cache_snapshot["page_evictions"]),
                    cached_tokens=int(prefix_cache_snapshot["cached_tokens"]),
                ),
            ),
            "speculative_decoding": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.LEWLM_OWNED
                    if speculative_supported
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=speculative_supported and self._speculative_request_count > 0,
                modes=[mode.value for mode in speculative_modes],
                reason=(
                    "LewLM owns the MLX draft-model draft/verify loop on the first-class path and uses explicit passthrough only for frontier adapters."
                    if speculative_supported
                    else "Installed MLX streaming generation entrypoints do not expose compatible speculative decoding hooks."
                ),
                notes=(
                    []
                    if self.settings.speculative_decoding_enabled
                    else [
                        "Enable `LEWLM_SPECULATIVE_DECODING_ENABLED=true` to allow MLX draft-model and compatible frontier speculation.",
                    ]
                ),
                metrics=_compact_runtime_metrics(
                    request_count=self._speculative_request_count,
                    controller_owned_requests=self._controller_speculative_request_count,
                    backend_passthrough_requests=self._backend_passthrough_speculative_request_count,
                    configured_num_draft_tokens=self.settings.speculative_decoding_num_draft_tokens,
                    drafted_tokens=self._drafted_token_count,
                    accepted_tokens=self._accepted_token_count,
                    verified_tokens=self._verified_token_count,
                    rejected_tokens=self._rejected_token_count,
                    rollback_tokens=self._rollback_token_count,
                    fallback_count=self._speculation_fallback_count,
                ),
            ),
            "constrained_decoding": runtime_performance_feature_report(
                ownership=PerformanceFeatureOwnership.PARTIAL,
                active=False,
                modes=["prompt_guided"],
                reason=(
                    "LewLM preserves structured-output requests on MLX text through prompt-guided validation fallback, "
                    "but the installed backend does not expose portable decode-time constrained decoding hooks."
                ),
                metrics={
                    "decoder_enforced": False,
                    "fallback_used": True,
                    "enforcement": "prompt_guided",
                },
                notes=[
                    "MLX text keeps the structured-output contract visible without over-claiming decode-time enforcement parity.",
                ],
            ),
            "paged_kv_cache": runtime_performance_feature_report(
                ownership=PerformanceFeatureOwnership.LEWLM_OWNED,
                active=self._paged_kv_request_count > 0,
                reason=(
                    "LewLM tracks first-class MLX text paged-KV residency with page reuse, eviction, and lane-aware pressure control."
                    if paged_kv_supported
                    else "LewLM tracks first-class MLX text paged-KV residency even when the installed MLX backend does not expose explicit native page controls."
                ),
                notes=[
                    "Runtime-native paged-KV knobs remain backend-dependent; the residency snapshot below reports the allocator state LewLM owns directly.",
                ],
                metrics=_compact_runtime_metrics(
                    page_size_tokens=int(paged_kv_snapshot["page_size_tokens"]),
                    max_pages=(
                        int(paged_kv_snapshot["max_pages"])
                        if "max_pages" in paged_kv_snapshot
                        else self.settings.kv_cache_max_pages
                    ),
                    native_control_supported=paged_kv_supported,
                    requests_using_paged_kv=self._paged_kv_request_count,
                    paged_prompt_tokens=self._paged_kv_prompt_tokens,
                    resident_pages=int(paged_kv_snapshot["resident_pages"]),
                    active_pages=int(paged_kv_snapshot["active_pages"]),
                    active_decode_pages=int(paged_kv_snapshot["active_decode_pages"]),
                    active_prefill_pages=int(paged_kv_snapshot["active_prefill_pages"]),
                    resident_decode_pages=int(paged_kv_snapshot["resident_decode_pages"]),
                    resident_prefill_pages=int(paged_kv_snapshot["resident_prefill_pages"]),
                    decode_lane_reservations=int(paged_kv_snapshot["decode_lane_reservations"]),
                    prefill_lane_reservations=int(paged_kv_snapshot["prefill_lane_reservations"]),
                    reused_pages=int(paged_kv_snapshot["reused_pages"]),
                    new_pages=int(paged_kv_snapshot["new_pages"]),
                    evicted_pages=int(paged_kv_snapshot["evicted_pages"]),
                    prefill_evicted_pages=int(paged_kv_snapshot["prefill_evicted_pages"]),
                    decode_evicted_pages=int(paged_kv_snapshot["decode_evicted_pages"]),
                    decode_headroom_preservation_events=int(
                        paged_kv_snapshot["decode_headroom_preservation_events"],
                    ),
                    prefill_decode_tradeoff_events=int(
                        paged_kv_snapshot["prefill_decode_tradeoff_events"],
                    ),
                    overflow_events=int(paged_kv_snapshot["overflow_events"]),
                    overflow_pages=int(paged_kv_snapshot["overflow_pages"]),
                    high_pressure_events=int(paged_kv_snapshot["high_pressure_events"]),
                    peak_resident_pages=int(paged_kv_snapshot["peak_resident_pages"]),
                    peak_total_pages=int(paged_kv_snapshot["peak_total_pages"]),
                    pressure_ratio=float(paged_kv_snapshot["pressure_ratio"]),
                    peak_pressure_ratio=float(paged_kv_snapshot["peak_pressure_ratio"]),
                    pressure_level=str(paged_kv_snapshot["pressure_level"]),
                ),
            ),
            "kv_cache_quantization": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.BACKEND_NATIVE
                    if kv_quantization_supported
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=kv_quantization_supported and self._quantized_kv_request_count > 0,
                reason=(
                    "MLX text generation exposes explicit KV-cache quantization controls on the installed backend entrypoints."
                    if kv_quantization_supported
                    else "Installed MLX text entrypoints do not expose explicit KV-cache quantization controls."
                ),
                notes=[
                    "LewLM applies KV-cache quantization separately from model-weight selection when the backend exposes a dedicated hook.",
                ],
                metrics=_compact_runtime_metrics(
                    quantization_bits=self.settings.kv_cache_quantization_bits,
                    requests_using_quantized_kv=self._quantized_kv_request_count,
                ),
            ),
            "prefill_optimization": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.BACKEND_NATIVE
                    if prefill_supported
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=prefill_supported and self._prefill_optimized_request_count > 0,
                reason=(
                    "MLX text generation exposes explicit prefill batch sizing or tokenized-prompt controls on the installed backend entrypoints."
                    if prefill_supported
                    else "Installed MLX text entrypoints do not expose explicit prefill batch sizing or tokenized-prompt controls."
                ),
                notes=[
                    "Prefill optimization stays local to the runtime request lifecycle; it does not add prefix-cache reuse or speculative decoding.",
                ],
                metrics=_compact_runtime_metrics(
                    prefill_token_batch_size=self.settings.prefill_token_batch_size,
                    optimized_requests=self._prefill_optimized_request_count,
                    optimized_prompt_tokens=self._prefill_prompt_tokens,
                    prefill_batches_planned=self._prefill_batch_count,
                ),
            ),
            "chunked_prefill": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.LEWLM_OWNED
                    if prefill_supported
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=prefill_supported and self._chunked_prefill_request_count > 0,
                reason=(
                    "MLX text generation can split long prompt ingest into bounded token chunks on the installed backend entrypoints."
                    if prefill_supported
                    else "Installed MLX text entrypoints do not expose chunked prefill or prompt-token controls."
                ),
                notes=[
                    "Chunked prefill only becomes active when a prompt estimate exceeds the configured prefill token batch size.",
                ],
                metrics=_compact_runtime_metrics(
                    prefill_token_batch_size=self.settings.prefill_token_batch_size,
                    chunked_requests=self._chunked_prefill_request_count,
                    chunked_prompt_tokens=self._chunked_prefill_prompt_tokens,
                    chunk_count=self._chunked_prefill_chunk_count,
                ),
            ),
            "prefill_isolation": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.LEWLM_OWNED
                    if self.supports_prefill_isolation(CapabilityName.CHAT)
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=self.supports_prefill_isolation(CapabilityName.CHAT) and self.settings.prefill_isolation_enabled,
                reason=(
                    "MLX text can combine chunked prefill with continuous batching so the scheduler may reserve decode headroom while long-prefill requests are active."
                    if self.supports_prefill_isolation(CapabilityName.CHAT)
                    else "Installed MLX text entrypoints do not currently expose the combined chunked-prefill and continuous-batching hooks needed for truthful single-host prefill isolation."
                ),
                notes=[
                    "Isolation is scheduler-driven and only becomes active when LewLM enables prefill isolation for long prompts.",
                ],
                metrics=_compact_runtime_metrics(
                    prefill_token_batch_size=self.settings.prefill_token_batch_size,
                    chunked_requests=self._chunked_prefill_request_count,
                    chunk_count=self._chunked_prefill_chunk_count,
                ),
            ),
        }
        feature_snapshot.update(
            self._frontier_execution.performance_feature_snapshot(),
        )
        feature_snapshot.update(
            self._acceleration.performance_feature_snapshot(callables=(generate, generate_stream)),
        )
        return feature_snapshot

    def supports_capability(self, capability: CapabilityName) -> bool:
        if not self.is_available():
            return False
        module = import_module("mlx_lm")
        if capability == CapabilityName.CHAT:
            return resolve_backend_callable(module, ("generate", "chat", "generate_text"), required=False) is not None
        if capability == CapabilityName.STREAMING:
            return resolve_backend_callable(module, ("generate_stream", "stream_generate"), required=False) is not None
        if capability == CapabilityName.EMBEDDINGS:
            return _resolve_semantic_callable(module, capability) is not None
        if capability == CapabilityName.RERANK:
            return _resolve_semantic_callable(module, capability) is not None
        return False

    def supports_continuous_batching(self, capability: CapabilityName) -> bool:
        if not self.is_available():
            return False
        module = import_module("mlx_lm")
        if capability == CapabilityName.CHAT:
            return _resolve_mlx_batch_generator_class(module) is not None or _resolve_mlx_batch_generate(module) is not None
        if capability == CapabilityName.STREAMING:
            return _resolve_mlx_batch_generator_class(module) is not None
        return False

    def continuous_batching_ownership(self, capability: CapabilityName) -> str:
        if not self.is_available():
            return "unsupported"
        module = import_module("mlx_lm")
        batch_generator_class = _resolve_mlx_batch_generator_class(module)
        batch_generate = _resolve_mlx_batch_generate(module)
        if capability in {CapabilityName.CHAT, CapabilityName.STREAMING} and batch_generator_class is not None:
            return "lewlm_owned"
        if capability == CapabilityName.CHAT and batch_generate is not None:
            return "backend_native"
        return "unsupported"

    def supports_chunked_prefill(self, capability: CapabilityName) -> bool:
        if capability not in {CapabilityName.CHAT, CapabilityName.STREAMING} or not self.is_available():
            return False
        module = import_module("mlx_lm")
        generate = resolve_backend_callable(module, ("generate", "chat", "generate_text"), required=False)
        generate_stream = resolve_backend_callable(module, ("generate_stream", "stream_generate"), required=False)
        generation_parameter_names = _callable_parameter_names(generate) | _callable_parameter_names(generate_stream)
        return bool(
            generation_parameter_names
            & set(
                _PREFILL_CONFIG_PARAMETERS
                + _PREFILL_BATCH_SIZE_PARAMETERS
                + _PROMPT_TOKEN_LIST_PARAMETERS
                + _PROMPT_TOKEN_COUNT_PARAMETERS
            ),
        )

    def supports_prefill_isolation(self, capability: CapabilityName) -> bool:
        return self.supports_chunked_prefill(capability) and self.supports_continuous_batching(capability)

    async def _load_model(self, manifest: ModelManifest) -> None:
        module = import_module("mlx_lm")
        load = resolve_backend_callable(module, ("load", "load_model", "load_pipeline"))
        load_options, loaded_feature_usage, loaded_controls = self._kv_cache_load_options(load)
        client = invoke_with_signature(
            load,
            {
                "path_or_hf_repo": manifest.source_path,
                "path": manifest.source_path,
                "model_path": manifest.source_path,
                "source_path": manifest.source_path,
                **load_options,
            },
            capability="model_load",
        )
        self._clients[manifest.model_id] = _normalize_loaded_client(client)
        self._loaded_feature_usage[manifest.model_id] = {**loaded_feature_usage, "prefill_optimization": False}
        self._loaded_performance_controls[manifest.model_id] = loaded_controls
        self._frontier_execution.register_manifest(manifest)

    async def _unload_model(self, model_id: str) -> None:
        self._close_owned_continuous_batch_controllers(model_id=model_id)
        self._clients.pop(model_id, None)
        self._loaded_feature_usage.pop(model_id, None)
        self._loaded_performance_controls.pop(model_id, None)
        self._prefix_cache.invalidate(model_id=model_id)
        self._frontier_execution.unregister_model(model_id)
        self._paged_kv_manager.unregister_model(model_id)

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        self._record_frontier_execution(request)
        self._record_structured_output_runtime(request)
        if self._should_use_owned_continuous_batching(request=request, capability=CapabilityName.CHAT):
            return await self._generate_controller(request=request).enqueue(request)
        if request.speculation is not None:
            return await self._generate_with_speculation(request)
        module = import_module("mlx_lm")
        generate = resolve_backend_callable(module, ("generate", "chat", "generate_text"))
        model, tokenizer = self._client_components(request.model_id)
        prompt = _messages_to_prompt(request.messages, tokenizer)
        prompt_tokens = _prompt_token_ids(tokenizer, prompt)
        kv_reservation = self._reserve_paged_kv_residency(request=request, prompt_tokens=prompt_tokens)
        try:
            prompt_value, prefix_cache_payload = self._prepare_prompt_cache_invocation(
                module=module,
                model=model,
                tokenizer=tokenizer,
                request=request,
                prompt_tokens=prompt_tokens,
            )
            generation_options = _mlx_text_generation_options(request.temperature)
            performance_options, feature_usage, generation_controls = self._generation_performance_options(
                generate,
                prompt_tokens=prompt_tokens,
            )
            feature_usage["paged_kv_cache"] = True
            feature_usage = self._merge_loaded_feature_usage(request.model_id, feature_usage)
            request.metadata["performance_controls"] = self._request_performance_controls(
                model_id=request.model_id,
                generation_controls=generation_controls,
            )
            response = self._invoke_generate_response(
                request=request,
                generate=generate,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_value,
                max_tokens=request.max_tokens,
                prompt_cache=prefix_cache_payload,
                generation_options=generation_options,
                performance_options=performance_options,
            )
            self._record_feature_usage(prompt_token_count=len(prompt_tokens), feature_usage=feature_usage)
            return response
        finally:
            self._paged_kv_manager.release(kv_reservation)

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        self._record_frontier_execution(request)
        self._record_structured_output_runtime(request)
        if self._should_use_owned_continuous_batching(request=request, capability=CapabilityName.STREAMING):
            queued_request = self._stream_controller(request=request).enqueue(request)
            try:
                while True:
                    chunk = await queued_request.queue.get()
                    if chunk is _OWNED_STREAM_END:
                        break
                    if isinstance(chunk, BaseException):
                        raise chunk
                    if isinstance(chunk, str) and chunk:
                        yield chunk
            finally:
                if not queued_request.completion.done():
                    queued_request.cancelled = True
            return
        if request.speculation is not None:
            async for chunk in self._stream_generate_with_speculation(request):
                yield chunk
            return
        module = import_module("mlx_lm")
        generate_stream = resolve_backend_callable(module, ("generate_stream", "stream_generate"))
        model, tokenizer = self._client_components(request.model_id)
        prompt = _messages_to_prompt(request.messages, tokenizer)
        prompt_tokens = _prompt_token_ids(tokenizer, prompt)
        kv_reservation = self._reserve_paged_kv_residency(request=request, prompt_tokens=prompt_tokens)
        try:
            prompt_value, prefix_cache_payload = self._prepare_prompt_cache_invocation(
                module=module,
                model=model,
                tokenizer=tokenizer,
                request=request,
                prompt_tokens=prompt_tokens,
            )
            generation_options = _mlx_text_generation_options(request.temperature)
            performance_options, feature_usage, generation_controls = self._generation_performance_options(
                generate_stream,
                prompt_tokens=prompt_tokens,
            )
            feature_usage["paged_kv_cache"] = True
            feature_usage = self._merge_loaded_feature_usage(request.model_id, feature_usage)
            request.metadata["performance_controls"] = self._request_performance_controls(
                model_id=request.model_id,
                generation_controls=generation_controls,
            )
            chunks = self._acceleration.invoke(
                request=request,
                callable_obj=generate_stream,
                callable_key="generate_stream",
                provided_values={
                    "client": {"model": model, "tokenizer": tokenizer},
                    "model": model,
                    "tokenizer": tokenizer,
                    "prompt": prompt_value,
                    "messages": [{"role": message.role, "content": message.content} for message in request.messages],
                    "max_tokens": request.max_tokens,
                    "verbose": False,
                    "prompt_cache": prefix_cache_payload,
                    **generation_options,
                    **performance_options,
                },
                capability=CapabilityName.STREAMING.value,
                passthrough_keys=("sampler", "temperature", "temp"),
            )
            self._record_feature_usage(prompt_token_count=len(prompt_tokens), feature_usage=feature_usage)
            for chunk in chunks:
                text = _mlx_chunk_to_text(chunk)
                if text:
                    yield text
        finally:
            self._paged_kv_manager.release(kv_reservation)

    def _invoke_generate_response(
        self,
        *,
        request: GenerateRequest,
        generate: Any,
        model: Any,
        tokenizer: Any | None,
        prompt: list[int] | str,
        max_tokens: int,
        prompt_cache: Any | None = None,
        generation_options: dict[str, Any] | None = None,
        performance_options: dict[str, Any] | None = None,
        callable_key: str = "generate",
        phase: str = "decode",
    ) -> GenerateResponse:
        result = self._acceleration.invoke(
            request=request,
            callable_obj=generate,
            callable_key=callable_key,
            provided_values={
                "client": {"model": model, "tokenizer": tokenizer},
                "model": model,
                "tokenizer": tokenizer,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "verbose": False,
                "prompt_cache": prompt_cache,
                **(generation_options or {}),
                **(performance_options or {}),
            },
            capability=CapabilityName.CHAT.value,
            passthrough_keys=("max_tokens", "sampler", "temperature", "temp"),
            phase=phase,
        )
        return _generate_response_from_result(result=result, model_id=request.model_id)

    def _record_structured_output_runtime(self, request: GenerateRequest) -> None:
        status = self.structured_output_runtime_status(request.structured_output)
        if status is None:
            return
        request.metadata["structured_output_runtime"] = status.model_dump(mode="json")

    def structured_output_runtime_status(
        self,
        contract: StructuredOutputRequest | None,
    ) -> StructuredOutputRuntimeStatus | None:
        if contract is None or contract.type == "text":
            return None
        fallback_reason = (
            "MLX text does not expose decode-time constrained decoding for JSON-schema output; "
            "LewLM records the contract and validates generated JSON after generation."
            if contract.type == "json_schema"
            else "MLX text does not expose grammar-based decode-time constrained decoding."
        )
        return StructuredOutputRuntimeStatus(
            runtime=self.name,
            mode=contract.type,
            enforcement="prompt_guided",
            decoder_enforced=False,
            fallback_used=True,
            fallback_reason=fallback_reason,
        )

    async def generate_batch(self, requests: Sequence[GenerateRequest]) -> list[GenerateResponse]:
        self._ensure_available()
        if not requests:
            return []
        for request in requests:
            self._record_structured_output_runtime(request)
        self._record_frontier_batch_requests(requests)
        model_id = self._validate_batch_requests(requests)
        module = import_module("mlx_lm")
        batch_generator_class = _resolve_mlx_batch_generator_class(module)
        if batch_generator_class is not None:
            return await self._generate_batch_with_batch_generator(
                batch_generator_class=batch_generator_class,
                requests=requests,
                model_id=model_id,
            )
        batch_generate = _resolve_mlx_batch_generate(module)
        if batch_generate is None:
            raise ConfigurationError("Installed MLX text package does not expose a native batched chat entrypoint.")
        return await self._generate_batch_with_batch_generate(
            batch_generate=batch_generate,
            requests=requests,
            model_id=model_id,
        )

    async def stream_generate_batch(self, requests: Sequence[GenerateRequest]) -> AsyncIterator[tuple[int, str]]:
        self._ensure_available()
        if not requests:
            return
        for request in requests:
            self._record_structured_output_runtime(request)
        self._record_frontier_batch_requests(requests)
        model_id = self._validate_batch_requests(requests)
        module = import_module("mlx_lm")
        batch_generator_class = _resolve_mlx_batch_generator_class(module)
        if batch_generator_class is None:
            raise ConfigurationError("Installed MLX text package does not expose a native batched streaming entrypoint.")
        stream_state = self._prepare_native_batch_state(requests=requests, model_id=model_id)
        generator = self._create_native_batch_generator(
            batch_generator_class=batch_generator_class,
            model=stream_state["model"],
            tokenizer=stream_state["tokenizer"],
            temperature=requests[0].temperature,
        )
        try:
            uid_order = list(
                generator.insert(
                    prompts=stream_state["prompt_values"],
                    max_tokens=stream_state["max_tokens"],
                    caches=stream_state["prompt_caches"],
                ),
            )
            uid_to_index = {uid: index for index, uid in enumerate(uid_order)}
            decoded_tokens: list[list[int]] = [[] for _ in requests]
            emitted_text: list[str] = ["" for _ in requests]
            for prompt_token_count, feature_usage in zip(
                stream_state["prompt_token_counts"],
                stream_state["feature_usages"],
                strict=True,
            ):
                self._record_feature_usage(prompt_token_count=prompt_token_count, feature_usage=feature_usage)
            self._native_batch_stream_calls += 1
            self._native_batch_request_count += len(requests)
            self._native_batch_max_size = max(self._native_batch_max_size, len(requests))
            for request in requests:
                self._record_native_batching_metadata(
                    request=request,
                    capability=CapabilityName.STREAMING,
                    backend="mlx_lm.BatchGenerator",
                    batch_size=len(requests),
                )
            while responses := generator.next_generated():
                for response in responses:
                    request_index = uid_to_index.get(response.uid)
                    if request_index is None:
                        continue
                    token_value = getattr(response, "token", None)
                    if isinstance(token_value, (int, float)):
                        decoded_tokens[request_index].append(int(token_value))
                    rendered_text = _decode_generated_tokens(
                        tokenizer=stream_state["tokenizer"],
                        tokens=decoded_tokens[request_index],
                    )
                    delta = rendered_text[len(emitted_text[request_index]) :]
                    emitted_text[request_index] = rendered_text
                    if delta:
                        yield request_index, delta
                await asyncio.sleep(0)
        finally:
            close = getattr(generator, "close", None)
            if callable(close):
                close()
            self._release_paged_kv_reservations(stream_state["kv_reservations"])

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        module = import_module("mlx_lm")
        embed = _resolve_semantic_callable(module, CapabilityName.EMBEDDINGS)
        model, tokenizer = self._client_components(request.model_id)
        result = invoke_with_signature(
            embed,
            {
                "client": {"model": model, "tokenizer": tokenizer},
                "model": model,
                "tokenizer": tokenizer,
                "inputs": request.inputs,
                "texts": request.inputs,
                "sentences": request.inputs,
                "documents": request.inputs,
            },
            capability=CapabilityName.EMBEDDINGS.value,
        )
        return _embedding_response_from_result(result, request)

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        module = import_module("mlx_lm")
        rerank = _resolve_semantic_callable(module, CapabilityName.RERANK)
        model, tokenizer = self._client_components(request.model_id)
        result = invoke_with_signature(
            rerank,
            {
                "client": {"model": model, "tokenizer": tokenizer},
                "model": model,
                "tokenizer": tokenizer,
                "query": request.query,
                "documents": request.documents,
                "inputs": request.documents,
                "texts": request.documents,
                "top_n": request.top_n,
            },
            capability=CapabilityName.RERANK.value,
        )
        return _rerank_response_from_result(result, request)

    def _tokenize(self, text: str) -> list[int]:
        _, tokenizer = self._client_components(next(iter(self._clients)))
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            return list(tokenizer.encode(text))
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens: Sequence[int]) -> str:
        _, tokenizer = self._client_components(next(iter(self._clients)))
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return str(tokenizer.decode(list(tokens)))
        return bytes(tokens).decode("utf-8")

    def _client_components(self, model_id: str) -> tuple[Any, Any | None]:
        return self._clients[model_id]

    def _validate_batch_requests(self, requests: Sequence[GenerateRequest]) -> str:
        model_ids = {request.model_id for request in requests}
        if len(model_ids) != 1:
            raise ConfigurationError(
                "MLX native batching requires all requests in the batch to target the same loaded model.",
            )
        if any(request.speculation is not None for request in requests):
            raise ConfigurationError("MLX native continuous batching does not combine speculative chat requests.")
        model_id = next(iter(model_ids))
        self._ensure_loaded(model_id)
        self._touch_model(model_id)
        return model_id

    def _prepare_native_batch_state(
        self,
        *,
        requests: Sequence[GenerateRequest],
        model_id: str,
    ) -> dict[str, Any]:
        module = import_module("mlx_lm")
        model, tokenizer = self._client_components(model_id)
        prompt_values: list[list[int]] = []
        prompt_caches: list[Any | None] = []
        prompt_token_counts: list[int] = []
        max_tokens: list[int] = []
        feature_usages: list[dict[str, bool]] = []
        kv_reservations: list[PagedKVReservation] = []
        for request in requests:
            prompt = _messages_to_prompt(request.messages, tokenizer)
            prompt_tokens = _prompt_token_ids(tokenizer, prompt)
            kv_reservation = self._reserve_paged_kv_residency(request=request, prompt_tokens=prompt_tokens)
            kv_reservations.append(kv_reservation)
            prompt_value, prefix_cache_payload = self._prepare_prompt_cache_invocation(
                module=module,
                model=model,
                tokenizer=tokenizer,
                request=request,
                prompt_tokens=prompt_tokens,
            )
            feature_usage = self._merge_loaded_feature_usage(
                request.model_id,
                {
                    **self._loaded_feature_usage.get(request.model_id, {}),
                    "paged_kv_cache": True,
                    "prefill_optimization": True,
                },
            )
            prompt_values.append(
                _prompt_token_ids(tokenizer, prompt_value) if isinstance(prompt_value, str) else list(prompt_value),
            )
            prompt_caches.append(prefix_cache_payload)
            prompt_token_counts.append(len(prompt_tokens))
            max_tokens.append(request.max_tokens)
            feature_usages.append(feature_usage)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "prompt_values": prompt_values,
            "prompt_caches": prompt_caches,
            "prompt_token_counts": prompt_token_counts,
            "max_tokens": max_tokens,
            "feature_usages": feature_usages,
            "kv_reservations": kv_reservations,
        }

    def _create_native_batch_generator(
        self,
        *,
        batch_generator_class: Any,
        model: Any,
        tokenizer: Any | None,
        temperature: float,
    ) -> Any:
        stop_tokens = _mlx_stop_tokens(tokenizer)
        generation_options = _mlx_text_generation_options(temperature)
        kwargs = {
            "max_tokens": 0,
            "stop_tokens": stop_tokens,
            "prefill_step_size": self.settings.prefill_token_batch_size,
            "prefill_batch_size": self.settings.continuous_batch_max_batch_size,
            "completion_batch_size": self.settings.continuous_batch_max_batch_size,
            **generation_options,
        }
        return batch_generator_class(
            model,
            **{key: value for key, value in kwargs.items() if value is not None},
        )

    @staticmethod
    def _record_native_batching_metadata(
        *,
        request: GenerateRequest,
        capability: CapabilityName,
        backend: str,
        batch_size: int,
        ownership: str = "backend_native",
        backend_primitive: str | None = None,
        queue_delay_seconds: float | None = None,
        persistent_scheduler: bool = False,
    ) -> None:
        payload: dict[str, Any] = {
            "capability": capability.value,
            "supported": True,
            "active": True,
            "backend": backend,
            "batch_size": batch_size,
            "stock_single_request_path": False,
            "fallback": False,
            "ownership": ownership,
        }
        if backend_primitive is not None:
            payload["backend_primitive"] = backend_primitive
        if queue_delay_seconds is not None:
            payload["queue_delay_milliseconds"] = int(round(max(queue_delay_seconds, 0.0) * 1000))
        if persistent_scheduler:
            payload["persistent_scheduler"] = True
        request.metadata["native_batching"] = payload

    def _release_paged_kv_reservations(self, reservations: Sequence[PagedKVReservation]) -> None:
        for reservation in reservations:
            self._paged_kv_manager.release(reservation)

    async def _generate_batch_with_batch_generator(
        self,
        *,
        batch_generator_class: Any,
        requests: Sequence[GenerateRequest],
        model_id: str,
    ) -> list[GenerateResponse]:
        state = self._prepare_native_batch_state(requests=requests, model_id=model_id)
        generator = self._create_native_batch_generator(
            batch_generator_class=batch_generator_class,
            model=state["model"],
            tokenizer=state["tokenizer"],
            temperature=requests[0].temperature,
        )
        try:
            uid_order = list(
                generator.insert(
                    prompts=state["prompt_values"],
                    max_tokens=state["max_tokens"],
                    caches=state["prompt_caches"],
                ),
            )
            uid_to_index = {uid: index for index, uid in enumerate(uid_order)}
            completion_tokens: dict[int, list[int]] = {uid: [] for uid in uid_order}
            finish_reasons: dict[int, str] = {}
            while responses := generator.next_generated():
                for response in responses:
                    if response.uid not in completion_tokens:
                        continue
                    token_value = getattr(response, "token", None)
                    if isinstance(token_value, (int, float)):
                        completion_tokens[response.uid].append(int(token_value))
                    if response.finish_reason is not None:
                        finish_reasons[response.uid] = str(response.finish_reason)
            self._native_batch_generate_calls += 1
            self._native_batch_request_count += len(requests)
            self._native_batch_max_size = max(self._native_batch_max_size, len(requests))
            for prompt_token_count, feature_usage in zip(
                state["prompt_token_counts"],
                state["feature_usages"],
                strict=True,
            ):
                self._record_feature_usage(prompt_token_count=prompt_token_count, feature_usage=feature_usage)
            for request in requests:
                self._record_native_batching_metadata(
                    request=request,
                    capability=CapabilityName.CHAT,
                    backend="mlx_lm.BatchGenerator",
                    batch_size=len(requests),
                )
            responses: list[GenerateResponse] = []
            for uid in uid_order:
                request_index = uid_to_index[uid]
                generated_tokens = completion_tokens[uid]
                prompt_token_count = state["prompt_token_counts"][request_index]
                responses.append(
                    GenerateResponse(
                        model_id=requests[request_index].model_id,
                        output_text=_decode_generated_tokens(tokenizer=state["tokenizer"], tokens=generated_tokens),
                        finish_reason=finish_reasons.get(uid, "stop"),
                        usage={
                            "prompt_tokens": prompt_token_count,
                            "completion_tokens": len(generated_tokens),
                            "total_tokens": prompt_token_count + len(generated_tokens),
                        },
                    ),
                )
            return responses
        finally:
            close = getattr(generator, "close", None)
            if callable(close):
                close()
            self._release_paged_kv_reservations(state["kv_reservations"])

    async def _generate_batch_with_batch_generate(
        self,
        *,
        batch_generate: Any,
        requests: Sequence[GenerateRequest],
        model_id: str,
    ) -> list[GenerateResponse]:
        state = self._prepare_native_batch_state(requests=requests, model_id=model_id)
        try:
            result = invoke_with_signature(
                batch_generate,
                {
                    "model": state["model"],
                    "tokenizer": state["tokenizer"],
                    "prompts": state["prompt_values"],
                    "prompt_caches": state["prompt_caches"],
                    "max_tokens": state["max_tokens"],
                    "verbose": False,
                    "return_prompt_caches": False,
                    "prefill_step_size": self.settings.prefill_token_batch_size,
                    "prefill_batch_size": self.settings.continuous_batch_max_batch_size,
                    "completion_batch_size": self.settings.continuous_batch_max_batch_size,
                    **_mlx_text_generation_options(requests[0].temperature),
                },
                capability="native_batch_generate",
                passthrough_keys=(
                    "max_tokens",
                    "verbose",
                    "return_prompt_caches",
                    "sampler",
                    "temperature",
                    "temp",
                    "prefill_step_size",
                    "prefill_batch_size",
                    "completion_batch_size",
                ),
            )
            texts = list(getattr(result, "texts", []))
            if len(texts) != len(requests):
                raise ConfigurationError(
                    f"Expected {len(requests)} MLX batched chat results, received {len(texts)}.",
                )
            self._native_batch_generate_calls += 1
            self._native_batch_request_count += len(requests)
            self._native_batch_max_size = max(self._native_batch_max_size, len(requests))
            for prompt_token_count, feature_usage in zip(
                state["prompt_token_counts"],
                state["feature_usages"],
                strict=True,
            ):
                self._record_feature_usage(prompt_token_count=prompt_token_count, feature_usage=feature_usage)
            for request in requests:
                self._record_native_batching_metadata(
                    request=request,
                    capability=CapabilityName.CHAT,
                    backend="mlx_lm.batch_generate",
                    batch_size=len(requests),
                )
            responses: list[GenerateResponse] = []
            for request, text, prompt_token_count in zip(
                requests,
                texts,
                state["prompt_token_counts"],
                strict=True,
            ):
                completion_tokens = _prompt_token_ids(state["tokenizer"], str(text))
                responses.append(
                    GenerateResponse(
                        model_id=request.model_id,
                        output_text=str(text),
                        finish_reason="stop",
                        usage={
                            "prompt_tokens": prompt_token_count,
                            "completion_tokens": len(completion_tokens),
                            "total_tokens": prompt_token_count + len(completion_tokens),
                        },
                    ),
                )
            return responses
        finally:
            self._release_paged_kv_reservations(state["kv_reservations"])

    def _request_performance_controls(
        self,
        *,
        model_id: str,
        generation_controls: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        payload: dict[str, dict[str, dict[str, Any]]] = {}
        loaded_controls = self._loaded_performance_controls.get(model_id)
        if loaded_controls:
            payload["load"] = copy.deepcopy(loaded_controls)
        if generation_controls:
            payload["generate"] = copy.deepcopy(generation_controls)
        return payload

    def _loaded_manifest_memory_mb(self, manifest: ModelManifest) -> int | None:
        return self._frontier_execution.loaded_memory_override(manifest.model_id) or manifest.estimated_memory_mb

    def _record_frontier_execution(self, request: GenerateRequest) -> None:
        manifest = self._loaded_manifests.get(request.model_id)
        if manifest is None:
            return
        self._frontier_execution.annotate_request(manifest=manifest, request=request)

    def _record_frontier_batch_requests(self, requests: Sequence[GenerateRequest]) -> None:
        for request in requests:
            self._record_frontier_execution(request)

    def _continuous_batch_controller_key(
        self,
        *,
        request: GenerateRequest,
        capability: CapabilityName,
    ) -> tuple[int, str, str, str]:
        loop = asyncio.get_running_loop()
        return (id(loop), capability.value, request.model_id, f"{request.temperature:.4f}")

    def _close_owned_continuous_batch_controllers(self, *, model_id: str | None = None) -> None:
        for controllers in (self._owned_generate_controllers, self._owned_stream_controllers):
            for key, controller in list(controllers.items()):
                if model_id is not None and key[2] != model_id:
                    continue
                controller.close()
                controllers.pop(key, None)

    def _prune_owned_continuous_batch_controllers(self) -> None:
        for controllers in (self._owned_generate_controllers, self._owned_stream_controllers):
            for key, controller in list(controllers.items()):
                if controller.closed:
                    controllers.pop(key, None)

    def _should_use_owned_continuous_batching(
        self,
        *,
        request: GenerateRequest,
        capability: CapabilityName,
    ) -> bool:
        return request.speculation is None and self.continuous_batching_ownership(capability) == "lewlm_owned"

    def _generate_controller(self, *, request: GenerateRequest) -> _MLXTextGenerateController:
        self._prune_owned_continuous_batch_controllers()
        key = self._continuous_batch_controller_key(request=request, capability=CapabilityName.CHAT)
        controller = self._owned_generate_controllers.get(key)
        if controller is None or controller.closed:
            controller = _MLXTextGenerateController(
                runtime=self,
                model_id=request.model_id,
                temperature=request.temperature,
            )
            self._owned_generate_controllers[key] = controller
        return controller

    def _stream_controller(self, *, request: GenerateRequest) -> _MLXTextStreamController:
        self._prune_owned_continuous_batch_controllers()
        key = self._continuous_batch_controller_key(request=request, capability=CapabilityName.STREAMING)
        controller = self._owned_stream_controllers.get(key)
        if controller is None or controller.closed:
            controller = _MLXTextStreamController(
                runtime=self,
                model_id=request.model_id,
                temperature=request.temperature,
            )
            self._owned_stream_controllers[key] = controller
        return controller

    def _kv_cache_load_options(self, load_callable: Any) -> tuple[dict[str, Any], dict[str, bool], dict[str, dict[str, Any]]]:
        parameter_names = _callable_parameter_names(load_callable)
        options: dict[str, Any] = {}
        feature_usage = {
            "paged_kv_cache": False,
            "kv_cache_quantization": False,
        }
        kv_cache_config = _compact_runtime_metrics(
            page_size=self.settings.kv_cache_page_size,
            max_pages=self.settings.kv_cache_max_pages,
            quantization_bits=self.settings.kv_cache_quantization_bits,
        )
        config_parameter = _first_matching_parameter(parameter_names, _KV_CACHE_CONFIG_PARAMETERS)
        paged_applied_parameters: list[str] = []
        quantization_applied_parameters: list[str] = []
        paged_rejected_parameters: list[str] = []
        quantization_rejected_parameters: list[str] = []
        if config_parameter is not None and kv_cache_config:
            options[config_parameter] = kv_cache_config
            feature_usage["paged_kv_cache"] = True
            paged_applied_parameters.append(config_parameter)
            if self.settings.kv_cache_quantization_bits is not None:
                feature_usage["kv_cache_quantization"] = True
                quantization_applied_parameters.append(config_parameter)
        else:
            page_size_parameter = _first_matching_parameter(parameter_names, _KV_CACHE_PAGE_SIZE_PARAMETERS)
            if page_size_parameter is not None:
                options[page_size_parameter] = self.settings.kv_cache_page_size
                feature_usage["paged_kv_cache"] = True
                paged_applied_parameters.append(page_size_parameter)
            else:
                paged_rejected_parameters.append("kv_cache_page_size")
            max_pages_parameter = _first_matching_parameter(parameter_names, _KV_CACHE_MAX_PAGES_PARAMETERS)
            if max_pages_parameter is not None and self.settings.kv_cache_max_pages is not None:
                options[max_pages_parameter] = self.settings.kv_cache_max_pages
                feature_usage["paged_kv_cache"] = True
                paged_applied_parameters.append(max_pages_parameter)
            elif self.settings.kv_cache_max_pages is not None:
                paged_rejected_parameters.append("kv_cache_max_pages")
            quantization_parameter = _first_matching_parameter(parameter_names, _KV_CACHE_QUANTIZATION_PARAMETERS)
            if quantization_parameter is not None and self.settings.kv_cache_quantization_bits is not None:
                options[quantization_parameter] = self.settings.kv_cache_quantization_bits
                feature_usage["kv_cache_quantization"] = True
                quantization_applied_parameters.append(quantization_parameter)
            elif self.settings.kv_cache_quantization_bits is not None:
                quantization_rejected_parameters.append("kv_cache_quantization_bits")
        paged_effective = (
            "enabled"
            if config_parameter is not None
            or (
                "kv_cache_page_size" not in paged_rejected_parameters
                and "kv_cache_max_pages" not in paged_rejected_parameters
            )
            else "partial" if paged_applied_parameters
            else "unsupported"
        )
        quantization_supported = bool(config_parameter or _first_matching_parameter(parameter_names, _KV_CACHE_QUANTIZATION_PARAMETERS))
        quantization_effective = (
            "disabled"
            if self.settings.kv_cache_quantization_bits is None
            else "enabled" if quantization_applied_parameters
            else "unsupported"
        )
        control_snapshot = {
            "paged_kv_cache": _performance_control_payload(
                requested=True,
                supported=bool(config_parameter or paged_applied_parameters),
                effective=paged_effective,
                reason=(
                    "MLX model load accepts paged KV-cache configuration."
                    if config_parameter is not None or paged_applied_parameters
                    else "Installed MLX model load entrypoints do not expose paged KV-cache configuration parameters."
                ),
                applied_parameters=tuple(paged_applied_parameters),
                rejected_parameters=tuple(dict.fromkeys(paged_rejected_parameters)),
                requested_page_size_tokens=self.settings.kv_cache_page_size,
                requested_max_pages=self.settings.kv_cache_max_pages,
                effective_page_size_tokens=self.settings.kv_cache_page_size if paged_applied_parameters else None,
                effective_max_pages=(
                    self.settings.kv_cache_max_pages if "kv_cache_max_pages" not in paged_rejected_parameters else None
                ),
            ),
            "kv_cache_quantization": _performance_control_payload(
                requested=self.settings.kv_cache_quantization_bits is not None,
                supported=quantization_supported,
                effective=quantization_effective,
                reason=(
                    "MLX model load accepts KV-cache quantization configuration."
                    if quantization_supported
                    else "Installed MLX model load entrypoints do not expose KV-cache quantization parameters."
                ),
                applied_parameters=tuple(quantization_applied_parameters),
                rejected_parameters=tuple(dict.fromkeys(quantization_rejected_parameters)),
                requested_quantization_bits=self.settings.kv_cache_quantization_bits,
                effective_quantization_bits=(
                    self.settings.kv_cache_quantization_bits if quantization_applied_parameters else None
                ),
            ),
        }
        return options, feature_usage, control_snapshot

    def _generation_performance_options(
        self,
        callable_obj: Any,
        *,
        prompt_tokens: list[int],
    ) -> tuple[dict[str, Any], dict[str, bool], dict[str, dict[str, Any]]]:
        parameter_names = _callable_parameter_names(callable_obj)
        options: dict[str, Any] = {}
        feature_usage = {
            "paged_kv_cache": False,
            "kv_cache_quantization": False,
            "prefill_optimization": False,
        }
        paged_applied_parameters: list[str] = []
        quantization_applied_parameters: list[str] = []
        prefill_applied_parameters: list[str] = []
        paged_rejected_parameters: list[str] = []
        quantization_rejected_parameters: list[str] = []
        prefill_rejected_parameters: list[str] = []
        kv_cache_config = _compact_runtime_metrics(
            page_size=self.settings.kv_cache_page_size,
            max_pages=self.settings.kv_cache_max_pages,
            quantization_bits=self.settings.kv_cache_quantization_bits,
        )
        config_parameter = _first_matching_parameter(parameter_names, _KV_CACHE_CONFIG_PARAMETERS)
        if config_parameter is not None and kv_cache_config:
            options[config_parameter] = kv_cache_config
            feature_usage["paged_kv_cache"] = True
            feature_usage["kv_cache_quantization"] = self.settings.kv_cache_quantization_bits is not None
            paged_applied_parameters.append(config_parameter)
            if self.settings.kv_cache_quantization_bits is not None:
                quantization_applied_parameters.append(config_parameter)
        else:
            page_size_parameter = _first_matching_parameter(parameter_names, _KV_CACHE_PAGE_SIZE_PARAMETERS)
            if page_size_parameter is not None:
                options[page_size_parameter] = self.settings.kv_cache_page_size
                feature_usage["paged_kv_cache"] = True
                paged_applied_parameters.append(page_size_parameter)
            else:
                paged_rejected_parameters.append("kv_cache_page_size")
            max_pages_parameter = _first_matching_parameter(parameter_names, _KV_CACHE_MAX_PAGES_PARAMETERS)
            if max_pages_parameter is not None and self.settings.kv_cache_max_pages is not None:
                options[max_pages_parameter] = self.settings.kv_cache_max_pages
                feature_usage["paged_kv_cache"] = True
                paged_applied_parameters.append(max_pages_parameter)
            elif self.settings.kv_cache_max_pages is not None:
                paged_rejected_parameters.append("kv_cache_max_pages")
            quantization_parameter = _first_matching_parameter(parameter_names, _KV_CACHE_QUANTIZATION_PARAMETERS)
            if quantization_parameter is not None and self.settings.kv_cache_quantization_bits is not None:
                options[quantization_parameter] = self.settings.kv_cache_quantization_bits
                feature_usage["kv_cache_quantization"] = True
                quantization_applied_parameters.append(quantization_parameter)
            elif self.settings.kv_cache_quantization_bits is not None:
                quantization_rejected_parameters.append("kv_cache_quantization_bits")
        prefill_config_parameter = _first_matching_parameter(parameter_names, _PREFILL_CONFIG_PARAMETERS)
        if prefill_config_parameter is not None:
            options[prefill_config_parameter] = {
                "token_batch_size": self.settings.prefill_token_batch_size,
                "prompt_token_count": len(prompt_tokens),
            }
            feature_usage["prefill_optimization"] = True
            prefill_applied_parameters.append(prefill_config_parameter)
        else:
            prefill_rejected_parameters.append("prefill_config")
        batch_size_parameter = _first_matching_parameter(parameter_names, _PREFILL_BATCH_SIZE_PARAMETERS)
        if batch_size_parameter is not None:
            options[batch_size_parameter] = self.settings.prefill_token_batch_size
            feature_usage["prefill_optimization"] = True
            prefill_applied_parameters.append(batch_size_parameter)
        else:
            prefill_rejected_parameters.append("prefill_token_batch_size")
        prompt_token_parameter = _first_matching_parameter(parameter_names, _PROMPT_TOKEN_LIST_PARAMETERS)
        if prompt_token_parameter is not None:
            options[prompt_token_parameter] = prompt_tokens
            feature_usage["prefill_optimization"] = True
            prefill_applied_parameters.append(prompt_token_parameter)
        else:
            prefill_rejected_parameters.append("prompt_tokens")
        prompt_token_count_parameter = _first_matching_parameter(parameter_names, _PROMPT_TOKEN_COUNT_PARAMETERS)
        if prompt_token_count_parameter is not None:
            options[prompt_token_count_parameter] = len(prompt_tokens)
            feature_usage["prefill_optimization"] = True
            prefill_applied_parameters.append(prompt_token_count_parameter)
        else:
            prefill_rejected_parameters.append("prompt_token_count")
        prefill_batch_supported = bool(prefill_config_parameter or batch_size_parameter)
        prompt_token_supported = bool(prompt_token_parameter or prompt_token_count_parameter)
        control_snapshot = {
            "paged_kv_cache": _performance_control_payload(
                requested=True,
                supported=bool(config_parameter or paged_applied_parameters),
                effective=(
                    "enabled"
                    if config_parameter is not None
                    or (
                        "kv_cache_page_size" not in paged_rejected_parameters
                        and "kv_cache_max_pages" not in paged_rejected_parameters
                    )
                    else "partial" if paged_applied_parameters
                    else "unsupported"
                ),
                reason=(
                    "MLX generation accepts paged KV-cache configuration."
                    if config_parameter is not None or paged_applied_parameters
                    else "Installed MLX generation entrypoints do not expose paged KV-cache configuration parameters."
                ),
                applied_parameters=tuple(paged_applied_parameters),
                rejected_parameters=tuple(dict.fromkeys(paged_rejected_parameters)),
                requested_page_size_tokens=self.settings.kv_cache_page_size,
                requested_max_pages=self.settings.kv_cache_max_pages,
                effective_page_size_tokens=self.settings.kv_cache_page_size if paged_applied_parameters else None,
                effective_max_pages=(
                    self.settings.kv_cache_max_pages if "kv_cache_max_pages" not in paged_rejected_parameters else None
                ),
            ),
            "kv_cache_quantization": _performance_control_payload(
                requested=self.settings.kv_cache_quantization_bits is not None,
                supported=bool(config_parameter or _first_matching_parameter(parameter_names, _KV_CACHE_QUANTIZATION_PARAMETERS)),
                effective=(
                    "disabled"
                    if self.settings.kv_cache_quantization_bits is None
                    else "enabled" if quantization_applied_parameters
                    else "unsupported"
                ),
                reason=(
                    "MLX generation accepts KV-cache quantization configuration."
                    if config_parameter is not None or _first_matching_parameter(parameter_names, _KV_CACHE_QUANTIZATION_PARAMETERS)
                    else "Installed MLX generation entrypoints do not expose KV-cache quantization parameters."
                ),
                applied_parameters=tuple(quantization_applied_parameters),
                rejected_parameters=tuple(dict.fromkeys(quantization_rejected_parameters)),
                requested_quantization_bits=self.settings.kv_cache_quantization_bits,
                effective_quantization_bits=(
                    self.settings.kv_cache_quantization_bits if quantization_applied_parameters else None
                ),
            ),
            "prefill_optimization": _performance_control_payload(
                requested=True,
                supported=bool(prefill_applied_parameters),
                effective="enabled" if prefill_batch_supported else "partial" if prompt_token_supported else "unsupported",
                reason=(
                    "MLX generation accepts prompt-token or prefill batch sizing controls."
                    if prefill_applied_parameters
                    else "Installed MLX generation entrypoints do not expose prefill batch or prompt-token controls."
                ),
                applied_parameters=tuple(prefill_applied_parameters),
                rejected_parameters=tuple(dict.fromkeys(prefill_rejected_parameters)),
                requested_prefill_token_batch_size=self.settings.prefill_token_batch_size,
                effective_prefill_token_batch_size=(
                    self.settings.prefill_token_batch_size if prefill_batch_supported else None
                ),
                prompt_token_count=len(prompt_tokens),
                tokenized_prompt_supplied=prompt_token_supported,
            ),
            "chunked_prefill": _performance_control_payload(
                requested=True,
                supported=bool(prefill_applied_parameters),
                effective=(
                    "enabled"
                    if prefill_batch_supported and len(prompt_tokens) > self.settings.prefill_token_batch_size
                    else "disabled"
                    if prefill_batch_supported
                    else "partial"
                    if prompt_token_supported
                    else "unsupported"
                ),
                reason=(
                    "MLX generation accepts explicit token-batch sizing for prompt ingest."
                    if prefill_applied_parameters
                    else "Installed MLX generation entrypoints do not expose chunked prefill controls."
                ),
                applied_parameters=tuple(prefill_applied_parameters),
                rejected_parameters=tuple(dict.fromkeys(prefill_rejected_parameters)),
                requested_prefill_token_batch_size=self.settings.prefill_token_batch_size,
                effective_prefill_token_batch_size=(
                    self.settings.prefill_token_batch_size if prefill_batch_supported else None
                ),
                prompt_token_count=len(prompt_tokens),
                chunk_count=max(1, math.ceil(len(prompt_tokens) / self.settings.prefill_token_batch_size)),
            ),
        }
        return options, feature_usage, control_snapshot

    def _supports_prefix_cache(self, *, module: Any | None = None) -> bool:
        if not self.is_available():
            return False
        module = module or import_module("mlx_lm")
        return (
            resolve_backend_callable(module, ("generate", "chat", "generate_text"), required=False) is not None
            and _mlx_cache_helpers() is not None
        )

    def prefix_cache_admission_preview(
        self,
        *,
        model_id: str,
        messages: Sequence[GenerateMessage],
    ) -> dict[str, object]:
        if model_id not in self._clients:
            return {}
        try:
            module = import_module("mlx_lm")
        except ImportError:
            return {}
        if not self._supports_prefix_cache(module=module):
            return {}
        _, tokenizer = self._client_components(model_id)
        prompt = _messages_to_prompt(messages, tokenizer)
        prompt_tokens = _prompt_token_ids(tokenizer, prompt)
        cached_prompt_tokens = prompt_tokens[:-1] if len(prompt_tokens) > 1 else prompt_tokens
        preview = self._prefix_cache.preview(model_id=model_id, prompt_tokens=cached_prompt_tokens)
        cached_prefix_tokens = preview.prefix_length if preview is not None else 0
        effective_prefill_tokens = max(len(cached_prompt_tokens) - cached_prefix_tokens, 0)
        return {
            "supported": True,
            "total_prompt_tokens": len(prompt_tokens),
            "effective_prefill_tokens": effective_prefill_tokens,
            "cached_prefix_tokens": cached_prefix_tokens,
            "cached_pages": preview.matched_page_count if preview is not None else 0,
            "page_size_tokens": preview.page_size_tokens if preview is not None else self.settings.kv_cache_page_size,
            "cache_key": preview.cache_key if preview is not None else None,
            "lookup_source": preview.source if preview is not None else "miss",
        }

    def _prepare_prompt_cache_invocation(
        self,
        *,
        module: Any,
        model: Any,
        tokenizer: Any | None,
        request: GenerateRequest,
        prompt_tokens: list[int],
    ) -> tuple[list[int] | str, Any | None]:
        helpers = _mlx_cache_helpers()
        if helpers is None or len(prompt_tokens) <= 1:
            request.metadata["prefix_cache"] = {
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_saves": 0,
                "resident_cache_hits": 0,
                "persistent_cache_hits": 0,
                "page_size_tokens": self.settings.kv_cache_page_size,
                "cached_pages": 0,
                "resident_page_hits": 0,
                "persistent_page_hits": 0,
                "restored_pages": 0,
                "stored_pages": 0,
                "copy_on_write_reused_pages": 0,
                "cache_restores": 0,
                "cached_tokens": 0,
                "saved_prefill_tokens": 0,
                "max_saved_prefill_tokens": 0,
                "lookup_source": "miss",
                "prefilled_uncached_tokens": 0,
                "total_prompt_tokens": len(prompt_tokens),
                "effective_prefill_tokens": 0,
            }
            return prompt_tokens, None
        lookup = self._prefix_cache.lookup(model_id=request.model_id, prompt_tokens=prompt_tokens[:-1])
        matched_prefix_tokens = 0
        cache_payload = helpers["make_prompt_cache"](model)
        lookup_source = "miss"
        if lookup is not None:
            cache_payload = copy.deepcopy(lookup.entry.payload)
            matched_prefix_tokens = lookup.prefix_length
            lookup_source = lookup.entry.source
            trim_count = len(lookup.entry.prefix_tokens) - matched_prefix_tokens
            if trim_count > 0:
                helpers["trim_prompt_cache"](cache_payload, trim_count)
        uncached_prefix_tokens = prompt_tokens[matched_prefix_tokens:-1]
        if uncached_prefix_tokens:
            generate = resolve_backend_callable(module, ("generate", "chat", "generate_text"))
            self._acceleration.invoke(
                request=request,
                callable_obj=generate,
                callable_key="generate",
                provided_values={
                    "client": {"model": model, "tokenizer": tokenizer},
                    "model": model,
                    "tokenizer": tokenizer,
                    "prompt": uncached_prefix_tokens,
                    "max_tokens": 0,
                    "verbose": False,
                    "prompt_cache": cache_payload,
                },
                capability="prefix_cache_prefill",
                passthrough_keys=("max_tokens",),
                phase="prefill",
            )
        stored_cache = copy.deepcopy(cache_payload)
        stored_entry = self._prefix_cache.save(
            model_id=request.model_id,
            prefix_tokens=prompt_tokens[:-1],
            payload=stored_cache,
        )
        request.metadata["prefix_cache"] = {
            "cache_hits": 1 if lookup is not None else 0,
            "cache_misses": 0 if lookup is not None else 1,
            "cache_saves": 1,
            "resident_cache_hits": 1 if lookup is not None and lookup.entry.source == "resident" else 0,
            "persistent_cache_hits": 1 if lookup is not None and lookup.entry.source == "persisted" else 0,
            "page_size_tokens": stored_entry.page_size_tokens if stored_entry is not None else self.settings.kv_cache_page_size,
            "cached_pages": lookup.matched_page_count if lookup is not None else 0,
            "resident_page_hits": lookup.resident_page_hits if lookup is not None else 0,
            "persistent_page_hits": lookup.persisted_page_hits if lookup is not None else 0,
            "restored_pages": lookup.restored_page_count if lookup is not None else 0,
            "stored_pages": stored_entry.page_count if stored_entry is not None else 0,
            "copy_on_write_reused_pages": lookup.matched_page_count if lookup is not None else 0,
            "cache_restores": 1 if lookup is not None and lookup.entry.source == "persisted" else 0,
            "saved_prefill_tokens": matched_prefix_tokens,
            "cached_tokens": matched_prefix_tokens,
            "max_saved_prefill_tokens": matched_prefix_tokens,
            "cache_key": stored_entry.cache_key if stored_entry is not None else None,
            "lookup_source": lookup_source,
            "prefilled_uncached_tokens": len(uncached_prefix_tokens),
            "total_prompt_tokens": len(prompt_tokens),
            "effective_prefill_tokens": len(uncached_prefix_tokens),
        }
        return [prompt_tokens[-1]], cache_payload

    def _reserve_paged_kv_residency(
        self,
        *,
        request: GenerateRequest,
        prompt_tokens: Sequence[int],
    ) -> PagedKVReservation:
        reservation = self._paged_kv_manager.reserve(
            model_id=request.model_id,
            prompt_tokens=prompt_tokens,
            max_tokens=request.max_tokens,
            scheduling_lane=self._scheduling_lane_for_request(request=request, prompt_tokens=prompt_tokens),
        )
        request.metadata["kv_residency"] = _compact_runtime_metrics(
            page_size_tokens=self.settings.kv_cache_page_size,
            max_pages=self.settings.kv_cache_max_pages,
            queue_lane=reservation.scheduling_lane,
            requested_pages=reservation.requested_pages,
            prompt_pages=reservation.prompt_pages,
            decode_pages=reservation.decode_pages,
            reused_pages=reservation.reused_pages,
            new_pages=reservation.new_pages,
            evicted_pages=reservation.evicted_pages,
            overflow_pages=reservation.overflow_pages,
            resident_pages=reservation.resident_pages_after,
            active_pages=reservation.active_pages_after,
            resident_decode_pages=reservation.resident_decode_pages_after,
            resident_prefill_pages=reservation.resident_prefill_pages_after,
            active_decode_pages=reservation.active_decode_pages_after,
            active_prefill_pages=reservation.active_prefill_pages_after,
            pressure_ratio=reservation.pressure_ratio,
            pressure_level=reservation.pressure_level,
        )
        scheduling = request.metadata.get("scheduling")
        if isinstance(scheduling, dict):
            scheduling.update(
                {
                    "kv_requested_pages": reservation.requested_pages,
                    "kv_reused_pages": reservation.reused_pages,
                    "kv_evicted_pages": reservation.evicted_pages,
                    "kv_overflow_pages": reservation.overflow_pages,
                    "kv_resident_pages": reservation.resident_pages_after,
                    "kv_active_pages": reservation.active_pages_after,
                    "kv_active_decode_pages": reservation.active_decode_pages_after,
                    "kv_active_prefill_pages": reservation.active_prefill_pages_after,
                    "kv_pressure_ratio": reservation.pressure_ratio,
                    "kv_pressure_level": reservation.pressure_level,
                },
            )
        return reservation

    def _scheduling_lane_for_request(
        self,
        *,
        request: GenerateRequest,
        prompt_tokens: Sequence[int],
    ) -> Literal["decode", "prefill"]:
        scheduling = request.metadata.get("scheduling")
        if isinstance(scheduling, dict):
            queue_lane = scheduling.get("queue_lane")
            if queue_lane in {"decode", "prefill"}:
                return queue_lane
        prompt_token_count = max(len(prompt_tokens), 1)
        prefill_chunk_count = max(
            1,
            math.ceil(prompt_token_count / self.settings.prefill_token_batch_size),
        )
        return (
            "prefill"
            if prompt_token_count >= self.settings.long_prefill_token_threshold or prefill_chunk_count > 1
            else "decode"
        )

    def _record_feature_usage(self, *, prompt_token_count: int, feature_usage: dict[str, bool]) -> None:
        normalized_prompt_tokens = max(prompt_token_count, 0)
        if feature_usage["paged_kv_cache"]:
            self._paged_kv_request_count += 1
            self._paged_kv_prompt_tokens += normalized_prompt_tokens
        if feature_usage["kv_cache_quantization"]:
            self._quantized_kv_request_count += 1
        if feature_usage["prefill_optimization"]:
            chunk_count = max(
                1,
                math.ceil(normalized_prompt_tokens / self.settings.prefill_token_batch_size),
            )
            self._prefill_optimized_request_count += 1
            self._prefill_prompt_tokens += normalized_prompt_tokens
            self._prefill_batch_count += chunk_count
            if chunk_count > 1:
                self._chunked_prefill_request_count += 1
                self._chunked_prefill_prompt_tokens += normalized_prompt_tokens
                self._chunked_prefill_chunk_count += chunk_count

    def _supports_draft_model_speculation(self, generate: Any | None = None) -> bool:
        if not self.is_available():
            return False
        if generate is None:
            module = import_module("mlx_lm")
            generate = resolve_backend_callable(module, ("generate", "chat", "generate_text"), required=False)
        return generate is not None and _mlx_cache_helpers() is not None and _callable_accepts_parameter(
            generate,
            ("prompt", "input_ids", "tokens"),
        )

    def _supported_speculation_modes(self, generate_stream: Any | None = None) -> set[SpeculationMode]:
        if not self.is_available():
            return set()
        module = import_module("mlx_lm")
        generate = resolve_backend_callable(module, ("generate", "chat", "generate_text"), required=False)
        if generate_stream is None:
            generate_stream = resolve_backend_callable(module, ("generate_stream", "stream_generate"), required=False)
        modes: set[SpeculationMode] = set()
        if self._supports_draft_model_speculation(generate) or (
            _callable_accepts_parameter(generate_stream, _DRAFT_MODEL_PARAMETERS)
            and _callable_accepts_parameter(generate_stream, _NUM_DRAFT_TOKENS_PARAMETERS)
        ):
            modes.add(SpeculationMode.DRAFT_MODEL)
        parameter_names = _callable_parameter_names(generate_stream)
        if (
            _first_matching_parameter(parameter_names, ("draft_model", "draft", "draft_client")) is not None
            and _first_matching_parameter(parameter_names, ("num_draft_tokens", "draft_tokens")) is not None
        ):
            modes.add(SpeculationMode.DRAFT_MODEL)
        for mode, aliases in _FRONTIER_SPECULATION_PARAMETER_ALIASES.items():
            if _first_matching_parameter(parameter_names, aliases) is not None:
                modes.add(mode)
        return modes

    async def _generate_with_speculation(self, request: GenerateRequest) -> GenerateResponse:
        speculation = request.speculation
        if speculation is None:
            raise ConfigurationError("Speculative generation requires a speculation configuration.")
        if speculation.mode == SpeculationMode.DRAFT_MODEL and self._supports_draft_model_speculation():
            return await self._generate_with_owned_draft_controller(request)
        self._set_speculation_execution_metadata(
            request,
            ownership="backend_passthrough",
            execution_path="backend_passthrough",
            controller="backend_adapter",
        )
        output_parts: list[str] = []
        drafted_tokens = 0
        accepted_tokens = 0
        verified_tokens = 0
        rejected_tokens = 0
        rollback_tokens = 0
        usage: dict[str, int] = {}
        async for text, chunk_drafted_tokens, chunk_accepted_tokens, chunk_verified_tokens, chunk_usage in self._iter_backend_speculation_chunks(request):
            if text:
                output_parts.append(text)
            drafted_tokens += chunk_drafted_tokens
            accepted_tokens += chunk_accepted_tokens
            verified_tokens += chunk_verified_tokens
            rejected_tokens += max(chunk_drafted_tokens - chunk_accepted_tokens, 0)
            rollback_tokens += int(chunk_usage.get("rollback_tokens", 0)) if chunk_usage else 0
            if chunk_usage:
                usage.update(chunk_usage)
        if drafted_tokens > 0 and rollback_tokens <= 0:
            rollback_tokens = max(rejected_tokens, 0)
        usage.update(
            _compact_runtime_metrics(
                drafted_tokens=drafted_tokens,
                accepted_tokens=accepted_tokens,
                verified_tokens=verified_tokens,
                rejected_tokens=rejected_tokens,
                rollback_tokens=rollback_tokens,
            ),
        )
        self._set_speculation_execution_metadata(
            request,
            drafted_tokens=drafted_tokens,
            accepted_tokens=accepted_tokens,
            verified_tokens=verified_tokens,
            rejected_tokens=rejected_tokens,
            rollback_tokens=rollback_tokens,
            fallback_count=0,
        )
        self._record_speculative_usage(
            drafted_tokens=drafted_tokens,
            accepted_tokens=accepted_tokens,
            verified_tokens=verified_tokens,
            rejected_tokens=rejected_tokens,
            rollback_tokens=rollback_tokens,
            fallback_count=0,
            ownership="backend_passthrough",
        )
        return GenerateResponse(
            model_id=request.model_id,
            output_text="".join(output_parts),
            finish_reason="stop",
            usage=usage,
        )

    async def _stream_generate_with_speculation(self, request: GenerateRequest) -> AsyncIterator[str]:
        speculation = request.speculation
        if speculation is None:
            raise ConfigurationError("Speculative generation requires a speculation configuration.")
        if speculation.mode == SpeculationMode.DRAFT_MODEL and self._supports_draft_model_speculation():
            async for chunk in self._stream_generate_with_owned_draft_controller(request):
                yield chunk
            return
        self._set_speculation_execution_metadata(
            request,
            ownership="backend_passthrough",
            execution_path="backend_passthrough",
            controller="backend_adapter",
        )
        drafted_tokens = 0
        accepted_tokens = 0
        verified_tokens = 0
        rejected_tokens = 0
        rollback_tokens = 0
        async for text, chunk_drafted_tokens, chunk_accepted_tokens, chunk_verified_tokens, chunk_usage in self._iter_backend_speculation_chunks(request):
            drafted_tokens += chunk_drafted_tokens
            accepted_tokens += chunk_accepted_tokens
            verified_tokens += chunk_verified_tokens
            rejected_tokens += max(chunk_drafted_tokens - chunk_accepted_tokens, 0)
            rollback_tokens += int(chunk_usage.get("rollback_tokens", 0)) if chunk_usage else 0
            if text:
                yield text
        if drafted_tokens > 0 and rollback_tokens <= 0:
            rollback_tokens = max(rejected_tokens, 0)
        self._set_speculation_execution_metadata(
            request,
            drafted_tokens=drafted_tokens,
            accepted_tokens=accepted_tokens,
            verified_tokens=verified_tokens,
            rejected_tokens=rejected_tokens,
            rollback_tokens=rollback_tokens,
            fallback_count=0,
        )
        self._record_speculative_usage(
            drafted_tokens=drafted_tokens,
            accepted_tokens=accepted_tokens,
            verified_tokens=verified_tokens,
            rejected_tokens=rejected_tokens,
            rollback_tokens=rollback_tokens,
            fallback_count=0,
            ownership="backend_passthrough",
        )

    async def _iter_backend_speculation_chunks(
        self,
        request: GenerateRequest,
    ) -> AsyncIterator[tuple[str, int, int, int, dict[str, int]]]:
        speculation = request.speculation
        if speculation is None:
            raise ConfigurationError("Speculative generation requires a speculation configuration.")
        module = import_module("mlx_lm")
        generate_stream = resolve_backend_callable(module, ("generate_stream", "stream_generate"))
        supported_modes = self._supported_speculation_modes(generate_stream)
        if speculation.mode not in supported_modes:
            raise ConfigurationError(
                f"Installed MLX backend does not support `{speculation.mode.value}` speculative decoding.",
                details={"supported_modes": sorted(mode.value for mode in supported_modes)},
            )
        model, tokenizer = self._client_components(request.model_id)
        prompt = _messages_to_prompt(request.messages, tokenizer)
        prompt_tokens = _prompt_token_ids(tokenizer, prompt)
        generation_options = _mlx_text_generation_options(request.temperature)
        performance_options, feature_usage, generation_controls = self._generation_performance_options(
            generate_stream,
            prompt_tokens=prompt_tokens,
        )
        feature_usage = self._merge_loaded_feature_usage(request.model_id, feature_usage)
        request.metadata["performance_controls"] = self._request_performance_controls(
            model_id=request.model_id,
            generation_controls=generation_controls,
        )
        chunks = self._acceleration.invoke(
            request=request,
            callable_obj=generate_stream,
            callable_key="speculative_generate_stream",
            provided_values={
                "client": {"model": model, "tokenizer": tokenizer},
                "model": model,
                "tokenizer": tokenizer,
                "prompt": prompt,
                "messages": [{"role": message.role, "content": message.content} for message in request.messages],
                "max_tokens": request.max_tokens,
                "verbose": False,
                **generation_options,
                **performance_options,
                **self._speculation_invoke_options(generate_stream=generate_stream, request=request),
            },
            capability=CapabilityName.STREAMING.value,
            passthrough_keys=("sampler", "temperature", "temp", *self._speculation_passthrough_keys(request)),
        )
        self._record_feature_usage(prompt_token_count=len(prompt_tokens), feature_usage=feature_usage)
        for chunk in chunks:
            text = _mlx_chunk_to_text(chunk)
            from_draft = bool(text) and _mlx_chunk_from_draft(chunk)
            chunk_usage = _normalize_usage(_mlx_chunk_usage(chunk))
            chunk_drafted_tokens = int(chunk_usage.get("drafted_tokens", 0))
            chunk_accepted_tokens = int(chunk_usage.get("accepted_tokens", chunk_usage.get("verified_tokens", 0)))
            chunk_verified_tokens = int(chunk_usage.get("verified_tokens", chunk_accepted_tokens))
            if chunk_drafted_tokens <= 0 and from_draft:
                chunk_drafted_tokens = 1
            if chunk_verified_tokens <= 0 and text and not from_draft:
                chunk_verified_tokens = 1
            if chunk_accepted_tokens <= 0 and chunk_verified_tokens > 0 and chunk_drafted_tokens > 0:
                chunk_accepted_tokens = min(chunk_verified_tokens, chunk_drafted_tokens)
            yield (
                text,
                chunk_drafted_tokens,
                chunk_accepted_tokens,
                chunk_verified_tokens,
                chunk_usage,
            )

    async def _generate_with_owned_draft_controller(self, request: GenerateRequest) -> GenerateResponse:
        output_parts: list[str] = []
        async for chunk in self._iter_owned_draft_controller_chunks(request):
            if chunk:
                output_parts.append(chunk)
        runtime_summary = self._speculation_runtime_summary(request)
        prompt_tokens = _coerce_int(runtime_summary.get("prompt_tokens"))
        completion_tokens = _coerce_int(runtime_summary.get("completion_tokens"))
        usage = _compact_runtime_metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            drafted_tokens=_coerce_int(runtime_summary.get("drafted_tokens")),
            accepted_tokens=_coerce_int(runtime_summary.get("accepted_tokens")),
            verified_tokens=_coerce_int(runtime_summary.get("verified_tokens")),
            rejected_tokens=_coerce_int(runtime_summary.get("rejected_tokens")),
            rollback_tokens=_coerce_int(runtime_summary.get("rollback_tokens")),
            fallback_count=_coerce_int(runtime_summary.get("fallback_count")),
        )
        return GenerateResponse(
            model_id=request.model_id,
            output_text="".join(output_parts),
            finish_reason="stop",
            usage=usage,
        )

    async def _stream_generate_with_owned_draft_controller(self, request: GenerateRequest) -> AsyncIterator[str]:
        async for chunk in self._iter_owned_draft_controller_chunks(request):
            if chunk:
                yield chunk

    async def _iter_owned_draft_controller_chunks(self, request: GenerateRequest) -> AsyncIterator[str]:
        speculation = request.speculation
        if speculation is None or speculation.mode != SpeculationMode.DRAFT_MODEL:
            raise ConfigurationError("LewLM-owned draft verification requires a draft-model speculation payload.")
        module = import_module("mlx_lm")
        generate = resolve_backend_callable(module, ("generate", "chat", "generate_text"), required=False)
        if generate is None or not self._supports_draft_model_speculation(generate):
            raise ConfigurationError("Installed MLX backend does not expose the primitives required for LewLM-owned draft verification.")
        model, tokenizer = self._client_components(request.model_id)
        if tokenizer is None or not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise ConfigurationError("LewLM-owned draft verification requires a tokenizer with encode/decode support.")
        prompt = _messages_to_prompt(request.messages, tokenizer)
        prompt_tokens = _prompt_token_ids(tokenizer, prompt)
        prompt_value, prompt_cache = self._prepare_prompt_cache_invocation(
            module=module,
            model=model,
            tokenizer=tokenizer,
            request=request,
            prompt_tokens=prompt_tokens,
        )
        generation_options = _mlx_text_generation_options(request.temperature)
        performance_options, feature_usage, generation_controls = self._generation_performance_options(
            generate,
            prompt_tokens=prompt_tokens,
        )
        feature_usage = self._merge_loaded_feature_usage(request.model_id, feature_usage)
        request.metadata["performance_controls"] = self._request_performance_controls(
            model_id=request.model_id,
            generation_controls=generation_controls,
        )
        self._record_feature_usage(prompt_token_count=len(prompt_tokens), feature_usage=feature_usage)
        self._set_speculation_execution_metadata(
            request,
            ownership="lewlm_controller",
            execution_path="lewlm_controller",
            controller="draft_verify",
            drafted_tokens=0,
            accepted_tokens=0,
            verified_tokens=0,
            rejected_tokens=0,
            rollback_tokens=0,
            fallback_count=0,
            prompt_tokens=len(prompt_tokens),
            completion_tokens=0,
        )
        companion_model_id = speculation.companion_model_id or speculation.draft_model_id
        if not companion_model_id:
            raise ConfigurationError("LewLM-owned draft verification requires a loaded draft model.")
        draft_model, draft_tokenizer = self._client_components(companion_model_id)
        completion_tokens: list[int] = []
        completion_text = ""
        cached_prefix_tokens = max(len(prompt_tokens) - 1, 0)
        if not isinstance(prompt_value, list) or not prompt_value:
            raise ConfigurationError("LewLM-owned draft verification requires prompt-cache priming on the selected MLX backend.")
        while len(completion_tokens) < request.max_tokens:
            remaining_tokens = request.max_tokens - len(completion_tokens)
            draft_window = min(max(speculation.num_draft_tokens or 1, 1), remaining_tokens)
            draft_response = self._invoke_generate_response(
                request=request,
                generate=generate,
                model=draft_model,
                tokenizer=draft_tokenizer or tokenizer,
                prompt=prompt + completion_text,
                max_tokens=draft_window,
                generation_options=generation_options,
                callable_key="draft_verify_generate_draft",
            )
            draft_tokens = _suffix_generated_tokens(
                tokenizer=tokenizer,
                prefix_tokens=completion_tokens,
                generated_text=draft_response.output_text,
            )
            if not draft_tokens:
                fallback_response = self._invoke_generate_response(
                    request=request,
                    generate=generate,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt + completion_text,
                    max_tokens=remaining_tokens,
                    generation_options=generation_options,
                    performance_options=performance_options,
                    callable_key="draft_verify_generate_fallback",
                )
                fallback_tokens = _suffix_generated_tokens(
                    tokenizer=tokenizer,
                    prefix_tokens=completion_tokens,
                    generated_text=fallback_response.output_text,
                )
                if not fallback_tokens:
                    break
                completion_tokens.extend(fallback_tokens)
                fallback_text = _decode_generated_tokens(tokenizer=tokenizer, tokens=fallback_tokens)
                completion_text += fallback_text
                self._set_speculation_execution_metadata(
                    request,
                    fallback_count=_coerce_int(self._speculation_runtime_summary(request).get("fallback_count")) + 1,
                    verified_tokens=len(completion_tokens),
                    completion_tokens=len(completion_tokens),
                )
                yield fallback_text
                break
            verify_prompt, cached_prefix_tokens = self._advance_speculation_prompt_cache(
                request=request,
                generate=generate,
                model=model,
                tokenizer=tokenizer,
                prompt_cache=prompt_cache,
                target_tokens=[*prompt_tokens, *completion_tokens],
                cached_prefix_tokens=cached_prefix_tokens,
            )
            verify_response = self._invoke_generate_response(
                request=request,
                generate=generate,
                model=model,
                tokenizer=tokenizer,
                prompt=verify_prompt,
                max_tokens=min(len(draft_tokens), remaining_tokens),
                prompt_cache=prompt_cache,
                generation_options=generation_options,
                performance_options=performance_options,
                callable_key="draft_verify_generate_verify",
            )
            verify_tokens = _suffix_generated_tokens(
                tokenizer=tokenizer,
                prefix_tokens=completion_tokens,
                generated_text=verify_response.output_text,
            )
            if not verify_tokens:
                break
            accepted_tokens = longest_token_prefix(draft_tokens, verify_tokens)
            rejected_tokens = max(len(draft_tokens) - accepted_tokens, 0)
            rollback_tokens = max(rejected_tokens, 0)
            committed_tokens = (
                verify_tokens[: accepted_tokens + 1]
                if accepted_tokens < len(verify_tokens)
                else verify_tokens
            )
            if not committed_tokens:
                break
            completion_tokens.extend(committed_tokens)
            committed_text = _decode_generated_tokens(tokenizer=tokenizer, tokens=committed_tokens)
            completion_text += committed_text
            runtime_summary = self._speculation_runtime_summary(request)
            self._set_speculation_execution_metadata(
                request,
                drafted_tokens=_coerce_int(runtime_summary.get("drafted_tokens")) + len(draft_tokens),
                accepted_tokens=_coerce_int(runtime_summary.get("accepted_tokens")) + accepted_tokens,
                verified_tokens=_coerce_int(runtime_summary.get("verified_tokens")) + len(verify_tokens),
                rejected_tokens=_coerce_int(runtime_summary.get("rejected_tokens")) + rejected_tokens,
                rollback_tokens=_coerce_int(runtime_summary.get("rollback_tokens")) + rollback_tokens,
                completion_tokens=len(completion_tokens),
            )
            yield committed_text
            if len(verify_tokens) < min(len(draft_tokens), remaining_tokens):
                break
        runtime_summary = self._speculation_runtime_summary(request)
        drafted_tokens = _coerce_int(runtime_summary.get("drafted_tokens"))
        accepted_tokens = _coerce_int(runtime_summary.get("accepted_tokens"))
        verified_tokens = _coerce_int(runtime_summary.get("verified_tokens"))
        rejected_tokens = _coerce_int(runtime_summary.get("rejected_tokens"))
        rollback_tokens = _coerce_int(runtime_summary.get("rollback_tokens"))
        fallback_count = _coerce_int(runtime_summary.get("fallback_count"))
        self._record_speculative_usage(
            drafted_tokens=drafted_tokens,
            accepted_tokens=accepted_tokens,
            verified_tokens=verified_tokens,
            rejected_tokens=rejected_tokens,
            rollback_tokens=rollback_tokens,
            fallback_count=fallback_count,
            ownership="lewlm_controller",
        )

    def _advance_speculation_prompt_cache(
        self,
        *,
        request: GenerateRequest,
        generate: Any,
        model: Any,
        tokenizer: Any | None,
        prompt_cache: Any,
        target_tokens: list[int],
        cached_prefix_tokens: int,
    ) -> tuple[list[int], int]:
        if not target_tokens:
            return [], 0
        uncached_prefix_tokens = target_tokens[cached_prefix_tokens:-1]
        if uncached_prefix_tokens:
            self._invoke_generate_response(
                request=request,
                generate=generate,
                model=model,
                tokenizer=tokenizer,
                prompt=list(uncached_prefix_tokens),
                max_tokens=0,
                prompt_cache=prompt_cache,
                callable_key="draft_verify_prefill",
                phase="prefill",
            )
            cached_prefix_tokens = len(target_tokens) - 1
        return [int(target_tokens[-1])], cached_prefix_tokens

    def _speculation_invoke_options(self, *, generate_stream: Any, request: GenerateRequest) -> dict[str, Any]:
        speculation = request.speculation
        if speculation is None:
            return {}
        parameter_names = _callable_parameter_names(generate_stream)
        options: dict[str, Any] = {}
        if speculation.mode == SpeculationMode.DRAFT_MODEL:
            draft_parameter = _first_matching_parameter(parameter_names, _DRAFT_MODEL_PARAMETERS)
            draft_tokens_parameter = _first_matching_parameter(parameter_names, _NUM_DRAFT_TOKENS_PARAMETERS)
            if draft_parameter is None or draft_tokens_parameter is None:
                raise ConfigurationError("Installed MLX backend does not expose draft-model speculation parameters.")
            options[draft_parameter] = self._speculation_companion_client(request)
            options[draft_tokens_parameter] = speculation.num_draft_tokens
            return options
        backend_parameter = speculation.parameters.get("backend_parameter")
        parameter_name = backend_parameter if isinstance(backend_parameter, str) and backend_parameter in parameter_names else None
        if parameter_name is None:
            parameter_name = _first_matching_parameter(
                parameter_names,
                _FRONTIER_SPECULATION_PARAMETER_ALIASES.get(speculation.mode, ()),
            )
        if parameter_name is None:
            raise ConfigurationError(
                f"Installed MLX backend does not expose a compatible `{speculation.mode.value}` parameter.",
            )
        backend_value = speculation.parameters.get("backend_value")
        if speculation.companion_model_id or speculation.draft_model_id:
            backend_value = self._speculation_companion_client(request)
        options[parameter_name] = backend_value
        for key, value in speculation.parameters.items():
            if key in {"backend_parameter", "backend_value"} or key not in parameter_names:
                continue
            options[key] = value
        return options

    def _speculation_passthrough_keys(self, request: GenerateRequest) -> tuple[str, ...]:
        speculation = request.speculation
        if speculation is None:
            return ()
        if speculation.mode == SpeculationMode.DRAFT_MODEL:
            return ("draft_model", "draft", "draft_client", "num_draft_tokens", "draft_tokens")
        keys = [key for key in speculation.parameters if isinstance(key, str)]
        backend_parameter = speculation.parameters.get("backend_parameter")
        if isinstance(backend_parameter, str):
            keys.append(backend_parameter)
        return tuple(dict.fromkeys(keys))

    def _speculation_companion_client(self, request: GenerateRequest) -> Any:
        speculation = request.speculation
        if speculation is None:
            raise KeyError("Speculative generation requested without a speculation payload.")
        companion_model_id = speculation.companion_model_id or speculation.draft_model_id
        if not companion_model_id:
            raise ConfigurationError(
                f"`{speculation.mode.value}` speculative decoding requires a companion model identifier.",
            )
        companion_model, _ = self._client_components(companion_model_id)
        return companion_model

    def _set_speculation_execution_metadata(self, request: GenerateRequest, **values: int | str | bool | None) -> None:
        runtime_summary = request.metadata.get("speculation_runtime")
        if not isinstance(runtime_summary, dict):
            runtime_summary = {}
            request.metadata["speculation_runtime"] = runtime_summary
        runtime_summary.update({key: value for key, value in values.items() if value is not None})
        execution_path = runtime_summary.get("execution_path")
        if isinstance(execution_path, str):
            request.metadata["speculation_execution_path"] = execution_path
        fallback_count = runtime_summary.get("fallback_count")
        if fallback_count is not None:
            request.metadata["speculation_fallback_count"] = _coerce_int(fallback_count)

    @staticmethod
    def _speculation_runtime_summary(request: GenerateRequest) -> dict[str, Any]:
        runtime_summary = request.metadata.get("speculation_runtime")
        return runtime_summary if isinstance(runtime_summary, dict) else {}

    def _record_speculative_usage(
        self,
        *,
        drafted_tokens: int,
        accepted_tokens: int,
        verified_tokens: int,
        rejected_tokens: int,
        rollback_tokens: int,
        fallback_count: int,
        ownership: str,
    ) -> None:
        self._speculative_request_count += 1
        if ownership == "lewlm_controller":
            self._controller_speculative_request_count += 1
        else:
            self._backend_passthrough_speculative_request_count += 1
        self._drafted_token_count += max(drafted_tokens, 0)
        self._accepted_token_count += max(accepted_tokens, 0)
        self._verified_token_count += max(verified_tokens, 0)
        self._rejected_token_count += max(rejected_tokens, 0)
        self._rollback_token_count += max(rollback_tokens, 0)
        self._speculation_fallback_count += max(fallback_count, 0)

    def _merge_loaded_feature_usage(self, model_id: str, feature_usage: dict[str, bool]) -> dict[str, bool]:
        merged = dict(feature_usage)
        for key, value in self._loaded_feature_usage.get(model_id, {}).items():
            merged[key] = merged.get(key, False) or value
        return merged


def _normalize_loaded_client(client: Any) -> tuple[Any, Any | None]:
    if isinstance(client, tuple):
        model = client[0] if len(client) > 0 else None
        tokenizer = client[1] if len(client) > 1 else None
        return model, tokenizer
    if isinstance(client, dict):
        return client.get("model", client), client.get("tokenizer")
    return client, None


def _resolve_semantic_callable(module: Any, capability: CapabilityName):
    if capability == CapabilityName.EMBEDDINGS:
        return resolve_backend_callable(
            module,
            ("embed", "embeddings", "encode", "encode_text", "get_embeddings"),
            required=False,
        )
    if capability == CapabilityName.RERANK:
        return resolve_backend_callable(
            module,
            ("rerank", "rank", "score", "score_documents"),
            required=False,
        )
    return None


def _mlx_text_generation_options(temperature: float) -> dict[str, Any]:
    if temperature <= 0:
        return {}
    try:
        sample_utils = import_module("mlx_lm.sample_utils")
    except ImportError:
        return {"temperature": temperature, "temp": temperature}
    make_sampler = getattr(sample_utils, "make_sampler", None)
    if callable(make_sampler):
        return {"sampler": make_sampler(temp=temperature)}
    return {"temperature": temperature, "temp": temperature}


def _mlx_lm_supports_manifest(source_path: str) -> bool:
    config = _load_manifest_config(source_path)
    if not config:
        return True
    try:
        utils = import_module("mlx_lm.utils")
    except ImportError:
        return True
    get_model_classes = getattr(utils, "get_model_classes", None) or getattr(utils, "_get_classes", None)
    if not callable(get_model_classes):
        return True
    try:
        get_model_classes(config=config)
    except (ImportError, KeyError, TypeError, ValueError):
        return False
    return True


def _load_manifest_config(source_path: str) -> dict[str, Any]:
    path = Path(source_path).expanduser().resolve(strict=False)
    config_path = path / "config.json" if path.is_dir() else path.with_name("config.json")
    if not config_path.exists():
        return {}
    try:
        with config_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _embedding_response_from_result(result: Any, request: EmbeddingRequest) -> EmbeddingResponse:
    if isinstance(result, EmbeddingResponse):
        return result
    usage: dict[str, int] = {}
    payload = result
    if isinstance(result, dict):
        usage = _normalize_usage(result.get("usage"))
        payload = result.get("data", result.get("embeddings", result.get("vectors", result.get("results", result))))
    vectors = _normalize_embedding_vectors(payload)
    prompt_tokens = usage.get("prompt_tokens", sum(max(1, len(text.split())) for text in request.inputs))
    normalized_usage = {
        "prompt_tokens": prompt_tokens,
        "total_tokens": usage.get("total_tokens", prompt_tokens),
    }
    return EmbeddingResponse(
        model_id=request.model_id,
        data=[EmbeddingVector(index=index, embedding=vector) for index, vector in enumerate(vectors)],
        usage=normalized_usage,
    )


def _normalize_embedding_vectors(payload: Any) -> list[list[float]]:
    if payload is None:
        return []
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    if _is_numeric_sequence(payload):
        return [[float(value) for value in payload]]
    vectors: list[list[float]] = []
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            if isinstance(item, EmbeddingVector):
                vectors.append([float(value) for value in item.embedding])
                continue
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, dict):
                vector_payload = item.get("embedding", item.get("vector", item.get("values", [])))
                if hasattr(vector_payload, "tolist"):
                    vector_payload = vector_payload.tolist()
                if _is_numeric_sequence(vector_payload):
                    vectors.append([float(value) for value in vector_payload])
                continue
            if _is_numeric_sequence(item):
                vectors.append([float(value) for value in item])
    return vectors


def _rerank_response_from_result(result: Any, request: RerankRequest) -> RerankResponse:
    if isinstance(result, RerankResponse):
        return result
    payload = result
    if isinstance(result, dict):
        payload = result.get("results", result.get("data", result.get("scores", result)))
    results = _normalize_rerank_results(payload, request)
    if request.top_n is not None:
        results = results[: request.top_n]
    return RerankResponse(model_id=request.model_id, results=results)


def _normalize_rerank_results(payload: Any, request: RerankRequest) -> list[RerankResult]:
    if payload is None:
        return []
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if all(isinstance(item, (int, float)) for item in payload):
            return [
                RerankResult(index=index, relevance_score=float(score), document=request.documents[index])
                for index, score in enumerate(payload)
            ]
        results: list[RerankResult] = []
        for index, item in enumerate(payload):
            if isinstance(item, RerankResult):
                results.append(item)
                continue
            if isinstance(item, dict):
                item_index = item.get("index", index)
                document = item.get("document")
                if not isinstance(document, str) and isinstance(item_index, int) and 0 <= item_index < len(request.documents):
                    document = request.documents[item_index]
                results.append(
                    RerankResult(
                        index=int(item_index),
                        relevance_score=float(item.get("relevance_score", item.get("score", item.get("similarity", 0.0)))),
                        document=document if isinstance(document, str) else None,
                    ),
                )
        return sorted(results, key=lambda item: (-item.relevance_score, item.index))
    return []


def _normalize_usage(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            normalized[key] = int(value)
    return normalized


def _generate_response_from_result(*, result: Any, model_id: str) -> GenerateResponse:
    if isinstance(result, GenerateResponse):
        return result
    if isinstance(result, dict):
        return GenerateResponse(
            model_id=model_id,
            output_text=str(result.get("text", result.get("output_text", ""))),
            finish_reason=str(result.get("finish_reason", "stop")),
            usage=_normalize_usage(result.get("usage")),
        )
    return GenerateResponse(
        model_id=model_id,
        output_text=str(result),
        finish_reason="stop",
        usage={},
    )


def _suffix_generated_tokens(
    *,
    tokenizer: Any | None,
    prefix_tokens: Sequence[int],
    generated_text: str,
) -> list[int]:
    if not generated_text:
        return []
    prefix_text = _decode_generated_tokens(tokenizer=tokenizer, tokens=prefix_tokens)
    combined_tokens = _prompt_token_ids(tokenizer, prefix_text + generated_text)
    normalized_prefix = [int(token) for token in prefix_tokens]
    if len(combined_tokens) >= len(normalized_prefix) and combined_tokens[: len(normalized_prefix)] == normalized_prefix:
        return combined_tokens[len(normalized_prefix) :]
    return _prompt_token_ids(tokenizer, generated_text)


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


def _is_numeric_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and all(
        isinstance(item, (int, float)) for item in value
    )


def _messages_to_prompt(messages: list[Any], tokenizer: Any | None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": message.role, "content": message.content} for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    rendered = []
    for message in messages:
        rendered.append(f"{message.role}: {message.content}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def _mlx_chunk_to_text(chunk: object) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        text = chunk.get("text")
        return text if isinstance(text, str) else ""
    text = getattr(chunk, "text", None)
    return text if isinstance(text, str) else ""


def _mlx_chunk_usage(chunk: object) -> Any:
    if isinstance(chunk, dict):
        return chunk.get("usage")
    return getattr(chunk, "usage", None)


def _resolve_mlx_batch_module(module: Any) -> Any | None:
    if getattr(module, "BatchGenerator", None) is not None or getattr(module, "batch_generate", None) is not None:
        return module
    try:
        return import_module("mlx_lm.generate")
    except ImportError:
        return None


def _resolve_mlx_batch_generate(module: Any) -> Any | None:
    batch_module = _resolve_mlx_batch_module(module)
    if batch_module is None:
        return None
    return resolve_backend_callable(batch_module, ("batch_generate",), required=False)


def _resolve_mlx_batch_generator_class(module: Any) -> Any | None:
    batch_module = _resolve_mlx_batch_module(module)
    if batch_module is None:
        return None
    candidate = getattr(batch_module, "BatchGenerator", None)
    return candidate if callable(candidate) else None


def _mlx_stop_tokens(tokenizer: Any | None) -> list[list[int]] | None:
    if tokenizer is None:
        return None
    eos_tokens = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(eos_tokens, Sequence) and not isinstance(eos_tokens, (str, bytes, bytearray)):
        normalized = [int(token) for token in eos_tokens if isinstance(token, (int, float))]
        return [[token] for token in normalized] if normalized else None
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, (int, float)):
        return [[int(eos_token_id)]]
    return None


def _decode_generated_tokens(*, tokenizer: Any | None, tokens: Sequence[int]) -> str:
    if not tokens:
        return ""
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(list(tokens)))
    return bytes(int(token) for token in tokens).decode("utf-8", errors="ignore")


def _mlx_chunk_from_draft(chunk: object) -> bool:
    if isinstance(chunk, dict):
        return bool(chunk.get("from_draft", chunk.get("draft", chunk.get("is_draft", False))))
    return bool(
        getattr(
            chunk,
            "from_draft",
            getattr(chunk, "draft", getattr(chunk, "is_draft", False)),
        ),
    )


def _callable_parameter_names(callable_obj: Any | None) -> set[str]:
    if callable_obj is None:
        return set()
    try:
        return set(inspect.signature(callable_obj).parameters)
    except (TypeError, ValueError):
        return set()


def _first_matching_parameter(parameter_names: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in parameter_names:
            return candidate
    return None


def _callable_accepts_parameter(callable_obj: Any | None, candidates: tuple[str, ...]) -> bool:
    if callable_obj is None:
        return False
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    if any(candidate in signature.parameters for candidate in candidates):
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _prompt_token_ids(tokenizer: Any | None, prompt: str) -> list[int]:
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(prompt)
        if isinstance(encoded, Sequence) and not isinstance(encoded, (str, bytes, bytearray)):
            return [int(token) for token in encoded if isinstance(token, (int, float))]
    return list(prompt.encode("utf-8"))


def _mlx_cache_helpers() -> dict[str, Any] | None:
    try:
        cache_module = import_module("mlx_lm.models.cache")
    except ImportError:
        return None
    make_prompt_cache = getattr(cache_module, "make_prompt_cache", None)
    trim_prompt_cache = getattr(cache_module, "trim_prompt_cache", None)
    if not callable(make_prompt_cache) or not callable(trim_prompt_cache):
        return None
    return {
        "make_prompt_cache": make_prompt_cache,
        "trim_prompt_cache": trim_prompt_cache,
    }


def _compact_runtime_metrics(**values: int | float | str | bool | None) -> dict[str, int | float | str | bool]:
    return {key: value for key, value in values.items() if value is not None}


def _performance_control_payload(
    *,
    requested: bool,
    supported: bool,
    effective: str,
    reason: str,
    applied_parameters: tuple[str, ...] = (),
    rejected_parameters: tuple[str, ...] = (),
    **details: int | float | str | bool | None,
) -> dict[str, Any]:
    return {
        "requested": requested,
        "supported": supported,
        "effective": effective,
        "reason": reason,
        "applied_parameters": list(applied_parameters),
        "rejected_parameters": list(rejected_parameters),
        **_compact_runtime_metrics(**details),
    }
