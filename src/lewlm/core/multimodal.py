"""Capability orchestration for embeddings, rerank, and audio APIs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from io import BytesIO
import math
import time
from typing import TypeVar, cast
from uuid import uuid4
import wave

from lewlm.core.contracts import (
    AudioSpeechRequest,
    AudioSpeechResponse,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioTranscriptionSegment,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelManifest,
    RerankRequest,
    RerankResponse,
    RoutingDecision,
    RuntimeContract,
    utc_now,
)
from lewlm.core.errors import ConfigurationError
from lewlm.core.execution_metadata import (
    ExecutionMetadata,
    ExecutionTimingMetadata,
    build_routed_execution_metadata,
    milliseconds_from_seconds,
)
from lewlm.documents.ingest.models import DocumentChunk, IngestedDocumentSource
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.routing.service import ModelRouter
from lewlm.runtime.request_coalescer import InFlightRequestCoalescer
from lewlm.runtime.response_cache import RuntimeResponseCache
from lewlm.runtime.scheduler import RuntimeRequestAdmission, RuntimeRequestScheduler
from lewlm.telemetry.runtime_metrics import RuntimeMetricsRecorder


ResponseT = TypeVar(
    "ResponseT",
    EmbeddingResponse,
    RerankResponse,
    AudioTranscriptionResponse,
    AudioSpeechResponse,
)

_AUDIO_TRANSCRIPTION_CHUNK_SECONDS = 1.0
_EMBEDDING_BATCH_WINDOW_SECONDS = 0.01


@dataclass(slots=True)
class EmbeddingExecution:
    request_id: str
    created_at: int
    response: EmbeddingResponse
    routing: RoutingDecision
    metadata: ExecutionMetadata


@dataclass(slots=True)
class RerankExecution:
    request_id: str
    created_at: int
    response: RerankResponse
    routing: RoutingDecision
    metadata: ExecutionMetadata


@dataclass(slots=True)
class AudioTranscriptionExecution:
    request_id: str
    created_at: int
    response: AudioTranscriptionResponse
    routing: RoutingDecision
    metadata: ExecutionMetadata


@dataclass(slots=True)
class AudioSpeechExecution:
    request_id: str
    created_at: int
    response: AudioSpeechResponse
    routing: RoutingDecision
    metadata: ExecutionMetadata


@dataclass(slots=True)
class RetrievalStageExecution:
    request_id: str
    created_at: int
    model_id: str
    routing: RoutingDecision
    metadata: ExecutionMetadata
    usage: dict[str, int] | None = None


@dataclass(slots=True)
class RetrievalContextItem:
    rank: int
    score: float
    chunk: DocumentChunk
    source: IngestedDocumentSource | None
    embedding_score: float | None = None
    rerank_score: float | None = None


@dataclass(slots=True)
class RetrievalContextExecution:
    request_id: str
    created_at: int
    query: str
    strategy: str
    items: list[RetrievalContextItem]
    sources: list[IngestedDocumentSource]
    metadata: ExecutionMetadata
    embedding_stage: RetrievalStageExecution | None = None
    rerank_stage: RetrievalStageExecution | None = None


@dataclass(slots=True)
class _AudioTranscriptionChunk:
    index: int
    total: int
    audio_bytes: bytes
    start_seconds: float
    end_seconds: float


@dataclass(slots=True)
class _AudioTranscriptionChunkPlan:
    chunks: tuple[_AudioTranscriptionChunk, ...]
    duration_seconds: float | None

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def is_chunked(self) -> bool:
        return self.chunk_count > 1


@dataclass(slots=True)
class _EmbeddingBatchResult:
    response: EmbeddingResponse
    load_seconds: float
    execution_seconds: float
    batch_size: int
    cache_hit: bool = False


@dataclass(slots=True)
class _PendingEmbeddingBatchItem:
    request_id: str
    requested_model_id: str | None
    created_at: int
    cache_key: str
    inputs: list[str]
    routing: RoutingDecision


@dataclass(slots=True)
class _ScoredRetrievalCandidate:
    chunk: DocumentChunk
    source: IngestedDocumentSource | None
    original_index: int
    score: float
    embedding_score: float | None = None
    rerank_score: float | None = None


class _EmbeddingBatchExecutionError(Exception):
    def __init__(
        self,
        error: Exception,
        *,
        load_seconds: float,
        execution_seconds: float,
        batch_size: int,
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.load_seconds = load_seconds
        self.execution_seconds = execution_seconds
        self.batch_size = batch_size


class MultimodalOrchestrator:
    """Execute multimodal capability requests while publishing lifecycle events."""

    def __init__(
        self,
        *,
        model_router: ModelRouter,
        event_bus: EventBus,
        runtime_request_scheduler: RuntimeRequestScheduler,
        model_load_scheduler: RuntimeRequestScheduler,
        runtime_metrics_recorder: RuntimeMetricsRecorder,
        runtime_response_cache: RuntimeResponseCache,
        runtime_request_coalescer: InFlightRequestCoalescer[object],
    ) -> None:
        self.model_router = model_router
        self.event_bus = event_bus
        self.runtime_request_scheduler = runtime_request_scheduler
        self.model_load_scheduler = model_load_scheduler
        self.runtime_metrics_recorder = runtime_metrics_recorder
        self.runtime_response_cache = runtime_response_cache
        self.runtime_request_coalescer = runtime_request_coalescer
        self._pending_embedding_batches: dict[str, list[_PendingEmbeddingBatchItem]] = {}
        self._embedding_batch_tasks: dict[str, asyncio.Task[None]] = {}

    async def embed(self, *, model_id: str | None, inputs: list[str]) -> EmbeddingExecution:
        manifest, runtime, routing = self.model_router.route_embeddings(model_id, inputs=inputs)
        request_id = str(uuid4())
        created_at = int(utc_now().timestamp())
        cache_key = self.runtime_response_cache.embedding_cache_key(model_id=manifest.model_id, inputs=inputs)
        is_owner, shared_future = self.runtime_request_coalescer.claim(cache_key)
        if not is_owner:
            try:
                shared_result = cast(_EmbeddingBatchResult, await shared_future)
            except _EmbeddingBatchExecutionError as exc:
                await self._publish_embedding_request_failed(
                    request_id=request_id,
                    manifest=manifest,
                    runtime=runtime,
                    error=exc.error,
                    batch_size=exc.batch_size,
                )
                self._record_embedding_failure(
                    manifest=manifest,
                    runtime=runtime,
                    inputs=inputs,
                    load_seconds=0.0,
                    execution_seconds=0.0,
                    extra_measurements={"coalesced_requests": 1},
                )
                raise exc.error from exc
            coalesced_response = shared_result.response.model_copy(deep=True)
            await self._publish_coalesced_request_events(
                request_id=request_id,
                requested_model_id=model_id,
                manifest=manifest,
                runtime=runtime,
                capability="embeddings",
            )
            self._record_embedding_success(
                manifest=manifest,
                runtime=runtime,
                inputs=inputs,
                response=coalesced_response,
                load_seconds=0.0,
                execution_seconds=0.0,
                extra_measurements={"coalesced_requests": 1},
            )
            return EmbeddingExecution(
                request_id=request_id,
                created_at=created_at,
                response=coalesced_response,
                routing=routing,
                metadata=build_routed_execution_metadata(
                    request_id=request_id,
                    created=created_at,
                    requested_model_id=model_id,
                    routing=routing,
                    result_origin="coalesced",
                ),
            )
        try:
            cached_response = self.runtime_response_cache.get_embedding_response_by_cache_key(cache_key)
            if cached_response is not None:
                cached_copy = cached_response.model_copy(deep=True)
                await self._publish_cached_request_events(
                    request_id=request_id,
                    requested_model_id=model_id,
                    manifest=manifest,
                    runtime=runtime,
                    capability="embeddings",
                )
                self._record_embedding_success(
                    manifest=manifest,
                    runtime=runtime,
                    inputs=inputs,
                    response=cached_copy,
                    load_seconds=0.0,
                    execution_seconds=0.0,
                    extra_measurements={"cache_hits": 1},
                )
                self.runtime_request_coalescer.resolve(
                    cache_key,
                    _EmbeddingBatchResult(
                        response=cached_copy.model_copy(deep=True),
                        load_seconds=0.0,
                        execution_seconds=0.0,
                        batch_size=1,
                        cache_hit=True,
                    ),
                )
                return EmbeddingExecution(
                    request_id=request_id,
                    created_at=created_at,
                    response=cached_copy.model_copy(deep=True),
                    routing=routing,
                    metadata=build_routed_execution_metadata(
                        request_id=request_id,
                        created=created_at,
                        requested_model_id=model_id,
                        routing=routing,
                        result_origin="cache_hit",
                    ),
                )
            await self._enqueue_embedding_batch(
                manifest=manifest,
                runtime=runtime,
                item=_PendingEmbeddingBatchItem(
                    request_id=request_id,
                    requested_model_id=model_id,
                    created_at=created_at,
                    cache_key=cache_key,
                    inputs=list(inputs),
                    routing=routing,
                ),
            )
            batch_result = cast(_EmbeddingBatchResult, await shared_future)
            response = batch_result.response.model_copy(deep=True)
            await self._publish_embedding_request_completed(
                request_id=request_id,
                manifest=manifest,
                runtime=runtime,
                batch_size=batch_result.batch_size,
            )
            success_measurements: dict[str, int] = {"cache_misses": 1}
            if batch_result.batch_size > 1:
                success_measurements["batched_requests"] = 1
                success_measurements["batch_size"] = batch_result.batch_size
            self._record_embedding_success(
                manifest=manifest,
                runtime=runtime,
                inputs=inputs,
                response=response,
                load_seconds=batch_result.load_seconds,
                execution_seconds=batch_result.execution_seconds,
                extra_measurements=success_measurements,
            )
            return EmbeddingExecution(
                request_id=request_id,
                created_at=created_at,
                response=response,
                routing=routing,
                metadata=build_routed_execution_metadata(
                    request_id=request_id,
                    created=created_at,
                    requested_model_id=model_id,
                    routing=routing,
                    load_milliseconds=milliseconds_from_seconds(batch_result.load_seconds),
                    execute_milliseconds=milliseconds_from_seconds(batch_result.execution_seconds),
                ),
            )
        except _EmbeddingBatchExecutionError as exc:
            await self._publish_embedding_request_failed(
                request_id=request_id,
                manifest=manifest,
                runtime=runtime,
                error=exc.error,
                batch_size=exc.batch_size,
            )
            failure_measurements: dict[str, int] = {"cache_misses": 1}
            if exc.batch_size > 1:
                failure_measurements["batched_requests"] = 1
                failure_measurements["batch_size"] = exc.batch_size
            self._record_embedding_failure(
                manifest=manifest,
                runtime=runtime,
                inputs=inputs,
                load_seconds=exc.load_seconds,
                execution_seconds=exc.execution_seconds,
                extra_measurements=failure_measurements,
            )
            raise exc.error from exc
        except Exception as exc:
            self.runtime_request_coalescer.reject(cache_key, exc)
            raise

    async def rerank(
        self,
        *,
        model_id: str | None,
        query: str,
        documents: list[str],
        top_n: int | None,
    ) -> RerankExecution:
        manifest, runtime, routing = self.model_router.route_rerank(
            model_id,
            query=query,
            documents=documents,
        )
        request_id = str(uuid4())
        cache_key = self.runtime_response_cache.rerank_cache_key(
            model_id=manifest.model_id,
            query=query,
            documents=documents,
            top_n=top_n,
        )
        is_owner, shared_future = self.runtime_request_coalescer.claim(cache_key)
        if not is_owner:
            response = await shared_future
            created_at = int(utc_now().timestamp())
            coalesced_response = response.model_copy(deep=True)
            await self._publish_coalesced_request_events(
                request_id=request_id,
                requested_model_id=model_id,
                manifest=manifest,
                runtime=runtime,
                capability="rerank",
            )
            self.runtime_metrics_recorder.record_success(
                model_id=manifest.model_id,
                runtime=runtime.name,
                capability="rerank",
                load_seconds=0.0,
                execution_seconds=0.0,
                measurements={
                    "document_count": len(documents),
                    "query_characters": len(query),
                    "document_characters": sum(len(item) for item in documents),
                    "result_count": len(coalesced_response.results),
                    "coalesced_requests": 1,
                },
            )
            return RerankExecution(
                request_id=request_id,
                created_at=created_at,
                response=coalesced_response,
                routing=routing,
                metadata=build_routed_execution_metadata(
                    request_id=request_id,
                    created=created_at,
                    requested_model_id=model_id,
                    routing=routing,
                    result_origin="coalesced",
                ),
            )
        try:
            cached_response = self.runtime_response_cache.get_rerank_response_by_cache_key(cache_key)
            if cached_response is not None:
                created_at = int(utc_now().timestamp())
                cached_copy = cached_response.model_copy(deep=True)
                await self._publish_cached_request_events(
                    request_id=request_id,
                    requested_model_id=model_id,
                    manifest=manifest,
                    runtime=runtime,
                    capability="rerank",
                )
                self.runtime_metrics_recorder.record_success(
                    model_id=manifest.model_id,
                    runtime=runtime.name,
                    capability="rerank",
                    load_seconds=0.0,
                    execution_seconds=0.0,
                    measurements={
                        "document_count": len(documents),
                        "query_characters": len(query),
                        "document_characters": sum(len(item) for item in documents),
                        "result_count": len(cached_copy.results),
                        "cache_hits": 1,
                    },
                )
                self.runtime_request_coalescer.resolve(cache_key, cached_copy)
                return RerankExecution(
                    request_id=request_id,
                    created_at=created_at,
                    response=cached_copy.model_copy(deep=True),
                    routing=routing,
                    metadata=build_routed_execution_metadata(
                        request_id=request_id,
                        created=created_at,
                        requested_model_id=model_id,
                        routing=routing,
                        result_origin="cache_hit",
                    ),
                )
            request_id, created_at, response, routing, metadata = await self._execute(
                capability="rerank",
                requested_model_id=model_id,
                manifest=manifest,
                runtime=runtime,
                routing=routing,
                request_id=request_id,
                request_measurements={
                    "document_count": len(documents),
                    "query_characters": len(query),
                    "document_characters": sum(len(item) for item in documents),
                    "cache_misses": 1,
                },
                response_measurements=lambda response: {"result_count": len(response.results)},
                invoke=lambda manifest, runtime, resolved_request_id: runtime.rerank(
                    RerankRequest(
                        model_id=manifest.model_id,
                        query=query,
                        documents=documents,
                        top_n=top_n,
                        request_id=resolved_request_id,
                    ),
                ),
            )
            self.runtime_response_cache.put_rerank_response(
                model_id=manifest.model_id,
                query=query,
                documents=documents,
                top_n=top_n,
                response=response,
            )
            self.runtime_request_coalescer.resolve(cache_key, response.model_copy(deep=True))
            return RerankExecution(
                request_id=request_id,
                created_at=created_at,
                response=response,
                routing=routing,
                metadata=metadata,
            )
        except Exception as exc:
            self.runtime_request_coalescer.reject(cache_key, exc)
            raise

    async def retrieve_context(
        self,
        *,
        query: str,
        candidate_chunks: list[DocumentChunk],
        candidate_sources: list[IngestedDocumentSource] | None = None,
        top_k: int,
        use_embeddings: bool = True,
        use_rerank: bool = True,
        embedding_model_id: str | None = None,
        rerank_model_id: str | None = None,
    ) -> RetrievalContextExecution:
        if not candidate_chunks:
            raise ConfigurationError(
                "Retrieval requests require at least one candidate chunk.",
                details={"candidate_count": 0},
            )
        if top_k < 1:
            raise ConfigurationError("Retrieval requests require top_k >= 1.", details={"top_k": top_k})
        if not use_embeddings and not use_rerank:
            raise ConfigurationError(
                "Retrieval requests must enable embeddings, rerank, or both.",
                details={"use_embeddings": use_embeddings, "use_rerank": use_rerank},
            )

        request_id = str(uuid4())
        created_at = int(utc_now().timestamp())
        documents = [chunk.text for chunk in candidate_chunks]
        candidate_sources = list(candidate_sources or ())
        embedding_stage: RetrievalStageExecution | None = None
        rerank_stage: RetrievalStageExecution | None = None
        embedding_scores: dict[int, float] | None = None
        rerank_scores: dict[int, float] | None = None

        if use_embeddings:
            embedding_execution = await self.embed(
                model_id=embedding_model_id,
                inputs=[query, *documents],
            )
            if len(embedding_execution.response.data) != len(candidate_chunks) + 1:
                raise ValueError(
                    "Embedding retrieval scoring returned an unexpected vector count "
                    f"for {len(candidate_chunks)} candidates."
                )
            query_embedding = embedding_execution.response.data[0].embedding
            embedding_scores = {
                index: _cosine_similarity(query_embedding, datum.embedding)
                for index, datum in enumerate(embedding_execution.response.data[1:])
            }
            embedding_stage = RetrievalStageExecution(
                request_id=embedding_execution.request_id,
                created_at=embedding_execution.created_at,
                model_id=embedding_execution.response.model_id,
                routing=embedding_execution.routing,
                metadata=embedding_execution.metadata,
                usage=embedding_execution.response.usage,
            )

        if use_rerank:
            rerank_execution = await self.rerank(
                model_id=rerank_model_id,
                query=query,
                documents=documents,
                top_n=None,
            )
            rerank_scores = {
                item.index: item.relevance_score
                for item in rerank_execution.response.results
            }
            rerank_stage = RetrievalStageExecution(
                request_id=rerank_execution.request_id,
                created_at=rerank_execution.created_at,
                model_id=rerank_execution.response.model_id,
                routing=rerank_execution.routing,
                metadata=rerank_execution.metadata,
            )

        ranked_candidates = _rank_retrieval_candidates(
            candidate_chunks=candidate_chunks,
            candidate_sources=candidate_sources,
            embedding_scores=embedding_scores,
            rerank_scores=rerank_scores,
        )
        limited_candidates = ranked_candidates[: min(top_k, len(ranked_candidates))]
        items = [
            RetrievalContextItem(
                rank=index,
                score=candidate.score,
                chunk=candidate.chunk,
                source=candidate.source,
                embedding_score=candidate.embedding_score,
                rerank_score=candidate.rerank_score,
            )
            for index, candidate in enumerate(limited_candidates, start=1)
        ]
        metadata = _build_retrieval_metadata(
            request_id=request_id,
            created_at=created_at,
            embedding_stage=embedding_stage,
            rerank_stage=rerank_stage,
        )
        return RetrievalContextExecution(
            request_id=request_id,
            created_at=created_at,
            query=query,
            strategy=_retrieval_strategy_label(
                use_embeddings=use_embeddings,
                use_rerank=use_rerank,
            ),
            items=items,
            sources=_matched_retrieval_sources(items),
            metadata=metadata,
            embedding_stage=embedding_stage,
            rerank_stage=rerank_stage,
        )

    async def _enqueue_embedding_batch(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        item: _PendingEmbeddingBatchItem,
    ) -> None:
        queue = self._pending_embedding_batches.setdefault(manifest.model_id, [])
        queue.append(item)
        task = self._embedding_batch_tasks.get(manifest.model_id)
        if task is None or task.done():
            self._embedding_batch_tasks[manifest.model_id] = asyncio.create_task(
                self._drain_embedding_batches(manifest=manifest, runtime=runtime),
            )

    async def _drain_embedding_batches(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
    ) -> None:
        model_id = manifest.model_id
        try:
            while True:
                await asyncio.sleep(_EMBEDDING_BATCH_WINDOW_SECONDS)
                items = self._pending_embedding_batches.pop(model_id, [])
                if not items:
                    return
                await self._execute_embedding_batch(manifest=manifest, runtime=runtime, items=items)
                if not self._pending_embedding_batches.get(model_id):
                    return
        finally:
            self._embedding_batch_tasks.pop(model_id, None)
            if not self._pending_embedding_batches.get(model_id):
                self._pending_embedding_batches.pop(model_id, None)

    async def _execute_embedding_batch(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        items: list[_PendingEmbeddingBatchItem],
    ) -> None:
        batch_size = len(items)
        batch_payload = self._embedding_batch_payload(batch_size)
        try:
            admission = await self.runtime_request_scheduler.acquire()
        except Exception as exc:
            failure = _EmbeddingBatchExecutionError(
                exc,
                load_seconds=0.0,
                execution_seconds=0.0,
                batch_size=batch_size,
            )
            for item in items:
                self.runtime_request_coalescer.reject(item.cache_key, failure)
            return
        try:
            if admission.was_queued:
                for item in items:
                    await self._publish(
                        EventType.REQUEST_QUEUED,
                        {
                            "request_id": item.request_id,
                            "model_id": manifest.model_id,
                            "runtime": runtime.name,
                            "wait_seconds": admission.wait_seconds,
                            "capability": "embeddings",
                            **batch_payload,
                        },
                    )
            for item in items:
                await self._publish(
                    EventType.REQUEST_ACCEPTED,
                    {
                        "request_id": item.request_id,
                        "requested_model_id": item.requested_model_id,
                        "model_id": manifest.model_id,
                        "capability": "embeddings",
                        **batch_payload,
                    },
                )
            load_started_at = time.perf_counter()
            load_admission = await self._acquire_model_load_admission_for_batch(
                manifest=manifest,
                runtime=runtime,
                items=items,
                capability="embeddings",
                batch_payload=batch_payload,
            )
            for item in items:
                await self._publish(
                    EventType.MODEL_LOADING,
                    {
                        "request_id": item.request_id,
                        "model_id": manifest.model_id,
                        "runtime": runtime.name,
                        "capability": "embeddings",
                        **batch_payload,
                    },
                )
            await runtime.load_model(manifest)
            await self.model_router.runtime_catalog.prepare_runtime_for_request(
                manifest,
                runtime,
                policy=self.model_router.settings.runtime_policy,
            )
            load_seconds = time.perf_counter() - load_started_at
            for item in items:
                await self._publish(
                    EventType.MODEL_LOADED,
                    {
                        "request_id": item.request_id,
                        "model_id": manifest.model_id,
                        "runtime": runtime.name,
                        "capability": "embeddings",
                        **batch_payload,
                    },
                )
            execution_started_at = time.perf_counter()
            response = await runtime.embed(
                EmbeddingRequest(
                    model_id=manifest.model_id,
                    inputs=[entry for item in items for entry in item.inputs],
                    request_id=items[0].request_id,
                ),
            )
            execution_seconds = time.perf_counter() - execution_started_at
        except Exception as exc:
            now = time.perf_counter()
            load_seconds = now - load_started_at if "load_started_at" in locals() else 0.0
            execution_seconds = now - execution_started_at if "execution_started_at" in locals() else 0.0
            if "execution_started_at" in locals():
                load_seconds = max(load_seconds - execution_seconds, 0.0)
            failure = _EmbeddingBatchExecutionError(
                exc,
                load_seconds=load_seconds / batch_size if batch_size else 0.0,
                execution_seconds=execution_seconds / batch_size if batch_size else 0.0,
                batch_size=batch_size,
            )
            for item in items:
                self.runtime_request_coalescer.reject(item.cache_key, failure)
        else:
            try:
                self._resolve_embedding_batch_items(
                    manifest=manifest,
                    items=items,
                    response=response,
                    load_seconds=load_seconds,
                    execution_seconds=execution_seconds,
                )
            except Exception as exc:
                failure = _EmbeddingBatchExecutionError(
                    exc,
                    load_seconds=load_seconds / batch_size if batch_size else 0.0,
                    execution_seconds=execution_seconds / batch_size if batch_size else 0.0,
                    batch_size=batch_size,
                )
                for item in items:
                    self.runtime_request_coalescer.reject(item.cache_key, failure)
        finally:
            await self.model_router.runtime_catalog.finalize_runtime_for_request(
                manifest,
                runtime,
                policy=self.model_router.settings.runtime_policy,
            )
            if "load_admission" in locals() and load_admission is not None:
                load_admission.release()
            admission.release()

    def _resolve_embedding_batch_items(
        self,
        *,
        manifest: ModelManifest,
        items: list[_PendingEmbeddingBatchItem],
        response: EmbeddingResponse,
        load_seconds: float,
        execution_seconds: float,
    ) -> None:
        total_vectors = sum(len(item.inputs) for item in items)
        if len(response.data) != total_vectors:
            raise ValueError(
                f"Embedding batch for {manifest.model_id} returned {len(response.data)} vectors for {total_vectors} inputs."
            )
        usage_chunks = self._split_embedding_usage(response.usage, items)
        offset = 0
        per_request_load_seconds = load_seconds / len(items)
        per_request_execution_seconds = execution_seconds / len(items)
        for index, item in enumerate(items):
            vector_slice = response.data[offset : offset + len(item.inputs)]
            offset += len(item.inputs)
            item_response = EmbeddingResponse(
                model_id=response.model_id,
                data=[
                    vector.model_copy(update={"index": item_index})
                    for item_index, vector in enumerate(vector_slice)
                ],
                usage=usage_chunks[index],
            )
            self.runtime_response_cache.put_embedding_response(
                model_id=manifest.model_id,
                inputs=item.inputs,
                response=item_response,
            )
            self.runtime_request_coalescer.resolve(
                item.cache_key,
                _EmbeddingBatchResult(
                    response=item_response,
                    load_seconds=per_request_load_seconds,
                    execution_seconds=per_request_execution_seconds,
                    batch_size=len(items),
                ),
            )

    def _split_embedding_usage(
        self,
        usage: dict[str, int],
        items: list[_PendingEmbeddingBatchItem],
    ) -> list[dict[str, int]]:
        if not usage:
            return [{} for _ in items]
        weights = [max(sum(len(value) for value in item.inputs), len(item.inputs), 1) for item in items]
        total_weight = sum(weights)
        allocations: list[dict[str, int]] = [{} for _ in items]
        for key, total in usage.items():
            remaining = total
            remaining_weight = total_weight
            for index, weight in enumerate(weights):
                if index == len(weights) - 1 or remaining_weight <= 0:
                    share = remaining
                else:
                    share = int(round(total * (weight / total_weight)))
                    share = min(share, remaining)
                allocations[index][key] = share
                remaining -= share
                remaining_weight -= weight
        return allocations

    def _embedding_batch_payload(self, batch_size: int) -> dict[str, object]:
        if batch_size <= 1:
            return {}
        return {"batched": True, "batch_size": batch_size}

    async def _publish_embedding_request_completed(
        self,
        *,
        request_id: str,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        batch_size: int,
    ) -> None:
        await self._publish(
            EventType.REQUEST_COMPLETED,
            {
                "request_id": request_id,
                "model_id": manifest.model_id,
                "runtime": runtime.name,
                "capability": "embeddings",
                **self._embedding_batch_payload(batch_size),
            },
        )

    async def _publish_embedding_request_failed(
        self,
        *,
        request_id: str,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        error: Exception,
        batch_size: int,
    ) -> None:
        await self._publish(
            EventType.REQUEST_FAILED,
            {
                "request_id": request_id,
                "model_id": manifest.model_id,
                "runtime": runtime.name,
                "capability": "embeddings",
                "error": str(error),
                **self._embedding_batch_payload(batch_size),
            },
        )

    def _record_embedding_success(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        inputs: list[str],
        response: EmbeddingResponse,
        load_seconds: float,
        execution_seconds: float,
        extra_measurements: dict[str, int] | None = None,
    ) -> None:
        self.runtime_metrics_recorder.record_success(
            model_id=manifest.model_id,
            runtime=runtime.name,
            capability="embeddings",
            load_seconds=load_seconds,
            execution_seconds=execution_seconds,
            usage=response.usage,
            measurements={
                "input_count": len(inputs),
                "input_characters": sum(len(item) for item in inputs),
                "vector_count": len(response.data),
                "vector_dimensions": len(response.data[0].embedding) if response.data else 0,
                **(extra_measurements or {}),
            },
        )

    def _record_embedding_failure(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        inputs: list[str],
        load_seconds: float,
        execution_seconds: float,
        extra_measurements: dict[str, int] | None = None,
    ) -> None:
        self.runtime_metrics_recorder.record_failure(
            model_id=manifest.model_id,
            runtime=runtime.name,
            capability="embeddings",
            load_seconds=load_seconds,
            execution_seconds=execution_seconds,
            measurements={
                "input_count": len(inputs),
                "input_characters": sum(len(item) for item in inputs),
                **(extra_measurements or {}),
            },
        )

    async def transcribe_audio(
        self,
        *,
        model_id: str | None,
        audio_bytes: bytes,
        file_name: str,
        language: str | None,
        prompt: str | None,
    ) -> AudioTranscriptionExecution:
        manifest, runtime, routing = self.model_router.route_audio_transcription(model_id)
        request_id = str(uuid4())
        created_at = int(utc_now().timestamp())
        chunk_plan = _plan_audio_transcription_chunks(audio_bytes)
        total_progress_steps = (chunk_plan.chunk_count * 2) + 2 if chunk_plan.is_chunked else 2
        cache_key = self.runtime_response_cache.audio_transcription_cache_key(
            model_id=manifest.model_id,
            audio_bytes=audio_bytes,
            file_name=file_name,
            language=language,
            prompt=prompt,
        )
        request_measurements = {
            "audio_input_bytes": len(audio_bytes),
            "prompt_characters": len(prompt or ""),
            "chunk_count": chunk_plan.chunk_count,
        }

        async def on_accepted(resolved_request_id: str) -> None:
            await self._publish(
                EventType.AUDIO_TRANSCRIPTION_STARTED,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "file_name": file_name,
                    "language": language,
                    "chunk_count": chunk_plan.chunk_count,
                },
            )
            await self._publish_progress(
                request_id=resolved_request_id,
                operation="audio.transcription",
                stage="chunks_planned" if chunk_plan.is_chunked else "input_buffered",
                completed_steps=1,
                total_steps=total_progress_steps,
                file_name=file_name,
                audio_input_bytes=len(audio_bytes),
                prompt_characters=len(prompt or ""),
                chunk_count=chunk_plan.chunk_count,
                duration_seconds=chunk_plan.duration_seconds,
            )

        async def on_success(resolved_request_id: str, response: AudioTranscriptionResponse) -> None:
            await self._publish_progress(
                request_id=resolved_request_id,
                operation="audio.transcription",
                stage="segments_ready",
                completed_steps=total_progress_steps,
                total_steps=total_progress_steps,
                chunk_count=chunk_plan.chunk_count,
                segment_count=len(response.segments),
                output_characters=len(response.text),
                duration_seconds=response.duration_seconds,
            )
            await self._publish(
                EventType.AUDIO_TRANSCRIPTION_COMPLETED,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "file_name": file_name,
                    "language": response.language,
                    "chunk_count": chunk_plan.chunk_count,
                    "segment_count": len(response.segments),
                    "duration_seconds": response.duration_seconds,
                },
            )

        async def on_failure(resolved_request_id: str, exc: Exception) -> None:
            await self._publish(
                EventType.AUDIO_TRANSCRIPTION_FAILED,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "file_name": file_name,
                    "language": language,
                    "error": str(exc),
                },
            )

        is_owner, shared_future = self.runtime_request_coalescer.claim(cache_key)
        if not is_owner:
            try:
                shared_response = cast(AudioTranscriptionResponse, await shared_future)
            except Exception as exc:
                await on_failure(request_id, exc)
                self.runtime_metrics_recorder.record_failure(
                    model_id=manifest.model_id,
                    runtime=runtime.name,
                    capability="audio_transcription",
                    load_seconds=0.0,
                    execution_seconds=0.0,
                    measurements={**request_measurements, "coalesced_requests": 1},
                )
                await self._publish(
                    EventType.REQUEST_FAILED,
                    {"request_id": request_id, "model_id": manifest.model_id, "error": str(exc)},
                )
                raise
            coalesced_response = shared_response.model_copy(deep=True)
            await self._publish_coalesced_request_events(
                request_id=request_id,
                requested_model_id=model_id,
                manifest=manifest,
                runtime=runtime,
                capability="audio_transcription",
                after_accepted=lambda: on_accepted(request_id),
                before_completed=lambda: on_success(request_id, coalesced_response),
            )
            self.runtime_metrics_recorder.record_success(
                model_id=manifest.model_id,
                runtime=runtime.name,
                capability="audio_transcription",
                load_seconds=0.0,
                execution_seconds=0.0,
                measurements={
                    **request_measurements,
                    "segment_count": len(coalesced_response.segments),
                    "output_characters": len(coalesced_response.text),
                    "coalesced_requests": 1,
                },
            )
            return AudioTranscriptionExecution(
                request_id=request_id,
                created_at=created_at,
                response=coalesced_response,
                routing=routing,
                metadata=build_routed_execution_metadata(
                    request_id=request_id,
                    created=created_at,
                    requested_model_id=model_id,
                    routing=routing,
                    result_origin="coalesced",
                ),
            )
        try:
            cached_response = self.runtime_response_cache.get_audio_transcription_response_by_cache_key(cache_key)
            if cached_response is not None:
                cached_copy = cached_response.model_copy(deep=True)
                await self._publish_cached_request_events(
                    request_id=request_id,
                    requested_model_id=model_id,
                    manifest=manifest,
                    runtime=runtime,
                    capability="audio_transcription",
                    after_accepted=lambda: on_accepted(request_id),
                    before_completed=lambda: on_success(request_id, cached_copy),
                )
                self.runtime_metrics_recorder.record_success(
                    model_id=manifest.model_id,
                    runtime=runtime.name,
                    capability="audio_transcription",
                    load_seconds=0.0,
                    execution_seconds=0.0,
                    measurements={
                        **request_measurements,
                        "segment_count": len(cached_copy.segments),
                        "output_characters": len(cached_copy.text),
                        "cache_hits": 1,
                    },
                )
                self.runtime_request_coalescer.resolve(cache_key, cached_copy.model_copy(deep=True))
                return AudioTranscriptionExecution(
                    request_id=request_id,
                    created_at=created_at,
                    response=cached_copy,
                    routing=routing,
                    metadata=build_routed_execution_metadata(
                        request_id=request_id,
                        created=created_at,
                        requested_model_id=model_id,
                        routing=routing,
                        result_origin="cache_hit",
                    ),
                )
            request_id, created_at, response, routing, metadata = await self._execute(
                capability="audio_transcription",
                requested_model_id=model_id,
                manifest=manifest,
                runtime=runtime,
                routing=routing,
                request_id=request_id,
                request_measurements={**request_measurements, "cache_misses": 1},
                response_measurements=lambda response: {
                    "segment_count": len(response.segments),
                    "output_characters": len(response.text),
                },
                invoke=lambda manifest, runtime, resolved_request_id: self._transcribe_audio_with_progress(
                    manifest=manifest,
                    runtime=runtime,
                    request_id=resolved_request_id,
                    file_name=file_name,
                    language=language,
                    prompt=prompt,
                    chunk_plan=chunk_plan,
                    total_progress_steps=total_progress_steps,
                    request_measurements=request_measurements,
                ),
                on_accepted=on_accepted,
                on_success=on_success,
                on_failure=on_failure,
            )
            self.runtime_response_cache.put_audio_transcription_response(
                model_id=manifest.model_id,
                audio_bytes=audio_bytes,
                file_name=file_name,
                language=language,
                prompt=prompt,
                response=response,
            )
            self.runtime_request_coalescer.resolve(cache_key, response.model_copy(deep=True))
            return AudioTranscriptionExecution(
                request_id=request_id,
                created_at=created_at,
                response=response,
                routing=routing,
                metadata=metadata,
            )
        except Exception as exc:
            self.runtime_request_coalescer.reject(cache_key, exc)
            raise

    async def _publish_cached_request_events(
        self,
        *,
        request_id: str,
        requested_model_id: str | None,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        capability: str,
        after_accepted: Callable[[], Awaitable[None]] | None = None,
        before_completed: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        await self._publish(
            EventType.REQUEST_ACCEPTED,
            {
                "request_id": request_id,
                "requested_model_id": requested_model_id,
                "model_id": manifest.model_id,
                "capability": capability,
            },
        )
        if after_accepted is not None:
            await after_accepted()
        if before_completed is not None:
            await before_completed()
        await self._publish(
            EventType.REQUEST_COMPLETED,
            {
                "request_id": request_id,
                "model_id": manifest.model_id,
                "runtime": runtime.name,
                "capability": capability,
                "cache_hit": True,
            },
        )

    async def _publish_coalesced_request_events(
        self,
        *,
        request_id: str,
        requested_model_id: str | None,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        capability: str,
        after_accepted: Callable[[], Awaitable[None]] | None = None,
        before_completed: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        await self._publish(
            EventType.REQUEST_ACCEPTED,
            {
                "request_id": request_id,
                "requested_model_id": requested_model_id,
                "model_id": manifest.model_id,
                "capability": capability,
            },
        )
        if after_accepted is not None:
            await after_accepted()
        if before_completed is not None:
            await before_completed()
        await self._publish(
            EventType.REQUEST_COMPLETED,
            {
                "request_id": request_id,
                "model_id": manifest.model_id,
                "runtime": runtime.name,
                "capability": capability,
                "coalesced": True,
            },
        )

    async def synthesize_speech(
        self,
        *,
        model_id: str | None,
        input_text: str,
        voice: str | None,
        audio_format: str,
    ) -> AudioSpeechExecution:
        manifest, runtime, routing = self.model_router.route_audio_speech(model_id)
        request_id = str(uuid4())
        created_at = int(utc_now().timestamp())
        cache_key = self.runtime_response_cache.audio_speech_cache_key(
            model_id=manifest.model_id,
            input_text=input_text,
            voice=voice,
            audio_format=audio_format,
        )
        request_measurements = {"input_characters": len(input_text)}

        async def on_accepted(resolved_request_id: str) -> None:
            await self._publish(
                EventType.AUDIO_SPEECH_STARTED,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "voice": voice,
                    "audio_format": audio_format,
                },
            )
            await self._publish_progress(
                request_id=resolved_request_id,
                operation="audio.speech",
                stage="input_buffered",
                completed_steps=1,
                total_steps=2,
                input_characters=len(input_text),
                voice=voice,
                audio_format=audio_format,
            )

        async def on_success(resolved_request_id: str, response: AudioSpeechResponse) -> None:
            await self._publish_progress(
                request_id=resolved_request_id,
                operation="audio.speech",
                stage="audio_generated",
                completed_steps=2,
                total_steps=2,
                audio_output_bytes=len(response.audio_bytes),
                duration_seconds=response.duration_seconds,
                voice=response.voice,
                media_type=response.media_type,
            )
            await self._publish(
                EventType.AUDIO_SPEECH_COMPLETED,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "voice": response.voice,
                    "audio_format": audio_format,
                    "media_type": response.media_type,
                    "duration_seconds": response.duration_seconds,
                },
            )

        async def on_failure(resolved_request_id: str, exc: Exception) -> None:
            await self._publish(
                EventType.AUDIO_SPEECH_FAILED,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "voice": voice,
                    "audio_format": audio_format,
                    "error": str(exc),
                },
            )

        is_owner, shared_future = self.runtime_request_coalescer.claim(cache_key)
        if not is_owner:
            try:
                shared_response = cast(AudioSpeechResponse, await shared_future)
            except Exception as exc:
                await on_failure(request_id, exc)
                self.runtime_metrics_recorder.record_failure(
                    model_id=manifest.model_id,
                    runtime=runtime.name,
                    capability="audio_speech",
                    load_seconds=0.0,
                    execution_seconds=0.0,
                    measurements={**request_measurements, "coalesced_requests": 1},
                )
                await self._publish(
                    EventType.REQUEST_FAILED,
                    {"request_id": request_id, "model_id": manifest.model_id, "error": str(exc)},
                )
                raise
            coalesced_response = shared_response.model_copy(deep=True)
            await self._publish_coalesced_request_events(
                request_id=request_id,
                requested_model_id=model_id,
                manifest=manifest,
                runtime=runtime,
                capability="audio_speech",
                after_accepted=lambda: on_accepted(request_id),
                before_completed=lambda: on_success(request_id, coalesced_response),
            )
            self.runtime_metrics_recorder.record_success(
                model_id=manifest.model_id,
                runtime=runtime.name,
                capability="audio_speech",
                load_seconds=0.0,
                execution_seconds=0.0,
                measurements={
                    **request_measurements,
                    "audio_output_bytes": len(coalesced_response.audio_bytes),
                    "coalesced_requests": 1,
                },
            )
            return AudioSpeechExecution(
                request_id=request_id,
                created_at=created_at,
                response=coalesced_response,
                routing=routing,
                metadata=build_routed_execution_metadata(
                    request_id=request_id,
                    created=created_at,
                    requested_model_id=model_id,
                    routing=routing,
                    result_origin="coalesced",
                ),
            )
        try:
            cached_response = self.runtime_response_cache.get_audio_speech_response_by_cache_key(cache_key)
            if cached_response is not None:
                cached_copy = cached_response.model_copy(deep=True)
                await self._publish_cached_request_events(
                    request_id=request_id,
                    requested_model_id=model_id,
                    manifest=manifest,
                    runtime=runtime,
                    capability="audio_speech",
                    after_accepted=lambda: on_accepted(request_id),
                    before_completed=lambda: on_success(request_id, cached_copy),
                )
                self.runtime_metrics_recorder.record_success(
                    model_id=manifest.model_id,
                    runtime=runtime.name,
                    capability="audio_speech",
                    load_seconds=0.0,
                    execution_seconds=0.0,
                    measurements={
                        **request_measurements,
                        "audio_output_bytes": len(cached_copy.audio_bytes),
                        "cache_hits": 1,
                    },
                )
                self.runtime_request_coalescer.resolve(cache_key, cached_copy.model_copy(deep=True))
                return AudioSpeechExecution(
                    request_id=request_id,
                    created_at=created_at,
                    response=cached_copy,
                    routing=routing,
                    metadata=build_routed_execution_metadata(
                        request_id=request_id,
                        created=created_at,
                        requested_model_id=model_id,
                        routing=routing,
                        result_origin="cache_hit",
                    ),
                )

            request_id, created_at, response, routing, metadata = await self._execute(
                capability="audio_speech",
                requested_model_id=model_id,
                manifest=manifest,
                runtime=runtime,
                routing=routing,
                request_id=request_id,
                request_measurements={**request_measurements, "cache_misses": 1},
                response_measurements=lambda response: {
                    "audio_output_bytes": len(response.audio_bytes),
                },
                invoke=lambda manifest, runtime, resolved_request_id: runtime.synthesize_speech(
                    AudioSpeechRequest(
                        model_id=manifest.model_id,
                        input_text=input_text,
                        voice=voice,
                        audio_format=audio_format,
                        request_id=resolved_request_id,
                    ),
                ),
                on_accepted=on_accepted,
                on_success=on_success,
                on_failure=on_failure,
            )
            self.runtime_response_cache.put_audio_speech_response(
                model_id=manifest.model_id,
                input_text=input_text,
                voice=voice,
                audio_format=audio_format,
                response=response,
            )
            self.runtime_request_coalescer.resolve(cache_key, response.model_copy(deep=True))
            return AudioSpeechExecution(
                request_id=request_id,
                created_at=created_at,
                response=response,
                routing=routing,
                metadata=metadata,
            )
        except Exception as exc:
            self.runtime_request_coalescer.reject(cache_key, exc)
            raise

    async def _transcribe_audio_with_progress(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        request_id: str,
        file_name: str,
        language: str | None,
        prompt: str | None,
        chunk_plan: _AudioTranscriptionChunkPlan,
        total_progress_steps: int,
        request_measurements: dict[str, int],
    ) -> AudioTranscriptionResponse:
        if not chunk_plan.is_chunked:
            request = AudioTranscriptionRequest(
                model_id=manifest.model_id,
                audio_bytes=chunk_plan.chunks[0].audio_bytes,
                file_name=file_name,
                language=language,
                prompt=prompt,
                request_id=request_id,
                metadata={"source_locator": f"audio:{file_name}"},
            )
            response = await runtime.transcribe_audio(request)
            request_measurements.update(_audio_encoder_cache_measurements(request))
            return response
        responses: list[AudioTranscriptionResponse] = []
        accumulated_segments = 0
        for chunk in chunk_plan.chunks:
            chunk_started_step = 2 + ((chunk.index - 1) * 2)
            await self._publish_progress(
                request_id=request_id,
                operation="audio.transcription",
                stage="chunk_started",
                completed_steps=chunk_started_step,
                total_steps=total_progress_steps,
                file_name=file_name,
                chunk_index=chunk.index,
                total_chunks=chunk.total,
                chunk_start_seconds=chunk.start_seconds,
                chunk_end_seconds=chunk.end_seconds,
            )
            request = AudioTranscriptionRequest(
                model_id=manifest.model_id,
                audio_bytes=chunk.audio_bytes,
                file_name=file_name,
                language=language,
                prompt=prompt,
                request_id=request_id,
                metadata={"source_locator": f"audio:{file_name}#chunk-{chunk.index}"},
            )
            response = await runtime.transcribe_audio(request)
            for metric_name, metric_value in _audio_encoder_cache_measurements(request).items():
                request_measurements[metric_name] = request_measurements.get(metric_name, 0) + metric_value
            responses.append(response)
            accumulated_segments += len(response.segments)
            await self._publish(
                EventType.AUDIO_CHUNK,
                {
                    "request_id": request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "operation": "audio.transcription",
                    "file_name": file_name,
                    "chunk_index": chunk.index,
                    "total_chunks": chunk.total,
                    "chunk_start_seconds": chunk.start_seconds,
                    "chunk_end_seconds": chunk.end_seconds,
                    "segment_count": len(response.segments),
                },
            )
            await self._publish_progress(
                request_id=request_id,
                operation="audio.transcription",
                stage="chunk_processed",
                completed_steps=chunk_started_step + 1,
                total_steps=total_progress_steps,
                file_name=file_name,
                chunk_index=chunk.index,
                total_chunks=chunk.total,
                chunk_start_seconds=chunk.start_seconds,
                chunk_end_seconds=chunk.end_seconds,
                accumulated_segment_count=accumulated_segments,
            )
        return _merge_audio_transcription_chunks(
            model_id=manifest.model_id,
            chunk_plan=chunk_plan,
            responses=responses,
        )

    async def _execute(
        self,
        *,
        capability: str,
        requested_model_id: str | None,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        routing: RoutingDecision,
        request_measurements: dict[str, int] | None,
        response_measurements: Callable[[ResponseT], dict[str, int]] | None,
        invoke: Callable[[ModelManifest, RuntimeContract, str], Awaitable[ResponseT]],
        request_id: str | None = None,
        on_accepted: Callable[[str], Awaitable[None]] | None = None,
        on_success: Callable[[str, ResponseT], Awaitable[None]] | None = None,
        on_failure: Callable[[str, Exception], Awaitable[None]] | None = None,
    ) -> tuple[str, int, ResponseT, RoutingDecision, ExecutionMetadata]:
        resolved_request_id = request_id or str(uuid4())
        created_at = int(utc_now().timestamp())
        try:
            admission = await self.runtime_request_scheduler.acquire()
        except Exception as exc:
            if on_failure is not None:
                await on_failure(resolved_request_id, exc)
            await self._publish(
                EventType.REQUEST_FAILED,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "capability": capability,
                    "error": str(exc),
                },
            )
            raise
        if admission.was_queued:
            await self._publish(
                EventType.REQUEST_QUEUED,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "wait_seconds": admission.wait_seconds,
                    "capability": capability,
                },
            )
        await self._publish(
            EventType.REQUEST_ACCEPTED,
            {
                "request_id": resolved_request_id,
                "requested_model_id": requested_model_id,
                "model_id": manifest.model_id,
                "runtime": runtime.name,
                "capability": capability,
            },
        )
        if on_accepted is not None:
            await on_accepted(resolved_request_id)
        load_admission: RuntimeRequestAdmission | None = None
        try:
            load_started_at = time.perf_counter()
            load_admission = await self._acquire_model_load_admission(
                request_id=resolved_request_id,
                requested_model_id=requested_model_id,
                manifest=manifest,
                runtime=runtime,
                capability=capability,
            )
            await self._publish(
                EventType.MODEL_LOADING,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "capability": capability,
                },
            )
            await runtime.load_model(manifest)
            await self.model_router.runtime_catalog.prepare_runtime_for_request(
                manifest,
                runtime,
                policy=self.model_router.settings.runtime_policy,
            )
            load_seconds = time.perf_counter() - load_started_at
            await self._publish(
                EventType.MODEL_LOADED,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "capability": capability,
                },
            )
            execution_started_at = time.perf_counter()
            response = await invoke(manifest, runtime, resolved_request_id)
            execution_seconds = time.perf_counter() - execution_started_at
            raw_usage = getattr(response, "usage", None)
            usage = raw_usage if isinstance(raw_usage, dict) else None
            self.runtime_metrics_recorder.record_success(
                model_id=manifest.model_id,
                runtime=runtime.name,
                capability=capability,
                load_seconds=load_seconds,
                execution_seconds=execution_seconds,
                usage=usage,
                measurements={
                    **(request_measurements or {}),
                    **(response_measurements(response) if response_measurements is not None else {}),
                },
            )
            if on_success is not None:
                await on_success(resolved_request_id, response)
            await self._publish(
                EventType.REQUEST_COMPLETED,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "capability": capability,
                },
            )
            metadata = build_routed_execution_metadata(
                request_id=resolved_request_id,
                created=created_at,
                requested_model_id=requested_model_id,
                routing=routing,
                queue_milliseconds=milliseconds_from_seconds(
                    admission.wait_seconds + (load_admission.wait_seconds if load_admission is not None else 0.0)
                ),
                load_milliseconds=milliseconds_from_seconds(load_seconds),
                execute_milliseconds=milliseconds_from_seconds(execution_seconds),
            )
            return resolved_request_id, created_at, response, routing, metadata
        except Exception as exc:
            now = time.perf_counter()
            load_seconds = now - load_started_at if "load_started_at" in locals() else 0.0
            execution_seconds = now - execution_started_at if "execution_started_at" in locals() else 0.0
            if "execution_started_at" in locals():
                load_seconds = max(load_seconds - execution_seconds, 0.0)
            self.runtime_metrics_recorder.record_failure(
                model_id=manifest.model_id,
                runtime=runtime.name,
                capability=capability,
                load_seconds=load_seconds,
                execution_seconds=execution_seconds,
                measurements=request_measurements,
            )
            if on_failure is not None:
                await on_failure(resolved_request_id, exc)
            await self._publish(
                EventType.REQUEST_FAILED,
                {
                    "request_id": resolved_request_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "capability": capability,
                    "error": str(exc),
                },
            )
            raise
        finally:
            await self.model_router.runtime_catalog.finalize_runtime_for_request(
                manifest,
                runtime,
                policy=self.model_router.settings.runtime_policy,
            )
            if load_admission is not None:
                load_admission.release()
            admission.release()

    async def _acquire_model_load_admission(
        self,
        *,
        request_id: str,
        requested_model_id: str | None,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        capability: str,
    ) -> RuntimeRequestAdmission | None:
        if runtime.is_model_loaded(manifest.model_id):
            return None
        admission = await self.model_load_scheduler.acquire()
        if admission.was_queued:
            await self._publish(
                EventType.REQUEST_QUEUED,
                {
                    "request_id": request_id,
                    "requested_model_id": requested_model_id,
                    "model_id": manifest.model_id,
                    "runtime": runtime.name,
                    "wait_seconds": admission.wait_seconds,
                    "capability": capability,
                    "queue_type": "model_load",
                },
            )
        return admission

    async def _acquire_model_load_admission_for_batch(
        self,
        *,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        items: list[_PendingEmbeddingBatchItem],
        capability: str,
        batch_payload: dict[str, object],
    ) -> RuntimeRequestAdmission | None:
        if runtime.is_model_loaded(manifest.model_id):
            return None
        admission = await self.model_load_scheduler.acquire()
        if admission.was_queued:
            for item in items:
                await self._publish(
                    EventType.REQUEST_QUEUED,
                    {
                        "request_id": item.request_id,
                        "requested_model_id": item.requested_model_id,
                        "model_id": manifest.model_id,
                        "runtime": runtime.name,
                        "wait_seconds": admission.wait_seconds,
                        "capability": capability,
                        "queue_type": "model_load",
                        **batch_payload,
                    },
                )
        return admission

    async def _publish(self, event_type: EventType, payload: dict[str, object]) -> None:
        normalized_payload = dict(payload)
        capability = normalized_payload.get("capability")
        if not isinstance(capability, str):
            if event_type in {
                EventType.AUDIO_CHUNK,
                EventType.AUDIO_TRANSCRIPTION_STARTED,
                EventType.AUDIO_TRANSCRIPTION_COMPLETED,
                EventType.AUDIO_TRANSCRIPTION_FAILED,
            }:
                capability = "audio_transcription"
            elif event_type in {
                EventType.AUDIO_SPEECH_STARTED,
                EventType.AUDIO_SPEECH_COMPLETED,
                EventType.AUDIO_SPEECH_FAILED,
            }:
                capability = "audio_speech"
        if isinstance(capability, str):
            normalized_payload.setdefault("capability", capability)
            normalized_payload.setdefault("operation", _operation_for_capability(capability))
        await self.event_bus.publish(StreamEvent(type=event_type, scope=EventScope.REQUEST, payload=normalized_payload))

    async def _publish_progress(
        self,
        *,
        request_id: str,
        operation: str,
        stage: str,
        completed_steps: int,
        total_steps: int,
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
                **payload,
            },
        )


def _plan_audio_transcription_chunks(audio_bytes: bytes) -> _AudioTranscriptionChunkPlan:
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as handle:
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
            channel_count = handle.getnchannels()
            sample_width = handle.getsampwidth()
            if frame_rate <= 0 or frame_count <= 0 or channel_count <= 0 or sample_width <= 0:
                raise wave.Error("invalid wav metadata")
            duration_seconds = frame_count / frame_rate
            frames_per_chunk = max(int(frame_rate * _AUDIO_TRANSCRIPTION_CHUNK_SECONDS), 1)
            chunk_count = math.ceil(frame_count / frames_per_chunk)
            if chunk_count <= 1:
                return _AudioTranscriptionChunkPlan(
                    chunks=(
                        _AudioTranscriptionChunk(
                            index=1,
                            total=1,
                            audio_bytes=audio_bytes,
                            start_seconds=0.0,
                            end_seconds=duration_seconds,
                        ),
                    ),
                    duration_seconds=duration_seconds,
                )
            chunks: list[_AudioTranscriptionChunk] = []
            for chunk_index in range(chunk_count):
                chunk_frame_count = min(frames_per_chunk, frame_count - (chunk_index * frames_per_chunk))
                chunk_frames = handle.readframes(chunk_frame_count)
                if not chunk_frames:
                    break
                start_seconds = chunk_index * frames_per_chunk / frame_rate
                actual_frame_count = len(chunk_frames) // (channel_count * sample_width)
                end_seconds = start_seconds + (actual_frame_count / frame_rate)
                buffer = BytesIO()
                with wave.open(buffer, "wb") as chunk_writer:
                    chunk_writer.setnchannels(channel_count)
                    chunk_writer.setsampwidth(sample_width)
                    chunk_writer.setframerate(frame_rate)
                    chunk_writer.writeframes(chunk_frames)
                chunks.append(
                    _AudioTranscriptionChunk(
                        index=chunk_index + 1,
                        total=chunk_count,
                        audio_bytes=buffer.getvalue(),
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                    ),
                )
            if len(chunks) <= 1:
                return _AudioTranscriptionChunkPlan(
                    chunks=(
                        _AudioTranscriptionChunk(
                            index=1,
                            total=1,
                            audio_bytes=audio_bytes,
                            start_seconds=0.0,
                            end_seconds=duration_seconds,
                        ),
                    ),
                    duration_seconds=duration_seconds,
                )
            normalized_chunks = tuple(
                _AudioTranscriptionChunk(
                    index=offset,
                    total=len(chunks),
                    audio_bytes=chunk.audio_bytes,
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                )
                for offset, chunk in enumerate(chunks, start=1)
            )
            return _AudioTranscriptionChunkPlan(
                chunks=normalized_chunks,
                duration_seconds=duration_seconds,
            )
    except wave.Error:
        return _AudioTranscriptionChunkPlan(
            chunks=(
                _AudioTranscriptionChunk(
                    index=1,
                    total=1,
                    audio_bytes=audio_bytes,
                    start_seconds=0.0,
                    end_seconds=0.0,
                ),
            ),
            duration_seconds=None,
        )


def _merge_audio_transcription_chunks(
    *,
    model_id: str,
    chunk_plan: _AudioTranscriptionChunkPlan,
    responses: list[AudioTranscriptionResponse],
) -> AudioTranscriptionResponse:
    if not chunk_plan.is_chunked or len(responses) == 1:
        return responses[0]
    text_parts: list[str] = []
    merged_segments: list[AudioTranscriptionSegment] = []
    language: str | None = None
    fallback_duration = 0.0
    for chunk, response in zip(chunk_plan.chunks, responses, strict=True):
        if response.text.strip():
            text_parts.append(response.text.strip())
        if language is None and response.language is not None:
            language = response.language
        if response.segments:
            for segment in response.segments:
                start_seconds = (
                    chunk.start_seconds + segment.start_seconds
                    if segment.start_seconds is not None
                    else chunk.start_seconds
                )
                end_seconds = (
                    chunk.start_seconds + segment.end_seconds if segment.end_seconds is not None else chunk.end_seconds
                )
                merged_segments.append(
                    AudioTranscriptionSegment(
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        text=segment.text,
                    ),
                )
                fallback_duration = max(fallback_duration, end_seconds or fallback_duration)
        elif response.text.strip():
            merged_segments.append(
                AudioTranscriptionSegment(
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    text=response.text.strip(),
                ),
            )
            fallback_duration = max(fallback_duration, chunk.end_seconds)
    return AudioTranscriptionResponse(
        model_id=model_id,
        text="\n".join(text_parts),
        language=language,
        duration_seconds=chunk_plan.duration_seconds or fallback_duration or None,
        segments=merged_segments,
    )


def _audio_encoder_cache_measurements(request: AudioTranscriptionRequest) -> dict[str, int]:
    payload = request.metadata.get("encoder_cache")
    if not isinstance(payload, dict):
        return {}
    measurements: dict[str, int] = {}
    for source_key, metric_key in (
        ("cache_hits", "multimodal_encoder_cache_hits"),
        ("cache_misses", "multimodal_encoder_cache_misses"),
        ("input_bytes", "multimodal_encoder_input_bytes"),
    ):
        value = payload.get(source_key)
        if isinstance(value, bool):
            measurements[metric_key] = int(value)
        elif isinstance(value, int):
            measurements[metric_key] = value
    return measurements


def _operation_for_capability(capability: str) -> str:
    if capability == "audio_transcription":
        return "audio.transcription"
    if capability == "audio_speech":
        return "audio.speech"
    return capability


def _retrieval_strategy_label(*, use_embeddings: bool, use_rerank: bool) -> str:
    if use_embeddings and use_rerank:
        return "hybrid"
    if use_embeddings:
        return "embeddings"
    return "rerank"


def _build_retrieval_metadata(
    *,
    request_id: str,
    created_at: int,
    embedding_stage: RetrievalStageExecution | None,
    rerank_stage: RetrievalStageExecution | None,
) -> ExecutionMetadata:
    primary_stage = rerank_stage or embedding_stage
    if primary_stage is None:
        raise ValueError("Retrieval metadata requires at least one retrieval stage.")
    stages = [stage for stage in (embedding_stage, rerank_stage) if stage is not None]
    queue_milliseconds = sum(stage.metadata.timing.queue_milliseconds for stage in stages)
    load_milliseconds = sum(stage.metadata.timing.load_milliseconds for stage in stages)
    execute_milliseconds = sum(stage.metadata.timing.execute_milliseconds for stage in stages)
    result_origins = [stage.metadata.result_origin for stage in stages]
    result_origin = (
        result_origins[0]
        if result_origins and all(origin == result_origins[0] for origin in result_origins)
        else "runtime"
    )
    return primary_stage.metadata.model_copy(
        update={
            "request_id": request_id,
            "created": created_at,
            "result_origin": result_origin,
            "timing": ExecutionTimingMetadata(
                queue_milliseconds=queue_milliseconds,
                load_milliseconds=load_milliseconds,
                execute_milliseconds=execute_milliseconds,
                total_milliseconds=queue_milliseconds + load_milliseconds + execute_milliseconds,
            ),
        },
    )


def _rank_retrieval_candidates(
    *,
    candidate_chunks: list[DocumentChunk],
    candidate_sources: list[IngestedDocumentSource],
    embedding_scores: dict[int, float] | None,
    rerank_scores: dict[int, float] | None,
) -> list[_ScoredRetrievalCandidate]:
    sources_by_id = {source.source_id: source for source in candidate_sources}
    ranked = [
        _ScoredRetrievalCandidate(
            chunk=chunk,
            source=sources_by_id.get(chunk.source_id),
            original_index=index,
            score=_retrieval_primary_score(
                embedding_score=(embedding_scores or {}).get(index),
                rerank_score=(rerank_scores or {}).get(index),
            ),
            embedding_score=(embedding_scores or {}).get(index),
            rerank_score=(rerank_scores or {}).get(index),
        )
        for index, chunk in enumerate(candidate_chunks)
    ]
    ranked.sort(
        key=lambda item: (
            -_sortable_score(item.rerank_score if rerank_scores is not None else item.embedding_score),
            -_sortable_score(item.embedding_score),
            item.original_index,
        ),
    )
    return ranked


def _retrieval_primary_score(*, embedding_score: float | None, rerank_score: float | None) -> float:
    if rerank_score is not None:
        return rerank_score
    if embedding_score is not None:
        return embedding_score
    return 0.0


def _sortable_score(score: float | None) -> float:
    if score is None:
        return float("-inf")
    return score


def _matched_retrieval_sources(items: list[RetrievalContextItem]) -> list[IngestedDocumentSource]:
    matched: list[IngestedDocumentSource] = []
    seen_source_ids: set[str] = set()
    for item in items:
        source = item.source
        if source is None or source.source_id in seen_source_ids:
            continue
        matched.append(source)
        seen_source_ids.add(source.source_id)
    return matched


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            "Embedding retrieval scoring requires vectors of equal length "
            f"(got {len(left)} and {len(right)})."
        )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    return dot_product / (left_norm * right_norm)
