from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from pydantic import SecretStr

from lewlm.api.app import create_app
from lewlm.config.settings import LewLMSettings, reset_settings_cache
from lewlm.conversion.backend import ConversionExecutionArtifact, ConversionExecutionResult
from lewlm.conversion.models import ConversionCompatibilityReport, ConversionPolicy, LayeredConversionArtifact
from lewlm.core.bootstrap import bootstrap_services
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
    ModelArtifactRole,
    ModelFormat,
    ModelManifest,
    ModelModality,
    QuantizationProfile,
    QuantizationStrategy,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RuntimeAffinity,
)
from lewlm.documents.ir.models import (
    CalloutBlock,
    Citation,
    DocumentIR,
    DocumentOutputFormat,
    DocumentSection,
    HeaderFooterContent,
    ListBlock,
    ParagraphBlock,
    TableBlock,
)
from lewlm.documents.service import DocumentGenerationService
from lewlm.runtime.base import ManagedAudioRuntime, ManagedTextRuntime
from lewlm.runtime.paged_kv import PagedKVResidencyManager, PagedKVReservation
from lewlm.runtime.prefix_cache import InMemoryTokenPrefixCache
from lewlm.storage.block_cache import MultimodalEncoderCache
from lewlm.storage import FrontierExecutionTracker, PersistentPrefixCacheStore
from lewlm.structured_output import GrammarResponseFormat, JSONSchemaResponseFormat, StructuredOutputRuntimeStatus

_TEST_MODELS_DIR_ENV = "LEWLM_TEST_MODELS_DIR"
_EXPECTED_EXTERNAL_MODEL_FIXTURES = (
    "Gemma-4-26B-A4B-it",
    "Gemma-4-31B-it",
    "Gemma-4-31B-JANG_4M",
    "Gemma-4-E2B-Hauhau",
    "Gemma-4-E4B-Hauhau",
)


def emit_benchmark_case_report(
    *,
    label: str,
    payload: dict[str, object],
    feature_names: Sequence[str] = (),
    scenario_names: Sequence[str] = (),
) -> None:
    feature_map = {
        str(item.get("feature")): item
        for item in payload.get("performance_features", [])
        if isinstance(item, dict) and isinstance(item.get("feature"), str)
    }
    scenario_map = {
        str(item.get("scenario")): item
        for item in payload.get("scenarios", [])
        if isinstance(item, dict) and isinstance(item.get("scenario"), str)
    }
    print(
        "BENCHMARK_CASE "
        f"{label} "
        f"model={payload.get('model_id')} "
        f"capability={payload.get('capability')} "
        f"runtime={payload.get('runtime')} "
        f"total={payload.get('total_seconds')}",
    )
    for feature_name in feature_names:
        feature = feature_map.get(feature_name, {})
        metrics = feature.get("metrics", {})
        print(
            "BENCHMARK_FEATURE "
            f"{label} "
            f"{feature_name} "
            f"supported={feature.get('supported')} "
            f"active={feature.get('active')} "
            f"metrics={json.dumps(metrics, sort_keys=True)}",
        )
    for scenario_name in scenario_names:
        scenario = scenario_map.get(scenario_name, {})
        metrics = scenario.get("metrics", {})
        print(
            "BENCHMARK_SCENARIO "
            f"{label} "
            f"{scenario_name} "
            f"status={scenario.get('status')} "
            f"metrics={json.dumps(metrics, sort_keys=True)}",
        )


def emit_benchmark_suite_report(
    entries: Sequence[tuple[str, dict[str, object], Sequence[str], Sequence[str]]],
) -> None:
    print(f"BENCHMARK_SUITE case_count={len(entries)}")
    for label, payload, feature_names, scenario_names in entries:
        feature_map = {
            str(item.get("feature")): item
            for item in payload.get("performance_features", [])
            if isinstance(item, dict) and isinstance(item.get("feature"), str)
        }
        scenario_map = {
            str(item.get("scenario")): item
            for item in payload.get("scenarios", [])
            if isinstance(item, dict) and isinstance(item.get("scenario"), str)
        }
        active_features = ",".join(
            feature_name
            for feature_name in feature_names
            if isinstance(feature_map.get(feature_name), dict) and feature_map[feature_name].get("active") is True
        )
        observed_scenarios = ",".join(
            scenario_name
            for scenario_name in scenario_names
            if isinstance(scenario_map.get(scenario_name), dict) and scenario_map[scenario_name].get("status") == "observed"
        )
        print(
            "BENCHMARK_SUITE_CASE "
            f"{label} "
            f"runtime={payload.get('runtime')} "
            f"total={payload.get('total_seconds')} "
            f"active_features={active_features or '-'} "
            f"observed_scenarios={observed_scenarios or '-'}",
        )


class FakeLlamaCppRuntime(ManagedTextRuntime):
    name = "fake_llamacpp"
    affinity = RuntimeAffinity.LLAMACPP
    supported_formats = (ModelFormat.GGUF,)

    def __init__(self) -> None:
        super().__init__()
        self._prefix_cache = InMemoryTokenPrefixCache(page_size_tokens=8)
        self.batch_generate_calls = 0
        self.batch_stream_calls = 0
        self.max_generate_batch_size = 0
        self.max_stream_batch_size = 0

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, None

    def performance_feature_snapshot(self) -> dict[str, object]:
        prefix_cache_metrics = self._prefix_cache.snapshot()
        return {
            "continuous_batching": {
                "supported": True,
                "active": self.batch_generate_calls > 0 or self.batch_stream_calls > 0,
                "ownership": "backend_native",
                "reason": "Fake llama.cpp runtime exposes backend-native batched chat and streaming entrypoints for tests.",
                "metrics": {
                    "batch_calls": self.batch_generate_calls + self.batch_stream_calls,
                    "batched_requests": self.max_generate_batch_size if self.max_generate_batch_size > 1 else 0,
                    "max_batch_size": max(self.max_generate_batch_size, self.max_stream_batch_size),
                },
            },
            "prefix_cache": {
                "supported": True,
                "active": bool(prefix_cache_metrics["active"]),
                "reason": "Fake llama.cpp runtime reuses in-memory prompt-prefix entries for tests.",
                "metrics": prefix_cache_metrics,
            },
        }

    async def _load_model(self, manifest: ModelManifest) -> None:
        return None

    async def _unload_model(self, model_id: str) -> None:
        return None

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        _apply_fake_structured_output_runtime(request)
        _record_fake_prefix_cache(self._prefix_cache, request)
        output = _fake_output_text(request)
        return GenerateResponse(
            model_id=request.model_id,
            output_text=output,
            finish_reason="stop",
            usage={
                "prompt_tokens": len(request.messages),
                "completion_tokens": len(output.split()),
                "total_tokens": len(request.messages) + len(output.split()),
            },
        )

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        _apply_fake_structured_output_runtime(request)
        _record_fake_prefix_cache(self._prefix_cache, request)
        for chunk in _fake_stream_output_chunks(request):
            if chunk:
                yield chunk

    def supports_continuous_batching(self, capability: CapabilityName) -> bool:
        return capability in {CapabilityName.CHAT, CapabilityName.STREAMING}

    def supports_chunked_prefill(self, capability: CapabilityName) -> bool:
        return capability in {CapabilityName.CHAT, CapabilityName.STREAMING}

    def supports_prefill_isolation(self, capability: CapabilityName) -> bool:
        return capability in {CapabilityName.CHAT, CapabilityName.STREAMING}

    async def generate_batch(self, requests: Sequence[GenerateRequest]) -> list[GenerateResponse]:
        self.batch_generate_calls += 1
        self.max_generate_batch_size = max(self.max_generate_batch_size, len(requests))
        for request in requests:
            request.metadata["native_batching"] = {
                "capability": CapabilityName.CHAT.value,
                "supported": True,
                "active": True,
                "backend": "fake_llamacpp.generate_batch",
                "batch_size": len(requests),
                "stock_single_request_path": False,
                "fallback": False,
                "ownership": "backend_native",
            }
        return [await self._generate(request) for request in requests]

    async def stream_generate_batch(self, requests: Sequence[GenerateRequest]) -> AsyncIterator[tuple[int, str]]:
        self.batch_stream_calls += 1
        self.max_stream_batch_size = max(self.max_stream_batch_size, len(requests))
        for request in requests:
            _apply_fake_structured_output_runtime(request)
            request.metadata["native_batching"] = {
                "capability": CapabilityName.STREAMING.value,
                "supported": True,
                "active": True,
                "backend": "fake_llamacpp.stream_generate_batch",
                "batch_size": len(requests),
                "stock_single_request_path": False,
                "fallback": False,
                "ownership": "backend_native",
            }
        rendered_chunks = [list(_fake_stream_output_chunks(request)) for request in requests]
        for chunk_index in range(max((len(chunks) for chunks in rendered_chunks), default=0)):
            for request_index, chunks in enumerate(rendered_chunks):
                if chunk_index >= len(chunks):
                    continue
                chunk = chunks[chunk_index]
                if chunk:
                    yield request_index, chunk
                    await asyncio.sleep(0)

    def _tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens: Sequence[int]) -> str:
        return bytes(tokens).decode("utf-8")


def _fake_output_text(request: GenerateRequest) -> str:
    structured_output = _fake_structured_output_text(request)
    if structured_output is not None:
        return structured_output
    _, rendered_user_text = _fake_user_text(request)
    output = f"Echo: {rendered_user_text}{_fake_citation_suffix(request)}"
    if _should_emit_reasoning(request):
        return "<think>Inspect the prompt before replying.</think>" + output
    return output


def _fake_prompt_text(request: GenerateRequest) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in request.messages)


def _record_fake_prefix_cache(prefix_cache: InMemoryTokenPrefixCache, request: GenerateRequest) -> None:
    prompt_tokens = list(_fake_prompt_text(request).encode("utf-8"))
    cached_tokens = prompt_tokens[:-1] if len(prompt_tokens) > 1 else prompt_tokens
    lookup = prefix_cache.lookup(model_id=request.model_id, prompt_tokens=cached_tokens)
    stored_entry = prefix_cache.save(
        model_id=request.model_id,
        prefix_tokens=cached_tokens,
        payload={"prompt_text": _fake_prompt_text(request)},
        estimated_size_bytes=len(cached_tokens),
    )
    request.metadata["prefix_cache"] = {
        "cache_hits": 1 if lookup is not None else 0,
        "cache_misses": 0 if lookup is not None else 1,
        "cache_saves": 1 if cached_tokens else 0,
        "resident_cache_hits": 1 if lookup is not None and lookup.entry.source == "resident" else 0,
        "persistent_cache_hits": 1 if lookup is not None and lookup.entry.source == "persisted" else 0,
        "page_size_tokens": stored_entry.page_size_tokens if stored_entry is not None else prefix_cache.snapshot()["page_size_tokens"],
        "cached_pages": lookup.matched_page_count if lookup is not None else 0,
        "resident_page_hits": lookup.resident_page_hits if lookup is not None else 0,
        "persistent_page_hits": lookup.persisted_page_hits if lookup is not None else 0,
        "restored_pages": lookup.restored_page_count if lookup is not None else 0,
        "stored_pages": stored_entry.page_count if stored_entry is not None else 0,
        "copy_on_write_reused_pages": lookup.matched_page_count if lookup is not None else 0,
        "cache_restores": 1 if lookup is not None and lookup.entry.source == "persisted" else 0,
        "saved_prefill_tokens": lookup.prefix_length if lookup is not None else 0,
        "cached_tokens": lookup.prefix_length if lookup is not None else 0,
        "max_saved_prefill_tokens": lookup.prefix_length if lookup is not None else 0,
        "cache_key": stored_entry.cache_key if stored_entry is not None else None,
        "lookup_source": lookup.entry.source if lookup is not None else "miss",
    }


def _fake_user_text(request: GenerateRequest) -> tuple[str, str]:
    last_user_message = next(
        (message.content for message in reversed(request.messages) if message.role == "user"),
        request.messages[-1].content if request.messages else "",
    )
    rendered_user_text = last_user_message.replace("[emit-reasoning]", "").strip()
    return last_user_message, rendered_user_text


def _should_emit_reasoning(request: GenerateRequest) -> bool:
    last_user_message, _ = _fake_user_text(request)
    return "[emit-reasoning]" in last_user_message


def _fake_stream_output_chunks(request: GenerateRequest) -> tuple[str, ...]:
    structured_output = _fake_structured_output_text(request)
    if structured_output is not None:
        midpoint = max(1, len(structured_output) // 2)
        return (structured_output[:midpoint], structured_output[midpoint:])
    _, rendered_user_text = _fake_user_text(request)
    citation_suffix = _fake_citation_suffix(request)
    if _should_emit_reasoning(request):
        return (
            "<thi",
            "nk>Inspect the prompt ",
            "before replying.</thi",
            "nk>",
            "Echo",
            ": ",
            rendered_user_text,
            citation_suffix,
        )
    return ("Echo", ": ", rendered_user_text, citation_suffix)


def _apply_fake_structured_output_runtime(request: GenerateRequest) -> None:
    contract = request.structured_output
    if contract is None or contract.type == "text":
        return
    request.metadata["structured_output_runtime"] = StructuredOutputRuntimeStatus(
        runtime="fake_llamacpp",
        mode=contract.type,
        enforcement="decode_time",
        decoder_enforced=True,
        fallback_used=False,
    ).model_dump(mode="json")


def _fake_structured_output_text(request: GenerateRequest) -> str | None:
    contract = request.structured_output
    runtime_status = request.metadata.get("structured_output_runtime")
    if contract is None or contract.type == "text" or not isinstance(runtime_status, dict):
        return None
    if not bool(runtime_status.get("decoder_enforced")):
        return None
    if isinstance(contract, GrammarResponseFormat):
        return "ok"
    if isinstance(contract, JSONSchemaResponseFormat):
        return json.dumps(_fake_value_from_schema(contract.schema_payload), separators=(",", ":"))
    return None


def _fake_value_from_schema(schema: dict[str, object]) -> object:
    const = schema.get("const")
    if const is not None:
        return const
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]
    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        result: dict[str, object] = {}
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str):
                    property_schema = properties.get(key, {}) if isinstance(properties, dict) else {}
                    if isinstance(property_schema, dict):
                        result[key] = _fake_value_from_schema(property_schema)
        elif isinstance(properties, dict):
            for key, property_schema in properties.items():
                if isinstance(key, str) and isinstance(property_schema, dict):
                    result[key] = _fake_value_from_schema(property_schema)
        return result
    if schema_type == "array":
        items = schema.get("items")
        return [_fake_value_from_schema(items)] if isinstance(items, dict) else []
    if schema_type == "string":
        return "ok"
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 1.0
    if schema_type == "boolean":
        return True
    if schema_type == "null":
        return None
    return "ok"


def _fake_citation_suffix(request: GenerateRequest) -> str:
    citation_context = request.metadata.get("citation_context")
    if not isinstance(citation_context, dict):
        return ""
    chunk_ids = citation_context.get("chunk_ids")
    if isinstance(chunk_ids, list):
        for chunk_id in chunk_ids:
            if isinstance(chunk_id, str) and chunk_id:
                return f"[[cite:{chunk_id}]]"
    source_ids = citation_context.get("source_ids")
    if isinstance(source_ids, list):
        for source_id in source_ids:
            if isinstance(source_id, str) and source_id:
                return f"[[cite:{source_id}]]"
    return ""


def _fake_vision_output_text(request: GenerateRequest) -> str:
    last_user_message = next(
        (message for message in reversed(request.messages) if message.role == "user"),
        request.messages[-1] if request.messages else None,
    )
    if last_user_message is None:
        return "Vision echo:"
    image_names = [
        attachment.name
        for attachment in last_user_message.attachments
        if attachment.attachment_type == "image"
    ]
    image_suffix = f" [images: {', '.join(image_names)}]" if image_names else ""
    return f"Vision echo: {last_user_message.content}{image_suffix}"


def _record_fake_multimodal_encoder_cache(
    encoder_cache: MultimodalEncoderCache | None,
    runtime: ManagedTextRuntime,
    request: GenerateRequest,
) -> None:
    if encoder_cache is None:
        return
    manifest = getattr(runtime, "_loaded_manifests", {}).get(request.model_id)
    if manifest is None:
        return
    image_paths: list[Path] = []
    source_locators: list[str] = []
    bundle_count = 0
    for message in request.messages:
        for attachment in message.attachments:
            if attachment.attachment_type != "image" or not attachment.source_path:
                continue
            source_path = Path(attachment.source_path)
            if source_path.is_dir():
                frame_paths = sorted(
                    child for child in source_path.iterdir() if child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg"}
                )
                image_paths.extend(frame_paths)
                source_locators.append(f"bundle:{source_path}")
                bundle_count += 1
            else:
                image_paths.append(source_path)
                source_locators.append(f"image:{source_path}")
    if not image_paths:
        return
    digest = hashlib.sha256()
    total_input_bytes = 0
    for image_path in image_paths:
        payload = image_path.read_bytes() if image_path.exists() else b""
        total_input_bytes += len(payload)
        digest.update(str(image_path).encode("utf-8"))
        digest.update(payload)
    cache_key = encoder_cache.cache_key_for_feature(
        runtime=runtime.name,
        model_id=request.model_id,
        model_fingerprint=manifest.fingerprint,
        modality="frame_bundle" if bundle_count else "image",
        content_sha256=digest.hexdigest(),
        preprocessing_fingerprint="fake-vision-v1",
    )
    cached_feature = encoder_cache.get_feature(cache_key=cache_key)
    if cached_feature is None:
        encoder_cache.put_feature(
            cache_key=cache_key,
            runtime=runtime.name,
            model_id=request.model_id,
            model_fingerprint=manifest.fingerprint,
            modality="frame_bundle" if bundle_count else "image",
            content_sha256=digest.hexdigest(),
            preprocessing_fingerprint="fake-vision-v1",
            feature={"image_count": len(image_paths)},
            source_locator="|".join(source_locators),
            metadata={
                "image_count": len(image_paths),
                "frame_count": len(image_paths) if bundle_count else 0,
                "bundle_count": bundle_count,
                "input_bytes": total_input_bytes,
            },
        )
        request.metadata["encoder_cache"] = {
            "cache_hits": 0,
            "cache_misses": 1,
            "image_input_count": len(image_paths),
            "frame_count": len(image_paths) if bundle_count else 0,
            "bundle_count": bundle_count,
            "input_bytes": total_input_bytes,
        }
    else:
        request.metadata["encoder_cache"] = {
            "cache_hits": 1,
            "cache_misses": 0,
            "image_input_count": len(image_paths),
            "frame_count": len(image_paths) if bundle_count else 0,
            "bundle_count": bundle_count,
            "input_bytes": total_input_bytes,
        }


def _fake_embedding_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [
        round(int.from_bytes(digest[index : index + 4], "big") / 4_294_967_295, 6)
        for index in range(0, 32, 4)
    ]


def _token_overlap_score(query: str, document: str) -> float:
    query_terms = {token.casefold() for token in query.split() if token}
    document_terms = {token.casefold() for token in document.split() if token}
    if not query_terms or not document_terms:
        return 0.0
    return round(len(query_terms & document_terms) / len(query_terms), 4)


def _fake_wav_bytes(payload: bytes) -> bytes:
    sample_rate = 16_000
    channels = 1
    bits_per_sample = 8
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    header = (
        b"RIFF"
        + (36 + len(payload)).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + channels.to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + block_align.to_bytes(2, "little")
        + bits_per_sample.to_bytes(2, "little")
        + b"data"
        + len(payload).to_bytes(4, "little")
    )
    return header + payload


class FakeMLXSemanticRuntime(ManagedTextRuntime):
    name = "fake_mlx_semantic"
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

    def __init__(self, settings: LewLMSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or LewLMSettings(environment="test", privacy_mode=True)
        self._prefix_cache = InMemoryTokenPrefixCache(
            page_size_tokens=self.settings.kv_cache_page_size,
            persistent_store=PersistentPrefixCacheStore(
                cache_root=self.settings.cache_dir,
                namespace=self.name,
                page_size_tokens=self.settings.kv_cache_page_size,
            ),
        )
        self._frontier_execution = FrontierExecutionTracker(settings=self.settings)
        self._paged_kv_manager = PagedKVResidencyManager(
            page_size_tokens=self.settings.kv_cache_page_size,
            max_pages=self.settings.kv_cache_max_pages,
        )
        self._paged_kv_request_count = 0
        self._paged_kv_prompt_tokens = 0
        self._quantized_kv_request_count = 0
        self._prefill_optimized_request_count = 0
        self._prefill_prompt_tokens = 0
        self._prefill_batch_count = 0
        self.batch_generate_calls = 0
        self.batch_stream_calls = 0
        self.max_generate_batch_size = 0
        self.max_stream_batch_size = 0

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, None

    def performance_feature_snapshot(self) -> dict[str, object]:
        prefix_cache_metrics = self._prefix_cache.snapshot()
        paged_kv_metrics = self._paged_kv_manager.snapshot()
        return {
            "continuous_batching": {
                "supported": True,
                "active": self.batch_generate_calls > 0 or self.batch_stream_calls > 0,
                "ownership": "backend_native",
                "reason": "Fake MLX semantic runtime exposes backend-native batched chat and streaming entrypoints for tests.",
                "metrics": {
                    "batch_calls": self.batch_generate_calls + self.batch_stream_calls,
                    "max_batch_size": max(self.max_generate_batch_size, self.max_stream_batch_size),
                },
            },
            "prefix_cache": {
                "supported": True,
                "active": bool(prefix_cache_metrics["active"]),
                "reason": "Fake MLX semantic runtime reuses in-memory prompt-prefix entries for tests.",
                "metrics": prefix_cache_metrics,
            },
            "persistent_multi_context_cache": {
                "supported": bool(prefix_cache_metrics["restart_resilient"]),
                "active": bool(prefix_cache_metrics["persisted_cache_entries"])
                or int(prefix_cache_metrics["persistent_cache_hits"]) > 0,
                "reason": "Fake MLX semantic runtime persists prompt-prefix entries on disk for restart-resilience tests.",
                "metrics": {
                    "page_size_tokens": prefix_cache_metrics["page_size_tokens"],
                    "resident_cache_entries": prefix_cache_metrics["resident_cache_entries"],
                    "persisted_cache_entries": prefix_cache_metrics["persisted_cache_entries"],
                    "persisted_cache_size_bytes": prefix_cache_metrics["persisted_cache_size_bytes"],
                    "resident_cache_hits": prefix_cache_metrics["resident_cache_hits"],
                    "persistent_cache_hits": prefix_cache_metrics["persistent_cache_hits"],
                    "resident_page_count": prefix_cache_metrics["resident_page_count"],
                    "resident_page_size_bytes": prefix_cache_metrics["resident_page_size_bytes"],
                    "persisted_page_count": prefix_cache_metrics["persisted_page_count"],
                    "persisted_page_size_bytes": prefix_cache_metrics["persisted_page_size_bytes"],
                    "persistent_page_hits": prefix_cache_metrics["persistent_page_hits"],
                    "cache_restores": prefix_cache_metrics["cache_restores"],
                    "page_restores": prefix_cache_metrics["page_restores"],
                    "cache_evictions": prefix_cache_metrics["cache_evictions"],
                    "cache_invalidations": prefix_cache_metrics["cache_invalidations"],
                    "page_evictions": prefix_cache_metrics["page_evictions"],
                    "cached_tokens": prefix_cache_metrics["cached_tokens"],
                },
                "notes": [
                    "Fake runtime tests can create a new runtime instance against the same cache directory to validate restart-safe prompt reuse.",
                ],
            },
            "paged_kv_cache": {
                "supported": True,
                "active": self._paged_kv_request_count > 0,
                "reason": "Fake MLX semantic runtime exposes LewLM-owned first-class paged-KV residency reporting for tests.",
                "metrics": {
                    "page_size_tokens": int(paged_kv_metrics["page_size_tokens"]),
                    "max_pages": int(paged_kv_metrics["max_pages"]),
                    "native_control_supported": True,
                    "requests_using_paged_kv": self._paged_kv_request_count,
                    "paged_prompt_tokens": self._paged_kv_prompt_tokens,
                    "resident_pages": int(paged_kv_metrics["resident_pages"]),
                    "active_pages": int(paged_kv_metrics["active_pages"]),
                    "active_decode_pages": int(paged_kv_metrics["active_decode_pages"]),
                    "active_prefill_pages": int(paged_kv_metrics["active_prefill_pages"]),
                    "resident_decode_pages": int(paged_kv_metrics["resident_decode_pages"]),
                    "resident_prefill_pages": int(paged_kv_metrics["resident_prefill_pages"]),
                    "decode_lane_reservations": int(paged_kv_metrics["decode_lane_reservations"]),
                    "prefill_lane_reservations": int(paged_kv_metrics["prefill_lane_reservations"]),
                    "reused_pages": int(paged_kv_metrics["reused_pages"]),
                    "new_pages": int(paged_kv_metrics["new_pages"]),
                    "evicted_pages": int(paged_kv_metrics["evicted_pages"]),
                    "prefill_evicted_pages": int(paged_kv_metrics["prefill_evicted_pages"]),
                    "decode_evicted_pages": int(paged_kv_metrics["decode_evicted_pages"]),
                    "decode_headroom_preservation_events": int(
                        paged_kv_metrics["decode_headroom_preservation_events"],
                    ),
                    "prefill_decode_tradeoff_events": int(paged_kv_metrics["prefill_decode_tradeoff_events"]),
                    "overflow_events": int(paged_kv_metrics["overflow_events"]),
                    "overflow_pages": int(paged_kv_metrics["overflow_pages"]),
                    "high_pressure_events": int(paged_kv_metrics["high_pressure_events"]),
                    "peak_resident_pages": int(paged_kv_metrics["peak_resident_pages"]),
                    "peak_total_pages": int(paged_kv_metrics["peak_total_pages"]),
                    "pressure_ratio": float(paged_kv_metrics["pressure_ratio"]),
                    "peak_pressure_ratio": float(paged_kv_metrics["peak_pressure_ratio"]),
                    "pressure_level": str(paged_kv_metrics["pressure_level"]),
                },
            },
            "kv_cache_quantization": {
                "supported": True,
                "active": self._quantized_kv_request_count > 0,
                "reason": "Fake MLX semantic runtime exposes runtime-local KV-cache quantization controls for tests.",
                "metrics": {
                    "quantization_bits": self.settings.kv_cache_quantization_bits,
                    "requests_using_quantized_kv": self._quantized_kv_request_count,
                },
            },
            "prefill_optimization": {
                "supported": True,
                "active": self._prefill_optimized_request_count > 0,
                "reason": "Fake MLX semantic runtime exposes runtime-local prefill optimization controls for tests.",
                "metrics": {
                    "prefill_token_batch_size": self.settings.prefill_token_batch_size,
                    "optimized_requests": self._prefill_optimized_request_count,
                    "optimized_prompt_tokens": self._prefill_prompt_tokens,
                    "prefill_batches_planned": self._prefill_batch_count,
                },
            },
            **self._frontier_execution.performance_feature_snapshot(),
        }

    async def _load_model(self, manifest: ModelManifest) -> None:
        self._frontier_execution.register_manifest(manifest)
        return None

    async def _unload_model(self, model_id: str) -> None:
        self._prefix_cache.invalidate(model_id=model_id)
        self._frontier_execution.unregister_model(model_id)
        return None

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        _record_fake_prefix_cache(self._prefix_cache, request)
        await self._record_frontier_execution(request)
        reservation = self._record_prefill_usage(request)
        try:
            output = _fake_output_text(request)
            return GenerateResponse(
                model_id=request.model_id,
                output_text=output,
                finish_reason="stop",
                usage={
                    "prompt_tokens": len(request.messages),
                    "completion_tokens": len(output.split()),
                    "total_tokens": len(request.messages) + len(output.split()),
                },
            )
        finally:
            self._paged_kv_manager.release(reservation)

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        _record_fake_prefix_cache(self._prefix_cache, request)
        await self._record_frontier_execution(request)
        reservation = self._record_prefill_usage(request)
        try:
            for chunk in _fake_stream_output_chunks(request):
                if chunk:
                    yield chunk
        finally:
            self._paged_kv_manager.release(reservation)

    def supports_continuous_batching(self, capability: CapabilityName) -> bool:
        return capability in {CapabilityName.CHAT, CapabilityName.STREAMING}

    async def generate_batch(self, requests: Sequence[GenerateRequest]) -> list[GenerateResponse]:
        self.batch_generate_calls += 1
        self.max_generate_batch_size = max(self.max_generate_batch_size, len(requests))
        for request in requests:
            request.metadata["native_batching"] = {
                "capability": CapabilityName.CHAT.value,
                "supported": True,
                "active": True,
                "backend": "fake_mlx_semantic.generate_batch",
                "batch_size": len(requests),
                "stock_single_request_path": False,
                "fallback": False,
                "ownership": "backend_native",
            }
        return [await self._generate(request) for request in requests]

    async def stream_generate_batch(self, requests: Sequence[GenerateRequest]) -> AsyncIterator[tuple[int, str]]:
        self.batch_stream_calls += 1
        self.max_stream_batch_size = max(self.max_stream_batch_size, len(requests))
        for request in requests:
            request.metadata["native_batching"] = {
                "capability": CapabilityName.STREAMING.value,
                "supported": True,
                "active": True,
                "backend": "fake_mlx_semantic.stream_generate_batch",
                "batch_size": len(requests),
                "stock_single_request_path": False,
                "fallback": False,
                "ownership": "backend_native",
            }
        rendered_chunks = [list(_fake_stream_output_chunks(request)) for request in requests]
        for chunk_index in range(max((len(chunks) for chunks in rendered_chunks), default=0)):
            for request_index, chunks in enumerate(rendered_chunks):
                if chunk_index >= len(chunks):
                    continue
                chunk = chunks[chunk_index]
                if chunk:
                    yield request_index, chunk
                    await asyncio.sleep(0)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        vectors = [
            EmbeddingVector(index=index, embedding=_fake_embedding_vector(text))
            for index, text in enumerate(request.inputs)
        ]
        prompt_tokens = sum(max(1, len(text.split())) for text in request.inputs)
        return EmbeddingResponse(
            model_id=request.model_id,
            data=vectors,
            usage={
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        )

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        ranked = sorted(
            (
                RerankResult(
                    index=index,
                    relevance_score=_token_overlap_score(request.query, document),
                    document=document,
                )
                for index, document in enumerate(request.documents)
            ),
            key=lambda item: (-item.relevance_score, item.index),
        )
        if request.top_n is not None:
            ranked = ranked[: request.top_n]
        return RerankResponse(model_id=request.model_id, results=ranked)

    def _tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens: Sequence[int]) -> str:
        return bytes(tokens).decode("utf-8")

    def prefix_cache_admission_preview(
        self,
        *,
        model_id: str,
        messages: Sequence[object],
    ) -> dict[str, object]:
        if not self.is_model_loaded(model_id):
            return {}
        prompt_text = "\n".join(
            f"{getattr(message, 'role', 'user')}: {getattr(message, 'content', '')}"
            for message in messages
        )
        prompt_tokens = list(prompt_text.encode("utf-8"))
        cached_tokens = prompt_tokens[:-1] if len(prompt_tokens) > 1 else prompt_tokens
        preview = self._prefix_cache.preview(model_id=model_id, prompt_tokens=cached_tokens)
        cached_prefix_tokens = preview.prefix_length if preview is not None else 0
        effective_prefill_tokens = max(len(cached_tokens) - cached_prefix_tokens, 0)
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

    def _loaded_manifest_memory_mb(self, manifest: ModelManifest) -> int | None:
        return self._frontier_execution.loaded_memory_override(manifest.model_id) or manifest.estimated_memory_mb

    async def _record_frontier_execution(self, request: GenerateRequest) -> None:
        manifest = self._loaded_manifests.get(request.model_id)
        if manifest is None:
            return
        execution = self._frontier_execution.annotate_request(manifest=manifest, request=request)
        if execution is None:
            return
        state_cache_miss = int(bool(execution.get("state_cache_misses")))
        swap_count = int(execution.get("expert_swap_count", 0))
        if state_cache_miss or swap_count:
            await asyncio.sleep(0.0005 * state_cache_miss + 0.0008 * swap_count)

    def _record_prefill_usage(self, request: GenerateRequest) -> PagedKVReservation:
        prompt_text = "\n".join(f"{message.role}: {message.content}" for message in request.messages) + "\nassistant:"
        prompt_tokens = self._tokenize(prompt_text)
        reservation = self._paged_kv_manager.reserve(
            model_id=request.model_id,
            prompt_tokens=prompt_tokens,
            max_tokens=request.max_tokens,
            scheduling_lane="prefill" if len(prompt_tokens) >= self.settings.long_prefill_token_threshold else "decode",
        )
        self._paged_kv_request_count += 1
        self._paged_kv_prompt_tokens += len(prompt_tokens)
        self._quantized_kv_request_count += 1
        self._prefill_optimized_request_count += 1
        self._prefill_prompt_tokens += len(prompt_tokens)
        self._prefill_batch_count += max(
            1,
            (len(prompt_tokens) + self.settings.prefill_token_batch_size - 1) // self.settings.prefill_token_batch_size,
        )
        request.metadata["kv_residency"] = {
            "page_size_tokens": self.settings.kv_cache_page_size,
            "max_pages": self.settings.kv_cache_max_pages,
            "queue_lane": reservation.scheduling_lane,
            "requested_pages": reservation.requested_pages,
            "prompt_pages": reservation.prompt_pages,
            "decode_pages": reservation.decode_pages,
            "reused_pages": reservation.reused_pages,
            "new_pages": reservation.new_pages,
            "evicted_pages": reservation.evicted_pages,
            "overflow_pages": reservation.overflow_pages,
            "resident_pages": reservation.resident_pages_after,
            "active_pages": reservation.active_pages_after,
            "resident_decode_pages": reservation.resident_decode_pages_after,
            "resident_prefill_pages": reservation.resident_prefill_pages_after,
            "active_decode_pages": reservation.active_decode_pages_after,
            "active_prefill_pages": reservation.active_prefill_pages_after,
            "pressure_ratio": reservation.pressure_ratio,
            "pressure_level": reservation.pressure_level,
        }
        return reservation


class FakeMLXAudioRuntime(ManagedAudioRuntime):
    name = "fake_mlx_audio"
    affinity = RuntimeAffinity.MLX_AUDIO
    supported_formats = (ModelFormat.AUDIO_FOLDER, ModelFormat.MLX)

    def __init__(self, *, multimodal_encoder_cache: MultimodalEncoderCache | None = None) -> None:
        super().__init__()
        self._multimodal_encoder_cache = multimodal_encoder_cache
        self._native_batch_generate_calls = 0
        self._native_batch_stream_calls = 0
        self._native_batch_request_count = 0
        self._native_batch_max_size = 0

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, None

    def performance_feature_snapshot(self) -> dict[str, object]:
        return {
            "continuous_batching": {
                "supported": True,
                "active": False,
                "ownership": "backend_native",
                "reason": "Fake MLX audio runtime keeps the benchmark feature surface aligned with runtime stats tests.",
                "metrics": {},
            },
            "multimodal_encoder_caching": {
                "supported": self._multimodal_encoder_cache is not None,
                "active": self._multimodal_encoder_cache is not None,
                "reason": "Fake MLX audio runtime reuses synthetic encoder features for repeated audio-transcription tests.",
                "metrics": {},
            },
        }

    async def _load_model(self, manifest: ModelManifest) -> None:
        return None

    async def _unload_model(self, model_id: str) -> None:
        return None

    async def _transcribe_audio(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse:
        if self._multimodal_encoder_cache is not None:
            manifest = self._loaded_manifests[request.model_id]
            content_sha256 = hashlib.sha256(request.audio_bytes).hexdigest()
            cache_key = self._multimodal_encoder_cache.cache_key_for_feature(
                runtime=self.name,
                model_id=request.model_id,
                model_fingerprint=manifest.fingerprint,
                modality="audio",
                content_sha256=content_sha256,
                preprocessing_fingerprint="fake-audio-v1",
            )
            cached_feature = self._multimodal_encoder_cache.get_feature(cache_key=cache_key)
            if cached_feature is None:
                self._multimodal_encoder_cache.put_feature(
                    cache_key=cache_key,
                    runtime=self.name,
                    model_id=request.model_id,
                    model_fingerprint=manifest.fingerprint,
                    modality="audio",
                    content_sha256=content_sha256,
                    preprocessing_fingerprint="fake-audio-v1",
                    feature={"bytes": len(request.audio_bytes)},
                    source_locator=str(request.metadata.get("source_locator", f"audio:{request.file_name}")),
                    metadata={"input_bytes": len(request.audio_bytes)},
                )
                request.metadata["encoder_cache"] = {"cache_hits": 0, "cache_misses": 1, "input_bytes": len(request.audio_bytes)}
            else:
                request.metadata["encoder_cache"] = {"cache_hits": 1, "cache_misses": 0, "input_bytes": len(request.audio_bytes)}
        text = f"Transcribed {request.file_name} ({len(request.audio_bytes)} bytes)"
        return AudioTranscriptionResponse(
            model_id=request.model_id,
            text=text,
            language=request.language or "en",
            duration_seconds=1.0,
            segments=[AudioTranscriptionSegment(start_seconds=0.0, end_seconds=1.0, text=text)],
        )

    async def _synthesize_speech(self, request: AudioSpeechRequest) -> AudioSpeechResponse:
        audio_bytes = _fake_wav_bytes(request.input_text.encode("utf-8"))
        return AudioSpeechResponse(
            model_id=request.model_id,
            audio_bytes=audio_bytes,
            media_type="audio/wav",
            voice=request.voice or "alloy",
            duration_seconds=1.0,
        )


class FakeMLXVisionRuntime(ManagedTextRuntime):
    name = "fake_mlx_vision"
    affinity = RuntimeAffinity.MLX_VISION
    supported_formats = (ModelFormat.MLX,)
    supported_modalities = (
        ModelModality.TEXT,
        ModelModality.VISION,
        ModelModality.MULTIMODAL,
    )
    supported_capabilities = frozenset(
        {
            CapabilityName.CHAT,
            CapabilityName.STREAMING,
            CapabilityName.VISION,
        },
    )

    def __init__(self, *, multimodal_encoder_cache: MultimodalEncoderCache | None = None) -> None:
        super().__init__()
        self._multimodal_encoder_cache = multimodal_encoder_cache
        self._native_batch_generate_calls = 0
        self._native_batch_stream_calls = 0
        self._native_batch_request_count = 0
        self._native_batch_max_size = 0

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, None

    def performance_feature_snapshot(self) -> dict[str, object]:
        return {
            "continuous_batching": {
                "supported": True,
                "active": self._native_batch_request_count > 0,
                "ownership": "backend_native",
                "reason": "Fake MLX vision runtime exposes batched chat and streaming entrypoints for scheduler tests.",
                "metrics": {
                    "chat_batch_calls": self._native_batch_generate_calls,
                    "stream_batch_calls": self._native_batch_stream_calls,
                    "batched_requests": self._native_batch_request_count,
                    "max_batch_size": self._native_batch_max_size,
                },
            },
            "multimodal_encoder_caching": {
                "supported": self._multimodal_encoder_cache is not None,
                "active": self._multimodal_encoder_cache is not None,
                "reason": "Fake MLX vision runtime reuses synthetic image encoder features for repeated multimodal chat tests.",
                "metrics": {},
            },
        }

    async def _load_model(self, manifest: ModelManifest) -> None:
        return None

    async def _unload_model(self, model_id: str) -> None:
        return None

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        _record_fake_multimodal_encoder_cache(self._multimodal_encoder_cache, self, request)
        output = _fake_vision_output_text(request)
        return GenerateResponse(
            model_id=request.model_id,
            output_text=output,
            finish_reason="stop",
            usage={
                "prompt_tokens": len(request.messages),
                "completion_tokens": len(output.split()),
                "total_tokens": len(request.messages) + len(output.split()),
            },
        )

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        _record_fake_multimodal_encoder_cache(self._multimodal_encoder_cache, self, request)
        output = _fake_vision_output_text(request)
        rendered_text = output.removeprefix("Vision echo: ")
        for chunk in ("Vision echo", ": ", rendered_text):
            if chunk:
                yield chunk

    def supports_continuous_batching(self, capability: CapabilityName) -> bool:
        return capability in {CapabilityName.CHAT, CapabilityName.STREAMING}

    async def generate_batch(self, requests: Sequence[GenerateRequest]) -> list[GenerateResponse]:
        self._native_batch_generate_calls += 1
        self._native_batch_request_count += len(requests)
        self._native_batch_max_size = max(self._native_batch_max_size, len(requests))
        for request in requests:
            request.metadata["native_batching"] = {
                "capability": CapabilityName.CHAT.value,
                "supported": True,
                "active": True,
                "backend": "fake_mlx_vision.generate_batch",
                "batch_size": len(requests),
                "stock_single_request_path": False,
                "fallback": False,
                "ownership": "backend_native",
            }
        return [await self._generate(request) for request in requests]

    async def stream_generate_batch(self, requests: Sequence[GenerateRequest]) -> AsyncIterator[tuple[int, str]]:
        self._native_batch_stream_calls += 1
        self._native_batch_request_count += len(requests)
        self._native_batch_max_size = max(self._native_batch_max_size, len(requests))
        for request in requests:
            request.metadata["native_batching"] = {
                "capability": CapabilityName.STREAMING.value,
                "supported": True,
                "active": True,
                "backend": "fake_mlx_vision.stream_generate_batch",
                "batch_size": len(requests),
                "stock_single_request_path": False,
                "fallback": False,
                "ownership": "backend_native",
            }
        rendered_chunks = [
            ["Vision echo", ": ", _fake_vision_output_text(request).removeprefix("Vision echo: ")]
            for request in requests
        ]
        for chunk_index in range(max((len(chunks) for chunks in rendered_chunks), default=0)):
            for request_index, chunks in enumerate(rendered_chunks):
                if chunk_index >= len(chunks):
                    continue
                chunk = chunks[chunk_index]
                if chunk:
                    yield request_index, chunk
                    await asyncio.sleep(0)

    def _tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens: Sequence[int]) -> str:
        return bytes(tokens).decode("utf-8")


class UnavailableMLXTextRuntime(FakeMLXSemanticRuntime):
    name = "unavailable_mlx_text"

    def _check_environment(self) -> tuple[bool, str | None]:
        return False, "Disabled for deterministic test coverage."

    def performance_feature_snapshot(self) -> dict[str, object]:
        return {}


class FakeExternalSemanticRuntime(FakeMLXSemanticRuntime):
    name = "local_external_adapter"
    affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR
    supported_formats = (ModelFormat.MLX, ModelFormat.GGUF)


class FakeMLXConversionBackend:
    name = "fake_mlx_lm"

    def is_available(self) -> bool:
        return True

    def availability_reason(self) -> str | None:
        return None

    def compatibility_report(
        self,
        manifest: ModelManifest,
        *,
        settings: LewLMSettings,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile: QuantizationProfile | None,
        cache_key: str,
        output_path: Path,
    ) -> ConversionCompatibilityReport:
        can_convert = manifest.format_type == ModelFormat.HUGGINGFACE and (
            quantization_profile is None or quantization_profile.strategy == QuantizationStrategy.WEIGHT_ONLY
        )
        layered_output = ModelModality.VISION in manifest.modality and ModelModality.TEXT in manifest.modality
        return ConversionCompatibilityReport(
            model_id=manifest.model_id,
            source_format=manifest.format_type,
            backend_name=self.name,
            can_convert=can_convert,
            reason=(
                "Fake conversion backend can convert Hugging Face bundles."
                if can_convert
                else (
                    "Fake conversion backend only supports weight-only Hugging Face conversion profiles."
                    if quantization_profile is not None and quantization_profile.strategy != QuantizationStrategy.WEIGHT_ONLY
                    else "Fake conversion backend only supports Hugging Face bundles."
                )
            ),
            cache_key=cache_key,
            output_path=str(output_path),
            quantization_mode="4bit" if policy != ConversionPolicy.MAX_QUALITY else None,
            custom_bits=custom_bits,
            requested_profile=quantization_profile,
            resolved_profile=quantization_profile,
            layered_output=layered_output,
            artifact_plans=(
                [
                    LayeredConversionArtifact(
                        artifact_key="multimodal",
                        role=ModelArtifactRole.MULTIMODAL_RUNNABLE,
                        display_name=f"{manifest.display_name} (multimodal mlx)",
                        relative_path="multimodal",
                        format_type=ModelFormat.MLX,
                        modality=manifest.modality,
                        runtime_affinity=(RuntimeAffinity.MLX_VISION,),
                        quantization="4bit" if policy != ConversionPolicy.MAX_QUALITY else None,
                        quantization_profile=quantization_profile,
                    ),
                    LayeredConversionArtifact(
                        artifact_key="text",
                        role=ModelArtifactRole.TEXT_RUNNABLE,
                        display_name=f"{manifest.display_name} (text mlx)",
                        relative_path="text",
                        format_type=ModelFormat.MLX,
                        modality=(ModelModality.TEXT,),
                        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
                        derived_from="multimodal",
                        quantization="4bit" if policy != ConversionPolicy.MAX_QUALITY else None,
                        quantization_profile=quantization_profile,
                    ),
                ]
                if layered_output
                else [
                    LayeredConversionArtifact(
                        artifact_key="standalone",
                        role=ModelArtifactRole.STANDALONE,
                        display_name=manifest.display_name,
                        relative_path=".",
                        format_type=ModelFormat.MLX,
                        modality=manifest.modality,
                        runtime_affinity=((RuntimeAffinity.MLX_VISION,) if ModelModality.VISION in manifest.modality else (RuntimeAffinity.MLX_TEXT,)),
                        quantization="4bit" if policy != ConversionPolicy.MAX_QUALITY else None,
                        quantization_profile=quantization_profile,
                    ),
                ]
            ),
        )

    def convert(
        self,
        manifest: ModelManifest,
        *,
        settings: LewLMSettings,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile: QuantizationProfile | None,
        output_path: Path,
        work_dir: Path,
    ) -> ConversionExecutionResult:
        source_path = Path(manifest.source_path)
        layered_output = ModelModality.VISION in manifest.modality and ModelModality.TEXT in manifest.modality
        if layered_output:
            output_path.mkdir(parents=True, exist_ok=True)
            multimodal_path = output_path / "multimodal"
            text_path = output_path / "text"
            self._write_fake_mlx_bundle(
                source_path=source_path,
                output_path=multimodal_path,
                include_processor=True,
                include_vision=True,
            )
            self._write_fake_mlx_bundle(
                source_path=source_path,
                output_path=text_path,
                include_processor=False,
                include_vision=False,
            )
            return ConversionExecutionResult(
                output_path=output_path,
                logs=[f"Converted {manifest.model_id} into paired fake MLX artifacts in pid {os.getpid()}."],
                artifacts=(
                    ConversionExecutionArtifact(
                        artifact_key="multimodal",
                        role=ModelArtifactRole.MULTIMODAL_RUNNABLE,
                        display_name=f"{manifest.display_name} (multimodal mlx)",
                        output_path=multimodal_path,
                        format_type=ModelFormat.MLX,
                        modality=manifest.modality,
                        runtime_affinity=(RuntimeAffinity.MLX_VISION,),
                    ),
                    ConversionExecutionArtifact(
                        artifact_key="text",
                        role=ModelArtifactRole.TEXT_RUNNABLE,
                        display_name=f"{manifest.display_name} (text mlx)",
                        output_path=text_path,
                        format_type=ModelFormat.MLX,
                        modality=(ModelModality.TEXT,),
                        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
                        derived_from="multimodal",
                    ),
                ),
            )

        self._write_fake_mlx_bundle(
            source_path=source_path,
            output_path=output_path,
            include_processor=ModelModality.VISION in manifest.modality,
            include_vision=ModelModality.VISION in manifest.modality,
        )
        return ConversionExecutionResult(
            output_path=output_path,
            logs=[f"Converted {manifest.model_id} with fake backend in pid {os.getpid()}."],
            artifacts=(
                ConversionExecutionArtifact(
                    artifact_key="standalone",
                    role=ModelArtifactRole.STANDALONE,
                    display_name=manifest.display_name,
                    output_path=output_path,
                    format_type=ModelFormat.MLX,
                    modality=manifest.modality,
                    runtime_affinity=((RuntimeAffinity.MLX_VISION,) if ModelModality.VISION in manifest.modality else (RuntimeAffinity.MLX_TEXT,)),
                ),
            ),
        )

    @staticmethod
    def _write_fake_mlx_bundle(
        *,
        source_path: Path,
        output_path: Path,
        include_processor: bool,
        include_vision: bool,
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        copied_metadata = False
        config_payload: dict[str, object] | None = None
        config_file = source_path / "config.json"
        if config_file.exists():
            config_payload = json.loads(config_file.read_text(encoding="utf-8"))
            config_payload.pop("vision_config", None)
            if include_vision:
                original_payload = json.loads(config_file.read_text(encoding="utf-8"))
                if "vision_config" in original_payload:
                    config_payload["vision_config"] = original_payload["vision_config"]
        if source_path.is_dir():
            for file_name in (
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ):
                source_file = source_path / file_name
                if source_file.exists():
                    shutil.copy2(source_file, output_path / file_name)
                    copied_metadata = True
            if include_processor:
                for file_name in ("processor_config.json", "preprocessor_config.json"):
                    source_file = source_path / file_name
                    if source_file.exists():
                        shutil.copy2(source_file, output_path / file_name)
                        copied_metadata = True
        if config_payload is None:
            config_payload = {"model_type": "phi3"}
        (output_path / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")
        if not copied_metadata and not (output_path / "tokenizer.json").exists():
            (output_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        (output_path / "weights.safetensors").write_bytes(b"converted-weights")


@pytest.fixture(autouse=True)
def reset_cached_settings() -> None:
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def temp_settings(tmp_path: Path) -> LewLMSettings:
    data_dir = tmp_path / "state"
    return LewLMSettings(
        environment="test",
        data_dir=data_dir,
        models_dir=(data_dir / "models",),
        privacy_mode=True,
        api_keys=(SecretStr("test-key"),),
    )


def _materialize_external_model_fixture(*, source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name.startswith(".") or child.is_dir():
            continue
        destination = target / child.name
        if child.suffix in {".json", ".jinja", ".md"}:
            shutil.copy2(child, destination)
            continue
        if child.name.lower().endswith(".gguf"):
            destination.write_bytes(b"gguf-fixture")
            continue
        if child.name.endswith(".safetensors") or child.name in {"weights.safetensors", "weights.npz"}:
            destination.write_bytes(b"fixture-weights")
            continue
        shutil.copy2(child, destination)


@pytest.fixture
def external_models_source_root() -> Path:
    configured = os.environ.get(_TEST_MODELS_DIR_ENV)
    root = (
        Path(configured).expanduser().resolve(strict=False)
        if configured and configured.strip()
        else Path.home() / ".lewlm" / "models"
    )
    missing = [name for name in _EXPECTED_EXTERNAL_MODEL_FIXTURES if not (root / name).exists()]
    if missing:
        pytest.skip(
            "External model fixtures are unavailable. "
            f"Populate {root} or set {_TEST_MODELS_DIR_ENV} to a directory containing: "
            + ", ".join(_EXPECTED_EXTERNAL_MODEL_FIXTURES),
        )
    return root


@pytest.fixture
def external_models_root(tmp_path: Path, external_models_source_root: Path) -> Path:
    root = tmp_path / "external-model-fixtures"
    root.mkdir(parents=True, exist_ok=True)
    for name in _EXPECTED_EXTERNAL_MODEL_FIXTURES:
        _materialize_external_model_fixture(
            source=external_models_source_root / name,
            target=root / name,
        )
    return root


@pytest.fixture
def external_models_settings(temp_settings: LewLMSettings, external_models_root: Path) -> LewLMSettings:
    return temp_settings.with_updates(
        models_dir=(external_models_root, temp_settings.cache_dir),
        file_access_roots=(temp_settings.data_dir, external_models_root),
    )


@pytest.fixture
def external_models_multimodal_settings(
    temp_settings: LewLMSettings,
    external_models_root: Path,
    sample_multimodal_models_root: Path,
) -> LewLMSettings:
    return temp_settings.with_updates(
        models_dir=(external_models_root, sample_multimodal_models_root, temp_settings.cache_dir),
        file_access_roots=(temp_settings.data_dir, external_models_root, sample_multimodal_models_root),
    )


@pytest.fixture
def secured_settings(temp_settings: LewLMSettings) -> LewLMSettings:
    return temp_settings.with_updates(api_key_required=True)


@pytest.fixture
def session_enabled_settings(temp_settings: LewLMSettings) -> LewLMSettings:
    return temp_settings.with_updates(privacy_mode=False)


@pytest.fixture
def limited_settings(temp_settings: LewLMSettings) -> LewLMSettings:
    return temp_settings.with_updates(request_max_bytes=120)


@pytest.fixture
def rate_limited_settings(temp_settings: LewLMSettings) -> LewLMSettings:
    return temp_settings.with_updates(rate_limit_requests=2, rate_limit_window_seconds=60)


@pytest.fixture
def tool_authorized_settings(temp_settings: LewLMSettings) -> LewLMSettings:
    return temp_settings.with_updates(
        audit_log_enabled=True,
        tool_authorization_required=True,
        parser_sandbox_timeout_seconds=10,
    )


@pytest.fixture
def encrypted_persistence_settings(temp_settings: LewLMSettings) -> LewLMSettings:
    return temp_settings.with_updates(
        audit_log_enabled=True,
        persistence_encryption_enabled=True,
        persistence_encryption_passphrase=SecretStr("correct horse battery staple"),
        persistence_encryption_kdf_iterations=100_000,
    )


@pytest.fixture
def sample_models_root(temp_settings: LewLMSettings) -> Path:
    temp_settings.prepare_directories()
    root = temp_settings.models_dir[0]

    gguf_file = root / "llama-3.2-3b-instruct-q4_k_m.gguf"
    gguf_file.write_bytes(b"gguf-model")

    mlx_dir = root / "qwen2.5-1.5b-instruct-mlx"
    mlx_dir.mkdir(parents=True)
    (mlx_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "max_position_embeddings": 32768}),
        encoding="utf-8",
    )
    (mlx_dir / "weights.safetensors").write_bytes(b"mlx-weights")
    (mlx_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    hf_dir = root / "phi-3-mini-hf"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text(
        json.dumps({"architectures": ["Phi3ForCausalLM"], "max_position_embeddings": 4096}),
        encoding="utf-8",
    )
    (hf_dir / "model.safetensors").write_bytes(b"hf-weights")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    return root


@pytest.fixture
def sample_multimodal_models_root(temp_settings: LewLMSettings) -> Path:
    temp_settings.prepare_directories()
    root = temp_settings.models_dir[0]

    embedding_dir = root / "e5-small-embed-mlx"
    embedding_dir.mkdir(parents=True)
    (embedding_dir / "config.json").write_text(
        json.dumps({"model_type": "e5", "max_position_embeddings": 8192}),
        encoding="utf-8",
    )
    (embedding_dir / "weights.safetensors").write_bytes(b"embed-weights")
    (embedding_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    rerank_dir = root / "bge-reranker-base-mlx"
    rerank_dir.mkdir(parents=True)
    (rerank_dir / "config.json").write_text(
        json.dumps({"model_type": "bge", "max_position_embeddings": 8192}),
        encoding="utf-8",
    )
    (rerank_dir / "weights.safetensors").write_bytes(b"rerank-weights")
    (rerank_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    audio_dir = root / "whisper-mini-audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "config.json").write_text(
        json.dumps({"model_type": "whisper"}),
        encoding="utf-8",
    )
    (audio_dir / "processor_config.json").write_text("{}", encoding="utf-8")

    return root


@pytest.fixture
def sample_audio_bytes() -> bytes:
    return _fake_wav_bytes(b"hello world")


@pytest.fixture
def long_sample_audio_bytes() -> bytes:
    return _fake_wav_bytes(b"a" * 32_000)


@pytest.fixture
def sample_chat_models_root(
    temp_settings: LewLMSettings,
    sample_models_root: Path,
    sample_multimodal_models_root: Path,
) -> Path:
    vision_dir = temp_settings.models_dir[0] / "qwen2-vl-vision-mlx"
    vision_dir.mkdir(parents=True, exist_ok=True)
    (vision_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2_vl", "vision_config": {"image_size": 448}}),
        encoding="utf-8",
    )
    (vision_dir / "weights.safetensors").write_bytes(b"vision-weights")
    (vision_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return temp_settings.models_dir[0]


@pytest.fixture
def services_with_fake_runtime(temp_settings: LewLMSettings, sample_models_root: Path):
    return bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )


@pytest.fixture
def services_with_fake_runtime_and_conversion(temp_settings: LewLMSettings, sample_models_root: Path):
    return bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_TEXT: UnavailableMLXTextRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
        conversion_backend=FakeMLXConversionBackend(),
    )


@pytest.fixture
def app_with_fake_runtime(temp_settings: LewLMSettings, services_with_fake_runtime):
    return create_app(temp_settings, services=services_with_fake_runtime)


@pytest.fixture
def services_with_fake_runtime_session_enabled(session_enabled_settings: LewLMSettings, sample_models_root: Path):
    return bootstrap_services(
        session_enabled_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )


@pytest.fixture
def app_with_fake_runtime_session_enabled(
    session_enabled_settings: LewLMSettings,
    services_with_fake_runtime_session_enabled,
):
    return create_app(session_enabled_settings, services=services_with_fake_runtime_session_enabled)


@pytest.fixture
def services_with_fake_multimodal_runtime(temp_settings: LewLMSettings, sample_multimodal_models_root: Path):
    return bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )


@pytest.fixture
def app_with_fake_multimodal_runtime(temp_settings: LewLMSettings, services_with_fake_multimodal_runtime):
    return create_app(temp_settings, services=services_with_fake_multimodal_runtime)


@pytest.fixture
def services_with_fake_attachment_runtime(temp_settings: LewLMSettings, sample_chat_models_root: Path):
    return bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
            RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
        },
    )


@pytest.fixture
def app_with_fake_attachment_runtime(temp_settings: LewLMSettings, services_with_fake_attachment_runtime):
    return create_app(temp_settings, services=services_with_fake_attachment_runtime)


@pytest.fixture
def services_with_fake_external_semantic_runtime(
    temp_settings: LewLMSettings,
    sample_multimodal_models_root: Path,
):
    return bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.EXTERNAL_ACCELERATOR: FakeExternalSemanticRuntime(settings=temp_settings),
            RuntimeAffinity.MLX_TEXT: UnavailableMLXTextRuntime(settings=temp_settings),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )


@pytest.fixture
def app_with_fake_external_semantic_runtime(
    temp_settings: LewLMSettings,
    services_with_fake_external_semantic_runtime,
):
    return create_app(temp_settings, services=services_with_fake_external_semantic_runtime)


@pytest.fixture
def app_with_fake_runtime_and_conversion(temp_settings: LewLMSettings, services_with_fake_runtime_and_conversion):
    return create_app(temp_settings, services=services_with_fake_runtime_and_conversion)


@pytest.fixture
def services_with_authorized_runtime_and_conversion(tool_authorized_settings: LewLMSettings, sample_models_root: Path):
    return bootstrap_services(
        tool_authorized_settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime()},
        conversion_backend=FakeMLXConversionBackend(),
    )


@pytest.fixture
def app_with_authorized_runtime_and_conversion(
    tool_authorized_settings: LewLMSettings,
    services_with_authorized_runtime_and_conversion,
):
    return create_app(tool_authorized_settings, services=services_with_authorized_runtime_and_conversion)


@pytest.fixture
def services_with_encrypted_runtime_and_conversion(encrypted_persistence_settings: LewLMSettings, sample_models_root: Path):
    return bootstrap_services(
        encrypted_persistence_settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime()},
        conversion_backend=FakeMLXConversionBackend(),
    )


@pytest.fixture
def app_with_encrypted_runtime_and_conversion(
    encrypted_persistence_settings: LewLMSettings,
    services_with_encrypted_runtime_and_conversion,
):
    return create_app(encrypted_persistence_settings, services=services_with_encrypted_runtime_and_conversion)


@pytest.fixture
def services_with_external_models_runtime_and_conversion(
    external_models_multimodal_settings: LewLMSettings,
):
    return bootstrap_services(
        external_models_multimodal_settings,
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
            RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
        },
        conversion_backend=FakeMLXConversionBackend(),
    )


@pytest.fixture
def app_with_external_models_runtime_and_conversion(
    external_models_multimodal_settings: LewLMSettings,
    services_with_external_models_runtime_and_conversion,
):
    return create_app(external_models_multimodal_settings, services=services_with_external_models_runtime_and_conversion)


@pytest.fixture
def sample_document_ir() -> DocumentIR:
    return DocumentIR(
        title="Quarterly Operations Summary",
        metadata={"department": "Operations"},
        header=HeaderFooterContent(left="LewLM", center="Quarterly Summary"),
        footer=HeaderFooterContent(right="Internal"),
        sections=[
            DocumentSection(
                heading="Summary",
                level=1,
                blocks=[
                    ParagraphBlock(text="Operations remained on track throughout the quarter."),
                    ListBlock(items=["Closed onboarding backlog", "Improved SLA response times"]),
                    CalloutBlock(kind="info", title="Status", body="All milestones stayed within budget."),
                ],
            ),
            DocumentSection(
                heading="Budget",
                level=1,
                blocks=[
                    TableBlock(
                        headers=["Category", "Amount"],
                        rows=[["Hosting", "1200"], ["Licenses", "800"]],
                        caption="Quarterly spend by category",
                    ),
                ],
            ),
        ],
        citations=[Citation(label="1", text="Internal finance worksheet")],
    )


@pytest.fixture
def sample_document_payload(sample_document_ir: DocumentIR) -> dict[str, object]:
    return sample_document_ir.model_dump(mode="json")


@pytest.fixture
def contract_transform_payload() -> dict[str, object]:
    return {
        "skill": "contract_text_replacement",
        "output_format": "docx",
        "input": {
            "title": "Service Agreement",
            "template_text": "This agreement is between {{vendor}} and {{client}}.\n\nPayment terms are [[terms]].",
            "replacements": {
                "vendor": "LewLM",
                "client": "Acme Corp",
                "terms": "Net 30 days",
            },
        },
    }


@pytest.fixture
def receipt_transform_payload() -> dict[str, object]:
    return {
        "skill": "receipt_extraction",
        "output_format": "csv",
        "input": {
            "title": "Cafe Receipt",
            "vendor": "Downtown Cafe",
            "receipt_number": "R-1001",
            "purchased_at": "2026-04-14",
            "currency": "USD",
            "subtotal": "12.50",
            "tax": "1.00",
            "total": "13.50",
            "items": [
                {"description": "Coffee", "quantity": "2", "unit_price": "4.00", "total": "8.00"},
                {"description": "Bagel", "quantity": "1", "unit_price": "4.50", "total": "4.50"},
            ],
        },
    }


@pytest.fixture
def ocr_assisted_extraction_payload() -> dict[str, object]:
    return {
        "skill": "ocr_assisted_extraction",
        "output_format": "markdown",
        "input": {
            "title": "Scanned Invoice Extraction",
            "source_title": "Scanned Vendor Invoice",
            "document_type": "invoice",
            "ocr_text": (
                "Invoice No: INV-2048\n"
                "Invoice Date: 2026-04-12\n"
                "Due Date: 2026-05-12\n"
                "Vendor: Northwind Supplies\n"
                "Bill To: Acme Corp\n"
                "Total Due: USD 1,240.00\n"
                "PO Number: PO-7781\n"
                "Notes\n"
                "Urgent handling for April restock."
            ),
            "expected_fields": [
                {"field": "Invoice Number", "aliases": ["invoice no"], "required": True},
                {"field": "Invoice Date", "required": True},
                {"field": "Due Date", "required": True},
                {"field": "Vendor", "required": True},
                {"field": "Bill To", "aliases": ["bill to"], "required": True},
                {"field": "Total Due", "aliases": ["total due"], "required": True},
                {"field": "PO Number", "aliases": ["po number"], "required": False},
                {"field": "Payment Terms", "aliases": ["terms"], "required": False},
            ],
        },
    }


@pytest.fixture
def branded_document_template_payload(temp_settings: LewLMSettings) -> dict[str, object]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for branded document fixtures.") from exc

    branding_dir = temp_settings.data_dir / "branding"
    branding_dir.mkdir(parents=True, exist_ok=True)
    logo_path = branding_dir / "company-logo.png"
    hero_path = branding_dir / "overview-chart.png"
    Image.new("RGB", (96, 96), color=(18, 72, 140)).save(logo_path)
    Image.new("RGB", (240, 120), color=(235, 242, 252)).save(hero_path)

    return {
        "skill": "branded_document_template",
        "output_format": "json",
        "input": {
            "title": "LewLM Product Launch Brief",
            "settings": {
                "organization_name": "LewLM",
                "subtitle": "Spring launch overview for cross-functional stakeholders.",
                "audience": "Leadership Team",
                "issued_on": "2026-05-01",
                "contact_line": "ops@lewlm.local",
                "header_text": "Product Launch Brief",
                "footer_text": "Internal planning use only",
                "logo_path": str(logo_path),
                "hero_image_path": str(hero_path),
            },
            "summary": "This brief packages launch status, rollout themes, and follow-up ownership into a branded document artifact.",
            "key_points": [
                "Keep API and CLI rollout dates aligned.",
                "Use the same launch messaging across docs and support channels.",
                "Track onboarding questions in the first-week operations review.",
            ],
            "sections": [
                {
                    "heading": "Launch Priorities",
                    "paragraphs": [
                        "The initial release focuses on a stable local-first workflow across chat, tool execution, and document transforms.",
                    ],
                    "bullets": [
                        "Finalize onboarding examples.",
                        "Validate operator guidance for customer pilots.",
                    ],
                    "callout_title": "Owner Note",
                    "callout_body": "Platform operations owns the final go-live checklist.",
                },
                {
                    "heading": "Customer Readiness",
                    "paragraphs": [
                        "Support and product teams will use the same release brief so external messaging stays consistent.",
                    ],
                    "bullets": [
                        "Publish updated docs.",
                        "Share the migration FAQ with the customer success team.",
                    ],
                },
            ],
        },
    }


@pytest.fixture
def file_template_transform_payload(temp_settings: LewLMSettings) -> dict[str, object]:
    template_dir = temp_settings.data_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    template_path = template_dir / "project-template.json"
    template_document = DocumentIR(
        title="Engagement Summary for {{client}}",
        sections=[
            DocumentSection(
                heading="Overview",
                blocks=[
                    ParagraphBlock(text="Prepared by {{owner}}."),
                    TableBlock(
                        headers=["Field", "Value"],
                        rows=[
                            ["Client", "{{client}}"],
                            ["Status", "{{status}}"],
                        ],
                        caption="Project status snapshot",
                    ),
                ],
            ),
        ],
    )
    template_path.write_text(template_document.model_dump_json(indent=2), encoding="utf-8")
    return {
        "skill": "file_template",
        "output_format": "xlsx",
        "template_path": str(template_path),
        "input": {
            "replacements": {
                "client": "Acme Corp",
                "owner": "LewLM",
                "status": "On Track",
            },
        },
    }


@pytest.fixture
def document_compare_transform_payload() -> dict[str, object]:
    return {
        "skill": "document_comparison",
        "output_format": "xlsx",
        "input": {
            "title": "Agreement Comparison",
            "left_title": "Baseline Agreement",
            "left_text": (
                "Shared scope line.\n\n"
                "Legacy pricing remains in effect.\n\n"
                "Termination notice requires 30 days."
            ),
            "right_title": "Revised Agreement",
            "right_text": (
                "Shared scope line.\n\n"
                "Updated pricing schedule applies.\n\n"
                "Termination notice requires 30 days.\n\n"
                "Security review is now mandatory."
            ),
        },
    }


@pytest.fixture
def meeting_transcript_notes_payload() -> dict[str, object]:
    return {
        "skill": "meeting_transcript_notes",
        "output_format": "markdown",
        "input": {
            "title": "Project Kickoff Notes",
            "meeting_date": "2026-04-15",
            "participants": ["Maya", "Jon", "Priya"],
            "transcript_text": (
                "Maya: We need the beta timeline locked by Friday.\n"
                "Jon: Agreed, we will keep scope focused on the API and CLI polish.\n"
                "Priya: Decision: postpone OCR improvements to the next sprint.\n"
                "ACTION: Maya | Send revised delivery timeline | 2026-04-20\n"
                "ACTION: Jon | Prepare rollout checklist | 2026-04-22\n"
                "Priya: I will draft the stakeholder update once the timeline is approved."
            ),
        },
    }


@pytest.fixture
def long_document_memo_payload() -> dict[str, object]:
    return {
        "skill": "long_document_memo",
        "output_format": "markdown",
        "input": {
            "title": "Platform Readiness Memo",
            "source_title": "Quarterly Platform Review",
            "source_text": (
                "The platform review confirmed that document workflows are stable across the API and CLI. "
                "Adoption increased after the team added inspectable session and tool surfaces.\n\n"
                "The remaining delivery risk is mostly tied to OCR depth and broader backend validation on "
                "non-Darwin hosts. The current roadmap keeps those items in later milestones to avoid "
                "destabilizing the core serving path.\n\n"
                "What operator guidance is still missing for Linux and Windows rollouts?\n\n"
                "The review recommended keeping future milestone slices narrow and evidence-backed so each "
                "increment can land with complete tests and docs."
            ),
        },
    }


@pytest.fixture
def speech_transcript_cleanup_payload() -> dict[str, object]:
    return {
        "skill": "speech_transcript_cleanup",
        "output_format": "markdown",
        "input": {
            "title": "Customer Call Cleanup",
            "language": "en",
            "transcript_text": (
                "agent: thanks everyone for joining today\n"
                "customer: i have two questions about rollout timing\n"
                "agent: we can confirm the friday deployment window\n"
                "follow up with legal for the final notice"
            ),
        },
    }


@pytest.fixture
def sample_ingest_sources(temp_settings: LewLMSettings, sample_document_ir: DocumentIR) -> dict[str, Path]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for sample ingest fixtures.") from exc

    temp_settings.prepare_directories()
    ingest_dir = temp_settings.data_dir / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    generation_service = DocumentGenerationService()

    csv_path = ingest_dir / "sample.csv"
    docx_path = ingest_dir / "sample.docx"
    markdown_path = ingest_dir / "sample.md"
    pdf_path = ingest_dir / "sample.pdf"
    text_path = ingest_dir / "sample.txt"
    xlsx_path = ingest_dir / "sample.xlsx"
    for output_format, output_path in (
        (DocumentOutputFormat.CSV, csv_path),
        (DocumentOutputFormat.DOCX, docx_path),
        (DocumentOutputFormat.PDF, pdf_path),
        (DocumentOutputFormat.XLSX, xlsx_path),
    ):
        artifact = generation_service.generate(sample_document_ir, output_format=output_format, file_name=output_path.name)
        output_path.write_bytes(artifact.content)

    image_bundle_dir = ingest_dir / "images"
    image_bundle_dir.mkdir(parents=True, exist_ok=True)
    image_one = image_bundle_dir / "receipt-front.png"
    image_two = image_bundle_dir / "receipt-back.png"
    Image.new("RGB", (120, 60), color=(255, 255, 255)).save(image_one)
    Image.new("RGB", (80, 80), color=(240, 240, 240)).save(image_two)
    text_path.write_text(
        "Quarterly operations summary\n\nOperations remained on track across support and platform readiness.\n\nEscalate Linux host audit follow-up next.",
        encoding="utf-8",
    )
    markdown_path.write_text(
        "# Quarterly operations summary\n\nOperations remained on track across support and platform readiness.\n\n## Action Items\n- Confirm Linux validation host booking\n- Refresh milestone tracker\n\n## Notes\n1. Keep the roadmap local-first.\n2. Avoid new product surface.\n",
        encoding="utf-8",
    )

    return {
        "csv": csv_path,
        "docx": docx_path,
        "markdown": markdown_path,
        "pdf": pdf_path,
        "text": text_path,
        "xlsx": xlsx_path,
        "image_bundle": image_bundle_dir,
        "image_one": image_one,
        "image_two": image_two,
    }


@pytest.fixture
def sample_attachment_sources(
    sample_ingest_sources: dict[str, Path],
    sample_audio_bytes: bytes,
) -> dict[str, Path]:
    ingest_dir = sample_ingest_sources["pdf"].parent
    text_path = ingest_dir / "attachment-note.txt"
    text_path.write_text("Attachment note: summarize the local progress update.", encoding="utf-8")
    audio_path = ingest_dir / "voice-note.wav"
    audio_path.write_bytes(sample_audio_bytes)
    return {
        **sample_ingest_sources,
        "text": text_path,
        "audio": audio_path,
    }


@pytest.fixture
def sample_prompt_assets(temp_settings: LewLMSettings) -> dict[str, Path]:
    prompt_dir = temp_settings.data_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    pretext_path = prompt_dir / "pretext.txt"
    pretext_path.write_text("Always organize the response into a concise local-first summary.", encoding="utf-8")

    skill_path = prompt_dir / "skill.json"
    skill_path.write_text(
        json.dumps(
            {
                "name": "milestone_brief",
                "description": "Summarize milestone work with status and next actions.",
                "version": "1.0.0",
                "applicable_modalities": ["text", "document"],
                "supported_inputs": ["chat_messages", "local_attachments"],
                "supported_outputs": ["json"],
                "prompt_scaffolding": "Focus on delivered work, current status, and open risks.",
                "validation_rules": ["Do not invent completed work.", "Keep file references explicit when mentioned."],
                "deterministic_formatting_hints": ["Use stable field names.", "Prefer terse strings over prose."],
                "tool_permissions": ["documents.ingest"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_schema_path = prompt_dir / "output-schema.json"
    output_schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "status": {"type": "string"},
                },
                "required": ["summary", "status"],
                "additionalProperties": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    response_format_path = prompt_dir / "response-format.json"
    response_format_path.write_text(
        json.dumps(
            {
                "type": "json_schema",
                "name": "milestone_summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "status": {"type": "string"},
                    },
                    "required": ["summary", "status"],
                    "additionalProperties": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tools_path = prompt_dir / "tools.json"
    tools_path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "local_lookup",
                        "description": "Searches local indexed project notes.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mcp_tools_path = prompt_dir / "local-mcp-tools.json"
    mcp_tools_path.write_text(
        json.dumps(
            {
                "mcp_tools": [
                    {
                        "server": "roadmap",
                        "name": "search_milestones",
                        "description": "Searches locally indexed milestone notes from an MCP-compatible catalog.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "pretext": pretext_path,
        "skill": skill_path,
        "response_format": response_format_path,
        "output_schema": output_schema_path,
        "tools": tools_path,
        "mcp_tools": mcp_tools_path,
    }
