"""Shared chat measurement and structured-output helpers."""

from __future__ import annotations

from lewlm.core.citations import CitationContextPackage
from lewlm.core.contracts import GenerateMessage, GenerateRequest
from lewlm.prompting import PromptCompilationTrace
from lewlm.runtime.scheduler import FrontierBatchMetrics
from lewlm.structured_output import StructuredOutputResult, analyze_structured_output


def _chat_measurements(
    messages: list[GenerateMessage],
    *,
    output_characters: int | None = None,
    delta_count: int | None = None,
) -> dict[str, int]:
    measurements = {
        "message_count": len(messages),
        "attachment_count": sum(len(message.attachments) for message in messages),
        "input_characters": sum(len(message.content) for message in messages),
    }
    if output_characters is not None:
        measurements["output_characters"] = output_characters
    if delta_count is not None:
        measurements["delta_count"] = delta_count
    return measurements


def _structured_output_result(
    request: GenerateRequest,
    prompt_trace: PromptCompilationTrace,
    output_text: str,
) -> StructuredOutputResult | None:
    contract = prompt_trace.output_contract
    return analyze_structured_output(
        format=contract.format,
        output_text=output_text,
        schema=contract.schema_payload,
        grammar=contract.grammar,
        syntax=contract.syntax,
        name=contract.name,
        strict=contract.strict,
        runtime_status=request.metadata.get("structured_output_runtime"),
    )


def _citation_context_metadata(citation_context: CitationContextPackage | None) -> dict[str, object]:
    if citation_context is None or not citation_context.has_entries():
        return {}
    section_ids = list(dict.fromkeys(chunk.section_id for chunk in citation_context.chunks))
    return {
        "source_count": len(citation_context.sources),
        "chunk_count": len(citation_context.chunks),
        "source_ids": [source.source_id for source in citation_context.sources],
        "chunk_ids": [chunk.chunk_id for chunk in citation_context.chunks],
        "section_ids": section_ids,
    }


def _request_cache_measurements(*, request: GenerateRequest) -> dict[str, int]:
    measurements = {
        **_cache_measurements(
            payload=request.metadata.get("prefix_cache"),
            mapping=(
                ("cache_hits", "prefix_cache_hits"),
                ("cache_misses", "prefix_cache_misses"),
                ("cache_saves", "prefix_cache_saves"),
                ("resident_cache_hits", "prefix_resident_cache_hits"),
                ("persistent_cache_hits", "prefix_persistent_cache_hits"),
                ("cached_pages", "prefix_cached_pages"),
                ("resident_page_hits", "prefix_resident_page_hits"),
                ("persistent_page_hits", "prefix_persistent_page_hits"),
                ("restored_pages", "prefix_restored_pages"),
                ("copy_on_write_reused_pages", "prefix_copy_on_write_reused_pages"),
                ("cache_restores", "prefix_cache_restores"),
                ("cached_tokens", "prefix_cached_tokens"),
                ("saved_prefill_tokens", "prefix_saved_prefill_tokens"),
                ("max_saved_prefill_tokens", "prefix_max_saved_prefill_tokens"),
            ),
        ),
        **_cache_measurements(
            payload=request.metadata.get("encoder_cache"),
            mapping=(
                ("cache_hits", "multimodal_encoder_cache_hits"),
                ("cache_misses", "multimodal_encoder_cache_misses"),
                ("image_input_count", "multimodal_encoder_image_inputs"),
                ("frame_count", "multimodal_encoder_frame_inputs"),
                ("bundle_count", "multimodal_encoder_bundle_requests"),
                ("input_bytes", "multimodal_encoder_input_bytes"),
            ),
        ),
    }
    return measurements


def _request_scheduling_measurements(*, request: GenerateRequest) -> dict[str, int | float]:
    scheduling = request.metadata.get("scheduling")
    if not isinstance(scheduling, dict):
        return {}
    measurements: dict[str, int | float] = {}
    bool_mapping = (
        ("prefill_heavy", "prefill_heavy"),
        ("decode_priority_requested", "decode_priority_requested"),
        ("decode_priority_active", "decode_priority_active"),
        ("prefix_cache_candidate", "prefix_cache_candidate"),
        ("prefill_isolation_requested", "prefill_isolation_requested"),
        ("prefill_isolation_active", "prefill_isolation_active"),
        ("chunked_prefill_requested", "chunked_prefill_requested"),
        ("chunked_prefill_active", "chunked_prefill_active"),
    )
    for source_key, metric_key in bool_mapping:
        value = scheduling.get(source_key)
        if isinstance(value, bool):
            measurements[metric_key] = int(value)
    int_mapping = (
        ("prompt_token_estimate", "prompt_token_estimate"),
        ("total_prompt_tokens", "total_prompt_tokens"),
        ("cached_prefix_tokens", "cached_prefix_tokens"),
        ("cached_pages", "cached_pages"),
        ("chunk_count", "prefill_chunk_count"),
        ("scheduler_wait_milliseconds", "scheduler_wait_milliseconds"),
    )
    for source_key, metric_key in int_mapping:
        value = scheduling.get(source_key)
        if isinstance(value, int) and not isinstance(value, bool):
            measurements[metric_key] = value
    return measurements


def _cache_measurements(
    *,
    payload: object,
    mapping: tuple[tuple[str, str], ...],
) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    measurements: dict[str, int] = {}
    for source_key, metric_key in mapping:
        value = payload.get(source_key)
        if isinstance(value, bool):
            measurements[metric_key] = int(value)
        elif isinstance(value, int):
            measurements[metric_key] = value
    return measurements


def _continuous_batch_measurements(
    *,
    batch_metrics: FrontierBatchMetrics | None,
) -> dict[str, int | float]:
    if batch_metrics is None:
        return {}
    return {
        "queue_delay_seconds": batch_metrics.queue_delay_seconds,
        "batch_window_milliseconds": batch_metrics.batch_window_milliseconds,
        "batch_size": batch_metrics.batch_size,
        "batch_utilization": batch_metrics.batch_utilization,
        "batched_requests": 1 if batch_metrics.batch_size > 1 else 0,
        "coalesced_requests": 1 if batch_metrics.batch_size > 1 and batch_metrics.batch_position > 0 else 0,
    }


def _chat_coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _chat_coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None
