"""Benchmark helpers for managed runtimes and external adapters."""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

from lewlm.core.contracts import CapabilityName, GenerateRequest, GenerateResponse, ModelManifest, RuntimeContract


def benchmark_runtime_chat_manifest(
    runtime: RuntimeContract,
    manifest: ModelManifest,
    *,
    prompt: str,
    max_tokens: int,
    warmup_run_count: int = 1,
) -> dict[str, Any]:
    """Measure a managed runtime with cold load, warm generate, and TTFT evidence."""

    return asyncio.run(
        _benchmark_runtime_chat_manifest(
            runtime=runtime,
            manifest=manifest,
            prompt=prompt,
            max_tokens=max_tokens,
            warmup_run_count=warmup_run_count,
        ),
    )


async def _benchmark_runtime_chat_manifest(
    *,
    runtime: RuntimeContract,
    manifest: ModelManifest,
    prompt: str,
    max_tokens: int,
    warmup_run_count: int,
) -> dict[str, Any]:
    if runtime.is_model_loaded(manifest.model_id):
        await runtime.unload_model(manifest.model_id)

    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    load_start = time.perf_counter()
    await runtime.load_model(manifest)
    load_seconds = round(time.perf_counter() - load_start, 4)

    cold_start = time.perf_counter()
    cold_response = await cast(Any, runtime).generate(request)
    cold_total_seconds = round(time.perf_counter() - cold_start + load_seconds, 4)

    for _ in range(max(0, warmup_run_count)):
        await cast(Any, runtime).generate(request)

    warm_start = time.perf_counter()
    warm_response = await cast(Any, runtime).generate(request)
    warm_total_seconds = round(time.perf_counter() - warm_start, 4)

    ttft_seconds, streamed_output = await _measure_streaming_ttft(runtime=runtime, request=request)
    completion_tokens = _completion_tokens(runtime=runtime, response=warm_response)
    steady_state_decode_seconds = (
        round(max(0.0, warm_total_seconds - ttft_seconds), 4)
        if ttft_seconds is not None
        else None
    )
    steady_state_decode_tokens_per_second = (
        round(completion_tokens / steady_state_decode_seconds, 4)
        if steady_state_decode_seconds not in {None, 0.0}
        else None
    )
    output_text = streamed_output or warm_response.output_text

    return {
        "status": "completed",
        "model_id": manifest.model_id,
        "display_name": manifest.display_name,
        "runtime": runtime.name,
        "runtime_affinity": runtime.affinity.value,
        "prompt": prompt,
        "output_text": output_text,
        "usage": warm_response.usage,
        "phase_breakdown": {
            "cold_load_seconds": load_seconds,
            "cold_total_seconds": cold_total_seconds,
            "warm_total_seconds": warm_total_seconds,
            "ttft_seconds": ttft_seconds,
            "steady_state_decode_seconds": steady_state_decode_seconds,
            "steady_state_decode_tokens_per_second": steady_state_decode_tokens_per_second,
        },
        "performance_features": runtime.performance_feature_snapshot(),
    }


async def _measure_streaming_ttft(
    *,
    runtime: RuntimeContract,
    request: GenerateRequest,
) -> tuple[float | None, str]:
    if not runtime.supports_capability(CapabilityName.STREAMING):
        return None, ""
    start = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []
    async for delta in cast(Any, runtime).stream_generate(request):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        chunks.append(delta)
    if first_token_at is None:
        return None, "".join(chunks)
    return round(first_token_at - start, 4), "".join(chunks)


def _completion_tokens(*, runtime: RuntimeContract, response: GenerateResponse) -> int:
    value = response.usage.get("completion_tokens")
    if isinstance(value, int):
        return value
    return len(runtime.tokenize(response.output_text))
