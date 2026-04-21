"""Direct backend benchmark helpers for CLI comparisons."""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from lewlm.core.contracts import GenerateMessage, ModelFormat, ModelManifest, RuntimeAffinity
from lewlm.runtime.introspection import invoke_with_signature, resolve_backend_callable
from lewlm.runtime.mlx_vision.runtime import load_mlx_vlm_backend_client


@dataclass(slots=True)
class _LoadedMLXTextClient:
    model: Any
    tokenizer: Any | None


@dataclass(slots=True)
class _LoadedMLXVisionClient:
    model: Any
    processor: Any | None


def benchmark_direct_chat_manifest(
    manifest: ModelManifest,
    *,
    prompt: str,
    max_tokens: int = 128,
    warmup_run_count: int = 1,
) -> dict[str, Any]:
    messages = (GenerateMessage(role="user", content=prompt),)
    load_started = time.perf_counter()
    runner = _direct_runner_for_manifest(manifest)
    cold_load_seconds = round(time.perf_counter() - load_started, 4)
    try:
        cold_result = runner.run(messages=messages, max_tokens=max_tokens)
        for _ in range(max(warmup_run_count, 0)):
            runner.run(messages=messages, max_tokens=max_tokens)
        warm_result = runner.run(messages=messages, max_tokens=max_tokens)
        stream_result = runner.measure_stream(messages=messages, max_tokens=max_tokens)
    finally:
        runner.close()
    cold_total_seconds = round(cold_load_seconds + cold_result["generate_seconds"], 4)
    warm_total_seconds = round(warm_result["generate_seconds"], 4)
    ttft_seconds = None if stream_result is None else stream_result.get("ttft_seconds")
    streamed_total_seconds = None if stream_result is None else stream_result.get("elapsed_seconds")
    completion_tokens = 0
    warm_usage = warm_result.get("usage", {})
    if isinstance(warm_usage, dict):
        completion_tokens = int(warm_usage.get("completion_tokens", 0) or 0)
    steady_state_decode_seconds = (
        round(max(float(streamed_total_seconds) - float(ttft_seconds), 0.0), 4)
        if isinstance(streamed_total_seconds, (int, float)) and isinstance(ttft_seconds, (int, float))
        else None
    )
    steady_state_decode_tokens_per_second = (
        round(completion_tokens / steady_state_decode_seconds, 4)
        if completion_tokens > 0 and steady_state_decode_seconds not in (None, 0.0)
        else None
    )
    return {
        "model_id": manifest.model_id,
        "display_name": manifest.display_name,
        "runtime": runner.runtime_name,
        "prompt": prompt,
        "load_seconds": cold_load_seconds,
        "generate_seconds": cold_result["generate_seconds"],
        "total_seconds": cold_total_seconds,
        "output_text": warm_result["output_text"],
        "usage": warm_result["usage"],
        "phase_breakdown": {
            "cold_load_seconds": cold_load_seconds,
            "cold_generate_seconds": cold_result["generate_seconds"],
            "cold_total_seconds": cold_total_seconds,
            "warm_load_seconds": 0.0,
            "warm_generate_seconds": warm_result["generate_seconds"],
            "warm_total_seconds": warm_total_seconds,
            "ttft_seconds": ttft_seconds,
            "steady_state_decode_seconds": steady_state_decode_seconds,
            "steady_state_decode_tokens_per_second": steady_state_decode_tokens_per_second,
            "warmup_run_count": max(warmup_run_count, 0),
            "streamed_total_seconds": streamed_total_seconds,
        },
    }


def _direct_runner_for_manifest(manifest: ModelManifest) -> "_DirectRunner":
    if manifest.format_type == ModelFormat.GGUF:
        return _DirectLlamaCppRunner(manifest)
    if RuntimeAffinity.MLX_VISION in manifest.runtime_affinity:
        return _DirectMLXVisionRunner(manifest)
    return _DirectMLXTextRunner(manifest)


class _DirectRunner:
    runtime_name: str

    def run(self, *, messages: tuple[GenerateMessage, ...], max_tokens: int) -> dict[str, Any]:
        raise NotImplementedError

    def measure_stream(self, *, messages: tuple[GenerateMessage, ...], max_tokens: int) -> dict[str, Any] | None:
        return None

    def close(self) -> None:
        raise NotImplementedError


class _DirectLlamaCppRunner(_DirectRunner):
    runtime_name = "llama_cpp_direct"

    def __init__(self, manifest: ModelManifest) -> None:
        llama_cpp = import_module("llama_cpp")
        llama_class = getattr(llama_cpp, "Llama")
        self._client = llama_class(
            model_path=manifest.source_path,
            n_ctx=manifest.context_length or 4096,
            verbose=False,
        )

    def run(self, *, messages: tuple[GenerateMessage, ...], max_tokens: int) -> dict[str, Any]:
        started = time.perf_counter()
        response = self._client.create_chat_completion(
            messages=[{"role": message.role, "content": message.content} for message in messages],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
        )
        generate_seconds = round(time.perf_counter() - started, 4)
        usage = response.get("usage", {})
        return {
            "generate_seconds": generate_seconds,
            "output_text": str(response["choices"][0]["message"].get("content", "")),
            "usage": {str(key): int(value) for key, value in usage.items() if isinstance(value, (int, float))},
        }

    def measure_stream(self, *, messages: tuple[GenerateMessage, ...], max_tokens: int) -> dict[str, Any] | None:
        started = time.perf_counter()
        ttft_seconds: float | None = None
        output_chunks: list[str] = []
        stream = self._client.create_chat_completion(
            messages=[{"role": message.role, "content": message.content} for message in messages],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        )
        for chunk in stream:
            text = _llama_stream_delta_text(chunk)
            if not text:
                continue
            if ttft_seconds is None:
                ttft_seconds = round(time.perf_counter() - started, 4)
            output_chunks.append(text)
        if ttft_seconds is None:
            return None
        return {
            "ttft_seconds": ttft_seconds,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "output_text": "".join(output_chunks),
        }

    def close(self) -> None:
        self._client = None


class _DirectMLXTextRunner(_DirectRunner):
    runtime_name = "mlx_lm_direct"

    def __init__(self, manifest: ModelManifest) -> None:
        self._module = import_module("mlx_lm")
        load = resolve_backend_callable(self._module, ("load", "load_model", "load_pipeline"))
        loaded = invoke_with_signature(
            load,
            {
                "path_or_hf_repo": manifest.source_path,
                "path": manifest.source_path,
                "model_path": manifest.source_path,
                "source_path": manifest.source_path,
            },
            capability="direct_model_load",
        )
        self._client = _normalize_mlx_text_client(loaded)

    def run(self, *, messages: tuple[GenerateMessage, ...], max_tokens: int) -> dict[str, Any]:
        generate = resolve_backend_callable(self._module, ("generate", "chat", "generate_text"))
        prompt = _render_mlx_text_prompt(messages, self._client.tokenizer)
        started = time.perf_counter()
        result = invoke_with_signature(
            generate,
            {
                "client": {"model": self._client.model, "tokenizer": self._client.tokenizer},
                "model": self._client.model,
                "tokenizer": self._client.tokenizer,
                "prompt": prompt,
                "messages": [{"role": message.role, "content": message.content} for message in messages],
                "max_tokens": max_tokens,
                "verbose": False,
            },
            capability="direct_chat",
            passthrough_keys=("max_tokens",),
        )
        return {
            "generate_seconds": round(time.perf_counter() - started, 4),
            "output_text": _text_from_generation_result(result),
            "usage": _usage_from_generation_result(result),
        }

    def close(self) -> None:
        self._client = _LoadedMLXTextClient(model=None, tokenizer=None)


class _DirectMLXVisionRunner(_DirectRunner):
    runtime_name = "mlx_vlm_direct"

    def __init__(self, manifest: ModelManifest) -> None:
        self._module = import_module("mlx_vlm")
        loaded = load_mlx_vlm_backend_client(manifest.source_path, capability="direct_model_load")
        if loaded is None:
            self._client = _LoadedMLXVisionClient(model=manifest.source_path, processor=None)
            return
        self._client = _normalize_mlx_vision_client(loaded)

    def run(self, *, messages: tuple[GenerateMessage, ...], max_tokens: int) -> dict[str, Any]:
        generate = resolve_backend_callable(self._module, ("generate", "chat", "generate_text"))
        started = time.perf_counter()
        result = invoke_with_signature(
            generate,
            {
                "client": {"model": self._client.model, "processor": self._client.processor},
                "model": self._client.model,
                "processor": self._client.processor,
                "prompt": _render_mlx_vision_prompt(messages),
                "messages": [{"role": message.role, "content": message.content} for message in messages],
                "images": [],
                "image": None,
                "image_paths": [],
                "max_tokens": max_tokens,
                "verbose": False,
            },
            capability="direct_chat",
            passthrough_keys=(),
        )
        return {
            "generate_seconds": round(time.perf_counter() - started, 4),
            "output_text": _text_from_generation_result(result),
            "usage": _usage_from_generation_result(result),
        }

    def close(self) -> None:
        self._client = _LoadedMLXVisionClient(model=None, processor=None)


def _normalize_mlx_text_client(loaded: Any) -> _LoadedMLXTextClient:
    if isinstance(loaded, tuple):
        model = loaded[0] if len(loaded) > 0 else None
        tokenizer = loaded[1] if len(loaded) > 1 else None
        return _LoadedMLXTextClient(model=model, tokenizer=tokenizer)
    if isinstance(loaded, dict):
        return _LoadedMLXTextClient(model=loaded.get("model", loaded), tokenizer=loaded.get("tokenizer"))
    return _LoadedMLXTextClient(model=loaded, tokenizer=None)


def _normalize_mlx_vision_client(loaded: Any) -> _LoadedMLXVisionClient:
    if isinstance(loaded, tuple):
        model = loaded[0] if len(loaded) > 0 else None
        processor = loaded[1] if len(loaded) > 1 else None
        return _LoadedMLXVisionClient(model=model, processor=processor)
    if isinstance(loaded, dict):
        return _LoadedMLXVisionClient(model=loaded.get("model", loaded), processor=loaded.get("processor"))
    return _LoadedMLXVisionClient(model=loaded, processor=None)


def _render_mlx_text_prompt(messages: tuple[GenerateMessage, ...], tokenizer: Any | None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": message.role, "content": message.content} for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    return _render_mlx_vision_prompt(messages)


def _render_mlx_vision_prompt(messages: tuple[GenerateMessage, ...]) -> str:
    rendered = [f"{message.role}: {message.content}" for message in messages]
    rendered.append("assistant:")
    return "\n".join(rendered)


def _text_from_generation_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        text = result.get("text", result.get("output_text", result.get("response", "")))
        return text if isinstance(text, str) else str(result)
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else str(result)


def _usage_from_generation_result(result: Any) -> dict[str, int]:
    if not isinstance(result, dict):
        return {}
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        str(key): int(value)
        for key, value in usage.items()
        if isinstance(value, (int, float))
    }


def _llama_stream_delta_text(chunk: Any) -> str:
    if not isinstance(chunk, dict):
        return ""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        return content if isinstance(content, str) else ""
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else ""
    text = choice.get("text")
    return text if isinstance(text, str) else ""
