from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    ConversionStatus,
    GenerateMessage,
    GenerateRequest,
    GenerateSpeculation,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    SpeculationMode,
    ValidationState,
)
from lewlm.runtime.llamacpp.runtime import LlamaCppRuntime, _InstrumentedLlamaRamCache


class FakePromptLookupDecoding:
    def __init__(self, *, max_ngram_size: int, num_pred_tokens: int) -> None:
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens


def test_llamacpp_runtime_enables_prompt_lookup_helper(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            captured["load_kwargs"] = kwargs

        def create_chat_completion(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return {
                "choices": [{"message": {"content": "prompt lookup output"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }

        def tokenize(self, payload: bytes) -> list[int]:
            return list(payload)

        def detokenize(self, tokens: list[int]) -> bytes:
            return bytes(tokens)

    def fake_import(name: str):
        if name == "llama_cpp":
            return SimpleNamespace(Llama=FakeLlama)
        if name == "llama_cpp.llama_speculative":
            return SimpleNamespace(LlamaPromptLookupDecoding=FakePromptLookupDecoding)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.llamacpp.runtime.import_module", fake_import)

    runtime = LlamaCppRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            prompt_lookup_speculation_enabled=True,
            prompt_lookup_max_ngram_size=4,
            prompt_lookup_num_pred_tokens=12,
        ),
    )
    manifest = _manifest()

    asyncio.run(runtime.load_model(manifest))
    response = asyncio.run(
        runtime.generate(
            GenerateRequest(
                model_id=manifest.model_id,
                messages=[GenerateMessage(role="user", content="hello")],
                max_tokens=16,
                temperature=0.0,
                speculation=GenerateSpeculation(
                    mode=SpeculationMode.PROMPT_LOOKUP,
                    prompt_lookup_max_ngram_size=4,
                    prompt_lookup_num_pred_tokens=12,
                ),
            ),
        ),
    )
    health = asyncio.run(runtime.health_check())

    draft_model = captured["load_kwargs"]["draft_model"]
    assert isinstance(draft_model, FakePromptLookupDecoding)
    assert draft_model.max_ngram_size == 4
    assert draft_model.num_pred_tokens == 12
    assert response.usage["prompt_lookup_requests"] == 1
    assert response.usage["prompt_lookup_max_ngram_size"] == 4
    assert response.usage["prompt_lookup_num_pred_tokens"] == 12
    assert health["performance_features"]["prompt_lookup_speculation"]["supported"] is True


def test_llamacpp_runtime_applies_prefill_controls_and_surfaces_rejections(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeLlama:
        def __init__(self, *, model_path: str, n_ctx: int, verbose: bool, n_batch: int) -> None:
            captured["load_kwargs"] = {
                "model_path": model_path,
                "n_ctx": n_ctx,
                "verbose": verbose,
                "n_batch": n_batch,
            }

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [{"message": {"content": "prefill output"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }

        def tokenize(self, payload: bytes) -> list[int]:
            return list(payload)

        def detokenize(self, tokens: list[int]) -> bytes:
            return bytes(tokens)

    def fake_import(name: str):
        if name == "llama_cpp":
            return SimpleNamespace(Llama=FakeLlama)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.llamacpp.runtime.import_module", fake_import)

    runtime = LlamaCppRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            prefill_token_batch_size=48,
            kv_cache_page_size=128,
            kv_cache_max_pages=24,
            kv_cache_quantization_bits=6,
        ),
    )
    manifest = _manifest()

    asyncio.run(runtime.load_model(manifest))
    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=16,
        temperature=0.0,
    )
    asyncio.run(runtime.generate(request))
    health = asyncio.run(runtime.health_check())

    assert captured["load_kwargs"]["n_batch"] == 48
    assert request.metadata["performance_controls"]["load"]["prefill_optimization"]["effective"] == "enabled"
    assert request.metadata["performance_controls"]["load"]["paged_kv_cache"]["effective"] == "rejected"
    assert request.metadata["performance_controls"]["load"]["kv_cache_quantization"]["effective"] == "rejected"
    assert health["performance_features"]["prefill_optimization"]["supported"] is True
    assert health["performance_features"]["prefill_optimization"]["active"] is True
    assert health["performance_features"]["prefill_optimization"]["metrics"]["requested_prefill_token_batch_size"] == 48
    assert health["performance_features"]["paged_kv_cache"]["supported"] is False
    assert health["performance_features"]["kv_cache_quantization"]["supported"] is False


def test_llamacpp_prefix_cache_wrapper_uses_longest_prefix_matches() -> None:
    class FakeRamCache:
        def __init__(self) -> None:
            self.cache_state: dict[tuple[int, ...], str] = {}
            self.cache_size = 0

        def __contains__(self, key: Sequence[int]) -> bool:
            return tuple(key) in self.cache_state

        def __getitem__(self, key: Sequence[int]) -> str:
            return self.cache_state[tuple(key)]

        def __setitem__(self, key: Sequence[int], value: str) -> None:
            normalized_key = tuple(key)
            self.cache_state[normalized_key] = value
            self.cache_size = len(self.cache_state)

    cache = _InstrumentedLlamaRamCache(cache_class=FakeRamCache)
    cache[(1, 2, 3)] = "cached-state"

    assert (1, 2, 3, 4) in cache
    assert cache[(1, 2, 3, 4)] == "cached-state"


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_id="gguf-model",
        display_name="gguf-model",
        architecture_family="llama",
        modality=(ModelModality.TEXT,),
        source_path="/tmp/gguf-model.gguf",
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
        estimated_memory_mb=1024,
        context_length=4096,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )
