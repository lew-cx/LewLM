from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    CapabilityName,
    ConversionStatus,
    EmbeddingRequest,
    GenerateMessage,
    GenerateRequest,
    GenerateSpeculation,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RerankRequest,
    RuntimeAffinity,
    SpeculationMode,
    ValidationState,
    utc_now,
)
from lewlm.core.errors import RuntimeUnavailableError
from lewlm.serving_profiles import resolve_serving_profile_application
from lewlm.storage.metadata import MetadataStore
from lewlm.runtime.llamacpp.runtime import LlamaCppRuntime, _InstrumentedLlamaRamCache


def test_llamacpp_runtime_reports_actionable_windows_install_reason_when_backend_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("lewlm.runtime.llamacpp.runtime.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.runtime.import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.runtime.shutil.which",
        lambda command: None,
    )

    runtime = LlamaCppRuntime(settings=LewLMSettings(data_dir=tmp_path / "state"))

    reason = runtime.availability_reason()

    assert reason is not None
    assert "Install the `llamacpp` extra" in reason
    assert "Microsoft C++ Build Tools" in reason
    assert "Missing required build tool(s) on PATH: cmake." in reason
    assert "Optional tool for faster builds: ninja." in reason


def test_llamacpp_runtime_wraps_backend_model_load_failure(monkeypatch, tmp_path: Path) -> None:
    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            raise OSError("[WinError -1073741795] Windows Error 0xc000001d")

    def fake_import(name: str):
        if name == "llama_cpp":
            return SimpleNamespace(Llama=FakeLlama)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.llamacpp.runtime.import_module", fake_import)

    runtime = LlamaCppRuntime(settings=LewLMSettings(data_dir=tmp_path / "state"))

    with pytest.raises(RuntimeUnavailableError) as exc_info:
        asyncio.run(runtime.load_model(_manifest()))

    assert "CPU instructions" in str(exc_info.value)
    assert exc_info.value.details["runtime"] == "llamacpp"
    assert exc_info.value.details["model_id"] == "gguf-model"
    assert exc_info.value.details["cause_type"] == "OSError"


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


def test_llamacpp_runtime_reports_prompt_lookup_fallback_when_load_surface_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeLlama:
        def __init__(self, *, model_path: str, n_ctx: int, verbose: bool) -> None:
            captured["load_kwargs"] = {
                "model_path": model_path,
                "n_ctx": n_ctx,
                "verbose": verbose,
            }

        def create_chat_completion(self, **kwargs):
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
    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=16,
        temperature=0.0,
    )
    asyncio.run(runtime.generate(request))
    health = asyncio.run(runtime.health_check())

    assert "draft_model" not in captured["load_kwargs"]
    assert request.metadata["runtime_load"]["prompt_lookup"]["supported"] is False
    assert "draft_model" in request.metadata["runtime_load"]["prompt_lookup"]["reason"]
    assert health["performance_features"]["prompt_lookup_speculation"]["supported"] is False
    assert "draft_model" in health["performance_features"]["prompt_lookup_speculation"]["reason"]


def test_llamacpp_runtime_supports_packaged_embeddings_and_rerank_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"embedding_inputs": []}

    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            captured["load_kwargs"] = kwargs

        def create_embedding(self, input):
            captured["embedding_inputs"].append(list(input))
            if input == ["alpha", "beta"]:
                return {
                    "data": [
                        {"embedding": [1.0, 0.0]},
                        {"embedding": [0.0, 1.0]},
                    ],
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                }
            if input == ["semantic query", "first doc", "second doc"]:
                return {
                    "data": [
                        {"embedding": [1.0, 0.0]},
                        {"embedding": [0.8, 0.0]},
                        {"embedding": [0.0, 1.0]},
                    ],
                }
            raise AssertionError(f"Unexpected embedding input payload: {input!r}")

        def tokenize(self, payload: bytes) -> list[int]:
            return list(payload)

        def detokenize(self, tokens: list[int]) -> bytes:
            return bytes(tokens)

    def fake_import(name: str):
        if name == "llama_cpp":
            return SimpleNamespace(Llama=FakeLlama)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.llamacpp.runtime.import_module", fake_import)

    runtime = LlamaCppRuntime(settings=LewLMSettings(data_dir=tmp_path / "state"))
    manifest = _manifest(modality=(ModelModality.EMBEDDING, ModelModality.RERANK))

    asyncio.run(runtime.load_model(manifest))
    embedding_request = EmbeddingRequest(model_id=manifest.model_id, inputs=["alpha", "beta"])
    rerank_request = RerankRequest(
        model_id=manifest.model_id,
        query="semantic query",
        documents=["first doc", "second doc"],
    )

    embedding_response = asyncio.run(runtime.embed(embedding_request))
    rerank_response = asyncio.run(runtime.rerank(rerank_request))

    assert captured["load_kwargs"]["embedding"] is True
    assert captured["embedding_inputs"] == [["alpha", "beta"], ["semantic query", "first doc", "second doc"]]
    assert runtime.supports_manifest(manifest) is True
    assert runtime.supports_capability(CapabilityName.EMBEDDINGS) is True
    assert runtime.supports_capability(CapabilityName.RERANK) is True
    assert embedding_request.metadata["runtime_load"]["semantic_text"]["embedding_mode_enabled"] is True
    assert embedding_request.metadata["semantic_runtime"]["execution_mode"] == "packaged_embedding"
    assert [item.embedding for item in embedding_response.data] == [[1.0, 0.0], [0.0, 1.0]]
    assert embedding_response.usage["prompt_tokens"] == 2
    assert rerank_request.metadata["semantic_runtime"]["execution_mode"] == "embedding_similarity_fallback"
    assert rerank_response.results[0].document == "first doc"
    assert rerank_response.results[0].relevance_score > rerank_response.results[1].relevance_score


def test_llamacpp_runtime_reports_missing_packaged_semantic_surface(monkeypatch, tmp_path: Path) -> None:
    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def fake_import(name: str):
        if name == "llama_cpp":
            return SimpleNamespace(Llama=FakeLlama)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.llamacpp.runtime.import_module", fake_import)

    runtime = LlamaCppRuntime(settings=LewLMSettings(data_dir=tmp_path / "state"))
    manifest = _manifest(modality=(ModelModality.EMBEDDING,))

    assert runtime.supports_capability(CapabilityName.EMBEDDINGS) is False
    assert runtime.supports_manifest(manifest) is False
    assert "create_embedding" in str(runtime.manifest_capability_reason(manifest, CapabilityName.EMBEDDINGS))


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
    assert health["performance_features"]["prefill_optimization"]["ownership"] == "backend_native"
    assert health["performance_features"]["prefill_optimization"]["active"] is True
    assert health["performance_features"]["prefill_optimization"]["metrics"]["requested_prefill_token_batch_size"] == 48
    assert health["performance_features"]["paged_kv_cache"]["supported"] is False
    assert health["performance_features"]["paged_kv_cache"]["ownership"] == "unsupported"
    assert health["performance_features"]["kv_cache_quantization"]["supported"] is False
    assert health["performance_features"]["kv_cache_quantization"]["ownership"] == "unsupported"


def test_llamacpp_runtime_normalizes_model_path_and_reports_runtime_load(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeLlama:
        cache = None

        def __init__(self, *, model_path: str, n_ctx: int, verbose: bool) -> None:
            captured["load_kwargs"] = {
                "model_path": model_path,
                "n_ctx": n_ctx,
                "verbose": verbose,
            }

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [{"message": {"content": "path output"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }

        def tokenize(self, payload: bytes) -> list[int]:
            return [1, 2, 3, 4]

        def detokenize(self, tokens: list[int]) -> bytes:
            return bytes(tokens)

    class FakeRamCache(dict):
        cache_state: dict[tuple[int, ...], str]
        cache_size: int

        def __init__(self) -> None:
            super().__init__()
            self.cache_state = {}
            self.cache_size = 0

        def __setitem__(self, key, value) -> None:
            normalized_key = tuple(key)
            self.cache_state[normalized_key] = value
            self.cache_size = len(self.cache_state)
            super().__setitem__(normalized_key, value)

    def fake_import(name: str):
        if name == "llama_cpp":
            return SimpleNamespace(Llama=FakeLlama, LlamaRAMCache=FakeRamCache)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.llamacpp.runtime.import_module", fake_import)

    runtime = LlamaCppRuntime(settings=LewLMSettings(data_dir=tmp_path / "state"))
    requested_path = str((tmp_path / "models" / "gguf-model.gguf")).replace("\\", "/")
    manifest = _manifest(source_path=requested_path)

    asyncio.run(runtime.load_model(manifest))
    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=16,
        temperature=0.0,
    )
    asyncio.run(runtime.generate(request))

    expected_path = str(Path(requested_path).expanduser())
    assert captured["load_kwargs"]["model_path"] == expected_path
    assert request.metadata["runtime_load"]["requested_model_path"] == requested_path
    assert request.metadata["runtime_load"]["effective_model_path"] == expected_path
    assert request.metadata["runtime_load"]["path_normalized"] == (requested_path != expected_path)
    assert "model_path" in request.metadata["runtime_load"]["load_option_names"]
    assert request.metadata["runtime_load"]["prefix_cache"]["supported"] is True


def test_llamacpp_runtime_accepts_prefill_serving_profile_before_model_load(monkeypatch, tmp_path: Path) -> None:
    class FakeLlama:
        def __init__(self, *, model_path: str, n_ctx: int, verbose: bool, n_batch: int) -> None:
            self.model_path = model_path

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

    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        prefill_token_batch_size=48,
    )
    runtime = LlamaCppRuntime(settings=settings)
    store = MetadataStore(settings.data_dir / "metadata.db")
    store.initialize()
    host_platform = {
        "system": "Windows",
        "release": "11",
        "machine": "AMD64",
        "python_version": "3.11.9",
    }
    store.upsert_serving_profile(
        model_id="gguf-model",
        capability="chat",
        host_platform=host_platform,
        runtime_name="llamacpp",
        workload_class="text_only",
        payload={
            "profile_id": "prefill-profile",
            "model_id": "gguf-model",
            "capability": "chat",
            "workload_class": "text_only",
            "runtime": "llamacpp",
            "recommended_at": utc_now().isoformat(),
            "reason": "Benchmarks preferred the smaller prefill batch size.",
            "settings_overrides": {"prefill_token_batch_size": 48},
        },
    )

    serving_profile = resolve_serving_profile_application(
        settings=settings,
        metadata_store=store,
        host_platform=host_platform,
        runtime=runtime,
        model_id="gguf-model",
        request_capability=CapabilityName.CHAT,
        apply_serving_profile=True,
        workload_class="text_only",
    )

    assert runtime.supports_chunked_prefill(CapabilityName.CHAT) is True
    assert serving_profile.status == "selected"
    assert serving_profile.accepted_settings["prefill_token_batch_size"] == 48


def test_llamacpp_streaming_records_prefix_prefill_metadata_and_handles_message_chunks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeLlama:
        cache = None

        def __init__(self, *, model_path: str, n_ctx: int, verbose: bool) -> None:
            self.cache = None

        def create_chat_completion(self, **kwargs):
            if kwargs["stream"]:
                assert self.cache is not None
                self.cache[(1, 2, 3)] = "cached-state"
                _ = self.cache[(1, 2, 3, 4)]
                return iter(
                    (
                        {"choices": [{"message": {"content": "prefix "}}]},
                        {"choices": [{"delta": {"content": "stream"}}]},
                    ),
                )
            return {
                "choices": [{"message": {"content": "stream output"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }

        def tokenize(self, payload: bytes) -> list[int]:
            return [1, 2, 3, 4]

        def detokenize(self, tokens: list[int]) -> bytes:
            return bytes(tokens)

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

    def fake_import(name: str):
        if name == "llama_cpp":
            return SimpleNamespace(Llama=FakeLlama, LlamaRAMCache=FakeRamCache)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.llamacpp.runtime.import_module", fake_import)

    runtime = LlamaCppRuntime(settings=LewLMSettings(data_dir=tmp_path / "state"))
    manifest = _manifest()

    asyncio.run(runtime.load_model(manifest))
    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=16,
        temperature=0.0,
    )

    async def collect() -> str:
        return "".join([chunk async for chunk in runtime.stream_generate(request)])

    output = asyncio.run(collect())
    health = asyncio.run(runtime.health_check())

    assert output == "prefix stream"
    assert request.metadata["prefix_cache"]["cache_hits"] == 1
    assert request.metadata["prefix_cache"]["saved_prefill_tokens"] == 3
    assert request.metadata["prefix_cache"]["prefilled_uncached_tokens"] == 1
    assert request.metadata["prefix_cache"]["total_prompt_tokens"] == 4
    assert health["performance_features"]["prefix_cache"]["metrics"]["saved_prefill_tokens"] == 3
    assert health["performance_features"]["prefix_cache"]["metrics"]["prefilled_uncached_tokens"] == 1
    assert health["performance_features"]["prefix_cache"]["metrics"]["total_prompt_tokens"] == 4


def test_llamacpp_runtime_reports_prefix_cache_fallback_without_attachment_surface(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeLlama:
        def __init__(self, *, model_path: str, n_ctx: int, verbose: bool) -> None:
            self.model_path = model_path

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [{"message": {"content": "cache output"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }

        def tokenize(self, payload: bytes) -> list[int]:
            return [1, 2, 3, 4]

        def detokenize(self, tokens: list[int]) -> bytes:
            return bytes(tokens)

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

    def fake_import(name: str):
        if name == "llama_cpp":
            return SimpleNamespace(Llama=FakeLlama, LlamaRAMCache=FakeRamCache)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.llamacpp.runtime.import_module", fake_import)

    runtime = LlamaCppRuntime(settings=LewLMSettings(data_dir=tmp_path / "state"))
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

    assert request.metadata["runtime_load"]["prefix_cache"]["supported"] is False
    assert "set_cache()" in request.metadata["runtime_load"]["prefix_cache"]["reason"]
    assert health["performance_features"]["prefix_cache"]["supported"] is False
    assert "set_cache()" in health["performance_features"]["prefix_cache"]["reason"]


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


def _manifest(
    *,
    source_path: str = "/tmp/gguf-model.gguf",
    modality: tuple[ModelModality, ...] = (ModelModality.TEXT,),
) -> ModelManifest:
    return ModelManifest(
        model_id="gguf-model",
        display_name="gguf-model",
        architecture_family="llama",
        modality=modality,
        source_path=source_path,
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
        estimated_memory_mb=1024,
        context_length=4096,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )
