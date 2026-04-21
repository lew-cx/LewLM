from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    CapabilityName,
    ConversionStatus,
    EmbeddingRequest,
    GenerateSpeculation,
    GenerateMessage,
    GenerateRequest,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RerankRequest,
    RuntimeAffinity,
    SpeculationMode,
    ValidationState,
)
from lewlm.runtime.mlx_text.runtime import MLXTextRuntime
from lewlm.structured_output import JSONSchemaResponseFormat


class FakeTokenizer:
    vocab_size = 1024

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        rendered = [f"{message['role']}: {message['content']}" for message in messages]
        if add_generation_prompt:
            rendered.append("assistant:")
        return "\n".join(rendered)

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")


def test_mlx_text_runtime_supports_embeddings_and_rerank(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizer()
    fake_module = SimpleNamespace(
        load=lambda path: ("fake-model", fake_tokenizer),
        generate=lambda **kwargs: "unused",
        generate_stream=lambda **kwargs: [],
        embed=lambda model, tokenizer, inputs: [[float(index), float(index + 1)] for index, _ in enumerate(inputs)],
        rerank=lambda model, tokenizer, query, documents, top_n=None: [
            {
                "index": index,
                "relevance_score": float(len(document)),
                "document": document,
            }
            for index, document in enumerate(documents)
        ],
    )
    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", lambda name: fake_module)

    runtime = MLXTextRuntime()
    manifest = _manifest(modality=(ModelModality.EMBEDDING, ModelModality.RERANK))

    asyncio.run(runtime.load_model(manifest))

    assert runtime.supports_capability(CapabilityName.EMBEDDINGS) is True
    assert runtime.supports_capability(CapabilityName.RERANK) is True

    embedding_response = asyncio.run(
        runtime.embed(EmbeddingRequest(model_id=manifest.model_id, inputs=["alpha", "beta"])),
    )
    rerank_response = asyncio.run(
        runtime.rerank(
            RerankRequest(
                model_id=manifest.model_id,
                query="alpha",
                documents=["short", "a bit longer"],
                top_n=1,
            ),
        ),
    )

    assert [item.embedding for item in embedding_response.data] == [[0.0, 1.0], [1.0, 2.0]]
    assert embedding_response.usage["prompt_tokens"] >= 2
    assert rerank_response.results[0].document == "a bit longer"
    assert rerank_response.results[0].index == 1


def test_mlx_text_runtime_supports_real_mlx_load_signature_and_sampling(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizer()
    captured: dict[str, object] = {}

    def fake_load(*, path_or_hf_repo: str):
        captured["path_or_hf_repo"] = path_or_hf_repo
        return "fake-model", fake_tokenizer

    def fake_generate(*, model, tokenizer, prompt, verbose=False, **kwargs):
        captured["generate"] = {
            "model": model,
            "tokenizer": tokenizer,
            "prompt": prompt,
            "verbose": verbose,
            "kwargs": kwargs,
        }
        return "sampled output"

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(
                load=fake_load,
                generate=fake_generate,
                generate_stream=lambda **kwargs: [],
            )
        if name == "mlx_lm.sample_utils":
            return SimpleNamespace(make_sampler=lambda *, temp: {"temp": temp})
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime()
    manifest = _manifest(modality=(ModelModality.TEXT,))

    asyncio.run(runtime.load_model(manifest))
    response = asyncio.run(
        runtime.generate(
            GenerateRequest(
                model_id=manifest.model_id,
                messages=[GenerateMessage(role="user", content="hello")],
                max_tokens=64,
                temperature=0.4,
            ),
        ),
    )

    assert captured["path_or_hf_repo"] == manifest.source_path
    assert captured["generate"] == {
        "model": "fake-model",
        "tokenizer": fake_tokenizer,
        "prompt": list("user: hello\nassistant:".encode("utf-8")),
        "verbose": False,
        "kwargs": {"max_tokens": 64, "sampler": {"temp": 0.4}},
    }
    assert response.output_text == "sampled output"


def test_mlx_text_runtime_records_prompt_guided_structured_output_metadata(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizer()

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(
                load=lambda path_or_hf_repo: ("fake-model", fake_tokenizer),
                generate=lambda **kwargs: '{"summary":"ok"}',
                generate_stream=lambda **kwargs: ['{"summary":"ok"}'],
            )
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime()
    manifest = _manifest(modality=(ModelModality.TEXT,))
    asyncio.run(runtime.load_model(manifest))

    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=32,
        temperature=0.0,
        structured_output=JSONSchemaResponseFormat(
            schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
            name="summary",
        ),
    )

    response = asyncio.run(runtime.generate(request))

    async def collect_stream() -> str:
        return "".join([chunk async for chunk in runtime.stream_generate(request)])

    streamed = asyncio.run(collect_stream())

    assert response.output_text == '{"summary":"ok"}'
    assert streamed == '{"summary":"ok"}'
    assert request.metadata["structured_output_runtime"]["runtime"] == runtime.name
    assert request.metadata["structured_output_runtime"]["mode"] == "json_schema"
    assert request.metadata["structured_output_runtime"]["enforcement"] == "prompt_guided"
    assert request.metadata["structured_output_runtime"]["decoder_enforced"] is False
    assert request.metadata["structured_output_runtime"]["fallback_used"] is True
    assert "does not expose decode-time constrained decoding" in request.metadata["structured_output_runtime"]["fallback_reason"]


def test_mlx_text_runtime_rejects_manifest_when_backend_cannot_describe_model_type(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "gemma4-bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text('{"model_type":"gemma4"}', encoding="utf-8")

    def fake_import(name: str):
        if name == "mlx_lm.utils":
            return SimpleNamespace(
                _get_classes=lambda config: (_ for _ in ()).throw(ValueError("Model type gemma4 not supported.")),
            )
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime()
    manifest = _manifest(
        model_id="gemma4-text-fast-path",
        source_path=str(bundle_dir),
        modality=(ModelModality.TEXT, ModelModality.MULTIMODAL),
    )

    assert runtime.supports_manifest(manifest) is False


def test_mlx_text_runtime_accepts_manifest_when_backend_supports_model_type(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "supported-bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text('{"model_type":"supported"}', encoding="utf-8")

    monkeypatch.setattr(
        "lewlm.runtime.mlx_text.runtime.import_module",
        lambda name: SimpleNamespace(_get_classes=lambda config: ("Model", "Args")),
    )

    runtime = MLXTextRuntime()
    manifest = _manifest(
        model_id="supported-text-model",
        source_path=str(bundle_dir),
        modality=(ModelModality.TEXT,),
    )

    assert runtime.supports_manifest(manifest) is True


def test_mlx_text_runtime_applies_kv_cache_and_prefill_controls(monkeypatch, tmp_path: Path) -> None:
    fake_tokenizer = FakeTokenizer()
    captured: dict[str, object] = {}

    def fake_load(*, path_or_hf_repo: str, kv_cache_config: dict[str, int]):
        captured["load"] = {
            "path_or_hf_repo": path_or_hf_repo,
            "kv_cache_config": kv_cache_config,
        }
        return "fake-model", fake_tokenizer

    def fake_generate(
        *,
        model,
        tokenizer,
        prompt,
        max_tokens,
        kv_cache_quantization_bits: int,
        prefill_token_batch_size: int,
        prompt_tokens: list[int],
        verbose: bool = False,
    ):
        captured["generate"] = {
            "model": model,
            "tokenizer": tokenizer,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "kv_cache_quantization_bits": kv_cache_quantization_bits,
            "prefill_token_batch_size": prefill_token_batch_size,
            "prompt_tokens": prompt_tokens,
            "verbose": verbose,
        }
        return {
            "text": "optimized output",
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": 2,
                "total_tokens": len(prompt_tokens) + 2,
            },
        }

    fake_module = SimpleNamespace(
        load=fake_load,
        generate=fake_generate,
        generate_stream=lambda **kwargs: [],
    )
    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", lambda name: fake_module)

    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        kv_cache_page_size=128,
        kv_cache_max_pages=24,
        kv_cache_quantization_bits=6,
        prefill_token_batch_size=48,
    )
    runtime = MLXTextRuntime(settings=settings)
    manifest = _manifest(model_id="optimized-model", modality=(ModelModality.TEXT,))

    asyncio.run(runtime.load_model(manifest))
    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=32,
        temperature=0.0,
    )
    response = asyncio.run(
        runtime.generate(request),
    )
    snapshot = runtime.performance_feature_snapshot()

    assert captured["load"] == {
        "path_or_hf_repo": manifest.source_path,
        "kv_cache_config": {"page_size": 128, "max_pages": 24, "quantization_bits": 6},
    }
    assert captured["generate"] == {
        "model": "fake-model",
        "tokenizer": fake_tokenizer,
        "prompt": list("user: hello\nassistant:".encode("utf-8")),
        "max_tokens": 32,
        "kv_cache_quantization_bits": 6,
        "prefill_token_batch_size": 48,
        "prompt_tokens": list("user: hello\nassistant:".encode("utf-8")),
        "verbose": False,
    }
    assert response.output_text == "optimized output"
    assert request.metadata["performance_controls"]["load"]["paged_kv_cache"]["effective"] == "enabled"
    assert request.metadata["performance_controls"]["load"]["kv_cache_quantization"]["effective"] == "enabled"
    assert request.metadata["performance_controls"]["generate"]["prefill_optimization"]["effective"] == "enabled"
    assert request.metadata["performance_controls"]["generate"]["prefill_optimization"]["effective_prefill_token_batch_size"] == 48
    assert request.metadata["prefix_cache"]["page_size_tokens"] == 128
    assert request.metadata["kv_residency"]["queue_lane"] == "decode"
    assert request.metadata["kv_residency"]["requested_pages"] >= 1
    assert request.metadata["kv_residency"]["active_pages"] >= 1
    assert request.metadata["kv_residency"]["pressure_level"] in {"low", "medium", "high", "overflow"}
    assert snapshot["paged_kv_cache"]["supported"] is True
    assert snapshot["paged_kv_cache"]["active"] is True
    assert snapshot["paged_kv_cache"]["metrics"]["requests_using_paged_kv"] == 1
    assert snapshot["paged_kv_cache"]["metrics"]["resident_pages"] >= 1
    assert snapshot["paged_kv_cache"]["metrics"]["active_pages"] == 0
    assert snapshot["paged_kv_cache"]["metrics"]["native_control_supported"] is True
    assert snapshot["kv_cache_quantization"]["metrics"]["requests_using_quantized_kv"] == 1
    assert snapshot["prefill_optimization"]["metrics"]["optimized_requests"] == 1


def test_mlx_text_runtime_runs_lewlm_owned_draft_controller(monkeypatch, tmp_path: Path) -> None:
    fake_tokenizer = FakeTokenizer()
    captured: dict[str, object] = {"prefill": [], "primary": [], "draft": [], "stream_called": False}

    def fake_load(*, path_or_hf_repo: str):
        return f"model:{path_or_hf_repo}", fake_tokenizer

    def fake_generate(*, model, tokenizer, prompt, max_tokens, verbose=False, prompt_cache=None, **kwargs):
        prompt_tokens = list(prompt) if isinstance(prompt, list) else list(str(prompt).encode("utf-8"))
        cached_tokens = list(prompt_cache.get("tokens", [])) if isinstance(prompt_cache, dict) else []
        if isinstance(prompt_cache, dict) and max_tokens == 0:
            prompt_cache["tokens"] = [*cached_tokens, *prompt_tokens]
            prefill_calls = captured["prefill"]
            assert isinstance(prefill_calls, list)
            prefill_calls.append({"model": model, "prompt": prompt_tokens})
            return {"text": "", "usage": {}}
        call_record = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "prompt_cache_tokens": cached_tokens,
            "verbose": verbose,
        }
        if model == "model:/tmp/draft-model":
            draft_calls = captured["draft"]
            assert isinstance(draft_calls, list)
            draft_calls.append(call_record)
            return {"text": "OK", "usage": {}}
        primary_calls = captured["primary"]
        assert isinstance(primary_calls, list)
        primary_calls.append(call_record)
        return {"text": "OK", "usage": {}}

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(
                load=fake_load,
                generate=fake_generate,
                generate_stream=lambda **kwargs: (_ for _ in ()).throw(AssertionError("generate_stream should not be used")),
            )
        if name == "mlx_lm.models.cache":
            return SimpleNamespace(
                make_prompt_cache=lambda model: {"tokens": []},
                trim_prompt_cache=lambda cache, trim_count: cache.update(
                    {"tokens": list(cache.get("tokens", []))[:-trim_count] if trim_count > 0 else list(cache.get("tokens", []))}
                ),
            )
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            speculative_decoding_enabled=True,
            speculative_decoding_num_draft_tokens=2,
        ),
    )
    primary_manifest = _manifest(
        model_id="primary-model",
        source_path="/tmp/primary-model",
        modality=(ModelModality.TEXT,),
    )
    draft_manifest = _manifest(
        model_id="draft-model",
        source_path="/tmp/draft-model",
        modality=(ModelModality.TEXT,),
    )

    asyncio.run(runtime.load_model(primary_manifest))
    asyncio.run(runtime.load_model(draft_manifest))
    request = GenerateRequest(
        model_id=primary_manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=2,
        temperature=0.0,
        speculation=GenerateSpeculation(
            mode=SpeculationMode.DRAFT_MODEL,
            draft_model_id=draft_manifest.model_id,
            num_draft_tokens=2,
        ),
    )
    response = asyncio.run(runtime.generate(request))
    snapshot = runtime.performance_feature_snapshot()

    prefill_calls = captured["prefill"]
    primary_calls = captured["primary"]
    draft_calls = captured["draft"]
    assert isinstance(prefill_calls, list)
    assert isinstance(primary_calls, list)
    assert isinstance(draft_calls, list)
    assert len(prefill_calls) == 1
    assert primary_calls == [
        {
            "model": "model:/tmp/primary-model",
            "prompt": [ord(":")],
            "max_tokens": 2,
            "prompt_cache_tokens": list("user: hello\nassistant".encode("utf-8")),
            "verbose": False,
        },
    ]
    assert draft_calls == [
        {
            "model": "model:/tmp/draft-model",
            "prompt": "user: hello\nassistant:",
            "max_tokens": 2,
            "prompt_cache_tokens": [],
            "verbose": False,
        },
    ]
    assert response.output_text == "OK"
    assert response.usage["drafted_tokens"] == 2
    assert response.usage["accepted_tokens"] == 2
    assert response.usage["verified_tokens"] == 2
    assert response.usage["rejected_tokens"] == 0
    assert request.metadata["speculation_execution_path"] == "lewlm_controller"
    assert request.metadata["speculation_runtime"]["controller"] == "draft_verify"
    assert snapshot["speculative_decoding"]["supported"] is True
    assert snapshot["speculative_decoding"]["metrics"]["controller_owned_requests"] == 1
    assert snapshot["speculative_decoding"]["metrics"]["accepted_tokens"] == 2


def test_mlx_text_runtime_tracks_rejected_tokens_in_owned_draft_controller(monkeypatch, tmp_path: Path) -> None:
    fake_tokenizer = FakeTokenizer()

    def fake_load(*, path_or_hf_repo: str):
        return f"model:{path_or_hf_repo}", fake_tokenizer

    def fake_generate(*, model, tokenizer, prompt, max_tokens, verbose=False, prompt_cache=None, **kwargs):
        prompt_tokens = list(prompt) if isinstance(prompt, list) else list(str(prompt).encode("utf-8"))
        if isinstance(prompt_cache, dict) and max_tokens == 0:
            prompt_cache["tokens"] = [*list(prompt_cache.get("tokens", [])), *prompt_tokens]
            return {"text": "", "usage": {}}
        if model == "model:/tmp/draft-model":
            return {"text": "OX", "usage": {}}
        return {"text": "OY", "usage": {}}

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(
                load=fake_load,
                generate=fake_generate,
                generate_stream=lambda **kwargs: (_ for _ in ()).throw(AssertionError("generate_stream should not be used")),
            )
        if name == "mlx_lm.models.cache":
            return SimpleNamespace(
                make_prompt_cache=lambda model: {"tokens": []},
                trim_prompt_cache=lambda cache, trim_count: cache.update(
                    {"tokens": list(cache.get("tokens", []))[:-trim_count] if trim_count > 0 else list(cache.get("tokens", []))}
                ),
            )
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            speculative_decoding_enabled=True,
            speculative_decoding_num_draft_tokens=2,
        ),
    )
    primary_manifest = _manifest(model_id="primary-model", source_path="/tmp/primary-model", modality=(ModelModality.TEXT,))
    draft_manifest = _manifest(model_id="draft-model", source_path="/tmp/draft-model", modality=(ModelModality.TEXT,))

    asyncio.run(runtime.load_model(primary_manifest))
    asyncio.run(runtime.load_model(draft_manifest))
    request = GenerateRequest(
        model_id=primary_manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=2,
        temperature=0.0,
        speculation=GenerateSpeculation(
            mode=SpeculationMode.DRAFT_MODEL,
            draft_model_id=draft_manifest.model_id,
            num_draft_tokens=2,
        ),
    )
    response = asyncio.run(runtime.generate(request))

    assert response.output_text == "OY"
    assert response.usage["drafted_tokens"] == 2
    assert response.usage["accepted_tokens"] == 1
    assert response.usage["verified_tokens"] == 2
    assert response.usage["rejected_tokens"] == 1
    assert response.usage["rollback_tokens"] == 1


def test_mlx_text_runtime_passes_heterogeneous_vocab_parameter_to_stream_generate(monkeypatch, tmp_path: Path) -> None:
    fake_tokenizer = FakeTokenizer()
    captured: dict[str, object] = {}

    def fake_load(*, path_or_hf_repo: str):
        return f"model:{path_or_hf_repo}", fake_tokenizer

    def fake_generate_stream(*, model, tokenizer, prompt, max_tokens, swift, verbose=False):
        captured["generate_stream"] = {
            "model": model,
            "tokenizer": tokenizer,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "swift": swift,
            "verbose": verbose,
        }
        return [SimpleNamespace(text="heterogeneous output", from_draft=False)]

    fake_module = SimpleNamespace(
        load=fake_load,
        generate=lambda **kwargs: "unused",
        generate_stream=fake_generate_stream,
    )
    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", lambda name: fake_module)

    runtime = MLXTextRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            speculative_decoding_enabled=True,
        ),
    )
    primary_manifest = _manifest(
        model_id="primary-model",
        source_path="/tmp/primary-model",
        modality=(ModelModality.TEXT,),
    )

    asyncio.run(runtime.load_model(primary_manifest))
    response = asyncio.run(
        runtime.generate(
            GenerateRequest(
                model_id=primary_manifest.model_id,
                messages=[GenerateMessage(role="user", content="hello")],
                max_tokens=16,
                temperature=0.0,
                speculation=GenerateSpeculation(
                    mode=SpeculationMode.HETEROGENEOUS_VOCAB,
                    parameters={"backend_parameter": "swift", "backend_value": "swift-head"},
                ),
            ),
        ),
    )

    assert captured["generate_stream"] == {
        "model": "model:/tmp/primary-model",
        "tokenizer": fake_tokenizer,
        "prompt": "user: hello\nassistant:",
        "max_tokens": 16,
        "swift": "swift-head",
        "verbose": False,
    }
    assert response.output_text == "heterogeneous output"


def test_mlx_text_runtime_restores_persisted_prefix_cache(monkeypatch, tmp_path: Path) -> None:
    fake_tokenizer = FakeTokenizer()
    captured: list[dict[str, object]] = []

    def fake_load(*, path_or_hf_repo: str):
        return f"model:{path_or_hf_repo}", fake_tokenizer

    def fake_generate(*, model, tokenizer, prompt, max_tokens, verbose=False, prompt_cache=None, **kwargs):
        prompt_tokens = list(prompt) if isinstance(prompt, list) else list(str(prompt).encode("utf-8"))
        cache_tokens = []
        if isinstance(prompt_cache, dict):
            cache_tokens = list(prompt_cache.get("tokens", []))
            if max_tokens == 0:
                prompt_cache["tokens"] = [*cache_tokens, *prompt_tokens]
                cache_tokens = list(prompt_cache["tokens"])
        captured.append(
            {
                "model": model,
                "tokenizer": tokenizer,
                "prompt": prompt_tokens,
                "max_tokens": max_tokens,
                "verbose": verbose,
                "prompt_cache_tokens": cache_tokens,
            },
        )
        return {
            "text": "cached output",
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": 2,
                "total_tokens": len(prompt_tokens) + 2,
            },
        }

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(load=fake_load, generate=fake_generate, generate_stream=lambda **kwargs: [])
        if name == "mlx_lm.models.cache":
            return SimpleNamespace(
                make_prompt_cache=lambda model: {"tokens": []},
                trim_prompt_cache=lambda cache, trim_count: cache.update(
                    {"tokens": list(cache.get("tokens", []))[:-trim_count] if trim_count > 0 else list(cache.get("tokens", []))}
                ),
            )
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    settings = LewLMSettings(data_dir=tmp_path / "state")
    manifest = _manifest(model_id="persistent-model", modality=(ModelModality.TEXT,))

    runtime_one = MLXTextRuntime(settings=settings)
    asyncio.run(runtime_one.load_model(manifest))
    asyncio.run(
        runtime_one.generate(
            GenerateRequest(
                model_id=manifest.model_id,
                messages=[GenerateMessage(role="user", content="hello")],
                max_tokens=8,
                temperature=0.0,
            ),
        ),
    )

    captured.clear()
    runtime_two = MLXTextRuntime(settings=settings)
    asyncio.run(runtime_two.load_model(manifest))
    response = asyncio.run(
        runtime_two.generate(
            GenerateRequest(
                model_id=manifest.model_id,
                messages=[GenerateMessage(role="user", content="help")],
                max_tokens=8,
                temperature=0.0,
            ),
        ),
    )
    snapshot = runtime_two.performance_feature_snapshot()

    assert response.output_text == "cached output"
    assert len(captured) == 2
    assert captured[0]["max_tokens"] == 0
    assert captured[1]["max_tokens"] == 8
    assert captured[1]["prompt"] == [ord(":")]
    assert snapshot["persistent_multi_context_cache"]["supported"] is True
    assert snapshot["persistent_multi_context_cache"]["active"] is True
    assert snapshot["persistent_multi_context_cache"]["metrics"]["persistent_cache_hits"] == 1
    assert snapshot["persistent_multi_context_cache"]["metrics"]["cache_restores"] == 1
    assert snapshot["persistent_multi_context_cache"]["metrics"]["page_restores"] == 1
    assert snapshot["persistent_multi_context_cache"]["metrics"]["persisted_page_count"] == 2


def test_mlx_text_runtime_exposes_prefix_cache_admission_preview(monkeypatch, tmp_path: Path) -> None:
    fake_tokenizer = FakeTokenizer()

    def fake_load(*, path_or_hf_repo: str):
        return f"model:{path_or_hf_repo}", fake_tokenizer

    def fake_generate(*, model, tokenizer, prompt, max_tokens, verbose=False, prompt_cache=None, **kwargs):
        prompt_tokens = list(prompt) if isinstance(prompt, list) else list(str(prompt).encode("utf-8"))
        if isinstance(prompt_cache, dict) and max_tokens == 0:
            prompt_cache["tokens"] = [*list(prompt_cache.get("tokens", [])), *prompt_tokens]
        return {"text": "cached output", "usage": {"prompt_tokens": len(prompt_tokens), "completion_tokens": 1}}

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(load=fake_load, generate=fake_generate, generate_stream=lambda **kwargs: [])
        if name == "mlx_lm.models.cache":
            return SimpleNamespace(
                make_prompt_cache=lambda model: {"tokens": []},
                trim_prompt_cache=lambda cache, trim_count: cache.update(
                    {"tokens": list(cache.get("tokens", []))[:-trim_count] if trim_count > 0 else list(cache.get("tokens", []))}
                ),
            )
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime(settings=LewLMSettings(data_dir=tmp_path / "state"))
    manifest = _manifest(model_id="preview-model", modality=(ModelModality.TEXT,))
    request_messages = [GenerateMessage(role="user", content="hello")]

    asyncio.run(runtime.load_model(manifest))
    asyncio.run(
        runtime.generate(
            GenerateRequest(
                model_id=manifest.model_id,
                messages=request_messages,
                max_tokens=8,
                temperature=0.0,
            ),
        ),
    )

    preview = runtime.prefix_cache_admission_preview(model_id=manifest.model_id, messages=request_messages)

    assert preview["supported"] is True
    assert preview["lookup_source"] == "resident"
    assert int(preview["cached_prefix_tokens"]) > 0
    assert preview["effective_prefill_tokens"] == 0
    assert int(preview["total_prompt_tokens"]) > int(preview["cached_prefix_tokens"])


def test_mlx_text_runtime_unload_invalidates_resident_prefix_cache(monkeypatch, tmp_path: Path) -> None:
    fake_tokenizer = FakeTokenizer()

    def fake_load(*, path_or_hf_repo: str):
        return f"model:{path_or_hf_repo}", fake_tokenizer

    def fake_generate(*, model, tokenizer, prompt, max_tokens, verbose=False, prompt_cache=None, **kwargs):
        prompt_tokens = list(prompt) if isinstance(prompt, list) else list(str(prompt).encode("utf-8"))
        if isinstance(prompt_cache, dict) and max_tokens == 0:
            prompt_cache["tokens"] = [*list(prompt_cache.get("tokens", [])), *prompt_tokens]
        return {"text": "cached output", "usage": {"prompt_tokens": len(prompt_tokens), "completion_tokens": 1}}

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(load=fake_load, generate=fake_generate, generate_stream=lambda **kwargs: [])
        if name == "mlx_lm.models.cache":
            return SimpleNamespace(
                make_prompt_cache=lambda model: {"tokens": []},
                trim_prompt_cache=lambda cache, trim_count: cache.update(
                    {"tokens": list(cache.get("tokens", []))[:-trim_count] if trim_count > 0 else list(cache.get("tokens", []))}
                ),
            )
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime(settings=LewLMSettings(data_dir=tmp_path / "state"))
    manifest = _manifest(model_id="invalidate-model", modality=(ModelModality.TEXT,))

    asyncio.run(runtime.load_model(manifest))
    asyncio.run(
        runtime.generate(
            GenerateRequest(
                model_id=manifest.model_id,
                messages=[GenerateMessage(role="user", content="hello")],
                max_tokens=8,
                temperature=0.0,
            ),
        ),
    )

    before = runtime.performance_feature_snapshot()
    asyncio.run(runtime.unload_model(manifest.model_id))
    after = runtime.performance_feature_snapshot()

    assert before["prefix_cache"]["metrics"]["resident_cache_entries"] == 1
    assert before["persistent_multi_context_cache"]["metrics"]["persisted_cache_entries"] == 1
    assert after["prefix_cache"]["metrics"]["resident_cache_entries"] == 0
    assert after["persistent_multi_context_cache"]["metrics"]["persisted_cache_entries"] == 1
    assert after["prefix_cache"]["metrics"]["cache_invalidations"] == 1


def test_mlx_text_runtime_uses_graph_compile_and_attention_kernel(monkeypatch, tmp_path: Path) -> None:
    fake_tokenizer = FakeTokenizer()
    captured: dict[str, object] = {"compiled_calls": 0}

    def fake_load(*, path_or_hf_repo: str):
        return "fake-model", fake_tokenizer

    def fake_generate(*, model, tokenizer, prompt, max_tokens, attention_kernel: str | None = None, verbose: bool = False):
        captured["generate"] = {
            "model": model,
            "tokenizer": tokenizer,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "attention_kernel": attention_kernel,
            "verbose": verbose,
        }
        return {
            "text": "accelerated output",
            "usage": {
                "completion_tokens": 2,
                "prompt_tokens": len(prompt),
                "total_tokens": len(prompt) + 2,
            },
        }

    def fake_compile(callable_obj):
        def compiled(**kwargs):
            captured["compiled_calls"] = int(captured["compiled_calls"]) + 1
            return callable_obj(**kwargs)

        return compiled

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(load=fake_load, generate=fake_generate, generate_stream=lambda **kwargs: [])
        if name == "mlx.core":
            return SimpleNamespace(compile=fake_compile)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            mlx_graph_compile_enabled=True,
            mlx_attention_kernel_mode="flash_attention",
        ),
    )
    manifest = _manifest(model_id="accelerated-model", modality=(ModelModality.TEXT,))

    asyncio.run(runtime.load_model(manifest))
    response = asyncio.run(
        runtime.generate(
            GenerateRequest(
                model_id=manifest.model_id,
                messages=[GenerateMessage(role="user", content="hello")],
                max_tokens=16,
                temperature=0.0,
            ),
        ),
    )
    snapshot = runtime.performance_feature_snapshot()

    assert captured["compiled_calls"] == 1
    assert captured["generate"] == {
        "model": "fake-model",
        "tokenizer": fake_tokenizer,
        "prompt": list("user: hello\nassistant:".encode("utf-8")),
        "max_tokens": 16,
        "attention_kernel": "flash_attention",
        "verbose": False,
    }
    assert response.output_text == "accelerated output"
    assert snapshot["graph_compilation"]["supported"] is True
    assert snapshot["graph_compilation"]["active"] is True
    assert snapshot["graph_compilation"]["metrics"]["compiled_requests"] == 1
    assert snapshot["attention_kernel_acceleration"]["supported"] is True
    assert snapshot["attention_kernel_acceleration"]["active"] is True
    assert snapshot["attention_kernel_acceleration"]["metrics"]["flash_attention_requests"] == 1
    assert snapshot["attention_kernel_acceleration"]["metrics"]["last_kernel_path"] == "flash_attention"


def test_mlx_text_runtime_accelerates_prefix_cache_prefill_and_decode(monkeypatch, tmp_path: Path) -> None:
    fake_tokenizer = FakeTokenizer()
    captured: dict[str, object] = {"compiled_calls": 0, "calls": []}

    def fake_load(*, path_or_hf_repo: str):
        return "fake-model", fake_tokenizer

    def fake_generate(
        *,
        model,
        tokenizer,
        prompt,
        max_tokens,
        prompt_cache=None,
        attention_kernel: str | None = None,
        verbose: bool = False,
    ):
        prompt_tokens = list(prompt) if isinstance(prompt, list) else list(str(prompt).encode("utf-8"))
        if isinstance(prompt_cache, dict) and max_tokens == 0:
            prompt_cache["tokens"] = [*list(prompt_cache.get("tokens", [])), *prompt_tokens]
        calls = captured["calls"]
        assert isinstance(calls, list)
        calls.append(
            {
                "prompt": prompt_tokens,
                "max_tokens": max_tokens,
                "attention_kernel": attention_kernel,
                "prompt_cache_tokens": list(prompt_cache.get("tokens", [])) if isinstance(prompt_cache, dict) else [],
                "verbose": verbose,
            },
        )
        return {
            "text": "accelerated cached output",
            "usage": {
                "completion_tokens": 2,
                "prompt_tokens": len(prompt_tokens),
                "total_tokens": len(prompt_tokens) + 2,
            },
        }

    def fake_compile(callable_obj):
        def compiled(**kwargs):
            captured["compiled_calls"] = int(captured["compiled_calls"]) + 1
            return callable_obj(**kwargs)

        return compiled

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(load=fake_load, generate=fake_generate, generate_stream=lambda **kwargs: [])
        if name == "mlx_lm.models.cache":
            return SimpleNamespace(
                make_prompt_cache=lambda model: {"tokens": []},
                trim_prompt_cache=lambda cache, trim_count: cache.update(
                    {"tokens": list(cache.get("tokens", []))[:-trim_count] if trim_count > 0 else list(cache.get("tokens", []))}
                ),
            )
        if name == "mlx.core":
            return SimpleNamespace(compile=fake_compile)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            mlx_graph_compile_enabled=True,
            mlx_attention_kernel_mode="flash_attention",
        ),
    )
    manifest = _manifest(model_id="cached-accelerated-model", modality=(ModelModality.TEXT,))
    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=8,
        temperature=0.0,
    )

    asyncio.run(runtime.load_model(manifest))
    response = asyncio.run(runtime.generate(request))

    calls = captured["calls"]
    assert isinstance(calls, list)
    assert captured["compiled_calls"] == 2
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 0
    assert calls[0]["attention_kernel"] == "flash_attention"
    assert calls[1]["max_tokens"] == 8
    assert calls[1]["prompt"] == [ord(":")]
    assert calls[1]["attention_kernel"] == "flash_attention"
    assert response.output_text == "accelerated cached output"
    assert request.metadata["prefix_cache"]["prefilled_uncached_tokens"] > 0
    assert request.metadata["mlx_acceleration"]["compile_state"] == "prefill+decode"
    assert request.metadata["mlx_acceleration"]["phase_details"]["prefill"]["effective_graph_compile"] is True
    assert request.metadata["mlx_acceleration"]["phase_details"]["prefill"]["effective_kernel_path"] == "flash_attention"
    assert request.metadata["mlx_acceleration"]["phase_details"]["decode"]["effective_graph_compile"] is True
    assert request.metadata["mlx_acceleration"]["phase_details"]["decode"]["effective_kernel_path"] == "flash_attention"


def test_mlx_text_runtime_falls_back_to_stock_when_acceleration_fails(monkeypatch, tmp_path: Path) -> None:
    fake_tokenizer = FakeTokenizer()
    captured: dict[str, object] = {"calls": []}

    def fake_load(*, path_or_hf_repo: str):
        return "fake-model", fake_tokenizer

    def fake_generate(
        *,
        model,
        tokenizer,
        prompt,
        max_tokens,
        attention_kernel: str | None = None,
        verbose: bool = False,
    ):
        calls = captured["calls"]
        assert isinstance(calls, list)
        calls.append(attention_kernel or "stock")
        if attention_kernel is not None:
            raise RuntimeError("accelerated path rejected")
        return {"text": "stock output"}

    def fake_compile(callable_obj):
        def compiled(**kwargs):
            return callable_obj(**kwargs)

        return compiled

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(load=fake_load, generate=fake_generate, generate_stream=lambda **kwargs: [])
        if name == "mlx.core":
            return SimpleNamespace(compile=fake_compile)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            mlx_graph_compile_enabled=True,
            mlx_attention_kernel_mode="flash_attention",
        ),
    )
    manifest = _manifest(model_id="fallback-model", modality=(ModelModality.TEXT,))

    asyncio.run(runtime.load_model(manifest))
    response = asyncio.run(
        runtime.generate(
            GenerateRequest(
                model_id=manifest.model_id,
                messages=[GenerateMessage(role="user", content="hello")],
                max_tokens=16,
                temperature=0.0,
            ),
        ),
    )
    snapshot = runtime.performance_feature_snapshot()

    assert captured["calls"] == ["flash_attention", "stock"]
    assert response.output_text == "stock output"
    assert snapshot["graph_compilation"]["metrics"]["compile_fallback_requests"] >= 1
    assert snapshot["graph_compilation"]["metrics"]["compile_failures"] >= 1
    assert snapshot["attention_kernel_acceleration"]["metrics"]["kernel_fallback_requests"] == 1
    assert snapshot["attention_kernel_acceleration"]["metrics"]["stock_requests"] == 1
    assert snapshot["attention_kernel_acceleration"]["metrics"]["last_kernel_path"] == "stock"


def test_mlx_text_runtime_generate_batch_uses_native_batch_generate(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizer()
    captured: dict[str, object] = {}

    def fake_batch_generate(
        *,
        model,
        tokenizer,
        prompts,
        prompt_caches,
        max_tokens,
        verbose=False,
        return_prompt_caches=False,
        prefill_step_size=None,
        prefill_batch_size=None,
        completion_batch_size=None,
    ):
        captured["batch_generate"] = {
            "model": model,
            "tokenizer": tokenizer,
            "prompts": prompts,
            "prompt_caches": prompt_caches,
            "max_tokens": max_tokens,
            "verbose": verbose,
            "return_prompt_caches": return_prompt_caches,
            "prefill_step_size": prefill_step_size,
            "prefill_batch_size": prefill_batch_size,
            "completion_batch_size": completion_batch_size,
        }
        return SimpleNamespace(texts=["first", "second"])

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(
                load=lambda path_or_hf_repo: ("fake-model", fake_tokenizer),
                generate=lambda **kwargs: "unused",
                generate_stream=lambda **kwargs: [],
            )
        if name == "mlx_lm.generate":
            return SimpleNamespace(batch_generate=fake_batch_generate)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime()
    manifest = _manifest(model_id="batched-model", modality=(ModelModality.TEXT,))
    asyncio.run(runtime.load_model(manifest))
    requests = [
        GenerateRequest(
            model_id=manifest.model_id,
            messages=[GenerateMessage(role="user", content="hello")],
            max_tokens=8,
            temperature=0.0,
        ),
        GenerateRequest(
            model_id=manifest.model_id,
            messages=[GenerateMessage(role="user", content="world")],
            max_tokens=6,
            temperature=0.0,
        ),
    ]

    responses = asyncio.run(runtime.generate_batch(requests))
    snapshot = runtime.performance_feature_snapshot()

    assert runtime.supports_continuous_batching(CapabilityName.CHAT) is True
    assert runtime.supports_continuous_batching(CapabilityName.STREAMING) is False
    assert captured["batch_generate"] == {
        "model": "fake-model",
        "tokenizer": fake_tokenizer,
        "prompts": [
            list("user: hello\nassistant:".encode("utf-8")),
            list("user: world\nassistant:".encode("utf-8")),
        ],
        "prompt_caches": [None, None],
        "max_tokens": [8, 6],
        "verbose": False,
        "return_prompt_caches": False,
        "prefill_step_size": 512,
        "prefill_batch_size": 4,
        "completion_batch_size": 4,
    }
    assert [response.output_text for response in responses] == ["first", "second"]
    assert snapshot["continuous_batching"]["metrics"]["chat_batch_calls"] == 1
    assert snapshot["continuous_batching"]["metrics"]["batched_requests"] == 2
    assert snapshot["continuous_batching"]["metrics"]["max_batch_size"] == 2
    assert snapshot["continuous_batching"]["ownership"] == "backend_native"
    assert snapshot["paged_kv_cache"]["metrics"]["requests_using_paged_kv"] == 2
    assert snapshot["paged_kv_cache"]["metrics"]["resident_pages"] >= 1
    assert snapshot["paged_kv_cache"]["metrics"]["active_pages"] == 0
    assert all(request.metadata["kv_residency"]["requested_pages"] >= 1 for request in requests)
    assert all(
        request.metadata["native_batching"]
        == {
            "capability": CapabilityName.CHAT.value,
            "supported": True,
            "active": True,
            "backend": "mlx_lm.batch_generate",
            "batch_size": 2,
            "stock_single_request_path": False,
            "fallback": False,
            "ownership": "backend_native",
        }
        for request in requests
    )


def test_mlx_text_runtime_generate_uses_lewlm_owned_persistent_batch_controller(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizer()
    captured: dict[str, object] = {"insert_calls": []}

    class FakeBatchGenerator:
        def __init__(self, model, **kwargs):
            captured["generator_init"] = {"model": model, "kwargs": kwargs}
            self._uids: list[int] = []
            self._events: list[list[SimpleNamespace]] = []

        def insert(self, *, prompts, max_tokens, caches):
            if not self._uids and len(prompts) == 2:
                uids = [11, 22]
            elif not self._uids:
                uids = [11]
            else:
                uids = [22]
            self._uids.extend(uids)
            insert_calls = captured["insert_calls"]
            assert isinstance(insert_calls, list)
            insert_calls.append(
                {
                    "prompts": prompts,
                    "max_tokens": max_tokens,
                    "caches": caches,
                    "uids": uids,
                },
            )
            if len(self._uids) >= 2 and not any(self._events):
                self._events.extend(
                    [
                        [SimpleNamespace(uid=11, token=79, finish_reason=None), SimpleNamespace(uid=22, token=80, finish_reason=None)],
                        [SimpleNamespace(uid=11, finish_reason="stop"), SimpleNamespace(uid=22, finish_reason="stop")],
                    ],
                )
            else:
                self._events.append([])
            return uids

        def next_generated(self):
            return self._events.pop(0) if self._events else []

        def close(self):
            captured["closed"] = True

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(
                load=lambda path_or_hf_repo: ("fake-model", fake_tokenizer),
                generate=lambda **kwargs: "unused",
                generate_stream=lambda **kwargs: [],
                BatchGenerator=FakeBatchGenerator,
            )
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime()
    manifest = _manifest(model_id="owned-batched-model", modality=(ModelModality.TEXT,))
    asyncio.run(runtime.load_model(manifest))

    async def run_requests():
        first = GenerateRequest(
            model_id=manifest.model_id,
            messages=[GenerateMessage(role="user", content="hello")],
            max_tokens=8,
            temperature=0.0,
        )
        second = GenerateRequest(
            model_id=manifest.model_id,
            messages=[GenerateMessage(role="user", content="world")],
            max_tokens=8,
            temperature=0.0,
        )
        responses = await asyncio.gather(runtime.generate(first), runtime.generate(second))
        return first, second, responses

    first_request, second_request, responses = asyncio.run(run_requests())
    snapshot = runtime.performance_feature_snapshot()

    assert [response.output_text for response in responses] == ["O", "P"]
    assert runtime.continuous_batching_ownership(CapabilityName.CHAT) == "lewlm_owned"
    assert snapshot["continuous_batching"]["ownership"] == "lewlm_owned"
    assert snapshot["continuous_batching"]["metrics"]["chat_batch_calls"] >= 1
    assert snapshot["continuous_batching"]["metrics"]["batched_requests"] == 2
    assert captured["closed"] is True
    assert all(
        request.metadata["native_batching"]["ownership"] == "lewlm_owned"
        and request.metadata["native_batching"]["backend"] == "lewlm.mlx_text_continuous_batch_scheduler"
        and request.metadata["native_batching"]["backend_primitive"] == "mlx_lm.BatchGenerator"
        and request.metadata["native_batching"]["persistent_scheduler"] is True
        for request in (first_request, second_request)
    )
    assert any(request.metadata["native_batching"]["batch_size"] == 2 for request in (first_request, second_request))


def test_mlx_text_runtime_stream_generate_batch_uses_native_batch_generator(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizer()
    generator_plans = [
        [
            [SimpleNamespace(uid=11, token=79, finish_reason=None), SimpleNamespace(uid=22, token=80, finish_reason=None)],
            [SimpleNamespace(uid=11, token=75, finish_reason=None), SimpleNamespace(uid=22, token=76, finish_reason=None)],
            [SimpleNamespace(uid=11, finish_reason="stop"), SimpleNamespace(uid=22, finish_reason="stop")],
            [],
        ],
    ]
    captured: dict[str, object] = {}

    class FakeBatchGenerator:
        def __init__(self, model, **kwargs):
            captured["generator_init"] = {"model": model, "kwargs": kwargs}
            self._events = [list(batch) for batch in generator_plans.pop(0)]

        def insert(self, *, prompts, max_tokens, caches):
            captured["insert"] = {
                "prompts": prompts,
                "max_tokens": max_tokens,
                "caches": caches,
            }
            return [11, 22]

        def next_generated(self):
            return self._events.pop(0)

        def close(self):
            captured["closed"] = True

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(
                load=lambda path_or_hf_repo: ("fake-model", fake_tokenizer),
                generate=lambda **kwargs: "unused",
                generate_stream=lambda **kwargs: [],
                BatchGenerator=FakeBatchGenerator,
            )
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime()
    manifest = _manifest(model_id="stream-batched-model", modality=(ModelModality.TEXT,))
    asyncio.run(runtime.load_model(manifest))
    requests = [
        GenerateRequest(
            model_id=manifest.model_id,
            messages=[GenerateMessage(role="user", content="hello")],
            max_tokens=8,
            temperature=0.0,
        ),
        GenerateRequest(
            model_id=manifest.model_id,
            messages=[GenerateMessage(role="user", content="world")],
            max_tokens=8,
            temperature=0.0,
        ),
    ]

    async def collect_chunks() -> list[tuple[int, str]]:
        return [
            (index, chunk)
            async for index, chunk in runtime.stream_generate_batch(requests)
        ]

    chunks = asyncio.run(collect_chunks())
    snapshot = runtime.performance_feature_snapshot()

    assert runtime.supports_continuous_batching(CapabilityName.CHAT) is True
    assert runtime.supports_continuous_batching(CapabilityName.STREAMING) is True
    assert captured["generator_init"] == {
        "model": "fake-model",
        "kwargs": {
            "max_tokens": 0,
            "prefill_step_size": 512,
            "prefill_batch_size": 4,
            "completion_batch_size": 4,
        },
    }
    assert captured["insert"] == {
        "prompts": [
            list("user: hello\nassistant:".encode("utf-8")),
            list("user: world\nassistant:".encode("utf-8")),
        ],
        "max_tokens": [8, 8],
        "caches": [None, None],
    }
    assert chunks == [(0, "O"), (1, "P"), (0, "K"), (1, "L")]
    assert captured["closed"] is True
    assert snapshot["continuous_batching"]["metrics"]["stream_batch_calls"] == 1
    assert snapshot["continuous_batching"]["metrics"]["batched_requests"] == 2
    assert snapshot["continuous_batching"]["metrics"]["max_batch_size"] == 2
    assert snapshot["continuous_batching"]["ownership"] == "lewlm_owned"
    assert snapshot["paged_kv_cache"]["metrics"]["requests_using_paged_kv"] == 2
    assert snapshot["paged_kv_cache"]["metrics"]["resident_pages"] >= 1
    assert snapshot["paged_kv_cache"]["metrics"]["active_pages"] == 0
    assert all(request.metadata["kv_residency"]["requested_pages"] >= 1 for request in requests)
    assert all(
        request.metadata["native_batching"]
        == {
            "capability": CapabilityName.STREAMING.value,
            "supported": True,
            "active": True,
            "backend": "mlx_lm.BatchGenerator",
            "batch_size": 2,
            "stock_single_request_path": False,
            "fallback": False,
            "ownership": "backend_native",
        }
        for request in requests
    )


def test_mlx_text_runtime_stream_generate_uses_lewlm_owned_persistent_batch_controller(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizer()
    captured: dict[str, object] = {"insert_calls": []}

    class FakeBatchGenerator:
        def __init__(self, model, **kwargs):
            captured["generator_init"] = {"model": model, "kwargs": kwargs}
            self._uids: list[int] = []
            self._events: list[list[SimpleNamespace]] = []

        def insert(self, *, prompts, max_tokens, caches):
            if not self._uids and len(prompts) == 2:
                uids = [11, 22]
            elif not self._uids:
                uids = [11]
            else:
                uids = [22]
            self._uids.extend(uids)
            insert_calls = captured["insert_calls"]
            assert isinstance(insert_calls, list)
            insert_calls.append(
                {
                    "prompts": prompts,
                    "max_tokens": max_tokens,
                    "caches": caches,
                    "uids": uids,
                },
            )
            if len(self._uids) >= 2 and not any(self._events):
                self._events.extend(
                    [
                        [SimpleNamespace(uid=11, token=79, finish_reason=None), SimpleNamespace(uid=22, token=80, finish_reason=None)],
                        [SimpleNamespace(uid=11, token=75, finish_reason=None), SimpleNamespace(uid=22, token=76, finish_reason=None)],
                        [SimpleNamespace(uid=11, finish_reason="stop"), SimpleNamespace(uid=22, finish_reason="stop")],
                    ],
                )
            else:
                self._events.append([])
            return uids

        def next_generated(self):
            return self._events.pop(0) if self._events else []

        def close(self):
            captured["closed"] = True

    def fake_import(name: str):
        if name == "mlx_lm":
            return SimpleNamespace(
                load=lambda path_or_hf_repo: ("fake-model", fake_tokenizer),
                generate=lambda **kwargs: "unused",
                generate_stream=lambda **kwargs: [],
                BatchGenerator=FakeBatchGenerator,
            )
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_text.runtime.import_module", fake_import)

    runtime = MLXTextRuntime()
    manifest = _manifest(model_id="owned-stream-model", modality=(ModelModality.TEXT,))
    asyncio.run(runtime.load_model(manifest))

    async def collect_output(request: GenerateRequest) -> str:
        return "".join([chunk async for chunk in runtime.stream_generate(request)])

    async def run_streams():
        first = GenerateRequest(
            model_id=manifest.model_id,
            messages=[GenerateMessage(role="user", content="hello")],
            max_tokens=8,
            temperature=0.0,
        )
        second = GenerateRequest(
            model_id=manifest.model_id,
            messages=[GenerateMessage(role="user", content="world")],
            max_tokens=8,
            temperature=0.0,
        )
        outputs = await asyncio.gather(collect_output(first), collect_output(second))
        return first, second, outputs

    first_request, second_request, outputs = asyncio.run(run_streams())
    snapshot = runtime.performance_feature_snapshot()

    assert outputs == ["OK", "PL"]
    assert runtime.continuous_batching_ownership(CapabilityName.STREAMING) == "lewlm_owned"
    assert snapshot["continuous_batching"]["ownership"] == "lewlm_owned"
    assert snapshot["continuous_batching"]["metrics"]["stream_batch_calls"] >= 1
    assert snapshot["continuous_batching"]["metrics"]["batched_requests"] == 2
    assert captured["closed"] is True
    assert all(
        request.metadata["native_batching"]["ownership"] == "lewlm_owned"
        and request.metadata["native_batching"]["backend"] == "lewlm.mlx_text_continuous_batch_scheduler"
        and request.metadata["native_batching"]["backend_primitive"] == "mlx_lm.BatchGenerator"
        and request.metadata["native_batching"]["persistent_scheduler"] is True
        for request in (first_request, second_request)
    )
    assert any(request.metadata["native_batching"]["batch_size"] == 2 for request in (first_request, second_request))


def _manifest(*, model_id: str = "semantic-model", source_path: str = "/tmp/semantic-model", modality: tuple[ModelModality, ...]) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        display_name=model_id,
        architecture_family="e5",
        modality=modality,
        source_path=source_path,
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
        estimated_memory_mb=512,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="fingerprint",
        last_validation_result=ModelValidationResult(
            status=ValidationState.VALID,
            message="ok",
        ),
    )
