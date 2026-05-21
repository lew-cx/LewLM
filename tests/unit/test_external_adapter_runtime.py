from __future__ import annotations

import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path, PosixPath
import threading
from typing import cast
from urllib.error import URLError
import wave

import asyncio
import pytest

import lewlm.runtime.adapters.openai_compatible as openai_compatible_runtime
from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    CapabilityName,
    ConversionStatus,
    GenerateAttachment,
    GenerateMessage,
    GenerateRequest,
    EmbeddingRequest,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RerankRequest,
    RuntimeAffinity,
    RuntimeReadinessState,
    ValidationState,
)
from lewlm.core.errors import RuntimeUnavailableError
from lewlm.runtime.adapters import LocalOpenAICompatibleAdapterRuntime, summarize_feature_preservation
from lewlm.structured_output import JSONSchemaResponseFormat


def test_external_adapter_runtime_requires_loopback_endpoint(tmp_path: Path) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://example.com:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)

    assert runtime.is_available() is False
    assert "loopback-only" in str(runtime.availability_reason())


def test_external_adapter_runtime_matches_advertised_model_and_reports_feature_preservation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
        external_accelerator_profile="vllm_mlx",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = ModelManifest(
        model_id="demo-model",
        display_name="Demo Model",
        architecture_family="llama",
        modality=(ModelModality.TEXT,),
        source_path=str(tmp_path / "demo-model"),
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
        estimated_memory_mb=512,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-model-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )

    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda method, path, payload: {"data": [{"id": "demo-model"}]},
    )

    report = runtime.candidate_report(manifest)
    feature_preservation = summarize_feature_preservation(
        native_features={
            "continuous_batching": {"supported": True},
            "prefix_cache": {"supported": True},
            "kv_cache_quantization": {"supported": True},
        },
        external_features=runtime.performance_feature_snapshot(),
    )

    assert report.available is True
    assert report.supports_manifest is True
    assert "continuous_batching" in feature_preservation["preserved"]
    assert "kv_cache_quantization" in feature_preservation["degraded"]


@pytest.mark.parametrize(
    ("system", "machine"),
    [
        ("Darwin", "arm64"),
        ("Linux", "x86_64"),
        ("Linux", "arm64"),
        ("Windows", "AMD64"),
    ],
)
def test_external_adapter_runtime_supports_documented_cross_platform_targets(
    tmp_path: Path,
    system: str,
    machine: str,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)

    assert runtime.supports_target_platform(system, machine) is True


def test_external_adapter_runtime_matches_gguf_manifest_by_file_stem(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
        external_accelerator_profile="vllm_local",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = ModelManifest(
        model_id="demo-gguf",
        display_name="Demo GGUF",
        architecture_family="llama",
        modality=(ModelModality.TEXT,),
        source_path=str(tmp_path / "demo-server-id.gguf"),
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
        estimated_memory_mb=512,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-gguf-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )

    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda method, path, payload: {"data": [{"id": "demo-server-id"}]},
    )

    report = runtime.candidate_report(manifest)

    assert report.available is True
    assert report.supports_manifest is True
    assert runtime.performance_feature_snapshot()["paged_kv_cache"]["metrics"]["adapter_profile"] == "vllm_local"


def test_external_adapter_runtime_matches_windows_style_gguf_paths_with_posix_pathlib(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
        external_accelerator_profile="vllm_local",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = ModelManifest(
        model_id="demo-gguf",
        display_name="Demo GGUF",
        architecture_family="llama",
        modality=(ModelModality.TEXT,),
        source_path="X:\\models\\demo-server-id.gguf",
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
        estimated_memory_mb=512,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-gguf-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )

    monkeypatch.setattr("lewlm.runtime.adapters.openai_compatible.Path", PosixPath)
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda method, path, payload: {"data": [{"id": "demo-server-id"}]},
    )

    report = runtime.candidate_report(manifest)

    assert report.available is True
    assert report.supports_manifest is True


def test_external_adapter_runtime_matches_converted_source_metadata_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = ModelManifest(
        model_id="phi-3-mini-hf_converted",
        display_name="phi-3-mini-hf (converted)",
        architecture_family="phi3",
        modality=(ModelModality.TEXT,),
        source_path=str(tmp_path / "phi-3-mini-hf_converted"),
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
        estimated_memory_mb=512,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="phi-3-mini-hf-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
        metadata={
            "source_model_id": "phi-3-mini-hf-a2a814fb3dbd",
            "source_display_name": "phi-3-mini-hf",
        },
    )

    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda method, path, payload: {"data": [{"id": "phi-3-mini-hf-a2a814fb3dbd"}]},
    )

    assert runtime.supports_manifest(manifest) is True


def test_external_adapter_runtime_surfaces_non_apple_local_profile_features(tmp_path: Path) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
        external_accelerator_profile="sglang_local",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)

    snapshot = runtime.performance_feature_snapshot()

    assert snapshot["continuous_batching"]["supported"] is True
    assert snapshot["continuous_batching"]["ownership"] == "backend_native"
    assert snapshot["prefix_cache"]["support_level"] == "partial"
    assert snapshot["prefix_cache"]["ownership"] == "partial"
    assert snapshot["paged_kv_cache"]["supported"] is True
    assert snapshot["paged_kv_cache"]["ownership"] == "partial"
    assert snapshot["speculative_decoding"]["supported"] is False
    assert snapshot["speculative_decoding"]["ownership"] == "unsupported"
    assert snapshot["speculative_decoding"]["metrics"]["adapter_profile"] == "sglang_local"
    assert snapshot["constrained_decoding"]["supported"] is True
    assert snapshot["constrained_decoding"]["ownership"] == "partial"
    assert snapshot["constrained_decoding"]["metrics"]["fallback_used"] is True


@pytest.mark.parametrize("profile", ["ollama_local", "llamacpp_server"])
def test_external_adapter_runtime_supports_generic_bridge_alias_profiles(
    tmp_path: Path,
    profile: str,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
        external_accelerator_profile=profile,
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)

    snapshot = runtime.performance_feature_snapshot()

    assert snapshot["continuous_batching"]["ownership"] == "partial"
    assert snapshot["prefix_cache"]["ownership"] == "unsupported"
    assert snapshot["constrained_decoding"]["ownership"] == "partial"
    assert snapshot["constrained_decoding"]["metrics"]["adapter_profile"] == profile


def test_external_adapter_runtime_records_prompt_guided_structured_output_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = _manifest(tmp_path, model_id="demo-model", display_name="Demo Model")

    def fake_request_json(method: str, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        if method == "GET" and path == "/v1/models":
            return {"data": [{"id": "demo-model"}]}
        assert method == "POST"
        assert path == "/v1/chat/completions"
        return {
            "choices": [{"message": {"content": '{"summary":"ok"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }

    monkeypatch.setattr(runtime, "_request_json", fake_request_json)

    asyncio.run(runtime.load_model(manifest))
    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[GenerateMessage(role="user", content="Return a summary")],
        max_tokens=8,
        temperature=0.0,
        structured_output=JSONSchemaResponseFormat(
            name="status",
            schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        ),
    )
    response = asyncio.run(runtime.generate(request))

    assert response.output_text == '{"summary":"ok"}'
    status = request.metadata["structured_output_runtime"]
    assert status["runtime"] == runtime.name
    assert status["enforcement"] == "prompt_guided"
    assert status["decoder_enforced"] is False
    assert status["fallback_used"] is True
    assert "adapter boundary" in status["fallback_reason"]


def test_external_adapter_runtime_reports_prompt_guided_structured_output_status(
    tmp_path: Path,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
        external_accelerator_profile="vllm_local",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)

    status = runtime.structured_output_runtime_status(
        JSONSchemaResponseFormat(
            name="status",
            schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        ),
    )

    assert status is not None
    assert status.runtime == runtime.name
    assert status.enforcement == "prompt_guided"
    assert status.decoder_enforced is False
    assert status.fallback_used is True
    assert "adapter contract" in status.fallback_reason


def test_external_adapter_runtime_matches_advertised_model_metadata_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
        external_accelerator_profile="vllm_local",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = _manifest(
        tmp_path,
        model_id="demo-gguf",
        display_name="Demo GGUF",
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
        source_name="demo-server-id.gguf",
    )

    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda method, path, payload: {
            "data": [
                {
                    "id": "mlx-server-42",
                    "root": "X:\\cache\\demo-server-id.gguf",
                    "metadata": {"aliases": ["demo-gguf"]},
                },
            ],
        },
    )

    report = runtime.candidate_report(manifest)

    assert report.available is True
    assert report.supports_manifest is True


def test_external_adapter_runtime_executes_embeddings_and_rerank(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
        external_accelerator_profile="vllm_local",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    embedding_manifest = ModelManifest(
        model_id="demo-embed",
        display_name="Demo Embed",
        architecture_family="e5",
        modality=(ModelModality.EMBEDDING,),
        source_path=str(tmp_path / "demo-embed"),
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
        estimated_memory_mb=512,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-embed-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )
    rerank_manifest = ModelManifest(
        model_id="demo-rerank",
        display_name="Demo Rerank",
        architecture_family="bge",
        modality=(ModelModality.RERANK,),
        source_path=str(tmp_path / "demo-rerank"),
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
        estimated_memory_mb=512,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-rerank-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )

    def fake_request_json(method: str, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        if method == "GET" and path == "/v1/models":
            return {"data": [{"id": "demo-embed"}, {"id": "demo-rerank"}]}
        if method == "POST" and path == "/v1/embeddings":
            if payload == {"model": "demo-embed", "input": ["LewLM semantic capability probe"]}:
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            assert payload == {"model": "demo-embed", "input": ["alpha", "beta"]}
            return {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            }
        if method == "POST" and path == "/v1/rerank":
            if payload == {
                "model": "demo-embed",
                "query": "LewLM semantic capability probe",
                "documents": ["LewLM semantic capability probe"],
                "top_n": 1,
            }:
                raise RuntimeUnavailableError(
                    "External accelerator request failed with HTTP 404.",
                    details={"runtime": runtime.name, "path": path, "body": "not found"},
                )
            if payload == {
                "model": "demo-rerank",
                "query": "LewLM semantic capability probe",
                "documents": ["LewLM semantic capability probe"],
                "top_n": 1,
            }:
                return {"results": [{"index": 0, "relevance_score": 1.0}]}
            assert payload == {
                "model": "demo-rerank",
                "query": "query",
                "documents": ["beta", "alpha"],
                "top_n": 1,
            }
            return {"results": [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.5}]}
        raise AssertionError((method, path, payload))

    monkeypatch.setattr(runtime, "_request_json", fake_request_json)

    assert runtime.supports_capability(CapabilityName.EMBEDDINGS) is True
    assert runtime.supports_capability(CapabilityName.RERANK) is True
    assert runtime.supports_manifest_capability(embedding_manifest, CapabilityName.EMBEDDINGS) is True
    assert runtime.supports_manifest_capability(rerank_manifest, CapabilityName.RERANK) is True

    asyncio.run(runtime.load_model(embedding_manifest))
    asyncio.run(runtime.load_model(rerank_manifest))
    embedding_response = asyncio.run(
        runtime.embed(EmbeddingRequest(model_id=embedding_manifest.model_id, inputs=["alpha", "beta"])),
    )
    rerank_response = asyncio.run(
        runtime.rerank(
            RerankRequest(
                model_id=rerank_manifest.model_id,
                query="query",
                documents=["beta", "alpha"],
                top_n=1,
            ),
        ),
    )

    assert embedding_response.usage["prompt_tokens"] == 4
    assert len(embedding_response.data) == 2
    assert embedding_response.data[0].embedding == [0.1, 0.2, 0.3]
    assert len(rerank_response.results) == 1
    assert rerank_response.results[0].index == 1
    assert rerank_response.results[0].document == "alpha"


def test_external_adapter_runtime_rejects_invalid_embedding_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = _manifest(
        tmp_path,
        model_id="demo-embed",
        display_name="Demo Embed",
        architecture_family="e5",
        modality=(ModelModality.EMBEDDING,),
    )

    def fake_request_json(method: str, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        if method == "GET" and path == "/v1/models":
            return {"data": [{"id": "demo-embed"}]}
        if method == "POST" and path == "/v1/embeddings":
            if payload == {"model": "demo-embed", "input": ["LewLM semantic capability probe"]}:
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        raise AssertionError((method, path, payload))

    monkeypatch.setattr(runtime, "_request_json", fake_request_json)

    asyncio.run(runtime.load_model(manifest))
    with pytest.raises(RuntimeUnavailableError, match="invalid `embeddings` payload"):
        asyncio.run(
            runtime.embed(
                EmbeddingRequest(model_id=manifest.model_id, inputs=["alpha", "beta"]),
            ),
        )


def test_external_adapter_runtime_supports_vision_chat_payloads(tmp_path: Path, monkeypatch) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlAbI4AAAAASUVORK5CYII=",
        ),
    )
    manifest = ModelManifest(
        model_id="demo-vision",
        display_name="Demo Vision",
        architecture_family="qwen2_vl",
        modality=(ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL),
        source_path=str(tmp_path / "demo-vision"),
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_VISION,),
        estimated_memory_mb=768,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-vision-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )
    chat_payloads: list[dict[str, object]] = []

    def fake_request_json(method: str, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        if path == "/v1/models":
            return {"data": [{"id": "demo-vision"}]}
        assert method == "POST"
        assert path == "/v1/chat/completions"
        assert payload is not None
        chat_payloads.append(payload)
        return {
            "choices": [{"message": {"content": "vision-ready"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(runtime, "_request_json", fake_request_json)

    assert runtime.supports_manifest_capability(manifest, CapabilityName.VISION) is True

    asyncio.run(runtime.load_model(manifest))
    response = asyncio.run(
        runtime.generate(
            GenerateRequest(
                model_id=manifest.model_id,
                messages=[
                    GenerateMessage(
                        role="user",
                        content="Describe the image",
                        attachments=[
                            GenerateAttachment(
                                attachment_type="image",
                                name=image_path.name,
                                source_path=str(image_path),
                                media_type="image/png",
                            ),
                        ],
                    ),
                ],
                max_tokens=8,
                temperature=0.0,
            ),
        ),
    )

    assert response.output_text == "vision-ready"
    final_payload = chat_payloads[-1]
    message_payload = final_payload["messages"][0]
    assert message_payload["role"] == "user"
    assert any(part["type"] == "image_url" for part in message_payload["content"])


def test_external_adapter_runtime_preserves_image_detail_in_chat_payload(tmp_path: Path, monkeypatch) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlAbI4AAAAASUVORK5CYII=",
        ),
    )
    manifest = ModelManifest(
        model_id="demo-vision",
        display_name="Demo Vision",
        architecture_family="qwen2_vl",
        modality=(ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL),
        source_path=str(tmp_path / "demo-vision"),
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_VISION,),
        estimated_memory_mb=768,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-vision-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )
    chat_payloads: list[dict[str, object]] = []

    def fake_request_json(method: str, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        if path == "/v1/models":
            return {"data": [{"id": "demo-vision"}]}
        assert method == "POST"
        assert path == "/v1/chat/completions"
        assert payload is not None
        chat_payloads.append(payload)
        return {
            "choices": [{"message": {"content": "vision-ready"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(runtime, "_request_json", fake_request_json)

    asyncio.run(runtime.load_model(manifest))
    asyncio.run(
        runtime.generate(
            GenerateRequest(
                model_id=manifest.model_id,
                messages=[
                    GenerateMessage(
                        role="user",
                        content="Describe the image",
                        attachments=[
                            GenerateAttachment(
                                attachment_type="image",
                                name=image_path.name,
                                source_path=str(image_path),
                                media_type="image/png",
                                detail="high",
                            ),
                        ],
                    ),
                ],
                max_tokens=8,
                temperature=0.0,
            ),
        ),
    )

    message_payload = chat_payloads[-1]["messages"][0]
    image_part = next(part for part in message_payload["content"] if part["type"] == "image_url")
    assert image_part["image_url"]["detail"] == "high"


def test_external_adapter_runtime_surfaces_explicit_vision_probe_endpoint_reason(tmp_path: Path, monkeypatch) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = _manifest(
        tmp_path,
        model_id="demo-vision",
        display_name="Demo Vision",
        architecture_family="qwen2_vl",
        modality=(ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL),
        runtime_affinity=(RuntimeAffinity.MLX_VISION,),
    )

    def fake_request_json(method: str, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        if method == "GET" and path == "/v1/models":
            return {"data": [{"id": "demo-vision"}]}
        if method == "POST" and path == "/v1/chat/completions":
            raise RuntimeUnavailableError(
                "External accelerator request failed with HTTP 404.",
                details={"runtime": runtime.name, "path": path, "body": "not found", "status_code": 404},
            )
        raise AssertionError((method, path, payload))

    monkeypatch.setattr(runtime, "_request_json", fake_request_json)

    assert runtime.supports_manifest_capability(manifest, CapabilityName.VISION) is False
    reason = runtime.manifest_capability_reason(manifest, CapabilityName.VISION)

    assert reason is not None
    assert "`/v1/chat/completions`" in reason
    assert "OpenAI-style image content blocks" in reason


def test_external_adapter_runtime_rejects_missing_image_attachment_path(tmp_path: Path, monkeypatch) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = _manifest(
        tmp_path,
        model_id="demo-vision",
        display_name="Demo Vision",
        architecture_family="qwen2_vl",
        modality=(ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL),
        runtime_affinity=(RuntimeAffinity.MLX_VISION,),
    )

    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda method, path, payload: {"data": [{"id": "demo-vision"}]}
        if path == "/v1/models"
        else {"choices": [{"message": {"content": "unreachable"}, "finish_reason": "stop"}]},
    )

    asyncio.run(runtime.load_model(manifest))
    with pytest.raises(RuntimeUnavailableError, match="does not exist"):
        asyncio.run(
            runtime.generate(
                GenerateRequest(
                    model_id=manifest.model_id,
                    messages=[
                        GenerateMessage(
                            role="user",
                            content="Describe the image",
                            attachments=[
                                GenerateAttachment(
                                    attachment_type="image",
                                    name="missing.png",
                                    source_path=str(tmp_path / "missing.png"),
                                    media_type="image/png",
                                ),
                            ],
                        ),
                    ],
                    max_tokens=8,
                    temperature=0.0,
                ),
            ),
        )


def test_external_adapter_runtime_supports_audio_transcription_and_speech(tmp_path: Path, monkeypatch) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 160)
    sample_wav = buffer.getvalue()
    manifest = ModelManifest(
        model_id="demo-audio",
        display_name="Demo Audio",
        architecture_family="whisper",
        modality=(ModelModality.AUDIO,),
        source_path=str(tmp_path / "demo-audio"),
        format_type=ModelFormat.AUDIO_FOLDER,
        runtime_affinity=(RuntimeAffinity.MLX_AUDIO,),
        estimated_memory_mb=512,
        context_length=4096,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-audio-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )
    transcription_calls: list[tuple[dict[str, object], dict[str, tuple[str, bytes, str]]]] = []

    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda method, path, payload: {"data": [{"id": "demo-audio"}]},
    )

    def fake_request_multipart_json(
        method: str,
        path: str,
        fields: dict[str, object],
        files: dict[str, tuple[str, bytes, str]],
    ) -> dict[str, object]:
        assert method == "POST"
        assert path == "/v1/audio/transcriptions"
        transcription_calls.append((fields, files))
        return {
            "text": "hello world",
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        }

    monkeypatch.setattr(runtime, "_request_multipart_json", fake_request_multipart_json)
    monkeypatch.setattr(
        runtime,
        "_request_bytes",
        lambda method, path, payload: (sample_wav, "audio/wav"),
    )

    assert runtime.supports_manifest_capability(manifest, CapabilityName.AUDIO_TRANSCRIPTION) is True
    assert runtime.supports_manifest_capability(manifest, CapabilityName.AUDIO_SPEECH) is True

    asyncio.run(runtime.load_model(manifest))
    transcription = asyncio.run(
        runtime.transcribe_audio(
            AudioTranscriptionRequest(
                model_id=manifest.model_id,
                audio_bytes=sample_wav,
                file_name="probe.wav",
                language="en",
            ),
        ),
    )
    speech = asyncio.run(
        runtime.synthesize_speech(
            AudioSpeechRequest(
                model_id=manifest.model_id,
                input_text="hello world",
                voice="alloy",
                audio_format="wav",
            ),
        ),
    )

    assert transcription.text == "hello world"
    assert transcription.language == "en"
    assert transcription_calls[-1][0]["model"] == "demo-audio"
    assert speech.media_type == "audio/wav"
    assert speech.voice == "alloy"


def test_external_adapter_runtime_reports_bridge_only_audio_transcription_probe_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 160)
    sample_wav = buffer.getvalue()
    manifest = ModelManifest(
        model_id="demo-audio",
        display_name="Demo Audio",
        architecture_family="whisper",
        modality=(ModelModality.AUDIO,),
        source_path=str(tmp_path / "demo-audio"),
        format_type=ModelFormat.AUDIO_FOLDER,
        runtime_affinity=(RuntimeAffinity.MLX_AUDIO,),
        estimated_memory_mb=512,
        context_length=4096,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-audio-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )

    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda method, path, payload: {"data": [{"id": "demo-audio"}]},
    )

    def failing_request_multipart_json(
        method: str,
        path: str,
        fields: dict[str, object],
        files: dict[str, tuple[str, bytes, str]],
    ) -> dict[str, object]:
        raise RuntimeUnavailableError(
            "External accelerator request failed with HTTP 404.",
            details={"runtime": runtime.name, "path": path, "body": "not found", "status_code": 404},
        )

    monkeypatch.setattr(runtime, "_request_multipart_json", failing_request_multipart_json)

    assert runtime.supports_manifest_capability(manifest, CapabilityName.AUDIO_TRANSCRIPTION) is False
    reason = runtime.manifest_capability_reason(manifest, CapabilityName.AUDIO_TRANSCRIPTION)
    assert reason is not None
    assert "/v1/audio/transcriptions" in reason
    assert "bridge-only non-Apple audio" in reason

    asyncio.run(runtime.load_model(manifest))
    with pytest.raises(RuntimeUnavailableError) as exc_info:
        asyncio.run(
            runtime.transcribe_audio(
                AudioTranscriptionRequest(
                    model_id=manifest.model_id,
                    audio_bytes=sample_wav,
                    file_name="sample.wav",
                    language="en",
                ),
            ),
        )

    assert exc_info.value.details["support_path"] == "bridge"
    assert exc_info.value.details["expected_endpoint"] == "/v1/audio/transcriptions"
    assert exc_info.value.details["bridge_only"] is True
    assert any("/v1/audio/transcriptions" in item for item in exc_info.value.details["fallback_guidance"])


def test_external_adapter_runtime_reports_bridge_only_audio_speech_probe_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = ModelManifest(
        model_id="demo-audio",
        display_name="Demo Audio",
        architecture_family="kokoro",
        modality=(ModelModality.AUDIO,),
        source_path=str(tmp_path / "demo-audio"),
        format_type=ModelFormat.AUDIO_FOLDER,
        runtime_affinity=(RuntimeAffinity.MLX_AUDIO,),
        estimated_memory_mb=512,
        context_length=4096,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-audio-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )

    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda method, path, payload: {"data": [{"id": "demo-audio"}]},
    )
    monkeypatch.setattr(
        runtime,
        "_request_multipart_json",
        lambda method, path, fields, files: {"text": "hello world", "language": "en"},
    )

    def failing_request_bytes(
        method: str,
        path: str,
        payload: dict[str, object] | None,
    ) -> tuple[bytes, str]:
        raise RuntimeUnavailableError(
            "External accelerator request failed with HTTP 404.",
            details={"runtime": runtime.name, "path": path, "body": "not found", "status_code": 404},
        )

    monkeypatch.setattr(runtime, "_request_bytes", failing_request_bytes)

    assert runtime.supports_manifest_capability(manifest, CapabilityName.AUDIO_SPEECH) is False
    reason = runtime.manifest_capability_reason(manifest, CapabilityName.AUDIO_SPEECH)
    assert reason is not None
    assert "/v1/audio/speech" in reason
    assert "bridge-only non-Apple" in reason

    asyncio.run(runtime.load_model(manifest))
    with pytest.raises(RuntimeUnavailableError) as exc_info:
        asyncio.run(
            runtime.synthesize_speech(
                AudioSpeechRequest(
                    model_id=manifest.model_id,
                    input_text="hello world",
                    voice="alloy",
                    audio_format="wav",
                ),
            ),
        )

    assert exc_info.value.details["support_path"] == "bridge"
    assert exc_info.value.details["expected_endpoint"] == "/v1/audio/speech"
    assert exc_info.value.details["bridge_only"] is True
    assert any("/v1/audio/speech" in item for item in exc_info.value.details["fallback_guidance"])


def test_external_adapter_runtime_reports_connection_refusal_in_candidate_report(tmp_path: Path, monkeypatch) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = _manifest(tmp_path, model_id="demo-model", display_name="Demo Model")
    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda method, path, payload: (_ for _ in ()).throw(
            RuntimeUnavailableError(
                "The configured external accelerator endpoint `http://127.0.0.1:8080/v1/models` refused the connection.",
                details={"runtime": runtime.name, "path": path},
            ),
        ),
    )

    report = runtime.candidate_report(manifest)

    assert report.available is False
    assert report.supports_manifest is False
    assert report.readiness_state == RuntimeReadinessState.RUNTIME_UNAVAILABLE
    assert "refused the connection" in str(report.availability_reason)


def test_external_adapter_runtime_maps_connection_refusal_to_endpoint_specific_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    monkeypatch.setattr(
        openai_compatible_runtime,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            URLError(ConnectionRefusedError(10061, "Connection refused")),
        ),
    )

    with pytest.raises(RuntimeUnavailableError, match=r"refused the connection"):
        runtime._available_remote_models()


def test_external_adapter_runtime_streams_server_sent_events_from_loopback_server(tmp_path: Path) -> None:
    streamed_payloads: list[dict[str, object]] = []
    with _loopback_server(
        {
            ("GET", "/v1/models"): lambda handler, request: _write_json_response(
                handler,
                {"data": [{"id": "demo-stream"}]},
            ),
            ("POST", "/v1/chat/completions"): lambda handler, request: _write_streaming_response(
                handler,
                streamed_payloads,
                request,
            ),
        },
    ) as base_url:
        settings = LewLMSettings(
            data_dir=tmp_path / "state",
            external_accelerator_enabled=True,
            external_accelerator_base_url=base_url,
        )
        runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
        manifest = _manifest(tmp_path, model_id="demo-stream", display_name="Demo Stream")
        asyncio.run(runtime.load_model(manifest))

        async def collect_stream() -> list[str]:
            chunks: list[str] = []
            async for chunk in runtime.stream_generate(
                GenerateRequest(
                    model_id=manifest.model_id,
                    messages=[GenerateMessage(role="user", content="Stream a greeting")],
                    max_tokens=8,
                    temperature=0.0,
                ),
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect_stream())

    assert "".join(chunks) == "hello world"
    assert streamed_payloads[-1]["stream"] is True


def test_external_adapter_runtime_posts_multipart_transcription_to_loopback_server(tmp_path: Path) -> None:
    sample_wav = _sample_wav_bytes()
    recorded_requests: list[dict[str, object]] = []
    with _loopback_server(
        {
            ("GET", "/v1/models"): lambda handler, request: _write_json_response(
                handler,
                {"data": [{"id": "demo-audio"}]},
            ),
            ("POST", "/v1/audio/transcriptions"): lambda handler, request: _write_json_response(
                handler,
                {"text": "hello world", "language": "en", "segments": [{"start": 0.0, "end": 0.5, "text": "hello world"}]},
                recorded_requests,
                request,
            ),
        },
    ) as base_url:
        settings = LewLMSettings(
            data_dir=tmp_path / "state",
            external_accelerator_enabled=True,
            external_accelerator_base_url=base_url,
        )
        runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
        manifest = _manifest(
            tmp_path,
            model_id="demo-audio",
            display_name="Demo Audio",
            architecture_family="whisper",
            modality=(ModelModality.AUDIO,),
            format_type=ModelFormat.AUDIO_FOLDER,
            runtime_affinity=(RuntimeAffinity.MLX_AUDIO,),
            context_length=4096,
        )
        asyncio.run(runtime.load_model(manifest))
        transcription = asyncio.run(
            runtime.transcribe_audio(
                AudioTranscriptionRequest(
                    model_id=manifest.model_id,
                    audio_bytes=sample_wav,
                    file_name="clip.wav",
                    language="en",
                    prompt="LewLM test",
                ),
            ),
        )

    assert transcription.text == "hello world"
    request = recorded_requests[-1]
    headers = request["headers"]
    body = request["body"]
    assert isinstance(headers, dict)
    assert isinstance(body, bytes)
    content_type = str(headers["Content-Type"])
    assert "multipart/form-data; boundary=" in content_type
    boundary = content_type.split("boundary=", 1)[1]
    assert f'name="model"\r\n\r\ndemo-audio'.encode("utf-8") in body
    assert f'name="language"\r\n\r\nen'.encode("utf-8") in body
    assert b'filename="clip.wav"' in body
    assert b"Content-Type: audio/wav" in body
    assert body.startswith(f"--{boundary}".encode("utf-8"))


def test_external_adapter_runtime_reports_semantic_capability_probe_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = ModelManifest(
        model_id="demo-rerank",
        display_name="Demo Rerank",
        architecture_family="bge",
        modality=(ModelModality.RERANK,),
        source_path=str(tmp_path / "demo-rerank"),
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
        estimated_memory_mb=512,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-rerank-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )

    def fake_request_json(method: str, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        if method == "GET" and path == "/v1/models":
            return {"data": [{"id": "demo-rerank"}]}
        if method == "POST" and path == "/v1/rerank":
            raise RuntimeUnavailableError(
                "External accelerator request failed with HTTP 404.",
                details={"runtime": runtime.name, "path": path, "body": "not found"},
            )
        raise AssertionError((method, path, payload))

    monkeypatch.setattr(runtime, "_request_json", fake_request_json)

    assert runtime.supports_capability(CapabilityName.RERANK) is False
    assert runtime.supports_manifest_capability(manifest, CapabilityName.RERANK) is False
    assert "/v1/rerank" in str(runtime.manifest_capability_reason(manifest, CapabilityName.RERANK))


def _manifest(
    tmp_path: Path,
    *,
    model_id: str,
    display_name: str,
    architecture_family: str = "llama",
    modality: tuple[ModelModality, ...] = (ModelModality.TEXT,),
    format_type: ModelFormat = ModelFormat.MLX,
    runtime_affinity: tuple[RuntimeAffinity, ...] = (RuntimeAffinity.MLX_TEXT,),
    estimated_memory_mb: int = 512,
    context_length: int = 8192,
    source_name: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        display_name=display_name,
        architecture_family=architecture_family,
        modality=modality,
        source_path=str(tmp_path / (source_name or model_id)),
        format_type=format_type,
        runtime_affinity=runtime_affinity,
        estimated_memory_mb=estimated_memory_mb,
        context_length=context_length,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint=f"{model_id}-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
        metadata=metadata or {},
    )


def _sample_wav_bytes() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()

@contextmanager
def _loopback_server(routes):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._handle_request()

        def do_POST(self) -> None:  # noqa: N802
            self._handle_request()

        def _handle_request(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(content_length) if content_length else b""
            route = routes[(self.command, self.path)]
            route(
                self,
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": dict(self.headers.items()),
                    "body": body,
                },
            )

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _write_json_response(
    handler: BaseHTTPRequestHandler,
    payload: dict[str, object],
    recorded_requests: list[dict[str, object]] | None = None,
    request: dict[str, object] | None = None,
) -> None:
    if recorded_requests is not None and request is not None:
        recorded_requests.append(request)
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _write_streaming_response(
    handler: BaseHTTPRequestHandler,
    streamed_payloads: list[dict[str, object]],
    request: dict[str, object],
) -> None:
    streamed_payloads.append(json.loads(cast(bytes, request["body"]).decode("utf-8")))
    chunks = (
        ": keepalive\n\n"
        'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":[{"type":"output_text","text":{"value":" world"}}]}}]}\n\n'
        "data: [DONE]\n\n"
    ).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Content-Length", str(len(chunks)))
    handler.end_headers()
    handler.wfile.write(chunks)
    handler.wfile.flush()
