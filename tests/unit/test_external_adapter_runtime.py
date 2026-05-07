from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import wave

import asyncio
import pytest

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
    ValidationState,
)
from lewlm.core.errors import RuntimeUnavailableError
from lewlm.runtime.adapters import LocalOpenAICompatibleAdapterRuntime, summarize_feature_preservation


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
