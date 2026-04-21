from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from lewlm.core.contracts import (
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    CapabilityName,
    ConversionStatus,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.runtime.mlx_audio.runtime import MLXAudioRuntime
from lewlm.storage import BlockDiskCache, MetadataStore, MultimodalEncoderCache


def test_mlx_audio_runtime_supports_stt_submodule_layout(monkeypatch, sample_audio_bytes: bytes) -> None:
    captured: dict[str, object] = {}

    class FakeSTTModel:
        def generate(
            self,
            audio: str,
            *,
            verbose: bool = False,
            language: str | None = None,
            prompt: str | None = None,
            text: str | None = None,
        ):
            captured["audio_path"] = audio
            captured["audio_bytes"] = Path(audio).read_bytes()
            captured["verbose"] = verbose
            captured["language"] = language
            captured["prompt"] = prompt
            captured["text"] = text
            return SimpleNamespace(
                text="hello world",
                language=language or "en",
                segments=[{"text": "hello world", "start": 0.0, "end": 1.0}],
            )

    def fake_load(*, model_path: str):
        captured["model_path"] = model_path
        return FakeSTTModel()

    def fake_import(name: str):
        if name == "mlx_audio":
            return SimpleNamespace()
        if name == "mlx_audio.stt":
            return SimpleNamespace(load=fake_load)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_audio.runtime.import_module", fake_import)

    runtime = MLXAudioRuntime()
    manifest = _manifest()

    asyncio.run(runtime.load_model(manifest))

    assert runtime.supports_capability(CapabilityName.AUDIO_TRANSCRIPTION) is True

    response = asyncio.run(
        runtime.transcribe_audio(
            AudioTranscriptionRequest(
                model_id=manifest.model_id,
                audio_bytes=sample_audio_bytes,
                file_name="sample.wav",
                language="en",
                prompt="speaker one",
            ),
        ),
    )

    assert captured["model_path"] == manifest.source_path
    assert captured["audio_bytes"] == sample_audio_bytes
    assert captured["verbose"] is False
    assert captured["language"] == "en"
    assert captured["prompt"] == "speaker one"
    assert captured["text"] == "speaker one"
    assert response.text == "hello world"
    assert response.language == "en"
    assert response.segments[0].text == "hello world"
    assert response.duration_seconds is not None


def test_mlx_audio_runtime_supports_tts_submodule_layout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTTSModel:
        pass

    def fake_load(*, model_path: str):
        captured["model_path"] = model_path
        return FakeTTSModel()

    def fake_generate_audio(
        *,
        model,
        text: str,
        voice: str | None = None,
        output_path: str,
        file_prefix: str,
        audio_format: str,
        join_audio: bool,
        play: bool,
        save: bool,
        verbose: bool,
    ) -> None:
        captured["generate_audio"] = {
            "model": model,
            "text": text,
            "voice": voice,
            "output_path": output_path,
            "file_prefix": file_prefix,
            "audio_format": audio_format,
            "join_audio": join_audio,
            "play": play,
            "save": save,
            "verbose": verbose,
        }
        output_file = Path(output_path) / f"{file_prefix}.{audio_format}"
        output_file.write_bytes(b"fake-audio")

    def fake_import(name: str):
        if name == "mlx_audio":
            return SimpleNamespace()
        if name == "mlx_audio.tts":
            return SimpleNamespace(load=fake_load)
        if name == "mlx_audio.tts.generate":
            return SimpleNamespace(generate_audio=fake_generate_audio)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_audio.runtime.import_module", fake_import)

    runtime = MLXAudioRuntime()
    manifest = _manifest()

    asyncio.run(runtime.load_model(manifest))

    assert runtime.supports_capability(CapabilityName.AUDIO_SPEECH) is True

    response = asyncio.run(
        runtime.synthesize_speech(
            AudioSpeechRequest(
                model_id=manifest.model_id,
                input_text="Ship the milestone",
                voice="alloy",
                audio_format="wav",
            ),
        ),
    )

    assert captured["model_path"] == manifest.source_path
    assert captured["generate_audio"] == {
        "model": captured["generate_audio"]["model"],
        "text": "Ship the milestone",
        "voice": "alloy",
        "output_path": captured["generate_audio"]["output_path"],
        "file_prefix": "speech",
        "audio_format": "wav",
        "join_audio": True,
        "play": False,
        "save": False,
        "verbose": False,
    }
    assert isinstance(captured["generate_audio"]["model"], FakeTTSModel)
    assert response.audio_bytes == b"fake-audio"
    assert response.media_type == "audio/wav"
    assert response.voice == "alloy"


def test_mlx_audio_runtime_reuses_encoder_features_across_identical_audio_with_different_file_names(
    monkeypatch,
    tmp_path: Path,
    sample_audio_bytes: bytes,
) -> None:
    captured: dict[str, object] = {"encode_calls": 0}

    class FakeSTTModel:
        def encode_audio(
            self,
            audio: str,
            *,
            language: str | None = None,
            prompt: str | None = None,
            text: str | None = None,
        ) -> dict[str, object]:
            captured["encode_calls"] = int(captured["encode_calls"]) + 1
            return {"bytes": len(Path(audio).read_bytes()), "language": language, "prompt": prompt, "text": text}

        def generate(
            self,
            audio: str,
            *,
            cached_audio_features: dict[str, object] | None = None,
            verbose: bool = False,
            language: str | None = None,
            prompt: str | None = None,
            text: str | None = None,
        ):
            return SimpleNamespace(
                text=f"decoded:{cached_audio_features is not None}",
                language=language or "en",
                segments=[{"text": "decoded", "start": 0.0, "end": 1.0}],
            )

    def fake_load(*, model_path: str):
        return FakeSTTModel()

    def fake_import(name: str):
        if name == "mlx_audio":
            return SimpleNamespace()
        if name == "mlx_audio.stt":
            return SimpleNamespace(load=fake_load)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_audio.runtime.import_module", fake_import)

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    metadata_store = MetadataStore(state_dir / "metadata.sqlite3")
    metadata_store.initialize()
    encoder_cache = MultimodalEncoderCache(
        block_disk_cache=BlockDiskCache(cache_root=state_dir, metadata_store=metadata_store),
    )
    runtime = MLXAudioRuntime(multimodal_encoder_cache=encoder_cache)
    manifest = _manifest()

    first_request = AudioTranscriptionRequest(
        model_id=manifest.model_id,
        audio_bytes=sample_audio_bytes,
        file_name="first.wav",
        language="en",
        prompt="speaker one",
    )
    second_request = AudioTranscriptionRequest(
        model_id=manifest.model_id,
        audio_bytes=sample_audio_bytes,
        file_name="second.wav",
        language="en",
        prompt="speaker one",
    )

    asyncio.run(runtime.load_model(manifest))
    asyncio.run(runtime.transcribe_audio(first_request))
    asyncio.run(runtime.transcribe_audio(second_request))

    assert captured["encode_calls"] == 1
    assert first_request.metadata["encoder_cache"]["cache_misses"] == 1
    assert second_request.metadata["encoder_cache"]["cache_hits"] == 1


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_id="audio-model",
        display_name="whisper-mini-audio",
        architecture_family="whisper",
        modality=(ModelModality.AUDIO,),
        source_path="/tmp/audio-model",
        format_type=ModelFormat.AUDIO_FOLDER,
        runtime_affinity=(RuntimeAffinity.MLX_AUDIO,),
        estimated_memory_mb=512,
        context_length=None,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="audio-fingerprint",
        last_validation_result=ModelValidationResult(
            status=ValidationState.VALID,
            message="ok",
        ),
    )
