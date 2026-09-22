from __future__ import annotations

import asyncio
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest

from lewlm.core.contracts import (
    AudioCapabilityRole,
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioVoiceSource,
    CapabilityName,
    ConversionStatus,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.core.errors import UnsupportedCapabilityError
from lewlm.runtime.mlx_audio.runtime import MLXAudioRuntime
from lewlm.storage import BlockDiskCache, MetadataStore, MultimodalEncoderCache

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="MLX runtimes are macOS-only.")


@pytest.fixture(autouse=True)
def fake_mlx_package_discovery(monkeypatch):
    """These runtime tests use fake modules; discovery must use the same fixture."""
    monkeypatch.setattr(
        "lewlm.runtime.mlx_audio.runtime.find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[]),
    )


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
    manifest = _speech_manifest()

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


def test_mlx_audio_runtime_reports_capability_per_manifest_not_per_runtime() -> None:
    runtime = MLXAudioRuntime()
    transcription_manifest = _manifest()
    speech_manifest = _speech_manifest()

    assert runtime.supports_manifest_capability(transcription_manifest, CapabilityName.AUDIO_TRANSCRIPTION) is True
    assert runtime.supports_manifest_capability(transcription_manifest, CapabilityName.AUDIO_SPEECH) is False
    assert runtime.supports_manifest_capability(speech_manifest, CapabilityName.AUDIO_SPEECH) is True
    assert runtime.supports_manifest_capability(speech_manifest, CapabilityName.AUDIO_TRANSCRIPTION) is False
    assert "audio_speech" in str(runtime.manifest_capability_reason(transcription_manifest, CapabilityName.AUDIO_SPEECH))


def test_mlx_audio_runtime_refuses_a_capability_the_model_does_not_serve(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.runtime.mlx_audio.runtime.import_module", lambda name: SimpleNamespace())

    runtime = MLXAudioRuntime()
    manifest = _manifest()
    asyncio.run(runtime.load_model(manifest))

    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        asyncio.run(
            runtime.synthesize_speech(
                AudioSpeechRequest(model_id=manifest.model_id, input_text="Hello", audio_format="wav"),
            ),
        )

    # The refusal names the request as inappropriate rather than failing inside the backend.
    assert excinfo.value.details["capability"] == CapabilityName.AUDIO_SPEECH.value
    assert excinfo.value.details["audio_roles"] == ["transcription"]


def test_mlx_audio_runtime_lists_voice_packs_from_the_bundle_and_backend_cache(tmp_path: Path, monkeypatch) -> None:
    bundle_dir = tmp_path / "Kokoro-82M"
    (bundle_dir / "voices").mkdir(parents=True)
    (bundle_dir / "voices" / "af_bundled.safetensors").write_bytes(b"voice")
    (bundle_dir / "bf_loose.pt").write_bytes(b"voice")
    (bundle_dir / "kokoro-v1_0.safetensors").write_bytes(b"weights")

    cache_root = tmp_path / "hub"
    snapshot_voices = cache_root / "models--prince-canuma--Kokoro-82M" / "snapshots" / "abc123" / "voices"
    snapshot_voices.mkdir(parents=True)
    (snapshot_voices / "am_cached.safetensors").write_bytes(b"voice")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))

    runtime = MLXAudioRuntime()
    voices = runtime.speech_voices(_speech_manifest(source_path=str(bundle_dir)))

    assert [voice.voice_id for voice in voices] == ["af_bundled", "am_cached", "bf_loose"]
    assert {voice.voice_id: voice.source for voice in voices} == {
        "af_bundled": AudioVoiceSource.BUNDLE,
        "bf_loose": AudioVoiceSource.BUNDLE,
        "am_cached": AudioVoiceSource.BACKEND_CACHE,
    }
    # The model weights are not a voice pack.
    assert "kokoro-v1_0" not in {voice.voice_id for voice in voices}


def test_mlx_audio_runtime_lists_only_wav_without_the_mlx_audio_encoder(monkeypatch) -> None:
    """Without `mlx_audio.tts.generate` LewLM encodes WAV itself and refuses the rest."""

    monkeypatch.setattr("lewlm.runtime.mlx_audio.runtime._import_optional_module", lambda name: None)

    support = MLXAudioRuntime().speech_formats(_speech_manifest())

    assert [(item.format, item.media_type, item.verified) for item in support.formats] == [("wav", "audio/wav", True)]
    assert support.exhaustive is True
    assert support.accepts("WAV") and not support.accepts("mp3")


def test_mlx_audio_runtime_lists_the_helper_encodings_as_unverified(monkeypatch) -> None:
    """With the helper the other encodings are reachable, but LewLM has not probed each encoder."""

    monkeypatch.setattr(
        "lewlm.runtime.mlx_audio.runtime._import_optional_module",
        lambda name: SimpleNamespace(generate_audio=lambda **_: None) if name == "mlx_audio.tts.generate" else None,
    )

    support = MLXAudioRuntime().speech_formats(_speech_manifest())

    assert [(item.format, item.verified) for item in support.formats] == [
        ("wav", True),
        ("mp3", False),
        ("flac", False),
        ("ogg", False),
    ]
    assert {item.format: item.media_type for item in support.formats} == {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }
    # No other name reaches the backend, so the list is exhaustive either way.
    assert support.exhaustive is True


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_id="audio-model",
        display_name="whisper-mini-audio",
        architecture_family="whisper",
        modality=(ModelModality.AUDIO,),
        audio_roles=(AudioCapabilityRole.TRANSCRIPTION,),
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


def _speech_manifest(*, source_path: str = "/tmp/speech-model") -> ModelManifest:
    return _manifest().model_copy(
        update={
            "model_id": "speech-model",
            "display_name": "Kokoro-82M",
            "architecture_family": "kokoro",
            "audio_roles": (AudioCapabilityRole.SPEECH,),
            "source_path": source_path,
            "format_type": ModelFormat.MLX,
        },
    )
