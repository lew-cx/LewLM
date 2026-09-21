"""First-pass MLX audio runtime adapter."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib import import_module
from importlib.util import find_spec
import inspect
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import wave

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    SPEECH_FORMAT_MEDIA_TYPES,
    AudioSpeechFormat,
    AudioSpeechFormatSupport,
    AudioSpeechRequest,
    AudioSpeechResponse,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioTranscriptionSegment,
    AudioVoice,
    AudioVoiceSource,
    CapabilityName,
    ModelFormat,
    ModelManifest,
    RuntimeAffinity,
    speech_media_type,
    wav_speech_format_support,
)
from lewlm.core.errors import NotImplementedLewLMError
from lewlm.runtime.base import ManagedAudioRuntime, collect_voice_files
from lewlm.runtime.introspection import invoke_with_signature, resolve_backend_callable
from lewlm.storage.block_cache import MultimodalEncoderCache


_AUDIO_FEATURE_PARAMETER_NAMES = ("cached_audio_features", "audio_features", "input_features", "encoder_outputs")
_AUDIO_ENCODER_METHOD_NAMES = ("encode_audio", "extract_features", "encode", "embed_audio")


@dataclass(slots=True)
class _AudioClientState:
    source_path: str
    stt_model: Any | None = None
    tts_model: Any | None = None


@dataclass(slots=True)
class _AudioEncoderCacheContext:
    provided_values: dict[str, Any]
    cache_hits: int
    cache_misses: int
    input_bytes: int

    def metrics(self) -> dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "input_bytes": self.input_bytes,
        }


class MLXAudioRuntime(ManagedAudioRuntime):
    """Adapter for MLX-native speech transcription and synthesis."""

    name = "mlx_audio"
    affinity = RuntimeAffinity.MLX_AUDIO
    supported_formats = (ModelFormat.MLX, ModelFormat.AUDIO_FOLDER)
    supported_systems = ("Darwin",)
    supported_machines = ("arm64", "aarch64")
    platform_guidance = "Install the `mlx` extra on Apple Silicon macOS to enable MLX-native audio inference."

    def __init__(
        self,
        *,
        settings: LewLMSettings | None = None,
        multimodal_encoder_cache: MultimodalEncoderCache | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings or LewLMSettings()
        self._clients: dict[str, _AudioClientState] = {}
        self._multimodal_encoder_cache = multimodal_encoder_cache
        self._encoder_cache_request_count = 0
        self._encoder_cache_hits = 0
        self._encoder_cache_misses = 0

    def performance_feature_snapshot(self) -> dict[str, object]:
        if not self.is_available():
            return {}
        supported = self._audio_encoder_cache_supported()
        return {
            "multimodal_encoder_caching": {
                "supported": supported,
                "active": supported and self._encoder_cache_request_count > 0,
                "reason": (
                    "LewLM reuses cached audio encoder features when the installed mlx-audio backend exposes compatible feature hooks."
                    if supported
                    else "Installed mlx-audio entrypoints do not expose a compatible cached-audio feature hook."
                ),
                "notes": (
                    [
                        "Audio encoder reuse activates only for backends that accept cached feature tensors and expose a compatible encoder method.",
                    ]
                    if supported
                    else []
                ),
                "metrics": _compact_metrics(
                    request_count=self._encoder_cache_request_count,
                    cache_hits=self._encoder_cache_hits,
                    cache_misses=self._encoder_cache_misses,
                ),
            },
        }

    def _check_environment(self) -> tuple[bool, str | None]:
        if self.loaded_model_ids:
            return True, None
        if self.settings.backend_feature_probes_enabled:
            try:
                import_module("mlx_audio")
            except ImportError:
                return False, "mlx-audio is not installed"
            return True, None
        try:
            spec = find_spec("mlx_audio")
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            return False, "mlx-audio is not installed"
        return True, None

    def supports_capability(self, capability: CapabilityName) -> bool:
        if not super().supports_capability(capability):
            return False
        if not self.settings.backend_feature_probes_enabled and not self.loaded_model_ids:
            return self.is_available()
        module = import_module("mlx_audio")
        if capability == CapabilityName.AUDIO_TRANSCRIPTION:
            return _has_audio_transcription_support(module)
        if capability == CapabilityName.AUDIO_SPEECH:
            return _has_audio_speech_support(module)
        return False

    async def _load_model(self, manifest: ModelManifest) -> None:
        self._clients[manifest.model_id] = _AudioClientState(source_path=manifest.source_path)

    async def _unload_model(self, model_id: str) -> None:
        self._clients.pop(model_id, None)
        if self._multimodal_encoder_cache is not None:
            self._multimodal_encoder_cache.drop_runtime_resident_features(runtime=self.name, model_id=model_id)

    async def _transcribe_audio(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse:
        module = import_module("mlx_audio")
        transcribe = resolve_backend_callable(module, ("transcribe", "transcribe_audio"), required=False)
        with _temporary_audio_input(request.audio_bytes, request.file_name) as audio_path:
            if transcribe is not None:
                client = self._stt_model_for(request.model_id)
                cache_context = self._build_audio_encoder_cache_context(
                    request=request,
                    audio_path=audio_path,
                    callable_obj=transcribe,
                    client=client,
                )
                try:
                    result = invoke_with_signature(
                        transcribe,
                        {
                            "client": client,
                            "model": client,
                            "pipeline": client,
                            "audio": audio_path,
                            "audio_path": audio_path,
                            "audio_input": audio_path,
                            "audio_or_path": audio_path,
                            "file_path": audio_path,
                            "path": audio_path,
                            "audio_bytes": request.audio_bytes,
                            "audio_data": request.audio_bytes,
                            "bytes": request.audio_bytes,
                            "file_name": request.file_name,
                            "language": request.language,
                            "lang_code": request.language,
                            "prompt": request.prompt,
                            "text": request.prompt,
                            **(cache_context.provided_values if cache_context is not None else {}),
                        },
                        capability=CapabilityName.AUDIO_TRANSCRIPTION.value,
                    )
                finally:
                    self._record_audio_encoder_cache_request(request=request, cache_context=cache_context)
            else:
                client = self._stt_model_for(request.model_id)
                generate = getattr(client, "generate", None)
                if not callable(generate):
                    raise NotImplementedLewLMError(
                        "LewLM could not find a transcription entrypoint for the installed mlx-audio backend.",
                        details={"capability": CapabilityName.AUDIO_TRANSCRIPTION.value},
                    )
                cache_context = self._build_audio_encoder_cache_context(
                    request=request,
                    audio_path=audio_path,
                    callable_obj=generate,
                    client=client,
                )
                try:
                    result = invoke_with_signature(
                        generate,
                        {
                            "audio": audio_path,
                            "audio_path": audio_path,
                            "audio_input": audio_path,
                            "audio_or_path": audio_path,
                            "file_path": audio_path,
                            "path": audio_path,
                            "verbose": False,
                            "language": request.language,
                            "lang_code": request.language,
                            "prompt": request.prompt,
                            "text": request.prompt,
                            **(cache_context.provided_values if cache_context is not None else {}),
                        },
                        capability=CapabilityName.AUDIO_TRANSCRIPTION.value,
                    )
                finally:
                    self._record_audio_encoder_cache_request(request=request, cache_context=cache_context)
        return _audio_transcription_response_from_result(result, request)

    async def _synthesize_speech(self, request: AudioSpeechRequest) -> AudioSpeechResponse:
        module = import_module("mlx_audio")
        synthesize = resolve_backend_callable(module, ("synthesize", "synthesize_speech", "generate_audio"), required=False)
        media_type = _media_type_for_format(request.audio_format)
        if synthesize is not None:
            client = self._tts_model_for(request.model_id)
            result = invoke_with_signature(
                synthesize,
                {
                    "client": client,
                    "model": client,
                    "pipeline": client,
                    "text": request.input_text,
                    "input": request.input_text,
                    "input_text": request.input_text,
                    "voice": request.voice,
                    "format": request.audio_format,
                    "audio_format": request.audio_format,
                },
                capability=CapabilityName.AUDIO_SPEECH.value,
            )
            return _audio_speech_response_from_result(
                result,
                request=request,
                media_type=media_type,
            )

        tts_generate_module = _import_optional_module("mlx_audio.tts.generate")
        generate_audio = (
            resolve_backend_callable(tts_generate_module, ("generate_audio",), required=False)
            if tts_generate_module is not None
            else None
        )
        if generate_audio is not None:
            client = self._tts_model_for(request.model_id)
            with TemporaryDirectory(prefix="lewlm-mlx-tts-") as output_dir:
                invoke_with_signature(
                    generate_audio,
                    {
                        "model": client,
                        "text": request.input_text,
                        "voice": request.voice,
                        "output_path": output_dir,
                        "file_prefix": "speech",
                        "audio_format": request.audio_format,
                        "format": request.audio_format,
                        "join_audio": True,
                        "play": False,
                        "save": False,
                        "verbose": False,
                    },
                    capability=CapabilityName.AUDIO_SPEECH.value,
                )
                output_path = _generated_audio_path(output_dir, request.audio_format)
                audio_bytes = output_path.read_bytes()
            return AudioSpeechResponse(
                model_id=request.model_id,
                audio_bytes=audio_bytes,
                media_type=media_type,
                voice=request.voice,
                duration_seconds=_duration_seconds_from_wav_bytes(audio_bytes),
            )

        client = self._tts_model_for(request.model_id)
        generate = getattr(client, "generate", None)
        if not callable(generate):
            raise NotImplementedLewLMError(
                "LewLM could not find a speech-synthesis entrypoint for the installed mlx-audio backend.",
                details={"capability": CapabilityName.AUDIO_SPEECH.value},
            )
        chunks = invoke_with_signature(
            generate,
            {
                "text": request.input_text,
                "voice": request.voice,
                "verbose": False,
            },
            capability=CapabilityName.AUDIO_SPEECH.value,
        )
        _, audio_bytes, duration_seconds = _speech_bytes_from_generated_chunks(
            chunks,
            audio_format=request.audio_format,
        )
        return AudioSpeechResponse(
            model_id=request.model_id,
            audio_bytes=audio_bytes,
            media_type=media_type,
            voice=request.voice,
            duration_seconds=duration_seconds,
        )

    def speech_voices(self, manifest: ModelManifest) -> list[AudioVoice]:
        """List voice packs this host can resolve for `manifest` without downloading.

        mlx-audio resolves a voice name to `voices/{name}.safetensors` inside
        the backend's own Hugging Face cache for the bundle's repo, so the same
        model directory offers different voices on different machines. Both
        places are read here, and each voice reports where it was found.
        """

        voices: dict[str, AudioVoice] = {voice.voice_id: voice for voice in super().speech_voices(manifest)}
        # An MLX bundle stores its weights as `.safetensors`/`.npz`, so a loose
        # `.pt` beside them is a voice pack rather than part of the model.
        collect_voice_files(voices, Path(manifest.source_path), source=AudioVoiceSource.BUNDLE, suffixes=(".pt",))
        for snapshot_path in self._backend_voice_directories(manifest):
            collect_voice_files(voices, snapshot_path, source=AudioVoiceSource.BACKEND_CACHE)
        return sorted(voices.values(), key=lambda voice: voice.voice_id)

    def speech_formats(self, manifest: ModelManifest) -> AudioSpeechFormatSupport:
        """WAV always; the other encodings only through mlx-audio's own helper.

        Without `mlx_audio.tts.generate` LewLM encodes the model's samples to
        WAV itself and refuses anything else, so listing more would be a lie.
        With it, the helper writes the file and LewLM has not probed each
        encoder, so those formats are reachable but unverified. The list is
        exhaustive either way: no other name reaches the backend.
        """

        support = wav_speech_format_support()
        if _import_optional_module("mlx_audio.tts.generate") is not None:
            support.formats.extend(
                AudioSpeechFormat(format=name, media_type=media_type, verified=False)
                for name, media_type in SPEECH_FORMAT_MEDIA_TYPES.items()
                if name != "wav"
            )
        return support

    def _backend_voice_directories(self, manifest: ModelManifest) -> list[Path]:
        cache_root = _huggingface_cache_root()
        if cache_root is None or not cache_root.is_dir():
            return []
        repo_ids, repo_names = self._voice_repo_matchers(manifest)
        directories: list[Path] = []
        for repo_directory in sorted(cache_root.glob("models--*")):
            repo_id = repo_directory.name.removeprefix("models--").replace("--", "/").casefold()
            if repo_id not in repo_ids and repo_id.rpartition("/")[2] not in repo_names:
                continue
            directories.extend(
                path for path in sorted(repo_directory.glob("snapshots/*/voices")) if path.is_dir()
            )
        return directories

    def _voice_repo_matchers(self, manifest: ModelManifest) -> tuple[set[str], set[str]]:
        """Hugging Face repos whose cached voice packs belong to this bundle."""

        repo_ids: set[str] = set()
        state = self._clients.get(manifest.model_id)
        client = state.tts_model if state is not None else None
        for candidate in (
            getattr(client, "repo_id", None),
            getattr(type(client), "REPO_ID", None) if client is not None else None,
            _load_json_file(Path(manifest.source_path) / "config.json").get("repo_id"),
        ):
            if isinstance(candidate, str) and candidate:
                repo_ids.add(candidate.casefold())
        # A bundle copied out of a repo keeps the repo name as its directory
        # name, which is enough to find that repo's voices in the cache.
        repo_names = {Path(manifest.source_path).name.casefold()}
        return repo_ids, repo_names

    def _stt_model_for(self, model_id: str) -> Any:
        state = self._clients[model_id]
        if state.stt_model is None:
            state.stt_model = _load_audio_model(
                state.source_path,
                module_names=("mlx_audio.stt", "mlx_audio"),
            )
        return state.stt_model

    def _tts_model_for(self, model_id: str) -> Any:
        state = self._clients[model_id]
        if state.tts_model is None:
            state.tts_model = _load_audio_model(
                state.source_path,
                module_names=("mlx_audio.tts", "mlx_audio"),
            )
        return state.tts_model

    def _audio_encoder_cache_supported(self) -> bool:
        if self._multimodal_encoder_cache is None:
            return False
        module = import_module("mlx_audio")
        inference_callables = [
            resolve_backend_callable(module, ("transcribe", "transcribe_audio"), required=False),
        ]
        state = next(iter(self._clients.values()), None)
        encoder_callables: list[Any] = []
        if state is not None and state.stt_model is not None:
            inference_callables.append(getattr(state.stt_model, "generate", None))
            encoder_callables.extend(getattr(state.stt_model, name, None) for name in _AUDIO_ENCODER_METHOD_NAMES)
        feature_ready = any(_audio_feature_parameter_name(callable_obj) is not None for callable_obj in inference_callables)
        encoder_ready = any(callable(method) for method in encoder_callables)
        return feature_ready and encoder_ready

    def _build_audio_encoder_cache_context(
        self,
        *,
        request: AudioTranscriptionRequest,
        audio_path: str,
        callable_obj: Any,
        client: Any,
    ) -> "_AudioEncoderCacheContext | None":
        if self._multimodal_encoder_cache is None:
            return None
        feature_parameter = _audio_feature_parameter_name(callable_obj)
        if feature_parameter is None:
            return None
        encoder_callable = _resolve_audio_encoder_callable(client)
        if encoder_callable is None:
            return None
        manifest = self._loaded_manifests[request.model_id]
        content_sha256 = hashlib.sha256(request.audio_bytes).hexdigest()
        preprocessing_fingerprint = _stable_digest(
            {
                "language": request.language,
                "prompt_present": bool(request.prompt),
                "encoder_callable": getattr(encoder_callable, "__name__", type(encoder_callable).__name__),
                "feature_parameter": feature_parameter,
                "model_type": type(client).__name__,
            }
        )
        cache_key = self._multimodal_encoder_cache.cache_key_for_feature(
            runtime=self.name,
            model_id=request.model_id,
            model_fingerprint=manifest.fingerprint,
            modality="audio",
            content_sha256=content_sha256,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )
        cached_feature = self._multimodal_encoder_cache.get_feature(cache_key=cache_key)
        if cached_feature is not None:
            return _AudioEncoderCacheContext(
                provided_values={feature_parameter: cached_feature},
                cache_hits=1,
                cache_misses=0,
                input_bytes=len(request.audio_bytes),
            )
        encoder_values = {
            "client": client,
            "model": client,
            "pipeline": client,
            "audio": audio_path,
            "audio_path": audio_path,
            "audio_input": audio_path,
            "audio_or_path": audio_path,
            "file_path": audio_path,
            "path": audio_path,
            "audio_bytes": request.audio_bytes,
            "audio_data": request.audio_bytes,
            "bytes": request.audio_bytes,
            "file_name": request.file_name,
            "language": request.language,
            "lang_code": request.language,
            "prompt": request.prompt,
            "text": request.prompt,
        }
        if not _callable_supported_with_values(encoder_callable, encoder_values):
            return None
        features = invoke_with_signature(
            encoder_callable,
            encoder_values,
            capability=f"{CapabilityName.AUDIO_TRANSCRIPTION.value}_encoder",
        )
        source_locator = request.metadata.get("source_locator")
        if not isinstance(source_locator, str) or not source_locator:
            source_locator = f"audio:{request.file_name}"
        self._multimodal_encoder_cache.put_feature(
            cache_key=cache_key,
            runtime=self.name,
            model_id=request.model_id,
            model_fingerprint=manifest.fingerprint,
            modality="audio",
            content_sha256=content_sha256,
            preprocessing_fingerprint=preprocessing_fingerprint,
            feature=features,
            source_locator=source_locator,
            metadata={"input_bytes": len(request.audio_bytes)},
        )
        return _AudioEncoderCacheContext(
            provided_values={feature_parameter: features},
            cache_hits=0,
            cache_misses=1,
            input_bytes=len(request.audio_bytes),
        )

    def _record_audio_encoder_cache_request(
        self,
        *,
        request: AudioTranscriptionRequest,
        cache_context: "_AudioEncoderCacheContext | None",
    ) -> None:
        if cache_context is None:
            return
        request.metadata["encoder_cache"] = cache_context.metrics()
        self._encoder_cache_request_count += 1
        self._encoder_cache_hits += cache_context.cache_hits
        self._encoder_cache_misses += cache_context.cache_misses


def _huggingface_cache_root() -> Path | None:
    for variable, suffix in (("HF_HUB_CACHE", ""), ("HUGGINGFACE_HUB_CACHE", ""), ("HF_HOME", "hub")):
        value = os.environ.get(variable)
        if value:
            return Path(value).expanduser() / suffix if suffix else Path(value).expanduser()
    return Path.home() / ".cache" / "huggingface" / "hub"


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _has_audio_transcription_support(module: Any) -> bool:
    if resolve_backend_callable(module, ("transcribe", "transcribe_audio"), required=False) is not None:
        return True
    stt_module = _import_optional_module("mlx_audio.stt")
    if stt_module is None:
        return False
    return resolve_backend_callable(stt_module, ("load", "load_model", "load_pipeline"), required=False) is not None


def _has_audio_speech_support(module: Any) -> bool:
    if resolve_backend_callable(module, ("synthesize", "synthesize_speech", "generate_audio"), required=False) is not None:
        return True
    tts_module = _import_optional_module("mlx_audio.tts")
    if tts_module is not None and resolve_backend_callable(tts_module, ("load", "load_model", "load_pipeline"), required=False) is not None:
        return True
    tts_generate_module = _import_optional_module("mlx_audio.tts.generate")
    if tts_generate_module is None:
        return False
    return resolve_backend_callable(tts_generate_module, ("generate_audio",), required=False) is not None


def _audio_feature_parameter_name(callable_obj: Any) -> str | None:
    if not callable(callable_obj):
        return None
    try:
        parameter_names = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return None
    return next((name for name in _AUDIO_FEATURE_PARAMETER_NAMES if name in parameter_names), None)


def _resolve_audio_encoder_callable(client: Any) -> Any | None:
    for name in _AUDIO_ENCODER_METHOD_NAMES:
        candidate = getattr(client, name, None)
        if callable(candidate):
            return candidate
    return None


def _callable_supported_with_values(callable_obj: Any, provided_values: dict[str, Any]) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if parameter.name in provided_values:
            continue
        if parameter.default is inspect.Signature.empty:
            return False
    return True


def _stable_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compact_metrics(**values: int) -> dict[str, int]:
    return {key: value for key, value in values.items() if value}


def _load_audio_model(source_path: str, *, module_names: tuple[str, ...]) -> Any:
    for module_name in module_names:
        module = _import_optional_module(module_name)
        if module is None:
            continue
        load = resolve_backend_callable(module, ("load", "load_model", "load_pipeline"), required=False)
        if load is None:
            continue
        return invoke_with_signature(
            load,
            {
                "path_or_hf_repo": source_path,
                "path": source_path,
                "model_path": source_path,
                "source_path": source_path,
            },
            capability="model_load",
        )
    raise NotImplementedLewLMError(
        "LewLM could not find a supported model-loading entrypoint in the installed mlx-audio backend.",
        details={"module_names": list(module_names)},
    )


def _audio_transcription_response_from_result(
    result: Any,
    request: AudioTranscriptionRequest,
) -> AudioTranscriptionResponse:
    if isinstance(result, AudioTranscriptionResponse):
        return result
    if isinstance(result, dict):
        return AudioTranscriptionResponse(
            model_id=request.model_id,
            text=str(result.get("text", "")),
            language=_string_or_none(result.get("language")),
            duration_seconds=_first_duration_value(
                result.get("duration_seconds"),
                _duration_seconds_from_wav_bytes(request.audio_bytes),
            ),
            segments=_normalize_transcription_segments(result.get("segments")),
        )
    return AudioTranscriptionResponse(
        model_id=request.model_id,
        text=_result_text(result),
        language=_string_or_none(getattr(result, "language", None)),
        duration_seconds=_first_duration_value(
            getattr(result, "duration_seconds", None),
            getattr(result, "audio_duration", None),
            _duration_seconds_from_wav_bytes(request.audio_bytes),
        ),
        segments=_normalize_transcription_segments(getattr(result, "segments", None)),
    )


def _audio_speech_response_from_result(
    result: Any,
    *,
    request: AudioSpeechRequest,
    media_type: str,
) -> AudioSpeechResponse:
    if isinstance(result, AudioSpeechResponse):
        return result
    if isinstance(result, dict):
        raw_audio = result.get("audio_bytes", result.get("audio", b""))
        audio_bytes = bytes(raw_audio)
        return AudioSpeechResponse(
            model_id=request.model_id,
            audio_bytes=audio_bytes,
            media_type=str(result.get("media_type", media_type)),
            voice=_string_or_none(result.get("voice")) or request.voice,
            duration_seconds=_first_duration_value(
                result.get("duration_seconds"),
                _duration_seconds_from_wav_bytes(audio_bytes),
            ),
        )
    if isinstance(result, (bytes, bytearray)):
        audio_bytes = bytes(result)
        return AudioSpeechResponse(
            model_id=request.model_id,
            audio_bytes=audio_bytes,
            media_type=media_type,
            voice=request.voice,
            duration_seconds=_duration_seconds_from_wav_bytes(audio_bytes),
        )
    audio_bytes = _bytes_like_to_bytes(getattr(result, "audio_bytes", None) or getattr(result, "audio", None))
    if audio_bytes is not None:
        return AudioSpeechResponse(
            model_id=request.model_id,
            audio_bytes=audio_bytes,
            media_type=media_type,
            voice=_string_or_none(getattr(result, "voice", None)) or request.voice,
            duration_seconds=_first_duration_value(
                getattr(result, "duration_seconds", None),
                getattr(result, "audio_duration", None),
                _duration_seconds_from_wav_bytes(audio_bytes),
            ),
        )
    raise NotImplementedLewLMError(
        "LewLM could not normalize the installed mlx-audio speech response.",
        details={"response_type": type(result).__name__},
    )


def _normalize_transcription_segments(payload: Any) -> list[AudioTranscriptionSegment]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        return []
    segments: list[AudioTranscriptionSegment] = []
    for item in payload:
        if isinstance(item, AudioTranscriptionSegment):
            segments.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if not isinstance(text, str):
                continue
            segments.append(
                AudioTranscriptionSegment(
                    start_seconds=_duration_value(item.get("start_seconds", item.get("start"))),
                    end_seconds=_duration_value(item.get("end_seconds", item.get("end"))),
                    text=text,
                ),
            )
            continue
        text = getattr(item, "text", None)
        if not isinstance(text, str):
            continue
        segments.append(
            AudioTranscriptionSegment(
                start_seconds=_first_duration_value(
                    getattr(item, "start_seconds", None),
                    getattr(item, "start_time", None),
                ),
                end_seconds=_first_duration_value(
                    getattr(item, "end_seconds", None),
                    getattr(item, "end_time", None),
                ),
                text=text,
            ),
        )
    return segments


def _speech_bytes_from_generated_chunks(
    chunks: Any,
    *,
    audio_format: str,
) -> tuple[int, bytes, float | None]:
    normalized_format = audio_format.casefold()
    if normalized_format != "wav":
        raise NotImplementedLewLMError(
            "LewLM can only encode direct mlx-audio model output to WAV without the package TTS helper.",
            details={"requested_format": audio_format},
        )
    if not isinstance(chunks, Sequence):
        chunks = list(chunks)
    sample_rate: int | None = None
    sample_sequences: list[list[float]] = []
    for chunk in chunks:
        chunk_audio = _coerce_audio_sequence(getattr(chunk, "audio", None))
        if chunk_audio is None:
            continue
        sample_sequences.append(chunk_audio)
        chunk_sample_rate = getattr(chunk, "sample_rate", None)
        if isinstance(chunk_sample_rate, int) and chunk_sample_rate > 0:
            sample_rate = chunk_sample_rate
    if not sample_sequences:
        return 16_000, b"", None
    resolved_sample_rate = sample_rate or 16_000
    flattened_samples = [sample for chunk in sample_sequences for sample in chunk]
    audio_bytes = _wav_bytes_from_samples(flattened_samples, resolved_sample_rate)
    duration_seconds = len(flattened_samples) / resolved_sample_rate if resolved_sample_rate > 0 else None
    return resolved_sample_rate, audio_bytes, duration_seconds


def _wav_bytes_from_samples(samples: list[float], sample_rate: int) -> bytes:
    pcm = bytearray()
    for sample in samples:
        normalized = max(-1.0, min(1.0, float(sample)))
        pcm.extend(int(normalized * 32767).to_bytes(2, "little", signed=True))
    with TemporaryDirectory(prefix="lewlm-mlx-audio-wav-") as temp_dir:
        wav_path = Path(temp_dir) / "speech.wav"
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(bytes(pcm))
        return wav_path.read_bytes()


def _coerce_audio_sequence(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [float(item) for item in value if isinstance(item, (int, float))]
    return None


@contextmanager
def _temporary_audio_input(audio_bytes: bytes, file_name: str) -> Iterator[str]:
    suffix = Path(file_name).suffix or ".wav"
    with TemporaryDirectory(prefix="lewlm-mlx-audio-in-") as temp_dir:
        audio_path = Path(temp_dir) / f"input{suffix}"
        audio_path.write_bytes(audio_bytes)
        yield str(audio_path)


def _generated_audio_path(output_dir: str, audio_format: str) -> Path:
    primary = Path(output_dir) / f"speech.{audio_format}"
    if primary.exists():
        return primary
    matches = sorted(Path(output_dir).glob(f"*.{audio_format}"))
    if matches:
        return matches[0]
    raise NotImplementedLewLMError(
        "LewLM could not find the expected synthesized audio artifact from mlx-audio.",
        details={"output_dir": output_dir, "audio_format": audio_format},
    )


def _duration_seconds_from_wav_bytes(audio_bytes: bytes) -> float | None:
    if not audio_bytes:
        return None
    with TemporaryDirectory(prefix="lewlm-mlx-audio-duration-") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        wav_path.write_bytes(audio_bytes)
        try:
            with wave.open(str(wav_path), "rb") as handle:
                frame_rate = handle.getframerate()
                if frame_rate <= 0:
                    return None
                return handle.getnframes() / frame_rate
        except wave.Error:
            return None


def _duration_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _first_duration_value(*values: Any) -> float | None:
    for value in values:
        duration = _duration_value(value)
        if duration is not None:
            return duration
    return None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        text = result.get("text")
        return text if isinstance(text, str) else ""
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else str(result)


def _bytes_like_to_bytes(value: Any) -> bytes | None:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if hasattr(value, "tobytes"):
        return value.tobytes()
    return None


def _import_optional_module(name: str) -> Any | None:
    try:
        return import_module(name)
    except ImportError:
        return None


def _media_type_for_format(audio_format: str) -> str:
    return speech_media_type(audio_format)
