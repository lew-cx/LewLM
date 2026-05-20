"""MLX vision-language runtime adapter with cache-aware multimodal batching hooks."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import fields
from functools import partial
import inspect
import hashlib
from importlib import import_module
import json
from pathlib import Path
from typing import Any, get_args, get_origin

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    CapabilityName,
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    ModelModality,
    RuntimeAffinity,
)
from lewlm.core.errors import ConfigurationError
from lewlm.runtime.base import ManagedTextRuntime
from lewlm.runtime.introspection import invoke_with_signature, resolve_backend_callable
from lewlm.runtime.metal import MLXAccelerationTracker
from lewlm.storage.block_cache import MultimodalEncoderCache


_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def load_mlx_vlm_backend_client(source_path: str, *, capability: str) -> Any | None:
    module = import_module("mlx_vlm")
    load = resolve_backend_callable(module, ("load", "load_model", "load_pipeline"), required=False)
    if load is None:
        return None
    provided_values = {
        "path_or_hf_repo": source_path,
        "path": source_path,
        "model_path": source_path,
        "source_path": source_path,
        "strict": False,
    }
    try:
        return invoke_with_signature(
            load,
            provided_values,
            capability=capability,
            passthrough_keys=("strict",),
        )
    except ValueError as exc:
        if "parameters not in model" not in str(exc):
            raise
    local_bundle_path = _local_gemma4_bundle_path(source_path)
    if local_bundle_path is None:
        raise
    return _load_local_gemma4_bundle(local_bundle_path)


def _local_gemma4_bundle_path(source_path: str) -> Path | None:
    candidate = Path(source_path).expanduser().resolve(strict=False)
    if not candidate.is_dir():
        return None
    config_path = candidate / "config.json"
    if not config_path.exists():
        return None
    try:
        config = _read_json_file(config_path)
    except ValueError:
        return None
    if config.get("model_type") != "gemma4":
        return None
    return candidate


def _read_json_file(path: Path) -> dict[str, Any]:
    import json

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def _load_local_gemma4_bundle(model_path: Path) -> Any:
    mlx_vlm_utils = import_module("mlx_vlm.utils")
    mx = import_module("mlx.core")
    nn = import_module("mlx.nn")
    config = getattr(mlx_vlm_utils, "load_config")(model_path)
    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", None)
    model_class, _ = getattr(mlx_vlm_utils, "get_model_and_args")(config=config)
    model_config = _hydrate_model_config(model_class.ModelConfig, config)
    model = model_class.Model(model_config)
    weights = _load_local_safetensor_weights(model_path, mx)
    if _mlx_vlm_weights_need_sanitization(weights):
        weights = getattr(mlx_vlm_utils, "sanitize_weights")(model, weights)
        if hasattr(model_class, "VisionModel") and getattr(model_config, "vision_config", None) is not None:
            weights = getattr(mlx_vlm_utils, "sanitize_weights")(model_class.VisionModel, weights, model_config.vision_config)
        if hasattr(model_class, "LanguageModel") and getattr(model_config, "text_config", None) is not None:
            weights = getattr(mlx_vlm_utils, "sanitize_weights")(model_class.LanguageModel, weights, model_config.text_config)
        if hasattr(model_class, "AudioModel") and getattr(model_config, "audio_config", None) is not None:
            weights = getattr(mlx_vlm_utils, "sanitize_weights")(model_class.AudioModel, weights, model_config.audio_config)
    quantization_specs = _infer_quantization_specs(
        parameter_shapes=_parameter_shapes(model.parameters()),
        weights=weights,
    )
    base_quantization = config.get("quantization", {})
    if quantization_specs:
        nn.quantize(
            model,
            group_size=int(base_quantization.get("group_size", 64)),
            bits=int(base_quantization.get("bits", 4)),
            mode=str(base_quantization.get("mode", "affine")),
            class_predicate=lambda path, module: _quantization_spec_for_path(
                path,
                module=module,
                quantization_specs=quantization_specs,
            ),
        )
    model.load_weights(list(weights.items()))
    processor = getattr(mlx_vlm_utils, "load_processor")(
        model_path,
        True,
        eos_token_ids=config.get("eos_token_id"),
    )
    load_image_processor = getattr(mlx_vlm_utils, "load_image_processor", None)
    if callable(load_image_processor):
        image_processor = load_image_processor(model_path)
        if image_processor is not None:
            processor.image_processor = image_processor
    return model, processor


def _hydrate_model_config(config_class: type[Any], raw_config: dict[str, Any]) -> Any:
    config = config_class.from_dict(raw_config)
    for field in fields(config_class):
        raw_value = raw_config.get(field.name)
        if not isinstance(raw_value, dict):
            continue
        nested_class = _config_field_class(field.type)
        if nested_class is None or not hasattr(nested_class, "from_dict"):
            continue
        setattr(config, field.name, nested_class.from_dict(raw_value))
    return config


def _config_field_class(field_type: Any) -> type[Any] | None:
    origin = get_origin(field_type)
    if origin is None:
        return field_type if isinstance(field_type, type) else None
    for candidate in get_args(field_type):
        if candidate is type(None):
            continue
        if isinstance(candidate, type):
            return candidate
    return None


def _load_local_safetensor_weights(model_path: Path, mx: Any) -> dict[str, Any]:
    weights: dict[str, Any] = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        if shard.name == "consolidated.safetensors":
            continue
        weights.update(mx.load(str(shard)))
    return weights


def _mlx_vlm_weights_need_sanitization(weights: dict[str, Any]) -> bool:
    return any(key.startswith("model.") for key in weights)


def _parameter_shapes(parameters: Any, prefix: str = "") -> dict[str, tuple[int, ...]]:
    tree_flatten = getattr(import_module("mlx.utils"), "tree_flatten")
    flat_parameters = tree_flatten(parameters, is_leaf=lambda _: False)
    return {
        name: tuple(value.shape)
        for name, value in flat_parameters
        if getattr(value, "shape", None) is not None
    }


def _infer_quantization_specs(
    *,
    parameter_shapes: dict[str, tuple[int, ...]],
    weights: dict[str, Any],
) -> dict[str, dict[str, int]]:
    specs: dict[str, dict[str, int]] = {}
    for key, value in weights.items():
        if not key.endswith(".weight"):
            continue
        path = key.removesuffix(".weight")
        scales_key = f"{path}.scales"
        if scales_key not in weights or key not in parameter_shapes:
            continue
        target_shape = parameter_shapes[key]
        stored_shape = getattr(value, "shape", None)
        scales_shape = getattr(weights[scales_key], "shape", None)
        if (
            stored_shape is None
            or scales_shape is None
            or len(target_shape) != 2
            or len(stored_shape) != 2
            or len(scales_shape) != 2
            or stored_shape[1] == 0
            or scales_shape[1] == 0
        ):
            continue
        pack_ratio = target_shape[1] / stored_shape[1]
        group_ratio = target_shape[1] / scales_shape[1]
        bits = int(round(32 / pack_ratio))
        group_size = int(round(group_ratio))
        if bits < 1 or group_size < 1:
            continue
        specs[path] = {"bits": bits, "group_size": group_size}
    return specs


def _quantization_spec_for_path(
    path: str,
    *,
    module: Any,
    quantization_specs: dict[str, dict[str, int]],
) -> dict[str, int] | bool:
    if path not in quantization_specs:
        return False
    if not hasattr(module, "to_quantized"):
        return False
    return quantization_specs[path]


def _resolve_mlx_vlm_batch_generate(module: Any) -> Any | None:
    return resolve_backend_callable(module, ("batch_generate", "generate_batch", "batch_chat"), required=False)


def _resolve_mlx_vlm_batch_stream_generate(module: Any) -> Any | None:
    return resolve_backend_callable(
        module,
        ("batch_stream_generate", "stream_generate_batch", "generate_stream_batch"),
        required=False,
    )


class MLXVisionRuntime(ManagedTextRuntime):
    """Adapter for MLX-native vision-language generation on Apple Silicon."""

    name = "mlx_vision"
    affinity = RuntimeAffinity.MLX_VISION
    supported_formats = (ModelFormat.MLX,)
    supported_modalities = (ModelModality.VISION, ModelModality.MULTIMODAL)
    supported_capabilities = frozenset(
        {
            CapabilityName.CHAT,
            CapabilityName.STREAMING,
            CapabilityName.VISION,
        },
    )
    supported_systems = ("Darwin",)
    supported_machines = ("arm64", "aarch64")
    platform_guidance = "Install the `mlx` extra on Apple Silicon macOS to enable MLX-native vision inference."

    def __init__(
        self,
        *,
        settings: LewLMSettings | None = None,
        multimodal_encoder_cache: MultimodalEncoderCache | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings or LewLMSettings()
        self._multimodal_encoder_cache = multimodal_encoder_cache
        self._clients: dict[str, Any] = {}
        self._acceleration = MLXAccelerationTracker(
            settings=self.settings,
            runtime_name=self.name,
            import_module_fn=import_module,
        )
        self._compile_safe_generation_wrappers: dict[tuple[str, str], Any] = {}
        self._encoder_cache_request_count = 0
        self._encoder_cache_hits = 0
        self._encoder_cache_misses = 0
        self._encoder_cache_saved_images = 0
        self._encoder_cache_saved_frames = 0
        self._encoder_cache_saved_bundles = 0
        self._native_batch_generate_calls = 0
        self._native_batch_stream_calls = 0
        self._native_batch_request_count = 0
        self._native_batch_max_size = 0
        self._stock_single_request_fallback_batches = 0
        self._stock_single_request_fallback_requests = 0

    def _check_environment(self) -> tuple[bool, str | None]:
        try:
            import_module("mlx_vlm")
        except ImportError:
            return False, "mlx-vlm is not installed"
        return True, None

    def supports_capability(self, capability: CapabilityName) -> bool:
        if not super().supports_capability(capability):
            return False
        module = import_module("mlx_vlm")
        if capability in {CapabilityName.CHAT, CapabilityName.VISION}:
            return resolve_backend_callable(module, ("generate", "chat", "generate_text"), required=False) is not None
        if capability == CapabilityName.STREAMING:
            return resolve_backend_callable(module, ("generate_stream", "stream_generate"), required=False) is not None
        return False

    @staticmethod
    def _accepts_parameter(callable_obj: Any, name: str) -> bool:
        if callable_obj is None:
            return False
        try:
            return name in inspect.signature(callable_obj).parameters
        except (TypeError, ValueError):
            return False

    def performance_feature_snapshot(self) -> dict[str, Any]:
        if not self.is_available():
            return {}
        module = import_module("mlx_vlm")
        generate = resolve_backend_callable(module, ("generate", "chat", "generate_text"), required=False)
        generate_stream = resolve_backend_callable(module, ("generate_stream", "stream_generate"), required=False)
        batch_generate = _resolve_mlx_vlm_batch_generate(module)
        batch_stream_generate = _resolve_mlx_vlm_batch_stream_generate(module)
        snapshot = self._acceleration.performance_feature_snapshot(callables=(generate, generate_stream))
        batching_supported = batch_generate is not None or batch_stream_generate is not None
        if batch_generate is not None and batch_stream_generate is not None:
            batching_reason = (
                "Installed mlx-vlm exposes native multimodal batching entrypoints for both chat and streaming requests."
            )
            batching_notes = [
                "LewLM routes same-model text-only and single-image multimodal bursts through mlx-vlm's detected batch hooks.",
            ]
        elif batch_generate is not None:
            batching_reason = (
                "Installed mlx-vlm exposes native multimodal batched chat generation through `batch_generate`."
            )
            batching_notes = [
                "LewLM only forwards text-only or single-image requests through the native chat batch path.",
                "Streaming stays on the stock single-request path until the installed mlx-vlm package exposes a compatible batched streaming entrypoint.",
                "Frame-bundle and mixed attachment-layout requests fall back clearly instead of pretending native batch support exists.",
            ]
        elif batch_stream_generate is not None:
            batching_reason = (
                "Installed mlx-vlm exposes a native batched streaming entrypoint, but not a matching batched chat callable."
            )
            batching_notes = [
                "LewLM keeps non-streaming multimodal chat on the stock single-request path when mlx-vlm does not expose `batch_generate`.",
            ]
        else:
            batching_reason = "The current MLX vision adapter does not detect an explicit batched chat or streaming entrypoint."
            batching_notes = [
                "LewLM can only batch multimodal chat when the selected vision runtime exposes native batched generation hooks.",
            ]
        if self._stock_single_request_fallback_requests:
            batching_notes.append(
                "Recent multimodal batch candidates fell back to stock single-request generation because the request shape was not compatible with the detected native batch hook.",
            )
        snapshot["continuous_batching"] = {
            "supported": batching_supported,
            "active": self._native_batch_request_count > 0,
            "ownership": "backend_native" if batching_supported else "unsupported",
            "reason": batching_reason,
            "notes": batching_notes,
            "metrics": _compact_runtime_metrics(
                chat_batch_calls=self._native_batch_generate_calls,
                stream_batch_calls=self._native_batch_stream_calls,
                batched_requests=self._native_batch_request_count,
                max_batch_size=self._native_batch_max_size,
                stock_single_request_fallback_batches=self._stock_single_request_fallback_batches,
                stock_single_request_fallback_requests=self._stock_single_request_fallback_requests,
            ),
        }
        encoder_cache_supported = self._multimodal_encoder_cache is not None and any(
            self._accepts_parameter(callable_obj, "vision_cache")
            for callable_obj in (generate, generate_stream)
        )
        snapshot["multimodal_encoder_caching"] = {
            "supported": encoder_cache_supported,
            "active": encoder_cache_supported and self._encoder_cache_request_count > 0,
            "reason": (
                "LewLM forwards content-addressed image and frame-bundle requests through mlx-vlm's native `vision_cache` hook."
                if encoder_cache_supported
                else "Installed mlx-vlm generation entrypoints do not expose a compatible `vision_cache` hook."
            ),
            "notes": (
                [
                    "Image-bundle directories are expanded into stable frame lists before cache lookup.",
                    "Encoder cache keys combine content hashes with a preprocessing fingerprint so edited sources invalidate stale features.",
                ]
                if encoder_cache_supported
                else []
            ),
            "metrics": _compact_runtime_metrics(
                request_count=self._encoder_cache_request_count,
                cache_hits=self._encoder_cache_hits,
                cache_misses=self._encoder_cache_misses,
                cached_image_inputs=self._encoder_cache_saved_images,
                cached_frame_inputs=self._encoder_cache_saved_frames,
                cached_bundle_requests=self._encoder_cache_saved_bundles,
            ),
        }
        return snapshot

    def supports_continuous_batching(self, capability: CapabilityName) -> bool:
        if not self.is_available():
            return False
        module = import_module("mlx_vlm")
        if capability == CapabilityName.CHAT:
            return _resolve_mlx_vlm_batch_generate(module) is not None
        if capability == CapabilityName.STREAMING:
            return _resolve_mlx_vlm_batch_stream_generate(module) is not None
        return False

    async def _load_model(self, manifest: ModelManifest) -> None:
        client = load_mlx_vlm_backend_client(manifest.source_path, capability="model_load")
        if client is None:
            self._clients[manifest.model_id] = {"model": manifest.source_path}
            return
        if isinstance(client, tuple):
            model = client[0] if len(client) > 0 else manifest.source_path
            processor = client[1] if len(client) > 1 else None
            self._clients[manifest.model_id] = {"model": model, "processor": processor}
            return
        self._clients[manifest.model_id] = {"model": client}

    async def _unload_model(self, model_id: str) -> None:
        self._clients.pop(model_id, None)
        for phase in ("decode", "stream"):
            self._compile_safe_generation_wrappers.pop((model_id, phase), None)
            self._acceleration.clear_compiled_callable(
                callable_key=self._vision_acceleration_callable_key(model_id=model_id, phase=phase),
            )
        if self._multimodal_encoder_cache is not None:
            self._multimodal_encoder_cache.drop_runtime_resident_features(runtime=self.name, model_id=model_id)

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        module = import_module("mlx_vlm")
        generate = resolve_backend_callable(module, ("generate", "chat", "generate_text"))
        client = self._clients[request.model_id]
        prompt = _messages_to_prompt(request.messages)
        cache_bridge, image_paths, cache_lookup_source = self._build_encoder_cache_bridge(
            request=request,
            client=client,
            callable_obj=generate,
        )
        callable_obj, callable_key, provided_values, passthrough_keys = self._generation_invocation(
            request=request,
            callable_obj=generate,
            client=client,
            prompt=prompt,
            image_paths=image_paths,
            cache_lookup_source=cache_lookup_source,
            cache_bridge=cache_bridge,
            phase="decode",
        )
        try:
            result = self._acceleration.invoke(
                request=request,
                callable_obj=callable_obj,
                callable_key=callable_key,
                provided_values=provided_values,
                capability=CapabilityName.CHAT.value,
                passthrough_keys=passthrough_keys,
                phase="decode",
            )
        finally:
            self._record_encoder_cache_request(request=request, cache_bridge=cache_bridge)
        output_text = _generation_text_from_result(result)
        return GenerateResponse(
            model_id=request.model_id,
            output_text=output_text,
            finish_reason="stop",
            usage={},
        )

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        module = import_module("mlx_vlm")
        generate_stream = resolve_backend_callable(module, ("generate_stream", "stream_generate"))
        client = self._clients[request.model_id]
        prompt = _messages_to_prompt(request.messages)
        cache_bridge, image_paths, cache_lookup_source = self._build_encoder_cache_bridge(
            request=request,
            client=client,
            callable_obj=generate_stream,
        )
        callable_obj, callable_key, provided_values, passthrough_keys = self._generation_invocation(
            request=request,
            callable_obj=generate_stream,
            client=client,
            prompt=prompt,
            image_paths=image_paths,
            cache_lookup_source=cache_lookup_source,
            cache_bridge=cache_bridge,
            phase="stream",
        )
        try:
            chunks = self._acceleration.invoke(
                request=self._stream_acceleration_request(request),
                callable_obj=callable_obj,
                callable_key=callable_key,
                provided_values=provided_values,
                capability=CapabilityName.STREAMING.value,
                passthrough_keys=passthrough_keys,
                phase="stream",
            )
            for chunk in chunks:
                text = _chunk_to_text(chunk)
                if text:
                    yield text
        finally:
            self._record_encoder_cache_request(request=request, cache_bridge=cache_bridge)

    async def generate_batch(self, requests: Sequence[GenerateRequest]) -> list[GenerateResponse]:
        self._ensure_available()
        if not requests:
            return []
        model_id = self._validate_batch_requests(requests)
        module = import_module("mlx_vlm")
        batch_generate = _resolve_mlx_vlm_batch_generate(module)
        if batch_generate is None:
            raise ConfigurationError("Installed MLX vision package does not expose a native batched chat entrypoint.")
        prompt_values, image_inputs, fallback_reason = self._prepare_native_batch_inputs(requests)
        if fallback_reason is not None:
            return await self._generate_batch_stock_fallback(requests=requests, reason=fallback_reason)
        client = self._clients[model_id]
        sampling_options = _mlx_sampling_options(requests[0].temperature)
        result = invoke_with_signature(
            batch_generate,
            {
                "client": client,
                "model": client.get("model"),
                "processor": client.get("processor"),
                "prompts": prompt_values,
                "prompt": prompt_values,
                "images": image_inputs,
                "image": image_inputs,
                "image_paths": image_inputs,
                "audios": None,
                "audio": None,
                "max_tokens": [request.max_tokens for request in requests],
                **sampling_options,
                "verbose": False,
                "group_by_shape": True,
                "track_image_sizes": False,
                "prefill_step_size": self.settings.prefill_token_batch_size,
            },
            capability="native_batch_generate",
            passthrough_keys=(
                "max_tokens",
                "sampler",
                "verbose",
                "group_by_shape",
                "track_image_sizes",
                "prefill_step_size",
            ),
        )
        texts = _batch_generation_texts(result)
        if len(texts) != len(requests):
            raise ConfigurationError(
                f"Expected {len(requests)} MLX multimodal batched chat results, received {len(texts)}.",
            )
        self._native_batch_generate_calls += 1
        self._native_batch_request_count += len(requests)
        self._native_batch_max_size = max(self._native_batch_max_size, len(requests))
        responses: list[GenerateResponse] = []
        for request, text in zip(requests, texts, strict=True):
            self._record_native_batching_metadata(
                request=request,
                capability=CapabilityName.CHAT,
                active=True,
                backend="mlx_vlm.batch_generate",
                batch_size=len(requests),
            )
            responses.append(
                GenerateResponse(
                    model_id=request.model_id,
                    output_text=text,
                    finish_reason="stop",
                    usage={},
                ),
            )
        return responses

    async def stream_generate_batch(self, requests: Sequence[GenerateRequest]) -> AsyncIterator[tuple[int, str]]:
        self._ensure_available()
        if not requests:
            return
        model_id = self._validate_batch_requests(requests)
        module = import_module("mlx_vlm")
        batch_stream_generate = _resolve_mlx_vlm_batch_stream_generate(module)
        if batch_stream_generate is None:
            raise ConfigurationError("Installed MLX vision package does not expose a native batched streaming entrypoint.")
        prompt_values, image_inputs, fallback_reason = self._prepare_native_batch_inputs(requests)
        if fallback_reason is not None:
            async for item in self._stream_batch_stock_fallback(requests=requests, reason=fallback_reason):
                yield item
            return
        client = self._clients[model_id]
        self._native_batch_stream_calls += 1
        self._native_batch_request_count += len(requests)
        self._native_batch_max_size = max(self._native_batch_max_size, len(requests))
        for request in requests:
            self._record_native_batching_metadata(
                request=request,
                capability=CapabilityName.STREAMING,
                active=True,
                backend="mlx_vlm.batch_stream_generate",
                batch_size=len(requests),
            )
        result = invoke_with_signature(
            batch_stream_generate,
            {
                "client": client,
                "model": client.get("model"),
                "processor": client.get("processor"),
                "prompts": prompt_values,
                "prompt": prompt_values,
                "images": image_inputs,
                "image": image_inputs,
                "image_paths": image_inputs,
                "audios": None,
                "audio": None,
                "max_tokens": [request.max_tokens for request in requests],
                **_mlx_sampling_options(requests[0].temperature),
                "verbose": False,
                "prefill_step_size": self.settings.prefill_token_batch_size,
            },
            capability="native_batch_stream_generate",
            passthrough_keys=(
                "max_tokens",
                "sampler",
                "verbose",
                "prefill_step_size",
            ),
        )
        for item in result:
            request_index, raw_chunk = _batch_stream_item(item)
            if request_index is None or request_index < 0 or request_index >= len(requests):
                continue
            text = _chunk_to_text(raw_chunk)
            if text:
                yield request_index, text

    def _tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens) -> str:
        return bytes(tokens).decode("utf-8")

    def _build_encoder_cache_bridge(
        self,
        *,
        request: GenerateRequest,
        client: dict[str, Any],
        callable_obj: Any,
    ) -> tuple["_VisionEncoderCacheBridge | None", list[str], str | None]:
        image_paths, modality, source_locator, content_sha256, total_input_bytes, bundle_count = _request_image_paths(request)
        if (
            self._multimodal_encoder_cache is None
            or not image_paths
            or not self._accepts_parameter(callable_obj, "vision_cache")
        ):
            return None, image_paths, image_paths[0] if image_paths else None
        manifest = self._loaded_manifests[request.model_id]
        preprocessing_fingerprint = _vision_preprocessing_fingerprint(
            client=client,
            image_paths=image_paths,
            modality=modality,
            bundle_count=bundle_count,
        )
        cache_key = self._multimodal_encoder_cache.cache_key_for_feature(
            runtime=self.name,
            model_id=request.model_id,
            model_fingerprint=manifest.fingerprint,
            modality=modality,
            content_sha256=content_sha256,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )
        return (
            _VisionEncoderCacheBridge(
                encoder_cache=self._multimodal_encoder_cache,
                cache_key=cache_key,
                runtime=self.name,
                model_id=request.model_id,
                model_fingerprint=manifest.fingerprint,
                modality=modality,
                content_sha256=content_sha256,
                preprocessing_fingerprint=preprocessing_fingerprint,
                source_locator=source_locator,
                image_count=len(image_paths),
                frame_count=len(image_paths) if modality == "frame_bundle" else 0,
                bundle_count=bundle_count,
                total_input_bytes=total_input_bytes,
            ),
            image_paths,
            image_paths[0],
        )

    def _generation_invocation(
        self,
        *,
        request: GenerateRequest,
        callable_obj: Any,
        client: dict[str, Any],
        prompt: str,
        image_paths: list[str],
        cache_lookup_source: str | None,
        cache_bridge: "_VisionEncoderCacheBridge | None",
        phase: str,
    ) -> tuple[Any, str, dict[str, Any], tuple[str, ...]]:
        callable_key = self._vision_acceleration_callable_key(model_id=request.model_id, phase=phase)
        passthrough_keys = ("max_tokens", "temperature", "vision_cache")
        provided_values = {
            "client": client,
            "model": client.get("model"),
            "processor": client.get("processor"),
            "prompt": prompt,
            "messages": [{"role": message.role, "content": message.content} for message in request.messages],
            "images": image_paths,
            "image": cache_lookup_source,
            "image_paths": image_paths,
            "vision_cache": cache_bridge,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "temp": request.temperature,
            "verbose": False,
        }
        if not self._request_graph_compile_enabled(request):
            return callable_obj, callable_key, provided_values, passthrough_keys
        prepared_values = self._prepare_compile_safe_generation_values(
            prompt=prompt,
            image_paths=image_paths,
            cache_lookup_source=cache_lookup_source,
            client=client,
        )
        compile_safe_callable = self._compile_safe_generation_callable(
            model_id=request.model_id,
            callable_obj=callable_obj,
            client=client,
            phase=phase,
        )
        if prepared_values is None or compile_safe_callable is callable_obj:
            return callable_obj, callable_key, provided_values, passthrough_keys
        prepared_passthrough_keys = tuple(
            key
            for key in prepared_values
            if key
            not in {
                "client",
                "model",
                "processor",
                "prompt",
                "messages",
                "images",
                "image",
                "image_paths",
                "max_tokens",
                "temperature",
                "temp",
                "verbose",
            }
        )
        return (
            compile_safe_callable,
            callable_key,
            {
                key: value
                for key, value in provided_values.items()
                if key not in {"client", "model", "processor", "messages", "images", "image_paths"}
            }
            | {
                **prepared_values,
            },
            passthrough_keys + prepared_passthrough_keys,
        )

    def _prepare_compile_safe_generation_values(
        self,
        *,
        prompt: str,
        image_paths: list[str],
        cache_lookup_source: str | None,
        client: dict[str, Any],
    ) -> dict[str, Any] | None:
        processor = client.get("processor")
        model = client.get("model")
        if processor is None or model is None:
            return None
        try:
            mlx_vlm_utils = import_module("mlx_vlm.utils")
        except ImportError:
            return None
        prepare_inputs = getattr(mlx_vlm_utils, "prepare_inputs", None)
        if not callable(prepare_inputs):
            return None
        model_config = getattr(model, "config", None)
        model_type = getattr(model_config, "model_type", None)
        image_source = _generation_image_argument(image_paths=image_paths, cache_lookup_source=cache_lookup_source)
        prepared_inputs = prepare_inputs(
            processor,
            images=image_source,
            prompts=prompt,
            image_token_index=getattr(model_config, "image_token_index", None),
            add_special_tokens=(
                getattr(processor, "chat_template", None) is None
                if model_type in {"gemma3", "gemma3n", "gemma4"}
                else True
            ),
        )
        prepared_mapping = dict(prepared_inputs)
        if prepared_mapping.get("input_ids") is None:
            return None
        passthrough_inputs = {
            key: value
            for key, value in prepared_mapping.items()
            if key not in {"input_ids", "pixel_values", "attention_mask"}
        }
        return {
            "prompt": prompt,
            "image": image_source,
            "input_ids": prepared_mapping.get("input_ids"),
            "pixel_values": prepared_mapping.get("pixel_values"),
            "mask": prepared_mapping.get("attention_mask"),
            **passthrough_inputs,
        }

    def _compile_safe_generation_callable(
        self,
        *,
        model_id: str,
        callable_obj: Any,
        client: dict[str, Any],
        phase: str,
    ) -> Any:
        cache_key = (model_id, phase)
        cached = self._compile_safe_generation_wrappers.get(cache_key)
        if cached is not None:
            return cached
        bound_values = {}
        if self._accepts_parameter(callable_obj, "client"):
            bound_values["client"] = client
        if self._accepts_parameter(callable_obj, "model"):
            bound_values["model"] = client.get("model")
        if self._accepts_parameter(callable_obj, "processor"):
            bound_values["processor"] = client.get("processor")
        if not bound_values:
            return callable_obj
        wrapper = partial(callable_obj, **bound_values)
        self._compile_safe_generation_wrappers[cache_key] = wrapper
        return wrapper

    def _request_graph_compile_enabled(self, request: GenerateRequest) -> bool:
        acceleration_payload = request.metadata.get("mlx_acceleration")
        if isinstance(acceleration_payload, dict):
            override_value = acceleration_payload.get("graph_compile_enabled")
            if isinstance(override_value, bool):
                return override_value
        return self.settings.mlx_graph_compile_enabled

    @staticmethod
    def _stream_acceleration_request(request: GenerateRequest) -> GenerateRequest:
        acceleration_payload = request.metadata.get("mlx_acceleration")
        overrides = dict(acceleration_payload) if isinstance(acceleration_payload, dict) else {}
        overrides["graph_compile_enabled"] = False
        overrides.setdefault(
            "graph_compile_override_reason",
            "LewLM keeps MLX vision streaming on the stock path because backend stream generators can fail lazily after graph compilation.",
        )
        request.metadata["mlx_acceleration"] = overrides
        return request

    @staticmethod
    def _vision_acceleration_callable_key(*, model_id: str, phase: str) -> str:
        return f"mlx_vision:{model_id}:{phase}"

    def _record_encoder_cache_request(
        self,
        *,
        request: GenerateRequest,
        cache_bridge: "_VisionEncoderCacheBridge | None",
    ) -> None:
        if cache_bridge is None:
            return
        metrics = cache_bridge.metrics()
        request.metadata["encoder_cache"] = metrics
        self._encoder_cache_request_count += 1
        self._encoder_cache_hits += metrics["cache_hits"]
        self._encoder_cache_misses += metrics["cache_misses"]
        self._encoder_cache_saved_images += metrics["image_input_count"]
        self._encoder_cache_saved_frames += metrics["frame_count"]
        self._encoder_cache_saved_bundles += metrics["bundle_count"]

    def _validate_batch_requests(self, requests: Sequence[GenerateRequest]) -> str:
        model_ids = {request.model_id for request in requests}
        if len(model_ids) != 1:
            raise ConfigurationError("MLX multimodal native batching requires all requests to target the same model.")
        model_id = next(iter(model_ids))
        self._ensure_loaded(model_id)
        self._touch_model(model_id)
        return model_id

    def _prepare_native_batch_inputs(
        self,
        requests: Sequence[GenerateRequest],
    ) -> tuple[list[str], list[str] | None, str | None]:
        prompt_values: list[str] = []
        image_inputs: list[str] = []
        saw_images = False
        saw_text_only = False
        for request in requests:
            prompt_values.append(_messages_to_prompt(request.messages))
            image_paths, modality, _, _, _, bundle_count = _request_image_paths(request)
            if not image_paths:
                saw_text_only = True
                continue
            if modality != "image" or bundle_count or len(image_paths) != 1:
                return (
                    prompt_values,
                    None,
                    "Installed mlx-vlm native chat batching currently supports only text-only or single-image requests; frame bundles and multi-image prompts stay on the stock single-request path.",
                )
            saw_images = True
            image_inputs.append(image_paths[0])
        if saw_images and saw_text_only:
            return (
                prompt_values,
                None,
                "Installed mlx-vlm native chat batching cannot mix text-only and image-conditioned requests inside the same backend batch.",
            )
        return prompt_values, image_inputs if saw_images else None, None

    async def _generate_batch_stock_fallback(
        self,
        *,
        requests: Sequence[GenerateRequest],
        reason: str,
    ) -> list[GenerateResponse]:
        self._stock_single_request_fallback_batches += 1
        self._stock_single_request_fallback_requests += len(requests)
        responses: list[GenerateResponse] = []
        for request in requests:
            self._record_native_batching_metadata(
                request=request,
                capability=CapabilityName.CHAT,
                active=False,
                backend="stock_single_request",
                batch_size=len(requests),
                fallback_reason=reason,
            )
            responses.append(await self._generate(request))
        return responses

    async def _stream_batch_stock_fallback(
        self,
        *,
        requests: Sequence[GenerateRequest],
        reason: str,
    ) -> AsyncIterator[tuple[int, str]]:
        self._stock_single_request_fallback_batches += 1
        self._stock_single_request_fallback_requests += len(requests)
        for request in requests:
            self._record_native_batching_metadata(
                request=request,
                capability=CapabilityName.STREAMING,
                active=False,
                backend="stock_single_request",
                batch_size=len(requests),
                fallback_reason=reason,
            )
        for index, request in enumerate(requests):
            async for delta in self._stream_generate(request):
                if delta:
                    yield index, delta

    @staticmethod
    def _record_native_batching_metadata(
        *,
        request: GenerateRequest,
        capability: CapabilityName,
        active: bool,
        backend: str,
        batch_size: int,
        fallback_reason: str | None = None,
    ) -> None:
        request.metadata["native_batching"] = {
            "capability": capability.value,
            "supported": True,
            "active": active,
            "backend": backend,
            "batch_size": batch_size,
            "stock_single_request_path": not active,
            "fallback": not active,
            "fallback_reason": fallback_reason,
            "ownership": "backend_native",
        }


class _VisionEncoderCacheBridge:
    def __init__(
        self,
        *,
        encoder_cache: MultimodalEncoderCache,
        cache_key: str,
        runtime: str,
        model_id: str,
        model_fingerprint: str,
        modality: str,
        content_sha256: str,
        preprocessing_fingerprint: str,
        source_locator: str,
        image_count: int,
        frame_count: int,
        bundle_count: int,
        total_input_bytes: int,
    ) -> None:
        self._encoder_cache = encoder_cache
        self._cache_key = cache_key
        self._runtime = runtime
        self._model_id = model_id
        self._model_fingerprint = model_fingerprint
        self._modality = modality
        self._content_sha256 = content_sha256
        self._preprocessing_fingerprint = preprocessing_fingerprint
        self._source_locator = source_locator
        self._image_count = image_count
        self._frame_count = frame_count
        self._bundle_count = bundle_count
        self._total_input_bytes = total_input_bytes
        self._cache_hit = 0
        self._cache_miss = 0

    def get(self, _image_source: object) -> Any | None:
        feature = self._encoder_cache.get_feature(cache_key=self._cache_key)
        if feature is None:
            self._cache_miss = 1
        else:
            self._cache_hit = 1
        return feature

    def put(self, _image_source: object, features: Any) -> None:
        self._encoder_cache.put_feature(
            cache_key=self._cache_key,
            runtime=self._runtime,
            model_id=self._model_id,
            model_fingerprint=self._model_fingerprint,
            modality=self._modality,
            content_sha256=self._content_sha256,
            preprocessing_fingerprint=self._preprocessing_fingerprint,
            feature=features,
            source_locator=self._source_locator,
            metadata={
                "image_count": self._image_count,
                "frame_count": self._frame_count,
                "bundle_count": self._bundle_count,
                "input_bytes": self._total_input_bytes,
            },
        )

    def metrics(self) -> dict[str, int]:
        return {
            "cache_hits": self._cache_hit,
            "cache_misses": self._cache_miss,
            "image_input_count": self._image_count,
            "frame_count": self._frame_count,
            "bundle_count": self._bundle_count,
            "input_bytes": self._total_input_bytes,
        }


def _request_image_paths(request: GenerateRequest) -> tuple[list[str], str, str, str, int, int]:
    image_paths: list[str] = []
    source_locators: list[str] = []
    content_parts: list[dict[str, Any]] = []
    total_input_bytes = 0
    bundle_count = 0
    for message in request.messages:
        for attachment in message.attachments:
            if attachment.attachment_type == "image" and attachment.source_path:
                source_path = Path(attachment.source_path).expanduser().resolve(strict=False)
                expanded_paths = _expanded_image_paths(source_path)
                if len(expanded_paths) > 1 or source_path.is_dir():
                    bundle_count += 1
                    source_locators.append(f"bundle:{source_path}")
                else:
                    source_locators.append(f"image:{source_path}")
                resolved_paths = expanded_paths or [str(source_path)]
                image_paths.extend(resolved_paths)
                file_parts, attachment_bytes = _image_content_parts(source_path, resolved_paths)
                total_input_bytes += attachment_bytes
                content_parts.append({"files": file_parts})
    content_sha256 = _stable_digest(content_parts)
    modality = "frame_bundle" if bundle_count else "image"
    return image_paths, modality, "|".join(source_locators), content_sha256, total_input_bytes, bundle_count


def _messages_to_prompt(messages: list[Any]) -> str:
    rendered = []
    for message in messages:
        rendered.append(f"{message.role}: {message.content}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def _mlx_sampling_options(temperature: float) -> dict[str, Any]:
    if temperature <= 0:
        return {}
    try:
        sample_utils = import_module("mlx_lm.sample_utils")
    except ImportError:
        return {}
    make_sampler = getattr(sample_utils, "make_sampler", None)
    if callable(make_sampler):
        return {"sampler": make_sampler(temp=temperature)}
    return {}


def _generation_image_argument(*, image_paths: list[str], cache_lookup_source: str | None) -> str | list[str] | None:
    if len(image_paths) > 1:
        return image_paths
    return cache_lookup_source


def _generation_text_from_result(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        text = result.get("text", result.get("output_text"))
        return text if isinstance(text, str) else str(result)
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else str(result)


def _chunk_to_text(chunk: object) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        text = chunk.get("text")
        return text if isinstance(text, str) else ""
    text = getattr(chunk, "text", None)
    return text if isinstance(text, str) else ""


def _batch_generation_texts(result: object) -> list[str]:
    if isinstance(result, dict):
        texts = result.get("texts", result.get("output_texts"))
        if isinstance(texts, list):
            return [str(item) for item in texts]
    texts = getattr(result, "texts", None)
    if isinstance(texts, list):
        return [str(item) for item in texts]
    if isinstance(result, list):
        return [str(item) for item in result]
    raise ConfigurationError("Installed MLX vision batch entrypoint returned an unsupported result shape.")


def _batch_stream_item(item: object) -> tuple[int | None, object]:
    if isinstance(item, tuple) and len(item) == 2:
        index = item[0]
        return (int(index), item[1]) if isinstance(index, (int, float)) else (None, item[1])
    if isinstance(item, dict):
        index = item.get("request_index", item.get("index", item.get("uid")))
        return (int(index), item.get("chunk", item)) if isinstance(index, (int, float)) else (None, item)
    for attribute in ("request_index", "index", "uid"):
        index = getattr(item, attribute, None)
        if isinstance(index, (int, float)):
            chunk = getattr(item, "chunk", getattr(item, "delta", item))
            return int(index), chunk
    return None, item


def _expanded_image_paths(source_path: Path) -> list[str]:
    if not source_path.exists():
        return [str(source_path)]
    if source_path.is_dir():
        image_paths = sorted(
            str(candidate)
            for candidate in source_path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in _IMAGE_SUFFIXES
        )
        return image_paths or [str(source_path)]
    return [str(source_path)]


def _image_content_parts(source_path: Path, resolved_paths: list[str]) -> tuple[list[dict[str, Any]], int]:
    parts: list[dict[str, Any]] = []
    total_size = 0
    for raw_path in resolved_paths:
        candidate = Path(raw_path)
        stat = candidate.stat() if candidate.exists() else None
        total_size += stat.st_size if stat is not None else 0
        part: dict[str, Any] = {
            "size": stat.st_size if stat is not None else None,
            "digest": _file_digest(candidate),
        }
        if stat is None:
            part["source_locator"] = str(candidate)
        parts.append(part)
    if source_path.is_dir() and not parts:
        parts.append({"source_locator": str(source_path), "missing": True})
    return parts, total_size


def _file_digest(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vision_preprocessing_fingerprint(
    *,
    client: dict[str, Any],
    image_paths: list[str],
    modality: str,
    bundle_count: int,
) -> str:
    processor = client.get("processor")
    model = client.get("model")
    config = {
        "processor_type": type(processor).__name__ if processor is not None else None,
        "model_type": type(model).__name__ if model is not None else None,
        "processor_fields": _dataclass_public_fields(processor),
        "image_count": len(image_paths),
        "modality": modality,
        "bundle_count": bundle_count,
    }
    return _stable_digest(config)


def _dataclass_public_fields(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        return {field.name: getattr(value, field.name) for field in fields(value)}
    except TypeError:
        return None


def _callable_accepts_parameter(callable_obj: Any, name: str) -> bool:
    if callable_obj is None:
        return False
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def _compact_runtime_metrics(**values: int) -> dict[str, int]:
    return {key: value for key, value in values.items() if value}
