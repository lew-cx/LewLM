from __future__ import annotations

import asyncio
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    ConversionStatus,
    GenerateAttachment,
    GenerateMessage,
    GenerateRequest,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.runtime.mlx_vision.runtime import (
    MLXVisionRuntime,
    _infer_quantization_specs,
    _mlx_vlm_weights_need_sanitization,
    load_mlx_vlm_backend_client,
)
from lewlm.storage import BlockDiskCache, MetadataStore, MultimodalEncoderCache

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="MLX runtimes are macOS-only.")


def test_mlx_vision_runtime_supports_real_mlx_vlm_signatures(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load(*, path_or_hf_repo: str, **kwargs):
        captured["path_or_hf_repo"] = path_or_hf_repo
        captured["load_kwargs"] = kwargs
        return "vision-model", "vision-processor"

    def fake_generate(*, model, processor, prompt, image=None, verbose=False, **kwargs):
        captured["generate"] = {
            "model": model,
            "processor": processor,
            "prompt": prompt,
            "image": image,
            "verbose": verbose,
            "kwargs": kwargs,
        }
        return SimpleNamespace(text="vision output")

    fake_module = SimpleNamespace(
        load=fake_load,
        generate=fake_generate,
        stream_generate=lambda **kwargs: [],
    )
    monkeypatch.setattr("lewlm.runtime.mlx_vision.runtime.import_module", lambda name: fake_module)

    runtime = MLXVisionRuntime()
    manifest = _manifest()

    asyncio.run(runtime.load_model(manifest))
    response = asyncio.run(
        runtime.generate(
            GenerateRequest(
                model_id=manifest.model_id,
                messages=[
                    GenerateMessage(
                        role="user",
                        content="describe this image",
                        attachments=[
                            GenerateAttachment(
                                attachment_type="image",
                                name="sample.png",
                                source_path="/tmp/sample.png",
                            ),
                        ],
                    ),
                ],
                max_tokens=32,
                temperature=0.2,
            ),
        ),
    )

    assert captured["path_or_hf_repo"] == manifest.source_path
    assert captured["load_kwargs"] == {"strict": False}
    assert captured["generate"] == {
        "model": "vision-model",
        "processor": "vision-processor",
        "prompt": "user: describe this image\nassistant:",
        "image": str(Path("/tmp/sample.png").resolve(strict=False)),
        "verbose": False,
        "kwargs": {"max_tokens": 32, "temperature": 0.2},
    }
    assert response.output_text == "vision output"


def test_mlx_vlm_weights_need_sanitization_detects_raw_namespace() -> None:
    assert _mlx_vlm_weights_need_sanitization({"model.language_model.layers.0.self_attn.q_proj.weight": object()})
    assert not _mlx_vlm_weights_need_sanitization(
        {"language_model.model.layers.0.self_attn.q_proj.weight": object()},
    )


def test_infer_quantization_specs_detects_mixed_4_and_8_bit_weights() -> None:
    parameter_shapes = {
        "language_model.model.layers.0.self_attn.q_proj.weight": (8192, 5376),
        "language_model.model.layers.0.mlp.gate_proj.weight": (21504, 5376),
        "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": (1152, 1152),
    }
    weights = {
        "language_model.model.layers.0.self_attn.q_proj.weight": SimpleNamespace(shape=(8192, 1344)),
        "language_model.model.layers.0.self_attn.q_proj.scales": SimpleNamespace(shape=(8192, 84)),
        "language_model.model.layers.0.mlp.gate_proj.weight": SimpleNamespace(shape=(21504, 672)),
        "language_model.model.layers.0.mlp.gate_proj.scales": SimpleNamespace(shape=(21504, 84)),
        "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": SimpleNamespace(shape=(1152, 1152)),
    }

    assert _infer_quantization_specs(parameter_shapes=parameter_shapes, weights=weights) == {
        "language_model.model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 64},
        "language_model.model.layers.0.mlp.gate_proj": {"bits": 4, "group_size": 64},
    }


def test_load_mlx_vlm_backend_client_uses_local_gemma4_loader_after_namespace_mismatch(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        load=lambda **kwargs: (_ for _ in ()).throw(ValueError("Received 2010 parameters not in model:\nfoo.")),
    )
    monkeypatch.setattr("lewlm.runtime.mlx_vision.runtime.import_module", lambda name: fake_module)
    monkeypatch.setattr(
        "lewlm.runtime.mlx_vision.runtime._local_gemma4_bundle_path",
        lambda source_path: Path("/tmp/gemma4-bundle"),
    )
    monkeypatch.setattr(
        "lewlm.runtime.mlx_vision.runtime._load_local_gemma4_bundle",
        lambda model_path: ("fixed-model", "fixed-processor"),
    )

    assert load_mlx_vlm_backend_client("/tmp/gemma4-bundle", capability="model_load") == (
        "fixed-model",
        "fixed-processor",
    )


def test_mlx_vision_runtime_uses_graph_compile_and_custom_sdpa(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {"compiled_calls": 0}
    fake_model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4", image_token_index=42))
    fake_processor = SimpleNamespace(chat_template=None)

    def fake_load(*, path_or_hf_repo: str, **kwargs):
        return fake_model, fake_processor

    def fake_prepare_inputs(
        processor,
        images=None,
        prompts=None,
        image_token_index=None,
        add_special_tokens=False,
        **kwargs,
    ):
        captured["prepare_inputs"] = {
            "processor": processor,
            "images": images,
            "prompts": prompts,
            "image_token_index": image_token_index,
            "add_special_tokens": add_special_tokens,
        }
        return {
            "input_ids": "prepared-input-ids",
            "pixel_values": "prepared-pixel-values",
            "attention_mask": "prepared-mask",
            "image_sizes": "prepared-image-sizes",
        }

    def fake_generate(
        *,
        model,
        processor,
        prompt,
        image=None,
        input_ids=None,
        pixel_values=None,
        mask=None,
        image_sizes=None,
        max_tokens: int,
        sdpa_kernel: str | None = None,
        vision_cache=None,
        verbose: bool = False,
        **kwargs,
    ):
        captured["generate"] = {
            "model": model,
            "processor": processor,
            "prompt": prompt,
            "image": image,
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "mask": mask,
            "image_sizes": image_sizes,
            "max_tokens": max_tokens,
            "sdpa_kernel": sdpa_kernel,
            "vision_cache": vision_cache,
            "verbose": verbose,
        }
        return SimpleNamespace(text="accelerated vision output")

    def fake_compile(callable_obj):
        def compiled(**kwargs):
            captured["compiled_calls"] = int(captured["compiled_calls"]) + 1
            captured["compiled_kwargs"] = dict(kwargs)
            return callable_obj(**kwargs)

        return compiled

    def fake_import(name: str):
        if name == "mlx_vlm":
            return SimpleNamespace(load=fake_load, generate=fake_generate, stream_generate=lambda **kwargs: [])
        if name == "mlx_vlm.utils":
            return SimpleNamespace(prepare_inputs=fake_prepare_inputs)
        if name == "mlx.core":
            return SimpleNamespace(compile=fake_compile)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_vision.runtime.import_module", fake_import)

    runtime = MLXVisionRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            mlx_graph_compile_enabled=True,
            mlx_attention_kernel_mode="custom_sdpa",
        ),
    )
    manifest = _manifest()
    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[
            GenerateMessage(
                role="user",
                content="describe this image",
                attachments=[
                    GenerateAttachment(
                        attachment_type="image",
                        name="sample.png",
                        source_path="/tmp/sample.png",
                    ),
                ],
            ),
        ],
        max_tokens=16,
        temperature=0.0,
    )

    asyncio.run(runtime.load_model(manifest))
    response = asyncio.run(runtime.generate(request))
    snapshot = runtime.performance_feature_snapshot()

    assert captured["compiled_calls"] == 1
    assert captured["prepare_inputs"] == {
        "processor": fake_processor,
        "images": str(Path("/tmp/sample.png").resolve(strict=False)),
        "prompts": "user: describe this image\nassistant:",
        "image_token_index": 42,
        "add_special_tokens": True,
    }
    assert captured["compiled_kwargs"] == {
        "prompt": "user: describe this image\nassistant:",
        "image": str(Path("/tmp/sample.png").resolve(strict=False)),
        "input_ids": "prepared-input-ids",
        "pixel_values": "prepared-pixel-values",
        "mask": "prepared-mask",
        "image_sizes": "prepared-image-sizes",
        "max_tokens": 16,
        "sdpa_kernel": "custom_sdpa",
        "temperature": 0.0,
        "verbose": False,
    }
    assert captured["generate"] == {
        "model": fake_model,
        "processor": fake_processor,
        "prompt": "user: describe this image\nassistant:",
        "image": str(Path("/tmp/sample.png").resolve(strict=False)),
        "input_ids": "prepared-input-ids",
        "pixel_values": "prepared-pixel-values",
        "mask": "prepared-mask",
        "image_sizes": "prepared-image-sizes",
        "max_tokens": 16,
        "sdpa_kernel": "custom_sdpa",
        "vision_cache": None,
        "verbose": False,
    }
    assert response.output_text == "accelerated vision output"
    assert snapshot["graph_compilation"]["active"] is True
    assert snapshot["attention_kernel_acceleration"]["active"] is True
    assert snapshot["attention_kernel_acceleration"]["metrics"]["custom_sdpa_requests"] == 1
    assert request.metadata["mlx_acceleration"]["compile_state"] == "decode"
    assert "model" not in captured["compiled_kwargs"]
    assert "processor" not in captured["compiled_kwargs"]


def test_mlx_vision_runtime_keeps_streaming_on_stock_path_even_when_graph_compile_is_enabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"compiled_calls": 0}

    def fake_load(*, path_or_hf_repo: str, **kwargs):
        return "vision-model", "vision-processor"

    def fake_stream_generate(*, model, processor, prompt, image=None, verbose=False, **kwargs):
        captured["stream_generate"] = {
            "model": model,
            "processor": processor,
            "prompt": prompt,
            "image": image,
            "verbose": verbose,
            "kwargs": kwargs,
        }
        return [SimpleNamespace(text="first"), SimpleNamespace(text=" second")]

    def fake_compile(callable_obj):
        captured["compiled_calls"] = int(captured["compiled_calls"]) + 1
        return callable_obj

    def fake_import(name: str):
        if name == "mlx_vlm":
            return SimpleNamespace(load=fake_load, generate=lambda **kwargs: None, stream_generate=fake_stream_generate)
        if name == "mlx.core":
            return SimpleNamespace(compile=fake_compile)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_vision.runtime.import_module", fake_import)

    runtime = MLXVisionRuntime(settings=LewLMSettings(data_dir=tmp_path / "state", mlx_graph_compile_enabled=True))
    manifest = _manifest()
    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[GenerateMessage(role="user", content="stream this")],
        max_tokens=16,
        temperature=0.0,
    )

    asyncio.run(runtime.load_model(manifest))

    async def collect() -> list[str]:
        return [chunk async for chunk in runtime.stream_generate(request)]

    chunks = asyncio.run(collect())

    assert chunks == ["first", " second"]
    assert captured["compiled_calls"] == 0
    assert request.metadata["mlx_acceleration"]["phase_details"]["stream"]["effective_graph_compile"] is False


def test_mlx_vision_runtime_disables_failed_graph_compile_after_first_fallback(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, int] = {"compiled_calls": 0, "stock_calls": 0}
    fake_model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4", image_token_index=42))
    fake_processor = SimpleNamespace(chat_template=None)

    def fake_load(*, path_or_hf_repo: str, **kwargs):
        return fake_model, fake_processor

    def fake_prepare_inputs(
        processor,
        images=None,
        prompts=None,
        image_token_index=None,
        add_special_tokens=False,
        **kwargs,
    ):
        return {
            "input_ids": "prepared-input-ids",
            "pixel_values": "prepared-pixel-values",
            "attention_mask": "prepared-mask",
        }

    def fake_generate(
        *,
        model,
        processor,
        prompt,
        image=None,
        input_ids=None,
        pixel_values=None,
        mask=None,
        max_tokens: int,
        verbose: bool = False,
        **kwargs,
    ):
        captured["stock_calls"] = int(captured["stock_calls"]) + 1
        return SimpleNamespace(text="fallback vision output")

    def fake_compile(callable_obj):
        def compiled(**kwargs):
            captured["compiled_calls"] = int(captured["compiled_calls"]) + 1
            raise ValueError("processor objects are not compile-safe")

        return compiled

    def fake_import(name: str):
        if name == "mlx_vlm":
            return SimpleNamespace(load=fake_load, generate=fake_generate, stream_generate=lambda **kwargs: [])
        if name == "mlx_vlm.utils":
            return SimpleNamespace(prepare_inputs=fake_prepare_inputs)
        if name == "mlx.core":
            return SimpleNamespace(compile=fake_compile)
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.mlx_vision.runtime.import_module", fake_import)

    runtime = MLXVisionRuntime(
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            mlx_graph_compile_enabled=True,
        ),
    )
    manifest = _manifest()
    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[
            GenerateMessage(
                role="user",
                content="describe this image",
                attachments=[
                    GenerateAttachment(
                        attachment_type="image",
                        name="sample.png",
                        source_path="/tmp/sample.png",
                    ),
                ],
            ),
        ],
        max_tokens=16,
        temperature=0.0,
    )

    asyncio.run(runtime.load_model(manifest))
    first = asyncio.run(runtime.generate(request))
    second_request = request.model_copy(deep=True)
    second = asyncio.run(runtime.generate(second_request))
    snapshot = runtime.performance_feature_snapshot()

    assert first.output_text == "fallback vision output"
    assert second.output_text == "fallback vision output"
    assert captured["compiled_calls"] == 1
    assert captured["stock_calls"] == 2
    assert snapshot["graph_compilation"]["active"] is False
    assert snapshot["graph_compilation"]["metrics"]["compile_attempts"] == 1
    assert snapshot["graph_compilation"]["metrics"]["compile_failures"] == 1
    assert snapshot["graph_compilation"]["metrics"]["compile_fallback_requests"] == 1
    assert request.metadata["mlx_acceleration"]["fallback_reason"] == "ValueError: processor objects are not compile-safe"
    assert second_request.metadata["mlx_acceleration"]["fallback_reason"] == "ValueError: processor objects are not compile-safe"


def test_mlx_vision_runtime_reuses_encoder_features_across_identical_images_at_different_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"encode_calls": 0, "features": []}

    def fake_load(*, path_or_hf_repo: str, **kwargs):
        return "vision-model", "vision-processor"

    def fake_generate(*, model, processor, prompt, image=None, vision_cache=None, verbose=False, **kwargs):
        feature = vision_cache.get(image) if vision_cache is not None else None
        if feature is None:
            captured["encode_calls"] = int(captured["encode_calls"]) + 1
            feature = {"digest": Path(image).read_bytes().hex()}
            if vision_cache is not None:
                vision_cache.put(image, feature)
        cast_features = captured["features"]
        assert isinstance(cast_features, list)
        cast_features.append(feature)
        return SimpleNamespace(text="vision output")

    fake_module = SimpleNamespace(
        load=fake_load,
        generate=fake_generate,
        stream_generate=lambda **kwargs: [],
    )
    monkeypatch.setattr("lewlm.runtime.mlx_vision.runtime.import_module", lambda name: fake_module)

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    metadata_store = MetadataStore(state_dir / "metadata.sqlite3")
    metadata_store.initialize()
    encoder_cache = MultimodalEncoderCache(
        block_disk_cache=BlockDiskCache(cache_root=state_dir, metadata_store=metadata_store),
    )
    runtime = MLXVisionRuntime(multimodal_encoder_cache=encoder_cache)
    manifest = _manifest()

    first_image = tmp_path / "sample-a.png"
    second_image = tmp_path / "sample-b.png"
    first_image.write_bytes(b"same-image")
    second_image.write_bytes(b"same-image")

    first_request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[
            GenerateMessage(
                role="user",
                content="describe image a",
                attachments=[
                    GenerateAttachment(
                        attachment_type="image",
                        name=first_image.name,
                        source_path=str(first_image),
                    ),
                ],
            ),
        ],
        max_tokens=16,
        temperature=0.0,
    )
    second_request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[
            GenerateMessage(
                role="user",
                content="describe image b",
                attachments=[
                    GenerateAttachment(
                        attachment_type="image",
                        name=second_image.name,
                        source_path=str(second_image),
                    ),
                ],
            ),
        ],
        max_tokens=16,
        temperature=0.0,
    )

    asyncio.run(runtime.load_model(manifest))
    asyncio.run(runtime.generate(first_request))
    asyncio.run(runtime.generate(second_request))

    assert captured["encode_calls"] == 1
    assert first_request.metadata["encoder_cache"]["cache_misses"] == 1
    assert second_request.metadata["encoder_cache"]["cache_hits"] == 1


def test_mlx_vision_runtime_uses_native_batch_generate_for_single_image_requests(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_load(*, path_or_hf_repo: str, **kwargs):
        return "vision-model", "vision-processor"

    def fake_generate(*, model, processor, prompt, image=None, verbose=False, **kwargs):
        return SimpleNamespace(text="single vision output")

    def fake_batch_generate(*, model, processor, prompts, images=None, max_tokens=None, verbose=False, **kwargs):
        captured["batch_generate"] = {
            "model": model,
            "processor": processor,
            "prompts": prompts,
            "images": images,
            "max_tokens": max_tokens,
            "verbose": verbose,
            "kwargs": kwargs,
        }
        return SimpleNamespace(texts=["batched output a", "batched output b"])

    fake_module = SimpleNamespace(
        load=fake_load,
        generate=fake_generate,
        stream_generate=lambda **kwargs: [],
        batch_generate=fake_batch_generate,
    )
    monkeypatch.setattr("lewlm.runtime.mlx_vision.runtime.import_module", lambda name: fake_module)

    runtime = MLXVisionRuntime()
    manifest = _manifest()
    first_image = tmp_path / "batch-a.png"
    second_image = tmp_path / "batch-b.png"
    first_image.write_bytes(b"first-image")
    second_image.write_bytes(b"second-image")
    requests = [
        GenerateRequest(
            model_id=manifest.model_id,
            messages=[
                GenerateMessage(
                    role="user",
                    content="describe image a",
                    attachments=[
                        GenerateAttachment(
                            attachment_type="image",
                            name=first_image.name,
                            source_path=str(first_image),
                        ),
                    ],
                ),
            ],
            max_tokens=24,
            temperature=0.0,
        ),
        GenerateRequest(
            model_id=manifest.model_id,
            messages=[
                GenerateMessage(
                    role="user",
                    content="describe image b",
                    attachments=[
                        GenerateAttachment(
                            attachment_type="image",
                            name=second_image.name,
                            source_path=str(second_image),
                        ),
                    ],
                ),
            ],
            max_tokens=24,
            temperature=0.0,
        ),
    ]

    asyncio.run(runtime.load_model(manifest))
    responses = asyncio.run(runtime.generate_batch(requests))
    snapshot = runtime.performance_feature_snapshot()

    assert [response.output_text for response in responses] == ["batched output a", "batched output b"]
    batch_payload = captured["batch_generate"]
    assert batch_payload["model"] == "vision-model"
    assert batch_payload["processor"] == "vision-processor"
    assert batch_payload["prompts"] == [
        "user: describe image a\nassistant:",
        "user: describe image b\nassistant:",
    ]
    assert batch_payload["images"] == [
        str(first_image.resolve(strict=False)),
        str(second_image.resolve(strict=False)),
    ]
    assert batch_payload["max_tokens"] == [24, 24]
    assert batch_payload["verbose"] is False
    assert isinstance(batch_payload["kwargs"], dict)
    assert "temperature" not in batch_payload["kwargs"]
    assert "temp" not in batch_payload["kwargs"]
    assert "sampler" not in batch_payload["kwargs"]
    assert batch_payload["kwargs"]["group_by_shape"] is True
    assert batch_payload["kwargs"]["track_image_sizes"] is False
    assert batch_payload["kwargs"]["prefill_step_size"] == runtime.settings.prefill_token_batch_size
    assert "prefill_batch_size" not in batch_payload["kwargs"]
    assert "completion_batch_size" not in batch_payload["kwargs"]
    assert snapshot["continuous_batching"]["supported"] is True
    assert snapshot["continuous_batching"]["active"] is True
    assert snapshot["continuous_batching"]["metrics"]["chat_batch_calls"] == 1
    assert snapshot["continuous_batching"]["metrics"]["batched_requests"] == 2
    assert snapshot["continuous_batching"]["metrics"]["max_batch_size"] == 2
    assert all(request.metadata["native_batching"]["active"] is True for request in requests)
    assert all(request.metadata["native_batching"]["backend"] == "mlx_vlm.batch_generate" for request in requests)


def test_mlx_vision_runtime_batch_generate_falls_back_for_frame_bundle_requests(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, int] = {"batch_calls": 0, "stock_calls": 0}

    def fake_load(*, path_or_hf_repo: str, **kwargs):
        return "vision-model", "vision-processor"

    def fake_generate(*, model, processor, prompt, image=None, verbose=False, **kwargs):
        captured["stock_calls"] += 1
        return SimpleNamespace(text=f"stock {captured['stock_calls']}")

    def fake_batch_generate(**kwargs):
        captured["batch_calls"] += 1
        return SimpleNamespace(texts=["unexpected"])

    fake_module = SimpleNamespace(
        load=fake_load,
        generate=fake_generate,
        stream_generate=lambda **kwargs: [],
        batch_generate=fake_batch_generate,
    )
    monkeypatch.setattr("lewlm.runtime.mlx_vision.runtime.import_module", lambda name: fake_module)

    runtime = MLXVisionRuntime()
    manifest = _manifest()
    first_bundle = tmp_path / "bundle-a"
    second_bundle = tmp_path / "bundle-b"
    first_bundle.mkdir()
    second_bundle.mkdir()
    (first_bundle / "frame-1.png").write_bytes(b"frame-a1")
    (first_bundle / "frame-2.png").write_bytes(b"frame-a2")
    (second_bundle / "frame-1.png").write_bytes(b"frame-b1")
    (second_bundle / "frame-2.png").write_bytes(b"frame-b2")
    requests = [
        GenerateRequest(
            model_id=manifest.model_id,
            messages=[
                GenerateMessage(
                    role="user",
                    content="describe video a",
                    attachments=[
                        GenerateAttachment(
                            attachment_type="image",
                            name=first_bundle.name,
                            source_path=str(first_bundle),
                        ),
                    ],
                ),
            ],
            max_tokens=24,
            temperature=0.0,
        ),
        GenerateRequest(
            model_id=manifest.model_id,
            messages=[
                GenerateMessage(
                    role="user",
                    content="describe video b",
                    attachments=[
                        GenerateAttachment(
                            attachment_type="image",
                            name=second_bundle.name,
                            source_path=str(second_bundle),
                        ),
                    ],
                ),
            ],
            max_tokens=24,
            temperature=0.0,
        ),
    ]

    asyncio.run(runtime.load_model(manifest))
    responses = asyncio.run(runtime.generate_batch(requests))
    snapshot = runtime.performance_feature_snapshot()

    assert [response.output_text for response in responses] == ["stock 1", "stock 2"]
    assert captured["batch_calls"] == 0
    assert captured["stock_calls"] == 2
    assert snapshot["continuous_batching"]["supported"] is True
    assert snapshot["continuous_batching"]["active"] is False
    assert snapshot["continuous_batching"]["metrics"]["stock_single_request_fallback_batches"] == 1
    assert snapshot["continuous_batching"]["metrics"]["stock_single_request_fallback_requests"] == 2
    assert all(request.metadata["native_batching"]["stock_single_request_path"] is True for request in requests)
    assert all(
        "frame bundles and multi-image prompts stay on the stock single-request path"
        in str(request.metadata["native_batching"]["fallback_reason"])
        for request in requests
    )


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_id="vision-model",
        display_name="vision-model",
        architecture_family="qwen-vl",
        modality=(ModelModality.VISION, ModelModality.MULTIMODAL),
        source_path="/tmp/vision-model",
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_VISION,),
        estimated_memory_mb=1024,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="vision-fingerprint",
        last_validation_result=ModelValidationResult(
            status=ValidationState.VALID,
            message="ok",
        ),
    )
