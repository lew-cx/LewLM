from __future__ import annotations

import asyncio
from pathlib import Path

from conftest import FakeLlamaCppRuntime, FakeMLXAudioRuntime, FakeMLXSemanticRuntime
from lewlm.core.contracts import (
    ArchitectureSubtype,
    CapabilityName,
    ConversionStatus,
    GenerateAttachment,
    GenerateMessage,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RequestModality,
    RoutingModalityPath,
    RuntimeCandidateReport,
    RuntimeAffinity,
    RuntimeReadinessState,
    RuntimeSupportPath,
    ValidationState,
)
from lewlm.routing.service import ModelRouter
from lewlm.runtime.catalog import RuntimeCatalog
from lewlm.runtime.experimental import FrontierExperimentalRuntime
from conftest import FakeMLXVisionRuntime
from lewlm.storage.metadata import MetadataStore


def test_model_router_prefers_context_fitting_candidate(
    temp_settings,
    monkeypatch,
) -> None:
    router = ModelRouter(
        model_registry=_StaticRegistry(
            [
                _manifest(
                    model_id="unknown-context",
                    display_name="aaa-unknown-context",
                    estimated_memory_mb=256,
                    context_length=None,
                ),
                _manifest(
                    model_id="known-long-context",
                    display_name="zzz-known-long-context",
                    estimated_memory_mb=512,
                    context_length=32_768,
                    format_type=ModelFormat.MLX,
                    runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
                ),
            ],
        ),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
                RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            },
        ),
        settings=temp_settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 8_192)

    manifest, runtime, decision = router.route_chat(
        None,
        messages=[GenerateMessage(role="user", content="x" * 100_000)],
        max_tokens=1_000,
    )

    assert manifest.model_id == "known-long-context"
    assert runtime.name == "fake_mlx_semantic"
    assert "context fit" in decision.reason
    assert any("context length is unknown" in item for item in decision.alternatives)


def test_model_router_respects_memory_budget_for_automatic_selection(temp_settings, monkeypatch) -> None:
    runtime = FakeLlamaCppRuntime()
    settings = temp_settings.with_updates(runtime_policy="aggressive_unload")
    router = ModelRouter(
        model_registry=_StaticRegistry(
            [
                _manifest(
                    model_id="large-model",
                    display_name="aaa-large-model",
                    estimated_memory_mb=1_500,
                ),
                _manifest(
                    model_id="small-model",
                    display_name="zzz-small-model",
                    estimated_memory_mb=256,
                ),
            ],
        ),
        runtime_catalog=RuntimeCatalog({RuntimeAffinity.LLAMACPP: runtime}),
        settings=settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 2_048)

    manifest, _, decision = router.route_chat(
        None,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=64,
    )

    assert manifest.model_id == "small-model"
    assert any("large-model" in item and "routing budget" in item for item in decision.alternatives)


def test_model_router_prefers_loaded_model_under_keep_warm_policy(temp_settings, monkeypatch) -> None:
    runtime = FakeLlamaCppRuntime()
    settings = temp_settings.with_updates(runtime_policy="keep_warm")
    warm_model = _manifest(model_id="warm-model", display_name="zzz-warm-model", estimated_memory_mb=512)
    cold_model = _manifest(model_id="cold-model", display_name="aaa-cold-model", estimated_memory_mb=512)
    router = ModelRouter(
        model_registry=_StaticRegistry([cold_model, warm_model]),
        runtime_catalog=RuntimeCatalog({RuntimeAffinity.LLAMACPP: runtime}),
        settings=settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 8_192)
    asyncio.run(runtime.load_model(warm_model))

    manifest, _, decision = router.route_chat(
        None,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=64,
    )

    assert manifest.model_id == "warm-model"
    assert "already loaded" in decision.reason


def test_model_router_reports_frontier_bounded_memory_plan(temp_settings, monkeypatch) -> None:
    settings = temp_settings.with_updates(
        moe_bounded_memory_mode="expert_streaming",
        moe_resident_expert_count=8,
    )
    runtime = FakeMLXSemanticRuntime(settings=settings)
    frontier_manifest = _manifest(
        model_id="frontier-moe-model",
        display_name="frontier-moe-model",
        estimated_memory_mb=24_576,
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.EXPERIMENTAL, RuntimeAffinity.MLX_TEXT),
    ).model_copy(
        update={
            "architecture_family": "mixtral",
            "architecture_subtype": ArchitectureSubtype.MOE,
            "metadata": {
                "expert_count": 64,
                "active_expert_count": 8,
                "expert_routing_type": "top-8",
                "cache_state_handling": "expert_resident_window",
            },
        },
    )
    router = ModelRouter(
        model_registry=_StaticRegistry([frontier_manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.EXPERIMENTAL: FrontierExperimentalRuntime(settings=settings),
                RuntimeAffinity.MLX_TEXT: runtime,
            },
        ),
        settings=settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 24_576)

    manifest, selected_runtime, decision = router.route_chat(
        frontier_manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=64,
    )
    capability_report = router.model_capability_report(frontier_manifest.model_id)

    assert manifest.model_id == frontier_manifest.model_id
    assert selected_runtime.affinity == RuntimeAffinity.MLX_TEXT
    assert "expert_streaming" in decision.reason
    assert "resident experts" in decision.reason
    assert "10602/13516 mb" in decision.reason
    assert capability_report.capabilities[0].supported is True
    assert any("bounded-memory planning" in note for note in capability_report.capabilities[0].notes)


def test_model_router_keeps_first_class_runtime_over_benchmark_selected_external_adapter(
    temp_settings,
    monkeypatch,
    tmp_path: Path,
) -> None:
    native_runtime = FakeMLXSemanticRuntime(settings=temp_settings)
    external_runtime = _FakeExternalAcceleratorRuntime(settings=temp_settings)
    metadata_store = MetadataStore(tmp_path / "metadata.sqlite3")
    metadata_store.initialize()
    manifest = _manifest(
        model_id="mlx-demo",
        display_name="mlx-demo",
        estimated_memory_mb=768,
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
    )
    host_platform = RuntimeCatalog.host_platform_snapshot().model_dump(mode="json")
    metadata_store.upsert_runtime_preference(
        model_id=manifest.model_id,
        capability="chat",
        host_platform=host_platform,
        payload={
            "selected_runtime_affinity": RuntimeAffinity.EXTERNAL_ACCELERATOR.value,
            "selected_runtime_name": external_runtime.name,
            "baseline_runtime_affinity": RuntimeAffinity.MLX_TEXT.value,
            "baseline_runtime_name": native_runtime.name,
            "primary_metric": "warm_total_seconds",
            "selected_metric_value": 0.4,
            "baseline_metric_value": 0.9,
            "source": "compare_external_adapter",
            "feature_preservation": {
                "preserved": ["continuous_batching"],
                "degraded": [],
                "rejected": [],
            },
        },
    )
    router = ModelRouter(
        model_registry=_StaticRegistry([manifest], metadata_store=metadata_store),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.MLX_TEXT: native_runtime,
                RuntimeAffinity.EXTERNAL_ACCELERATOR: external_runtime,
            },
        ),
        settings=temp_settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 16_384)

    selected_manifest, selected_runtime, decision = router.route_chat(
        manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello accelerator")],
        max_tokens=32,
    )
    capability_report = router.model_capability_report(manifest.model_id)

    assert selected_manifest.model_id == manifest.model_id
    assert selected_runtime.affinity == RuntimeAffinity.MLX_TEXT
    assert "measured downgrade kept `fake_mlx_semantic`" in decision.reason
    assert capability_report.capabilities[0].runtime_affinity == RuntimeAffinity.MLX_TEXT
    assert "productized default" in " ".join(capability_report.capabilities[0].notes)


def test_model_router_downgrades_partial_support_external_adapter_preference(
    temp_settings,
    monkeypatch,
    tmp_path: Path,
) -> None:
    native_runtime = FakeMLXSemanticRuntime(settings=temp_settings)
    external_runtime = _FakeExternalAcceleratorRuntime(settings=temp_settings)
    metadata_store = MetadataStore(tmp_path / "metadata.sqlite3")
    metadata_store.initialize()
    manifest = _manifest(
        model_id="mlx-demo",
        display_name="mlx-demo",
        estimated_memory_mb=768,
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
    )
    host_platform = RuntimeCatalog.host_platform_snapshot().model_dump(mode="json")
    metadata_store.upsert_runtime_preference(
        model_id=manifest.model_id,
        capability="chat",
        host_platform=host_platform,
        payload={
            "selected_runtime_affinity": RuntimeAffinity.EXTERNAL_ACCELERATOR.value,
            "selected_runtime_name": external_runtime.name,
            "baseline_runtime_affinity": RuntimeAffinity.MLX_TEXT.value,
            "baseline_runtime_name": native_runtime.name,
            "primary_metric": "warm_total_seconds",
            "selected_metric_value": 0.4,
            "baseline_metric_value": 0.9,
            "winner": "external",
            "source": "compare_external_adapter",
            "feature_preservation": {
                "preserved": ["continuous_batching"],
                "degraded": ["kv_cache_quantization"],
                "rejected": [],
            },
        },
    )
    router = ModelRouter(
        model_registry=_StaticRegistry([manifest], metadata_store=metadata_store),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.MLX_TEXT: native_runtime,
                RuntimeAffinity.EXTERNAL_ACCELERATOR: external_runtime,
            },
        ),
        settings=temp_settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 16_384)

    selected_manifest, selected_runtime, decision = router.route_chat(
        manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello accelerator")],
        max_tokens=32,
    )
    capability_report = router.model_capability_report(manifest.model_id)

    assert selected_manifest.model_id == manifest.model_id
    assert selected_runtime.affinity == RuntimeAffinity.MLX_TEXT
    assert "measured downgrade kept `fake_mlx_semantic`" in decision.reason
    assert capability_report.capabilities[0].runtime_affinity == RuntimeAffinity.MLX_TEXT
    assert "downgraded an adapter-backed path" in capability_report.capabilities[0].reason
    assert any("partial feature coverage" in note for note in capability_report.capabilities[0].notes)


def test_model_router_can_route_gguf_text_models_through_external_adapter(temp_settings, monkeypatch) -> None:
    manifest = _manifest(
        model_id="gguf-demo",
        display_name="gguf-demo",
        estimated_memory_mb=512,
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
    )
    router = ModelRouter(
        model_registry=_StaticRegistry([manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.EXTERNAL_ACCELERATOR: _FakeExternalTextBridgeRuntime(),
            },
        ),
        settings=temp_settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 16_384)

    selected_manifest, selected_runtime, decision = router.route_chat(
        manifest.model_id,
        messages=[GenerateMessage(role="user", content="hello accelerator")],
        max_tokens=32,
    )
    capability_report = router.model_capability_report(manifest.model_id)
    readiness = router.capability_readiness(CapabilityName.CHAT)

    assert selected_manifest.model_id == manifest.model_id
    assert selected_runtime.affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert decision.runtime_affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert capability_report.capabilities[0].runtime_affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert any(
        candidate.runtime_affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR and candidate.supports_manifest
        for candidate in capability_report.runtime_candidates
    )
    assert readiness.ready is True
    assert readiness.available_runtime_names == ["local_external_adapter"]


def test_model_router_can_route_semantic_models_through_external_adapter(temp_settings) -> None:
    embedding_manifest = _manifest(
        model_id="embed-demo",
        display_name="embed-demo",
        estimated_memory_mb=256,
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
    ).model_copy(
        update={
            "architecture_family": "e5",
            "modality": (ModelModality.EMBEDDING,),
            "source_path": "/tmp/embed-demo",
        },
    )
    rerank_manifest = _manifest(
        model_id="rerank-demo",
        display_name="rerank-demo",
        estimated_memory_mb=256,
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
    ).model_copy(
        update={
            "architecture_family": "bge",
            "modality": (ModelModality.RERANK,),
            "source_path": "/tmp/rerank-demo",
        },
    )

    class _FakeExternalSemanticBridgeRuntime(FakeMLXSemanticRuntime):
        name = "local_external_adapter"
        affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR
        supported_formats = (ModelFormat.MLX, ModelFormat.GGUF)

    router = ModelRouter(
        model_registry=_StaticRegistry([embedding_manifest, rerank_manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.EXTERNAL_ACCELERATOR: _FakeExternalSemanticBridgeRuntime(),
            },
        ),
        settings=temp_settings,
    )

    selected_embedding_manifest, embedding_runtime, embedding_decision = router.route_embeddings(
        embedding_manifest.model_id,
        inputs=["semantic bridge"],
    )
    selected_rerank_manifest, rerank_runtime, rerank_decision = router.route_rerank(
        rerank_manifest.model_id,
        query="semantic bridge",
        documents=["one", "two"],
    )
    embedding_readiness = router.capability_readiness(CapabilityName.EMBEDDINGS)
    rerank_readiness = router.capability_readiness(CapabilityName.RERANK)

    assert selected_embedding_manifest.model_id == embedding_manifest.model_id
    assert embedding_runtime.affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert embedding_decision.runtime_affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert selected_rerank_manifest.model_id == rerank_manifest.model_id
    assert rerank_runtime.affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert rerank_decision.runtime_affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert embedding_readiness.ready is True
    assert embedding_readiness.available_runtime_names == ["local_external_adapter"]
    assert rerank_readiness.ready is True
    assert rerank_readiness.available_runtime_names == ["local_external_adapter"]


def test_model_router_prefers_packaged_semantic_gguf_path_and_reports_bridge_alternative(temp_settings) -> None:
    embedding_manifest = _manifest(
        model_id="gguf-embed-demo",
        display_name="gguf-embed-demo",
        estimated_memory_mb=256,
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
    ).model_copy(
        update={
            "architecture_family": "bge",
            "modality": (ModelModality.EMBEDDING,),
            "source_path": "X:\\models\\gguf-embed-demo.gguf",
        },
    )
    rerank_manifest = _manifest(
        model_id="gguf-rerank-demo",
        display_name="gguf-rerank-demo",
        estimated_memory_mb=256,
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
    ).model_copy(
        update={
            "architecture_family": "bge",
            "modality": (ModelModality.RERANK,),
            "source_path": "X:\\models\\gguf-rerank-demo.gguf",
        },
    )

    class _FakePackagedSemanticGGUFRuntime(FakeMLXSemanticRuntime):
        name = "fake_llamacpp_semantic"
        affinity = RuntimeAffinity.LLAMACPP
        supported_formats = (ModelFormat.GGUF,)
        supported_modalities = (ModelModality.TEXT, ModelModality.EMBEDDING, ModelModality.RERANK)

    class _FakeExternalSemanticBridgeRuntime(_FakePackagedSemanticGGUFRuntime):
        name = "local_external_adapter"
        affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR

    router = ModelRouter(
        model_registry=_StaticRegistry([embedding_manifest, rerank_manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.LLAMACPP: _FakePackagedSemanticGGUFRuntime(settings=temp_settings),
                RuntimeAffinity.EXTERNAL_ACCELERATOR: _FakeExternalSemanticBridgeRuntime(settings=temp_settings),
            },
        ),
        settings=temp_settings,
    )

    _, embedding_runtime, embedding_decision = router.route_embeddings(
        embedding_manifest.model_id,
        inputs=["semantic packaged"],
    )
    _, rerank_runtime, rerank_decision = router.route_rerank(
        rerank_manifest.model_id,
        query="semantic packaged",
        documents=["one", "two"],
    )
    embedding_readiness = router.capability_readiness(CapabilityName.EMBEDDINGS)
    rerank_readiness = router.capability_readiness(CapabilityName.RERANK)

    assert embedding_runtime.affinity == RuntimeAffinity.LLAMACPP
    assert embedding_decision.support_path == RuntimeSupportPath.PACKAGED
    assert rerank_runtime.affinity == RuntimeAffinity.LLAMACPP
    assert rerank_decision.support_path == RuntimeSupportPath.PACKAGED
    assert embedding_readiness.bridge_only is False
    assert embedding_readiness.available_support_paths == [RuntimeSupportPath.PACKAGED, RuntimeSupportPath.BRIDGE]
    assert embedding_readiness.packaged_runtime_names == ["fake_llamacpp_semantic"]
    assert embedding_readiness.bridge_runtime_names == ["local_external_adapter"]
    assert rerank_readiness.bridge_only is False
    assert rerank_readiness.available_support_paths == [RuntimeSupportPath.PACKAGED, RuntimeSupportPath.BRIDGE]


def test_model_router_can_route_audio_models_through_external_adapter(temp_settings) -> None:
    manifest = _manifest(
        model_id="audio-demo",
        display_name="audio-demo",
        estimated_memory_mb=256,
        format_type=ModelFormat.AUDIO_FOLDER,
        runtime_affinity=(RuntimeAffinity.MLX_AUDIO,),
    ).model_copy(
        update={
            "architecture_family": "whisper",
            "modality": (ModelModality.AUDIO,),
            "source_path": "/tmp/audio-demo",
        },
    )
    router = ModelRouter(
        model_registry=_StaticRegistry([manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.EXTERNAL_ACCELERATOR: _FakeExternalAudioBridgeRuntime(),
            },
        ),
        settings=temp_settings,
    )

    selected_manifest, transcription_runtime, transcription_decision = router.route_audio_transcription(manifest.model_id)
    _, speech_runtime, speech_decision = router.route_audio_speech(manifest.model_id)
    capability_report = router.model_capability_report(manifest.model_id)
    transcription_readiness = router.capability_readiness(CapabilityName.AUDIO_TRANSCRIPTION)
    speech_readiness = router.capability_readiness(CapabilityName.AUDIO_SPEECH)
    capability_by_name = {item.capability: item for item in capability_report.capabilities}

    assert selected_manifest.model_id == manifest.model_id
    assert transcription_runtime.affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert transcription_decision.support_path == RuntimeSupportPath.BRIDGE
    assert speech_runtime.affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert speech_decision.support_path == RuntimeSupportPath.BRIDGE
    assert capability_by_name[CapabilityName.AUDIO_TRANSCRIPTION].support_path == RuntimeSupportPath.BRIDGE
    assert capability_by_name[CapabilityName.AUDIO_SPEECH].support_path == RuntimeSupportPath.BRIDGE
    assert transcription_readiness.ready is True
    assert transcription_readiness.bridge_only is True
    assert transcription_readiness.available_support_paths == [RuntimeSupportPath.BRIDGE]
    assert transcription_readiness.bridge_runtime_names == ["local_external_adapter"]
    assert "bridge-backed external audio path" in transcription_readiness.reason
    assert any("/v1/audio/transcriptions" in note for note in transcription_readiness.notes)
    assert speech_readiness.ready is True
    assert speech_readiness.bridge_only is True
    assert "bridge-backed external audio path" in speech_readiness.reason
    assert any("/v1/audio/speech" in note for note in speech_readiness.notes)


def test_model_router_prefers_decode_time_structured_output_path_when_auto_selecting(
    temp_settings,
    monkeypatch,
) -> None:
    router = ModelRouter(
        model_registry=_StaticRegistry(
            [
                _manifest(
                    model_id="mlx-chat",
                    display_name="aaa-mlx-chat",
                    estimated_memory_mb=512,
                    format_type=ModelFormat.MLX,
                    runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
                ),
                _manifest(
                    model_id="gguf-chat",
                    display_name="zzz-gguf-chat",
                    estimated_memory_mb=512,
                    format_type=ModelFormat.GGUF,
                    runtime_affinity=(RuntimeAffinity.LLAMACPP,),
                ),
            ],
        ),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(settings=temp_settings),
                RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
            },
        ),
        settings=temp_settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 16_384)

    plain_manifest, plain_runtime, _ = router.route_chat(
        None,
        messages=[GenerateMessage(role="user", content="hello")],
        max_tokens=32,
    )
    structured_manifest, structured_runtime, structured_decision = router.route_chat(
        None,
        messages=[GenerateMessage(role="user", content="return JSON")],
        max_tokens=32,
        structured_output_requested=True,
    )
    explicit_manifest, explicit_runtime, _ = router.route_chat(
        "mlx-chat",
        messages=[GenerateMessage(role="user", content="return JSON")],
        max_tokens=32,
        structured_output_requested=True,
    )

    assert plain_manifest.model_id == "mlx-chat"
    assert plain_runtime.affinity == RuntimeAffinity.MLX_TEXT
    assert structured_manifest.model_id == "gguf-chat"
    assert structured_runtime.affinity == RuntimeAffinity.LLAMACPP
    assert "decode-time structured output available" in structured_decision.reason
    assert explicit_manifest.model_id == "mlx-chat"
    assert explicit_runtime.affinity == RuntimeAffinity.MLX_TEXT


def test_model_router_reports_bridge_backed_vision_support_on_external_adapter(temp_settings, monkeypatch) -> None:
    manifest = _multimodal_manifest()
    router = ModelRouter(
        model_registry=_StaticRegistry([manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.EXTERNAL_ACCELERATOR: _FakeExternalVisionBridgeRuntime(),
            },
        ),
        settings=temp_settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 16_384)

    _, runtime, decision = router.route_chat(
        manifest.model_id,
        messages=[
            GenerateMessage(
                role="user",
                content="Describe this image",
                attachments=[GenerateAttachment(attachment_type="image", name="sample.png", source_path="/tmp/sample.png")],
            ),
        ],
        max_tokens=32,
    )
    capability_report = router.model_capability_report(manifest.model_id)
    vision_capability = next(item for item in capability_report.capabilities if item.capability == CapabilityName.VISION)
    readiness = router.capability_readiness(CapabilityName.VISION)

    assert runtime.affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert decision.request_modality == RequestModality.IMAGE_CONDITIONED
    assert decision.support_path == RuntimeSupportPath.BRIDGE
    assert vision_capability.runtime_affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert vision_capability.support_path == RuntimeSupportPath.BRIDGE
    assert "bridge-backed" in vision_capability.reason
    assert readiness.bridge_only is True
    assert readiness.available_support_paths == [RuntimeSupportPath.BRIDGE]
    assert "bridge-backed" in readiness.reason


def test_model_router_readiness_surfaces_external_adapter_model_mismatch_reason(temp_settings) -> None:
    manifest = _manifest(
        model_id="mlx-demo",
        display_name="mlx-demo",
        estimated_memory_mb=512,
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
    )
    router = ModelRouter(
        model_registry=_StaticRegistry([manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.EXTERNAL_ACCELERATOR: _MismatchedExternalAdapterRuntime(),
            },
        ),
        settings=temp_settings,
    )

    readiness = router.capability_readiness(CapabilityName.CHAT)
    capability_report = router.model_capability_report(manifest.model_id)

    assert readiness.ready is False
    assert readiness.available_runtime_names == []
    assert any("did not advertise a compatible local model id" in note for note in readiness.notes)
    assert any(
        "did not advertise a compatible local model id" in item
        for item in capability_report.capabilities[0].alternatives
    )


def test_model_router_prefers_text_fast_path_for_text_only_multimodal_bundle(temp_settings, monkeypatch) -> None:
    manifest = _multimodal_manifest()
    router = ModelRouter(
        model_registry=_StaticRegistry([manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(settings=temp_settings),
                RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
            },
        ),
        settings=temp_settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 16_384)

    selected_manifest, selected_runtime, decision = router.route_chat(
        manifest.model_id,
        messages=[GenerateMessage(role="user", content="Route me through text only")],
        max_tokens=32,
    )

    assert selected_manifest.model_id == manifest.model_id
    assert selected_runtime.affinity == RuntimeAffinity.MLX_TEXT
    assert decision.request_modality == RequestModality.TEXT_ONLY
    assert decision.modality_path == RoutingModalityPath.TEXT_FAST_PATH
    assert "Gemma-safe same-bundle text runtime" in str(decision.modality_path_reason)


def test_model_router_falls_back_to_multimodal_runtime_when_text_fast_path_cannot_load_manifest(
    temp_settings,
    monkeypatch,
) -> None:
    class UnsupportedGemmaTextRuntime(FakeMLXSemanticRuntime):
        def supports_manifest(self, manifest: ModelManifest) -> bool:
            return False

    manifest = _multimodal_manifest()
    router = ModelRouter(
        model_registry=_StaticRegistry([manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.MLX_TEXT: UnsupportedGemmaTextRuntime(settings=temp_settings),
                RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
            },
        ),
        settings=temp_settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 16_384)

    selected_manifest, selected_runtime, decision = router.route_chat(
        manifest.model_id,
        messages=[GenerateMessage(role="user", content="Route me through text only")],
        max_tokens=32,
    )

    assert selected_manifest.model_id == manifest.model_id
    assert selected_runtime.affinity == RuntimeAffinity.MLX_VISION
    assert decision.request_modality == RequestModality.TEXT_ONLY
    assert decision.modality_path == RoutingModalityPath.MULTIMODAL_DEFAULT
    assert "No safe text-only runtime" in str(decision.modality_path_reason)


def test_model_router_keeps_multimodal_runtime_for_image_conditioned_bundle(temp_settings, monkeypatch) -> None:
    manifest = _multimodal_manifest()
    router = ModelRouter(
        model_registry=_StaticRegistry([manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(settings=temp_settings),
                RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
            },
        ),
        settings=temp_settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 16_384)

    _, selected_runtime, decision = router.route_chat(
        manifest.model_id,
        messages=[
            GenerateMessage(
                role="user",
                content="Describe this image",
                attachments=[GenerateAttachment(attachment_type="image", name="sample.png", source_path="/tmp/sample.png")],
            ),
        ],
        max_tokens=32,
    )

    assert selected_runtime.affinity == RuntimeAffinity.MLX_VISION
    assert decision.request_modality == RequestModality.IMAGE_CONDITIONED
    assert decision.modality_path == RoutingModalityPath.MULTIMODAL_DEFAULT
    assert "Image attachments require" in str(decision.modality_path_reason)


def test_model_router_classifies_frame_bundle_requests(temp_settings, monkeypatch) -> None:
    manifest = _multimodal_manifest()
    router = ModelRouter(
        model_registry=_StaticRegistry([manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(settings=temp_settings),
                RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
            },
        ),
        settings=temp_settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 16_384)

    _, selected_runtime, decision = router.route_chat(
        manifest.model_id,
        messages=[
            GenerateMessage(
                role="user",
                content="Summarize these frames",
                attachments=[
                    GenerateAttachment(
                        attachment_type="image",
                        name="frames",
                        source_path="/tmp/frames",
                        metadata={"source_type": "image_bundle"},
                    ),
                ],
            ),
        ],
        max_tokens=32,
    )

    assert selected_runtime.affinity == RuntimeAffinity.MLX_VISION
    assert decision.request_modality == RequestModality.FRAME_BUNDLE_VIDEO
    assert decision.modality_path == RoutingModalityPath.MULTIMODAL_DEFAULT
    assert "Frame-bundle" in str(decision.modality_path_reason)


def test_model_router_classifies_audio_conditioned_requests(temp_settings, monkeypatch) -> None:
    manifest = _multimodal_manifest()
    router = ModelRouter(
        model_registry=_StaticRegistry([manifest]),
        runtime_catalog=RuntimeCatalog(
            {
                RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(settings=temp_settings),
                RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
            },
        ),
        settings=temp_settings,
    )
    monkeypatch.setattr(router, "_host_memory_mb", lambda: 16_384)

    _, selected_runtime, decision = router.route_chat(
        manifest.model_id,
        messages=[
            GenerateMessage(
                role="user",
                content="Use this transcript",
                attachments=[GenerateAttachment(attachment_type="audio", name="clip.wav", source_path="/tmp/clip.wav")],
            ),
        ],
        max_tokens=32,
    )

    assert selected_runtime.affinity == RuntimeAffinity.MLX_VISION
    assert decision.request_modality == RequestModality.AUDIO_CONDITIONED
    assert decision.modality_path == RoutingModalityPath.MULTIMODAL_DEFAULT
    assert "disable" in str(decision.modality_path_reason).casefold()


class _StaticRegistry:
    def __init__(self, manifests: list[ModelManifest], *, metadata_store: MetadataStore | None = None) -> None:
        self._manifests = {manifest.model_id: manifest for manifest in manifests}
        self._ordered = list(manifests)
        self.metadata_store = metadata_store or _EmptyPreferenceStore()

    def list_manifests(self) -> list[ModelManifest]:
        return list(self._ordered)

    def get_manifest(self, model_id: str) -> ModelManifest:
        return self._manifests[model_id]


class _EmptyPreferenceStore:
    def get_runtime_preference(self, *, model_id: str, capability: str, host_platform: dict[str, object]) -> dict[str, object] | None:
        return None

    def upsert_capability_probe_record(
        self,
        *,
        category: str,
        probe_name: str,
        host_platform: dict[str, object],
        status: str,
        source: str,
        runtime_name: str,
        runtime_affinity: str,
        model_id: str | None = None,
        workload_class: str | None = None,
        reason: str | None = None,
        details: dict[str, object] | None = None,
        recorded_at=None,
    ) -> None:
        return None

    def list_capability_probe_records(
        self,
        *,
        limit: int = 100,
        host_platform: dict[str, object],
        model_id: str | None = None,
        runtime_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, object]]:
        return []


class _FakeExternalAcceleratorRuntime(FakeMLXSemanticRuntime):
    name = "fake_external_accelerator"
    affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR


class _FakeExternalTextBridgeRuntime(FakeLlamaCppRuntime):
    name = "local_external_adapter"
    affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR
    supported_formats = (ModelFormat.MLX, ModelFormat.GGUF)


class _FakeExternalAudioBridgeRuntime(FakeMLXAudioRuntime):
    name = "local_external_adapter"
    affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR
    supported_formats = (ModelFormat.MLX, ModelFormat.GGUF, ModelFormat.AUDIO_FOLDER)


class _FakeExternalVisionBridgeRuntime(FakeMLXVisionRuntime):
    name = "local_external_adapter"
    affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR
    supported_formats = (ModelFormat.MLX, ModelFormat.GGUF)


class _MismatchedExternalAdapterRuntime(_FakeExternalTextBridgeRuntime):
    def supports_manifest(self, manifest: ModelManifest) -> bool:
        return False

    def candidate_report(self, manifest: ModelManifest | None = None) -> RuntimeCandidateReport:
        return RuntimeCandidateReport(
            runtime_name=self.name,
            runtime_affinity=self.affinity,
            readiness_state=RuntimeReadinessState.MANIFEST_UNSUPPORTED,
            registered=True,
            available=True,
            availability_reason=(
                "The configured external accelerator endpoint did not advertise a compatible local model id. "
                "Available ids: ['server-demo']."
            ),
            host_platform_supported=True,
            supported_systems=["Darwin", "Linux", "Windows"],
            supported_machines=[],
            support_path=RuntimeSupportPath.BRIDGE,
            supports_manifest=False,
        )



def _manifest(
    *,
    model_id: str,
    display_name: str,
    estimated_memory_mb: int,
    context_length: int | None = 8_192,
    format_type: ModelFormat = ModelFormat.GGUF,
    runtime_affinity: tuple[RuntimeAffinity, ...] = (RuntimeAffinity.LLAMACPP,),
) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        display_name=display_name,
        architecture_family="llama",
        modality=(ModelModality.TEXT,),
        source_path=f"/tmp/{model_id}.gguf",
        format_type=format_type,
        runtime_affinity=runtime_affinity,
        estimated_memory_mb=estimated_memory_mb,
        context_length=context_length,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint=f"{model_id}-fingerprint",
        last_validation_result=ModelValidationResult(
            status=ValidationState.VALID,
            message="ok",
        ),
    )


def _multimodal_manifest() -> ModelManifest:
    return _manifest(
        model_id="gemma4-multimodal",
        display_name="Gemma4 multimodal",
        estimated_memory_mb=768,
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_VISION,),
    ).model_copy(
        update={
            "modality": (ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL),
            "text_only_runtime_affinity": (RuntimeAffinity.MLX_TEXT,),
            "text_only_runtime_source": "same_bundle",
            "text_only_runtime_reason": "Gemma-safe same-bundle text runtime.",
        },
    )
