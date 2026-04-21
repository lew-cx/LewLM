from __future__ import annotations

import asyncio
from pathlib import Path

from conftest import FakeLlamaCppRuntime, FakeMLXSemanticRuntime
from lewlm.core.contracts import (
    ArchitectureSubtype,
    ConversionStatus,
    GenerateAttachment,
    GenerateMessage,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RequestModality,
    RoutingModalityPath,
    RuntimeAffinity,
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


def test_model_router_prefers_benchmark_selected_external_adapter(temp_settings, monkeypatch, tmp_path: Path) -> None:
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
    assert selected_runtime.affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert "benchmark preferred" in decision.reason
    assert capability_report.capabilities[0].runtime_affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR
    assert "benchmark-backed local routing preference" in capability_report.capabilities[0].reason


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
    assert "downgraded a partial-support path" in capability_report.capabilities[0].reason
    assert any("partial feature coverage" in note for note in capability_report.capabilities[0].notes)


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
