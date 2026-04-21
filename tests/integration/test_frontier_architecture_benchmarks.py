from __future__ import annotations

import json

import pytest

from conftest import FakeMLXAudioRuntime, FakeMLXSemanticRuntime, FakeMLXVisionRuntime, emit_benchmark_case_report
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import GenerateMessage, RuntimeAffinity
from lewlm.telemetry.stats import PerformanceFeatureName


@pytest.mark.asyncio
@pytest.mark.long_running
async def test_benchmark_reports_frontier_architecture_modes(temp_settings: LewLMSettings) -> None:
    ssm_dir = temp_settings.models_dir[0] / "hybrid-mamba-mlx"
    ssm_dir.mkdir(parents=True)
    (ssm_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gateddeltanet",
                "d_state": 128,
                "num_attention_heads": 8,
                "max_position_embeddings": 8192,
            },
        ),
        encoding="utf-8",
    )
    (ssm_dir / "weights.safetensors").write_bytes(b"ssm-weights")
    (ssm_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    moe_dir = temp_settings.models_dir[0] / "giant-mixtral-mlx"
    moe_dir.mkdir(parents=True)
    (moe_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mixtral",
                "num_experts": 64,
                "experts_per_token": 8,
                "max_position_embeddings": 32768,
            },
        ),
        encoding="utf-8",
    )
    (moe_dir / "weights.safetensors").write_bytes(b"m" * (16 * 1024 * 1024))
    (moe_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    settings = temp_settings.with_updates(
        moe_bounded_memory_mode="expert_streaming",
        moe_resident_expert_count=8,
    )
    services = bootstrap_services(
        settings,
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(settings=settings),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
            RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
        },
    )
    try:
        manifests = services.model_registry.scan().manifests
        ssm_model_id = next(manifest.model_id for manifest in manifests if manifest.source_path == str(ssm_dir))
        moe_model_id = next(manifest.model_id for manifest in manifests if manifest.source_path == str(moe_dir))
        ssm_chat = await services.chat_orchestrator.complete(
            model_id=ssm_model_id,
            messages=[GenerateMessage(role="user", content="frontier ssm chat")],
            max_tokens=64,
            temperature=0.0,
        )
        moe_chat = await services.chat_orchestrator.complete(
            model_id=moe_model_id,
            messages=[GenerateMessage(role="user", content="frontier moe chat")],
            max_tokens=64,
            temperature=0.0,
        )

        ssm_result = await services.telemetry_service.benchmark(
            model_id=ssm_model_id,
            prompt="frontier benchmark",
            capability="chat",
        )
        moe_result = await services.telemetry_service.benchmark(
            model_id=moe_model_id,
            prompt="frontier benchmark",
            capability="chat",
        )
        runtime_stats = await services.telemetry_service.runtime_stats()
        emit_benchmark_case_report(
            label="frontier-hybrid-ssm-feature",
            payload=ssm_result.model_dump(mode="json"),
            feature_names=("hybrid_ssm_routing", "ssm_state_cache_handling"),
            scenario_names=("frontier_architecture_modes",),
        )
        emit_benchmark_case_report(
            label="frontier-moe-feature",
            payload=moe_result.model_dump(mode="json"),
            feature_names=("moe_bounded_memory_serving",),
            scenario_names=("frontier_architecture_modes",),
        )

        benchmark_features = {item.feature.value: item for item in moe_result.performance_features}
        runtime_features = {item.feature.value: item for item in runtime_stats.performance_features}
        runtime_snapshot = next(item for item in runtime_stats.runtimes if item["name"] == "fake_mlx_semantic")
        loaded_models = {item["model_id"]: item for item in runtime_snapshot["loaded_models"]}
        ssm_scenario = next(item for item in ssm_result.scenarios if item.scenario == "frontier_architecture_modes")
        moe_scenario = next(item for item in moe_result.scenarios if item.scenario == "frontier_architecture_modes")
        moe_sample = next(item for item in moe_scenario.samples if item.metrics["architecture_subtype"] == "moe")
        ssm_sample = next(item for item in ssm_scenario.samples if item.metrics["architecture_subtype"] == "hybrid_ssm")

        assert benchmark_features[PerformanceFeatureName.HYBRID_SSM_ROUTING.value].supported is True
        assert benchmark_features[PerformanceFeatureName.HYBRID_SSM_ROUTING.value].active is True
        assert benchmark_features[PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING.value].supported is True
        assert benchmark_features[PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING.value].active is True
        assert runtime_features[PerformanceFeatureName.SSM_STATE_CACHE_HANDLING.value].supported is True
        assert runtime_features[PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING.value].active is True
        assert ssm_chat.request_metadata["frontier_architecture"]["planning_only"] is False
        assert ssm_chat.request_metadata["frontier_architecture"]["state_cache_bytes"] > 0
        assert moe_chat.request_metadata["frontier_architecture"]["planning_only"] is False
        assert moe_chat.request_metadata["frontier_architecture"]["effective_loaded_memory_mb"] == moe_chat.request_metadata["frontier_architecture"]["planned_memory_mb"]
        assert ssm_result.measurements["frontier_state_cache_bytes"] > 0
        assert moe_result.measurements["frontier_effective_loaded_memory_mb"] == moe_result.measurements["frontier_planned_memory_mb"]
        assert ssm_scenario.feature == PerformanceFeatureName.HYBRID_SSM_ROUTING
        assert ssm_scenario.status == "observed"
        assert ssm_scenario.metrics["hybrid_ssm_model_count"] == 1
        assert moe_scenario.feature == PerformanceFeatureName.MOE_BOUNDED_MEMORY_SERVING
        assert moe_scenario.status == "observed"
        assert moe_scenario.metrics["moe_model_count"] == 1
        assert moe_sample.metrics["planned_memory_mb"] < moe_sample.metrics["full_estimated_memory_mb"]
        assert moe_sample.metrics["effective_loaded_memory_mb"] == moe_sample.metrics["planned_memory_mb"]
        assert moe_sample.metrics["expert_swap_count"] > 0
        assert moe_sample.metrics["resident_expert_count"] == 8
        assert loaded_models[moe_model_id]["estimated_memory_mb"] == moe_sample.metrics["planned_memory_mb"]
        assert ssm_sample.metrics["cache_state_handling"] == "hybrid_attention_state"
        assert ssm_sample.metrics["planning_only"] is False
        assert ssm_sample.metrics["state_cache_bytes"] > 0
    finally:
        await services.aclose()
