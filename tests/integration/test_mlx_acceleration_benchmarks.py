from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from conftest import emit_benchmark_case_report
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import (
    CapabilityName,
    ConversionStatus,
    GenerateMessage,
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.runtime.base import ManagedTextRuntime
from lewlm.telemetry.stats import PerformanceFeatureName


class AccelerationBenchmarkRuntime(ManagedTextRuntime):
    name = "fake_mlx_acceleration"
    affinity = RuntimeAffinity.MLX_TEXT
    supported_formats = (ModelFormat.MLX,)
    supported_modalities = (ModelModality.TEXT,)
    supported_capabilities = frozenset({CapabilityName.CHAT, CapabilityName.STREAMING})

    def __init__(self, *, settings: LewLMSettings) -> None:
        super().__init__()
        self.settings = settings
        self.compiled_requests = 0
        self.stock_requests = 0
        self.flash_attention_requests = 0
        self.kernel_fallback_requests = 0
        self.last_kernel_path = "stock"

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, None

    def performance_feature_snapshot(self) -> dict[str, object]:
        return {
            "graph_compilation": {
                "supported": True,
                "active": self.compiled_requests > 0,
                "reason": "Fake MLX runtime exposes graph compilation for benchmark proof.",
                "metrics": {
                    "configured_enabled": self.settings.mlx_graph_compile_enabled,
                    "compile_attempts": self.compiled_requests,
                    "compiled_requests": self.compiled_requests,
                    "compile_fallback_requests": 0,
                    "compile_failures": 0,
                    "compiled_callable_count": 1 if self.compiled_requests else 0,
                },
                "notes": [],
            },
            "attention_kernel_acceleration": {
                "supported": True,
                "active": self.flash_attention_requests > 0,
                "reason": "Fake MLX runtime exposes accelerated attention hooks for benchmark proof.",
                "metrics": {
                    "configured_mode": self.settings.mlx_attention_kernel_mode,
                    "preferred_mode": "flash_attention",
                    "supported_modes": "flash_attention,custom_sdpa",
                    "kernel_parameter": "attention_kernel",
                    "stock_requests": self.stock_requests,
                    "flash_attention_requests": self.flash_attention_requests,
                    "custom_sdpa_requests": 0,
                    "kernel_fallback_requests": self.kernel_fallback_requests,
                    "last_kernel_path": self.last_kernel_path,
                },
                "notes": [],
            },
        }

    async def _load_model(self, manifest: ModelManifest) -> None:
        return None

    async def _unload_model(self, model_id: str) -> None:
        return None

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        overrides = request.metadata.get("mlx_acceleration", {})
        if not isinstance(overrides, dict):
            overrides = {}
        graph_compile = bool(overrides.get("graph_compile_enabled", self.settings.mlx_graph_compile_enabled))
        kernel_mode = str(overrides.get("attention_kernel_mode", self.settings.mlx_attention_kernel_mode))
        effective_kernel = kernel_mode if kernel_mode in {"flash_attention", "custom_sdpa"} else "stock"
        if graph_compile:
            self.compiled_requests += 1
        if effective_kernel == "flash_attention":
            self.flash_attention_requests += 1
        else:
            self.stock_requests += 1
        self.last_kernel_path = effective_kernel
        request.metadata["mlx_acceleration"] = {
            **overrides,
            "requested_graph_compile": graph_compile,
            "graph_compile_supported": True,
            "effective_graph_compile": graph_compile,
            "requested_kernel_mode": kernel_mode,
            "effective_kernel_path": effective_kernel,
            "attention_kernel_supported": True,
            "preferred_kernel_mode": "flash_attention",
            "kernel_parameter": "attention_kernel",
            "acceleration_fallback": False,
            "compile_state": "decode" if graph_compile else "stock",
            "phase_details": {
                "decode": {
                    "phase": "decode",
                    "requested_graph_compile": graph_compile,
                    "graph_compile_supported": True,
                    "effective_graph_compile": graph_compile,
                    "requested_kernel_mode": kernel_mode,
                    "effective_kernel_path": effective_kernel,
                    "attention_kernel_supported": True,
                    "preferred_kernel_mode": "flash_attention",
                    "kernel_parameter": "attention_kernel",
                    "acceleration_fallback": False,
                    "phase_compile_state": "compiled" if graph_compile else "stock",
                },
            },
        }
        await asyncio.sleep(0.002 if graph_compile or effective_kernel != "stock" else 0.03)
        output = f"Echo: {request.messages[-1].content}"
        return GenerateResponse(
            model_id=request.model_id,
            output_text=output,
            finish_reason="stop",
            usage={
                "prompt_tokens": len(request.messages),
                "completion_tokens": len(output.split()),
                "total_tokens": len(request.messages) + len(output.split()),
            },
        )

    async def _stream_generate(self, request: GenerateRequest):
        for chunk in ("Echo", ": ", request.messages[-1].content):
            yield chunk

    def _tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens) -> str:
        return bytes(tokens).decode("utf-8")


@pytest.mark.asyncio
@pytest.mark.long_running
async def test_benchmark_reports_mlx_acceleration_paths(temp_settings: LewLMSettings) -> None:
    model_dir = temp_settings.models_dir[0] / "benchmark-acceleration-mlx"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "max_position_embeddings": 8192}),
        encoding="utf-8",
    )
    (model_dir / "weights.safetensors").write_bytes(b"mlx-weights")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    settings = temp_settings.with_updates(
        mlx_graph_compile_enabled=True,
        mlx_attention_kernel_mode="flash_attention",
    )
    runtime = AccelerationBenchmarkRuntime(settings=settings)
    services = bootstrap_services(
        settings,
        runtime_overrides={RuntimeAffinity.MLX_TEXT: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        model_id = next(
            manifest.model_id
            for manifest in manifests
            if manifest.source_path == str(model_dir)
        )
        result = await services.telemetry_service.benchmark(model_id=model_id, prompt="acceleration benchmark")
        runtime_stats = await services.telemetry_service.runtime_stats()
        emit_benchmark_case_report(
            label="mlx-acceleration-feature",
            payload=result.model_dump(mode="json"),
            feature_names=("graph_compilation", "attention_kernel_acceleration"),
            scenario_names=("mlx_acceleration_paths",),
        )

        benchmark_features = {item.feature.value: item for item in result.performance_features}
        runtime_features = {item.feature.value: item for item in runtime_stats.performance_features}
        scenario = next(item for item in result.scenarios if item.scenario == "mlx_acceleration_paths")

        assert benchmark_features["graph_compilation"].supported is True
        assert benchmark_features["graph_compilation"].active is True
        assert benchmark_features["attention_kernel_acceleration"].supported is True
        assert benchmark_features["attention_kernel_acceleration"].active is True
        assert runtime_features["graph_compilation"].active is True
        assert runtime_features["attention_kernel_acceleration"].active is True
        assert scenario.feature == PerformanceFeatureName.ATTENTION_KERNEL_ACCELERATION
        assert scenario.status == "observed"
        assert scenario.metrics["average_accelerated_generate_seconds"] < scenario.metrics["average_stock_generate_seconds"]
        assert scenario.metrics["average_time_saved_seconds"] > 0
        assert scenario.metrics["compiled_sample_count"] == 1
        assert scenario.metrics["fallback_sample_count"] == 0
        assert scenario.metrics["compile_states"] == "decode"
        assert scenario.metrics["kernel_paths"] == "flash_attention"
        assert scenario.samples[0].metrics["compile_state"] == "decode"
        assert scenario.samples[0].metrics["kernel_path"] == "flash_attention"
        assert scenario.samples[0].metrics["graph_compile_used"] is True
    finally:
        await services.aclose()
