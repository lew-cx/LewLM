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
    GenerateMessage,
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    ModelModality,
    RuntimeAffinity,
    SpeculationMode,
)
from lewlm.core.speculation import chat_speculation_workload_class, speculation_benchmark_preference_key
from lewlm.runtime.base import ManagedTextRuntime


class _BaseBenchmarkRuntime(ManagedTextRuntime):
    supported_modalities = (ModelModality.TEXT,)
    supported_capabilities = frozenset({CapabilityName.CHAT, CapabilityName.STREAMING})

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, None

    async def _load_model(self, manifest: ModelManifest) -> None:
        return None

    async def _unload_model(self, model_id: str) -> None:
        return None

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
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


class SpeculativeBenchmarkMLXRuntime(_BaseBenchmarkRuntime):
    name = "fake_mlx_speculative"
    affinity = RuntimeAffinity.MLX_TEXT
    supported_formats = (ModelFormat.MLX,)

    def __init__(self) -> None:
        super().__init__()
        self.speculative_request_count = 0
        self.last_draft_model_id: str | None = None

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        if request.speculation is None:
            await asyncio.sleep(0.02)
        else:
            await asyncio.sleep(0.001)
        response = await super().generate(request)
        if request.speculation is None or request.speculation.mode != SpeculationMode.DRAFT_MODEL:
            return response
        self.speculative_request_count += 1
        self.last_draft_model_id = request.speculation.draft_model_id
        usage = {
            **response.usage,
            "drafted_tokens": 2,
            "verified_tokens": 1,
        }
        return response.model_copy(update={"usage": usage})

    async def health_check(self) -> dict[str, object]:
        payload = await super().health_check()
        payload["performance_features"] = {
            "speculative_decoding": {
                "supported": True,
                "active": self.speculative_request_count > 0,
                "modes": ["draft_model"],
                "reason": "Fake MLX runtime exposes draft-model speculation for benchmark proof.",
                "metrics": {
                    "request_count": self.speculative_request_count,
                    "drafted_tokens": 2 if self.speculative_request_count else 0,
                    "verified_tokens": 1 if self.speculative_request_count else 0,
                    "configured_num_draft_tokens": 2,
                },
                "notes": (
                    [f"Last active draft model: `{self.last_draft_model_id}`."]
                    if self.last_draft_model_id is not None
                    else []
                ),
            },
        }
        return payload


class PromptLookupBenchmarkLlamaRuntime(_BaseBenchmarkRuntime):
    name = "fake_llamacpp_prompt_lookup"
    affinity = RuntimeAffinity.LLAMACPP
    supported_formats = (ModelFormat.GGUF,)

    def __init__(self) -> None:
        super().__init__()
        self.prompt_lookup_request_count = 0

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        response = await super().generate(request)
        if request.speculation is None or request.speculation.mode != SpeculationMode.PROMPT_LOOKUP:
            return response
        self.prompt_lookup_request_count += 1
        usage = {
            **response.usage,
            "prompt_lookup_requests": 1,
            "prompt_lookup_max_ngram_size": request.speculation.prompt_lookup_max_ngram_size or 0,
            "prompt_lookup_num_pred_tokens": request.speculation.prompt_lookup_num_pred_tokens or 0,
        }
        return response.model_copy(update={"usage": usage})

    async def health_check(self) -> dict[str, object]:
        payload = await super().health_check()
        payload["performance_features"] = {
            "prompt_lookup_speculation": {
                "supported": True,
                "active": self.prompt_lookup_request_count > 0,
                "modes": ["prompt_lookup"],
                "reason": "Fake llama.cpp runtime exposes prompt-lookup speculation for benchmark proof.",
                "metrics": {
                    "request_count": self.prompt_lookup_request_count,
                    "configured_max_ngram_size": 4,
                    "configured_num_pred_tokens": 12,
                },
                "notes": [],
            },
        }
        return payload


class SelectingSpeculativeBenchmarkMLXRuntime(_BaseBenchmarkRuntime):
    name = "fake_mlx_speculation_selector"
    affinity = RuntimeAffinity.MLX_TEXT
    supported_formats = (ModelFormat.MLX,)

    def __init__(self) -> None:
        super().__init__()
        self.last_draft_model_id: str | None = None

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        response = await super().generate(request)
        if request.speculation is None:
            await asyncio.sleep(0.02)
            return response
        await asyncio.sleep(0.001)
        self.last_draft_model_id = request.speculation.draft_model_id
        return response.model_copy(
            update={
                "usage": {
                    **response.usage,
                    "drafted_tokens": 4,
                    "verified_tokens": 3,
                },
            },
        )


class FrontierSpeculativeBenchmarkMLXRuntime(_BaseBenchmarkRuntime):
    name = "fake_mlx_frontier_selector"
    affinity = RuntimeAffinity.MLX_TEXT
    supported_formats = (ModelFormat.MLX,)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        response = await super().generate(request)
        speculation = request.speculation
        if speculation is None:
            await asyncio.sleep(0.03)
            return response
        if speculation.mode == SpeculationMode.MEDUSA:
            await asyncio.sleep(0.002)
            return response.model_copy(
                update={
                    "usage": {
                        **response.usage,
                        "drafted_tokens": 6,
                        "verified_tokens": 5,
                    },
                },
            )
        if speculation.mode == SpeculationMode.EAGLE:
            await asyncio.sleep(0.001)
            return response.model_copy(
                update={
                    "output_text": "Diverged output",
                    "usage": {
                        **response.usage,
                        "drafted_tokens": 7,
                        "verified_tokens": 3,
                    },
                },
            )
        return response


@pytest.mark.asyncio
@pytest.mark.long_running
async def test_benchmark_reports_mlx_draft_speculation(
    temp_settings: LewLMSettings,
    sample_models_root: Path,
) -> None:
    draft_dir = temp_settings.models_dir[0] / "qwen2.5-0.5b-instruct-draft-mlx"
    draft_dir.mkdir(parents=True)
    (draft_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "max_position_embeddings": 32768}),
        encoding="utf-8",
    )
    (draft_dir / "weights.safetensors").write_bytes(b"draft-weights")
    (draft_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    runtime = SpeculativeBenchmarkMLXRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(
            speculative_decoding_enabled=True,
            speculative_decoding_num_draft_tokens=2,
        ),
        runtime_overrides={RuntimeAffinity.MLX_TEXT: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        primary_model_id = next(
            manifest.model_id
            for manifest in manifests
            if manifest.display_name == "qwen2.5-1.5b-instruct-mlx"
        )

        result = await services.telemetry_service.benchmark(model_id=primary_model_id, prompt="speculative benchmark")
        runtime_stats = await services.telemetry_service.runtime_stats()
        emit_benchmark_case_report(
            label="speculation-draft-feature",
            payload=result.model_dump(mode="json"),
            feature_names=("speculative_decoding",),
            scenario_names=("speculation_selection",),
        )

        benchmark_features = {item.feature.value: item for item in result.performance_features}
        stats_features = {item.feature.value: item for item in runtime_stats.performance_features}

        assert benchmark_features["speculative_decoding"].supported is True
        assert benchmark_features["speculative_decoding"].active is True
        assert benchmark_features["speculative_decoding"].metrics["request_count"] == 1
        assert stats_features["speculative_decoding"].supported is True
        assert stats_features["speculative_decoding"].active is True
        assert runtime.last_draft_model_id is not None
        assert runtime.last_draft_model_id != primary_model_id
    finally:
        await services.aclose()


@pytest.mark.asyncio
@pytest.mark.long_running
async def test_benchmark_reports_prompt_lookup_speculation(
    temp_settings: LewLMSettings,
    sample_models_root: Path,
) -> None:
    runtime = PromptLookupBenchmarkLlamaRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(
            prompt_lookup_speculation_enabled=True,
            prompt_lookup_max_ngram_size=4,
            prompt_lookup_num_pred_tokens=12,
        ),
        runtime_overrides={RuntimeAffinity.LLAMACPP: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        gguf_model_id = next(manifest.model_id for manifest in manifests if manifest.format_type.value == "gguf")

        result = await services.telemetry_service.benchmark(model_id=gguf_model_id, prompt="prompt lookup benchmark")
        runtime_stats = await services.telemetry_service.runtime_stats()
        emit_benchmark_case_report(
            label="speculation-prompt-lookup-feature",
            payload=result.model_dump(mode="json"),
            feature_names=("prompt_lookup_speculation",),
            scenario_names=("speculation_selection",),
        )

        benchmark_features = {item.feature.value: item for item in result.performance_features}
        stats_features = {item.feature.value: item for item in runtime_stats.performance_features}

        assert benchmark_features["prompt_lookup_speculation"].supported is True
        assert benchmark_features["prompt_lookup_speculation"].active is True
        assert benchmark_features["prompt_lookup_speculation"].metrics["request_count"] == 1
        assert stats_features["prompt_lookup_speculation"].supported is True
        assert stats_features["prompt_lookup_speculation"].active is True
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_benchmark_selects_and_persists_fastest_safe_speculation_mode(
    temp_settings: LewLMSettings,
    sample_models_root: Path,
) -> None:
    draft_dir = temp_settings.models_dir[0] / "qwen2.5-0.5b-instruct-draft-mlx"
    draft_dir.mkdir(parents=True)
    (draft_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "max_position_embeddings": 32768}),
        encoding="utf-8",
    )
    (draft_dir / "weights.safetensors").write_bytes(b"draft-weights")
    (draft_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    runtime = SelectingSpeculativeBenchmarkMLXRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(
            speculative_decoding_enabled=True,
            speculative_decoding_num_draft_tokens=2,
        ),
        runtime_overrides={RuntimeAffinity.MLX_TEXT: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        primary_model_id = next(
            manifest.model_id
            for manifest in manifests
            if manifest.display_name == "qwen2.5-1.5b-instruct-mlx"
        )

        result = await services.telemetry_service.benchmark(model_id=primary_model_id, prompt="select best speculation")
        scenario = next(item for item in result.scenarios if item.scenario == "speculation_selection")
        workload_class = chat_speculation_workload_class(
            messages=[GenerateMessage(role="user", content="select best speculation")],
            max_tokens=128,
        )
        stored_preference = services.metadata_store.get_value(
            speculation_benchmark_preference_key(
                model_id=primary_model_id,
                runtime_name=runtime.name,
                workload_class=workload_class,
            ),
        )

        assert result.output_text == "Echo: select best speculation"
        assert scenario.metrics["selected_mode"] == "draft_model"
        assert scenario.metrics["candidate_count"] >= 1
        assert scenario.metrics["workload_class"] == workload_class
        assert runtime.last_draft_model_id is not None
        assert stored_preference is not None
        assert stored_preference["selected_mode"] == "draft_model"
        assert stored_preference["workload_class"] == workload_class
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_benchmark_reports_skipped_and_lost_frontier_speculation_modes(
    temp_settings: LewLMSettings,
    sample_models_root: Path,
) -> None:
    runtime = FrontierSpeculativeBenchmarkMLXRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(speculative_decoding_enabled=True),
        runtime_overrides={RuntimeAffinity.MLX_TEXT: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        primary_manifest = next(
            manifest
            for manifest in manifests
            if manifest.display_name == "qwen2.5-1.5b-instruct-mlx"
        )
        services.metadata_store.replace_model_manifests(
            [
                primary_manifest.model_copy(
                    update={
                        "metadata": {
                            "speculation_modes": [
                                {"mode": "medusa", "parameters": {"backend_parameter": "medusa"}},
                                {"mode": "eagle", "parameters": {"backend_parameter": "eagle"}},
                                {
                                    "mode": "suffix_decoding",
                                    "required_modules": ["definitely_missing_frontier_module"],
                                    "parameters": {"backend_parameter": "suffix_decoding"},
                                },
                            ],
                        },
                    },
                ),
            ],
            stale_source_paths=(),
        )

        result = await services.telemetry_service.benchmark(model_id=primary_manifest.model_id, prompt="implement a helper")
        scenario = next(item for item in result.scenarios if item.scenario == "speculation_selection")
        sample_metrics = [sample.metrics for sample in scenario.samples]

        assert scenario.metrics["selected_mode"] == "medusa"
        assert scenario.metrics["skipped_candidate_count"] >= 1
        assert scenario.metrics["selected_acceptance_rate"] == pytest.approx(5 / 6, rel=1e-4)
        assert scenario.metrics["selected_verified_tokens"] == 5
        assert scenario.metrics["selected_rollback_tokens"] == 1
        assert any(
            metrics.get("mode") == "eagle"
            and metrics.get("selection_status") == "lost"
            and "changed the output" in str(metrics.get("outcome_reason"))
            for metrics in sample_metrics
        )
        assert any(
            metrics.get("mode") == "suffix_decoding"
            and metrics.get("selection_status") == "skipped"
            and "optional local modules" in str(metrics.get("outcome_reason"))
            for metrics in sample_metrics
        )
    finally:
        await services.aclose()
