from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from conftest import (
    FakeLlamaCppRuntime,
    FakeMLXAudioRuntime,
    FakeMLXSemanticRuntime,
    FakeMLXVisionRuntime,
    emit_benchmark_case_report,
    emit_benchmark_suite_report,
)
from lewlm.cli.main import main
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import (
    CapabilityName,
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    ModelModality,
    RuntimeAffinity,
    SpeculationMode,
)
from lewlm.runtime.base import ManagedTextRuntime


class AccelerationDiagnosticRuntime(ManagedTextRuntime):
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
                "reason": "Fake MLX runtime exposes graph compilation for benchmark diagnostics.",
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
                "reason": "Fake MLX runtime exposes accelerated attention hooks for benchmark diagnostics.",
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


class ServingProfileBenchmarkRuntime(FakeMLXSemanticRuntime):
    def __init__(self, *, settings: LewLMSettings | None = None) -> None:
        super().__init__(settings=settings)

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        if self.settings.prefill_token_batch_size != 256:
            await asyncio.sleep(0.002)
        response = await super()._generate(request)
        return response.model_copy(
            update={"output_text": f"{response.output_text} [prefill={self.settings.prefill_token_batch_size}]"},
        )

    async def _stream_generate(self, request: GenerateRequest):
        response = await self._generate(request)
        yield response.output_text


class SpeculationDiagnosticRuntime(FakeMLXSemanticRuntime):
    name = "fake_mlx_speculation_selector"
    affinity = RuntimeAffinity.MLX_TEXT
    supported_formats = (ModelFormat.MLX,)

    def __init__(self, *, settings: LewLMSettings | None = None) -> None:
        super().__init__(settings=settings)
        self.last_draft_model_id: str | None = None
        self.speculative_request_count = 0

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        response = await super().generate(request)
        if request.speculation is None:
            await asyncio.sleep(0.02)
            return response
        self.speculative_request_count += 1
        self.last_draft_model_id = request.speculation.draft_model_id
        await asyncio.sleep(0.001)
        return response.model_copy(
            update={
                "usage": {
                    **response.usage,
                    "drafted_tokens": 4,
                    "verified_tokens": 3,
                },
            },
        )

    async def health_check(self) -> dict[str, object]:
        payload = await super().health_check()
        performance_features = payload.get("performance_features")
        if not isinstance(performance_features, dict):
            performance_features = {}
            payload["performance_features"] = performance_features
        performance_features["speculative_decoding"] = {
            "supported": True,
            "active": self.speculative_request_count > 0,
            "modes": ["draft_model"],
            "reason": "Fake MLX runtime exposes draft-model speculation for benchmark diagnostics.",
            "metrics": {
                "request_count": self.speculative_request_count,
                "drafted_tokens": 4 if self.speculative_request_count else 0,
                "verified_tokens": 3 if self.speculative_request_count else 0,
                "configured_num_draft_tokens": 2,
            },
            "notes": (
                [f"Last active draft model: `{self.last_draft_model_id}`."]
                if self.last_draft_model_id is not None
                else []
            ),
        }
        return payload


class PromptLookupDiagnosticRuntime(FakeLlamaCppRuntime):
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
        return response.model_copy(
            update={
                "usage": {
                    **response.usage,
                    "prompt_lookup_requests": 1,
                    "prompt_lookup_max_ngram_size": request.speculation.prompt_lookup_max_ngram_size or 0,
                    "prompt_lookup_num_pred_tokens": request.speculation.prompt_lookup_num_pred_tokens or 0,
                },
            },
        )

    async def health_check(self) -> dict[str, object]:
        payload = await super().health_check()
        performance_features = payload.get("performance_features")
        if not isinstance(performance_features, dict):
            performance_features = {}
            payload["performance_features"] = performance_features
        performance_features["prompt_lookup_speculation"] = {
            "supported": True,
            "active": self.prompt_lookup_request_count > 0,
            "modes": ["prompt_lookup"],
            "reason": "Fake llama.cpp runtime exposes prompt-lookup speculation for benchmark diagnostics.",
            "metrics": {
                "request_count": self.prompt_lookup_request_count,
                "configured_max_ngram_size": 4,
                "configured_num_pred_tokens": 12,
            },
            "notes": [],
        }
        return payload


def _scan_payload(*, settings: LewLMSettings, services, capsys) -> dict[str, object]:
    exit_code = main(["scan", "--json"], settings=settings, services=services)
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    return payload


def _benchmark_payload(*, settings: LewLMSettings, services, capsys, args: list[str]) -> dict[str, object]:
    exit_code = main(["benchmark", *args, "--json"], settings=settings, services=services)
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    return payload


def _telemetry_benchmark_payload(
    *,
    services,
    model_id: str,
    prompt: str = "Benchmark ping",
    capability: str = CapabilityName.CHAT.value,
    warmup_run_count: int = 1,
) -> dict[str, object]:
    result = asyncio.run(
        services.telemetry_service.benchmark(
            model_id=model_id,
            prompt=prompt,
            capability=capability,
            warmup_run_count=warmup_run_count,
        ),
    )
    return result.model_dump(mode="json")


def _autotune_payload(*, settings: LewLMSettings, services, capsys, model_id: str) -> dict[str, object]:
    exit_code = main(["autotune", "--model", model_id, "--json"], settings=settings, services=services)
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    return payload


def _manifest_by_display_name(scan_payload: dict[str, object], display_name: str) -> dict[str, object]:
    return next(
        manifest
        for manifest in scan_payload["manifests"]
        if manifest["display_name"] == display_name
    )


def _manifest_by_modality(scan_payload: dict[str, object], modality: str) -> dict[str, object]:
    return next(
        manifest
        for manifest in scan_payload["manifests"]
        if modality in manifest["modality"]
    )


def _scenario(payload: dict[str, object], scenario_name: str) -> dict[str, object]:
    return next(item for item in payload["scenarios"] if item["scenario"] == scenario_name)


def _feature(payload: dict[str, object], feature_name: str) -> dict[str, object]:
    return next(item for item in payload["performance_features"] if item["feature"] == feature_name)


def _evidence(payload: dict[str, object], family_name: str) -> dict[str, object]:
    return next(item for item in payload["performance_core_evidence"] if item["family"] == family_name)


def test_cli_benchmark_diagnostics_text_cache_batching_and_prefill(
    temp_settings,
    services_with_fake_attachment_runtime,
    capsys,
) -> None:
    scan_payload = _scan_payload(
        settings=temp_settings,
        services=services_with_fake_attachment_runtime,
        capsys=capsys,
    )
    manifest = _manifest_by_display_name(scan_payload, "qwen2.5-1.5b-instruct-mlx")

    payload = _telemetry_benchmark_payload(
        services=services_with_fake_attachment_runtime,
        model_id=manifest["model_id"],
    )
    emit_benchmark_case_report(
        label="text-cache-batching-prefill",
        payload=payload,
        feature_names=(
            "serving_core",
            "continuous_batching",
            "prefix_cache",
            "persistent_multi_context_cache",
            "paged_kv_cache",
            "kv_cache_quantization",
            "prefill_optimization",
        ),
        scenario_names=("repeated_prefix", "continuous_batching", "warm_chat_cache"),
    )

    repeated_prefix = _scenario(payload, "repeated_prefix")
    continuous_batching = _scenario(payload, "continuous_batching")
    warm_cache = _scenario(payload, "warm_chat_cache")

    assert repeated_prefix["status"] == "observed"
    assert repeated_prefix["metrics"]["average_second_over_first_ratio"] is not None
    assert repeated_prefix["metrics"]["total_second_cached_tokens"] >= 0
    assert continuous_batching["status"] == "observed"
    assert continuous_batching["metrics"]["native_batched_request_delta"] >= 1
    assert continuous_batching["metrics"]["frontier_batched_request_delta"] >= 1
    assert warm_cache["status"] == "observed"
    assert warm_cache["metrics"]["average_warm_over_cold_ttft_ratio"] is not None
    assert warm_cache["metrics"]["total_cache_restores"] >= 0

    assert _feature(payload, "serving_core")["supported"] is True
    assert "total_sequences_started" in _feature(payload, "serving_core")["metrics"]
    assert _feature(payload, "continuous_batching")["supported"] is True
    assert _feature(payload, "continuous_batching")["ownership_modes"] == ["backend_native"]
    assert _feature(payload, "prefix_cache")["active"] is True
    assert "ownership_modes" in _feature(payload, "prefix_cache")
    assert _feature(payload, "persistent_multi_context_cache")["supported"] is True
    assert _feature(payload, "paged_kv_cache")["supported"] is True
    assert _feature(payload, "kv_cache_quantization")["supported"] is True
    assert _feature(payload, "prefill_optimization")["active"] is True
    assert _evidence(payload, "continuous_batching")["family"] == "continuous_batching"
    assert _evidence(payload, "prefill_isolation")["family"] == "prefill_isolation"


def test_cli_benchmark_diagnostics_gguf_constrained_decoding(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    scan_payload = _scan_payload(
        settings=temp_settings,
        services=services_with_fake_runtime,
        capsys=capsys,
    )
    manifest = next(item for item in scan_payload["manifests"] if item["format_type"] == "gguf")

    payload = _telemetry_benchmark_payload(
        services=services_with_fake_runtime,
        model_id=manifest["model_id"],
    )
    emit_benchmark_case_report(
        label="gguf-constrained-decoding",
        payload=payload,
        scenario_names=("constrained_decoding",),
    )

    constrained_decoding = _scenario(payload, "constrained_decoding")

    assert constrained_decoding["status"] == "observed"
    assert constrained_decoding["metrics"]["decoder_enforced"] is True
    assert constrained_decoding["metrics"]["fallback_used"] is False
    assert constrained_decoding["metrics"]["validation_state"] == "valid"
    assert _evidence(payload, "constrained_decoding")["family"] == "constrained_decoding"


def test_cli_benchmark_diagnostics_multimodal_chat_encoder_reuse(
    temp_settings,
    services_with_fake_attachment_runtime,
    capsys,
) -> None:
    scan_payload = _scan_payload(
        settings=temp_settings,
        services=services_with_fake_attachment_runtime,
        capsys=capsys,
    )
    manifest = _manifest_by_display_name(scan_payload, "qwen2-vl-vision-mlx")

    payload = _telemetry_benchmark_payload(
        services=services_with_fake_attachment_runtime,
        model_id=manifest["model_id"],
    )
    emit_benchmark_case_report(
        label="multimodal-vision-batching-and-encoder-reuse",
        payload=payload,
        feature_names=("continuous_batching", "multimodal_encoder_caching"),
        scenario_names=("continuous_batching", "multimodal_encoder_reuse"),
    )

    batching = _scenario(payload, "continuous_batching")
    scenario = _scenario(payload, "multimodal_encoder_reuse")

    assert batching["status"] == "observed"
    assert batching["metrics"]["request_shape"] == "single_image"
    assert batching["metrics"]["runtime_native_batch_request_count"] >= 2
    assert batching["metrics"]["runtime_stock_single_request_fallback_request_count"] == 0
    assert batching["metrics"]["runtime_native_batch_backend"] == "fake_mlx_vision.generate_batch"
    assert _feature(payload, "continuous_batching")["supported"] is True
    assert scenario["status"] == "observed"
    assert scenario["metrics"]["sample_count"] == 2
    assert scenario["metrics"]["multimodal_feature_cache_hit_delta"] is not None
    assert scenario["metrics"]["multimodal_feature_cache_miss_delta"] is not None
    assert scenario["metrics"]["multimodal_encoder_cache_hit_delta"] >= 2
    assert scenario["metrics"]["multimodal_encoder_cache_miss_delta"] >= 4
    assert scenario["metrics"]["average_cold_elapsed_seconds"] is not None
    assert scenario["metrics"]["average_encoder_advantage_seconds"] is not None
    assert {sample["metrics"]["sample_type"] for sample in scenario["samples"]} == {"image", "frame_bundle"}
    assert all("cold_elapsed_seconds" in sample["metrics"] for sample in scenario["samples"])
    assert all("first_feature_cache_hit_delta" in sample["metrics"] for sample in scenario["samples"])
    assert all("second_feature_cache_hit_delta" in sample["metrics"] for sample in scenario["samples"])
    assert _feature(payload, "multimodal_encoder_caching")["supported"] is True


def test_cli_benchmark_diagnostics_audio_encoder_reuse(
    temp_settings,
    services_with_fake_attachment_runtime,
    capsys,
) -> None:
    scan_payload = _scan_payload(
        settings=temp_settings,
        services=services_with_fake_attachment_runtime,
        capsys=capsys,
    )
    manifest = _manifest_by_modality(scan_payload, "audio")

    payload = _telemetry_benchmark_payload(
        services=services_with_fake_attachment_runtime,
        model_id=manifest["model_id"],
        capability=CapabilityName.AUDIO_TRANSCRIPTION.value,
    )
    emit_benchmark_case_report(
        label="multimodal-audio-encoder-reuse",
        payload=payload,
        feature_names=("multimodal_encoder_caching",),
        scenario_names=("multimodal_encoder_reuse",),
    )

    scenario = _scenario(payload, "multimodal_encoder_reuse")

    assert payload["capability"] == "audio_transcription"
    assert scenario["status"] == "observed"
    assert scenario["metrics"]["sample_count"] == 1
    assert scenario["metrics"]["multimodal_encoder_cache_hit_delta"] >= 1
    assert scenario["metrics"]["multimodal_encoder_cache_miss_delta"] >= 1
    assert scenario["metrics"]["average_chunk_count"] > 1
    assert scenario["metrics"]["average_second_over_first_ratio"] is not None
    assert scenario["samples"][0]["metrics"]["sample_type"] == "audio"
    assert scenario["samples"][0]["metrics"]["chunk_count"] > 1
    assert _feature(payload, "multimodal_encoder_caching")["supported"] is True


@pytest.mark.filterwarnings("ignore:builtin type SwigPyPacked has no __module__ attribute:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:builtin type SwigPyObject has no __module__ attribute:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:builtin type swigvarlink has no __module__ attribute:DeprecationWarning")
def test_cli_benchmark_diagnostics_speculation_selection(
    temp_settings,
    sample_models_root: Path,
    capsys,
) -> None:
    draft_dir = temp_settings.models_dir[0] / "qwen2.5-0.5b-instruct-draft-mlx"
    draft_dir.mkdir(parents=True)
    (draft_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "max_position_embeddings": 32768}),
        encoding="utf-8",
    )
    (draft_dir / "weights.safetensors").write_bytes(b"draft-weights")
    (draft_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    settings = temp_settings.with_updates(
        speculative_decoding_enabled=True,
        speculative_decoding_num_draft_tokens=2,
    )
    runtime = SpeculationDiagnosticRuntime(settings=settings)
    services = bootstrap_services(
        settings,
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: runtime,
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )
    try:
        scan_payload = _scan_payload(settings=settings, services=services, capsys=capsys)
        manifest = _manifest_by_display_name(scan_payload, "qwen2.5-1.5b-instruct-mlx")

        payload = _telemetry_benchmark_payload(
            services=services,
            model_id=manifest["model_id"],
        )
        emit_benchmark_case_report(
            label="draft-speculation-selection",
            payload=payload,
            feature_names=("speculative_decoding",),
            scenario_names=("speculation_selection",),
        )

        scenario = _scenario(payload, "speculation_selection")

        assert scenario["status"] == "observed"
        assert scenario["metrics"]["selected_mode"] == "draft_model"
        assert scenario["metrics"]["safe_candidate_count"] >= 1
        assert _feature(payload, "speculative_decoding")["active"] is True
        assert runtime.last_draft_model_id is not None
        assert runtime.last_draft_model_id != manifest["model_id"]
    finally:
        services.close()


def test_cli_benchmark_diagnostics_mlx_acceleration_paths(
    temp_settings,
    capsys,
) -> None:
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
    runtime = AccelerationDiagnosticRuntime(settings=settings)
    services = bootstrap_services(
        settings,
        runtime_overrides={RuntimeAffinity.MLX_TEXT: runtime},
    )
    try:
        scan_payload = _scan_payload(settings=settings, services=services, capsys=capsys)
        manifest = _manifest_by_display_name(scan_payload, "benchmark-acceleration-mlx")

        payload = _telemetry_benchmark_payload(
            services=services,
            model_id=manifest["model_id"],
        )
        emit_benchmark_case_report(
            label="mlx-acceleration",
            payload=payload,
            feature_names=("graph_compilation", "attention_kernel_acceleration"),
            scenario_names=("mlx_acceleration_paths",),
        )

        scenario = _scenario(payload, "mlx_acceleration_paths")

        assert scenario["status"] == "observed"
        assert (
            scenario["metrics"]["average_accelerated_generate_seconds"]
            < scenario["metrics"]["average_stock_generate_seconds"]
        )
        assert scenario["metrics"]["average_time_saved_seconds"] > 0
        assert scenario["metrics"]["compiled_sample_count"] == 1
        assert scenario["metrics"]["fallback_sample_count"] == 0
        assert scenario["metrics"]["compile_states"] == "decode"
        assert scenario["metrics"]["kernel_paths"] == "flash_attention"
        assert scenario["samples"][0]["metrics"]["compile_state"] == "decode"
        assert scenario["samples"][0]["metrics"]["kernel_path"] == "flash_attention"
        assert scenario["samples"][0]["metrics"]["graph_compile_used"] is True
        assert _feature(payload, "graph_compilation")["active"] is True
        assert _feature(payload, "attention_kernel_acceleration")["active"] is True
    finally:
        services.close()


def test_cli_benchmark_stdout_omits_internal_acceleration_diagnostics(
    temp_settings,
    capsys,
) -> None:
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
    runtime = AccelerationDiagnosticRuntime(settings=settings)
    services = bootstrap_services(
        settings,
        runtime_overrides={RuntimeAffinity.MLX_TEXT: runtime},
    )
    try:
        scan_payload = _scan_payload(settings=settings, services=services, capsys=capsys)
        manifest = _manifest_by_display_name(scan_payload, "benchmark-acceleration-mlx")

        exit_code = main(["benchmark", "--model", manifest["model_id"]], settings=settings, services=services)
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert "mlx_acceleration_paths" not in stdout
        assert "kernels=flash_attention" not in stdout
        assert "compile_state=decode" not in stdout
        assert "compiled=1/1" not in stdout
        assert "saved=" not in stdout
    finally:
        services.close()


@pytest.mark.parametrize(
    ("display_name", "config_payload", "weights", "expected_subtype"),
    (
        (
            "hybrid-mamba-mlx",
            {
                "model_type": "gateddeltanet",
                "d_state": 128,
                "num_attention_heads": 8,
                "max_position_embeddings": 8192,
            },
            b"ssm-weights",
            "hybrid_ssm",
        ),
        (
            "giant-mixtral-mlx",
            {
                "model_type": "mixtral",
                "num_experts": 64,
                "experts_per_token": 8,
                "max_position_embeddings": 32768,
            },
            b"m" * (16 * 1024 * 1024),
            "moe",
        ),
    ),
    ids=("hybrid-ssm", "moe"),
)
def test_cli_benchmark_diagnostics_frontier_architecture_modes(
    temp_settings,
    capsys,
    display_name: str,
    config_payload: dict[str, object],
    weights: bytes,
    expected_subtype: str,
) -> None:
    model_dir = temp_settings.models_dir[0] / display_name
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")
    (model_dir / "weights.safetensors").write_bytes(weights)
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    settings = temp_settings.with_updates(
        moe_bounded_memory_mode="expert_streaming",
        moe_resident_expert_count=8,
    )
    services = bootstrap_services(
        settings,
        runtime_overrides={RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(settings=settings)},
    )
    try:
        scan_payload = _scan_payload(settings=settings, services=services, capsys=capsys)
        manifest = _manifest_by_display_name(scan_payload, display_name)

        payload = _telemetry_benchmark_payload(
            services=services,
            model_id=manifest["model_id"],
        )
        emit_benchmark_case_report(
            label=f"frontier-{expected_subtype}",
            payload=payload,
            feature_names=(
                ("hybrid_ssm_routing", "ssm_state_cache_handling")
                if expected_subtype == "hybrid_ssm"
                else ("moe_bounded_memory_serving",)
            ),
            scenario_names=("frontier_architecture_modes",),
        )

        scenario = _scenario(payload, "frontier_architecture_modes")
        sample_metrics = scenario["samples"][0]["metrics"]

        assert scenario["status"] == "observed"
        assert sample_metrics["architecture_subtype"] == expected_subtype
        assert sample_metrics["planning_only"] is False
        if expected_subtype == "hybrid_ssm":
            assert scenario["metrics"]["hybrid_ssm_model_count"] == 1
            assert sample_metrics["cache_state_handling"] == "hybrid_attention_state"
            assert sample_metrics["state_cache_bytes"] > 0
        else:
            assert scenario["metrics"]["moe_model_count"] == 1
            assert sample_metrics["resident_expert_count"] == 8
            assert sample_metrics["planned_memory_mb"] < sample_metrics["full_estimated_memory_mb"]
            assert sample_metrics["effective_loaded_memory_mb"] == sample_metrics["planned_memory_mb"]
            assert sample_metrics["expert_swap_count"] > 0
    finally:
        services.close()


@pytest.mark.filterwarnings("ignore:builtin type SwigPyPacked has no __module__ attribute:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:builtin type SwigPyObject has no __module__ attribute:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:builtin type swigvarlink has no __module__ attribute:DeprecationWarning")
def test_optimization_benchmark_suite_logs_combined_results(
    temp_settings,
    sample_chat_models_root: Path,
) -> None:
    suite_entries: list[tuple[str, dict[str, object], tuple[str, ...], tuple[str, ...]]] = []

    attachment_services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
            RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
        },
    )
    try:
        manifests = attachment_services.model_registry.scan().manifests
        text_model_id = next(
            manifest.model_id
            for manifest in manifests
            if manifest.display_name == "qwen2.5-1.5b-instruct-mlx"
        )
        vision_model_id = next(
            manifest.model_id
            for manifest in manifests
            if manifest.display_name == "qwen2-vl-vision-mlx"
        )
        audio_model_id = next(
            manifest.model_id
            for manifest in manifests
            if manifest.display_name == "whisper-mini-audio"
        )

        text_payload = _telemetry_benchmark_payload(
            services=attachment_services,
            model_id=text_model_id,
        )
        text_features = (
            "continuous_batching",
            "prefix_cache",
            "persistent_multi_context_cache",
            "paged_kv_cache",
            "kv_cache_quantization",
            "prefill_optimization",
        )
        text_scenarios = ("repeated_prefix", "continuous_batching", "warm_chat_cache")
        emit_benchmark_case_report(
            label="suite-text-cache-batching-prefill",
            payload=text_payload,
            feature_names=text_features,
            scenario_names=text_scenarios,
        )
        suite_entries.append(("suite-text-cache-batching-prefill", text_payload, text_features, text_scenarios))
        assert _feature(text_payload, "continuous_batching")["active"] is True
        assert _scenario(text_payload, "continuous_batching")["status"] == "observed"
        assert text_payload["optimization_attribution"]["modality_routing"]["status"] == "default"
        assert text_payload["optimization_attribution"]["modality_routing"]["metrics"] == {
            "request_modality": "text_only",
            "modality_path": "text_default",
            "runtime_affinity": "mlx_text",
        }

        vision_payload = _telemetry_benchmark_payload(
            services=attachment_services,
            model_id=vision_model_id,
        )
        vision_features = ("continuous_batching", "multimodal_encoder_caching")
        vision_scenarios = ("continuous_batching", "multimodal_encoder_reuse")
        emit_benchmark_case_report(
            label="suite-multimodal-vision",
            payload=vision_payload,
            feature_names=vision_features,
            scenario_names=vision_scenarios,
        )
        suite_entries.append(("suite-multimodal-vision", vision_payload, vision_features, vision_scenarios))
        assert _feature(vision_payload, "continuous_batching")["active"] is True
        assert _scenario(vision_payload, "continuous_batching")["status"] == "observed"
        assert _feature(vision_payload, "multimodal_encoder_caching")["supported"] is True
        assert _scenario(vision_payload, "multimodal_encoder_reuse")["status"] == "observed"
        assert vision_payload["optimization_attribution"]["modality_routing"]["status"] == "active"
        assert vision_payload["optimization_attribution"]["modality_routing"]["metrics"] == {
            "request_modality": "text_only",
            "modality_path": "multimodal_default",
            "runtime_affinity": "mlx_vision",
        }
        assert (
            "No safe text-only runtime"
            in vision_payload["optimization_attribution"]["modality_routing"]["detail"]
        )

        audio_payload = _telemetry_benchmark_payload(
            services=attachment_services,
            model_id=audio_model_id,
            capability=CapabilityName.AUDIO_TRANSCRIPTION.value,
        )
        audio_features = ("multimodal_encoder_caching",)
        audio_scenarios = ("multimodal_encoder_reuse",)
        emit_benchmark_case_report(
            label="suite-multimodal-audio",
            payload=audio_payload,
            feature_names=audio_features,
            scenario_names=audio_scenarios,
        )
        suite_entries.append(("suite-multimodal-audio", audio_payload, audio_features, audio_scenarios))
        assert _feature(audio_payload, "multimodal_encoder_caching")["supported"] is True
        assert _scenario(audio_payload, "multimodal_encoder_reuse")["status"] == "observed"
    finally:
        attachment_services.close()

    draft_dir = temp_settings.models_dir[0] / "qwen2.5-0.5b-instruct-draft-mlx"
    draft_dir.mkdir(parents=True, exist_ok=True)
    (draft_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "max_position_embeddings": 32768}),
        encoding="utf-8",
    )
    (draft_dir / "weights.safetensors").write_bytes(b"draft-weights")
    (draft_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    speculation_settings = temp_settings.with_updates(
        speculative_decoding_enabled=True,
        speculative_decoding_num_draft_tokens=2,
    )
    speculation_runtime = SpeculationDiagnosticRuntime(settings=speculation_settings)
    speculation_services = bootstrap_services(
        speculation_settings,
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: speculation_runtime,
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )
    try:
        manifests = speculation_services.model_registry.scan().manifests
        model_id = next(
            manifest.model_id
            for manifest in manifests
            if manifest.display_name == "qwen2.5-1.5b-instruct-mlx"
        )
        speculation_payload = _telemetry_benchmark_payload(
            services=speculation_services,
            model_id=model_id,
        )
        speculation_features = ("speculative_decoding",)
        speculation_scenarios = ("speculation_selection",)
        emit_benchmark_case_report(
            label="suite-draft-speculation",
            payload=speculation_payload,
            feature_names=speculation_features,
            scenario_names=speculation_scenarios,
        )
        suite_entries.append(("suite-draft-speculation", speculation_payload, speculation_features, speculation_scenarios))
        assert _feature(speculation_payload, "speculative_decoding")["active"] is True
        assert _scenario(speculation_payload, "speculation_selection")["metrics"]["selected_mode"] == "draft_model"
    finally:
        speculation_services.close()

    prompt_lookup_runtime = PromptLookupDiagnosticRuntime()
    prompt_lookup_settings = temp_settings.with_updates(
        prompt_lookup_speculation_enabled=True,
        prompt_lookup_max_ngram_size=4,
        prompt_lookup_num_pred_tokens=12,
    )
    prompt_lookup_services = bootstrap_services(
        prompt_lookup_settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: prompt_lookup_runtime},
    )
    try:
        manifests = prompt_lookup_services.model_registry.scan().manifests
        model_id = next(manifest.model_id for manifest in manifests if manifest.format_type.value == "gguf")
        prompt_lookup_payload = _telemetry_benchmark_payload(
            services=prompt_lookup_services,
            model_id=model_id,
        )
        prompt_lookup_features = ("prompt_lookup_speculation",)
        prompt_lookup_scenarios = ("speculation_selection",)
        emit_benchmark_case_report(
            label="suite-prompt-lookup",
            payload=prompt_lookup_payload,
            feature_names=prompt_lookup_features,
            scenario_names=prompt_lookup_scenarios,
        )
        suite_entries.append(("suite-prompt-lookup", prompt_lookup_payload, prompt_lookup_features, prompt_lookup_scenarios))
        assert _feature(prompt_lookup_payload, "prompt_lookup_speculation")["active"] is True
        assert _scenario(prompt_lookup_payload, "speculation_selection")["metrics"]["selected_mode"] == "prompt_lookup"
    finally:
        prompt_lookup_services.close()

    acceleration_model_dir = temp_settings.models_dir[0] / "benchmark-acceleration-mlx"
    acceleration_model_dir.mkdir(parents=True, exist_ok=True)
    (acceleration_model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "max_position_embeddings": 8192}),
        encoding="utf-8",
    )
    (acceleration_model_dir / "weights.safetensors").write_bytes(b"mlx-weights")
    (acceleration_model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    acceleration_settings = temp_settings.with_updates(
        mlx_graph_compile_enabled=True,
        mlx_attention_kernel_mode="flash_attention",
    )
    acceleration_runtime = AccelerationDiagnosticRuntime(settings=acceleration_settings)
    acceleration_services = bootstrap_services(
        acceleration_settings,
        runtime_overrides={RuntimeAffinity.MLX_TEXT: acceleration_runtime},
    )
    try:
        manifests = acceleration_services.model_registry.scan().manifests
        model_id = next(
            manifest.model_id
            for manifest in manifests
            if manifest.source_path == str(acceleration_model_dir)
        )
        acceleration_payload = _telemetry_benchmark_payload(
            services=acceleration_services,
            model_id=model_id,
        )
        acceleration_features = ("graph_compilation", "attention_kernel_acceleration")
        acceleration_scenarios = ("mlx_acceleration_paths",)
        emit_benchmark_case_report(
            label="suite-mlx-acceleration",
            payload=acceleration_payload,
            feature_names=acceleration_features,
            scenario_names=acceleration_scenarios,
        )
        suite_entries.append(("suite-mlx-acceleration", acceleration_payload, acceleration_features, acceleration_scenarios))
        assert _feature(acceleration_payload, "graph_compilation")["active"] is True
        assert _scenario(acceleration_payload, "mlx_acceleration_paths")["status"] == "observed"
    finally:
        acceleration_services.close()

    ssm_dir = temp_settings.models_dir[0] / "hybrid-mamba-mlx"
    ssm_dir.mkdir(parents=True, exist_ok=True)
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
    moe_dir.mkdir(parents=True, exist_ok=True)
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

    frontier_settings = temp_settings.with_updates(
        moe_bounded_memory_mode="expert_streaming",
        moe_resident_expert_count=8,
    )
    frontier_services = bootstrap_services(
        frontier_settings,
        runtime_overrides={RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(settings=frontier_settings)},
    )
    try:
        manifests = frontier_services.model_registry.scan().manifests
        ssm_model_id = next(manifest.model_id for manifest in manifests if manifest.source_path == str(ssm_dir))
        moe_model_id = next(manifest.model_id for manifest in manifests if manifest.source_path == str(moe_dir))

        ssm_payload = _telemetry_benchmark_payload(
            services=frontier_services,
            model_id=ssm_model_id,
        )
        ssm_features = ("hybrid_ssm_routing", "ssm_state_cache_handling")
        ssm_scenarios = ("frontier_architecture_modes",)
        emit_benchmark_case_report(
            label="suite-frontier-hybrid-ssm",
            payload=ssm_payload,
            feature_names=ssm_features,
            scenario_names=ssm_scenarios,
        )
        suite_entries.append(("suite-frontier-hybrid-ssm", ssm_payload, ssm_features, ssm_scenarios))
        assert _feature(ssm_payload, "hybrid_ssm_routing")["supported"] is True
        assert _scenario(ssm_payload, "frontier_architecture_modes")["status"] == "observed"

        moe_payload = _telemetry_benchmark_payload(
            services=frontier_services,
            model_id=moe_model_id,
        )
        moe_features = ("moe_bounded_memory_serving",)
        moe_scenarios = ("frontier_architecture_modes",)
        emit_benchmark_case_report(
            label="suite-frontier-moe",
            payload=moe_payload,
            feature_names=moe_features,
            scenario_names=moe_scenarios,
        )
        suite_entries.append(("suite-frontier-moe", moe_payload, moe_features, moe_scenarios))
        assert _feature(moe_payload, "moe_bounded_memory_serving")["supported"] is True
        assert _scenario(moe_payload, "frontier_architecture_modes")["status"] == "observed"
    finally:
        frontier_services.close()

    emit_benchmark_suite_report(suite_entries)
    assert len(suite_entries) == 8


def test_cli_benchmark_compare_direct_diagnostics_surface_profile_metrics(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    scan_payload = _scan_payload(
        settings=temp_settings,
        services=services_with_fake_runtime,
        capsys=capsys,
    )
    manifest = next(
        item
        for item in scan_payload["manifests"]
        if item["format_type"] == "huggingface"
    )

    def fake_convert(services, source_manifest, request):
        label = request.policy.value
        result_dir = tmp_path / label
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
        (result_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (result_dir / "weights.safetensors").write_bytes(b"1234" if label == "balanced" else b"12345678")
        benchmark_manifest = type(source_manifest).model_validate(
            {
                **source_manifest.model_dump(mode="json"),
                "model_id": f"{source_manifest.model_id}-{label}",
                "display_name": f"{source_manifest.display_name}-{label}",
                "source_path": str(result_dir),
                "format_type": "mlx",
                "conversion_status": "runnable",
                "runtime_affinity": ["mlx_text"],
            },
        )
        return (
            {
                "needed": True,
                "status": "completed",
                "request": request.model_dump(mode="json"),
                "profile_label": label,
                "cache_hit": False,
                "duration_seconds": 11.0 if label == "balanced" else 13.0,
                "result_path": str(result_dir),
                "job_id": f"job-{label}",
                "logs_tail": [f"converted-{label}"],
            },
            benchmark_manifest,
        )

    monkeypatch.setattr("lewlm.cli.main._convert_manifest_for_benchmark", fake_convert)
    monkeypatch.setattr(
        "lewlm.cli.main.benchmark_direct_chat_manifest",
        lambda model_manifest, *, prompt, max_tokens, warmup_run_count: {
            "model_id": model_manifest.model_id,
            "display_name": model_manifest.display_name,
            "runtime": "mlx_lm_direct",
            "prompt": prompt,
            "load_seconds": 3.0,
            "generate_seconds": 2.0,
            "total_seconds": 5.0,
            "output_text": "direct baseline",
            "usage": {"completion_tokens": 2},
        },
    )

    def fake_managed_benchmark(*, services, model_id, prompt, warmup_run_count):
        is_balanced = model_id.endswith("balanced")
        return {
            "status": "completed",
            "model_id": model_id,
            "runtime": "fake_mlx_semantic",
            "capability": "chat",
            "load_seconds": 1.0,
            "generate_seconds": 1.0 if is_balanced else 2.0,
            "total_seconds": 2.0 if is_balanced else 3.0,
            "ttft_seconds": 0.4 if is_balanced else 0.6,
            "completion_tokens_per_second": 11.0 if is_balanced else 8.0,
            "output_text": "hello there" if is_balanced else "hello world",
            "usage": {"completion_tokens": 2},
            "serving_profile": (
                {
                    "status": "selected",
                    "profile_id": "profile-balanced",
                    "accepted_settings": {"prefill_token_batch_size": 256},
                    "rejected_settings": {},
                    "effective_settings": {"prefill_token_batch_size": 256},
                    "reason": "Applied the persisted serving-profile overrides selected by autotuning.",
                }
                if is_balanced
                else {
                    "status": "runtime_mismatch",
                    "profile_id": "profile-max-quality",
                    "accepted_settings": {},
                    "rejected_settings": {
                        "mlx_attention_kernel_mode": {
                            "requested_value": "flash_attention",
                            "reason": "Profile was benchmarked against runtime `mlx_text`, but this request routed elsewhere.",
                        },
                    },
                    "effective_settings": {"prefill_token_batch_size": 512},
                    "reason": "Persisted profile runtime does not match the active routed runtime.",
                }
            ),
        }

    monkeypatch.setattr("lewlm.cli.main._run_managed_benchmark_once", fake_managed_benchmark)

    payload = _benchmark_payload(
        settings=temp_settings,
        services=services_with_fake_runtime,
        capsys=capsys,
        args=[
            "--model",
            manifest["model_id"],
            "--compare-direct",
            "--convert-missing",
            "--convert-policy",
            "balanced",
            "--convert-policy",
            "max_quality",
        ],
    )

    assert payload["benchmark_type"] == "direct_chat_comparison"
    assert payload["benchmark_count"] == 2
    assert payload["model_count"] == 1
    assert payload["comparison_controls"]["warmup_run_count"] == 1
    assert payload["summary"]["converted_model_count"] == 2
    assert payload["summary"]["conversion_total_seconds"] == 24.0
    assert payload["summary"]["metric_summaries"]["warm_total_seconds"]["status"] == "completed"

    profiles = {
        item["conversion"]["profile_label"]: item["profile_metrics"]
        for item in payload["models"][0]["profiles"]
    }
    assert profiles["balanced"]["quality_proxy"]["reference_profile"] == "max_quality"
    assert profiles["balanced"]["quality_proxy"]["exact_match"] is False
    assert profiles["max_quality"]["quality_proxy"]["exact_match"] is True
    assert profiles["balanced"]["model_size_bytes"] is not None
    assert profiles["max_quality"]["model_size_bytes"] is not None
    assert profiles["max_quality"]["model_size_bytes"] > profiles["balanced"]["model_size_bytes"]
    assert profiles["balanced"]["serving_profile_compatibility"]["classification"] == "fully_supported"
    recommendation = payload["models"][0]["profile_recommendation"]
    assert recommendation["status"] == "recommended"
    assert recommendation["profile_label"] == "balanced"
    assert recommendation["supporting_metrics"]["serving_profile_compatibility"]["classification"] == "fully_supported"


def test_cli_autotune_diagnostics_surface_candidate_metrics(
    temp_settings,
    services_with_fake_attachment_runtime,
    capsys,
) -> None:
    scan_payload = _scan_payload(
        settings=temp_settings,
        services=services_with_fake_attachment_runtime,
        capsys=capsys,
    )
    manifest = _manifest_by_display_name(scan_payload, "qwen2.5-1.5b-instruct-mlx")

    payload = _autotune_payload(
        settings=temp_settings,
        services=services_with_fake_attachment_runtime,
        capsys=capsys,
        model_id=manifest["model_id"],
    )

    assert payload["artifact"]["artifact_path"]
    assert any(item["continuous_batching_throughput"] is not None for item in payload["candidate_summaries"])
    assert "prefix_cache" in payload["active_cache_features"]
    assert "persistent_multi_context_cache" in payload["active_cache_features"]
    assert any(item["name"] == "batching_disabled" for item in payload["candidate_summaries"])
    assert any(item["name"].startswith("prefill_") for item in payload["candidate_summaries"])


def test_cli_benchmark_reuses_persisted_serving_profile_and_reports_it(
    temp_settings,
    sample_chat_models_root,
    capsys,
) -> None:
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_TEXT: ServingProfileBenchmarkRuntime(settings=temp_settings),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
            RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
        },
    )
    scan_payload = _scan_payload(
        settings=temp_settings,
        services=services,
        capsys=capsys,
    )
    manifest = _manifest_by_display_name(scan_payload, "qwen2.5-1.5b-instruct-mlx")
    recommendation = asyncio.run(
        services.telemetry_service.autotune(
            model_id=manifest["model_id"],
            prompt="Serving profile benchmark probe",
        ),
    )

    exit_code = main(["benchmark", "--model", manifest["model_id"]], settings=temp_settings, services=services)
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert f"serving profile: {recommendation.profile_id}" in stdout
    assert "workload=text_only" in stdout
    assert "prefill_token_batch_size=256" in stdout

    payload = _benchmark_payload(
        settings=temp_settings,
        services=services,
        capsys=capsys,
        args=["--model", manifest["model_id"]],
    )
    assert payload["serving_profile"]["profile_id"] == recommendation.profile_id
    assert payload["workload_class"] == "text_only"
    assert payload["serving_profile"]["workload_class"] == "text_only"
    assert payload["serving_profile"]["effective_settings"]["prefill_token_batch_size"] == 256
    assert payload["output_text"].endswith("[prefill=256]")

    disabled_payload = _benchmark_payload(
        settings=temp_settings,
        services=services,
        capsys=capsys,
        args=["--model", manifest["model_id"], "--disable-serving-profile"],
    )
    assert disabled_payload["serving_profile"]["status"] == "disabled"
    assert disabled_payload["output_text"].endswith("[prefill=512]")
    services.close()


def test_cli_benchmark_supports_multimodal_workload_class_profiles(
    temp_settings,
    sample_chat_models_root,
    capsys,
) -> None:
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_TEXT: ServingProfileBenchmarkRuntime(settings=temp_settings),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
            RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
        },
    )
    scan_payload = _scan_payload(
        settings=temp_settings,
        services=services,
        capsys=capsys,
    )
    manifest = _manifest_by_display_name(scan_payload, "qwen2-vl-vision-mlx")
    recommendation = asyncio.run(
        services.telemetry_service.autotune(
            model_id=manifest["model_id"],
            prompt="Serving profile multimodal benchmark probe",
            workload_class="single_image",
        ),
    )

    exit_code = main(
        ["benchmark", "--model", manifest["model_id"], "--workload-class", "single_image"],
        settings=temp_settings,
        services=services,
    )
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "workload class: single_image" in stdout
    assert f"serving profile: {recommendation.profile_id}" in stdout
    assert "workload=single_image" in stdout

    payload = _benchmark_payload(
        settings=temp_settings,
        services=services,
        capsys=capsys,
        args=["--model", manifest["model_id"], "--workload-class", "single_image"],
    )
    assert payload["runtime"] == "fake_mlx_vision"
    assert payload["workload_class"] == "single_image"
    assert payload["serving_profile"]["profile_id"] == recommendation.profile_id
    assert payload["serving_profile"]["workload_class"] == "single_image"
    services.close()
