from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from conftest import FakeLlamaCppRuntime, FakeMLXSemanticRuntime
from lewlm.conversion.models import CONVERSION_OUTPUT_METADATA_FILENAME, ConversionArtifactRecord, ConversionPolicy
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.errors import ConfigurationError
from lewlm.cli.main import (
    _annotate_profile_metrics,
    _benchmark_conversion_requests,
    _conversion_quantization_profile_from_args,
    _filter_benchmark_stderr,
    _persist_external_adapter_runtime_preference,
    _print_benchmark_table,
    _print_benchmark_scenario_highlights,
    main,
)
from lewlm.core.contracts import (
    ConversionStatus,
    GenerateResponse,
    HostPlatformSnapshot,
    ModelArtifactLayer,
    ModelArtifactRole,
    ModelFormat,
    ModelInventory,
    ModelModality,
    RuntimeAffinity,
)


class PromptGuidedLlamaRuntime(FakeLlamaCppRuntime):
    def structured_output_runtime_status(self, contract):
        status = super().structured_output_runtime_status(contract)
        if status is None:
            return None
        status.enforcement = "prompt_guided"
        status.decoder_enforced = False
        status.fallback_used = True
        status.fallback_reason = "Fake runtime preserves structured-output requests without decoder enforcement."
        return status

    async def _generate(self, request) -> GenerateResponse:
        prompt = request.messages[-1].content if request.messages else ""
        return GenerateResponse(
            model_id=request.model_id,
            output_text=f"Echo: {prompt}",
            finish_reason="stop",
            usage={"prompt_tokens": len(request.messages), "completion_tokens": 2, "total_tokens": len(request.messages) + 2},
        )


def test_cli_scan_emits_json_summary(temp_settings, sample_models_root: Path, capsys) -> None:
    exit_code = main(["scan", "--json"], settings=temp_settings)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["discovered_count"] == 3


def test_cli_list_models_emits_registered_inventory(temp_settings, sample_models_root: Path, capsys) -> None:
    main(["scan"], settings=temp_settings)
    capsys.readouterr()

    exit_code = main(["list-models", "--json"], settings=temp_settings)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["count"] == 3


def test_cli_list_models_groups_related_artifacts_in_default_view(
    temp_settings,
    services_with_fake_runtime_and_conversion,
    monkeypatch,
    capsys,
) -> None:
    services_with_fake_runtime_and_conversion.model_registry.scan()
    source_manifest = next(
        manifest
        for manifest in services_with_fake_runtime_and_conversion.model_registry.list_manifests()
        if manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION
    )
    source_layer = ModelArtifactLayer(
        artifact_key="source_bundle",
        role=ModelArtifactRole.SOURCE_BUNDLE,
        display_name=source_manifest.display_name,
        format_type=source_manifest.format_type,
        source_path=source_manifest.source_path,
        modality=source_manifest.modality,
        runtime_affinity=source_manifest.runtime_affinity,
        tokenizer_path=source_manifest.tokenizer_path,
        processor_path=source_manifest.processor_path,
        quantization=source_manifest.quantization,
        quantization_profile=source_manifest.quantization_profile,
    )
    converted_root = temp_settings.cache_dir / "conversions" / "grouped-demo"
    converted_multimodal = source_manifest.model_copy(
        update={
            "model_id": f"{source_manifest.model_id}-multimodal",
            "display_name": f"{source_manifest.display_name} (multimodal mlx)",
            "source_path": str(converted_root / "vision"),
            "format_type": ModelFormat.MLX,
            "conversion_status": ConversionStatus.RUNNABLE,
            "runtime_affinity": (RuntimeAffinity.MLX_VISION,),
            "artifact_key": "multimodal",
            "artifact_role": ModelArtifactRole.MULTIMODAL_RUNNABLE,
            "artifact_family_id": "family-demo",
            "artifact_lineage": [source_layer],
        },
    )
    converted_text = source_manifest.model_copy(
        update={
            "model_id": f"{source_manifest.model_id}-text",
            "display_name": f"{source_manifest.display_name} (text mlx)",
            "source_path": str(converted_root / "text"),
            "format_type": ModelFormat.MLX,
            "modality": (ModelModality.TEXT,),
            "conversion_status": ConversionStatus.RUNNABLE,
            "runtime_affinity": (RuntimeAffinity.MLX_TEXT,),
            "artifact_key": "text",
            "artifact_role": ModelArtifactRole.TEXT_RUNNABLE,
            "artifact_family_id": "family-demo",
            "artifact_lineage": [source_layer],
        },
    )
    inventory = ModelInventory(count=3, items=[source_manifest, converted_multimodal, converted_text])
    monkeypatch.setattr(services_with_fake_runtime_and_conversion.model_registry, "inventory", lambda: inventory)

    exit_code = main(["list-models"], settings=temp_settings, services=services_with_fake_runtime_and_conversion)
    captured = capsys.readouterr()

    assert exit_code == 0
    output_lines = captured.out.splitlines()
    assert output_lines[0] == converted_text.display_name
    assert "  model: mlx [" in captured.out
    assert "  source: " not in captured.out
    assert "other runnable variants:" in captured.out
    assert "multimodal runnable:" in captured.out
    assert "text runnable:" not in captured.out
    assert f"  source path: {source_manifest.source_path}" in captured.out


def test_cli_list_models_hides_source_entry_when_standalone_conversion_exists(
    temp_settings,
    services_with_fake_runtime_and_conversion,
    monkeypatch,
    capsys,
) -> None:
    services_with_fake_runtime_and_conversion.model_registry.scan()
    source_manifest = next(
        manifest
        for manifest in services_with_fake_runtime_and_conversion.model_registry.list_manifests()
        if manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION
    )
    converted_manifest = source_manifest.model_copy(
        update={
            "model_id": "phi-3-mini-hf_converted",
            "display_name": f"{source_manifest.display_name} (converted)",
            "source_path": str(temp_settings.cache_dir / "conversions" / "phi-3-mini-hf"),
            "format_type": ModelFormat.MLX,
            "conversion_status": ConversionStatus.RUNNABLE,
            "runtime_affinity": (RuntimeAffinity.MLX_TEXT,),
            "metadata": {
                **source_manifest.metadata,
                "converted_output": True,
                "source_display_name": source_manifest.display_name,
                "source_model_id": source_manifest.model_id,
            },
        },
    )
    inventory = ModelInventory(count=2, items=[source_manifest, converted_manifest])
    monkeypatch.setattr(services_with_fake_runtime_and_conversion.model_registry, "inventory", lambda: inventory)

    exit_code = main(["list-models"], settings=temp_settings, services=services_with_fake_runtime_and_conversion)
    captured = capsys.readouterr()

    assert exit_code == 0
    output_lines = captured.out.splitlines()
    assert output_lines[0] == converted_manifest.display_name
    assert f"  use: {converted_manifest.model_id}" in captured.out
    assert "other runnable variants:" not in captured.out
    assert f"  path: {converted_manifest.source_path}" in captured.out
    assert f"  source path: {source_manifest.source_path}" in captured.out


def test_cli_list_models_all_preserves_raw_artifact_view(temp_settings, sample_models_root: Path, capsys) -> None:
    main(["scan", "--json"], settings=temp_settings)
    scan_payload = json.loads(capsys.readouterr().out)

    exit_code = main(["list-models", "--all"], settings=temp_settings)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert any(f"{manifest['model_id']}:" in captured.out for manifest in scan_payload["manifests"])


def test_cli_capabilities_accepts_hashless_model_selector(temp_settings, sample_models_root: Path, capsys) -> None:
    main(["scan", "--json"], settings=temp_settings)
    scan_payload = json.loads(capsys.readouterr().out)
    gguf_manifest = next(
        manifest
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    )
    selector = re.sub(r"[^a-z0-9]+", "-", gguf_manifest["display_name"].casefold()).strip("-")

    exit_code = main(["capabilities", selector, "--json"], settings=temp_settings)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["model_id"] == gguf_manifest["model_id"]


def test_cli_capabilities_emits_model_report(temp_settings, sample_models_root: Path, capsys) -> None:
    main(["scan", "--json"], settings=temp_settings)
    scan_payload = json.loads(capsys.readouterr().out)
    gguf_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    )

    exit_code = main(["capabilities", gguf_model_id, "--json"], settings=temp_settings)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["model_id"] == gguf_model_id
    assert payload["host_platform"]["system"]
    assert payload["target_platforms"]
    assert any(item["capability"] == "chat" for item in payload["capabilities"])


def test_cli_capabilities_explain_disabled_runtime_pack(temp_settings, sample_models_root: Path, capsys) -> None:
    disabled_settings = temp_settings.with_updates(disabled_runtime_packs=("llamacpp",))
    main(["scan", "--json"], settings=disabled_settings)
    scan_payload = json.loads(capsys.readouterr().out)
    gguf_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    )

    exit_code = main(["capabilities", gguf_model_id, "--json"], settings=disabled_settings)
    payload = json.loads(capsys.readouterr().out)
    runtime_candidate = next(
        item
        for item in payload["runtime_candidates"]
        if item["runtime_affinity"] == RuntimeAffinity.LLAMACPP.value
    )

    assert exit_code == 0
    assert runtime_candidate["registered"] is False
    assert runtime_candidate["availability_reason"] == "Runtime pack `llamacpp` is disabled."


def test_cli_benchmark_emits_persisted_record(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0

    gguf_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    )

    exit_code = main(
        ["benchmark", "--model", gguf_model_id, "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["model_id"] == gguf_model_id
    assert payload["benchmark_id"]
    assert payload["created_at"]
    assert payload["artifact"]["artifact_path"]
    assert payload["regression"]["status"] == "no_baseline"
    assert all(item["scenario"] != "repeated_prefix" for item in payload["scenarios"])
    assert all(item["scenario"] != "continuous_batching" for item in payload["scenarios"])
    assert all(item["scenario"] != "warm_chat_cache" for item in payload["scenarios"])
    performance_features = {item["feature"]: item for item in payload["performance_features"]}
    assert performance_features["continuous_batching"]["supported"] is True
    assert performance_features["request_scheduling_and_backpressure"]["supported"] is True
    assert performance_features["model_load_admission_control"]["supported"] is True
    assert performance_features["prefix_cache"]["supported"] is True
    assert performance_features["prefix_cache"]["active"] is True
    assert performance_features["prefix_cache"]["metrics"]["cache_saves"] >= 1


def test_cli_capabilities_include_measured_registry_after_benchmark(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0
    gguf_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    )

    benchmark_code = main(
        ["benchmark", "--model", gguf_model_id, "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    capsys.readouterr()
    assert benchmark_code == 0

    exit_code = main(["capabilities", gguf_model_id, "--json"], settings=temp_settings, services=services_with_fake_runtime)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    measured = {item["category"]: item for item in payload["measured_capabilities"]}
    assert measured["batching"]["status"] == "supported"
    assert measured["cache_reuse"]["status"] == "supported"
    assert measured["constrained_decoding"]["status"] == "supported"


def test_cli_benchmark_prints_multimodal_encoder_scenario_summary(
    capsys,
) -> None:
    _print_benchmark_scenario_highlights(
        {
            "scenarios": [
                {
                    "scenario": "multimodal_encoder_reuse",
                    "status": "observed",
                    "metrics": {
                        "multimodal_encoder_cache_hit_delta": 3,
                        "multimodal_encoder_cache_miss_delta": 2,
                        "multimodal_feature_cache_hit_delta": 2,
                        "average_second_over_first_ratio": 0.5,
                        "average_encoder_advantage_seconds": 0.1,
                    },
                }
            ]
        }
    )
    output = capsys.readouterr().out

    assert "scenarios:" in output
    assert "multimodal_encoder_reuse" in output
    assert "encoder_hits=" in output
    assert "encoder_misses=" in output


def test_cli_benchmark_prints_mlx_acceleration_scenario_summary(
    capsys,
) -> None:
    _print_benchmark_scenario_highlights(
        {
            "scenarios": [
                {
                    "scenario": "mlx_acceleration_paths",
                    "status": "observed",
                    "metrics": {
                        "sample_count": 1,
                        "compiled_sample_count": 1,
                        "fallback_sample_count": 1,
                        "kernel_paths": "flash_attention",
                        "average_accelerated_over_stock_ratio": 0.4,
                        "average_time_saved_seconds": 0.02,
                    },
                    "samples": [
                        {
                            "metrics": {
                                "fallback_reason": "RuntimeError: flash attention rejected prompt length",
                            }
                        }
                    ],
                }
            ]
        }
    )
    output = capsys.readouterr().out

    assert "scenarios:" in output
    assert "mlx_acceleration_paths" in output
    assert "kernels=flash_attention" in output
    assert "compiled=1/1" in output
    assert "saved=0.02s" in output
    assert "fallback_reason=RuntimeError: flash attention rejected prompt length" in output


def test_cli_benchmark_all_emits_suite_payload(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
    capsys,
) -> None:
    second_model_path = temp_settings.models_dir[0] / "mistral-7b-instruct-q4_k_m.gguf"
    second_model_path.write_bytes(b"gguf-model-2")
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0
    gguf_model_ids = [
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    ]
    assert len(gguf_model_ids) == 2
    monkeypatch.setattr(
        services_with_fake_runtime.telemetry_service,
        "_benchmark_candidate_model_ids",
        lambda *, capability: list(gguf_model_ids),
    )
    expected_model_count = len(gguf_model_ids)

    exit_code = main(
        ["benchmark", "--all", "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["benchmark_count"] == expected_model_count
    assert payload["model_count"] == expected_model_count
    assert payload["repeat_count"] == 1
    assert len(payload["results"]) == expected_model_count
    assert len(payload["models"]) == expected_model_count
    assert payload["artifact"]["artifact_path"]
    assert payload["regression"]["status"] == "no_baseline"
    assert all(item["capability"] == "chat" for item in payload["results"])


def test_cli_benchmark_all_embeddings_emits_capability_suite_payload(
    temp_settings,
    services_with_fake_multimodal_runtime,
    capsys,
) -> None:
    second_embedding_dir = temp_settings.models_dir[0] / "gte-small-embed-mlx"
    second_embedding_dir.mkdir(parents=True)
    (second_embedding_dir / "config.json").write_text(
        json.dumps({"model_type": "gte", "max_position_embeddings": 8192}),
        encoding="utf-8",
    )
    (second_embedding_dir / "weights.safetensors").write_bytes(b"embed-weights-2")
    (second_embedding_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_multimodal_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0
    expected_model_count = sum(
        1
        for manifest in scan_payload["manifests"]
        if any(
            capability["capability"] == "embeddings" and capability["supported"] is True
            for capability in services_with_fake_multimodal_runtime.model_router.model_capability_report(
                manifest["model_id"],
            ).model_dump(mode="json")["capabilities"]
        )
    )
    assert expected_model_count >= 2

    exit_code = main(
        ["benchmark", "--all", "--capability", "embeddings", "--json"],
        settings=temp_settings,
        services=services_with_fake_multimodal_runtime,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["capability"] == "embeddings"
    assert payload["benchmark_count"] == expected_model_count
    assert payload["model_count"] == expected_model_count
    assert len(payload["results"]) == expected_model_count
    assert payload["repeat_count"] == 1
    assert all(item["runtime"] == "fake_mlx_semantic" for item in payload["results"])
    assert all(item["capability"] == "embeddings" for item in payload["results"])
    assert payload["artifact"]["artifact_path"]
    assert payload["scenarios"] == []
    performance_features = {item["feature"]: item for item in payload["performance_features"]}
    assert performance_features["continuous_batching"]["supported"] is True
    assert performance_features["disk_backed_cache"]["supported"] is True


def test_cli_doctor_prints_optimization_default_summary(
    temp_settings,
    services_with_fake_attachment_runtime,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_attachment_runtime)
    capsys.readouterr()
    assert scan_code == 0

    exit_code = main(["doctor"], settings=temp_settings, services=services_with_fake_attachment_runtime)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "optimization defaults:" in output
    assert "runtime strategy:" in output
    assert "non-apple=gguf_llamacpp" in output
    assert "recommended feature paths:" in output
    assert "structured output:" in output
    assert "default " in output
    assert "workloads:" in output


def test_cli_config_prints_default_path_context(temp_settings, capsys) -> None:
    exit_code = main(["config"], settings=temp_settings)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"home dir: {Path.home().resolve(strict=False)}" in output
    assert f"default data dir: {Path.home().resolve(strict=False) / '.lewlm'}" in output
    assert f"default model root: {Path.home().resolve(strict=False) / '.lewlm' / 'models'}" in output
    assert f"data dir: {temp_settings.data_dir}" in output


def test_cli_doctor_prints_default_path_context(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    exit_code = main(["doctor"], settings=temp_settings, services=services_with_fake_runtime)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"home dir: {Path.home().resolve(strict=False)}" in output
    assert f"default data dir: {Path.home().resolve(strict=False) / '.lewlm'}" in output
    assert f"default model root: {Path.home().resolve(strict=False) / '.lewlm' / 'models'}" in output
    assert f"data dir: {temp_settings.data_dir}" in output


def test_cli_doctor_reports_host_memory_diagnostics(
    temp_settings,
    services_with_fake_attachment_runtime,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        services_with_fake_attachment_runtime.runtime_catalog,
        "host_platform_snapshot",
        lambda: HostPlatformSnapshot(
            system="Windows",
            release="11",
            machine="AMD64",
            python_version="3.11.9",
            total_memory_mb=None,
            total_memory_reason="Windows GlobalMemoryStatusEx failed.",
        ),
    )

    exit_code = main(["doctor"], settings=temp_settings, services=services_with_fake_attachment_runtime)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "host memory: unavailable (Windows GlobalMemoryStatusEx failed.)" in output
    assert "install guidance:" in output


def test_cli_doctor_json_includes_multimodal_workload_defaults(
    temp_settings,
    services_with_fake_attachment_runtime,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_attachment_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0
    vision_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["display_name"] == "qwen2-vl-vision-mlx"
    )
    text_profile = asyncio.run(
        services_with_fake_attachment_runtime.telemetry_service.autotune(
            model_id=vision_model_id,
            prompt="Doctor multimodal text defaults",
            workload_class="text_only_multimodal",
        )
    )
    image_profile = asyncio.run(
        services_with_fake_attachment_runtime.telemetry_service.autotune(
            model_id=vision_model_id,
            prompt="Doctor multimodal image defaults",
            workload_class="single_image",
        )
    )

    exit_code = main(["doctor", "--json"], settings=temp_settings, services=services_with_fake_attachment_runtime)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    defaults = payload["runtime_stats"]["optimization_defaults"]["models"]
    model_defaults = next(item for item in defaults if item["model_id"] == vision_model_id)
    workload_defaults = {item["workload_class"]: item for item in model_defaults["workload_defaults"]}

    assert model_defaults["default_workload_class"] == "text_only_multimodal"
    assert workload_defaults["text_only_multimodal"]["runtime"]
    assert workload_defaults["text_only_multimodal"]["profile_id"] == text_profile.profile_id
    assert workload_defaults["text_only_multimodal"]["profile_status"] == "selected"
    assert workload_defaults["single_image"]["runtime"] == "fake_mlx_vision"
    assert workload_defaults["single_image"]["profile_id"] == image_profile.profile_id
    assert workload_defaults["single_image"]["profile_status"] == "selected"
    assert workload_defaults["frame_bundle_video"]["runtime"] == "fake_mlx_vision"
    assert model_defaults["decisions"]["multimodal_default_selection"]["status"] == "deferred"
    assert model_defaults["decisions"]["multimodal_default_selection"]["benchmark_backed"] is False


def test_cli_doctor_marks_multimodal_default_selection_benchmark_backed_when_all_workloads_are_profiled(
    temp_settings,
    services_with_fake_attachment_runtime,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_attachment_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0
    vision_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["display_name"] == "qwen2-vl-vision-mlx"
    )
    for workload_class, prompt in (
        ("text_only_multimodal", "Doctor multimodal text defaults"),
        ("single_image", "Doctor multimodal image defaults"),
        ("repeated_image", "Doctor multimodal repeated image defaults"),
        ("frame_bundle_video", "Doctor multimodal bundle defaults"),
        ("audio_conditioned", "Doctor multimodal audio defaults"),
    ):
        asyncio.run(
            services_with_fake_attachment_runtime.telemetry_service.autotune(
                model_id=vision_model_id,
                prompt=prompt,
                workload_class=workload_class,
            )
        )

    exit_code = main(["doctor", "--json"], settings=temp_settings, services=services_with_fake_attachment_runtime)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    defaults = payload["runtime_stats"]["optimization_defaults"]["models"]
    model_defaults = next(item for item in defaults if item["model_id"] == vision_model_id)
    decision = model_defaults["decisions"]["multimodal_default_selection"]
    workload_defaults = {item["workload_class"]: item for item in model_defaults["workload_defaults"]}

    assert decision["status"] == "adopted"
    assert decision["benchmark_backed"] is True
    assert sorted(decision["metrics"]["benchmark_backed_workloads"]) == [
        "audio_conditioned",
        "frame_bundle_video",
        "repeated_image",
        "single_image",
        "text_only_multimodal",
    ]
    assert all(item["profile_status"] == "selected" for item in workload_defaults.values())


def test_cli_chat_routes_image_requests_through_external_bridge(
    temp_settings,
    services_with_fake_external_vision_runtime,
    sample_attachment_sources,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_external_vision_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0
    vision_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["display_name"] == "qwen2-vl-vision-mlx"
    )

    exit_code = main(
        [
            "chat",
            "Describe the attached image",
            "--model",
            vision_model_id,
            "--attach-image",
            str(sample_attachment_sources["image_one"]),
            "--json",
        ],
        settings=temp_settings,
        services=services_with_fake_external_vision_runtime,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["output_text"].startswith("Adapter vision echo:")
    bridge_runtime = services_with_fake_external_vision_runtime.runtime_catalog.get_runtime(RuntimeAffinity.EXTERNAL_ACCELERATOR)
    assert bridge_runtime is not None
    user_message = next(
        message for message in bridge_runtime.captured_requests[-1].messages if message.role == "user" and message.attachments
    )
    assert user_message.attachments[0].name == sample_attachment_sources["image_one"].name


def test_cli_benchmark_all_repeat_emits_multi_pass_suite_payload(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
    capsys,
) -> None:
    second_model_path = temp_settings.models_dir[0] / "mistral-7b-instruct-q4_k_m.gguf"
    second_model_path.write_bytes(b"gguf-model-2")
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0
    gguf_model_ids = [
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    ]
    assert len(gguf_model_ids) == 2
    monkeypatch.setattr(
        services_with_fake_runtime.telemetry_service,
        "_benchmark_candidate_model_ids",
        lambda *, capability: list(gguf_model_ids),
    )
    expected_model_count = len(gguf_model_ids)

    exit_code = main(
        ["benchmark", "--all", "--repeat", "2", "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["benchmark_count"] == expected_model_count * 2
    assert payload["model_count"] == expected_model_count
    assert payload["repeat_count"] == 2
    assert len(payload["results"]) == expected_model_count * 2
    assert len(payload["models"]) == expected_model_count
    assert payload["artifact"]["artifact_path"]
    assert all(item["run_count"] == 2 for item in payload["models"])
    performance_features = {item["feature"]: item for item in payload["performance_features"]}
    assert performance_features["balanced_residency_mode"]["supported"] is True
    assert "speculative_decoding" in performance_features


def test_cli_autotune_emits_persisted_serving_profile(
    temp_settings,
    services_with_fake_attachment_runtime,
    capsys,
) -> None:
    text_model_dir = temp_settings.models_dir[0] / "qwen2.5-1.5b-instruct-mlx"
    text_model_dir.mkdir(parents=True, exist_ok=True)
    (text_model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "max_position_embeddings": 32768}),
        encoding="utf-8",
    )
    (text_model_dir / "weights.safetensors").write_bytes(b"mlx-text-weights")
    (text_model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_attachment_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0
    text_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["display_name"] == "qwen2.5-1.5b-instruct-mlx"
    )

    exit_code = main(
        ["autotune", "--model", text_model_id, "--json"],
        settings=temp_settings,
        services=services_with_fake_attachment_runtime,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["model_id"] == text_model_id
    assert payload["capability"] == "chat"
    assert payload["artifact"]["artifact_path"]
    assert payload["candidate_summaries"]
    assert any(item["name"] == "batching_disabled" for item in payload["candidate_summaries"])
    assert "runtime_policy" in payload["effective_settings"]


def test_cli_benchmark_compare_direct_emits_json_payload(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0

    gguf_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    )

    monkeypatch.setattr(
        "lewlm.cli.main.benchmark_direct_chat_manifest",
        lambda manifest, *, prompt, max_tokens, warmup_run_count: {
            "model_id": manifest.model_id,
            "display_name": manifest.display_name,
            "runtime": "llama_cpp_direct",
            "prompt": prompt,
            "load_seconds": 3.0,
            "generate_seconds": 2.0,
            "total_seconds": 5.0,
            "output_text": "baseline",
            "usage": {"completion_tokens": 2},
        },
    )

    exit_code = main(
        ["benchmark", "--model", gguf_model_id, "--compare-direct", "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["benchmark_type"] == "direct_chat_comparison"
    assert payload["model_count"] == 1
    assert payload["benchmark_count"] == 1
    assert payload["comparison_controls"]["primary_metric"] == "cold_total_seconds"
    assert payload["comparison_controls"]["warmup_run_count"] == 1
    assert payload["artifact"]["artifact_path"]
    assert payload["summary"]["completed_run_count"] == 1
    assert payload["summary"]["time_saved_seconds"] is not None
    assert payload["results"][0]["comparison"]["metric_summaries"]["warm_total_seconds"]["status"] == "completed"
    assert payload["results"][0]["direct"]["runtime"] == "llama_cpp_direct"
    assert payload["results"][0]["optimized"]["runtime"] == "fake_llamacpp"
    assert payload["results"][0]["optimized"]["phase_breakdown"]["ttft_seconds"] is not None
    assert payload["results"][0]["optimized"]["optimization_attribution"]["serving_profile_defaults"]["status"] == "active"
    assert payload["results"][0]["comparison"]["status"] == "completed"


def test_cli_benchmark_compare_direct_prints_human_summary(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
    capsys,
) -> None:
    main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    gguf_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    )

    monkeypatch.setattr(
        "lewlm.cli.main.benchmark_direct_chat_manifest",
        lambda manifest, *, prompt, max_tokens, warmup_run_count: {
            "model_id": manifest.model_id,
            "display_name": manifest.display_name,
            "runtime": "llama_cpp_direct",
            "prompt": prompt,
            "load_seconds": 4.0,
            "generate_seconds": 1.0,
            "total_seconds": 5.0,
            "output_text": "baseline",
            "usage": {},
        },
    )

    exit_code = main(
        ["benchmark", "--model", gguf_model_id, "--compare-direct"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "LewLM benchmark vs direct inference" in output
    assert "artifact:" in output
    assert "Model" in output
    assert "Convert" in output
    assert "Saved" in output
    assert "Evidence" not in output


def test_print_benchmark_table_aligns_ansi_styled_numeric_columns(capsys) -> None:
    _print_benchmark_table(
        headers=("Metric", "Saved", "Delta"),
        rows=[("cold total", "\033[32m+1.2500s\033[0m", "\033[31m-12.50%\033[0m")],
        alignments=("left", "right", "right"),
    )

    output_lines = capsys.readouterr().out.splitlines()
    stripped_row = (
        output_lines[2]
        .replace("\033[32m", "")
        .replace("\033[31m", "")
        .replace("\033[0m", "")
    )

    assert stripped_row == "cold total  +1.2500s  -12.50%"


def test_cli_benchmark_compare_external_adapter_prints_human_summary(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
    capsys,
) -> None:
    main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    model_id = next(manifest["model_id"] for manifest in scan_payload["manifests"])

    monkeypatch.setattr(
        "lewlm.cli.main._run_external_adapter_benchmark",
        lambda args, settings, services: {
            "status": "completed",
            "benchmark_type": "external_adapter_comparison",
            "model_id": model_id,
            "native": {"status": "completed", "runtime": "fake_mlx_semantic"},
            "external_adapter": {"status": "completed", "runtime": "local_external_adapter"},
            "comparison": {
                "status": "completed",
                "primary_metric": "warm_total_seconds",
                "winner": "external",
                "metric_summaries": {
                    "warm_total_seconds": {
                        "status": "completed",
                        "native": 0.9,
                        "external": 0.4,
                    },
                },
            },
            "feature_preservation": {
                "preserved": ["continuous_batching"],
                "degraded": ["prefill_optimization"],
                "rejected": ["kv_cache_quantization"],
            },
            "routing_preference": {
                "applied": False,
                "reason": (
                    "Persisted benchmark evidence, but LewLM keeps `fake_mlx_semantic` because "
                    "external accelerators remain a measured bridge path and do not replace the "
                    "first-class local runtime default on this host."
                ),
            },
            "artifact": {"artifact_path": "/tmp/external-adapter-benchmark.json"},
        },
    )

    exit_code = main(
        ["benchmark", "--model", model_id, "--compare-external-adapter"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "LewLM native vs local external adapter" in output
    assert "winner: external" in output
    assert "preserved: continuous_batching" in output
    assert "routing preference: Persisted benchmark evidence, but LewLM keeps" in output
    assert "first-class local runtime default" in output
    assert "artifact: /tmp/external-adapter-benchmark.json" in output


def test_cli_benchmark_compare_external_adapter_accepts_non_apple_profile(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
    capsys,
) -> None:
    settings = temp_settings.with_updates(external_accelerator_profile="vllm_local")
    main(["scan", "--json"], settings=settings, services=services_with_fake_runtime)
    model_id = next(manifest["model_id"] for manifest in json.loads(capsys.readouterr().out)["manifests"])

    def _fake_benchmark(args, benchmark_settings, services):
        assert benchmark_settings.external_accelerator_profile == "vllm_local"
        return {
            "status": "completed",
            "benchmark_type": "external_adapter_comparison",
            "model_id": model_id,
            "native": {"status": "completed", "runtime": "fake_mlx_semantic"},
            "external_adapter": {"status": "completed", "runtime": "local_external_adapter"},
            "comparison": {"status": "completed", "primary_metric": "warm_total_seconds", "winner": "external"},
            "feature_preservation": {"preserved": ["continuous_batching"], "degraded": [], "rejected": []},
            "routing_preference": {
                "applied": True,
                "selected_runtime_name": "local_external_adapter",
                "selected_runtime_affinity": "external_accelerator",
            },
            "artifact": {"artifact_path": "/tmp/external-adapter-benchmark.json"},
        }

    monkeypatch.setattr("lewlm.cli.main._run_external_adapter_benchmark", _fake_benchmark)

    exit_code = main(
        ["benchmark", "--model", model_id, "--compare-external-adapter"],
        settings=settings,
        services=services_with_fake_runtime,
    )

    assert exit_code == 0


def test_cli_persists_but_downgrades_partial_support_external_adapter_preference(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    model_id = next(manifest["model_id"] for manifest in scan_payload["manifests"])
    manifest = services_with_fake_runtime.model_registry.get_manifest(model_id)
    native_runtime = FakeMLXSemanticRuntime(settings=temp_settings)

    class FakeExternalAdapterRuntime(FakeMLXSemanticRuntime):
        name = "local_external_adapter"
        affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR

    external_runtime = FakeExternalAdapterRuntime(settings=temp_settings)

    result = _persist_external_adapter_runtime_preference(
        services=services_with_fake_runtime,
        manifest=manifest,
        native_runtime=native_runtime,
        external_runtime=external_runtime,
        comparison={
            "status": "completed",
            "winner": "external",
            "primary_metric": "warm_total_seconds",
            "selected_metric_value": 0.4,
            "baseline_metric_value": 0.9,
        },
        feature_preservation={
            "preserved": ["continuous_batching"],
            "degraded": ["kv_cache_quantization"],
            "rejected": [],
        },
        created_at="2026-04-18T20:00:00Z",
    )

    host_platform = services_with_fake_runtime.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
    persisted = services_with_fake_runtime.model_registry.metadata_store.get_runtime_preference(
        model_id=manifest.model_id,
        capability="chat",
        host_platform=host_platform,
    )

    assert result["applied"] is False
    assert result["persisted"] is True
    assert result["effective_runtime_name"] == native_runtime.name
    assert "LewLM keeps `fake_mlx_semantic`" in result["reason"]
    assert persisted is not None
    assert persisted["preference_status"] == "downgraded"
    assert persisted["effective_runtime_name"] == native_runtime.name
    assert "partially preserved" in persisted["downgrade_reason"]


def test_cli_benchmark_compare_direct_surfaces_scheduler_and_residency_evidence(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
    capsys,
) -> None:
    main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    gguf_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    )

    monkeypatch.setattr(
        "lewlm.cli.main.benchmark_direct_chat_manifest",
        lambda manifest, *, prompt, max_tokens, warmup_run_count=1: {
            "model_id": manifest.model_id,
            "display_name": manifest.display_name,
            "runtime": "llama_cpp_direct",
            "prompt": prompt,
            "load_seconds": 1.0,
            "generate_seconds": 0.9,
            "total_seconds": 1.9,
            "output_text": "baseline",
            "usage": {"completion_tokens": 4},
            "phase_breakdown": {
                "cold_load_seconds": 1.0,
                "cold_total_seconds": 1.9,
                "warm_total_seconds": 0.9,
                "ttft_seconds": 0.3,
                "steady_state_decode_seconds": 0.6,
                "steady_state_decode_tokens_per_second": 6.6667,
            },
        },
    )
    monkeypatch.setattr(
        "lewlm.cli.main._run_managed_benchmark_once",
        lambda *, services, model_id, prompt, warmup_run_count=1: {
            "status": "completed",
            "model_id": model_id,
            "runtime": "fake_llamacpp",
            "capability": "chat",
            "load_seconds": 0.8,
            "generate_seconds": 0.7,
            "total_seconds": 1.5,
            "output_text": "optimized",
            "phase_breakdown": {
                "cold_load_seconds": 0.8,
                "cold_total_seconds": 1.5,
                "warm_total_seconds": 0.7,
                "ttft_seconds": 0.2,
                "steady_state_decode_seconds": 0.5,
                "steady_state_decode_tokens_per_second": 8.0,
            },
            "scenarios": [
                {
                    "scenario": "continuous_batching",
                    "status": "observed",
                    "metrics": {
                        "concurrency": 2,
                        "throughput_requests_per_second": 2.4,
                        "native_batch_count_delta": 1,
                        "native_batched_request_delta": 2,
                        "frontier_batch_count_delta": 1,
                        "frontier_batched_request_delta": 2,
                        "average_batch_size": 2.0,
                        "average_batch_utilization": 0.5,
                        "average_queue_delay_seconds": 0.01,
                        "single_request_elapsed_seconds": 0.82,
                    },
                },
                {
                    "scenario": "warm_chat_cache",
                    "status": "observed",
                    "metrics": {
                        "average_cold_ttft_seconds": 0.31,
                        "average_warm_ttft_seconds": 0.18,
                        "average_warm_over_cold_ttft_ratio": 0.58,
                        "average_cold_elapsed_seconds": 0.92,
                        "average_warm_elapsed_seconds": 0.63,
                        "total_cache_restores": 1,
                        "total_persistent_cache_hits": 1,
                        "total_warm_saved_prefill_tokens": 24,
                    },
                },
            ],
        },
    )

    exit_code = main(
        ["benchmark", "--model", gguf_model_id, "--compare-direct", "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    comparison_evidence = payload["results"][0]["comparison"]["evidence"]
    assert comparison_evidence["overall_status"] == "proved"
    assert comparison_evidence["scheduler"]["status"] == "proved"
    assert comparison_evidence["residency"]["status"] == "proved"
    assert payload["models"][0]["evidence_summary"]["overall_status"] == "proved"
    assert payload["summary"]["evidence_summary"]["overall_status"] == "proved"


def test_cli_benchmark_compare_direct_filters_known_backend_noise() -> None:
    stderr = "\n".join(
        (
            "llama_context: n_ctx_seq (4096) < n_ctx_train (131072) -- the full capacity of the model will not be utilized",
            "llama_kv_cache_iswa: using full-size SWA cache (ref: https://example.test)",
            "llama_kv_cache: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to 1024",
            "real backend error",
        ),
    )

    filtered = _filter_benchmark_stderr(stderr)

    assert "real backend error" in filtered
    assert "llama_context:" not in filtered
    assert "llama_kv_cache_iswa:" not in filtered
    assert "padding V cache" not in filtered


def test_cli_benchmark_compare_direct_convert_missing_emits_conversion_payload(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0

    hf_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] != "gguf"
    )

    def fake_convert(services, manifest, request):
        return (
            {
                "needed": True,
                "status": "completed",
                "request": request.model_dump(mode="json"),
                "profile_label": "int4",
                "cache_hit": False,
                "duration_seconds": 12.5,
                "result_path": "/tmp/converted-model",
                "job_id": "job-convert-1",
                "logs_tail": ["converted"],
            },
            type(manifest).model_validate(
                {
                    **manifest.model_dump(mode="json"),
                    "model_id": f"{manifest.model_id}-converted",
                    "source_path": "/tmp/converted-model",
                    "format_type": "mlx",
                    "conversion_status": "runnable",
                    "runtime_affinity": ["mlx_text"],
                },
            ),
        )

    monkeypatch.setattr("lewlm.cli.main._convert_manifest_for_benchmark", fake_convert)
    monkeypatch.setattr(
        "lewlm.cli.main.benchmark_direct_chat_manifest",
        lambda manifest, *, prompt, max_tokens, warmup_run_count: {
            "model_id": manifest.model_id,
            "display_name": manifest.display_name,
            "runtime": "mlx_lm_direct",
            "prompt": prompt,
            "load_seconds": 2.0,
            "generate_seconds": 1.0,
            "total_seconds": 3.0,
            "output_text": "baseline",
            "usage": {},
        },
    )
    monkeypatch.setattr(
        "lewlm.cli.main._run_managed_benchmark_once",
        lambda *, services, model_id, prompt, warmup_run_count: {
            "status": "completed",
            "model_id": model_id,
            "runtime": "mlx_text",
            "capability": "chat",
            "load_seconds": 1.0,
            "generate_seconds": 1.0,
            "total_seconds": 2.0,
            "output_text": "optimized",
        },
    )

    exit_code = main(
        ["benchmark", "--model", hf_model_id, "--compare-direct", "--convert-missing", "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["summary"]["converted_model_count"] == 1
    assert payload["summary"]["conversion_total_seconds"] == 12.5
    assert payload["models"][0]["conversion"]["duration_seconds"] == 12.5
    assert payload["results"][0]["benchmark_model_id"].endswith("-converted")
    assert payload["results"][0]["conversion"]["needed"] is True
    assert payload["results"][0]["comparison"]["time_saved_seconds"] == 1.0


def test_cli_benchmark_compare_direct_all_ignores_cached_conversion_artifacts(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0

    gguf_manifest = next(
        manifest
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    )
    cache_manifest = {
        **gguf_manifest,
        "model_id": f"{gguf_manifest['model_id']}-cached",
        "display_name": f"{gguf_manifest['display_name']} converted",
        "source_path": str(temp_settings.cache_dir / "conversions" / "converted-model"),
        "format_type": "mlx",
        "conversion_status": "runnable",
        "runtime_affinity": ["mlx_text"],
    }
    services_with_fake_runtime.metadata_store.replace_model_manifests(
        [
            *services_with_fake_runtime.model_registry.list_manifests(),
            type(services_with_fake_runtime.model_registry.list_manifests()[0]).model_validate(cache_manifest),
        ],
        stale_source_paths=(),
    )

    monkeypatch.setattr(
        "lewlm.cli.main.benchmark_direct_chat_manifest",
        lambda manifest, *, prompt, max_tokens, warmup_run_count: {
            "model_id": manifest.model_id,
            "display_name": manifest.display_name,
            "runtime": "llama_cpp_direct",
            "prompt": prompt,
            "load_seconds": 1.0,
            "generate_seconds": 1.0,
            "total_seconds": 2.0,
            "output_text": "baseline",
            "usage": {},
        },
    )
    monkeypatch.setattr(
        "lewlm.cli.main._run_managed_benchmark_once",
        lambda *, services, model_id, prompt, warmup_run_count: {
            "status": "completed",
            "model_id": model_id,
            "runtime": "fake_runtime",
            "capability": "chat",
            "load_seconds": 1.0,
            "generate_seconds": 0.5,
            "total_seconds": 1.5,
            "output_text": "optimized",
        },
    )

    exit_code = main(
        ["benchmark", "--all", "--compare-direct", "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    payload = json.loads(capsys.readouterr().out)
    expected_model_count = sum(
        1
        for manifest in services_with_fake_runtime.model_registry.list_manifests()
        if not str(manifest.source_path).startswith(str(temp_settings.cache_dir / "conversions"))
        and manifest.conversion_status == "runnable"
    )

    assert exit_code == 0
    assert payload["model_count"] == expected_model_count
    assert all(not result["model_id"].endswith("-cached") for result in payload["results"])


def test_cli_benchmark_compare_direct_all_human_emits_progress(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0

    def fake_convert(services, manifest, request):
        converted_manifest = manifest.model_copy(
            update={
                "model_id": f"{manifest.model_id}-{request.policy.value}-converted",
                "display_name": f"{manifest.display_name} converted",
                "format_type": "mlx",
                "conversion_status": "runnable",
                "runtime_affinity": ["mlx_text"],
                "source_path": str(Path(manifest.source_path).with_name(f"{Path(manifest.source_path).name}-converted")),
            },
        )
        return (
            {
                "needed": True,
                "status": "completed",
                "request": request.model_dump(mode="json"),
                "profile_label": request.policy.value,
                "cache_hit": False,
                "duration_seconds": 4.0,
                "result_path": converted_manifest.source_path,
                "job_id": f"{manifest.model_id}-job",
                "logs_tail": [],
            },
            converted_manifest,
        )

    monkeypatch.setattr("lewlm.cli.main._convert_manifest_for_benchmark", fake_convert)
    monkeypatch.setattr(
        "lewlm.cli.main.benchmark_direct_chat_manifest",
        lambda manifest, *, prompt, max_tokens, warmup_run_count: {
            "model_id": manifest.model_id,
            "display_name": manifest.display_name,
            "runtime": "mlx_lm_direct",
            "prompt": prompt,
            "load_seconds": 1.0,
            "generate_seconds": 1.0,
            "total_seconds": 2.0,
            "output_text": "baseline",
            "usage": {},
        },
    )
    monkeypatch.setattr(
        "lewlm.cli.main._run_managed_benchmark_once",
        lambda *, services, model_id, prompt, warmup_run_count: {
            "status": "completed",
            "model_id": model_id,
            "runtime": "mlx_text",
            "capability": "chat",
            "load_seconds": 0.5,
            "generate_seconds": 0.5,
            "total_seconds": 1.0,
            "output_text": "optimized",
        },
    )

    exit_code = main(
        ["benchmark", "--all", "--compare-direct", "--convert-missing"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "preparing direct benchmark targets:" in output
    assert "converting" in output
    assert "running direct baseline:" in output
    assert "running LewLM-managed benchmark:" in output
    assert "LewLM benchmark vs direct inference" in output


def test_cli_benchmark_compare_direct_nonrunnable_model_requires_convert_missing(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0

    hf_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] != "gguf"
    )

    exit_code = main(
        ["benchmark", "--model", hf_model_id, "--compare-direct"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--convert-missing" in captured.err


def test_cli_benchmark_compare_direct_rejects_non_chat_capability(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    exit_code = main(
        ["benchmark", "--all", "--compare-direct", "--capability", "embeddings"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "chat benchmarks only" in captured.err


def test_cli_quantization_profile_parser_builds_mixed_precision_profile() -> None:
    profile = _conversion_quantization_profile_from_args(
        argparse.Namespace(
            profile="mixed_precision",
            profile_name="mixed-a",
            weight_precision="int4",
            activation_precision=None,
            compute_precision="bf16",
            kv_cache_precision=None,
            calibration_samples=None,
            layer_override=["layers.0.attn.q_proj:int8::bf16"],
            external_quantizer=None,
            external_profile=None,
            external_module=None,
        ),
    )

    assert profile is not None
    assert profile.strategy == "mixed_precision"
    assert profile.name == "mixed-a"
    assert profile.layer_overrides[0].weight_precision == "int8"
    assert profile.layer_overrides[0].compute_precision == "bf16"


def test_cli_benchmark_conversion_requests_expand_multiple_policies_and_profiles() -> None:
    manifest = argparse.Namespace(model_id="model-1")
    requests = _benchmark_conversion_requests(
        argparse.Namespace(
            convert_policies=["balanced", "max_quality"],
            convert_profiles=["mixed_precision", "hybrid_fp8"],
        ),
        manifest,
    )

    assert [request.policy.value for request in requests] == ["balanced", "max_quality", "balanced", "balanced"]
    assert all(request.model_id == "model-1" for request in requests)
    assert requests[2].quantization_profile is not None
    assert requests[2].quantization_profile.strategy == "mixed_precision"
    assert requests[3].quantization_profile is not None
    assert requests[3].quantization_profile.strategy == "hybrid_fp8"


def test_cli_profile_metrics_annotation_uses_max_quality_reference(tmp_path: Path) -> None:
    balanced_dir = tmp_path / "balanced"
    balanced_dir.mkdir()
    (balanced_dir / "weights.safetensors").write_bytes(b"1234")
    max_quality_dir = tmp_path / "max-quality"
    max_quality_dir.mkdir()
    (max_quality_dir / "weights.safetensors").write_bytes(b"12345678")
    results = [
        {
            "model_id": "source-model",
            "source_path": str(tmp_path / "source"),
            "conversion": {
                "needed": True,
                "profile_label": "balanced",
                "request": {"policy": "balanced"},
                "result_path": str(balanced_dir),
            },
            "optimized": {
                "status": "completed",
                "output_text": "hello there",
                "ttft_seconds": 0.4,
                "decode_tokens_per_second": 11.0,
                "phase_breakdown": {
                    "cold_load_seconds": 1.5,
                    "warm_total_seconds": 0.8,
                },
                "serving_profile": {
                    "status": "selected",
                    "accepted_settings": {"prefill_token_batch_size": 256},
                    "rejected_settings": {},
                    "effective_settings": {"prefill_token_batch_size": 256},
                    "reason": "Applied the persisted serving-profile overrides selected by autotuning.",
                },
            },
        },
        {
            "model_id": "source-model",
            "source_path": str(tmp_path / "source"),
            "conversion": {
                "needed": True,
                "profile_label": "max_quality",
                "request": {"policy": "max_quality"},
                "result_path": str(max_quality_dir),
            },
            "optimized": {
                "status": "completed",
                "output_text": "hello world",
                "ttft_seconds": 0.6,
                "decode_tokens_per_second": 8.0,
                "phase_breakdown": {
                    "cold_load_seconds": 2.0,
                    "warm_total_seconds": 1.0,
                },
                "serving_profile": {
                    "status": "not_found",
                    "accepted_settings": {},
                    "rejected_settings": {},
                    "effective_settings": {"prefill_token_batch_size": 512},
                    "reason": "No persisted serving profile is available for this host/model pair.",
                },
            },
        },
    ]

    _annotate_profile_metrics(results)

    assert results[0]["profile_metrics"]["quality_proxy"]["reference_profile"] == "max_quality"
    assert results[0]["profile_metrics"]["quality_proxy"]["exact_match"] is False
    assert results[1]["profile_metrics"]["quality_proxy"]["exact_match"] is True
    assert results[1]["profile_metrics"]["model_size_bytes"] == 8
    assert results[0]["profile_metrics"]["cold_load_seconds"] == 1.5
    assert results[0]["profile_metrics"]["warm_total_seconds"] == 0.8
    assert results[0]["profile_metrics"]["serving_profile_compatibility"]["classification"] == "fully_supported"


def test_cli_cache_emits_performance_features_json(
    temp_settings,
    services_with_fake_multimodal_runtime,
    capsys,
) -> None:
    exit_code = main(["cache", "--json"], settings=temp_settings, services=services_with_fake_multimodal_runtime)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    performance_features = {item["feature"]: item for item in payload["performance_features"]}
    assert performance_features["disk_backed_cache"]["supported"] is True
    assert performance_features["prefix_cache"]["supported"] is True
    assert performance_features["paged_kv_cache"]["supported"] is True
    assert performance_features["kv_cache_quantization"]["supported"] is True
    assert performance_features["block_disk_cache"]["supported"] is True


def test_cli_cache_clear_conversions_prunes_registry_and_artifact_records(
    temp_settings,
    services_with_fake_runtime_and_conversion,
    capsys,
) -> None:
    services_with_fake_runtime_and_conversion.model_registry.scan()
    conversion_root = temp_settings.cache_dir / "conversions"
    converted_dir = conversion_root / "cache-key"
    converted_dir.mkdir(parents=True, exist_ok=True)
    (converted_dir / "config.json").write_text('{"model_type":"phi3"}', encoding="utf-8")
    (converted_dir / "weights.safetensors").write_bytes(b"weights")
    (converted_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (converted_dir / CONVERSION_OUTPUT_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "source_display_name": "phi-3-mini-hf",
                "source_model_id": "phi-3-mini-hf-source",
                "display_name": "phi-3-mini-hf (converted)",
                "artifact_role": "standalone",
                "artifact_family_id": "cache-key",
            },
        ),
        encoding="utf-8",
    )
    services_with_fake_runtime_and_conversion.metadata_store.upsert_conversion_artifact(
        ConversionArtifactRecord(
            cache_key="cache-key",
            model_id="phi-3-mini-hf-source",
            output_path=str(converted_dir),
            policy=ConversionPolicy.BALANCED,
            metadata={"storage_mode": "directory"},
        ),
    )
    services_with_fake_runtime_and_conversion.model_registry.scan(
        roots=[*temp_settings.models_dir, conversion_root],
    )
    assert any(
        Path(manifest.source_path).is_relative_to(conversion_root)
        for manifest in services_with_fake_runtime_and_conversion.model_registry.inventory().items
    )
    assert services_with_fake_runtime_and_conversion.metadata_store.list_conversion_artifacts()

    exit_code = main(
        ["cache", "clear-conversions", "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime_and_conversion,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["cleared_artifact_records"] == 1
    assert Path(payload["cache_root"]).exists()
    assert list(Path(payload["cache_root"]).iterdir()) == []
    assert payload["scan"]["removed_count"] >= 1
    inventory = services_with_fake_runtime_and_conversion.model_registry.inventory()
    assert inventory.count == 3
    assert all(
        not Path(manifest.source_path).is_relative_to(conversion_root)
        for manifest in inventory.items
    )
    assert services_with_fake_runtime_and_conversion.metadata_store.list_conversion_artifacts() == []


def test_cli_doctor_emits_performance_features_json(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    exit_code = main(["doctor", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["install_profiles"]["active_profile_ids"][0] == "core_only"
    assert "total_memory_mb" in payload["runtime_stats"]["platform"]
    assert "total_memory_source" in payload["runtime_stats"]["platform"]
    assert "total_memory_reason" in payload["runtime_stats"]["platform"]
    performance_features = {
        item["feature"]: item
        for item in payload["runtime_stats"]["performance_features"]
    }
    performance_core_evidence = {
        item["family"]: item
        for item in payload["runtime_stats"]["performance_core_evidence"]
    }
    assert "artifact_summary" in payload["runtime_stats"]["benchmark_summary"]
    assert performance_features["request_scheduling_and_backpressure"]["supported"] is True
    assert performance_features["prefix_cache"]["supported"] is True
    assert performance_features["prefix_cache"]["active"] is False
    assert "measured_capability_registry" in payload["runtime_stats"]
    assert "continuous_batching" in performance_core_evidence
    assert "constrained_decoding" in performance_core_evidence


def test_cli_doctor_reports_mlx_kv_cache_and_prefill_support(
    temp_settings,
    services_with_fake_multimodal_runtime,
    capsys,
) -> None:
    exit_code = main(["doctor", "--json"], settings=temp_settings, services=services_with_fake_multimodal_runtime)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    performance_features = {
        item["feature"]: item
        for item in payload["runtime_stats"]["performance_features"]
    }
    assert performance_features["paged_kv_cache"]["supported"] is True
    assert performance_features["kv_cache_quantization"]["supported"] is True
    assert performance_features["prefill_optimization"]["supported"] is True


def test_cli_chat_json_includes_reasoning_when_requested(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0
    gguf_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    )

    exit_code = main(
        [
            "chat",
            "[emit-reasoning] Explain the CLI plan",
            "--model",
            gguf_model_id,
            "--reasoning-visibility",
            "summarized",
            "--json",
        ],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["output_text"] == "Echo: Explain the CLI plan"
    assert payload["reasoning"] == {
        "visibility": "summarized",
        "available": True,
        "content": None,
        "summary": "Inspect the prompt before replying.",
    }


def test_cli_chat_json_reports_decode_time_structured_output(
    temp_settings,
    services_with_fake_runtime,
    tmp_path: Path,
    capsys,
) -> None:
    response_format_path = tmp_path / "response-format.json"
    response_format_path.write_text(
        json.dumps(
            {
                "type": "json_schema",
                "name": "cli_probe",
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "const": "ok"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        ),
        encoding="utf-8",
    )
    scan_code = main(["scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_code == 0
    gguf_model_id = next(
        manifest["model_id"]
        for manifest in scan_payload["manifests"]
        if manifest["format_type"] == "gguf"
    )

    exit_code = main(
        [
            "chat",
            "Return the probe object.",
            "--model",
            gguf_model_id,
            "--response-format-file",
            str(response_format_path),
            "--json",
        ],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["structured_output"]["contract"]["type"] == "json_schema"
    assert payload["structured_output"]["enforcement"] == "decode_time"
    assert payload["structured_output"]["decoder_enforced"] is True
    assert payload["structured_output"]["fallback_used"] is False
    assert payload["structured_output"]["validation"]["state"] == "valid"


def test_cli_chat_human_output_reports_prompt_guided_structured_output_fallback(
    temp_settings,
    sample_models_root: Path,
    tmp_path: Path,
    capsys,
) -> None:
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: PromptGuidedLlamaRuntime()},
    )
    response_format_path = tmp_path / "response-format.json"
    response_format_path.write_text(
        json.dumps(
            {
                "type": "json_schema",
                "name": "cli_probe",
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "const": "ok"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        ),
        encoding="utf-8",
    )
    try:
        scan_code = main(["scan", "--json"], settings=temp_settings, services=services)
        scan_payload = json.loads(capsys.readouterr().out)
        assert scan_code == 0
        gguf_model_id = next(
            manifest["model_id"]
            for manifest in scan_payload["manifests"]
            if manifest["format_type"] == "gguf"
        )

        exit_code = main(
            [
                "chat",
                "Return the probe object.",
                "--model",
                gguf_model_id,
                "--response-format-file",
                str(response_format_path),
            ],
            settings=temp_settings,
            services=services,
        )
        output = capsys.readouterr().out
    finally:
        services.close()

    assert exit_code == 0
    assert "structured output: json_schema (prompt_guided, validation=invalid) fallback" in output


def test_cli_main_closes_owned_bootstrapped_services(temp_settings, monkeypatch, capsys) -> None:
    close_calls = 0

    class FakeScanSummary:
        def model_dump_json(self, *, indent: int = 2) -> str:
            return json.dumps({"discovered_count": 0}, indent=indent)

    class FakeModelRegistry:
        def scan(self, roots=None):
            return FakeScanSummary()

    class FakeServices:
        def __init__(self) -> None:
            self.model_registry = FakeModelRegistry()

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    fake_services = FakeServices()
    monkeypatch.setattr("lewlm.cli.main.bootstrap_services", lambda settings: fake_services)

    exit_code = main(["scan", "--json"], settings=temp_settings)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["discovered_count"] == 0
    assert close_calls == 1


def test_cli_main_preserves_injected_services_and_still_closes_owned_failures(
    temp_settings,
    monkeypatch,
    capsys,
) -> None:
    owned_close_calls = 0
    injected_close_calls = 0

    class FailingModelRegistry:
        def scan(self, roots=None):
            raise ConfigurationError("scan failed")

    class FakeServices:
        def __init__(self, *, failing: bool) -> None:
            self.model_registry = FailingModelRegistry() if failing else object()

        def close(self) -> None:
            nonlocal owned_close_calls, injected_close_calls
            if self is owned_services:
                owned_close_calls += 1
            else:
                injected_close_calls += 1

    owned_services = FakeServices(failing=True)
    monkeypatch.setattr("lewlm.cli.main.bootstrap_services", lambda settings: owned_services)

    exit_code = main(["scan", "--json"], settings=temp_settings)
    error_payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert error_payload["error"]["message"] == "scan failed"
    assert owned_close_calls == 1

    class SuccessScanSummary:
        def model_dump_json(self, *, indent: int = 2) -> str:
            return json.dumps({"discovered_count": 1}, indent=indent)

    class SuccessModelRegistry:
        def scan(self, roots=None):
            return SuccessScanSummary()

    injected_services = FakeServices(failing=False)
    injected_services.model_registry = SuccessModelRegistry()

    exit_code = main(["scan", "--json"], settings=temp_settings, services=injected_services)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["discovered_count"] == 1
    assert injected_close_calls == 0
