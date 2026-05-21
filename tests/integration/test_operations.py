from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from conftest import (
    FakeExternalSemanticRuntime,
    FakeLlamaCppRuntime,
    FakeMLXAudioRuntime,
    FakeMLXConversionBackend,
    FakeMLXSemanticRuntime,
    UnavailableMLXTextRuntime,
)
from lewlm.api.app import create_app
from lewlm.conversion.models import ConversionJobRequest, JobStatus
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import CapabilityName, GenerateMessage, GenerateRequest, GenerateResponse, ModelFormat, ModelManifest, RuntimeAffinity
from lewlm.core.errors import BackpressureError
from lewlm.security.persistence import ENCRYPTED_FILE_MAGIC


class SlowFakeLlamaCppRuntime(FakeLlamaCppRuntime):
    async def _load_model(self, manifest: ModelManifest) -> None:
        await asyncio.sleep(0.02)

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        await asyncio.sleep(0.05)
        return await super()._generate(request)


class FakeExternalAudioRuntime(FakeMLXAudioRuntime):
    name = "local_external_adapter"
    affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR
    supported_formats = (ModelFormat.MLX, ModelFormat.GGUF, ModelFormat.AUDIO_FOLDER)


def _write_external_validation_manifest(
    path: Path,
    *,
    capability_report: dict[str, object],
    system: str,
    machine: str,
) -> None:
    external_model = dict(capability_report)
    external_model["target_platforms"] = [
        {
            "system": system,
            "machine": machine,
            "supported": True,
            "readiness_state": "verified",
            "verification_method": "host_probe",
            "runtime_affinities": ["llamacpp"],
            "reason": f"Validated on {system} {machine}.",
            "fallback_available": False,
            "fallback_reason": None,
            "install_hints": [],
            "validation_manifest_count": 0,
            "verified_hosts": [],
            "notes": [],
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format": "lewlm-release-manifest-v1",
                "generated_at": "2026-04-15T00:00:00+00:00",
                "git_commit": "abc1234def5678",
                "platform": {"system": system, "machine": machine, "release": "validated-host"},
                "registered_models": [external_model],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_conversion_jobs_and_stats_endpoints(app_with_fake_runtime_and_conversion) -> None:
    with TestClient(app_with_fake_runtime_and_conversion) as client:
        services = client.app.state.services

        def drop_promoted_manifests() -> None:
            retained = [
                manifest
                for manifest in services.model_registry.list_manifests()
                if not Path(manifest.source_path).is_relative_to(result_path)
            ]
            stale_sources = [
                source_path
                for source_path, _ in services.metadata_store.list_model_manifest_records()
                if Path(source_path).is_relative_to(result_path)
            ]
            services.metadata_store.replace_model_manifests(retained, stale_source_paths=stale_sources)

        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        gguf_model_id = next(manifest["model_id"] for manifest in manifests if manifest["format_type"] == "gguf")
        hf_model_id = next(manifest["model_id"] for manifest in manifests if manifest["format_type"] == "huggingface")

        submit_response = client.post(
            "/v1/models/convert",
            json={"model_id": hf_model_id, "idempotency_key": "convert-job-1"},
        )
        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]

        job_payload = None
        for _ in range(30):
            job_response = client.get(f"/v1/jobs/{job_id}")
            job_payload = job_response.json()
            if job_payload["status"] == "completed":
                break
            time.sleep(0.05)

        assert job_payload is not None
        assert job_payload["status"] == "completed"
        assert job_payload["idempotency_key"] == "convert-job-1"
        assert job_payload["payload"]["result_path"]
        assert job_payload["payload"]["sandboxed"] is True
        assert job_payload["payload"]["storage_mode"] == "directory"
        assert job_payload["payload"]["cache_encrypted"] is False
        result_path = Path(job_payload["payload"]["result_path"])
        list_response = client.get("/v1/models")
        scoped_scan_response = client.post("/v1/models/scan", json={"paths": [str(result_path)]})

        assert list_response.status_code == 200
        assert any(
            Path(manifest["source_path"]).is_relative_to(result_path)
            for manifest in list_response.json()["items"]
        )
        assert scoped_scan_response.status_code == 403

        drop_promoted_manifests()

        replay_response = client.post(
            "/v1/models/convert",
            json={"model_id": hf_model_id, "idempotency_key": "convert-job-1"},
        )
        assert replay_response.status_code == 200
        assert replay_response.json()["job_id"] == job_id
        assert replay_response.json()["idempotency_key"] == "convert-job-1"
        assert replay_response.json()["idempotent_replay"] is True
        replayed_list_response = client.get("/v1/models")
        assert any(
            Path(manifest["source_path"]).is_relative_to(result_path)
            for manifest in replayed_list_response.json()["items"]
        )

        drop_promoted_manifests()

        repeat_response = client.post("/v1/models/convert", json={"model_id": hf_model_id})
        assert repeat_response.status_code == 200
        assert repeat_response.json()["payload"]["cache_hit"] is True
        assert repeat_response.json()["payload"]["sandboxed"] is True
        assert repeat_response.json()["payload"]["storage_mode"] == "directory"
        assert repeat_response.json()["payload"]["cache_encrypted"] is False
        repeated_list_response = client.get("/v1/models")
        assert any(
            Path(manifest["source_path"]).is_relative_to(result_path)
            for manifest in repeated_list_response.json()["items"]
        )

        cache_response = client.get("/v1/cache/stats")
        runtime_response = client.get("/v1/runtime/stats")
        capabilities_response = client.get(f"/v1/models/{gguf_model_id}/capabilities")

    assert cache_response.status_code == 200
    assert cache_response.json()["artifact_count"] == 1
    assert cache_response.json()["cache_hits"] >= 1
    assert runtime_response.status_code == 200
    assert runtime_response.json()["platform"]["system"]
    assert runtime_response.json()["readiness"]["status"] == "partial"
    assert runtime_response.json()["readiness"]["ready_capability_count"] >= 1
    assert "host_platform_supported" in runtime_response.json()["runtimes"][0]
    assert "readiness_state" in runtime_response.json()["runtimes"][0]
    assert runtime_response.json()["queue_depth"] == 0
    assert capabilities_response.status_code == 200
    assert capabilities_response.json()["model_id"] == gguf_model_id
    assert capabilities_response.json()["host_platform"]["python_version"]
    assert capabilities_response.json()["target_platforms"]
    assert capabilities_response.json()["runtime_candidates"][0]["runtime_name"] == "fake_llamacpp"
    assert any(
        item["capability"] == "chat"
        and item["supported"] is True
        and item["readiness_state"] == "ready"
        for item in capabilities_response.json()["capabilities"]
    )


def test_conversion_endpoint_rejects_conflicting_idempotency_keys(app_with_fake_runtime_and_conversion) -> None:
    with TestClient(app_with_fake_runtime_and_conversion) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        hf_model_id = next(manifest["model_id"] for manifest in manifests if manifest["format_type"] == "huggingface")

        first_response = client.post(
            "/v1/models/convert",
            json={"model_id": hf_model_id, "idempotency_key": "convert-conflict-1"},
        )
        assert first_response.status_code == 200

        conflicting_response = client.post(
            "/v1/models/convert",
            json={"model_id": hf_model_id, "force": True, "idempotency_key": "convert-conflict-1"},
        )

    assert conflicting_response.status_code == 409
    assert conflicting_response.json()["error"]["code"] == "idempotency_conflict"
    assert conflicting_response.json()["error"]["details"]["idempotency_key"] == "convert-conflict-1"
    assert conflicting_response.json()["error"]["details"]["fallback_guidance"]


def test_runtime_stats_include_benchmark_history_and_target_platforms(
    temp_settings,
    services_with_fake_runtime_and_conversion,
) -> None:
    app = create_app(temp_settings, services=services_with_fake_runtime_and_conversion)
    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        gguf_model_id = next(manifest["model_id"] for manifest in manifests if manifest["format_type"] == "gguf")
        hf_model_id = next(manifest["model_id"] for manifest in manifests if manifest["format_type"] == "huggingface")

        benchmark = asyncio.run(
            services_with_fake_runtime_and_conversion.telemetry_service.benchmark(
                model_id=gguf_model_id,
                prompt="Benchmark ping",
            ),
        )
        runtime_response = client.get("/v1/runtime/stats")

    assert benchmark.runtime == "fake_llamacpp"
    assert benchmark.artifact is not None
    assert benchmark.regression is not None
    assert benchmark.regression.status == "no_baseline"
    assert runtime_response.status_code == 200
    runtime_payload = runtime_response.json()
    assert "total_memory_mb" in runtime_payload["platform"]
    assert "total_memory_source" in runtime_payload["platform"]
    assert "total_memory_reason" in runtime_payload["platform"]
    assert runtime_payload["benchmark_summary"]["total_runs"] == 1
    assert runtime_payload["benchmark_summary"]["recent_runs"][0]["model_id"] == gguf_model_id
    assert runtime_payload["benchmark_summary"]["artifact_summary"]["total_artifacts"] == 1
    readiness = {item["capability"]: item for item in runtime_payload["readiness"]["capabilities"]}
    assert readiness["chat"]["ready"] is True
    assert readiness["chat"]["readiness_state"] == "ready"
    benchmark_features = {
        item["feature"]: item
        for item in runtime_payload["benchmark_summary"]["recent_runs"][0]["performance_features"]
    }
    benchmark_scenarios = {
        item.scenario: item
        for item in benchmark.scenarios
    }
    assert "continuous_batching" in benchmark_scenarios
    assert benchmark_scenarios["continuous_batching"].metrics["throughput_requests_per_second"] is not None
    assert benchmark_features["request_scheduling_and_backpressure"]["supported"] is True
    assert benchmark_features["continuous_batching"]["supported"] is True
    assert benchmark_features["prefix_cache"]["supported"] is True
    assert benchmark_features["prefix_cache"]["active"] is True
    assert benchmark_features["prefix_cache"]["metrics"]["cache_saves"] >= 1
    runtime_features = {item["feature"]: item for item in runtime_payload["performance_features"]}
    assert runtime_features["balanced_residency_mode"]["supported"] is True
    assert runtime_features["continuous_batching"]["supported"] is True
    assert "chat" in runtime_features["continuous_batching"]["supported_capabilities"]
    assert sorted(runtime_features["continuous_batching"]["ownership_modes"]) == ["backend_native", "partial"]
    assert runtime_features["continuous_batching"]["metrics"]["chat_streaming_ownership_mode"] == "mixed"
    assert runtime_features["continuous_batching"]["metrics"]["backend_native_runtime_count"] >= 1
    assert runtime_features["continuous_batching"]["metrics"]["partial_runtime_count"] >= 1
    assert runtime_features["continuous_batching"]["metrics"]["lewlm_owned_runtime_count"] == 0
    runtime_strategy = runtime_payload["runtime_support_strategy"]
    assert runtime_strategy["first_class_non_apple_path_id"] == "gguf_llamacpp"
    gguf_path = next(path for path in runtime_strategy["paths"] if path["path_id"] == "gguf_llamacpp")
    assert gguf_path["role"] == "first_class_non_apple"
    assert gguf_path["benchmark_backed_defaults"] is False
    assert "serving-profile/autotune default adoption" in " ".join(gguf_path["lewlm_managed_layers"])
    targets = {
        (item["system"], item["machine"]): item
        for item in runtime_payload["target_platforms"]
    }
    assert ("Linux", "x86_64") in targets
    assert ("Windows", "AMD64") in targets
    assert gguf_model_id in targets[("Linux", "x86_64")]["compatible_models"]
    assert hf_model_id in targets[("Linux", "x86_64")]["fallback_models"]
    assert hf_model_id in targets[("Windows", "AMD64")]["fallback_models"]
    assert "readiness_state" in targets[("Linux", "x86_64")]
    assert "verification_method" in targets[("Linux", "x86_64")]
    assert "fallback_model_count" in targets[("Linux", "x86_64")]
    assert "readiness_state" in targets[("Linux", "x86_64")]["runtimes"][0]


def test_runtime_stats_readiness_reports_multimodal_surfaces(app_with_fake_attachment_runtime) -> None:
    with TestClient(app_with_fake_attachment_runtime) as client:
        scan_response = client.post("/v1/models/scan", json={})
        runtime_response = client.get("/v1/runtime/stats")

    assert scan_response.status_code == 200
    assert runtime_response.status_code == 200
    readiness = {item["capability"]: item for item in runtime_response.json()["readiness"]["capabilities"]}
    assert readiness["chat"]["ready"] is True
    assert readiness["embeddings"]["ready"] is True
    assert readiness["rerank"]["ready"] is True
    assert readiness["audio_transcription"]["ready"] is True
    assert readiness["audio_speech"]["ready"] is True


def test_runtime_stats_mark_nonapple_behavior_defaults_per_family(
    temp_settings,
    services_with_fake_runtime,
) -> None:
    app = create_app(temp_settings, services=services_with_fake_runtime)
    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        gguf_model_id = next(manifest["model_id"] for manifest in manifests if manifest["format_type"] == "gguf")
        recommendation = asyncio.run(
            services_with_fake_runtime.telemetry_service.autotune(
                model_id=gguf_model_id,
                prompt="Non-Apple serving defaults",
            ),
        )
        runtime_response = client.get("/v1/runtime/stats")

    assert recommendation.runtime == "fake_llamacpp"
    assert runtime_response.status_code == 200
    runtime_payload = runtime_response.json()
    standards_contract = runtime_payload["standards_acceptance_contract"]
    assert standards_contract["format"] == "lewlm-standards-acceptance-contract-v1"
    assert any(item["name"] == "cuda13_ready" for item in standards_contract["vocabulary"])
    assert any(item["name"] == "document_ocr_transformer" for item in standards_contract["vocabulary"])
    optimization_defaults = runtime_payload["optimization_defaults"]["models"]
    model_defaults = next(item for item in optimization_defaults if item["model_id"] == gguf_model_id)
    assert model_defaults["decisions"]["prefix_reuse"]["status"] == "adopted"
    assert model_defaults["decisions"]["prefix_reuse"]["benchmark_backed"] is True

    runtime_strategy = runtime_payload["runtime_support_strategy"]
    gguf_path = next(path for path in runtime_strategy["paths"] if path["path_id"] == "gguf_llamacpp")
    evidence = {item["family"]: item for item in gguf_path["performance_core_evidence"]}

    assert gguf_path["benchmark_backed_defaults"] is True
    assert evidence["continuous_batching"]["mode"] == "backend_native"
    assert evidence["continuous_batching"]["benchmark_backed"] is True
    assert evidence["prefix_reuse"]["mode"] == "backend_native"
    assert evidence["prefix_reuse"]["benchmark_backed"] is True
    assert evidence["tiered_kv"]["benchmark_backed"] is False


def test_runtime_stats_readiness_reports_external_semantic_bridge_surfaces(
    temp_settings,
    sample_multimodal_models_root: Path,
) -> None:
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.EXTERNAL_ACCELERATOR: FakeExternalSemanticRuntime(settings=temp_settings),
            RuntimeAffinity.MLX_TEXT: UnavailableMLXTextRuntime(settings=temp_settings),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )
    app = create_app(temp_settings, services=services)
    try:
        with TestClient(app) as client:
            scan_response = client.post("/v1/models/scan", json={})
            runtime_response = client.get("/v1/runtime/stats")
    finally:
        services.close()

    assert scan_response.status_code == 200
    readiness = {item["capability"]: item for item in runtime_response.json()["readiness"]["capabilities"]}
    assert readiness["chat"]["ready"] is False
    assert readiness["embeddings"]["ready"] is True
    assert readiness["embeddings"]["available_runtime_names"] == ["local_external_adapter"]
    assert readiness["rerank"]["ready"] is True
    assert readiness["rerank"]["available_runtime_names"] == ["local_external_adapter"]


def test_runtime_stats_readiness_reports_external_audio_bridge_surfaces(
    temp_settings,
    sample_multimodal_models_root: Path,
) -> None:
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.EXTERNAL_ACCELERATOR: FakeExternalAudioRuntime(),
        },
    )
    app = create_app(temp_settings, services=services)
    try:
        with TestClient(app) as client:
            scan_response = client.post("/v1/models/scan", json={})
            runtime_response = client.get("/v1/runtime/stats")
    finally:
        services.close()

    assert scan_response.status_code == 200
    readiness = {item["capability"]: item for item in runtime_response.json()["readiness"]["capabilities"]}
    assert readiness["audio_transcription"]["ready"] is True
    assert readiness["audio_transcription"]["bridge_only"] is True
    assert readiness["audio_transcription"]["available_support_paths"] == ["bridge"]
    assert readiness["audio_transcription"]["bridge_runtime_names"] == ["local_external_adapter"]
    assert readiness["audio_speech"]["ready"] is True
    assert readiness["audio_speech"]["bridge_only"] is True
    assert "bridge-backed only" in " ".join(readiness["audio_speech"]["notes"])


def test_runtime_stats_report_mlx_kv_cache_and_prefill_support(
    temp_settings,
    services_with_fake_attachment_runtime,
) -> None:
    text_model_dir = temp_settings.models_dir[0] / "qwen2.5-1.5b-instruct-mlx"
    text_model_dir.mkdir(parents=True, exist_ok=True)
    (text_model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "max_position_embeddings": 32768}),
        encoding="utf-8",
    )
    (text_model_dir / "weights.safetensors").write_bytes(b"mlx-text-weights")
    (text_model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    app = create_app(temp_settings, services=services_with_fake_attachment_runtime)
    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        text_model_id = next(
            manifest["model_id"]
            for manifest in manifests
            if "mlx_text" in manifest["runtime_affinity"]
            and any(
                capability["capability"] == "chat"
                and capability["supported"] is True
                and capability["runtime_name"] == "fake_mlx_semantic"
                for capability in services_with_fake_attachment_runtime.model_router.model_capability_report(
                    manifest["model_id"],
                ).model_dump(mode="json")["capabilities"]
            )
        )

        benchmark = asyncio.run(
            services_with_fake_attachment_runtime.telemetry_service.benchmark(
                model_id=text_model_id,
                prompt="Prefill benchmark",
            ),
        )
        cache_response = client.get("/v1/cache/stats")
        runtime_response = client.get("/v1/runtime/stats")

    assert benchmark.runtime == "fake_mlx_semantic"
    benchmark_features = {item.feature.value: item for item in benchmark.performance_features}
    assert benchmark_features["prefix_cache"].supported is True
    assert benchmark_features["prefix_cache"].active is True
    assert benchmark_features["persistent_multi_context_cache"].supported is True
    assert any(item.scenario == "warm_chat_cache" for item in benchmark.scenarios)
    assert benchmark_features["paged_kv_cache"].supported is True
    assert benchmark_features["kv_cache_quantization"].supported is True
    assert benchmark_features["prefill_optimization"].supported is True
    assert benchmark_features["prefill_optimization"].active is True

    cache_features = {item["feature"]: item for item in cache_response.json()["performance_features"]}
    assert cache_features["prefix_cache"]["supported"] is True
    assert cache_features["persistent_multi_context_cache"]["supported"] is True
    assert cache_features["paged_kv_cache"]["supported"] is True
    assert cache_features["kv_cache_quantization"]["supported"] is True

    runtime_features = {item["feature"]: item for item in runtime_response.json()["performance_features"]}
    assert runtime_features["prefix_cache"]["supported"] is True
    assert runtime_features["prefix_cache"]["active"] is True
    assert runtime_features["prefix_cache"]["metrics"]["cache_hits"] >= 1
    assert runtime_features["persistent_multi_context_cache"]["supported"] is True
    assert runtime_features["persistent_multi_context_cache"]["active"] is True
    assert runtime_features["persistent_multi_context_cache"]["metrics"]["persisted_cache_entries"] >= 1
    assert runtime_features["paged_kv_cache"]["supported"] is True
    assert runtime_features["paged_kv_cache"]["metrics"]["requests_using_paged_kv"] >= 1
    assert runtime_features["paged_kv_cache"]["metrics"]["resident_pages"] >= 1
    assert runtime_features["paged_kv_cache"]["metrics"]["active_pages"] == 0
    assert runtime_features["paged_kv_cache"]["metrics"]["pressure_level"] in {
        "low",
        "medium",
        "high",
        "overflow",
        "unbounded",
    }
    assert runtime_features["kv_cache_quantization"]["supported"] is True
    assert runtime_features["prefill_optimization"]["supported"] is True
    assert runtime_features["prefill_optimization"]["active"] is True
    assert runtime_features["prefill_optimization"]["metrics"]["optimized_requests"] >= 1


def test_autotune_endpoint_persists_recommended_serving_profile(
    temp_settings,
    services_with_fake_attachment_runtime,
) -> None:
    text_model_dir = temp_settings.models_dir[0] / "qwen2.5-1.5b-instruct-mlx"
    text_model_dir.mkdir(parents=True, exist_ok=True)
    (text_model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "max_position_embeddings": 32768}),
        encoding="utf-8",
    )
    (text_model_dir / "weights.safetensors").write_bytes(b"mlx-text-weights")
    (text_model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    app = create_app(temp_settings, services=services_with_fake_attachment_runtime)
    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        text_model_id = next(
            manifest["model_id"]
            for manifest in manifests
            if manifest["display_name"] == "qwen2.5-1.5b-instruct-mlx"
        )
        response = client.post(
            "/v1/benchmarks/autotune",
            json={"model_id": text_model_id, "prompt": "Autotune ping"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == text_model_id
    assert payload["capability"] == "chat"
    assert payload["workload_class"] == "text_only"
    assert payload["profile_id"]
    assert payload["artifact"]["artifact_path"]
    assert payload["candidate_summaries"]
    assert any(item["name"] == "prefill_256" for item in payload["candidate_summaries"])
    assert payload["effective_settings"]["prefill_token_batch_size"] >= 1
    stored_profile = services_with_fake_attachment_runtime.metadata_store.get_serving_profile(
        model_id=text_model_id,
        capability="chat",
        host_platform=payload["host_platform"],
        runtime_name=payload["runtime"],
        workload_class=payload["workload_class"],
    )
    assert stored_profile is not None
    assert stored_profile["profile_id"] == payload["profile_id"]


def test_autotune_endpoint_persists_multimodal_workload_specific_profile(
    temp_settings,
    services_with_fake_attachment_runtime,
) -> None:
    app = create_app(temp_settings, services=services_with_fake_attachment_runtime)
    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        vision_model_id = next(
            manifest["model_id"]
            for manifest in manifests
            if manifest["display_name"] == "qwen2-vl-vision-mlx"
        )
        response = client.post(
            "/v1/benchmarks/autotune",
            json={
                "model_id": vision_model_id,
                "prompt": "Autotune image workload",
                "workload_class": "single_image",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == vision_model_id
    assert payload["runtime"] == "fake_mlx_vision"
    assert payload["workload_class"] == "single_image"
    stored_profile = services_with_fake_attachment_runtime.metadata_store.get_serving_profile(
        model_id=vision_model_id,
        capability="chat",
        host_platform=payload["host_platform"],
        runtime_name=payload["runtime"],
        workload_class=payload["workload_class"],
    )
    assert stored_profile is not None
    assert stored_profile["profile_id"] == payload["profile_id"]


def test_benchmark_suite_records_multiple_chat_models(
    temp_settings,
    sample_models_root: Path,
) -> None:
    second_model_path = temp_settings.models_dir[0] / "mistral-7b-instruct-q4_k_m.gguf"
    second_model_path.write_bytes(b"gguf-model-2")
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_TEXT: UnavailableMLXTextRuntime(),
        },
        conversion_backend=FakeMLXConversionBackend(),
    )
    app = create_app(temp_settings, services=services)
    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        assert scan_response.status_code == 200
        suite = asyncio.run(
            services.telemetry_service.benchmark_suite(
                prompt="Benchmark suite ping",
            ),
        )
        runtime_response = client.get("/v1/runtime/stats")

    assert suite.benchmark_count == 2
    assert suite.model_count == 2
    assert suite.repeat_count == 1
    assert len(suite.results) == 2
    assert len(suite.models) == 2
    assert suite.artifact is not None
    assert suite.regression is not None
    assert all(item.run_count == 1 for item in suite.models)
    assert all(result.runtime == "fake_llamacpp" for result in suite.results)
    assert services.metadata_store.benchmark_record_count() == 2
    runtime_payload = runtime_response.json()
    assert runtime_payload["benchmark_summary"]["total_runs"] == 2
    assert len(runtime_payload["benchmark_summary"]["models"]) == 2


def test_benchmark_suite_records_multiple_embedding_models(
    temp_settings,
    sample_multimodal_models_root: Path,
) -> None:
    second_embedding_dir = temp_settings.models_dir[0] / "gte-small-embed-mlx"
    second_embedding_dir.mkdir(parents=True)
    (second_embedding_dir / "config.json").write_text(
        json.dumps({"model_type": "gte", "max_position_embeddings": 8192}),
        encoding="utf-8",
    )
    (second_embedding_dir / "weights.safetensors").write_bytes(b"embed-weights-2")
    (second_embedding_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )
    app = create_app(temp_settings, services=services)
    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        assert scan_response.status_code == 200
        scan_payload = scan_response.json()
        expected_model_count = sum(
            1
            for manifest in scan_payload["manifests"]
            if any(
                capability["capability"] == "embeddings" and capability["supported"] is True
                for capability in services.model_router.model_capability_report(
                    manifest["model_id"],
                ).model_dump(mode="json")["capabilities"]
            )
        )
        suite = asyncio.run(
            services.telemetry_service.benchmark_suite(
                prompt="Benchmark semantic suite",
                capability=CapabilityName.EMBEDDINGS.value,
            ),
        )
        runtime_response = client.get("/v1/runtime/stats")

    assert suite.capability == CapabilityName.EMBEDDINGS.value
    assert expected_model_count >= 2
    assert suite.benchmark_count == expected_model_count
    assert suite.model_count == expected_model_count
    assert len(suite.results) == expected_model_count
    assert all(result.runtime == "fake_mlx_semantic" for result in suite.results)
    assert all(result.capability == CapabilityName.EMBEDDINGS.value for result in suite.results)
    assert any(item.scenario == "multimodal_reuse" for item in suite.scenarios)
    suite_features = {
        item.feature.value: item
        for item in suite.performance_features
    }
    assert suite_features["continuous_batching"].supported is True
    assert suite_features["disk_backed_cache"].supported is True
    assert services.metadata_store.benchmark_record_count() == expected_model_count
    runtime_payload = runtime_response.json()
    assert runtime_payload["benchmark_summary"]["total_runs"] == expected_model_count
    assert runtime_payload["benchmark_summary"]["capability_counts"]["embeddings"] == expected_model_count
    assert runtime_payload["benchmark_summary"]["recent_runs"][0]["capability"] == "embeddings"


def test_benchmark_suite_repeat_count_records_multiple_passes(
    temp_settings,
    sample_models_root: Path,
) -> None:
    second_model_path = temp_settings.models_dir[0] / "mistral-7b-instruct-q4_k_m.gguf"
    second_model_path.write_bytes(b"gguf-model-2")
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_TEXT: UnavailableMLXTextRuntime(),
        },
        conversion_backend=FakeMLXConversionBackend(),
    )
    app = create_app(temp_settings, services=services)
    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        assert scan_response.status_code == 200
        suite = asyncio.run(
            services.telemetry_service.benchmark_suite(
                prompt="Repeated benchmark suite ping",
                repeat_count=2,
            ),
        )
        runtime_response = client.get("/v1/runtime/stats")

    assert suite.repeat_count == 2
    assert suite.benchmark_count == 4
    assert suite.model_count == 2
    assert len(suite.results) == 4
    assert len(suite.models) == 2
    assert suite.artifact is not None
    assert all(item.run_count == 2 for item in suite.models)
    assert all(item.capability_counts == {"chat": 2} for item in suite.models)
    assert services.metadata_store.benchmark_record_count() == 4
    runtime_payload = runtime_response.json()
    assert runtime_payload["benchmark_summary"]["total_runs"] == 4
    assert runtime_payload["benchmark_summary"]["capability_counts"]["chat"] == 4


def test_benchmark_records_rerank_capability_run(
    temp_settings,
    sample_multimodal_models_root: Path,
) -> None:
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )
    app = create_app(temp_settings, services=services)
    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        assert scan_response.status_code == 200
        benchmark = asyncio.run(
            services.telemetry_service.benchmark(
                model_id=None,
                prompt="alpha beta query",
                capability=CapabilityName.RERANK.value,
            ),
        )
        runtime_response = client.get("/v1/runtime/stats")

    assert benchmark.capability == CapabilityName.RERANK.value
    assert benchmark.runtime == "fake_mlx_semantic"
    assert any(item.scenario == "multimodal_reuse" for item in benchmark.scenarios)
    assert benchmark.measurements["document_count"] == 3
    assert benchmark.measurements["result_count"] == 3
    runtime_payload = runtime_response.json()
    assert runtime_payload["benchmark_summary"]["total_runs"] == 1
    assert runtime_payload["benchmark_summary"]["capability_counts"]["rerank"] == 1
    assert runtime_payload["benchmark_summary"]["recent_runs"][0]["capability"] == "rerank"


def test_benchmark_regression_uses_prior_artifact_baseline(
    temp_settings,
    sample_models_root: Path,
) -> None:
    baseline_services = bootstrap_services(
        temp_settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime()},
    )
    baseline_app = create_app(temp_settings, services=baseline_services)
    with TestClient(baseline_app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        assert scan_response.status_code == 200
        manifests = scan_response.json()["manifests"]
        gguf_model_id = next(manifest["model_id"] for manifest in manifests if manifest["format_type"] == "gguf")
        baseline = asyncio.run(
            baseline_services.telemetry_service.benchmark(
                model_id=gguf_model_id,
                prompt="Regression check prompt",
            ),
        )

    slow_services = bootstrap_services(
        temp_settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: SlowFakeLlamaCppRuntime()},
    )
    slow_app = create_app(temp_settings, services=slow_services)
    with TestClient(slow_app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        assert scan_response.status_code == 200
        manifests = scan_response.json()["manifests"]
        gguf_model_id = next(manifest["model_id"] for manifest in manifests if manifest["format_type"] == "gguf")
        regressed = asyncio.run(
            slow_services.telemetry_service.benchmark(
                model_id=gguf_model_id,
                prompt="Regression check prompt",
            ),
        )

    assert baseline.regression is not None
    assert baseline.regression.status == "no_baseline"
    assert regressed.regression is not None
    assert regressed.regression.status == "failed"
    assert regressed.artifact is not None
    assert regressed.regression.compared_to_artifact_id == baseline.artifact.artifact_id
    assert regressed.regression.failure_count >= 1


def test_external_validation_manifests_upgrade_target_readiness(
    temp_settings,
    sample_models_root: Path,
) -> None:
    seed_services = bootstrap_services(
        temp_settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime()},
    )
    manifests = seed_services.model_registry.scan().manifests
    gguf_model_id = next(manifest.model_id for manifest in manifests if manifest.format_type.value == "gguf")
    capability_report = seed_services.model_router.model_capability_report(gguf_model_id).model_dump(mode="json")
    validation_manifest_path = temp_settings.data_dir / "validations" / "linux-x86_64.json"
    _write_external_validation_manifest(
        validation_manifest_path,
        capability_report=capability_report,
        system="Linux",
        machine="x86_64",
    )

    validated_settings = temp_settings.with_updates(validation_manifest_paths=(validation_manifest_path,))
    validated_services = bootstrap_services(
        validated_settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime()},
    )
    app = create_app(validated_settings, services=validated_services)

    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        gguf_model_id = next(
            manifest["model_id"]
            for manifest in scan_response.json()["manifests"]
            if manifest["format_type"] == "gguf"
        )
        capabilities_response = client.get(f"/v1/models/{gguf_model_id}/capabilities")
        runtime_response = client.get("/v1/runtime/stats")

    assert capabilities_response.status_code == 200
    assert runtime_response.status_code == 200
    runtime_payload = runtime_response.json()
    assert runtime_payload["validation_manifest_count"] == 1
    linux_target = next(
        target
        for target in runtime_payload["target_platforms"]
        if target["system"] == "Linux" and target["machine"] == "x86_64"
    )
    assert linux_target["readiness_state"] == "verified_external"
    assert linux_target["verification_method"] == "external_release_manifest"
    assert linux_target["verified_model_count"] >= 1
    assert linux_target["verified_hosts"]

    capability_target = next(
        target
        for target in capabilities_response.json()["target_platforms"]
        if target["system"] == "Linux" and target["machine"] == "x86_64"
    )
    assert capability_target["readiness_state"] == "verified_external"
    assert capability_target["verification_method"] == "external_release_manifest"
    assert capability_target["validation_manifest_count"] == 1
    assert capability_target["verified_hosts"]


def test_conversion_required_text_bundles_get_cross_platform_fallback_guidance(
    temp_settings,
    services_with_fake_runtime_and_conversion,
) -> None:
    app = create_app(temp_settings, services=services_with_fake_runtime_and_conversion)

    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        hf_model_id = next(
            manifest["model_id"]
            for manifest in scan_response.json()["manifests"]
            if manifest["format_type"] == "huggingface"
        )
        capabilities_response = client.get(f"/v1/models/{hf_model_id}/capabilities")

    assert capabilities_response.status_code == 200
    targets = {
        (target["system"], target["machine"]): target
        for target in capabilities_response.json()["target_platforms"]
    }
    linux_target = targets[("Linux", "x86_64")]
    windows_target = targets[("Windows", "AMD64")]
    assert linux_target["readiness_state"] == "fallback_guided"
    assert linux_target["fallback_available"] is True
    assert "GGUF build" in linux_target["fallback_reason"]
    assert linux_target["notes"]
    assert windows_target["readiness_state"] == "fallback_guided"
    assert windows_target["fallback_available"] is True
    assert "fake_llamacpp" in windows_target["fallback_reason"]


def test_api_key_and_request_size_guards(secured_settings, limited_settings) -> None:
    secured_app = create_app(secured_settings)
    with TestClient(secured_app) as client:
        assert client.get("/v1/health").status_code == 200
        unauthorized = client.get("/v1/models")
        authorized = client.get("/v1/models", headers={"x-api-key": "test-key"})

    limited_app = create_app(limited_settings)
    with TestClient(limited_app) as client:
        oversized = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "x" * 1000}],
            },
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert oversized.status_code == 413


def test_runtime_stats_include_request_metrics(app_with_fake_multimodal_runtime, sample_audio_bytes: bytes) -> None:
    encoded_audio = base64.b64encode(sample_audio_bytes).decode("ascii")
    with TestClient(app_with_fake_multimodal_runtime) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        embedding_model_id = next(
            manifest["model_id"]
            for manifest in manifests
            if "embedding" in manifest["modality"]
        )
        rerank_model_id = next(
            manifest["model_id"]
            for manifest in manifests
            if "rerank" in manifest["modality"]
        )
        audio_model_id = next(
            manifest["model_id"]
            for manifest in manifests
            if "audio" in manifest["modality"]
        )

        embeddings_response = client.post(
            "/v1/embeddings",
            json={"model": embedding_model_id, "input": ["alpha", "beta"]},
        )
        rerank_response = client.post(
            "/v1/rerank",
            json={
                "model": rerank_model_id,
                "query": "alpha beta",
                "documents": ["alpha beta overlap", "audio sample", "runtime stats"],
            },
        )
        transcription_response = client.post(
            "/v1/audio/transcriptions",
            json={
                "model": audio_model_id,
                "file_name": "sample.wav",
                "audio_base64": encoded_audio,
            },
        )
        runtime_response = client.get("/v1/runtime/stats")

    assert embeddings_response.status_code == 200
    assert rerank_response.status_code == 200
    assert transcription_response.status_code == 200
    assert runtime_response.status_code == 200
    request_metrics = runtime_response.json()["request_metrics"]
    assert request_metrics["total_requests"] == 3
    assert request_metrics["failure_count"] == 0
    assert request_metrics["success_count"] == 3
    assert request_metrics["average_execution_seconds"] is not None
    assert request_metrics["total_prompt_tokens"] > 0
    model_metrics = {item["model_id"]: item for item in request_metrics["models"]}
    assert model_metrics[embedding_model_id]["capability_counts"]["embeddings"] == 1
    assert model_metrics[rerank_model_id]["capability_counts"]["rerank"] == 1
    assert model_metrics[audio_model_id]["capability_counts"]["audio_transcription"] == 1
    capability_metrics = {item["capability"]: item for item in request_metrics["capabilities"]}
    assert capability_metrics["embeddings"]["metric_totals"]["input_count"] == 2
    assert capability_metrics["embeddings"]["metric_totals"]["vector_count"] == 2
    assert capability_metrics["rerank"]["metric_totals"]["document_count"] == 3
    assert capability_metrics["audio_transcription"]["metric_totals"]["audio_input_bytes"] > 0
    scheduler_metrics = runtime_response.json()["request_scheduler"]
    assert scheduler_metrics["max_concurrent_requests"] == 4
    assert scheduler_metrics["peak_active_requests"] >= 1
    assert scheduler_metrics["rejected_requests"] == 0
    load_scheduler_metrics = runtime_response.json()["load_scheduler"]
    assert load_scheduler_metrics["max_concurrent_requests"] == 1
    assert load_scheduler_metrics["peak_active_requests"] >= 1
    assert load_scheduler_metrics["rejected_requests"] == 0


class CountingFakeMLXSemanticRuntime(FakeMLXSemanticRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_calls = 0
        self.rerank_calls = 0
        self.embedding_batch_input_counts: list[int] = []

    async def embed(self, request):
        self.embedding_calls += 1
        self.embedding_batch_input_counts.append(len(request.inputs))
        return await super().embed(request)

    async def rerank(self, request):
        self.rerank_calls += 1
        return await super().rerank(request)


class SlowCountingFakeMLXSemanticRuntime(CountingFakeMLXSemanticRuntime):
    async def embed(self, request):
        self.embedding_calls += 1
        self.embedding_batch_input_counts.append(len(request.inputs))
        await asyncio.sleep(0.1)
        return await FakeMLXSemanticRuntime.embed(self, request)

    async def rerank(self, request):
        self.rerank_calls += 1
        await asyncio.sleep(0.1)
        return await FakeMLXSemanticRuntime.rerank(self, request)


class CountingFakeMLXAudioRuntime(FakeMLXAudioRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.transcription_calls = 0
        self.speech_calls = 0

    async def transcribe_audio(self, request):
        self.transcription_calls += 1
        return await super().transcribe_audio(request)

    async def synthesize_speech(self, request):
        self.speech_calls += 1
        return await super().synthesize_speech(request)


class SlowCountingFakeMLXAudioRuntime(CountingFakeMLXAudioRuntime):
    async def transcribe_audio(self, request):
        self.transcription_calls += 1
        await asyncio.sleep(0.1)
        return await FakeMLXAudioRuntime.transcribe_audio(self, request)

    async def synthesize_speech(self, request):
        self.speech_calls += 1
        await asyncio.sleep(0.1)
        return await FakeMLXAudioRuntime.synthesize_speech(self, request)


def test_runtime_response_cache_reuses_embeddings_and_rerank_results(
    temp_settings,
    sample_multimodal_models_root: Path,
) -> None:
    runtime = CountingFakeMLXSemanticRuntime()
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: runtime,
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )
    app = create_app(temp_settings, services=services)

    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        embedding_model_id = next(
            manifest["model_id"]
            for manifest in manifests
            if manifest["display_name"] == "e5-small-embed-mlx"
        )
        rerank_model_id = next(
            manifest["model_id"]
            for manifest in manifests
            if manifest["display_name"] == "bge-reranker-base-mlx"
        )

        first_embeddings = client.post(
            "/v1/embeddings",
            json={"model": embedding_model_id, "input": ["alpha", "beta"]},
        )
        second_embeddings = client.post(
            "/v1/embeddings",
            json={"model": embedding_model_id, "input": ["alpha", "beta"]},
        )
        first_rerank = client.post(
            "/v1/rerank",
            json={
                "model": rerank_model_id,
                "query": "alpha beta",
                "documents": ["alpha beta overlap", "audio sample", "runtime stats"],
            },
        )
        second_rerank = client.post(
            "/v1/rerank",
            json={
                "model": rerank_model_id,
                "query": "alpha beta",
                "documents": ["alpha beta overlap", "audio sample", "runtime stats"],
            },
        )
        cache_response = client.get("/v1/cache/stats")
        runtime_response = client.get("/v1/runtime/stats")

    assert first_embeddings.status_code == 200
    assert second_embeddings.status_code == 200
    assert first_embeddings.json()["request_id"] != second_embeddings.json()["request_id"]
    assert first_embeddings.json()["data"] == second_embeddings.json()["data"]
    assert first_embeddings.json()["model"] == second_embeddings.json()["model"]
    assert first_embeddings.json()["routing"]["model_id"] == embedding_model_id
    assert second_embeddings.json()["routing"]["model_id"] == embedding_model_id
    assert first_rerank.status_code == 200
    assert second_rerank.status_code == 200
    assert first_rerank.json()["request_id"] != second_rerank.json()["request_id"]
    assert first_rerank.json()["results"] == second_rerank.json()["results"]
    assert first_rerank.json()["model"] == second_rerank.json()["model"]
    assert first_rerank.json()["routing"]["model_id"] == rerank_model_id
    assert second_rerank.json()["routing"]["model_id"] == rerank_model_id
    assert runtime.embedding_calls == 1
    assert runtime.rerank_calls == 1

    cache_payload = cache_response.json()
    assert cache_payload["runtime_response_count"] == 2
    assert cache_payload["runtime_response_bytes"] > 0
    assert cache_payload["runtime_cache_hits"] == 2
    assert cache_payload["runtime_cache_misses"] == 2
    assert cache_payload["cache_hits"] >= 2
    assert cache_payload["cache_misses"] >= 2
    cache_features = {item["feature"]: item for item in cache_payload["performance_features"]}
    assert cache_features["disk_backed_cache"]["supported"] is True
    assert cache_features["block_disk_cache"]["supported"] is True

    runtime_payload = runtime_response.json()
    runtime_features = {item["feature"]: item for item in runtime_payload["performance_features"]}
    assert runtime_features["continuous_batching"]["supported"] is True
    assert runtime_features["request_scheduling_and_backpressure"]["supported"] is True
    assert runtime_features["prefix_cache"]["supported"] is True
    capability_metrics = {
        item["capability"]: item
        for item in runtime_payload["request_metrics"]["capabilities"]
    }
    assert capability_metrics["embeddings"]["metric_totals"]["cache_hits"] == 1
    assert capability_metrics["embeddings"]["metric_totals"]["cache_misses"] == 1
    assert capability_metrics["rerank"]["metric_totals"]["cache_hits"] == 1
    assert capability_metrics["rerank"]["metric_totals"]["cache_misses"] == 1


def test_runtime_response_cache_reuses_audio_transcription_results(
    temp_settings,
    sample_multimodal_models_root: Path,
    sample_audio_bytes: bytes,
) -> None:
    audio_runtime = CountingFakeMLXAudioRuntime()
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            RuntimeAffinity.MLX_AUDIO: audio_runtime,
        },
    )
    app = create_app(temp_settings, services=services)
    encoded_audio = base64.b64encode(sample_audio_bytes).decode("ascii")

    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        audio_model_id = next(
            manifest["model_id"]
            for manifest in manifests
            if "audio" in manifest["modality"]
        )

        first_transcription = client.post(
            "/v1/audio/transcriptions",
            json={
                "model": audio_model_id,
                "file_name": "sample.wav",
                "audio_base64": encoded_audio,
                "language": "en",
                "prompt": "normalize casing",
            },
        )
        second_transcription = client.post(
            "/v1/audio/transcriptions",
            json={
                "model": audio_model_id,
                "file_name": "sample.wav",
                "audio_base64": encoded_audio,
                "language": "en",
                "prompt": "normalize casing",
            },
        )
        cache_response = client.get("/v1/cache/stats")
        runtime_response = client.get("/v1/runtime/stats")

    assert first_transcription.status_code == 200
    assert second_transcription.status_code == 200
    assert first_transcription.json()["request_id"] != second_transcription.json()["request_id"]
    assert first_transcription.json()["text"] == second_transcription.json()["text"]
    assert first_transcription.json()["segments"] == second_transcription.json()["segments"]
    assert first_transcription.json()["model"] == second_transcription.json()["model"]
    assert first_transcription.json()["routing"]["model_id"] == audio_model_id
    assert second_transcription.json()["routing"]["model_id"] == audio_model_id
    assert audio_runtime.transcription_calls == 1

    cache_payload = cache_response.json()
    assert cache_payload["runtime_response_count"] == 1
    assert cache_payload["runtime_cache_hits"] == 1
    assert cache_payload["runtime_cache_misses"] == 1
    assert cache_payload["multimodal_encoder_count"] == 1
    assert cache_payload["multimodal_encoder_cache_hits"] == 0
    assert cache_payload["multimodal_encoder_cache_misses"] == 1

    capability_metrics = {
        item["capability"]: item
        for item in runtime_response.json()["request_metrics"]["capabilities"]
    }
    assert capability_metrics["audio_transcription"]["metric_totals"]["cache_hits"] == 1
    assert capability_metrics["audio_transcription"]["metric_totals"]["cache_misses"] == 1
    assert capability_metrics["audio_transcription"]["metric_totals"]["audio_input_bytes"] == len(sample_audio_bytes) * 2


def test_runtime_response_cache_reuses_audio_speech_results(
    temp_settings,
    sample_multimodal_models_root: Path,
) -> None:
    audio_runtime = CountingFakeMLXAudioRuntime()
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            RuntimeAffinity.MLX_AUDIO: audio_runtime,
        },
    )
    app = create_app(temp_settings, services=services)
    input_text = "Ship the next milestone"

    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        audio_model_id = next(
            manifest["model_id"]
            for manifest in manifests
            if "audio" in manifest["modality"]
        )

        first_speech = client.post(
            "/v1/audio/speech",
            json={"model": audio_model_id, "input": input_text, "voice": "alloy", "format": "wav"},
        )
        second_speech = client.post(
            "/v1/audio/speech",
            json={"model": audio_model_id, "input": input_text, "voice": "alloy", "format": "wav"},
        )
        cache_response = client.get("/v1/cache/stats")
        runtime_response = client.get("/v1/runtime/stats")

    assert first_speech.status_code == 200
    assert second_speech.status_code == 200
    assert first_speech.json()["request_id"] != second_speech.json()["request_id"]
    assert first_speech.json()["audio_base64"] == second_speech.json()["audio_base64"]
    assert first_speech.json()["media_type"] == second_speech.json()["media_type"]
    assert first_speech.json()["model"] == second_speech.json()["model"]
    assert first_speech.json()["routing"]["model_id"] == audio_model_id
    assert second_speech.json()["routing"]["model_id"] == audio_model_id
    assert audio_runtime.speech_calls == 1

    cache_payload = cache_response.json()
    assert cache_payload["runtime_response_count"] == 1
    assert cache_payload["runtime_cache_hits"] == 1
    assert cache_payload["runtime_cache_misses"] == 1

    capability_metrics = {
        item["capability"]: item
        for item in runtime_response.json()["request_metrics"]["capabilities"]
    }
    assert capability_metrics["audio_speech"]["metric_totals"]["cache_hits"] == 1
    assert capability_metrics["audio_speech"]["metric_totals"]["cache_misses"] == 1
    assert capability_metrics["audio_speech"]["metric_totals"]["input_characters"] == len(input_text) * 2
    assert capability_metrics["audio_speech"]["metric_totals"]["audio_output_bytes"] > 0


@pytest.mark.asyncio
async def test_multimodal_orchestrator_coalesces_concurrent_duplicate_requests(
    temp_settings,
    sample_multimodal_models_root: Path,
) -> None:
    runtime = SlowCountingFakeMLXSemanticRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(
            max_concurrent_runtime_requests=1,
            runtime_request_queue_limit=1,
            runtime_request_queue_timeout_seconds=1,
        ),
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: runtime,
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )
    try:
        manifests = services.model_registry.scan().manifests
        embedding_model_id = next(
            manifest.model_id
            for manifest in manifests
            if manifest.display_name == "e5-small-embed-mlx"
        )
        rerank_model_id = next(
            manifest.model_id
            for manifest in manifests
            if manifest.display_name == "bge-reranker-base-mlx"
        )

        first_embedding = asyncio.create_task(
            services.multimodal_orchestrator.embed(
                model_id=embedding_model_id,
                inputs=["alpha", "beta"],
            ),
        )
        await asyncio.sleep(0.02)
        second_embedding = asyncio.create_task(
            services.multimodal_orchestrator.embed(
                model_id=embedding_model_id,
                inputs=["alpha", "beta"],
            ),
        )
        embedding_a, embedding_b = await asyncio.gather(first_embedding, second_embedding)

        first_rerank = asyncio.create_task(
            services.multimodal_orchestrator.rerank(
                model_id=rerank_model_id,
                query="alpha beta",
                documents=["alpha beta overlap", "audio sample", "runtime stats"],
                top_n=2,
            ),
        )
        await asyncio.sleep(0.02)
        second_rerank = asyncio.create_task(
            services.multimodal_orchestrator.rerank(
                model_id=rerank_model_id,
                query="alpha beta",
                documents=["alpha beta overlap", "audio sample", "runtime stats"],
                top_n=2,
            ),
        )
        rerank_a, rerank_b = await asyncio.gather(first_rerank, second_rerank)

        assert embedding_a.response.model_dump(mode="json") == embedding_b.response.model_dump(mode="json")
        assert rerank_a.response.model_dump(mode="json") == rerank_b.response.model_dump(mode="json")
        assert runtime.embedding_calls == 1
        assert runtime.rerank_calls == 1
        scheduler_metrics = services.runtime_request_scheduler.snapshot()
        assert scheduler_metrics["total_queued_requests"] == 0
        capability_metrics = {
            item["capability"]: item
            for item in services.runtime_metrics_recorder.snapshot()["capabilities"]
        }
        assert capability_metrics["embeddings"]["metric_totals"]["coalesced_requests"] == 1
        assert capability_metrics["embeddings"]["metric_totals"]["cache_misses"] == 1
        assert capability_metrics["rerank"]["metric_totals"]["coalesced_requests"] == 1
        assert capability_metrics["rerank"]["metric_totals"]["cache_misses"] == 1
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_multimodal_orchestrator_coalesces_concurrent_duplicate_audio_transcriptions(
    temp_settings,
    sample_multimodal_models_root: Path,
    sample_audio_bytes: bytes,
) -> None:
    runtime = SlowCountingFakeMLXAudioRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(
            max_concurrent_runtime_requests=1,
            runtime_request_queue_limit=1,
            runtime_request_queue_timeout_seconds=1,
        ),
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            RuntimeAffinity.MLX_AUDIO: runtime,
        },
    )
    try:
        manifests = services.model_registry.scan().manifests
        audio_model_id = next(
            manifest.model_id
            for manifest in manifests
            if "audio" in manifest.modality
        )

        first_transcription = asyncio.create_task(
            services.multimodal_orchestrator.transcribe_audio(
                model_id=audio_model_id,
                audio_bytes=sample_audio_bytes,
                file_name="sample.wav",
                language="en",
                prompt="normalize casing",
            ),
        )
        await asyncio.sleep(0.02)
        second_transcription = asyncio.create_task(
            services.multimodal_orchestrator.transcribe_audio(
                model_id=audio_model_id,
                audio_bytes=sample_audio_bytes,
                file_name="sample.wav",
                language="en",
                prompt="normalize casing",
            ),
        )
        transcription_a, transcription_b = await asyncio.gather(first_transcription, second_transcription)

        assert transcription_a.response.model_dump(mode="json") == transcription_b.response.model_dump(mode="json")
        assert runtime.transcription_calls == 1
        scheduler_metrics = services.runtime_request_scheduler.snapshot()
        assert scheduler_metrics["total_queued_requests"] == 0
        capability_metrics = {
            item["capability"]: item
            for item in services.runtime_metrics_recorder.snapshot()["capabilities"]
        }
        assert capability_metrics["audio_transcription"]["metric_totals"]["coalesced_requests"] == 1
        assert capability_metrics["audio_transcription"]["metric_totals"]["cache_misses"] == 1
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_multimodal_orchestrator_coalesces_concurrent_duplicate_audio_speech_requests(
    temp_settings,
    sample_multimodal_models_root: Path,
) -> None:
    runtime = SlowCountingFakeMLXAudioRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(
            max_concurrent_runtime_requests=1,
            runtime_request_queue_limit=1,
            runtime_request_queue_timeout_seconds=1,
        ),
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            RuntimeAffinity.MLX_AUDIO: runtime,
        },
    )
    try:
        manifests = services.model_registry.scan().manifests
        audio_model_id = next(
            manifest.model_id
            for manifest in manifests
            if "audio" in manifest.modality
        )

        first_speech = asyncio.create_task(
            services.multimodal_orchestrator.synthesize_speech(
                model_id=audio_model_id,
                input_text="Ship the next milestone",
                voice="alloy",
                audio_format="wav",
            ),
        )
        await asyncio.sleep(0.02)
        second_speech = asyncio.create_task(
            services.multimodal_orchestrator.synthesize_speech(
                model_id=audio_model_id,
                input_text="Ship the next milestone",
                voice="alloy",
                audio_format="wav",
            ),
        )
        speech_a, speech_b = await asyncio.gather(first_speech, second_speech)

        assert speech_a.response.model_dump(mode="python") == speech_b.response.model_dump(mode="python")
        assert runtime.speech_calls == 1
        scheduler_metrics = services.runtime_request_scheduler.snapshot()
        assert scheduler_metrics["total_queued_requests"] == 0
        capability_metrics = {
            item["capability"]: item
            for item in services.runtime_metrics_recorder.snapshot()["capabilities"]
        }
        assert capability_metrics["audio_speech"]["metric_totals"]["coalesced_requests"] == 1
        assert capability_metrics["audio_speech"]["metric_totals"]["cache_misses"] == 1
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_multimodal_orchestrator_batches_distinct_concurrent_embedding_requests(
    temp_settings,
    sample_multimodal_models_root: Path,
) -> None:
    runtime = SlowCountingFakeMLXSemanticRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(
            max_concurrent_runtime_requests=1,
            runtime_request_queue_limit=1,
            runtime_request_queue_timeout_seconds=1,
        ),
        runtime_overrides={
            RuntimeAffinity.MLX_TEXT: runtime,
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )
    try:
        manifests = services.model_registry.scan().manifests
        embedding_model_id = next(
            manifest.model_id
            for manifest in manifests
            if manifest.display_name == "e5-small-embed-mlx"
        )

        first_embedding = asyncio.create_task(
            services.multimodal_orchestrator.embed(
                model_id=embedding_model_id,
                inputs=["alpha request"],
            ),
        )
        await asyncio.sleep(0.002)
        second_embedding = asyncio.create_task(
            services.multimodal_orchestrator.embed(
                model_id=embedding_model_id,
                inputs=["beta request"],
            ),
        )
        embedding_a, embedding_b = await asyncio.gather(first_embedding, second_embedding)

        assert runtime.embedding_calls == 1
        assert runtime.embedding_batch_input_counts == [2]
        assert len(embedding_a.response.data) == 1
        assert len(embedding_b.response.data) == 1
        assert embedding_a.response.model_dump(mode="json") != embedding_b.response.model_dump(mode="json")
        scheduler_metrics = services.runtime_request_scheduler.snapshot()
        assert scheduler_metrics["total_queued_requests"] == 0
        capability_metrics = {
            item["capability"]: item
            for item in services.runtime_metrics_recorder.snapshot()["capabilities"]
        }
        assert capability_metrics["embeddings"]["metric_totals"]["batched_requests"] == 2
        assert capability_metrics["embeddings"]["metric_totals"]["cache_misses"] == 2
    finally:
        await services.aclose()


class SlowFakeLlamaCppRuntime(FakeLlamaCppRuntime):
    async def _generate(self, request):
        await asyncio.sleep(0.15)
        return await super()._generate(request)


class SlowLoadingFakeLlamaCppRuntime(FakeLlamaCppRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.active_loads = 0
        self.max_concurrent_loads = 0

    async def _load_model(self, manifest: ModelManifest) -> None:
        self.active_loads += 1
        self.max_concurrent_loads = max(self.max_concurrent_loads, self.active_loads)
        try:
            await asyncio.sleep(0.1)
        finally:
            self.active_loads -= 1


async def test_chat_orchestrator_rejects_when_runtime_scheduler_queue_is_full(
    temp_settings,
    sample_models_root: Path,
) -> None:
    constrained_settings = temp_settings.with_updates(
        max_concurrent_runtime_requests=1,
        runtime_request_queue_limit=0,
        runtime_request_queue_timeout_seconds=1,
        continuous_batch_max_batch_size=1,
    )
    services = bootstrap_services(
        constrained_settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: SlowFakeLlamaCppRuntime()},
    )
    try:
        manifests = services.model_registry.scan().manifests
        gguf_model_id = next(manifest.model_id for manifest in manifests if manifest.format_type.value == "gguf")
        first_task = asyncio.create_task(
            services.chat_orchestrator.complete(
                model_id=gguf_model_id,
                messages=[GenerateMessage(role="user", content="first request")],
                max_tokens=64,
                temperature=0.0,
            ),
        )
        await asyncio.sleep(0.03)
        with pytest.raises(BackpressureError):
            await services.chat_orchestrator.complete(
                model_id=gguf_model_id,
                messages=[GenerateMessage(role="user", content="second request")],
                max_tokens=64,
                temperature=0.0,
            )
        await first_task
        snapshot = services.runtime_request_scheduler.snapshot()
        assert snapshot["rejected_requests"] == 1
        assert snapshot["peak_active_requests"] == 1
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_chat_orchestrator_serializes_model_loads_with_load_admission_control(
    temp_settings,
    sample_models_root: Path,
) -> None:
    second_model_path = temp_settings.models_dir[0] / "mistral-7b-instruct-q4_k_m.gguf"
    second_model_path.write_bytes(b"gguf-model-2")
    runtime = SlowLoadingFakeLlamaCppRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(
            max_concurrent_runtime_requests=2,
            max_concurrent_model_loads=1,
            runtime_request_queue_limit=2,
            runtime_request_queue_timeout_seconds=1,
        ),
        runtime_overrides={RuntimeAffinity.LLAMACPP: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        gguf_model_ids = [
            manifest.model_id
            for manifest in manifests
            if manifest.format_type.value == "gguf"
        ]
        assert len(gguf_model_ids) == 2

        first_task = asyncio.create_task(
            services.chat_orchestrator.complete(
                model_id=gguf_model_ids[0],
                messages=[GenerateMessage(role="user", content="first request")],
                max_tokens=64,
                temperature=0.0,
            ),
        )
        await asyncio.sleep(0.02)
        second_task = asyncio.create_task(
            services.chat_orchestrator.complete(
                model_id=gguf_model_ids[1],
                messages=[GenerateMessage(role="user", content="second request")],
                max_tokens=64,
                temperature=0.0,
            ),
        )
        await asyncio.gather(first_task, second_task)

        request_snapshot = services.runtime_request_scheduler.snapshot()
        load_snapshot = services.model_load_scheduler.snapshot()
        runtime_stats = await services.telemetry_service.runtime_stats()

        assert runtime.max_concurrent_loads == 1
        assert request_snapshot["peak_active_requests"] == 2
        assert request_snapshot["total_queued_requests"] == 0
        assert load_snapshot["total_queued_requests"] == 1
        assert load_snapshot["peak_active_requests"] == 1
        assert runtime_stats.load_scheduler.total_queued_requests == 1
        assert runtime_stats.load_scheduler.max_concurrent_requests == 1
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_chat_orchestrator_batches_chat_requests_with_backend_native_batching(
    temp_settings,
    sample_models_root: Path,
) -> None:
    runtime = FakeLlamaCppRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(
            continuous_batch_window_milliseconds=25,
            continuous_batch_max_batch_size=2,
        ),
        runtime_overrides={RuntimeAffinity.LLAMACPP: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        gguf_model_id = next(manifest.model_id for manifest in manifests if manifest.format_type.value == "gguf")
        first_task = asyncio.create_task(
            services.chat_orchestrator.complete(
                model_id=gguf_model_id,
                messages=[GenerateMessage(role="user", content="first batched request")],
                max_tokens=64,
                temperature=0.0,
            ),
        )
        second_task = asyncio.create_task(
            services.chat_orchestrator.complete(
                model_id=gguf_model_id,
                messages=[GenerateMessage(role="user", content="second batched request")],
                max_tokens=64,
                temperature=0.0,
            ),
        )
        first_result, second_result = await asyncio.gather(first_task, second_task)
        request_snapshot = services.runtime_request_scheduler.snapshot()
        runtime_stats = await services.telemetry_service.runtime_stats()
        continuous_batching = next(
            feature for feature in runtime_stats.performance_features if feature.feature.value == "continuous_batching"
        )

        assert "first batched request" in first_result.response.output_text
        assert "second batched request" in second_result.response.output_text
        assert runtime.batch_generate_calls == 1
        assert runtime.max_generate_batch_size == 2
        assert request_snapshot["native_total_batches"] == 1
        assert request_snapshot["native_batched_requests"] == 2
        assert request_snapshot["native_average_batch_size"] == 2.0
        assert request_snapshot["native_average_batch_utilization"] == 1.0
        assert request_snapshot["frontier_total_batches"] == 1
        assert request_snapshot["frontier_batched_requests"] == 2
        assert request_snapshot["frontier_average_batch_size"] == 2.0
        assert request_snapshot["frontier_average_batch_utilization"] == 1.0
        assert continuous_batching.supported is True
        assert sorted(continuous_batching.ownership_modes) == ["backend_native", "partial"]
        assert continuous_batching.metrics["chat_streaming_ownership_mode"] == "mixed"
        assert continuous_batching.metrics["backend_native_runtime_count"] >= 1
        assert continuous_batching.metrics["partial_runtime_count"] >= 1
        assert continuous_batching.metrics["lewlm_owned_runtime_count"] == 0
        assert continuous_batching.metrics["native_total_batches"] == 1
        assert continuous_batching.metrics["native_average_batch_utilization"] == 1.0
        assert continuous_batching.metrics["frontier_total_batches"] == 1
        assert continuous_batching.metrics["frontier_average_batch_utilization"] == 1.0
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_chat_orchestrator_batches_streaming_requests_with_backend_native_batching(
    temp_settings,
    sample_models_root: Path,
) -> None:
    runtime = FakeLlamaCppRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(
            continuous_batch_window_milliseconds=25,
            continuous_batch_max_batch_size=2,
        ),
        runtime_overrides={RuntimeAffinity.LLAMACPP: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        gguf_model_id = next(manifest.model_id for manifest in manifests if manifest.format_type.value == "gguf")
        first_task = asyncio.create_task(
            services.chat_orchestrator.stream(
                model_id=gguf_model_id,
                messages=[GenerateMessage(role="user", content="first stream request")],
                max_tokens=64,
                temperature=0.0,
            ),
        )
        second_task = asyncio.create_task(
            services.chat_orchestrator.stream(
                model_id=gguf_model_id,
                messages=[GenerateMessage(role="user", content="second stream request")],
                max_tokens=64,
                temperature=0.0,
            ),
        )
        first_session, second_session = await asyncio.gather(first_task, second_task)

        async def collect(stream) -> str:
            chunks: list[str] = []
            async for item in stream:
                chunks.append(item)
            return "".join(chunks)

        first_output, second_output = await asyncio.gather(
            collect(first_session.stream),
            collect(second_session.stream),
        )
        request_snapshot = services.runtime_request_scheduler.snapshot()

        assert "first stream request" in first_output
        assert "second stream request" in second_output
        assert runtime.batch_stream_calls == 1
        assert runtime.max_stream_batch_size == 2
        assert request_snapshot["native_total_batches"] == 1
        assert request_snapshot["native_batched_requests"] == 2
        assert request_snapshot["frontier_total_batches"] == 1
        assert request_snapshot["frontier_batched_requests"] == 2
    finally:
        await services.aclose()


def test_rate_limit_and_content_type_guards(rate_limited_settings, temp_settings) -> None:
    rate_limited_app = create_app(rate_limited_settings)
    with TestClient(rate_limited_app) as client:
        first = client.get("/v1/models")
        second = client.get("/v1/models")
        third = client.get("/v1/models")

    content_type_app = create_app(temp_settings)
    with TestClient(content_type_app) as client:
        wrong_type = client.post(
            "/v1/chat/completions",
            content=json.dumps({"messages": [{"role": "user", "content": "hello"}]}),
            headers={"content-type": "text/plain"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert wrong_type.status_code == 415


def test_api_filesystem_scope_guards(temp_settings, file_template_transform_payload) -> None:
    outside_model_root = temp_settings.data_dir.parent / "outside-models"
    outside_model_root.mkdir(parents=True)
    outside_template_path = temp_settings.data_dir.parent / "outside-template.json"
    outside_template_path.write_text(
        json.dumps(
            {
                "title": "Outside {{client}}",
                "sections": [{"heading": "Overview", "blocks": [{"type": "paragraph", "text": "Owner {{owner}}", "style_tokens": []}]}],
                "metadata": {},
                "style_tokens": [],
                "citations": [],
            },
        ),
        encoding="utf-8",
    )
    payload = dict(file_template_transform_payload)
    payload["template_path"] = str(outside_template_path)

    app = create_app(temp_settings)
    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={"paths": [str(outside_model_root)]})
        template_response = client.post("/v1/documents/transform", json=payload)

    assert scan_response.status_code == 403
    assert template_response.status_code == 403


def test_tool_authorization_and_audit_log(
    app_with_authorized_runtime_and_conversion,
    tool_authorized_settings,
    sample_document_payload,
    file_template_transform_payload,
) -> None:
    with TestClient(app_with_authorized_runtime_and_conversion) as client:
        denied_generate = client.post(
            "/v1/documents/generate",
            json={
                "output_format": "csv",
                "file_name": "report.csv",
                "document": sample_document_payload,
            },
        )
        allowed_generate = client.post(
            "/v1/documents/generate",
            json={
                "output_format": "csv",
                "file_name": "report.csv",
                "document": sample_document_payload,
                "authorized_actions": ["document_generate"],
            },
        )

        denied_transform = client.post("/v1/documents/transform", json=file_template_transform_payload)
        allowed_transform_payload = dict(file_template_transform_payload)
        allowed_transform_payload["authorized_actions"] = ["document_transform"]
        allowed_transform = client.post("/v1/documents/transform", json=allowed_transform_payload)
        denied_tool_execute = client.post(
            "/v1/tools/execute",
            json={"tool": "documents.transform", "input": file_template_transform_payload},
        )
        allowed_tool_execute_payload = dict(file_template_transform_payload)
        allowed_tool_execute_payload["authorized_actions"] = ["document_transform"]
        allowed_tool_execute = client.post(
            "/v1/tools/execute",
            json={"tool": "documents.transform", "input": allowed_tool_execute_payload},
        )

        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        hf_model_id = next(manifest["model_id"] for manifest in manifests if manifest["format_type"] == "huggingface")

        denied_convert = client.post("/v1/models/convert", json={"model_id": hf_model_id})
        allowed_convert = client.post(
            "/v1/models/convert",
            json={"model_id": hf_model_id, "authorized_actions": ["model_conversion"]},
        )

        job_payload = None
        if allowed_convert.status_code == 200:
            for _ in range(30):
                job_response = client.get(f"/v1/jobs/{allowed_convert.json()['job_id']}")
                job_payload = job_response.json()
                if job_payload["status"] == "completed":
                    break
                time.sleep(0.05)

    assert denied_generate.status_code == 403
    assert allowed_generate.status_code == 200
    assert denied_transform.status_code == 403
    assert allowed_transform.status_code == 200
    assert denied_tool_execute.status_code == 403
    assert allowed_tool_execute.status_code == 200
    assert denied_convert.status_code == 403
    assert allowed_convert.status_code == 200
    assert job_payload is not None
    assert job_payload["status"] == "completed"

    audit_events = [
        json.loads(line)
        for line in tool_authorized_settings.audit_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["action"] == "document_generate" and event["outcome"] == "denied" for event in audit_events)
    assert any(event["action"] == "document_generate" and event["outcome"] == "success" for event in audit_events)
    assert any(event["action"] == "document_transform" and event["outcome"] == "success" for event in audit_events)
    assert any(event["action"] == "model_conversion" and event["outcome"] == "completed" for event in audit_events)


def test_prompt_override_requests_are_audited(
    tool_authorized_settings,
    sample_models_root: Path,
    sample_prompt_assets,
) -> None:
    services = bootstrap_services(
        tool_authorized_settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime()},
    )
    app = create_app(tool_authorized_settings, services=services)

    with TestClient(app) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        gguf_model_id = next(manifest["model_id"] for manifest in manifests if manifest["format_type"] == "gguf")

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": gguf_model_id,
                "messages": [{"role": "user", "content": "Audit these overrides"}],
                "developer_prompt": "Keep it terse.",
                "pretext_path": str(sample_prompt_assets["pretext"]),
                "output_schema_path": str(sample_prompt_assets["output_schema"]),
                "include_prompt_trace": True,
            },
        )

    assert response.status_code == 200
    audit_events = [
        json.loads(line)
        for line in tool_authorized_settings.audit_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompt_event = next(event for event in audit_events if event["action"] == "prompt_override")
    assert prompt_event["outcome"] == "applied"
    assert prompt_event["actor"] == "api"
    assert prompt_event["details"]["override_count"] == 3
    assert prompt_event["details"]["selected_template"] == "structured_output"


def test_authorized_tool_failures_are_audited(
    app_with_authorized_runtime_and_conversion,
    tool_authorized_settings,
    file_template_transform_payload,
) -> None:
    failing_payload = dict(file_template_transform_payload)
    failing_payload["authorized_actions"] = ["document_transform"]
    failing_payload["input"] = {"replacements": {"client": "Acme Corp", "owner": "LewLM"}}

    with TestClient(app_with_authorized_runtime_and_conversion) as client:
        response = client.post(
            "/v1/tools/execute",
            json={"tool": "documents.transform", "input": failing_payload},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "document_validation_error"

    audit_events = [
        json.loads(line)
        for line in tool_authorized_settings.audit_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failure_event = next(
        event
        for event in audit_events
        if event["action"] == "document_transform" and event["outcome"] == "failed"
    )
    assert failure_event["actor"] == "api"
    assert failure_event["details"]["tool"] == "documents.transform"
    assert failure_event["details"]["sandboxed"] is True
    assert "missing a required placeholder value" in failure_event["details"]["error"]


def test_encrypted_conversion_cache_artifacts(
    app_with_encrypted_runtime_and_conversion,
    encrypted_persistence_settings,
) -> None:
    with TestClient(app_with_encrypted_runtime_and_conversion) as client:
        scan_response = client.post("/v1/models/scan", json={})
        manifests = scan_response.json()["manifests"]
        hf_model_id = next(manifest["model_id"] for manifest in manifests if manifest["format_type"] == "huggingface")

        submit_response = client.post("/v1/models/convert", json={"model_id": hf_model_id})
        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]

        job_payload = None
        for _ in range(30):
            job_response = client.get(f"/v1/jobs/{job_id}")
            job_payload = job_response.json()
            if job_payload["status"] == "completed":
                break
            time.sleep(0.05)

        repeat_response = client.post("/v1/models/convert", json={"model_id": hf_model_id})
        cache_response = client.get("/v1/cache/stats")

    assert job_payload is not None
    assert job_payload["status"] == "completed"
    assert job_payload["payload"]["sandboxed"] is True
    assert job_payload["payload"]["storage_mode"] == "encrypted_archive"
    assert job_payload["payload"]["cache_encrypted"] is True
    assert job_payload["payload"]["result_path"].endswith(".lewlmcache")
    assert Path(job_payload["payload"]["result_path"]).is_file()

    raw_bytes = Path(job_payload["payload"]["result_path"]).read_bytes()
    assert raw_bytes.startswith(ENCRYPTED_FILE_MAGIC)

    assert repeat_response.status_code == 200
    assert repeat_response.json()["payload"]["cache_hit"] is True
    assert repeat_response.json()["payload"]["sandboxed"] is True
    assert repeat_response.json()["payload"]["storage_mode"] == "encrypted_archive"
    assert repeat_response.json()["payload"]["cache_encrypted"] is True
    assert cache_response.status_code == 200
    assert cache_response.json()["artifact_count"] == 1
    assert cache_response.json()["file_count"] == 1


def test_encrypted_bootstrap_migrates_plain_conversion_cache(temp_settings, sample_models_root: Path) -> None:
    plaintext_services = bootstrap_services(
        temp_settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime()},
        conversion_backend=FakeMLXConversionBackend(),
    )
    try:
        manifests = plaintext_services.model_registry.scan().manifests
        hf_model_id = next(manifest.model_id for manifest in manifests if manifest.format_type == "huggingface")
        job = plaintext_services.conversion_service.submit(ConversionJobRequest(model_id=hf_model_id))
        if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
            for _ in range(30):
                job = plaintext_services.conversion_service.get_job(job.job_id)
                if job.status == JobStatus.COMPLETED:
                    break
                time.sleep(0.05)

        assert job.status == JobStatus.COMPLETED
        plaintext_path = Path(job.payload["result_path"])
        assert plaintext_path.is_dir()

        encrypted_settings = temp_settings.with_updates(
            persistence_encryption_enabled=True,
            persistence_encryption_passphrase=SecretStr("correct horse battery staple"),
            persistence_encryption_kdf_iterations=100_000,
        )
        encrypted_services = bootstrap_services(
            encrypted_settings,
            runtime_overrides={RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime()},
            conversion_backend=FakeMLXConversionBackend(),
        )
        try:
            artifacts = encrypted_services.metadata_store.list_conversion_artifacts()

            assert len(artifacts) == 1
            assert artifacts[0].metadata["storage_mode"] == "encrypted_archive"
            assert Path(artifacts[0].output_path).is_file()
            assert not plaintext_path.exists()
        finally:
            encrypted_services.close()
    finally:
        plaintext_services.close()
