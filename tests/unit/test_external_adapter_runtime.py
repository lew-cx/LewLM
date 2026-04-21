from __future__ import annotations

from pathlib import Path

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    ConversionStatus,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.runtime.adapters import LocalOpenAICompatibleAdapterRuntime, summarize_feature_preservation


def test_external_adapter_runtime_requires_loopback_endpoint(tmp_path: Path) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://example.com:8080",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)

    assert runtime.is_available() is False
    assert "loopback-only" in str(runtime.availability_reason())


def test_external_adapter_runtime_matches_advertised_model_and_reports_feature_preservation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
        external_accelerator_profile="vllm_mlx",
    )
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    manifest = ModelManifest(
        model_id="demo-model",
        display_name="Demo Model",
        architecture_family="llama",
        modality=(ModelModality.TEXT,),
        source_path=str(tmp_path / "demo-model"),
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
        estimated_memory_mb=512,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="demo-model-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )

    monkeypatch.setattr(
        runtime,
        "_request_json",
        lambda method, path, payload: {"data": [{"id": "demo-model"}]},
    )

    report = runtime.candidate_report(manifest)
    feature_preservation = summarize_feature_preservation(
        native_features={
            "continuous_batching": {"supported": True},
            "prefix_cache": {"supported": True},
            "kv_cache_quantization": {"supported": True},
        },
        external_features=runtime.performance_feature_snapshot(),
    )

    assert report.available is True
    assert report.supports_manifest is True
    assert "continuous_batching" in feature_preservation["preserved"]
    assert "kv_cache_quantization" in feature_preservation["degraded"]
