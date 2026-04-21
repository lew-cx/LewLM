from __future__ import annotations

import json
import shutil
from types import SimpleNamespace
from pathlib import Path
import sys

from lewlm.config.settings import LewLMSettings
from lewlm.conversion.backend import MLXConversionBackend
from lewlm.conversion.service import ConversionService
from lewlm.conversion.models import ConversionCompatibilityReport, ConversionPolicy
from lewlm.core.errors import ConversionError
from lewlm.core.contracts import (
    ConversionStatus,
    ModelArtifactRole,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    QuantizationPrecision,
    QuantizationProfile,
    QuantizationStrategy,
    RuntimeAffinity,
    ValidationState,
)


def _manifest(
    path: Path,
    *,
    model_id: str,
    format_type: ModelFormat,
    modality: tuple[ModelModality, ...] = (ModelModality.TEXT,),
) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        display_name=model_id,
        architecture_family="gemma",
        modality=modality,
        source_path=str(path),
        format_type=format_type,
        runtime_affinity=(
            ((RuntimeAffinity.MLX_VISION,) if ModelModality.VISION in modality else (RuntimeAffinity.MLX_TEXT,))
            if format_type == ModelFormat.MLX
            else (
                (RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_VISION)
                if ModelModality.VISION in modality
                else (RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_TEXT)
            )
        ),
        conversion_status=(
            ConversionStatus.RUNNABLE if format_type == ModelFormat.MLX else ConversionStatus.REQUIRES_CONVERSION
        ),
        fingerprint=f"{model_id}-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )


def _conversion_service(temp_settings: LewLMSettings):
    recorded_artifacts = []
    service = object.__new__(ConversionService)
    service.settings = temp_settings
    service.backend = SimpleNamespace(name="mlx")
    service.audit_logger = SimpleNamespace(record=lambda **kwargs: None)
    service.metadata_store = SimpleNamespace(
        encryptor=None,
        upsert_conversion_artifact=recorded_artifacts.append,
    )
    return service, recorded_artifacts


def _compatibility(manifest: ModelManifest, output_path: Path) -> ConversionCompatibilityReport:
    return ConversionCompatibilityReport(
        model_id=manifest.model_id,
        source_format=manifest.format_type,
        backend_name="mlx",
        can_convert=True,
        reason="ok",
        cache_key="cache-key",
        output_path=str(output_path),
    )


def test_mlx_conversion_backend_reports_already_runnable_mlx_models(temp_settings: LewLMSettings, tmp_path: Path) -> None:
    source_dir = tmp_path / "gemma-mlx"
    source_dir.mkdir()
    backend = MLXConversionBackend()

    report = backend.compatibility_report(
        _manifest(source_dir, model_id="gemma-mlx", format_type=ModelFormat.MLX),
        settings=temp_settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is False
    assert report.already_runnable is True
    assert report.reason == "Model is already in MLX format and does not need conversion."
    assert report.output_path == str(source_dir)


def test_mlx_conversion_backend_reports_quantization_and_warning_for_supported_text_bundles(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    backend = MLXConversionBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "availability_reason", lambda: None)
    monkeypatch.setattr(backend, "_conversion_backend_available", lambda _backend: True)

    report = backend.compatibility_report(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=temp_settings.with_updates(allow_outbound_network=False),
        policy=ConversionPolicy.BALANCED,
        custom_bits=4,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is True
    assert report.quantization_mode == "4bit"
    assert report.custom_bits == 4
    assert report.warnings == ["Custom bits were provided without selecting the `custom_bits` policy."]


def test_mlx_conversion_backend_rejects_non_four_bit_custom_quantization(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    backend = MLXConversionBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "availability_reason", lambda: None)
    monkeypatch.setattr(backend, "_conversion_backend_available", lambda _backend: True)

    report = backend.compatibility_report(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=temp_settings,
        policy=ConversionPolicy.CUSTOM_BITS,
        custom_bits=8,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is False
    assert report.custom_bits == 8
    assert report.reason == "The current MLX conversion integration only supports 4-bit custom quantization requests."


def test_mlx_conversion_backend_reports_hybrid_fp8_profile_as_unsupported(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    backend = MLXConversionBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "availability_reason", lambda: None)
    monkeypatch.setattr(backend, "_conversion_backend_available", lambda _backend: True)

    report = backend.compatibility_report(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=temp_settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=QuantizationProfile(
            strategy=QuantizationStrategy.HYBRID_FP8,
            weight_precision=QuantizationPrecision.INT4,
            compute_precision=QuantizationPrecision.FP8_E4M3,
        ),
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is False
    assert report.profile_support[0].requires_native_fp8 is True
    assert "does not expose native FP8" in report.reason


def test_mlx_conversion_backend_requires_local_source_when_network_is_disabled(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = MLXConversionBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "availability_reason", lambda: None)
    monkeypatch.setattr(backend, "_conversion_backend_available", lambda _backend: True)

    report = backend.compatibility_report(
        _manifest(tmp_path / "missing-model", model_id="remote-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=temp_settings.with_updates(allow_outbound_network=False),
        policy=ConversionPolicy.MAX_QUALITY,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is False
    assert report.reason == "Conversion requires a local model path when outbound network access is disabled."


def test_mlx_conversion_backend_uses_current_cli_flags(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    output_dir = tmp_path / "converted"
    backend = MLXConversionBackend()
    recorded: dict[str, object] = {}

    def fake_run(command, *, capture_output, text, cwd, check):
        recorded["command"] = command
        recorded["cwd"] = cwd
        output_dir.mkdir()
        return SimpleNamespace(returncode=0, stdout="converted\n", stderr="")

    monkeypatch.setattr("lewlm.conversion.backend.subprocess.run", fake_run)

    result = backend.convert(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=temp_settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=4,
        quantization_profile=None,
        output_path=output_dir,
        work_dir=tmp_path,
    )

    assert result.output_path == output_dir
    assert recorded["cwd"] == tmp_path
    assert recorded["command"] == [
        sys.executable,
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        str(source_dir),
        "--mlx-path",
        str(output_dir),
        "-q",
        "--q-bits",
        "4",
    ]


def test_mlx_conversion_backend_routes_vision_bundles_to_mlx_vlm(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma4-vl"
    source_dir.mkdir()
    output_dir = tmp_path / "converted"
    backend = MLXConversionBackend()
    recorded: list[list[str]] = []

    def fake_run(command, *, capture_output, text, cwd, check):
        recorded.append(command)
        Path(command[command.index("--mlx-path") + 1]).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="converted\n", stderr="")

    monkeypatch.setattr("lewlm.conversion.backend.subprocess.run", fake_run)
    monkeypatch.setattr(backend, "_conversion_backend_available", lambda _backend: True)

    manifest = _manifest(
        source_dir,
        model_id="gemma4-vl",
        format_type=ModelFormat.HUGGINGFACE,
        modality=(ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL),
    )
    report = backend.compatibility_report(
        manifest,
        settings=temp_settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=output_dir,
    )
    result = backend.convert(
        manifest,
        settings=temp_settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        output_path=output_dir,
        work_dir=tmp_path,
    )

    assert report.can_convert is True
    assert report.backend_name == "mlx_vlm"
    assert report.layered_output is True
    assert [artifact.role for artifact in report.artifact_plans] == [
        ModelArtifactRole.MULTIMODAL_RUNNABLE,
        ModelArtifactRole.TEXT_RUNNABLE,
    ]
    assert [command[:4] for command in recorded] == [
        [sys.executable, "-m", "mlx_vlm", "convert"],
        [sys.executable, "-m", "mlx_lm", "convert"],
    ]
    assert result.output_path == output_dir
    assert [artifact.role for artifact in result.artifacts] == [
        ModelArtifactRole.MULTIMODAL_RUNNABLE,
        ModelArtifactRole.TEXT_RUNNABLE,
    ]
    assert result.artifacts[0].output_path == output_dir / "multimodal"
    assert result.artifacts[1].output_path == output_dir / "text"


def test_conversion_service_write_output_path_cleans_partial_cache_on_no_space(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _ = _conversion_service(temp_settings)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "weights.safetensors").write_bytes(b"weights")
    target_dir = tmp_path / "target"

    def fake_copytree(source: Path, target: Path):
        target.mkdir(parents=True, exist_ok=True)
        (target / "partial.safetensors").write_bytes(b"partial")
        raise shutil.Error(
            [
                (
                    str(source / "weights.safetensors"),
                    str(target / "weights.safetensors"),
                    "[Errno 28] No space left on device",
                ),
            ],
        )

    monkeypatch.setattr("lewlm.conversion.service.shutil.copytree", fake_copytree)

    try:
        service._write_output_path(source_dir, target_dir, force=False)
    except ConversionError as exc:
        assert "Insufficient disk space" in str(exc)
    else:
        raise AssertionError("expected ConversionError")

    assert not target_dir.exists()


def test_conversion_service_recovers_valid_orphaned_conversion_cache(
    temp_settings: LewLMSettings,
    tmp_path: Path,
) -> None:
    service, recorded_artifacts = _conversion_service(temp_settings)
    source_dir = tmp_path / "source-model"
    source_dir.mkdir()
    output_dir = tmp_path / "converted-model"
    output_dir.mkdir()
    (output_dir / "config.json").write_text(json.dumps({"model_type": "gemma"}))
    (output_dir / "weights.safetensors").write_bytes(b"weights")

    artifact = service._recover_or_cleanup_orphaned_output(
        manifest=_manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        cache_key="cache-key",
        output_path=output_dir,
        policy=ConversionPolicy.BALANCED,
        quantization_profile=None,
        compatibility=_compatibility(
            _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
            output_dir,
        ),
    )

    assert artifact is not None
    assert artifact.output_path == str(output_dir)
    assert len(recorded_artifacts) == 1
    assert recorded_artifacts[0].output_path == str(output_dir)


def test_conversion_service_removes_invalid_orphaned_conversion_cache(
    temp_settings: LewLMSettings,
    tmp_path: Path,
) -> None:
    service, recorded_artifacts = _conversion_service(temp_settings)
    source_dir = tmp_path / "source-model"
    source_dir.mkdir()
    output_dir = tmp_path / "converted-model"
    output_dir.mkdir()
    (output_dir / "config.json").write_text("{}")

    artifact = service._recover_or_cleanup_orphaned_output(
        manifest=_manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        cache_key="cache-key",
        output_path=output_dir,
        policy=ConversionPolicy.BALANCED,
        quantization_profile=None,
        compatibility=_compatibility(
            _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
            output_dir,
        ),
    )

    assert artifact is None
    assert recorded_artifacts == []
    assert not output_dir.exists()


def test_conversion_service_cache_key_includes_quantization_profile(temp_settings: LewLMSettings) -> None:
    service, _ = _conversion_service(temp_settings)

    baseline = service._build_cache_key("fingerprint", ConversionPolicy.BALANCED, None, None)
    mixed_precision = service._build_cache_key(
        "fingerprint",
        ConversionPolicy.BALANCED,
        None,
        QuantizationProfile(
            strategy=QuantizationStrategy.MIXED_PRECISION,
            layer_overrides=[
                {
                    "layer_pattern": "layers.0.attn.q_proj",
                    "weight_precision": "int8",
                    "compute_precision": "bf16",
                },
            ],
        ),
    )

    assert baseline != mixed_precision


def test_conversion_service_clear_cache_removes_artifacts_and_idempotency_records(temp_settings: LewLMSettings) -> None:
    service = object.__new__(ConversionService)
    service.settings = temp_settings
    service.backend = SimpleNamespace(name="mlx")
    service.audit_logger = SimpleNamespace(record=lambda **kwargs: None)
    cleared: dict[str, object] = {}
    service.metadata_store = SimpleNamespace(
        clear_conversion_artifacts=lambda: 2,
        clear_idempotent_operation_results=lambda *, operation_name: (
            cleared.setdefault("operation_name", operation_name),
            1,
        )[1],
    )
    conversion_root = temp_settings.cache_dir / "conversions"
    nested_dir = conversion_root / "cache-key"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "weights.safetensors").write_bytes(b"weights")
    (conversion_root / "cache-key.lewlmcache").write_bytes(b"archive")

    payload = service.clear_cache()

    assert payload["cache_root"] == str(conversion_root)
    assert payload["removed_entries"] == 2
    assert payload["cleared_artifact_records"] == 2
    assert payload["cleared_idempotent_records"] == 1
    assert cleared["operation_name"] == "models.convert"
    assert conversion_root.exists()
    assert list(conversion_root.iterdir()) == []


def test_conversion_service_removes_discoverable_but_nonrunnable_orphaned_cache(
    temp_settings: LewLMSettings,
    tmp_path: Path,
) -> None:
    service, recorded_artifacts = _conversion_service(temp_settings)
    source_dir = tmp_path / "source-model"
    source_dir.mkdir()
    output_dir = tmp_path / "converted-model"
    output_dir.mkdir()
    (output_dir / "config.json").write_text(json.dumps({"model_type": "gemma"}))
    (output_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (output_dir / "tokenizer.json").write_text("{}")

    artifact = service._recover_or_cleanup_orphaned_output(
        manifest=_manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        cache_key="cache-key",
        output_path=output_dir,
        policy=ConversionPolicy.BALANCED,
        quantization_profile=None,
        compatibility=_compatibility(
            _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
            output_dir,
        ),
    )

    assert artifact is None
    assert recorded_artifacts == []
    assert not output_dir.exists()
