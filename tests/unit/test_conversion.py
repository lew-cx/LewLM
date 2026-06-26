from __future__ import annotations

import json
import shutil
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

from lewlm.config.settings import LewLMSettings
from lewlm.conversion.backend import (
    AutoConversionBackend,
    LlamaCppConversionBackend,
    MLXConversionBackend,
    OnnxGenAIConversionBackend,
)
from lewlm.conversion.jang import JangNormalizationResult, normalize_jang_bundle
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
from lewlm.registry.discovery import discover_models


def _manifest(
    path: Path,
    *,
    model_id: str,
    format_type: ModelFormat,
    modality: tuple[ModelModality, ...] = (ModelModality.TEXT,),
) -> ModelManifest:
    if format_type == ModelFormat.GGUF:
        runtime_affinity = (RuntimeAffinity.LLAMACPP,)
        conversion_status = ConversionStatus.RUNNABLE
    elif format_type == ModelFormat.ONNX_GENAI:
        runtime_affinity = (RuntimeAffinity.ONNX_GENAI,)
        conversion_status = ConversionStatus.RUNNABLE
    elif format_type == ModelFormat.MLX:
        runtime_affinity = (RuntimeAffinity.MLX_VISION,) if ModelModality.VISION in modality else (RuntimeAffinity.MLX_TEXT,)
        conversion_status = ConversionStatus.RUNNABLE
    else:
        runtime_affinity = (
            (RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_VISION)
            if ModelModality.VISION in modality
            else (RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_TEXT)
        )
        conversion_status = ConversionStatus.REQUIRES_CONVERSION
    return ModelManifest(
        model_id=model_id,
        display_name=model_id,
        architecture_family="gemma",
        modality=modality,
        source_path=str(path),
        format_type=format_type,
        runtime_affinity=runtime_affinity,
        conversion_status=conversion_status,
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


def test_mlx_conversion_backend_reports_already_runnable_mlx_models(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma-mlx"
    source_dir.mkdir()
    backend = MLXConversionBackend()
    monkeypatch.setattr("lewlm.conversion.backend.platform.system", lambda: "Darwin")
    monkeypatch.setattr("lewlm.conversion.backend.platform.machine", lambda: "arm64")

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


def test_mlx_conversion_backend_reports_mlx_host_unsupported_on_windows(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma-mlx"
    source_dir.mkdir()
    backend = MLXConversionBackend()
    monkeypatch.setattr("lewlm.conversion.backend.platform.system", lambda: "Windows")
    monkeypatch.setattr("lewlm.conversion.backend.platform.machine", lambda: "AMD64")

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
    assert report.already_runnable is False
    assert "Apple Silicon macOS" in report.reason


def test_llamacpp_conversion_backend_reports_hf_to_gguf_plan(
    temp_settings: LewLMSettings,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    converter = tmp_path / "convert_hf_to_gguf.py"
    quantizer = tmp_path / "llama-quantize.exe"
    converter.write_text("# converter", encoding="utf-8")
    quantizer.write_text("quantizer", encoding="utf-8")
    settings = temp_settings.with_updates(
        llamacpp_convert_hf_to_gguf_path=converter,
        llamacpp_quantize_path=quantizer,
    )
    backend = LlamaCppConversionBackend()

    report = backend.compatibility_report(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is True
    assert report.target_format == "gguf"
    assert report.backend_name == "llamacpp_gguf"
    assert report.quantization_mode == "q4_k_m"
    assert report.artifact_plans[0].format_type == ModelFormat.GGUF
    assert report.artifact_plans[0].runtime_affinity == (RuntimeAffinity.LLAMACPP,)
    assert report.artifact_plans[0].relative_path.endswith("-q4_k_m.gguf")


def test_llamacpp_conversion_backend_reports_missing_quantizer_for_quantized_exports(
    temp_settings: LewLMSettings,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("# converter", encoding="utf-8")
    backend = LlamaCppConversionBackend()

    report = backend.compatibility_report(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=temp_settings.with_updates(llamacpp_convert_hf_to_gguf_path=converter),
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is False
    assert "llama.cpp `llama-quantize`" in report.reason


def test_llamacpp_conversion_backend_allows_direct_q8_export_without_quantizer(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    output_dir = tmp_path / "converted"
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("# converter", encoding="utf-8")
    settings = temp_settings.with_updates(llamacpp_convert_hf_to_gguf_path=converter)
    backend = LlamaCppConversionBackend()
    recorded: list[list[str]] = []

    report = backend.compatibility_report(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=settings,
        policy=ConversionPolicy.CUSTOM_BITS,
        custom_bits=8,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=output_dir,
    )

    def fake_run(command, *, capture_output, text, cwd, check):
        recorded.append(command)
        Path(command[command.index("--outfile") + 1]).write_bytes(b"q8-gguf")
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr("lewlm.conversion.backend.subprocess.run", fake_run)
    result = backend.convert(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=settings,
        policy=ConversionPolicy.CUSTOM_BITS,
        custom_bits=8,
        quantization_profile=None,
        output_path=output_dir,
        work_dir=tmp_path,
    )

    assert report.can_convert is True
    assert report.quantization_mode == "q8_0"
    assert result.artifacts[0].output_path == output_dir / "gemma-hf-q8_0.gguf"
    assert recorded == [
        [
            sys.executable,
            str(converter),
            str(source_dir),
            "--outfile",
            str(output_dir / "gemma-hf-q8_0.gguf"),
            "--outtype",
            "q8_0",
        ],
    ]


def test_llamacpp_conversion_backend_detects_local_llamacpp_checkout(
    temp_settings: LewLMSettings,
) -> None:
    llama_root = temp_settings.data_dir / "tools" / "llama.cpp"
    (llama_root / "gguf-py").mkdir(parents=True)
    (llama_root / "conversion").mkdir()
    converter = llama_root / "convert_hf_to_gguf.py"
    converter.write_text("# converter", encoding="utf-8")

    tool = LlamaCppConversionBackend._converter_tool(temp_settings)

    assert tool is not None
    assert tool.command == (sys.executable, str(converter))
    assert tool.cwd == llama_root
    assert tool.env is not None
    assert str(llama_root) in tool.env["PYTHONPATH"]


def test_llamacpp_conversion_backend_reports_jang_multimodal_bundle_convertible(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma4-jang"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "quantization": {"bits": 4, "group_size": 64},
                "vision_config": {"image_size": 896},
            },
        ),
        encoding="utf-8",
    )
    (source_dir / "jang_config.json").write_text(json.dumps({"format": "jang"}), encoding="utf-8")
    converter = tmp_path / "convert_hf_to_gguf.py"
    quantizer = tmp_path / "llama-quantize.exe"
    converter.write_text("# converter", encoding="utf-8")
    quantizer.write_text("# quantizer", encoding="utf-8")
    monkeypatch.setattr("lewlm.conversion.backend.missing_jang_dependencies", lambda: [])

    report = LlamaCppConversionBackend().compatibility_report(
        _manifest(
            source_dir,
            model_id="gemma4-jang",
            format_type=ModelFormat.HUGGINGFACE,
            modality=(ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL),
        ),
        settings=temp_settings.with_updates(
            llamacpp_convert_hf_to_gguf_path=converter,
            llamacpp_quantize_path=quantizer,
        ),
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is True
    assert "normalized from JANG" in report.reason
    assert report.artifact_plans[0].modality == (ModelModality.TEXT,)
    assert report.artifact_plans[0].metadata["source_preprocessing"] == "jang_normalization"
    assert report.artifact_plans[0].metadata["multimodal_support"] == "probe_gated"
    assert any("JANG-packed" in warning for warning in report.warnings)


def test_llamacpp_conversion_backend_rejects_unmapped_custom_bits(
    temp_settings: LewLMSettings,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    backend = LlamaCppConversionBackend()

    report = backend.compatibility_report(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=temp_settings,
        policy=ConversionPolicy.CUSTOM_BITS,
        custom_bits=7,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is False
    assert "7-bit weights" in report.reason


def test_llamacpp_conversion_backend_normalizes_jang_before_export(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma4-jang"
    source_dir.mkdir()
    (source_dir / "jang_config.json").write_text(json.dumps({"format": "jang"}), encoding="utf-8")
    output_dir = tmp_path / "converted"
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("# converter", encoding="utf-8")
    settings = temp_settings.with_updates(llamacpp_convert_hf_to_gguf_path=converter)
    backend = LlamaCppConversionBackend()
    recorded: dict[str, object] = {}

    def fake_normalize(source_path: Path, output_path: Path) -> JangNormalizationResult:
        output_path.mkdir(parents=True)
        return JangNormalizationResult(
            source_path=output_path,
            logs=["normalized jang"],
            metadata={"converted_tensors": 1},
        )

    def fake_run(command, *, capture_output, text, cwd, check):
        recorded["command"] = command
        recorded["cwd"] = cwd
        Path(command[command.index("--outfile") + 1]).write_bytes(b"f16-gguf")
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr("lewlm.conversion.backend.normalize_jang_bundle", fake_normalize)
    monkeypatch.setattr("lewlm.conversion.backend.subprocess.run", fake_run)

    result = backend.convert(
        _manifest(
            source_dir,
            model_id="gemma4-jang",
            format_type=ModelFormat.HUGGINGFACE,
            modality=(ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL),
        ),
        settings=settings,
        policy=ConversionPolicy.MAX_QUALITY,
        custom_bits=None,
        quantization_profile=None,
        output_path=output_dir,
        work_dir=tmp_path,
    )

    assert recorded["command"][2] == str(tmp_path / "jang-normalized-hf")
    assert recorded["cwd"] == tmp_path
    assert result.logs[0] == "normalized jang"
    assert result.artifacts[0].metadata["source_preprocessing"] == "jang_normalization"
    assert result.artifacts[0].metadata["converted_tensors"] == 1


def test_llamacpp_conversion_backend_runs_export_and_quantize_commands(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    output_dir = tmp_path / "converted"
    converter = tmp_path / "convert_hf_to_gguf.py"
    quantizer = tmp_path / "llama-quantize.exe"
    converter.write_text("# converter", encoding="utf-8")
    quantizer.write_text("quantizer", encoding="utf-8")
    settings = temp_settings.with_updates(
        llamacpp_convert_hf_to_gguf_path=converter,
        llamacpp_quantize_path=quantizer,
    )
    backend = LlamaCppConversionBackend()
    recorded: list[list[str]] = []

    def fake_run(command, *, capture_output, text, cwd, check):
        recorded.append(command)
        if "--outfile" in command:
            Path(command[command.index("--outfile") + 1]).write_bytes(b"f16-gguf")
        else:
            Path(command[2]).write_bytes(b"q4-gguf")
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr("lewlm.conversion.backend.subprocess.run", fake_run)

    result = backend.convert(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        output_path=output_dir,
        work_dir=tmp_path,
    )

    assert result.output_path == output_dir
    assert result.artifacts[0].format_type == ModelFormat.GGUF
    assert result.artifacts[0].runtime_affinity == (RuntimeAffinity.LLAMACPP,)
    assert result.artifacts[0].output_path == output_dir / "gemma-hf-q4_k_m.gguf"
    assert recorded[0][:2] == [sys.executable, str(converter)]
    assert recorded[1] == [str(quantizer), str(tmp_path / "gemma-hf-q4_k_m-f16.gguf"), str(output_dir / "gemma-hf-q4_k_m.gguf"), "Q4_K_M"]


def test_normalize_jang_bundle_dequantizes_tiny_fixture(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    from safetensors import safe_open
    from safetensors.numpy import save_file

    source_dir = tmp_path / "jang-source"
    output_dir = tmp_path / "normalized"
    source_dir.mkdir()
    packed_values = sum(int(value) << (4 * index) for index, value in enumerate(range(8)))
    save_file(
        {
            "model.layers.0.weight": np.array([[packed_values]], dtype=np.uint32),
            "model.layers.0.scales": np.array([[1.0, 1.0]], dtype=np.float16),
            "model.layers.0.biases": np.array([[0.0, 0.0]], dtype=np.float16),
            "model.norm.weight": np.array([1.0, 2.0], dtype=np.float16),
        },
        str(source_dir / "model-00001-of-00001.safetensors"),
        metadata={"format": "jang"},
    )
    (source_dir / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "quantization": {"bits": 4, "group_size": 4}}),
        encoding="utf-8",
    )
    (source_dir / "jang_config.json").write_text(json.dumps({"format": "jang"}), encoding="utf-8")
    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"format": "jang"},
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.scales": "model-00001-of-00001.safetensors",
                    "model.layers.0.biases": "model-00001-of-00001.safetensors",
                    "model.norm.weight": "model-00001-of-00001.safetensors",
                },
            },
        ),
        encoding="utf-8",
    )

    result = normalize_jang_bundle(source_dir, output_dir)
    output_index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_file = output_dir / output_index["weight_map"]["model.layers.0.weight"]

    assert result.metadata["converted_tensors"] == 1
    assert result.metadata["skipped_sidecar_tensors"] == 2
    assert "quantization" not in json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert "model.layers.0.scales" not in output_index["weight_map"]
    assert "model.layers.0.biases" not in output_index["weight_map"]
    with safe_open(str(weight_file), framework="numpy") as output:
        assert output.get_tensor("model.layers.0.weight").tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]


def test_auto_conversion_backend_prefers_llamacpp_gguf_on_windows(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    converter = tmp_path / "convert_hf_to_gguf.py"
    quantizer = tmp_path / "llama-quantize.exe"
    converter.write_text("# converter", encoding="utf-8")
    quantizer.write_text("quantizer", encoding="utf-8")
    monkeypatch.setattr("lewlm.conversion.backend.platform.system", lambda: "Windows")
    monkeypatch.setattr("lewlm.conversion.backend.platform.machine", lambda: "AMD64")

    report = AutoConversionBackend().compatibility_report(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=temp_settings.with_updates(
            llamacpp_convert_hf_to_gguf_path=converter,
            llamacpp_quantize_path=quantizer,
        ),
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is True
    assert report.backend_name == "llamacpp_gguf"
    assert report.target_format == "gguf"


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


def test_mlx_conversion_backend_reports_host_unsupported_for_non_macos_conversion(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    backend = MLXConversionBackend()
    monkeypatch.setattr("lewlm.conversion.backend.platform.system", lambda: "Windows")
    monkeypatch.setattr("lewlm.conversion.backend.platform.machine", lambda: "AMD64")

    report = backend.compatibility_report(
        _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=temp_settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is False
    assert report.reason == "MLX conversion is only supported on Apple Silicon macOS hosts."


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


def test_mlx_conversion_backend_preserves_dual_artifact_plan_for_discovered_sharded_gemma4_bundle(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma4-vl"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "quantization": {"bits": 4, "group_size": 64},
                "text_config": {"hidden_size": 2048},
                "vision_config": {"image_size": 448},
            },
        ),
        encoding="utf-8",
    )
    (source_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (source_dir / "model-00001-of-00002.safetensors").write_bytes(b"weights-1")
    (source_dir / "model-00002-of-00002.safetensors").write_bytes(b"weights-2")
    (source_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source_dir / "processor_config.json").write_text("{}", encoding="utf-8")

    manifests = discover_models([tmp_path])
    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.format_type == ModelFormat.HUGGINGFACE

    backend = MLXConversionBackend()
    monkeypatch.setattr(backend, "availability_reason", lambda: None)
    monkeypatch.setattr(backend, "_conversion_backend_available", lambda _backend: True)

    report = backend.compatibility_report(
        manifest,
        settings=temp_settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
    )

    assert report.can_convert is True
    assert report.layered_output is True
    assert [artifact.role for artifact in report.artifact_plans] == [
        ModelArtifactRole.MULTIMODAL_RUNNABLE,
        ModelArtifactRole.TEXT_RUNNABLE,
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
    monkeypatch.setattr(backend, "availability_reason", lambda: None)
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


def test_conversion_service_plans_executable_gguf_and_install_gated_onnx_targets(
    temp_settings: LewLMSettings,
    tmp_path: Path,
) -> None:
    converter = tmp_path / "convert_hf_to_gguf.py"
    quantizer = tmp_path / "llama-quantize.exe"
    converter.write_text("# converter", encoding="utf-8")
    quantizer.write_text("# quantizer", encoding="utf-8")
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    manifest = _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE)
    service, _ = _conversion_service(
        temp_settings.with_updates(
            llamacpp_convert_hf_to_gguf_path=converter,
            llamacpp_quantize_path=quantizer,
        ),
    )
    service.model_registry = SimpleNamespace(get_manifest=lambda model_id: manifest)

    report = service.plan_targets(manifest.model_id)
    targets = {target.target_id: target for target in report.targets}

    assert report.default_target_id == "gguf_llamacpp"
    assert targets["gguf_llamacpp"].can_convert is True
    assert targets["gguf_llamacpp"].state == "available"
    assert targets["gguf_llamacpp"].runtime_provider.value == "llamacpp"
    # onnxruntime-genai is not installed in the test environment, so the ONNX
    # target is install-gated rather than executable here.
    assert targets["onnx_genai"].can_convert is False
    assert targets["onnx_genai"].state == "requires_install"
    assert "onnxruntime-genai" in str(targets["onnx_genai"].reason)
    assert targets["onnx_genai"].backend_name == "onnx_genai_builder"


def test_conversion_service_plans_existing_onnx_bundle_as_probe_gated_runnable(
    temp_settings: LewLMSettings,
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "phi-onnx", model_id="phi-onnx", format_type=ModelFormat.ONNX_GENAI)
    service, _ = _conversion_service(temp_settings)
    service.model_registry = SimpleNamespace(get_manifest=lambda model_id: manifest)

    report = service.plan_targets(manifest.model_id)
    targets = {target.target_id: target for target in report.targets}

    assert report.default_target_id == "onnx_genai"
    assert targets["onnx_genai"].already_runnable is True
    assert targets["onnx_genai"].state == "already_runnable"


def _force_onnx_builder_available(monkeypatch) -> None:
    monkeypatch.setattr(
        "lewlm.conversion.backend.importlib.util.find_spec",
        lambda name, *args, **kwargs: object(),
    )


def test_onnx_genai_conversion_backend_requires_install_when_missing(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("lewlm.conversion.backend.importlib.util.find_spec", lambda name, *a, **k: None)
    backend = OnnxGenAIConversionBackend()
    assert backend.is_available() is False
    report = backend.compatibility_report(
        _manifest(tmp_path / "phi-hf", model_id="phi-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=temp_settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache",
        output_path=tmp_path / "out",
    )
    assert report.can_convert is False
    assert "onnxruntime-genai" in report.reason


def test_onnx_genai_conversion_backend_reports_hf_to_onnx_plan(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    _force_onnx_builder_available(monkeypatch)
    backend = OnnxGenAIConversionBackend()
    assert backend.is_available() is True
    report = backend.compatibility_report(
        _manifest(tmp_path / "phi-hf", model_id="phi-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=temp_settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache",
        output_path=tmp_path / "out",
    )
    assert report.can_convert is True
    assert report.backend_name == "onnx_genai_builder"
    assert report.target_format == ModelFormat.ONNX_GENAI.value
    assert report.quantization_mode == "int4"
    assert report.artifact_plans[0].format_type == ModelFormat.ONNX_GENAI


def test_onnx_genai_conversion_backend_reports_already_runnable_onnx_bundle(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    _force_onnx_builder_available(monkeypatch)
    report = OnnxGenAIConversionBackend().compatibility_report(
        _manifest(tmp_path / "phi-onnx", model_id="phi-onnx", format_type=ModelFormat.ONNX_GENAI),
        settings=temp_settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache",
        output_path=tmp_path / "out",
    )
    assert report.already_runnable is True
    assert report.can_convert is False


def test_onnx_genai_conversion_backend_runs_builder_command(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    _force_onnx_builder_available(monkeypatch)
    source_dir = tmp_path / "phi-hf"
    source_dir.mkdir()
    output_dir = tmp_path / "converted-onnx"
    recorded: list[list[str]] = []

    def fake_run(command, *, capture_output, text, cwd, check):
        recorded.append(command)
        Path(command[command.index("-o") + 1]).mkdir(parents=True, exist_ok=True)
        (Path(command[command.index("-o") + 1]) / "model.onnx").write_bytes(b"onnx")
        return SimpleNamespace(returncode=0, stdout="built\n", stderr="")

    monkeypatch.setattr("lewlm.conversion.backend.subprocess.run", fake_run)

    settings = temp_settings.with_updates(onnx_genai_conversion_execution_provider="dml")
    result = OnnxGenAIConversionBackend().convert(
        _manifest(source_dir, model_id="phi-hf", format_type=ModelFormat.HUGGINGFACE),
        settings=settings,
        policy=ConversionPolicy.MAX_QUALITY,
        custom_bits=None,
        quantization_profile=None,
        output_path=output_dir,
        work_dir=tmp_path,
    )

    assert result.output_path == output_dir
    assert result.artifacts[0].format_type == ModelFormat.ONNX_GENAI
    assert result.artifacts[0].runtime_affinity == (RuntimeAffinity.ONNX_GENAI,)
    assert result.artifacts[0].metadata["precision"] == "fp16"
    assert result.artifacts[0].metadata["execution_provider"] == "dml"
    command = recorded[0]
    assert command[:3] == [sys.executable, "-m", "onnxruntime_genai.models.builder"]
    assert command[command.index("-i") + 1] == str(source_dir)
    assert command[command.index("-o") + 1] == str(output_dir)
    assert command[command.index("-p") + 1] == "fp16"
    assert command[command.index("-e") + 1] == "dml"


def test_onnx_genai_conversion_backend_rejects_non_text_source(
    temp_settings: LewLMSettings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    _force_onnx_builder_available(monkeypatch)
    report = OnnxGenAIConversionBackend().compatibility_report(
        _manifest(
            tmp_path / "vision-hf",
            model_id="vision-hf",
            format_type=ModelFormat.HUGGINGFACE,
            modality=(ModelModality.VISION,),
        ),
        settings=temp_settings,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache",
        output_path=tmp_path / "out",
    )
    assert report.can_convert is False
    assert "text generation models" in report.reason


def test_conversion_service_selects_explicit_gguf_target_backend(
    temp_settings: LewLMSettings,
    tmp_path: Path,
) -> None:
    converter = tmp_path / "convert_hf_to_gguf.py"
    quantizer = tmp_path / "llama-quantize.exe"
    converter.write_text("# converter", encoding="utf-8")
    quantizer.write_text("# quantizer", encoding="utf-8")
    source_dir = tmp_path / "gemma-hf"
    source_dir.mkdir()
    manifest = _manifest(source_dir, model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE)
    service, _ = _conversion_service(
        temp_settings.with_updates(
            llamacpp_convert_hf_to_gguf_path=converter,
            llamacpp_quantize_path=quantizer,
        ),
    )

    compatibility, backend = service._compatibility_for_request(
        manifest,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
        target_id="gguf_llamacpp",
    )

    assert backend.name == "llamacpp_gguf"
    assert compatibility.backend_name == "llamacpp_gguf"
    assert compatibility.can_convert is True


def test_conversion_service_reports_install_gated_onnx_target_for_hf_sources(
    temp_settings: LewLMSettings,
    tmp_path: Path,
) -> None:
    # onnxruntime-genai is not installed in the test environment, so an ONNX
    # target request routes to the real builder backend and reports the install
    # requirement rather than a hard "planned-only" rejection.
    manifest = _manifest(tmp_path / "gemma-hf", model_id="gemma-hf", format_type=ModelFormat.HUGGINGFACE)
    service, _ = _conversion_service(temp_settings)

    compatibility, backend = service._compatibility_for_request(
        manifest,
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=tmp_path / "converted",
        target_id="onnx_genai",
    )

    assert backend.name == "onnx_genai_builder"
    assert compatibility.backend_name == "onnx_genai_builder"
    assert compatibility.target_format == "onnx_genai"
    assert compatibility.can_convert is False
    assert "onnxruntime-genai" in compatibility.reason


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
