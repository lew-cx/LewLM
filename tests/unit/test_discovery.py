from __future__ import annotations

from pathlib import Path
import json

from lewlm.conversion.models import CONVERSION_OUTPUT_METADATA_FILENAME, LAYERED_CONVERSION_MANIFEST_FILENAME
from lewlm.core.contracts import (
    ArchitectureSubtype,
    ConversionStatus,
    ModelArtifactRole,
    ModelFormat,
    ModelModality,
    QuantizationPrecision,
    QuantizationStrategy,
    RuntimeAffinity,
)
from lewlm.registry.discovery import discover_models
from lewlm.registry.discovery import (
    _build_converted_model_id,
    _build_model_id,
    _fingerprint_path,
    _infer_modalities,
    _infer_quantization,
    _infer_runtime_affinity,
)


def test_model_id_builder_normalizes_display_names() -> None:
    assert _build_model_id("Gemma 4 / 31B IT", "abcdef1234567890") == "gemma-4-31b-it-abcdef123456"
    assert _build_model_id("!!!", "1234567890abcdef") == "model-1234567890ab"


def test_converted_model_id_builder_uses_readable_suffixes() -> None:
    assert _build_converted_model_id("Gemma 4 / 31B IT", artifact_role=ModelArtifactRole.MULTIMODAL_RUNNABLE) == (
        "gemma-4-31b-it_converted"
    )
    assert _build_converted_model_id("Gemma 4 / 31B IT", artifact_role=ModelArtifactRole.TEXT_RUNNABLE) == (
        "gemma-4-31b-it_converted_text"
    )


def test_discovery_reads_bom_prefixed_conversion_output_metadata_for_gguf(tmp_path: Path) -> None:
    output_dir = tmp_path / "converted"
    output_dir.mkdir()
    (output_dir / "gemma-q8_0.gguf").write_bytes(b"gguf")
    (output_dir / CONVERSION_OUTPUT_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "source_display_name": "Gemma Source",
                "source_model_id": "gemma-source",
                "display_name": "Gemma Source (converted)",
                "artifact_role": "standalone",
                "artifact_family_id": "cache-key",
                "metadata": {"source_preprocessing": "jang_normalization"},
            },
        ),
        encoding="utf-8-sig",
    )

    manifests = discover_models([output_dir])

    assert len(manifests) == 1
    assert manifests[0].model_id == "gemma-source_converted"
    assert manifests[0].display_name == "Gemma Source (converted)"
    assert manifests[0].metadata["converted_output"] is True
    assert manifests[0].metadata["source_preprocessing"] == "jang_normalization"


def test_fingerprint_path_is_stable_until_bundle_contents_change(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "model"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text('{"model_type":"gemma"}', encoding="utf-8")
    (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    first = _fingerprint_path(bundle_dir)
    second = _fingerprint_path(bundle_dir)
    (bundle_dir / "config.json").write_text('{"model_type":"gemma","revision":2}', encoding="utf-8")
    third = _fingerprint_path(bundle_dir)

    assert first == second
    assert third != first


def test_quantization_inference_prefers_filename_then_config_bits() -> None:
    assert _infer_quantization("llama-3-q4_k_m.gguf") == "q4_k_m"
    assert _infer_quantization("gemma-mlx", {"quantization": {"bits": 8}}) == "int8"


def test_modality_and_runtime_inference_cover_vision_and_audio_paths(tmp_path: Path) -> None:
    vision_modalities = _infer_modalities(
        path=tmp_path / "qwen2-vl-vision-mlx",
        file_names={"config.json", "weights.safetensors", "tokenizer.json"},
        config_data={"model_type": "qwen2_vl", "vision_config": {"image_size": 448}},
        processor_data={},
    )
    audio_modalities = _infer_modalities(
        path=tmp_path / "whisper-mini-audio",
        file_names={"config.json", "processor.json"},
        config_data={"model_type": "whisper"},
        processor_data={"processor_class": "WhisperProcessor"},
    )

    assert vision_modalities == (ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL)
    assert _infer_runtime_affinity(format_type=ModelFormat.MLX, modalities=vision_modalities) == (RuntimeAffinity.MLX_VISION,)
    assert audio_modalities == (ModelModality.AUDIO,)
    assert _infer_runtime_affinity(format_type=ModelFormat.AUDIO_FOLDER, modalities=audio_modalities) == (
        RuntimeAffinity.CONVERSION,
        RuntimeAffinity.MLX_AUDIO,
    )


def test_discovery_treats_quantized_sharded_outputs_as_huggingface_conversion_sources(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "converted-gemma4"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text(
        (
            '{"model_type":"gemma4","quantization":{"bits":4,"group_size":64},'
            '"text_config":{"hidden_size":2048},"vision_config":{"image_size":448}}'
        ),
        encoding="utf-8",
    )
    (bundle_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "model-00001-of-00002.safetensors").write_bytes(b"weights-1")
    (bundle_dir / "model-00002-of-00002.safetensors").write_bytes(b"weights-2")
    (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "processor_config.json").write_text("{}", encoding="utf-8")

    manifests = discover_models([tmp_path])

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.format_type == ModelFormat.HUGGINGFACE
    assert manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION
    assert manifest.runtime_affinity == (RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_VISION)
    assert manifest.text_only_runtime_affinity == ()
    assert manifest.text_only_runtime_source is None
    assert manifest.text_only_runtime_reason is None


def test_discovery_detects_onnx_genai_bundles_as_windows_native_candidates(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "phi-3-mini-onnx"
    bundle_dir.mkdir()
    (bundle_dir / "genai_config.json").write_text(
        json.dumps({"model": {"type": "phi3"}, "search": {"max_length": 4096}}),
        encoding="utf-8",
    )
    (bundle_dir / "model.onnx").write_bytes(b"onnx")
    (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    manifests = discover_models([tmp_path])

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.format_type == ModelFormat.ONNX_GENAI
    assert manifest.conversion_status == ConversionStatus.RUNNABLE
    assert manifest.runtime_affinity == (RuntimeAffinity.ONNX_GENAI,)
    assert manifest.modality == (ModelModality.TEXT,)


def test_discovery_expands_layered_conversion_bundle_into_text_and_multimodal_artifacts(tmp_path: Path) -> None:
    layered_root = tmp_path / "gemma4-layered"
    multimodal_dir = layered_root / "multimodal"
    text_dir = layered_root / "text"
    multimodal_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    (multimodal_dir / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "quantization": {"bits": 4}, "vision_config": {"image_size": 448}}),
        encoding="utf-8",
    )
    (multimodal_dir / "weights.safetensors").write_bytes(b"weights")
    (multimodal_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (multimodal_dir / "processor_config.json").write_text("{}", encoding="utf-8")
    (text_dir / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "quantization": {"bits": 4}}),
        encoding="utf-8",
    )
    (text_dir / "weights.safetensors").write_bytes(b"weights")
    (text_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (layered_root / LAYERED_CONVERSION_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "artifact_family_id": "cache-key",
                "display_name": "Gemma-4-31B-it",
                "source_path": str(tmp_path / "source-gemma4"),
                "source_format": "huggingface",
                "source_modality": ["text", "vision", "multimodal"],
                "source_runtime_affinity": ["conversion", "mlx_vision"],
                "artifacts": [
                    {
                        "artifact_key": "multimodal",
                        "role": "multimodal_runnable",
                        "display_name": "Gemma-4-31B-it (multimodal mlx)",
                        "relative_path": "multimodal",
                        "format_type": "mlx",
                        "modality": ["text", "vision", "multimodal"],
                        "runtime_affinity": ["mlx_vision"],
                        "processor_path": "multimodal/processor_config.json",
                        "tokenizer_path": "multimodal/tokenizer.json",
                        "quantization": "4bit",
                    },
                    {
                        "artifact_key": "text",
                        "role": "text_runnable",
                        "display_name": "Gemma-4-31B-it (text mlx)",
                        "relative_path": "text",
                        "format_type": "mlx",
                        "modality": ["text"],
                        "runtime_affinity": ["mlx_text"],
                        "derived_from": "multimodal",
                        "tokenizer_path": "text/tokenizer.json",
                        "quantization": "4bit",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    manifests = discover_models([tmp_path])

    assert len(manifests) == 2
    by_role = {manifest.artifact_role: manifest for manifest in manifests}
    assert by_role[ModelArtifactRole.MULTIMODAL_RUNNABLE].runtime_affinity == (RuntimeAffinity.MLX_VISION,)
    assert by_role[ModelArtifactRole.TEXT_RUNNABLE].runtime_affinity == (RuntimeAffinity.MLX_TEXT,)
    assert by_role[ModelArtifactRole.TEXT_RUNNABLE].modality == (ModelModality.TEXT,)
    assert by_role[ModelArtifactRole.MULTIMODAL_RUNNABLE].artifact_family_id == "cache-key"
    assert by_role[ModelArtifactRole.MULTIMODAL_RUNNABLE].model_id == "gemma-4-31b-it_converted"
    assert by_role[ModelArtifactRole.TEXT_RUNNABLE].model_id == "gemma-4-31b-it_converted_text"
    assert [layer.role for layer in by_role[ModelArtifactRole.TEXT_RUNNABLE].artifact_lineage] == [
        ModelArtifactRole.SOURCE_BUNDLE,
        ModelArtifactRole.MULTIMODAL_RUNNABLE,
        ModelArtifactRole.TEXT_RUNNABLE,
    ]
    assert by_role[ModelArtifactRole.TEXT_RUNNABLE].artifact_lineage[-1].derived_from == "multimodal"


def test_discovery_uses_conversion_output_metadata_for_standalone_converted_bundles(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "cache-key"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text('{"model_type":"phi3"}', encoding="utf-8")
    (bundle_dir / "weights.safetensors").write_bytes(b"weights")
    (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (bundle_dir / CONVERSION_OUTPUT_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "source_display_name": "phi-3-mini-hf",
                "source_model_id": "phi-3-mini-hf-a2a814fb3dbd",
                "display_name": "phi-3-mini-hf (converted)",
                "artifact_role": "standalone",
                "artifact_family_id": "cache-key",
            },
        ),
        encoding="utf-8",
    )

    manifests = discover_models([tmp_path])

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.display_name == "phi-3-mini-hf (converted)"
    assert manifest.model_id == "phi-3-mini-hf_converted"
    assert manifest.metadata["converted_output"] is True
    assert manifest.metadata["source_display_name"] == "phi-3-mini-hf"


def test_discovery_disambiguates_duplicate_converted_model_ids(tmp_path: Path) -> None:
    for folder_name in ("cache-a", "cache-b"):
        bundle_dir = tmp_path / folder_name
        bundle_dir.mkdir()
        (bundle_dir / "config.json").write_text('{"model_type":"phi3"}', encoding="utf-8")
        (bundle_dir / "weights.safetensors").write_bytes(folder_name.encode("utf-8"))
        (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (bundle_dir / CONVERSION_OUTPUT_METADATA_FILENAME).write_text(
            json.dumps(
                {
                    "source_display_name": "phi-3-mini-hf",
                    "source_model_id": f"{folder_name}-source",
                    "display_name": "phi-3-mini-hf (converted)",
                    "artifact_role": "standalone",
                    "artifact_family_id": folder_name,
                },
            ),
            encoding="utf-8",
        )

    manifests = discover_models([tmp_path])
    model_ids = sorted(manifest.model_id for manifest in manifests)

    assert len(manifests) == 2
    assert len(set(model_ids)) == 2
    assert all(model_id.startswith("phi-3-mini-hf_converted") for model_id in model_ids)


def test_discovery_detects_hybrid_ssm_architecture_and_frontier_runtime(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "gated-deltanet-mlx"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text(
        '{"model_type":"gateddeltanet","d_state":128,"num_attention_heads":8,"max_position_embeddings":8192}',
        encoding="utf-8",
    )
    (bundle_dir / "weights.safetensors").write_bytes(b"weights")
    (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    manifests = discover_models([tmp_path])

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.architecture_subtype == ArchitectureSubtype.HYBRID_SSM
    assert manifest.runtime_affinity == (RuntimeAffinity.EXPERIMENTAL, RuntimeAffinity.MLX_TEXT)
    assert manifest.metadata["cache_state_handling"] == "hybrid_attention_state"
    assert manifest.metadata["state_size"] == 128


def test_discovery_detects_moe_architecture_and_expert_metadata(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "mixtral-frontier-mlx"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text(
        '{"model_type":"mixtral","num_experts":64,"experts_per_token":8,"max_position_embeddings":32768}',
        encoding="utf-8",
    )
    (bundle_dir / "weights.safetensors").write_bytes(b"weights")
    (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    manifests = discover_models([tmp_path])

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.architecture_subtype == ArchitectureSubtype.MOE
    assert manifest.runtime_affinity == (RuntimeAffinity.EXPERIMENTAL, RuntimeAffinity.MLX_TEXT)
    assert manifest.metadata["expert_count"] == 64
    assert manifest.metadata["active_expert_count"] == 8
    assert manifest.metadata["expert_routing_type"] == "top-8"


def test_discovery_preserves_sidecar_mixed_precision_quantization_profile(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "mixed-profile-model"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text('{"model_type":"gemma4","quantization":{"bits":4}}', encoding="utf-8")
    (bundle_dir / "weights.safetensors").write_bytes(b"weights")
    (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "lewlm.quantization_profile.json").write_text(
        """
        {
          "strategy": "mixed_precision",
          "weight_precision": "int4",
          "compute_precision": "bf16",
          "layer_overrides": [
            {"layer_pattern": "layers.0.attn.q_proj", "weight_precision": "int8", "compute_precision": "bf16"},
            {"layer_pattern": "layers.0.mlp.gate_proj", "weight_precision": "int4", "compute_precision": "fp16"}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    manifest = discover_models([tmp_path])[0]

    assert manifest.quantization == "mixed-2-layer"
    assert manifest.quantization_profile is not None
    assert manifest.quantization_profile.strategy == QuantizationStrategy.MIXED_PRECISION
    assert len(manifest.quantization_profile.layer_overrides) == 2
    assert manifest.quantization_profile.layer_overrides[0].weight_precision == QuantizationPrecision.INT8
