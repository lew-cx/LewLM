"""Filesystem discovery for local model manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from math import ceil
from pathlib import Path
from typing import Any

from lewlm.conversion.models import (
    CONVERSION_OUTPUT_METADATA_FILENAME,
    ConversionOutputMetadata,
    LAYERED_CONVERSION_MANIFEST_FILENAME,
    LayeredConversionManifest,
    QUANTIZATION_PROFILE_METADATA_FILENAME,
)
from lewlm.core.contracts import (
    ArchitectureSubtype,
    ConversionStatus,
    ExternalQuantizerReference,
    LayerQuantizationOverride,
    ModelArtifactLayer,
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
    quantization_profile_label,
)
from lewlm.runtime.experimental import extract_architecture_metadata, infer_architecture_subtype


TOKENIZER_FILENAMES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
PROCESSOR_FILENAMES = ("processor.json", "processor_config.json", "preprocessor_config.json")
ONNX_GENAI_CONFIG_FILENAMES = ("genai_config.json",)
QUANTIZATION_PROFILE_FILENAMES = (QUANTIZATION_PROFILE_METADATA_FILENAME, "quantization_profile.json")
DISTRIBUTED_PIPELINE_FILENAMES = ("distributed_pipeline.json",)
AUDIO_KEYWORDS = {"asr", "audio", "bark", "kokoro", "parler", "speech", "stt", "transcribe", "tts", "wav2vec", "whisper", "xtts"}
VISION_KEYWORDS = {"fuyu", "idefics", "llava", "moondream", "pixtral", "qwen2-vl", "vision", "vl"}
EMBEDDING_KEYWORDS = {"bge", "e5", "embed", "embedding", "gte"}
RERANK_KEYWORDS = {"rerank", "reranker"}
IDENTIFIER_KEYS = (
    "model_type",
    "architectures",
    "processor_class",
    "image_processor_type",
    "video_processor_type",
    "feature_extractor_type",
)
QUANTIZATION_RE = re.compile(r"(q\d(?:_[a-z0-9]+)+|q\d(?:_[a-z0-9]+)?|int4|int8|fp16|bf16)", re.IGNORECASE)
CONTEXT_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "seq_length",
    "context_length",
)


def discover_models(roots: Iterable[Path]) -> list[ModelManifest]:
    """Discover model bundles and GGUF files below the provided roots."""

    manifests: list[ModelManifest] = []
    for root in roots:
        manifests.extend(_discover_root(root))
    manifests = _ensure_unique_model_ids(manifests)
    manifests.sort(key=lambda manifest: (manifest.display_name.casefold(), manifest.model_id))
    return manifests


def _discover_root(root: Path) -> list[ModelManifest]:
    manifests: list[ModelManifest] = []
    for current_root, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [dirname for dirname in dirnames if not dirname.startswith(".")]
        path = Path(current_root)
        file_names = set(filenames)
        if LAYERED_CONVERSION_MANIFEST_FILENAME in file_names:
            layered_payload = _load_json(path / LAYERED_CONVERSION_MANIFEST_FILENAME)
            try:
                layered_manifest = LayeredConversionManifest.model_validate(layered_payload)
            except Exception:
                layered_manifest = None
            if layered_manifest is not None:
                manifests.extend(_build_layered_manifests(path=path, layered_manifest=layered_manifest))
                dirnames[:] = []
                continue
        conversion_output = _load_conversion_output_metadata(path, file_names)
        config_data = _load_json(path / "config.json") if "config.json" in file_names else {}

        bundle_format = _detect_bundle_format(file_names)
        if bundle_format is not None:
            manifests.append(
                _build_directory_manifest(
                    path=path,
                    file_names=file_names,
                    format_type=bundle_format,
                    config_data=config_data,
                    conversion_output=conversion_output,
                ),
            )
            dirnames[:] = []
            continue

        gguf_files = sorted(filename for filename in filenames if filename.lower().endswith(".gguf"))
        if conversion_output is not None and len(gguf_files) == 1:
            manifests.append(_build_converted_gguf_manifest(path / gguf_files[0], conversion_output=conversion_output))
            dirnames[:] = []
            continue
        for filename in gguf_files:
            manifests.append(_build_gguf_manifest(path / filename))
    return manifests


def _detect_bundle_format(file_names: set[str]) -> ModelFormat | None:
    normalized_names = {name.casefold() for name in file_names}
    if {"adapter_config.json"} <= file_names and any(name.startswith("adapter_model") for name in file_names):
        return ModelFormat.ADAPTER_BUNDLE
    if any(name in normalized_names for name in ONNX_GENAI_CONFIG_FILENAMES) or (
        any(name.endswith(".onnx") for name in normalized_names)
        and (
            "config.json" in normalized_names
            or any(name in normalized_names for name in TOKENIZER_FILENAMES)
            or any(name in normalized_names for name in PROCESSOR_FILENAMES)
        )
    ):
        return ModelFormat.ONNX_GENAI
    if "config.json" in file_names and any(name in {"weights.safetensors", "weights.npz"} for name in file_names):
        return ModelFormat.MLX
    if "config.json" in file_names and (
        any(name.endswith(".safetensors") for name in file_names)
        or any(name.endswith(".bin") for name in file_names)
    ) and (
        any(name in TOKENIZER_FILENAMES for name in file_names)
        or any(name in PROCESSOR_FILENAMES for name in file_names)
        or "generation_config.json" in file_names
    ):
        return ModelFormat.HUGGINGFACE
    if "config.json" in file_names and any(name in PROCESSOR_FILENAMES for name in file_names):
        return ModelFormat.AUDIO_FOLDER
    return None


def _build_gguf_manifest(path: Path) -> ModelManifest:
    display_name = path.stem
    fingerprint = _fingerprint_path(path)
    quantization = _infer_quantization(display_name)
    architecture_family = _infer_architecture_from_name(display_name)
    architecture_subtype = infer_architecture_subtype(name=display_name, config_data={})
    estimated_memory_mb = ceil(path.stat().st_size / (1024 * 1024))
    validation = ModelValidationResult(
        status=ValidationState.VALID,
        message="GGUF model file discovered and ready for llama.cpp compatibility checks.",
        details={"source_kind": "file"},
    )
    return ModelManifest(
        model_id=_build_model_id(display_name, fingerprint),
        display_name=display_name,
        architecture_family=architecture_family,
        architecture_subtype=architecture_subtype,
        modality=(ModelModality.TEXT,),
        source_path=str(path),
        format_type=ModelFormat.GGUF,
        quantization=quantization,
        runtime_affinity=_frontier_runtime_affinity(
            base_affinity=(RuntimeAffinity.LLAMACPP,),
            architecture_subtype=architecture_subtype,
            modalities=(ModelModality.TEXT,),
        ),
        estimated_memory_mb=estimated_memory_mb,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint=fingerprint,
        last_validation_result=validation,
        metadata={
            "source_kind": "file",
            "size_bytes": path.stat().st_size,
            **extract_architecture_metadata(
                name=display_name,
                config_data={},
                architecture_subtype=architecture_subtype,
            ),
        },
    )


def _build_converted_gguf_manifest(path: Path, *, conversion_output: ConversionOutputMetadata) -> ModelManifest:
    base_manifest = _build_gguf_manifest(path)
    return base_manifest.model_copy(
        update={
            "model_id": _build_converted_model_id(
                conversion_output.source_display_name,
                artifact_role=conversion_output.artifact_role,
            ),
            "display_name": conversion_output.display_name,
            "artifact_role": conversion_output.artifact_role,
            "artifact_family_id": conversion_output.artifact_family_id,
            "metadata": {
                **base_manifest.metadata,
                "converted_output": True,
                "source_display_name": conversion_output.source_display_name,
                "source_model_id": conversion_output.source_model_id,
                **conversion_output.metadata,
            },
        },
    )


def _build_directory_manifest(
    path: Path,
    file_names: set[str],
    format_type: ModelFormat,
    *,
    config_data: dict[str, Any] | None = None,
    conversion_output: ConversionOutputMetadata | None = None,
) -> ModelManifest:
    config_data = config_data or _load_json(path / "config.json")
    display_name = conversion_output.display_name if conversion_output is not None else path.name
    fingerprint = _fingerprint_path(path)
    tokenizer_path = _resolve_optional_child(path, TOKENIZER_FILENAMES)
    processor_path = _resolve_optional_child(path, PROCESSOR_FILENAMES)
    processor_data = _load_json(processor_path) if processor_path is not None else {}
    distributed_pipeline_path = _resolve_optional_child(path, DISTRIBUTED_PIPELINE_FILENAMES)
    distributed_pipeline = _load_json(distributed_pipeline_path) if distributed_pipeline_path is not None else {}
    quantization_profile = _resolve_quantization_profile(path=path, config_data=config_data)
    architecture_family = _infer_architecture(path=path, config_data=config_data)
    architecture_subtype = infer_architecture_subtype(name=display_name, config_data=config_data)
    modalities = _infer_modalities(
        path=path,
        file_names=file_names,
        config_data=config_data,
        processor_data=processor_data,
    )
    runtime_affinity = _frontier_runtime_affinity(
        base_affinity=_infer_runtime_affinity(format_type=format_type, modalities=modalities),
        architecture_subtype=architecture_subtype,
        modalities=modalities,
    )
    if distributed_pipeline:
        distributed_only = bool(distributed_pipeline.get("distributed_only", False))
        runtime_affinity = (
            (RuntimeAffinity.DISTRIBUTED_EXPERIMENTAL,)
            if distributed_only
            else (RuntimeAffinity.DISTRIBUTED_EXPERIMENTAL, *runtime_affinity)
        )
    validation = ModelValidationResult(
        status=ValidationState.VALID,
        message=(
            "Model bundle discovered with experimental distributed pipeline metadata."
            if distributed_pipeline
            else "Model bundle discovered. Runtime compatibility remains subject to backend-specific validation."
        ),
        details={"source_kind": "directory", "file_count": len(file_names)},
    )
    size_bytes = _estimate_size_bytes(path)
    text_only_runtime_affinity, text_only_runtime_source, text_only_runtime_reason = _infer_text_only_runtime_profile(
        format_type=format_type,
        modalities=modalities,
        config_data=config_data,
        tokenizer_path=tokenizer_path,
    )
    model_id = (
        _build_converted_model_id(
            conversion_output.source_display_name,
            artifact_role=conversion_output.artifact_role,
        )
        if conversion_output is not None
        else _build_model_id(display_name, fingerprint)
    )
    return ModelManifest(
        model_id=model_id,
        display_name=display_name,
        architecture_family=architecture_family,
        architecture_subtype=architecture_subtype,
        modality=modalities,
        source_path=str(path),
        format_type=format_type,
        quantization=quantization_profile_label(quantization_profile) or _infer_quantization(display_name, config_data),
        quantization_profile=quantization_profile,
        tokenizer_path=str(tokenizer_path) if tokenizer_path is not None else None,
        processor_path=str(processor_path) if processor_path is not None else None,
        runtime_affinity=runtime_affinity,
        text_only_runtime_affinity=text_only_runtime_affinity,
        text_only_runtime_source=text_only_runtime_source,
        text_only_runtime_reason=text_only_runtime_reason,
        required_extra_files=_required_extra_files(format_type),
        estimated_memory_mb=ceil(size_bytes / (1024 * 1024)) if size_bytes else None,
        context_length=_extract_context_length(config_data),
        conversion_status=_infer_conversion_status(format_type),
        fingerprint=fingerprint,
        last_validation_result=validation,
        metadata={
            "source_kind": "directory",
            "file_count": len(file_names),
            "size_bytes": size_bytes,
            "config_present": bool(config_data),
            **(
                {
                    "converted_output": True,
                    "source_display_name": conversion_output.source_display_name,
                    "source_model_id": conversion_output.source_model_id,
                }
                if conversion_output is not None
                else {}
            ),
            **extract_architecture_metadata(
                name=display_name,
                config_data=config_data,
                architecture_subtype=architecture_subtype,
            ),
            **({"distributed_pipeline": distributed_pipeline} if distributed_pipeline else {}),
        },
    )


def _infer_text_only_runtime_profile(
    *,
    format_type: ModelFormat,
    modalities: tuple[ModelModality, ...],
    config_data: dict[str, Any],
    tokenizer_path: Path | None,
) -> tuple[tuple[RuntimeAffinity, ...], str | None, str | None]:
    if (
        format_type != ModelFormat.MLX
        or tokenizer_path is None
        or ModelModality.TEXT not in modalities
        or ModelModality.VISION not in modalities
        or ModelModality.MULTIMODAL not in modalities
    ):
        return (), None, None
    text_config = config_data.get("text_config") or config_data.get("llm_config")
    if not isinstance(text_config, dict):
        return (), None, None
    model_type = str(config_data.get("model_type") or "").casefold().replace("-", "").replace("_", "")
    architectures = {
        str(item).casefold().replace("-", "").replace("_", "")
        for item in config_data.get("architectures", [])
        if isinstance(item, str)
    }
    if model_type == "gemma4" or "gemma4forconditionalgeneration" in architectures:
        return (
            (RuntimeAffinity.MLX_TEXT,),
            "same_bundle",
            (
                "Gemma 4 MLX multimodal bundles expose nested text configuration and a tokenizer, "
                "so LewLM can prefer the MLX text runtime for text-only prompts."
            ),
        )
    return (), None, None


def _build_layered_manifests(*, path: Path, layered_manifest: LayeredConversionManifest) -> list[ModelManifest]:
    lineage = _layered_lineage(path=path, layered_manifest=layered_manifest)
    manifests: list[ModelManifest] = []
    for artifact in layered_manifest.artifacts:
        artifact_path = path if artifact.relative_path in {"", ".", "./"} else path / artifact.relative_path
        if not artifact_path.exists():
            continue
        file_names = {child.name for child in artifact_path.iterdir()} if artifact_path.is_dir() else {artifact_path.name}
        config_data = _load_json(artifact_path / "config.json") if artifact_path.is_dir() and "config.json" in file_names else {}
        if artifact.format_type == ModelFormat.GGUF and artifact_path.is_file():
            base_manifest = _build_gguf_manifest(artifact_path)
        elif artifact_path.is_dir():
            base_manifest = _build_directory_manifest(
                path=artifact_path,
                file_names=file_names,
                format_type=artifact.format_type,
                config_data=config_data,
            )
        else:
            continue
        display_name = artifact.display_name
        manifests.append(
            base_manifest.model_copy(
                update={
                    "model_id": _build_converted_model_id(
                        layered_manifest.display_name,
                        artifact_role=artifact.role,
                    ),
                    "display_name": display_name,
                    "modality": artifact.modality,
                    "runtime_affinity": artifact.runtime_affinity,
                    "tokenizer_path": (
                        str(path / artifact.tokenizer_path)
                        if artifact.tokenizer_path is not None and not Path(artifact.tokenizer_path).is_absolute()
                        else artifact.tokenizer_path
                    )
                    or base_manifest.tokenizer_path,
                    "processor_path": (
                        str(path / artifact.processor_path)
                        if artifact.processor_path is not None and not Path(artifact.processor_path).is_absolute()
                        else artifact.processor_path
                    )
                    or base_manifest.processor_path,
                    "quantization": artifact.quantization or base_manifest.quantization,
                    "quantization_profile": artifact.quantization_profile or base_manifest.quantization_profile,
                    "artifact_key": artifact.artifact_key,
                    "artifact_role": artifact.role,
                    "artifact_family_id": layered_manifest.artifact_family_id,
                    "artifact_lineage": lineage,
                    "metadata": {
                        **base_manifest.metadata,
                        "artifact_family_id": layered_manifest.artifact_family_id,
                        "artifact_role": artifact.role.value,
                        "artifact_key": artifact.artifact_key,
                        "artifact_relative_path": artifact.relative_path,
                    },
                },
            ),
        )
    return manifests


def _infer_modalities(
    *,
    path: Path,
    file_names: set[str],
    config_data: dict[str, Any],
    processor_data: dict[str, Any],
) -> tuple[ModelModality, ...]:
    searchable_text = " ".join(_model_identifiers(path=path, file_names=file_names, config_data=config_data, processor_data=processor_data))
    modalities: list[ModelModality] = []
    if any(keyword in searchable_text for keyword in EMBEDDING_KEYWORDS):
        modalities.append(ModelModality.EMBEDDING)
    if any(keyword in searchable_text for keyword in RERANK_KEYWORDS):
        modalities.append(ModelModality.RERANK)
    if any(keyword in searchable_text for keyword in AUDIO_KEYWORDS):
        modalities.append(ModelModality.AUDIO)
    if "vision_config" in config_data or any(keyword in searchable_text for keyword in VISION_KEYWORDS):
        modalities.extend((ModelModality.TEXT, ModelModality.VISION))
    if not modalities:
        modalities.append(ModelModality.TEXT)
    unique_modalities = list(dict.fromkeys(modalities))
    if ModelModality.TEXT in unique_modalities and len(unique_modalities) > 1:
        unique_modalities.append(ModelModality.MULTIMODAL)
    return tuple(unique_modalities)


def _infer_runtime_affinity(
    *,
    format_type: ModelFormat,
    modalities: tuple[ModelModality, ...],
) -> tuple[RuntimeAffinity, ...]:
    if format_type == ModelFormat.GGUF:
        return (RuntimeAffinity.LLAMACPP,)
    if format_type == ModelFormat.ONNX_GENAI:
        return (RuntimeAffinity.ONNX_GENAI,)
    if format_type == ModelFormat.MLX:
        if ModelModality.VISION in modalities:
            return (RuntimeAffinity.MLX_VISION,)
        if _is_audio_only(modalities):
            return (RuntimeAffinity.MLX_AUDIO,)
        return (RuntimeAffinity.MLX_TEXT,)
    if format_type == ModelFormat.AUDIO_FOLDER:
        return (RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_AUDIO)
    if format_type == ModelFormat.HUGGINGFACE:
        if ModelModality.VISION in modalities:
            return (RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_VISION)
        if _is_audio_only(modalities):
            return (RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_AUDIO)
        return (RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_TEXT)
    if format_type == ModelFormat.ADAPTER_BUNDLE:
        return (RuntimeAffinity.CONVERSION,)
    return (RuntimeAffinity.EXPERIMENTAL,)


def _layered_lineage(*, path: Path, layered_manifest: LayeredConversionManifest) -> list[ModelArtifactLayer]:
    layers = [
        ModelArtifactLayer(
            artifact_key="source_bundle",
            role=ModelArtifactRole.SOURCE_BUNDLE,
            display_name=layered_manifest.display_name,
            format_type=layered_manifest.source_format,
            source_path=layered_manifest.source_path,
            modality=layered_manifest.source_modality,
            runtime_affinity=layered_manifest.source_runtime_affinity,
            tokenizer_path=layered_manifest.source_tokenizer_path,
            processor_path=layered_manifest.source_processor_path,
            quantization=layered_manifest.source_quantization,
            quantization_profile=layered_manifest.source_quantization_profile,
        ),
    ]
    for artifact in layered_manifest.artifacts:
        artifact_path = path if artifact.relative_path in {"", ".", "./"} else path / artifact.relative_path
        tokenizer_path = (
            path / artifact.tokenizer_path
            if artifact.tokenizer_path is not None and not Path(artifact.tokenizer_path).is_absolute()
            else Path(artifact.tokenizer_path)
            if artifact.tokenizer_path is not None
            else _resolve_optional_child(artifact_path, TOKENIZER_FILENAMES)
        )
        processor_path = (
            path / artifact.processor_path
            if artifact.processor_path is not None and not Path(artifact.processor_path).is_absolute()
            else Path(artifact.processor_path)
            if artifact.processor_path is not None
            else _resolve_optional_child(artifact_path, PROCESSOR_FILENAMES)
        )
        layers.append(
            ModelArtifactLayer(
                artifact_key=artifact.artifact_key,
                role=artifact.role,
                display_name=artifact.display_name,
                format_type=artifact.format_type,
                source_path=str(artifact_path),
                modality=artifact.modality,
                runtime_affinity=artifact.runtime_affinity,
                tokenizer_path=str(tokenizer_path) if tokenizer_path is not None else None,
                processor_path=str(processor_path) if processor_path is not None else None,
                quantization=artifact.quantization,
                quantization_profile=artifact.quantization_profile,
                derived_from=artifact.derived_from or "source_bundle",
                metadata=artifact.metadata,
            ),
        )
    return layers


def _infer_conversion_status(format_type: ModelFormat) -> ConversionStatus:
    if format_type in {ModelFormat.GGUF, ModelFormat.MLX, ModelFormat.ONNX_GENAI, ModelFormat.AUDIO_FOLDER}:
        return ConversionStatus.RUNNABLE
    if format_type in {ModelFormat.HUGGINGFACE, ModelFormat.ADAPTER_BUNDLE}:
        return ConversionStatus.REQUIRES_CONVERSION
    return ConversionStatus.UNKNOWN


def _required_extra_files(format_type: ModelFormat) -> list[str]:
    if format_type == ModelFormat.ADAPTER_BUNDLE:
        return ["base_model_reference"]
    return []


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_optional_child(path: Path, candidates: Iterable[str]) -> Path | None:
    for name in candidates:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def _extract_context_length(config_data: dict[str, Any]) -> int | None:
    values = list(_context_lengths(config_data))
    if not values:
        return None
    return max(values)


def _infer_quantization(name: str, config_data: dict[str, Any] | None = None) -> str | None:
    match = QUANTIZATION_RE.search(name)
    if match is not None:
        return match.group(1).lower()
    if config_data:
        profile = _quantization_profile_from_config(config_data)
        if profile is not None:
            return quantization_profile_label(profile)
        quantization = config_data.get("quantization")
        if isinstance(quantization, str):
            return quantization
        if isinstance(quantization, dict):
            bits = quantization.get("bits")
            if isinstance(bits, int):
                return f"int{bits}"
    return None


def _resolve_quantization_profile(*, path: Path, config_data: dict[str, Any]) -> QuantizationProfile | None:
    for filename in QUANTIZATION_PROFILE_FILENAMES:
        profile_payload = _load_json(path / filename)
        if profile_payload:
            try:
                return QuantizationProfile.model_validate(profile_payload)
            except Exception:
                continue
    return _quantization_profile_from_config(config_data)


def _quantization_profile_from_config(config_data: dict[str, Any]) -> QuantizationProfile | None:
    quantization = config_data.get("quantization")
    if isinstance(quantization, str):
        precision = _precision_from_value(quantization)
        return QuantizationProfile(
            name=quantization,
            strategy=QuantizationStrategy.WEIGHT_ONLY,
            weight_precision=precision,
        )
    if not isinstance(quantization, dict):
        return None
    weight_precision = _precision_from_value(
        quantization.get("bits") or quantization.get("weight_bits") or quantization.get("precision"),
    )
    activation_precision = _precision_from_value(
        quantization.get("activation_bits") or quantization.get("activation_dtype"),
    )
    compute_precision = _precision_from_value(
        quantization.get("compute_dtype") or quantization.get("dtype"),
    )
    kv_cache_precision = _precision_from_value(quantization.get("kv_cache_bits") or quantization.get("kv_cache_dtype"))
    layer_overrides = [
        override
        for layer_name, value in quantization.items()
        if (override := _layer_override_from_spec(layer_name, value)) is not None
    ]
    external_quantizer = _external_quantizer_from_config(quantization.get("external_quantizer"))
    strategy = QuantizationStrategy.WEIGHT_ONLY
    if layer_overrides:
        strategy = QuantizationStrategy.MIXED_PRECISION
    elif any(precision in {QuantizationPrecision.FP8_E4M3, QuantizationPrecision.FP8_E5M2} for precision in (weight_precision, compute_precision)):
        strategy = QuantizationStrategy.HYBRID_FP8
    elif activation_precision is not None:
        strategy = QuantizationStrategy.ACTIVATION_AWARE
    elif external_quantizer is not None:
        strategy = QuantizationStrategy.EXTERNAL_ADAPTIVE
    metadata = {
        key: value
        for key, value in quantization.items()
        if key
        not in {
            "bits",
            "weight_bits",
            "precision",
            "group_size",
            "name",
            "activation_bits",
            "activation_dtype",
            "compute_dtype",
            "dtype",
            "kv_cache_bits",
            "kv_cache_dtype",
            "external_quantizer",
        }
        and not (isinstance(value, dict) and ("bits" in value or "weight_bits" in value or "precision" in value))
    }
    return QuantizationProfile(
        name=str(quantization.get("name")) if isinstance(quantization.get("name"), str) else None,
        strategy=strategy,
        weight_precision=weight_precision,
        activation_precision=activation_precision,
        kv_cache_precision=kv_cache_precision,
        compute_precision=compute_precision,
        calibration_samples=_coerce_int(quantization.get("calibration_samples")),
        group_size=_coerce_int(quantization.get("group_size")),
        layer_overrides=layer_overrides,
        external_quantizer=external_quantizer,
        metadata=metadata,
    )


def _layer_override_from_spec(layer_name: str, value: Any) -> LayerQuantizationOverride | None:
    if not isinstance(value, dict):
        return None
    if not any(key in value for key in ("bits", "weight_bits", "precision", "activation_bits", "activation_dtype", "compute_dtype", "dtype")):
        return None
    return LayerQuantizationOverride(
        layer_pattern=layer_name,
        weight_precision=_precision_from_value(value.get("bits") or value.get("weight_bits") or value.get("precision")),
        activation_precision=_precision_from_value(value.get("activation_bits") or value.get("activation_dtype")),
        compute_precision=_precision_from_value(value.get("compute_dtype") or value.get("dtype")),
    )


def _external_quantizer_from_config(value: Any) -> ExternalQuantizerReference | None:
    if isinstance(value, str):
        return ExternalQuantizerReference(name=value)
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    if not isinstance(name, str) or not name:
        return None
    required_packages = value.get("required_packages")
    return ExternalQuantizerReference(
        name=name,
        profile=value.get("profile") if isinstance(value.get("profile"), str) else None,
        module=value.get("module") if isinstance(value.get("module"), str) else None,
        required_packages=[str(item) for item in required_packages] if isinstance(required_packages, list) else [],
    )


def _precision_from_value(value: Any) -> QuantizationPrecision | None:
    if isinstance(value, int):
        try:
            return QuantizationPrecision(f"int{value}")
        except ValueError:
            return None
    if not isinstance(value, str):
        return None
    normalized = value.casefold().replace("-", "_")
    aliases = {
        "fp8": QuantizationPrecision.FP8_E4M3,
        "e4m3": QuantizationPrecision.FP8_E4M3,
        "e5m2": QuantizationPrecision.FP8_E5M2,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return QuantizationPrecision(normalized)
    except ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _infer_architecture(path: Path, config_data: dict[str, Any]) -> str:
    model_type = config_data.get("model_type")
    if isinstance(model_type, str) and model_type:
        return model_type
    architectures = config_data.get("architectures")
    if isinstance(architectures, list) and architectures:
        first_arch = architectures[0]
        if isinstance(first_arch, str):
            return first_arch.removesuffix("ForCausalLM").removesuffix("Model")
    return _infer_architecture_from_name(path.name)


def _infer_architecture_from_name(name: str) -> str:
    normalized = re.sub(r"[-_]+", " ", name.casefold())
    for token in ("qwen", "llama", "mistral", "phi", "gemma", "deepseek", "mixtral", "whisper", "mamba", "jamba"):
        if token in normalized:
            return token
    if "gateddeltanet" in normalized or "gated delta net" in normalized:
        return "gateddeltanet"
    return "unknown"


def _frontier_runtime_affinity(
    *,
    base_affinity: tuple[RuntimeAffinity, ...],
    architecture_subtype: ArchitectureSubtype,
    modalities: tuple[ModelModality, ...],
) -> tuple[RuntimeAffinity, ...]:
    if (
        architecture_subtype.value not in {"ssm_mamba", "hybrid_ssm", "moe", "hybrid_moe"}
        or set(modalities) != {ModelModality.TEXT}
        or RuntimeAffinity.EXPERIMENTAL in base_affinity
    ):
        return base_affinity
    return (RuntimeAffinity.EXPERIMENTAL, *base_affinity)


def _estimate_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _fingerprint_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        return digest.hexdigest()

    for child in sorted((candidate for candidate in path.rglob("*") if candidate.is_file()), key=lambda item: item.as_posix()):
        stat = child.stat()
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _load_conversion_output_metadata(path: Path, file_names: set[str]) -> ConversionOutputMetadata | None:
    if CONVERSION_OUTPUT_METADATA_FILENAME not in file_names:
        return None
    payload = _load_json(path / CONVERSION_OUTPUT_METADATA_FILENAME)
    try:
        return ConversionOutputMetadata.model_validate(payload)
    except Exception:
        return None


def _build_model_id(display_name: str, fingerprint: str) -> str:
    slug = _slugify(display_name)
    return f"{slug or 'model'}-{fingerprint[:12]}"


def _build_converted_model_id(display_name: str, *, artifact_role: ModelArtifactRole) -> str:
    slug = _slugify(display_name) or "model"
    suffix = "_converted_text" if artifact_role == ModelArtifactRole.TEXT_RUNNABLE else "_converted"
    return f"{slug}{suffix}"


def _ensure_unique_model_ids(manifests: list[ModelManifest]) -> list[ModelManifest]:
    by_model_id: dict[str, list[ModelManifest]] = {}
    for manifest in manifests:
        by_model_id.setdefault(manifest.model_id, []).append(manifest)
    return [
        manifest
        if len(by_model_id[manifest.model_id]) == 1
        else manifest.model_copy(update={"model_id": f"{manifest.model_id}_{manifest.fingerprint[:6]}"})
        for manifest in manifests
    ]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _model_identifiers(
    *,
    path: Path,
    file_names: set[str],
    config_data: dict[str, Any],
    processor_data: dict[str, Any],
) -> list[str]:
    identifiers = [path.name.casefold(), *(name.casefold() for name in file_names)]
    _extend_identifiers(identifiers, config_data)
    _extend_identifiers(identifiers, processor_data)
    return identifiers


def _extend_identifiers(identifiers: list[str], payload: dict[str, Any]) -> None:
    for key in IDENTIFIER_KEYS:
        _append_identifier_value(identifiers, payload.get(key))
    for nested_key in ("text_config", "vision_config", "audio_config"):
        nested_payload = payload.get(nested_key)
        if isinstance(nested_payload, dict):
            _extend_identifiers(identifiers, nested_payload)


def _append_identifier_value(identifiers: list[str], value: Any) -> None:
    if isinstance(value, str):
        identifiers.append(value.casefold())
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                identifiers.append(item.casefold())


def _context_lengths(payload: Any) -> Iterable[int]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in CONTEXT_KEYS and isinstance(value, int):
                yield value
            else:
                yield from _context_lengths(value)
        return
    if isinstance(payload, list):
        for item in payload:
            yield from _context_lengths(item)


def _is_audio_only(modalities: tuple[ModelModality, ...]) -> bool:
    return ModelModality.AUDIO in modalities and not any(
        modality in modalities
        for modality in (
            ModelModality.TEXT,
            ModelModality.VISION,
            ModelModality.MULTIMODAL,
            ModelModality.EMBEDDING,
            ModelModality.RERANK,
        )
    )
