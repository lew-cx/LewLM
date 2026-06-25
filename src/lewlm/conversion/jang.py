"""Helpers for normalizing JANG-packed Hugging Face bundles before export."""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lewlm.core.errors import ConversionError


JANG_CONFIG_FILENAME = "jang_config.json"
SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"
DEFAULT_JANG_GROUP_SIZE = 64


@dataclass(slots=True)
class JangNormalizationResult:
    """Result of materializing a standard HF safetensors view from JANG weights."""

    source_path: Path
    logs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def is_jang_bundle(source_path: Path) -> bool:
    """Return whether a local HF bundle appears to use JANG-packed weights."""

    if not source_path.is_dir():
        return False
    if (source_path / JANG_CONFIG_FILENAME).exists():
        return True
    index = _load_json(source_path / SAFETENSORS_INDEX_FILENAME)
    metadata = index.get("metadata")
    if isinstance(metadata, dict) and str(metadata.get("format") or "").casefold() == "jang":
        return True
    config = _load_json(source_path / "config.json")
    quantization = config.get("quantization")
    if isinstance(quantization, dict):
        quantization_text = " ".join(str(value).casefold() for value in quantization.values())
        return "jang" in quantization_text
    return False


def missing_jang_dependencies() -> list[str]:
    """Return optional Python packages required by the JANG normalization path."""

    return [
        module_name
        for module_name in ("numpy", "safetensors")
        if importlib.util.find_spec(module_name) is None
    ]


def normalize_jang_bundle(source_path: Path, output_path: Path) -> JangNormalizationResult:
    """Dequantize JANG-packed tensor triplets into a standard HF safetensors bundle.

    JANG stores many matrices as `weight` uint32 payloads with sibling `scales` and
    `biases` tensors. llama.cpp's upstream exporter understands Gemma4, but it
    expects regular HF tensor names and therefore rejects the JANG-only siblings.
    This helper writes a temporary normalized bundle with those triplets expanded
    to fp16 `weight` tensors and leaves all non-packed tensors unchanged.
    """

    missing = missing_jang_dependencies()
    if missing:
        raise ConversionError(
            "JANG normalization requires optional conversion dependencies.",
            details={"missing_packages": missing},
        )

    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    if not source_path.is_dir():
        raise ConversionError("JANG normalization requires a local directory source path.")

    output_path.mkdir(parents=True, exist_ok=True)
    config_data = _copy_hf_metadata_files(source_path=source_path, output_path=output_path)
    jang_config = _load_json(source_path / JANG_CONFIG_FILENAME)
    group_size = _jang_group_size(config_data=config_data, jang_config=jang_config)
    source_weight_map = _source_weight_map(source_path)
    if not source_weight_map:
        raise ConversionError(
            "JANG normalization could not find safetensors weights to normalize.",
            details={"source_path": str(source_path)},
        )

    output_weight_map: dict[str, str] = {}
    total_size = 0
    converted_count = 0
    copied_count = 0
    skipped_count = 0
    ordinal = 0
    tensors_by_file = _group_tensors_by_file(source_weight_map)

    for shard_name, tensor_names in tensors_by_file.items():
        shard_path = source_path / shard_name
        if not shard_path.exists():
            raise ConversionError(
                "JANG normalization found a missing safetensors shard.",
                details={"missing_shard": str(shard_path)},
            )
        with safe_open(str(shard_path), framework="numpy") as shard:
            available_names = set(shard.keys())
            for tensor_name in tensor_names:
                if tensor_name.endswith(".scales") or tensor_name.endswith(".biases"):
                    skipped_count += 1
                    continue

                scales_name = _sibling_tensor_name(tensor_name, "scales")
                biases_name = _sibling_tensor_name(tensor_name, "biases")
                if (
                    tensor_name.endswith(".weight")
                    and scales_name in available_names
                    and biases_name in available_names
                ):
                    tensor = _dequantize_jang_tensor(
                        packed=shard.get_tensor(tensor_name),
                        scales=shard.get_tensor(scales_name),
                        biases=shard.get_tensor(biases_name),
                        group_size=group_size,
                        np=np,
                    )
                    converted_count += 1
                else:
                    tensor = shard.get_tensor(tensor_name)
                    copied_count += 1

                ordinal += 1
                output_name = f"model-{ordinal:05d}.safetensors"
                save_file({tensor_name: tensor}, str(output_path / output_name), metadata={"format": "pt"})
                output_weight_map[tensor_name] = output_name
                total_size += int(getattr(tensor, "nbytes", 0))

    index_payload = {
        "metadata": {
            "format": "pt",
            "total_size": total_size,
            "lewlm_normalized_from": "jang",
            "jang_group_size": group_size,
        },
        "weight_map": output_weight_map,
    }
    (output_path / SAFETENSORS_INDEX_FILENAME).write_text(
        json.dumps(index_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return JangNormalizationResult(
        source_path=output_path,
        logs=[
            f"Normalized JANG bundle from {source_path} to {output_path}.",
            f"Dequantized {converted_count} packed tensor(s), copied {copied_count} tensor(s), skipped {skipped_count} JANG sidecar tensor(s).",
        ],
        metadata={
            "converted_tensors": converted_count,
            "copied_tensors": copied_count,
            "skipped_sidecar_tensors": skipped_count,
            "total_size": total_size,
            "group_size": group_size,
        },
    )


def _copy_hf_metadata_files(*, source_path: Path, output_path: Path) -> dict[str, Any]:
    config_data: dict[str, Any] = {}
    for child in source_path.iterdir():
        if child.is_dir():
            continue
        if child.name == "config.json":
            config_data = _load_json(child)
            normalized_config = dict(config_data)
            normalized_config.pop("quantization", None)
            normalized_config.pop("quantization_config", None)
            normalized_config.pop("compressed_tensors_config", None)
            normalized_config["lewlm_normalized_from"] = "jang"
            (output_path / child.name).write_text(
                json.dumps(normalized_config, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            continue
        if child.name in {JANG_CONFIG_FILENAME, SAFETENSORS_INDEX_FILENAME}:
            continue
        if child.suffix.casefold() in {".safetensors", ".bin"}:
            continue
        shutil.copy2(child, output_path / child.name)
    return config_data


def _source_weight_map(source_path: Path) -> dict[str, str]:
    index_payload = _load_json(source_path / SAFETENSORS_INDEX_FILENAME)
    weight_map = index_payload.get("weight_map")
    if isinstance(weight_map, dict):
        return {
            str(tensor_name): str(shard_name)
            for tensor_name, shard_name in weight_map.items()
            if isinstance(tensor_name, str) and isinstance(shard_name, str)
        }

    from safetensors import safe_open

    discovered: dict[str, str] = {}
    for shard_path in sorted(source_path.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="numpy") as shard:
            for tensor_name in shard.keys():
                discovered[str(tensor_name)] = shard_path.name
    return discovered


def _group_tensors_by_file(weight_map: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for tensor_name, shard_name in sorted(weight_map.items(), key=lambda item: (item[1], item[0])):
        grouped.setdefault(shard_name, []).append(tensor_name)
    return grouped


def _sibling_tensor_name(tensor_name: str, suffix: str) -> str:
    if tensor_name.endswith(".weight"):
        return f"{tensor_name.removesuffix('.weight')}.{suffix}"
    return f"{tensor_name}.{suffix}"


def _jang_group_size(*, config_data: dict[str, Any], jang_config: dict[str, Any]) -> int:
    quantization = config_data.get("quantization")
    if isinstance(quantization, dict):
        group_size = _positive_int(quantization.get("group_size"))
        if group_size is not None:
            return group_size
    jang_quantization = jang_config.get("quantization")
    if isinstance(jang_quantization, dict):
        block_size = _positive_int(jang_quantization.get("block_size"))
        if block_size is not None:
            return block_size
    return DEFAULT_JANG_GROUP_SIZE


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _dequantize_jang_tensor(*, packed, scales, biases, group_size: int, np):
    if str(packed.dtype) != "uint32":
        raise ConversionError(
            "JANG packed tensor expected uint32 storage.",
            details={"dtype": str(packed.dtype)},
        )
    if packed.ndim < 2 or scales.ndim < 2 or biases.ndim < 2:
        raise ConversionError(
            "JANG packed tensors must have at least two dimensions.",
            details={
                "packed_shape": tuple(int(dim) for dim in packed.shape),
                "scales_shape": tuple(int(dim) for dim in scales.shape),
                "biases_shape": tuple(int(dim) for dim in biases.shape),
            },
        )
    if scales.shape != biases.shape or scales.shape[:-1] != packed.shape[:-1]:
        raise ConversionError(
            "JANG scale/bias tensor shapes do not match packed tensor rows.",
            details={
                "packed_shape": tuple(int(dim) for dim in packed.shape),
                "scales_shape": tuple(int(dim) for dim in scales.shape),
                "biases_shape": tuple(int(dim) for dim in biases.shape),
            },
        )

    leading_shape = tuple(int(dim) for dim in packed.shape[:-1])
    packed_2d = packed.reshape((-1, int(packed.shape[-1])))
    scales_2d = scales.reshape((-1, int(scales.shape[-1]))).astype(np.float32, copy=False)
    biases_2d = biases.reshape((-1, int(biases.shape[-1]))).astype(np.float32, copy=False)
    output_columns = int(scales_2d.shape[-1]) * group_size
    if output_columns % int(packed_2d.shape[-1]) != 0:
        raise ConversionError(
            "JANG packed tensor shape does not map cleanly to the configured group size.",
            details={
                "packed_columns": int(packed_2d.shape[-1]),
                "scale_groups": int(scales_2d.shape[-1]),
                "group_size": group_size,
            },
        )

    values_per_word = output_columns // int(packed_2d.shape[-1])
    if values_per_word not in {4, 8, 16}:
        raise ConversionError(
            "Unsupported JANG packing width.",
            details={"values_per_uint32": values_per_word},
        )
    bits_per_value = 32 // values_per_word
    mask = (1 << bits_per_value) - 1
    shifts = (np.arange(values_per_word, dtype=np.uint32) * bits_per_value).reshape((1, 1, values_per_word))
    row_count = int(packed_2d.shape[0])
    dequantized = np.empty((row_count, output_columns), dtype=np.float16)
    target_chunk_values = 32_000_000
    rows_per_chunk = max(1, min(row_count, target_chunk_values // max(1, output_columns)))
    for row_start in range(0, row_count, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, row_count)
        packed_chunk = packed_2d[row_start:row_end]
        unpacked = ((packed_chunk[:, :, None] >> shifts) & mask).reshape((packed_chunk.shape[0], -1))
        unpacked = unpacked[:, :output_columns].astype(np.float32, copy=False)
        expanded_scales = np.repeat(scales_2d[row_start:row_end], group_size, axis=1)[:, :output_columns]
        expanded_biases = np.repeat(biases_2d[row_start:row_end], group_size, axis=1)[:, :output_columns]
        dequantized[row_start:row_end] = unpacked * expanded_scales + expanded_biases
    return dequantized.reshape((*leading_shape, output_columns))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
