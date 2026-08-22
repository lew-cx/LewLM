"""Normalize Hugging Face bundles whose tensors do not match their declared architecture.

Sentence-transformers publishes embedding and reranker models by saving the
*backbone* it wrapped rather than the full causal-LM head, while leaving
`config.json` advertising the original `*ForCausalLM` architecture. The result is
a bundle that every downstream exporter — llama.cpp, mlx-lm, onnxruntime-genai —
rejects, because the tensors are named `layers.0...` instead of
`model.layers.0...`, and because `tie_word_embeddings: false` promises an
`lm_head.weight` that the export dropped.

The helpers here rewrite such a bundle into the canonical layout the exporters
expect. Renaming a tensor never moves its bytes, so the safetensors payload is
copied verbatim and only the JSON header is rebuilt. That keeps normalization
dependency-free (stdlib only) and bounded in memory regardless of model size.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lewlm.core.errors import ConversionError


SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"
BACKBONE_TENSOR_PREFIX = "model."
_COPY_BUFFER_BYTES = 8 * 1024 * 1024

# Top-level tensor names a decoder backbone exposes once its `model.` prefix has
# been stripped. Seeing these without the prefix is what identifies the bundle.
_BACKBONE_ROOT_NAMES = frozenset({"embed_tokens", "layers", "norm", "rotary_emb"})
_WEIGHT_SUFFIXES = (".safetensors",)
# Architectures whose weights are a decoder stack. mlx-lm and the onnxruntime-genai
# builder both construct decoders, so a semantic model wrapping one is convertible
# on those paths even though it is never used to generate text.
_DECODER_ARCHITECTURE_MARKERS = ("forcausallm", "forconditionalgeneration", "lmheadmodel")


@dataclass(slots=True)
class CheckpointLayout:
    """What a bundle's weights actually look like on disk."""

    tensor_names: tuple[str, ...] = ()
    shard_names: tuple[str, ...] = ()
    has_index: bool = False
    backbone_prefix_missing: bool = False
    has_lm_head: bool = False
    declares_tied_embeddings: bool | None = None
    architectures: tuple[str, ...] = ()

    @property
    def needs_tensor_prefix(self) -> bool:
        return self.backbone_prefix_missing

    @property
    def needs_tie_reconciliation(self) -> bool:
        """True when config promises an untied `lm_head` the checkpoint does not carry."""

        return self.declares_tied_embeddings is False and not self.has_lm_head

    @property
    def needs_normalization(self) -> bool:
        return self.needs_tensor_prefix or self.needs_tie_reconciliation

    @property
    def is_decoder(self) -> bool:
        """Whether the weights form a decoder stack rather than an encoder.

        Encoder embedding models (BERT, XLM-R and friends) name their weights
        `encoder.layer.*`, carry no decoder backbone, and declare a plain
        `*Model` architecture, so they stay off the causal-LM exporters.
        """

        if any(
            marker in architecture.casefold()
            for architecture in self.architectures
            for marker in _DECODER_ARCHITECTURE_MARKERS
        ):
            return True
        roots = {name.split(".", 1)[0] for name in self.tensor_names}
        if "model" in roots:
            roots |= {name.split(".")[1] for name in self.tensor_names if name.startswith(BACKBONE_TENSOR_PREFIX)}
        return "layers" in roots and "embed_tokens" in roots


@dataclass(slots=True)
class CheckpointNormalizationResult:
    """Result of materializing a canonical causal-LM view of a bundle."""

    source_path: Path
    normalized: bool = False
    logs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def inspect_checkpoint_layout(source_path: Path) -> CheckpointLayout:
    """Describe a local HF bundle's tensor layout without loading any weights."""

    if not source_path.is_dir():
        return CheckpointLayout()
    config_data = _load_json(source_path / "config.json")
    architectures = tuple(
        str(entry) for entry in config_data.get("architectures", []) if isinstance(entry, str)
    )
    declared_tie = config_data.get("tie_word_embeddings")
    tensor_names, shard_names, has_index = _read_tensor_inventory(source_path)
    if not tensor_names:
        return CheckpointLayout(
            architectures=architectures,
            declares_tied_embeddings=declared_tie if isinstance(declared_tie, bool) else None,
        )
    roots = {name.split(".", 1)[0] for name in tensor_names}
    prefix_missing = (
        not any(name.startswith(BACKBONE_TENSOR_PREFIX) for name in tensor_names)
        and bool(roots & _BACKBONE_ROOT_NAMES)
    )
    return CheckpointLayout(
        tensor_names=tensor_names,
        shard_names=shard_names,
        has_index=has_index,
        backbone_prefix_missing=prefix_missing,
        has_lm_head=any(name.split(".", 1)[0] == "lm_head" for name in tensor_names),
        declares_tied_embeddings=declared_tie if isinstance(declared_tie, bool) else None,
        architectures=architectures,
    )


def is_decoder_bundle(source_path: Path) -> bool:
    """Return whether a bundle's weights form a decoder stack."""

    return inspect_checkpoint_layout(source_path).is_decoder


def is_backbone_checkpoint(source_path: Path) -> bool:
    """Return whether a bundle needs normalization before an exporter can read it."""

    return inspect_checkpoint_layout(source_path).needs_normalization


def checkpoint_normalization_warnings(layout: CheckpointLayout) -> list[str]:
    """Human-readable notes describing what normalization will change."""

    warnings: list[str] = []
    if layout.needs_tensor_prefix:
        warnings.append(
            "Source bundle stores a bare transformer backbone (sentence-transformers layout); LewLM will "
            f"re-prefix its tensors with `{BACKBONE_TENSOR_PREFIX}` before export."
        )
    if layout.needs_tie_reconciliation:
        warnings.append(
            "Source config declares `tie_word_embeddings: false` but ships no `lm_head` tensor; LewLM will tie "
            "the output projection to the input embeddings so the exporter can build a complete model."
        )
    return warnings


def normalize_checkpoint_bundle(source_path: Path, output_path: Path) -> CheckpointNormalizationResult:
    """Write a canonical causal-LM view of `source_path` into `output_path`.

    Shards whose tensor names already match the canonical layout are hard-linked
    rather than copied, so a bundle that only needs its `config.json` corrected
    costs no extra disk.
    """

    if not source_path.is_dir():
        raise ConversionError("Checkpoint normalization requires a local directory source path.")
    source_root = source_path.resolve()
    output_root = output_path.resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ConversionError(
            "Checkpoint normalization output must be outside the source bundle.",
            details={"source_path": str(source_path), "output_path": str(output_path)},
        )
    layout = inspect_checkpoint_layout(source_path)
    if not layout.needs_normalization:
        return CheckpointNormalizationResult(source_path=source_path, normalized=False)

    output_path.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    rename: Callable[[str], str] = _prefix_backbone_tensor if layout.needs_tensor_prefix else (lambda name: name)

    weight_files = _weight_file_names(source_path, layout)
    for shard_name in weight_files:
        source_shard = source_path / shard_name
        target_shard = output_path / shard_name
        target_shard.parent.mkdir(parents=True, exist_ok=True)
        if layout.needs_tensor_prefix:
            _rewrite_safetensors_names(source_shard, target_shard, rename)
        else:
            _link_or_copy(source_shard, target_shard)
    if weight_files:
        logs.append(
            f"Normalized {len(weight_files)} safetensors shard(s) into a canonical causal-LM tensor layout."
        )

    _copy_support_files(source_path=source_path, output_path=output_path, weight_files=weight_files)
    if layout.has_index:
        _rewrite_index(source_path=source_path, output_path=output_path, rename=rename)
    config_changes = _rewrite_config(source_path=source_path, output_path=output_path, layout=layout)
    logs.extend(config_changes)

    metadata: dict[str, Any] = {"source_preprocessing": "backbone_normalization"}
    if layout.needs_tensor_prefix:
        metadata["tensor_prefix_applied"] = BACKBONE_TENSOR_PREFIX
    if layout.needs_tie_reconciliation:
        metadata["tie_word_embeddings_forced"] = True
    if layout.architectures:
        metadata["source_architectures"] = list(layout.architectures)
    return CheckpointNormalizationResult(
        source_path=output_path,
        normalized=True,
        logs=logs,
        metadata=metadata,
    )


def _read_tensor_inventory(source_path: Path) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Collect tensor and shard names from the index when present, else the lone shard."""

    index = _load_json(source_path / SAFETENSORS_INDEX_FILENAME)
    weight_map = index.get("weight_map")
    if isinstance(weight_map, dict) and weight_map:
        names = tuple(str(name) for name in weight_map)
        shards = _validated_index_shard_names(source_path, weight_map)
        return names, shards, True
    shards = tuple(
        sorted(entry.name for entry in source_path.iterdir() if entry.name.endswith(_WEIGHT_SUFFIXES))
    )
    names: list[str] = []
    for shard_name in shards:
        try:
            header, _ = _read_safetensors_header(source_path / shard_name)
        except ConversionError:
            continue
        names.extend(key for key in header if key != "__metadata__")
    return tuple(names), shards, False


def _validated_index_shard_names(source_path: Path, weight_map: dict[Any, Any]) -> tuple[str, ...]:
    """Return unique shard filenames after containing them to the source bundle."""

    source_root = source_path.resolve()
    shards: list[str] = []
    for raw_name in weight_map.values():
        if not isinstance(raw_name, str) or not raw_name:
            raise ConversionError(
                "Safetensors index entries must name a safe relative filename.",
                details={"source_path": str(source_path), "shard_name": raw_name},
            )
        shard_path = Path(raw_name)
        if (
            shard_path.is_absolute()
            or shard_path.name != raw_name
            or "/" in raw_name
            or "\\" in raw_name
            or not raw_name.endswith(_WEIGHT_SUFFIXES)
        ):
            raise ConversionError(
                "Safetensors index entries must name a safe relative filename.",
                details={"source_path": str(source_path), "shard_name": raw_name},
            )
        resolved_shard = (source_path / shard_path).resolve()
        try:
            resolved_shard.relative_to(source_root)
        except ValueError as error:
            raise ConversionError(
                "Safetensors index entries must resolve inside the source bundle.",
                details={"source_path": str(source_path), "shard_name": raw_name},
            ) from error
        if not resolved_shard.is_file():
            raise ConversionError(
                "Safetensors index references a missing shard.",
                details={"source_path": str(source_path), "shard_name": raw_name},
            )
        if raw_name not in shards:
            shards.append(raw_name)
    return tuple(shards)


def _prefix_backbone_tensor(name: str) -> str:
    """Move only decoder-body tensors under `model.`, leaving output heads top-level."""

    root = name.split(".", 1)[0]
    return f"{BACKBONE_TENSOR_PREFIX}{name}" if root in _BACKBONE_ROOT_NAMES else name


def _weight_file_names(source_path: Path, layout: CheckpointLayout) -> tuple[str, ...]:
    if layout.shard_names:
        return tuple(name for name in layout.shard_names if (source_path / name).exists())
    return tuple(
        sorted(entry.name for entry in source_path.iterdir() if entry.name.endswith(_WEIGHT_SUFFIXES))
    )


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    """Return a shard's JSON header and the byte offset at which its payload starts."""

    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ConversionError(
                    "Safetensors shard is truncated before its header length.",
                    details={"path": str(path)},
                )
            header_length = struct.unpack("<Q", raw_length)[0]
            # A non-safetensors file reads as a nonsense header length, so the
            # declared size is checked against the file before allocating for it.
            if header_length > file_size - 8:
                raise ConversionError(
                    "Safetensors shard declares a header larger than the file.",
                    details={"path": str(path), "declared_header_bytes": header_length, "file_bytes": file_size},
                )
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise ConversionError(
                    "Safetensors shard is truncated inside its header.",
                    details={"path": str(path)},
                )
            header = json.loads(header_bytes)
    except OSError as error:
        raise ConversionError(
            "Could not read safetensors shard during normalization.",
            details={"path": str(path), "error": str(error)},
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ConversionError(
            "Safetensors shard has an unreadable header.",
            details={"path": str(path), "error": str(error)},
        ) from error
    if not isinstance(header, dict):
        raise ConversionError(
            "Safetensors shard header is not a JSON object.",
            details={"path": str(path)},
        )
    return header, 8 + header_length


def _rewrite_safetensors_names(source: Path, target: Path, rename: Callable[[str], str]) -> None:
    """Copy a shard with renamed tensors.

    Tensor payloads are addressed by offsets relative to the end of the header,
    and renaming a key moves no bytes, so the original payload is streamed through
    unchanged beneath a freshly serialized header.
    """

    header, payload_start = _read_safetensors_header(source)
    renamed: dict[str, Any] = {}
    for key, value in header.items():
        if key == "__metadata__":
            renamed[key] = value
            continue
        new_key = rename(key)
        if new_key in renamed:
            raise ConversionError(
                "Tensor rename collided during checkpoint normalization.",
                details={"path": str(source), "tensor": key, "renamed_to": new_key},
            )
        renamed[new_key] = value
    header_bytes = json.dumps(renamed, separators=(",", ":")).encode("utf-8")
    # safetensors starts the payload on an 8-byte boundary; pad with spaces so the
    # recorded data offsets stay valid against the copied payload.
    header_bytes += b" " * (-len(header_bytes) % 8)
    with source.open("rb") as reader, target.open("wb") as writer:
        writer.write(struct.pack("<Q", len(header_bytes)))
        writer.write(header_bytes)
        reader.seek(payload_start)
        shutil.copyfileobj(reader, writer, _COPY_BUFFER_BYTES)


def _rewrite_index(*, source_path: Path, output_path: Path, rename: Callable[[str], str]) -> None:
    index = _load_json(source_path / SAFETENSORS_INDEX_FILENAME)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return
    index["weight_map"] = {rename(str(name)): value for name, value in weight_map.items()}
    _write_json(output_path / SAFETENSORS_INDEX_FILENAME, index)


def _rewrite_config(*, source_path: Path, output_path: Path, layout: CheckpointLayout) -> list[str]:
    config_data = _load_json(source_path / "config.json")
    if not config_data:
        return []
    logs: list[str] = []
    if layout.needs_tie_reconciliation:
        config_data["tie_word_embeddings"] = True
        logs.append(
            "Set `tie_word_embeddings: true` because the bundle ships no `lm_head` tensor to load."
        )
    _write_json(output_path / "config.json", config_data)
    return logs


def _copy_support_files(*, source_path: Path, output_path: Path, weight_files: tuple[str, ...]) -> None:
    """Mirror tokenizer, template, and config files the exporters read alongside weights."""

    skip = {*weight_files, SAFETENSORS_INDEX_FILENAME, "config.json"}
    for entry in source_path.iterdir():
        if entry.name in skip or entry.name.startswith("."):
            continue
        if entry.is_dir():
            shutil.copytree(entry, output_path / entry.name, dirs_exist_ok=True)
            continue
        if entry.name.endswith(_WEIGHT_SUFFIXES):
            continue
        _link_or_copy(entry, output_path / entry.name)


def _link_or_copy(source: Path, target: Path) -> None:
    """Prefer a hard link so unchanged shards cost no additional disk."""

    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
