"""Block-level disk cache services for reusable local artifacts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from lewlm.core.contracts import GenerateAttachment
from lewlm.core.errors import StorageError
from lewlm.storage.metadata import MetadataStore


class BlockDiskCache:
    """Persist reusable JSON cache blocks on disk and index them in metadata storage."""

    def __init__(self, *, cache_root: Path, metadata_store: MetadataStore) -> None:
        self.cache_root = cache_root / "blocks"
        self.metadata_store = metadata_store

    def get_json_block(self, *, cache_key: str, block_kind: str) -> dict[str, Any] | None:
        record = self.metadata_store.get_cache_block(cache_key)
        if record is None:
            self._record_miss(block_kind)
            return None
        if record["block_kind"] != block_kind:
            raise StorageError(
                "Cache block kind mismatch.",
                details={
                    "cache_key": cache_key,
                    "expected_block_kind": block_kind,
                    "actual_block_kind": record["block_kind"],
                },
            )
        block_path = self.cache_root / str(record["storage_path"])
        if not block_path.is_file():
            raise StorageError(
                "Cached block file is missing on disk.",
                details={"cache_key": cache_key, "block_kind": block_kind, "storage_path": str(block_path)},
            )
        self._record_hit(block_kind)
        try:
            return json.loads(block_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StorageError(
                "Cached block payload is invalid JSON.",
                details={"cache_key": cache_key, "block_kind": block_kind, "storage_path": str(block_path)},
            ) from exc

    def put_json_block(
        self,
        *,
        cache_key: str,
        block_kind: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        stored_size_bytes: int | None = None,
    ) -> None:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        block_path = self._block_path(cache_key=cache_key, block_kind=block_kind)
        block_path.parent.mkdir(parents=True, exist_ok=True)
        block_path.write_text(serialized, encoding="utf-8")
        self.metadata_store.upsert_cache_block(
            cache_key=cache_key,
            block_kind=block_kind,
            storage_path=block_path.relative_to(self.cache_root).as_posix(),
            size_bytes=stored_size_bytes if stored_size_bytes is not None else len(serialized.encode("utf-8")),
            metadata=metadata or {},
        )

    def delete_json_block(self, *, cache_key: str, block_kind: str) -> bool:
        record = self.metadata_store.get_cache_block(cache_key)
        if record is None or record["block_kind"] != block_kind:
            return False
        block_path = self.cache_root / str(record["storage_path"])
        if block_path.exists():
            block_path.unlink()
        self.metadata_store.delete_cache_block(cache_key)
        return True

    def stats(self) -> dict[str, int]:
        return {
            "block_cache_count": self.metadata_store.cache_block_count(),
            "block_cache_bytes": self.metadata_store.cache_block_size_bytes(),
            "block_cache_hits": self.metadata_store.get_counter("block_cache_hits"),
            "block_cache_misses": self.metadata_store.get_counter("block_cache_misses"),
            "multimodal_feature_count": self.metadata_store.cache_block_count(block_kind="multimodal_feature"),
            "multimodal_feature_bytes": self.metadata_store.cache_block_size_bytes(block_kind="multimodal_feature"),
            "multimodal_feature_cache_hits": self.metadata_store.get_counter("multimodal_feature_cache_hits"),
            "multimodal_feature_cache_misses": self.metadata_store.get_counter("multimodal_feature_cache_misses"),
            "multimodal_encoder_count": self.metadata_store.cache_block_count(block_kind="multimodal_encoder"),
            "multimodal_encoder_bytes": self.metadata_store.cache_block_size_bytes(block_kind="multimodal_encoder"),
            "multimodal_encoder_cache_hits": self.metadata_store.get_counter("multimodal_encoder_cache_hits"),
            "multimodal_encoder_cache_misses": self.metadata_store.get_counter("multimodal_encoder_cache_misses"),
            "multimodal_encoder_cache_invalidations": self.metadata_store.get_counter(
                "multimodal_encoder_cache_invalidations",
            ),
        }

    def _block_path(self, *, cache_key: str, block_kind: str) -> Path:
        return self.cache_root / block_kind / cache_key[:2] / f"{cache_key}.json"

    def _record_hit(self, block_kind: str) -> None:
        self.metadata_store.increment_counter("block_cache_hits")
        if block_kind == "multimodal_feature":
            self.metadata_store.increment_counter("multimodal_feature_cache_hits")
        if block_kind == "multimodal_encoder":
            self.metadata_store.increment_counter("multimodal_encoder_cache_hits")

    def _record_miss(self, block_kind: str) -> None:
        self.metadata_store.increment_counter("block_cache_misses")
        if block_kind == "multimodal_feature":
            self.metadata_store.increment_counter("multimodal_feature_cache_misses")
        if block_kind == "multimodal_encoder":
            self.metadata_store.increment_counter("multimodal_encoder_cache_misses")


class MultimodalFeatureCache:
    """Persist prompt-ready attachment features for reuse across multimodal requests."""

    _BLOCK_KIND = "multimodal_feature"
    _CACHE_VERSION = 1

    def __init__(self, *, block_disk_cache: BlockDiskCache) -> None:
        self.block_disk_cache = block_disk_cache

    def get_attachment(self, *, cache_key: str, name: str, source_path: str) -> GenerateAttachment | None:
        payload = self.block_disk_cache.get_json_block(cache_key=cache_key, block_kind=self._BLOCK_KIND)
        if payload is None:
            return None
        return GenerateAttachment(
            attachment_type=str(payload["attachment_type"]),
            name=name,
            source_path=source_path,
            media_type=payload.get("media_type"),
            extracted_text=payload.get("extracted_text"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def put_attachment(
        self,
        *,
        cache_key: str,
        attachment: GenerateAttachment,
        cache_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.block_disk_cache.put_json_block(
            cache_key=cache_key,
            block_kind=self._BLOCK_KIND,
            payload={
                "attachment_type": attachment.attachment_type,
                "media_type": attachment.media_type,
                "extracted_text": attachment.extracted_text,
                "metadata": attachment.metadata,
            },
            metadata=cache_metadata or {},
        )

    def cache_key_for_path_attachment(self, *, raw_bytes: bytes, suffix: str) -> str:
        return self._cache_key(
            {
                "scope": "path_attachment",
                "suffix": suffix,
                "content_sha256": self._content_digest(raw_bytes),
                "input_bytes": len(raw_bytes),
            },
        )

    def cache_key_for_audio_attachment(
        self,
        *,
        raw_bytes: bytes,
        file_name: str,
        suffix: str,
        language: str | None,
        prompt: str | None,
    ) -> str:
        return self._cache_key(
            {
                "scope": "audio_attachment",
                "file_name": file_name,
                "suffix": suffix,
                "language": language,
                "prompt": prompt,
                "content_sha256": self._content_digest(raw_bytes),
                "input_bytes": len(raw_bytes),
            },
        )

    @classmethod
    def _cache_key(cls, payload: dict[str, Any]) -> str:
        serialized = json.dumps(
            {"version": cls._CACHE_VERSION, **payload},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256()
        digest.update(serialized.encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _content_digest(raw_bytes: bytes) -> str:
        digest = hashlib.sha256()
        digest.update(raw_bytes)
        return digest.hexdigest()


@dataclass(slots=True)
class _ResidentEncoderFeature:
    cache_key: str
    feature: Any
    source_locator: str | None
    feature_bytes: int


class MultimodalEncoderCache:
    """Track reusable multimodal encoder outputs with explicit invalidation."""

    _BLOCK_KIND = "multimodal_encoder"
    _CACHE_VERSION = 1

    def __init__(self, *, block_disk_cache: BlockDiskCache, max_entries: int = 64) -> None:
        self.block_disk_cache = block_disk_cache
        self.max_entries = max_entries
        self._resident: OrderedDict[str, _ResidentEncoderFeature] = OrderedDict()

    def cache_key_for_feature(
        self,
        *,
        runtime: str,
        model_id: str,
        model_fingerprint: str,
        modality: str,
        content_sha256: str,
        preprocessing_fingerprint: str,
    ) -> str:
        return _cache_key(
            {
                "scope": "multimodal_encoder",
                "runtime": runtime,
                "model_id": model_id,
                "model_fingerprint": model_fingerprint,
                "modality": modality,
                "content_sha256": content_sha256,
                "preprocessing_fingerprint": preprocessing_fingerprint,
            },
            version=self._CACHE_VERSION,
        )

    def get_feature(self, *, cache_key: str) -> Any | None:
        record = self.block_disk_cache.metadata_store.get_cache_block(cache_key)
        if record is None or record["block_kind"] != self._BLOCK_KIND:
            self._record_miss()
            self._resident.pop(cache_key, None)
            return None
        resident = self._resident.get(cache_key)
        if resident is None:
            self._record_miss()
            return None
        self._resident.move_to_end(cache_key)
        self._record_hit()
        return resident.feature

    def put_feature(
        self,
        *,
        cache_key: str,
        runtime: str,
        model_id: str,
        model_fingerprint: str,
        modality: str,
        content_sha256: str,
        preprocessing_fingerprint: str,
        feature: Any,
        feature_bytes: int | None = None,
        source_locator: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if source_locator is not None:
            self.invalidate_source(
                source_locator=source_locator,
                runtime=runtime,
                model_id=model_id,
                keep_cache_key=cache_key,
            )
        estimated_bytes = feature_bytes if feature_bytes is not None else estimate_feature_bytes(feature)
        descriptor_metadata = {
            "runtime": runtime,
            "model_id": model_id,
            "model_fingerprint": model_fingerprint,
            "modality": modality,
            "content_sha256": content_sha256,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "source_locator": source_locator,
            "feature_bytes": estimated_bytes,
            **(metadata or {}),
        }
        self.block_disk_cache.put_json_block(
            cache_key=cache_key,
            block_kind=self._BLOCK_KIND,
            payload={
                "version": self._CACHE_VERSION,
                "runtime": runtime,
                "model_id": model_id,
                "model_fingerprint": model_fingerprint,
                "modality": modality,
                "content_sha256": content_sha256,
                "preprocessing_fingerprint": preprocessing_fingerprint,
                "source_locator": source_locator,
                "feature_bytes": estimated_bytes,
            },
            metadata=descriptor_metadata,
            stored_size_bytes=estimated_bytes if estimated_bytes > 0 else None,
        )
        self._resident[cache_key] = _ResidentEncoderFeature(
            cache_key=cache_key,
            feature=feature,
            source_locator=source_locator,
            feature_bytes=estimated_bytes,
        )
        self._resident.move_to_end(cache_key)
        self._evict_if_needed()

    def invalidate_source(
        self,
        *,
        source_locator: str,
        runtime: str | None = None,
        model_id: str | None = None,
        keep_cache_key: str | None = None,
    ) -> int:
        removed = 0
        for record in self.block_disk_cache.metadata_store.list_cache_blocks(block_kind=self._BLOCK_KIND):
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if metadata.get("source_locator") != source_locator:
                continue
            if runtime is not None and metadata.get("runtime") != runtime:
                continue
            if model_id is not None and metadata.get("model_id") != model_id:
                continue
            if keep_cache_key is not None and record["cache_key"] == keep_cache_key:
                continue
            if self.block_disk_cache.delete_json_block(cache_key=str(record["cache_key"]), block_kind=self._BLOCK_KIND):
                removed += 1
            self._resident.pop(str(record["cache_key"]), None)
        if removed:
            self.block_disk_cache.metadata_store.increment_counter("multimodal_encoder_cache_invalidations", removed)
        return removed

    def drop_runtime_resident_features(self, *, runtime: str | None = None, model_id: str | None = None) -> int:
        removed = 0
        for record in self.block_disk_cache.metadata_store.list_cache_blocks(block_kind=self._BLOCK_KIND):
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if runtime is not None and metadata.get("runtime") != runtime:
                continue
            if model_id is not None and metadata.get("model_id") != model_id:
                continue
            cache_key = str(record["cache_key"])
            if self._resident.pop(cache_key, None) is not None:
                removed += 1
        return removed

    def stats(self) -> dict[str, int]:
        resident_bytes = sum(entry.feature_bytes for entry in self._resident.values())
        return {
            "multimodal_encoder_count": self.block_disk_cache.metadata_store.cache_block_count(block_kind=self._BLOCK_KIND),
            "multimodal_encoder_bytes": self.block_disk_cache.metadata_store.cache_block_size_bytes(
                block_kind=self._BLOCK_KIND,
            ),
            "multimodal_encoder_cache_hits": self.block_disk_cache.metadata_store.get_counter(
                "multimodal_encoder_cache_hits",
            ),
            "multimodal_encoder_cache_misses": self.block_disk_cache.metadata_store.get_counter(
                "multimodal_encoder_cache_misses",
            ),
            "multimodal_encoder_cache_invalidations": self.block_disk_cache.metadata_store.get_counter(
                "multimodal_encoder_cache_invalidations",
            ),
            "multimodal_encoder_resident_count": len(self._resident),
            "multimodal_encoder_resident_bytes": resident_bytes,
        }

    def _record_hit(self) -> None:
        self.block_disk_cache.metadata_store.increment_counter("multimodal_encoder_cache_hits")

    def _record_miss(self) -> None:
        self.block_disk_cache.metadata_store.increment_counter("multimodal_encoder_cache_misses")

    def _evict_if_needed(self) -> None:
        while len(self._resident) > self.max_entries:
            self._resident.popitem(last=False)


def estimate_feature_bytes(feature: Any) -> int:
    raw_nbytes = getattr(feature, "nbytes", None)
    if isinstance(raw_nbytes, int):
        return max(raw_nbytes, 0)
    size = getattr(feature, "size", None)
    itemsize = getattr(feature, "itemsize", None)
    if isinstance(size, int) and isinstance(itemsize, int):
        return max(size * itemsize, 0)
    if isinstance(feature, (bytes, bytearray)):
        return len(feature)
    return 0


def _cache_key(payload: dict[str, Any], *, version: int) -> str:
    serialized = json.dumps(
        {"version": version, **payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256()
    digest.update(serialized.encode("utf-8"))
    return digest.hexdigest()
