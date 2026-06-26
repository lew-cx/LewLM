"""Persistent content-addressed storage for paged prompt-prefix cache entries."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from lewlm.core.contracts import utc_now


_CACHE_VERSION = 2
_SAFE_MODEL_ID_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_CACHE_NAMESPACE_SEGMENT_LENGTH = 12
_MAX_LEGACY_CACHE_NAMESPACE_SEGMENT_LENGTH = 32
_MAX_CACHE_MODEL_SEGMENT_LENGTH = 20
_CACHE_SEGMENT_DIGEST_LENGTH = 12
_CACHE_KEY_FILENAME_LENGTH = 20


@dataclass(slots=True)
class StoredPrefixCachePage:
    page_key: str
    model_id: str
    parent_page_key: str | None
    page_index: int
    page_size_tokens: int
    tokens: tuple[int, ...]
    estimated_size_bytes: int
    ref_count: int
    access_count: int
    created_at: str
    updated_at: str
    last_used_at: str


@dataclass(slots=True)
class StoredPrefixCacheEntry:
    cache_key: str
    model_id: str
    prefix_tokens: tuple[int, ...]
    page_keys: tuple[str, ...]
    page_size_tokens: int
    payload: Any | None
    payload_persisted: bool
    estimated_size_bytes: int
    access_count: int
    created_at: str
    updated_at: str
    last_used_at: str


@dataclass(slots=True)
class StoredPrefixCacheSaveResult:
    entry: StoredPrefixCacheEntry
    new_page_count: int
    shared_page_count: int


@dataclass(slots=True)
class PersistentPrefixCacheEvictionSummary:
    entries: tuple[StoredPrefixCacheEntry, ...]
    page_eviction_count: int


@dataclass(slots=True)
class _PrefixCachePageDescriptor:
    page_key: str
    parent_page_key: str | None
    page_index: int
    tokens: tuple[int, ...]
    page_size_tokens: int
    estimated_size_bytes: int


class PersistentPrefixCacheStore:
    """Persist paged prompt-prefix cache entries on disk."""

    def __init__(self, *, cache_root: Path, namespace: str, page_size_tokens: int = 16) -> None:
        self.cache_root = cache_root / "ppc" / _compact_cache_path_segment(
            namespace,
            fallback="runtime",
            max_length=_MAX_CACHE_NAMESPACE_SEGMENT_LENGTH,
        )
        self._legacy_cache_roots = (
            cache_root / "prompt-prefix-cache" / _compact_cache_path_segment(
                namespace,
                fallback="runtime",
                max_length=_MAX_LEGACY_CACHE_NAMESPACE_SEGMENT_LENGTH,
            ),
        )
        self.page_size_tokens = max(int(page_size_tokens), 1)
        self._entries_by_model: dict[str, dict[str, StoredPrefixCacheEntry]] = {}
        self._pages_by_model: dict[str, dict[str, StoredPrefixCachePage]] = {}
        self._loaded_models: set[str] = set()
        self._lock = Lock()

    def lookup(self, *, model_id: str, prompt_tokens: tuple[int, ...]) -> StoredPrefixCacheEntry | None:
        normalized_prompt = tuple(int(token) for token in prompt_tokens)
        with self._lock:
            best_entry, _, best_page_length = self._best_entry_match_locked(
                model_id=model_id,
                prompt_tokens=normalized_prompt,
            )
            if best_entry is None or not best_entry.payload_persisted:
                return None
            now = utc_now().isoformat()
            best_entry.access_count += 1
            best_entry.last_used_at = now
            best_entry.updated_at = now
            self._write_entry_locked(best_entry)
            pages = self._pages_for_model_locked(model_id)
            for page_key in best_entry.page_keys[:best_page_length]:
                page = pages.get(page_key)
                if page is None:
                    continue
                page.access_count += 1
                page.last_used_at = now
                page.updated_at = now
                self._write_page_locked(page)
            return _copy_stored_entry(best_entry)

    def preview(self, *, model_id: str, prompt_tokens: tuple[int, ...]) -> StoredPrefixCacheEntry | None:
        normalized_prompt = tuple(int(token) for token in prompt_tokens)
        with self._lock:
            best_entry, _, _ = self._best_entry_match_locked(
                model_id=model_id,
                prompt_tokens=normalized_prompt,
            )
            if best_entry is None or not best_entry.payload_persisted:
                return None
            return _copy_stored_entry(best_entry)

    def put(
        self,
        *,
        model_id: str,
        prefix_tokens: tuple[int, ...],
        payload: Any,
        estimated_size_bytes: int | None = None,
    ) -> StoredPrefixCacheSaveResult:
        normalized_prefix = tuple(int(token) for token in prefix_tokens)
        if not normalized_prefix:
            raise ValueError("prefix_tokens must not be empty.")
        cache_key = self.cache_key_for(model_id=model_id, prefix_tokens=normalized_prefix)
        payload_record = _normalize_payload(payload)
        payload_persisted = payload_record is not None
        descriptors = self._page_descriptors_for(model_id=model_id, prefix_tokens=normalized_prefix)
        page_keys = tuple(descriptor.page_key for descriptor in descriptors)
        timestamp = utc_now().isoformat()
        entry = StoredPrefixCacheEntry(
            cache_key=cache_key,
            model_id=model_id,
            prefix_tokens=normalized_prefix,
            page_keys=page_keys,
            page_size_tokens=self.page_size_tokens,
            payload=payload_record,
            payload_persisted=payload_persisted,
            estimated_size_bytes=max(int(estimated_size_bytes or 0), 0),
            access_count=0,
            created_at=timestamp,
            updated_at=timestamp,
            last_used_at=timestamp,
        )
        with self._lock:
            entries = self._entries_for_model_locked(model_id)
            pages = self._pages_for_model_locked(model_id)
            existing = entries.get(cache_key)
            existing_page_keys = set(existing.page_keys) if existing is not None else set()
            shared_page_count = 0
            prefix_still_shared = True
            new_page_count = 0
            for descriptor in descriptors:
                existing_page = pages.get(descriptor.page_key)
                if existing_page is not None and prefix_still_shared:
                    shared_page_count += 1
                else:
                    prefix_still_shared = False
                if existing_page is None:
                    new_page_count += 1
                    page = StoredPrefixCachePage(
                        page_key=descriptor.page_key,
                        model_id=model_id,
                        parent_page_key=descriptor.parent_page_key,
                        page_index=descriptor.page_index,
                        page_size_tokens=descriptor.page_size_tokens,
                        tokens=descriptor.tokens,
                        estimated_size_bytes=max(int(descriptor.estimated_size_bytes), 0),
                        ref_count=1,
                        access_count=0,
                        created_at=timestamp,
                        updated_at=timestamp,
                        last_used_at=timestamp,
                    )
                    pages[descriptor.page_key] = page
                else:
                    page = existing_page
                    if descriptor.page_key not in existing_page_keys:
                        page.ref_count += 1
                    page.updated_at = timestamp
                    page.last_used_at = timestamp
                self._write_page_locked(page)
            if existing is not None:
                entry.created_at = existing.created_at
                entry.access_count = existing.access_count
                removed_page_keys = [
                    page_key
                    for page_key in existing.page_keys
                    if page_key not in set(page_keys)
                ]
                self._release_page_keys_locked(model_id=model_id, page_keys=removed_page_keys)
            if entry.estimated_size_bytes <= 0:
                entry.estimated_size_bytes = _serialized_size(self._serialize_entry(entry))
            entries[cache_key] = entry
            self._write_entry_locked(entry)
        return StoredPrefixCacheSaveResult(
            entry=_copy_stored_entry(entry),
            new_page_count=new_page_count,
            shared_page_count=shared_page_count,
        )

    def enforce_limit(self, *, model_id: str, max_entries: int) -> PersistentPrefixCacheEvictionSummary:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")
        evicted_entries: list[StoredPrefixCacheEntry] = []
        page_evictions = 0
        with self._lock:
            entries = self._entries_for_model_locked(model_id)
            while len(entries) > max_entries:
                cache_key, entry = min(
                    entries.items(),
                    key=lambda item: (item[1].last_used_at, item[1].updated_at, item[0]),
                )
                evicted_entries.append(_copy_stored_entry(entry))
                entries.pop(cache_key, None)
                self._unlink_entry_paths(model_id=model_id, cache_key=cache_key)
                page_evictions += self._release_page_keys_locked(model_id=model_id, page_keys=entry.page_keys)
        return PersistentPrefixCacheEvictionSummary(
            entries=tuple(evicted_entries),
            page_eviction_count=page_evictions,
        )

    def invalidate(
        self,
        *,
        model_id: str,
        cache_key: str | None = None,
    ) -> PersistentPrefixCacheEvictionSummary:
        evicted_entries: list[StoredPrefixCacheEntry] = []
        page_evictions = 0
        with self._lock:
            entries = self._entries_for_model_locked(model_id)
            matching_entries = [
                entry
                for entry in entries.values()
                if cache_key is None or entry.cache_key == cache_key
            ]
            for entry in matching_entries:
                evicted_entries.append(_copy_stored_entry(entry))
                entries.pop(entry.cache_key, None)
                self._unlink_entry_paths(model_id=model_id, cache_key=entry.cache_key)
                page_evictions += self._release_page_keys_locked(model_id=model_id, page_keys=entry.page_keys)
        return PersistentPrefixCacheEvictionSummary(
            entries=tuple(evicted_entries),
            page_eviction_count=page_evictions,
        )

    def entry_count(self, *, model_id: str | None = None) -> int:
        if model_id is not None:
            with self._lock:
                return len(self._entries_for_model_locked(model_id))
        total = 0
        for model_dir in self.cache_root.glob("*"):
            if not model_dir.is_dir():
                continue
            entry_dir = model_dir / "entries"
            if entry_dir.is_dir():
                total += sum(1 for path in entry_dir.glob("*.json") if path.is_file())
            total += sum(1 for path in model_dir.glob("*.json") if path.is_file())
        return total

    def total_size_bytes(self, *, model_id: str | None = None) -> int:
        if model_id is not None:
            with self._lock:
                return sum(
                    entry.estimated_size_bytes
                    for entry in self._entries_for_model_locked(model_id).values()
                )
        total = 0
        for model_dir in self.cache_root.glob("*"):
            if not model_dir.is_dir():
                continue
            entry_dir = model_dir / "entries"
            if entry_dir.is_dir():
                total += sum(path.stat().st_size for path in entry_dir.glob("*.json") if path.is_file())
            total += sum(path.stat().st_size for path in model_dir.glob("*.json") if path.is_file())
        return total

    def page_count(self, *, model_id: str | None = None) -> int:
        if model_id is not None:
            with self._lock:
                return len(self._pages_for_model_locked(model_id))
        total = 0
        for path in self.cache_root.glob("*/pages/*.json"):
            if path.is_file():
                total += 1
        return total

    def total_page_size_bytes(self, *, model_id: str | None = None) -> int:
        if model_id is not None:
            with self._lock:
                return sum(
                    page.estimated_size_bytes
                    for page in self._pages_for_model_locked(model_id).values()
                )
        total = 0
        for path in self.cache_root.glob("*/pages/*.json"):
            if path.is_file():
                total += path.stat().st_size
        return total

    @classmethod
    def cache_key_for(cls, *, model_id: str, prefix_tokens: tuple[int, ...]) -> str:
        serialized = json.dumps(
            {
                "version": _CACHE_VERSION,
                "model_id": model_id,
                "prefix_tokens": list(prefix_tokens),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256()
        digest.update(serialized.encode("utf-8"))
        return digest.hexdigest()

    @classmethod
    def page_key_for(
        cls,
        *,
        model_id: str,
        parent_page_key: str | None,
        page_index: int,
        page_tokens: tuple[int, ...],
        page_size_tokens: int,
    ) -> str:
        serialized = json.dumps(
            {
                "version": _CACHE_VERSION,
                "model_id": model_id,
                "parent_page_key": parent_page_key,
                "page_index": page_index,
                "page_size_tokens": page_size_tokens,
                "page_tokens": list(page_tokens),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256()
        digest.update(serialized.encode("utf-8"))
        return digest.hexdigest()

    def _entries_for_model_locked(self, model_id: str) -> dict[str, StoredPrefixCacheEntry]:
        self._ensure_model_loaded_locked(model_id)
        return self._entries_by_model.setdefault(model_id, {})

    def _pages_for_model_locked(self, model_id: str) -> dict[str, StoredPrefixCachePage]:
        self._ensure_model_loaded_locked(model_id)
        return self._pages_by_model.setdefault(model_id, {})

    def _ensure_model_loaded_locked(self, model_id: str) -> None:
        if model_id in self._loaded_models:
            return
        entries = self._load_model_entries_locked(model_id)
        pages = self._load_model_pages_locked(model_id)
        if not pages and entries:
            pages = self._rebuild_pages_from_entries(model_id=model_id, entries=entries)
        self._entries_by_model[model_id] = entries
        self._pages_by_model[model_id] = pages
        self._loaded_models.add(model_id)

    def _best_entry_match_locked(
        self,
        *,
        model_id: str,
        prompt_tokens: tuple[int, ...],
    ) -> tuple[StoredPrefixCacheEntry | None, int, int]:
        entries = self._entries_for_model_locked(model_id)
        if not entries:
            return None, 0, 0
        prompt_page_keys = self._page_keys_for(model_id=model_id, prefix_tokens=prompt_tokens)
        best_entry: StoredPrefixCacheEntry | None = None
        best_prefix_length = 0
        best_page_length = 0
        for entry in entries.values():
            prefix_length = _longest_token_prefix(entry.prefix_tokens, prompt_tokens)
            if prefix_length <= 0:
                continue
            page_length = _matched_page_count(entry.page_keys, prompt_page_keys)
            if prefix_length > best_prefix_length or (
                prefix_length == best_prefix_length and page_length > best_page_length
            ):
                best_entry = entry
                best_prefix_length = prefix_length
                best_page_length = page_length
        return best_entry, best_prefix_length, best_page_length

    def _load_model_entries_locked(self, model_id: str) -> dict[str, StoredPrefixCacheEntry]:
        paths: list[Path] = []
        for model_dir in self._model_dirs(model_id):
            entry_dir = model_dir / "entries"
            if model_dir.is_dir():
                paths.extend(
                    sorted(path for path in model_dir.glob("*.json") if path.is_file())
                )
            if entry_dir.is_dir():
                paths.extend(
                    sorted(path for path in entry_dir.glob("*.json") if path.is_file())
                )
        entries: dict[str, StoredPrefixCacheEntry] = {}
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            prefix_tokens = tuple(int(token) for token in payload.get("prefix_tokens", []))
            if not prefix_tokens:
                continue
            page_size_tokens = max(int(payload.get("page_size_tokens", self.page_size_tokens)), 1)
            page_keys_payload = payload.get("page_keys", [])
            if isinstance(page_keys_payload, list) and page_keys_payload:
                page_keys = tuple(str(key) for key in page_keys_payload)
            else:
                page_keys = tuple(
                    descriptor.page_key
                    for descriptor in self._page_descriptors_for(
                        model_id=model_id,
                        prefix_tokens=prefix_tokens,
                        page_size_tokens=page_size_tokens,
                    )
                )
            entry = StoredPrefixCacheEntry(
                cache_key=str(payload["cache_key"]),
                model_id=str(payload["model_id"]),
                prefix_tokens=prefix_tokens,
                page_keys=page_keys,
                page_size_tokens=page_size_tokens,
                payload=payload.get("payload"),
                payload_persisted=bool(payload.get("payload_persisted")),
                estimated_size_bytes=max(int(payload.get("estimated_size_bytes", 0)), int(path.stat().st_size)),
                access_count=max(int(payload.get("access_count", 0)), 0),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
                last_used_at=str(payload.get("last_used_at") or payload["updated_at"]),
            )
            entries[entry.cache_key] = entry
        return entries

    def _load_model_pages_locked(self, model_id: str) -> dict[str, StoredPrefixCachePage]:
        pages: dict[str, StoredPrefixCachePage] = {}
        for page_dir in self._page_dirs(model_id):
            if not page_dir.is_dir():
                continue
            for path in page_dir.glob("*.json"):
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                tokens = tuple(int(token) for token in payload.get("tokens", []))
                if not tokens:
                    continue
                page = StoredPrefixCachePage(
                    page_key=str(payload["page_key"]),
                    model_id=str(payload["model_id"]),
                    parent_page_key=(
                        str(payload["parent_page_key"])
                        if payload.get("parent_page_key") is not None
                        else None
                    ),
                    page_index=max(int(payload.get("page_index", 0)), 0),
                    page_size_tokens=max(int(payload.get("page_size_tokens", self.page_size_tokens)), 1),
                    tokens=tokens,
                    estimated_size_bytes=max(int(payload.get("estimated_size_bytes", 0)), int(path.stat().st_size)),
                    ref_count=max(int(payload.get("ref_count", 1)), 1),
                    access_count=max(int(payload.get("access_count", 0)), 0),
                    created_at=str(payload["created_at"]),
                    updated_at=str(payload["updated_at"]),
                    last_used_at=str(payload.get("last_used_at") or payload["updated_at"]),
                )
                pages[page.page_key] = page
        return pages

    def _rebuild_pages_from_entries(
        self,
        *,
        model_id: str,
        entries: dict[str, StoredPrefixCacheEntry],
    ) -> dict[str, StoredPrefixCachePage]:
        pages: dict[str, StoredPrefixCachePage] = {}
        for entry in entries.values():
            for descriptor in self._page_descriptors_for(
                model_id=model_id,
                prefix_tokens=entry.prefix_tokens,
                page_size_tokens=entry.page_size_tokens,
            ):
                page = pages.get(descriptor.page_key)
                if page is None:
                    page = StoredPrefixCachePage(
                        page_key=descriptor.page_key,
                        model_id=model_id,
                        parent_page_key=descriptor.parent_page_key,
                        page_index=descriptor.page_index,
                        page_size_tokens=descriptor.page_size_tokens,
                        tokens=descriptor.tokens,
                        estimated_size_bytes=descriptor.estimated_size_bytes,
                        ref_count=1,
                        access_count=entry.access_count,
                        created_at=entry.created_at,
                        updated_at=entry.updated_at,
                        last_used_at=entry.last_used_at,
                    )
                    pages[descriptor.page_key] = page
                else:
                    page.ref_count += 1
                    page.access_count = max(page.access_count, entry.access_count)
                    page.updated_at = max(page.updated_at, entry.updated_at)
                    page.last_used_at = max(page.last_used_at, entry.last_used_at)
        return pages

    def _write_entry_locked(self, entry: StoredPrefixCacheEntry) -> None:
        path = self._entry_path(model_id=entry.model_id, cache_key=entry.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._serialize_entry(entry), indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        self._unlink_legacy_paths(path, self._legacy_entry_paths(model_id=entry.model_id, cache_key=entry.cache_key))
        entry.estimated_size_bytes = max(entry.estimated_size_bytes, int(path.stat().st_size))

    def _write_page_locked(self, page: StoredPrefixCachePage) -> None:
        path = self._page_path(model_id=page.model_id, page_key=page.page_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._serialize_page(page), indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        self._unlink_legacy_paths(path, self._legacy_page_paths(model_id=page.model_id, page_key=page.page_key))
        page.estimated_size_bytes = max(page.estimated_size_bytes, int(path.stat().st_size))

    def _release_page_keys_locked(self, *, model_id: str, page_keys: tuple[str, ...] | list[str]) -> int:
        pages = self._pages_for_model_locked(model_id)
        removed = 0
        for page_key in page_keys:
            page = pages.get(page_key)
            if page is None:
                continue
            if page.ref_count > 1:
                page.ref_count -= 1
                page.updated_at = utc_now().isoformat()
                self._write_page_locked(page)
                continue
            pages.pop(page_key, None)
            self._unlink_page_paths(model_id=model_id, page_key=page_key)
            removed += 1
        return removed

    def _page_descriptors_for(
        self,
        *,
        model_id: str,
        prefix_tokens: tuple[int, ...],
        page_size_tokens: int | None = None,
    ) -> tuple[_PrefixCachePageDescriptor, ...]:
        normalized_page_size = max(int(page_size_tokens or self.page_size_tokens), 1)
        descriptors: list[_PrefixCachePageDescriptor] = []
        parent_page_key: str | None = None
        for page_index, start in enumerate(range(0, len(prefix_tokens), normalized_page_size)):
            tokens = prefix_tokens[start : start + normalized_page_size]
            page_key = self.page_key_for(
                model_id=model_id,
                parent_page_key=parent_page_key,
                page_index=page_index,
                page_tokens=tokens,
                page_size_tokens=normalized_page_size,
            )
            descriptor = _PrefixCachePageDescriptor(
                page_key=page_key,
                parent_page_key=parent_page_key,
                page_index=page_index,
                tokens=tokens,
                page_size_tokens=normalized_page_size,
                estimated_size_bytes=_serialized_size(
                    {
                        "page_key": page_key,
                        "model_id": model_id,
                        "parent_page_key": parent_page_key,
                        "page_index": page_index,
                        "page_size_tokens": normalized_page_size,
                        "tokens": list(tokens),
                    },
                ),
            )
            descriptors.append(descriptor)
            parent_page_key = page_key
        return tuple(descriptors)

    def _page_keys_for(
        self,
        *,
        model_id: str,
        prefix_tokens: tuple[int, ...],
        page_size_tokens: int | None = None,
    ) -> tuple[str, ...]:
        return tuple(
            descriptor.page_key
            for descriptor in self._page_descriptors_for(
                model_id=model_id,
                prefix_tokens=prefix_tokens,
                page_size_tokens=page_size_tokens,
            )
        )

    @staticmethod
    def _serialize_entry(entry: StoredPrefixCacheEntry) -> dict[str, Any]:
        return {
            "cache_key": entry.cache_key,
            "model_id": entry.model_id,
            "prefix_tokens": list(entry.prefix_tokens),
            "page_keys": list(entry.page_keys),
            "page_size_tokens": entry.page_size_tokens,
            "payload": entry.payload,
            "payload_persisted": entry.payload_persisted,
            "estimated_size_bytes": entry.estimated_size_bytes,
            "access_count": entry.access_count,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "last_used_at": entry.last_used_at,
        }

    @staticmethod
    def _serialize_page(page: StoredPrefixCachePage) -> dict[str, Any]:
        return {
            "page_key": page.page_key,
            "model_id": page.model_id,
            "parent_page_key": page.parent_page_key,
            "page_index": page.page_index,
            "page_size_tokens": page.page_size_tokens,
            "tokens": list(page.tokens),
            "estimated_size_bytes": page.estimated_size_bytes,
            "ref_count": page.ref_count,
            "access_count": page.access_count,
            "created_at": page.created_at,
            "updated_at": page.updated_at,
            "last_used_at": page.last_used_at,
        }

    def _model_dir(self, model_id: str) -> Path:
        return self.cache_root / _compact_cache_path_segment(
            model_id,
            fallback="model",
            max_length=_MAX_CACHE_MODEL_SEGMENT_LENGTH,
        )

    def _model_dirs(self, model_id: str) -> tuple[Path, ...]:
        current = self._model_dir(model_id)
        legacy_dirs = self._legacy_model_dirs(model_id)
        return _unique_paths((current, *legacy_dirs))

    def _entry_dir(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "entries"

    def _page_dir(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "pages"

    def _page_dirs(self, model_id: str) -> tuple[Path, ...]:
        return tuple(model_dir / "pages" for model_dir in self._model_dirs(model_id))

    def _entry_path(self, *, model_id: str, cache_key: str) -> Path:
        return self._entry_dir(model_id) / f"{_compact_cache_key_filename(cache_key)}.json"

    def _page_path(self, *, model_id: str, page_key: str) -> Path:
        return self._page_dir(model_id) / f"{_compact_cache_key_filename(page_key)}.json"

    def _legacy_model_dirs(self, model_id: str) -> tuple[Path, ...]:
        legacy_dirs: list[Path] = []
        for cache_root in self._legacy_cache_roots:
            legacy_dirs.append(
                cache_root / _compact_cache_path_segment(
                    model_id,
                    fallback="model",
                    max_length=_MAX_CACHE_MODEL_SEGMENT_LENGTH,
                ),
            )
            legacy_dirs.append(self._legacy_model_dir(cache_root, model_id))
        return tuple(legacy_dirs)

    @staticmethod
    def _legacy_model_dir(cache_root: Path, model_id: str) -> Path:
        normalized = _SAFE_MODEL_ID_PATTERN.sub("-", model_id).strip("-") or "model"
        digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:10]
        return cache_root / f"{normalized[:48]}-{digest}"

    def _legacy_entry_paths(self, *, model_id: str, cache_key: str) -> tuple[Path, ...]:
        paths: list[Path] = []
        for model_dir in self._model_dirs(model_id):
            paths.append(model_dir / "entries" / f"{cache_key}.json")
            paths.append(model_dir / f"{cache_key}.json")
        return tuple(paths)

    def _legacy_page_paths(self, *, model_id: str, page_key: str) -> tuple[Path, ...]:
        return tuple(page_dir / f"{page_key}.json" for page_dir in self._page_dirs(model_id))

    def _unlink_entry_paths(self, *, model_id: str, cache_key: str) -> None:
        paths = (
            self._entry_path(model_id=model_id, cache_key=cache_key),
            *self._legacy_entry_paths(model_id=model_id, cache_key=cache_key),
        )
        self._unlink_legacy_paths(None, paths)

    def _unlink_page_paths(self, *, model_id: str, page_key: str) -> None:
        paths = (
            self._page_path(model_id=model_id, page_key=page_key),
            *self._legacy_page_paths(model_id=model_id, page_key=page_key),
        )
        self._unlink_legacy_paths(None, paths)

    @staticmethod
    def _unlink_legacy_paths(primary_path: Path | None, paths: tuple[Path, ...]) -> None:
        seen: set[Path] = set()
        for path in paths:
            if primary_path is not None and path == primary_path:
                continue
            if path in seen:
                continue
            seen.add(path)
            path.unlink(missing_ok=True)


def _compact_cache_path_segment(value: str, *, fallback: str, max_length: int) -> str:
    normalized = _SAFE_MODEL_ID_PATTERN.sub("-", value).strip("-") or fallback
    if len(normalized) <= max_length:
        return normalized
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:_CACHE_SEGMENT_DIGEST_LENGTH]
    prefix_length = max(max_length - len(digest) - 1, 0)
    prefix = normalized[:prefix_length].rstrip("-._")
    if not prefix:
        return digest[:max_length]
    return f"{prefix}-{digest}"


def _compact_cache_key_filename(key: str) -> str:
    if len(key) <= _CACHE_KEY_FILENAME_LENGTH:
        return key
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:_CACHE_KEY_FILENAME_LENGTH]


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _normalize_payload(payload: Any) -> Any | None:
    if payload is None:
        return None
    if isinstance(payload, (bool, int, float, str)):
        return payload
    if isinstance(payload, list):
        items = [_normalize_payload(item) for item in payload]
        return items if all(item is not None or original is None for item, original in zip(items, payload)) else None
    if isinstance(payload, tuple):
        items = [_normalize_payload(item) for item in payload]
        return items if all(item is not None or original is None for item, original in zip(items, payload)) else None
    if isinstance(payload, dict):
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                return None
            normalized_value = _normalize_payload(value)
            if normalized_value is None and value is not None:
                return None
            normalized[key] = normalized_value
        return normalized
    return None


def _copy_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _copy_json_value(item) for key, item in value.items()}
    return value


def _copy_stored_entry(entry: StoredPrefixCacheEntry) -> StoredPrefixCacheEntry:
    return StoredPrefixCacheEntry(
        cache_key=entry.cache_key,
        model_id=entry.model_id,
        prefix_tokens=entry.prefix_tokens,
        page_keys=entry.page_keys,
        page_size_tokens=entry.page_size_tokens,
        payload=_copy_json_value(entry.payload),
        payload_persisted=entry.payload_persisted,
        estimated_size_bytes=entry.estimated_size_bytes,
        access_count=entry.access_count,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        last_used_at=entry.last_used_at,
    )


def _serialized_size(payload: dict[str, Any]) -> int:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return len(serialized.encode("utf-8"))


def _matched_page_count(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    matched = 0
    for left_page_key, right_page_key in zip(left, right):
        if left_page_key != right_page_key:
            break
        matched += 1
    return matched


def _longest_token_prefix(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    prefix_length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        prefix_length += 1
    return prefix_length
