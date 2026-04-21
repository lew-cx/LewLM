"""Helpers for runtime-local paged prompt-prefix caches."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, Sequence

from lewlm.storage.prefix_cache_store import PersistentPrefixCacheStore, StoredPrefixCacheSaveResult


@dataclass(slots=True)
class PrefixCacheEntry:
    model_id: str
    prefix_tokens: tuple[int, ...]
    payload: Any
    estimated_size_bytes: int | None = None
    cache_key: str | None = None
    source: Literal["resident", "persisted"] = "resident"
    page_keys: tuple[str, ...] = ()
    page_size_tokens: int = 0
    page_count: int = 0
    shared_page_count: int = 0
    new_page_count: int = 0


@dataclass(slots=True)
class PrefixCacheLookup:
    entry: PrefixCacheEntry
    prefix_length: int
    matched_page_count: int = 0
    resident_page_hits: int = 0
    persisted_page_hits: int = 0
    restored_page_count: int = 0


@dataclass(slots=True)
class PrefixCachePreview:
    model_id: str
    prefix_tokens: tuple[int, ...]
    cache_key: str | None = None
    source: Literal["resident", "persisted"] = "resident"
    prefix_length: int = 0
    matched_page_count: int = 0
    resident_page_hits: int = 0
    persisted_page_hits: int = 0
    page_size_tokens: int = 0
    page_count: int = 0
    uncached_token_count: int = 0


@dataclass(slots=True)
class _ResidentPrefixCachePage:
    page_key: str
    estimated_size_bytes: int
    ref_count: int = 1


@dataclass(slots=True)
class _ResidentInsertSummary:
    new_page_count: int
    shared_page_count: int
    evicted_entry_count: int
    evicted_page_count: int


def longest_token_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    prefix_length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        prefix_length += 1
    return prefix_length


class InMemoryTokenPrefixCache:
    """Small paged hot-tier cache with optional persisted cold storage."""

    def __init__(
        self,
        *,
        max_entries_per_model: int = 4,
        max_persisted_entries_per_model: int | None = None,
        persistent_store: PersistentPrefixCacheStore | None = None,
        page_size_tokens: int = 16,
    ) -> None:
        self._max_entries_per_model = max(max_entries_per_model, 1)
        self._max_persisted_entries_per_model = max(
            max_persisted_entries_per_model or self._max_entries_per_model * 8,
            self._max_entries_per_model,
        )
        self._persistent_store = persistent_store
        self._page_size_tokens = max(
            int(page_size_tokens or getattr(persistent_store, "page_size_tokens", 0) or 1),
            1,
        )
        self._entries: dict[str, list[PrefixCacheEntry]] = {}
        self._pages: dict[str, dict[str, _ResidentPrefixCachePage]] = {}
        self._lookup_count = 0
        self._hit_count = 0
        self._miss_count = 0
        self._save_count = 0
        self._saved_prefill_tokens = 0
        self._max_saved_prefill_tokens = 0
        self._resident_hit_count = 0
        self._persistent_hit_count = 0
        self._restore_count = 0
        self._eviction_count = 0
        self._page_hit_count = 0
        self._resident_page_hit_count = 0
        self._persistent_page_hit_count = 0
        self._page_save_count = 0
        self._page_restore_count = 0
        self._resident_page_eviction_count = 0
        self._persistent_page_eviction_count = 0
        self._copy_on_write_reused_pages = 0
        self._invalidation_count = 0
        self._lock = Lock()

    def lookup(self, *, model_id: str, prompt_tokens: Sequence[int]) -> PrefixCacheLookup | None:
        normalized_prompt = tuple(int(token) for token in prompt_tokens)
        with self._lock:
            self._lookup_count += 1
            resident_lookup = self._resident_lookup_locked(model_id=model_id, prompt_tokens=normalized_prompt)
            if resident_lookup is not None:
                self._resident_hit_count += 1
                self._record_hit_locked(resident_lookup)
                return resident_lookup
        persistent_lookup = self._persistent_lookup(model_id=model_id, prompt_tokens=normalized_prompt)
        if persistent_lookup is not None:
            with self._lock:
                self._persistent_hit_count += 1
                self._restore_count += 1
                restored_page_count = self._insert_resident_locked(persistent_lookup.entry).new_page_count
                persistent_lookup.restored_page_count = restored_page_count
                self._page_restore_count += restored_page_count
                self._record_hit_locked(persistent_lookup)
            return persistent_lookup
        with self._lock:
            self._miss_count += 1
        return None

    def preview(self, *, model_id: str, prompt_tokens: Sequence[int]) -> PrefixCachePreview | None:
        normalized_prompt = tuple(int(token) for token in prompt_tokens)
        if not normalized_prompt:
            return None
        with self._lock:
            resident_preview = self._resident_preview_locked(
                model_id=model_id,
                prompt_tokens=normalized_prompt,
            )
            if resident_preview is not None:
                resident_preview.uncached_token_count = max(
                    len(normalized_prompt) - resident_preview.prefix_length,
                    0,
                )
                return resident_preview
        persistent_preview = self._persistent_preview(model_id=model_id, prompt_tokens=normalized_prompt)
        if persistent_preview is not None:
            persistent_preview.uncached_token_count = max(
                len(normalized_prompt) - persistent_preview.prefix_length,
                0,
            )
        return persistent_preview

    def save(
        self,
        *,
        model_id: str,
        prefix_tokens: Sequence[int],
        payload: Any,
        estimated_size_bytes: int | None = None,
    ) -> PrefixCacheEntry | None:
        normalized_prefix = tuple(int(token) for token in prefix_tokens)
        if not normalized_prefix:
            return None
        stored_page_keys = self._page_keys_for(model_id=model_id, prefix_tokens=normalized_prefix)
        resident_shared_page_count = 0
        if stored_page_keys:
            with self._lock:
                resident_shared_page_count = self._shared_page_count_locked(model_id=model_id, page_keys=stored_page_keys)
        persisted_result: StoredPrefixCacheSaveResult | None = None
        if self._persistent_store is not None:
            persisted_result = self._persistent_store.put(
                model_id=model_id,
                prefix_tokens=normalized_prefix,
                payload=payload,
                estimated_size_bytes=estimated_size_bytes,
            )
            evicted = self._persistent_store.enforce_limit(
                model_id=model_id,
                max_entries=self._max_persisted_entries_per_model,
            )
            self._eviction_count += len(evicted.entries)
            self._persistent_page_eviction_count += evicted.page_eviction_count
        with self._lock:
            entry = PrefixCacheEntry(
                model_id=model_id,
                prefix_tokens=normalized_prefix,
                payload=payload,
                estimated_size_bytes=estimated_size_bytes,
                cache_key=persisted_result.entry.cache_key if persisted_result is not None else None,
                page_keys=stored_page_keys,
                page_size_tokens=self._page_size_tokens,
                page_count=len(stored_page_keys),
                shared_page_count=(
                    persisted_result.shared_page_count
                    if persisted_result is not None
                    else resident_shared_page_count
                ),
                new_page_count=(
                    persisted_result.new_page_count
                    if persisted_result is not None
                    else max(len(stored_page_keys) - resident_shared_page_count, 0)
                ),
            )
            insert_summary = self._insert_resident_locked(entry)
            self._page_save_count += insert_summary.new_page_count
            self._copy_on_write_reused_pages += insert_summary.shared_page_count
            self._resident_page_eviction_count += insert_summary.evicted_page_count
            self._eviction_count += insert_summary.evicted_entry_count
            self._save_count += 1
        return PrefixCacheEntry(
            model_id=entry.model_id,
            prefix_tokens=entry.prefix_tokens,
            payload=entry.payload,
            estimated_size_bytes=entry.estimated_size_bytes,
            cache_key=entry.cache_key,
            source=entry.source,
            page_keys=entry.page_keys,
            page_size_tokens=entry.page_size_tokens,
            page_count=entry.page_count,
            shared_page_count=entry.shared_page_count,
            new_page_count=entry.new_page_count,
        )

    def invalidate(
        self,
        *,
        model_id: str,
        cache_key: str | None = None,
        include_persisted: bool = False,
    ) -> dict[str, int]:
        resident_entry_count = 0
        resident_page_count = 0
        with self._lock:
            resident_entries = self._entries.get(model_id, [])
            retained_entries: list[PrefixCacheEntry] = []
            for entry in resident_entries:
                if cache_key is not None and entry.cache_key != cache_key:
                    retained_entries.append(entry)
                    continue
                resident_entry_count += 1
                resident_page_count += self._release_resident_page_keys_locked(
                    model_id=model_id,
                    page_keys=entry.page_keys,
                )
            if retained_entries:
                self._entries[model_id] = retained_entries
            else:
                self._entries.pop(model_id, None)
            if not self._pages.get(model_id):
                self._pages.pop(model_id, None)
            self._invalidation_count += resident_entry_count
        persisted_entry_count = 0
        persisted_page_count = 0
        if include_persisted and self._persistent_store is not None:
            persisted = self._persistent_store.invalidate(model_id=model_id, cache_key=cache_key)
            persisted_entry_count = len(persisted.entries)
            persisted_page_count = persisted.page_eviction_count
            with self._lock:
                self._invalidation_count += persisted_entry_count
        return {
            "resident_entries": resident_entry_count,
            "resident_pages": resident_page_count,
            "persisted_entries": persisted_entry_count,
            "persisted_pages": persisted_page_count,
        }

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            total_entries = sum(len(entries) for entries in self._entries.values())
            resident_size_bytes = sum(
                entry.estimated_size_bytes or 0
                for entries in self._entries.values()
                for entry in entries
            )
            resident_page_count = sum(len(pages) for pages in self._pages.values())
            resident_page_size_bytes = sum(
                page.estimated_size_bytes
                for pages in self._pages.values()
                for page in pages.values()
            )
        persisted_entries = 0
        persisted_size_bytes = 0
        persisted_page_count = 0
        persisted_page_size_bytes = 0
        if self._persistent_store is not None:
            persisted_entries = self._persistent_store.entry_count()
            persisted_size_bytes = self._persistent_store.total_size_bytes()
            persisted_page_count = self._persistent_store.page_count()
            persisted_page_size_bytes = self._persistent_store.total_page_size_bytes()
        return {
            "supported": True,
            "active": total_entries > 0 or persisted_entries > 0,
            "page_size_tokens": self._page_size_tokens,
            "cache_entries": total_entries,
            "cache_size_bytes": resident_size_bytes,
            "resident_cache_entries": total_entries,
            "resident_cache_hits": self._resident_hit_count,
            "persisted_cache_entries": persisted_entries,
            "persisted_cache_size_bytes": persisted_size_bytes,
            "persistent_cache_hits": self._persistent_hit_count,
            "cache_restores": self._restore_count,
            "cache_evictions": self._eviction_count,
            "cache_hits": self._hit_count,
            "cache_misses": self._miss_count,
            "cache_saves": self._save_count,
            "saved_prefill_tokens": self._saved_prefill_tokens,
            "cached_tokens": self._saved_prefill_tokens,
            "max_saved_prefill_tokens": self._max_saved_prefill_tokens,
            "resident_page_count": resident_page_count,
            "resident_page_size_bytes": resident_page_size_bytes,
            "persisted_page_count": persisted_page_count,
            "persisted_page_size_bytes": persisted_page_size_bytes,
            "page_hits": self._page_hit_count,
            "resident_page_hits": self._resident_page_hit_count,
            "persistent_page_hits": self._persistent_page_hit_count,
            "page_saves": self._page_save_count,
            "page_restores": self._page_restore_count,
            "resident_page_evictions": self._resident_page_eviction_count,
            "persisted_page_evictions": self._persistent_page_eviction_count,
            "page_evictions": self._resident_page_eviction_count + self._persistent_page_eviction_count,
            "copy_on_write_reused_pages": self._copy_on_write_reused_pages,
            "cache_invalidations": self._invalidation_count,
            "content_addressed": self._persistent_store is not None,
            "restart_resilient": self._persistent_store is not None,
        }

    def _resident_lookup_locked(self, *, model_id: str, prompt_tokens: tuple[int, ...]) -> PrefixCacheLookup | None:
        best_index, best_prefix_length, best_matched_pages = self._best_resident_match_locked(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
        )
        if best_index is None or best_prefix_length <= 0:
            return None
        entries = self._entries[model_id]
        entry = entries.pop(best_index)
        entries.append(entry)
        return PrefixCacheLookup(
            entry=entry,
            prefix_length=best_prefix_length,
            matched_page_count=best_matched_pages,
            resident_page_hits=best_matched_pages,
        )

    def _resident_preview_locked(
        self,
        *,
        model_id: str,
        prompt_tokens: tuple[int, ...],
    ) -> PrefixCachePreview | None:
        best_index, best_prefix_length, best_matched_pages = self._best_resident_match_locked(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
        )
        if best_index is None or best_prefix_length <= 0:
            return None
        entry = self._entries[model_id][best_index]
        return PrefixCachePreview(
            model_id=entry.model_id,
            prefix_tokens=entry.prefix_tokens,
            cache_key=entry.cache_key,
            source="resident",
            prefix_length=best_prefix_length,
            matched_page_count=best_matched_pages,
            resident_page_hits=best_matched_pages,
            page_size_tokens=entry.page_size_tokens,
            page_count=len(entry.page_keys),
        )

    def _persistent_lookup(self, *, model_id: str, prompt_tokens: tuple[int, ...]) -> PrefixCacheLookup | None:
        if self._persistent_store is None:
            return None
        stored = self._persistent_store.lookup(model_id=model_id, prompt_tokens=prompt_tokens)
        if stored is None or not stored.payload_persisted:
            return None
        prefix_length = longest_token_prefix(stored.prefix_tokens, prompt_tokens)
        if prefix_length <= 0:
            return None
        prompt_page_keys = self._page_keys_for(
            model_id=model_id,
            prefix_tokens=prompt_tokens,
            page_size_tokens=stored.page_size_tokens,
        )
        matched_pages = _matched_page_count(stored.page_keys, prompt_page_keys)
        return PrefixCacheLookup(
            entry=PrefixCacheEntry(
                model_id=stored.model_id,
                prefix_tokens=stored.prefix_tokens,
                payload=stored.payload,
                estimated_size_bytes=stored.estimated_size_bytes,
                cache_key=stored.cache_key,
                source="persisted",
                page_keys=stored.page_keys,
                page_size_tokens=stored.page_size_tokens,
                page_count=len(stored.page_keys),
                shared_page_count=matched_pages,
                new_page_count=0,
            ),
            prefix_length=prefix_length,
            matched_page_count=matched_pages,
            persisted_page_hits=matched_pages,
        )

    def _persistent_preview(
        self,
        *,
        model_id: str,
        prompt_tokens: tuple[int, ...],
    ) -> PrefixCachePreview | None:
        if self._persistent_store is None:
            return None
        stored = self._persistent_store.preview(model_id=model_id, prompt_tokens=prompt_tokens)
        if stored is None or not stored.payload_persisted:
            return None
        prefix_length = longest_token_prefix(stored.prefix_tokens, prompt_tokens)
        if prefix_length <= 0:
            return None
        prompt_page_keys = self._page_keys_for(
            model_id=model_id,
            prefix_tokens=prompt_tokens,
            page_size_tokens=stored.page_size_tokens,
        )
        matched_pages = _matched_page_count(stored.page_keys, prompt_page_keys)
        return PrefixCachePreview(
            model_id=stored.model_id,
            prefix_tokens=stored.prefix_tokens,
            cache_key=stored.cache_key,
            source="persisted",
            prefix_length=prefix_length,
            matched_page_count=matched_pages,
            persisted_page_hits=matched_pages,
            page_size_tokens=stored.page_size_tokens,
            page_count=len(stored.page_keys),
        )

    def _record_hit_locked(self, lookup: PrefixCacheLookup) -> None:
        self._hit_count += 1
        self._saved_prefill_tokens += lookup.prefix_length
        self._max_saved_prefill_tokens = max(self._max_saved_prefill_tokens, lookup.prefix_length)
        self._page_hit_count += lookup.matched_page_count
        self._resident_page_hit_count += lookup.resident_page_hits
        self._persistent_page_hit_count += lookup.persisted_page_hits
        self._copy_on_write_reused_pages += lookup.matched_page_count

    def _insert_resident_locked(self, entry: PrefixCacheEntry) -> _ResidentInsertSummary:
        entries = self._entries.setdefault(entry.model_id, [])
        pages = self._pages.setdefault(entry.model_id, {})
        existing_index = next(
            (
                index
                for index, resident in enumerate(entries)
                if resident.prefix_tokens == entry.prefix_tokens
            ),
            None,
        )
        existing_entry = entries[existing_index] if existing_index is not None else None
        existing_page_keys = set(existing_entry.page_keys) if existing_entry is not None else set()
        shared_page_count = self._shared_page_count_locked(model_id=entry.model_id, page_keys=entry.page_keys)
        new_page_count = 0
        for page_key in entry.page_keys:
            resident_page = pages.get(page_key)
            if resident_page is None:
                pages[page_key] = _ResidentPrefixCachePage(
                    page_key=page_key,
                    estimated_size_bytes=_estimate_page_size(page_key, entry.page_size_tokens),
                    ref_count=1,
                )
                new_page_count += 1
            elif page_key not in existing_page_keys:
                resident_page.ref_count += 1
        if existing_index is not None and existing_entry is not None:
            entries.pop(existing_index)
            removed_page_keys = [
                page_key
                for page_key in existing_entry.page_keys
                if page_key not in set(entry.page_keys)
            ]
            self._release_resident_page_keys_locked(model_id=entry.model_id, page_keys=removed_page_keys)
        entries.append(
            PrefixCacheEntry(
                model_id=entry.model_id,
                prefix_tokens=entry.prefix_tokens,
                payload=entry.payload,
                estimated_size_bytes=entry.estimated_size_bytes,
                cache_key=entry.cache_key,
                source="resident",
                page_keys=entry.page_keys,
                page_size_tokens=entry.page_size_tokens,
                page_count=len(entry.page_keys),
                shared_page_count=shared_page_count,
                new_page_count=new_page_count,
            ),
        )
        evicted_entry_count = 0
        evicted_page_count = 0
        while len(entries) > self._max_entries_per_model:
            evicted_entry = entries.pop(0)
            evicted_entry_count += 1
            evicted_page_count += self._release_resident_page_keys_locked(
                model_id=entry.model_id,
                page_keys=evicted_entry.page_keys,
            )
        return _ResidentInsertSummary(
            new_page_count=new_page_count,
            shared_page_count=shared_page_count,
            evicted_entry_count=evicted_entry_count,
            evicted_page_count=evicted_page_count,
        )

    def _release_resident_page_keys_locked(self, *, model_id: str, page_keys: tuple[str, ...] | list[str]) -> int:
        pages = self._pages.setdefault(model_id, {})
        removed = 0
        for page_key in page_keys:
            resident_page = pages.get(page_key)
            if resident_page is None:
                continue
            if resident_page.ref_count > 1:
                resident_page.ref_count -= 1
                continue
            pages.pop(page_key, None)
            removed += 1
        return removed

    def _shared_page_count_locked(self, *, model_id: str, page_keys: tuple[str, ...]) -> int:
        resident_pages = self._pages.get(model_id, {})
        shared = 0
        for page_key in page_keys:
            if page_key not in resident_pages:
                break
            shared += 1
        return shared

    def _best_resident_match_locked(
        self,
        *,
        model_id: str,
        prompt_tokens: tuple[int, ...],
    ) -> tuple[int | None, int, int]:
        prompt_page_keys = self._page_keys_for(model_id=model_id, prefix_tokens=prompt_tokens)
        best_index: int | None = None
        best_prefix_length = 0
        best_matched_pages = 0
        for index, entry in enumerate(self._entries.get(model_id, [])):
            prefix_length = longest_token_prefix(entry.prefix_tokens, prompt_tokens)
            if prefix_length <= 0:
                continue
            matched_pages = _matched_page_count(entry.page_keys, prompt_page_keys)
            if prefix_length > best_prefix_length or (
                prefix_length == best_prefix_length and matched_pages > best_matched_pages
            ):
                best_index = index
                best_prefix_length = prefix_length
                best_matched_pages = matched_pages
        return best_index, best_prefix_length, best_matched_pages

    def _page_keys_for(
        self,
        *,
        model_id: str,
        prefix_tokens: tuple[int, ...],
        page_size_tokens: int | None = None,
    ) -> tuple[str, ...]:
        normalized_page_size = max(int(page_size_tokens or self._page_size_tokens), 1)
        page_keys: list[str] = []
        parent_page_key: str | None = None
        for page_index, start in enumerate(range(0, len(prefix_tokens), normalized_page_size)):
            tokens = prefix_tokens[start : start + normalized_page_size]
            page_key = PersistentPrefixCacheStore.page_key_for(
                model_id=model_id,
                parent_page_key=parent_page_key,
                page_index=page_index,
                page_tokens=tokens,
                page_size_tokens=normalized_page_size,
            )
            page_keys.append(page_key)
            parent_page_key = page_key
        return tuple(page_keys)


def _matched_page_count(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    matched = 0
    for left_page_key, right_page_key in zip(left, right):
        if left_page_key != right_page_key:
            break
        matched += 1
    return matched


def _estimate_page_size(page_key: str, page_size_tokens: int) -> int:
    return max(len(page_key), 64) + page_size_tokens * 4
