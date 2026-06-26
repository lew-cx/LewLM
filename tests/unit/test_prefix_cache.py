from __future__ import annotations

from lewlm.runtime.prefix_cache import InMemoryTokenPrefixCache, longest_token_prefix
from lewlm.storage import PersistentPrefixCacheStore


def test_longest_token_prefix_counts_shared_prefix() -> None:
    assert longest_token_prefix([1, 2, 3, 9], [1, 2, 3, 4]) == 3
    assert longest_token_prefix([7, 8], [9, 10]) == 0


def test_in_memory_token_prefix_cache_tracks_hits_and_saved_tokens() -> None:
    cache = InMemoryTokenPrefixCache(page_size_tokens=2)

    assert cache.lookup(model_id="model", prompt_tokens=[1, 2, 3]) is None

    cache.save(model_id="model", prefix_tokens=[1, 2, 3, 4], payload={"state": "cached"}, estimated_size_bytes=16)
    lookup = cache.lookup(model_id="model", prompt_tokens=[1, 2, 3, 9])

    assert lookup is not None
    assert lookup.prefix_length == 3
    assert lookup.matched_page_count == 1
    snapshot = cache.snapshot()
    assert snapshot["cache_entries"] == 1
    assert snapshot["cache_hits"] == 1
    assert snapshot["cache_misses"] == 1
    assert snapshot["cache_saves"] == 1
    assert snapshot["saved_prefill_tokens"] == 3
    assert snapshot["max_saved_prefill_tokens"] == 3
    assert snapshot["cache_size_bytes"] == 16
    assert snapshot["page_size_tokens"] == 2
    assert snapshot["resident_page_count"] == 2
    assert snapshot["page_hits"] == 1
    assert snapshot["copy_on_write_reused_pages"] == 1


def test_in_memory_token_prefix_cache_reuses_shared_pages_copy_on_write() -> None:
    cache = InMemoryTokenPrefixCache(page_size_tokens=2)

    first_entry = cache.save(
        model_id="model",
        prefix_tokens=[1, 2, 3, 4],
        payload={"id": "first"},
        estimated_size_bytes=16,
    )
    second_entry = cache.save(
        model_id="model",
        prefix_tokens=[1, 2, 3, 9],
        payload={"id": "second"},
        estimated_size_bytes=16,
    )

    assert first_entry is not None
    assert second_entry is not None
    assert first_entry.page_count == 2
    assert second_entry.page_count == 2
    assert second_entry.shared_page_count == 1
    assert second_entry.new_page_count == 1
    snapshot = cache.snapshot()
    assert snapshot["cache_entries"] == 2
    assert snapshot["resident_page_count"] == 3
    assert snapshot["page_saves"] == 3
    assert snapshot["copy_on_write_reused_pages"] == 1


def test_in_memory_token_prefix_cache_preview_is_non_mutating() -> None:
    cache = InMemoryTokenPrefixCache(page_size_tokens=2)
    cache.save(model_id="model", prefix_tokens=[1, 2, 3, 4], payload={"id": "cached"}, estimated_size_bytes=16)

    preview = cache.preview(model_id="model", prompt_tokens=[1, 2, 3, 9])

    assert preview is not None
    assert preview.prefix_length == 3
    assert preview.matched_page_count == 1
    assert preview.uncached_token_count == 1
    snapshot = cache.snapshot()
    assert snapshot["cache_hits"] == 0
    assert snapshot["cache_misses"] == 0
    assert snapshot["saved_prefill_tokens"] == 0


def test_in_memory_token_prefix_cache_restores_persisted_entries(tmp_path) -> None:
    store = PersistentPrefixCacheStore(cache_root=tmp_path, namespace="prefix-cache-tests", page_size_tokens=2)
    cold_cache = InMemoryTokenPrefixCache(max_entries_per_model=1, persistent_store=store, page_size_tokens=2)

    cold_cache.save(
        model_id="model",
        prefix_tokens=[1, 2, 3, 4],
        payload={"state": ["cached"]},
        estimated_size_bytes=24,
    )

    warm_cache = InMemoryTokenPrefixCache(max_entries_per_model=1, persistent_store=store, page_size_tokens=2)
    lookup = warm_cache.lookup(model_id="model", prompt_tokens=[1, 2, 3, 9])

    assert lookup is not None
    assert lookup.entry.source == "persisted"
    assert lookup.prefix_length == 3
    assert lookup.matched_page_count == 1
    assert lookup.restored_page_count == 2
    snapshot = warm_cache.snapshot()
    assert snapshot["persistent_cache_hits"] == 1
    assert snapshot["cache_restores"] == 1
    assert snapshot["persisted_cache_entries"] == 1
    assert snapshot["persisted_page_count"] == 2
    assert snapshot["resident_page_count"] == 2
    assert snapshot["page_restores"] == 2
    assert snapshot["restart_resilient"] is True


def test_in_memory_token_prefix_cache_invalidates_resident_entries_only(tmp_path) -> None:
    store = PersistentPrefixCacheStore(cache_root=tmp_path, namespace="prefix-cache-invalidation", page_size_tokens=2)
    cache = InMemoryTokenPrefixCache(max_entries_per_model=1, persistent_store=store, page_size_tokens=2)
    cache.save(model_id="model", prefix_tokens=[1, 2, 3, 4], payload={"state": "cached"}, estimated_size_bytes=24)

    invalidated = cache.invalidate(model_id="model")

    assert invalidated == {
        "resident_entries": 1,
        "resident_pages": 2,
        "persisted_entries": 0,
        "persisted_pages": 0,
    }
    snapshot = cache.snapshot()
    assert snapshot["resident_cache_entries"] == 0
    assert snapshot["persisted_cache_entries"] == 1
    assert snapshot["cache_invalidations"] == 1


def test_in_memory_token_prefix_cache_tracks_resident_evictions(tmp_path) -> None:
    cache = InMemoryTokenPrefixCache(
        max_entries_per_model=1,
        page_size_tokens=2,
        persistent_store=PersistentPrefixCacheStore(
            cache_root=tmp_path,
            namespace="prefix-cache-evictions",
            page_size_tokens=2,
        ),
    )

    cache.save(model_id="model", prefix_tokens=[1, 2, 3], payload={"id": 1}, estimated_size_bytes=8)
    cache.save(model_id="model", prefix_tokens=[1, 2, 4], payload={"id": 2}, estimated_size_bytes=8)

    snapshot = cache.snapshot()
    assert snapshot["resident_cache_entries"] == 1
    assert snapshot["persisted_cache_entries"] == 2
    assert snapshot["cache_evictions"] == 1
    assert snapshot["resident_page_count"] == 2
    assert snapshot["persisted_page_count"] == 3
    assert snapshot["resident_page_evictions"] == 1
    assert snapshot["page_evictions"] == 1


def test_persistent_prefix_cache_keeps_deep_autotune_paths_windows_safe(tmp_path) -> None:
    deep_cache_root = (
        tmp_path
        / "state"
        / "autotune-candidates"
        / "8ba8496a2525"
        / "benchmark-probes"
        / ("d" * 32)
        / "cache"
    )
    store = PersistentPrefixCacheStore(
        cache_root=deep_cache_root,
        namespace="fake_mlx_semantic",
        page_size_tokens=2,
    )
    model_id = "qwen2.5-1.5b-instruct-mlx-ea3264c7089b-5320058eaf-with-extra-windows-path-pressure"

    result = store.put(
        model_id=model_id,
        prefix_tokens=(1, 2, 3, 4),
        payload={"state": "cached"},
        estimated_size_bytes=24,
    )

    page_path = store._page_path(model_id=model_id, page_key=result.entry.page_keys[0])
    assert len(page_path.parent.parent.name) <= 20
    assert len(page_path.stem) <= 20
    # Budget LewLM's path contribution (the deep autotune layout + cache page
    # path) against a realistic Windows data-dir root rather than pytest's deep
    # temp prefix, so the Windows MAX_PATH check is independent of the runner.
    realistic_windows_root_len = len(r"C:\Users\operator\.lewlm")
    contributed_len = len(str(page_path)) - len(str(tmp_path))
    assert realistic_windows_root_len + contributed_len < 260
    assert page_path.is_file()
