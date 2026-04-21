from __future__ import annotations

import sqlite3
from pathlib import Path

from lewlm.core.contracts import IdempotentOperationRecord
from lewlm.security.persistence import PersistenceEncryptor
from lewlm.storage.metadata import SCHEMA_VERSION, MetadataStore


def test_metadata_store_initializes_and_round_trips_values(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")

    store.initialize()
    store.set_value("health", {"ok": True})
    store.increment_counter("cache_hits")

    assert store.ping() is True
    assert store.get_schema_version() == SCHEMA_VERSION
    assert store.get_value("health") == {"ok": True}
    assert store.get_counter("cache_hits") == 1
    snapshot = store.snapshot()
    assert snapshot["model_count"] == 0
    assert snapshot["job_count"] == 0
    assert snapshot["conversion_artifact_count"] == 0
    assert snapshot["runtime_response_cache_count"] == 0
    assert snapshot["cache_block_count"] == 0
    assert snapshot["benchmark_record_count"] == 0
    assert snapshot["benchmark_artifact_count"] == 0
    assert snapshot["capability_probe_record_count"] == 0


def test_metadata_store_persists_benchmark_records(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()

    store.append_benchmark_record(
        {
            "benchmark_id": "bench-1",
            "model_id": "demo-model",
            "runtime": "fake_llamacpp",
            "reason": "Requested model selected.",
            "prompt": "Benchmark ping",
            "output_text": "Echo: Benchmark ping",
            "load_seconds": 0.01,
            "generate_seconds": 0.02,
            "total_seconds": 0.03,
            "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
            "completion_tokens_per_second": 150.0,
            "created_at": "2024-01-01T00:00:00+00:00",
        },
    )

    records = store.list_benchmark_records()

    assert store.benchmark_record_count() == 1
    assert records[0]["model_id"] == "demo-model"
    assert records[0]["runtime"] == "fake_llamacpp"


def test_metadata_store_persists_benchmark_artifacts(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()

    store.append_benchmark_artifact(
        {
            "artifact_id": "artifact-1",
            "artifact_path": str(tmp_path / "benchmarks" / "artifact-1.json"),
            "workload_signature": "chat-single-1m-r1-demo",
            "created_at": "2024-01-01T00:00:00+00:00",
            "capability": "chat",
            "benchmark_count": 1,
            "model_count": 1,
            "result": {"model_id": "demo-model", "total_seconds": 0.03},
            "scenarios": [],
            "regression": {"status": "no_baseline"},
        },
    )

    artifacts = store.list_benchmark_artifacts()
    latest = store.latest_benchmark_artifact(workload_signature="chat-single-1m-r1-demo")
    fetched = store.get_benchmark_artifact("artifact-1")

    assert store.benchmark_artifact_count() == 1
    assert artifacts[0]["artifact_id"] == "artifact-1"
    assert latest is not None
    assert latest["artifact_path"].endswith("artifact-1.json")
    assert fetched is not None
    assert fetched["artifact_id"] == "artifact-1"


def test_metadata_store_persists_capability_probe_records(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    host_platform = {
        "system": "Darwin",
        "release": "24.0.0",
        "machine": "arm64",
        "python_version": "3.12.0",
    }

    store.upsert_capability_probe_record(
        category="batching",
        probe_name="continuous_batching",
        host_platform=host_platform,
        status="supported",
        source="benchmark_scenario",
        reason="Observed native batching during the concurrent probe burst.",
        runtime_name="fake_llamacpp",
        runtime_affinity="llamacpp",
        model_id="demo-model",
        details={"benchmark_id": "bench-1"},
        recorded_at="2024-01-01T00:00:00+00:00",
    )

    records = store.list_capability_probe_records(host_platform=host_platform)

    assert store.capability_probe_record_count(host_platform=host_platform) == 1
    assert records[0]["category"] == "batching"
    assert records[0]["probe_name"] == "continuous_batching"
    assert records[0]["runtime_name"] == "fake_llamacpp"


def test_metadata_store_persists_serving_profiles(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    host_platform = {
        "system": "Darwin",
        "release": "24.0.0",
        "machine": "arm64",
        "python_version": "3.12.0",
    }

    store.upsert_serving_profile(
        model_id="demo-model",
        capability="chat",
        host_platform=host_platform,
        runtime_name="fake_mlx_text",
        workload_class="text_only",
        payload={
            "profile_id": "profile-1",
            "model_id": "demo-model",
            "capability": "chat",
            "workload_class": "text_only",
            "runtime": "fake_mlx_text",
            "host_platform": host_platform,
            "recommended_at": "2024-01-01T00:00:00+00:00",
            "reason": "Lowest latency.",
            "settings_overrides": {"runtime_policy": "keep_warm"},
            "effective_settings": {"runtime_policy": "keep_warm"},
            "metrics": {"total_seconds": 0.25},
            "candidate_summaries": [],
        },
    )

    profile = store.get_serving_profile(
        model_id="demo-model",
        capability="chat",
        host_platform=host_platform,
        runtime_name="fake_mlx_text",
        workload_class="text_only",
    )

    assert profile is not None
    assert profile["profile_id"] == "profile-1"
    assert profile["settings_overrides"]["runtime_policy"] == "keep_warm"
    assert store.list_serving_profiles(limit=1)[0]["model_id"] == "demo-model"


def test_metadata_store_serving_profiles_distinguish_runtime_and_workload(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    host_platform = {
        "system": "Darwin",
        "release": "24.0.0",
        "machine": "arm64",
        "python_version": "3.12.0",
    }

    store.upsert_serving_profile(
        model_id="demo-model",
        capability="chat",
        host_platform=host_platform,
        runtime_name="fake_mlx_text",
        workload_class="text_only",
        payload={"profile_id": "profile-text", "runtime": "fake_mlx_text", "workload_class": "text_only"},
    )
    store.upsert_serving_profile(
        model_id="demo-model",
        capability="chat",
        host_platform=host_platform,
        runtime_name="fake_mlx_vision",
        workload_class="single_image",
        payload={"profile_id": "profile-image", "runtime": "fake_mlx_vision", "workload_class": "single_image"},
    )

    assert (
        store.get_serving_profile(
            model_id="demo-model",
            capability="chat",
            host_platform=host_platform,
            runtime_name="fake_mlx_text",
            workload_class="text_only",
        )["profile_id"]
        == "profile-text"
    )
    assert (
        store.get_serving_profile(
            model_id="demo-model",
            capability="chat",
            host_platform=host_platform,
            runtime_name="fake_mlx_vision",
            workload_class="single_image",
        )["profile_id"]
        == "profile-image"
    )


def test_metadata_store_serving_profile_lookup_falls_back_to_legacy_text_key(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    host_platform = {
        "system": "Darwin",
        "release": "24.0.0",
        "machine": "arm64",
        "python_version": "3.12.0",
    }

    store.upsert_serving_profile(
        model_id="demo-model",
        capability="chat",
        host_platform=host_platform,
        payload={"profile_id": "legacy-profile", "runtime": "fake_mlx_text"},
    )

    profile = store.get_serving_profile(
        model_id="demo-model",
        capability="chat",
        host_platform=host_platform,
        runtime_name="fake_mlx_text",
        workload_class="text_only_multimodal",
    )

    assert profile is not None
    assert profile["profile_id"] == "legacy-profile"


def test_metadata_store_persists_runtime_preferences(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()
    host_platform = {
        "system": "Darwin",
        "release": "24.0.0",
        "machine": "arm64",
        "python_version": "3.12.0",
    }

    store.upsert_runtime_preference(
        model_id="demo-model",
        capability="chat",
        host_platform=host_platform,
        payload={
            "selected_runtime_affinity": "external_accelerator",
            "selected_runtime_name": "local_external_adapter",
            "baseline_runtime_affinity": "mlx_text",
            "baseline_runtime_name": "fake_mlx_semantic",
            "primary_metric": "warm_total_seconds",
            "selected_metric_value": 0.41,
            "baseline_metric_value": 0.73,
        },
    )

    preference = store.get_runtime_preference(
        model_id="demo-model",
        capability="chat",
        host_platform=host_platform,
    )

    assert preference is not None
    assert preference["selected_runtime_affinity"] == "external_accelerator"
    assert preference["primary_metric"] == "warm_total_seconds"
    assert store.list_runtime_preferences(limit=1)[0]["selected_runtime_name"] == "local_external_adapter"


def test_metadata_store_encrypts_persisted_values(encrypted_persistence_settings) -> None:
    encryptor = PersistenceEncryptor(encrypted_persistence_settings)
    store = MetadataStore(encrypted_persistence_settings.database_path, encryptor=encryptor)

    store.initialize()
    store.set_value("health", {"ok": True})

    with sqlite3.connect(encrypted_persistence_settings.database_path) as connection:
        row = connection.execute(
            "SELECT value FROM app_kv WHERE key = ?",
            ("health",),
        ).fetchone()

    assert row is not None
    assert row[0].startswith("enc::v1::")
    assert store.get_value("health") == {"ok": True}


def test_metadata_store_persists_runtime_response_cache_entries(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()

    store.upsert_runtime_response_cache_entry(
        cache_key="cache-1",
        capability="embeddings",
        model_id="embed-model",
        response_payload={
            "model_id": "embed-model",
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        },
    )

    record = store.get_runtime_response_cache_entry("cache-1")

    assert record is not None
    assert record["capability"] == "embeddings"
    assert record["model_id"] == "embed-model"
    assert record["response_payload"]["data"][0]["index"] == 0
    assert store.runtime_response_cache_count() == 1
    assert store.runtime_response_cache_size_bytes() > 0


def test_metadata_store_persists_cache_blocks(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()

    store.upsert_cache_block(
        cache_key="block-1",
        block_kind="multimodal_feature",
        storage_path="blocks/multimodal_feature/ab/block-1.json",
        size_bytes=128,
        metadata={"source_kind": "image", "input_bytes": 64},
    )

    record = store.get_cache_block("block-1")

    assert record is not None
    assert record["block_kind"] == "multimodal_feature"
    assert record["storage_path"] == "blocks/multimodal_feature/ab/block-1.json"
    assert record["size_bytes"] == 128
    assert record["metadata"]["source_kind"] == "image"
    assert store.cache_block_count() == 1
    assert store.cache_block_count(block_kind="multimodal_feature") == 1
    assert store.cache_block_size_bytes() == 128
    assert store.cache_block_size_bytes(block_kind="multimodal_feature") == 128


def test_metadata_store_persists_idempotent_operation_results(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    store.initialize()

    store.upsert_idempotent_operation_result(
        IdempotentOperationRecord(
            operation_name="documents.transform",
            idempotency_key="idem-1",
            request_hash="hash-1",
            response_payload={"request_id": "req-1", "tool": "documents.transform"},
        ),
    )

    record = store.get_idempotent_operation_result("documents.transform", "idem-1")

    assert record is not None
    assert record.operation_name == "documents.transform"
    assert record.idempotency_key == "idem-1"
    assert record.request_hash == "hash-1"
    assert record.response_payload["request_id"] == "req-1"
