from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import ConversionStatus, ModelFormat, ModelModality, RuntimeAffinity
from lewlm.conversion.models import ConversionJobRequest, ConversionPolicy, JobStatus


def test_registry_scan_discovers_supported_model_layouts(
    temp_settings,
    sample_models_root: Path,
) -> None:
    services = bootstrap_services(temp_settings)

    summary = services.model_registry.scan()

    assert summary.discovered_count == 3
    assert summary.new_count == 3
    assert {manifest.format_type for manifest in summary.manifests} == {
        ModelFormat.GGUF,
        ModelFormat.HUGGINGFACE,
        ModelFormat.MLX,
    }
    assert any(manifest.conversion_status == ConversionStatus.RUNNABLE for manifest in summary.manifests)


def test_registry_scan_tracks_unchanged_models(
    temp_settings,
    sample_models_root: Path,
) -> None:
    services = bootstrap_services(temp_settings)

    first = services.model_registry.scan()
    second = services.model_registry.scan()

    assert first.discovered_count == 3
    assert second.discovered_count == 3
    assert second.unchanged_count == 3
    assert second.new_count == 0


def test_registry_scan_supports_encrypted_persistence(
    encrypted_persistence_settings,
    sample_models_root: Path,
) -> None:
    services = bootstrap_services(encrypted_persistence_settings)

    summary = services.model_registry.scan()

    assert summary.discovered_count == 3
    inventory = services.model_registry.inventory()
    assert inventory.count == 3

    first_source_path = summary.manifests[0].source_path
    first_display_name = summary.manifests[0].display_name
    with sqlite3.connect(encrypted_persistence_settings.database_path) as connection:
        row = connection.execute(
            """
            SELECT source_path, source_path_encrypted, display_name, manifest_json
            FROM model_manifests
            WHERE model_id = ?
            """,
            (summary.manifests[0].model_id,),
        ).fetchone()

    assert row is not None
    assert row[0] != first_source_path
    assert row[1].startswith("enc::v1::")
    assert row[2].startswith("enc::v1::")
    assert row[3].startswith("enc::v1::")
    assert services.metadata_store.get_model_manifest(summary.manifests[0].model_id) is not None


def test_registry_encryption_migrates_existing_plaintext_rows(
    temp_settings,
    encrypted_persistence_settings,
    sample_models_root: Path,
) -> None:
    plaintext_services = bootstrap_services(temp_settings)
    first = plaintext_services.model_registry.scan()

    encrypted_services = bootstrap_services(encrypted_persistence_settings)
    second = encrypted_services.model_registry.scan()

    assert first.discovered_count == 3
    assert second.discovered_count == 3
    assert second.unchanged_count == 3


def test_registry_scan_detects_embedding_rerank_and_audio_models(
    temp_settings,
    sample_multimodal_models_root: Path,
) -> None:
    services = bootstrap_services(temp_settings)

    summary = services.model_registry.scan()

    assert summary.discovered_count == 3
    modalities_by_name = {manifest.display_name: set(manifest.modality) for manifest in summary.manifests}
    assert modalities_by_name["e5-small-embed-mlx"] == {ModelModality.EMBEDDING}
    assert modalities_by_name["bge-reranker-base-mlx"] == {ModelModality.EMBEDDING, ModelModality.RERANK}
    assert modalities_by_name["whisper-mini-audio"] == {ModelModality.AUDIO}
    runtime_affinity_by_name = {manifest.display_name: manifest.runtime_affinity for manifest in summary.manifests}
    assert runtime_affinity_by_name["whisper-mini-audio"] == (RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_AUDIO)


def test_registry_inventory_prefers_converted_artifacts_after_conversion(
    services_with_fake_runtime_and_conversion,
) -> None:
    services = services_with_fake_runtime_and_conversion
    services.model_registry.scan()
    source_manifest = next(
        manifest
        for manifest in services.model_registry.list_manifests()
        if manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION
    )

    job = services.conversion_service.submit(
        ConversionJobRequest(model_id=source_manifest.model_id, policy=ConversionPolicy.BALANCED),
    )
    for _ in range(30):
        job = services.conversion_service.get_job(job.job_id)
        if job.status == JobStatus.COMPLETED:
            break
        time.sleep(0.05)

    assert job.status == JobStatus.COMPLETED
    inventory = services.model_registry.inventory()
    assert inventory.count == 3
    assert source_manifest.model_id not in {manifest.model_id for manifest in inventory.items}
    assert any(
        Path(manifest.source_path).is_relative_to(Path(job.payload["result_path"]))
        for manifest in inventory.items
    )


def test_registry_default_scan_includes_persisted_conversion_artifact_roots(
    temp_settings,
    services_with_fake_runtime_and_conversion,
) -> None:
    services = services_with_fake_runtime_and_conversion
    services.model_registry.scan()
    source_manifest = next(
        manifest
        for manifest in services.model_registry.list_manifests()
        if manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION
    )

    job = services.conversion_service.submit(
        ConversionJobRequest(model_id=source_manifest.model_id, policy=ConversionPolicy.BALANCED),
    )
    for _ in range(30):
        job = services.conversion_service.get_job(job.job_id)
        if job.status == JobStatus.COMPLETED:
            break
        time.sleep(0.05)

    assert job.status == JobStatus.COMPLETED
    reloaded_services = bootstrap_services(temp_settings)
    summary = reloaded_services.model_registry.scan()
    inventory = reloaded_services.model_registry.inventory()

    assert any(
        Path(manifest.source_path).is_relative_to(Path(job.payload["result_path"]))
        for manifest in summary.manifests
    )
    assert source_manifest.model_id not in {manifest.model_id for manifest in inventory.items}
    assert any(
        Path(manifest.source_path).is_relative_to(Path(job.payload["result_path"]))
        for manifest in inventory.items
    )


def test_registry_scan_updates_model_source_path_when_model_moves_between_roots(
    temp_settings,
) -> None:
    original_root = temp_settings.models_dir[0]
    original_root.mkdir(parents=True, exist_ok=True)
    original_model = original_root / "llama-3.2-3b-instruct-q4_k_m.gguf"
    original_model.write_bytes(b"gguf-model")

    initial_services = bootstrap_services(temp_settings)
    initial_summary = initial_services.model_registry.scan()

    moved_root = temp_settings.data_dir / "moved-models"
    moved_root.mkdir(parents=True, exist_ok=True)
    moved_model = moved_root / original_model.name
    shutil.copy2(original_model, moved_model)
    original_model.unlink()

    moved_settings = temp_settings.with_updates(models_dir=(moved_root,))
    moved_services = bootstrap_services(moved_settings)
    moved_summary = moved_services.model_registry.scan()

    assert initial_summary.discovered_count == 1
    assert moved_summary.discovered_count == 1
    inventory = moved_services.model_registry.inventory()
    assert inventory.count == 1
    assert inventory.items[0].source_path == str(moved_model)
