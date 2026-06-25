"""Background model conversion and cache management."""

from __future__ import annotations

import errno
import hashlib
import json
import platform
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from uuid import uuid4

from lewlm.config.settings import LewLMSettings
from lewlm.conversion.backend import (
    AutoConversionBackend,
    ConversionBackend,
    LlamaCppConversionBackend,
    MLXConversionBackend,
    backend_descriptor,
    run_isolated_conversion,
)
from lewlm.conversion.models import (
    CONVERSION_OUTPUT_METADATA_FILENAME,
    ConversionOutputMetadata,
    ConversionArtifactRecord,
    ConversionCompatibilityReport,
    ConversionJobRequest,
    ConversionPolicy,
    ConversionTargetPlanningReport,
    JobRecord,
    JobStatus,
    JobType,
    LAYERED_CONVERSION_MANIFEST_FILENAME,
    LayeredConversionArtifact,
    LayeredConversionManifest,
    QUANTIZATION_PROFILE_METADATA_FILENAME,
    quantization_mode_from_profile,
    quantization_profile_cache_payload,
)
from lewlm.core.contracts import (
    ConversionStatus,
    ConversionTarget,
    IdempotentOperationRecord,
    ModelArtifactRole,
    ModelFormat,
    ModelManifest,
    ModelModality,
    RuntimeAffinity,
    RuntimeProvider,
    RuntimeSupportPath,
)
from lewlm.core.errors import ConversionError, IdempotencyConflictError, JobNotFoundError
from lewlm.events.bus import EventBus
from lewlm.events.schema import EventScope, EventType, StreamEvent
from lewlm.registry.discovery import PROCESSOR_FILENAMES, TOKENIZER_FILENAMES, discover_models
from lewlm.registry.service import ModelRegistry
from lewlm.security.audit import AuditLogger
from lewlm.security.persistence import ENCRYPTED_FILE_MAGIC
from lewlm.security.sandbox import run_in_subprocess
from lewlm.security.workspace import secure_workspace
from lewlm.storage.metadata import MetadataStore


class ConversionService:
    """Manage background conversion jobs and cached conversion artifacts."""

    _IDEMPOTENT_OPERATION_NAME = "models.convert"

    def __init__(
        self,
        *,
        settings: LewLMSettings,
        model_registry: ModelRegistry,
        metadata_store: MetadataStore,
        event_bus: EventBus,
        backend: ConversionBackend | None = None,
        audit_logger: AuditLogger,
    ) -> None:
        self.settings = settings
        self.model_registry = model_registry
        self.metadata_store = metadata_store
        self.event_bus = event_bus
        self.backend = backend or AutoConversionBackend()
        self.audit_logger = audit_logger
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, settings.conversion_worker_count),
            thread_name_prefix="lewlm-convert",
        )
        self._queued_jobs: set[str] = set()
        self._running_jobs: set[str] = set()
        self._lock = Lock()
        self._closed = False
        self._migrate_encrypted_cache_artifacts()

    def close(self) -> None:
        """Release worker resources owned by the conversion service."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)

    def clear_cache(self) -> dict[str, object]:
        """Remove persisted conversion artifacts and related conversion cache metadata."""

        cache_root = self.settings.cache_dir / "conversions"
        if cache_root.exists() and not cache_root.is_dir():
            raise ConversionError(
                "Conversion cache root is not a directory.",
                details={"path": str(cache_root)},
            )
        removed_entries = 0
        if cache_root.exists():
            for child in cache_root.iterdir():
                removed_entries += 1
                if child.is_symlink() or child.is_file():
                    child.unlink(missing_ok=True)
                else:
                    shutil.rmtree(child)
        cache_root.mkdir(parents=True, exist_ok=True)
        cleared_artifact_records = self.metadata_store.clear_conversion_artifacts()
        cleared_idempotent_records = self.metadata_store.clear_idempotent_operation_results(
            operation_name=self._IDEMPOTENT_OPERATION_NAME,
        )
        self.audit_logger.record(
            action="conversion_cache_clear",
            outcome="completed",
            actor="service",
            details={
                "cache_root": str(cache_root),
                "removed_entries": removed_entries,
                "cleared_artifact_records": cleared_artifact_records,
                "cleared_idempotent_records": cleared_idempotent_records,
            },
        )
        return {
            "cache_root": str(cache_root),
            "removed_entries": removed_entries,
            "cleared_artifact_records": cleared_artifact_records,
            "cleared_idempotent_records": cleared_idempotent_records,
        }

    def plan_targets(
        self,
        model_id: str,
        *,
        policy: ConversionPolicy = ConversionPolicy.BALANCED,
        custom_bits: int | None = None,
        quantization_profile=None,
    ) -> ConversionTargetPlanningReport:
        """Return non-executing conversion target options for one registered model."""

        manifest = self.model_registry.get_manifest(model_id)
        cache_key = self._build_cache_key(
            manifest.fingerprint,
            policy,
            custom_bits,
            quantization_profile,
        )
        output_path = self._output_path(cache_key)
        targets = [
            _target_from_compatibility_report(
                target_id=_target_id_for_backend(backend.name),
                report=backend.compatibility_report(
                    manifest,
                    settings=self.settings,
                    policy=policy,
                    custom_bits=custom_bits,
                    quantization_profile=quantization_profile,
                    cache_key=cache_key,
                    output_path=output_path,
                ),
            )
            for backend in _planning_backend_order(manifest)
        ]
        targets.append(_onnx_genai_planning_target(manifest))
        default_target_id = _default_conversion_target_id(manifest, targets)
        notes = [
            "This is a read-only target plan. It does not queue conversion work or materialize artifacts.",
            "Use `lewlm convert` without `--plan` or POST `/v1/lewlm/conversions` to queue an executable conversion.",
        ]
        if any(target.target_id == "onnx_genai" and target.state == "planned" for target in targets):
            notes.append(
                "ONNX Runtime GenAI is included as a planned Windows-native target, but HF-to-ONNX preparation is not executable yet.",
            )
        return ConversionTargetPlanningReport(
            model_id=manifest.model_id,
            source_format=manifest.format_type,
            conversion_status=manifest.conversion_status.value,
            default_target_id=default_target_id,
            targets=targets,
            notes=notes,
        )

    def submit(self, request: ConversionJobRequest) -> JobRecord:
        manifest = self.model_registry.get_manifest(request.model_id)
        cache_key = self._build_cache_key(
            manifest.fingerprint,
            request.policy,
            request.custom_bits,
            request.quantization_profile,
            target_id=request.target_id,
        )
        replayed_job = self._lookup_idempotent_job(request, cache_key=cache_key)
        if replayed_job is not None:
            self._promote_job_output(replayed_job)
            self.audit_logger.record(
                action="model_conversion",
                outcome="idempotent_replay",
                actor="service",
                details={
                    "job_id": replayed_job.job_id,
                    "model_id": request.model_id,
                    "idempotency_key": request.idempotency_key,
                },
            )
            return replayed_job
        output_path = self._output_path(cache_key)
        compatibility, conversion_backend = self._compatibility_for_request(
            manifest,
            policy=request.policy,
            custom_bits=request.custom_bits,
            quantization_profile=request.quantization_profile,
            cache_key=cache_key,
            output_path=output_path,
            target_id=request.target_id,
        )

        if not request.force:
            existing_job = self.metadata_store.find_job_by_cache_key(
                job_type=JobType.MODEL_CONVERSION,
                cache_key=cache_key,
                statuses=(JobStatus.QUEUED, JobStatus.RUNNING),
            )
            if existing_job is not None:
                return self._store_idempotent_job(request, cache_key=cache_key, job=existing_job)

            artifact = self.metadata_store.get_conversion_artifact(cache_key)
            if artifact is not None and self._is_cached_conversion_artifact(Path(artifact.output_path)):
                self._register_conversion_output(Path(artifact.output_path))
                self.metadata_store.increment_counter("cache_hits")
                job = self._build_job(
                    status=JobStatus.COMPLETED,
                    cache_key=cache_key,
                    payload={
                        "request": request.model_dump(mode="json"),
                        "compatibility": compatibility.model_dump(mode="json"),
                        "result_path": artifact.output_path,
                        "cache_hit": True,
                        "cache_encrypted": bool(artifact.metadata.get("cache_encrypted", False)),
                        "storage_mode": artifact.metadata.get("storage_mode", "directory"),
                        "sandboxed": bool(artifact.metadata.get("sandboxed", self.settings.conversion_sandbox_enabled)),
                        "artifacts": list(artifact.metadata.get("artifacts", [])),
                        "logs": ["Using cached conversion artifact."],
                    },
                )
                self.metadata_store.upsert_job(job)
                self.audit_logger.record(
                    action="model_conversion",
                    outcome="cache_hit",
                    actor="service",
                    details={"job_id": job.job_id, "model_id": request.model_id, "cache_key": cache_key},
                )
                return self._store_idempotent_job(request, cache_key=cache_key, job=job)
            recovered_artifact = self._recover_or_cleanup_orphaned_output(
                manifest=manifest,
                cache_key=cache_key,
                output_path=output_path,
                policy=request.policy,
                quantization_profile=compatibility.resolved_profile,
                compatibility=compatibility,
            )
            if recovered_artifact is not None:
                self._register_conversion_output(Path(recovered_artifact.output_path))
                self.metadata_store.increment_counter("cache_hits")
                job = self._build_job(
                    status=JobStatus.COMPLETED,
                    cache_key=cache_key,
                    payload={
                        "request": request.model_dump(mode="json"),
                        "compatibility": compatibility.model_dump(mode="json"),
                        "result_path": recovered_artifact.output_path,
                        "cache_hit": True,
                        "cache_encrypted": bool(recovered_artifact.metadata.get("cache_encrypted", False)),
                        "storage_mode": recovered_artifact.metadata.get("storage_mode", "directory"),
                        "sandboxed": bool(
                            recovered_artifact.metadata.get("sandboxed", self.settings.conversion_sandbox_enabled)
                        ),
                        "artifacts": list(recovered_artifact.metadata.get("artifacts", [])),
                        "logs": ["Recovered an existing conversion artifact from the cache path."],
                    },
                )
                self.metadata_store.upsert_job(job)
                self.audit_logger.record(
                    action="model_conversion",
                    outcome="cache_recovered",
                    actor="service",
                    details={"job_id": job.job_id, "model_id": request.model_id, "cache_key": cache_key},
                )
                return self._store_idempotent_job(request, cache_key=cache_key, job=job)

        if compatibility.already_runnable:
            job = self._build_job(
                status=JobStatus.COMPLETED,
                cache_key=cache_key,
                payload={
                    "request": request.model_dump(mode="json"),
                    "compatibility": compatibility.model_dump(mode="json"),
                    "result_path": compatibility.output_path,
                    "cache_hit": False,
                    "cache_encrypted": False,
                    "storage_mode": "external",
                    "sandboxed": False,
                    "logs": [compatibility.reason],
                },
            )
            self.metadata_store.upsert_job(job)
            self.audit_logger.record(
                action="model_conversion",
                outcome="already_runnable",
                actor="service",
                details={"job_id": job.job_id, "model_id": request.model_id, "cache_key": cache_key},
            )
            return self._store_idempotent_job(request, cache_key=cache_key, job=job)

        if not compatibility.can_convert:
            job = self._build_job(
                status=JobStatus.FAILED,
                cache_key=cache_key,
                payload={
                    "request": request.model_dump(mode="json"),
                    "compatibility": compatibility.model_dump(mode="json"),
                    "error": compatibility.reason,
                    "logs": [compatibility.reason],
                },
            )
            self.metadata_store.upsert_job(job)
            self.audit_logger.record(
                action="model_conversion",
                outcome="rejected",
                actor="service",
                details={
                    "job_id": job.job_id,
                    "model_id": request.model_id,
                    "cache_key": cache_key,
                    "reason": compatibility.reason,
                },
            )
            return self._store_idempotent_job(request, cache_key=cache_key, job=job)

        job = self._build_job(
            status=JobStatus.QUEUED,
            cache_key=cache_key,
            payload={
                "request": request.model_dump(mode="json"),
                "compatibility": compatibility.model_dump(mode="json"),
                "logs": ["Conversion job queued."],
            },
        )
        self.metadata_store.upsert_job(job)
        self.audit_logger.record(
            action="model_conversion",
            outcome="queued",
            actor="service",
            details={"job_id": job.job_id, "model_id": request.model_id, "cache_key": cache_key},
        )
        with self._lock:
            self._queued_jobs.add(job.job_id)
        job = self._store_idempotent_job(request, cache_key=cache_key, job=job)
        self._emit_event(
            EventType.REQUEST_QUEUED,
            {"job_id": job.job_id, "model_id": manifest.model_id, "cache_key": cache_key},
        )
        self.executor.submit(self._run_job, job.job_id, manifest.model_id, request, compatibility, conversion_backend)
        return job

    def get_job(self, job_id: str) -> JobRecord:
        job = self.metadata_store.get_job(job_id)
        if job is None:
            raise JobNotFoundError("Requested job was not found.", details={"job_id": job_id})
        return job

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._queued_jobs)

    def active_job_count(self) -> int:
        with self._lock:
            return len(self._queued_jobs) + len(self._running_jobs)

    def cache_stats(self) -> dict[str, object]:
        artifacts = self.metadata_store.list_conversion_artifacts()
        file_count = 0
        total_size_bytes = 0
        for artifact in artifacts:
            artifact_path = Path(artifact.output_path)
            if artifact_path.is_file():
                file_count += 1
                total_size_bytes += artifact_path.stat().st_size
            elif artifact_path.is_dir():
                for child in artifact_path.rglob("*"):
                    if child.is_file():
                        file_count += 1
                        total_size_bytes += child.stat().st_size
        return {
            "cache_dir": str(self.settings.cache_dir),
            "artifact_count": len(artifacts),
            "file_count": file_count,
            "total_size_bytes": total_size_bytes,
            "cache_hits": self.metadata_store.get_counter("cache_hits"),
            "cache_misses": self.metadata_store.get_counter("cache_misses"),
        }

    def _compatibility_for_request(
        self,
        manifest: ModelManifest,
        *,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile,
        cache_key: str,
        output_path: Path,
        target_id: str | None,
    ) -> tuple[ConversionCompatibilityReport, ConversionBackend]:
        backend = _backend_for_target_id(target_id) or self.backend
        if _normalize_target_id(target_id) == "onnx_genai":
            return (
                _planned_onnx_conversion_report(
                    manifest=manifest,
                    cache_key=cache_key,
                    output_path=output_path,
                ),
                backend,
            )
        if target_id is not None and _backend_for_target_id(target_id) is None:
            return (
                ConversionCompatibilityReport(
                    model_id=manifest.model_id,
                    source_format=manifest.format_type,
                    target_format=str(target_id),
                    backend_name="unknown",
                    can_convert=False,
                    reason=f"Unknown conversion target `{target_id}`. Run `lewlm convert {manifest.model_id} --plan` to list supported targets.",
                    cache_key=cache_key,
                    output_path=str(output_path),
                ),
                backend,
            )
        return (
            backend.compatibility_report(
                manifest,
                settings=self.settings,
                policy=policy,
                custom_bits=custom_bits,
                quantization_profile=quantization_profile,
                cache_key=cache_key,
                output_path=output_path,
            ),
            backend,
        )

    def _run_job(
        self,
        job_id: str,
        model_id: str,
        request: ConversionJobRequest,
        compatibility: ConversionCompatibilityReport,
        conversion_backend: ConversionBackend,
    ) -> None:
        started = time.perf_counter()
        manifest = self.model_registry.get_manifest(model_id)
        work_root = self.settings.temp_dir
        output_path = Path(compatibility.output_path)
        self._transition_job(job_id, JobStatus.RUNNING, log_message="Conversion worker started.")
        self._emit_event(
            EventType.TOOL_STARTED,
            {"job_id": job_id, "model_id": model_id, "tool": "model_conversion"},
        )
        sandboxed = self.settings.conversion_sandbox_enabled
        try:
            with secure_workspace(work_root, prefix="convert-") as temp_dir:
                temp_output = temp_dir / "conversion-output"
                result = self._run_conversion_backend(
                    manifest,
                    backend=conversion_backend,
                    policy=request.policy,
                    custom_bits=request.custom_bits,
                    quantization_profile=request.quantization_profile,
                    work_dir=temp_dir,
                    output_path=temp_output,
                )
                self._persist_conversion_metadata(
                    manifest=manifest,
                    compatibility=compatibility,
                    result=result,
                )
                storage_mode = self._write_output_path(result.output_path, output_path, force=request.force)
                materialized_artifacts = self._materialize_artifacts(
                    output_path=output_path,
                    compatibility=compatibility,
                    result=result,
                )
                artifact = ConversionArtifactRecord(
                    cache_key=compatibility.cache_key,
                    model_id=model_id,
                    output_path=str(output_path),
                    policy=request.policy,
                    metadata={
                        "backend_name": conversion_backend.name,
                        "source_path": manifest.source_path,
                        "quantization_mode": compatibility.quantization_mode,
                        "quantization_profile": (
                            compatibility.resolved_profile.model_dump(mode="json", exclude_none=True)
                            if compatibility.resolved_profile is not None
                            else None
                        ),
                        "cache_encrypted": storage_mode == "encrypted_archive",
                        "storage_mode": storage_mode,
                        "sandboxed": sandboxed,
                        "layered_output": compatibility.layered_output,
                        "artifacts": materialized_artifacts,
                    },
                )
                self.metadata_store.upsert_conversion_artifact(artifact)
                self._register_conversion_output(output_path)
                self.metadata_store.increment_counter("cache_misses")
                duration_seconds = round(time.perf_counter() - started, 4)
                self._complete_job(
                    job_id,
                    status=JobStatus.COMPLETED,
                    payload_updates={
                        "result_path": str(output_path),
                        "duration_seconds": duration_seconds,
                        "cache_encrypted": storage_mode == "encrypted_archive",
                        "storage_mode": storage_mode,
                        "sandboxed": sandboxed,
                        "artifacts": materialized_artifacts,
                        "logs": result.logs,
                    },
                )
                self._emit_event(
                    EventType.TOOL_FINISHED,
                    {"job_id": job_id, "model_id": model_id, "tool": "model_conversion", "status": "completed"},
                )
                self.audit_logger.record(
                    action="model_conversion",
                    outcome="completed",
                    actor="service",
                    details={
                        "job_id": job_id,
                        "model_id": model_id,
                        "cache_key": compatibility.cache_key,
                        "output_path": str(output_path),
                        "duration_seconds": duration_seconds,
                    },
                )
        except Exception as exc:
            duration_seconds = round(time.perf_counter() - started, 4)
            self._complete_job(
                job_id,
                status=JobStatus.FAILED,
                payload_updates={
                    "error": str(exc),
                    "duration_seconds": duration_seconds,
                },
            )
            self._emit_event(
                EventType.REQUEST_FAILED,
                {"job_id": job_id, "model_id": model_id, "error": str(exc)},
            )
            self.audit_logger.record(
                action="model_conversion",
                outcome="failed",
                actor="service",
                details={
                    "job_id": job_id,
                    "model_id": model_id,
                    "cache_key": compatibility.cache_key,
                    "error": str(exc),
                    "duration_seconds": duration_seconds,
                },
            )

    def _transition_job(self, job_id: str, status: JobStatus, *, log_message: str | None = None) -> None:
        job = self.get_job(job_id)
        payload = dict(job.payload)
        logs = list(payload.get("logs", []))
        if log_message is not None:
            logs.append(log_message)
        payload["logs"] = logs
        updated = job.model_copy(update={"status": status, "payload": payload})
        self.metadata_store.upsert_job(updated)
        with self._lock:
            self._queued_jobs.discard(job_id)
            if status == JobStatus.RUNNING:
                self._running_jobs.add(job_id)

    def _complete_job(self, job_id: str, *, status: JobStatus, payload_updates: dict[str, object]) -> None:
        job = self.get_job(job_id)
        payload = dict(job.payload)
        payload.update(payload_updates)
        updated = job.model_copy(update={"status": status, "payload": payload})
        self.metadata_store.upsert_job(updated)
        with self._lock:
            self._queued_jobs.discard(job_id)
            self._running_jobs.discard(job_id)

    def _register_conversion_output(self, output_path: Path) -> list[ModelManifest]:
        if not output_path.exists() or not output_path.is_dir():
            return []
        discovered = discover_models((output_path,))
        if not discovered:
            return []
        retained_manifests = [
            manifest
            for manifest in self.model_registry.list_manifests()
            if not self._is_under_root(Path(manifest.source_path), output_path)
        ]
        existing_sources = {source_path for source_path, _ in self.metadata_store.list_model_manifest_records()}
        discovered_sources = {manifest.source_path for manifest in discovered}
        stale_sources = [
            source_path
            for source_path in existing_sources
            if self._is_under_root(Path(source_path), output_path) and source_path not in discovered_sources
        ]
        self.metadata_store.replace_model_manifests(
            [*retained_manifests, *discovered],
            stale_source_paths=stale_sources,
        )
        return discovered

    def _promote_job_output(self, job: JobRecord) -> None:
        if job.status != JobStatus.COMPLETED:
            return
        result_path = job.payload.get("result_path")
        if not isinstance(result_path, str):
            return
        self._register_conversion_output(Path(result_path))

    def _write_output_path(self, source_path: Path, target_path: Path, *, force: bool) -> str:
        if target_path.exists():
            if not force:
                raise ConversionError(
                    "Target conversion cache path already exists.",
                    details={"output_path": str(target_path)},
                )
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        encryptor = self.metadata_store.encryptor
        try:
            if encryptor is not None:
                with secure_workspace(self.settings.temp_dir, prefix="cache-archive-") as temp_dir:
                    archive_path = temp_dir / "artifact.tar"
                    self._archive_output(source_path, archive_path)
                    encryptor.encrypt_file(archive_path, target_path)
                return "encrypted_archive"
            if source_path.is_dir():
                shutil.copytree(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
            return "directory"
        except Exception as exc:
            self._cleanup_failed_output_path(target_path)
            if self._is_no_space_error(exc):
                available_bytes = shutil.disk_usage(target_path.parent).free
                raise ConversionError(
                    "Insufficient disk space while writing the conversion artifact.",
                    details={
                        "output_path": str(target_path),
                        "available_bytes": available_bytes,
                    },
                ) from exc
            raise

    def _recover_or_cleanup_orphaned_output(
        self,
        *,
        manifest: ModelManifest,
        cache_key: str,
        output_path: Path,
        policy: ConversionPolicy,
        quantization_profile,
        compatibility: ConversionCompatibilityReport,
    ) -> ConversionArtifactRecord | None:
        if not output_path.exists():
            return None
        if self._is_cached_conversion_artifact(output_path):
            self._persist_quantization_profile_metadata(output_path, quantization_profile)
            if compatibility.layered_output:
                for artifact in compatibility.artifact_plans:
                    artifact_output_path = (
                        output_path
                        if artifact.relative_path in {"", ".", "./"}
                        else output_path / artifact.relative_path
                    )
                    self._persist_quantization_profile_metadata(artifact_output_path, quantization_profile)
                self._write_layered_conversion_manifest(
                    root_output_path=output_path,
                    source_manifest=manifest,
                    compatibility=compatibility,
                )
            else:
                self._write_conversion_output_metadata(
                    output_path=output_path,
                    source_manifest=manifest,
                    compatibility=compatibility,
                )
            materialized_artifacts = self._materialize_artifacts(
                output_path=output_path,
                compatibility=compatibility,
            )
            storage_mode = "encrypted_archive" if self._is_encrypted_conversion_archive(output_path) else ("directory" if output_path.is_dir() else "file")
            artifact = ConversionArtifactRecord(
                cache_key=cache_key,
                model_id=manifest.model_id,
                output_path=str(output_path),
                policy=policy,
                metadata={
                    "backend_name": compatibility.backend_name,
                    "source_path": manifest.source_path,
                    "quantization_mode": quantization_mode_from_profile(quantization_profile),
                    "quantization_profile": (
                        quantization_profile.model_dump(mode="json", exclude_none=True)
                        if quantization_profile is not None
                        else None
                    ),
                    "cache_encrypted": storage_mode == "encrypted_archive",
                    "storage_mode": storage_mode,
                    "sandboxed": self.settings.conversion_sandbox_enabled,
                    "layered_output": compatibility.layered_output,
                    "artifacts": materialized_artifacts,
                },
            )
            self.metadata_store.upsert_conversion_artifact(artifact)
            return artifact
        self._cleanup_failed_output_path(output_path)
        self.audit_logger.record(
            action="model_conversion",
            outcome="stale_cache_cleanup",
            actor="service",
            details={"model_id": manifest.model_id, "cache_key": cache_key, "output_path": str(output_path)},
        )
        return None

    @staticmethod
    def _cleanup_failed_output_path(target_path: Path) -> None:
        if not target_path.exists():
            return
        if target_path.is_dir():
            shutil.rmtree(target_path, ignore_errors=True)
            return
        target_path.unlink(missing_ok=True)

    @staticmethod
    def _is_no_space_error(error: Exception) -> bool:
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            return True
        if isinstance(error, shutil.Error):
            return any("No space left on device" in str(item[-1]) for item in error.args[0] if item)
        return "No space left on device" in str(error)

    @staticmethod
    def _is_runnable_conversion_output(output_path: Path) -> bool:
        if not output_path.exists() or not output_path.is_dir():
            return False
        discovered = discover_models((output_path,))
        return any(manifest.conversion_status == ConversionStatus.RUNNABLE for manifest in discovered)

    @staticmethod
    def _is_encrypted_conversion_archive(output_path: Path) -> bool:
        if not output_path.exists() or not output_path.is_file():
            return False
        try:
            with output_path.open("rb") as handle:
                return handle.read(len(ENCRYPTED_FILE_MAGIC)) == ENCRYPTED_FILE_MAGIC
        except OSError:
            return False

    @classmethod
    def _is_cached_conversion_artifact(cls, output_path: Path) -> bool:
        return cls._is_runnable_conversion_output(output_path) or cls._is_encrypted_conversion_archive(output_path)

    def _run_conversion_backend(
        self,
        manifest,
        *,
        backend: ConversionBackend | None = None,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile,
        work_dir: Path,
        output_path: Path,
    ):
        selected_backend = backend or self.backend
        if not self.settings.conversion_sandbox_enabled:
            return selected_backend.convert(
                manifest,
                settings=self.settings,
                policy=policy,
                custom_bits=custom_bits,
                quantization_profile=quantization_profile,
                output_path=output_path,
                work_dir=work_dir,
            )
        backend_module, backend_qualname = backend_descriptor(selected_backend)
        return run_in_subprocess(
            run_isolated_conversion,
            operation="Conversion sandbox worker",
            timeout_seconds=self.settings.conversion_sandbox_timeout_seconds,
            enabled=True,
            clear_environment=self.settings.conversion_sandbox_clear_environment,
            workspace_root=self.settings.conversion_sandbox_dir,
            backend_module=backend_module,
            backend_qualname=backend_qualname,
            manifest_payload=manifest.model_dump(mode="python"),
            settings_payload=self.settings.model_dump(mode="python", exclude_computed_fields=True),
            policy=policy.value,
            custom_bits=custom_bits,
            quantization_profile_payload=(
                quantization_profile.model_dump(mode="python", exclude_none=True)
                if quantization_profile is not None
                else None
            ),
            output_path=str(output_path),
            work_dir=str(work_dir),
        )

    @staticmethod
    def _persist_quantization_profile_metadata(output_path: Path, quantization_profile) -> None:
        if quantization_profile is None or not output_path.is_dir():
            return
        (output_path / QUANTIZATION_PROFILE_METADATA_FILENAME).write_text(
            json.dumps(
                quantization_profile.model_dump(mode="json", exclude_none=True),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _persist_conversion_metadata(
        self,
        *,
        manifest: ModelManifest,
        compatibility: ConversionCompatibilityReport,
        result,
    ) -> None:
        artifact_paths = [artifact.output_path for artifact in getattr(result, "artifacts", ())] or [result.output_path]
        for artifact_path in artifact_paths:
            self._persist_quantization_profile_metadata(artifact_path, compatibility.resolved_profile)
        if compatibility.layered_output:
            self._write_layered_conversion_manifest(
                root_output_path=result.output_path,
                source_manifest=manifest,
                compatibility=compatibility,
            )
            return
        self._write_conversion_output_metadata(
            output_path=result.output_path,
            source_manifest=manifest,
            compatibility=compatibility,
        )

    def _write_layered_conversion_manifest(
        self,
        *,
        root_output_path: Path,
        source_manifest: ModelManifest,
        compatibility: ConversionCompatibilityReport,
    ) -> None:
        if not root_output_path.is_dir():
            return
        layered_manifest = LayeredConversionManifest(
            artifact_family_id=compatibility.cache_key,
            display_name=source_manifest.display_name,
            source_path=source_manifest.source_path,
            source_format=source_manifest.format_type,
            source_modality=source_manifest.modality,
            source_runtime_affinity=source_manifest.runtime_affinity,
            source_tokenizer_path=source_manifest.tokenizer_path,
            source_processor_path=source_manifest.processor_path,
            source_quantization=source_manifest.quantization,
            source_quantization_profile=source_manifest.quantization_profile,
            artifacts=[
                plan.model_copy(
                    update={
                        "tokenizer_path": self._relative_to_root(
                            root_output_path,
                            self._resolve_optional_artifact_file(
                                root_output_path if plan.relative_path in {"", ".", "./"} else root_output_path / plan.relative_path,
                                TOKENIZER_FILENAMES,
                            ),
                        ),
                        "processor_path": self._relative_to_root(
                            root_output_path,
                            self._resolve_optional_artifact_file(
                                root_output_path if plan.relative_path in {"", ".", "./"} else root_output_path / plan.relative_path,
                                PROCESSOR_FILENAMES,
                            ),
                        ),
                        "quantization": plan.quantization or compatibility.quantization_mode or source_manifest.quantization,
                        "quantization_profile": plan.quantization_profile or compatibility.resolved_profile,
                    },
                )
                for plan in compatibility.artifact_plans
            ],
            metadata={
                "backend_name": compatibility.backend_name,
                "layered_output": compatibility.layered_output,
            },
        )
        (root_output_path / LAYERED_CONVERSION_MANIFEST_FILENAME).write_text(
            json.dumps(layered_manifest.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_conversion_output_metadata(
        self,
        *,
        output_path: Path,
        source_manifest: ModelManifest,
        compatibility: ConversionCompatibilityReport,
    ) -> None:
        if not output_path.is_dir():
            return
        artifact_metadata = (
            dict(compatibility.artifact_plans[0].metadata)
            if len(compatibility.artifact_plans) == 1
            else {}
        )
        metadata = ConversionOutputMetadata(
            source_display_name=source_manifest.display_name,
            source_model_id=source_manifest.model_id,
            display_name=f"{source_manifest.display_name} (converted)",
            artifact_role=ModelArtifactRole.STANDALONE,
            artifact_family_id=compatibility.cache_key,
            metadata={
                **artifact_metadata,
                "backend_name": compatibility.backend_name,
                "layered_output": compatibility.layered_output,
                "quantization_mode": compatibility.quantization_mode,
                "target_format": compatibility.target_format,
            },
        )
        (output_path / CONVERSION_OUTPUT_METADATA_FILENAME).write_text(
            json.dumps(metadata.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _materialize_artifacts(
        self,
        *,
        output_path: Path,
        compatibility: ConversionCompatibilityReport,
        result=None,
    ) -> list[dict[str, object]]:
        descriptors = {artifact.artifact_key: artifact for artifact in getattr(result, "artifacts", ())}
        materialized: list[dict[str, object]] = []
        for plan in compatibility.artifact_plans:
            descriptor = descriptors.get(plan.artifact_key)
            relative_path = plan.relative_path
            if descriptor is not None and result is not None:
                relative_path = self._relative_artifact_path(result.output_path, descriptor.output_path)
            artifact_output_path = output_path if relative_path in {"", ".", "./"} else output_path / relative_path
            tokenizer_path = self._resolve_optional_artifact_file(artifact_output_path, TOKENIZER_FILENAMES)
            processor_path = self._resolve_optional_artifact_file(artifact_output_path, PROCESSOR_FILENAMES)
            materialized.append(
                {
                    "artifact_key": plan.artifact_key,
                    "role": plan.role.value,
                    "display_name": plan.display_name,
                    "relative_path": relative_path,
                    "output_path": str(artifact_output_path),
                    "format_type": plan.format_type.value,
                    "modality": [modality.value for modality in plan.modality],
                    "runtime_affinity": [affinity.value for affinity in plan.runtime_affinity],
                    "derived_from": plan.derived_from,
                    "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
                    "processor_path": str(processor_path) if processor_path is not None else None,
                    "quantization": plan.quantization or compatibility.quantization_mode,
                    "quantization_profile": (
                        (plan.quantization_profile or compatibility.resolved_profile).model_dump(mode="json", exclude_none=True)
                        if (plan.quantization_profile or compatibility.resolved_profile) is not None
                        else None
                    ),
                },
            )
        return materialized

    @staticmethod
    def _resolve_optional_artifact_file(path: Path, candidates: tuple[str, ...]) -> Path | None:
        for candidate_name in candidates:
            candidate = path / candidate_name
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _relative_to_root(root_output_path: Path, path: Path | None) -> str | None:
        if path is None:
            return None
        return str(path.relative_to(root_output_path))

    @staticmethod
    def _relative_artifact_path(root_output_path: Path, artifact_output_path: Path) -> str:
        if artifact_output_path == root_output_path:
            return "."
        return str(artifact_output_path.relative_to(root_output_path))

    @staticmethod
    def _is_under_root(source_path: Path, root: Path) -> bool:
        try:
            source_path.relative_to(root)
            return True
        except ValueError:
            return False

    def _archive_output(self, source_path: Path, archive_path: Path) -> None:
        if source_path.is_dir():
            with tarfile.open(archive_path, "w") as archive:
                for child in sorted(source_path.iterdir()):
                    archive.add(child, arcname=child.name)
            return
        with tarfile.open(archive_path, "w") as archive:
            archive.add(source_path, arcname=source_path.name)

    def _migrate_encrypted_cache_artifacts(self) -> None:
        encryptor = self.metadata_store.encryptor
        if encryptor is None:
            return
        artifacts = self.metadata_store.list_conversion_artifacts()
        for artifact in artifacts:
            storage_mode = str(artifact.metadata.get("storage_mode", "directory"))
            if storage_mode == "encrypted_archive" and Path(artifact.output_path).is_file():
                continue
            source_path = Path(artifact.output_path)
            if not source_path.exists():
                continue
            target_path = self._output_path(artifact.cache_key)
            if source_path == target_path and source_path.is_file():
                with source_path.open("rb") as handle:
                    if handle.read(len(ENCRYPTED_FILE_MAGIC)) != ENCRYPTED_FILE_MAGIC:
                        continue
                updated_metadata = dict(artifact.metadata)
                updated_metadata["cache_encrypted"] = True
                updated_metadata["storage_mode"] = "encrypted_archive"
                self.metadata_store.upsert_conversion_artifact(
                    artifact.model_copy(update={"metadata": updated_metadata}),
                )
                continue
            if source_path == target_path:
                continue
            with secure_workspace(self.settings.temp_dir, prefix="cache-migrate-") as temp_dir:
                archive_path = temp_dir / "artifact.tar"
                self._archive_output(source_path, archive_path)
                encryptor.encrypt_file(archive_path, target_path)
            if source_path.is_dir():
                shutil.rmtree(source_path)
            else:
                source_path.unlink()
            updated_metadata = dict(artifact.metadata)
            updated_metadata["cache_encrypted"] = True
            updated_metadata["storage_mode"] = "encrypted_archive"
            self.metadata_store.upsert_conversion_artifact(
                artifact.model_copy(update={"output_path": str(target_path), "metadata": updated_metadata}),
            )

    def _build_job(self, *, status: JobStatus, cache_key: str | None, payload: dict[str, object]) -> JobRecord:
        return JobRecord(
            job_id=str(uuid4()),
            job_type=JobType.MODEL_CONVERSION,
            status=status,
            cache_key=cache_key,
            payload=payload,
        )

    def _lookup_idempotent_job(self, request: ConversionJobRequest, *, cache_key: str) -> JobRecord | None:
        if request.idempotency_key is None:
            return None
        record = self.metadata_store.get_idempotent_operation_result(
            self._IDEMPOTENT_OPERATION_NAME,
            request.idempotency_key,
        )
        if record is None:
            return None
        request_hash = self._request_hash(request, cache_key=cache_key)
        if record.request_hash != request_hash:
            raise IdempotencyConflictError(
                "The supplied idempotency key has already been used for a different request payload.",
                details={
                    "operation_name": self._IDEMPOTENT_OPERATION_NAME,
                    "idempotency_key": request.idempotency_key,
                    "fallback_guidance": [
                        "Reuse the same request body when retrying an idempotent conversion.",
                        "Choose a new idempotency key for a materially different conversion request.",
                    ],
                },
            )
        response_payload = record.response_payload
        stored_job_id = response_payload.get("job_id")
        if isinstance(stored_job_id, str):
            current_job = self.metadata_store.get_job(stored_job_id)
            if current_job is not None:
                return current_job.model_copy(
                    update={"idempotency_key": request.idempotency_key, "idempotent_replay": True},
                )
        return JobRecord.model_validate(response_payload).model_copy(
            update={"idempotency_key": request.idempotency_key, "idempotent_replay": True},
        )

    def _store_idempotent_job(self, request: ConversionJobRequest, *, cache_key: str, job: JobRecord) -> JobRecord:
        if request.idempotency_key is None:
            return job
        persisted_job = job.model_copy(update={"idempotency_key": request.idempotency_key, "idempotent_replay": False})
        self.metadata_store.upsert_idempotent_operation_result(
            IdempotentOperationRecord(
                operation_name=self._IDEMPOTENT_OPERATION_NAME,
                idempotency_key=request.idempotency_key,
                request_hash=self._request_hash(request, cache_key=cache_key),
                response_payload=persisted_job.model_dump(mode="json"),
            ),
        )
        return persisted_job

    def _request_hash(self, request: ConversionJobRequest, *, cache_key: str) -> str:
        payload = request.model_dump(mode="json")
        authorizations = payload.get("authorized_actions")
        if isinstance(authorizations, list):
            payload["authorized_actions"] = sorted({str(item) for item in authorizations})
        serialized = json.dumps(
            {"operation_name": self._IDEMPOTENT_OPERATION_NAME, "cache_key": cache_key, **payload},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _build_cache_key(
        self,
        fingerprint: str,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile,
        target_id: str | None = None,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(fingerprint.encode("utf-8"))
        digest.update(policy.value.encode("utf-8"))
        digest.update(str(target_id or "").encode("utf-8"))
        digest.update(str(custom_bits or "").encode("utf-8"))
        digest.update(quantization_profile_cache_payload(quantization_profile).encode("utf-8"))
        return digest.hexdigest()

    def _output_path(self, cache_key: str) -> Path:
        if self.metadata_store.encryptor is not None:
            return self.settings.cache_dir / "conversions" / f"{cache_key}.lewlmcache"
        return self.settings.cache_dir / "conversions" / cache_key

    def _emit_event(self, event_type: EventType, payload: dict[str, object]) -> None:
        self.event_bus.publish_threadsafe(StreamEvent(type=event_type, scope=EventScope.JOB, payload=payload))


def _planning_backend_order(manifest: ModelManifest) -> tuple[ConversionBackend, ...]:
    gguf_backend = LlamaCppConversionBackend()
    mlx_backend = MLXConversionBackend()
    if manifest.format_type == ModelFormat.MLX:
        return (mlx_backend, gguf_backend)
    if platform.system() == "Darwin" and platform.machine().casefold() == "arm64":
        return (mlx_backend, gguf_backend)
    return (gguf_backend, mlx_backend)


def _backend_for_target_id(target_id: str | None) -> ConversionBackend | None:
    normalized = _normalize_target_id(target_id)
    if normalized is None:
        return None
    if normalized in {"gguf_llamacpp", "llamacpp", "llamacpp_gguf", "gguf"}:
        return LlamaCppConversionBackend()
    if normalized in {"mlx", "mlx_lm", "mlx_vlm"}:
        return MLXConversionBackend()
    return None


def _normalize_target_id(target_id: str | None) -> str | None:
    if target_id is None:
        return None
    normalized = target_id.strip().casefold().replace("-", "_")
    return normalized or None


def _planned_onnx_conversion_report(
    *,
    manifest: ModelManifest,
    cache_key: str,
    output_path: Path,
) -> ConversionCompatibilityReport:
    target = _onnx_genai_planning_target(manifest)
    return ConversionCompatibilityReport(
        model_id=manifest.model_id,
        source_format=manifest.format_type,
        target_format=ModelFormat.ONNX_GENAI.value,
        backend_name="onnx_genai_planned",
        can_convert=False,
        already_runnable=target.already_runnable,
        reason=target.reason or "ONNX Runtime GenAI conversion is not executable in this build.",
        cache_key=cache_key,
        output_path=str(output_path if not target.already_runnable else manifest.source_path),
        artifact_plans=[],
        warnings=[
            "ONNX Runtime GenAI is currently a planned target. Run `lewlm convert --plan` to inspect target status.",
        ],
    )


def _target_from_compatibility_report(
    *,
    target_id: str,
    report: ConversionCompatibilityReport,
) -> ConversionTarget:
    provider = _provider_for_conversion_backend(report.backend_name)
    runtime_affinity = _runtime_affinity_for_conversion_report(report)
    return ConversionTarget(
        target_id=target_id,
        target_format=report.target_format,
        runtime_provider=provider,
        runtime_affinity=runtime_affinity,
        backend_name=report.backend_name,
        state=_conversion_target_state(report),
        can_convert=report.can_convert,
        already_runnable=report.already_runnable,
        reason=report.reason,
        support_path=RuntimeSupportPath.PACKAGED,
        optimization_profiles=_optimization_profiles_for_target(target_id),
        artifact_plans=[artifact.model_dump(mode="json") for artifact in report.artifact_plans],
        notes=[*report.warnings],
    )


def _onnx_genai_planning_target(manifest: ModelManifest) -> ConversionTarget:
    text_like = any(
        modality in manifest.modality
        for modality in (ModelModality.TEXT, ModelModality.EMBEDDING, ModelModality.RERANK, ModelModality.MULTIMODAL)
    )
    if manifest.format_type == ModelFormat.ONNX_GENAI:
        state = "already_runnable"
        reason = "Model is already an ONNX Runtime GenAI bundle; runtime load/generate evidence is still probe-gated."
        already_runnable = True
    elif manifest.format_type == ModelFormat.HUGGINGFACE and text_like:
        state = "planned"
        reason = (
            "HF-to-ONNX Runtime GenAI preparation is planned for Windows-native CPU, DirectML, and CUDA paths, "
            "but this build does not include an executable ONNX conversion adapter yet."
        )
        already_runnable = False
    else:
        state = "unsupported"
        reason = "ONNX Runtime GenAI planning currently targets ONNX bundles or text-like Hugging Face sources."
        already_runnable = False
    return ConversionTarget(
        target_id="onnx_genai",
        target_format=ModelFormat.ONNX_GENAI.value,
        runtime_provider=RuntimeProvider.ONNX_GENAI,
        runtime_affinity=RuntimeAffinity.ONNX_GENAI,
        backend_name="onnx_genai_planned",
        state=state,
        can_convert=False,
        already_runnable=already_runnable,
        reason=reason,
        support_path=RuntimeSupportPath.PACKAGED,
        optimization_profiles=["fp16", "int8", "int4", "directml_auto"],
        artifact_plans=[
            {
                "artifact_key": "onnx_genai",
                "role": ModelArtifactRole.STANDALONE.value,
                "display_name": f"{manifest.display_name} (ONNX GenAI)",
                "relative_path": ".",
                "format_type": ModelFormat.ONNX_GENAI.value,
                "modality": [modality.value for modality in manifest.modality],
                "runtime_affinity": [RuntimeAffinity.ONNX_GENAI.value],
                "metadata": {"target_family": "onnxruntime_genai"},
            },
        ],
        notes=[
            "Listed so operators can see the Windows-native target direction without treating it as executable conversion support.",
            "Execution requires future load/generate smoke evidence from the ONNX Runtime GenAI adapter.",
        ],
    )


def _default_conversion_target_id(manifest: ModelManifest, targets: list[ConversionTarget]) -> str | None:
    if manifest.format_type == ModelFormat.GGUF:
        return "gguf_llamacpp"
    if manifest.format_type == ModelFormat.MLX:
        return "mlx"
    if manifest.format_type == ModelFormat.ONNX_GENAI:
        return "onnx_genai"
    for target in targets:
        if target.can_convert:
            return target.target_id
    for target in targets:
        if target.already_runnable:
            return target.target_id
    return None


def _target_id_for_backend(backend_name: str) -> str:
    if backend_name == "llamacpp_gguf":
        return "gguf_llamacpp"
    if backend_name in {"mlx_lm", "mlx_vlm"}:
        return "mlx"
    return backend_name


def _provider_for_conversion_backend(backend_name: str) -> RuntimeProvider:
    if backend_name == "llamacpp_gguf":
        return RuntimeProvider.LLAMACPP
    if backend_name in {"mlx_lm", "mlx_vlm"}:
        return RuntimeProvider.MLX
    if "onnx" in backend_name:
        return RuntimeProvider.ONNX_GENAI
    return RuntimeProvider.UNKNOWN


def _runtime_affinity_for_conversion_report(report: ConversionCompatibilityReport) -> RuntimeAffinity | None:
    for artifact in report.artifact_plans:
        if artifact.runtime_affinity:
            return artifact.runtime_affinity[0]
    if report.backend_name == "llamacpp_gguf":
        return RuntimeAffinity.LLAMACPP
    if report.backend_name == "mlx_vlm":
        return RuntimeAffinity.MLX_VISION
    if report.backend_name == "mlx_lm":
        return RuntimeAffinity.MLX_TEXT
    return None


def _conversion_target_state(report: ConversionCompatibilityReport) -> str:
    if report.already_runnable:
        return "already_runnable"
    if report.can_convert:
        return "available"
    reason = report.reason.casefold()
    if "missing" in reason or "not found" in reason or "install" in reason or "unavailable" in reason:
        return "requires_install"
    return "unsupported"


def _optimization_profiles_for_target(target_id: str) -> list[str]:
    if target_id == "gguf_llamacpp":
        return ["q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"]
    if target_id == "mlx":
        return ["fp16", "int4_weight_only"]
    if target_id == "onnx_genai":
        return ["fp16", "int8", "int4", "directml_auto"]
    return []
