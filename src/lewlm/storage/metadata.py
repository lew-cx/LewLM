"""SQLite-backed metadata storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lewlm.conversion.models import ConversionArtifactRecord, JobRecord, JobStatus, JobType
from lewlm.core.contracts import GenerateMessage, IdempotentOperationRecord, ModelManifest, utc_now
from lewlm.core.errors import StorageError
from lewlm.history.models import SessionRecord, SessionTurnRecord
from lewlm.security.persistence import PersistenceEncryptor, is_encrypted_value


SCHEMA_VERSION = 10

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS app_kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_records (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    cache_key TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_manifests (
    model_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    source_path_encrypted TEXT,
    fingerprint TEXT NOT NULL,
    format_type TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_manifests_display_name
    ON model_manifests (display_name);

CREATE TABLE IF NOT EXISTS conversion_artifacts (
    cache_key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    output_path TEXT NOT NULL,
    policy TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_records_cache_key
    ON job_records (cache_key);

CREATE TABLE IF NOT EXISTS runtime_response_cache (
    cache_key TEXT PRIMARY KEY,
    capability TEXT NOT NULL,
    model_id TEXT NOT NULL,
    response_json TEXT NOT NULL,
    response_size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runtime_response_cache_capability_model
    ON runtime_response_cache (capability, model_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS cache_blocks (
    cache_key TEXT PRIMARY KEY,
    block_kind TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cache_blocks_kind_updated_at
    ON cache_blocks (block_kind, updated_at DESC);

CREATE TABLE IF NOT EXISTS idempotent_operation_results (
    operation_name TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    response_size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (operation_name, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_idempotent_operation_results_updated_at
    ON idempotent_operation_results (updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT,
    context_policy TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    turn_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at
    ON chat_sessions (updated_at DESC);

CREATE TABLE IF NOT EXISTS session_turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    request_kind TEXT NOT NULL,
    input_messages_json TEXT NOT NULL,
    response_message_json TEXT NOT NULL,
    requested_model_id TEXT,
    model_id TEXT NOT NULL,
    max_tokens INTEGER NOT NULL,
    temperature REAL NOT NULL,
    finish_reason TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_session_turns_session_id_created_at
    ON session_turns (session_id, created_at);

CREATE TABLE IF NOT EXISTS benchmark_records (
    benchmark_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    runtime TEXT NOT NULL,
    benchmark_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_benchmark_records_created_at
    ON benchmark_records (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_benchmark_records_model_id
    ON benchmark_records (model_id, created_at DESC);

CREATE TABLE IF NOT EXISTS benchmark_artifacts (
    artifact_id TEXT PRIMARY KEY,
    workload_signature TEXT NOT NULL,
    created_at TEXT NOT NULL,
    artifact_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_benchmark_artifacts_created_at
    ON benchmark_artifacts (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_benchmark_artifacts_signature
    ON benchmark_artifacts (workload_signature, created_at DESC);

CREATE TABLE IF NOT EXISTS capability_probe_records (
    probe_key TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    probe_name TEXT NOT NULL,
    runtime_name TEXT,
    runtime_affinity TEXT,
    model_id TEXT,
    workload_class TEXT,
    host_signature TEXT NOT NULL,
    status TEXT NOT NULL,
    source TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    probe_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_capability_probe_records_host_category
    ON capability_probe_records (host_signature, category, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_capability_probe_records_model
    ON capability_probe_records (model_id, recorded_at DESC);

CREATE TABLE IF NOT EXISTS runtime_probe_records (
    probe_key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    capability TEXT NOT NULL,
    mode TEXT NOT NULL,
    runtime_name TEXT,
    runtime_affinity TEXT,
    host_signature TEXT NOT NULL,
    state TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    probe_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runtime_probe_records_model
    ON runtime_probe_records (model_id, capability, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_runtime_probe_records_host
    ON runtime_probe_records (host_signature, recorded_at DESC);
"""


class MetadataStore:
    """Low-level metadata store used by registry and API services."""

    def __init__(self, database_path: Path, *, encryptor: PersistenceEncryptor | None = None) -> None:
        self.database_path = database_path
        self.encryptor = encryptor

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise StorageError(
                "SQLite operation failed.",
                details={"database_path": str(self.database_path), "error": str(exc)},
            ) from exc
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(SCHEMA_SQL)
            self._ensure_column(connection, "job_records", "cache_key", "TEXT")
            self._ensure_column(connection, "model_manifests", "source_path_encrypted", "TEXT")
            connection.execute(
                """
                INSERT INTO app_kv(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE
                SET value = excluded.value, updated_at = excluded.updated_at
                """,
                ("schema_version", self._encode_json_value(SCHEMA_VERSION), utc_now().isoformat()),
            )
            if self.encryptor is not None:
                self._migrate_encrypted_persistence(connection)

    def _ensure_column(self, connection: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def ping(self) -> bool:
        with self.connection() as connection:
            connection.execute("SELECT 1").fetchone()
        return True

    def get_schema_version(self) -> int:
        value = self.get_value("schema_version")
        if value is None:
            raise StorageError("Metadata store has not been initialized.")
        return int(value)

    def set_value(self, key: str, value: Any) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO app_kv(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE
                SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, self._encode_json_value(value), utc_now().isoformat()),
            )

    def get_value(self, key: str) -> Any | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT value FROM app_kv WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return self._decode_json_value(row["value"], field_name="app_kv.value")

    def increment_counter(self, key: str, amount: int = 1) -> int:
        current_value = self.get_counter(key)
        new_value = current_value + amount
        self.set_value(key, new_value)
        return new_value

    def get_counter(self, key: str) -> int:
        value = self.get_value(key)
        if isinstance(value, int):
            return value
        return 0

    def append_benchmark_record(self, payload: dict[str, Any]) -> None:
        benchmark_id = str(payload["benchmark_id"])
        model_id = str(payload["model_id"])
        runtime = str(payload["runtime"])
        created_at = str(payload["created_at"])
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO benchmark_records(benchmark_id, model_id, runtime, benchmark_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(benchmark_id) DO UPDATE
                SET model_id = excluded.model_id,
                    runtime = excluded.runtime,
                    benchmark_json = excluded.benchmark_json,
                    created_at = excluded.created_at
                """,
                (
                    benchmark_id,
                    model_id,
                    runtime,
                    self._encode_json_value(payload),
                    created_at,
                ),
            )

    def list_benchmark_records(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT benchmark_json
                FROM benchmark_records
                ORDER BY created_at DESC, benchmark_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            self._decode_json_value(row["benchmark_json"], field_name="benchmark_records.benchmark_json")
            for row in rows
        ]

    def benchmark_record_count(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM benchmark_records",
            ).fetchone()
        return int(row["count"])

    def append_benchmark_artifact(self, payload: dict[str, Any]) -> None:
        artifact_id = str(payload["artifact_id"])
        workload_signature = str(payload["workload_signature"])
        created_at = str(payload["created_at"])
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO benchmark_artifacts(artifact_id, workload_signature, created_at, artifact_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    workload_signature,
                    created_at,
                    self._encode_json_value(payload),
                ),
            )

    def list_benchmark_artifacts(
        self,
        *,
        limit: int = 20,
        workload_signature: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            if workload_signature is None:
                rows = connection.execute(
                    """
                    SELECT artifact_json
                    FROM benchmark_artifacts
                    ORDER BY created_at DESC, artifact_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT artifact_json
                    FROM benchmark_artifacts
                    WHERE workload_signature = ?
                    ORDER BY created_at DESC, artifact_id DESC
                    LIMIT ?
                    """,
                    (workload_signature, limit),
                ).fetchall()
        return [
            self._decode_json_value(row["artifact_json"], field_name="benchmark_artifacts.artifact_json")
            for row in rows
        ]

    def latest_benchmark_artifact(
        self,
        *,
        workload_signature: str,
        exclude_artifact_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            if exclude_artifact_id is None:
                row = connection.execute(
                    """
                    SELECT artifact_json
                    FROM benchmark_artifacts
                    WHERE workload_signature = ?
                    ORDER BY created_at DESC, artifact_id DESC
                    LIMIT 1
                    """,
                    (workload_signature,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT artifact_json
                    FROM benchmark_artifacts
                    WHERE workload_signature = ? AND artifact_id != ?
                    ORDER BY created_at DESC, artifact_id DESC
                    LIMIT 1
                    """,
                    (workload_signature, exclude_artifact_id),
                ).fetchone()
        if row is None:
            return None
        return self._decode_json_value(
            row["artifact_json"],
            field_name="benchmark_artifacts.artifact_json",
        )

    def get_benchmark_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT artifact_json
                FROM benchmark_artifacts
                WHERE artifact_id = ?
                """,
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return self._decode_json_value(
            row["artifact_json"],
            field_name="benchmark_artifacts.artifact_json",
        )

    def benchmark_artifact_count(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM benchmark_artifacts",
            ).fetchone()
        return int(row["count"])

    def upsert_capability_probe_record(
        self,
        *,
        category: str,
        probe_name: str,
        host_platform: dict[str, Any],
        status: str,
        source: str,
        reason: str,
        runtime_name: str | None = None,
        runtime_affinity: str | None = None,
        model_id: str | None = None,
        workload_class: str | None = None,
        details: dict[str, Any] | None = None,
        recorded_at: str | None = None,
    ) -> None:
        payload = {
            "probe_key": self._capability_probe_key(
                category=category,
                probe_name=probe_name,
                host_platform=host_platform,
                source=source,
                runtime_name=runtime_name,
                model_id=model_id,
                workload_class=workload_class,
            ),
            "category": category,
            "probe_name": probe_name,
            "status": status,
            "source": source,
            "reason": reason,
            "runtime_name": runtime_name,
            "runtime_affinity": runtime_affinity,
            "model_id": model_id,
            "workload_class": workload_class,
            "details": details or {},
            "recorded_at": recorded_at or utc_now().isoformat(),
        }
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO capability_probe_records(
                    probe_key,
                    category,
                    probe_name,
                    runtime_name,
                    runtime_affinity,
                    model_id,
                    workload_class,
                    host_signature,
                    status,
                    source,
                    recorded_at,
                    probe_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(probe_key) DO UPDATE
                SET category = excluded.category,
                    probe_name = excluded.probe_name,
                    runtime_name = excluded.runtime_name,
                    runtime_affinity = excluded.runtime_affinity,
                    model_id = excluded.model_id,
                    workload_class = excluded.workload_class,
                    host_signature = excluded.host_signature,
                    status = excluded.status,
                    source = excluded.source,
                    recorded_at = excluded.recorded_at,
                    probe_json = excluded.probe_json
                """,
                (
                    str(payload["probe_key"]),
                    category,
                    probe_name,
                    runtime_name,
                    runtime_affinity,
                    model_id,
                    workload_class,
                    self._host_signature(host_platform),
                    status,
                    source,
                    str(payload["recorded_at"]),
                    self._encode_json_value(payload),
                ),
            )

    def list_capability_probe_records(
        self,
        *,
        limit: int = 100,
        host_platform: dict[str, Any] | None = None,
        model_id: str | None = None,
        runtime_name: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT probe_json
            FROM capability_probe_records
            WHERE 1 = 1
        """
        parameters: list[Any] = []
        if host_platform is not None:
            query += " AND host_signature = ?"
            parameters.append(self._host_signature(host_platform))
        if model_id is not None:
            query += " AND model_id = ?"
            parameters.append(model_id)
        if runtime_name is not None:
            query += " AND runtime_name = ?"
            parameters.append(runtime_name)
        if category is not None:
            query += " AND category = ?"
            parameters.append(category)
        query += " ORDER BY recorded_at DESC, probe_key DESC LIMIT ?"
        parameters.append(limit)
        with self.connection() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [
            self._decode_json_value(row["probe_json"], field_name="capability_probe_records.probe_json")
            for row in rows
        ]

    def capability_probe_record_count(
        self,
        *,
        host_platform: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> int:
        query = "SELECT COUNT(*) AS count FROM capability_probe_records WHERE 1 = 1"
        parameters: list[Any] = []
        if host_platform is not None:
            query += " AND host_signature = ?"
            parameters.append(self._host_signature(host_platform))
        if model_id is not None:
            query += " AND model_id = ?"
            parameters.append(model_id)
        with self.connection() as connection:
            row = connection.execute(query, tuple(parameters)).fetchone()
        return int(row["count"])

    def upsert_runtime_probe_record(
        self,
        *,
        model_id: str,
        capability: str,
        mode: str,
        host_platform: dict[str, Any],
        evidence: dict[str, Any],
        recorded_at: str | None = None,
    ) -> str:
        runtime_name = evidence.get("runtime_name")
        runtime_affinity = evidence.get("runtime_affinity")
        state = str(evidence["state"])
        recorded = recorded_at or str(evidence.get("recorded_at") or utc_now().isoformat())
        probe_key = self._runtime_probe_key(
            model_id=model_id,
            capability=capability,
            mode=mode,
            host_platform=host_platform,
            runtime_name=str(runtime_name) if runtime_name is not None else None,
        )
        evidence_payload = {
            **evidence,
            "probe_key": probe_key,
            "recorded_at": recorded,
        }
        payload = {
            "probe_key": probe_key,
            "model_id": model_id,
            "capability": capability,
            "mode": mode,
            "runtime_name": runtime_name,
            "runtime_affinity": runtime_affinity,
            "host_platform": host_platform,
            "state": state,
            "recorded_at": recorded,
            "evidence": evidence_payload,
        }
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO runtime_probe_records(
                    probe_key,
                    model_id,
                    capability,
                    mode,
                    runtime_name,
                    runtime_affinity,
                    host_signature,
                    state,
                    recorded_at,
                    probe_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(probe_key) DO UPDATE
                SET model_id = excluded.model_id,
                    capability = excluded.capability,
                    mode = excluded.mode,
                    runtime_name = excluded.runtime_name,
                    runtime_affinity = excluded.runtime_affinity,
                    host_signature = excluded.host_signature,
                    state = excluded.state,
                    recorded_at = excluded.recorded_at,
                    probe_json = excluded.probe_json
                """,
                (
                    probe_key,
                    model_id,
                    capability,
                    mode,
                    str(runtime_name) if runtime_name is not None else None,
                    str(runtime_affinity) if runtime_affinity is not None else None,
                    self._host_signature(host_platform),
                    state,
                    recorded,
                    self._encode_json_value(payload),
                ),
            )
        return probe_key

    def list_runtime_probe_records(
        self,
        *,
        limit: int = 100,
        host_platform: dict[str, Any] | None = None,
        model_id: str | None = None,
        capability: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT probe_json
            FROM runtime_probe_records
            WHERE 1 = 1
        """
        parameters: list[Any] = []
        if host_platform is not None:
            query += " AND host_signature = ?"
            parameters.append(self._host_signature(host_platform))
        if model_id is not None:
            query += " AND model_id = ?"
            parameters.append(model_id)
        if capability is not None:
            query += " AND capability = ?"
            parameters.append(capability)
        query += " ORDER BY recorded_at DESC, probe_key DESC LIMIT ?"
        parameters.append(limit)
        with self.connection() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [
            self._decode_json_value(row["probe_json"], field_name="runtime_probe_records.probe_json")
            for row in rows
        ]

    def runtime_probe_record_count(
        self,
        *,
        host_platform: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> int:
        query = "SELECT COUNT(*) AS count FROM runtime_probe_records WHERE 1 = 1"
        parameters: list[Any] = []
        if host_platform is not None:
            query += " AND host_signature = ?"
            parameters.append(self._host_signature(host_platform))
        if model_id is not None:
            query += " AND model_id = ?"
            parameters.append(model_id)
        with self.connection() as connection:
            row = connection.execute(query, tuple(parameters)).fetchone()
        return int(row["count"])

    def upsert_serving_profile(
        self,
        *,
        model_id: str,
        capability: str,
        host_platform: dict[str, Any],
        runtime_name: str | None = None,
        workload_class: str | None = None,
        payload: dict[str, Any],
    ) -> None:
        self.set_value(
            self._serving_profile_key(
                model_id=model_id,
                capability=capability,
                host_platform=host_platform,
                runtime_name=runtime_name,
                workload_class=workload_class,
            ),
            payload,
        )

    def get_serving_profile(
        self,
        *,
        model_id: str,
        capability: str,
        host_platform: dict[str, Any],
        runtime_name: str | None = None,
        workload_class: str | None = None,
    ) -> dict[str, Any] | None:
        for key in self._serving_profile_lookup_keys(
            model_id=model_id,
            capability=capability,
            host_platform=host_platform,
            runtime_name=runtime_name,
            workload_class=workload_class,
        ):
            value = self.get_value(key)
            if isinstance(value, dict):
                return value
        return None

    def list_serving_profiles(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT value
                FROM app_kv
                WHERE key LIKE 'serving_profile:%'
                ORDER BY updated_at DESC, key DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            self._decode_json_value(row["value"], field_name="app_kv.value")
            for row in rows
        ]

    def upsert_runtime_preference(
        self,
        *,
        model_id: str,
        capability: str,
        host_platform: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        self.set_value(
            self._runtime_preference_key(
                model_id=model_id,
                capability=capability,
                host_platform=host_platform,
            ),
            payload,
        )

    def get_runtime_preference(
        self,
        *,
        model_id: str,
        capability: str,
        host_platform: dict[str, Any],
    ) -> dict[str, Any] | None:
        value = self.get_value(
            self._runtime_preference_key(
                model_id=model_id,
                capability=capability,
                host_platform=host_platform,
            ),
        )
        return value if isinstance(value, dict) else None

    def list_runtime_preferences(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT value
                FROM app_kv
                WHERE key LIKE 'runtime_preference:%'
                ORDER BY updated_at DESC, key DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            self._decode_json_value(row["value"], field_name="app_kv.value")
            for row in rows
        ]

    def list_model_manifest_records(self) -> list[tuple[str, str]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT source_path, source_path_encrypted, fingerprint FROM model_manifests",
            ).fetchall()
        return [
            (
                self._decode_source_path(
                    source_path=str(row["source_path"]),
                    encrypted_source_path=row["source_path_encrypted"],
                ),
                str(row["fingerprint"]),
            )
            for row in rows
        ]

    def list_model_manifests(self) -> list[ModelManifest]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT manifest_json
                FROM model_manifests
                ORDER BY model_id ASC
                """,
            ).fetchall()
        manifests = [
            ModelManifest.model_validate_json(
                self._decode_text_value(row["manifest_json"], field_name="model_manifests.manifest_json"),
            )
            for row in rows
        ]
        return sorted(manifests, key=lambda manifest: (manifest.display_name.casefold(), manifest.model_id))

    def get_model_manifest(self, model_id: str) -> ModelManifest | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT manifest_json FROM model_manifests WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        if row is None:
            return None
        return ModelManifest.model_validate_json(
            self._decode_text_value(row["manifest_json"], field_name="model_manifests.manifest_json"),
        )

    def upsert_job(self, job: JobRecord) -> None:
        timestamp = utc_now().isoformat()
        job = job.model_copy(update={"updated_at": utc_now()})
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO job_records(job_id, job_type, status, cache_key, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_type = excluded.job_type,
                    status = excluded.status,
                    cache_key = excluded.cache_key,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    job.job_id,
                    job.job_type.value,
                    job.status.value,
                    job.cache_key,
                    self._encode_json_value(job.payload),
                    job.created_at.isoformat(),
                    timestamp,
                ),
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT job_id, job_type, status, cache_key, payload_json, created_at, updated_at
                FROM job_records
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return self._job_record_from_row(row)

    def find_job_by_cache_key(
        self,
        *,
        job_type: JobType,
        cache_key: str,
        statuses: Iterable[JobStatus],
    ) -> JobRecord | None:
        status_values = tuple(status.value for status in statuses)
        if not status_values:
            return None
        placeholders = ", ".join("?" for _ in status_values)
        parameters = (job_type.value, cache_key, *status_values)
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT job_id, job_type, status, cache_key, payload_json, created_at, updated_at
                FROM job_records
                WHERE job_type = ? AND cache_key = ? AND status IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT 1
                """,
                parameters,
            ).fetchone()
        if row is None:
            return None
        return self._job_record_from_row(row)

    def _job_record_from_row(self, row: sqlite3.Row) -> JobRecord:
        payload = self._decode_json_value(row["payload_json"], field_name="job_records.payload_json")
        idempotency_key = None
        if isinstance(payload.get("request"), dict):
            request_idempotency_key = payload["request"].get("idempotency_key")
            if isinstance(request_idempotency_key, str):
                idempotency_key = request_idempotency_key
        return JobRecord(
            job_id=row["job_id"],
            job_type=JobType(row["job_type"]),
            status=JobStatus(row["status"]),
            cache_key=row["cache_key"],
            idempotency_key=idempotency_key,
            payload=payload,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert_conversion_artifact(self, artifact: ConversionArtifactRecord) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO conversion_artifacts(
                    cache_key, model_id, output_path, policy, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    model_id = excluded.model_id,
                    output_path = excluded.output_path,
                    policy = excluded.policy,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    artifact.cache_key,
                    artifact.model_id,
                    self._encode_text_value(artifact.output_path),
                    artifact.policy.value,
                    self._encode_json_value(artifact.metadata),
                    artifact.created_at.isoformat(),
                    artifact.updated_at.isoformat(),
                ),
            )

    def get_conversion_artifact(self, cache_key: str) -> ConversionArtifactRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT cache_key, model_id, output_path, policy, metadata_json, created_at, updated_at
                FROM conversion_artifacts
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return ConversionArtifactRecord(
            cache_key=row["cache_key"],
            model_id=row["model_id"],
            output_path=self._decode_text_value(row["output_path"], field_name="conversion_artifacts.output_path"),
            policy=row["policy"],
            metadata=self._decode_json_value(row["metadata_json"], field_name="conversion_artifacts.metadata_json"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_conversion_artifacts(self) -> list[ConversionArtifactRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT cache_key, model_id, output_path, policy, metadata_json, created_at, updated_at
                FROM conversion_artifacts
                ORDER BY updated_at DESC
                """,
            ).fetchall()
        return [
            ConversionArtifactRecord(
                cache_key=row["cache_key"],
                model_id=row["model_id"],
                output_path=self._decode_text_value(row["output_path"], field_name="conversion_artifacts.output_path"),
                policy=row["policy"],
                metadata=self._decode_json_value(row["metadata_json"], field_name="conversion_artifacts.metadata_json"),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def clear_conversion_artifacts(self) -> int:
        with self.connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM conversion_artifacts").fetchone()
            connection.execute("DELETE FROM conversion_artifacts")
        return int(row["count"])

    def upsert_runtime_response_cache_entry(
        self,
        *,
        cache_key: str,
        capability: str,
        model_id: str,
        response_payload: dict[str, Any],
    ) -> None:
        response_json = json.dumps(response_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        timestamp = utc_now().isoformat()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO runtime_response_cache(
                    cache_key, capability, model_id, response_json, response_size_bytes, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    capability = excluded.capability,
                    model_id = excluded.model_id,
                    response_json = excluded.response_json,
                    response_size_bytes = excluded.response_size_bytes,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    capability,
                    model_id,
                    self._encode_text_value(response_json),
                    len(response_json.encode("utf-8")),
                    timestamp,
                    timestamp,
                ),
            )

    def get_runtime_response_cache_entry(self, cache_key: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT cache_key, capability, model_id, response_json, response_size_bytes, created_at, updated_at
                FROM runtime_response_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "cache_key": row["cache_key"],
            "capability": row["capability"],
            "model_id": row["model_id"],
            "response_payload": json.loads(
                self._decode_text_value(row["response_json"], field_name="runtime_response_cache.response_json"),
            ),
            "response_size_bytes": int(row["response_size_bytes"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def runtime_response_cache_count(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM runtime_response_cache",
            ).fetchone()
        return int(row["count"])

    def runtime_response_cache_size_bytes(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(response_size_bytes), 0) AS total FROM runtime_response_cache",
            ).fetchone()
        return int(row["total"])

    def upsert_cache_block(
        self,
        *,
        cache_key: str,
        block_kind: str,
        storage_path: str,
        size_bytes: int,
        metadata: dict[str, Any],
    ) -> None:
        timestamp = utc_now().isoformat()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO cache_blocks(
                    cache_key, block_kind, storage_path, size_bytes, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    block_kind = excluded.block_kind,
                    storage_path = excluded.storage_path,
                    size_bytes = excluded.size_bytes,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    block_kind,
                    self._encode_text_value(storage_path),
                    size_bytes,
                    self._encode_json_value(metadata),
                    timestamp,
                    timestamp,
                ),
            )

    def get_cache_block(self, cache_key: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT cache_key, block_kind, storage_path, size_bytes, metadata_json, created_at, updated_at
                FROM cache_blocks
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "cache_key": row["cache_key"],
            "block_kind": row["block_kind"],
            "storage_path": self._decode_text_value(row["storage_path"], field_name="cache_blocks.storage_path"),
            "size_bytes": int(row["size_bytes"]),
            "metadata": self._decode_json_value(row["metadata_json"], field_name="cache_blocks.metadata_json"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def cache_block_count(self, *, block_kind: str | None = None) -> int:
        with self.connection() as connection:
            if block_kind is None:
                row = connection.execute("SELECT COUNT(*) AS count FROM cache_blocks").fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM cache_blocks WHERE block_kind = ?",
                    (block_kind,),
                ).fetchone()
        return int(row["count"])

    def cache_block_size_bytes(self, *, block_kind: str | None = None) -> int:
        with self.connection() as connection:
            if block_kind is None:
                row = connection.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM cache_blocks",
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM cache_blocks WHERE block_kind = ?",
                    (block_kind,),
                ).fetchone()
        return int(row["total"])

    def list_cache_blocks(self, *, block_kind: str | None = None) -> list[dict[str, Any]]:
        with self.connection() as connection:
            if block_kind is None:
                rows = connection.execute(
                    """
                    SELECT cache_key, block_kind, storage_path, size_bytes, metadata_json, created_at, updated_at
                    FROM cache_blocks
                    ORDER BY updated_at DESC, cache_key ASC
                    """,
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT cache_key, block_kind, storage_path, size_bytes, metadata_json, created_at, updated_at
                    FROM cache_blocks
                    WHERE block_kind = ?
                    ORDER BY updated_at DESC, cache_key ASC
                    """,
                    (block_kind,),
                ).fetchall()
        return [
            {
                "cache_key": row["cache_key"],
                "block_kind": row["block_kind"],
                "storage_path": self._decode_text_value(row["storage_path"], field_name="cache_blocks.storage_path"),
                "size_bytes": int(row["size_bytes"]),
                "metadata": self._decode_json_value(row["metadata_json"], field_name="cache_blocks.metadata_json"),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def delete_cache_block(self, cache_key: str) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM cache_blocks WHERE cache_key = ?", (cache_key,))

    def upsert_idempotent_operation_result(self, record: IdempotentOperationRecord) -> None:
        response_json = json.dumps(record.response_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        timestamp = utc_now().isoformat()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO idempotent_operation_results(
                    operation_name, idempotency_key, request_hash, response_json, response_size_bytes, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(operation_name, idempotency_key) DO UPDATE SET
                    request_hash = excluded.request_hash,
                    response_json = excluded.response_json,
                    response_size_bytes = excluded.response_size_bytes,
                    updated_at = excluded.updated_at
                """,
                (
                    record.operation_name,
                    record.idempotency_key,
                    record.request_hash,
                    self._encode_text_value(response_json),
                    len(response_json.encode("utf-8")),
                    record.created_at.isoformat(),
                    timestamp,
                ),
            )

    def get_idempotent_operation_result(
        self,
        operation_name: str,
        idempotency_key: str,
    ) -> IdempotentOperationRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT operation_name, idempotency_key, request_hash, response_json, created_at, updated_at
                FROM idempotent_operation_results
                WHERE operation_name = ? AND idempotency_key = ?
                """,
                (operation_name, idempotency_key),
            ).fetchone()
        if row is None:
            return None
        return IdempotentOperationRecord(
            operation_name=row["operation_name"],
            idempotency_key=row["idempotency_key"],
            request_hash=row["request_hash"],
            response_payload=json.loads(
                self._decode_text_value(
                    row["response_json"],
                    field_name="idempotent_operation_results.response_json",
                ),
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def clear_idempotent_operation_results(self, *, operation_name: str | None = None) -> int:
        with self.connection() as connection:
            if operation_name is None:
                row = connection.execute("SELECT COUNT(*) AS count FROM idempotent_operation_results").fetchone()
                connection.execute("DELETE FROM idempotent_operation_results")
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM idempotent_operation_results WHERE operation_name = ?",
                    (operation_name,),
                ).fetchone()
                connection.execute(
                    "DELETE FROM idempotent_operation_results WHERE operation_name = ?",
                    (operation_name,),
                )
        return int(row["count"])

    def upsert_session(self, session: SessionRecord) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions(
                    session_id, title, context_policy, metadata_json, message_count, turn_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title = excluded.title,
                    context_policy = excluded.context_policy,
                    metadata_json = excluded.metadata_json,
                    message_count = excluded.message_count,
                    turn_count = excluded.turn_count,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    session.session_id,
                    self._encode_text_value(session.title) if session.title is not None else None,
                    session.context_policy,
                    self._encode_json_value(session.metadata),
                    session.message_count,
                    session.turn_count,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )

    def list_sessions(self) -> list[SessionRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT session_id, title, context_policy, metadata_json, message_count, turn_count, created_at, updated_at
                FROM chat_sessions
                ORDER BY updated_at DESC, created_at DESC
                """,
            ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT session_id, title, context_policy, metadata_json, message_count, turn_count, created_at, updated_at
                FROM chat_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def delete_session(self, session_id: str) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))

    def append_session_turn(self, turn: SessionTurnRecord) -> None:
        with self.connection() as connection:
            session_row = connection.execute(
                """
                SELECT message_count, turn_count
                FROM chat_sessions
                WHERE session_id = ?
                """,
                (turn.session_id,),
            ).fetchone()
            if session_row is None:
                raise StorageError("Cannot append to a missing session.", details={"session_id": turn.session_id})
            connection.execute(
                """
                INSERT INTO session_turns(
                    turn_id,
                    session_id,
                    request_kind,
                    input_messages_json,
                    response_message_json,
                    requested_model_id,
                    model_id,
                    max_tokens,
                    temperature,
                    finish_reason,
                    usage_json,
                    metadata_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_id,
                    turn.session_id,
                    turn.request_kind,
                    self._encode_json_value([message.model_dump(mode="json") for message in turn.input_messages]),
                    self._encode_json_value(turn.response_message.model_dump(mode="json")),
                    turn.requested_model_id,
                    turn.model_id,
                    turn.max_tokens,
                    turn.temperature,
                    turn.finish_reason,
                    self._encode_json_value(turn.usage),
                    self._encode_json_value(turn.metadata),
                    turn.created_at.isoformat(),
                ),
            )
            connection.execute(
                """
                UPDATE chat_sessions
                SET message_count = ?, turn_count = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    int(session_row["message_count"]) + len(turn.input_messages) + 1,
                    int(session_row["turn_count"]) + 1,
                    turn.created_at.isoformat(),
                    turn.session_id,
                ),
            )

    def list_session_turns(self, session_id: str) -> list[SessionTurnRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    turn_id,
                    session_id,
                    request_kind,
                    input_messages_json,
                    response_message_json,
                    requested_model_id,
                    model_id,
                    max_tokens,
                    temperature,
                    finish_reason,
                    usage_json,
                    metadata_json,
                    created_at
                FROM session_turns
                WHERE session_id = ?
                ORDER BY created_at ASC, turn_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [self._row_to_turn(row) for row in rows]

    def replace_model_manifests(
        self,
        manifests: Iterable[ModelManifest],
        *,
        stale_source_paths: Iterable[str],
    ) -> None:
        manifest_list = list(manifests)
        stale_sources = list(stale_source_paths)
        timestamp = utc_now().isoformat()
        with self.connection() as connection:
            for manifest in manifest_list:
                stored_source_path = self._lookup_source_path(manifest.source_path)
                connection.execute(
                    "DELETE FROM model_manifests WHERE model_id = ? AND source_path != ?",
                    (manifest.model_id, stored_source_path),
                )
                connection.execute(
                    """
                    INSERT INTO model_manifests(
                        model_id,
                        display_name,
                        source_path,
                        source_path_encrypted,
                        fingerprint,
                        format_type,
                        manifest_json,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_path) DO UPDATE SET
                        model_id = excluded.model_id,
                        display_name = excluded.display_name,
                        source_path = excluded.source_path,
                        source_path_encrypted = excluded.source_path_encrypted,
                        fingerprint = excluded.fingerprint,
                        format_type = excluded.format_type,
                        manifest_json = excluded.manifest_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        manifest.model_id,
                        self._encode_text_value(manifest.display_name),
                        stored_source_path,
                        self._encrypted_source_path(manifest.source_path),
                        manifest.fingerprint,
                        manifest.format_type.value,
                        self._encode_text_value(manifest.model_dump_json()),
                        timestamp,
                    ),
                )
            if stale_sources:
                connection.executemany(
                    "DELETE FROM model_manifests WHERE source_path = ?",
                    [(self._lookup_source_path(source_path),) for source_path in stale_sources],
                )

    def _serving_profile_key(
        self,
        *,
        model_id: str,
        capability: str,
        host_platform: dict[str, Any],
        runtime_name: str | None = None,
        workload_class: str | None = None,
    ) -> str:
        host_signature = hashlib.sha256(json.dumps(host_platform, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        model_signature = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:16]
        key = f"serving_profile:{capability}:{host_signature}:{model_signature}"
        if runtime_name:
            runtime_signature = hashlib.sha256(runtime_name.encode("utf-8")).hexdigest()[:12]
            key = f"{key}:{runtime_signature}"
        if workload_class:
            key = f"{key}:{workload_class}"
        return key

    def _serving_profile_lookup_keys(
        self,
        *,
        model_id: str,
        capability: str,
        host_platform: dict[str, Any],
        runtime_name: str | None,
        workload_class: str | None,
    ) -> tuple[str, ...]:
        keys: list[str] = []
        if runtime_name and workload_class:
            keys.append(
                self._serving_profile_key(
                    model_id=model_id,
                    capability=capability,
                    host_platform=host_platform,
                    runtime_name=runtime_name,
                    workload_class=workload_class,
                ),
            )
        if runtime_name and workload_class in {None, "text_only", "text_only_multimodal"}:
            keys.append(
                self._serving_profile_key(
                    model_id=model_id,
                    capability=capability,
                    host_platform=host_platform,
                    runtime_name=runtime_name,
                ),
            )
        if workload_class in {None, "text_only", "text_only_multimodal"}:
            keys.append(
                self._serving_profile_key(
                    model_id=model_id,
                    capability=capability,
                    host_platform=host_platform,
                ),
            )
        deduped_keys: list[str] = []
        seen_keys: set[str] = set()
        for key in keys:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_keys.append(key)
        return tuple(deduped_keys)

    def _runtime_preference_key(
        self,
        *,
        model_id: str,
        capability: str,
        host_platform: dict[str, Any],
    ) -> str:
        host_signature = self._host_signature(host_platform)
        model_signature = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:16]
        return f"runtime_preference:{capability}:{host_signature}:{model_signature}"

    def _capability_probe_key(
        self,
        *,
        category: str,
        probe_name: str,
        host_platform: dict[str, Any],
        source: str,
        runtime_name: str | None,
        model_id: str | None,
        workload_class: str | None,
    ) -> str:
        discriminator = json.dumps(
            {
                "category": category,
                "probe_name": probe_name,
                "host_signature": self._host_signature(host_platform),
                "source": source,
                "runtime_name": runtime_name or "",
                "model_id": model_id or "",
                "workload_class": workload_class or "",
            },
            sort_keys=True,
        )
        return f"capability_probe:{hashlib.sha256(discriminator.encode('utf-8')).hexdigest()}"

    def _runtime_probe_key(
        self,
        *,
        model_id: str,
        capability: str,
        mode: str,
        host_platform: dict[str, Any],
        runtime_name: str | None,
    ) -> str:
        discriminator = json.dumps(
            {
                "model_id": model_id,
                "capability": capability,
                "mode": mode,
                "host_signature": self._host_signature(host_platform),
                "runtime_name": runtime_name or "",
            },
            sort_keys=True,
        )
        return f"runtime_probe:{hashlib.sha256(discriminator.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _host_signature(host_platform: dict[str, Any]) -> str:
        normalized_payload = {
            key: host_platform.get(key)
            for key in ("system", "release", "machine", "python_version")
            if key in host_platform
        }
        return hashlib.sha256(json.dumps(normalized_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def snapshot(self) -> dict[str, Any]:
        with self.connection() as connection:
            model_count = connection.execute(
                "SELECT COUNT(*) AS count FROM model_manifests",
            ).fetchone()["count"]
            job_count = connection.execute(
                "SELECT COUNT(*) AS count FROM job_records",
            ).fetchone()["count"]
            artifact_count = connection.execute(
                "SELECT COUNT(*) AS count FROM conversion_artifacts",
            ).fetchone()["count"]
            session_count = connection.execute(
                "SELECT COUNT(*) AS count FROM chat_sessions",
            ).fetchone()["count"]
            session_turn_count = connection.execute(
                "SELECT COUNT(*) AS count FROM session_turns",
            ).fetchone()["count"]
            benchmark_count = connection.execute(
                "SELECT COUNT(*) AS count FROM benchmark_records",
            ).fetchone()["count"]
            benchmark_artifact_count = connection.execute(
                "SELECT COUNT(*) AS count FROM benchmark_artifacts",
            ).fetchone()["count"]
            capability_probe_count = connection.execute(
                "SELECT COUNT(*) AS count FROM capability_probe_records",
            ).fetchone()["count"]
            runtime_probe_count = connection.execute(
                "SELECT COUNT(*) AS count FROM runtime_probe_records",
            ).fetchone()["count"]
            runtime_response_cache_count = connection.execute(
                "SELECT COUNT(*) AS count FROM runtime_response_cache",
            ).fetchone()["count"]
            cache_block_count = connection.execute(
                "SELECT COUNT(*) AS count FROM cache_blocks",
            ).fetchone()["count"]
        return {
            "database_path": str(self.database_path),
            "schema_version": self.get_schema_version(),
            "model_count": int(model_count),
            "job_count": int(job_count),
            "conversion_artifact_count": int(artifact_count),
            "runtime_response_cache_count": int(runtime_response_cache_count),
            "cache_block_count": int(cache_block_count),
            "session_count": int(session_count),
            "session_turn_count": int(session_turn_count),
            "benchmark_record_count": int(benchmark_count),
            "benchmark_artifact_count": int(benchmark_artifact_count),
            "capability_probe_record_count": int(capability_probe_count),
            "runtime_probe_record_count": int(runtime_probe_count),
        }

    def _migrate_encrypted_persistence(self, connection: sqlite3.Connection) -> None:
        for row in connection.execute("SELECT key, value FROM app_kv").fetchall():
            if not is_encrypted_value(row["value"]):
                connection.execute(
                    "UPDATE app_kv SET value = ? WHERE key = ?",
                    (self._encode_text_value(str(row["value"])), row["key"]),
                )

        for row in connection.execute("SELECT job_id, payload_json FROM job_records").fetchall():
            if not is_encrypted_value(row["payload_json"]):
                connection.execute(
                    "UPDATE job_records SET payload_json = ? WHERE job_id = ?",
                    (self._encode_text_value(str(row["payload_json"])), row["job_id"]),
                )

        for row in connection.execute(
            "SELECT cache_key, output_path, metadata_json FROM conversion_artifacts",
        ).fetchall():
            updates: list[tuple[str, str]] = []
            if not is_encrypted_value(row["output_path"]):
                updates.append(("output_path", self._encode_text_value(str(row["output_path"]))))
            if not is_encrypted_value(row["metadata_json"]):
                updates.append(("metadata_json", self._encode_text_value(str(row["metadata_json"]))))
            if updates:
                assignments = ", ".join(f"{column} = ?" for column, _ in updates)
                parameters = [value for _, value in updates]
                parameters.append(row["cache_key"])
                connection.execute(
                    f"UPDATE conversion_artifacts SET {assignments} WHERE cache_key = ?",
                    parameters,
                )

        for row in connection.execute(
            "SELECT cache_key, response_json FROM runtime_response_cache",
        ).fetchall():
            if not is_encrypted_value(row["response_json"]):
                connection.execute(
                    "UPDATE runtime_response_cache SET response_json = ? WHERE cache_key = ?",
                    (self._encode_text_value(str(row["response_json"])), row["cache_key"]),
                )

        for row in connection.execute(
            "SELECT cache_key, storage_path, metadata_json FROM cache_blocks",
        ).fetchall():
            updates: list[tuple[str, str]] = []
            if not is_encrypted_value(str(row["storage_path"])):
                updates.append(("storage_path", self._encode_text_value(str(row["storage_path"]))))
            if not is_encrypted_value(str(row["metadata_json"])):
                updates.append(("metadata_json", self._encode_text_value(str(row["metadata_json"]))))
            if updates:
                assignments = ", ".join(f"{column} = ?" for column, _ in updates)
                parameters = [value for _, value in updates]
                parameters.append(row["cache_key"])
                connection.execute(
                    f"UPDATE cache_blocks SET {assignments} WHERE cache_key = ?",
                    parameters,
                )

        for row in connection.execute(
            """
            SELECT model_id, display_name, source_path, source_path_encrypted, manifest_json
            FROM model_manifests
            """,
        ).fetchall():
            actual_source_path = (
                self._decode_text_value(
                    row["source_path_encrypted"],
                    field_name="model_manifests.source_path_encrypted",
                )
                if row["source_path_encrypted"]
                else str(row["source_path"])
            )
            expected_lookup = self._lookup_source_path(actual_source_path)
            if (
                row["source_path_encrypted"]
                and str(row["source_path"]) == expected_lookup
                and is_encrypted_value(str(row["display_name"]))
                and is_encrypted_value(str(row["manifest_json"]))
            ):
                continue
            display_name = self._decode_text_value(str(row["display_name"]), field_name="model_manifests.display_name")
            manifest_json = self._decode_text_value(str(row["manifest_json"]), field_name="model_manifests.manifest_json")
            connection.execute(
                """
                UPDATE model_manifests
                SET display_name = ?, source_path = ?, source_path_encrypted = ?, manifest_json = ?
                WHERE model_id = ?
                """,
                (
                    self._encode_text_value(display_name),
                    expected_lookup,
                    self._encrypted_source_path(actual_source_path),
                    self._encode_text_value(manifest_json),
                    row["model_id"],
                ),
            )

        for row in connection.execute(
            """
            SELECT session_id, title, metadata_json
            FROM chat_sessions
            """,
        ).fetchall():
            updates: list[tuple[str, str]] = []
            if row["title"] is not None and not is_encrypted_value(str(row["title"])):
                updates.append(("title", self._encode_text_value(str(row["title"]))))
            if not is_encrypted_value(row["metadata_json"]):
                updates.append(("metadata_json", self._encode_text_value(str(row["metadata_json"]))))
            if updates:
                assignments = ", ".join(f"{column} = ?" for column, _ in updates)
                parameters = [value for _, value in updates]
                parameters.append(row["session_id"])
                connection.execute(
                    f"UPDATE chat_sessions SET {assignments} WHERE session_id = ?",
                    parameters,
                )

        for row in connection.execute(
            """
            SELECT benchmark_id, benchmark_json
            FROM benchmark_records
            """,
        ).fetchall():
            if not is_encrypted_value(row["benchmark_json"]):
                connection.execute(
                    "UPDATE benchmark_records SET benchmark_json = ? WHERE benchmark_id = ?",
                    (self._encode_text_value(str(row["benchmark_json"])), row["benchmark_id"]),
                )

        for row in connection.execute(
            """
            SELECT artifact_id, artifact_json
            FROM benchmark_artifacts
            """,
        ).fetchall():
            if not is_encrypted_value(row["artifact_json"]):
                connection.execute(
                    "UPDATE benchmark_artifacts SET artifact_json = ? WHERE artifact_id = ?",
                    (self._encode_text_value(str(row["artifact_json"])), row["artifact_id"]),
                )

        for row in connection.execute(
            """
            SELECT turn_id, input_messages_json, response_message_json, usage_json, metadata_json
            FROM session_turns
            """,
        ).fetchall():
            updates: list[tuple[str, str]] = []
            if not is_encrypted_value(row["input_messages_json"]):
                updates.append(("input_messages_json", self._encode_text_value(str(row["input_messages_json"]))))
            if not is_encrypted_value(row["response_message_json"]):
                updates.append(("response_message_json", self._encode_text_value(str(row["response_message_json"]))))
            if not is_encrypted_value(row["usage_json"]):
                updates.append(("usage_json", self._encode_text_value(str(row["usage_json"]))))
            if not is_encrypted_value(row["metadata_json"]):
                updates.append(("metadata_json", self._encode_text_value(str(row["metadata_json"]))))
            if updates:
                assignments = ", ".join(f"{column} = ?" for column, _ in updates)
                parameters = [value for _, value in updates]
                parameters.append(row["turn_id"])
                connection.execute(
                    f"UPDATE session_turns SET {assignments} WHERE turn_id = ?",
                    parameters,
                )

    def _encode_json_value(self, value: Any) -> str:
        serialized = json.dumps(value)
        return self._encode_text_value(serialized)

    def _decode_json_value(self, value: str, *, field_name: str) -> Any:
        return json.loads(self._decode_text_value(value, field_name=field_name))

    def _encode_text_value(self, value: str) -> str:
        if self.encryptor is None:
            return value
        return self.encryptor.encrypt_text(value)

    def _decode_text_value(self, value: str, *, field_name: str) -> str:
        if is_encrypted_value(value):
            if self.encryptor is None:
                raise StorageError(
                    "Encrypted persistence data requires a configured passphrase.",
                    details={"database_path": str(self.database_path), "field": field_name},
                )
            return self.encryptor.decrypt_text(value)
        return value

    def _lookup_source_path(self, source_path: str) -> str:
        if self.encryptor is None:
            return source_path
        return self.encryptor.stable_digest(source_path)

    def _encrypted_source_path(self, source_path: str) -> str | None:
        if self.encryptor is None:
            return None
        return self.encryptor.encrypt_text(source_path)

    def _decode_source_path(self, *, source_path: str, encrypted_source_path: str | None) -> str:
        if encrypted_source_path:
            return self._decode_text_value(
                encrypted_source_path,
                field_name="model_manifests.source_path_encrypted",
            )
        return source_path

    def _row_to_session(self, row: sqlite3.Row) -> SessionRecord:
        raw_title = row["title"]
        return SessionRecord(
            session_id=row["session_id"],
            title=None if raw_title is None else self._decode_text_value(str(raw_title), field_name="chat_sessions.title"),
            context_policy=row["context_policy"],
            metadata=self._decode_json_value(row["metadata_json"], field_name="chat_sessions.metadata_json"),
            message_count=int(row["message_count"]),
            turn_count=int(row["turn_count"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_turn(self, row: sqlite3.Row) -> SessionTurnRecord:
        return SessionTurnRecord(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            request_kind=row["request_kind"],
            input_messages=[
                GenerateMessage.model_validate(payload)
                for payload in self._decode_json_value(
                    row["input_messages_json"],
                    field_name="session_turns.input_messages_json",
                )
            ],
            response_message=GenerateMessage.model_validate(
                self._decode_json_value(
                    row["response_message_json"],
                    field_name="session_turns.response_message_json",
                ),
            ),
            requested_model_id=row["requested_model_id"],
            model_id=row["model_id"],
            max_tokens=int(row["max_tokens"]),
            temperature=float(row["temperature"]),
            finish_reason=row["finish_reason"],
            usage=self._decode_json_value(row["usage_json"], field_name="session_turns.usage_json"),
            metadata=self._decode_json_value(row["metadata_json"], field_name="session_turns.metadata_json"),
            created_at=row["created_at"],
        )
