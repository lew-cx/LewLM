"""Structured audit logging for opt-in LewLM deployments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
import os
from threading import Lock
from typing import Any

from lewlm.config.settings import LewLMSettings
from lewlm.security.persistence import PersistenceEncryptor


class AuditLogger:
    """Append structured audit events to a local JSONL file when enabled."""

    def __init__(self, settings: LewLMSettings, *, encryptor: PersistenceEncryptor | None = None) -> None:
        self.settings = settings
        self.encryptor = encryptor
        self._lock = Lock()

    def record(
        self,
        *,
        action: str,
        outcome: str,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.settings.audit_log_enabled:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "outcome": outcome,
            "actor": actor,
            "details": _redact_audit_value("details", details or {}),
        }
        self._append(payload)

    def _append(self, payload: dict[str, Any]) -> None:
        audit_path = self.settings.audit_log_path
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if not audit_path.exists():
            audit_path.touch()
        audit_path.chmod(0o600)
        line = json.dumps(payload, default=str)
        if self.encryptor is not None:
            line = self.encryptor.encrypt_text(line)
        with self._lock:
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")


_SECRET_KEY_PARTS = {
    "api_key",
    "authorization",
    "cookie",
    "password",
    "passphrase",
    "secret",
    "token",
}
_PATH_KEY_SUFFIXES = ("_path", "_paths", "_root", "_roots", "_dir", "_dirs")
_TEXT_KEY_PARTS = {"content", "prompt", "summary"}
_BLOB_KEY_PARTS = {"audio_base64", "content_base64", "file_bytes"}


def _redact_audit_value(key: str, value: Any) -> Any:
    normalized_key = key.casefold()
    if isinstance(value, dict):
        return {item_key: _redact_audit_value(str(item_key), item_value) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact_audit_value(key, item) for item in value]
    if isinstance(value, tuple):
        return [_redact_audit_value(key, item) for item in value]
    if not isinstance(value, str):
        return value
    if _is_secret_key(normalized_key):
        return _redacted_marker("secret", value)
    if _is_blob_key(normalized_key):
        return _redacted_marker("blob", value)
    if _is_path_key(normalized_key) or _looks_like_path(value):
        return _redacted_path(value)
    if _is_text_key(normalized_key) or _looks_like_large_text(value):
        return _redacted_marker("text", value)
    return value


def _is_secret_key(key: str) -> bool:
    return any(part in key for part in _SECRET_KEY_PARTS)


def _is_blob_key(key: str) -> bool:
    return any(part in key for part in _BLOB_KEY_PARTS)


def _is_path_key(key: str) -> bool:
    return key == "path" or key == "paths" or key == "roots" or key.endswith(_PATH_KEY_SUFFIXES)


def _is_text_key(key: str) -> bool:
    return any(part == key or key.endswith(f"_{part}") for part in _TEXT_KEY_PARTS)


def _looks_like_path(value: str) -> bool:
    if not value or value.startswith("enc::v1::"):
        return False
    if os.sep in value or (os.altsep and os.altsep in value):
        return True
    if len(value) > 2 and value[1:3] == ":\\":
        return True
    return value.startswith("./") or value.startswith("../") or value.startswith("~/")


def _looks_like_large_text(value: str) -> bool:
    return len(value) > 120 or "\n" in value


def _redacted_marker(kind: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"<redacted_{kind}:len={len(value)}:sha256={digest}>"


def _redacted_path(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    normalized = value.rstrip("/\\")
    tail = normalized.split("/")[-1].split("\\")[-1] if normalized else "root"
    return f"<path:{tail}:sha256={digest}>"
