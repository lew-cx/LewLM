"""Helpers for opt-in encrypted local persistence."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import TypeAlias

from lewlm.config.settings import LewLMSettings
from lewlm.core.errors import ConfigurationError, StorageError


ENCRYPTED_VALUE_PREFIX = "enc::v1::"
ENCRYPTED_FILE_MAGIC = b"LEWLMF1\0"
ENCRYPTED_FILE_NONCE_BYTES = 12
ENCRYPTED_FILE_TAG_BYTES = 16
ENCRYPTED_FILE_CHUNK_BYTES = 1024 * 1024
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def is_encrypted_value(value: str | None) -> bool:
    return bool(value) and value.startswith(ENCRYPTED_VALUE_PREFIX)


class PersistenceEncryptor:
    """Encrypt persisted payloads with a passphrase-derived key."""

    def __init__(self, settings: LewLMSettings) -> None:
        self.settings = settings
        if not settings.persistence_encryption_enabled:
            raise ConfigurationError("PersistenceEncryptor requires persistence encryption to be enabled.")
        passphrase = settings.persistence_encryption_passphrase
        if passphrase is None:
            raise ConfigurationError("Persistence encryption requires a passphrase.")
        self._base_key = self._derive_base_key(passphrase.get_secret_value())
        self._fernet = self._build_fernet(self._base_key)

    def encrypt_text(self, value: str) -> str:
        ciphertext = self._fernet.encrypt(value.encode("utf-8")).decode("utf-8")
        return f"{ENCRYPTED_VALUE_PREFIX}{ciphertext}"

    def decrypt_text(self, value: str) -> str:
        if not is_encrypted_value(value):
            return value
        try:
            return self._fernet.decrypt(value.removeprefix(ENCRYPTED_VALUE_PREFIX).encode("utf-8")).decode("utf-8")
        except self._invalid_token_exception() as exc:
            raise StorageError(
                "Encrypted persistence value could not be decrypted.",
                details={"persistence_salt_path": str(self.settings.persistence_salt_path)},
            ) from exc

    def encrypt_json(self, value: JSONValue) -> str:
        return self.encrypt_text(json.dumps(value))

    def decrypt_json(self, value: str) -> JSONValue:
        return json.loads(self.decrypt_text(value))

    def stable_digest(self, value: str) -> str:
        return hmac.new(self._base_key, value.encode("utf-8"), hashlib.sha256).hexdigest()

    def encrypt_file(self, source_path: Path, target_path: Path) -> None:
        cipher = self._build_cipher
        nonce = os.urandom(ENCRYPTED_FILE_NONCE_BYTES)
        encryptor = cipher(self._base_key, nonce=nonce)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open("rb") as source_handle, target_path.open("wb") as target_handle:
            target_handle.write(ENCRYPTED_FILE_MAGIC)
            target_handle.write(nonce)
            while True:
                chunk = source_handle.read(ENCRYPTED_FILE_CHUNK_BYTES)
                if not chunk:
                    break
                target_handle.write(encryptor.update(chunk))
            target_handle.write(encryptor.finalize())
            target_handle.write(encryptor.tag)
        target_path.chmod(0o600)

    def decrypt_file(self, source_path: Path, target_path: Path) -> None:
        cipher = self._build_cipher
        file_size = source_path.stat().st_size
        header_size = len(ENCRYPTED_FILE_MAGIC) + ENCRYPTED_FILE_NONCE_BYTES
        minimum_size = header_size + ENCRYPTED_FILE_TAG_BYTES
        if file_size < minimum_size:
            raise StorageError(
                "Encrypted file payload is truncated.",
                details={"path": str(source_path)},
            )

        with source_path.open("rb") as source_handle:
            magic = source_handle.read(len(ENCRYPTED_FILE_MAGIC))
            if magic != ENCRYPTED_FILE_MAGIC:
                raise StorageError(
                    "Encrypted file payload used an unknown format.",
                    details={"path": str(source_path)},
                )
            nonce = source_handle.read(ENCRYPTED_FILE_NONCE_BYTES)
            source_handle.seek(file_size - ENCRYPTED_FILE_TAG_BYTES)
            tag = source_handle.read(ENCRYPTED_FILE_TAG_BYTES)
            source_handle.seek(header_size)
            decryptor = cipher(self._base_key, nonce=nonce, tag=tag)
            remaining = file_size - minimum_size

            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target_path.open("wb") as target_handle:
                    while remaining > 0:
                        chunk = source_handle.read(min(ENCRYPTED_FILE_CHUNK_BYTES, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        target_handle.write(decryptor.update(chunk))
                    target_handle.write(decryptor.finalize())
            except self._invalid_tag_exception() as exc:
                if target_path.exists():
                    target_path.unlink()
                raise StorageError(
                    "Encrypted file payload could not be decrypted.",
                    details={"path": str(source_path)},
                ) from exc
        target_path.chmod(0o600)

    def _derive_base_key(self, passphrase: str) -> bytes:
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        except ImportError as exc:
            raise ConfigurationError("Persistence encryption requires the `cryptography` dependency.") from exc

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._load_or_create_salt(self.settings.persistence_salt_path),
            iterations=max(100_000, self.settings.persistence_encryption_kdf_iterations),
        )
        return kdf.derive(passphrase.encode("utf-8"))

    def _build_fernet(self, base_key: bytes):
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise ConfigurationError("Persistence encryption requires the `cryptography` dependency.") from exc
        return Fernet(base64.urlsafe_b64encode(base_key))

    def _build_cipher(self, base_key: bytes, *, nonce: bytes, tag: bytes | None = None):
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError as exc:
            raise ConfigurationError("Persistence encryption requires the `cryptography` dependency.") from exc
        mode = modes.GCM(nonce, tag) if tag is not None else modes.GCM(nonce)
        return Cipher(algorithms.AES(base_key), mode).encryptor() if tag is None else Cipher(algorithms.AES(base_key), mode).decryptor()

    def _invalid_token_exception(self):
        try:
            from cryptography.fernet import InvalidToken
        except ImportError as exc:
            raise ConfigurationError("Persistence encryption requires the `cryptography` dependency.") from exc
        return InvalidToken

    def _invalid_tag_exception(self):
        try:
            from cryptography.exceptions import InvalidTag
        except ImportError as exc:
            raise ConfigurationError("Persistence encryption requires the `cryptography` dependency.") from exc
        return InvalidTag

    @staticmethod
    def _load_or_create_salt(salt_path: Path) -> bytes:
        salt_path.parent.mkdir(parents=True, exist_ok=True)
        if salt_path.exists():
            salt_path.chmod(0o600)
            return salt_path.read_bytes()

        salt = os.urandom(16)
        try:
            with salt_path.open("xb") as handle:
                handle.write(salt)
        except FileExistsError:
            salt = salt_path.read_bytes()
        salt_path.chmod(0o600)
        return salt
