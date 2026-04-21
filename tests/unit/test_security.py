from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest

from lewlm.core.errors import FileAccessError, SandboxExecutionError, ToolAuthorizationError, UnsupportedMediaTypeError
from lewlm.security.audit import AuditLogger
from lewlm.security.authorization import ToolAction, ToolAuthorizer
from lewlm.security.persistence import ENCRYPTED_FILE_MAGIC, PersistenceEncryptor
from lewlm.documents.ir.models import DocumentIR, DocumentSection, ImageBlock
from lewlm.security.files import read_scoped_text_file, resolve_scoped_path, scope_document_paths, validate_audio_bytes
from lewlm.security.sandbox import run_in_subprocess


def _sandbox_echo(value: str) -> str:
    return value


def _sandbox_sleep(_: str) -> str:
    time.sleep(2)
    return "done"


def _sandbox_environment_snapshot(variable_name: str) -> dict[str, str | None]:
    return {
        "cwd": os.getcwd(),
        "custom_value": os.environ.get(variable_name),
        "path": os.environ.get("PATH"),
    }


def test_read_scoped_text_file_rejects_out_of_scope_json(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_path = tmp_path / "outside.json"
    outside_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileAccessError):
        read_scoped_text_file(
            outside_path,
            allowed_roots=(allowed_root,),
            purpose="JSON input",
            media_type="application/json",
        )


def test_read_scoped_text_file_rejects_binary_prompt(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_bytes(b"\x00binary")

    with pytest.raises(UnsupportedMediaTypeError):
        read_scoped_text_file(
            prompt_path,
            allowed_roots=(tmp_path,),
            purpose="System prompt",
            media_type="text/plain",
        )


def test_scope_document_paths_resolves_relative_images_within_root(tmp_path: Path) -> None:
    image_path = tmp_path / "assets" / "chart.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    document = DocumentIR(
        title="Image Report",
        sections=[
            DocumentSection(
                heading="Assets",
                blocks=[ImageBlock(alt_text="Chart", path="assets/chart.png")],
            ),
        ],
    )

    scoped = scope_document_paths(document, allowed_roots=(tmp_path,), base_dir=tmp_path)

    image_block = scoped.sections[0].blocks[0]
    assert isinstance(image_block, ImageBlock)
    assert image_block.path == str(image_path.resolve(strict=False))


def test_resolve_scoped_path_supports_any_expectation_for_files_and_directories(tmp_path: Path) -> None:
    nested_dir = tmp_path / "bundle"
    nested_dir.mkdir()
    nested_file = nested_dir / "note.txt"
    nested_file.write_text("hello", encoding="utf-8")

    assert resolve_scoped_path(nested_file, allowed_roots=(tmp_path,), purpose="Scoped file", expect="any") == nested_file
    assert resolve_scoped_path(nested_dir, allowed_roots=(tmp_path,), purpose="Scoped dir", expect="any") == nested_dir


def test_resolve_scoped_path_rejects_unknown_expectation(tmp_path: Path) -> None:
    document = tmp_path / "note.txt"
    document.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported scoped-path expectation"):
        resolve_scoped_path(document, allowed_roots=(tmp_path,), purpose="Scoped file", expect="bogus")  # type: ignore[arg-type]


def test_tool_authorizer_requires_explicit_action(tool_authorized_settings) -> None:
    authorizer = ToolAuthorizer(
        settings=tool_authorized_settings,
        audit_logger=AuditLogger(tool_authorized_settings),
    )

    with pytest.raises(ToolAuthorizationError):
        authorizer.require(ToolAction.DOCUMENT_GENERATE, authorizations=[], actor="test")

    authorizer.require(
        ToolAction.DOCUMENT_GENERATE,
        authorizations=[ToolAction.DOCUMENT_GENERATE.value],
        actor="test",
    )

    audit_events = [
        json.loads(line)
        for line in tool_authorized_settings.audit_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["action"] == "document_generate" and event["outcome"] == "denied" for event in audit_events)
    assert any(event["action"] == "document_generate" and event["outcome"] == "authorized" for event in audit_events)


def test_run_in_subprocess_enforces_timeout() -> None:
    assert run_in_subprocess(
        _sandbox_echo,
        "ready",
        operation="sandbox test",
        timeout_seconds=2,
        enabled=True,
    ) == "ready"

    with pytest.raises(SandboxExecutionError):
        run_in_subprocess(
            _sandbox_sleep,
            "ignored",
            operation="sandbox timeout test",
            timeout_seconds=1,
            enabled=True,
        )


def test_run_in_subprocess_uses_ephemeral_workspace_and_clears_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace_root = tmp_path / "sandbox-root"
    monkeypatch.setenv("LEWLM_SECRET_TOKEN", "keep-out")

    snapshot = run_in_subprocess(
        _sandbox_environment_snapshot,
        "LEWLM_SECRET_TOKEN",
        operation="sandbox isolation test",
        timeout_seconds=2,
        enabled=True,
        clear_environment=True,
        workspace_root=workspace_root,
    )

    sandbox_cwd = Path(snapshot["cwd"] or "")
    assert snapshot["custom_value"] is None
    assert snapshot["path"]
    assert sandbox_cwd.parent == workspace_root
    assert not sandbox_cwd.exists()


def test_audit_logger_redacts_paths_and_prompt_content(tool_authorized_settings) -> None:
    logger = AuditLogger(tool_authorized_settings)

    logger.record(
        action="prompt_override",
        outcome="applied",
        actor="test",
        details={
            "input_path": "/tmp/lewlm/private/request.json",
            "developer_prompt": "Keep this private and do not leak the prompt body.",
            "api_key": "super-secret",
            "selected_template": "structured_output",
        },
    )

    event = json.loads(tool_authorized_settings.audit_log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["details"]["input_path"].startswith("<path:request.json:")
    assert event["details"]["developer_prompt"].startswith("<redacted_text:")
    assert event["details"]["api_key"].startswith("<redacted_secret:")
    assert event["details"]["selected_template"] == "structured_output"


def test_audit_logger_encrypts_lines_when_enabled(encrypted_persistence_settings) -> None:
    encryptor = PersistenceEncryptor(encrypted_persistence_settings)
    logger = AuditLogger(encrypted_persistence_settings, encryptor=encryptor)

    logger.record(action="document_generate", outcome="success", actor="test", details={"file_name": "report.csv"})

    raw_line = encrypted_persistence_settings.audit_log_path.read_text(encoding="utf-8").strip()
    assert raw_line.startswith("enc::v1::")

    payload = json.loads(encryptor.decrypt_text(raw_line))
    assert payload["action"] == "document_generate"
    assert payload["outcome"] == "success"


def test_persistence_encryptor_round_trips_files(encrypted_persistence_settings, tmp_path: Path) -> None:
    encryptor = PersistenceEncryptor(encrypted_persistence_settings)
    source_path = tmp_path / "artifact.txt"
    encrypted_path = tmp_path / "artifact.lewlmcache"
    decrypted_path = tmp_path / "artifact.out"

    source_path.write_text("secret artifact payload", encoding="utf-8")
    encryptor.encrypt_file(source_path, encrypted_path)

    assert encrypted_path.read_bytes().startswith(ENCRYPTED_FILE_MAGIC)

    encryptor.decrypt_file(encrypted_path, decrypted_path)
    assert decrypted_path.read_text(encoding="utf-8") == "secret artifact payload"


def test_validate_audio_bytes_accepts_wav_signature() -> None:
    wav_bytes = b"RIFF\x2a\x00\x00\x00WAVEfmt " + b"\x00" * 16

    assert validate_audio_bytes(wav_bytes, purpose="Audio input", file_name="sample.wav") == "audio/wav"
