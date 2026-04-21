"""File-scoping and local media validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal
import zipfile

from lewlm.core.errors import FileAccessError, UnsupportedMediaTypeError
from lewlm.documents.ir.models import DocumentIR, ImageBlock


JSON_FILE_SUFFIXES = {".json"}
PDF_FILE_SUFFIXES = {".pdf"}
DOCX_FILE_SUFFIXES = {".docx"}
XLSX_FILE_SUFFIXES = {".xlsx"}
IMAGE_SIGNATURES: tuple[tuple[str, tuple[bytes, ...]], ...] = (
    ("image/png", (b"\x89PNG\r\n\x1a\n",)),
    ("image/jpeg", (b"\xff\xd8\xff",)),
    ("image/gif", (b"GIF87a", b"GIF89a")),
)
AUDIO_SIGNATURES: tuple[tuple[str, tuple[bytes, ...]], ...] = (
    ("audio/flac", (b"fLaC",)),
    ("audio/ogg", (b"OggS",)),
    ("audio/mpeg", (b"ID3",)),
)
ScopedPathExpectation = Literal["file", "dir", "any"]


def normalize_roots(roots: Sequence[Path | str]) -> tuple[Path, ...]:
    return tuple(Path(root).expanduser().resolve(strict=False) for root in roots)


def resolve_scoped_path(
    path: Path | str,
    *,
    allowed_roots: Sequence[Path | str],
    purpose: str,
    base_dir: Path | str | None = None,
    must_exist: bool = True,
    expect: ScopedPathExpectation = "file",
) -> Path:
    _validate_expectation(expect)
    normalized_roots = normalize_roots(allowed_roots)
    if not normalized_roots:
        raise FileAccessError(f"{purpose} requires at least one allowed root.")
    resolved = _resolve_candidate(Path(path).expanduser(), normalized_roots=normalized_roots, base_dir=base_dir)
    if not any(_is_within_root(resolved, root) for root in normalized_roots):
        raise FileAccessError(
            f"{purpose} is outside the allowed filesystem scope.",
            details={"path": str(resolved), "allowed_roots": [str(root) for root in normalized_roots]},
        )
    if must_exist and not resolved.exists():
        raise FileAccessError(f"{purpose} does not exist.", details={"path": str(resolved)})
    if must_exist and expect == "file" and not resolved.is_file():
        raise FileAccessError(f"{purpose} must be a file.", details={"path": str(resolved)})
    if must_exist and expect == "dir" and not resolved.is_dir():
        raise FileAccessError(f"{purpose} must be a directory.", details={"path": str(resolved)})
    return resolved


def _validate_expectation(expect: ScopedPathExpectation | str) -> None:
    if expect not in {"file", "dir", "any"}:
        raise ValueError(f"Unsupported scoped-path expectation: {expect!r}.")


def read_scoped_text_file(
    path: Path | str,
    *,
    allowed_roots: Sequence[Path | str],
    purpose: str,
    media_type: str,
    base_dir: Path | str | None = None,
) -> tuple[Path, str]:
    resolved = resolve_scoped_path(
        path,
        allowed_roots=allowed_roots,
        purpose=purpose,
        base_dir=base_dir,
        must_exist=True,
        expect="file",
    )
    raw = resolved.read_bytes()
    if media_type == "application/json":
        _validate_json_bytes(raw, purpose=purpose, path=resolved)
    elif media_type in {"text/plain", "text/markdown"}:
        _validate_text_bytes(raw, purpose=purpose, path=resolved)
    else:
        raise UnsupportedMediaTypeError(
            "Unsupported local file validation media type.",
            details={"media_type": media_type, "path": str(resolved)},
        )
    return resolved, raw.decode("utf-8")


def validate_scoped_image_file(
    path: Path | str,
    *,
    allowed_roots: Sequence[Path | str],
    purpose: str,
    base_dir: Path | str | None = None,
) -> Path:
    resolved = resolve_scoped_path(
        path,
        allowed_roots=allowed_roots,
        purpose=purpose,
        base_dir=base_dir,
        must_exist=True,
        expect="file",
    )
    with resolved.open("rb") as handle:
        header = handle.read(16)
    if _matches_known_image_signature(header):
        return resolved
    raise UnsupportedMediaTypeError(
        "Image file does not match a supported signature.",
        details={"path": str(resolved)},
    )


def validate_scoped_binary_file(
    path: Path | str,
    *,
    allowed_roots: Sequence[Path | str],
    purpose: str,
    media_type: str,
    base_dir: Path | str | None = None,
) -> Path:
    resolved = resolve_scoped_path(
        path,
        allowed_roots=allowed_roots,
        purpose=purpose,
        base_dir=base_dir,
        must_exist=True,
        expect="file",
    )
    if media_type == "application/pdf":
        _validate_pdf_file(resolved, purpose=purpose)
    elif media_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        _validate_zip_file(resolved, purpose=purpose, suffixes=DOCX_FILE_SUFFIXES, expected_member="word/document.xml")
    elif media_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        _validate_zip_file(resolved, purpose=purpose, suffixes=XLSX_FILE_SUFFIXES, expected_member="xl/workbook.xml")
    else:
        raise UnsupportedMediaTypeError(
            "Unsupported local binary file validation media type.",
            details={"media_type": media_type, "path": str(resolved)},
        )
    return resolved


def validate_audio_bytes(
    raw: bytes,
    *,
    purpose: str,
    file_name: str | None = None,
) -> str:
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "audio/wav"
    if len(raw) >= 2 and raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    for media_type, signatures in AUDIO_SIGNATURES:
        if any(raw.startswith(signature) for signature in signatures):
            return media_type
    raise UnsupportedMediaTypeError(
        f"{purpose} does not match a supported audio signature.",
        details={"file_name": file_name},
    )


def scope_document_paths(
    document: DocumentIR,
    *,
    allowed_roots: Sequence[Path | str],
    base_dir: Path | str | None = None,
) -> DocumentIR:
    scoped_sections = []
    for section in document.sections:
        scoped_blocks = []
        for block in section.blocks:
            if isinstance(block, ImageBlock) and block.path is not None:
                scoped_path = validate_scoped_image_file(
                    block.path,
                    allowed_roots=allowed_roots,
                    purpose="Document image",
                    base_dir=base_dir,
                )
                scoped_blocks.append(block.model_copy(update={"path": str(scoped_path)}))
            else:
                scoped_blocks.append(block)
        scoped_sections.append(section.model_copy(update={"blocks": scoped_blocks}))
    return document.model_copy(update={"sections": scoped_sections})


def _resolve_candidate(
    path: Path,
    *,
    normalized_roots: tuple[Path, ...],
    base_dir: Path | str | None,
) -> Path:
    if path.is_absolute():
        return path.resolve(strict=False)
    if base_dir is not None:
        return (Path(base_dir).expanduser().resolve(strict=False) / path).resolve(strict=False)
    for root in normalized_roots:
        candidate = (root / path).resolve(strict=False)
        if candidate.exists():
            return candidate
    return (normalized_roots[0] / path).resolve(strict=False)


def _is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_text_bytes(raw: bytes, *, purpose: str, path: Path) -> None:
    if b"\x00" in raw:
        raise UnsupportedMediaTypeError(
            f"{purpose} must be a text file.",
            details={"path": str(path)},
        )
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedMediaTypeError(
            f"{purpose} must be UTF-8 text.",
            details={"path": str(path)},
        ) from exc


def _validate_json_bytes(raw: bytes, *, purpose: str, path: Path) -> None:
    if path.suffix.casefold() not in JSON_FILE_SUFFIXES:
        raise UnsupportedMediaTypeError(
            f"{purpose} must use a .json file.",
            details={"path": str(path)},
        )
    _validate_text_bytes(raw, purpose=purpose, path=path)
    stripped = raw.lstrip()
    if not stripped or stripped[:1] not in {b"{", b"["}:
        raise UnsupportedMediaTypeError(
            f"{purpose} must contain JSON text.",
            details={"path": str(path)},
        )


def _validate_pdf_file(path: Path, *, purpose: str) -> None:
    if path.suffix.casefold() not in PDF_FILE_SUFFIXES:
        raise UnsupportedMediaTypeError(
            f"{purpose} must use a .pdf file.",
            details={"path": str(path)},
        )
    with path.open("rb") as handle:
        header = handle.read(5)
    if not header.startswith(b"%PDF"):
        raise UnsupportedMediaTypeError(
            f"{purpose} does not match a PDF signature.",
            details={"path": str(path)},
        )


def _validate_zip_file(path: Path, *, purpose: str, suffixes: set[str], expected_member: str) -> None:
    if path.suffix.casefold() not in suffixes:
        raise UnsupportedMediaTypeError(
            f"{purpose} uses an unsupported file extension.",
            details={"path": str(path)},
        )
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise UnsupportedMediaTypeError(
            f"{purpose} does not contain a valid Office archive.",
            details={"path": str(path)},
        ) from exc
    if expected_member not in names:
        raise UnsupportedMediaTypeError(
            f"{purpose} does not match the expected Office document signature.",
            details={"path": str(path)},
        )


def _matches_known_image_signature(header: bytes) -> bool:
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return True
    for _, signatures in IMAGE_SIGNATURES:
        if any(header.startswith(signature) for signature in signatures):
            return True
    return False
