"""Best-effort local OCR helpers for document ingest."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from io import BytesIO


@dataclass(frozen=True, slots=True)
class OcrBackendStatus:
    """Availability details for the local OCR backend."""

    available: bool
    backend_name: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class OcrExtractionResult:
    """OCR attempt result for a single image."""

    text: str | None
    backend_name: str | None = None
    reason: str | None = None


def detect_ocr_backend() -> OcrBackendStatus:
    """Return the local OCR backend status."""

    try:
        import pytesseract
    except ImportError:
        return OcrBackendStatus(available=False, reason="The `pytesseract` dependency is not installed.")

    if shutil.which("tesseract") is None:
        return OcrBackendStatus(available=False, backend_name="pytesseract", reason="The `tesseract` binary is not installed.")

    try:
        pytesseract.get_tesseract_version()
    except RuntimeError as exc:
        return OcrBackendStatus(available=False, backend_name="pytesseract", reason=f"Tesseract is unavailable: {exc}")
    return OcrBackendStatus(available=True, backend_name="pytesseract+tesseract")


def perform_ocr_on_image_bytes(image_bytes: bytes, *, language: str = "eng") -> OcrExtractionResult:
    """Attempt OCR on an image payload using the local OCR backend."""

    backend = detect_ocr_backend()
    if not backend.available:
        return OcrExtractionResult(text=None, backend_name=backend.backend_name, reason=backend.reason)

    from PIL import Image, ImageOps
    import pytesseract

    with Image.open(BytesIO(image_bytes)) as image:
        normalized = ImageOps.exif_transpose(image).convert("L")
        normalized = normalized.point(lambda pixel: 0 if pixel < 180 else 255)
        try:
            extracted = pytesseract.image_to_string(normalized, lang=language, config="--psm 6")
        except RuntimeError as exc:
            return OcrExtractionResult(text=None, backend_name=backend.backend_name, reason=f"OCR extraction failed: {exc}")

    collapsed = _normalize_ocr_text(extracted)
    if not collapsed:
        return OcrExtractionResult(text=None, backend_name=backend.backend_name, reason="OCR completed but no text was detected.")
    return OcrExtractionResult(text=collapsed, backend_name=backend.backend_name)


def _normalize_ocr_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
