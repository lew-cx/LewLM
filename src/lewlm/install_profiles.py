"""Install-profile summaries for operator-facing readiness surfaces."""

from __future__ import annotations

import importlib.util
import platform

from pydantic import BaseModel, Field

from lewlm.documents.ingest.ocr import detect_ocr_backend


class InstallProfileStatus(BaseModel):
    """Machine-readable status for one documented install profile."""

    profile: str
    label: str
    extras: list[str] = Field(default_factory=list)
    install_spec: str
    installed: bool
    ready: bool
    summary: str
    notes: list[str] = Field(default_factory=list)


class InstallProfileSummary(BaseModel):
    """Current host summary for LewLM's documented install profiles."""

    active_profile_ids: list[str] = Field(default_factory=list)
    recommended_profile_id: str | None = None
    profiles: list[InstallProfileStatus] = Field(default_factory=list)


def summarize_install_profiles() -> InstallProfileSummary:
    """Return a host-local summary of the documented install profiles."""

    system = platform.system()
    machine = platform.machine().casefold()
    apple_silicon_host = system == "Darwin" and machine in {"arm64", "aarch64"}

    mlx_missing = _missing_modules(("mlx", "mlx_lm", "mlx_vlm", "mlx_audio"))
    gguf_missing = _missing_modules(("llama_cpp",))
    documents_missing = _missing_modules(("openpyxl", "PIL", "pytesseract", "pypdf", "docx", "reportlab", "weasyprint"))

    ocr_note: str | None = None
    if not documents_missing:
        ocr_status = detect_ocr_backend()
        if not ocr_status.available and ocr_status.reason:
            ocr_note = f"OCR-style flows still need a working local backend: {ocr_status.reason}"

    profiles = [
        InstallProfileStatus(
            profile="core_only",
            label="Core only",
            extras=[],
            install_spec=".",
            installed=True,
            ready=True,
            summary="Base middleware package with CLI, local API, registry, routing, and readiness surfaces but no optional runtime or document packages.",
            notes=[
                "Add a runtime profile before expecting local chat, embeddings, rerank, or audio execution.",
            ],
        ),
        InstallProfileStatus(
            profile="mlx_local_backend",
            label="MLX local app backend",
            extras=["mlx"],
            install_spec=".[mlx]",
            installed=not mlx_missing,
            ready=apple_silicon_host and not mlx_missing,
            summary="Apple Silicon runtime profile for MLX text, multimodal, and audio adapters on supported hosts.",
            notes=[
                *(
                    ["Requires macOS on Apple Silicon to become ready on this host."]
                    if not apple_silicon_host
                    else []
                ),
                *([f"Missing Python modules: {', '.join(mlx_missing)}"] if mlx_missing else []),
            ],
        ),
        InstallProfileStatus(
            profile="gguf_fallback_backend",
            label="GGUF fallback backend",
            extras=["llamacpp"],
            install_spec=".[llamacpp]",
            installed=not gguf_missing,
            ready=not gguf_missing,
            summary="Cross-platform GGUF runtime profile backed by llama.cpp for LewLM's main fallback serving path.",
            notes=[f"Missing Python modules: {', '.join(gguf_missing)}"] if gguf_missing else [],
        ),
        InstallProfileStatus(
            profile="documents_enabled_backend",
            label="Documents-enabled backend",
            extras=["documents"],
            install_spec=".[documents]",
            installed=not documents_missing,
            ready=not documents_missing,
            summary="Document ingest, rendering, and transform profile for PDF, DOCX, XLSX, OCR-oriented, and deterministic artifact workflows.",
            notes=[
                *([f"Missing Python modules: {', '.join(documents_missing)}"] if documents_missing else []),
                *([ocr_note] if ocr_note else []),
                "This profile is additive; pair it with an inference runtime profile when you also want local model execution.",
            ],
        ),
    ]
    return InstallProfileSummary(
        active_profile_ids=[profile.profile for profile in profiles if profile.installed],
        recommended_profile_id="mlx_local_backend" if apple_silicon_host else "gguf_fallback_backend",
        profiles=profiles,
    )


def _missing_modules(module_names: tuple[str, ...]) -> list[str]:
    return [module_name for module_name in module_names if importlib.util.find_spec(module_name) is None]
