"""Install-profile summaries for operator-facing readiness surfaces."""

from __future__ import annotations

import importlib.util
import platform
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from lewlm.documents.ingest.ocr import detect_ocr_backend

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_EXTERNAL_ACCELERATOR_PROFILE_NOTES = {
    "openai_compatible": "Generic OpenAI-compatible bridge profile for a local loopback server.",
    "vmlx": "MLX-oriented bridge profile for a compatible local loopback server.",
    "omlx": "Optimized-MLX bridge profile for a compatible local loopback server.",
    "vllm_mlx": "vLLM-style bridge profile for a compatible local loopback server.",
    "vllm_local": "vLLM-style bridge profile for a compatible local loopback server.",
    "sglang_local": "SGLang-style bridge profile for a compatible local loopback server.",
}


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
    notes: list[str] = Field(default_factory=list)


def summarize_install_profiles(settings: Any | None = None) -> InstallProfileSummary:
    """Return a host-local summary of the documented install profiles."""

    system = platform.system()
    machine = platform.machine()
    normalized_machine = machine.casefold()
    apple_silicon_host = system == "Darwin" and normalized_machine in {"arm64", "aarch64"}
    gguf_host_supported = system in {"Darwin", "Linux", "Windows"}
    host_label = f"{system} {machine}".strip()

    mlx_missing = _missing_modules(("mlx", "mlx_lm", "mlx_vlm", "mlx_audio"))
    gguf_missing = _missing_modules(("llama_cpp",))
    documents_missing = _missing_modules(("openpyxl", "PIL", "pytesseract", "pypdf", "docx", "reportlab", "weasyprint"))
    external_enabled = bool(getattr(settings, "external_accelerator_enabled", False))
    external_base_url = getattr(settings, "external_accelerator_base_url", None)
    external_profile = str(getattr(settings, "external_accelerator_profile", "openai_compatible"))

    ocr_note: str | None = None
    if not documents_missing:
        ocr_status = detect_ocr_backend()
        if not ocr_status.available and ocr_status.reason:
            ocr_note = f"OCR-style flows still need a working local backend: {ocr_status.reason}"

    summary_notes: list[str] = []
    if apple_silicon_host:
        summary_notes.append("This Apple Silicon host can use MLX as LewLM's first-class local runtime profile.")
    elif gguf_host_supported:
        summary_notes.append(
            f"{host_label} should use the cross-platform GGUF profile as LewLM's first-class non-Apple runtime family.",
        )
        summary_notes.append(
            "External accelerators remain the supported bridge path for compatible loopback-only local servers, including NVIDIA-oriented operators, and do not replace the packaged GGUF/llama.cpp default when it is available.",
        )
    else:
        summary_notes.append(
            f"{host_label} is outside LewLM's documented local runtime host matrix; readiness notes below stay conservative on purpose.",
        )

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
                "Add Apple MLX, cross-platform GGUF, or the external accelerator bridge before expecting local chat, embeddings, rerank, or audio execution.",
            ],
        ),
        InstallProfileStatus(
            profile="mlx_local_backend",
            label="Apple MLX local backend",
            extras=["mlx"],
            install_spec=".[mlx]",
            installed=not mlx_missing,
            ready=apple_silicon_host and not mlx_missing,
            summary="Apple Silicon runtime profile for LewLM's first-class MLX text, vision, and audio adapters on supported hosts.",
            notes=[
                *(["Recommended packaged runtime on Apple Silicon hosts."] if apple_silicon_host else []),
                *(
                    [
                        (
                            "Requires macOS on Apple Silicon to become ready on this host. "
                            "Use `.[llamacpp]` for the documented cross-platform packaged runtime path on non-MLX hosts."
                        ),
                    ]
                    if not apple_silicon_host
                    else []
                ),
                *([f"Missing Python modules: {', '.join(mlx_missing)}"] if mlx_missing else []),
            ],
        ),
        InstallProfileStatus(
            profile="gguf_fallback_backend",
            label="Cross-platform GGUF backend",
            extras=["llamacpp"],
            install_spec=".[llamacpp]",
            installed=not gguf_missing,
            ready=gguf_host_supported and not gguf_missing,
            summary="Cross-platform packaged GGUF runtime profile backed by llama.cpp; LewLM's first-class non-Apple runtime family.",
            notes=[
                *(["Recommended first-class non-Apple runtime on Linux and Windows today."] if system in {"Linux", "Windows"} else []),
                *(
                    [
                        f"Current host {host_label} is outside the documented GGUF support matrix (macOS, Linux, Windows).",
                    ]
                    if not gguf_host_supported
                    else []
                ),
                *([f"Missing Python modules: {', '.join(gguf_missing)}"] if gguf_missing else []),
                *(
                    [f"Install `.[llamacpp]` for the documented local runtime path on {host_label}."]
                    if gguf_host_supported and gguf_missing
                    else []
                ),
            ],
        ),
        InstallProfileStatus(
            profile="external_accelerator_bridge_backend",
            label="Cross-platform external accelerator bridge",
            extras=[],
            install_spec=".",
            installed=external_enabled,
            ready=external_enabled and _is_loopback_url(external_base_url) and system in {"Darwin", "Linux", "Windows"},
            summary=(
                "Bridge profile for LewLM talking to a loopback-only OpenAI-compatible local server "
                "instead of importing an in-process runtime package."
            ),
            notes=_external_accelerator_notes(
                system=system,
                host_label=host_label,
                apple_silicon_host=apple_silicon_host,
                enabled=external_enabled,
                base_url=external_base_url,
                profile=external_profile,
            ),
        ),
        InstallProfileStatus(
            profile="documents_enabled_backend",
            label="Documents add-on",
            extras=["documents"],
            install_spec=".[documents]",
            installed=not documents_missing,
            ready=not documents_missing,
            summary="Document ingest, rendering, and transform profile for PDF, DOCX, XLSX, OCR-oriented, and deterministic artifact workflows.",
            notes=[
                *([f"Missing Python modules: {', '.join(documents_missing)}"] if documents_missing else []),
                *([ocr_note] if ocr_note else []),
                "Python extras do not install the OCR engine itself; verify the local backend separately when you need OCR workflows.",
                "This profile is additive; pair it with Apple MLX, cross-platform GGUF, or the external accelerator bridge when you also want local model execution.",
            ],
        ),
    ]
    return InstallProfileSummary(
        active_profile_ids=[profile.profile for profile in profiles if profile.installed],
        recommended_profile_id=(
            "mlx_local_backend"
            if apple_silicon_host
            else "gguf_fallback_backend"
            if gguf_host_supported
            else None
        ),
        profiles=profiles,
        notes=summary_notes,
    )


def _missing_modules(module_names: tuple[str, ...]) -> list[str]:
    return [module_name for module_name in module_names if importlib.util.find_spec(module_name) is None]


def _is_loopback_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        return False
    return parsed.hostname in _LOOPBACK_HOSTS


def _external_accelerator_notes(
    *,
    system: str,
    host_label: str,
    apple_silicon_host: bool,
    enabled: bool,
    base_url: str | None,
    profile: str,
) -> list[str]:
    notes = [
        "Use this bridge when LewLM should front a loopback-only OpenAI-compatible local server rather than importing a runtime package directly.",
        _EXTERNAL_ACCELERATOR_PROFILE_NOTES.get(
            profile,
            f"Configured bridge profile: `{profile}`.",
        ),
        "LewLM does not install or manage the external server itself.",
        "Text and streaming use `/v1/chat/completions`; image-conditioned chat uses the same route with OpenAI-style image content blocks.",
        "Audio execution depends on the loopback server exposing `/v1/audio/transcriptions` and `/v1/audio/speech`.",
        "LewLM reports narrower bridge depth explicitly; it does not claim MLX-level multimodal optimization or telemetry parity through this path.",
    ]
    if system in {"Linux", "Windows"}:
        notes.append(
            "Linux and Windows operators, including NVIDIA-backed local servers that expose an OpenAI-compatible loopback endpoint, fit this bridge path."
        )
    notes.append(
        "Semantic text support on this bridge is adapter-backed: embeddings require a compatible local `/v1/embeddings` endpoint, and rerank requires a compatible local `/v1/rerank` endpoint or equivalent extension."
    )
    if not apple_silicon_host:
        notes.append(
            "Use the packaged GGUF runtime when you want LewLM-managed local execution; use this bridge when you already run the local server yourself."
        )
        notes.append(
            "Benchmarks and probes can still record bridge evidence here, but LewLM keeps the packaged GGUF/llama.cpp path as the first-class non-Apple default when that runtime is available."
        )
    if not enabled:
        notes.append("Set LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true to activate this bridge profile.")
        return notes
    notes.append(f"Configured bridge profile: `{profile}`.")
    if base_url is None:
        notes.append("Set LEWLM_EXTERNAL_ACCELERATOR_BASE_URL to a loopback URL such as http://127.0.0.1:8000.")
    elif not _is_loopback_url(base_url):
        notes.append("LEWLM_EXTERNAL_ACCELERATOR_BASE_URL must target a loopback-only local host.")
    else:
        notes.append(f"{host_label} is configured for a loopback-only external accelerator endpoint.")
    return notes
