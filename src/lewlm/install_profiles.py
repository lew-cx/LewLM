"""Install-profile summaries for operator-facing readiness surfaces."""

from __future__ import annotations

import importlib.util
import platform
import shutil
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from lewlm.core.contracts import RuntimeSupportPath, StandardsAcceptanceContract, build_standards_acceptance_contract
from lewlm.documents.ingest.ocr import detect_ocr_backend

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_EXTERNAL_ACCELERATOR_PROFILE_NOTES = {
    "openai_compatible": "Generic OpenAI-compatible bridge profile for a local loopback server.",
    "vmlx": "MLX-oriented bridge profile for a compatible local loopback server.",
    "omlx": "Optimized-MLX bridge profile for a compatible local loopback server.",
    "vllm_mlx": "vLLM-style bridge profile for a compatible local loopback server.",
    "vllm_local": "vLLM-style bridge profile for a compatible local loopback server.",
    "sglang_local": "SGLang-style bridge profile for a compatible local loopback server.",
    "tensorrt_llm_server": "TensorRT-LLM bridge profile for a compatible local loopback server.",
    "openvino_model_server": "OpenVINO Model Server bridge profile for a compatible local loopback server.",
    "ollama_local": "Ollama-compatible bridge profile for a local loopback server that preserves the generic OpenAI-compatible contract.",
    "llamacpp_server": "llama.cpp-server-compatible bridge profile for a local loopback server that preserves the generic OpenAI-compatible contract.",
}


def _has_command(command: str) -> bool:
    return shutil.which(command) is not None


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
    recommended_feature_paths: list["FeaturePathRecommendation"] = Field(default_factory=list)
    standards_acceptance_contract: StandardsAcceptanceContract = Field(
        default_factory=build_standards_acceptance_contract,
    )
    profiles: list[InstallProfileStatus] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class FeaturePathRecommendation(BaseModel):
    """Recommended current-host path for one public operator-facing feature class."""

    feature_class: str
    profile: str
    label: str
    support_path: RuntimeSupportPath
    summary: str
    fallback_guidance: list[str] = Field(default_factory=list)


InstallProfileSummary.model_rebuild()


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
    onnx_genai_missing = _missing_modules(("onnxruntime_genai",))
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
            "External accelerators remain the supported bridge path for compatible loopback-only local servers, including NVIDIA-oriented operators, and do not replace the packaged GGUF/llama.cpp default when it is available. Non-Apple audio transcription and speech still use this bridge path, while semantic GGUF models can stay packaged on the llama.cpp path.",
        )
        if system == "Windows":
            summary_notes.append(
                "ONNX Runtime GenAI is tracked as LewLM's Windows-native DirectML/CUDA/CPU candidate path for already-prepared ONNX bundles; model-specific load and generation probes still decide evidence strength.",
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
            summary=(
                "Cross-platform packaged GGUF runtime profile backed by llama.cpp; "
                "LewLM's first-class non-Apple runtime family for chat, embeddings, and packaged rerank fallback."
            ),
            notes=[
                *(["Recommended first-class non-Apple runtime on Linux and Windows today."] if system in {"Linux", "Windows"} else []),
                *(
                    [
                        "Packaged GGUF support on non-Apple hosts covers LewLM's first-class text runtime family plus embedding-capable semantic GGUF models; rerank stays packaged through LewLM's embedding-similarity fallback when the backend lacks a native rerank API. Non-Apple audio transcription and speech intentionally remain bridge-only through the external accelerator path.",
                    ]
                    if system in {"Linux", "Windows"}
                    else []
                ),
                *(
                    [
                        "On Windows, the `llamacpp` extra also installs CMake and Ninja helper packages, but source builds still require Microsoft C++ Build Tools when no prebuilt llama-cpp-python wheel is available.",
                    ]
                    if system == "Windows"
                    else []
                ),
                *(
                    [
                        "CMake is not currently on PATH. Existing Windows environments may still need it available before retrying a local llama.cpp source build.",
                    ]
                    if system == "Windows" and gguf_missing and not _has_command("cmake")
                    else []
                ),
                *(
                    [
                        "Ninja is optional for faster Windows llama.cpp source builds.",
                    ]
                    if system == "Windows" and gguf_missing and not _has_command("ninja")
                    else []
                ),
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
            profile="onnx_genai_backend",
            label="ONNX Runtime GenAI backend",
            extras=["onnx_genai"],
            install_spec=".[onnx_genai]",
            installed=not onnx_genai_missing,
            ready=system in {"Darwin", "Linux", "Windows"} and not onnx_genai_missing,
            summary=(
                "Packaged ONNX Runtime GenAI profile for already-prepared model bundles, with Windows-native CPU, DirectML, "
                "and CUDA execution-provider planning metadata."
            ),
            notes=[
                *(
                    [
                        "On Windows, this is the DirectML-native prepared-bundle route alongside the stable GGUF baseline.",
                    ]
                    if system == "Windows"
                    else []
                ),
                "LewLM can load and generate from compatible ONNX GenAI bundles when the installed package exposes the expected Python API; per-model probes and benchmarks still upgrade evidence beyond package readiness.",
                *([f"Missing Python modules: {', '.join(onnx_genai_missing)}"] if onnx_genai_missing else []),
                *(
                    ["Install `.[onnx_genai]` to prepare ONNX Runtime GenAI bundle probing on this host."]
                    if onnx_genai_missing
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
        recommended_feature_paths=_recommended_feature_paths(
            system=system,
            apple_silicon_host=apple_silicon_host,
            gguf_host_supported=gguf_host_supported,
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
        "Audio execution depends on the loopback server exposing `/v1/audio/transcriptions` and `/v1/audio/speech`, and LewLM probes those bridge endpoints separately.",
        "This bridge is LewLM's current bridge-only non-Apple public audio parity path for transcription and speech requests.",
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


def _recommended_feature_paths(
    *,
    system: str,
    apple_silicon_host: bool,
    gguf_host_supported: bool,
) -> list[FeaturePathRecommendation]:
    if apple_silicon_host:
        return [
            FeaturePathRecommendation(
                feature_class="chat",
                profile="mlx_local_backend",
                label="Apple MLX local backend",
                support_path=RuntimeSupportPath.PACKAGED,
                summary="Use the packaged Apple MLX path as the default chat and streaming route on Apple Silicon macOS.",
                fallback_guidance=[
                    "If the MLX profile is unavailable on this host, switch to the packaged GGUF profile for chat.",
                ],
            ),
            FeaturePathRecommendation(
                feature_class="semantic_text",
                profile="mlx_local_backend",
                label="Apple MLX local backend",
                support_path=RuntimeSupportPath.PACKAGED,
                summary="Use the Apple MLX path for embeddings and rerank on Apple Silicon macOS.",
                fallback_guidance=[
                    "If you need a loopback semantic backend instead, use the external accelerator bridge with compatible `/v1/embeddings` and `/v1/rerank` endpoints.",
                ],
            ),
            FeaturePathRecommendation(
                feature_class="vision",
                profile="mlx_local_backend",
                label="Apple MLX local backend",
                support_path=RuntimeSupportPath.PACKAGED,
                summary="Use the Apple MLX vision path for image-conditioned chat on Apple Silicon macOS.",
                fallback_guidance=[
                    "If MLX vision is unavailable, the external accelerator bridge can still serve OpenAI-style image content blocks through LewLM's shared chat surfaces.",
                ],
            ),
            FeaturePathRecommendation(
                feature_class="audio",
                profile="mlx_local_backend",
                label="Apple MLX local backend",
                support_path=RuntimeSupportPath.PACKAGED,
                summary="Use the Apple MLX audio path for transcription and speech on Apple Silicon macOS.",
                fallback_guidance=[
                    "If MLX audio is unavailable, the external accelerator bridge can serve audio through compatible local `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints.",
                ],
            ),
            FeaturePathRecommendation(
                feature_class="structured_output",
                profile="gguf_fallback_backend",
                label="Cross-platform GGUF backend",
                support_path=RuntimeSupportPath.PACKAGED,
                summary="Use the packaged GGUF/llama.cpp path when you need decode-time JSON-schema or grammar enforcement; Apple MLX remains prompt-guided fallback for structured output.",
                fallback_guidance=[
                    "If GGUF is unavailable, LewLM still records structured-output contracts on MLX, but enforcement falls back to prompt-guided generation plus validation metadata.",
                ],
            ),
        ]
    if not gguf_host_supported:
        return []
    return [
        FeaturePathRecommendation(
            feature_class="chat",
            profile="gguf_fallback_backend",
            label="Cross-platform GGUF backend",
            support_path=RuntimeSupportPath.PACKAGED,
            summary="Use the packaged GGUF/llama.cpp path as the default chat and streaming route on this host.",
            fallback_guidance=[
                "If another loopback server already owns execution, the external accelerator bridge remains a supported adapter-backed alternative.",
            ],
        ),
        FeaturePathRecommendation(
            feature_class="semantic_text",
            profile="gguf_fallback_backend",
            label="Cross-platform GGUF backend",
            support_path=RuntimeSupportPath.PACKAGED,
            summary="Use the packaged GGUF/llama.cpp path for embeddings and packaged rerank fallback on this host.",
            fallback_guidance=[
                "Use compatible embedding or rerank GGUF models for semantic text; rerank stays honest by using packaged embedding-similarity fallback when the backend does not expose a native rerank API.",
                "If another loopback server already owns semantic execution, the external accelerator bridge remains a supported adapter-backed alternative with compatible `/v1/embeddings` and `/v1/rerank` endpoints.",
            ],
        ),
        FeaturePathRecommendation(
            feature_class="vision",
            profile="external_accelerator_bridge_backend",
            label="Cross-platform external accelerator bridge",
            support_path=RuntimeSupportPath.BRIDGE,
            summary=(
                "Use the loopback external accelerator bridge for bridge-only image-conditioned chat on this host; "
                "LewLM does not claim packaged non-Apple vision parity here."
            ),
            fallback_guidance=[
                "The local server must accept OpenAI-style image content blocks on `/v1/chat/completions`.",
            ],
        ),
        FeaturePathRecommendation(
            feature_class="audio",
            profile="external_accelerator_bridge_backend",
            label="Cross-platform external accelerator bridge",
            support_path=RuntimeSupportPath.BRIDGE,
            summary="Use the loopback external accelerator bridge for the current bridge-only transcription and speech path on this host.",
            fallback_guidance=[
                "The local server must expose compatible `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints.",
                "LewLM probes the transcription and speech bridge endpoints separately and keeps packaged-vs-bridge readiness explicit.",
            ],
        ),
        FeaturePathRecommendation(
            feature_class="structured_output",
            profile="gguf_fallback_backend",
            label="Cross-platform GGUF backend",
            support_path=RuntimeSupportPath.PACKAGED,
            summary="Use the packaged GGUF/llama.cpp path when you need decode-time JSON-schema or grammar enforcement on this host.",
            fallback_guidance=[
                "The external accelerator bridge preserves structured-output requests only through prompt-guided fallback; it is not the portable decode-time enforcement default.",
            ],
        ),
    ]
