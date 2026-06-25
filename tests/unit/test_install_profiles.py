from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import set_host_platform
from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    ModelTargetPlatformReport,
    PerformanceCoreEvidenceFamily,
    PerformanceCoreEvidenceMode,
    RuntimeSupportPath,
    StandardsAcceptanceState,
    StandardsVocabularyTerm,
)
from lewlm.documents.ingest.ocr import OcrBackendStatus
from lewlm.install_profiles import FeaturePathRecommendation, InstallProfileSummary, summarize_install_profiles
from lewlm.telemetry.models import RuntimeSupportPathSummary


def _stub_installed_modules(monkeypatch, installed: set[str]) -> None:
    monkeypatch.setattr(
        "lewlm.install_profiles._missing_modules",
        lambda module_names: [name for name in module_names if name not in installed],
    )
    monkeypatch.setattr(
        "lewlm.install_profiles.detect_ocr_backend",
        lambda: OcrBackendStatus(available=False, backend_name="pytesseract", reason="The `tesseract` binary is not installed."),
    )


def test_install_profiles_prefer_mlx_on_apple_silicon(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.install_profiles.platform.system", lambda: "Darwin")
    monkeypatch.setattr("lewlm.install_profiles.platform.machine", lambda: "arm64")

    def fake_missing_modules(module_names: tuple[str, ...]) -> list[str]:
        if module_names == ("mlx", "mlx_lm", "mlx_vlm", "mlx_audio"):
            return []
        if module_names == ("llama_cpp",):
            return ["llama_cpp"]
        if module_names == ("onnxruntime_genai",):
            return ["onnxruntime_genai"]
        return []

    monkeypatch.setattr("lewlm.install_profiles._missing_modules", fake_missing_modules)
    monkeypatch.setattr(
        "lewlm.install_profiles.detect_ocr_backend",
        lambda: OcrBackendStatus(available=False, backend_name="pytesseract", reason="The `tesseract` binary is not installed."),
    )

    summary = summarize_install_profiles()
    profiles = {profile.profile: profile for profile in summary.profiles}
    recommendations = {item.feature_class: item for item in summary.recommended_feature_paths}

    assert summary.recommended_profile_id == "mlx_local_backend"
    assert summary.active_profile_ids == ["core_only", "mlx_local_backend", "documents_enabled_backend"]
    assert "first-class local runtime profile" in summary.notes[0]
    assert profiles["mlx_local_backend"].label == "Apple MLX local backend"
    assert profiles["mlx_local_backend"].ready is True
    assert profiles["gguf_fallback_backend"].label == "Cross-platform GGUF backend"
    assert profiles["gguf_fallback_backend"].installed is False
    assert profiles["external_accelerator_bridge_backend"].installed is False
    assert profiles["documents_enabled_backend"].label == "Documents add-on"
    assert profiles["documents_enabled_backend"].ready is True
    assert "tesseract" in profiles["documents_enabled_backend"].notes[0]
    assert recommendations["chat"].profile == "mlx_local_backend"
    assert recommendations["chat"].support_path == RuntimeSupportPath.PACKAGED
    assert recommendations["structured_output"].profile == "gguf_fallback_backend"
    assert "decode-time" in recommendations["structured_output"].summary


def test_install_profiles_prefer_gguf_on_linux_hosts(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.install_profiles.platform.system", lambda: "Linux")
    monkeypatch.setattr("lewlm.install_profiles.platform.machine", lambda: "x86_64")

    def fake_missing_modules(module_names: tuple[str, ...]) -> list[str]:
        if module_names == ("mlx", "mlx_lm", "mlx_vlm", "mlx_audio"):
            return []
        if module_names == ("llama_cpp",):
            return []
        if module_names == ("onnxruntime_genai",):
            return ["onnxruntime_genai"]
        return ["openpyxl"]

    monkeypatch.setattr("lewlm.install_profiles._missing_modules", fake_missing_modules)

    summary = summarize_install_profiles()
    profiles = {profile.profile: profile for profile in summary.profiles}
    recommendations = {item.feature_class: item for item in summary.recommended_feature_paths}

    assert summary.recommended_profile_id == "gguf_fallback_backend"
    assert summary.active_profile_ids == ["core_only", "mlx_local_backend", "gguf_fallback_backend"]
    assert "first-class non-Apple runtime family" in summary.notes[0]
    assert "semantic GGUF models can stay packaged" in summary.notes[1]
    assert profiles["mlx_local_backend"].installed is True
    assert profiles["mlx_local_backend"].ready is False
    assert "Apple Silicon" in profiles["mlx_local_backend"].notes[0]
    assert profiles["gguf_fallback_backend"].ready is True
    assert "Recommended first-class non-Apple runtime on Linux and Windows today." in profiles["gguf_fallback_backend"].notes[0]
    assert "embedding-capable semantic GGUF models" in profiles["gguf_fallback_backend"].notes[1]
    assert profiles["external_accelerator_bridge_backend"].installed is False
    assert any(
        "NVIDIA-backed local servers" in note
        for note in profiles["external_accelerator_bridge_backend"].notes
    )
    assert profiles["documents_enabled_backend"].installed is False
    assert recommendations["chat"].profile == "gguf_fallback_backend"
    assert recommendations["chat"].support_path == RuntimeSupportPath.PACKAGED
    assert recommendations["semantic_text"].profile == "gguf_fallback_backend"
    assert recommendations["semantic_text"].support_path == RuntimeSupportPath.PACKAGED
    assert recommendations["vision"].profile == "external_accelerator_bridge_backend"
    assert recommendations["audio"].profile == "external_accelerator_bridge_backend"
    assert "bridge-only" in recommendations["audio"].summary
    assert any("probes the transcription and speech bridge endpoints separately" in item for item in recommendations["audio"].fallback_guidance)
    assert recommendations["structured_output"].profile == "gguf_fallback_backend"


@pytest.mark.parametrize(
    ("system", "machine", "settings", "expected_profile", "expected_feature_paths", "expected_bridge_ready"),
    [
        (
            "Darwin",
            "arm64",
            None,
            "mlx_local_backend",
            {
                "chat": "packaged",
                "semantic_text": "packaged",
                "vision": "packaged",
                "audio": "packaged",
                "structured_output": "packaged",
            },
            False,
        ),
        (
            "Linux",
            "x86_64",
            SimpleNamespace(
                external_accelerator_enabled=True,
                external_accelerator_base_url="http://127.0.0.1:8000",
                external_accelerator_profile="vllm_local",
            ),
            "gguf_fallback_backend",
            {
                "chat": "packaged",
                "semantic_text": "packaged",
                "vision": "bridge",
                "audio": "bridge",
                "structured_output": "packaged",
            },
            True,
        ),
        (
            "Windows",
            "AMD64",
            SimpleNamespace(
                external_accelerator_enabled=True,
                external_accelerator_base_url="http://127.0.0.1:8000",
                external_accelerator_profile="vllm_local",
            ),
            "gguf_fallback_backend",
            {
                "chat": "packaged",
                "semantic_text": "packaged",
                "vision": "bridge",
                "audio": "bridge",
                "structured_output": "packaged",
            },
            True,
        ),
    ],
)
def test_install_profiles_platform_matrix_is_host_proof(
    monkeypatch,
    system: str,
    machine: str,
    settings,
    expected_profile: str,
    expected_feature_paths: dict[str, str],
    expected_bridge_ready: bool,
) -> None:
    set_host_platform(monkeypatch, system=system, machine=machine)
    _stub_installed_modules(
        monkeypatch,
        {
            "mlx",
            "mlx_lm",
            "mlx_vlm",
            "mlx_audio",
            "llama_cpp",
        },
    )

    summary = summarize_install_profiles(settings)
    profiles = {profile.profile: profile for profile in summary.profiles}
    recommendations = {item.feature_class: item for item in summary.recommended_feature_paths}

    assert summary.recommended_profile_id == expected_profile
    assert profiles["mlx_local_backend"].ready is (system == "Darwin")
    assert profiles["gguf_fallback_backend"].ready is True
    assert profiles["external_accelerator_bridge_backend"].ready is expected_bridge_ready
    assert profiles["external_accelerator_bridge_backend"].installed is expected_bridge_ready
    for feature_class, support_path in expected_feature_paths.items():
        assert recommendations[feature_class].support_path.value == support_path


def test_install_profiles_report_configured_external_accelerator_on_apple_hosts(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.install_profiles.platform.system", lambda: "Darwin")
    monkeypatch.setattr("lewlm.install_profiles.platform.machine", lambda: "arm64")
    monkeypatch.setattr("lewlm.install_profiles._missing_modules", lambda module_names: list(module_names))

    settings = SimpleNamespace(
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8000",
        external_accelerator_profile="vllm_mlx",
    )

    summary = summarize_install_profiles(settings)
    profiles = {profile.profile: profile for profile in summary.profiles}
    external = profiles["external_accelerator_bridge_backend"]

    assert summary.recommended_profile_id == "mlx_local_backend"
    assert summary.active_profile_ids == ["core_only", "external_accelerator_bridge_backend"]
    assert external.installed is True
    assert external.ready is True
    assert any("vLLM-style bridge profile" in note for note in external.notes)
    assert any("loopback-only external accelerator endpoint" in note for note in external.notes)


def test_install_profiles_report_external_accelerator_ready_on_windows(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.install_profiles.platform.system", lambda: "Windows")
    monkeypatch.setattr("lewlm.install_profiles.platform.machine", lambda: "AMD64")
    monkeypatch.setattr("lewlm.install_profiles._missing_modules", lambda module_names: list(module_names))

    settings = SimpleNamespace(
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8000",
        external_accelerator_profile="vllm_mlx",
    )

    summary = summarize_install_profiles(settings)
    profiles = {profile.profile: profile for profile in summary.profiles}
    external = profiles["external_accelerator_bridge_backend"]
    recommendations = {item.feature_class: item for item in summary.recommended_feature_paths}

    assert summary.recommended_profile_id == "gguf_fallback_backend"
    assert summary.active_profile_ids == ["core_only", "external_accelerator_bridge_backend"]
    assert external.installed is True
    assert external.ready is True
    assert any("NVIDIA-backed local servers" in note for note in external.notes)
    assert any("/v1/rerank" in note for note in external.notes)
    assert any("bridge-only non-Apple public audio parity path" in note for note in external.notes)
    assert recommendations["chat"].profile == "gguf_fallback_backend"
    assert recommendations["semantic_text"].profile == "gguf_fallback_backend"
    assert recommendations["semantic_text"].support_path == RuntimeSupportPath.PACKAGED
    assert recommendations["audio"].profile == "external_accelerator_bridge_backend"


def test_install_profiles_report_windows_llamacpp_build_prerequisites_when_backend_is_missing(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.install_profiles.platform.system", lambda: "Windows")
    monkeypatch.setattr("lewlm.install_profiles.platform.machine", lambda: "AMD64")

    def fake_missing_modules(module_names: tuple[str, ...]) -> list[str]:
        if module_names == ("llama_cpp",):
            return ["llama_cpp"]
        if module_names == ("onnxruntime_genai",):
            return ["onnxruntime_genai"]
        return []

    monkeypatch.setattr("lewlm.install_profiles._missing_modules", fake_missing_modules)
    monkeypatch.setattr("lewlm.install_profiles._has_command", lambda command: False)
    monkeypatch.setattr(
        "lewlm.install_profiles.detect_ocr_backend",
        lambda: OcrBackendStatus(available=False, backend_name="pytesseract", reason="The `tesseract` binary is not installed."),
    )

    summary = summarize_install_profiles()
    profiles = {profile.profile: profile for profile in summary.profiles}
    gguf = profiles["gguf_fallback_backend"]

    assert gguf.ready is False
    assert any("Microsoft C++ Build Tools" in note for note in gguf.notes)
    assert any("CMake is not currently on PATH" in note for note in gguf.notes)
    assert any("Ninja is optional" in note for note in gguf.notes)


def test_parity_contract_fields_stay_machine_readable() -> None:
    install_profile_summary = InstallProfileSummary()
    feature = FeaturePathRecommendation(
        feature_class="semantic_text",
        profile="external_accelerator_bridge_backend",
        label="Cross-platform external accelerator bridge",
        support_path=RuntimeSupportPath.BRIDGE,
        summary="Bridge-backed semantic path.",
    )
    target = ModelTargetPlatformReport(
        system="Windows",
        machine="AMD64",
        supported=True,
        readiness_state="verified",
        verification_method="host_probe",
        reason="Verified on the current host.",
    )
    strategy = RuntimeSupportPathSummary(
        path_id="gguf_llamacpp",
        label="GGUF via llama.cpp",
        role="first_class_non_apple",
        host_scope="cross_platform",
        benchmark_backed_defaults=True,
        performance_core_evidence=[
            {
                "family": PerformanceCoreEvidenceFamily.CONSTRAINED_DECODING,
                "mode": PerformanceCoreEvidenceMode.BACKEND_NATIVE,
                "reason": "Decode-time constrained decoding is backend-native on this path.",
                "benchmark_backed": True,
            },
        ],
    )

    assert feature.model_dump(mode="json")["support_path"] == "bridge"
    assert target.model_dump(mode="json")["verification_method"] == "host_probe"
    standards_contract = install_profile_summary.model_dump(mode="json")["standards_acceptance_contract"]
    assert standards_contract["format"] == "lewlm-standards-acceptance-contract-v1"
    assert {item["state"] for item in standards_contract["acceptance_states"]} == {
        state.value for state in StandardsAcceptanceState
    }
    assert {item["name"] for item in standards_contract["vocabulary"]} >= {
        StandardsVocabularyTerm.KV_OFFLOAD.value,
        StandardsVocabularyTerm.RESPONSES_API_EVENTS.value,
        StandardsVocabularyTerm.LOCAL_AGENT_SANDBOX.value,
    }
    payload = strategy.model_dump(mode="json")
    assert payload["benchmark_backed_defaults"] is True
    assert payload["performance_core_evidence"][0]["benchmark_backed"] is True
    assert payload["performance_core_evidence"][0]["mode"] == "backend_native"


def test_install_profiles_describe_ollama_bridge_alias() -> None:
    settings = LewLMSettings(
        external_accelerator_enabled=True,
        external_accelerator_base_url="http://127.0.0.1:8080",
        external_accelerator_profile="ollama_local",
    )

    summary = summarize_install_profiles(settings)
    external = next(profile for profile in summary.profiles if profile.profile == "external_accelerator_bridge_backend")

    assert any("Ollama-compatible bridge profile" in note for note in external.notes)


def test_install_profile_docs_cover_cross_platform_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    docs_to_snippets = {
        repo_root / "README.md": (
            "Apple MLX local backend",
            "Cross-platform GGUF backend",
            "Cross-platform external accelerator bridge",
            "Documents add-on",
            "NVIDIA",
            "first-class non-Apple",
            "Recommended feature paths by platform",
            "semantic text",
            "structured output",
            "Parity acceptance contract",
            "support_path",
            "host_probe",
            "benchmark_backed",
            "standards_acceptance_contract",
            "kv_offload",
            "local_agent_sandbox",
            "unverified",
        ),
        repo_root / "docs" / "getting-started" / "installation.md": (
            "Apple MLX local backend",
            "Cross-platform GGUF backend",
            "Cross-platform external accelerator bridge",
            "LEWLM_EXTERNAL_ACCELERATOR_ENABLED",
            "NVIDIA",
            "first-class non-Apple",
            "Recommended default feature routes",
            "semantic text",
            "structured output",
            "Reading support states",
            "host_probe",
            "benchmark_backed",
            "support_path",
            "standards_acceptance_contract",
            "kv_offload",
            "unverified",
        ),
        repo_root / "docs" / "getting-started" / "quickstart.md": (
            "## Apple MLX local backend",
            "## Cross-platform GGUF backend",
            "## Cross-platform external accelerator bridge",
            "LEWLM_EXTERNAL_ACCELERATOR_BASE_URL",
            "first-class non-Apple",
            "Platform default feature guide",
            "structured output",
            "standards_acceptance_contract",
            "local_agent_sandbox",
        ),
        repo_root / "docs" / "architecture" / "runtime-routing-and-serving.md": (
            "Standards acceptance contract",
            "kv_offload",
            "local_agent_sandbox",
            "lewlm_owned",
            "unverified",
        ),
        repo_root / "docs" / "reference" / "runtime-capability-matrix.md": (
            "`external_accelerator`",
            "Bridge to a loopback-only OpenAI-compatible local server",
            "Darwin, Linux, Windows",
            "adapter-backed",
            "first-class non-Apple",
            "Recommended operator path by feature class",
            "semantic text",
            "structured output",
            "Acceptance state legend",
            "Full parity acceptance matrix",
            "host_probe",
            "benchmark_backed",
            "support_path",
            "standards_acceptance_contract",
            "kv_offload",
            "responses_api_events",
            "local_agent_sandbox",
            "unverified",
        ),
    }

    for path, snippets in docs_to_snippets.items():
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in text, f"{snippet!r} missing from {path}"
