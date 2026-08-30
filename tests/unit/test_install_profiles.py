from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import set_host_platform
from lewlm.config.settings import LewLMSettings
from lewlm.container import ContainerStatus
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
from lewlm.runtime.feature_probes import BackendFeatureProbe
from lewlm.runtime.llamacpp.build_flavor import LlamaCppBuildFlavor
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


def _stub_loadable_llamacpp_build(monkeypatch) -> None:
    """Pin the build flavor so a test reads its simulated host, not the real one.

    `_missing_modules` only answers whether the package is on disk. Readiness
    also asks whether its native library loads, and without this stub that
    second question is answered by whichever machine runs the test.
    """

    monkeypatch.setattr(
        "lewlm.install_profiles.detect_llamacpp_build_flavor",
        lambda: LlamaCppBuildFlavor(
            installed=True,
            gpu_offload_supported=False,
            accelerator_hints=[],
            system_info="AVX = 1",
            detection_state="detected",
            reason="llama.cpp build flavor detected from the installed backend's own reporting APIs.",
        ),
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
    _stub_loadable_llamacpp_build(monkeypatch)

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
    _stub_loadable_llamacpp_build(monkeypatch)

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
        backend_feature_probes_enabled=False,
    )

    summary = summarize_install_profiles(settings)
    external = next(profile for profile in summary.profiles if profile.profile == "external_accelerator_bridge_backend")

    assert any("Ollama-compatible bridge profile" in note for note in external.notes)


def test_backend_inventory_reports_installed_versions_honestly(monkeypatch) -> None:
    set_host_platform(monkeypatch, system="Windows", machine="AMD64")
    _stub_installed_modules(monkeypatch, installed={"llama_cpp"})
    monkeypatch.setattr(
        "lewlm.install_profiles._backend_distribution_version",
        lambda distribution: "0.3.9" if distribution == "llama-cpp-python" else None,
    )

    summary = summarize_install_profiles()
    inventory = {entry.module: entry for entry in summary.backend_inventory}

    assert set(inventory) == {"mlx", "mlx_lm", "mlx_vlm", "mlx_audio", "llama_cpp", "onnxruntime_genai"}
    assert inventory["llama_cpp"].profile == "gguf_fallback_backend"
    assert inventory["llama_cpp"].installed is True
    assert inventory["llama_cpp"].version == "0.3.9"
    assert "still decide capability evidence" in inventory["llama_cpp"].detail
    assert inventory["onnxruntime_genai"].installed is False
    assert inventory["onnxruntime_genai"].version is None
    assert "not importable" in inventory["onnxruntime_genai"].detail
    assert inventory["mlx"].profile == "mlx_local_backend"
    assert inventory["mlx"].installed is False


def test_backend_inventory_keeps_version_claims_honest_without_metadata(monkeypatch) -> None:
    set_host_platform(monkeypatch, system="Linux", machine="x86_64")
    _stub_installed_modules(monkeypatch, installed={"llama_cpp"})
    monkeypatch.setattr(
        "lewlm.install_profiles._backend_distribution_version",
        lambda distribution: None,
    )

    summary = summarize_install_profiles()
    llama_entry = next(entry for entry in summary.backend_inventory if entry.module == "llama_cpp")

    assert llama_entry.installed is True
    assert llama_entry.version is None
    assert "no version metadata" in llama_entry.detail
    assert "without a version claim" in llama_entry.detail


def test_install_profiles_do_not_report_a_blocked_gguf_backend_as_ready(monkeypatch) -> None:
    set_host_platform(monkeypatch, system="Windows", machine="AMD64")
    _stub_installed_modules(monkeypatch, installed={"llama_cpp"})
    blocked_reason = (
        "llama-cpp-python is installed, but its native llama.cpp library could not be loaded on "
        "this host: Failed to load shared library 'llama.dll': [WinError 4551] An Application "
        "Control policy has blocked this file."
    )
    monkeypatch.setattr(
        "lewlm.install_profiles.detect_llamacpp_build_flavor",
        lambda: LlamaCppBuildFlavor(
            installed=True,
            detection_state="unavailable",
            reason=blocked_reason,
        ),
    )

    summary = summarize_install_profiles()
    gguf = next(profile for profile in summary.profiles if profile.profile == "gguf_fallback_backend")

    # Present on disk, so `installed` stays true; nothing can load, so `ready`
    # must not claim otherwise.
    assert gguf.installed is True
    assert gguf.ready is False
    assert any("An Application Control policy has blocked this file" in note for note in gguf.notes)


def test_install_profiles_surface_llamacpp_build_flavor_when_installed(monkeypatch) -> None:
    set_host_platform(monkeypatch, system="Windows", machine="AMD64")
    _stub_installed_modules(monkeypatch, installed={"llama_cpp"})
    monkeypatch.setattr(
        "lewlm.install_profiles.detect_llamacpp_build_flavor",
        lambda: LlamaCppBuildFlavor(
            installed=True,
            gpu_offload_supported=True,
            accelerator_hints=["cuda"],
            system_info="AVX = 1 | CUDA = 1",
            detection_state="detected",
            reason="llama.cpp build flavor detected from the installed backend's own reporting APIs.",
        ),
    )

    summary = summarize_install_profiles()
    gguf = next(profile for profile in summary.profiles if profile.profile == "gguf_fallback_backend")

    assert summary.llamacpp_build is not None
    assert summary.llamacpp_build.accelerator_hints == ["cuda"]
    assert any("GPU offload support" in note and "cuda" in note for note in gguf.notes)
    assert any("probes and benchmarks decide capability evidence" in note for note in gguf.notes)


def test_install_profiles_recommend_accelerated_build_for_cpu_only_llamacpp(monkeypatch) -> None:
    set_host_platform(monkeypatch, system="Linux", machine="x86_64")
    _stub_installed_modules(monkeypatch, installed={"llama_cpp"})
    monkeypatch.setattr(
        "lewlm.install_profiles.detect_llamacpp_build_flavor",
        lambda: LlamaCppBuildFlavor(
            installed=True,
            gpu_offload_supported=False,
            accelerator_hints=[],
            system_info="AVX = 1 | CUDA = 0",
            detection_state="detected",
            reason="llama.cpp build flavor detected from the installed backend's own reporting APIs.",
        ),
    )

    summary = summarize_install_profiles()
    gguf = next(profile for profile in summary.profiles if profile.profile == "gguf_fallback_backend")

    assert any("CPU-only" in note for note in gguf.notes)
    assert any("Vulkan as the vendor-neutral option" in note for note in gguf.notes)


def test_install_profiles_skip_build_flavor_detection_when_llamacpp_missing(monkeypatch) -> None:
    set_host_platform(monkeypatch, system="Windows", machine="AMD64")
    _stub_installed_modules(monkeypatch, installed=set())

    def _unexpected_detection() -> LlamaCppBuildFlavor:
        raise AssertionError("build-flavor detection must not run when llama_cpp is missing")

    monkeypatch.setattr("lewlm.install_profiles.detect_llamacpp_build_flavor", _unexpected_detection)

    summary = summarize_install_profiles()

    assert summary.llamacpp_build is None


def test_install_profiles_surface_backend_feature_probes(monkeypatch) -> None:
    set_host_platform(monkeypatch, system="Windows", machine="AMD64")
    _stub_installed_modules(monkeypatch, installed=set())
    monkeypatch.setattr(
        "lewlm.install_profiles.probe_backend_features",
        lambda **_kwargs: [
            BackendFeatureProbe(
                profile="gguf_fallback_backend",
                backend="llama_cpp",
                feature="decode_time_grammar_enforcement",
                present=True,
                detail="`LlamaGrammar.from_string` and `.from_json_schema` are exposed for decode-time enforcement.",
            ),
        ],
    )

    summary = summarize_install_profiles()

    assert len(summary.backend_feature_probes) == 1
    probe = summary.backend_feature_probes[0]
    assert probe.feature == "decode_time_grammar_enforcement"
    assert probe.present is True


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


def _stub_container(monkeypatch, *, in_container: bool) -> None:
    """Pin container residency so guidance is not decided by the test host."""

    monkeypatch.setattr(
        "lewlm.install_profiles.detect_container",
        lambda: ContainerStatus(
            in_container=in_container,
            runtime="docker" if in_container else None,
            indicators=["`/.dockerenv` exists"] if in_container else [],
            reason="pinned for test",
        ),
    )


def test_non_apple_native_host_is_pointed_at_the_container_image(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.install_profiles.platform.system", lambda: "Windows")
    monkeypatch.setattr("lewlm.install_profiles.platform.machine", lambda: "AMD64")
    _stub_installed_modules(monkeypatch, {"llama_cpp"})
    _stub_loadable_llamacpp_build(monkeypatch)
    _stub_container(monkeypatch, in_container=False)

    summary = summarize_install_profiles()

    assert summary.container is not None
    assert summary.container.in_container is False
    assert "should run LewLM through the shipped container image" in summary.notes[0]
    # The existing non-Apple contract still has to read out of the same note.
    assert "first-class non-Apple runtime family" in summary.notes[0]


def test_non_apple_container_host_reports_it_is_already_on_the_promoted_path(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.install_profiles.platform.system", lambda: "Linux")
    monkeypatch.setattr("lewlm.install_profiles.platform.machine", lambda: "x86_64")
    _stub_installed_modules(monkeypatch, {"llama_cpp"})
    _stub_loadable_llamacpp_build(monkeypatch)
    _stub_container(monkeypatch, in_container=True)

    summary = summarize_install_profiles()

    assert summary.container is not None
    assert summary.container.in_container is True
    assert summary.container.runtime == "docker"
    assert "is running LewLM's container image" in summary.notes[0]
    assert "first-class non-Apple runtime family" in summary.notes[0]
    # The image is where the conversion tools live; that is the reason to say so.
    assert "conversion tools" in summary.notes[0]


def test_apple_silicon_guidance_is_left_alone(monkeypatch) -> None:
    """MLX stays the Apple recommendation; Docker cannot reach Metal."""

    monkeypatch.setattr("lewlm.install_profiles.platform.system", lambda: "Darwin")
    monkeypatch.setattr("lewlm.install_profiles.platform.machine", lambda: "arm64")
    _stub_installed_modules(monkeypatch, {"mlx", "mlx_lm", "mlx_vlm", "mlx_audio"})
    _stub_container(monkeypatch, in_container=False)

    summary = summarize_install_profiles()

    assert "first-class local runtime profile" in summary.notes[0]
    assert not any("container image" in note for note in summary.notes)


def test_cpu_only_llamacpp_build_points_at_the_cuda_image(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.install_profiles.platform.system", lambda: "Linux")
    monkeypatch.setattr("lewlm.install_profiles.platform.machine", lambda: "x86_64")
    _stub_installed_modules(monkeypatch, {"llama_cpp"})
    _stub_loadable_llamacpp_build(monkeypatch)
    _stub_container(monkeypatch, in_container=False)

    summary = summarize_install_profiles()
    gguf = {profile.profile: profile for profile in summary.profiles}["gguf_fallback_backend"]

    assert any("Dockerfile.cuda" in note for note in gguf.notes)
    # The native escape hatches stay documented rather than being replaced.
    assert any("Vulkan as the vendor-neutral option" in note for note in gguf.notes)
