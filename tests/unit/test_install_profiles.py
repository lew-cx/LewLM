from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lewlm.documents.ingest.ocr import OcrBackendStatus
from lewlm.install_profiles import summarize_install_profiles


def test_install_profiles_prefer_mlx_on_apple_silicon(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.install_profiles.platform.system", lambda: "Darwin")
    monkeypatch.setattr("lewlm.install_profiles.platform.machine", lambda: "arm64")

    def fake_missing_modules(module_names: tuple[str, ...]) -> list[str]:
        if module_names == ("mlx", "mlx_lm", "mlx_vlm", "mlx_audio"):
            return []
        if module_names == ("llama_cpp",):
            return ["llama_cpp"]
        return []

    monkeypatch.setattr("lewlm.install_profiles._missing_modules", fake_missing_modules)
    monkeypatch.setattr(
        "lewlm.install_profiles.detect_ocr_backend",
        lambda: OcrBackendStatus(available=False, backend_name="pytesseract", reason="The `tesseract` binary is not installed."),
    )

    summary = summarize_install_profiles()
    profiles = {profile.profile: profile for profile in summary.profiles}

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


def test_install_profiles_prefer_gguf_on_linux_hosts(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.install_profiles.platform.system", lambda: "Linux")
    monkeypatch.setattr("lewlm.install_profiles.platform.machine", lambda: "x86_64")

    def fake_missing_modules(module_names: tuple[str, ...]) -> list[str]:
        if module_names == ("mlx", "mlx_lm", "mlx_vlm", "mlx_audio"):
            return []
        if module_names == ("llama_cpp",):
            return []
        return ["openpyxl"]

    monkeypatch.setattr("lewlm.install_profiles._missing_modules", fake_missing_modules)

    summary = summarize_install_profiles()
    profiles = {profile.profile: profile for profile in summary.profiles}

    assert summary.recommended_profile_id == "gguf_fallback_backend"
    assert summary.active_profile_ids == ["core_only", "mlx_local_backend", "gguf_fallback_backend"]
    assert "first-class non-Apple runtime family" in summary.notes[0]
    assert "do not replace the packaged GGUF/llama.cpp default" in summary.notes[1]
    assert profiles["mlx_local_backend"].installed is True
    assert profiles["mlx_local_backend"].ready is False
    assert "Apple Silicon" in profiles["mlx_local_backend"].notes[0]
    assert profiles["gguf_fallback_backend"].ready is True
    assert "Recommended first-class non-Apple runtime on Linux and Windows today." in profiles["gguf_fallback_backend"].notes[0]
    assert profiles["external_accelerator_bridge_backend"].installed is False
    assert any(
        "NVIDIA-backed local servers" in note
        for note in profiles["external_accelerator_bridge_backend"].notes
    )
    assert profiles["documents_enabled_backend"].installed is False


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

    assert summary.recommended_profile_id == "gguf_fallback_backend"
    assert summary.active_profile_ids == ["core_only", "external_accelerator_bridge_backend"]
    assert external.installed is True
    assert external.ready is True
    assert any("NVIDIA-backed local servers" in note for note in external.notes)
    assert any("/v1/rerank" in note for note in external.notes)


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
        ),
        repo_root / "docs" / "getting-started" / "installation.md": (
            "Apple MLX local backend",
            "Cross-platform GGUF backend",
            "Cross-platform external accelerator bridge",
            "LEWLM_EXTERNAL_ACCELERATOR_ENABLED",
            "NVIDIA",
            "first-class non-Apple",
        ),
        repo_root / "docs" / "getting-started" / "quickstart.md": (
            "## Apple MLX local backend",
            "## Cross-platform GGUF backend",
            "## Cross-platform external accelerator bridge",
            "LEWLM_EXTERNAL_ACCELERATOR_BASE_URL",
            "first-class non-Apple",
        ),
        repo_root / "docs" / "reference" / "runtime-capability-matrix.md": (
            "`external_accelerator`",
            "Bridge to a loopback-only OpenAI-compatible local server",
            "Darwin, Linux, Windows",
            "adapter-backed",
            "first-class non-Apple",
        ),
    }

    for path, snippets in docs_to_snippets.items():
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in text, f"{snippet!r} missing from {path}"
