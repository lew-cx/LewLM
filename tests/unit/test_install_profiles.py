from __future__ import annotations

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
    assert profiles["mlx_local_backend"].installed is True
    assert profiles["mlx_local_backend"].ready is True
    assert profiles["gguf_fallback_backend"].installed is False
    assert profiles["documents_enabled_backend"].ready is True
    assert "tesseract" in profiles["documents_enabled_backend"].notes[0]


def test_install_profiles_report_host_blocked_mlx_on_non_apple_hosts(monkeypatch) -> None:
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
    assert profiles["mlx_local_backend"].installed is True
    assert profiles["mlx_local_backend"].ready is False
    assert "Apple Silicon" in profiles["mlx_local_backend"].notes[0]
    assert profiles["gguf_fallback_backend"].ready is True
    assert profiles["documents_enabled_backend"].installed is False
