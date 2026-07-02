from __future__ import annotations

from types import SimpleNamespace

from lewlm.runtime.llamacpp.build_flavor import detect_llamacpp_build_flavor


def _fail_import(name: str):
    raise ImportError(name)


def test_build_flavor_is_unavailable_when_backend_is_missing(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.runtime.llamacpp.build_flavor.import_module", _fail_import)

    flavor = detect_llamacpp_build_flavor()

    assert flavor.installed is False
    assert flavor.detection_state == "unavailable"
    assert flavor.gpu_offload_supported is None
    assert flavor.accelerator_hints == []
    assert "not installed" in flavor.reason


def test_build_flavor_detects_cuda_build_from_backend_reporting(monkeypatch) -> None:
    fake_llama_cpp = SimpleNamespace(
        llama_supports_gpu_offload=lambda: True,
        llama_print_system_info=lambda: b"AVX = 1 | AVX2 = 1 | CUDA = 1 | VULKAN = 0 | METAL = 0",
    )
    monkeypatch.setattr("lewlm.runtime.llamacpp.build_flavor.import_module", lambda name: fake_llama_cpp)

    flavor = detect_llamacpp_build_flavor()

    assert flavor.installed is True
    assert flavor.detection_state == "detected"
    assert flavor.gpu_offload_supported is True
    assert flavor.accelerator_hints == ["cuda"]
    assert "AVX2 = 1" in (flavor.system_info or "")
    assert "heuristic inventory evidence only" in flavor.reason


def test_build_flavor_does_not_hint_accelerators_from_zero_flags(monkeypatch) -> None:
    fake_llama_cpp = SimpleNamespace(
        llama_supports_gpu_offload=lambda: False,
        llama_print_system_info=lambda: b"AVX = 1 | CUDA = 0 | VULKAN = 0",
    )
    monkeypatch.setattr("lewlm.runtime.llamacpp.build_flavor.import_module", lambda name: fake_llama_cpp)

    flavor = detect_llamacpp_build_flavor()

    assert flavor.gpu_offload_supported is False
    assert flavor.accelerator_hints == []
    assert flavor.detection_state == "detected"


def test_build_flavor_hints_from_backend_section_style_output(monkeypatch) -> None:
    fake_llama_cpp = SimpleNamespace(
        llama_supports_gpu_offload=lambda: True,
        llama_print_system_info=lambda: "load_backend: loaded Vulkan backend\nload_backend: loaded CPU backend",
    )
    monkeypatch.setattr("lewlm.runtime.llamacpp.build_flavor.import_module", lambda name: fake_llama_cpp)

    flavor = detect_llamacpp_build_flavor()

    assert flavor.accelerator_hints == ["vulkan"]


def test_build_flavor_reports_partial_detection_honestly(monkeypatch) -> None:
    fake_llama_cpp = SimpleNamespace()
    monkeypatch.setattr("lewlm.runtime.llamacpp.build_flavor.import_module", lambda name: fake_llama_cpp)

    flavor = detect_llamacpp_build_flavor()

    assert flavor.installed is True
    assert flavor.detection_state == "partial"
    assert flavor.gpu_offload_supported is None
    assert flavor.system_info is None
    assert "do not expose `llama_supports_gpu_offload`" in flavor.reason
    assert "do not expose `llama_print_system_info`" in flavor.reason
