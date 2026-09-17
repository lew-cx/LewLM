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


def _blocked_import(name: str):
    raise RuntimeError(
        "Failed to load shared library 'llama.dll': "
        "[WinError 4551] An Application Control policy has blocked this file",
    )


def test_build_flavor_reports_a_blocked_library_as_installed_but_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.runtime.llamacpp.build_flavor.import_module", _blocked_import)

    flavor = detect_llamacpp_build_flavor()

    # Installed is the honest answer here: the package is on disk, so this is
    # not fixed by installing the extra again.
    assert flavor.installed is True
    assert flavor.detection_state == "unavailable"
    assert flavor.gpu_offload_supported is None
    assert "An Application Control policy has blocked this file" in flavor.reason
    assert "not installed" not in flavor.reason


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


def test_build_flavor_hints_metal_from_current_backend_section_names(monkeypatch) -> None:
    """Current llama.cpp names the Metal section `MTL`, as observed on an Apple Silicon host."""

    fake_llama_cpp = SimpleNamespace(
        llama_supports_gpu_offload=lambda: True,
        llama_print_system_info=lambda: b"MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | ACCELERATE = 1 | ",
    )
    monkeypatch.setattr("lewlm.runtime.llamacpp.build_flavor.import_module", lambda name: fake_llama_cpp)

    flavor = detect_llamacpp_build_flavor()

    assert flavor.accelerator_hints == ["metal"]


def test_build_flavor_hints_cuda_from_current_backend_section_names(monkeypatch) -> None:
    fake_llama_cpp = SimpleNamespace(
        llama_supports_gpu_offload=lambda: True,
        llama_print_system_info=lambda: b"CUDA : ARCHS = 890 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : SSE3 = 1 | AVX = 1 | ",
    )
    monkeypatch.setattr("lewlm.runtime.llamacpp.build_flavor.import_module", lambda name: fake_llama_cpp)

    flavor = detect_llamacpp_build_flavor()

    assert flavor.accelerator_hints == ["cuda"]


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
