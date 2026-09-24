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


def test_build_cpu_features_the_host_lacks_are_reported_only_when_checkable() -> None:
    from lewlm.runtime.llamacpp.build_flavor import _missing_cpu_features

    # The prebuilt cu130 Windows wheel (llama-cpp-python 0.3.35) on an Arrow Lake CPU.
    info = ("CUDA : ARCHS = 750,800,860,890,900 | USE_GRAPHS = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | "
            "F16C = 1 | FMA = 1 | AVX512 = 1 | AVX512_VNNI = 0 | LLAMAFILE = 1 |")
    windows = ({"SSE3", "SSSE3", "AVX", "AVX2"}, {"SSE3", "SSSE3", "AVX", "AVX2", "AVX512"})
    assert _missing_cpu_features(info, windows) == ["AVX512"], "F16C/FMA cannot be checked on Windows and are not claimed missing"
    assert _missing_cpu_features(info.replace("AVX512 = 1", "AVX512 = 0"), windows) == []
    assert _missing_cpu_features(info, None) is None, "an unreadable host stays unknown"
    assert _missing_cpu_features(None, windows) is None


def test_the_verifier_refuses_a_build_that_would_crash_at_first_load() -> None:
    import sys
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    from lewlm.runtime.llamacpp.build_flavor import LlamaCppBuildFlavor

    spec = spec_from_file_location("verify_llamacpp_build_isa", Path(__file__).resolve().parents[2] / "scripts" / "verify_llamacpp_build.py")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    flavor = LlamaCppBuildFlavor(installed=True, gpu_offload_supported=True, accelerator_hints=["cuda"],
                                 missing_cpu_features=["AVX512"], detection_state="detected", reason="detected")
    code, message = module.evaluate(flavor, expect="gpu", hint="cuda")
    assert code == 2 and "AVX512" in message and "illegal instruction" in message
