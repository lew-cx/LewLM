from __future__ import annotations

from types import SimpleNamespace

import pytest

from lewlm.runtime.llamacpp.import_guard import load_llama_cpp


def _absent(name: str):
    raise ImportError(name)


def _blocked(name: str):
    # What llama-cpp-python raises when ctypes cannot load the packaged
    # library: a RuntimeError chained from the host's own OSError.
    raise RuntimeError(
        "Failed to load shared library 'llama.dll': "
        "[WinError 4551] An Application Control policy has blocked this file",
    )


def _refused(name: str):
    raise OSError("libllama.so: cannot open shared object file")


def test_import_guard_reports_an_absent_package_without_a_reason() -> None:
    imported = load_llama_cpp(_absent)

    assert imported.module is None
    assert imported.installed is False
    # No reason: callers phrase their own install guidance for this case.
    assert imported.reason is None


def test_import_guard_separates_a_blocked_library_from_an_absent_package() -> None:
    imported = load_llama_cpp(_blocked)

    assert imported.module is None
    assert imported.installed is True
    assert "An Application Control policy has blocked this file" in imported.reason
    assert "will not change this" in imported.reason


def test_import_guard_catches_a_bare_os_error_from_the_loader() -> None:
    imported = load_llama_cpp(_refused)

    assert imported.module is None
    assert imported.installed is True
    assert "cannot open shared object file" in imported.reason


def test_import_guard_returns_the_module_when_the_load_succeeds() -> None:
    fake = SimpleNamespace(__name__="llama_cpp")

    imported = load_llama_cpp(lambda name: fake)

    assert imported.module is fake
    assert imported.installed is True
    assert imported.reason is None


@pytest.mark.parametrize("failure", [_blocked, _refused])
def test_import_guard_never_lets_a_host_refusal_escape(failure) -> None:
    # The bug this guards: these escaped as unhandled errors and 500ed
    # /v1/health and /v1/runtime/stats.
    assert load_llama_cpp(failure).module is None
