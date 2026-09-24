"""One place that knows llama-cpp-python can be installed and still not load.

`importlib.util.find_spec` proves a package is on disk. It does not prove the
native extension behind it will load, and for this wheel those are genuinely
different questions: the packaged `llama.dll`/`libllama.so` is loaded by
`ctypes` at import time, long after pip reported success. A host can refuse it
— Windows Application Control blocking an unsigned binary, a missing system
library, an architecture mismatch — and the refusal arrives as `OSError` or as
the `RuntimeError` llama-cpp-python raises in its place, never as `ImportError`.

Callers funnel through here so that refusal is reported in the host's own words
instead of escaping as an unhandled error or being mislabelled as an absent
install. Distinguishing the two matters: "not installed" is fixed by installing
the extra, and "installed but blocked" is not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
import sys
from typing import Any

_registered_dll_directories: list[str] | None = None


@dataclass(frozen=True)
class LlamaCppImport:
    """Outcome of importing `llama_cpp`, with absence separated from refusal.

    ``module`` is the imported module, or ``None`` when it could not be
    imported. ``installed`` stays ``True`` when the package is present but its
    native library would not load, in which case ``reason`` carries the host's
    own message. ``reason`` is ``None`` when the package is simply absent, so
    callers keep phrasing their own install guidance.
    """

    module: Any | None
    installed: bool
    reason: str | None


def load_llama_cpp(importer: Callable[[str], Any] = import_module) -> LlamaCppImport:
    """Import `llama_cpp` without letting a host-level refusal escape.

    Callers pass their own module-level ``import_module`` so each keeps the
    import seam its own tests already patch; only the knowledge of how this
    import fails lives here.
    """

    register_nvidia_wheel_dll_directories()
    try:
        return LlamaCppImport(module=importer("llama_cpp"), installed=True, reason=None)
    except ImportError:
        return LlamaCppImport(module=None, installed=False, reason=None)
    except (OSError, RuntimeError) as exc:
        reason = (
            "llama-cpp-python is installed, but its native llama.cpp library could not be "
            f"loaded on this host: {exc}. The package is present, so installing the "
            "`llamacpp` extra again will not change this; the load itself is being refused."
        )
        if sys.platform == "win32" and "Could not find module" in str(exc):
            reason += (
                " On Windows this usually means a dependency DLL is missing: a CUDA build of the "
                "wheel also needs NVIDIA's cuBLAS (for the cu13 wheels, `pip install \"nvidia-cublas>=13,<14\"` "
                "into the same environment, or the matching CUDA Toolkit on CUDA_PATH)."
            )
        return LlamaCppImport(module=None, installed=True, reason=reason)


def register_nvidia_wheel_dll_directories(search_path: list[str] | None = None) -> list[str]:
    """Let a CUDA llama.cpp wheel find NVIDIA runtime DLLs that pip installed.

    NVIDIA publishes cuBLAS and the CUDA runtime as wheels (``nvidia/<pkg>/bin``,
    ``nvidia/cu13/bin/x86_64``). On Linux the extension finds them through its
    rpath; on Windows nothing puts those folders on the DLL search path, so a
    CUDA wheel fails to load unless the full toolkit is installed. llama.cpp
    loads with the standard search order, so each folder goes on ``PATH`` as
    well as through ``os.add_dll_directory``. No-op off Windows or without such
    wheels; runs once per process.
    """

    global _registered_dll_directories
    if sys.platform != "win32":
        return []
    if _registered_dll_directories is not None and search_path is None:
        return _registered_dll_directories
    directories: list[str] = []
    for entry in search_path if search_path is not None else sys.path:
        root = Path(entry) / "nvidia"
        if not root.is_dir():
            continue
        for dll in sorted(root.glob("*/bin/**/*.dll")):
            folder = str(dll.parent)
            if folder not in directories:
                directories.append(folder)
    add_dll_directory = getattr(os, "add_dll_directory", None)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for folder in directories:
        if add_dll_directory is not None:
            add_dll_directory(folder)
        if folder not in path_entries:
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    if search_path is None:
        _registered_dll_directories = directories
    return directories
