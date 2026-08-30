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
from typing import Any


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

    try:
        return LlamaCppImport(module=importer("llama_cpp"), installed=True, reason=None)
    except ImportError:
        return LlamaCppImport(module=None, installed=False, reason=None)
    except (OSError, RuntimeError) as exc:
        return LlamaCppImport(
            module=None,
            installed=True,
            reason=(
                "llama-cpp-python is installed, but its native llama.cpp library could not be "
                f"loaded on this host: {exc}. The package is present, so installing the "
                "`llamacpp` extra again will not change this; the load itself is being refused."
            ),
        )
