"""Feature-detected build-flavor reporting for the installed llama-cpp-python wheel.

Detection is honest and version-tolerant: every field degrades to an explicit
reason instead of a guess, and nothing here widens capability claims. Runtime
load/generate probes and benchmark records remain the capability evidence.
"""

from __future__ import annotations

import ctypes
import platform
import re
import sys
from importlib import import_module
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from lewlm.runtime.llamacpp.import_guard import load_llama_cpp

_ACCELERATOR_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cuda", ("CUDA",)),
    ("rocm", ("ROCM", "HIP")),
    ("vulkan", ("VULKAN",)),
    ("metal", ("METAL", "MTL")),
    ("sycl", ("SYCL",)),
    ("kompute", ("KOMPUTE",)),
)
_SYSTEM_INFO_FLAG_PATTERN = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([01])")

# x86 features as llama.cpp's system info names them, and how each host OS can
# confirm them: Windows `IsProcessorFeaturePresent` ids (it has none for F16C,
# FMA, or BMI2, which stay unchecked there) and Linux /proc/cpuinfo flags.
_WINDOWS_PROCESSOR_FEATURES = {"SSE3": 13, "SSSE3": 36, "AVX": 39, "AVX2": 40, "AVX512": 41}
_LINUX_CPUINFO_FLAGS = {"SSE3": "pni", "SSSE3": "ssse3", "AVX": "avx", "AVX2": "avx2", "F16C": "f16c", "FMA": "fma",
                        "BMI2": "bmi2", "AVX512": "avx512f"}

LlamaCppBuildDetectionState = Literal["detected", "partial", "unavailable"]


class LlamaCppBuildFlavor(BaseModel):
    """Honest report of what the installed llama.cpp build exposes on this host.

    ``accelerator_hints`` are heuristic markers parsed from the backend's own
    system-info string; they are inventory evidence only, never a support claim.
    """

    installed: bool
    gpu_offload_supported: bool | None = None
    accelerator_hints: list[str] = Field(default_factory=list)
    system_info: str | None = None
    # CPU features the build reports using that this host verifiably lacks; such
    # a library imports fine and then dies with an illegal instruction at the
    # first model load. `None` when the host's features could not be checked.
    missing_cpu_features: list[str] | None = None
    detection_state: LlamaCppBuildDetectionState
    reason: str


def detect_llamacpp_build_flavor() -> LlamaCppBuildFlavor:
    """Inspect the installed llama-cpp-python build without loading a model."""

    imported = load_llama_cpp(import_module)
    if imported.module is None:
        return LlamaCppBuildFlavor(
            installed=imported.installed,
            detection_state="unavailable",
            reason=imported.reason
            or (
                "llama-cpp-python is not installed on this host; "
                "build-flavor detection stays unavailable until the `llamacpp` extra is installed."
            ),
        )
    llama_cpp = imported.module

    partial_reasons: list[str] = []

    gpu_offload_supported: bool | None = None
    supports_gpu_offload = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if callable(supports_gpu_offload):
        try:
            gpu_offload_supported = bool(supports_gpu_offload())
        except (OSError, RuntimeError) as exc:
            partial_reasons.append(f"`llama_supports_gpu_offload` failed on this host: {exc}.")
    else:
        partial_reasons.append("Installed bindings do not expose `llama_supports_gpu_offload`.")

    system_info: str | None = None
    print_system_info = getattr(llama_cpp, "llama_print_system_info", None)
    if callable(print_system_info):
        try:
            raw_info = print_system_info()
        except (OSError, RuntimeError) as exc:
            partial_reasons.append(f"`llama_print_system_info` failed on this host: {exc}.")
        else:
            if isinstance(raw_info, (bytes, bytearray)):
                system_info = bytes(raw_info).decode("utf-8", errors="replace")
            elif raw_info is not None:
                system_info = str(raw_info)
    else:
        partial_reasons.append("Installed bindings do not expose `llama_print_system_info`.")

    accelerator_hints = _accelerator_hints_from_system_info(system_info)
    missing_cpu_features = _missing_cpu_features(system_info, _host_cpu_features())

    if partial_reasons:
        detection_state: LlamaCppBuildDetectionState = "partial"
        reason = (
            "llama.cpp build-flavor detection is partial on this host: "
            + " ".join(partial_reasons)
            + " Accelerator hints are heuristic inventory evidence only; probes and benchmarks decide capability evidence."
        )
    else:
        detection_state = "detected"
        reason = (
            "llama.cpp build flavor detected from the installed backend's own reporting APIs. "
            "Accelerator hints are heuristic inventory evidence only; probes and benchmarks decide capability evidence."
        )

    return LlamaCppBuildFlavor(
        installed=True,
        gpu_offload_supported=gpu_offload_supported,
        accelerator_hints=accelerator_hints,
        system_info=system_info,
        missing_cpu_features=missing_cpu_features,
        detection_state=detection_state,
        reason=reason,
    )


def _host_cpu_features() -> tuple[set[str], set[str]] | None:
    """``(present, checkable)`` x86 features on this host, or ``None`` when it cannot be read."""

    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return None
    if sys.platform == "win32":
        try:
            is_present = ctypes.windll.kernel32.IsProcessorFeaturePresent  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return None
        present = {name for name, feature in _WINDOWS_PROCESSOR_FEATURES.items() if is_present(feature)}
        return present, set(_WINDOWS_PROCESSOR_FEATURES)
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    flags_line = next((line for line in cpuinfo.splitlines() if line.startswith("flags")), "")
    flags = set(flags_line.partition(":")[2].split())
    if not flags:
        return None
    return {name for name, flag in _LINUX_CPUINFO_FLAGS.items() if flag in flags}, set(_LINUX_CPUINFO_FLAGS)


def _missing_cpu_features(system_info: str | None, host: tuple[set[str], set[str]] | None) -> list[str] | None:
    if not system_info or host is None:
        return None
    present, checkable = host
    built = {name.upper() for name, value in _SYSTEM_INFO_FLAG_PATTERN.findall(system_info) if value == "1"}
    return sorted(feature for feature in built & checkable if feature not in present)


def _accelerator_hints_from_system_info(system_info: str | None) -> list[str]:
    if not system_info:
        return []
    flag_values: dict[str, str] = {
        name.upper(): value for name, value in _SYSTEM_INFO_FLAG_PATTERN.findall(system_info)
    }
    upper_info = system_info.upper()
    hints: list[str] = []
    for hint, markers in _ACCELERATOR_MARKERS:
        for marker in markers:
            if marker in flag_values:
                if flag_values[marker] == "1":
                    hints.append(hint)
                break
        else:
            # No explicit flag pair; fall back to substring presence for
            # backend-section style system-info output.
            if any(marker in upper_info for marker in markers):
                hints.append(hint)
    return hints
