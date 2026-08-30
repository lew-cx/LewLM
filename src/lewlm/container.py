"""Detect whether this LewLM process is running inside a container.

Deployment shape changes what advice is useful. On a non-Apple host LewLM's
packaged runtime family is GGUF/llama.cpp, and the reliable way to get a
GPU-capable llama.cpp plus the HF-to-GGUF conversion tools onto that host is
the shipped container image rather than a native build. Telling an operator
that when they are *already* inside that image is noise; telling them when they
are not is the whole point.

Detection is deliberately shallow and honest. Every signal here is an
indicator, never a guarantee: containers can be built without any of these
markers, and a host can carry one without being a container. `indicators`
carries what was actually observed so the report explains itself instead of
asking anyone to trust a bare boolean.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

#: Set by LewLM's own images so the container answer never depends on probing.
_EXPLICIT_MARKER_ENV = "LEWLM_IN_CONTAINER"

_MARKER_FILES: tuple[tuple[str, str], ...] = (
    ("/.dockerenv", "docker"),
    ("/run/.containerenv", "podman"),
)

_CGROUP_MARKERS: tuple[tuple[str, str], ...] = (
    ("docker", "docker"),
    ("containerd", "containerd"),
    ("kubepods", "kubernetes"),
    ("libpod", "podman"),
)

ContainerRuntimeName = Literal["docker", "podman", "containerd", "kubernetes", "unknown"]


class ContainerStatus(BaseModel):
    """What LewLM can honestly say about running inside a container.

    ``runtime`` names the container runtime only when a signal identified one;
    an explicit marker with no other evidence reports ``"unknown"``. ``reason``
    is operator-facing prose and always explains the verdict, including the
    negative one.
    """

    in_container: bool
    runtime: ContainerRuntimeName | None = None
    indicators: list[str] = Field(default_factory=list)
    reason: str


def detect_container() -> ContainerStatus:
    """Report container residency from the markers visible to this process."""

    indicators: list[str] = []
    runtimes: list[str] = []

    explicit = os.environ.get(_EXPLICIT_MARKER_ENV, "").strip().casefold()
    if explicit in {"1", "true", "yes", "on"}:
        indicators.append(f"`{_EXPLICIT_MARKER_ENV}` is set in the environment")

    for marker_path, runtime_name in _MARKER_FILES:
        if Path(marker_path).exists():
            indicators.append(f"`{marker_path}` exists")
            runtimes.append(runtime_name)

    cgroup_runtime = _cgroup_runtime()
    if cgroup_runtime is not None:
        indicators.append(f"`/proc/1/cgroup` names `{cgroup_runtime}`")
        runtimes.append(cgroup_runtime)

    if not indicators:
        return ContainerStatus(
            in_container=False,
            runtime=None,
            indicators=[],
            reason=(
                "No container markers were visible to this process, so LewLM treats this as a native host. "
                "Container images that strip the usual markers can still be reported this way."
            ),
        )

    runtime: ContainerRuntimeName = runtimes[0] if runtimes else "unknown"  # type: ignore[assignment]
    return ContainerStatus(
        in_container=True,
        runtime=runtime,
        indicators=indicators,
        reason=(
            f"LewLM is running inside a container ({runtime}); detected from: {', '.join(indicators)}. "
            "These are indicators, not proof, and they do not by themselves say which runtimes are installed."
        ),
    )


def _cgroup_runtime() -> str | None:
    """Return the container runtime named by PID 1's cgroup, when readable."""

    try:
        cgroup_text = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        # Absent on non-Linux hosts and unreadable under some sandboxes; both
        # are ordinary, and neither is evidence either way.
        return None
    lowered = cgroup_text.casefold()
    for marker, runtime_name in _CGROUP_MARKERS:
        if marker in lowered:
            return runtime_name
    return None
