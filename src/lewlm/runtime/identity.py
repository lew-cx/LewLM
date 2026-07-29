"""Stable identity for one model-owning LewLM service container."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
import hashlib
import os
from pathlib import Path
import socket
import subprocess
from uuid import uuid4

from pydantic import BaseModel, Field

from lewlm.core.contracts import utc_now

#: Version of the public HTTP/Python contract shape, independent of the package
#: version. A caller pins against this to know which surfaces it can rely on.
API_SCHEMA_VERSION = "v1"


class RuntimeBuildIdentity(BaseModel):
    """Proof of which implementation produced an artifact.

    A remote LewLM reporting only a package version cannot prove which build is
    running — an editable checkout, a patched wheel, and a release all report
    the same string. These fields let a caller distinguish them.
    """

    package_version: str
    api_schema_version: str = API_SCHEMA_VERSION
    source_commit: str | None = Field(
        default=None,
        description="Git commit of the running source tree, when it can be determined.",
    )
    source_dirty: bool | None = Field(
        default=None,
        description="Whether the working tree had uncommitted changes. Null when unknown.",
    )
    distribution_digest: str | None = Field(
        default=None,
        description="Stable digest of installed package metadata, for comparing two deployments.",
    )
    install_kind: str = Field(
        default="unknown",
        description="One of `editable`, `installed`, or `unknown`.",
    )
    release_build: bool = Field(
        default=False,
        description="True only for a clean, non-editable install with no local modifications.",
    )

    @classmethod
    def detect(cls, *, version: str) -> "RuntimeBuildIdentity":
        return _detect_build_identity(version)


def _run_git(argument: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *argument],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _source_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _distribution_digest() -> tuple[str | None, str]:
    """Return a metadata digest and the install kind for the running package."""

    try:
        dist = distribution("lewlm")
    except PackageNotFoundError:
        return None, "unknown"
    except Exception:  # pragma: no cover - defensive against metadata backends
        return None, "unknown"

    install_kind = "installed"
    try:
        if any(str(path).endswith("__editable__.lewlm.pth") for path in (dist.files or ())):
            install_kind = "editable"
        elif dist.read_text("direct_url.json") and '"editable": true' in (dist.read_text("direct_url.json") or ""):
            install_kind = "editable"
    except Exception:  # pragma: no cover - metadata layouts vary by installer
        pass

    digest = hashlib.sha256()
    for name in ("METADATA", "RECORD", "WHEEL"):
        try:
            content = dist.read_text(name)
        except Exception:  # pragma: no cover - optional metadata files
            content = None
        if content:
            digest.update(name.encode("utf-8"))
            digest.update(content.encode("utf-8"))
    computed = digest.hexdigest()
    # An all-empty digest carries no information, so report it as unknown.
    return (computed if computed != hashlib.sha256().hexdigest() else None), install_kind


@lru_cache(maxsize=1)
def _detect_build_identity(version: str) -> RuntimeBuildIdentity:
    """Detect build provenance once per process; none of it changes at runtime."""

    root = _source_root()
    commit = None
    dirty: bool | None = None
    if (root / ".git").exists():
        commit = _run_git(["rev-parse", "HEAD"], cwd=root)
        if commit is not None:
            status = _run_git(["status", "--porcelain"], cwd=root)
            dirty = bool(status) if status is not None else None

    digest, install_kind = _distribution_digest()
    return RuntimeBuildIdentity(
        package_version=version,
        source_commit=commit,
        source_dirty=dirty,
        distribution_digest=digest,
        install_kind=install_kind,
        release_build=install_kind == "installed" and dirty is not True,
    )


class RuntimeInstanceMetadata(BaseModel):
    """Process metadata that remains stable for a service-container lifetime."""

    runtime_instance_id: str
    started_at: datetime
    process_id: int
    hostname: str
    version: str
    build: RuntimeBuildIdentity

    @classmethod
    def create(cls, *, version: str) -> "RuntimeInstanceMetadata":
        return cls(
            runtime_instance_id=str(uuid4()),
            started_at=utc_now(),
            process_id=os.getpid(),
            hostname=socket.gethostname(),
            version=version,
            build=RuntimeBuildIdentity.detect(version=version),
        )


class RuntimeInfo(RuntimeInstanceMetadata):
    """Public runtime identity and live process summary."""

    status: str = "ready"
    loaded_model_count: int = 0
    active_request_count: int = 0
    enabled_features: list[str] = Field(
        default_factory=list,
        description="Public feature surfaces this build has enabled.",
    )
