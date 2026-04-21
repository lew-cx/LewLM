"""Secure local temporary workspace helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile


@contextmanager
def secure_workspace(root: Path, *, prefix: str) -> Iterator[Path]:
    workspace_root = root.expanduser().resolve(strict=False)
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(dir=workspace_root, prefix=prefix))
    workspace.chmod(0o700)
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace)
