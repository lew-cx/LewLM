"""Sandbox worker helpers for parser-style operations."""

from __future__ import annotations

import os
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, ParamSpec, TypeVar

from lewlm.core.errors import LewLMError, SandboxExecutionError


P = ParamSpec("P")
T = TypeVar("T")


def run_in_subprocess(
    target: Callable[P, T],
    /,
    *args: P.args,
    operation: str,
    timeout_seconds: int,
    enabled: bool,
    clear_environment: bool = True,
    workspace_root: str | Path | None = None,
    **kwargs: P.kwargs,
) -> T:
    """Execute a target function in a dedicated spawned worker process."""

    if not enabled:
        return target(*args, **kwargs)
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    resolved_workspace_root = Path(workspace_root).expanduser().resolve(strict=False) if workspace_root is not None else None
    process = ctx.Process(
        target=_sandbox_worker,
        args=(
            child_conn,
            target,
            args,
            kwargs,
            clear_environment,
            str(resolved_workspace_root) if resolved_workspace_root is not None else None,
        ),
    )
    process.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout=max(1, timeout_seconds)):
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            raise SandboxExecutionError(
                f"{operation} exceeded the sandbox timeout.",
                details={
                    "operation": operation,
                    "timeout_seconds": timeout_seconds,
                    "clear_environment": clear_environment,
                    "workspace_root": str(resolved_workspace_root) if resolved_workspace_root is not None else None,
                },
            )
        payload = parent_conn.recv()
    except EOFError as exc:
        raise SandboxExecutionError(
            f"{operation} failed in the sandbox worker.",
            details={
                "operation": operation,
                "error": "sandbox worker exited before returning a result",
                "clear_environment": clear_environment,
                "workspace_root": str(resolved_workspace_root) if resolved_workspace_root is not None else None,
            },
        ) from exc
    finally:
        parent_conn.close()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
    if payload["kind"] == "result":
        return payload["value"]
    if payload["kind"] == "lewlm_error":
        raise payload["error"]
    raise SandboxExecutionError(
        f"{operation} failed in the sandbox worker.",
        details={
            "operation": operation,
            "error": payload["error"],
            "error_type": payload["error_type"],
            "clear_environment": clear_environment,
            "workspace_root": str(resolved_workspace_root) if resolved_workspace_root is not None else None,
        },
    )


def _sandbox_worker(
    connection: Any,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    clear_environment: bool,
    workspace_root: str | None,
) -> None:
    try:
        if clear_environment:
            _clear_environment()
        workspace_dir: str | None = None
        if workspace_root is not None:
            workspace_path = Path(workspace_root)
            workspace_path.mkdir(parents=True, exist_ok=True)
            workspace_dir = str(workspace_path)
        with TemporaryDirectory(prefix="lewlm-sandbox-", dir=workspace_dir) as sandbox_dir:
            os.chdir(sandbox_dir)
            connection.send({"kind": "result", "value": target(*args, **kwargs)})
    except LewLMError as exc:
        connection.send({"kind": "lewlm_error", "error": exc})
    except Exception as exc:  # noqa: BLE001 - returned to parent as structured sandbox failure
        connection.send(
            {
                "kind": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
    finally:
        connection.close()


def _clear_environment() -> None:
    preserved = {
        key: value
        for key, value in os.environ.items()
        if key in {
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "PYTHONHOME",
            "PYTHONPATH",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "TZ",
            "USERPROFILE",
            "VIRTUAL_ENV",
            "WINDIR",
        }
    }
    os.environ.clear()
    os.environ.update(preserved)
