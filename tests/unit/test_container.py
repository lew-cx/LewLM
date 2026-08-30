from __future__ import annotations

from pathlib import Path

import pytest

from lewlm.container import ContainerStatus, detect_container


@pytest.fixture(autouse=True)
def _clear_marker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEWLM_IN_CONTAINER", raising=False)


def _no_marker_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "exists", lambda self: False)


def _no_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lewlm.container._cgroup_runtime", lambda: None)


def test_native_host_reports_no_container(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_marker_files(monkeypatch)
    _no_cgroup(monkeypatch)

    status = detect_container()

    assert status.in_container is False
    assert status.runtime is None
    assert status.indicators == []
    assert "native host" in status.reason


def test_explicit_marker_env_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_marker_files(monkeypatch)
    _no_cgroup(monkeypatch)
    monkeypatch.setenv("LEWLM_IN_CONTAINER", "true")

    status = detect_container()

    assert status.in_container is True
    # No signal named a runtime, so the report stays honest about not knowing.
    assert status.runtime == "unknown"
    assert any("LEWLM_IN_CONTAINER" in indicator for indicator in status.indicators)


def test_marker_file_names_the_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    # `as_posix()` so the comparison holds on Windows, where `str()` would
    # render the same path with a backslash.
    monkeypatch.setattr(Path, "exists", lambda self: self.as_posix() == "/.dockerenv")
    _no_cgroup(monkeypatch)

    status = detect_container()

    assert status.in_container is True
    assert status.runtime == "docker"
    assert any("/.dockerenv" in indicator for indicator in status.indicators)


def test_cgroup_names_the_runtime_without_marker_files(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_marker_files(monkeypatch)
    monkeypatch.setattr("lewlm.container._cgroup_runtime", lambda: "kubernetes")

    status = detect_container()

    assert status.in_container is True
    assert status.runtime == "kubernetes"
    assert any("cgroup" in indicator for indicator in status.indicators)


def test_unreadable_cgroup_is_not_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A host without `/proc/1/cgroup` must not be reported either way."""

    _no_marker_files(monkeypatch)

    def _raise(*_args: object, **_kwargs: object) -> str:
        raise OSError("no /proc on this host")

    monkeypatch.setattr(Path, "read_text", _raise)

    status = detect_container()

    assert status.in_container is False
    assert status.indicators == []


def test_status_model_round_trips() -> None:
    status = ContainerStatus(
        in_container=True,
        runtime="podman",
        indicators=["`/run/.containerenv` exists"],
        reason="test",
    )

    assert ContainerStatus.model_validate(status.model_dump()) == status
