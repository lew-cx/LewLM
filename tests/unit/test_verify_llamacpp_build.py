"""The build-flavor verifier refuses the wrong wheel instead of trusting pip."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from lewlm.runtime.llamacpp.build_flavor import LlamaCppBuildFlavor

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_verifier():
    spec = spec_from_file_location("lewlm_verify_llamacpp_build", REPO_ROOT / "scripts" / "verify_llamacpp_build.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flavor(*, offload: bool | None, hints: list[str] | None = None, state: str = "detected") -> LlamaCppBuildFlavor:
    return LlamaCppBuildFlavor(
        installed=True,
        gpu_offload_supported=offload,
        accelerator_hints=hints or [],
        detection_state=state,  # type: ignore[arg-type]
        reason="pinned for test",
    )


@pytest.mark.parametrize(
    ("flavor", "expect", "hint", "code"),
    [
        (_flavor(offload=False), "cpu", None, 0),
        (_flavor(offload=True, hints=["cuda"]), "cpu", None, 1),
        (_flavor(offload=True, hints=["cuda"]), "gpu", "cuda", 0),
        (_flavor(offload=True, hints=["metal"]), "gpu", "cuda", 1),
        (_flavor(offload=False), "gpu", None, 1),
        (_flavor(offload=None), "cpu", None, 1),
        (_flavor(offload=None), "any", None, 0),
    ],
)
def test_verdicts(flavor: LlamaCppBuildFlavor, expect: str, hint: str | None, code: int) -> None:
    verifier = _load_verifier()
    got, message = verifier.evaluate(flavor, expect=expect, hint=hint)
    assert got == code, message


def test_unloadable_backend_is_its_own_exit_code() -> None:
    verifier = _load_verifier()
    code, message = verifier.evaluate(_flavor(offload=None, state="unavailable"), expect="cpu", hint=None)
    assert code == 2
    assert "not usable" in message


def test_main_uses_live_detection_and_json(monkeypatch, capsys) -> None:
    verifier = _load_verifier()
    monkeypatch.setattr(
        "lewlm.runtime.llamacpp.build_flavor.detect_llamacpp_build_flavor",
        lambda: _flavor(offload=True, hints=["cuda"]),
    )

    assert verifier.main(["--expect", "gpu", "--hint", "cuda", "--json"]) == 0
    out = capsys.readouterr().out
    assert '"exit_code": 0' in out and '"accelerator_hints"' in out

    assert verifier.main(["--expect", "cpu"]) == 1
    assert "FAIL" in capsys.readouterr().err
