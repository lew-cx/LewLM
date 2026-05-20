from __future__ import annotations

from types import SimpleNamespace

from lewlm.core.contracts import CapabilityName
from lewlm.runtime.mlx_vision.runtime import MLXVisionRuntime


def test_mlx_vision_runtime_reports_vision_capability_when_generation_is_available(monkeypatch) -> None:
    runtime = MLXVisionRuntime()
    monkeypatch.setattr(runtime, "is_available", lambda: True)
    monkeypatch.setattr(
        "lewlm.runtime.mlx_vision.runtime.import_module",
        lambda name: SimpleNamespace(generate=lambda **kwargs: "ok"),
    )

    assert runtime.supports_capability(CapabilityName.VISION) is True

