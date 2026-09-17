"""A stream that fails after output has started ends with a structured terminal
event, not a reset socket, and is never replayed."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import FakeLlamaCppRuntime, FakeMLXAudioRuntime, UnavailableMLXTextRuntime, UnavailableMLXVisionRuntime
from lewlm.api.app import create_app
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import GenerateRequest, RuntimeAffinity
from lewlm.core.errors import RuntimeUnavailableError


class _DiesMidStreamRuntime(FakeLlamaCppRuntime):
    """Yields one delta, then fails the way an upstream socket closing does."""

    name = "fake_llamacpp"

    def __init__(self) -> None:
        super().__init__()
        self.stream_starts = 0

    def supports_continuous_batching(self, capability) -> bool:
        # Take the single-request stream path, which is the bridge's path.
        return False

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        self.stream_starts += 1
        yield "partial "
        raise RuntimeUnavailableError(
            "External accelerator stream failed.",
            details={"runtime": self.name, "path": "/v1/chat/completions", "body": "SECRET upstream body"},
        )


def _app(temp_settings: LewLMSettings, runtime: _DiesMidStreamRuntime):
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.LLAMACPP: runtime,
            RuntimeAffinity.MLX_TEXT: UnavailableMLXTextRuntime(settings=temp_settings),
            RuntimeAffinity.MLX_VISION: UnavailableMLXVisionRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )
    return create_app(temp_settings, services=services)


def _frames(response) -> list[dict]:
    frames = []
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        frames.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return frames


def test_chat_stream_failure_ends_with_error_chunk_and_done(temp_settings: LewLMSettings, sample_models_root: Path) -> None:
    runtime = _DiesMidStreamRuntime()
    with TestClient(_app(temp_settings, runtime)) as client:
        manifests = client.post("/v1/models/scan", json={}).json()["manifests"]
        model_id = next(item["model_id"] for item in manifests if item["format_type"] == "gguf")
        with client.stream(
            "POST", "/v1/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "Hello."}], "stream": True},
        ) as response:
            assert response.status_code == 200
            frames = _frames(response)

    assert frames[-1] == "[DONE]"
    terminal = frames[-2]
    assert terminal["choices"][0]["finish_reason"] == "error"
    assert terminal["error"]["code"] == "runtime_unavailable"
    assert terminal["error"]["partial_output"] is True
    assert "SECRET" not in json.dumps(terminal), "upstream bodies never reach the wire"
    assert any(frame != "[DONE]" and (frame["choices"][0]["delta"].get("content") or "") for frame in frames[:-2])
    assert runtime.stream_starts == 1, "no automatic replay after partial output"


def test_responses_stream_failure_ends_with_error_chunk_and_done(temp_settings: LewLMSettings, sample_models_root: Path) -> None:
    runtime = _DiesMidStreamRuntime()
    with TestClient(_app(temp_settings, runtime)) as client:
        manifests = client.post("/v1/models/scan", json={}).json()["manifests"]
        model_id = next(item["model_id"] for item in manifests if item["format_type"] == "gguf")
        with client.stream(
            "POST", "/v1/responses",
            json={"model": model_id, "input": "Hello.", "stream": True},
        ) as response:
            assert response.status_code == 200
            frames = _frames(response)

    assert frames[-1] == "[DONE]"
    terminal = frames[-2]
    assert terminal["done"] is True
    assert terminal["error"]["code"] == "runtime_unavailable"
    assert terminal["error"]["partial_output"] is True
    assert runtime.stream_starts == 1
