"""Abandoning a stream must stop generation, not just stop reading it."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def _gguf_model_id(client: TestClient) -> str:
    manifests = client.post("/v1/models/scan", json={}).json()["manifests"]
    return next(item["model_id"] for item in manifests if item["format_type"] == "gguf")


def _serving_core(app):
    return app.state.services.chat_orchestrator.serving_core


def test_fully_consumed_stream_is_not_reported_as_cancelled(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        model_id = _gguf_model_id(client)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hello."}],
                "stream": True,
            },
        ) as response:
            frames = [line for line in response.iter_lines() if line.startswith("data: ")]

        snapshot = _serving_core(app_with_fake_runtime).snapshot()

    assert frames[-1].endswith("[DONE]")
    assert snapshot.total_cancellation_requests == 0


def test_final_chunk_reports_usage(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        model_id = _gguf_model_id(client)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Count my tokens."}],
                "stream": True,
            },
        ) as response:
            payloads = [
                json.loads(line[len("data: ") :])
                for line in response.iter_lines()
                if line.startswith("data: ") and not line.endswith("[DONE]")
            ]

    # Usage is only knowable at the end, so it belongs on the final chunk only.
    assert all(item.get("usage") is None for item in payloads[:-1])
    usage = payloads[-1]["usage"]
    assert usage is not None
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    # The fake runtime exposes a tokenizer, so these are measured, not estimated.
    assert usage["measured"] is True


def test_responses_final_chunk_reports_usage(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        model_id = _gguf_model_id(client)
        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": model_id, "input": "Count my tokens.", "stream": True},
        ) as response:
            payloads = [
                json.loads(line[len("data: ") :])
                for line in response.iter_lines()
                if line.startswith("data: ") and not line.endswith("[DONE]")
            ]

    final = payloads[-1]
    assert final["done"] is True
    assert final["usage"]["total_tokens"] > 0
