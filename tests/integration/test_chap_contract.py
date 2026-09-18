"""Modernization step 10: the Chap-facing contract, proven over HTTP with no model.

- the smoke script passes in fixture mode (what CI and any OS can run);
- the integration bundle's generated parts match this checkout and its Chap
  examples validate against the public models;
- a narrowly configured browser origin gets CORS on JSON and SSE responses,
  another origin gets nothing, and no wildcard is ever emitted.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from lewlm.api.schemas.chat import ChatCompletionChunk, ChatCompletionResponse
from lewlm.api.schemas.health import HealthResponse
from lewlm.core.contracts import ModelCapabilityAvailability
from lewlm.core.errors import error_from_dict
from lewlm.runtime.identity import RuntimeInfo
from lewlm.testing import FakeBackendFixture

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "examples" / "integration-bundle.json"


def test_chap_smoke_passes_in_fixture_mode(tmp_path: Path) -> None:
    report_path = tmp_path / "chap-smoke.json"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "chap_backend_smoke.py"), "--fixture", "--stream-delay-ms", "5", "--output", str(report_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["format"] == "lewlm-chap-smoke-v1" and report["mode"] == "fixture"
    assert report["summary"]["failed"] == 0
    by_name = {check["name"]: check for check in report["checks"]}
    expected = {"service_health", "runtime_startup", "model_picker", "request_identity", "chat_streaming", "chat_nonstreaming",
                "structured_output", "tools", "reasoning", "cancellation", "failure", "unavailable_engine", "stream_interrupted"}
    assert expected <= set(by_name)
    assert all(by_name[name]["status"] == "passed" for name in expected), {n: by_name[n]["status"] for n in expected}
    assert by_name["unavailable_engine"]["observations"]["status"] == 503
    assert by_name["unavailable_engine"]["observations"]["engine_state_in_health"] in {"failed", "stale"}
    assert by_name["stream_interrupted"]["observations"]["terminal_finish_reason"] == "error"
    assert by_name["tools"]["observations"]["tool_calls"][0]["name"] == "get_weather"


def test_integration_bundle_generated_parts_match_the_checkout() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_integration_bundle.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


def test_a_drifted_bundle_is_a_contract_failure_that_blocks_the_gate(tmp_path: Path) -> None:
    """Step 11's deliberate contract failure: one changed schema fails `--check`."""

    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    bundle["schemas"]["chat.stream"]["properties"].pop("usage")
    drifted = tmp_path / "integration-bundle.json"
    drifted.write_text(json.dumps(bundle), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_integration_bundle.py"), "--check", "--bundle", str(drifted)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 1
    assert "out of date for: chat.stream" in completed.stderr


def test_bundle_chap_examples_validate_against_the_public_models() -> None:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    chap = bundle["chap"]
    assert chap["smoke_script"] == "examples/chap_backend_smoke.py"
    assert chap["fixture"].startswith("python -m lewlm.testing.fake_backend")
    for key in ("identity", "service_vs_engine_vs_model", "model_picker", "chat_metadata", "usage", "streaming", "incomplete_stream", "errors", "absent_fields"):
        assert key in chap["field_notes"]
    examples = chap["examples"]

    HealthResponse.model_validate(examples["health"])
    HealthResponse.model_validate(examples["health.engine_down"])
    assert examples["health"]["engines"][0]["state"] == "advertised"
    assert examples["health.engine_down"]["status"] == "ok"
    assert examples["health.engine_down"]["engines"][0]["state"] in {"failed", "stale"}
    RuntimeInfo.model_validate(examples["runtime"])
    assert examples["runtime"]["startup"]["engines"][0]["endpoint_id"] == "fixture"
    for item in examples["models.capability_availability"]:
        availability = ModelCapabilityAvailability.model_validate(item)
        assert availability.endpoint_id == "fixture" and availability.engine_state == "advertised"

    stream = examples["chat.stream.text"]
    ChatCompletionChunk.model_validate(stream["first_content_chunk"])
    terminal = ChatCompletionChunk.model_validate(stream["terminal_chunk"])
    assert terminal.choices[0].finish_reason == "stop" and terminal.usage is not None
    assert terminal.usage.cached_tokens == 8, "the fixture engine reports cached tokens; the bundle shows the field"

    tool = ChatCompletionResponse.model_validate(examples["chat.native_tool_call"])
    assert tool.tool_calls is not None and tool.choices[0].finish_reason == "tool_calls"
    output = ChatCompletionResponse.model_validate(examples["chat.json_output"])
    assert output.structured_output is not None and json.loads(output.choices[0].message.content)["city"] == "Paris"
    assert output.metadata.model.endpoint_id == "fixture" and output.metadata.model.engine_profile == "openai_compatible"

    cancelled = examples["chat.cancellation"]
    assert cancelled["cancel_response_after_first_chunk"]["state"] in {"cancelling", "cancelled"}
    assert cancelled["cancel_response_after_stream_ended"]["state"] == "cancelled"
    ChatCompletionChunk.model_validate(cancelled["terminal_chunk"])
    assert cancelled["terminal_chunk"]["choices"][0]["finish_reason"] == "cancelled"

    interrupted = ChatCompletionChunk.model_validate(examples["chat.stream.interrupted"]["terminal_chunk"])
    assert interrupted.choices[0].finish_reason == "error"
    assert interrupted.error is not None and interrupted.error.partial_output is True

    outage = examples["chat.unavailable_engine"]
    assert outage["http_status"] == 503
    rehydrated = error_from_dict(outage["body"]["error"])
    assert rehydrated.code == "runtime_unavailable" and rehydrated.details["endpoint_id"] == "fixture"
    assert examples["chat.model_not_found"]["body"]["error"]["code"] == "model_not_found"


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
def test_a_named_cancel_ends_the_stream_with_a_cancelled_terminal_chunk_then_done(route: str) -> None:
    with FakeBackendFixture() as fixture:
        client = httpx.Client(base_url=fixture.base_url, timeout=30)
        request_id = "chap-cancel-terminal"
        body = {"model": fixture.model_id, "max_tokens": 256, "stream": True}
        body["messages" if route.endswith("completions") else "input"] = (
            [{"role": "user", "content": "Write a long story."}] if route.endswith("completions") else "Write a long story."
        )
        frames: list[dict] = []
        cancelled_state = None
        with client.stream("POST", route, json=body, headers={"x-request-id": request_id}) as response:
            assert response.status_code == 200
            buffer = ""
            for raw in response.iter_text():
                buffer += raw
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.splitlines():
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            frames.append({"_done": True} if data == "[DONE]" else json.loads(data))
                if cancelled_state is None and len(frames) >= 2:
                    cancelled_state = client.post(f"/v1/requests/{request_id}/cancel").json()["state"]
        assert cancelled_state in {"cancelling", "cancelled"}
        assert frames[-1] == {"_done": True}, "the stream ends with [DONE], not a dropped socket"
        terminal = frames[-2]
        if route.endswith("completions"):
            assert terminal["choices"][0]["finish_reason"] == "cancelled"
        else:
            assert terminal["done"] is True and terminal["finish_reason"] == "cancelled"
        assert terminal.get("error") is None
        assert len(frames) < 200, "the stream stopped well before the 200-word reply finished"
        assert client.post(f"/v1/requests/{request_id}/cancel").json()["state"] == "cancelled"
        # The fake engine notices the closed socket on its next write.
        deadline = time.monotonic() + 10.0
        while fixture.engine.disconnects < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert fixture.engine.disconnects >= 1, "the upstream connection was closed, not left running"


@pytest.mark.parametrize("stream", [False, True])
def test_a_narrow_browser_origin_gets_cors_without_a_wildcard(stream: bool) -> None:
    allowed = "http://localhost:5173"
    with FakeBackendFixture(cors_allow_origins=(allowed,)) as fixture:
        client = httpx.Client(base_url=fixture.base_url, timeout=30)
        preflight = client.options(
            "/v1/chat/completions",
            headers={"Origin": allowed, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type,x-request-id"},
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == allowed
        assert "x-request-id" in preflight.headers["access-control-allow-headers"].lower()

        payload = {"model": fixture.model_id, "messages": [{"role": "user", "content": "count"}], "max_tokens": 16, "stream": stream}
        with client.stream("POST", "/v1/chat/completions", json=payload, headers={"Origin": allowed, "x-request-id": "chap-cors-1"}) as response:
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == allowed
            assert "x-request-id" in response.headers["access-control-expose-headers"].lower()
            assert response.headers["x-request-id"] == "chap-cors-1"
            if stream:
                assert response.headers["content-type"].startswith("text/event-stream")
            body = b"".join(response.iter_bytes())
        assert b"[DONE]" in body if stream else b"choices" in body

        other = client.get("/v1/health", headers={"Origin": "http://evil.example"})
        assert other.status_code == 200
        assert "access-control-allow-origin" not in other.headers
        assert "*" not in preflight.headers.get("access-control-allow-origin", "")
