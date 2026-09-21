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
from lewlm.testing import FakeBackendFixture, FakeOpenAIEngine

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


# --- a truncated reply is distinguishable on both surfaces (G32) --------------


def _sse_frames(response: httpx.Response) -> list[dict]:
    frames: list[dict] = []
    buffer = ""
    for raw in response.iter_text():
        buffer += raw
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            for line in event.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    frames.append({"_done": True} if data == "[DONE]" else json.loads(data))
    return frames


def test_responses_publishes_the_same_finish_reason_as_chat() -> None:
    """A reply that ran out of `max_output_tokens` must not look like one that finished.

    The chat surface has always said so in `choices[0].finish_reason`; the
    responses surface carried nothing on its sync body, so a truncation
    indicator could not be built for it. Both surfaces now report the value the
    runtime produced, and the streaming terminal chunk agrees with the sync body.
    """

    with FakeBackendFixture(engine=FakeOpenAIEngine(long_reply_words=40)) as fixture:
        client = httpx.Client(base_url=fixture.base_url, timeout=30)
        truncated = "Write a long story."
        finished = "hi"

        chat_truncated = client.post(
            "/v1/chat/completions",
            json={"model": fixture.model_id, "messages": [{"role": "user", "content": truncated}], "max_tokens": 256},
        ).json()
        sync_truncated = client.post(
            "/v1/responses",
            json={"model": fixture.model_id, "input": truncated, "max_output_tokens": 256},
        ).json()
        sync_finished = client.post(
            "/v1/responses",
            json={"model": fixture.model_id, "input": finished, "max_output_tokens": 16},
        ).json()
        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": fixture.model_id, "input": truncated, "max_output_tokens": 256, "stream": True},
        ) as response:
            assert response.status_code == 200
            frames = _sse_frames(response)

        assert chat_truncated["choices"][0]["finish_reason"] == "length"
        assert sync_truncated["finish_reason"] == "length", sync_truncated
        assert sync_finished["finish_reason"] == "stop", sync_finished
        terminal = frames[-2]
        assert frames[-1] == {"_done": True} and terminal["done"] is True
        assert terminal["finish_reason"] == "length"
        # The typed model round-trips the field, so a client built from the
        # bundle reads it without a cast.
        from lewlm.api.schemas.chat import ResponseCreateResponse

        assert ResponseCreateResponse.model_validate(sync_truncated).finish_reason == "length"


def test_a_native_tool_call_is_a_tool_calls_finish_on_the_responses_surface() -> None:
    with FakeBackendFixture() as fixture:
        client = httpx.Client(base_url=fixture.base_url, timeout=30)
        body = client.post(
            "/v1/responses",
            json={
                "model": fixture.model_id,
                "input": "What is the weather in Lisbon?",
                "max_output_tokens": 64,
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Current weather for a city.",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                    },
                ],
            },
        ).json()
        assert body["finish_reason"] == "tool_calls", body


# --- the event stream resumes exactly after a drop (G13) ----------------------


def _read_frame(lines) -> dict:
    """The next SSE event frame: its `id:` line (if any) beside its parsed body."""

    current: dict = {}
    for line in lines:
        if line.startswith("id: "):
            current["id"] = line[4:]
        elif line.startswith("data: "):
            current["data"] = json.loads(line[6:])
        elif line == "" and "data" in current:
            return current
    raise AssertionError("stream ended before a frame")


def _event_frames(client: httpx.Client, path: str, *, count: int, headers: dict | None = None) -> list[dict]:
    """Read `count` frames then drop the connection, which is what a client losing its stream does."""

    with client.stream("GET", path, headers=headers) as response:
        assert response.status_code == 200, response.read()
        lines = response.iter_lines()
        return [_read_frame(lines) for _ in range(count)]


def _events_marker(frame: dict) -> dict:
    assert frame["data"]["type"] == "events.resumed"
    assert "id" not in frame, "the marker never advances the client's cursor"
    return frame["data"]["payload"]


def test_an_sse_client_can_drop_and_resume_from_the_id_it_last_saw() -> None:
    """Every frame carries `id:`; sending it back replays what was missed, once, in order, then goes live."""

    with FakeBackendFixture() as fixture:
        client = httpx.Client(base_url=fixture.base_url, timeout=30)
        bus = fixture.services.event_bus

        def chat(text: str) -> str:
            response = client.post(
                "/v1/chat/completions",
                json={"model": fixture.model_id, "messages": [{"role": "user", "content": text}], "max_tokens": 8},
            )
            assert response.status_code == 200, response.text
            return response.headers["x-request-id"]

        chat("warm-up")  # the bus issues its first cursor with its first event
        before_both = bus.latest_cursor
        assert before_both is not None
        first_request_id = chat("one")
        second_request_id = chat("two")
        completions = f"types=request.completed&request_id={first_request_id},{second_request_id}"

        # Resume from before both chats, as `?after=` (a browser EventSource
        # cannot set headers itself): both completions replay, in order, each
        # with the cursor it was published under.
        marker, completed_one, completed_two = _event_frames(client, f"/v1/events?{completions}&after={before_both}", count=3)
        assert _events_marker(marker) == {
            "after": before_both, "replayed": 2, "lost": 0, "latest": bus.latest_cursor,
            "retained": _events_marker(marker)["retained"], "replay_buffer_size": 4096,
        }
        assert completed_one["data"]["payload"]["request_id"] == first_request_id
        assert completed_two["data"]["payload"]["request_id"] == second_request_id
        assert completed_one["id"] == completed_one["data"]["cursor"], "the id: line is the cursor in the body"

        # The client dropped after the first completion. Reconnect the way an
        # EventSource does: Last-Event-ID of the last frame it saw.
        marker, missed = _event_frames(client, f"/v1/events?{completions}", count=2, headers={"Last-Event-ID": completed_one["id"]})
        payload = _events_marker(marker)
        assert payload["after"] == completed_one["id"] and payload["replayed"] == 1 and payload["lost"] == 0
        assert missed["id"] == completed_two["id"], "exactly the missed frame, under its original cursor"

        # The header is the fresher cursor: it wins over a stale `?after=` kept in the URL.
        (marker,) = _event_frames(client, f"/v1/events?{completions}&after={before_both}", count=1, headers={"Last-Event-ID": completed_two["id"]})
        assert _events_marker(marker)["after"] == completed_two["id"]

        # A cursor from another server lifetime is not matched against this one.
        (marker,) = _event_frames(client, "/v1/events?types=request.completed&after=deadbeef:1", count=1)
        assert _events_marker(marker)["lost"] is None and _events_marker(marker)["replayed"] == 0

        # Live after replay: a stream opened at the latest cursor replays nothing
        # and the next chat arrives on it, under the bus's newest cursor.
        resume_point = bus.latest_cursor
        with client.stream("GET", f"/v1/events?types=request.completed&after={resume_point}") as response:
            lines = response.iter_lines()
            assert _events_marker(_read_frame(lines))["replayed"] == 0
            third_request_id = chat("three")
            live = _read_frame(lines)
        assert live["data"]["payload"]["request_id"] == third_request_id
        epoch, _, sequence = live["id"].partition(":")
        assert epoch == resume_point.partition(":")[0] and int(sequence) > int(resume_point.partition(":")[2])
