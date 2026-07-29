"""Contract-completeness coverage: error envelope, CORS, request IDs, sessions, sampling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lewlm.api.app import create_app
from lewlm.config.settings import LewLMSettings


@pytest.fixture()
def client(app_with_fake_runtime):
    with TestClient(app_with_fake_runtime, raise_server_exceptions=False) as test_client:
        test_client.post("/v1/models/scan", json={})
        yield test_client


@pytest.fixture()
def session_client(app_with_fake_runtime_session_enabled):
    with TestClient(app_with_fake_runtime_session_enabled, raise_server_exceptions=False) as test_client:
        test_client.post("/v1/models/scan", json={})
        yield test_client


def _gguf_model_id(client: TestClient) -> str:
    manifests = client.post("/v1/models/scan", json={}).json()["manifests"]
    return next(item["model_id"] for item in manifests if item["format_type"] == "gguf")


# --- every failure uses one envelope -----------------------------------------


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("wrong field type", {"messages": "not-a-list"}),
        ("missing required field", {}),
        ("bad nested type", {"messages": [{"role": 5, "content": "hi"}]}),
        ("out-of-range value", {"messages": [{"role": "user", "content": "hi"}], "sampling": {"top_p": 5}}),
    ],
)
def test_malformed_requests_return_the_error_envelope(client, label: str, payload: dict) -> None:
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 422, label
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    # A caller must be able to point a user at the offending field.
    assert error["details"]["fields"], label
    assert all({"field", "message", "type"} <= set(item) for item in error["details"]["fields"])


def test_malformed_json_body_returns_the_error_envelope(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=b"{not valid json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_unknown_route_uses_the_error_envelope(client) -> None:
    response = client.get("/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_wrong_method_uses_the_error_envelope(client) -> None:
    response = client.delete("/v1/health")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_error_envelopes_carry_the_request_id(client) -> None:
    response = client.get("/v1/does-not-exist", headers={"x-request-id": "trace-404"})
    assert response.headers["x-request-id"] == "trace-404"


# --- request identifiers -----------------------------------------------------


def test_caller_request_id_is_echoed(client) -> None:
    response = client.get("/v1/health", headers={"x-request-id": "caller-supplied-1"})
    assert response.headers["x-request-id"] == "caller-supplied-1"


def test_a_request_id_is_minted_when_absent(client) -> None:
    first = client.get("/v1/health").headers["x-request-id"]
    second = client.get("/v1/health").headers["x-request-id"]
    assert first and second and first != second


def test_request_id_and_correlation_id_are_independent(client) -> None:
    response = client.get(
        "/v1/health",
        headers={"x-request-id": "req-1", "x-lewlm-correlation-id": "corr-1"},
    )
    assert response.headers["x-request-id"] == "req-1"
    assert response.headers["x-lewlm-correlation-id"] == "corr-1"


# --- CORS --------------------------------------------------------------------


def test_cors_is_off_by_default(client) -> None:
    response = client.get("/v1/health", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_a_configured_origin(tmp_path: Path) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path,
        cors_enabled=True,
        cors_allow_origins=("http://localhost:3000",),
    )
    with TestClient(create_app(settings=settings)) as client:
        preflight = client.options(
            "/v1/health",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"

        response = client.get("/v1/health", headers={"Origin": "http://localhost:3000"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        # Browser callers need the correlation headers readable from JS.
        exposed = response.headers["access-control-expose-headers"]
        assert "x-request-id" in exposed
        assert "x-lewlm-correlation-id" in exposed


def test_cors_rejects_an_unconfigured_origin(tmp_path: Path) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path,
        cors_enabled=True,
        cors_allow_origins=("http://localhost:3000",),
    )
    with TestClient(create_app(settings=settings)) as client:
        response = client.get("/v1/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_requires_an_explicit_origin_list() -> None:
    with pytest.raises(Exception, match="cors_allow_origins"):
        LewLMSettings(cors_enabled=True)


def test_wildcard_origin_cannot_be_combined_with_credentials() -> None:
    # A loopback model server behind a wildcard-with-credentials CORS policy is
    # reachable by any page the operator visits.
    with pytest.raises(Exception, match="wildcard origin"):
        LewLMSettings(cors_enabled=True, cors_allow_origins=("*",), cors_allow_credentials=True)


# --- session rename ----------------------------------------------------------


def test_a_session_can_be_renamed(session_client) -> None:
    client = session_client
    session_id = client.post("/v1/sessions", json={"title": "Old", "metadata": {"keep": "yes"}}).json()["session_id"]

    response = client.patch(f"/v1/sessions/{session_id}", json={"title": "New"})

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "New"
    # Renaming must not quietly discard anything else about the session.
    assert body["metadata"] == {"keep": "yes"}


def test_session_metadata_merges_by_default(session_client) -> None:
    client = session_client
    session_id = client.post("/v1/sessions", json={"metadata": {"a": "1"}}).json()["session_id"]
    body = client.patch(f"/v1/sessions/{session_id}", json={"metadata": {"b": "2"}}).json()
    assert body["metadata"] == {"a": "1", "b": "2"}


def test_session_metadata_can_be_replaced_outright(session_client) -> None:
    client = session_client
    session_id = client.post("/v1/sessions", json={"metadata": {"a": "1"}}).json()["session_id"]
    body = client.patch(
        f"/v1/sessions/{session_id}",
        json={"metadata": {"b": "2"}, "replace_metadata": True},
    ).json()
    assert body["metadata"] == {"b": "2"}


def test_an_empty_session_update_is_rejected(session_client) -> None:
    client = session_client
    session_id = client.post("/v1/sessions", json={"title": "Keep"}).json()["session_id"]
    response = client.patch(f"/v1/sessions/{session_id}", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_renaming_a_missing_session_is_a_typed_404(session_client) -> None:
    client = session_client
    response = client.patch("/v1/sessions/does-not-exist", json={"title": "x"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_renaming_preserves_turn_history(session_client) -> None:
    client = session_client
    session_id = client.post("/v1/sessions", json={"title": "Chat"}).json()["session_id"]
    model_id = _gguf_model_id(client)
    client.post(
        "/v1/chat/completions",
        json={"model": model_id, "session_id": session_id, "messages": [{"role": "user", "content": "hi"}]},
    )
    before = client.get(f"/v1/sessions/{session_id}").json()

    client.patch(f"/v1/sessions/{session_id}", json={"title": "Renamed"})
    after = client.get(f"/v1/sessions/{session_id}").json()

    assert after["title"] == "Renamed"
    assert len(after["turns"]) == len(before["turns"]) == 1


# --- sampling controls -------------------------------------------------------


def test_sampling_controls_are_reported_as_applied_or_unsupported(client) -> None:
    model_id = _gguf_model_id(client)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "sampling": {"top_p": 0.9, "top_k": 40, "seed": 7},
        },
    )

    assert response.status_code == 200, response.text
    sampling = response.json()["metadata"]["sampling"]
    assert sampling is not None
    assert sampling["requested"] == {"top_p": 0.9, "top_k": 40, "seed": 7}
    # Every requested control is accounted for, either applied or unsupported.
    accounted = set(sampling["applied"]) | set(sampling["unsupported"])
    assert accounted == {"top_p", "top_k", "seed"}
    # Determinism is only claimed when the seed actually reached the backend.
    assert sampling["deterministic"] == ("seed" in sampling["applied"])


def test_no_sampling_report_when_no_controls_are_requested(client) -> None:
    model_id = _gguf_model_id(client)
    response = client.post(
        "/v1/chat/completions",
        json={"model": model_id, "messages": [{"role": "user", "content": "hi"}]},
    )
    sampling = response.json()["metadata"]["sampling"]
    assert sampling is None or sampling["requested"] == {}


@pytest.mark.parametrize(
    "controls",
    [
        {"top_p": 0},
        {"top_p": 1.5},
        {"top_k": 0},
        {"repetition_penalty": -1},
        {"presence_penalty": 9},
    ],
)
def test_out_of_range_sampling_controls_are_rejected(client, controls: dict) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "sampling": controls},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_responses_route_also_accepts_sampling_controls(client) -> None:
    model_id = _gguf_model_id(client)
    response = client.post(
        "/v1/responses",
        json={"model": model_id, "input": "hi", "sampling": {"top_p": 0.8}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["sampling"]["requested"] == {"top_p": 0.8}


# --- inspecting the prompt must not cost the transport (G23) ------------------


def _stream_chunks(client: TestClient, path: str, payload: dict) -> list[dict]:
    with client.stream("POST", path, json=payload) as response:
        assert response.status_code == 200
        lines = [
            line.removeprefix("data: ")
            for line in response.iter_lines()
            if line and line.startswith("data: ")
        ]
    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


def test_streaming_chat_carries_the_prompt_trace_on_its_final_chunk(client) -> None:
    model_id = _gguf_model_id(client)
    chunks = _stream_chunks(
        client,
        "/v1/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "include_prompt_trace": True,
        },
    )

    trace = chunks[-1]["prompt_trace"]
    assert trace is not None
    assert trace["serialized_model_prompt"]
    assert trace["output_contract"]["format"] == "text"
    # Same rule as `usage`: terminal chunk only, never on a delta.
    assert all(chunk.get("prompt_trace") is None for chunk in chunks[:-1])


def test_streaming_responses_carries_the_prompt_trace_on_its_final_chunk(client) -> None:
    model_id = _gguf_model_id(client)
    chunks = _stream_chunks(
        client,
        "/v1/responses",
        {"model": model_id, "input": "hi", "stream": True, "include_prompt_trace": True},
    )

    assert chunks[-1]["done"] is True
    assert chunks[-1]["prompt_trace"]["serialized_model_prompt"]
    assert all(chunk.get("prompt_trace") is None for chunk in chunks[:-1])


def test_streaming_omits_the_prompt_trace_when_it_was_not_requested(client) -> None:
    model_id = _gguf_model_id(client)
    chunks = _stream_chunks(
        client,
        "/v1/chat/completions",
        {"model": model_id, "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert all(chunk.get("prompt_trace") is None for chunk in chunks)


# --- the WebSocket event stream is guarded too (G14) --------------------------


def test_websocket_events_refuse_an_unauthenticated_handshake(secured_settings) -> None:
    """The HTTP middleware never runs for a WebSocket, so the route enforces it."""

    with TestClient(create_app(secured_settings)) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/v1/events"):
                pass

    assert refused.value.code == 1008
    assert refused.value.reason == "authentication_error"


def test_websocket_events_accept_a_keyed_handshake(secured_settings) -> None:
    with TestClient(create_app(secured_settings)) as client:
        # Reaching the body at all means the handshake was accepted.
        with client.websocket_connect("/v1/events", headers={"x-api-key": "test-key"}) as websocket:
            assert websocket is not None


def test_websocket_events_accept_the_key_as_a_subprotocol(secured_settings) -> None:
    """A browser cannot set handshake headers; the subprotocol list is all it has."""

    with TestClient(create_app(secured_settings)) as client:
        with client.websocket_connect(
            "/v1/events",
            subprotocols=["lewlm.api-key.test-key"],
        ) as websocket:
            # The key is never echoed back: that would put it in the response
            # headers too, and a client that offers one need not receive one.
            assert websocket.accepted_subprotocol is None

    with TestClient(create_app(secured_settings)) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/events", subprotocols=["lewlm.api-key.wrong"]):
                pass


def test_websocket_events_still_stream_when_no_key_is_required(client) -> None:
    with client.websocket_connect("/v1/events") as websocket:
        model_id = _gguf_model_id(client)
        client.post(
            "/v1/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert websocket.receive_json()["type"]


# --- roles are a closed set the templates can render (G17) --------------------


@pytest.mark.parametrize("role", ["system", "developer", "user", "assistant", "tool"])
def test_every_declared_role_is_accepted(client, role: str) -> None:
    model_id = _gguf_model_id(client)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [
                {"role": role, "content": "context"},
                {"role": "user", "content": "hi"},
            ],
            "include_prompt_trace": True,
        },
    )
    assert response.status_code == 200, response.text


def test_an_unknown_role_is_rejected_rather_than_rendered(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "moderator", "content": "hi"}]},
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["details"]["fields"][0]["field"].startswith("messages.0.role")


def test_developer_and_tool_roles_fold_onto_a_role_the_template_can_render(client) -> None:
    """No template has a `tool` turn token, so the text carries the distinction."""

    model_id = _gguf_model_id(client)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [
                {"role": "developer", "content": "Keep it terse."},
                {"role": "tool", "content": "lookup returned 42"},
                {"role": "user", "content": "hi"},
            ],
            "include_prompt_trace": True,
        },
    )

    prompt = response.json()["prompt_trace"]["serialized_model_prompt"]
    assert "Developer instructions:" in prompt
    assert "Tool result:" in prompt
    # The role names themselves never reach the template as turn tokens.
    assert "<developer>" not in prompt
    assert "<tool>" not in prompt


# --- one model is addressable on its own (G20) -------------------------------


def test_a_single_model_is_fetchable_without_filtering_the_inventory(client) -> None:
    model_id = _gguf_model_id(client)
    response = client.get(f"/v1/models/{model_id}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"]["model_id"] == model_id
    # The readiness annotation from `/v1/models` comes along, so a caller does
    # not have to fetch the list purely to learn whether this model can serve.
    assert payload["capability_availability"]["model_id"] == model_id
    assert "chat_ready" in payload["capability_availability"]


def test_an_unknown_model_returns_the_error_envelope(client) -> None:
    response = client.get("/v1/models/not-a-real-model")

    assert response.status_code == 404
    assert response.json()["error"]["code"]


def test_the_single_model_route_does_not_shadow_the_scan_route(client) -> None:
    assert client.post("/v1/models/scan", json={}).status_code == 200


# --- structured-output enforcement is knowable up front (G18) ----------------


def test_capabilities_report_what_a_response_format_will_actually_get(client) -> None:
    model_id = _gguf_model_id(client)
    report = client.get(f"/v1/models/{model_id}/capabilities").json()

    support = report["structured_output"]
    assert support is not None
    assert support["runtime_name"]
    assert support["reason"]
    # The per-mode entries are the same shape the response reports afterwards,
    # so a caller can compare the prediction against the outcome directly.
    for mode in ("json_schema", "grammar"):
        assert set(support[mode]) >= {"enforcement", "decoder_enforced", "fallback_used"}
        assert (mode in support["decode_time_modes"]) == support[mode]["decoder_enforced"]
