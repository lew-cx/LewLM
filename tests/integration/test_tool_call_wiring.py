"""Model-emitted tool calls surface on the chat and responses APIs."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Look up the weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _tool_call_text(payload: dict) -> str:
    return f"```json\n{json.dumps(payload)}\n```"


def _chat(client: TestClient, model_id: str, content: str, **extra) -> dict:
    response = client.post(
        "/v1/chat/completions",
        json={"model": model_id, "messages": [{"role": "user", "content": content}], **extra},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _gguf_model_id(client: TestClient) -> str:
    manifests = client.post("/v1/models/scan", json={}).json()["manifests"]
    return next(item["model_id"] for item in manifests if item["format_type"] == "gguf")


def test_chat_completion_reports_parsed_tool_call(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        model_id = _gguf_model_id(client)
        body = _chat(
            client,
            model_id,
            _tool_call_text({"name": "get_weather", "arguments": {"city": "Halifax"}}),
            tools=[_WEATHER_TOOL],
        )

    tool_calls = body["tool_calls"]
    assert tool_calls["status"] == "parsed"
    assert tool_calls["parser"] == "lewlm_strict_tool_parser"
    assert [call["name"] for call in tool_calls["tool_calls"]] == ["get_weather"]
    assert tool_calls["tool_calls"][0]["arguments"] == {"city": "Halifax"}
    assert tool_calls["issues"] == []
    assert tool_calls["parallel"] is False


def test_chat_completion_omits_tool_calls_when_no_tools_declared(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        model_id = _gguf_model_id(client)
        body = _chat(
            client,
            model_id,
            _tool_call_text({"name": "get_weather", "arguments": {"city": "Halifax"}}),
        )

    # Without declared tools the request never asked for tool calling, so the
    # response must not report tool-call issues for ordinary JSON output.
    assert body["tool_calls"] is None


def test_chat_completion_reports_schema_violation_without_inventing_a_call(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        model_id = _gguf_model_id(client)
        body = _chat(
            client,
            model_id,
            _tool_call_text({"name": "get_weather", "arguments": {"city": 7}}),
            tools=[_WEATHER_TOOL],
        )

    tool_calls = body["tool_calls"]
    assert tool_calls["status"] == "failed"
    assert tool_calls["tool_calls"] == []
    assert [issue["code"] for issue in tool_calls["issues"]] == ["schema_violation"]


def test_chat_completion_reports_unknown_tool(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        model_id = _gguf_model_id(client)
        body = _chat(
            client,
            model_id,
            _tool_call_text({"name": "launch_rocket", "arguments": {}}),
            tools=[_WEATHER_TOOL],
        )

    tool_calls = body["tool_calls"]
    assert tool_calls["status"] == "failed"
    assert [issue["code"] for issue in tool_calls["issues"]] == ["unknown_tool"]


def test_chat_completion_reports_parallel_tool_calls(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        model_id = _gguf_model_id(client)
        body = _chat(
            client,
            model_id,
            _tool_call_text(
                {
                    "tool_calls": [
                        {"name": "get_weather", "arguments": {"city": "Halifax"}},
                        {"name": "get_weather", "arguments": {"city": "Toronto"}},
                    ],
                },
            ),
            tools=[_WEATHER_TOOL],
        )

    tool_calls = body["tool_calls"]
    assert tool_calls["status"] == "parsed"
    assert tool_calls["parallel"] is True
    assert [call["arguments"]["city"] for call in tool_calls["tool_calls"]] == ["Halifax", "Toronto"]


def test_responses_route_reports_parsed_tool_call(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        model_id = _gguf_model_id(client)
        response = client.post(
            "/v1/responses",
            json={
                "model": model_id,
                "input": _tool_call_text({"name": "get_weather", "arguments": {"city": "Halifax"}}),
                "tools": [_WEATHER_TOOL],
            },
        )
        assert response.status_code == 200, response.text

    tool_calls = response.json()["tool_calls"]
    assert tool_calls["status"] == "parsed"
    assert [call["name"] for call in tool_calls["tool_calls"]] == ["get_weather"]


def test_streaming_chat_reports_tool_calls_on_final_chunk(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        model_id = _gguf_model_id(client)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": _tool_call_text({"name": "get_weather", "arguments": {"city": "Halifax"}}),
                    },
                ],
                "tools": [_WEATHER_TOOL],
                "stream": True,
            },
        ) as stream_response:
            assert stream_response.status_code == 200
            payloads = [
                json.loads(line[len("data: ") :])
                for line in stream_response.iter_lines()
                if line.startswith("data: ") and not line.endswith("[DONE]")
            ]

    final = next(item for item in reversed(payloads) if item.get("tool_calls") is not None)
    assert final["tool_calls"]["status"] == "parsed"
    assert [call["name"] for call in final["tool_calls"]["tool_calls"]] == ["get_weather"]


def _compiled_system_text(app, **request_kwargs) -> str:
    """Compile a prompt through the app's real prompt compiler."""

    from lewlm.core.contracts import GenerateMessage
    from lewlm.prompting import PromptCompilationRequest

    compiler = app.state.services.prompt_compiler
    result = compiler.compile(
        messages=[GenerateMessage(role="user", content="What is the weather in Halifax?")],
        request=PromptCompilationRequest(**request_kwargs),
    )
    return "\n\n".join(message.content for message in result.messages if message.role == "system")


def test_declaring_tools_puts_the_invocation_contract_in_the_compiled_prompt(app_with_fake_runtime) -> None:
    """The model must be told the shape the strict parser accepts.

    Without this the loop cannot close: the model emits bare arguments, the
    parser finds no candidate, and `no_tool_calls` reads as a refusal rather
    than a format mismatch.
    """

    with TestClient(app_with_fake_runtime):
        prompt = _compiled_system_text(app_with_fake_runtime, tools=[_WEATHER_TOOL])

    assert "get_weather" in prompt
    for marker in ("tool_call", "tool_calls", "name"):
        assert f'"{marker}"' in prompt, f"compiled prompt never names the `{marker}` shape"

    # The shape the prompt advertises must be the shape the parser accepts.
    advertised = '{"name": "get_weather", "arguments": {"city": "Halifax"}}'
    assert advertised.split(":")[0] in prompt
    with TestClient(app_with_fake_runtime) as client:
        model_id = _gguf_model_id(client)
        body = _chat(client, model_id, _tool_call_text(json.loads(advertised)), tools=[_WEATHER_TOOL])
    assert body["tool_calls"]["status"] == "parsed"


def test_tool_contract_is_absent_when_no_tools_are_declared(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime):
        prompt = _compiled_system_text(app_with_fake_runtime, system_prompt="Be brief.")

    assert "To call a tool" not in prompt
