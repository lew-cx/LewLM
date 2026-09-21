"""A fake OpenAI-compatible engine plus a LewLM server in front of it.

    python -m lewlm.testing.fake_backend --port 8080

serves LewLM at ``http://127.0.0.1:8080`` fronting a fake engine, with one
model ``fixture-chat`` bound to endpoint ``fixture``. Point a host app at it
and every public route behaves as it would with a real bridge-backed model:
discovery, capabilities, streaming with a terminal usage chunk, native tool
calls, ``json_schema`` output validated after generation, cancellation with
transport close, a 503 naming the endpoint when the engine is stopped, and
the stream-error envelope when it dies mid-stream.

The engine is deliberately small and deterministic. It is not a model: it
answers by rule (echo, count, tool call when a tool is offered, a schema-
shaped object when JSON is requested) so a UI can be checked against known
output.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

DEFAULT_MODEL_ID = "fixture-chat"
DEFAULT_ENDPOINT_ID = "fixture"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _message_text(content: Any) -> str:
    """Text of an OpenAI-style message whether it is a string or content parts."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text")
    return ""


def _schema_shaped_value(schema: dict[str, Any]) -> Any:
    """A value satisfying a simple JSON schema: enough for a UI to render."""

    kind = schema.get("type")
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if kind == "object" or "properties" in schema:
        return {name: _schema_shaped_value(sub) for name, sub in (schema.get("properties") or {}).items()}
    if kind == "array":
        return [_schema_shaped_value(schema.get("items") or {"type": "string"})]
    if kind == "integer":
        return 2_000_000
    if kind == "number":
        return 1.5
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    return "Paris"


class FakeOpenAIEngine:
    """Loopback ``/v1/models`` + ``/v1/chat/completions`` with rule-based replies.

    ``api_key`` gates ``/v1`` (``/health`` stays open, as vLLM and SGLang do).
    ``stream_delay_seconds`` spaces streamed chunks so a UI can render tokens
    and cancel mid-stream. ``long_reply_words`` is the length of a reply to a
    prompt asking for a *long* answer, or to any request with a large
    ``max_tokens``, so cancellation has something to interrupt.
    """

    def __init__(
        self,
        *,
        model_ids: tuple[str, ...] = (DEFAULT_MODEL_ID,),
        api_key: str | None = None,
        stream_delay_seconds: float = 0.02,
        long_reply_words: int = 200,
        max_model_len: int = 8192,
    ) -> None:
        self.model_ids = model_ids
        self.api_key = api_key
        self.stream_delay_seconds = stream_delay_seconds
        self.long_reply_words = long_reply_words
        self.max_model_len = max_model_len
        self.requests: list[dict[str, Any]] = []
        self.disconnects = 0
        #: Set to make every in-flight stream close without a terminal chunk on
        #: its next frame (a timing-based kill, for a live client).
        self.die_mid_stream = threading.Event()
        #: When not None, a stream closes after emitting this many frames (a
        #: deterministic kill, for in-process test clients that batch reads).
        self.die_after_frames: int | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    # ---- lifecycle ------------------------------------------------------------

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise RuntimeError("engine is not started")
        return f"http://127.0.0.1:{self._port}/v1"

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self, port: int | None = None) -> "FakeOpenAIEngine":
        if self._server is not None:
            return self
        engine = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, status: int, payload: dict[str, Any] | bytes, content_type: str = "application/json") -> None:
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self) -> bool:
                if engine.api_key is None or self.path.startswith("/health"):
                    return True
                if self.headers.get("Authorization") == f"Bearer {engine.api_key}":
                    return True
                self._send(401, {"error": {"message": "Unauthorized", "type": "authentication_error", "code": 401}})
                return False

            def _alive(self) -> bool:
                # A stopped engine must not keep answering on pooled keep-alive
                # sockets: drop the connection so the client sees a transport error.
                if engine._server is None:
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    self.connection.close()
                    return False
                return True

            def do_GET(self) -> None:  # noqa: N802
                if not self._alive() or not self._authorized():
                    return
                if self.path.startswith("/health"):
                    self._send(200, b"", "text/plain")
                    return
                self._send(200, {"object": "list", "data": [
                    {"id": model_id, "object": "model", "owned_by": "lewlm-fixture", "max_model_len": engine.max_model_len}
                    for model_id in engine.model_ids
                ]})

            def do_POST(self) -> None:  # noqa: N802
                if not self._alive() or not self._authorized():
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                engine.requests.append(payload)
                if payload.get("model") not in engine.model_ids:
                    self._send(404, {"error": {"message": f"The model `{payload.get('model')}` does not exist.", "type": "NotFoundError", "code": 404}})
                    return
                reply = engine._reply(payload)
                if payload.get("stream"):
                    self._stream(reply)
                else:
                    self._send(200, engine._completion(reply))

            def _stream(self, reply: dict[str, Any]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def emit(data: str) -> None:
                    frame = f"data: {data}\n\n".encode()
                    self.wfile.write(f"{len(frame):x}\r\n".encode() + frame + b"\r\n")
                    self.wfile.flush()

                try:
                    emitted = 0
                    for frame in engine._stream_frames(reply):
                        if engine.die_mid_stream.is_set() or (engine.die_after_frames is not None and emitted >= engine.die_after_frames):
                            # Simulate the engine process dying: shut the socket down so the
                            # client sees EOF/RST now, not at its read timeout, and do not
                            # wait for another keep-alive request on it.
                            self.close_connection = True
                            try:
                                self.connection.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass
                            self.connection.close()
                            return
                        emit(json.dumps(frame))
                        emitted += 1
                        time.sleep(engine.stream_delay_seconds)
                    emit("[DONE]")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    engine.disconnects += 1

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", port or 0), _Handler)
        self._port = self._server.server_port
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="lewlm-fake-engine")
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving (simulates the engine going away); ``start()`` brings it back on the same port."""

        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    # ---- rules ----------------------------------------------------------------

    def _reply(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        last = messages[-1] if messages else {}
        last_text = _message_text(last.get("content"))
        response_format = payload.get("response_format") or {}
        tools = payload.get("tools") or []

        if last.get("role") == "tool":
            return {"content": "The tool reported clear skies at 21 C in Lisbon.", "finish_reason": "stop"}
        if tools and payload.get("tool_choice") != "none":
            tool = tools[0].get("function", tools[0])
            name = tool.get("name", "tool")
            arguments = {"city": "Lisbon"} if "city" in json.dumps(tool.get("parameters") or tool.get("input_schema") or {}) else {}
            return {"content": "", "tool_call": {"id": "call_fixture_1", "name": name, "arguments": json.dumps(arguments)}, "finish_reason": "tool_calls"}
        if response_format.get("type") == "json_schema":
            schema = (response_format.get("json_schema") or {}).get("schema") or response_format.get("schema") or {}
            return {"content": json.dumps(_schema_shaped_value(schema)), "finish_reason": "stop"}
        lowered = last_text.lower()
        if "count" in lowered:
            return {"content": "one two three four five", "finish_reason": "stop"}
        if "long" in lowered or int(payload.get("max_tokens") or 0) >= 128:
            words = " ".join(f"word{i}" for i in range(1, self.long_reply_words + 1))
            return {"content": words, "finish_reason": "length"}
        if "step by step" in lowered:
            return {"content": "<think>17 plus 25 is 42.</think>The answer is 42.", "finish_reason": "stop"}
        return {"content": f"fixture reply: {last_text.strip()[:60] or 'ready'}", "finish_reason": "stop"}

    @staticmethod
    def _usage(reply: dict[str, Any]) -> dict[str, Any]:
        completion = max(1, len(reply.get("content", "").split()))
        return {"prompt_tokens": 12, "completion_tokens": completion, "total_tokens": 12 + completion,
                "prompt_tokens_details": {"cached_tokens": 8}}

    def _completion(self, reply: dict[str, Any]) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": reply.get("content", "")}
        if reply.get("tool_call"):
            call = reply["tool_call"]
            message["content"] = None
            message["tool_calls"] = [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]
        return {"id": "chatcmpl-fixture", "object": "chat.completion", "created": int(time.time()), "model": self.model_ids[0],
                "choices": [{"index": 0, "message": message, "finish_reason": reply["finish_reason"]}], "usage": self._usage(reply)}

    def _stream_frames(self, reply: dict[str, Any]) -> Iterator[dict[str, Any]]:
        base = {"id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": int(time.time()), "model": self.model_ids[0]}
        if reply.get("tool_call"):
            call = reply["tool_call"]
            arguments = call["arguments"]
            half = max(1, len(arguments) // 2)
            yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": arguments[:half]}}]}, "finish_reason": None}]}
            yield {**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": arguments[half:]}}]}, "finish_reason": None}]}
            yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        else:
            words = reply.get("content", "").split(" ")
            yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
            for index, word in enumerate(words):
                yield {**base, "choices": [{"index": 0, "delta": {"content": word + (" " if index < len(words) - 1 else "")}, "finish_reason": None}]}
            yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": reply["finish_reason"]}]}
        yield {**base, "choices": [], "usage": self._usage(reply)}


class FakeBackendFixture:
    """LewLM on a loopback port fronting a :class:`FakeOpenAIEngine`.

    Use as a context manager (``with FakeBackendFixture() as fixture:``) or
    call :meth:`start` / :meth:`stop`. ``fixture.base_url`` is what a host app
    points at; ``fixture.engine`` lets a test stop and restart the engine or
    make it die mid-stream.
    """

    def __init__(
        self,
        *,
        port: int | None = None,
        engine: FakeOpenAIEngine | None = None,
        endpoint_id: str = DEFAULT_ENDPOINT_ID,
        data_dir: Path | None = None,
        api_key: str | None = None,
        cors_allow_origins: tuple[str, ...] = (),
        settings_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine or FakeOpenAIEngine()
        self.endpoint_id = endpoint_id
        self._port = port
        self._data_dir = data_dir
        self._api_key = api_key
        self._cors_allow_origins = cors_allow_origins
        self._settings_overrides = dict(settings_overrides or {})
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self.services: Any = None
        self.base_url = ""
        self.model_id = ""

    def __enter__(self) -> "FakeBackendFixture":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def start(self) -> "FakeBackendFixture":
        import uvicorn
        from pydantic import SecretStr

        from lewlm.api.app import create_app
        from lewlm.config.endpoints import ExternalEndpoint
        from lewlm.config.settings import LewLMSettings
        from lewlm.core.bootstrap import bootstrap_services

        self.engine.start()
        if self._data_dir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="lewlm-fixture-")
            data_dir = Path(self._tempdir.name)
        else:
            data_dir = self._data_dir
        (data_dir / "state").mkdir(parents=True, exist_ok=True)
        (data_dir / "models").mkdir(parents=True, exist_ok=True)
        port = self._port or _free_port()
        settings = LewLMSettings(
            environment="test",
            host="127.0.0.1",
            port=port,
            data_dir=data_dir / "state",
            models_dir=(data_dir / "models",),
            runtime_packs=("external_accelerator",),
            external_endpoints=(ExternalEndpoint(endpoint_id=self.endpoint_id, profile="openai_compatible", base_url=self.engine.base_url, read_timeout_seconds=30),),
            backend_feature_probes_enabled=False,
            api_keys=(SecretStr(self._api_key),) if self._api_key else (),
            cors_enabled=bool(self._cors_allow_origins),
            cors_allow_origins=self._cors_allow_origins,
            **self._settings_overrides,
        )
        self.services = bootstrap_services(settings)
        self.services.model_registry.scan()
        manifests = self.services.model_registry.list_manifests()
        if not manifests:
            raise RuntimeError("the fake engine advertised no models")
        self.model_id = manifests[0].model_id
        app = create_app(settings, services=self.services)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="lewlm-fixture-server")
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 30
        import urllib.request
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{self.base_url}/v1/health", timeout=1) as response:
                    if response.status == 200:
                        return self
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(0.05)
        raise RuntimeError("LewLM fixture server did not become healthy")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=15)
        self._server, self._thread = None, None
        self.engine.stop()
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    @contextmanager
    def engine_stopped(self) -> Iterator[None]:
        """Stop the engine for the block, then bring it back on the same port."""

        port = self.engine._port
        self.engine.stop()
        try:
            yield
        finally:
            self.engine.start(port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8080, help="LewLM port on 127.0.0.1")
    parser.add_argument("--api-key", default=None, help="Require this LewLM API key (Authorization: Bearer)")
    parser.add_argument("--cors-origin", action="append", default=[], help="Allow this browser origin (repeatable); no wildcard")
    parser.add_argument("--stream-delay-ms", type=int, default=20, help="Delay between streamed chunks")
    args = parser.parse_args(argv)
    engine = FakeOpenAIEngine(stream_delay_seconds=args.stream_delay_ms / 1000.0)
    fixture = FakeBackendFixture(port=args.port, engine=engine, api_key=args.api_key, cors_allow_origins=tuple(args.cors_origin))
    with fixture:
        print(f"LewLM fixture: {fixture.base_url}  model: {fixture.model_id}  endpoint: {fixture.endpoint_id}  engine: {engine.base_url}")
        print("Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
