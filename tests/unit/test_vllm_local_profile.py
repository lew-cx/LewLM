"""Portable checks for the `vllm_local` bridge profile (modernization step 07).

No engine is involved: a fake loopback server stands in for `vllm serve` so
profile identity, structured-output forwarding, error translation, and the
scheduling contract (no LewLM microbatch window, concurrent upstream
submission, per-request cancellation) can be proven on any OS. Real vLLM
behaviour is a separate Linux/NVIDIA lane recorded in the recipe.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lewlm.config.endpoints import ExternalEndpoint
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import CapabilityName, GenerateMessage, RuntimeProvider
from lewlm.core.errors import RuntimeUnavailableError
from lewlm.core.serving_core import ServingRuntimeAdapterKind, continuous_batching_ownership, describe_serving_runtime_adapter
from lewlm.runtime.adapters import LocalOpenAICompatibleAdapterRuntime
from lewlm.structured_output import JSONSchemaResponseFormat


class _FakeVLLM:
    """Enough of `vllm serve` to exercise the bridge.

    `/v1/models` is key-gated (vLLM guards `/v1`, not `/health`). Streams block
    on a barrier before their first content chunk, so a stream can only finish
    if a second request is *also* in flight upstream: that is the concurrency
    proof, observed at the fake engine rather than inferred at LewLM.
    """

    def __init__(self, *, api_key: str, model_id: str, barrier_parties: int = 2) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.barrier = threading.Barrier(barrier_parties, timeout=5.0)
        self.arrivals: list[float] = []
        self.disconnects: list[str] = []
        self.chat_payloads: list[dict] = []
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _authorized(self) -> bool:
                if self.headers.get("Authorization") == f"Bearer {fake.api_key}":
                    return True
                body = b'{"error": "Unauthorized"}'
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return False

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers(); return
                if not self._authorized():
                    return
                payload = {"object": "list", "data": [{"id": fake.model_id, "object": "model", "owned_by": "vllm", "max_model_len": 8192}]}
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                if not self._authorized():
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                fake.chat_payloads.append(payload)
                if payload.get("tool_choice") is not None:
                    # vLLM without --enable-auto-tool-choice rejects the request up front.
                    body = json.dumps({"error": {"message": '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set', "type": "BadRequestError", "code": 400}}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                fake.arrivals.append(time.monotonic())
                request_id = payload.get("metadata", {}).get("request_id") or f"req-{len(fake.arrivals)}"
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
                    fake.barrier.wait()
                except threading.BrokenBarrierError:
                    pass
                try:
                    for index in range(6):
                        emit(json.dumps({"id": request_id, "choices": [{"index": 0, "delta": {"content": f"tok{index} "}, "finish_reason": None}]}))
                        time.sleep(0.05)
                    emit(json.dumps({"id": request_id, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}}))
                    emit("[DONE]")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    fake.disconnects.append(request_id)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def stop(self) -> None:
        self.barrier.abort()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _endpoint(base_url: str, *, endpoint_id: str = "vllm", profile: str = "vllm_local") -> ExternalEndpoint:
    return ExternalEndpoint(endpoint_id=endpoint_id, profile=profile, base_url=base_url, api_key_env="VLLM_API_KEY", read_timeout_seconds=10)


def test_vllm_local_is_distinct_from_vllm_mlx_in_every_evidence_surface(tmp_path: Path) -> None:
    local = _endpoint("http://127.0.0.1:8000/v1")
    apple = _endpoint("http://127.0.0.1:8001/v1", endpoint_id="vllm-mlx", profile="vllm_mlx")
    settings = LewLMSettings(data_dir=tmp_path, external_endpoints=(local, apple))
    local_runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings, endpoint=local)
    apple_runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings, endpoint=apple)

    # Same provider family, different profile ids everywhere a UI can read them.
    assert local_runtime.bridge_profile().provider is RuntimeProvider.VLLM
    assert apple_runtime.bridge_profile().provider is RuntimeProvider.VLLM
    assert local_runtime.bridge_profile().profile_id == "vllm_local"
    assert apple_runtime.bridge_profile().profile_id == "vllm_mlx"
    assert local_runtime.endpoint_snapshot()["profile"] == "vllm_local"
    assert apple_runtime.endpoint_snapshot()["profile"] == "vllm_mlx"
    assert local_runtime.name != apple_runtime.name

    features = local_runtime.performance_feature_snapshot()
    assert features["continuous_batching"]["ownership"] == "backend_native"
    assert features["prefix_cache"]["ownership"] == "backend_native"
    assert all(feature["active"] is False for feature in features.values()), "labels declare ownership, never activity"
    assert features["prefix_cache"]["metrics"]["adapter_profile"] == "vllm_local"


def test_json_schema_is_forwarded_natively_and_reported_as_upstream_enforced(tmp_path: Path) -> None:
    endpoint = _endpoint("http://127.0.0.1:8000/v1")
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,)), endpoint=endpoint)
    contract = JSONSchemaResponseFormat(name="answer", schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})
    status = runtime.structured_output_runtime_status(contract)
    assert status is not None
    assert status.enforcement == "decode_time"
    assert status.enforcement_evidence == "upstream_native"
    assert status.decoder_enforced is False, "LewLM never observes vLLM's decoder; validation happens after generation"
    assert status.fallback_used is False


def test_bridge_never_opens_a_lewlm_microbatch_window_but_reports_backend_batching(tmp_path: Path) -> None:
    endpoint = _endpoint("http://127.0.0.1:8000/v1")
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,)), endpoint=endpoint)

    for capability in (CapabilityName.CHAT, CapabilityName.STREAMING):
        # No LewLM-side batch API: the frontier scheduler's window never applies.
        assert runtime.supports_continuous_batching(capability) is False
        # ...but the engine batches, and the serving snapshot says so honestly.
        assert continuous_batching_ownership(runtime=runtime, capability=capability) == "backend_native"
        adapter = describe_serving_runtime_adapter(runtime=runtime, capability=capability)
        assert adapter.kind is ServingRuntimeAdapterKind.BACKEND_NATIVE_BATCH
        assert adapter.backend_batching is True
    assert continuous_batching_ownership(runtime=runtime, capability=CapabilityName.EMBEDDINGS) == "unsupported"

    generic = _endpoint("http://127.0.0.1:8002/v1", endpoint_id="generic", profile="openai_compatible")
    generic_runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(data_dir=tmp_path, external_endpoints=(generic,)), endpoint=generic)
    assert continuous_batching_ownership(runtime=generic_runtime, capability=CapabilityName.CHAT) == "unsupported"


def test_two_streams_reach_the_fake_engine_concurrently_and_cancelling_one_leaves_the_other(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeVLLM(api_key="inference-key", model_id="qwen2.5-0.5b-instruct")
    monkeypatch.setenv("VLLM_API_KEY", "inference-key")
    settings = LewLMSettings(
        data_dir=tmp_path / "state", models_dir=(tmp_path / "models",), runtime_packs=("external_accelerator",),
        external_endpoints=(_endpoint(fake.base_url),),
        max_concurrent_runtime_requests=4,
    )
    (tmp_path / "state").mkdir(); (tmp_path / "models").mkdir()
    services = bootstrap_services(settings)
    try:
        services.model_registry.scan()
        manifest = next(m for m in services.model_registry.list_manifests() if m.source_path == "external://vllm/qwen2.5-0.5b-instruct")
        assert manifest.metadata["external_profile"] == "vllm_local"
        assert manifest.context_length == 8192, "context comes from vLLM's max_model_len"

        async def run() -> tuple[str, list[str], str]:
            messages = [GenerateMessage(role="user", content="count")]

            async def open_stream():
                return await services.chat_orchestrator.stream(
                    model_id=manifest.model_id, messages=messages, max_tokens=8, temperature=0.0, apply_serving_profile=False,
                )

            first, second = await asyncio.gather(open_stream(), open_stream())
            # Both streams only produce output once both requests are inside
            # the fake engine (the barrier), so the first delta of each is the
            # concurrency evidence. Neither waited in a LewLM batch window.
            first_delta = await anext(first.stream)
            second_delta = await anext(second.stream)
            assert first_delta.startswith("tok0") and second_delta.startswith("tok0")
            for session in (first, second):
                scheduling = session.request.metadata.get("scheduling", {})
                assert scheduling.get("queue_type") != "continuous_batching"

            # Consumer-side cancellation of the first stream: transport closes.
            await first.stream.aclose()
            rest = [delta async for delta in second.stream]
            serving = (second.request_metadata or {}).get("serving", {})
            return first.request_id, rest, second.finish_reason, serving

        first_id, rest, finish_reason, serving = asyncio.run(run())
        assert serving.get("runtime_adapter", {}).get("kind") == "backend_native_batch"
        assert "".join(rest).strip().endswith("tok5"), "the surviving stream completed"
        assert finish_reason == "stop"
        assert len(fake.arrivals) == 2 and fake.arrivals[1] - fake.arrivals[0] < 2.0
        deadline = time.monotonic() + 5.0
        while len(fake.disconnects) < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(fake.disconnects) == 1, "exactly the cancelled stream's upstream connection was closed"
        stats = services.runtime_request_scheduler.snapshot()
        assert stats["peak_active_requests"] == 2, "both requests were admitted at once, not serialized"
        assert stats["active_requests"] == 0, "admission released for both requests"
    finally:
        fake.stop()


def test_tool_choice_rejected_upstream_is_a_structured_invalid_request_without_replay(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeVLLM(api_key="inference-key", model_id="qwen2.5-0.5b-instruct", barrier_parties=1)
    monkeypatch.setenv("VLLM_API_KEY", "inference-key")
    try:
        endpoint = _endpoint(fake.base_url)
        runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,)), endpoint=endpoint)
        runtime.advertised_model_records(refresh=True)

        async def run() -> None:
            await runtime._transport.request_json(
                "POST", "/v1/chat/completions",
                payload={"model": "qwen2.5-0.5b-instruct", "messages": [{"role": "user", "content": "hi"}],
                         "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}], "tool_choice": "auto"},
            )

        with pytest.raises(RuntimeUnavailableError) as excinfo:
            asyncio.run(run())
        assert excinfo.value.details["error_kind"] == "invalid_request"
        assert excinfo.value.details["status_code"] == 400
        assert len(fake.chat_payloads) == 1, "no automatic retry after an upstream rejection"
        assert "inference-key" not in str(excinfo.value)
    finally:
        fake.stop()
