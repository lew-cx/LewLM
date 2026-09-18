"""Portable checks for the `sglang_local` bridge profile (modernization step 08).

No engine is involved: a fake loopback server stands in for
`sglang.launch_server` with SGLang's actual auth shape (`/health` always open,
`Authorization: Bearer` required elsewhere), its `/v1/models` record
(`max_model_len`), and the `--enable-cache-report` usage detail. Real SGLang
behaviour is a separate Linux/NVIDIA lane recorded in the recipe.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lewlm.api.routes.chat import _completion_usage
from lewlm.config.endpoints import ExternalEndpoint
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import CapabilityName, GenerateMessage, RuntimeProvider
from lewlm.core.errors import RoutingError, RuntimeUnavailableError
from lewlm.core.serving_core import continuous_batching_ownership
from lewlm.runtime.adapters import LocalOpenAICompatibleAdapterRuntime
from lewlm.runtime.adapters.openai_compatible import _normalize_usage


class _FakeSGLang:
    def __init__(self, *, api_key: str, model_id: str, cache_report: bool) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.cache_report = cache_report
        self.requests: list[dict] = []
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, status: int, payload: dict | bytes, content_type: str = "application/json") -> None:
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self) -> bool:
                # SGLang's middleware at the pin: /health* and /metrics* always
                # pass; everything else needs "Bearer <api_key>".
                if self.path.startswith("/health") or self.path.startswith("/metrics"):
                    return True
                header = self.headers.get("Authorization", "")
                parts = header.split(" ", 1)
                if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1] == fake.api_key:
                    return True
                self._send(401, {"error": "Unauthorized"})
                return False

            def do_GET(self) -> None:  # noqa: N802
                if not self._authorized():
                    return
                if self.path.startswith("/health"):
                    self._send(200, b"", "text/plain")
                    return
                self._send(200, {"object": "list", "data": [{"id": fake.model_id, "object": "model", "created": 0, "owned_by": "sglang", "root": fake.model_id, "max_model_len": 8192}]})

            def do_POST(self) -> None:  # noqa: N802
                if not self._authorized():
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append(payload)
                usage = {"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43}
                if fake.cache_report:
                    usage["prompt_tokens_details"] = {"cached_tokens": 32}
                if payload.get("stream"):
                    frames = [
                        {"id": "r", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "ok "}, "finish_reason": None}]},
                        {"id": "r", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                        {"id": "r", "object": "chat.completion.chunk", "choices": [], "usage": usage},
                    ]
                    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
                    self._send(200, body.encode(), "text/event-stream")
                    return
                self._send(200, {"id": "r", "object": "chat.completion", "model": fake.model_id,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                                 "usage": usage})

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _endpoint(base_url: str) -> ExternalEndpoint:
    return ExternalEndpoint(endpoint_id="sglang", profile="sglang_local", base_url=base_url, api_key_env="SGLANG_API_KEY", read_timeout_seconds=10)


def _services(tmp_path: Path, base_url: str):
    settings = LewLMSettings(
        data_dir=tmp_path / "state", models_dir=(tmp_path / "models",), runtime_packs=("external_accelerator",),
        external_endpoints=(_endpoint(base_url),),
    )
    (tmp_path / "state").mkdir(parents=True, exist_ok=True); (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    return bootstrap_services(settings)


def test_profile_maps_to_the_sglang_provider_with_backend_owned_batching_and_partial_prefix_cache(tmp_path: Path) -> None:
    endpoint = _endpoint("http://127.0.0.1:30000/v1")
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,)), endpoint=endpoint)

    assert runtime.name == "local_external_adapter:sglang"
    assert runtime.bridge_profile().provider is RuntimeProvider.SGLANG
    assert runtime.bridge_profile().profile_id == "sglang_local"
    features = runtime.performance_feature_snapshot()
    assert features["continuous_batching"]["ownership"] == "backend_native"
    assert features["prefix_cache"]["ownership"] == "partial", "radix-cache reuse is upstream; hit counts only when the server reports them"
    assert all(feature["active"] is False for feature in features.values())
    assert continuous_batching_ownership(runtime=runtime, capability=CapabilityName.STREAMING) == "backend_native"
    assert runtime.supports_continuous_batching(CapabilityName.STREAMING) is False


def test_health_is_open_but_inventory_needs_the_bearer_key(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeSGLang(api_key="inference-key", model_id="qwen2.5-0.5b-instruct", cache_report=False)
    try:
        monkeypatch.setenv("SGLANG_API_KEY", "wrong-key")
        endpoint = _endpoint(fake.base_url)
        runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,)), endpoint=endpoint)
        with pytest.raises(RuntimeUnavailableError) as excinfo:
            runtime.advertised_model_records(refresh=True)
        assert excinfo.value.details["status_code"] == 401
        assert runtime.inventory_state == "failed"

        monkeypatch.setenv("SGLANG_API_KEY", "inference-key")
        records = runtime.advertised_model_records(refresh=True)
        assert records[0]["id"] == "qwen2.5-0.5b-instruct" and records[0]["max_model_len"] == 8192
        assert runtime.inventory_state == "advertised"
    finally:
        fake.stop()


def test_cache_report_counter_survives_the_bridge_and_absence_means_unknown(tmp_path: Path, monkeypatch) -> None:
    # Normalization keeps the one nested counter and nothing else nested.
    assert _normalize_usage({"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43, "prompt_tokens_details": {"cached_tokens": 32}}) == {
        "prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43, "cached_tokens": 32,
    }
    assert "cached_tokens" not in _normalize_usage({"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43})
    assert _normalize_usage({"prompt_tokens": 1, "prompt_tokens_details": {"cached_tokens": True}}) == {"prompt_tokens": 1}
    assert _completion_usage({"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43}).cached_tokens is None
    assert _completion_usage({"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43, "cached_tokens": 32}).cached_tokens == 32

    monkeypatch.setenv("SGLANG_API_KEY", "inference-key")
    for cache_report in (True, False):
        fake = _FakeSGLang(api_key="inference-key", model_id="qwen2.5-0.5b-instruct", cache_report=cache_report)
        services = _services(tmp_path / ("report" if cache_report else "silent"), fake.base_url)
        try:
            services.model_registry.scan()
            manifest = next(m for m in services.model_registry.list_manifests() if m.source_path == "external://sglang/qwen2.5-0.5b-instruct")
            assert manifest.context_length == 8192

            async def run() -> tuple[dict[str, int], dict[str, int]]:
                messages = [GenerateMessage(role="user", content="hi")]
                execution = await services.chat_orchestrator.complete(
                    model_id=manifest.model_id, messages=messages, max_tokens=8, temperature=0.0, apply_serving_profile=False,
                )
                session = await services.chat_orchestrator.stream(
                    model_id=manifest.model_id, messages=messages, max_tokens=8, temperature=0.0, apply_serving_profile=False,
                )
                async for _ in session.stream:
                    pass
                return execution.response.usage, session.usage

            complete_usage, stream_usage = asyncio.run(run())
            if cache_report:
                assert complete_usage["cached_tokens"] == 32 and stream_usage["cached_tokens"] == 32
            else:
                assert "cached_tokens" not in complete_usage and "cached_tokens" not in stream_usage
            assert stream_usage["prompt_tokens"] == 40 and complete_usage["total_tokens"] == 43
        finally:
            fake.stop()


def test_stopped_sglang_is_a_structured_endpoint_failure_that_keeps_native_candidates(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeSGLang(api_key="inference-key", model_id="qwen2.5-0.5b-instruct", cache_report=False)
    monkeypatch.setenv("SGLANG_API_KEY", "inference-key")
    services = _services(tmp_path, fake.base_url)
    try:
        services.model_registry.scan()
        manifest = next(m for m in services.model_registry.list_manifests() if m.source_path == "external://sglang/qwen2.5-0.5b-instruct")
        assert manifest.metadata["external_profile"] == "sglang_local"

        fake.stop()
        runtime = services.runtime_catalog.get_endpoint_runtime("sglang")
        runtime.invalidate_discovery_cache()
        with pytest.raises(RoutingError) as excinfo:
            services.model_router.route_chat(manifest.model_id)
        assert excinfo.value.details["endpoint_id"] == "sglang"
        assert excinfo.value.details["engine_profile"] == "sglang_local"
        assert excinfo.value.details["fallback_policy"] == "none"

        services.model_registry.scan()
        assert any(m.model_id == manifest.model_id for m in services.model_registry.list_manifests()), "an unreachable endpoint is not evidence its model was deleted"
        assert runtime.inventory_state in {"stale", "failed"}
        # Only this endpoint is unavailable; coexistence with Ollama/llama.cpp
        # candidates is proven by the same mechanism in the TabbyAPI suite.
        assert runtime.candidate_report(manifest).available is False
    finally:
        try:
            fake.stop()
        except Exception:
            pass
