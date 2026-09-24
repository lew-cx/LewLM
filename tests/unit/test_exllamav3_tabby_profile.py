"""Portable checks for the `exllamav3_tabby` bridge profile (modernization step 06).

No engine is involved: a fake loopback server stands in for TabbyAPI so profile
selection, format gating, authentication, error translation, and coexistence
with native/Ollama paths can be proven on any OS. Real ExLlamaV3 behavior is a
separate Linux/NVIDIA lane recorded in the recipe.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lewlm.config.endpoints import ExternalEndpoint
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import ModelFormat, RuntimeProvider, RuntimeSupportPath
from lewlm.core.errors import ConversionError, RoutingError, RuntimeUnavailableError
from lewlm.core.middleware import _provider_from_runtime_name
from lewlm.registry import ollama_inventory
from lewlm.registry.external_inventory import build_external_manifest
from lewlm.runtime.adapters import LocalOpenAICompatibleAdapterRuntime
from lewlm.runtime.llamacpp.runtime import LlamaCppRuntime


class _FakeTabby:
    """Enough of TabbyAPI's OpenAI surface to exercise the bridge: key-gated `/v1/models`."""

    def __init__(self, *, api_key: str, model_ids: list[str]) -> None:
        self.api_key = api_key
        self.model_ids = model_ids
        self.auth_headers: list[str | None] = []
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                fake.auth_headers.append(self.headers.get("Authorization"))
                if self.headers.get("Authorization") != f"Bearer {fake.api_key}":
                    body = json.dumps({"detail": "Invalid API key. SECRET-ADMIN-KEY must not leak"}).encode()
                    self.send_response(401)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                payload = {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "tabbyAPI"} for m in fake.model_ids]}
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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


def _endpoint(base_url: str, *, api_key_env: str | None = "TABBY_API_KEY") -> ExternalEndpoint:
    return ExternalEndpoint(endpoint_id="tabby", profile="exllamav3_tabby", base_url=base_url, api_key_env=api_key_env)


def test_profile_is_selectable_and_maps_to_the_exllamav3_provider(tmp_path: Path) -> None:
    endpoint = _endpoint("http://127.0.0.1:5000/v1")
    settings = LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,))
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings, endpoint=endpoint)

    assert runtime.name == "local_external_adapter:tabby"
    assert runtime.bridge_profile().provider is RuntimeProvider.EXLLAMAV3
    assert _provider_from_runtime_name(runtime.name, RuntimeSupportPath.BRIDGE, external_profile="exllamav3_tabby") is RuntimeProvider.EXLLAMAV3
    features = runtime.performance_feature_snapshot()
    # Profile labels declare ownership; nothing is active until observed.
    assert features["paged_kv_cache"]["ownership"] == "backend_native"
    assert all(feature["active"] is False for feature in features.values())
    assert features["constrained_decoding"]["metrics"]["evidence_state"] == "unverified"


def test_format_evidence_stays_unknown_and_exl3_never_reaches_llamacpp() -> None:
    endpoint = _endpoint("http://127.0.0.1:5000/v1")
    advertised = build_external_manifest({"id": "Qwen2.5-0.5B-Instruct-exl3-4.0bpw", "owned_by": "tabbyAPI"}, endpoint=endpoint, upstream_id="Qwen2.5-0.5B-Instruct-exl3-4.0bpw")
    assert advertised.format_type is ModelFormat.UNKNOWN, "a TabbyAPI listing does not reveal the artifact format"
    with_evidence = build_external_manifest({"id": "m", "metadata": {"format": "exl3"}}, endpoint=endpoint, upstream_id="m")
    assert with_evidence.format_type is ModelFormat.EXL3

    llamacpp = LlamaCppRuntime()
    assert llamacpp.supports_manifest(advertised) is False
    assert llamacpp.supports_manifest(with_evidence) is False
    assert ModelFormat.EXL3 not in llamacpp.supported_formats


def test_missing_inference_key_blocks_the_endpoint_before_any_request(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TABBY_API_KEY", raising=False)
    endpoint = _endpoint("http://127.0.0.1:5000/v1")
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,)), endpoint=endpoint)

    assert runtime.is_available() is False
    assert "TABBY_API_KEY" in (runtime.availability_reason() or "")


def test_wrong_key_is_a_structured_unavailable_error_without_leaking_the_body(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeTabby(api_key="inference-key", model_ids=["Qwen2.5-0.5B-Instruct-exl3-4.0bpw"])
    try:
        monkeypatch.setenv("TABBY_API_KEY", "wrong-key")
        endpoint = _endpoint(fake.base_url)
        runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,)), endpoint=endpoint)
        with pytest.raises(RuntimeUnavailableError) as excinfo:
            runtime.advertised_model_records(refresh=True)
        assert excinfo.value.details.get("status_code") == 401
        assert fake.auth_headers[-1] == "Bearer wrong-key"
        assert runtime.inventory_state == "failed"

        monkeypatch.setenv("TABBY_API_KEY", "inference-key")
        records = runtime.advertised_model_records(refresh=True)
        assert [record["id"] for record in records] == ["Qwen2.5-0.5B-Instruct-exl3-4.0bpw"]
        assert runtime.inventory_state == "advertised"
    finally:
        fake.stop()


def test_unavailable_tabby_endpoint_leaves_ollama_routable_and_conversion_refused(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeTabby(api_key="inference-key", model_ids=["Qwen2.5-0.5B-Instruct-exl3-4.0bpw"])
    monkeypatch.setenv("TABBY_API_KEY", "inference-key")
    monkeypatch.setattr(
        ollama_inventory, "_fetch_tags",
        lambda *_a, **_k: [{"name": "tiny:latest", "model": "tiny:latest", "digest": "a" * 64, "details": {"format": "gguf", "family": "llama"}, "capabilities": ["completion"]}],
    )
    settings = LewLMSettings(
        data_dir=tmp_path / "state", models_dir=(tmp_path / "models",), runtime_packs=("external_accelerator",),
        external_endpoints=(_endpoint(fake.base_url), ExternalEndpoint(endpoint_id="ollama", profile="ollama_local", base_url="http://127.0.0.1:11434")),
        ollama_discovery_enabled=True, ollama_base_url="http://127.0.0.1:11434",
    )
    (tmp_path / "state").mkdir(); (tmp_path / "models").mkdir()
    services = bootstrap_services(settings)
    try:
        services.model_registry.scan()
        manifests = {m.source_path: m for m in services.model_registry.list_manifests()}
        tabby_model = manifests["external://tabby/Qwen2.5-0.5B-Instruct-exl3-4.0bpw"]
        ollama_model = manifests["ollama://tiny:latest"]
        assert tabby_model.metadata["external_profile"] == "exllamav3_tabby"

        with pytest.raises(ConversionError):
            services.conversion_service.plan_targets(tabby_model.model_id)

        fake.stop()
        services.runtime_catalog.get_endpoint_runtime("tabby").invalidate_discovery_cache()
        with pytest.raises(RuntimeUnavailableError) as excinfo:
            services.model_router.route_chat(tabby_model.model_id)
        assert excinfo.value.details["endpoint_id"] == "tabby"
        assert excinfo.value.details["engine_profile"] == "exllamav3_tabby"

        ollama_runtime = services.runtime_catalog.get_endpoint_runtime("ollama")
        ollama_runtime._discovered_model_ids = ("tiny:latest",)
        ollama_runtime._discovered_model_records = ({"id": "tiny:latest"},)
        _, runtime, decision = services.model_router.route_chat(ollama_model.model_id)
        assert runtime.name == "local_external_adapter:ollama" and decision.endpoint_id == "ollama"
        assert excinfo.value.details["fallback_policy"] == "none", "no silent substitution for the Tabby model"
    finally:
        try:
            fake.stop()
        except Exception:
            pass


def test_tabby_sampling_map_matches_what_the_pinned_server_implements(tmp_path: Path) -> None:
    # TabbyAPI 53da7919's sampler request has top_k/min_p/repetition_penalty and
    # no seed (accepted by the schema, then dropped): never claim determinism.
    from lewlm.core.contracts import GenerateMessage, GenerateRequest, SamplingControls
    from lewlm.runtime.adapters import LocalOpenAICompatibleAdapterRuntime

    endpoint = ExternalEndpoint(endpoint_id="tabby", profile="exllamav3_tabby", base_url="http://127.0.0.1:5000/v1")
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,)), endpoint=endpoint)
    request = GenerateRequest(model_id="m", messages=[GenerateMessage(role="user", content="hi")], max_tokens=8, temperature=0.9,
                              sampling=SamplingControls(seed=7, top_k=20, min_p=0.05, repetition_penalty=1.1))
    payload = runtime._chat_payload(remote_model_id="qwen-exl3", request=request, stream=False)
    report = request.metadata["sampling_controls"]
    assert "seed" not in payload and (payload["top_k"], payload["min_p"], payload["repetition_penalty"]) == (20, 0.05, 1.1)
    assert report["unsupported"] == ["seed"] and report["deterministic"] is False
