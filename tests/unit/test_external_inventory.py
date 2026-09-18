"""External endpoint inventory, endpoint-qualified identity, path guards, and
explicit fallback routing (modernization step 04).

Two live loopback `/v1/models` servers and a stubbed Ollama daemon coexist in
one registry; every check runs through the real bootstrap so the registry,
catalog, router, and conversion service see exactly what an operator would.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lewlm.config.endpoints import ExternalEndpoint
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import ModelFormat, ModelModality, RuntimeAffinity
from lewlm.core.errors import ConversionError, RoutingError, RuntimeUnavailableError
from lewlm.core.execution_metadata import build_routed_execution_metadata
from lewlm.registry import ollama_inventory
from lewlm.registry.external_inventory import build_external_manifest
from lewlm.runtime.adapters import openai_compatible
from lewlm.runtime.llamacpp.runtime import LlamaCppRuntime
from lewlm.utils.model_identity import (
    external_model_id,
    external_source_path,
    is_external_source,
    is_uri_source,
    parse_external_source,
)


class _FakeEndpoint:
    """A loopback OpenAI-compatible `/v1/models` server whose list can change."""

    def __init__(self, model_ids: list[str]) -> None:
        self.model_ids = list(model_ids)
        self.requests: list[str] = []
        self.chat_requests = 0
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                fake.requests.append(self.path)
                if self.path != "/v1/models":
                    self.send_response(404)
                    self.end_headers()
                    return
                payload = {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "fake"} for model_id in fake.model_ids]}
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                fake.requests.append(self.path)
                fake.chat_requests += 1
                length = int(self.headers.get("Content-Length", "0") or 0)
                self.rfile.read(length)
                # A stream that dies after one fragment: the caller must see a
                # failure, and this server must see exactly one request.
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"index":0,"delta":{"content":"par"}}]}\n\n')
                self.wfile.flush()
                self.connection.close()

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.port = self._server.server_port
        self.base_url = f"http://127.0.0.1:{self.port}/v1"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@contextmanager
def _two_endpoints():
    alpha = _FakeEndpoint(["shared-name", "alpha-only"])
    beta = _FakeEndpoint(["shared-name", "beta-only.gguf"])
    try:
        yield alpha, beta
    finally:
        alpha.stop()
        beta.stop()


def _settings(tmp_path: Path, alpha: _FakeEndpoint, beta: _FakeEndpoint, **overrides) -> LewLMSettings:
    endpoints = (
        ExternalEndpoint(endpoint_id="alpha", profile="vllm_local", base_url=alpha.base_url),
        ExternalEndpoint(endpoint_id="beta", profile="llamacpp_server", base_url=beta.base_url, **overrides.pop("beta_kwargs", {})),
        ExternalEndpoint(endpoint_id="ollama", profile="ollama_local", base_url="http://127.0.0.1:11434"),
    )
    return LewLMSettings(
        data_dir=tmp_path / "state",
        models_dir=(tmp_path / "models",),
        runtime_packs=("external_accelerator",),
        external_endpoints=endpoints,
        ollama_discovery_enabled=True,
        ollama_base_url="http://127.0.0.1:11434",
        **overrides,
    )


@pytest.fixture
def stub_ollama(monkeypatch):
    monkeypatch.setattr(
        ollama_inventory,
        "_fetch_tags",
        lambda *_args, **_kwargs: [
            {"name": "tiny:latest", "model": "tiny:latest", "digest": "a" * 64,
             "details": {"format": "gguf", "family": "llama"}, "capabilities": ["completion"]},
            {"name": "shared-name", "model": "shared-name", "digest": "b" * 64,
             "details": {"format": "gguf", "family": "llama"}, "capabilities": ["completion"]},
        ],
    )


def _bootstrap(settings: LewLMSettings):
    (settings.data_dir).mkdir(parents=True, exist_ok=True)
    settings.models_dir[0].mkdir(parents=True, exist_ok=True)
    return bootstrap_services(settings)


# --- identity helpers -------------------------------------------------------


def test_external_source_uri_round_trips_and_keeps_slashes_in_one_segment() -> None:
    uri = external_source_path("gpu", "Qwen/Qwen2.5-0.5B-Instruct")
    assert uri == "external://gpu/Qwen%2FQwen2.5-0.5B-Instruct"
    assert parse_external_source(uri) == ("gpu", "Qwen/Qwen2.5-0.5B-Instruct")
    assert is_external_source(uri) and is_uri_source(uri) and is_uri_source("ollama://tiny:latest")
    assert not is_uri_source("/models/tiny.gguf")
    assert parse_external_source("external://gpu") is None


def test_external_model_ids_are_stable_and_endpoint_qualified() -> None:
    first = external_model_id("alpha", "shared-name")
    assert first == external_model_id("alpha", "shared-name")
    assert first != external_model_id("beta", "shared-name")
    assert external_model_id("alpha", "Org/Model-1") != external_model_id("alpha", "org-model_1")
    assert first.startswith("shared-name-alpha-")


# --- manifest evidence ------------------------------------------------------


def test_manifest_format_comes_from_evidence_not_profile_labels() -> None:
    generic = ExternalEndpoint(endpoint_id="a", profile="vllm_local", base_url="http://127.0.0.1:1/v1")
    gguf_server = ExternalEndpoint(endpoint_id="b", profile="llamacpp_server", base_url="http://127.0.0.1:2/v1")

    unknown = build_external_manifest({"id": "m"}, endpoint=generic, upstream_id="m")
    assert unknown.format_type is ModelFormat.UNKNOWN
    assert unknown.last_validation_result.details["format_evidence"] == "none"
    assert unknown.metadata["execution_locality"] == "loopback_unverified"
    assert unknown.metadata["external_upstream_model_id"] == "m"
    assert unknown.runtime_affinity == (RuntimeAffinity.EXTERNAL_ACCELERATOR,)

    by_name = build_external_manifest({"id": "m.gguf"}, endpoint=generic, upstream_id="m.gguf")
    assert by_name.format_type is ModelFormat.GGUF
    by_server = build_external_manifest({"id": "m"}, endpoint=gguf_server, upstream_id="m")
    assert by_server.format_type is ModelFormat.GGUF
    exl3 = build_external_manifest({"id": "m", "metadata": {"format": "exl3"}}, endpoint=generic, upstream_id="m")
    assert exl3.format_type is ModelFormat.EXL3
    with_context = build_external_manifest({"id": "m", "max_model_len": 32768}, endpoint=generic, upstream_id="m")
    assert with_context.context_length == 32768


# --- scan: coexistence, refresh, failure ------------------------------------


def test_two_live_endpoints_and_ollama_coexist_with_distinct_identities(tmp_path: Path, stub_ollama) -> None:
    with _two_endpoints() as (alpha, beta):
        services = _bootstrap(_settings(tmp_path, alpha, beta))
        summary = services.model_registry.scan()

        by_source = {m.source_path: m for m in services.model_registry.list_manifests()}
        assert set(by_source) == {
            "external://alpha/shared-name", "external://alpha/alpha-only",
            "external://beta/shared-name", "external://beta/beta-only.gguf",
            "ollama://tiny:latest", "ollama://shared-name",
        }
        # Same upstream name, three different servers, three different models.
        ids = {by_source[s].model_id for s in ("external://alpha/shared-name", "external://beta/shared-name", "ollama://shared-name")}
        assert len(ids) == 3
        assert by_source["external://alpha/shared-name"].metadata["external_endpoint_id"] == "alpha"
        assert by_source["external://beta/shared-name"].metadata["external_endpoint_id"] == "beta"
        assert by_source["ollama://tiny:latest"].metadata["external_endpoint_id"] == "ollama"
        assert by_source["external://beta/beta-only.gguf"].format_type is ModelFormat.GGUF
        assert by_source["external://alpha/alpha-only"].format_type is ModelFormat.UNKNOWN
        # The Ollama endpoint is inventoried once, by the Ollama path only.
        assert not any(s.startswith("external://ollama/") for s in by_source)
        assert any("`alpha` (vllm_local) advertised 2 model(s)" in note for note in summary.notes)
        assert summary.new_count == 6


def test_refresh_adds_and_removes_only_the_changed_endpoints_records(tmp_path: Path, stub_ollama) -> None:
    with _two_endpoints() as (alpha, beta):
        services = _bootstrap(_settings(tmp_path, alpha, beta))
        services.model_registry.scan()

        beta.model_ids = ["shared-name", "beta-new"]
        summary = services.model_registry.scan()

        sources = {m.source_path for m in services.model_registry.list_manifests()}
        assert "external://beta/beta-new" in sources
        assert "external://beta/beta-only.gguf" not in sources
        assert {"external://alpha/shared-name", "external://alpha/alpha-only", "ollama://tiny:latest"} <= sources
        assert summary.new_count == 1 and summary.removed_count == 1


def test_unreachable_endpoint_keeps_its_last_known_models_as_stale(tmp_path: Path, stub_ollama) -> None:
    with _two_endpoints() as (alpha, beta):
        services = _bootstrap(_settings(tmp_path, alpha, beta))
        services.model_registry.scan()
        alpha_runtime = services.runtime_catalog.get_endpoint_runtime("alpha")
        assert alpha_runtime.inventory_state == "advertised"

        alpha.stop()
        summary = services.model_registry.scan()

        sources = {m.source_path for m in services.model_registry.list_manifests()}
        assert {"external://alpha/shared-name", "external://alpha/alpha-only"} <= sources, "unreachable is not deleted"
        assert summary.removed_count == 0
        assert any("`alpha` could not be read" in note and "keeping 2 previously registered" in note for note in summary.notes)
        assert alpha_runtime.inventory_state == "stale"
        snapshot = alpha_runtime.endpoint_snapshot()
        assert snapshot["inventory_state"] == "stale"
        assert snapshot["advertised_model_ids"] == ["shared-name", "alpha-only"]
        assert snapshot["inventory_error"]
        # Beta was read normally; one endpoint's failure is confined to it.
        assert services.runtime_catalog.get_endpoint_runtime("beta").inventory_state == "advertised"
        beta.stop()


def test_disabled_endpoint_retires_its_advertised_models_on_the_next_scan(tmp_path: Path, stub_ollama) -> None:
    with _two_endpoints() as (alpha, beta):
        services = _bootstrap(_settings(tmp_path, alpha, beta))
        services.model_registry.scan()

        disabled = _bootstrap(_settings(tmp_path, alpha, beta, beta_kwargs={"enabled": False}))
        summary = disabled.model_registry.scan()

        sources = {m.source_path for m in disabled.model_registry.list_manifests()}
        assert not any(s.startswith("external://beta/") for s in sources)
        assert {"external://alpha/shared-name", "ollama://tiny:latest"} <= sources
        assert summary.removed_count == 2


# --- path guards ------------------------------------------------------------


def test_uri_backed_models_never_enter_conversion_or_packaged_runtimes(tmp_path: Path, stub_ollama) -> None:
    with _two_endpoints() as (alpha, beta):
        services = _bootstrap(_settings(tmp_path, alpha, beta))
        services.model_registry.scan()
        gguf_bound = next(m for m in services.model_registry.list_manifests() if m.source_path == "external://beta/beta-only.gguf")
        ollama_bound = next(m for m in services.model_registry.list_manifests() if m.source_path == "ollama://tiny:latest")

        with pytest.raises(ConversionError, match="no local artifact to convert"):
            services.conversion_service.plan_targets(gguf_bound.model_id)
        with pytest.raises(ConversionError):
            services.conversion_service.plan_targets(ollama_bound.model_id)
        # Declared GGUF or not, llama.cpp is never handed a URI as weights.
        assert LlamaCppRuntime().supports_manifest(gguf_bound) is False
        assert LlamaCppRuntime().supports_manifest(ollama_bound) is False
        # The owning endpoint serves it regardless of the unknown weight format.
        unknown_bound = next(m for m in services.model_registry.list_manifests() if m.source_path == "external://alpha/alpha-only")
        assert unknown_bound.format_type is ModelFormat.UNKNOWN
        assert services.runtime_catalog.get_endpoint_runtime("alpha").supports_manifest(unknown_bound) is True
        assert services.runtime_catalog.get_endpoint_runtime("beta").supports_manifest(unknown_bound) is False


# --- adapter inventory TTL --------------------------------------------------


def test_successful_inventory_is_reread_after_the_ttl_and_on_explicit_refresh(tmp_path: Path, monkeypatch) -> None:
    alpha = _FakeEndpoint(["m"])
    try:
        endpoint = ExternalEndpoint(endpoint_id="alpha", base_url=alpha.base_url)
        settings = LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,), external_inventory_ttl_seconds=30.0)
        runtime = openai_compatible.LocalOpenAICompatibleAdapterRuntime(settings=settings, endpoint=endpoint)
        clock = {"now": 1000.0}
        monkeypatch.setattr(openai_compatible, "monotonic", lambda: clock["now"])

        assert runtime.advertised_model_records() and alpha.requests.count("/v1/models") == 1
        clock["now"] += 10
        runtime.advertised_model_records()
        assert alpha.requests.count("/v1/models") == 1, "inside the TTL the cached list is used"
        clock["now"] += 25
        runtime.advertised_model_records()
        assert alpha.requests.count("/v1/models") == 2, "past the TTL the list is re-read"
        runtime.advertised_model_records(refresh=True)
        assert alpha.requests.count("/v1/models") == 3, "explicit refresh always re-reads"
        assert runtime.endpoint_snapshot()["inventory_ttl_seconds"] == 30.0
    finally:
        alpha.stop()


# --- routing ----------------------------------------------------------------


def test_explicit_routing_targets_the_bound_endpoint_and_records_evidence(tmp_path: Path, stub_ollama) -> None:
    with _two_endpoints() as (alpha, beta):
        services = _bootstrap(_settings(tmp_path, alpha, beta))
        services.model_registry.scan()
        beta_shared = next(m for m in services.model_registry.list_manifests() if m.source_path == "external://beta/shared-name")
        ollama_tiny = next(m for m in services.model_registry.list_manifests() if m.source_path == "ollama://tiny:latest")

        manifest, runtime, decision = services.model_router.route_chat(beta_shared.model_id)
        assert manifest.model_id == beta_shared.model_id
        assert runtime.name == "local_external_adapter:beta"
        assert decision.endpoint_id == "beta" and decision.engine_profile == "llamacpp_server"
        assert decision.execution_locality == "loopback_unverified"
        assert decision.fallback_from_model_id is None

        metadata = build_routed_execution_metadata(request_id="r", created=0, requested_model_id=beta_shared.model_id, routing=decision)
        assert metadata.model.endpoint_id == "beta" and metadata.model.engine_profile == "llamacpp_server"
        assert metadata.model.execution_locality == "loopback_unverified"

        # Ollama's own locality classification travels through unchanged.
        ollama_runtime = services.runtime_catalog.get_endpoint_runtime("ollama")
        ollama_runtime._discovered_model_ids = ("tiny:latest", "shared-name")
        ollama_runtime._discovered_model_records = ({"id": "tiny:latest"}, {"id": "shared-name"})
        _, _, ollama_decision = services.model_router.route_chat(ollama_tiny.model_id)
        assert ollama_decision.endpoint_id == "ollama" and ollama_decision.execution_locality == "host_local"


def test_unavailable_endpoint_surfaces_failure_by_default_and_leaves_other_paths_usable(tmp_path: Path, stub_ollama) -> None:
    with _two_endpoints() as (alpha, beta):
        services = _bootstrap(_settings(tmp_path, alpha, beta))
        services.model_registry.scan()
        alpha_only = next(m for m in services.model_registry.list_manifests() if m.source_path == "external://alpha/alpha-only")
        beta_shared = next(m for m in services.model_registry.list_manifests() if m.source_path == "external://beta/shared-name")
        alpha_runtime = services.runtime_catalog.get_endpoint_runtime("alpha")
        alpha.stop()
        alpha_runtime.invalidate_discovery_cache()

        with pytest.raises(RuntimeUnavailableError) as excinfo:
            services.model_router.route_chat(alpha_only.model_id)
        assert excinfo.value.details["endpoint_id"] == "alpha"
        assert excinfo.value.details["fallback_policy"] == "none"
        assert excinfo.value.details["endpoint"]["inventory_state"] == "failed"

        # The other endpoint keeps routing; a same-name model there is not a substitute.
        _, runtime, decision = services.model_router.route_chat(beta_shared.model_id)
        assert runtime.name == "local_external_adapter:beta" and decision.fallback_from_model_id is None


def test_explicit_alias_policy_substitutes_only_the_configured_registered_model(tmp_path: Path, stub_ollama) -> None:
    with _two_endpoints() as (alpha, beta):
        probe = _bootstrap(_settings(tmp_path, alpha, beta))
        probe.model_registry.scan()
        alpha_only = next(m for m in probe.model_registry.list_manifests() if m.source_path == "external://alpha/alpha-only")
        beta_shared = next(m for m in probe.model_registry.list_manifests() if m.source_path == "external://beta/shared-name")
        alpha_shared = next(m for m in probe.model_registry.list_manifests() if m.source_path == "external://alpha/shared-name")

        services = _bootstrap(_settings(
            tmp_path, alpha, beta,
            external_fallback_policy="explicit_alias",
            external_fallback_aliases={alpha_only.model_id: beta_shared.model_id, alpha_shared.model_id: "does-not-exist"},
        ))
        services.model_registry.scan()
        alpha.stop()
        services.runtime_catalog.get_endpoint_runtime("alpha").invalidate_discovery_cache()

        manifest, runtime, decision = services.model_router.route_chat(alpha_only.model_id)
        assert manifest.model_id == beta_shared.model_id
        assert runtime.name == "local_external_adapter:beta"
        assert decision.fallback_from_model_id == alpha_only.model_id
        assert "substituted the operator-configured alias" in decision.fallback_reason
        metadata = build_routed_execution_metadata(request_id="r", created=0, requested_model_id=alpha_only.model_id, routing=decision)
        assert metadata.routing.fallback_from_model_id == alpha_only.model_id
        assert metadata.model.requested_model_id == alpha_only.model_id and metadata.model.resolved_model_id == beta_shared.model_id

        # An alias to an unregistered model is not a fallback; the outage surfaces
        # as the endpoint's unavailability, with the unusable alias named.
        with pytest.raises(RuntimeUnavailableError) as excinfo:
            services.model_router.route_chat(alpha_shared.model_id)
        assert any("does-not-exist" in item for item in excinfo.value.details["alternatives"])


@pytest.mark.asyncio
async def test_a_stream_that_dies_mid_flight_is_not_retried(tmp_path: Path, stub_ollama) -> None:
    from lewlm.core.contracts import GenerateMessage, GenerateRequest

    with _two_endpoints() as (alpha, beta):
        services = _bootstrap(_settings(tmp_path, alpha, beta))
        services.model_registry.scan()
        beta_shared = next(m for m in services.model_registry.list_manifests() if m.source_path == "external://beta/shared-name")
        runtime = services.runtime_catalog.get_endpoint_runtime("beta")
        await runtime.load_model(beta_shared)

        received: list[str] = []
        with pytest.raises(Exception):
            async for chunk in runtime.stream_generate(
                GenerateRequest(model_id=beta_shared.model_id, messages=[GenerateMessage(role="user", content="hi")], max_tokens=8),
            ):
                received.append(chunk)
        assert beta.chat_requests == 1, "no automatic replay after partial output"
        assert received == ["par"] or received == []
