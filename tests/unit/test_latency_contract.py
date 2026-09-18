"""Portable checks for modernization step 09 (runtime latency contract).

Everything here runs without an engine or GPU. Hardware tuning itself (CPU
thread counts, CUDA graphs, compile caches) is measured on real hardware and
recorded separately; these tests pin down the contracts that make those
measurements usable and keep the middleware from adding waits of its own:

- serving-profile presets (`interactive` / `throughput`) are stored per preset
  and a recommendation is rejected as stale when its measured inputs change;
- many model aliases on one endpoint cannot bypass the aggregate admission cap;
- nothing warms a bridge model implicitly (scan, health, runtime info);
- the three startup phases are reported separately from cached state.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lewlm.api.app import create_app
from lewlm.config.endpoints import ExternalEndpoint
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import CapabilityName, GenerateMessage, utc_now
from lewlm.runtime.adapters import LocalOpenAICompatibleAdapterRuntime
from lewlm.serving_profiles import (
    SERVING_PROFILE_PRESETS,
    normalize_serving_profile_preset,
    resolve_serving_profile_application,
    serving_profile_fingerprint,
)
from lewlm.storage import MetadataStore


class _FakeEngine:
    """A loopback server advertising several aliases; streams hold until released."""

    def __init__(self, *, model_ids: list[str]) -> None:
        self.model_ids = model_ids
        self.in_flight = 0
        self.peak_in_flight = 0
        self.chat_calls = 0
        self.release = threading.Event()
        self._lock = threading.Lock()
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps({"object": "list", "data": [{"id": m, "object": "model", "max_model_len": 4096} for m in fake.model_ids]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                with fake._lock:
                    fake.chat_calls += 1
                    fake.in_flight += 1
                    fake.peak_in_flight = max(fake.peak_in_flight, fake.in_flight)
                try:
                    fake.release.wait(timeout=5.0)
                    frames = [
                        {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
                        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
                    ]
                    body = ("".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    with fake._lock:
                        fake.in_flight -= 1

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def stop(self) -> None:
        self.release.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _endpoint(base_url: str, profile: str = "vllm_local") -> ExternalEndpoint:
    return ExternalEndpoint(endpoint_id="engine", profile=profile, base_url=base_url, read_timeout_seconds=10)


def _services(tmp_path: Path, base_url: str, **overrides):
    settings = LewLMSettings(
        environment="test",
        data_dir=tmp_path / "state", models_dir=(tmp_path / "models",), runtime_packs=("external_accelerator",),
        external_endpoints=(_endpoint(base_url),), backend_feature_probes_enabled=False, **overrides,
    )
    (tmp_path / "state").mkdir(parents=True, exist_ok=True); (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    return bootstrap_services(settings)


# --- presets and fingerprints -------------------------------------------------

def test_presets_are_the_two_measured_workloads_and_normalize_strictly() -> None:
    assert SERVING_PROFILE_PRESETS == ("interactive", "throughput")
    assert normalize_serving_profile_preset(None) == "interactive"
    assert normalize_serving_profile_preset("Throughput") == "throughput"
    with pytest.raises(ValueError):
        normalize_serving_profile_preset("batch")


def test_fingerprint_captures_host_engine_model_revision_precision_workload_and_preset(tmp_path: Path) -> None:
    fake = _FakeEngine(model_ids=["m"])
    try:
        services = _services(tmp_path, fake.base_url)
        services.model_registry.scan()
        manifest = next(m for m in services.model_registry.list_manifests())
        runtime = services.runtime_catalog.get_endpoint_runtime("engine")
        host = services.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
        fingerprint = serving_profile_fingerprint(host_platform=host, runtime=runtime, manifest=manifest, workload_class="text_only", preset="interactive")
        assert set(fingerprint) == {"host", "runtime", "engine_profile", "engine_server", "model_revision", "precision", "workload_class", "preset"}
        assert fingerprint["runtime"] == "local_external_adapter:engine"
        assert fingerprint["engine_profile"] == "vllm_local"
        assert fingerprint["engine_server"] == fake.base_url
        assert fingerprint["model_revision"] == manifest.fingerprint
        assert fingerprint["preset"] == "interactive"
        assert fingerprint == serving_profile_fingerprint(host_platform=host, runtime=runtime, manifest=manifest, workload_class="text_only", preset="interactive")
    finally:
        fake.stop()


def test_recommendation_is_stale_when_a_measured_input_changes_and_presets_never_cross(tmp_path: Path) -> None:
    fake = _FakeEngine(model_ids=["m"])
    try:
        services = _services(tmp_path, fake.base_url)
        services.model_registry.scan()
        manifest = next(m for m in services.model_registry.list_manifests())
        runtime = services.runtime_catalog.get_endpoint_runtime("engine")
        host = services.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
        store: MetadataStore = services.metadata_store
        fingerprint = serving_profile_fingerprint(host_platform=host, runtime=runtime, manifest=manifest, workload_class="text_only", preset="interactive")
        payload = {
            "profile_id": "p1", "runtime": runtime.name, "recommended_at": utc_now().isoformat(), "reason": "measured",
            "settings_overrides": {"runtime_policy": "keep_warm"}, "preset": "interactive", "fingerprint": fingerprint,
        }
        store.upsert_serving_profile(model_id=manifest.model_id, capability="chat", host_platform=host, runtime_name=runtime.name,
                                     workload_class="text_only", preset="interactive", payload=payload)

        def resolve(**kw):
            kw.setdefault("manifest", manifest)
            return resolve_serving_profile_application(
                settings=services.settings, metadata_store=store, host_platform=host, runtime=runtime, model_id=manifest.model_id,
                request_capability=CapabilityName.CHAT, apply_serving_profile=True, workload_class="text_only", **kw,
            )

        selected = resolve()
        assert selected.status == "selected" and selected.preset == "interactive"
        assert selected.accepted_settings == {"runtime_policy": "keep_warm"}

        # The same model with a different artifact revision: stale, overrides rejected, inputs named.
        changed = manifest.model_copy(update={"fingerprint": "different-artifact"})
        stale = resolve(manifest=changed)
        assert stale.status == "stale"
        assert stale.stale_inputs == {"model_revision": (manifest.fingerprint, "different-artifact")}
        assert stale.accepted_settings == {} and "runtime_policy" in stale.rejected_settings
        assert "re-run autotune" in stale.rejected_settings["runtime_policy"].reason

        # A throughput request never adopts an interactive recommendation.
        other = resolve(preset="throughput")
        assert other.status == "not_found" and other.preset == "throughput"

        # The deployment-wide default preset comes from settings.
        throughput_settings = services.settings.with_updates(serving_profile_preset="throughput")
        assert resolve_serving_profile_application(
            settings=throughput_settings, metadata_store=store, host_platform=host, runtime=runtime, model_id=manifest.model_id,
            request_capability=CapabilityName.CHAT, apply_serving_profile=True, workload_class="text_only", manifest=manifest,
        ).status == "not_found"

        # Profiles recorded before fingerprints existed stay usable.
        legacy = dict(payload); legacy.pop("fingerprint"); legacy.pop("preset")
        store.upsert_serving_profile(model_id=manifest.model_id, capability="chat", host_platform=host, runtime_name=runtime.name,
                                     workload_class="text_only", payload=legacy)
        assert resolve(manifest=changed).status == "selected"
    finally:
        fake.stop()


# --- admission cap across aliases --------------------------------------------

def test_many_aliases_on_one_endpoint_cannot_bypass_the_aggregate_admission_cap(tmp_path: Path) -> None:
    fake = _FakeEngine(model_ids=[f"alias-{i}" for i in range(6)])
    try:
        services = _services(tmp_path, fake.base_url, max_concurrent_runtime_requests=2, runtime_request_queue_limit=16, runtime_request_queue_timeout_seconds=30)
        services.model_registry.scan()
        manifests = sorted(services.model_registry.list_manifests(), key=lambda m: m.model_id)
        assert len(manifests) == 6

        async def run() -> tuple[int, int]:
            async def one(manifest):
                session = await services.chat_orchestrator.stream(
                    model_id=manifest.model_id, messages=[GenerateMessage(role="user", content="x")], max_tokens=4, temperature=0.0, apply_serving_profile=False,
                )
                return [delta async for delta in session.stream]

            tasks = [asyncio.create_task(one(m)) for m in manifests]
            # Let the first admissions reach the engine, then release everything.
            deadline = time.monotonic() + 5.0
            while fake.in_flight < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)
            observed_peak = fake.peak_in_flight
            fake.release.set()
            await asyncio.gather(*tasks)
            return observed_peak, fake.chat_calls

        peak, calls = asyncio.run(run())
        assert calls == 6, "every alias was served"
        assert peak == 2, f"six aliases on one endpoint never exceeded the cap of 2 upstream (peak {peak})"
        stats = services.runtime_request_scheduler.snapshot()
        assert stats["peak_active_requests"] == 2 and stats["active_requests"] == 0
        assert stats["total_queued_requests"] >= 4, "the excess waited in LewLM's queue rather than reaching the engine"
    finally:
        fake.stop()


# --- no implicit warm, startup phases -----------------------------------------

def test_scan_health_and_runtime_info_never_warm_a_bridge_model_and_report_phases(tmp_path: Path) -> None:
    fake = _FakeEngine(model_ids=["m"])
    fake.release.set()
    try:
        services = _services(tmp_path, fake.base_url, api_keys=())
        assert services.runtime_instance.ready_at is not None
        assert services.runtime_instance.ready_at >= services.runtime_instance.started_at
        app = create_app(services=services)
        with TestClient(app) as client:
            assert client.get("/v1/health").status_code == 200
            services.model_registry.scan()
            models = client.get("/v1/models").json()
            assert models["count"] == 1
            info = client.get("/v1/runtime").json()
            assert fake.chat_calls == 0, "listing, health, and runtime info made no generation request"

            startup = info["startup"]
            assert startup["lewlm_ready_at"] is not None and startup["lewlm_ready_seconds"] >= 0.0
            [engine] = startup["engines"]
            assert engine["endpoint_id"] == "engine" and engine["profile"] == "vllm_local"
            assert engine["state"] == "advertised" and engine["advertised_model_count"] == 1
            assert engine["first_advertised_at"] is not None
            assert startup["warm_models"] == [] and startup["loading_models"] == []

            # Engine gone: runtime info still answers from cached state, without probing.
            fake.stop()
            info = client.get("/v1/runtime").json()
            assert info["startup"]["engines"][0]["state"] == "advertised", "cached; no probe was made"
            assert client.get("/v1/health").status_code == 200
    finally:
        try:
            fake.stop()
        except Exception:
            pass


def test_bridge_records_first_advertised_time_once(tmp_path: Path) -> None:
    fake = _FakeEngine(model_ids=["m"])
    try:
        endpoint = _endpoint(fake.base_url)
        runtime = LocalOpenAICompatibleAdapterRuntime(settings=LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,)), endpoint=endpoint)
        assert runtime.endpoint_snapshot()["first_advertised_at"] is None
        runtime.advertised_model_records(refresh=True)
        first = runtime.endpoint_snapshot()["first_advertised_at"]
        assert first is not None
        runtime.advertised_model_records(refresh=True)
        assert runtime.endpoint_snapshot()["first_advertised_at"] == first
    finally:
        fake.stop()
