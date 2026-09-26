"""Modernization step 12: enable an engine, serve, lose it, fall back, disable it.

Two fake engines stand in for a new accelerator (`primary`) and an existing
compatible path (`fallback`). Everything runs over LewLM's public HTTP API:

1. the new endpoint serves a request;
2. with the engine stopped, an explicitly configured alias serves the same
   model id through the existing path *before* generation is submitted;
3. an active stream that loses its engine fails transparently — terminal
   error chunk, no replay — and the next request goes to the fallback;
4. disabling the endpoint (the rollback) keeps the other path, the
   registered models, and the caches; doctor names what to run next.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import pytest

from lewlm.cli.main import handle_doctor
from lewlm.config.endpoints import ExternalEndpoint
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.testing import FakeOpenAIEngine


def _settings(data_dir: Path, primary: FakeOpenAIEngine, fallback: FakeOpenAIEngine, *, primary_enabled: bool, aliases: dict[str, str] | None) -> LewLMSettings:
    (data_dir / "state").mkdir(parents=True, exist_ok=True)
    (data_dir / "models").mkdir(parents=True, exist_ok=True)
    return LewLMSettings(
        environment="test", data_dir=data_dir / "state", models_dir=(data_dir / "models",), runtime_packs=("external_accelerator",),
        backend_feature_probes_enabled=False, api_keys=(),
        external_endpoints=(
            ExternalEndpoint(endpoint_id="primary", profile="vllm_local", base_url=primary.base_url, enabled=primary_enabled, read_timeout_seconds=30),
            ExternalEndpoint(endpoint_id="fallback", profile="openai_compatible", base_url=fallback.base_url, read_timeout_seconds=30),
        ),
        external_fallback_policy="explicit_alias" if aliases else "none",
        external_fallback_aliases=aliases or {},
    )


def _serve(settings: LewLMSettings):
    """LewLM's HTTP app over an in-process transport; no uvicorn needed."""

    from lewlm.api.app import create_app

    services = bootstrap_services(settings)
    services.model_registry.scan()
    app = create_app(settings, services=services)
    return services, app


def _read_sse(response) -> list[dict]:
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


@pytest.fixture
def engines():
    primary = FakeOpenAIEngine(model_ids=("accel-chat",), stream_delay_seconds=0.01).start()
    fallback = FakeOpenAIEngine(model_ids=("accel-chat",), stream_delay_seconds=0.01).start()
    try:
        yield primary, fallback
    finally:
        primary.stop()
        fallback.stop()


def test_enable_serve_lose_fall_back_and_disable(engines, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    primary, fallback = engines
    data_dir = tmp_path / "lewlm"

    # --- 1. enable the new endpoint and serve through it ------------------------
    services, app = _serve(_settings(data_dir, primary, fallback, primary_enabled=True, aliases=None))
    manifests = {m.metadata["external_endpoint_id"]: m for m in services.model_registry.list_manifests()}
    primary_model, fallback_model = manifests["primary"].model_id, manifests["fallback"].model_id
    assert primary_model != fallback_model, "same upstream name on two endpoints stays two models"
    with TestClient(app) as client:
        body = client.post("/v1/chat/completions", json={"model": primary_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).json()
        assert body["metadata"]["model"]["endpoint_id"] == "primary" and body["metadata"]["routing"]["fallback_from_model_id"] is None
        assert len(primary.requests) == 1 and len(fallback.requests) == 0

        # --- 2. engine gone, no alias configured: a 503 naming the endpoint, other path untouched
        primary.stop()
        client.post("/v1/models/scan", json={})
        outage = client.post("/v1/chat/completions", json={"model": primary_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        assert outage.status_code == 503 and outage.json()["error"]["details"]["endpoint_id"] == "primary"
        assert client.post("/v1/chat/completions", json={"model": fallback_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).status_code == 200
        health = client.get("/v1/health").json()
        assert health["status"] == "ok" and {e["endpoint_id"]: e["state"] for e in health["engines"]}["primary"] in {"failed", "stale"}
        assert any(item["model_id"] == primary_model for item in client.get("/v1/models").json()["items"]), "the model is not deleted"
    primary.start(primary._port)

    # --- 3. explicit alias: preflight fallback to the existing path when the engine is down
    services, app = _serve(_settings(data_dir, primary, fallback, primary_enabled=True, aliases={primary_model: fallback_model}))
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json={"model": primary_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).json()["metadata"]["model"]["endpoint_id"] == "primary"

        # An active stream loses its engine: transparent failure, no replay.
        primary_requests_before = len(primary.requests)
        fallback_requests_before = len(fallback.requests)
        primary.die_after_frames = 3
        try:
            with client.stream("POST", "/v1/chat/completions", json={"model": primary_model, "messages": [{"role": "user", "content": "Write a long story."}], "max_tokens": 256, "stream": True}) as response:
                frames = _read_sse(response)
        finally:
            primary.die_after_frames = None
        terminal = next(f for f in reversed(frames) if isinstance(f, dict) and f.get("choices") and f["choices"][0].get("finish_reason"))
        assert terminal["choices"][0]["finish_reason"] == "error" and terminal["error"]["partial_output"] is True
        assert frames[-1] == {"_done": True}
        assert len(primary.requests) == primary_requests_before + 1, "the interrupted stream was never replayed"
        assert len(fallback.requests) == fallback_requests_before, "no mid-stream substitution either"

        # Now the engine is down for good: the next request takes the configured alias.
        primary.stop()
        client.post("/v1/models/scan", json={})
        fallen = client.post("/v1/chat/completions", json={"model": primary_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).json()
        assert fallen["metadata"]["model"]["endpoint_id"] == "fallback"
        assert fallen["metadata"]["model"]["requested_model_id"] == primary_model and fallen["metadata"]["model"]["resolved_model_id"] == fallback_model
        assert fallen["metadata"]["routing"]["fallback_from_model_id"] == primary_model
        assert "alias" in fallen["metadata"]["routing"]["fallback_reason"]
        assert len(fallback.requests) == fallback_requests_before + 1

    # --- 4. rollback: disable the endpoint; the other path, models, and caches stay
    cache_artifacts_before = services.telemetry_service.cache_stats().artifact_count
    services, app = _serve(_settings(data_dir, primary, fallback, primary_enabled=False, aliases={primary_model: fallback_model}))
    with TestClient(app) as client:
        health = client.get("/v1/health").json()
        assert {e["endpoint_id"]: e["enabled"] for e in health["engines"]}["primary"] is False
        models = client.get("/v1/models").json()
        assert any(item["model_id"] == fallback_model for item in models["items"])
        assert client.post("/v1/chat/completions", json={"model": fallback_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).json()["metadata"]["model"]["endpoint_id"] == "fallback"
        # A disabled endpoint's advertised models leave the registry on the next
        # scan, so their ids are unknown from now on: clients point at the
        # fallback model directly (documented in the rollback guide).
        via_old_id = client.post("/v1/chat/completions", json={"model": primary_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        assert via_old_id.status_code == 404 and via_old_id.json()["error"]["code"] == "model_not_found"
        assert not any(item["model_id"] == primary_model for item in models["items"])
    assert services.telemetry_service.cache_stats().artifact_count == cache_artifacts_before

    # Doctor says exactly what the operator should do, without touching the engine.
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        handle_doctor(argparse.Namespace(json=True, no_probe=False), services.settings, services)
    guidance = json.loads(buffer.getvalue())["external_engines"]
    by_id = {engine["endpoint_id"]: engine for engine in guidance["engines"]}
    assert by_id["primary"]["state"] == "disabled" and '"enabled": true' in by_id["primary"]["next_command"]
    assert by_id["fallback"]["state"] == "ready"
    assert any("primary" in step for step in guidance["next_steps"])
    assert "rollout-and-rollback" in guidance["rollback"]


def test_legacy_singular_settings_still_work_and_migrate_to_one_endpoint(tmp_path: Path) -> None:
    """The documented migration: the singular variables synthesize `legacy-default`."""

    engine = FakeOpenAIEngine(model_ids=("accel-chat",)).start()
    try:
        (tmp_path / "state").mkdir(); (tmp_path / "models").mkdir()
        legacy = LewLMSettings(
            environment="test", data_dir=tmp_path / "state", models_dir=(tmp_path / "models",), runtime_packs=("external_accelerator",),
            backend_feature_probes_enabled=False,
            external_accelerator_enabled=True, external_accelerator_base_url=engine.base_url.removesuffix("/v1"), external_accelerator_profile="openai_compatible",
        )
        assert [e.endpoint_id for e in legacy.resolved_external_endpoints()] == ["legacy-default"]
        # `legacy-default` is reserved for the synthesized endpoint; a migration names its own id.
        migrated = legacy.with_updates(
            external_accelerator_enabled=False,
            external_endpoints=(ExternalEndpoint(endpoint_id="accelerator", profile="openai_compatible", base_url=engine.base_url),),
        )
        upstream_ids: list[set[str]] = []
        model_ids: list[set[str]] = []
        for settings in (legacy, migrated):
            services = bootstrap_services(settings)
            services.model_registry.scan()
            manifests = services.model_registry.list_manifests()
            upstream_ids.append({m.metadata["external_upstream_model_id"] for m in manifests})
            model_ids.append({m.model_id for m in manifests})
        assert upstream_ids[0] == upstream_ids[1] == {"accel-chat"}, "both forms register the same upstream models"
        assert model_ids[0] != model_ids[1], "the endpoint id is part of the LewLM model id, so a migration changes ids (documented)"
        with pytest.raises(ValueError):
            LewLMSettings(
                environment="test", data_dir=tmp_path / "state", models_dir=(tmp_path / "models",),
                external_accelerator_enabled=True, external_accelerator_base_url=engine.base_url,
                external_endpoints=(ExternalEndpoint(endpoint_id="x", profile="openai_compatible", base_url=engine.base_url),),
            )
    finally:
        engine.stop()


@pytest.mark.parametrize("with_alias", [True, False])
def test_availability_names_the_down_engine_and_the_alias_that_answers(engines, tmp_path: Path, with_alias: bool) -> None:
    """G38: a picker that obeys `chat_ready` can still offer a model the alias serves."""

    from fastapi.testclient import TestClient

    primary, fallback = engines
    data_dir = tmp_path / "lewlm"
    services, _ = _serve(_settings(data_dir, primary, fallback, primary_enabled=True, aliases=None))
    manifests = {m.metadata["external_endpoint_id"]: m for m in services.model_registry.list_manifests()}
    primary_model, fallback_model = manifests["primary"].model_id, manifests["fallback"].model_id
    _, app = _serve(_settings(data_dir, primary, fallback, primary_enabled=True, aliases={primary_model: fallback_model} if with_alias else None))
    with TestClient(app) as client:
        def entry(model_id: str) -> dict:
            return next(item for item in client.get("/v1/models").json()["capability_availability"] if item["model_id"] == model_id)

        assert entry(primary_model)["chat_ready"] is True and entry(primary_model)["fallback_model_id"] is None
        primary.stop()
        try:
            client.post("/v1/models/scan", json={})
            down = entry(primary_model)
            assert down["chat_ready"] is False
            assert down["engine_state"] in {"stale", "failed"}
            assert "Engine `primary` is" in down["reason"]
            if with_alias:
                assert down["fallback_model_id"] == fallback_model
                assert f"answered by the fallback alias `{fallback_model}`" in down["reason"]
                served = client.post("/v1/chat/completions", json={"model": primary_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).json()
                assert served["metadata"]["routing"]["fallback_from_model_id"] == primary_model
            else:
                assert down["fallback_model_id"] is None
            assert entry(fallback_model)["fallback_model_id"] is None
        finally:
            primary.start(primary._port)
