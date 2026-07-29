"""`/v1/models` reports which models can actually serve on this host."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_inventory_annotates_per_model_serving_readiness(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        client.post("/v1/models/scan", json={})
        response = client.get("/v1/models")
        assert response.status_code == 200, response.text
        body = response.json()

    assert body["count"] == len(body["items"])
    availability = body["capability_availability"]
    # Every listed model is annotated, so no caller needs a per-model fan-out.
    assert len(availability) == len(body["items"])
    assert [item["model_id"] for item in availability] == [item["model_id"] for item in body["items"]]

    for entry in availability:
        assert entry["reason"]
        assert entry["servable"] == bool(entry["ready_capabilities"])
        assert entry["chat_ready"] == ("chat" in entry["ready_capabilities"])
        assert not set(entry["ready_capabilities"]) & set(entry["blocked_capabilities"])

    assert body["chat_ready_count"] == sum(1 for item in availability if item["chat_ready"])
    assert body["servable_count"] == sum(1 for item in availability if item["servable"])
    # The fixture registry contains at least one chat-capable model.
    assert body["chat_ready_count"] >= 1


def test_inventory_availability_matches_the_capabilities_route(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        client.post("/v1/models/scan", json={})
        body = client.get("/v1/models").json()
        availability = {item["model_id"]: item for item in body["capability_availability"]}

        for model_id, entry in availability.items():
            report = client.get(f"/v1/models/{model_id}/capabilities").json()
            supported = {
                item["capability"] for item in report["capabilities"] if item["supported"]
            }
            # The cheap inventory annotation must agree with the authoritative
            # per-model report that callers previously had to fan out to.
            assert set(entry["ready_capabilities"]) == supported, model_id
            assert entry["chat_ready"] == ("chat" in supported), model_id


def test_non_runnable_models_are_reported_as_not_servable(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        client.post("/v1/models/scan", json={})
        body = client.get("/v1/models").json()

    manifests = {item["model_id"]: item for item in body["items"]}
    for entry in body["capability_availability"]:
        if manifests[entry["model_id"]]["conversion_status"] != "runnable":
            assert entry["servable"] is False
            assert entry["chat_ready"] is False
            assert entry["ready_capabilities"] == []
            assert "runnable" in entry["reason"]
