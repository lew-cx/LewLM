"""HTTP contract coverage for the shared-runtime baseline."""

from __future__ import annotations

import json
import time

from conftest import FakeLlamaCppRuntime
from fastapi.testclient import TestClient
from pydantic import SecretStr

from lewlm.api.app import create_app
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import RuntimeAffinity


def test_clients_share_runtime_identity_and_one_residency(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as rag_client, TestClient(app_with_fake_runtime) as document_client:
        scan = rag_client.post("/v1/models/scan", json={})
        model_id = next(
            item["model_id"]
            for item in scan.json()["manifests"]
            if item["format_type"] == "gguf"
        )

        rag_runtime = rag_client.get(
            "/v1/runtime",
            headers={"x-lewlm-application-id": "rag-chat"},
        )
        document_runtime = document_client.get(
            "/v1/runtime",
            headers={"x-lewlm-application-id": "document-generator"},
        )
        first_warm = rag_client.post(
            f"/v1/models/{model_id}/warm",
            headers={"x-lewlm-application-id": "rag-chat"},
        )
        second_warm = document_client.post(
            f"/v1/models/{model_id}/warm",
            headers={"x-lewlm-application-id": "document-generator"},
        )
        residencies = document_client.get("/v1/runtime/residencies")

        assert rag_runtime.status_code == 200
        assert document_runtime.status_code == 200
        assert rag_runtime.json()["runtime_instance_id"] == document_runtime.json()["runtime_instance_id"]
        assert first_warm.status_code == 200
        assert first_warm.json()["backend_operation_performed"] is True
        assert second_warm.status_code == 200
        assert second_warm.json()["backend_operation_performed"] is False
        assert second_warm.json()["runtime_instance_id"] == rag_runtime.json()["runtime_instance_id"]
        assert len(residencies.json()) == 1
        assert residencies.json()[0]["model_id"] == model_id
        assert residencies.json()[0]["load_attempt_count"] == 1


def test_application_usage_metrics_attribute_shared_model_requests(
    app_with_fake_runtime,
    services_with_fake_runtime,
) -> None:
    with TestClient(app_with_fake_runtime) as client:
        scan = client.post("/v1/models/scan", json={})
        model_id = next(
            item["model_id"]
            for item in scan.json()["manifests"]
            if item["format_type"] == "gguf"
        )
        for application_id in ("rag-chat", "document-generator"):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": f"request from {application_id}"}],
                },
                headers={"x-lewlm-application-id": application_id},
            )
            assert response.status_code == 200

    applications = {
        item["application_id"]: item
        for item in services_with_fake_runtime.runtime_metrics_recorder.snapshot()["applications"]
    }
    assert applications["rag-chat"]["request_count"] == 1
    assert applications["rag-chat"]["success_count"] == 1
    assert applications["rag-chat"]["lease_acquisition_count"] == 1
    assert applications["rag-chat"]["model_usage_counts"] == {model_id: 1}
    assert applications["document-generator"]["request_count"] == 1
    assert applications["document-generator"]["success_count"] == 1
    assert applications["document-generator"]["lease_acquisition_count"] == 1


def test_lifecycle_unload_is_idempotent(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        scan = client.post("/v1/models/scan", json={})
        model_id = next(
            item["model_id"]
            for item in scan.json()["manifests"]
            if item["format_type"] == "gguf"
        )
        assert client.post(f"/v1/models/{model_id}/warm").status_code == 200
        first = client.post(f"/v1/models/{model_id}/unload")
        second = client.post(f"/v1/models/{model_id}/unload")

        assert first.status_code == 200
        assert first.json()["backend_operation_performed"] is True
        assert second.status_code == 200
        assert second.json()["backend_operation_performed"] is False
        assert client.get("/v1/runtime/residencies").json() == []


def test_async_drain_operation_is_pollable_and_idempotent(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        scan = client.post("/v1/models/scan", json={})
        model_id = next(
            item["model_id"]
            for item in scan.json()["manifests"]
            if item["format_type"] == "gguf"
        )
        assert client.post(f"/v1/models/{model_id}/warm").status_code == 200

        created = client.post(
            f"/v1/models/{model_id}/drain-operations",
            json={"timeout_seconds": 2, "idempotency_key": "release-model-a"},
            headers={"x-lewlm-application-id": "runtime-operator"},
        )
        replay = client.post(
            f"/v1/models/{model_id}/drain-operations",
            json={"timeout_seconds": 2, "idempotency_key": "release-model-a"},
            headers={"x-lewlm-application-id": "runtime-operator"},
        )
        conflicting_replay = client.post(
            f"/v1/models/{model_id}/drain-operations",
            json={"timeout_seconds": 3, "idempotency_key": "release-model-a"},
            headers={"x-lewlm-application-id": "runtime-operator"},
        )

        assert created.status_code == 202
        assert replay.status_code == 202
        assert replay.json()["operation_id"] == created.json()["operation_id"]
        assert replay.json()["idempotent_replay"] is True
        assert conflicting_replay.status_code == 409

        operation_id = created.json()["operation_id"]
        deadline = time.monotonic() + 2
        while True:
            polled = client.get(f"/v1/model-lifecycle/operations/{operation_id}")
            assert polled.status_code == 200
            if polled.json()["status"] in {"succeeded", "failed", "cancelled"}:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)

        assert polled.json()["status"] == "succeeded"
        assert polled.json()["result"]["backend_operation_performed"] is True
        assert client.get("/v1/runtime/residencies").json() == []


def test_lifecycle_routes_support_non_chat_models(app_with_fake_multimodal_runtime) -> None:
    with TestClient(app_with_fake_multimodal_runtime) as client:
        scan = client.post("/v1/models/scan", json={})
        audio_model_id = next(
            item["model_id"]
            for item in scan.json()["manifests"]
            if item["format_type"] == "audio_folder"
        )

        warm = client.post(f"/v1/models/{audio_model_id}/warm")
        residency = client.get(f"/v1/models/{audio_model_id}/residency")
        unload = client.post(f"/v1/models/{audio_model_id}/unload")

        assert warm.status_code == 200
        assert warm.json()["runtime"] == "fake_mlx_audio"
        assert residency.status_code == 200
        assert residency.json()["state"] == "ready"
        assert unload.status_code == 200
        assert unload.json()["backend_operation_performed"] is True


def test_lifecycle_actions_require_explicit_authorization_and_keep_audits_content_free(
    app_with_authorized_runtime_and_conversion,
    tool_authorized_settings,
) -> None:
    sensitive_prompt = "sensitive prompt must not enter lifecycle audit"
    sensitive_document = "private document text must not enter lifecycle audit"
    with TestClient(app_with_authorized_runtime_and_conversion) as client:
        scan = client.post("/v1/models/scan", json={})
        model_id = next(
            item["model_id"]
            for item in scan.json()["manifests"]
            if item["format_type"] == "gguf"
        )
        denied = client.post(f"/v1/models/{model_id}/warm")
        allowed_warm = client.post(
            f"/v1/models/{model_id}/warm",
            headers={
                "x-lewlm-application-id": "runtime-operator",
                "x-lewlm-authorized-actions": "model_warm",
                "x-test-prompt": sensitive_prompt,
                "x-test-document": sensitive_document,
            },
        )
        allowed_drain = client.post(
            f"/v1/models/{model_id}/drain",
            headers={"x-lewlm-authorized-actions": "model_drain"},
        )
        allowed_warm_again = client.post(
            f"/v1/models/{model_id}/warm",
            headers={"x-lewlm-authorized-actions": "model_warm"},
        )
        allowed_unload = client.post(
            f"/v1/models/{model_id}/unload",
            headers={"x-lewlm-authorized-actions": "model_unload"},
        )

    assert denied.status_code == 403
    assert allowed_warm.status_code == 200
    assert allowed_drain.status_code == 200
    assert allowed_warm_again.status_code == 200
    assert allowed_unload.status_code == 200
    audit_text = tool_authorized_settings.audit_log_path.read_text(encoding="utf-8")
    audit_events = [json.loads(line) for line in audit_text.splitlines() if line.strip()]
    assert sensitive_prompt not in audit_text
    assert sensitive_document not in audit_text
    assert any(event["action"] == "model_warm" and event["outcome"] == "denied" for event in audit_events)
    assert any(event["action"] == "model_warm" and event["outcome"] == "authorized" for event in audit_events)
    assert any(event["action"] == "model_drain" and event["outcome"] == "authorized" for event in audit_events)
    assert any(event["action"] == "model_unload" and event["outcome"] == "authorized" for event in audit_events)


def test_scoped_lifecycle_credentials_separate_operator_and_administrator(
    temp_settings,
    sample_models_root,
) -> None:
    settings = temp_settings.with_updates(
        api_key_required=True,
        audit_log_enabled=True,
        lifecycle_operator_api_keys=(SecretStr("operator-key"),),
        lifecycle_administrator_api_keys=(SecretStr("administrator-key"),),
    )
    services = bootstrap_services(
        settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime()},
    )
    app = create_app(services=services)
    operator_headers = {
        "x-api-key": "operator-key",
        "x-lewlm-application-id": "application-alpha",
        "x-lewlm-client-instance-id": "alpha-1",
    }
    administrator_headers = {
        "x-api-key": "administrator-key",
        "x-lewlm-application-id": "application-beta",
        "x-lewlm-client-instance-id": "beta-1",
    }
    try:
        with TestClient(app) as client:
            scan = client.post("/v1/models/scan", json={}, headers=operator_headers)
            model_id = next(
                item["model_id"]
                for item in scan.json()["manifests"]
                if item["format_type"] == "gguf"
            )
            warm = client.post(f"/v1/models/{model_id}/warm", headers=operator_headers)
            inspect = client.get(f"/v1/models/{model_id}/residency", headers=operator_headers)
            denied_unload = client.post(f"/v1/models/{model_id}/unload", headers=operator_headers)
            generic_key_denied = client.get(
                f"/v1/models/{model_id}/residency",
                headers={"x-api-key": "test-key", "x-lewlm-application-id": "ordinary-application"},
            )
            admin_unload = client.post(f"/v1/models/{model_id}/unload", headers=administrator_headers)

        assert warm.status_code == 200
        assert inspect.status_code == 200
        assert denied_unload.status_code == 403
        assert denied_unload.json()["error"]["details"]["required_role"] == "administrator"
        assert generic_key_denied.status_code == 403
        assert admin_unload.status_code == 200

        audit_events = [
            json.loads(line)
            for line in settings.audit_log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        denied = next(
            event
            for event in audit_events
            if event["action"] == "model_unload" and event["outcome"] == "denied"
        )
        authorized = next(
            event
            for event in audit_events
            if event["action"] == "model_unload" and event["outcome"] == "authorized"
        )
        assert denied["details"]["application_id"] == "application-alpha"
        assert authorized["details"]["application_id"] == "application-beta"
        assert authorized["details"]["granted_role"] == "administrator"
    finally:
        services.close()
