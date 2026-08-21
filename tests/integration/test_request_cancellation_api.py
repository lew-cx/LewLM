"""HTTP coverage for the request-cancellation handle.

The scenario throughout is DocKtizo's: the process that wants to cancel is not
the process holding the request, so it can only address it by the handle it
supplied when the request was issued.
"""

from __future__ import annotations

import asyncio
from threading import Event, Thread

import pytest
from conftest import FakeLlamaCppRuntime
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse
from pydantic import SecretStr

from lewlm import LewLM
from lewlm.api.app import create_app
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import GenerateMessage, GenerateRequest, GenerateResponse, ModelManifest, RuntimeAffinity
from lewlm.core.errors import RequestCancelledError
from lewlm.runtime.cancellation import request_cancelled


@pytest.fixture()
def client(app_with_fake_runtime):
    with TestClient(app_with_fake_runtime) as test_client:
        yield test_client


ORCHESTRATOR = {"x-lewlm-application-id": "docktizo"}


def _generate_document(client: TestClient, *, handle: str) -> object:
    return client.post(
        "/v1/documents/generate",
        headers={**ORCHESTRATOR, "x-request-id": handle},
        json={
            "output_format": "markdown",
            "document": {
                "title": "Status report",
                "sections": [{"heading": "Intro", "blocks": [{"type": "paragraph", "text": "Body."}]}],
            },
        },
    )


def test_cancelling_an_unseen_handle_records_an_intent(client: TestClient) -> None:
    response = client.post("/v1/requests/req-unseen/cancel", headers=ORCHESTRATOR)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["request_id"] == "req-unseen"
    assert payload["state"] == "pending"
    assert payload["application_id"] == "docktizo"
    assert payload["runtime_instance_id"]
    assert payload["expires_at"] is not None


def test_a_document_request_cancelled_before_it_arrives_is_refused(client: TestClient) -> None:
    client.post("/v1/requests/req-doc/cancel", headers=ORCHESTRATOR)

    generated = _generate_document(client, handle="req-doc")

    assert generated.status_code == 499, generated.text
    error = generated.json()["error"]
    assert error["code"] == "request_cancelled"
    assert error["details"]["request_id"] == "req-doc"
    assert error["details"]["stage"] == "tool.documents.generate"
    assert generated.headers["x-request-id"] == "req-doc"


def test_the_acknowledgement_reports_that_the_request_stopped(client: TestClient) -> None:
    client.post("/v1/requests/req-ack/cancel", headers=ORCHESTRATOR)
    _generate_document(client, handle="req-ack")

    acknowledged = client.post("/v1/requests/req-ack/cancel", headers=ORCHESTRATOR)

    assert acknowledged.json()["state"] == "cancelled"
    assert acknowledged.json()["completed_at"] is not None


def test_a_request_that_completed_first_is_reported_as_completed(client: TestClient) -> None:
    generated = _generate_document(client, handle="req-done")
    assert generated.status_code == 200, generated.text
    assert generated.json()["request_id"] == "req-done"
    assert generated.json()["metadata"]["request_id"] == "req-done"

    acknowledged = client.post("/v1/requests/req-done/cancel", headers=ORCHESTRATOR)

    assert acknowledged.json()["state"] == "completed"


def test_responses_preserves_the_callers_request_handle(client: TestClient) -> None:
    manifests = client.post("/v1/models/scan", json={}).json()["manifests"]
    model_id = next(item["model_id"] for item in manifests if item["format_type"] == "gguf")
    generated = client.post(
        "/v1/responses",
        headers={**ORCHESTRATOR, "x-request-id": "req-response"},
        json={"model": model_id, "input": "Return a short status.", "max_output_tokens": 16},
    )

    assert generated.status_code == 200, generated.text
    assert generated.headers["x-request-id"] == "req-response"
    assert generated.json()["id"] == "req-response"
    assert generated.json()["metadata"]["request_id"] == "req-response"


def test_application_id_is_observability_metadata_not_authorization(client: TestClient) -> None:
    client.post("/v1/requests/req-owned/cancel", headers=ORCHESTRATOR)

    relabelled = client.post(
        "/v1/requests/req-owned/cancel",
        headers={"x-lewlm-application-id": "someone-else"},
    )

    # Open deployments have no authenticated owner to compare. The application
    # header is retained for tracing, but possession of the handle is what lets
    # a caller address it.
    assert relabelled.status_code == 200
    assert relabelled.json()["state"] == "pending"
    assert _generate_document(client, handle="req-owned").status_code == 499


def test_a_duplicate_active_handle_is_reported_as_a_conflict(client: TestClient) -> None:
    registry = client.app.state.services.request_cancellation_registry

    with registry.track("req-duplicate", application_id="docktizo") as original:
        duplicate = client.get(
            "/v1/models",
            headers={**ORCHESTRATOR, "x-request-id": "req-duplicate"},
        )

        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "request_handle_conflict"
        assert original.cancelled is False


def test_an_invalid_request_handle_is_refused_before_work_starts(client: TestClient) -> None:
    response = client.get("/v1/models", headers={"x-request-id": "not/path-safe"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.headers["x-request-id"] != "not/path-safe"


def test_a_stream_handle_stays_active_until_its_body_closes(temp_settings) -> None:
    app = create_app(temp_settings)
    body_started = Event()
    result: dict[str, object] = {}

    @app.get("/test/cancellable-stream")
    async def cancellable_stream() -> StreamingResponse:
        async def body():
            body_started.set()
            while not request_cancelled():
                await asyncio.sleep(0.005)
            if False:
                yield b""

        return StreamingResponse(body())

    with TestClient(app) as stream_client:
        def consume() -> None:
            try:
                with stream_client.stream(
                    "GET",
                    "/test/cancellable-stream",
                    headers={"x-request-id": "req-stream-live"},
                ) as response:
                    result["status_code"] = response.status_code
                    result["body"] = b"".join(response.iter_bytes())
            except BaseException as exc:  # surfaced on the test thread below
                result["error"] = exc

        consumer = Thread(target=consume, daemon=True)
        consumer.start()
        assert body_started.wait(timeout=2)

        registry = stream_client.app.state.services.request_cancellation_registry
        assert registry.active_request_count == 1
        duplicate = stream_client.get(
            "/v1/models",
            headers={"x-request-id": "req-stream-live"},
        )
        assert duplicate.status_code == 409

        cancelled = stream_client.post("/v1/requests/req-stream-live/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["state"] == "cancelling"

        consumer.join(timeout=2)
        assert not consumer.is_alive()
        assert "error" not in result
        assert result == {"status_code": 200, "body": b""}
        assert stream_client.post("/v1/requests/req-stream-live/cancel").json()["state"] == "cancelled"


def test_authenticated_handles_are_bound_to_the_api_credential(temp_settings) -> None:
    settings = temp_settings.with_updates(
        api_key_required=True,
        api_keys=(SecretStr("owner-key"), SecretStr("other-key")),
    )

    with TestClient(create_app(settings)) as secured_client:
        registry = secured_client.app.state.services.request_cancellation_registry
        with registry.track(
            "req-secure",
            application_id="docktizo",
            credential="owner-key",
        ) as token:
            refused = secured_client.post(
                "/v1/requests/req-secure/cancel",
                headers={
                    "x-api-key": "other-key",
                    # Matching this untrusted label must not grant authority.
                    "x-lewlm-application-id": "docktizo",
                },
            )
            accepted = secured_client.post(
                "/v1/requests/req-secure/cancel",
                headers={
                    "x-api-key": "owner-key",
                    # Conversely, changing the label must not remove authority.
                    "x-lewlm-application-id": "renamed-app",
                },
            )

        assert refused.status_code == 403
        assert refused.json()["error"]["code"] == "tool_authorization_error"
        assert accepted.status_code == 200
        assert accepted.json()["state"] == "cancelling"
        assert token.cancelled is True


def test_an_embedded_host_can_cancel_an_authenticated_request_before_it_arrives(temp_settings) -> None:
    settings = temp_settings.with_updates(
        api_key_required=True,
        api_keys=(SecretStr("owner-key"), SecretStr("other-key")),
    )

    with TestClient(create_app(settings)) as secured_client:
        embedded_client = LewLM(services=secured_client.app.state.services).app_client()
        pending = embedded_client.cancel_request("req-embedded-future")

        generated = secured_client.post(
            "/v1/documents/generate",
            headers={
                "x-api-key": "owner-key",
                "x-request-id": "req-embedded-future",
            },
            json={
                "output_format": "markdown",
                "document": {"title": "Cancelled before arrival", "sections": []},
            },
        )
        refused = secured_client.post(
            "/v1/requests/req-embedded-future/cancel",
            headers={"x-api-key": "other-key"},
        )
        acknowledged = secured_client.post(
            "/v1/requests/req-embedded-future/cancel",
            headers={"x-api-key": "owner-key"},
        )

        assert pending.state.value == "pending"
        assert generated.status_code == 499, generated.text
        assert refused.status_code == 403
        assert acknowledged.json()["state"] == "cancelled"


def test_reusing_one_handle_for_the_cancel_call_does_not_shadow_the_target(client: TestClient) -> None:
    # The cancel endpoint is not itself tracked, so a caller that stamps the
    # same handle on both calls still cancels the work it meant to.
    cancelled = client.post(
        "/v1/requests/req-shared/cancel",
        headers={**ORCHESTRATOR, "x-request-id": "req-shared"},
    )

    assert cancelled.json()["state"] == "pending"
    assert _generate_document(client, handle="req-shared").status_code == 499


def test_an_uncancelled_request_is_unaffected(client: TestClient) -> None:
    client.post("/v1/requests/req-other/cancel", headers=ORCHESTRATOR)

    generated = _generate_document(client, handle="req-live")

    assert generated.status_code == 200, generated.text


# --- cancellation of work already accepted -----------------------------------


async def _wait_until(condition, *, timeout_seconds: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("Timed out waiting for the scheduler to queue the second request.")
        await asyncio.sleep(0.005)


class SlowFakeLlamaCppRuntime(FakeLlamaCppRuntime):
    async def _load_model(self, manifest: ModelManifest) -> None:
        await asyncio.sleep(0.01)

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        # Long enough that the second request is provably still queued when the
        # cancellation lands, rather than racing the first request's completion.
        await asyncio.sleep(0.5)
        return await super()._generate(request)


async def test_a_generation_cancelled_while_queued_never_reaches_the_model(
    temp_settings,
    sample_models_root,
) -> None:
    services = bootstrap_services(
        temp_settings.with_updates(
            max_concurrent_runtime_requests=1,
            runtime_request_queue_limit=4,
            # Without batching, each request holds its own admission, which is
            # where a queued request is cancellable individually.
            continuous_batch_max_batch_size=1,
        ),
        runtime_overrides={RuntimeAffinity.LLAMACPP: SlowFakeLlamaCppRuntime()},
    )
    try:
        model_id = next(
            manifest.model_id
            for manifest in services.model_registry.scan().manifests
            if manifest.format_type.value == "gguf"
        )
        registry = services.request_cancellation_registry
        # Warm first so the two requests contend on runtime admission only, not
        # on the separate model-load scheduler.
        await services.model_router.warm_model_lifecycle(model_id)

        async def generate(handle: str) -> str:
            with registry.track(handle, application_id="docktizo"):
                execution = await services.chat_orchestrator.complete(
                    model_id=model_id,
                    messages=[GenerateMessage(role="user", content=handle)],
                    max_tokens=32,
                    temperature=0.0,
                )
                return execution.response.output_text

        first = asyncio.create_task(generate("req-first"))
        queued = asyncio.create_task(generate("req-queued"))
        await _wait_until(lambda: services.runtime_request_scheduler.snapshot()["queued_requests"] > 0)

        record = registry.cancel("req-queued", application_id="docktizo")
        with pytest.raises(RequestCancelledError):
            await queued

        assert record.state.value == "cancelling"
        # The request that was already admitted is unaffected.
        assert await first
        assert registry.cancel("req-queued", application_id="docktizo").state.value == "cancelled"
        assert services.runtime_request_scheduler.snapshot()["active_requests"] == 0
    finally:
        await services.aclose()
