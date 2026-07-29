"""HTTP-level coverage for the host-application integration surfaces."""

from __future__ import annotations

import base64
import hashlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(app_with_fake_runtime):
    with TestClient(app_with_fake_runtime) as test_client:
        test_client.post("/v1/models/scan", json={})
        yield test_client


# --- uploaded-source ingestion over HTTP -------------------------------------


def _upload_payload(source_id: str, body: bytes, *, file_name: str = "notes.md", **extra) -> dict:
    return {
        "source_id": source_id,
        "file_name": file_name,
        "content_base64": base64.b64encode(body).decode("ascii"),
        **extra,
    }


def test_documents_can_be_ingested_as_uploaded_bytes(client) -> None:
    body = b"# Uploaded\n\nThis never touched a shared mount.\n"
    response = client.post(
        "/v1/documents/ingest",
        json={
            "sources": [
                _upload_payload(
                    "caller-owned-1",
                    body,
                    media_type="text/markdown",
                    expected_sha256=hashlib.sha256(body).hexdigest(),
                ),
            ],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ingested_count"] == 1
    assert payload["failed_count"] == 0
    assert payload["partial"] is False
    assert [source["source_id"] for source in payload["sources"]] == ["caller-owned-1"]
    # No server-local filesystem detail may reach a remote caller.
    assert payload["sources"][0]["path"] is None
    assert all(chunk["source_path"] is None for chunk in payload["chunks"])


def test_ingest_without_any_source_is_rejected(client) -> None:
    response = client.post("/v1/documents/ingest", json={})
    assert response.status_code == 422


def test_partial_ingestion_reports_one_outcome_per_requested_source(client) -> None:
    response = client.post(
        "/v1/documents/ingest",
        json={
            "sources": [
                _upload_payload("good", b"# Good\n\nParses.\n"),
                _upload_payload("bad-digest", b"# Bad\n\nBody.\n", expected_sha256="0" * 64),
                _upload_payload("empty", b""),
            ],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["partial"] is True
    assert payload["ingested_count"] == 1
    assert payload["failed_count"] == 2

    outcomes = {item["source_id"]: item for item in payload["source_results"]}
    assert set(outcomes) == {"good", "bad-digest", "empty"}
    assert outcomes["good"]["status"] == "ingested"
    assert outcomes["good"]["chunk_count"] > 0
    assert outcomes["bad-digest"]["error_code"] == "checksum_mismatch"
    assert outcomes["empty"]["error_code"] == "empty_source"
    # Every failure states whether retrying unchanged could help.
    assert all("retryable" in item for item in payload["source_results"])


def test_invalid_base64_upload_is_rejected_with_a_typed_error(client) -> None:
    response = client.post(
        "/v1/documents/ingest",
        json={"sources": [{"source_id": "s1", "file_name": "a.md", "content_base64": "not base64!!"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["details"]["source_id"] == "s1"


# --- component provenance ----------------------------------------------------


def test_ingest_response_names_the_components_that_ran(client) -> None:
    response = client.post(
        "/v1/documents/ingest",
        json={"sources": [_upload_payload("s1", b"# Title\n\nBody.\n")]},
    )
    payload = response.json()

    components = payload["metadata"]["components"]
    assert components, "expected component provenance on the envelope"
    kinds = {item["kind"] for item in components}
    assert "parser" in kinds
    assert "chunker" in kinds
    for item in components:
        assert item["name"]
        assert item["version"]
    # The envelope version must remain distinguishable from component versions.
    assert payload["metadata"]["version"] == "v1"


def test_generated_document_names_its_renderer(client) -> None:
    response = client.post(
        "/v1/documents/generate",
        json={
            "output_format": "markdown",
            "document": {
                "title": "Report",
                "sections": [{"heading": "Intro", "blocks": [{"type": "paragraph", "text": "Body."}]}],
            },
        },
    )

    assert response.status_code == 200, response.text
    components = response.json()["metadata"]["components"]
    renderer = next(item for item in components if item["kind"] == "renderer")
    assert renderer["name"] == "markdown_renderer"
    assert renderer["version"]


# --- correlation identifiers -------------------------------------------------


def test_correlation_id_header_is_echoed_and_reaches_metadata(client) -> None:
    response = client.post(
        "/v1/documents/ingest",
        json={"sources": [_upload_payload("s1", b"# Title\n\nBody.\n")]},
        headers={"x-lewlm-correlation-id": "caller-trace-1"},
    )

    assert response.headers["x-lewlm-correlation-id"] == "caller-trace-1"
    assert response.json()["metadata"]["correlation_id"] == "caller-trace-1"


def test_correlation_id_can_be_supplied_in_the_request_body(client) -> None:
    response = client.post(
        "/v1/documents/ingest",
        json={
            "sources": [_upload_payload("s1", b"# Title\n\nBody.\n")],
            "correlation_id": "body-trace-1",
        },
    )
    assert response.json()["metadata"]["correlation_id"] == "body-trace-1"


def test_absent_correlation_id_stays_absent(client) -> None:
    response = client.post(
        "/v1/documents/ingest",
        json={"sources": [_upload_payload("s1", b"# Title\n\nBody.\n")]},
    )
    # LewLM never invents one, so a caller can tell its own ID from a request ID.
    assert response.json()["metadata"]["correlation_id"] is None
    assert "x-lewlm-correlation-id" not in response.headers


def test_chat_completion_carries_the_correlation_id(client) -> None:
    models = client.get("/v1/models").json()["items"]
    model_id = models[0]["model_id"]
    response = client.post(
        "/v1/chat/completions",
        json={"model": model_id, "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-lewlm-correlation-id": "chat-trace-1"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["correlation_id"] == "chat-trace-1"


def test_overlong_correlation_id_is_bounded(client) -> None:
    response = client.post(
        "/v1/documents/ingest",
        json={"sources": [_upload_payload("s1", b"# Title\n\nBody.\n")]},
        headers={"x-lewlm-correlation-id": "x" * 500},
    )
    assert len(response.json()["metadata"]["correlation_id"]) == 128


# --- tokenizer-aware counting ------------------------------------------------


def test_token_count_uses_the_model_tokenizer(client) -> None:
    models = client.get("/v1/models").json()["items"]
    model_id = models[0]["model_id"]
    text = "LewLM counts tokens exactly."

    response = client.post("/v1/tokenize/count", json={"model": model_id, "text": text})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["token_count"] > 0
    assert payload["character_count"] == len(text)
    assert payload["truncated"] is False
    assert payload["truncated_text"] is None
    tokenizer = next(item for item in payload["metadata"]["components"] if item["kind"] == "tokenizer")
    assert tokenizer["version"]


def test_token_count_returns_a_deterministic_truncation_boundary(client) -> None:
    models = client.get("/v1/models").json()["items"]
    model_id = models[0]["model_id"]
    text = "This sentence is long enough to be cut at an exact token boundary."

    response = client.post(
        "/v1/tokenize/count",
        json={"model": model_id, "text": text, "max_tokens": 5},
    )

    payload = response.json()
    assert payload["truncated"] is True
    assert payload["truncated_token_count"] == 5
    assert payload["truncated_text"]
    assert text.startswith(payload["truncated_text"])

    # The boundary must be reproducible: re-counting the truncated text
    # returns exactly the requested budget.
    recount = client.post(
        "/v1/tokenize/count",
        json={"model": model_id, "text": payload["truncated_text"]},
    ).json()
    assert recount["token_count"] == 5


def test_token_count_does_not_truncate_when_under_budget(client) -> None:
    models = client.get("/v1/models").json()["items"]
    response = client.post(
        "/v1/tokenize/count",
        json={"model": models[0]["model_id"], "text": "hi", "max_tokens": 4096},
    )
    payload = response.json()
    assert payload["truncated"] is False
    assert payload["truncated_text"] is None


def test_token_count_rejects_a_non_positive_budget(client) -> None:
    response = client.post("/v1/tokenize/count", json={"text": "hi", "max_tokens": 0})
    assert response.status_code == 422


# --- runtime build identity --------------------------------------------------


def test_runtime_identity_exposes_build_provenance(client) -> None:
    payload = client.get("/v1/runtime").json()

    build = payload["build"]
    assert build["package_version"]
    assert build["api_schema_version"] == "v1"
    assert build["install_kind"] in {"editable", "installed", "unknown"}
    assert "source_commit" in build
    assert "distribution_digest" in build
    assert isinstance(build["release_build"], bool)


# --- versioned retrieval scoring policy --------------------------------------


@pytest.fixture()
def semantic_client(app_with_fake_multimodal_runtime):
    with TestClient(app_with_fake_multimodal_runtime) as test_client:
        test_client.post("/v1/models/scan", json={})
        yield test_client


def _semantic_model_ids(client) -> tuple[str, str]:
    manifests = client.get("/v1/models").json()["items"]
    embedding_model = next(
        item["model_id"] for item in manifests if item["display_name"] == "e5-small-embed-mlx"
    )
    rerank_model = next(
        item["model_id"] for item in manifests if "rerank" in item["display_name"]
    )
    return embedding_model, rerank_model


def test_retrieval_response_names_and_versions_its_scoring_policy(semantic_client) -> None:
    client = semantic_client
    embedding_model, rerank_model = _semantic_model_ids(client)
    response = client.post(
        "/v1/retrieval/context",
        json={
            "embedding_model": embedding_model,
            "rerank_model": rerank_model,
            "query": "typed helpers",
            "candidate_sources": [
                {
                    "source_id": "source-1",
                    "source_type": "markdown",
                    "source_name": "notes.md",
                    "source_label": "notes.md",
                },
            ],
            "candidate_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "text": "LewLM exposes typed helper methods.",
                    "source_id": "source-1",
                    "section_id": "section-1",
                    "source_label": "notes.md",
                    "section_label": "notes.md / 1",
                },
                {
                    "chunk_id": "chunk-2",
                    "text": "Unrelated note about weather.",
                    "source_id": "source-1",
                    "section_id": "section-2",
                    "source_label": "notes.md",
                    "section_label": "notes.md / 2",
                },
            ],
            "top_k": 1,
        },
    )

    assert response.status_code == 200, response.text
    policy = response.json()["scoring_policy"]
    assert policy["name"] == "rerank_primary_embedding_tiebreak"
    assert policy["version"]
    assert policy["primary_signal"] in {"rerank", "embedding"}
    assert policy["final_tie_break"] == "original_order"
    assert policy["missing_score_behaviour"] == "rejected"


def test_retrieval_rejects_duplicate_candidate_chunks(semantic_client) -> None:
    client = semantic_client
    duplicate = {
        "chunk_id": "chunk-1",
        "text": "Same identity twice.",
        "source_id": "source-1",
        "section_id": "section-1",
        "source_label": "notes.md",
        "section_label": "notes.md / 1",
    }
    embedding_model, rerank_model = _semantic_model_ids(client)
    response = client.post(
        "/v1/retrieval/context",
        json={
            "embedding_model": embedding_model,
            "rerank_model": rerank_model,
            "query": "q",
            "candidate_chunks": [duplicate, dict(duplicate)],
            "top_k": 1,
        },
    )
    assert response.status_code == 400
    assert "unique chunk identities" in response.json()["error"]["message"]
