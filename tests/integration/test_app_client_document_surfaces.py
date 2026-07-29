"""The typed client must cover rendering, not stop at ingestion."""

from __future__ import annotations

import base64
import hashlib

import pytest

from lewlm import LewLM, LewLMAppClient
from lewlm.api.schemas.documents import DocumentGenerateRequest
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.documents.skills.models import ContractTextReplacementInput, ContractTextReplacementRequest


@pytest.fixture()
def app_client(services_with_fake_runtime) -> LewLMAppClient:
    return LewLM(services=services_with_fake_runtime).app_client()


def _document() -> DocumentIR:
    return DocumentIR.model_validate(
        {
            "title": "Quarterly Report",
            "sections": [
                {"heading": "Summary", "blocks": [{"type": "paragraph", "text": "Revenue grew."}]},
            ],
        },
    )


def test_client_renders_a_document_without_a_second_http_transport(app_client) -> None:
    response = app_client.generate_document(
        document=_document(),
        output_format=DocumentOutputFormat.MARKDOWN,
    )

    assert response.file_name.endswith(".md")
    assert response.size_bytes > 0
    # Decoding is part of the typed surface, so callers never hand-roll base64.
    content = LewLMAppClient.document_bytes(response)
    assert b"Quarterly Report" in content
    assert len(content) == response.size_bytes


def test_client_accepts_a_typed_generate_request(app_client) -> None:
    response = app_client.generate_document(
        DocumentGenerateRequest(
            output_format=DocumentOutputFormat.JSON,
            document=_document(),
            file_name="report.json",
            correlation_id="render-trace-1",
        ),
    )
    assert response.file_name == "report.json"
    assert response.metadata.correlation_id == "render-trace-1"


def test_generated_document_carries_renderer_provenance(app_client) -> None:
    response = app_client.generate_document(document=_document(), output_format="markdown")
    renderer = next(item for item in response.metadata.components if item.kind.value == "renderer")
    assert renderer.name == "markdown_renderer"
    assert renderer.version


def test_client_runs_a_document_skill(app_client) -> None:
    response = app_client.transform_document(
        ContractTextReplacementRequest(
            output_format=DocumentOutputFormat.MARKDOWN,
            input=ContractTextReplacementInput(
                title="Agreement",
                template_text="This agreement is with {{party}}.",
                replacements={"party": "Acme"},
            ),
        ),
    )

    assert response.skill == "contract_text_replacement"
    assert b"Acme" in LewLMAppClient.document_bytes(response)


def test_generate_document_rejects_mixed_argument_styles(app_client) -> None:
    with pytest.raises(ValueError, match="not both"):
        app_client.generate_document(
            DocumentGenerateRequest(output_format=DocumentOutputFormat.JSON, document=_document()),
            file_name="other.json",
        )


def test_generate_document_requires_a_document(app_client) -> None:
    with pytest.raises(ValueError, match="document and output_format are required"):
        app_client.generate_document()


def test_client_ingests_uploaded_bytes(app_client) -> None:
    body = b"# Uploaded\n\nNo shared mount needed.\n"
    source = app_client.upload_source("caller-1", body, file_name="notes.md", media_type="text/markdown")

    assert source.expected_sha256 == hashlib.sha256(body).hexdigest()
    assert base64.b64decode(source.content_base64) == body

    response = app_client.ingest_documents(sources=[source])
    assert [item.source_id for item in response.sources] == ["caller-1"]
    assert response.sources[0].path is None
    assert response.source_results[0].status == "ingested"


def test_client_reports_partial_ingestion(app_client) -> None:
    good = app_client.upload_source("good", b"# Good\n\nBody.\n", file_name="good.md")
    bad = app_client.upload_source(
        "bad",
        b"# Bad\n\nBody.\n",
        file_name="bad.md",
        expected_sha256="0" * 64,
        verify=False,
    )

    response = app_client.ingest_documents(sources=[good, bad])
    assert response.partial is True
    outcomes = {item.source_id: item for item in response.source_results}
    assert outcomes["bad"].error_code.value == "checksum_mismatch"


def test_client_counts_tokens_with_the_model_tokenizer(app_client, services_with_fake_runtime) -> None:
    services_with_fake_runtime.model_registry.scan()
    manifests = services_with_fake_runtime.model_registry.list_manifests()
    model_id = manifests[0].model_id

    result = app_client.count_tokens(text="count these tokens", model=model_id)
    assert result.token_count > 0
    assert result.character_count == len("count these tokens")


def test_client_truncates_at_an_exact_token_boundary(app_client, services_with_fake_runtime) -> None:
    services_with_fake_runtime.model_registry.scan()
    model_id = services_with_fake_runtime.model_registry.list_manifests()[0].model_id

    result = app_client.count_tokens(
        text="a longer sentence that will not fit inside the requested budget",
        model=model_id,
        max_tokens=6,
    )
    assert result.truncated is True
    assert result.truncated_token_count == 6
    recount = app_client.count_tokens(text=result.truncated_text, model=model_id)
    assert recount.token_count == 6
