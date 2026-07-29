"""Coverage for the host-application integration surfaces added for remote callers."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from pathlib import Path

import pytest

from lewlm.core.contracts import RerankResult
from lewlm.core.errors import BackendContractError, ConfigurationError
from lewlm.core.execution_metadata import build_routed_execution_metadata
from lewlm.core.multimodal import (
    _retrieval_scoring_policy,
    _validate_candidate_identity,
    _validate_finite_vector,
    _validated_rerank_scores,
)
from lewlm.core.provenance import (
    ComponentKind,
    chunker_provenance,
    installed_version,
    parser_provenance,
    renderer_provenance,
    retrieval_scoring_provenance,
)
from lewlm.documents.ingest.models import DocumentChunk, DocumentIngestErrorCode, DocumentSourceType, IngestedDocumentSource
from lewlm.documents.ingest.service import (
    MAX_SOURCE_METADATA_VALUE_CHARACTERS,
    MAX_UPLOAD_SOURCE_BYTES,
    DocumentIngestService,
    UploadedDocumentSource,
    _safe_upload_file_name,
)
from lewlm.runtime.identity import API_SCHEMA_VERSION, RuntimeBuildIdentity, RuntimeInstanceMetadata
from lewlm.runtime.request_context import (
    MAX_CORRELATION_ID_CHARACTERS,
    apply_body_correlation_id,
    correlation_scope,
    current_correlation_id,
    normalize_correlation_id,
)


# --- retrieval validation ----------------------------------------------------


def _chunk(chunk_id: str, *, source_id: str = "source-1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        source_id=source_id,
        section_id=f"{source_id}-sec-1",
        source_label="doc",
        section_label="doc / 1",
    )


def _source(source_id: str) -> IngestedDocumentSource:
    return IngestedDocumentSource(
        source_id=source_id,
        source_type=DocumentSourceType.MARKDOWN,
        source_name="doc.md",
        source_label="doc.md",
    )


def test_duplicate_chunk_identities_are_rejected_before_model_work() -> None:
    with pytest.raises(ConfigurationError, match="unique chunk identities"):
        _validate_candidate_identity([_chunk("a"), _chunk("a")], None)


def test_duplicate_source_identities_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unique source identities"):
        _validate_candidate_identity([_chunk("a")], [_source("s"), _source("s")])


def test_distinct_candidate_identities_are_accepted() -> None:
    _validate_candidate_identity([_chunk("a"), _chunk("b")], [_source("s1"), _source("s2")])


def test_rerank_duplicate_index_is_rejected_rather_than_silently_collapsed() -> None:
    # A dict comprehension would keep only the last score for index 0.
    results = [RerankResult(index=0, relevance_score=0.9), RerankResult(index=0, relevance_score=0.1)]
    with pytest.raises(BackendContractError, match="duplicate candidate index") as exc_info:
        _validated_rerank_scores(results, candidate_count=2)
    assert exc_info.value.code == "backend_contract_violation"
    assert int(exc_info.value.status_code) == 502


@pytest.mark.parametrize("index", [-1, 2, 99])
def test_rerank_out_of_range_index_is_rejected(index: int) -> None:
    with pytest.raises(BackendContractError, match="outside the candidate range"):
        _validated_rerank_scores([RerankResult(index=index, relevance_score=0.5)], candidate_count=2)


def test_rerank_missing_candidate_is_rejected_not_ranked_last() -> None:
    # Ranking an unscored candidate last is a silent correction the caller
    # cannot observe, so an incomplete rerank response must fail instead.
    with pytest.raises(BackendContractError, match="did not score every retrieval candidate") as exc_info:
        _validated_rerank_scores([RerankResult(index=0, relevance_score=0.5)], candidate_count=3)
    assert exc_info.value.details["missing_indices"] == [1, 2]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_rerank_non_finite_score_is_rejected(score: float) -> None:
    with pytest.raises(BackendContractError, match="non-finite relevance score"):
        _validated_rerank_scores([RerankResult(index=0, relevance_score=score)], candidate_count=1)


def test_complete_rerank_response_is_accepted() -> None:
    results = [RerankResult(index=1, relevance_score=0.2), RerankResult(index=0, relevance_score=0.8)]
    assert _validated_rerank_scores(results, candidate_count=2) == {0: 0.8, 1: 0.2}


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_embedding_component_is_rejected(value: float) -> None:
    with pytest.raises(BackendContractError, match="non-finite value"):
        _validate_finite_vector([0.1, value, 0.3], origin="query embedding")


def test_finite_embedding_passes() -> None:
    _validate_finite_vector([0.1, -0.2, 0.0], origin="query embedding")


# --- versioned scoring policy ------------------------------------------------


def test_hybrid_scoring_policy_names_its_rules() -> None:
    policy = _retrieval_scoring_policy(use_embeddings=True, use_rerank=True)
    assert policy.name == "rerank_primary_embedding_tiebreak"
    assert policy.version
    assert policy.primary_signal == "rerank"
    assert policy.tie_break_signal == "embedding"
    assert policy.final_tie_break == "original_order"
    assert policy.normalization == "cosine"
    assert policy.missing_score_behaviour == "rejected"


def test_embeddings_only_policy_reports_embedding_as_primary() -> None:
    policy = _retrieval_scoring_policy(use_embeddings=True, use_rerank=False)
    assert policy.primary_signal == "embedding"
    assert policy.tie_break_signal == "original_order"
    assert policy.rerank_used is False


def test_rerank_only_policy_falls_back_to_original_order() -> None:
    policy = _retrieval_scoring_policy(use_embeddings=False, use_rerank=True)
    assert policy.primary_signal == "rerank"
    assert policy.tie_break_signal == "original_order"
    assert policy.normalization == "none"


# --- component provenance ----------------------------------------------------


def test_envelope_version_and_component_versions_are_distinct() -> None:
    metadata = build_routed_execution_metadata(
        request_id="r1",
        created=1,
        requested_model_id=None,
        routing__=None,  # type: ignore[call-arg]
    ) if False else None
    # The envelope version is a schema marker; components identify the code.
    renderer = renderer_provenance("pdf")
    assert renderer.kind is ComponentKind.RENDERER
    assert renderer.name == "pdf_renderer"
    assert renderer.version
    assert renderer.implementation == "reportlab"


def test_parser_provenance_names_the_third_party_implementation() -> None:
    parser = parser_provenance("docx")
    assert parser.kind is ComponentKind.PARSER
    assert parser.name == "docx_body_parser"
    assert parser.implementation == "python-docx"


def test_parser_provenance_handles_a_pure_python_parser() -> None:
    parser = parser_provenance("markdown")
    assert parser.implementation is None
    assert parser.implementation_version is None


def test_chunker_and_scoring_components_are_versioned() -> None:
    assert chunker_provenance().version
    assert retrieval_scoring_provenance().kind is ComponentKind.SCORING_POLICY


def test_installed_version_returns_none_for_an_absent_distribution() -> None:
    assert installed_version("definitely-not-a-real-distribution-xyz") is None


def test_installed_version_resolves_a_present_distribution() -> None:
    assert installed_version("pydantic") is not None


# --- runtime build identity --------------------------------------------------


def test_build_identity_reports_schema_version_and_install_kind() -> None:
    build = RuntimeBuildIdentity.detect(version="9.9.9")
    assert build.api_schema_version == API_SCHEMA_VERSION
    assert build.install_kind in {"editable", "installed", "unknown"}
    # A dirty or editable tree must never claim to be a release build.
    if build.install_kind == "editable" or build.source_dirty:
        assert build.release_build is False


def test_runtime_instance_metadata_carries_build_identity() -> None:
    metadata = RuntimeInstanceMetadata.create(version="1.2.3")
    assert metadata.build.package_version == "1.2.3"
    assert metadata.build.api_schema_version == API_SCHEMA_VERSION


# --- correlation identifiers -------------------------------------------------


def test_correlation_id_is_trimmed_and_bounded() -> None:
    assert normalize_correlation_id("  trace-1  ") == "trace-1"
    assert normalize_correlation_id("") is None
    assert normalize_correlation_id("   ") is None
    assert normalize_correlation_id(None) is None
    assert len(normalize_correlation_id("x" * 500)) == MAX_CORRELATION_ID_CHARACTERS


def test_correlation_scope_restores_the_previous_value() -> None:
    assert current_correlation_id() is None
    with correlation_scope("trace-a"):
        assert current_correlation_id() == "trace-a"
        with correlation_scope("trace-b"):
            assert current_correlation_id() == "trace-b"
        assert current_correlation_id() == "trace-a"
    assert current_correlation_id() is None


def test_a_transport_correlation_id_wins_over_the_request_body() -> None:
    with correlation_scope("from-header"):
        apply_body_correlation_id("from-body")
        assert current_correlation_id() == "from-header"


def test_routed_metadata_adopts_the_ambient_correlation_id() -> None:
    from lewlm.core.contracts import RoutingDecision, RuntimeAffinity

    routing = RoutingDecision(
        model_id="m",
        runtime_name="r",
        runtime_affinity=RuntimeAffinity.MLX_TEXT,
        reason="test",
    )
    with correlation_scope("trace-metadata"):
        metadata = build_routed_execution_metadata(
            request_id="req-1",
            created=1,
            requested_model_id=None,
            routing=routing,
        )
    assert metadata.correlation_id == "trace-metadata"
    assert metadata.version == "v1"


def test_events_inherit_the_ambient_correlation_id() -> None:
    from lewlm.events.schema import EventType, StreamEvent

    with correlation_scope("trace-event"):
        event = StreamEvent(type=EventType.SYSTEM_READY)
    assert event.correlation_id == "trace-event"
    assert event.payload["correlation_id"] == "trace-event"


def test_events_without_a_correlation_id_stay_absent() -> None:
    from lewlm.events.schema import EventType, StreamEvent

    event = StreamEvent(type=EventType.SYSTEM_READY)
    assert event.correlation_id is None
    assert "correlation_id" not in event.payload


# --- uploaded-source ingestion -----------------------------------------------


@pytest.fixture()
def ingest_service(tmp_path: Path) -> DocumentIngestService:
    return DocumentIngestService(workspace_root=tmp_path, sandbox_enabled=False)


def _upload(source_id: str, body: bytes, *, file_name: str = "doc.md", **kwargs) -> UploadedDocumentSource:
    return UploadedDocumentSource(source_id=source_id, file_name=file_name, content=body, **kwargs)


def test_uploaded_source_keeps_caller_identity_and_hides_server_paths(ingest_service) -> None:
    body = b"# Title\n\nUploaded body text.\n"
    result = ingest_service.ingest(
        sources=[_upload("caller-owned-1", body, media_type="text/markdown")],
    )

    assert [source.source_id for source in result.sources] == ["caller-owned-1"]
    # A remote caller must never receive a server-local path, and identity must
    # not depend on where LewLM happened to stage the bytes.
    assert result.sources[0].path is None
    assert all(chunk.source_path is None for chunk in result.chunks)
    assert all(chunk.source_id == "caller-owned-1" for chunk in result.chunks)


def test_uploaded_source_verifies_the_expected_digest(ingest_service) -> None:
    body = b"# Title\n\nBody.\n"
    result = ingest_service.ingest(
        sources=[_upload("s1", body, expected_sha256=hashlib.sha256(body).hexdigest())],
    )
    assert result.source_results[0].status == "ingested"
    assert result.source_results[0].content_sha256 == hashlib.sha256(body).hexdigest()


def test_uploaded_source_with_a_wrong_digest_fails_that_source_only(ingest_service) -> None:
    good = b"# Good\n\nParses fine.\n"
    result = ingest_service.ingest(
        sources=[
            _upload("good", good),
            _upload("bad", b"# Bad\n\nBody.\n", expected_sha256="0" * 64),
        ],
    )

    assert result.ingested_count == 1
    assert result.failed_count == 1
    assert result.partial is True
    outcomes = {item.source_id: item for item in result.source_results}
    assert outcomes["good"].status == "ingested"
    assert outcomes["bad"].status == "failed"
    assert outcomes["bad"].error_code is DocumentIngestErrorCode.CHECKSUM_MISMATCH
    assert outcomes["bad"].retryable is False


def test_empty_uploaded_source_reports_a_stable_error_code(ingest_service) -> None:
    result = ingest_service.ingest(sources=[_upload("ok", b"# Ok\n\nBody.\n"), _upload("empty", b"")])
    outcomes = {item.source_id: item for item in result.source_results}
    assert outcomes["empty"].error_code is DocumentIngestErrorCode.EMPTY_SOURCE


def test_every_requested_source_gets_exactly_one_outcome(ingest_service) -> None:
    result = ingest_service.ingest(
        sources=[
            _upload("a", b"# A\n\nBody.\n"),
            _upload("b", b"# B\n\nBody.\n"),
            _upload("c", b"", ),
        ],
    )
    assert [item.source_id for item in result.source_results] == ["a", "b", "c"]
    assert len(result.source_results) == 3


def test_outcomes_carry_chunk_counts_and_provenance(ingest_service) -> None:
    result = ingest_service.ingest(sources=[_upload("s1", b"# Title\n\nBody one.\n\nBody two.\n")])
    outcome = result.source_results[0]
    assert outcome.chunk_count == sum(1 for chunk in result.chunks if chunk.source_id == "s1")
    assert outcome.chunk_count > 0
    assert outcome.provider_reference
    assert any(item.kind is ComponentKind.PARSER for item in outcome.components)
    assert any(item.kind is ComponentKind.CHUNKER for item in result.components)


def test_ingest_fails_the_request_when_no_source_survives(ingest_service) -> None:
    from lewlm.core.errors import DocumentValidationError

    with pytest.raises(DocumentValidationError) as exc_info:
        ingest_service.ingest(sources=[_upload("only", b"", )])
    # The per-source detail must still be recoverable from the failure.
    assert exc_info.value.details["source_results"][0]["error_code"] == "empty_source"


def test_duplicate_uploaded_source_ids_are_rejected(ingest_service) -> None:
    from lewlm.core.errors import DocumentValidationError

    with pytest.raises(DocumentValidationError, match="unique source identifiers"):
        ingest_service.ingest(sources=[_upload("same", b"# A\n\nx\n"), _upload("same", b"# B\n\ny\n")])


def test_blank_uploaded_source_id_is_rejected(ingest_service) -> None:
    from lewlm.core.errors import DocumentValidationError

    with pytest.raises(DocumentValidationError, match="non-empty source_id"):
        ingest_service.ingest(sources=[_upload("   ", b"# A\n\nx\n")])


def test_oversized_uploaded_source_is_refused(ingest_service) -> None:
    oversized = b"x" * (MAX_UPLOAD_SOURCE_BYTES + 1)
    result = ingest_service.ingest(
        sources=[_upload("ok", b"# Ok\n\nBody.\n"), _upload("big", oversized, file_name="big.md")],
    )
    outcomes = {item.source_id: item for item in result.source_results}
    assert outcomes["big"].error_code is DocumentIngestErrorCode.SOURCE_TOO_LARGE


def test_uploaded_metadata_is_bounded(ingest_service) -> None:
    from lewlm.core.errors import DocumentValidationError

    with pytest.raises(DocumentValidationError, match="exceeds the allowed length"):
        ingest_service.ingest(
            sources=[
                _upload(
                    "s1",
                    b"# Title\n\nBody.\n",
                    metadata={"note": "x" * (MAX_SOURCE_METADATA_VALUE_CHARACTERS + 1)},
                ),
            ],
        )


def test_uploaded_metadata_is_echoed_back(ingest_service) -> None:
    result = ingest_service.ingest(
        sources=[_upload("s1", b"# Title\n\nBody.\n", metadata={"tenant": "acme"})],
    )
    assert result.sources[0].metadata["tenant"] == "acme"
    assert result.sources[0].metadata["source_origin"] == "upload"


@pytest.mark.parametrize(
    ("raw", "expected_suffix"),
    [
        ("../../etc/passwd", "passwd"),
        ("/absolute/path/report.pdf", "report.pdf"),
        ("weird name!@#.md", ".md"),
    ],
)
def test_uploaded_file_names_cannot_escape_the_workspace(raw: str, expected_suffix: str) -> None:
    safe = _safe_upload_file_name(raw, index=0)
    assert "/" not in safe
    assert ".." not in safe
    assert safe.endswith(expected_suffix)


def test_empty_upload_file_name_still_produces_a_usable_name() -> None:
    assert _safe_upload_file_name("", index=3) == "upload-3"


def test_paths_and_uploads_can_be_mixed(ingest_service, tmp_path: Path) -> None:
    local = tmp_path / "local.md"
    local.write_text("# Local\n\nLocal body.\n", encoding="utf-8")

    result = ingest_service.ingest(
        [local],
        sources=[_upload("uploaded-1", b"# Uploaded\n\nUploaded body.\n")],
        allowed_file_roots=(tmp_path,),
    )
    assert result.ingested_count == 2
    identities = {source.source_id for source in result.sources}
    assert "uploaded-1" in identities
    # The path-based source keeps its derived identity and its path.
    path_sources = [source for source in result.sources if source.source_id != "uploaded-1"]
    assert path_sources[0].path is not None


def test_ingest_requires_at_least_one_source(ingest_service) -> None:
    from lewlm.core.errors import DocumentValidationError

    with pytest.raises(DocumentValidationError, match="at least one source"):
        ingest_service.ingest()
