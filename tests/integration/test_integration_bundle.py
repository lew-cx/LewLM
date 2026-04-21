from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from lewlm.api.app import create_app
from lewlm.api.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ResponseChunk,
    ResponseCreateRequest,
    ResponseCreateResponse,
)
from lewlm.api.schemas.documents import (
    DocumentGenerateRequest,
    DocumentGenerateResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentTransformResponse,
)
from lewlm.api.schemas.multimodal import (
    EmbeddingCreateRequest,
    EmbeddingCreateResponse,
    RetrievalContextRequest,
    RetrievalContextResponse,
    RerankCreateRequest,
    RerankCreateResponse,
)
from lewlm.documents.skills.models import DocumentTransformRequest
from lewlm.events.schema import StreamEvent


_BUNDLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "integration-bundle.json"


def _load_bundle() -> dict[str, object]:
    return json.loads(_BUNDLE_PATH.read_text(encoding="utf-8"))


def _parse_sse_payload(frame: str) -> dict[str, object]:
    for line in frame.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise AssertionError(f"Missing data line in SSE frame: {frame!r}")


def test_integration_bundle_schemas_match_current_models() -> None:
    bundle = _load_bundle()

    assert bundle["bundle_format"] == "lewlm-integration-bundle-v1"
    assert bundle["openapi_url"] == "/v1/openapi.json"
    assert bundle["example_client_path"] == "examples/http_api_integration.py"
    assert bundle["schemas"] == {
        "chat.request": ChatCompletionRequest.model_json_schema(),
        "chat.response": ChatCompletionResponse.model_json_schema(),
        "chat.stream": ChatCompletionChunk.model_json_schema(),
        "responses.request": ResponseCreateRequest.model_json_schema(),
        "responses.response": ResponseCreateResponse.model_json_schema(),
        "responses.stream": ResponseChunk.model_json_schema(),
        "embeddings.request": EmbeddingCreateRequest.model_json_schema(),
        "embeddings.response": EmbeddingCreateResponse.model_json_schema(),
        "retrieval.request": RetrievalContextRequest.model_json_schema(),
        "retrieval.response": RetrievalContextResponse.model_json_schema(),
        "rerank.request": RerankCreateRequest.model_json_schema(),
        "rerank.response": RerankCreateResponse.model_json_schema(),
        "documents.ingest.request": DocumentIngestRequest.model_json_schema(),
        "documents.ingest.response": DocumentIngestResponse.model_json_schema(),
        "documents.generate.request": DocumentGenerateRequest.model_json_schema(),
        "documents.generate.response": DocumentGenerateResponse.model_json_schema(),
        "documents.transform.request": TypeAdapter(DocumentTransformRequest).json_schema(),
        "documents.transform.response": DocumentTransformResponse.model_json_schema(),
        "events.stream": StreamEvent.model_json_schema(),
    }


def test_integration_bundle_examples_validate_against_models() -> None:
    bundle = _load_bundle()
    surfaces = {surface["name"]: surface for surface in bundle["surfaces"]}

    ChatCompletionRequest.model_validate(surfaces["chat"]["examples"]["request"])
    ChatCompletionResponse.model_validate(surfaces["chat"]["examples"]["response"])
    ChatCompletionChunk.model_validate(surfaces["chat"]["examples"]["stream_chunk"])
    ChatCompletionChunk.model_validate(_parse_sse_payload(surfaces["chat"]["examples"]["sse_frame"]))

    ResponseCreateRequest.model_validate(surfaces["responses"]["examples"]["request"])
    ResponseCreateResponse.model_validate(surfaces["responses"]["examples"]["response"])
    ResponseChunk.model_validate(surfaces["responses"]["examples"]["stream_chunk"])
    ResponseChunk.model_validate(_parse_sse_payload(surfaces["responses"]["examples"]["sse_frame"]))

    EmbeddingCreateRequest.model_validate(surfaces["embeddings"]["examples"]["request"])
    EmbeddingCreateResponse.model_validate(surfaces["embeddings"]["examples"]["response"])

    RetrievalContextRequest.model_validate(surfaces["retrieval"]["examples"]["request"])
    RetrievalContextResponse.model_validate(surfaces["retrieval"]["examples"]["response"])

    RerankCreateRequest.model_validate(surfaces["rerank"]["examples"]["request"])
    RerankCreateResponse.model_validate(surfaces["rerank"]["examples"]["response"])

    DocumentIngestRequest.model_validate(surfaces["documents.ingest"]["examples"]["request"])
    DocumentIngestResponse.model_validate(surfaces["documents.ingest"]["examples"]["response"])

    DocumentGenerateRequest.model_validate(surfaces["documents.generate"]["examples"]["request"])
    DocumentGenerateResponse.model_validate(surfaces["documents.generate"]["examples"]["response"])

    TypeAdapter(DocumentTransformRequest).validate_python(surfaces["documents.transform"]["examples"]["request"])
    DocumentTransformResponse.model_validate(surfaces["documents.transform"]["examples"]["response"])

    StreamEvent.model_validate(surfaces["events"]["examples"]["websocket_event"])
    StreamEvent.model_validate(_parse_sse_payload(surfaces["events"]["examples"]["sse_frame"]))


def test_integration_bundle_proves_out_host_app_surfaces() -> None:
    bundle = _load_bundle()
    surfaces = {surface["name"]: surface for surface in bundle["surfaces"]}

    chat_request = surfaces["chat"]["examples"]["request"]
    assert chat_request["include_prompt_trace"] is True
    assert chat_request["response_format"]["schema"]["required"] == ["summary"]
    assert chat_request["citation_context"]["chunks"][0]["chunk_id"] == "chunk-1"
    assert surfaces["chat"]["examples"]["response"]["prompt_trace"]["output_contract"]["format"] == "json_schema"
    assert surfaces["chat"]["examples"]["response"]["citations"][0]["chunk_id"] == "chunk-1"
    assert surfaces["chat"]["examples"]["response"]["prompt_trace"]["overrides"][-1]["source"] == "response_format"

    responses_request = surfaces["responses"]["examples"]["request"]
    assert responses_request["include_prompt_trace"] is True
    assert responses_request["response_format"]["schema"]["properties"]["summary"]["type"] == "string"
    assert surfaces["responses"]["examples"]["response"]["prompt_trace"]["output_contract"]["format"] == "json_schema"
    assert surfaces["responses"]["examples"]["response"]["citations"][0]["source_id"] == "source-1"
    assert surfaces["responses"]["examples"]["response"]["prompt_trace"]["overrides"][-1]["source"] == "response_format"

    ingest_response = surfaces["documents.ingest"]["examples"]["response"]
    assert ingest_response["sources"][0]["source_id"] == ingest_response["chunks"][0]["source_id"]
    assert ingest_response["chunks"][0]["section_id"] == ingest_response["document"]["sections"][0]["metadata"]["section_id"]
    assert ingest_response["chunks"][0]["section_label"].startswith(ingest_response["sources"][0]["source_label"])

    retrieval_response = surfaces["retrieval"]["examples"]["response"]
    assert retrieval_response["strategy"] == "hybrid"
    assert retrieval_response["items"][0]["chunk"]["chunk_id"] == "chunk-1"
    assert retrieval_response["items"][0]["source"]["source_id"] == "source-1"
    assert retrieval_response["embedding_stage"] is not None
    assert retrieval_response["rerank_stage"] is not None


def test_openapi_exposes_bundle_facing_request_and_stream_metadata() -> None:
    openapi = create_app().openapi()

    chat = openapi["paths"]["/v1/chat/completions"]["post"]
    assert set(chat["requestBody"]["content"]) == {"application/json", "multipart/form-data"}
    assert "text/event-stream" in chat["responses"]["200"]["content"]
    assert chat["requestBody"]["content"]["application/json"]["example"]["include_prompt_trace"] is True
    assert chat["requestBody"]["content"]["application/json"]["example"]["response_format"]["schema"]["required"] == ["summary"]

    responses = openapi["paths"]["/v1/responses"]["post"]
    assert set(responses["requestBody"]["content"]) == {"application/json", "multipart/form-data"}
    assert "text/event-stream" in responses["responses"]["200"]["content"]
    assert responses["requestBody"]["content"]["application/json"]["example"]["include_prompt_trace"] is True
    assert responses["requestBody"]["content"]["application/json"]["example"]["response_format"]["schema"]["required"] == ["summary"]
    assert responses["requestBody"]["content"]["application/json"]["example"]["citation_context"]["chunks"][0]["chunk_id"] == "chunk-1"

    retrieval = openapi["paths"]["/v1/retrieval/context"]["post"]
    assert set(retrieval["requestBody"]["content"]) == {"application/json"}
    assert retrieval["requestBody"]["content"]["application/json"]["example"]["candidate_chunks"][0]["chunk_id"] == "chunk-1"

    tools = openapi["paths"]["/v1/tools"]["get"]
    assert tools["responses"]["200"]["content"]["application/json"]["example"]["items"][0]["name"] == "documents.generate"

    tool = openapi["paths"]["/v1/tools/{tool_name}"]["get"]
    assert tool["responses"]["200"]["content"]["application/json"]["example"]["required_authorization"] == "document_generate"

    execute_tool = openapi["paths"]["/v1/tools/execute"]["post"]
    assert execute_tool["requestBody"]["content"]["application/json"]["examples"]["default"]["value"]["tool"] == "documents.generate"
    assert execute_tool["responses"]["200"]["content"]["application/json"]["example"]["trace"]["actor"] == "api"

    events = openapi["paths"]["/v1/events"]["get"]
    assert "text/event-stream" in events["responses"]["200"]["content"]
