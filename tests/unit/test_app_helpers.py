from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from lewlm import LewLM, LewLMAppClient
from lewlm.api.schemas.chat import ChatCompletionRequest, ChatMessage, ResponseCreateRequest
from lewlm.api.schemas.documents import DocumentIngestRequest, DocumentIngestResponse
from lewlm.api.schemas.health import ConfigurationHealth, HealthResponse, StorageHealth
from lewlm.api.schemas.multimodal import EmbeddingCreateRequest, RetrievalContextRequest, RerankCreateRequest
from lewlm.core.citations import CitationContextPackage, GeneratedCitationReference
from lewlm.core.contracts import GenerateResponse, ReasoningVisibility, RoutingDecision, RuntimeAffinity
from lewlm.core.execution_metadata import build_routed_execution_metadata
from lewlm.documents.ingest.models import DocumentChunk, DocumentSourceType, IngestedDocumentSource
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.install_profiles import InstallProfileStatus, InstallProfileSummary
from lewlm.prompting import PromptCompilationTrace
from lewlm.structured_output import JSONSchemaResponseFormat
from lewlm.telemetry.stats import RuntimeStats
from lewlm.tools.models import DocumentGenerateToolRequest, GenerateDocumentToolInput, LocalToolDescriptor


def _routing(model_id: str, *, affinity: RuntimeAffinity = RuntimeAffinity.MLX_TEXT) -> RoutingDecision:
    return RoutingDecision(
        model_id=model_id,
        runtime_name="fake-runtime",
        runtime_affinity=affinity,
        reason="unit-test routing",
    )


def _runtime_stats_payload() -> RuntimeStats:
    host_platform = {
        "system": "Darwin",
        "release": "24.0.0",
        "machine": "arm64",
        "python_version": "3.14.3",
    }
    scheduler = {
        "max_concurrent_requests": 4,
        "queue_limit": 8,
        "queue_timeout_seconds": 30,
        "active_requests": 0,
        "queued_requests": 0,
        "peak_active_requests": 1,
        "max_observed_queue_depth": 0,
        "total_queued_requests": 0,
        "rejected_requests": 0,
        "timed_out_requests": 0,
    }
    return RuntimeStats.model_validate(
        {
            "platform": host_platform,
            "readiness": {
                "status": "ready",
                "host_platform": host_platform,
                "discovered_model_count": 3,
                "runnable_model_count": 3,
                "capability_count": 4,
                "ready_capability_count": 4,
            },
            "runtime_policy": "balanced",
            "request_max_bytes": 1_048_576,
            "api_key_required": False,
            "active_sessions": 0,
            "queue_depth": 0,
            "active_jobs": 0,
            "current_loaded_models": [],
            "runtimes": [],
            "request_scheduler": scheduler,
            "load_scheduler": scheduler,
            "request_metrics": {},
            "benchmark_summary": {},
            "validation_manifest_count": 0,
            "target_platforms": [],
            "cluster": {"mode": "single_host"},
            "performance_features": [],
            "serving_core": {"version": "v1"},
            "optimization_defaults": None,
        },
    )


class _StubClusterStatus:
    def model_dump(self, mode: str = "json") -> dict[str, str]:
        return {"mode": "single_host"}


class _StubClusterService:
    def status(self) -> _StubClusterStatus:
        return _StubClusterStatus()


class _StubMetadataStore:
    def snapshot(self) -> dict[str, object]:
        return {
            "database_path": "/tmp/lewlm-test.db",
            "schema_version": 1,
            "model_count": 3,
        }


class _StubTelemetryService:
    def __init__(self, runtime_stats: RuntimeStats) -> None:
        self._runtime_stats = runtime_stats

    async def runtime_stats(self) -> RuntimeStats:
        return self._runtime_stats


class _StubChatOrchestrator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs["messages"]
        output_text = f"Echo: {messages[0].content}"
        citations: list[GeneratedCitationReference] = []
        citation_context = kwargs.get("citation_context")
        if isinstance(citation_context, CitationContextPackage) and citation_context.chunks:
            chunk = citation_context.chunks[0]
            citations = [
                GeneratedCitationReference(
                    reference_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    chunk_id=chunk.chunk_id,
                    section_id=chunk.section_id,
                    source_label=chunk.source_label,
                    section_label=chunk.section_label,
                ),
            ]
        return SimpleNamespace(
            request_id="chat-1",
            created_at=123,
            response=GenerateResponse(
                model_id=kwargs["model_id"] or "chat-model",
                output_text=output_text,
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                citations=citations,
            ),
            routing=_routing(kwargs["model_id"] or "chat-model"),
            prompt_trace=PromptCompilationTrace(
                selected_template="default",
                requested_model_id=kwargs["model_id"],
                resolved_model_id=kwargs["model_id"] or "chat-model",
                message_count=len(messages),
                message_roles=[message.role for message in messages],
            ),
            metadata=build_routed_execution_metadata(
                request_id="chat-1",
                created=123,
                requested_model_id=kwargs["model_id"],
                routing=_routing(kwargs["model_id"] or "chat-model"),
            ),
            request_metadata={"source": "unit-test"},
            structured_output=None,
            serving_profile=None,
        )


class _StubMultimodalOrchestrator:
    def __init__(self) -> None:
        self.embed_calls: list[tuple[str | None, list[str]]] = []
        self.rerank_calls: list[tuple[str | None, str, list[str], int | None]] = []
        self.retrieve_context_calls: list[dict[str, object]] = []

    async def embed(self, *, model_id: str | None, inputs: list[str]):
        self.embed_calls.append((model_id, inputs))
        return SimpleNamespace(
            request_id="embed-1",
            created_at=234,
            response=SimpleNamespace(
                model_id=model_id or "embed-model",
                data=[
                    SimpleNamespace(index=index, embedding=[float(index), 0.5, 1.0])
                    for index, _ in enumerate(inputs)
                ],
                usage={"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            ),
            routing=_routing(model_id or "embed-model"),
            metadata=build_routed_execution_metadata(
                request_id="embed-1",
                created=234,
                requested_model_id=model_id,
                routing=_routing(model_id or "embed-model"),
            ),
        )

    async def rerank(self, *, model_id: str | None, query: str, documents: list[str], top_n: int | None):
        self.rerank_calls.append((model_id, query, documents, top_n))
        ranked = [
            SimpleNamespace(index=0, relevance_score=0.9, document=documents[0]),
            SimpleNamespace(index=1, relevance_score=0.4, document=documents[1]),
        ]
        if top_n is not None:
            ranked = ranked[:top_n]
        return SimpleNamespace(
            request_id="rerank-1",
            created_at=345,
            response=SimpleNamespace(model_id=model_id or "rerank-model", results=ranked),
            routing=_routing(model_id or "rerank-model"),
            metadata=build_routed_execution_metadata(
                request_id="rerank-1",
                created=345,
                requested_model_id=model_id,
                routing=_routing(model_id or "rerank-model"),
            ),
        )

    async def retrieve_context(
        self,
        *,
        query: str,
        candidate_chunks: list[DocumentChunk],
        candidate_sources: list[IngestedDocumentSource],
        top_k: int,
        use_embeddings: bool,
        use_rerank: bool,
        embedding_model_id: str | None,
        rerank_model_id: str | None,
    ):
        self.retrieve_context_calls.append(
            {
                "query": query,
                "candidate_chunks": candidate_chunks,
                "candidate_sources": candidate_sources,
                "top_k": top_k,
                "use_embeddings": use_embeddings,
                "use_rerank": use_rerank,
                "embedding_model_id": embedding_model_id,
                "rerank_model_id": rerank_model_id,
            },
        )
        selected_chunk = candidate_chunks[0]
        selected_source = next(
            (source for source in candidate_sources if source.source_id == selected_chunk.source_id),
            None,
        )
        return SimpleNamespace(
            request_id="retrieval-1",
            created_at=456,
            query=query,
            strategy="hybrid",
            items=[
                SimpleNamespace(
                    rank=1,
                    score=0.95,
                    embedding_score=0.87,
                    rerank_score=0.95,
                    chunk=selected_chunk,
                    source=selected_source,
                ),
            ],
            sources=[selected_source] if selected_source is not None else [],
            embedding_stage=SimpleNamespace(
                request_id="embed-stage-1",
                created_at=234,
                model_id=embedding_model_id or "embed-model",
                routing=_routing(embedding_model_id or "embed-model"),
                metadata=build_routed_execution_metadata(
                    request_id="embed-stage-1",
                    created=234,
                    requested_model_id=embedding_model_id,
                    routing=_routing(embedding_model_id or "embed-model"),
                ),
                usage={"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
            ),
            rerank_stage=SimpleNamespace(
                request_id="rerank-stage-1",
                created_at=345,
                model_id=rerank_model_id or "rerank-model",
                routing=_routing(rerank_model_id or "rerank-model"),
                metadata=build_routed_execution_metadata(
                    request_id="rerank-stage-1",
                    created=345,
                    requested_model_id=rerank_model_id,
                    routing=_routing(rerank_model_id or "rerank-model"),
                ),
                usage=None,
            ),
            metadata=build_routed_execution_metadata(
                request_id="retrieval-1",
                created=456,
                requested_model_id=rerank_model_id,
                routing=_routing(rerank_model_id or "rerank-model"),
            ),
        )


class _StubToolExecutionService:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def execute(self, request, *, actor: str, allowed_file_roots, emit_tool_events: bool, base_dir=None, request_id=None):
        self.calls.append(
            {
                "request": request,
                "actor": actor,
                "allowed_file_roots": allowed_file_roots,
                "emit_tool_events": emit_tool_events,
                "base_dir": base_dir,
                "request_id": request_id,
            },
        )
        payload = request.input
        if request.tool == "documents.generate":
            return SimpleNamespace(
                request_id="tool-generate-1",
                idempotency_key=payload.idempotency_key,
                idempotent_replay=False,
                tool=request.tool,
                trace=SimpleNamespace(started_at=datetime.now(timezone.utc), duration_ms=8, actor=actor),
                result={
                    "artifact": {
                        "file_name": payload.file_name or "generated.md",
                        "output_format": payload.output_format.value,
                        "media_type": "text/markdown",
                    },
                    "document": payload.document.model_dump(mode="json"),
                },
            )
        document = DocumentIR(title=payload.title or "Ingested")
        source = IngestedDocumentSource(
            source_id="source-1",
            path=payload.paths[0],
            source_type=DocumentSourceType.MARKDOWN,
            source_name=Path(payload.paths[0]).name,
            source_label=Path(payload.paths[0]).name,
        )
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            text="hello",
            source_id=source.source_id,
            section_id="section-1",
            source_label=source.source_label,
            section_label=f"{source.source_label} / Section 1",
            source_name=source.source_name,
            source_path=payload.paths[0],
            source_type=DocumentSourceType.MARKDOWN,
        )
        response = DocumentIngestResponse(
            request_id="ingest-1",
            document=document,
            sources=[source],
            chunks=[chunk],
            metadata=build_routed_execution_metadata(
                request_id="ingest-1",
                created=456,
                requested_model_id=None,
                routing=_routing("documents.ingest"),
                result_origin="coalesced",
            ),
        )
        return SimpleNamespace(
            request_id=response.request_id,
            idempotency_key=None,
            idempotent_replay=False,
            tool="documents.ingest",
            trace=SimpleNamespace(started_at=datetime.now(timezone.utc), duration_ms=12, actor=actor),
            result=response.model_dump(mode="json", exclude={"request_id", "idempotency_key", "idempotent_replay"}),
        )


class _StubToolCatalogService:
    def __init__(self) -> None:
        self._tools = [
            LocalToolDescriptor(
                name="documents.generate",
                description="Render a structured document payload into a deterministic artifact.",
                required_authorization="document_generate",
                result_type="artifact",
                input_schema={"type": "object"},
                tags=["documents", "generation", "local"],
                aliases=["document_generate"],
            ),
            LocalToolDescriptor(
                name="documents.ingest",
                description="Ingest local files into structured document output.",
                required_authorization="document_ingest",
                result_type="document_ir",
                input_schema={"type": "object"},
                tags=["documents", "ingest", "local"],
                aliases=["document_ingest"],
            ),
        ]

    def list_tools(self) -> list[LocalToolDescriptor]:
        return list(self._tools)

    def get_tool(self, tool_name: str) -> LocalToolDescriptor:
        for descriptor in self._tools:
            if descriptor.name == tool_name:
                return descriptor
        raise LookupError(tool_name)


def _stub_services(temp_settings):
    runtime_stats = _runtime_stats_payload()
    chat_orchestrator = _StubChatOrchestrator()
    multimodal_orchestrator = _StubMultimodalOrchestrator()
    tool_execution_service = _StubToolExecutionService()
    tool_catalog_service = _StubToolCatalogService()
    services = SimpleNamespace(
        settings=temp_settings.with_updates(
            reasoning_visibility=ReasoningVisibility.HIDDEN,
            file_access_roots=(temp_settings.data_dir,),
        ),
        metadata_store=_StubMetadataStore(),
        model_router=SimpleNamespace(capability_readiness_summary=lambda: runtime_stats.readiness),
        cluster_service=_StubClusterService(),
        telemetry_service=_StubTelemetryService(runtime_stats),
        chat_orchestrator=chat_orchestrator,
        multimodal_orchestrator=multimodal_orchestrator,
        tool_catalog_service=tool_catalog_service,
        tool_execution_service=tool_execution_service,
    )
    return services, chat_orchestrator, multimodal_orchestrator, tool_execution_service, runtime_stats


def test_embedded_app_client_maps_common_workflows(temp_settings) -> None:
    services, chat_orchestrator, multimodal_orchestrator, tool_execution_service, runtime_stats = _stub_services(temp_settings)
    lewlm = LewLM(services=services)
    client = lewlm.app_client()
    output_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
        },
        "required": ["summary"],
        "additionalProperties": False,
    }
    citation_context = CitationContextPackage.model_validate(
        {
            "sources": [
                {
                    "source_id": "source-1",
                    "path": "/tmp/source.md",
                    "source_type": "markdown",
                    "source_name": "source.md",
                    "source_label": "Source One",
                    "metadata": {},
                }
            ],
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "text": "LewLM exposes a local-first backend package.",
                    "source_id": "source-1",
                    "section_id": "section-1",
                    "source_label": "Source One",
                    "section_label": "Source One / Summary",
                    "metadata": {},
                }
            ],
        },
    )

    raw_health = lewlm.health()
    health = client.health()
    chat = client.chat_completion(
        model="chat-model",
        messages=[ChatMessage(role="user", content="Hello from the embedded helper")],
        citation_context=citation_context,
        response_format=JSONSchemaResponseFormat(schema=output_schema, name="summary_contract"),
        include_prompt_trace=True,
    )
    responses = client.responses(
        request=ResponseCreateRequest(
            model="chat-model",
            input="Hello from the responses helper",
            response_format=JSONSchemaResponseFormat(schema=output_schema, name="summary_contract"),
            include_prompt_trace=True,
        ),
    )
    embeddings = client.embeddings(
        request=EmbeddingCreateRequest(model="embed-model", input=["alpha", "beta"]),
    )
    rerank = client.rerank(
        request=RerankCreateRequest(
            model="rerank-model",
            query="local model",
            documents=["local model routing", "remote api"],
            top_n=1,
        ),
    )
    retrieval = client.retrieve_context(
        request=RetrievalContextRequest(
            query="local model",
            candidate_sources=[
                IngestedDocumentSource(
                    source_id="source-1",
                    path="/tmp/source.md",
                    source_type=DocumentSourceType.MARKDOWN,
                    source_name="source.md",
                    source_label="source.md",
                ),
            ],
            candidate_chunks=[
                DocumentChunk(
                    chunk_id="chunk-1",
                    text="local model routing",
                    source_id="source-1",
                    section_id="section-1",
                    source_label="source.md",
                    section_label="source.md / Section 1",
                    source_name="source.md",
                    source_path="/tmp/source.md",
                    source_type=DocumentSourceType.MARKDOWN,
                ),
            ],
            embedding_model="embed-model",
            rerank_model="rerank-model",
            top_k=1,
        ),
    )
    ingest = client.ingest_documents(
        request=DocumentIngestRequest(paths=[str(Path(temp_settings.data_dir) / "sample.md")], title="Stub ingest"),
    )
    tools = client.list_tools()
    tool = client.get_tool("documents.generate")
    tool_execution = client.execute_tool(
        DocumentGenerateToolRequest(
            input=GenerateDocumentToolInput(
                output_format=DocumentOutputFormat.MARKDOWN,
                file_name="embedded-tool.md",
                document=DocumentIR(title="Embedded tool output"),
            ),
        ),
    )
    stats = client.runtime_stats()

    assert raw_health["configuration"]["tool_sandbox_enabled"] is True
    assert raw_health["cluster"] == {"mode": "single_host"}
    assert isinstance(health, HealthResponse)
    assert health.configuration == ConfigurationHealth.model_validate(raw_health["configuration"])
    assert health.storage == StorageHealth.model_validate(raw_health["storage"])
    assert chat.choices[0].message.content == "Echo: Hello from the embedded helper"
    assert chat.citations[0].chunk_id == "chunk-1"
    assert chat.citations[0].source_label == "Source One"
    assert chat.prompt_trace is not None
    assert chat.prompt_trace.message_count == 1
    assert chat_orchestrator.calls[0]["prompt_request"].actor == "api"
    assert chat_orchestrator.calls[0]["citation_context"] == citation_context
    assert chat_orchestrator.calls[0]["prompt_request"].response_format.type == "json_schema"
    assert chat_orchestrator.calls[0]["prompt_request"].include_trace is True
    assert responses.output_text == "Echo: Hello from the responses helper"
    assert embeddings.model == "embed-model"
    assert len(embeddings.data) == 2
    assert multimodal_orchestrator.embed_calls == [("embed-model", ["alpha", "beta"])]
    assert rerank.model == "rerank-model"
    assert [item.index for item in rerank.results] == [0]
    assert multimodal_orchestrator.rerank_calls == [
        ("rerank-model", "local model", ["local model routing", "remote api"], 1),
    ]
    assert retrieval.strategy == "hybrid"
    assert retrieval.items[0].chunk.chunk_id == "chunk-1"
    assert retrieval.items[0].source is not None
    assert retrieval.items[0].source.source_id == "source-1"
    assert retrieval.embedding_stage is not None
    assert retrieval.rerank_stage is not None
    assert multimodal_orchestrator.retrieve_context_calls[0]["top_k"] == 1
    assert ingest.request_id == "ingest-1"
    assert ingest.sources[0].path.endswith("sample.md")
    assert tools.count == 2
    assert tools.items[0].name == "documents.generate"
    assert tool.name == "documents.generate"
    assert tool_execution.tool == "documents.generate"
    assert tool_execution.trace.actor == "api"
    assert tool_execution.result["artifact"]["file_name"] == "embedded-tool.md"
    assert tool_execution_service.calls[0]["actor"] == "app_client"
    assert tool_execution_service.calls[1]["actor"] == "api"
    assert stats == runtime_stats


def test_app_client_rejects_mixed_request_and_keyword_arguments(temp_settings) -> None:
    services, _, _, _, _ = _stub_services(temp_settings)
    client = LewLM(services=services).app_client()

    with pytest.raises(ValueError, match="either `request` or keyword arguments"):
        client.chat_completion(
            request=ChatCompletionRequest(
                model="chat-model",
                messages=[ChatMessage(role="user", content="hello")],
            ),
            messages=[ChatMessage(role="user", content="override")],
        )
    with pytest.raises(ValueError, match="either `request` or keyword arguments"):
        client.embeddings(
            request=EmbeddingCreateRequest(model="embed-model", input="alpha"),
            inputs="beta",
        )


@contextmanager
def _serve_http_app_client_api() -> Iterator[tuple[str, list[dict[str, object]]]]:
    request_log: list[dict[str, object]] = []
    runtime_stats = _runtime_stats_payload().model_dump(mode="json")
    tools = {
        "count": 2,
        "items": [
            {
                "name": "documents.generate",
                "version": "1.0.0",
                "description": "Render a structured document payload into a deterministic artifact.",
                "execution_mode": "local",
                "required_authorization": "document_generate",
                "result_type": "artifact",
                "input_schema": {"type": "object"},
                "tags": ["documents", "generation", "local"],
                "aliases": ["document_generate"],
            },
            {
                "name": "documents.ingest",
                "version": "1.0.0",
                "description": "Ingest local files into structured document output.",
                "execution_mode": "local",
                "required_authorization": "document_ingest",
                "result_type": "document_ir",
                "input_schema": {"type": "object"},
                "tags": ["documents", "ingest", "local"],
                "aliases": ["document_ingest"],
            },
        ],
    }
    health = HealthResponse(
        status="ok",
        service="LewLM",
        version="0.1.0a0",
        time=datetime.now(timezone.utc),
        install_profiles=InstallProfileSummary(
            active_profile_ids=["core_only"],
            recommended_profile_id="gguf_fallback_backend",
            profiles=[
                InstallProfileStatus(
                    profile="core_only",
                    label="Core only",
                    install_spec=".",
                    installed=True,
                    ready=True,
                    summary="base install",
                ),
            ],
        ),
        readiness=_runtime_stats_payload().readiness,
        storage=StorageHealth(
            healthy=True,
            database_path="/tmp/lewlm-test.db",
            schema_version=1,
            model_count=3,
        ),
        configuration=ConfigurationHealth(
            data_dir="/tmp/lewlm",
            models_dir=["/tmp/lewlm/models"],
            privacy_mode=False,
            telemetry_enabled=False,
            allow_outbound_network=False,
            audit_log_enabled=True,
            persistence_encryption_enabled=False,
            tool_authorization_required=True,
            parser_sandbox_enabled=True,
            tool_sandbox_enabled=True,
            conversion_sandbox_enabled=True,
        ),
        cluster={"mode": "single_host"},
    ).model_dump(mode="json")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request_log.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
            if self.path == "/v1/health":
                payload = health
            elif self.path == "/v1/runtime/stats":
                payload = runtime_stats
            elif self.path == "/v1/tools":
                payload = tools
            elif self.path == "/v1/tools/documents.generate":
                payload = tools["items"][0]
            else:
                self.send_error(404)
                return
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            request_log.append(
                {
                    "method": "POST",
                    "path": self.path,
                    "headers": dict(self.headers),
                    "payload": payload,
                },
            )
            if self.path == "/v1/chat/completions":
                response = {
                    "id": "chat-remote-1",
                    "object": "chat.completion",
                    "created": 123,
                    "model": payload["model"],
                    "session_id": payload.get("session_id"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"Echo: {payload['messages'][0]['content']}"},
                            "finish_reason": "stop",
                        },
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                    "metadata": build_routed_execution_metadata(
                        request_id="chat-remote-1",
                        created=123,
                        requested_model_id=payload["model"],
                        routing=_routing(payload["model"]),
                    ).model_dump(mode="json"),
                    "prompt_trace": None,
                    "serving_profile": None,
                }
            elif self.path == "/v1/responses":
                response = {
                    "id": "response-remote-1",
                    "object": "response",
                    "created": 124,
                    "model": payload["model"],
                    "session_id": payload.get("session_id"),
                    "output": [
                        {
                            "type": "output_text",
                            "text": f"Echo: {payload['input']}",
                            "reasoning": None,
                        },
                    ],
                    "output_text": f"Echo: {payload['input']}",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                    "metadata": build_routed_execution_metadata(
                        request_id="response-remote-1",
                        created=124,
                        requested_model_id=payload["model"],
                        routing=_routing(payload["model"]),
                    ).model_dump(mode="json"),
                    "prompt_trace": None,
                    "serving_profile": None,
                }
            elif self.path == "/v1/embeddings":
                inputs = payload["input"] if isinstance(payload["input"], list) else [payload["input"]]
                response = {
                    "request_id": "embed-remote-1",
                    "created": 234,
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": [float(index), 0.5, 1.0], "index": index}
                        for index, _ in enumerate(inputs)
                    ],
                    "model": payload["model"],
                    "usage": {"prompt_tokens": len(inputs), "completion_tokens": 0, "total_tokens": len(inputs)},
                    "routing": _routing(payload["model"]).model_dump(mode="json"),
                    "metadata": build_routed_execution_metadata(
                        request_id="embed-remote-1",
                        created=234,
                        requested_model_id=payload["model"],
                        routing=_routing(payload["model"]),
                    ).model_dump(mode="json"),
                }
            elif self.path == "/v1/rerank":
                response = {
                    "request_id": "rerank-remote-1",
                    "created": 345,
                    "model": payload["model"],
                    "results": [
                        {"index": 0, "relevance_score": 0.9, "document": payload["documents"][0]},
                    ],
                    "routing": _routing(payload["model"]).model_dump(mode="json"),
                    "metadata": build_routed_execution_metadata(
                        request_id="rerank-remote-1",
                        created=345,
                        requested_model_id=payload["model"],
                        routing=_routing(payload["model"]),
                    ).model_dump(mode="json"),
                }
            elif self.path == "/v1/retrieval/context":
                response = {
                    "request_id": "retrieval-remote-1",
                    "created": 456,
                    "query": payload["query"],
                    "strategy": "hybrid",
                    "candidate_count": len(payload["candidate_chunks"]),
                    "returned_count": 1,
                    "items": [
                        {
                            "rank": 1,
                            "score": 0.95,
                            "embedding_score": 0.87,
                            "rerank_score": 0.95,
                            "chunk": payload["candidate_chunks"][0],
                            "source": payload["candidate_sources"][0],
                        },
                    ],
                    "sources": [payload["candidate_sources"][0]],
                    "embedding_stage": {
                        "request_id": "embed-stage-remote-1",
                        "created": 234,
                        "model": payload.get("embedding_model") or "embed-model",
                        "routing": _routing(payload.get("embedding_model") or "embed-model").model_dump(mode="json"),
                        "metadata": build_routed_execution_metadata(
                            request_id="embed-stage-remote-1",
                            created=234,
                            requested_model_id=payload.get("embedding_model"),
                            routing=_routing(payload.get("embedding_model") or "embed-model"),
                        ).model_dump(mode="json"),
                        "usage": {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
                    },
                    "rerank_stage": {
                        "request_id": "rerank-stage-remote-1",
                        "created": 345,
                        "model": payload.get("rerank_model") or "rerank-model",
                        "routing": _routing(payload.get("rerank_model") or "rerank-model").model_dump(mode="json"),
                        "metadata": build_routed_execution_metadata(
                            request_id="rerank-stage-remote-1",
                            created=345,
                            requested_model_id=payload.get("rerank_model"),
                            routing=_routing(payload.get("rerank_model") or "rerank-model"),
                        ).model_dump(mode="json"),
                    },
                    "metadata": build_routed_execution_metadata(
                        request_id="retrieval-remote-1",
                        created=456,
                        requested_model_id=payload.get("rerank_model"),
                        routing=_routing(payload.get("rerank_model") or "rerank-model"),
                    ).model_dump(mode="json"),
                }
            elif self.path == "/v1/documents/ingest":
                response = DocumentIngestResponse(
                    request_id="ingest-remote-1",
                    document=DocumentIR(title=payload.get("title") or "Remote ingest"),
                    sources=[
                        IngestedDocumentSource(
                            source_id="source-remote-1",
                            path=payload["paths"][0],
                            source_type=DocumentSourceType.MARKDOWN,
                            source_name=Path(payload["paths"][0]).name,
                            source_label=Path(payload["paths"][0]).name,
                        ),
                    ],
                    chunks=[],
                    metadata=build_routed_execution_metadata(
                        request_id="ingest-remote-1",
                        created=456,
                        requested_model_id=None,
                        routing=_routing("documents.ingest"),
                        result_origin="coalesced",
                    ),
                ).model_dump(mode="json")
            elif self.path == "/v1/tools/execute":
                response = {
                    "request_id": "tool-remote-1",
                    "tool": payload["tool"],
                    "idempotency_key": payload["input"].get("idempotency_key"),
                    "idempotent_replay": False,
                    "trace": {
                        "tool": payload["tool"],
                        "version": "1.0.0",
                        "execution_mode": "local",
                        "actor": "api",
                        "required_authorization": "document_generate",
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "duration_ms": 11,
                        "summary": "Generated one deterministic markdown artifact.",
                        "details": {"file_name": payload["input"].get("file_name")},
                    },
                    "result": {
                        "artifact": {
                            "file_name": payload["input"].get("file_name") or "generated.md",
                            "output_format": payload["input"]["output_format"],
                            "media_type": "text/markdown",
                        },
                    },
                }
            else:
                self.send_error(404)
                return
            raw = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", request_log
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_http_app_client_round_trips_typed_requests_and_responses() -> None:
    with _serve_http_app_client_api() as (base_url, request_log):
        client = LewLMAppClient.from_http(base_url, api_key="secret-key")
        health = client.health()
        chat = client.chat_completion(
            model="chat-model",
            messages=[ChatMessage(role="user", content="Hello from HTTP")],
            response_format=JSONSchemaResponseFormat(
                schema={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
                name="summary_contract",
            ),
            include_prompt_trace=True,
        )
        responses = client.responses(
            model="chat-model",
            input="Hello from responses over HTTP",
            response_format=JSONSchemaResponseFormat(
                schema={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
                name="summary_contract",
            ),
        )
        embeddings = client.embeddings(model="embed-model", inputs=["alpha", "beta"])
        rerank = client.rerank(
            model="rerank-model",
            query="local model",
            documents=["local model routing", "remote api"],
            top_n=1,
        )
        retrieval = client.retrieve_context(
            query="local model",
            candidate_sources=[
                IngestedDocumentSource(
                    source_id="source-1",
                    path="/tmp/source.md",
                    source_type=DocumentSourceType.MARKDOWN,
                    source_name="source.md",
                    source_label="source.md",
                ),
            ],
            candidate_chunks=[
                DocumentChunk(
                    chunk_id="chunk-1",
                    text="local model routing",
                    source_id="source-1",
                    section_id="section-1",
                    source_label="source.md",
                    section_label="source.md / Section 1",
                    source_name="source.md",
                    source_path="/tmp/source.md",
                    source_type=DocumentSourceType.MARKDOWN,
                ),
            ],
            embedding_model="embed-model",
            rerank_model="rerank-model",
            top_k=1,
        )
        ingest = client.ingest_documents(paths=["/tmp/sample.md"], title="Remote ingest")
        tools = client.list_tools()
        tool = client.get_tool("documents.generate")
        tool_execution = client.execute_tool(
            DocumentGenerateToolRequest(
                input=GenerateDocumentToolInput(
                    output_format=DocumentOutputFormat.MARKDOWN,
                    file_name="remote-tool.md",
                    document=DocumentIR(title="Remote tool output"),
                ),
            ),
        )
        runtime_stats = client.runtime_stats()

    assert health.status == "ok"
    assert health.configuration.tool_sandbox_enabled is True
    assert chat.model == "chat-model"
    assert chat.choices[0].message.content == "Echo: Hello from HTTP"
    assert responses.output_text == "Echo: Hello from responses over HTTP"
    assert embeddings.model == "embed-model"
    assert len(embeddings.data) == 2
    assert rerank.model == "rerank-model"
    assert rerank.results[0].document == "local model routing"
    assert retrieval.request_id == "retrieval-remote-1"
    assert retrieval.items[0].chunk.chunk_id == "chunk-1"
    assert retrieval.embedding_stage is not None
    assert ingest.request_id == "ingest-remote-1"
    assert ingest.sources[0].path == "/tmp/sample.md"
    assert tools.count == 2
    assert tool.name == "documents.generate"
    assert tool_execution.tool == "documents.generate"
    assert tool_execution.result["artifact"]["file_name"] == "remote-tool.md"
    assert runtime_stats.runtime_policy == "balanced"
    post_paths = [entry["path"] for entry in request_log if entry["method"] == "POST"]
    assert post_paths == [
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/retrieval/context",
        "/v1/documents/ingest",
        "/v1/tools/execute",
    ]
    get_paths = [entry["path"] for entry in request_log if entry["method"] == "GET"]
    assert "/v1/tools" in get_paths
    assert "/v1/tools/documents.generate" in get_paths
    chat_request = next(entry for entry in request_log if entry["method"] == "POST" and entry["path"] == "/v1/chat/completions")
    assert chat_request["payload"]["include_prompt_trace"] is True
    assert chat_request["payload"]["response_format"]["schema"]["required"] == ["summary"]
    tool_request = next(entry for entry in request_log if entry["method"] == "POST" and entry["path"] == "/v1/tools/execute")
    assert tool_request["payload"]["tool"] == "documents.generate"
    assert tool_request["payload"]["input"]["file_name"] == "remote-tool.md"
    assert all(entry["headers"].get("X-Api-Key") == "secret-key" for entry in request_log)
