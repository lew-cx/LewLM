from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from lewlm import LewLM, LewLMAppClient
from lewlm.api.schemas import ChatMessage
from lewlm.core.citations import CitationContextPackage
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat, DocumentSection, ParagraphBlock
from lewlm.structured_output import JSONSchemaResponseFormat
from lewlm.tools.models import DocumentGenerateToolRequest, GenerateDocumentToolInput


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["title", "summary"],
    "additionalProperties": False,
}


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2))


@contextmanager
def _client_context(base_url: str | None) -> Iterator[LewLMAppClient]:
    if base_url:
        yield LewLMAppClient.from_http(base_url)
        return
    with LewLM() as lewlm:
        yield lewlm.app_client()


def _starter_document(title: str, summary: str) -> DocumentIR:
    return DocumentIR(
        title=title,
        sections=[
            DocumentSection(
                heading="Summary",
                blocks=[ParagraphBlock(text=summary)],
            )
        ],
    )


def run_chat_app(client: LewLMAppClient, *, model: str, prompt: str) -> None:
    response = client.chat_completion(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        response_format=JSONSchemaResponseFormat(schema=SUMMARY_SCHEMA, name="chat_card"),
        include_prompt_trace=True,
    )
    _print_json(
        {
            "pattern": "chat-app",
            "text": response.choices[0].message.content,
            "structured_output": None if response.structured_output is None else response.structured_output.model_dump(mode="json"),
            "prompt_trace": None if response.prompt_trace is None else response.prompt_trace.model_dump(mode="json"),
            "metadata": response.metadata.model_dump(mode="json"),
        },
    )


def run_grounded_answer_app(
    client: LewLMAppClient,
    *,
    model: str,
    paths: list[str],
    question: str,
    embedding_model: str | None,
    rerank_model: str | None,
) -> None:
    ingest = client.ingest_documents(paths=paths, title="Grounded answer starter proof")
    context = CitationContextPackage(sources=ingest.sources, chunks=ingest.chunks)
    retrieval = None
    if ingest.chunks and (embedding_model is not None or rerank_model is not None):
        retrieval = client.retrieve_context(
            query=question,
            candidate_sources=ingest.sources,
            candidate_chunks=ingest.chunks,
            top_k=min(4, len(ingest.chunks)),
            use_embeddings=embedding_model is not None,
            use_rerank=rerank_model is not None,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
        )
        context = CitationContextPackage(
            sources=[item.source for item in retrieval.items if item.source is not None] or ingest.sources,
            chunks=[item.chunk for item in retrieval.items] or ingest.chunks,
        )
    response = client.responses(
        model=model,
        input=question,
        citation_context=context,
    )
    _print_json(
        {
            "pattern": "grounded-answer-app",
            "answer": response.output_text,
            "citations": [citation.model_dump(mode="json") for citation in (response.citations or [])],
            "ingest": {
                "request_id": ingest.request_id,
                "source_count": len(ingest.sources),
                "chunk_count": len(ingest.chunks),
            },
            "retrieval": None
            if retrieval is None
            else {
                "strategy": retrieval.strategy,
                "returned_count": retrieval.returned_count,
                "chunk_ids": [item.chunk.chunk_id for item in retrieval.items],
            },
        },
    )


def run_document_ingest_app(client: LewLMAppClient, *, paths: list[str], title: str | None) -> None:
    response = client.ingest_documents(paths=paths, title=title)
    _print_json(
        {
            "pattern": "document-ingest-app",
            "request_id": response.request_id,
            "document_title": response.document.title,
            "sources": [source.model_dump(mode="json") for source in response.sources],
            "chunks": [chunk.model_dump(mode="json") for chunk in response.chunks[:3]],
            "metadata": response.metadata.model_dump(mode="json"),
        },
    )


def run_local_tool_app(client: LewLMAppClient) -> None:
    tool_catalog = client.list_tools()
    document_tool = client.get_tool("documents.generate")
    tool_execution = client.execute_tool(
        DocumentGenerateToolRequest(
            input=GenerateDocumentToolInput(
                output_format=DocumentOutputFormat.MARKDOWN,
                file_name="starter-proof.md",
                document=_starter_document(
                    "Starter proof",
                    "LewLM can own deterministic local document work without forcing a workflow engine.",
                ),
                authorized_actions=["document_generate"],
            ),
        ),
    )
    _print_json(
        {
            "pattern": "local-tool-app",
            "catalog_count": tool_catalog.count,
            "selected_tool": document_tool.model_dump(mode="json"),
            "execution": tool_execution.model_dump(mode="json"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Starter proofs for common LewLM host-app integration patterns.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="LewLM HTTP base URL. Omit to run the same flows against an embedded LewLM instance.",
    )
    subparsers = parser.add_subparsers(dest="pattern", required=True)

    chat_parser = subparsers.add_parser("chat-app", help="Structured chat-app starter proof.")
    chat_parser.add_argument("--model", required=True, help="Chat-capable LewLM model id.")
    chat_parser.add_argument(
        "--prompt",
        default="Return a JSON card that explains LewLM for a host application.",
        help="Prompt to send through the chat-completions helper.",
    )

    grounded_parser = subparsers.add_parser("grounded-answer-app", help="Grounded-answer starter proof.")
    grounded_parser.add_argument("--model", required=True, help="Chat-capable LewLM model id.")
    grounded_parser.add_argument("paths", nargs="+", help="Local files LewLM is allowed to ingest.")
    grounded_parser.add_argument(
        "--question",
        default="Summarize the supplied material and ground the answer in the returned citation context.",
        help="Question to ask after ingest.",
    )
    grounded_parser.add_argument("--embedding-model", default=None, help="Optional embedding model for retrieval narrowing.")
    grounded_parser.add_argument("--rerank-model", default=None, help="Optional rerank model for retrieval narrowing.")

    ingest_parser = subparsers.add_parser("document-ingest-app", help="Document-ingest starter proof.")
    ingest_parser.add_argument("paths", nargs="+", help="Local files LewLM is allowed to ingest.")
    ingest_parser.add_argument("--title", default=None, help="Optional ingest title override.")

    subparsers.add_parser("local-tool-app", help="Local-tool starter proof using the shared tool contract.")

    args = parser.parse_args()
    with _client_context(args.base_url) as client:
        if args.pattern == "chat-app":
            run_chat_app(client, model=args.model, prompt=args.prompt)
            return
        if args.pattern == "grounded-answer-app":
            run_grounded_answer_app(
                client,
                model=args.model,
                paths=[str(Path(path).expanduser().resolve()) for path in args.paths],
                question=args.question,
                embedding_model=args.embedding_model,
                rerank_model=args.rerank_model,
            )
            return
        if args.pattern == "document-ingest-app":
            run_document_ingest_app(
                client,
                paths=[str(Path(path).expanduser().resolve()) for path in args.paths],
                title=args.title,
            )
            return
        if args.pattern == "local-tool-app":
            run_local_tool_app(client)
            return
    raise SystemExit(f"Unsupported starter proof: {args.pattern}")


if __name__ == "__main__":
    main()
