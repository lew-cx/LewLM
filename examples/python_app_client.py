from __future__ import annotations

from lewlm import LewLM, LewLMAppClient
from lewlm.api.schemas import ChatMessage
from lewlm.documents.ingest.models import DocumentChunk, DocumentSourceType, IngestedDocumentSource
from lewlm.structured_output import JSONSchemaResponseFormat


CHAT_MODEL_ID = "<your-chat-model-id>"
EMBEDDING_MODEL_ID = "<your-embedding-model-id>"
RERANK_MODEL_ID = "<your-rerank-model-id>"
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
    "additionalProperties": False,
}


def embedded_example() -> None:
    with LewLM() as lewlm:
        client = lewlm.app_client()
        print(f"health: {client.health().status}")
        print(f"runtime policy: {client.runtime_stats().runtime_policy}")

        if CHAT_MODEL_ID.startswith("<"):
            print("Set CHAT_MODEL_ID before running the embedded chat example.")
            return

        chat = client.chat_completion(
            model=CHAT_MODEL_ID,
            messages=[ChatMessage(role="user", content="Return a JSON summary of LewLM for a host application.")],
            response_format=JSONSchemaResponseFormat(schema=SUMMARY_SCHEMA, name="summary_contract"),
            include_prompt_trace=True,
        )
        print(chat.choices[0].message.content)
        if chat.structured_output is not None:
            print(f"structured output validation: {chat.structured_output.validation.state}")
        if chat.prompt_trace is not None:
            print(f"prompt template: {chat.prompt_trace.selected_template}")
            print(f"output contract: {chat.prompt_trace.output_contract.format}")
        if not EMBEDDING_MODEL_ID.startswith("<") and not RERANK_MODEL_ID.startswith("<"):
            retrieval = client.retrieve_context(
                query="local backend integration helper",
                embedding_model=EMBEDDING_MODEL_ID,
                rerank_model=RERANK_MODEL_ID,
                top_k=1,
                candidate_sources=[
                    IngestedDocumentSource(
                        source_id="source-1",
                        path="/tmp/app-notes.md",
                        source_type=DocumentSourceType.MARKDOWN,
                        source_name="app-notes.md",
                        source_label="app-notes.md",
                    ),
                ],
                candidate_chunks=[
                    DocumentChunk(
                        chunk_id="chunk-1",
                        text="LewLM exposes typed helper methods for host applications.",
                        source_id="source-1",
                        section_id="section-1",
                        source_label="app-notes.md",
                        section_label="app-notes.md / Section 1",
                    ),
                    DocumentChunk(
                        chunk_id="chunk-2",
                        text="This unrelated note is about weather forecasts.",
                        source_id="source-1",
                        section_id="section-2",
                        source_label="app-notes.md",
                        section_label="app-notes.md / Section 2",
                    ),
                ],
            )
            print(f"top retrieval chunk: {retrieval.items[0].chunk.chunk_id}")


def remote_example(base_url: str = "http://127.0.0.1:8080") -> None:
    client = LewLMAppClient.from_http(base_url)
    print(f"health: {client.health().status}")
    print(f"runtime policy: {client.runtime_stats().runtime_policy}")


if __name__ == "__main__":
    embedded_example()
