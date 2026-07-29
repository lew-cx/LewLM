"""RAG Chat and document-generation clients sharing one LewLM server process."""

from __future__ import annotations

from pydantic import BaseModel

from lewlm import LewLMAppClient
from lewlm.api.schemas import ChatMessage
from lewlm.documents.ir.models import DocumentIR, DocumentSection, ListBlock, ParagraphBlock


BASE_URL = "http://127.0.0.1:8080"
CHAT_MODEL_ID = "<your-chat-model-id>"


# This product schema belongs to the document application, not LewLM core.
class StatusReportSpec(BaseModel):
    title: str
    executive_summary: str
    accomplishments: list[str]
    risks: list[str]
    next_steps: list[str]


def status_report_to_document_ir(report: StatusReportSpec) -> DocumentIR:
    """Application-owned adapter into LewLM's generic deterministic IR."""

    return DocumentIR(
        title=report.title,
        sections=[
            DocumentSection(
                heading="Executive summary",
                blocks=[ParagraphBlock(text=report.executive_summary)],
            ),
            DocumentSection(heading="Accomplishments", blocks=[ListBlock(items=report.accomplishments)]),
            DocumentSection(heading="Risks", blocks=[ListBlock(items=report.risks)]),
            DocumentSection(heading="Next steps", blocks=[ListBlock(items=report.next_steps)]),
        ],
    )


def main() -> None:
    rag_client = LewLMAppClient.from_http(BASE_URL, application_id="rag-chat")
    document_client = LewLMAppClient.from_http(BASE_URL, application_id="document-generator")

    rag_runtime = rag_client.runtime_info()
    document_runtime = document_client.runtime_info()
    assert rag_runtime.runtime_instance_id == document_runtime.runtime_instance_id
    print(f"shared runtime: {rag_runtime.runtime_instance_id}")

    if CHAT_MODEL_ID.startswith("<"):
        print("Set CHAT_MODEL_ID to run the two capability calls.")
        return

    rag_answer = rag_client.chat_completion(
        model=CHAT_MODEL_ID,
        messages=[ChatMessage(role="user", content="Summarize the retrieved project context.")],
    )
    print(rag_answer.choices[0].message.content)

    generated = document_client.responses(
        model=CHAT_MODEL_ID,
        input="Create a status report from the supplied project facts.",
        output_schema=StatusReportSpec.model_json_schema(),
    )
    report = StatusReportSpec.model_validate_json(generated.output_text)
    document = status_report_to_document_ir(report)
    print(document.model_dump_json(indent=2))
    # The document application next posts this generic DocumentIR to /v1/documents/generate,
    # then owns the resulting artifact's product lifecycle.


if __name__ == "__main__":
    main()
