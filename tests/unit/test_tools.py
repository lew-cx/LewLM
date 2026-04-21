from __future__ import annotations

import asyncio
import json

import pytest

from lewlm.core.errors import DocumentValidationError
from lewlm.documents.service import DocumentGenerationService
from lewlm.documents.ingest.service import DocumentIngestService
from lewlm.documents.skills.catalog import DocumentSkillCatalogService
from lewlm.documents.skills.service import DocumentTransformService
from lewlm.events.bus import EventBus
from lewlm.security.audit import AuditLogger
from lewlm.security.authorization import ToolAuthorizer
from lewlm.storage.metadata import MetadataStore
from lewlm.tools import ToolCatalogService, ToolExecutionService, parse_tool_execution_request


def _build_tool_execution_service(temp_settings) -> ToolExecutionService:
    temp_settings.prepare_directories()
    audit_logger = AuditLogger(temp_settings)
    metadata_store = MetadataStore(temp_settings.database_path)
    metadata_store.initialize()
    return ToolExecutionService(
        settings=temp_settings,
        tool_catalog=ToolCatalogService(),
        document_generation_service=DocumentGenerationService(),
        document_ingest_service=DocumentIngestService(
            workspace_root=temp_settings.temp_dir,
            sandbox_enabled=False,
        ),
        document_transform_service=DocumentTransformService(),
        tool_authorizer=ToolAuthorizer(settings=temp_settings, audit_logger=audit_logger),
        audit_logger=audit_logger,
        metadata_store=metadata_store,
    )


def test_skill_and_tool_catalogs_expose_builtin_metadata() -> None:
    skill_catalog = DocumentSkillCatalogService()
    tool_catalog = ToolCatalogService()

    assert any(skill.name == "document_comparison" for skill in skill_catalog.list_skills())
    meeting_skill = skill_catalog.get_skill("meeting_transcript_notes")
    assert meeting_skill.example_path == "examples/meeting-transcript-notes.json"
    assert "markdown" in [output_format.value for output_format in meeting_skill.supported_output_formats]
    memo_skill = skill_catalog.get_skill("long_document_memo")
    assert memo_skill.example_path == "examples/long-document-memo.json"
    assert "memo" in memo_skill.tags
    branded_skill = skill_catalog.get_skill("branded_document_template")
    assert branded_skill.example_path == "examples/branded-document-template.json"
    assert "branding" in branded_skill.tags
    ocr_skill = skill_catalog.get_skill("ocr_assisted_extraction")
    assert ocr_skill.example_path == "examples/ocr-assisted-extraction.json"
    assert "ocr" in ocr_skill.tags
    cleanup_skill = skill_catalog.get_skill("speech_transcript_cleanup")
    assert cleanup_skill.example_path == "examples/speech-transcript-cleanup.json"
    assert "speech" in cleanup_skill.tags
    assert tool_catalog.get_tool("documents.transform").required_authorization == "document_transform"
    assert tool_catalog.get_tool("document_transform").name == "documents.transform"


def test_tool_execution_service_runs_registered_transform_tool(
    temp_settings,
    contract_transform_payload: dict[str, object],
) -> None:
    service = _build_tool_execution_service(temp_settings)
    payload = json.dumps({"tool": "documents.transform", "input": contract_transform_payload}, indent=2)
    request = parse_tool_execution_request(payload)

    response = service.execute(
        request,
        actor="cli",
        allowed_file_roots=(temp_settings.data_dir,),
        base_dir=temp_settings.data_dir,
    )

    assert response.request_id
    assert response.tool == "documents.transform"
    assert response.trace.required_authorization == "document_transform"
    assert response.trace.details["sandboxed"] is True
    assert response.result["skill"] == "contract_text_replacement"
    assert response.result["file_name"].endswith(".docx")
    assert response.result["content_base64"]


def test_tool_execution_service_replays_idempotent_requests(
    temp_settings,
    contract_transform_payload: dict[str, object],
) -> None:
    service = _build_tool_execution_service(temp_settings)
    payload = dict(contract_transform_payload)
    payload["idempotency_key"] = "tool-replay-1"
    request = parse_tool_execution_request(json.dumps({"tool": "documents.transform", "input": payload}, indent=2))

    first = service.execute(
        request,
        actor="cli",
        allowed_file_roots=(temp_settings.data_dir,),
        base_dir=temp_settings.data_dir,
    )
    second = service.execute(
        request,
        actor="cli",
        allowed_file_roots=(temp_settings.data_dir,),
        base_dir=temp_settings.data_dir,
    )

    assert first.idempotent_replay is False
    assert second.idempotent_replay is True
    assert first.idempotency_key == "tool-replay-1"
    assert second.idempotency_key == "tool-replay-1"
    assert second.request_id == first.request_id
    assert second.result == first.result


async def test_tool_execution_service_emits_request_correlated_events(
    temp_settings,
    contract_transform_payload: dict[str, object],
) -> None:
    temp_settings.prepare_directories()
    audit_logger = AuditLogger(temp_settings)
    metadata_store = MetadataStore(temp_settings.database_path)
    metadata_store.initialize()
    event_bus = EventBus()
    event_bus.attach_loop(asyncio.get_running_loop())
    service = ToolExecutionService(
        settings=temp_settings,
        tool_catalog=ToolCatalogService(),
        document_generation_service=DocumentGenerationService(event_bus=event_bus),
        document_ingest_service=DocumentIngestService(
            workspace_root=temp_settings.temp_dir,
            sandbox_enabled=False,
            event_bus=event_bus,
        ),
        document_transform_service=DocumentTransformService(event_bus=event_bus),
        tool_authorizer=ToolAuthorizer(settings=temp_settings, audit_logger=audit_logger),
        audit_logger=audit_logger,
        event_bus=event_bus,
        metadata_store=metadata_store,
    )
    subscription = event_bus.subscribe()
    payload = json.dumps({"tool": "documents.transform", "input": contract_transform_payload}, indent=2)
    request = parse_tool_execution_request(payload)

    response = service.execute(
        request,
        actor="cli",
        allowed_file_roots=(temp_settings.data_dir,),
        base_dir=temp_settings.data_dir,
    )
    events = [await subscription.get() for _ in range(10)]
    subscription.close()

    assert response.request_id
    assert [events[0].type.value, events[1].type.value, events[-1].type.value] == [
        "tool.pending",
        "tool.started",
        "tool.finished",
    ]
    assert [event.type.value for event in events] == [
        "tool.pending",
        "tool.started",
        "document.transform.started",
        "operation.progress",
        "document.render.started",
        "operation.progress",
        "operation.progress",
        "document.render.completed",
        "document.transform.completed",
        "tool.finished",
    ]
    assert all(event.payload["request_id"] == response.request_id for event in events)


async def test_tool_execution_service_replays_failure_events_from_sandbox(
    temp_settings,
    file_template_transform_payload: dict[str, object],
) -> None:
    temp_settings.prepare_directories()
    audit_logger = AuditLogger(temp_settings)
    metadata_store = MetadataStore(temp_settings.database_path)
    metadata_store.initialize()
    event_bus = EventBus()
    event_bus.attach_loop(asyncio.get_running_loop())
    service = ToolExecutionService(
        settings=temp_settings,
        tool_catalog=ToolCatalogService(),
        document_generation_service=DocumentGenerationService(event_bus=event_bus),
        document_ingest_service=DocumentIngestService(
            workspace_root=temp_settings.temp_dir,
            sandbox_enabled=False,
            event_bus=event_bus,
        ),
        document_transform_service=DocumentTransformService(event_bus=event_bus),
        tool_authorizer=ToolAuthorizer(settings=temp_settings, audit_logger=audit_logger),
        audit_logger=audit_logger,
        event_bus=event_bus,
        metadata_store=metadata_store,
    )
    subscription = event_bus.subscribe()
    payload = {
        "tool": "documents.transform",
        "input": {
            **file_template_transform_payload,
            "input": {"replacements": {"client": "Acme Corp", "owner": "LewLM"}},
        },
    }
    request = parse_tool_execution_request(json.dumps(payload, indent=2))

    with pytest.raises(DocumentValidationError):
        service.execute(
            request,
            actor="cli",
            allowed_file_roots=(temp_settings.data_dir,),
            base_dir=temp_settings.data_dir,
        )
    events = [await subscription.get() for _ in range(5)]
    subscription.close()

    assert [event.type.value for event in events] == [
        "tool.pending",
        "tool.started",
        "document.transform.started",
        "document.transform.failed",
        "tool.failed",
    ]
