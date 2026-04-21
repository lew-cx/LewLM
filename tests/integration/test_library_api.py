from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from conftest import FakeLlamaCppRuntime, FakeMLXAudioRuntime, FakeMLXSemanticRuntime, FakeMLXVisionRuntime
from lewlm import LewLM
from lewlm.cli.main import handle_config
from lewlm.conversion.models import ConversionPolicy, JobStatus
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import GenerateAttachment, GenerateMessage, ReasoningVisibility, RuntimeAffinity
from lewlm.core.errors import PrivacyModeError
from lewlm.documents.ingest.models import DocumentSourceType
from lewlm.documents.ir.models import DocumentOutputFormat, ListBlock
from lewlm.documents.skills.models import DOCUMENT_TRANSFORM_REQUEST_ADAPTER
from lewlm.prompting import PromptCompilationRequest
from lewlm.tools.models import DocumentGenerateToolRequest, GenerateDocumentToolInput


class TrackingLibraryRuntime(FakeLlamaCppRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.unload_calls: list[str] = []

    async def _unload_model(self, model_id: str) -> None:
        self.unload_calls.append(model_id)
        await super()._unload_model(model_id)


class ServingProfileLibraryRuntime(FakeMLXSemanticRuntime):
    def __init__(self, *, settings) -> None:
        super().__init__(settings=settings)

    async def _generate(self, request):
        if self.settings.prefill_token_batch_size != 256:
            await asyncio.sleep(0.002)
        response = await super()._generate(request)
        return response.model_copy(
            update={"output_text": f"{response.output_text} [prefill={self.settings.prefill_token_batch_size}]"},
        )

    async def _stream_generate(self, request):
        response = await self._generate(request)
        yield response.output_text


def _serving_profile_services(temp_settings):
    return bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_TEXT: ServingProfileLibraryRuntime(settings=temp_settings),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
            RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
        },
    )


def test_library_facade_supports_embeddable_workflows(
    sample_document_ir,
    services_with_fake_runtime_and_conversion,
) -> None:
    lewlm = LewLM(services=services_with_fake_runtime_and_conversion)

    scan_summary = lewlm.scan_models()
    assert scan_summary.discovered_count == 3

    inventory = lewlm.inventory()
    assert inventory.count == 3

    conversion_candidate = next(
        manifest for manifest in inventory.items if manifest.conversion_status.value == "requires_conversion"
    )
    job = lewlm.submit_conversion(model_id=conversion_candidate.model_id, policy=ConversionPolicy.BALANCED)
    job = lewlm.wait_for_job(job.job_id)
    assert job.status == JobStatus.COMPLETED

    runnable_model = next(manifest for manifest in lewlm.list_models() if manifest.conversion_status.value == "runnable")
    chat = lewlm.chat_sync(prompt="Hello from the package API", model_id=runnable_model.model_id)
    assert chat.response.output_text == "Echo: Hello from the package API"
    assert chat.metadata.model.resolved_model_id == runnable_model.model_id
    assert chat.metadata.routing.kind == "model_router"
    assert chat.metadata.timing.total_milliseconds >= 0
    assert chat.metadata.serving is not None
    assert chat.metadata.serving.phase == "completed"
    assert chat.metadata.serving.runtime_adapter_kind == "backend_native_batch"
    assert chat.request_metadata["serving"]["phase"] == "completed"
    assert chat.request_metadata["serving"]["runtime_adapter"]["kind"] == "backend_native_batch"

    artifact = lewlm.generate_document(
        sample_document_ir,
        output_format=DocumentOutputFormat.MARKDOWN,
        file_name="report.md",
    )
    assert artifact.file_name == "report.md"
    assert b"Quarterly Operations Summary" in artifact.content

    tool_envelope = lewlm.execute_tool(
        DocumentGenerateToolRequest(
            input=GenerateDocumentToolInput(
                output_format=DocumentOutputFormat.JSON,
                file_name="report.json",
                document=sample_document_ir,
            ),
        ),
    )
    assert tool_envelope.tool == "documents.generate"
    assert tool_envelope.result["output_format"] == "json"

    health = lewlm.health()
    assert health["status"] == "ok"
    assert health["install_profiles"]["active_profile_ids"][0] == "core_only"
    assert "readiness" in health
    assert health["storage"]["model_count"] == 3

    runtime_stats = lewlm.runtime_stats_sync()
    assert runtime_stats.runtime_policy == "balanced"
    assert runtime_stats.serving_core.total_sequences_completed >= 1
    assert any(sequence.request_id == chat.request_id for sequence in runtime_stats.serving_core.recent_sequences)

    with TestClient(lewlm.create_app()) as client:
        response = client.get("/v1/health")
    assert response.status_code == 200

    lewlm.close()
    services_with_fake_runtime_and_conversion.close()


def test_library_wait_for_job_async_supports_active_loops(
    services_with_fake_runtime_and_conversion,
) -> None:
    lewlm = LewLM(services=services_with_fake_runtime_and_conversion)
    lewlm.scan_models()
    inventory = lewlm.inventory()
    conversion_candidate = next(
        manifest for manifest in inventory.items if manifest.conversion_status.value == "requires_conversion"
    )
    job = lewlm.submit_conversion(model_id=conversion_candidate.model_id, policy=ConversionPolicy.BALANCED)

    async def run_wait() -> None:
        completed_job = await lewlm.wait_for_job_async(job.job_id)
        assert completed_job.status == JobStatus.COMPLETED

    asyncio.run(run_wait())

    lewlm.close()
    services_with_fake_runtime_and_conversion.close()


def test_library_facade_exposes_sessions_skills_and_tools(
    services_with_fake_runtime_session_enabled,
) -> None:
    lewlm = LewLM(services=services_with_fake_runtime_session_enabled)

    skills = lewlm.list_skills()
    tools = lewlm.list_tools()
    assert any(skill.name == "document_comparison" for skill in skills)
    assert any(tool.name == "documents.transform" for tool in tools)
    assert lewlm.get_skill("document_comparison").tool_name == "documents.transform"
    assert lewlm.get_tool("documents.transform").required_authorization == "document_transform"

    session = lewlm.create_session(title="Package API Session", context_policy="summary_and_last_turn")
    assert lewlm.get_session(session.session_id).title == "Package API Session"
    assert lewlm.list_sessions()[0].session_id == session.session_id

    bundle = lewlm.export_session(session.session_id)
    imported = lewlm.import_session(bundle, title="Imported Package Session")
    assert imported.title == "Imported Package Session"

    deleted = lewlm.delete_session(session.session_id)
    assert deleted.session_id == session.session_id
    assert [record.title for record in lewlm.list_sessions()] == ["Imported Package Session"]

    lewlm.close()
    services_with_fake_runtime_session_enabled.close()


def test_library_facade_supports_document_ingest_and_transform_workflows(
    temp_settings,
    sample_ingest_sources,
    meeting_transcript_notes_payload,
) -> None:
    lewlm = LewLM(temp_settings)

    ingest_result = lewlm.ingest_documents(
        sample_ingest_sources["markdown"],
        allowed_file_roots=(temp_settings.data_dir,),
        request_id="library-ingest-1",
    )
    transform_request = DOCUMENT_TRANSFORM_REQUEST_ADAPTER.validate_python(meeting_transcript_notes_payload)
    artifact = lewlm.transform_document(transform_request, request_id="library-transform-1")

    assert ingest_result.sources[0].source_type == DocumentSourceType.MARKDOWN
    assert ingest_result.sources[0].source_label == "sample.md"
    assert ingest_result.chunks[0].source_id == ingest_result.sources[0].source_id
    assert ingest_result.chunks[0].section_label.startswith("sample.md / ")
    assert any(
        isinstance(block, ListBlock)
        for section in ingest_result.document.sections
        for block in section.blocks
    )
    assert artifact.output_format == DocumentOutputFormat.MARKDOWN
    assert artifact.media_type == "text/markdown"
    assert artifact.file_name.endswith(".md")
    assert b"Project Kickoff Notes" in artifact.content

    lewlm.close()


def test_library_facade_blocks_session_history_when_privacy_mode_is_enabled(
    services_with_fake_runtime,
) -> None:
    lewlm = LewLM(services=services_with_fake_runtime)

    with pytest.raises(PrivacyModeError, match="privacy mode is enabled"):
        lewlm.create_session(title="Blocked Session")
    with pytest.raises(PrivacyModeError, match="privacy mode is enabled"):
        lewlm.list_sessions()

    lewlm.close()
    services_with_fake_runtime.close()


def test_library_facade_supports_reasoning_visibility_controls(
    services_with_fake_runtime,
) -> None:
    lewlm = LewLM(services=services_with_fake_runtime)
    scan_summary = lewlm.scan_models()
    assert scan_summary.discovered_count == 3
    runnable_model = next(manifest for manifest in lewlm.list_models() if manifest.conversion_status.value == "runnable")

    chat = lewlm.chat_sync(
        prompt="[emit-reasoning] Explain the package flow",
        model_id=runnable_model.model_id,
        reasoning_visibility=ReasoningVisibility.SUMMARIZED,
    )

    async def run_stream() -> tuple[object, str, str, object]:
        stream_session = await lewlm.stream_chat(
            prompt="[emit-reasoning] Stream the package flow",
            model_id=runnable_model.model_id,
            reasoning_visibility=ReasoningVisibility.RAW_MODEL_EMITTED,
        )
        streamed_text_parts: list[str] = []
        streamed_reasoning_parts: list[str] = []
        assert stream_session.stream_items is not None
        async for item in stream_session.stream_items:
            if item.content:
                streamed_text_parts.append(item.content)
            if item.reasoning:
                streamed_reasoning_parts.append(item.reasoning)
        return (
            stream_session,
            "".join(streamed_text_parts),
            "".join(streamed_reasoning_parts),
            stream_session.reasoning,
        )

    stream_session, streamed_text, streamed_reasoning, final_reasoning = asyncio.run(run_stream())

    assert chat.response.output_text == "Echo: Explain the package flow"
    assert chat.response.reasoning is not None
    assert chat.response.reasoning.visibility == ReasoningVisibility.SUMMARIZED
    assert chat.response.reasoning.summary == "Inspect the prompt before replying."
    assert streamed_text == "Echo: Stream the package flow"
    assert streamed_reasoning == "Inspect the prompt before replying."
    assert final_reasoning is not None
    assert final_reasoning.visibility == ReasoningVisibility.RAW_MODEL_EMITTED
    assert final_reasoning.content == "Inspect the prompt before replying."
    assert stream_session.request_metadata["serving"]["streaming"] is True
    assert stream_session.request_metadata["serving"]["phase"] == "completed"
    assert stream_session.metadata is not None
    assert stream_session.metadata.serving is not None
    assert stream_session.metadata.serving.streaming is True

    lewlm.close()
    services_with_fake_runtime.close()


def test_library_facade_applies_persisted_serving_profiles(
    temp_settings,
    sample_chat_models_root,
) -> None:
    services = _serving_profile_services(temp_settings)
    lewlm = LewLM(services=services)
    lewlm.scan_models()
    model_id = next(
        manifest.model_id
        for manifest in lewlm.list_models()
        if manifest.display_name == "qwen2.5-1.5b-instruct-mlx"
    )
    recommendation = asyncio.run(
        services.telemetry_service.autotune(
            model_id=model_id,
            prompt="Serving profile library probe",
        ),
    )

    chat = lewlm.chat_sync(prompt="Library serving profile", model_id=model_id)
    assert chat.serving_profile is not None
    assert chat.serving_profile.profile_id == recommendation.profile_id
    assert chat.serving_profile.accepted_settings["prefill_token_batch_size"] == 256
    assert chat.response.output_text.endswith("[prefill=256]")
    assert chat.request_metadata["serving_profile"]["profile_id"] == recommendation.profile_id

    disabled_chat = lewlm.chat_sync(
        prompt="Library serving profile disabled",
        model_id=model_id,
        apply_serving_profile=False,
    )
    assert disabled_chat.serving_profile is not None
    assert disabled_chat.serving_profile.status == "disabled"
    assert disabled_chat.response.output_text.endswith("[prefill=512]")

    async def run_stream() -> tuple[object, str]:
        session = await lewlm.stream_chat(prompt="Library serving profile stream", model_id=model_id)
        streamed_output = "".join([chunk async for chunk in session.stream])
        return session, streamed_output

    stream_session, streamed_output = asyncio.run(run_stream())
    assert stream_session.serving_profile is not None
    assert stream_session.serving_profile.profile_id == recommendation.profile_id
    assert stream_session.request_metadata["serving_profile"]["profile_id"] == recommendation.profile_id
    assert streamed_output == "Echo: Library serving profile stream"

    lewlm.close()
    services.close()


def test_library_facade_selects_distinct_multimodal_serving_profiles_by_workload(
    temp_settings,
    sample_chat_models_root,
) -> None:
    services = _serving_profile_services(temp_settings)
    lewlm = LewLM(services=services)
    lewlm.scan_models()
    vision_model_id = next(
        manifest.model_id
        for manifest in lewlm.list_models()
        if manifest.display_name == "qwen2-vl-vision-mlx"
    )
    text_profile = asyncio.run(
        services.telemetry_service.autotune(
            model_id=vision_model_id,
            prompt="Text-only multimodal serving profile",
            workload_class="text_only_multimodal",
        ),
    )
    image_profile = asyncio.run(
        services.telemetry_service.autotune(
            model_id=vision_model_id,
            prompt="Image-conditioned multimodal serving profile",
            workload_class="single_image",
        ),
    )
    image_path = Path(temp_settings.data_dir) / "sample-image.png"
    image_path.write_bytes(b"image-bytes")

    text_chat = lewlm.chat_sync(prompt="Use the multimodal model without attachments", model_id=vision_model_id)
    image_chat = lewlm.chat_sync(
        messages=[
            GenerateMessage(
                role="user",
                content="Describe the attached image",
                attachments=[
                    GenerateAttachment(
                        attachment_type="image",
                        name=image_path.name,
                        source_path=str(image_path),
                        media_type="image/png",
                    ),
                ],
            ),
        ],
        model_id=vision_model_id,
    )

    assert text_chat.serving_profile is not None
    assert text_chat.serving_profile.profile_id == text_profile.profile_id
    assert text_chat.serving_profile.workload_class == "text_only_multimodal"
    assert image_chat.serving_profile is not None
    assert image_chat.serving_profile.profile_id == image_profile.profile_id
    assert image_chat.serving_profile.workload_class == "single_image"

    lewlm.close()
    services.close()


def test_library_facade_routes_once_when_serving_profile_lookup_misses(
    temp_settings,
    sample_chat_models_root,
) -> None:
    services = _serving_profile_services(temp_settings)
    lewlm = LewLM(services=services)
    lewlm.scan_models()
    model_id = next(
        manifest.model_id
        for manifest in lewlm.list_models()
        if manifest.display_name == "qwen2.5-1.5b-instruct-mlx"
    )
    route_calls = 0
    original_route_chat = services.model_router.route_chat

    def counting_route_chat(*args, **kwargs):
        nonlocal route_calls
        route_calls += 1
        return original_route_chat(*args, **kwargs)

    services.model_router.route_chat = counting_route_chat
    try:
        chat = lewlm.chat_sync(prompt="One route only", model_id=model_id)
    finally:
        services.model_router.route_chat = original_route_chat

    assert chat.serving_profile is not None
    assert chat.serving_profile.status == "not_found"
    assert route_calls == 1

    lewlm.close()
    services.close()


def test_library_facade_only_stores_prompt_trace_metadata_when_requested(
    temp_settings,
    sample_chat_models_root,
) -> None:
    services = _serving_profile_services(temp_settings)
    lewlm = LewLM(services=services)
    lewlm.scan_models()
    model_id = next(
        manifest.model_id
        for manifest in lewlm.list_models()
        if manifest.display_name == "qwen2.5-1.5b-instruct-mlx"
    )

    default_chat = lewlm.chat_sync(prompt="No trace metadata", model_id=model_id)
    traced_chat = lewlm.chat_sync(
        prompt="Trace metadata",
        model_id=model_id,
        prompt_request=PromptCompilationRequest(actor="system", include_trace=True),
    )

    assert "prompt_trace" not in default_chat.request_metadata
    assert traced_chat.request_metadata["prompt_trace"]["message_count"] == 1
    assert traced_chat.prompt_trace.message_count == 1

    lewlm.close()
    services.close()


def test_chat_and_responses_api_surface_persisted_serving_profiles(
    temp_settings,
    sample_chat_models_root,
) -> None:
    services = _serving_profile_services(temp_settings)
    lewlm = LewLM(services=services)
    lewlm.scan_models()
    model_id = next(
        manifest.model_id
        for manifest in lewlm.list_models()
        if manifest.display_name == "qwen2.5-1.5b-instruct-mlx"
    )
    recommendation = asyncio.run(
        services.telemetry_service.autotune(
            model_id=model_id,
            prompt="Serving profile API probe",
        ),
    )

    with TestClient(lewlm.create_app()) as client:
        chat_response = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "API serving profile"}],
            },
        )
        responses_api = client.post(
            "/v1/responses",
            json={
                "model": model_id,
                "input": "Responses serving profile",
            },
        )
        disabled_chat_response = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Disabled API serving profile"}],
                "apply_serving_profile": False,
            },
        )
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Streaming API serving profile"}],
                "stream": True,
            },
        ) as stream_response:
            data_lines = [
                line.removeprefix("data: ")
                for line in stream_response.iter_lines()
                if line.startswith("data: ")
            ]

    assert chat_response.status_code == 200
    chat_payload = chat_response.json()
    expected_prefill_batch_size = recommendation.effective_settings["prefill_token_batch_size"]
    assert chat_payload["serving_profile"]["profile_id"] == recommendation.profile_id
    assert chat_payload["serving_profile"]["effective_settings"]["prefill_token_batch_size"] == expected_prefill_batch_size
    assert chat_payload["choices"][0]["message"]["content"].endswith(f"[prefill={expected_prefill_batch_size}]")

    assert responses_api.status_code == 200
    response_payload = responses_api.json()
    assert response_payload["serving_profile"]["profile_id"] == recommendation.profile_id
    assert response_payload["output_text"].endswith(f"[prefill={expected_prefill_batch_size}]")

    assert disabled_chat_response.status_code == 200
    disabled_payload = disabled_chat_response.json()
    assert disabled_payload["serving_profile"]["status"] == "disabled"
    assert disabled_payload["choices"][0]["message"]["content"].endswith("[prefill=512]")

    assert data_lines[-1] == "[DONE]"
    streamed_chunks = [json.loads(line) for line in data_lines[:-1]]
    assert streamed_chunks[0]["serving_profile"]["profile_id"] == recommendation.profile_id

    lewlm.close()
    services.close()


def test_library_facade_close_releases_only_owned_services(temp_settings, monkeypatch) -> None:
    lewlm = LewLM(temp_settings)
    close_calls = 0
    original_close = lewlm.services.conversion_service.close

    def wrapped_close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    monkeypatch.setattr(lewlm.services.conversion_service, "close", wrapped_close)

    lewlm.close()
    lewlm.close()

    assert close_calls == 1


def test_library_facade_close_preserves_injected_services(
    services_with_fake_runtime,
    monkeypatch,
) -> None:
    lewlm = LewLM(services=services_with_fake_runtime)
    close_calls = 0
    original_close = services_with_fake_runtime.conversion_service.close

    def wrapped_close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    monkeypatch.setattr(services_with_fake_runtime.conversion_service, "close", wrapped_close)

    lewlm.close()

    assert close_calls == 0

    services_with_fake_runtime.close()
    assert close_calls == 1


def test_library_sync_helpers_raise_clear_errors_inside_running_loop(
    services_with_fake_runtime,
) -> None:
    lewlm = LewLM(services=services_with_fake_runtime)
    lewlm.scan_models()
    runnable_model = next(manifest for manifest in lewlm.list_models() if manifest.conversion_status.value == "runnable")

    async def run_checks() -> None:
        with pytest.raises(RuntimeError, match=r"LewLM\.wait_for_job cannot run inside an active asyncio event loop"):
            lewlm.wait_for_job("job-123")
        with pytest.raises(RuntimeError, match=r"LewLM\.chat_sync cannot run inside an active asyncio event loop"):
            lewlm.chat_sync(prompt="hello", model_id=runnable_model.model_id)
        with pytest.raises(RuntimeError, match=r"LewLM\.warm_model_sync cannot run inside an active asyncio event loop"):
            lewlm.warm_model_sync(runnable_model.model_id)
        with pytest.raises(RuntimeError, match=r"LewLM\.unload_model_sync cannot run inside an active asyncio event loop"):
            lewlm.unload_model_sync(runnable_model.model_id)
        with pytest.raises(RuntimeError, match=r"LewLM\.runtime_stats_sync cannot run inside an active asyncio event loop"):
            lewlm.runtime_stats_sync()

    asyncio.run(run_checks())

    lewlm.close()
    services_with_fake_runtime.close()


def test_library_close_raises_clear_error_inside_running_loop(
    temp_settings,
    sample_models_root,
) -> None:
    runtime = TrackingLibraryRuntime()
    lewlm = LewLM(temp_settings, runtime_overrides={RuntimeAffinity.LLAMACPP: runtime})

    async def run_checks() -> None:
        with pytest.raises(RuntimeError, match=r"LewLM\.close cannot run inside an active asyncio event loop"):
            lewlm.close()
        await lewlm.aclose()

    asyncio.run(run_checks())


def test_library_aclose_unloads_loaded_runtime_models(
    temp_settings,
    sample_models_root,
) -> None:
    runtime = TrackingLibraryRuntime()
    lewlm = LewLM(temp_settings, runtime_overrides={RuntimeAffinity.LLAMACPP: runtime})
    lewlm.scan_models()
    runnable_model = next(manifest for manifest in lewlm.list_models() if manifest.conversion_status.value == "runnable")

    async def run_flow() -> None:
        await lewlm.warm_model(runnable_model.model_id)
        assert runtime.loaded_model_ids == (runnable_model.model_id,)
        await lewlm.aclose()

    asyncio.run(run_flow())

    assert runtime.loaded_model_ids == ()
    assert runtime.unload_calls == [runnable_model.model_id]


def test_config_command_prints_operator_focused_summary(temp_settings, capsys) -> None:
    exit_code = handle_config(argparse.Namespace(json=False), temp_settings)

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "model roots:" in captured
    assert "validation manifests:" in captured
    assert "release bundle command:" in captured
