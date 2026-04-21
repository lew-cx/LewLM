"""Chat, responses, and streaming routes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack
import logging
from pathlib import Path
from typing import TypeVar

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile

from lewlm.api.dependencies import get_services
from lewlm.api.message_normalization import normalize_chat_messages, normalize_response_input
from lewlm.api.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    ResponseChunk,
    ResponseCreateRequest,
    ResponseCreateResponse,
    ResponseOutputText,
)
from lewlm.core.chat import ChatStreamDelta
from lewlm.core.contracts import GenerateMessage, ReasoningOutput, ReasoningVisibility
from lewlm.core.errors import ConfigurationError
from lewlm.prompting import PromptCompilationRequest
from lewlm.security.workspace import secure_workspace


router = APIRouter(tags=["chat"])
_STREAM_HEARTBEAT_SECONDS = 1.0
StreamItemT = TypeVar("StreamItemT")
logger = logging.getLogger(__name__)

_CHAT_JSON_EXAMPLE = {
    "model": "local-chat-model",
    "messages": [
        {
            "role": "user",
            "content": "Return a grounded JSON summary of LewLM for a host application.",
        },
    ],
    "citation_context": {
        "sources": [
            {
                "source_id": "source-1",
                "path": "/tmp/integration-bundle.md",
                "source_type": "markdown",
                "source_name": "integration-bundle.md",
                "source_label": "LewLM Integration Bundle",
                "media_type": "text/markdown",
                "metadata": {},
            }
        ],
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "text": "This bundle is app-agnostic and checked in so host applications can map their own contracts onto LewLM without reading implementation files.",
                "source_id": "source-1",
                "section_id": "section-1",
                "source_label": "LewLM Integration Bundle",
                "section_label": "LewLM Integration Bundle / Summary",
                "section_heading": "Summary",
                "section_level": 1,
                "source_name": "integration-bundle.md",
                "source_path": "/tmp/integration-bundle.md",
                "source_type": "markdown",
                "metadata": {"chunk_index": 0},
            }
        ],
    },
    "response_format": {
        "type": "json_schema",
        "name": "lewlm_summary",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
    "include_prompt_trace": True,
    "stream": False,
}
_CHAT_MULTIPART_EXAMPLE = {
    "payload_json": (
        '{"messages":[{"role":"user","content":[{"type":"input_text","text":"Describe the upload named diagram_0."},'
        '{"type":"input_image","upload_name":"diagram_0"}]}],"stream":false}'
    ),
    "diagram_0": "(binary file part)",
}
_CHAT_STREAM_FRAME_EXAMPLE = (
    'data: {"id":"req-chat-001","object":"chat.completion.chunk","created":1760000000,'
    '"model":"local-chat-model","choices":[{"index":0,"delta":{"role":"assistant","content":"LewLM exposes a stable backend contract."}}],"citations":[]}\n\n'
)
_RESPONSES_JSON_EXAMPLE = {
    "model": "local-chat-model",
    "input": "Return a grounded JSON summary of LewLM for a host application.",
    "citation_context": _CHAT_JSON_EXAMPLE["citation_context"],
    "response_format": {
        "type": "json_schema",
        "name": "lewlm_summary",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
    "include_prompt_trace": True,
    "stream": False,
}
_RESPONSES_MULTIPART_EXAMPLE = {
    "payload_json": (
        '{"input":[{"role":"user","content":[{"type":"input_text","text":"Describe the upload named note_0."},'
        '{"type":"input_file","upload_name":"note_0"}]}],"stream":false}'
    ),
    "note_0": "(binary file part)",
}
_RESPONSES_STREAM_FRAME_EXAMPLE = (
    'data: {"id":"req-response-001","object":"response.chunk","created":1760000000,'
    '"model":"local-chat-model","delta":"LewLM exposes a stable backend contract.","done":false,"citations":[]}\n\n'
)


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        200: {
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": (
                            "Server-sent event frames whose `data:` payload lines contain "
                            "`ChatCompletionChunk` JSON objects, followed by `data: [DONE]`."
                        ),
                    },
                    "example": _CHAT_STREAM_FRAME_EXAMPLE,
                },
            },
        },
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": ChatCompletionRequest.model_json_schema(),
                    "example": _CHAT_JSON_EXAMPLE,
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["payload_json"],
                        "properties": {
                            "payload_json": {
                                "type": "string",
                                "description": "JSON-encoded ChatCompletionRequest payload.",
                            },
                        },
                        "additionalProperties": {
                            "type": "string",
                            "format": "binary",
                        },
                    },
                    "example": _CHAT_MULTIPART_EXAMPLE,
                },
            },
        },
    },
)
async def create_chat_completion(
    request: Request,
) -> ChatCompletionResponse | StreamingResponse:
    """Create a chat completion using the selected local runtime."""

    services = get_services(request)
    payload, uploaded_files, exit_stack = await _parse_structured_payload(request, ChatCompletionRequest, services=services)
    try:
        input_messages = await normalize_chat_messages(payload.messages, services, uploaded_files=uploaded_files)
        messages = _merge_session_messages(services, payload.session_id, input_messages)
        prompt_request = _prompt_request_from_payload(
            actor="api",
            system_prompt=payload.system_prompt,
            developer_prompt=payload.developer_prompt,
            pretext_path=payload.pretext_path,
            skills_path=payload.skills_path,
            response_format=payload.response_format,
            response_format_path=payload.response_format_path,
            output_schema=payload.output_schema,
            output_schema_path=payload.output_schema_path,
            tools=payload.tools,
            tools_path=payload.tools_path,
            mcp_tools=payload.mcp_tools,
            mcp_tools_path=payload.mcp_tools_path,
            include_trace=payload.include_prompt_trace,
        )
        if payload.stream:
            stream_session = await services.chat_orchestrator.stream(
                model_id=payload.model,
                messages=messages,
                citation_context=payload.citation_context,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                apply_serving_profile=payload.apply_serving_profile,
                reasoning_visibility=_reasoning_visibility_from_request(
                    payload.reasoning_visibility,
                    services.settings.reasoning_visibility,
                ),
                prompt_request=prompt_request,
            )
            return StreamingResponse(
                _chat_completion_stream(
                    stream_session,
                    on_close=exit_stack.close if exit_stack is not None else None,
                    on_complete=_session_completion_callback(
                        services,
                        session_id=payload.session_id,
                        request_kind="chat.completions",
                        input_messages=input_messages,
                        requested_model_id=payload.model,
                        resolved_model_id=stream_session.model_id,
                        max_tokens=payload.max_tokens,
                        temperature=payload.temperature,
                        finish_reason="stop",
                    ),
                ),
                media_type="text/event-stream",
            )

        execution = await services.chat_orchestrator.complete(
            model_id=payload.model,
            messages=messages,
            citation_context=payload.citation_context,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            apply_serving_profile=payload.apply_serving_profile,
            reasoning_visibility=_reasoning_visibility_from_request(
                payload.reasoning_visibility,
                services.settings.reasoning_visibility,
            ),
            prompt_request=prompt_request,
        )
    except Exception:
        if exit_stack is not None:
            exit_stack.close()
        raise

    try:
        _persist_session_turn(
            services,
            session_id=payload.session_id,
            request_kind="chat.completions",
            input_messages=input_messages,
            output_text=execution.response.output_text,
            requested_model_id=payload.model,
            resolved_model_id=execution.response.model_id,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            finish_reason=execution.response.finish_reason,
            usage=execution.response.usage,
            metadata=execution.request_metadata,
        )
    finally:
        if exit_stack is not None:
            exit_stack.close()
    return ChatCompletionResponse(
        id=execution.request_id,
        created=execution.created_at,
        model=execution.response.model_id,
        session_id=payload.session_id,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionChoiceMessage(
                    role="assistant",
                    content=execution.response.output_text,
                    reasoning=execution.response.reasoning,
                ),
                finish_reason=execution.response.finish_reason,
            ),
        ],
        usage=_completion_usage(execution.response.usage),
        metadata=execution.metadata,
        citations=execution.response.citations,
        structured_output=execution.structured_output,
        prompt_trace=execution.prompt_trace if payload.include_prompt_trace else None,
        serving_profile=execution.serving_profile,
    )


@router.post(
    "/v1/responses",
    response_model=ResponseCreateResponse,
    responses={
        200: {
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": (
                            "Server-sent event frames whose `data:` payload lines contain "
                            "`ResponseChunk` JSON objects, followed by `data: [DONE]`."
                        ),
                    },
                    "example": _RESPONSES_STREAM_FRAME_EXAMPLE,
                },
            },
        },
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": ResponseCreateRequest.model_json_schema(),
                    "example": _RESPONSES_JSON_EXAMPLE,
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["payload_json"],
                        "properties": {
                            "payload_json": {
                                "type": "string",
                                "description": "JSON-encoded ResponseCreateRequest payload.",
                            },
                        },
                        "additionalProperties": {
                            "type": "string",
                            "format": "binary",
                        },
                    },
                    "example": _RESPONSES_MULTIPART_EXAMPLE,
                },
            },
        },
    },
)
async def create_response(
    request: Request,
) -> ResponseCreateResponse | StreamingResponse:
    """Create a LewLM responses-style completion."""

    services = get_services(request)
    payload, uploaded_files, exit_stack = await _parse_structured_payload(request, ResponseCreateRequest, services=services)
    try:
        input_messages = await normalize_response_input(payload.input, services, uploaded_files=uploaded_files)
        messages = _merge_session_messages(services, payload.session_id, input_messages)
        prompt_request = _prompt_request_from_payload(
            actor="api",
            system_prompt=payload.system_prompt,
            developer_prompt=payload.developer_prompt,
            pretext_path=payload.pretext_path,
            skills_path=payload.skills_path,
            response_format=payload.response_format,
            response_format_path=payload.response_format_path,
            output_schema=payload.output_schema,
            output_schema_path=payload.output_schema_path,
            tools=payload.tools,
            tools_path=payload.tools_path,
            mcp_tools=payload.mcp_tools,
            mcp_tools_path=payload.mcp_tools_path,
            include_trace=payload.include_prompt_trace,
        )
        if payload.stream:
            stream_session = await services.chat_orchestrator.stream(
                model_id=payload.model,
                messages=messages,
                citation_context=payload.citation_context,
                max_tokens=payload.max_output_tokens,
                temperature=payload.temperature,
                apply_serving_profile=payload.apply_serving_profile,
                reasoning_visibility=_reasoning_visibility_from_request(
                    payload.reasoning_visibility,
                    services.settings.reasoning_visibility,
                ),
                prompt_request=prompt_request,
            )
            return StreamingResponse(
                _response_stream(
                    stream_session,
                    on_close=exit_stack.close if exit_stack is not None else None,
                    on_complete=_session_completion_callback(
                        services,
                        session_id=payload.session_id,
                        request_kind="responses",
                        input_messages=input_messages,
                        requested_model_id=payload.model,
                        resolved_model_id=stream_session.model_id,
                        max_tokens=payload.max_output_tokens,
                        temperature=payload.temperature,
                        finish_reason="stop",
                        metadata=stream_session.request_metadata,
                    ),
                ),
                media_type="text/event-stream",
            )

        execution = await services.chat_orchestrator.complete(
            model_id=payload.model,
            messages=messages,
            citation_context=payload.citation_context,
            max_tokens=payload.max_output_tokens,
            temperature=payload.temperature,
            apply_serving_profile=payload.apply_serving_profile,
            reasoning_visibility=_reasoning_visibility_from_request(
                payload.reasoning_visibility,
                services.settings.reasoning_visibility,
            ),
            prompt_request=prompt_request,
        )
    except Exception:
        if exit_stack is not None:
            exit_stack.close()
        raise

    try:
        _persist_session_turn(
            services,
            session_id=payload.session_id,
            request_kind="responses",
            input_messages=input_messages,
            output_text=execution.response.output_text,
            requested_model_id=payload.model,
            resolved_model_id=execution.response.model_id,
            max_tokens=payload.max_output_tokens,
            temperature=payload.temperature,
            finish_reason=execution.response.finish_reason,
            usage=execution.response.usage,
            metadata=execution.request_metadata,
        )
    finally:
        if exit_stack is not None:
            exit_stack.close()
    return ResponseCreateResponse(
        id=execution.request_id,
        created=execution.created_at,
        model=execution.response.model_id,
        session_id=payload.session_id,
        output=[
            ResponseOutputText(
                text=execution.response.output_text,
                reasoning=execution.response.reasoning,
            ),
        ],
        output_text=execution.response.output_text,
        usage=_completion_usage(execution.response.usage),
        metadata=execution.metadata,
        citations=execution.response.citations,
        structured_output=execution.structured_output,
        prompt_trace=execution.prompt_trace if payload.include_prompt_trace else None,
        serving_profile=execution.serving_profile,
    )


def _merge_session_messages(services, session_id: str | None, input_messages: list[GenerateMessage]) -> list[GenerateMessage]:
    if session_id is None:
        return input_messages
    return services.session_history_service.build_conversation_messages(
        session_id=session_id,
        new_messages=input_messages,
    )


def _persist_session_turn(
    services,
    *,
    session_id: str | None,
    request_kind: str,
    input_messages: list[GenerateMessage],
    output_text: str,
    requested_model_id: str | None,
    resolved_model_id: str,
    max_tokens: int,
    temperature: float,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    if session_id is None:
        return
    services.session_history_service.record_turn(
        session_id=session_id,
        request_kind=request_kind,
        input_messages=input_messages,
        response_message=GenerateMessage(role="assistant", content=output_text),
        requested_model_id=requested_model_id,
        model_id=resolved_model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        finish_reason=finish_reason,
        usage=usage,
        metadata=metadata,
    )


def _session_completion_callback(
    services,
    *,
    session_id: str | None,
    request_kind: str,
    input_messages: list[GenerateMessage],
    requested_model_id: str | None,
    resolved_model_id: str,
    max_tokens: int,
    temperature: float,
    finish_reason: str,
    metadata: dict[str, object] | None = None,
) -> Callable[[str], None] | None:
    if session_id is None:
        return None

    def callback(output_text: str) -> None:
        _persist_session_turn(
            services,
            session_id=session_id,
            request_kind=request_kind,
            input_messages=input_messages,
            output_text=output_text,
            requested_model_id=requested_model_id,
            resolved_model_id=resolved_model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            finish_reason=finish_reason,
            metadata=metadata,
        )

    return callback


async def _chat_completion_stream(stream_session, *, on_close=None, on_complete=None) -> AsyncIterator[str]:
    try:
        sent_role = False
        sent_serving_profile = False
        deltas: list[str] = []
        source_stream = stream_session.stream_items or _stream_items_from_content(stream_session.stream)
        async for item in _stream_with_heartbeat(source_stream):
            if item is None:
                yield ": keep-alive\n\n"
                continue
            if item.content is not None:
                deltas.append(item.content)
            chunk = ChatCompletionChunk(
                id=stream_session.request_id,
                created=stream_session.created_at,
                model=stream_session.model_id,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionDelta(
                            role=None if sent_role else "assistant",
                            content=item.content,
                            reasoning=_stream_reasoning_output(item.reasoning, stream_session.reasoning_visibility),
                        ),
                    ),
                ],
                serving_profile=stream_session.serving_profile if not sent_serving_profile else None,
            )
            sent_role = True
            sent_serving_profile = True
            yield f"data: {chunk.model_dump_json()}\n\n"
        if on_complete is not None:
            try:
                on_complete("".join(deltas))
            except Exception:
                logger.exception(
                    "Streaming chat session completion callback failed.",
                    extra={"request_id": stream_session.request_id, "model_id": stream_session.model_id},
                )
        final_chunk = ChatCompletionChunk(
            id=stream_session.request_id,
            created=stream_session.created_at,
            model=stream_session.model_id,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionDelta(reasoning=stream_session.reasoning),
                    finish_reason="stop",
                ),
            ],
            citations=stream_session.citations,
            metadata=stream_session.metadata,
            structured_output=stream_session.structured_output,
            serving_profile=stream_session.serving_profile if not sent_serving_profile else None,
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if on_close is not None:
            try:
                on_close()
            except Exception:
                logger.exception(
                    "Streaming chat cleanup callback failed.",
                    extra={"request_id": stream_session.request_id, "model_id": stream_session.model_id},
                )


async def _response_stream(stream_session, *, on_close=None, on_complete=None) -> AsyncIterator[str]:
    try:
        deltas: list[str] = []
        sent_serving_profile = False
        source_stream = stream_session.stream_items or _stream_items_from_content(stream_session.stream)
        async for item in _stream_with_heartbeat(source_stream):
            if item is None:
                yield ": keep-alive\n\n"
                continue
            if item.content is not None:
                deltas.append(item.content)
            chunk = ResponseChunk(
                id=stream_session.request_id,
                created=stream_session.created_at,
                model=stream_session.model_id,
                delta=item.content,
                reasoning=_stream_reasoning_output(item.reasoning, stream_session.reasoning_visibility),
                done=False,
                serving_profile=stream_session.serving_profile if not sent_serving_profile else None,
            )
            sent_serving_profile = True
            yield f"data: {chunk.model_dump_json()}\n\n"
        if on_complete is not None:
            try:
                on_complete("".join(deltas))
            except Exception:
                logger.exception(
                    "Streaming response session completion callback failed.",
                    extra={"request_id": stream_session.request_id, "model_id": stream_session.model_id},
                )
        yield (
            "data: "
            f"{ResponseChunk(id=stream_session.request_id, created=stream_session.created_at, model=stream_session.model_id, reasoning=stream_session.reasoning, done=True, citations=stream_session.citations, metadata=stream_session.metadata, structured_output=stream_session.structured_output, serving_profile=stream_session.serving_profile if not sent_serving_profile else None).model_dump_json()}\n\n"
        )
        yield "data: [DONE]\n\n"
    finally:
        if on_close is not None:
            try:
                on_close()
            except Exception:
                logger.exception(
                    "Streaming response cleanup callback failed.",
                    extra={"request_id": stream_session.request_id, "model_id": stream_session.model_id},
                )


async def _parse_structured_payload(request: Request, schema, *, services):
    content_type = request.headers.get("content-type", "")
    media_type = content_type.partition(";")[0].strip().casefold()
    if media_type == "multipart/form-data":
        form = await request.form()
        payload_json = form.get("payload_json")
        if not isinstance(payload_json, str) or not payload_json.strip():
            raise ConfigurationError("Multipart requests require a `payload_json` form field.")
        payload = schema.model_validate_json(payload_json)
        exit_stack = ExitStack()
        try:
            workspace = exit_stack.enter_context(secure_workspace(services.settings.temp_dir, prefix="api-upload-"))
            uploaded_files: dict[str, Path] = {}
            upload_index = 0
            for field_name, value in form.multi_items():
                if not isinstance(value, UploadFile):
                    continue
                try:
                    if field_name in uploaded_files:
                        raise ConfigurationError(
                            "Multipart upload field names must be unique.",
                            details={"field_name": field_name},
                        )
                    target_path = workspace / _safe_upload_name(index=upload_index, file_name=value.filename)
                    target_path.write_bytes(await value.read())
                    uploaded_files[field_name] = target_path
                    upload_index += 1
                finally:
                    await value.close()
            return payload, uploaded_files, exit_stack.pop_all()
        except Exception:
            exit_stack.close()
            raise

    body = await request.body()
    return schema.model_validate_json(body), {}, None


def _safe_upload_name(*, index: int, file_name: str | None) -> str:
    base_name = Path(file_name or f"upload-{index}").name
    sanitized = "".join(character if character.isalnum() or character in {".", "-", "_"} else "-" for character in base_name)
    cleaned = sanitized.strip("-.") or f"upload-{index}"
    return f"{index}-{cleaned}"


def _completion_usage(raw_usage: dict[str, int]) -> CompletionUsage:
    prompt_tokens = raw_usage.get("prompt_tokens", 0)
    completion_tokens = raw_usage.get("completion_tokens", 0)
    total_tokens = raw_usage.get("total_tokens", prompt_tokens + completion_tokens)
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _reasoning_visibility_from_request(
    requested_visibility: ReasoningVisibility | None,
    default_visibility: ReasoningVisibility,
) -> ReasoningVisibility:
    return requested_visibility or default_visibility


def _prompt_request_from_payload(
    *,
    actor: str,
    system_prompt: str | None,
    developer_prompt: str | None,
    pretext_path: str | None,
    skills_path: str | None,
    response_format,
    response_format_path: str | None,
    output_schema: dict[str, object] | None,
    output_schema_path: str | None,
    tools,
    tools_path: str | None,
    mcp_tools,
    mcp_tools_path: str | None,
    include_trace: bool,
) -> PromptCompilationRequest | None:
    prompt_request = PromptCompilationRequest(
        actor=actor,
        system_prompt=system_prompt,
        developer_prompt=developer_prompt,
        pretext_path=pretext_path,
        skills_path=skills_path,
        response_format=response_format,
        response_format_path=response_format_path,
        output_schema=output_schema,
        output_schema_path=output_schema_path,
        tools=list(tools),
        tools_path=tools_path,
        mcp_tools=list(mcp_tools),
        mcp_tools_path=mcp_tools_path,
        include_trace=include_trace,
    )
    if prompt_request.include_trace or prompt_request.has_requested_overrides():
        return prompt_request
    return None


async def _stream_with_heartbeat(
    stream: AsyncIterator[StreamItemT],
    *,
    heartbeat_seconds: float = _STREAM_HEARTBEAT_SECONDS,
) -> AsyncIterator[StreamItemT | None]:
    iterator = stream.__aiter__()
    pending: asyncio.Task[StreamItemT] | None = asyncio.create_task(anext(iterator))
    try:
        while pending is not None:
            try:
                item = await asyncio.wait_for(asyncio.shield(pending), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield None
                continue
            except StopAsyncIteration:
                pending = None
                break
            yield item
            pending = asyncio.create_task(anext(iterator))
    finally:
        if pending is not None and not pending.done():
            pending.cancel()


def _stream_reasoning_output(reasoning_delta: str | None, visibility: ReasoningVisibility):
    if reasoning_delta is None:
        return None
    return ReasoningOutput(
        visibility=visibility,
        available=True,
        content=reasoning_delta,
    )


async def _stream_items_from_content(stream: AsyncIterator[str]) -> AsyncIterator[ChatStreamDelta]:
    async for content in stream:
        yield ChatStreamDelta(content=content)
