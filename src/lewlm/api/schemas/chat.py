"""Chat and response API schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from lewlm.core.citations import CitationContextPackage, GeneratedCitationReference
from lewlm.core.contracts import ReasoningOutput, ReasoningVisibility, SamplingControls
from lewlm.core.execution_metadata import ExecutionMetadata
from lewlm.prompting import PromptCompilationTrace, PromptMCPToolDefinition, PromptToolDefinition
from lewlm.serving_profiles import ServingProfileApplication
from lewlm.structured_output import StructuredOutputRequest, StructuredOutputResult
from lewlm.tool_calls import ToolCallParseResult


class InputTextPart(BaseModel):
    type: Literal["text", "input_text"]
    text: str


class InputImagePart(BaseModel):
    type: Literal["input_image", "image"]
    path: str | None = None
    upload_name: str | None = None
    detail: Literal["auto", "low", "high"] = "auto"

    @model_validator(mode="after")
    def _validate_source(self) -> "InputImagePart":
        if not self.path and not self.upload_name:
            raise ValueError("Image content parts require either `path` or `upload_name`.")
        return self


class InputFilePart(BaseModel):
    type: Literal["input_file", "file"]
    path: str | None = None
    upload_name: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> "InputFilePart":
        if not self.path and not self.upload_name:
            raise ValueError("File content parts require either `path` or `upload_name`.")
        return self


class InputAudioPart(BaseModel):
    type: Literal["input_audio", "audio"]
    path: str | None = None
    upload_name: str | None = None
    language: str | None = None
    prompt: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> "InputAudioPart":
        if not self.path and not self.upload_name:
            raise ValueError("Audio content parts require either `path` or `upload_name`.")
        return self


MessageContentPart = Annotated[
    InputTextPart | InputImagePart | InputFilePart | InputAudioPart,
    Field(discriminator="type"),
]


#: The roles LewLM's prompt templates know how to render. Leaving this open
#: meant an unrecognized role reached the template and was serialized as a
#: literal tag the model had never seen, with nothing raised to say so.
MessageRole = Literal["system", "developer", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: MessageRole = "user"
    content: str | list[MessageContentPart]


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    measured: bool = Field(
        default=True,
        description=(
            "True when counts came from the model's own tokenizer. False when the "
            "backend exposed no tokenizer and LewLM had to estimate."
        ),
    )
    cached_tokens: int | None = Field(
        default=None,
        description=(
            "Prompt tokens the backend reported as served from its own prefix cache "
            "(OpenAI-style prompt_tokens_details.cached_tokens). Absent when the "
            "backend exposes no such counter; LewLM never infers it."
        ),
    )


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    session_id: str | None = None
    correlation_id: str | None = Field(
        default=None,
        description="Caller correlation identifier echoed back through metadata and events.",
    )
    messages: list[ChatMessage]
    citation_context: CitationContextPackage | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    sampling: SamplingControls | None = Field(
        default=None,
        description=(
            "Decode controls beyond temperature. Backends differ in support; "
            "`metadata.sampling` reports which were applied and which were not."
        ),
    )
    apply_serving_profile: bool = True
    stream: bool = False
    reasoning_visibility: ReasoningVisibility | None = None
    system_prompt: str | None = None
    developer_prompt: str | None = None
    pretext_path: str | None = None
    skills_path: str | None = None
    response_format: StructuredOutputRequest | None = None
    response_format_path: str | None = None
    output_schema: dict[str, Any] | None = None
    output_schema_path: str | None = None
    tools: list[PromptToolDefinition] = Field(default_factory=list)
    tools_path: str | None = None
    tool_choice: Literal["auto", "none", "required"] | dict[str, Any] | None = None
    mcp_tools: list[PromptMCPToolDefinition] = Field(default_factory=list)
    mcp_tools_path: str | None = None
    include_prompt_trace: bool = False

    @model_validator(mode="after")
    def _validate_structured_output_inputs(self) -> "ChatCompletionRequest":
        if self.response_format is not None and self.response_format_path is not None:
            raise ValueError("Specify either `response_format` or `response_format_path`, not both.")
        if (self.response_format is not None or self.response_format_path is not None) and (
            self.output_schema is not None or self.output_schema_path is not None
        ):
            raise ValueError(
                "Specify either `response_format` / `response_format_path` or legacy "
                "`output_schema` / `output_schema_path`, not both.",
            )
        if self.tool_choice is not None and not (self.tools or self.tools_path or self.mcp_tools or self.mcp_tools_path):
            raise ValueError("`tool_choice` requires at least one declared tool or tool definition path.")
        return self


class ChatCompletionChoiceMessage(BaseModel):
    role: str
    content: str
    reasoning: ReasoningOutput | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    session_id: str | None = None
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage
    metadata: ExecutionMetadata
    citations: list[GeneratedCitationReference] = Field(default_factory=list)
    structured_output: StructuredOutputResult | None = None
    tool_calls: ToolCallParseResult | None = None
    prompt_trace: PromptCompilationTrace | None = None
    serving_profile: ServingProfileApplication | None = None


class StreamErrorEnvelope(BaseModel):
    """Why a stream ended before its normal terminal chunk.

    Carried on a final chunk whose `finish_reason` is `error` (chat) or whose
    `done` is true (responses), followed by `[DONE]`, so a client sees a
    structured failure instead of a dropped connection. Any output already
    delivered stands; LewLM never replays the request. Raw backend payloads
    and credentials are never included.
    """

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    partial_output: bool = Field(
        default=False,
        description="True when at least one content delta had been delivered before the failure.",
    )


class ChatCompletionDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: ReasoningOutput | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    citations: list[GeneratedCitationReference] = Field(default_factory=list)
    usage: CompletionUsage | None = Field(
        default=None,
        description="Token accounting. Present on the final chunk only, since it is not knowable before then.",
    )
    metadata: ExecutionMetadata | None = None
    structured_output: StructuredOutputResult | None = None
    tool_calls: ToolCallParseResult | None = None
    error: StreamErrorEnvelope | None = Field(
        default=None,
        description="Present only on a terminal chunk with finish_reason `error`: the stream ended incompletely.",
    )
    prompt_trace: PromptCompilationTrace | None = Field(
        default=None,
        description=(
            "Compiled-prompt trace when `include_prompt_trace` was set. Present on the final chunk "
            "only, so inspecting the prompt does not cost the caller its stream."
        ),
    )
    serving_profile: ServingProfileApplication | None = None


class ResponseInputMessage(BaseModel):
    role: MessageRole = "user"
    content: str | list[MessageContentPart]


class ResponseCreateRequest(BaseModel):
    model: str | None = None
    session_id: str | None = None
    correlation_id: str | None = Field(
        default=None,
        description="Caller correlation identifier echoed back through metadata and events.",
    )
    input: str | list[ResponseInputMessage]
    citation_context: CitationContextPackage | None = None
    max_output_tokens: int = 512
    temperature: float = 0.7
    sampling: SamplingControls | None = Field(
        default=None,
        description=(
            "Decode controls beyond temperature. Backends differ in support; "
            "`metadata.sampling` reports which were applied and which were not."
        ),
    )
    apply_serving_profile: bool = True
    stream: bool = False
    reasoning_visibility: ReasoningVisibility | None = None
    system_prompt: str | None = None
    developer_prompt: str | None = None
    pretext_path: str | None = None
    skills_path: str | None = None
    response_format: StructuredOutputRequest | None = None
    response_format_path: str | None = None
    output_schema: dict[str, Any] | None = None
    output_schema_path: str | None = None
    tools: list[PromptToolDefinition] = Field(default_factory=list)
    tools_path: str | None = None
    tool_choice: Literal["auto", "none", "required"] | dict[str, Any] | None = None
    mcp_tools: list[PromptMCPToolDefinition] = Field(default_factory=list)
    mcp_tools_path: str | None = None
    include_prompt_trace: bool = False

    @model_validator(mode="after")
    def _validate_structured_output_inputs(self) -> "ResponseCreateRequest":
        if self.response_format is not None and self.response_format_path is not None:
            raise ValueError("Specify either `response_format` or `response_format_path`, not both.")
        if (self.response_format is not None or self.response_format_path is not None) and (
            self.output_schema is not None or self.output_schema_path is not None
        ):
            raise ValueError(
                "Specify either `response_format` / `response_format_path` or legacy "
                "`output_schema` / `output_schema_path`, not both.",
            )
        if self.tool_choice is not None and not (self.tools or self.tools_path or self.mcp_tools or self.mcp_tools_path):
            raise ValueError("`tool_choice` requires at least one declared tool or tool definition path.")
        return self


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    reasoning: ReasoningOutput | None = None


class ResponseCreateResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created: int
    model: str
    session_id: str | None = None
    output: list[ResponseOutputText]
    output_text: str
    usage: CompletionUsage = Field(default_factory=CompletionUsage)
    metadata: ExecutionMetadata
    citations: list[GeneratedCitationReference] = Field(default_factory=list)
    structured_output: StructuredOutputResult | None = None
    tool_calls: ToolCallParseResult | None = None
    prompt_trace: PromptCompilationTrace | None = None
    serving_profile: ServingProfileApplication | None = None


class ResponseChunk(BaseModel):
    id: str
    object: Literal["response.chunk"] = "response.chunk"
    created: int
    model: str
    delta: str | None = None
    reasoning: ReasoningOutput | None = None
    tool_call_delta: list[dict[str, Any]] | None = None
    done: bool = False
    citations: list[GeneratedCitationReference] = Field(default_factory=list)
    usage: CompletionUsage | None = Field(
        default=None,
        description="Token accounting. Present on the final chunk only, since it is not knowable before then.",
    )
    metadata: ExecutionMetadata | None = None
    structured_output: StructuredOutputResult | None = None
    tool_calls: ToolCallParseResult | None = None
    error: StreamErrorEnvelope | None = Field(
        default=None,
        description="Present only on a terminal chunk (`done` true) when the stream ended incompletely.",
    )
    prompt_trace: PromptCompilationTrace | None = Field(
        default=None,
        description=(
            "Compiled-prompt trace when `include_prompt_trace` was set. Present on the final chunk "
            "only, so inspecting the prompt does not cost the caller its stream."
        ),
    )
    serving_profile: ServingProfileApplication | None = None
