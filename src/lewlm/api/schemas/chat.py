"""Chat and response API schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from lewlm.core.citations import CitationContextPackage, GeneratedCitationReference
from lewlm.core.contracts import ReasoningOutput, ReasoningVisibility
from lewlm.core.execution_metadata import ExecutionMetadata
from lewlm.prompting import PromptCompilationTrace, PromptMCPToolDefinition, PromptToolDefinition
from lewlm.serving_profiles import ServingProfileApplication
from lewlm.structured_output import StructuredOutputRequest, StructuredOutputResult


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


class ChatMessage(BaseModel):
    role: str
    content: str | list[MessageContentPart]


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    session_id: str | None = None
    messages: list[ChatMessage]
    citation_context: CitationContextPackage | None = None
    max_tokens: int = 512
    temperature: float = 0.7
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
    prompt_trace: PromptCompilationTrace | None = None
    serving_profile: ServingProfileApplication | None = None


class ChatCompletionDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: ReasoningOutput | None = None


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
    metadata: ExecutionMetadata | None = None
    structured_output: StructuredOutputResult | None = None
    serving_profile: ServingProfileApplication | None = None


class ResponseInputMessage(BaseModel):
    role: str = "user"
    content: str | list[MessageContentPart]


class ResponseCreateRequest(BaseModel):
    model: str | None = None
    session_id: str | None = None
    input: str | list[ResponseInputMessage]
    citation_context: CitationContextPackage | None = None
    max_output_tokens: int = 512
    temperature: float = 0.7
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
    prompt_trace: PromptCompilationTrace | None = None
    serving_profile: ServingProfileApplication | None = None


class ResponseChunk(BaseModel):
    id: str
    object: Literal["response.chunk"] = "response.chunk"
    created: int
    model: str
    delta: str | None = None
    reasoning: ReasoningOutput | None = None
    done: bool = False
    citations: list[GeneratedCitationReference] = Field(default_factory=list)
    metadata: ExecutionMetadata | None = None
    structured_output: StructuredOutputResult | None = None
    serving_profile: ServingProfileApplication | None = None
