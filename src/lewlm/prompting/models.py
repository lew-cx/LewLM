"""Typed prompt compilation inputs and trace models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lewlm.core.contracts import GenerateMessage
from lewlm.structured_output import StructuredOutputRequest


class PromptToolDefinition(BaseModel):
    """Declarative tool metadata that can be folded into a prompt plan."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class PromptMCPToolDefinition(PromptToolDefinition):
    """Prompt-only local MCP tool metadata surfaced during compilation."""

    server: str


class PromptSkillDefinition(BaseModel):
    """Structured local skill file used by the prompt compiler."""

    name: str
    description: str | None = None
    version: str | None = None
    applicable_modalities: list[str] = Field(default_factory=list)
    supported_inputs: list[str] = Field(default_factory=list)
    supported_outputs: list[str] = Field(default_factory=list)
    prompt_scaffolding: str | None = None
    validation_rules: list[str] = Field(default_factory=list)
    post_processing_steps: list[str] = Field(default_factory=list)
    tool_permissions: list[str] = Field(default_factory=list)
    deterministic_formatting_hints: list[str] = Field(default_factory=list)
    failure_recovery_instructions: list[str] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None


class PromptOverrideRecord(BaseModel):
    """Inspectable record of an applied prompt override."""

    source: str
    scope: str
    summary: str
    path: str | None = None


class PromptAttachmentPlanEntry(BaseModel):
    """Attachment metadata surfaced by prompt compilation."""

    message_index: int
    role: str
    name: str
    attachment_type: str
    media_type: str | None = None
    source_path: str | None = None
    extracted_text_characters: int = 0


class PromptToolPlanEntry(BaseModel):
    """Tool metadata surfaced by prompt compilation."""

    name: str
    source: Literal["request", "tools_file", "skills_file", "mcp_request", "mcp_tools_file"] = "request"
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    registered: bool = False
    execution_mode: Literal["prompt_only", "local_tool"] = "prompt_only"
    version: str | None = None
    required_authorization: str | None = None
    mcp_server: str | None = None
    metadata_trusted: bool = True


class PromptModelTemplateSelection(BaseModel):
    """Selected model-aware prompt template used for serialization."""

    id: str
    version: str
    source: Literal["default", "architecture_family", "model_id"] = "default"
    matched_on: str | None = None


class PromptOutputContract(BaseModel):
    """Declared output contract for a compiled prompt."""

    model_config = ConfigDict(populate_by_name=True)

    format: Literal["text", "json_schema", "grammar"] = "text"
    name: str | None = None
    strict: bool | None = None
    schema_payload: dict[str, Any] | None = Field(default=None, alias="schema", serialization_alias="schema")
    grammar: str | None = None
    syntax: str | None = None


class PromptCompilationTrace(BaseModel):
    """Inspectable trace for a compiled prompt."""

    selected_template: str
    requested_model_id: str | None = None
    resolved_model_id: str | None = None
    model_prompt_template: PromptModelTemplateSelection | None = None
    serialized_model_prompt: str | None = None
    message_count: int
    message_roles: list[str] = Field(default_factory=list)
    attachment_plan: list[PromptAttachmentPlanEntry] = Field(default_factory=list)
    tool_plan: list[PromptToolPlanEntry] = Field(default_factory=list)
    output_contract: PromptOutputContract = Field(default_factory=PromptOutputContract)
    overrides: list[PromptOverrideRecord] = Field(default_factory=list)


class PromptCompilationRequest(BaseModel):
    """Optional prompt overrides supplied by an API or CLI caller."""

    actor: Literal["api", "cli", "system"] = "system"
    system_prompt: str | None = None
    developer_prompt: str | None = None
    pretext_path: str | None = None
    system_prompt_file_path: str | None = None
    skills_path: str | None = None
    response_format: StructuredOutputRequest | None = None
    response_format_path: str | None = None
    output_schema: dict[str, Any] | None = None
    output_schema_path: str | None = None
    tools: list[PromptToolDefinition] = Field(default_factory=list)
    tools_path: str | None = None
    mcp_tools: list[PromptMCPToolDefinition] = Field(default_factory=list)
    mcp_tools_path: str | None = None
    include_trace: bool = False

    def has_requested_overrides(self) -> bool:
        """Return whether the caller provided any explicit prompt override input."""

        return any(
            (
                bool(self.system_prompt and self.system_prompt.strip()),
                bool(self.developer_prompt and self.developer_prompt.strip()),
                self.pretext_path is not None,
                self.system_prompt_file_path is not None,
                self.skills_path is not None,
                self.response_format is not None and self.response_format.type != "text",
                self.response_format_path is not None,
                self.output_schema is not None,
                self.output_schema_path is not None,
                bool(self.tools),
                self.tools_path is not None,
                bool(self.mcp_tools),
                self.mcp_tools_path is not None,
            ),
        )


class PromptCompilationResult(BaseModel):
    """Compiled prompt messages plus an inspectable trace."""

    messages: list[GenerateMessage]
    trace: PromptCompilationTrace
