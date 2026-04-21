"""Model-aware prompt template resolution and serialization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from lewlm.core.contracts import GenerateMessage, ModelManifest
from lewlm.prompting.models import PromptModelTemplateSelection


@dataclass(frozen=True, slots=True)
class PromptTemplateDescriptor:
    """Static prompt-template metadata used for trace serialization."""

    template_id: str
    version: str
    render_style: Literal["generic", "chatml", "gemma", "llama"]
    architecture_markers: tuple[str, ...] = ()


class PromptTemplateCatalogService:
    """Resolve a deterministic prompt template for a selected model."""

    def __init__(self) -> None:
        self._templates = (
            PromptTemplateDescriptor(
                template_id="generic-chat-v1",
                version="1.0.0",
                render_style="generic",
            ),
            PromptTemplateDescriptor(
                template_id="llama-instruct-v1",
                version="1.0.0",
                render_style="llama",
                architecture_markers=("llama", "mistral", "mixtral"),
            ),
            PromptTemplateDescriptor(
                template_id="chatml-v1",
                version="1.0.0",
                render_style="chatml",
                architecture_markers=("qwen", "deepseek", "phi"),
            ),
            PromptTemplateDescriptor(
                template_id="gemma-chat-v1",
                version="1.0.0",
                render_style="gemma",
                architecture_markers=("gemma",),
            ),
        )
        self._default_template = self._templates[0]
        self._templates_by_id = {template.template_id: template for template in self._templates}

    def resolve_template(
        self,
        *,
        model_manifest: ModelManifest | None = None,
        resolved_model_id: str | None = None,
    ) -> PromptModelTemplateSelection:
        """Resolve the prompt template that best matches the selected model."""

        if model_manifest is not None:
            match = self._match_descriptor(model_manifest.architecture_family)
            if match is not None:
                return PromptModelTemplateSelection(
                    id=match.template_id,
                    version=match.version,
                    source="architecture_family",
                    matched_on=model_manifest.architecture_family,
                )

        if resolved_model_id:
            match = self._match_descriptor(resolved_model_id)
            if match is not None:
                return PromptModelTemplateSelection(
                    id=match.template_id,
                    version=match.version,
                    source="model_id",
                    matched_on=resolved_model_id,
                )

        return PromptModelTemplateSelection(
            id=self._default_template.template_id,
            version=self._default_template.version,
            source="default",
        )

    def serialize_messages(
        self,
        messages: list[GenerateMessage],
        template_selection: PromptModelTemplateSelection,
    ) -> str:
        """Serialize compiled messages with the selected prompt template."""

        descriptor = self._templates_by_id.get(template_selection.id, self._default_template)
        if descriptor.render_style == "llama":
            return _render_llama_prompt(messages)
        if descriptor.render_style == "chatml":
            return _render_chatml_prompt(messages)
        if descriptor.render_style == "gemma":
            return _render_gemma_prompt(messages)
        return _render_generic_prompt(messages)

    def _match_descriptor(self, value: str) -> PromptTemplateDescriptor | None:
        normalized = _normalize_marker(value)
        for template in self._templates:
            if any(marker in normalized for marker in template.architecture_markers):
                return template
        return None


def _normalize_marker(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _render_generic_prompt(messages: list[GenerateMessage]) -> str:
    parts = [_render_tagged_message(message, prefix="<", suffix=">") for message in messages]
    if not messages or messages[-1].role != "assistant":
        parts.append("<assistant>\n")
    return "\n".join(part for part in parts if part)


def _render_chatml_prompt(messages: list[GenerateMessage]) -> str:
    parts = [
        f"<|im_start|>{message.role}\n{message.content.strip()}\n<|im_end|>"
        for message in messages
        if message.content.strip()
    ]
    if not messages or messages[-1].role != "assistant":
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _render_gemma_prompt(messages: list[GenerateMessage]) -> str:
    parts = [
        f"<start_of_turn>{message.role}\n{message.content.strip()}<end_of_turn>"
        for message in messages
        if message.content.strip()
    ]
    if not messages or messages[-1].role != "assistant":
        parts.append("<start_of_turn>assistant\n")
    return "\n".join(parts)


def _render_llama_prompt(messages: list[GenerateMessage]) -> str:
    leading_system, remainder = _split_leading_system_messages(messages)
    system_text = "\n\n".join(message.content.strip() for message in leading_system if message.content.strip())
    rendered: list[str] = []
    first_user = True

    for message in remainder:
        content = message.content.strip()
        if not content:
            continue
        if message.role == "user":
            if first_user and system_text:
                content = f"<<SYS>>\n{system_text}\n<</SYS>>\n\n{content}"
            rendered.append(f"<s>[INST] {content} [/INST]")
            first_user = False
            continue
        if message.role == "assistant":
            if rendered:
                rendered.append(f" {content} </s>")
            else:
                rendered.append(f"<s>{content}</s>")
            continue
        rendered.append(f"<s>{message.role}: {content}</s>")

    if not rendered:
        if system_text:
            return f"<s>[INST] <<SYS>>\n{system_text}\n<</SYS>>\n\n [/INST]"
        return "<s>[INST]  [/INST]"
    return "".join(rendered)


def _split_leading_system_messages(
    messages: list[GenerateMessage],
) -> tuple[list[GenerateMessage], list[GenerateMessage]]:
    system_messages: list[GenerateMessage] = []
    remaining_index = 0
    for index, message in enumerate(messages):
        if message.role != "system":
            remaining_index = index
            break
        system_messages.append(message)
    else:
        remaining_index = len(messages)
    return system_messages, messages[remaining_index:]


def _render_tagged_message(message: GenerateMessage, *, prefix: str, suffix: str) -> str:
    content = message.content.strip()
    if not content:
        return ""
    return f"{prefix}{message.role}{suffix}\n{content}\n{prefix}/{message.role}{suffix}"
