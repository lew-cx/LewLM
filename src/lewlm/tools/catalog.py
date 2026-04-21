"""Catalog of local executable tools."""

from __future__ import annotations

from typing import Any

from lewlm.core.errors import PackUnavailableError, ToolNotFoundError
from lewlm.security.authorization import ToolAction
from lewlm.tools.descriptors import LocalToolDescriptor


class ToolCatalogService:
    """Expose registered local tool descriptors."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        disabled_reason: str | None = None,
    ) -> None:
        self._enabled = enabled
        self._disabled_reason = disabled_reason or "Local tools are disabled for this LewLM process."
        self._descriptors = {
            descriptor.name: descriptor
            for descriptor in (
                LocalToolDescriptor(
                    name="documents.generate",
                    description="Render a structured DocumentIR payload into a deterministic output artifact.",
                    required_authorization=ToolAction.DOCUMENT_GENERATE.value,
                    result_type="artifact",
                    input_schema=_artifact_input_schema(include_document=True),
                    tags=["documents", "generation", "local"],
                    aliases=[ToolAction.DOCUMENT_GENERATE.value],
                ),
                LocalToolDescriptor(
                    name="documents.ingest",
                    description="Ingest local files into structured DocumentIR output using the configured local parsers.",
                    required_authorization=ToolAction.DOCUMENT_INGEST.value,
                    result_type="document_ir",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "paths": {"type": "array", "items": {"type": "string"}},
                            "title": {"type": "string"},
                            "idempotency_key": {"type": "string"},
                        },
                        "required": ["paths"],
                    },
                    tags=["documents", "ingest", "local"],
                    aliases=[ToolAction.DOCUMENT_INGEST.value],
                ),
                LocalToolDescriptor(
                    name="documents.transform",
                    description="Run a built-in deterministic document skill and render its result as a document artifact.",
                    required_authorization=ToolAction.DOCUMENT_TRANSFORM.value,
                    result_type="artifact",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "skill": {"type": "string"},
                            "output_format": {"type": "string"},
                            "file_name": {"type": "string"},
                            "idempotency_key": {"type": "string"},
                            "input": {"type": "object"},
                        },
                        "required": ["skill", "output_format", "input"],
                    },
                    tags=["documents", "skills", "local"],
                    aliases=[ToolAction.DOCUMENT_TRANSFORM.value],
                ),
            )
        }
        self._aliases = {
            alias: descriptor.name
            for descriptor in self._descriptors.values()
            for alias in descriptor.aliases
        }

    def list_tools(self) -> list[LocalToolDescriptor]:
        if not self._enabled:
            return []
        return sorted(self._descriptors.values(), key=lambda descriptor: descriptor.name)

    def get_tool(self, tool_name: str) -> LocalToolDescriptor:
        if not self._enabled:
            raise PackUnavailableError(
                "Local tools are unavailable because the `documents` feature pack is disabled.",
                details={"pack": "documents", "reason": self._disabled_reason},
            )
        descriptor = self.find_tool(tool_name)
        if descriptor is None:
            raise ToolNotFoundError(
                "Local tool was not found.",
                details={"tool": tool_name},
            )
        return descriptor

    def find_tool(self, tool_name: str) -> LocalToolDescriptor | None:
        if not self._enabled:
            return None
        canonical_name = self._aliases.get(tool_name, tool_name)
        return self._descriptors.get(canonical_name)


def _artifact_input_schema(*, include_document: bool) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "output_format": {"type": "string"},
            "file_name": {"type": "string"},
            "idempotency_key": {"type": "string"},
        },
        "required": ["output_format"],
    }
    if include_document:
        schema["properties"]["document"] = {"type": "object"}
        schema["required"] = ["output_format", "document"]
    return schema
