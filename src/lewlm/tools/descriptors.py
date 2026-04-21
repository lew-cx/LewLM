"""Lightweight local tool descriptor models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LocalToolDescriptor(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str
    execution_mode: Literal["local"] = "local"
    required_authorization: str
    result_type: Literal["artifact", "document_ir"]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
