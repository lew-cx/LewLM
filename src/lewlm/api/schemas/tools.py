"""API schemas for local tool catalog routes."""

from __future__ import annotations

from pydantic import BaseModel

from lewlm.tools.descriptors import LocalToolDescriptor


class ToolListResponse(BaseModel):
    count: int
    items: list[LocalToolDescriptor]
