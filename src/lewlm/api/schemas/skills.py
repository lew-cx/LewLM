"""API schemas for built-in skill catalog routes."""

from __future__ import annotations

from pydantic import BaseModel

from lewlm.documents.skills.models import BuiltInSkillDescriptor


class SkillListResponse(BaseModel):
    count: int
    items: list[BuiltInSkillDescriptor]
