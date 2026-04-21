"""Built-in skill catalog routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from lewlm.api.dependencies import get_services
from lewlm.api.schemas.skills import SkillListResponse
from lewlm.documents.skills.models import BuiltInSkillDescriptor


router = APIRouter(tags=["skills"])


@router.get("/v1/skills", response_model=SkillListResponse)
def list_skills(request: Request) -> SkillListResponse:
    """List built-in deterministic skills."""

    services = get_services(request)
    skills = services.skill_catalog_service.list_skills()
    return SkillListResponse(count=len(skills), items=skills)


@router.get("/v1/skills/{skill_name}", response_model=BuiltInSkillDescriptor)
def get_skill(skill_name: str, request: Request) -> BuiltInSkillDescriptor:
    """Return a built-in skill descriptor."""

    services = get_services(request)
    return services.skill_catalog_service.get_skill(skill_name)
