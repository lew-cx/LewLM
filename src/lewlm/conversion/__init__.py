"""Conversion exports."""

from __future__ import annotations

from lewlm.conversion.models import (
    ConversionCompatibilityReport,
    ConversionJobRequest,
    ConversionPolicy,
    JobRecord,
    JobStatus,
    JobType,
)

__all__ = [
    "ConversionCompatibilityReport",
    "ConversionJobRequest",
    "ConversionPolicy",
    "ConversionService",
    "JobRecord",
    "JobStatus",
    "JobType",
]


def __getattr__(name: str):
    if name == "ConversionService":
        from lewlm.conversion.service import ConversionService

        return ConversionService
    raise AttributeError(name)
