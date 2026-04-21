"""Conversion jobs, compatibility reports, and cache records."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from lewlm.core.contracts import (
    ModelFormat,
    ModelArtifactRole,
    ModelModality,
    QuantizationPrecision,
    QuantizationProfile,
    QuantizationStrategy,
    RuntimeAffinity,
    quantization_profile_label,
    utc_now,
)


QUANTIZATION_PROFILE_METADATA_FILENAME = "lewlm.quantization_profile.json"
LAYERED_CONVERSION_MANIFEST_FILENAME = "lewlm.layered_manifest.json"
CONVERSION_OUTPUT_METADATA_FILENAME = "lewlm.conversion_output.json"


class ConversionPolicy(str, Enum):
    MAX_QUALITY = "max_quality"
    BALANCED = "balanced"
    MAX_FIT = "max_fit"
    CUSTOM_BITS = "custom_bits"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    MODEL_CONVERSION = "model_conversion"


class ConversionProfileSupport(BaseModel):
    requested_profile: QuantizationProfile | None = None
    supported: bool
    reason: str
    warnings: list[str] = Field(default_factory=list)
    missing_packages: list[str] = Field(default_factory=list)
    requires_calibration: bool = False
    requires_external_quantizer: bool = False
    requires_native_fp8: bool = False
    notes: list[str] = Field(default_factory=list)


class ConversionCompatibilityReport(BaseModel):
    model_id: str
    source_format: ModelFormat
    target_format: str = "mlx"
    backend_name: str
    can_convert: bool
    already_runnable: bool = False
    reason: str
    cache_key: str
    output_path: str
    quantization_mode: str | None = None
    custom_bits: int | None = None
    requested_profile: QuantizationProfile | None = None
    resolved_profile: QuantizationProfile | None = None
    profile_support: list[ConversionProfileSupport] = Field(default_factory=list)
    layered_output: bool = False
    artifact_plans: list["LayeredConversionArtifact"] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ConversionJobRequest(BaseModel):
    model_id: str
    policy: ConversionPolicy = ConversionPolicy.BALANCED
    custom_bits: int | None = None
    quantization_profile: QuantizationProfile | None = None
    force: bool = False
    idempotency_key: str | None = None
    authorized_actions: list[str] = Field(default_factory=list)


class JobRecord(BaseModel):
    job_id: str
    job_type: JobType
    status: JobStatus
    cache_key: str | None = None
    idempotency_key: str | None = None
    idempotent_replay: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ConversionArtifactRecord(BaseModel):
    cache_key: str
    model_id: str
    output_path: str
    policy: ConversionPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class LayeredConversionArtifact(BaseModel):
    artifact_key: str
    role: ModelArtifactRole
    display_name: str
    relative_path: str
    format_type: ModelFormat
    modality: tuple[ModelModality, ...]
    runtime_affinity: tuple[RuntimeAffinity, ...]
    derived_from: str | None = None
    tokenizer_path: str | None = None
    processor_path: str | None = None
    quantization: str | None = None
    quantization_profile: QuantizationProfile | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LayeredConversionManifest(BaseModel):
    artifact_family_id: str
    display_name: str
    source_path: str
    source_format: ModelFormat
    source_modality: tuple[ModelModality, ...]
    source_runtime_affinity: tuple[RuntimeAffinity, ...] = ()
    source_tokenizer_path: str | None = None
    source_processor_path: str | None = None
    source_quantization: str | None = None
    source_quantization_profile: QuantizationProfile | None = None
    artifacts: list[LayeredConversionArtifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversionOutputMetadata(BaseModel):
    source_display_name: str
    source_model_id: str
    display_name: str
    artifact_role: ModelArtifactRole = ModelArtifactRole.STANDALONE
    artifact_family_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


ConversionCompatibilityReport.model_rebuild()


def quantization_precision_from_bits(bits: int | None) -> QuantizationPrecision | None:
    if bits is None:
        return None
    try:
        return QuantizationPrecision(f"int{bits}")
    except ValueError:
        return None


def default_quantization_profile(
    *,
    policy: ConversionPolicy,
    custom_bits: int | None,
) -> QuantizationProfile:
    if policy == ConversionPolicy.MAX_QUALITY:
        return QuantizationProfile(
            name=policy.value,
            strategy=QuantizationStrategy.WEIGHT_ONLY,
            weight_precision=QuantizationPrecision.FP16,
        )
    if policy == ConversionPolicy.CUSTOM_BITS:
        return QuantizationProfile(
            name=policy.value,
            strategy=QuantizationStrategy.WEIGHT_ONLY,
            weight_precision=quantization_precision_from_bits(custom_bits),
            metadata={"custom_bits": custom_bits} if custom_bits is not None else {},
        )
    return QuantizationProfile(
        name=policy.value,
        strategy=QuantizationStrategy.WEIGHT_ONLY,
        weight_precision=QuantizationPrecision.INT4,
    )


def resolve_quantization_profile(
    *,
    policy: ConversionPolicy,
    custom_bits: int | None,
    requested_profile: QuantizationProfile | None,
) -> QuantizationProfile:
    resolved = requested_profile.model_copy(deep=True) if requested_profile is not None else default_quantization_profile(
        policy=policy,
        custom_bits=custom_bits,
    )
    if resolved.name is None:
        resolved.name = policy.value
    if resolved.strategy == QuantizationStrategy.WEIGHT_ONLY and resolved.weight_precision is None:
        resolved.weight_precision = (
            QuantizationPrecision.FP16
            if policy == ConversionPolicy.MAX_QUALITY
            else quantization_precision_from_bits(custom_bits) or QuantizationPrecision.INT4
        )
    if custom_bits is not None:
        resolved.metadata = {**resolved.metadata, "custom_bits": custom_bits}
    return resolved


def quantization_profile_cache_payload(profile: QuantizationProfile | None) -> str:
    if profile is None:
        return ""
    return json.dumps(
        profile.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def quantization_mode_from_profile(profile: QuantizationProfile | None) -> str | None:
    if profile is None:
        return None
    if profile.strategy == QuantizationStrategy.WEIGHT_ONLY:
        if profile.weight_precision in {None, QuantizationPrecision.FP16}:
            return None
        if profile.weight_precision == QuantizationPrecision.INT4:
            return "4bit"
    return quantization_profile_label(profile)
