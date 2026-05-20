"""Shared constrained-decoding probe contracts and classification helpers."""

from __future__ import annotations

from typing import Any

from lewlm.core.contracts import MeasuredCapabilityStatus
from lewlm.structured_output import JSONSchemaResponseFormat, StructuredOutputResult, StructuredOutputRuntimeStatus

# Keep the legacy probe name so truthful probe results overwrite older placeholder rows.
CONSTRAINED_DECODING_CODE_PROBE_NAME = "prompt_guided_structured_output"

CONSTRAINED_DECODING_PROBE_CONTRACT = JSONSchemaResponseFormat(
    schema={
        "type": "object",
        "properties": {
            "status": {"type": "string", "const": "ok"},
            "request_kind": {"type": "string", "const": "benchmark_probe"},
        },
        "required": ["status", "request_kind"],
        "additionalProperties": False,
    },
    name="lewlm_benchmark_probe",
    strict=True,
)


def classify_constrained_decoding_runtime_status(
    status: StructuredOutputRuntimeStatus | dict[str, Any] | None,
) -> tuple[MeasuredCapabilityStatus, str, dict[str, Any]]:
    """Classify lightweight runtime enforcement metadata for measured capability probes."""

    normalized = _coerce_runtime_status(status)
    if normalized is None:
        return (
            MeasuredCapabilityStatus.REJECTED,
            "Runtime probe did not expose structured-output enforcement metadata for this path.",
            {
                "enforcement": "none",
                "decoder_enforced": False,
                "fallback_used": False,
                "fallback_reason": None,
            },
        )
    details = {
        "enforcement": normalized.enforcement,
        "decoder_enforced": normalized.decoder_enforced,
        "fallback_used": normalized.fallback_used,
        "fallback_reason": normalized.fallback_reason,
    }
    if normalized.decoder_enforced:
        return (
            MeasuredCapabilityStatus.SUPPORTED,
            "Runtime probe reports decode-time constrained decoding on the routed runtime.",
            details,
        )
    if normalized.fallback_used or normalized.enforcement == "prompt_guided":
        return (
            MeasuredCapabilityStatus.FALLBACK,
            normalized.fallback_reason
            or "Runtime probe reports prompt-guided structured-output fallback on the routed runtime.",
            details,
        )
    return (
        MeasuredCapabilityStatus.REJECTED,
        "Runtime probe could not verify decode-time constrained decoding on the routed runtime.",
        details,
    )


def classify_constrained_decoding_result(
    result: StructuredOutputResult | None,
) -> tuple[MeasuredCapabilityStatus, str, dict[str, Any], list[str]]:
    """Classify a completed structured-output result for benchmark and probe reporting."""

    validation = result.validation if result is not None else None
    decoder_enforced = bool(result.decoder_enforced) if result is not None else False
    fallback_used = bool(result.fallback_used) if result is not None else False
    enforcement = result.enforcement if result is not None else "none"
    validation_state = validation.state if validation is not None else "unavailable"
    details = {
        "enforcement": enforcement,
        "decoder_enforced": decoder_enforced,
        "fallback_used": fallback_used,
        "validation_state": validation_state,
        "validation_issue_count": len(validation.issues) if validation is not None else 0,
    }
    notes: list[str] = []
    if validation is not None and validation.message:
        notes.append(validation.message)
    if result is not None and result.fallback_reason:
        notes.append(result.fallback_reason)
    if decoder_enforced and validation_state == "valid":
        return (
            MeasuredCapabilityStatus.SUPPORTED,
            "Benchmark verified decode-time constrained decoding on the routed runtime.",
            details,
            notes,
        )
    if fallback_used:
        return (
            MeasuredCapabilityStatus.FALLBACK,
            "Benchmark observed prompt-guided structured-output fallback instead of decode-time constrained decoding.",
            details,
            notes,
        )
    if validation_state == "valid":
        return (
            MeasuredCapabilityStatus.FALLBACK,
            "Benchmark observed structured output, but the runtime did not report decode-time decoder enforcement.",
            details,
            notes,
        )
    return (
        MeasuredCapabilityStatus.REJECTED,
        "Benchmark could not verify decode-time constrained decoding from the routed runtime response.",
        details,
        notes,
    )


def _coerce_runtime_status(
    status: StructuredOutputRuntimeStatus | dict[str, Any] | None,
) -> StructuredOutputRuntimeStatus | None:
    if isinstance(status, StructuredOutputRuntimeStatus):
        return status
    if isinstance(status, dict):
        return StructuredOutputRuntimeStatus.model_validate(status)
    return None
