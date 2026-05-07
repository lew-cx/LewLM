"""Helpers for applying measured runtime-preference evidence safely."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lewlm.core.contracts import RuntimeAffinity
from lewlm.runtime.support_strategy import external_adapter_demotion_reason


@dataclass(frozen=True, slots=True)
class RuntimePreferenceAssessment:
    """Interpreted runtime-preference payload with downgrade guidance."""

    source: str | None
    selected_runtime_name: str | None
    selected_runtime_affinity: str | None
    baseline_runtime_name: str | None
    baseline_runtime_affinity: str | None
    effective_runtime_name: str | None
    effective_runtime_affinity: str | None
    primary_metric: str | None
    selected_metric_value: int | float | None
    baseline_metric_value: int | float | None
    adopted: bool
    downgrade_reason: str | None = None
    notes: tuple[str, ...] = ()
    preserved_features: tuple[str, ...] = ()
    degraded_features: tuple[str, ...] = ()
    rejected_features: tuple[str, ...] = ()


def assess_runtime_preference(payload: Mapping[str, Any] | None) -> RuntimePreferenceAssessment | None:
    if not isinstance(payload, Mapping):
        return None
    source = _string_or_none(payload.get("source"))
    selected_runtime_name = _string_or_none(payload.get("selected_runtime_name"))
    selected_runtime_affinity = _string_or_none(payload.get("selected_runtime_affinity"))
    baseline_runtime_name = _string_or_none(payload.get("baseline_runtime_name"))
    baseline_runtime_affinity = _string_or_none(payload.get("baseline_runtime_affinity"))
    primary_metric = _string_or_none(payload.get("primary_metric"))
    selected_metric_value = _number_or_none(payload.get("selected_metric_value"))
    baseline_metric_value = _number_or_none(payload.get("baseline_metric_value"))
    effective_runtime_name = selected_runtime_name
    effective_runtime_affinity = selected_runtime_affinity
    adopted = True
    downgrade_reason = None
    notes: list[str] = []
    preserved_features: tuple[str, ...] = ()
    degraded_features: tuple[str, ...] = ()
    rejected_features: tuple[str, ...] = ()

    if source == "compare_external_adapter":
        feature_preservation = payload.get("feature_preservation")
        preserved_features = tuple(_string_list(feature_preservation, "preserved"))
        degraded_features = tuple(_string_list(feature_preservation, "degraded"))
        rejected_features = tuple(_string_list(feature_preservation, "rejected"))
        bridge_boundary_reason = external_adapter_demotion_reason(
            baseline_runtime_affinity=baseline_runtime_affinity,
        )
        if rejected_features:
            notes.append(
                "Measured external-adapter comparison rejected native feature coverage for "
                + ", ".join(f"`{name}`" for name in rejected_features)
                + "."
            )
        if degraded_features:
            notes.append(
                "Measured external-adapter comparison preserved only partial feature coverage for "
                + ", ".join(f"`{name}`" for name in degraded_features)
                + "."
            )
        if _is_external_runtime(runtime_name=selected_runtime_name, runtime_affinity=selected_runtime_affinity):
            if rejected_features:
                adopted = False
                downgrade_reason = (
                    "measured feature-preservation evidence rejected native coverage for "
                    + ", ".join(f"`{name}`" for name in rejected_features)
                )
            elif degraded_features:
                adopted = False
                downgrade_reason = (
                    "measured feature-preservation evidence only partially preserved "
                    + ", ".join(f"`{name}`" for name in degraded_features)
                )
            elif not preserved_features:
                adopted = False
                downgrade_reason = "the measured comparison did not record feature-preservation evidence for the adapter path"
            elif bridge_boundary_reason is not None:
                adopted = False
                downgrade_reason = bridge_boundary_reason
            if not adopted:
                effective_runtime_name = baseline_runtime_name
                effective_runtime_affinity = baseline_runtime_affinity
                if effective_runtime_name or effective_runtime_affinity:
                    notes.append(
                        "LewLM downgraded the adapter-backed routing winner and kept "
                        f"`{effective_runtime_name or effective_runtime_affinity}` as the safe measured default."
                    )
                    if bridge_boundary_reason is not None:
                        notes.append(
                            "LewLM keeps the first-class local runtime as the productized default and records the "
                            "external-adapter result as bridge evidence instead of promoting it to the default path."
                        )
                else:
                    notes.append(
                        "LewLM downgraded the adapter-backed routing winner because no measured safe baseline runtime was recorded."
                    )

    return RuntimePreferenceAssessment(
        source=source,
        selected_runtime_name=selected_runtime_name,
        selected_runtime_affinity=selected_runtime_affinity,
        baseline_runtime_name=baseline_runtime_name,
        baseline_runtime_affinity=baseline_runtime_affinity,
        effective_runtime_name=effective_runtime_name,
        effective_runtime_affinity=effective_runtime_affinity,
        primary_metric=primary_metric,
        selected_metric_value=selected_metric_value,
        baseline_metric_value=baseline_metric_value,
        adopted=adopted,
        downgrade_reason=downgrade_reason,
        notes=tuple(notes),
        preserved_features=preserved_features,
        degraded_features=degraded_features,
        rejected_features=rejected_features,
    )


def runtime_preference_matches(
    assessment: RuntimePreferenceAssessment,
    *,
    runtime_name: str,
    runtime_affinity: str,
) -> bool:
    return assessment.effective_runtime_affinity == runtime_affinity or assessment.effective_runtime_name == runtime_name


def runtime_preference_comparison_suffix(assessment: RuntimePreferenceAssessment) -> str:
    if (
        assessment.primary_metric is not None
        and assessment.selected_metric_value is not None
        and assessment.baseline_metric_value is not None
    ):
        return (
            f" ({assessment.primary_metric}: {assessment.selected_metric_value} vs {assessment.baseline_metric_value})"
        )
    return ""


def _is_external_runtime(*, runtime_name: str | None, runtime_affinity: str | None) -> bool:
    return runtime_affinity == RuntimeAffinity.EXTERNAL_ACCELERATOR.value or runtime_name == "local_external_adapter"


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _number_or_none(value: object) -> int | float | None:
    return value if isinstance(value, int | float) else None


def _string_list(payload: object, key: str) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]
