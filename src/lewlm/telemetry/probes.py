"""Helpers for summarizing measured capability probe evidence."""

from __future__ import annotations

from collections.abc import Sequence

from lewlm.core.contracts import (
    MeasuredCapabilityCategory,
    MeasuredCapabilityProbeRecord,
    MeasuredCapabilityStatus,
    MeasuredCapabilitySummary,
)

MEASURED_CAPABILITY_CATEGORY_ORDER: tuple[MeasuredCapabilityCategory, ...] = (
    MeasuredCapabilityCategory.BATCHING,
    MeasuredCapabilityCategory.CACHE_REUSE,
    MeasuredCapabilityCategory.SPECULATION,
    MeasuredCapabilityCategory.CONSTRAINED_DECODING,
    MeasuredCapabilityCategory.COMPILE_KERNELS,
    MeasuredCapabilityCategory.ADAPTER_PRESERVATION,
)


def summarize_measured_capabilities(
    records: Sequence[MeasuredCapabilityProbeRecord],
) -> list[MeasuredCapabilitySummary]:
    grouped: dict[MeasuredCapabilityCategory, list[MeasuredCapabilityProbeRecord]] = {
        category: []
        for category in MEASURED_CAPABILITY_CATEGORY_ORDER
    }
    for record in records:
        grouped.setdefault(record.category, []).append(record)
    return [
        _summarize_category(category, grouped.get(category, []))
        for category in (
            *MEASURED_CAPABILITY_CATEGORY_ORDER,
            *sorted(set(grouped) - set(MEASURED_CAPABILITY_CATEGORY_ORDER), key=lambda item: item.value),
        )
    ]


def _summarize_category(
    category: MeasuredCapabilityCategory,
    records: Sequence[MeasuredCapabilityProbeRecord],
) -> MeasuredCapabilitySummary:
    if not records:
        return MeasuredCapabilitySummary(
            category=category,
            status=MeasuredCapabilityStatus.UNMEASURED,
            reason="No measured probe or benchmark evidence has been recorded for this category on this host yet.",
        )
    ordered = sorted(records, key=lambda item: (item.recorded_at, item.probe_name), reverse=True)
    informative_statuses = {
        item.status
        for item in ordered
        if item.status != MeasuredCapabilityStatus.NOT_APPLICABLE
    }
    if not informative_statuses:
        status = MeasuredCapabilityStatus.NOT_APPLICABLE
    elif len(informative_statuses) == 1:
        status = next(iter(informative_statuses))
    else:
        status = MeasuredCapabilityStatus.MIXED
    return MeasuredCapabilitySummary(
        category=category,
        status=status,
        reason=_summary_reason(status=status, records=ordered),
        record_count=len(ordered),
        latest_recorded_at=ordered[0].recorded_at,
        runtime_names=sorted({item.runtime_name for item in ordered if item.runtime_name}),
        sources=sorted({item.source.value for item in ordered}),
        probes=list(ordered),
    )


def _summary_reason(
    *,
    status: MeasuredCapabilityStatus,
    records: Sequence[MeasuredCapabilityProbeRecord],
) -> str:
    if not records:
        return "No measured evidence is available."
    latest = records[0]
    if status == MeasuredCapabilityStatus.MIXED:
        status_labels = ", ".join(sorted({item.status.value for item in records}))
        return f"Recorded probes on this host show mixed measured outcomes: {status_labels}."
    if status == MeasuredCapabilityStatus.NOT_APPLICABLE:
        return "Recorded probes marked this category as not applicable on this host."
    return latest.reason
