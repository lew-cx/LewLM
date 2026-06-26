"""Middleware-facing capability and artifact evidence reports."""

from __future__ import annotations

from typing import Any

from lewlm.core.contracts import (
    CapabilityEvidence,
    CapabilityEvidenceState,
    CapabilityName,
    CapabilityOwnership,
    CapabilityReadinessState,
    ConversionStatus,
    LewLMMiddlewareCapabilitiesReport,
    ModelArtifactLineageReport,
    ModelCapabilityReport,
    ModelCapabilityStatus,
    RuntimeAffinity,
    RuntimeProvider,
    RuntimeProviderReport,
    RuntimeSupportPath,
    runtime_support_path_for_affinity,
)


def build_middleware_capabilities_report(services: Any) -> LewLMMiddlewareCapabilitiesReport:
    """Build LewLM's host-level evidence report without running inference."""

    manifests = services.model_registry.list_manifests()
    readiness = services.model_router.capability_readiness_summary()
    benchmark_records = services.metadata_store.list_benchmark_records(limit=500)
    benchmarked_capabilities = _benchmarked_capabilities(benchmark_records)
    capability_evidence: list[CapabilityEvidence] = []
    for item in readiness.capabilities:
        state = _host_evidence_state(item.readiness_state)
        if item.ready and item.capability.value in benchmarked_capabilities:
            state = CapabilityEvidenceState.BENCHMARK_PASSED
        capability_evidence.append(
            CapabilityEvidence(
                capability=item.capability,
                state=state,
                ownership=_host_ownership(item),
                reason=item.reason,
                runtime_name=(item.available_runtime_names[0] if item.available_runtime_names else None),
                provider=_provider_from_runtime_name(
                    item.available_runtime_names[0] if item.available_runtime_names else None,
                    item.available_support_paths[0] if item.available_support_paths else None,
                    external_profile=services.settings.external_accelerator_profile,
                ),
                source="host_readiness",
                details={
                    "ready": item.ready,
                    "readiness_state": item.readiness_state.value,
                    "available_runtime_names": item.available_runtime_names,
                    "available_support_paths": [support_path.value for support_path in item.available_support_paths],
                    "candidate_model_count": item.candidate_model_count,
                    "runnable_model_count": item.runnable_model_count,
                    "bridge_only": item.bridge_only,
                },
            ),
        )
    notes = [
        "LewLM reports routability as `discovered` until a load, generation, probe, or benchmark record upgrades the evidence state.",
        "Bridge-backed support is reported separately from LewLM-owned or packaged runtime support.",
    ]
    if not benchmark_records:
        notes.append("No benchmark records are stored yet; run `lewlm bench` or `lewlm benchmark` to upgrade evidence.")
    return LewLMMiddlewareCapabilitiesReport(
        service=services.settings.app_name,
        host_platform=services.runtime_catalog.host_platform_snapshot(),
        discovered_model_count=len(manifests),
        runnable_model_count=sum(1 for manifest in manifests if manifest.conversion_status == ConversionStatus.RUNNABLE),
        conversion_required_model_count=sum(
            1 for manifest in manifests if manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION
        ),
        capability_evidence=capability_evidence,
        runtime_providers=build_runtime_provider_reports(services),
        readiness=readiness,
        notes=notes,
    )


def build_runtime_provider_reports(services: Any) -> list[RuntimeProviderReport]:
    """Summarize configured runtime providers through LewLM's ownership vocabulary."""

    reports: list[RuntimeProviderReport] = []
    runtimes = getattr(services.runtime_catalog, "_runtimes", {})
    for runtime in runtimes.values():
        support_path = runtime_support_path_for_affinity(runtime.affinity) or RuntimeSupportPath.PACKAGED
        available = runtime.is_available()
        provider = _provider_from_runtime_name(
            runtime.name,
            support_path,
            affinity=runtime.affinity,
            external_profile=services.settings.external_accelerator_profile,
        )
        supported_capabilities = [
            capability
            for capability in CapabilityName
            if capability != CapabilityName.CONVERSION and runtime.supports_capability(capability)
        ]
        reports.append(
            RuntimeProviderReport(
                provider=provider,
                runtime_name=runtime.name,
                runtime_affinity=runtime.affinity,
                ownership=_provider_ownership(support_path=support_path, available=available),
                support_path=support_path,
                available=available,
                reason=runtime.availability_reason(),
                supported_capabilities=supported_capabilities,
                evidence_state=(
                    CapabilityEvidenceState.DISCOVERED
                    if available
                    else CapabilityEvidenceState.REQUIRES_INSTALL
                ),
                notes=_provider_notes(provider=provider, support_path=support_path),
            ),
        )
    reports.sort(key=lambda item: (item.support_path.value, item.provider.value, item.runtime_name))
    return reports


def build_model_artifact_lineage_report(services: Any, model_id: str) -> ModelArtifactLineageReport:
    """Return artifact, conversion, benchmark, and capability evidence for one model."""

    manifest = services.model_registry.get_manifest(model_id)
    capability_report = services.model_router.model_capability_report(manifest.model_id)
    conversion_artifacts = [
        artifact.model_dump(mode="json")
        for artifact in services.metadata_store.list_conversion_artifacts()
        if _conversion_artifact_matches_model(artifact.model_id, artifact.metadata, manifest.model_id)
    ]
    latest_benchmark = _latest_benchmark_for_model(services.metadata_store.list_benchmark_records(limit=500), manifest.model_id)
    runtime_probe_records = _runtime_probe_evidence_for_model(services, manifest.model_id)
    evidence = build_model_capability_evidence(
        capability_report,
        benchmark_records=services.metadata_store.list_benchmark_records(limit=500),
        runtime_probe_records=[item.model_dump(mode="json") for item in runtime_probe_records],
    )
    notes = []
    if manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION:
        notes.append("This source artifact still needs conversion before LewLM can claim runnable local inference.")
    if not runtime_probe_records:
        notes.append("No runtime smoke probe evidence is stored for this model yet.")
    if latest_benchmark is None:
        notes.append("No benchmark evidence is stored for this model yet.")
    return ModelArtifactLineageReport(
        model_id=manifest.model_id,
        display_name=manifest.display_name,
        source_path=manifest.source_path,
        format_type=manifest.format_type,
        artifact_role=manifest.artifact_role,
        artifact_family_id=manifest.artifact_family_id,
        artifact_lineage=manifest.artifact_lineage,
        conversion_artifacts=conversion_artifacts,
        runtime_probe_records=runtime_probe_records,
        latest_benchmark=latest_benchmark,
        capability_evidence=evidence,
        notes=notes,
    )


def build_model_capability_evidence(
    report: ModelCapabilityReport,
    *,
    benchmark_records: list[dict[str, Any]] | None = None,
    runtime_probe_records: list[dict[str, Any]] | None = None,
) -> list[CapabilityEvidence]:
    """Normalize a per-model capability report into evidence records."""

    benchmark_records = benchmark_records or []
    runtime_probe_records = runtime_probe_records or []
    evidence: list[CapabilityEvidence] = []
    for status in report.capabilities:
        latest_benchmark = _latest_benchmark_for_model(
            benchmark_records,
            report.model_id,
            capability=status.capability.value,
        )
        latest_probe = _latest_runtime_probe_for_capability(
            runtime_probe_records,
            capability=status.capability.value,
        )
        if latest_benchmark is None and latest_probe is not None:
            evidence.append(CapabilityEvidence.model_validate(latest_probe))
            continue
        state = _model_evidence_state(status, report.conversion_status)
        if latest_benchmark is not None and status.supported:
            state = CapabilityEvidenceState.BENCHMARK_PASSED
        evidence.append(
            CapabilityEvidence(
                capability=status.capability,
                state=state,
                ownership=_model_ownership(status),
                reason=status.reason,
                runtime_name=status.runtime_name,
                runtime_affinity=status.runtime_affinity,
                provider=_provider_from_runtime_name(
                    status.runtime_name,
                    status.support_path,
                    affinity=status.runtime_affinity,
                ),
                model_id=report.model_id,
                source=("benchmark" if latest_benchmark is not None and status.supported else "routing"),
                benchmark_id=(
                    str(latest_benchmark.get("benchmark_id"))
                    if latest_benchmark is not None and latest_benchmark.get("benchmark_id")
                    else None
                ),
                artifact_id=_benchmark_artifact_id(latest_benchmark),
                details={
                    "supported": status.supported,
                    "readiness_state": status.readiness_state.value,
                    "support_path": status.support_path.value if status.support_path is not None else None,
                    "estimated_memory_mb": status.estimated_memory_mb,
                    "notes": status.notes,
                },
            ),
        )
    return evidence


def _runtime_probe_evidence_for_model(services: Any, model_id: str) -> list[CapabilityEvidence]:
    metadata_store = getattr(services, "metadata_store", None)
    list_runtime_probe_records = getattr(metadata_store, "list_runtime_probe_records", None)
    if not callable(list_runtime_probe_records):
        return []
    host_platform = services.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
    records = list_runtime_probe_records(
        model_id=model_id,
        host_platform=host_platform,
        limit=50,
    )
    evidence: list[CapabilityEvidence] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("evidence"), dict):
            continue
        evidence.append(CapabilityEvidence.model_validate(record["evidence"]))
    return evidence


def _latest_runtime_probe_for_capability(
    runtime_probe_records: list[dict[str, Any]],
    *,
    capability: str,
) -> dict[str, Any] | None:
    for record in runtime_probe_records:
        payload = record.get("evidence") if isinstance(record.get("evidence"), dict) else record
        if not isinstance(payload, dict):
            continue
        if payload.get("capability") == capability:
            return payload
    return None


def _host_evidence_state(readiness_state: CapabilityReadinessState) -> CapabilityEvidenceState:
    if readiness_state == CapabilityReadinessState.READY:
        return CapabilityEvidenceState.DISCOVERED
    if readiness_state == CapabilityReadinessState.CONVERSION_REQUIRED:
        return CapabilityEvidenceState.REQUIRES_CONVERSION
    if readiness_state == CapabilityReadinessState.RUNTIME_UNAVAILABLE:
        return CapabilityEvidenceState.REQUIRES_INSTALL
    if readiness_state == CapabilityReadinessState.NO_MODELS:
        return CapabilityEvidenceState.UNSUPPORTED
    return CapabilityEvidenceState.UNSUPPORTED


def _model_evidence_state(
    status: ModelCapabilityStatus,
    conversion_status: ConversionStatus,
) -> CapabilityEvidenceState:
    if status.supported:
        return CapabilityEvidenceState.DISCOVERED
    if conversion_status == ConversionStatus.REQUIRES_CONVERSION:
        return CapabilityEvidenceState.REQUIRES_CONVERSION
    if status.readiness_state == CapabilityReadinessState.RUNTIME_UNAVAILABLE:
        return CapabilityEvidenceState.REQUIRES_INSTALL
    if status.readiness_state == CapabilityReadinessState.CONVERSION_REQUIRED:
        return CapabilityEvidenceState.REQUIRES_CONVERSION
    return CapabilityEvidenceState.UNSUPPORTED


def _host_ownership(item: Any) -> CapabilityOwnership:
    if not item.ready:
        if item.readiness_state == CapabilityReadinessState.CONVERSION_REQUIRED:
            return CapabilityOwnership.FALLBACK
        if item.readiness_state == CapabilityReadinessState.RUNTIME_UNAVAILABLE:
            return CapabilityOwnership.UNVERIFIED
        return CapabilityOwnership.UNSUPPORTED
    if item.bridge_only:
        return CapabilityOwnership.BRIDGE_VERIFIED
    if item.packaged_runtime_names:
        return CapabilityOwnership.BACKEND_NATIVE
    if item.bridge_runtime_names:
        return CapabilityOwnership.BRIDGE_VERIFIED
    return CapabilityOwnership.UNVERIFIED


def _model_ownership(status: ModelCapabilityStatus) -> CapabilityOwnership:
    if not status.supported:
        if status.readiness_state == CapabilityReadinessState.CONVERSION_REQUIRED:
            return CapabilityOwnership.FALLBACK
        if status.readiness_state == CapabilityReadinessState.RUNTIME_UNAVAILABLE:
            return CapabilityOwnership.UNVERIFIED
        return CapabilityOwnership.UNSUPPORTED
    if status.support_path == RuntimeSupportPath.BRIDGE:
        return CapabilityOwnership.BRIDGE_VERIFIED
    if status.support_path == RuntimeSupportPath.PACKAGED:
        return CapabilityOwnership.BACKEND_NATIVE
    return CapabilityOwnership.UNVERIFIED


def _provider_ownership(*, support_path: RuntimeSupportPath, available: bool) -> CapabilityOwnership:
    if not available:
        return CapabilityOwnership.UNVERIFIED
    if support_path == RuntimeSupportPath.BRIDGE:
        return CapabilityOwnership.BRIDGE_VERIFIED
    return CapabilityOwnership.BACKEND_NATIVE


def _provider_from_runtime_name(
    runtime_name: str | None,
    support_path: RuntimeSupportPath | None,
    *,
    affinity: RuntimeAffinity | None = None,
    external_profile: str | None = None,
) -> RuntimeProvider:
    normalized_name = (runtime_name or "").casefold()
    profile = (external_profile or "").casefold()
    if affinity in {RuntimeAffinity.MLX_TEXT, RuntimeAffinity.MLX_VISION, RuntimeAffinity.MLX_AUDIO}:
        return RuntimeProvider.MLX
    if affinity == RuntimeAffinity.LLAMACPP:
        return RuntimeProvider.LLAMACPP
    if support_path == RuntimeSupportPath.BRIDGE:
        bridge_key = profile or normalized_name
        if "sglang" in bridge_key:
            return RuntimeProvider.SGLANG
        if "vllm" in bridge_key:
            return RuntimeProvider.VLLM
        if "tensorrt" in bridge_key or "trt" in bridge_key:
            return RuntimeProvider.TENSORRT_LLM
        if "openvino" in bridge_key:
            return RuntimeProvider.OPENVINO
        if "ollama" in bridge_key:
            return RuntimeProvider.OLLAMA
        if "lm_studio" in bridge_key or "lm-studio" in bridge_key:
            return RuntimeProvider.LM_STUDIO
        if "llamacpp" in bridge_key or "llama.cpp" in bridge_key:
            return RuntimeProvider.LLAMACPP_SERVER
        return RuntimeProvider.OPENAI_COMPATIBLE
    if "onnx" in normalized_name:
        return RuntimeProvider.ONNX_GENAI
    if "openvino" in normalized_name:
        return RuntimeProvider.OPENVINO
    return RuntimeProvider.UNKNOWN


def _provider_notes(*, provider: RuntimeProvider, support_path: RuntimeSupportPath) -> list[str]:
    if support_path == RuntimeSupportPath.BRIDGE:
        return [
            f"`{provider.value}` is treated as a bridge-backed provider; LewLM verifies the adapter contract but does not own upstream kernels.",
        ]
    if provider == RuntimeProvider.LLAMACPP:
        return ["llama.cpp/GGUF is LewLM's first-class packaged non-Apple runtime path."]
    if provider == RuntimeProvider.MLX:
        return ["MLX remains LewLM's first-class packaged Apple Silicon runtime path."]
    return []


def _benchmarked_capabilities(records: list[dict[str, Any]]) -> set[str]:
    capabilities: set[str] = set()
    for record in records:
        capability = record.get("capability")
        if isinstance(capability, str) and capability:
            capabilities.add(capability)
    return capabilities


def _latest_benchmark_for_model(
    records: list[dict[str, Any]],
    model_id: str,
    *,
    capability: str | None = None,
) -> dict[str, Any] | None:
    for record in records:
        if record.get("model_id") != model_id:
            continue
        if capability is not None and record.get("capability") != capability:
            continue
        return record
    return None


def _benchmark_artifact_id(record: dict[str, Any] | None) -> str | None:
    if record is None:
        return None
    artifact = record.get("artifact")
    if not isinstance(artifact, dict):
        return None
    artifact_id = artifact.get("artifact_id")
    return str(artifact_id) if artifact_id else None


def _conversion_artifact_matches_model(
    artifact_model_id: str,
    metadata: dict[str, Any],
    model_id: str,
) -> bool:
    if artifact_model_id == model_id:
        return True
    source_model_id = metadata.get("source_model_id")
    if source_model_id == model_id:
        return True
    request = metadata.get("request")
    if isinstance(request, dict) and request.get("model_id") == model_id:
        return True
    return False
