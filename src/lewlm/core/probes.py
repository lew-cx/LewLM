"""Runtime smoke probes for evidence-backed capability reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from lewlm.core.contracts import (
    CapabilityEvidence,
    CapabilityEvidenceState,
    CapabilityName,
    CapabilityOwnership,
    GenerateMessage,
    GenerateRequest,
    ModelModality,
    RuntimeSupportPath,
    runtime_support_path_for_affinity,
)
from lewlm.core.middleware import _provider_from_runtime_name

RuntimeProbeMode = Literal["load", "generate"]


@dataclass(slots=True)
class RuntimeProbeOutcome:
    """Result of an opt-in runtime smoke probe."""

    model_id: str
    capability: CapabilityName
    mode: RuntimeProbeMode
    evidence: list[CapabilityEvidence]
    reason: str
    generated_text: str | None = None
    persisted: bool = False


async def run_model_smoke_probe(
    services: Any,
    *,
    model_id: str,
    capability: CapabilityName = CapabilityName.CHAT,
    mode: RuntimeProbeMode = "load",
    prompt: str = "LewLM runtime probe",
    max_tokens: int = 1,
) -> RuntimeProbeOutcome:
    """Run an explicit load or generation probe for one model."""

    if mode == "generate" and capability not in {CapabilityName.CHAT, CapabilityName.STREAMING}:
        outcome = _probe_failure(
            model_id=model_id,
            capability=capability,
            mode=mode,
            reason=f"Generation smoke probes currently support chat-like capabilities only, not `{capability.value}`.",
        )
        outcome.persisted = _persist_probe_outcome(services, outcome)
        return outcome
    try:
        manifest, runtime, _ = _route_probe_target(
            services,
            model_id=model_id,
            capability=capability,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        support_path = runtime_support_path_for_affinity(runtime.affinity) or RuntimeSupportPath.PACKAGED
        await runtime.load_model(manifest)
        generated_text: str | None = None
        state = CapabilityEvidenceState.LOAD_PASSED
        reason = f"`{runtime.name}` loaded `{manifest.model_id}` successfully."
        if mode == "generate":
            response = await runtime.generate(
                GenerateRequest(
                    model_id=manifest.model_id,
                    messages=[GenerateMessage(role="user", content=prompt)],
                    max_tokens=max_tokens,
                ),
            )
            generated_text = response.output_text
            state = CapabilityEvidenceState.GENERATE_PASSED
            reason = f"`{runtime.name}` loaded and generated from `{manifest.model_id}` successfully."
        evidence = [
            CapabilityEvidence(
                capability=capability,
                state=state,
                ownership=(
                    CapabilityOwnership.BRIDGE_VERIFIED
                    if support_path == RuntimeSupportPath.BRIDGE
                    else CapabilityOwnership.BACKEND_NATIVE
                ),
                reason=reason,
                runtime_name=runtime.name,
                runtime_affinity=runtime.affinity,
                provider=_provider_from_runtime_name(
                    runtime.name,
                    support_path,
                    affinity=runtime.affinity,
                    external_profile=services.settings.external_accelerator_profile,
                ),
                model_id=manifest.model_id,
                source="runtime_smoke_probe",
                details={
                    "mode": mode,
                    "support_path": support_path.value,
                    "max_tokens": max_tokens,
                },
            ),
        ]
        outcome = RuntimeProbeOutcome(
            model_id=manifest.model_id,
            capability=capability,
            mode=mode,
            evidence=evidence,
            reason=reason,
            generated_text=generated_text,
        )
        outcome.persisted = _persist_probe_outcome(services, outcome)
        return outcome
    except Exception as exc:  # noqa: BLE001 - probes must preserve backend-specific failure details.
        outcome = _probe_failure(
            model_id=model_id,
            capability=capability,
            mode=mode,
            reason=str(exc),
            details={"cause_type": type(exc).__name__},
        )
        outcome.persisted = _persist_probe_outcome(services, outcome)
        return outcome


def _route_probe_target(
    services: Any,
    *,
    model_id: str,
    capability: CapabilityName,
    prompt: str,
    max_tokens: int,
) -> tuple[Any, Any, Any]:
    if capability in {CapabilityName.CHAT, CapabilityName.STREAMING}:
        return services.model_router.route_chat(
            model_id,
            messages=[GenerateMessage(role="user", content=prompt)],
            max_tokens=max_tokens,
        )
    if capability == CapabilityName.VISION:
        return services.model_router.route_capability(
            capability=capability,
            requested_model_id=model_id,
            required_modalities=(ModelModality.VISION,),
        )
    if capability == CapabilityName.EMBEDDINGS:
        return services.model_router.route_embeddings(model_id, inputs=[prompt])
    if capability == CapabilityName.RERANK:
        return services.model_router.route_rerank(model_id, query=prompt, documents=[prompt])
    if capability == CapabilityName.AUDIO_TRANSCRIPTION:
        return services.model_router.route_audio_transcription(model_id)
    if capability == CapabilityName.AUDIO_SPEECH:
        return services.model_router.route_audio_speech(model_id)
    return services.model_router.route_capability(capability=capability, requested_model_id=model_id)


def _persist_probe_outcome(services: Any, outcome: RuntimeProbeOutcome) -> bool:
    metadata_store = getattr(services, "metadata_store", None)
    upsert_runtime_probe_record = getattr(metadata_store, "upsert_runtime_probe_record", None)
    if not callable(upsert_runtime_probe_record):
        return False
    try:
        host_platform_snapshot = services.runtime_catalog.host_platform_snapshot()
        host_platform = (
            host_platform_snapshot.model_dump(mode="json")
            if hasattr(host_platform_snapshot, "model_dump")
            else dict(host_platform_snapshot)
        )
        persisted = False
        for evidence in outcome.evidence:
            evidence_payload = evidence.model_dump(mode="json")
            probe_key = upsert_runtime_probe_record(
                model_id=outcome.model_id,
                capability=outcome.capability.value,
                mode=outcome.mode,
                host_platform=host_platform,
                evidence=evidence_payload,
            )
            evidence.probe_key = probe_key
            persisted = True
        return persisted
    except Exception:  # noqa: BLE001 - persistence should not rewrite runtime probe truth.
        return False


def _probe_failure(
    *,
    model_id: str,
    capability: CapabilityName,
    mode: RuntimeProbeMode,
    reason: str,
    details: dict[str, object] | None = None,
) -> RuntimeProbeOutcome:
    return RuntimeProbeOutcome(
        model_id=model_id,
        capability=capability,
        mode=mode,
        reason=reason,
        evidence=[
            CapabilityEvidence(
                capability=capability,
                state=CapabilityEvidenceState.PROBE_FAILED,
                ownership=CapabilityOwnership.UNVERIFIED,
                reason=reason,
                model_id=model_id,
                source="runtime_smoke_probe",
                details={"mode": mode, **(details or {})},
            ),
        ],
    )
