"""Model selection and routing services."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    CapabilityName,
    CapabilityReadinessState,
    ConversionStatus,
    GenerateMessage,
    HostCapabilityReadiness,
    MeasuredCapabilityCategory,
    MeasuredCapabilityEvidenceSource,
    MeasuredCapabilityProbeRecord,
    MeasuredCapabilityStatus,
    MeasuredCapabilitySummary,
    ModelCapabilityReport,
    ModelCapabilityStatus,
    ModelManifest,
    ModelModality,
    RequestModality,
    RoutingDecision,
    RoutingModalityPath,
    RuntimeContract,
    ServiceReadinessState,
    ServiceReadinessSummary,
)
from lewlm.core.errors import RoutingError
from lewlm.registry.service import ModelRegistry
from lewlm.telemetry.probes import summarize_measured_capabilities
from lewlm.routing.measured_preferences import (
    RuntimePreferenceAssessment,
    assess_runtime_preference,
    runtime_preference_comparison_suffix,
    runtime_preference_matches,
)
from lewlm.runtime.catalog import RuntimeCatalog
from lewlm.runtime.experimental import build_frontier_serving_plan, frontier_plan_notes, frontier_plan_summary
from lewlm.utils.model_identity import build_manifest_validation_key
from lewlm.utils.validation_manifests import (
    apply_external_validation_to_model_targets,
    load_validation_manifests,
)


@dataclass(slots=True)
class _ScoredCandidate:
    manifest: ModelManifest
    runtime: RuntimeContract
    score: float
    reasons: list[str]


@dataclass(slots=True)
class _ChatRequestProfile:
    request_modality: RequestModality
    required_modalities: tuple[ModelModality, ...]


class ModelRouter:
    """Select models and runtimes for incoming requests."""

    _capability_priority: tuple[CapabilityName, ...] = (
        CapabilityName.CHAT,
        CapabilityName.STREAMING,
        CapabilityName.VISION,
        CapabilityName.EMBEDDINGS,
        CapabilityName.RERANK,
        CapabilityName.AUDIO_TRANSCRIPTION,
        CapabilityName.AUDIO_SPEECH,
    )
    _consumer_readiness_capabilities: tuple[CapabilityName, ...] = (
        CapabilityName.CHAT,
        CapabilityName.STREAMING,
        CapabilityName.EMBEDDINGS,
        CapabilityName.RERANK,
        CapabilityName.AUDIO_TRANSCRIPTION,
        CapabilityName.AUDIO_SPEECH,
    )

    def __init__(
        self,
        *,
        model_registry: ModelRegistry,
        runtime_catalog: RuntimeCatalog,
        settings: LewLMSettings,
    ) -> None:
        self.model_registry = model_registry
        self.runtime_catalog = runtime_catalog
        self.settings = settings

    def route_chat(
        self,
        requested_model_id: str | None = None,
        *,
        messages: list[GenerateMessage] | None = None,
        max_tokens: int = 512,
    ) -> tuple[ModelManifest, RuntimeContract, RoutingDecision]:
        chat_profile = self._chat_request_profile(messages)
        return self.route_capability(
            capability=CapabilityName.CHAT,
            requested_model_id=requested_model_id,
            required_modalities=chat_profile.required_modalities,
            requested_context_tokens=self._estimate_chat_context_tokens(messages, max_tokens),
            request_modality=chat_profile.request_modality,
        )

    def route_embeddings(
        self,
        requested_model_id: str | None = None,
        *,
        inputs: list[str] | None = None,
    ) -> tuple[ModelManifest, RuntimeContract, RoutingDecision]:
        return self.route_capability(
            capability=CapabilityName.EMBEDDINGS,
            requested_model_id=requested_model_id,
            required_modalities=(ModelModality.EMBEDDING,),
            requested_context_tokens=self._estimate_embedding_context_tokens(inputs),
        )

    def route_rerank(
        self,
        requested_model_id: str | None = None,
        *,
        query: str | None = None,
        documents: list[str] | None = None,
    ) -> tuple[ModelManifest, RuntimeContract, RoutingDecision]:
        return self.route_capability(
            capability=CapabilityName.RERANK,
            requested_model_id=requested_model_id,
            required_modalities=(ModelModality.RERANK,),
            requested_context_tokens=self._estimate_rerank_context_tokens(query, documents),
        )

    def route_audio_transcription(
        self,
        requested_model_id: str | None = None,
    ) -> tuple[ModelManifest, RuntimeContract, RoutingDecision]:
        return self.route_capability(
            capability=CapabilityName.AUDIO_TRANSCRIPTION,
            requested_model_id=requested_model_id,
            required_modalities=(ModelModality.AUDIO,),
        )

    def route_audio_speech(self, requested_model_id: str | None = None) -> tuple[ModelManifest, RuntimeContract, RoutingDecision]:
        return self.route_capability(
            capability=CapabilityName.AUDIO_SPEECH,
            requested_model_id=requested_model_id,
            required_modalities=(ModelModality.AUDIO,),
        )

    def route_capability(
        self,
        *,
        capability: CapabilityName,
        requested_model_id: str | None = None,
        required_modalities: tuple[ModelModality, ...] = (),
        requested_context_tokens: int | None = None,
        request_modality: RequestModality | None = None,
    ) -> tuple[ModelManifest, RuntimeContract, RoutingDecision]:
        alternatives: list[str] = []
        candidates = self._candidate_manifests(
            requested_model_id,
            required_modalities=required_modalities,
            requested_context_tokens=requested_context_tokens,
        )

        if requested_model_id is not None:
            manifest = candidates[0]
            scored_candidates = self._rank_runtime_candidates(
                manifest,
                capability=capability,
                required_modalities=required_modalities,
                requested_context_tokens=requested_context_tokens,
                request_modality=request_modality,
                alternatives=alternatives,
            )
            if not scored_candidates:
                raise self._routing_error(
                    f"No {capability.value.replace('_', ' ')}-capable runtime is currently available for the requested model.",
                    capability=capability,
                    requested_model_id=requested_model_id,
                    required_modalities=required_modalities,
                    requested_context_tokens=requested_context_tokens,
                    alternatives=alternatives[:8],
                )
            selected = scored_candidates[0]
            decision = self._build_routing_decision(
                selected,
                capability=capability,
                requested_context_tokens=requested_context_tokens,
                request_modality=request_modality,
                explicit_request=True,
                alternatives=[
                    *alternatives[:4],
                    *[
                        f"{candidate.manifest.model_id} via {candidate.runtime.name}: lower routing score ({candidate.score:.1f})."
                        for candidate in scored_candidates[1:4]
                    ],
                ],
            )
            return selected.manifest, selected.runtime, decision

        scored_candidates: list[_ScoredCandidate] = []
        for manifest in candidates:
            manifest_candidates = self._rank_runtime_candidates(
                manifest,
                capability=capability,
                required_modalities=required_modalities,
                requested_context_tokens=requested_context_tokens,
                request_modality=request_modality,
                alternatives=alternatives,
            )
            scored_candidates.extend(manifest_candidates)

        if not scored_candidates:
            raise self._routing_error(
                f"No {capability.value.replace('_', ' ')}-capable model/runtime pair is currently available.",
                capability=capability,
                requested_model_id=requested_model_id,
                required_modalities=required_modalities,
                requested_context_tokens=requested_context_tokens,
                alternatives=alternatives[:8],
            )

        scored_candidates.sort(
            key=lambda item: (-item.score, item.manifest.display_name.casefold(), item.manifest.model_id),
        )
        selected = scored_candidates[0]
        decision = self._build_routing_decision(
            selected,
            capability=capability,
            requested_context_tokens=requested_context_tokens,
            request_modality=request_modality,
            explicit_request=False,
            alternatives=[
                *alternatives[:4],
                *[
                    f"{candidate.manifest.model_id} via {candidate.runtime.name}: lower routing score ({candidate.score:.1f})."
                    for candidate in scored_candidates[1:4]
                ],
            ],
        )
        return selected.manifest, selected.runtime, decision

    async def warm_model(self, model_id: str) -> RoutingDecision:
        manifest, runtime, decision = self.route_chat(model_id)
        await runtime.load_model(manifest)
        await runtime.warm_model(model_id)
        return decision

    async def unload_model(self, model_id: str) -> RoutingDecision:
        manifest, runtime, decision = self.route_chat(model_id)
        await runtime.unload_model(model_id)
        return decision

    def model_capability_report(self, model_id: str) -> ModelCapabilityReport:
        manifest = self.model_registry.get_manifest(model_id)
        validation_manifests = load_validation_manifests(self.settings.validation_manifest_paths)
        frontier_plan = build_frontier_serving_plan(manifest=manifest, settings=self.settings)
        capabilities: list[ModelCapabilityStatus] = []
        blocked_reason = None
        if manifest.conversion_status != ConversionStatus.RUNNABLE:
            blocked_reason = (
                f"Model is `{manifest.conversion_status.value}` and must become runnable before runtime capability checks can pass."
            )
        for capability in self._capabilities_for_manifest(manifest):
            if blocked_reason is not None:
                capabilities.append(
                    ModelCapabilityStatus(
                        capability=capability,
                        supported=False,
                        readiness_state=(
                            CapabilityReadinessState.CONVERSION_REQUIRED
                            if manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION
                            else CapabilityReadinessState.BLOCKED
                        ),
                        reason=blocked_reason,
                    ),
                )
                continue
            try:
                request_modality = (
                    RequestModality.TEXT_ONLY
                    if capability in {CapabilityName.CHAT, CapabilityName.STREAMING}
                    else None
                )
                alternatives: list[str] = []
                runtimes = self._candidate_runtimes(
                    manifest,
                    capability=capability,
                    request_modality=request_modality,
                    alternatives=alternatives,
                )
                runtime = self._preferred_runtime(
                    manifest,
                    capability=capability,
                    request_modality=request_modality,
                    runtimes=runtimes,
                )
                estimate = runtime.estimate_resources(manifest)
                runtime_preference = self._runtime_preference_assessment(
                    manifest,
                    capability,
                    request_modality=request_modality,
                )
                runtime_reason = f"Supported via `{runtime.name}`."
                if runtime_preference is not None and runtime_preference.adopted:
                    runtime_reason = f"{runtime_reason[:-1]} with benchmark-backed local routing preference."
                elif runtime_preference is not None and runtime_preference.downgrade_reason is not None:
                    runtime_reason = (
                        f"{runtime_reason[:-1]} after measured routing downgraded a partial-support path "
                        f"because {runtime_preference.downgrade_reason}."
                    )
                modality_path, modality_path_reason = self._modality_routing_details(
                    manifest,
                    runtime,
                    request_modality=request_modality,
                )
                capability_notes = [*estimate.notes, *frontier_plan_notes(frontier_plan)]
                if runtime_preference is not None:
                    capability_notes.extend(runtime_preference.notes)
                if modality_path is not None:
                    capability_notes.append(f"Default `{capability.value}` report assumes `{request_modality.value}` routing.")
                if modality_path_reason is not None:
                    capability_notes.append(modality_path_reason)
                capabilities.append(
                    ModelCapabilityStatus(
                        capability=capability,
                        supported=True,
                        readiness_state=CapabilityReadinessState.READY,
                        runtime_name=runtime.name,
                        runtime_affinity=runtime.affinity,
                        reason=runtime_reason,
                        estimated_memory_mb=estimate.estimated_memory_mb,
                        notes=capability_notes,
                    ),
                )
            except RoutingError as exc:
                capabilities.append(
                    ModelCapabilityStatus(
                        capability=capability,
                        supported=False,
                        readiness_state=CapabilityReadinessState.RUNTIME_UNAVAILABLE,
                        reason=str(exc),
                        alternatives=_string_list(exc.details.get("alternatives")),
                        notes=frontier_plan_notes(frontier_plan),
                    ),
                )
        return ModelCapabilityReport(
            model_id=manifest.model_id,
            display_name=manifest.display_name,
            architecture_family=manifest.architecture_family,
            format_type=manifest.format_type,
            modality=manifest.modality,
            quantization=manifest.quantization,
            quantization_profile=manifest.quantization_profile,
            validation_key=build_manifest_validation_key(manifest),
            conversion_status=manifest.conversion_status,
            host_platform=self.runtime_catalog.host_platform_snapshot(),
            runtime_candidates=self.runtime_catalog.describe_manifest_runtimes(manifest),
            target_platforms=apply_external_validation_to_model_targets(
                self.runtime_catalog.describe_manifest_targets(manifest),
                manifest=manifest,
                validation_manifests=validation_manifests,
            ),
            capabilities=capabilities,
            measured_capabilities=self._measured_capabilities_for_model(manifest.model_id),
        )

    def _measured_capabilities_for_model(self, model_id: str) -> list[MeasuredCapabilitySummary]:
        manifest = self.model_registry.get_manifest(model_id)
        host_platform = self.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
        if manifest.conversion_status == ConversionStatus.RUNNABLE:
            try:
                _, runtime, _ = self.route_chat(
                    manifest.model_id,
                    messages=[GenerateMessage(role="user", content="Measured constrained decoding probe")],
                    max_tokens=8,
                )
            except RoutingError:
                pass
            else:
                if not self._has_constrained_decoding_benchmark_evidence(
                    host_platform=host_platform,
                    model_id=manifest.model_id,
                    runtime_name=runtime.name,
                ):
                    self.model_registry.metadata_store.upsert_capability_probe_record(
                        category=MeasuredCapabilityCategory.CONSTRAINED_DECODING.value,
                        probe_name="prompt_guided_structured_output",
                        host_platform=host_platform,
                        status=MeasuredCapabilityStatus.REJECTED.value,
                        source=MeasuredCapabilityEvidenceSource.CODE_PROBE.value,
                        reason=(
                            "LewLM still records structured output as prompt-guided fallback on this host; "
                            "decode-time constrained decoding is not yet benchmark-verified for the routed runtime."
                        ),
                        runtime_name=runtime.name,
                        runtime_affinity=runtime.affinity.value,
                        model_id=manifest.model_id,
                        details={
                            "enforcement": "prompt_guided_fallback",
                            "capability": CapabilityName.CHAT.value,
                        },
                    )
        records = [
            MeasuredCapabilityProbeRecord.model_validate(payload)
            for payload in self.model_registry.metadata_store.list_capability_probe_records(
                host_platform=host_platform,
                model_id=model_id,
            )
        ]
        return summarize_measured_capabilities(records)

    def _has_constrained_decoding_benchmark_evidence(
        self,
        *,
        host_platform: dict[str, object],
        model_id: str,
        runtime_name: str,
    ) -> bool:
        records = [
            MeasuredCapabilityProbeRecord.model_validate(payload)
            for payload in self.model_registry.metadata_store.list_capability_probe_records(
                host_platform=host_platform,
                model_id=model_id,
                runtime_name=runtime_name,
                category=MeasuredCapabilityCategory.CONSTRAINED_DECODING.value,
            )
        ]
        return any(record.source == MeasuredCapabilityEvidenceSource.BENCHMARK_SCENARIO for record in records)

    def capability_readiness(self, capability: CapabilityName) -> HostCapabilityReadiness:
        manifests = self.model_registry.list_manifests()
        candidate_manifests = [manifest for manifest in manifests if capability in self._capabilities_for_manifest(manifest)]
        ready_model_ids: list[str] = []
        blocked_model_ids: list[str] = []
        conversion_required_model_ids: list[str] = []
        available_runtime_names: set[str] = set()
        notes: list[str] = []
        request_modality = (
            RequestModality.TEXT_ONLY if capability in {CapabilityName.CHAT, CapabilityName.STREAMING} else None
        )

        for manifest in candidate_manifests:
            if manifest.conversion_status == ConversionStatus.REQUIRES_CONVERSION:
                conversion_required_model_ids.append(manifest.model_id)
                continue
            if manifest.conversion_status != ConversionStatus.RUNNABLE:
                blocked_model_ids.append(manifest.model_id)
                continue
            alternatives: list[str] = []
            try:
                runtimes = self._candidate_runtimes(
                    manifest,
                    capability=capability,
                    request_modality=request_modality,
                    alternatives=alternatives,
                )
                runtime = self._preferred_runtime(
                    manifest,
                    capability=capability,
                    request_modality=request_modality,
                    runtimes=runtimes,
                )
                ready_model_ids.append(manifest.model_id)
                available_runtime_names.add(runtime.name)
            except RoutingError as exc:
                blocked_model_ids.append(manifest.model_id)
                notes.extend(_string_list(exc.details.get("alternatives")))

        runnable_model_count = sum(1 for manifest in candidate_manifests if manifest.conversion_status == ConversionStatus.RUNNABLE)
        if ready_model_ids:
            readiness_state = CapabilityReadinessState.READY
            reason = f"LewLM can serve `{capability.value}` on this host."
        elif not candidate_manifests:
            readiness_state = CapabilityReadinessState.NO_MODELS
            reason = f"No discovered models currently advertise `{capability.value}`."
        elif runnable_model_count:
            readiness_state = CapabilityReadinessState.RUNTIME_UNAVAILABLE
            reason = f"LewLM found runnable `{capability.value}` candidates, but no compatible local runtime is ready."
        elif conversion_required_model_ids:
            readiness_state = CapabilityReadinessState.CONVERSION_REQUIRED
            reason = f"LewLM found `{capability.value}` candidates that still require conversion before they can run."
        else:
            readiness_state = CapabilityReadinessState.BLOCKED
            reason = f"LewLM found `{capability.value}` candidates, but none are currently ready on this host."

        normalized_notes = _unique_list(notes)
        if ready_model_ids and blocked_model_ids:
            normalized_notes.append(f"{len(blocked_model_ids)} candidate model(s) are still blocked for this capability.")
        if conversion_required_model_ids:
            normalized_notes.append(
                f"{len(conversion_required_model_ids)} candidate model(s) would become eligible after conversion."
            )

        return HostCapabilityReadiness(
            capability=capability,
            ready=bool(ready_model_ids),
            readiness_state=readiness_state,
            reason=reason,
            available_runtime_names=sorted(available_runtime_names),
            candidate_model_count=len(candidate_manifests),
            runnable_model_count=runnable_model_count,
            ready_model_ids=ready_model_ids,
            blocked_model_ids=blocked_model_ids,
            conversion_required_model_ids=conversion_required_model_ids,
            notes=normalized_notes,
        )

    def capability_readiness_summary(self) -> ServiceReadinessSummary:
        manifests = self.model_registry.list_manifests()
        runnable_model_count = sum(1 for manifest in manifests if manifest.conversion_status == ConversionStatus.RUNNABLE)
        capabilities = [self.capability_readiness(capability) for capability in self._consumer_readiness_capabilities]
        ready_capability_count = sum(1 for capability in capabilities if capability.ready)
        if ready_capability_count == len(capabilities):
            status = ServiceReadinessState.READY
        elif ready_capability_count > 0:
            status = ServiceReadinessState.PARTIAL
        else:
            status = ServiceReadinessState.BLOCKED
        notes: list[str] = []
        if not manifests:
            notes.append("No models have been scanned yet.")
        elif runnable_model_count == 0:
            notes.append("Discovered models exist, but none are runnable yet on this host.")
        if any(item.readiness_state == CapabilityReadinessState.CONVERSION_REQUIRED for item in capabilities):
            notes.append("Some capabilities are one conversion step away from becoming ready.")
        return ServiceReadinessSummary(
            status=status,
            host_platform=self.runtime_catalog.host_platform_snapshot(),
            discovered_model_count=len(manifests),
            runnable_model_count=runnable_model_count,
            capability_count=len(capabilities),
            ready_capability_count=ready_capability_count,
            capabilities=capabilities,
            notes=notes,
        )

    def _candidate_manifests(
        self,
        requested_model_id: str | None,
        *,
        required_modalities: tuple[ModelModality, ...] = (),
        requested_context_tokens: int | None = None,
    ) -> list[ModelManifest]:
        if requested_model_id is not None:
            manifest = self.model_registry.get_manifest(requested_model_id)
            if manifest.conversion_status != ConversionStatus.RUNNABLE:
                raise self._routing_error(
                    "The requested model is not runnable yet.",
                    capability=None,
                    requested_model_id=requested_model_id,
                    required_modalities=required_modalities,
                    requested_context_tokens=requested_context_tokens,
                    extra_details={"conversion_status": manifest.conversion_status.value},
                )
            if required_modalities and not all(modality in manifest.modality for modality in required_modalities):
                raise self._routing_error(
                    "The requested model does not support the required modalities.",
                    capability=None,
                    requested_model_id=requested_model_id,
                    required_modalities=required_modalities,
                    requested_context_tokens=requested_context_tokens,
                    extra_details={"model_modalities": [modality.value for modality in manifest.modality]},
                )
            if (
                requested_context_tokens is not None
                and manifest.context_length is not None
                and manifest.context_length < requested_context_tokens
            ):
                raise self._routing_error(
                    "The requested model does not satisfy the estimated context requirement.",
                    capability=None,
                    requested_model_id=requested_model_id,
                    required_modalities=required_modalities,
                    requested_context_tokens=requested_context_tokens,
                    extra_details={"model_context_length": manifest.context_length},
                )
            return [manifest]

        manifests = [
            manifest
            for manifest in self.model_registry.list_manifests()
            if manifest.conversion_status == ConversionStatus.RUNNABLE
            and all(modality in manifest.modality for modality in required_modalities)
        ]
        if manifests:
            return manifests
        capability_hint = ", ".join(modality.value for modality in required_modalities) or "runnable"
        raise self._routing_error(
            f"No runnable {capability_hint}-capable models have been registered yet.",
            capability=None,
            requested_model_id=requested_model_id,
            required_modalities=required_modalities,
            requested_context_tokens=requested_context_tokens,
        )

    def _candidate_runtimes(
        self,
        manifest: ModelManifest,
        *,
        capability: CapabilityName,
        request_modality: RequestModality | None,
        alternatives: list[str],
    ) -> list[RuntimeContract]:
        runtimes, candidate_alternatives = self.runtime_catalog.compatible_runtimes(
            manifest,
            capability=capability,
            request_modality=request_modality,
        )
        if candidate_alternatives:
            alternatives.extend(f"{manifest.model_id}: {item}" for item in candidate_alternatives)
        if runtimes:
            return runtimes
        raise RoutingError(
            "No compatible runtime is currently available for the selected model.",
            details={
                "model_id": manifest.model_id,
                "requested_capability": capability.value,
                "alternatives": candidate_alternatives,
            },
        )

    def _rank_runtime_candidates(
        self,
        manifest: ModelManifest,
        *,
        capability: CapabilityName,
        required_modalities: tuple[ModelModality, ...],
        requested_context_tokens: int | None,
        request_modality: RequestModality | None,
        alternatives: list[str],
    ) -> list[_ScoredCandidate]:
        try:
            runtimes = self._candidate_runtimes(
                manifest,
                capability=capability,
                request_modality=request_modality,
                alternatives=alternatives,
            )
        except RoutingError:
            return []
        scored_candidates: list[_ScoredCandidate] = []
        for runtime in runtimes:
            candidate, rejected_reason = self._score_candidate(
                manifest,
                runtime,
                capability=capability,
                required_modalities=required_modalities,
                requested_context_tokens=requested_context_tokens,
                request_modality=request_modality,
            )
            if candidate is None:
                if rejected_reason is not None:
                    alternatives.append(rejected_reason)
                continue
            scored_candidates.append(candidate)
        scored_candidates.sort(
            key=lambda item: (-item.score, item.manifest.display_name.casefold(), item.manifest.model_id, item.runtime.name),
        )
        return scored_candidates

    def _preferred_runtime(
        self,
        manifest: ModelManifest,
        *,
        capability: CapabilityName,
        request_modality: RequestModality | None = None,
        runtimes: list[RuntimeContract],
    ) -> RuntimeContract:
        assessment = self._runtime_preference_assessment(
            manifest,
            capability,
            request_modality=request_modality,
        )
        if assessment is None:
            return runtimes[0]
        for runtime in runtimes:
            if runtime_preference_matches(
                assessment,
                runtime_name=runtime.name,
                runtime_affinity=runtime.affinity.value,
            ):
                return runtime
        return runtimes[0]

    def _routing_error(
        self,
        message: str,
        *,
        capability: CapabilityName | None,
        requested_model_id: str | None,
        required_modalities: tuple[ModelModality, ...],
        requested_context_tokens: int | None,
        alternatives: list[str] | None = None,
        extra_details: dict[str, object] | None = None,
    ) -> RoutingError:
        return RoutingError(
            message,
            details={
                "requested_capability": capability.value if capability is not None else None,
                "requested_model_id": requested_model_id,
                "required_modalities": [modality.value for modality in required_modalities],
                "estimated_context_tokens": requested_context_tokens,
                "alternatives": list(alternatives or []),
                "fallback_guidance": self._fallback_guidance(
                    capability=capability,
                    requested_model_id=requested_model_id,
                    required_modalities=required_modalities,
                ),
                **(extra_details or {}),
            },
        )

    def _enrich_routing_error(
        self,
        exc: RoutingError,
        *,
        capability: CapabilityName | None,
        requested_model_id: str | None,
        required_modalities: tuple[ModelModality, ...],
        requested_context_tokens: int | None,
    ) -> RoutingError:
        details = dict(exc.details)
        details.setdefault("requested_capability", capability.value if capability is not None else None)
        details.setdefault("requested_model_id", requested_model_id)
        details.setdefault("required_modalities", [modality.value for modality in required_modalities])
        details.setdefault("estimated_context_tokens", requested_context_tokens)
        details["fallback_guidance"] = self._fallback_guidance(
            capability=capability,
            requested_model_id=requested_model_id,
            required_modalities=required_modalities,
        )
        return RoutingError(str(exc), details=details)

    def _fallback_guidance(
        self,
        *,
        capability: CapabilityName | None,
        requested_model_id: str | None,
        required_modalities: tuple[ModelModality, ...],
    ) -> list[str]:
        guidance = ["Run `lewlm list-models --json` or call `GET /v1/models` to inspect runnable local models."]
        if requested_model_id is not None:
            guidance.append(
                f"Inspect `lewlm capabilities {requested_model_id}` or `GET /v1/models/{requested_model_id}/capabilities` "
                "for runtime and capability diagnostics.",
            )
        if capability is not None:
            guidance.append(f"Choose a model that supports `{capability.value}` on the current host.")
        if required_modalities:
            modality_hint = ", ".join(modality.value for modality in required_modalities)
            guidance.append(f"Retry with a model that includes these modalities: {modality_hint}.")
        if requested_model_id is None:
            guidance.append("If the registry looks stale, rerun `lewlm scan` before retrying the request.")
        return guidance

    def _score_candidate(
        self,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        *,
        capability: CapabilityName,
        required_modalities: tuple[ModelModality, ...],
        requested_context_tokens: int | None,
        request_modality: RequestModality | None,
    ) -> tuple[_ScoredCandidate | None, str | None]:
        score = 0.0
        reasons: list[str] = []
        policy = self.settings.runtime_policy

        if requested_context_tokens is not None:
            if manifest.context_length is None:
                if requested_context_tokens >= 4_096:
                    return (
                        None,
                        (
                            f"{manifest.model_id}: context length is unknown for an estimated request size of "
                            f"{requested_context_tokens} tokens."
                        ),
                    )
                score -= 2.0
                reasons.append("context length unknown")
            elif manifest.context_length < requested_context_tokens:
                return (
                    None,
                    (
                        f"{manifest.model_id}: context length {manifest.context_length} is below the estimated "
                        f"request size of {requested_context_tokens} tokens."
                    ),
                )
            else:
                headroom = manifest.context_length - requested_context_tokens
                score += min(20.0, headroom / 1024)
                reasons.append(f"context fit {requested_context_tokens}/{manifest.context_length} tokens")

        memory_budget_mb = self._memory_budget_mb()
        routing_memory_mb = self._routing_memory_estimate_mb(manifest)
        if manifest.estimated_memory_mb is None:
            score -= 1.0
            reasons.append("memory estimate unavailable")
        elif memory_budget_mb is not None:
            if routing_memory_mb is not None and routing_memory_mb > memory_budget_mb:
                return (
                    None,
                    (
                        f"{manifest.model_id}: estimated memory {routing_memory_mb} MB exceeds the "
                        f"`{policy}` routing budget of {memory_budget_mb} MB."
                    ),
                )
            fit_ratio = (routing_memory_mb or manifest.estimated_memory_mb) / max(1, memory_budget_mb)
            score += max(2.0, (1.0 - fit_ratio) * 24.0)
            reasons.append(f"memory fit {(routing_memory_mb or manifest.estimated_memory_mb)}/{memory_budget_mb} MB")

        if runtime.is_model_loaded(manifest.model_id):
            score += {"keep_warm": 45.0, "balanced": 20.0, "aggressive_unload": 6.0}[policy]
            reasons.append("already loaded")
        else:
            score += {"keep_warm": 0.0, "balanced": 4.0, "aggressive_unload": 1.0}[policy]
            reasons.append("not yet loaded")
        frontier_summary = frontier_plan_summary(build_frontier_serving_plan(manifest=manifest, settings=self.settings))
        if frontier_summary is not None:
            reasons.append(frontier_summary)

        affinity_rank = manifest.runtime_affinity.index(runtime.affinity) if runtime.affinity in manifest.runtime_affinity else len(manifest.runtime_affinity)
        score += max(0.0, 4.0 - affinity_rank)

        if policy == "aggressive_unload" and manifest.estimated_memory_mb is not None:
            score -= manifest.estimated_memory_mb / 256
        elif policy == "balanced" and manifest.estimated_memory_mb is not None:
            score -= manifest.estimated_memory_mb / 1024
        elif policy == "keep_warm" and manifest.estimated_memory_mb is not None:
            score -= manifest.estimated_memory_mb / 4096

        if capability == CapabilityName.CHAT and ModelModality.VISION in required_modalities and ModelModality.VISION in manifest.modality:
            score += 8.0
            reasons.append("includes required vision support")

        modality_path, modality_path_reason = self._modality_routing_details(
            manifest,
            runtime,
            request_modality=request_modality,
        )
        if request_modality == RequestModality.TEXT_ONLY and manifest.text_only_runtime_affinity:
            if runtime.affinity in manifest.text_only_runtime_affinity:
                score += 80.0
                reasons.append("text-only fast path enabled")
            elif runtime.affinity in manifest.runtime_affinity:
                score -= 40.0
                reasons.append("text-only request would stay on the default multimodal runtime")
        elif request_modality is not None and modality_path == RoutingModalityPath.MULTIMODAL_DEFAULT:
            reasons.append("attachment-bearing request stayed on the multimodal runtime")
        if modality_path_reason is not None and modality_path_reason not in reasons:
            reasons.append(modality_path_reason)

        runtime_preference = self._runtime_preference_assessment(
            manifest,
            capability,
            request_modality=request_modality,
        )
        if runtime_preference is not None:
            if runtime_preference_matches(
                runtime_preference,
                runtime_name=runtime.name,
                runtime_affinity=runtime.affinity.value,
            ):
                score += 32.0
                if runtime_preference.adopted:
                    preference_reason = f"benchmark preferred `{runtime.name}`"
                    preference_reason += runtime_preference_comparison_suffix(runtime_preference)
                else:
                    preference_reason = f"measured downgrade kept `{runtime.name}` as the safe default"
                    if runtime_preference.downgrade_reason is not None:
                        preference_reason += f" because {runtime_preference.downgrade_reason}"
                reasons.append(preference_reason)
            else:
                score -= 6.0
                preferred_runtime = runtime_preference.effective_runtime_name or runtime_preference.effective_runtime_affinity
                if runtime_preference.adopted:
                    reasons.append(f"benchmark preference favored `{preferred_runtime}`")
                else:
                    reasons.append(f"measured downgrade kept `{preferred_runtime}` as the safe default")

        return _ScoredCandidate(manifest=manifest, runtime=runtime, score=score, reasons=reasons), None

    def _build_routing_decision(
        self,
        candidate: _ScoredCandidate,
        *,
        capability: CapabilityName,
        requested_context_tokens: int | None,
        request_modality: RequestModality | None,
        explicit_request: bool,
        alternatives: list[str],
    ) -> RoutingDecision:
        modality_path, modality_path_reason = self._modality_routing_details(
            candidate.manifest,
            candidate.runtime,
            request_modality=request_modality,
        )
        reason = (
            self._build_explicit_reason(
                candidate.manifest,
                candidate.runtime,
                capability,
                requested_context_tokens=requested_context_tokens,
                score_reasons=candidate.reasons,
            )
            if explicit_request
            else self._build_candidate_reason(candidate, capability)
        )
        return RoutingDecision(
            model_id=candidate.manifest.model_id,
            runtime_name=candidate.runtime.name,
            runtime_affinity=candidate.runtime.affinity,
            reason=reason,
            request_modality=request_modality,
            modality_path=modality_path,
            modality_path_reason=modality_path_reason,
            alternatives=alternatives,
        )

    def _build_explicit_reason(
        self,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        capability: CapabilityName,
        *,
        requested_context_tokens: int | None,
        score_reasons: list[str] | None = None,
    ) -> str:
        notes = [f"using explicitly requested model `{manifest.model_id}` via `{runtime.name}` for `{capability.value}`"]
        if requested_context_tokens is not None and manifest.context_length is not None:
            notes.append(f"context fit {requested_context_tokens}/{manifest.context_length} tokens")
        if manifest.estimated_memory_mb is not None:
            budget = self._memory_budget_mb()
            routing_memory_mb = self._routing_memory_estimate_mb(manifest)
            if budget is not None:
                notes.append(
                    f"estimated memory {(routing_memory_mb or manifest.estimated_memory_mb)}/{budget} MB under `{self.settings.runtime_policy}` policy",
                )
            else:
                notes.append(f"estimated memory {routing_memory_mb or manifest.estimated_memory_mb} MB")
        if runtime.is_model_loaded(manifest.model_id):
            notes.append("model already loaded")
        if score_reasons:
            notes.extend(score_reasons)
        frontier_summary = frontier_plan_summary(build_frontier_serving_plan(manifest=manifest, settings=self.settings))
        if frontier_summary is not None:
            notes.append(frontier_summary)
        return "; ".join(notes).capitalize() + "."

    def _build_candidate_reason(self, candidate: _ScoredCandidate, capability: CapabilityName) -> str:
        detail = "; ".join(candidate.reasons)
        return (
            f"Automatically selected `{candidate.manifest.model_id}` via `{candidate.runtime.name}` for `{capability.value}` under "
            f"`{self.settings.runtime_policy}` policy (score {candidate.score:.1f}; {detail})."
        )

    def _runtime_preference_payload(
        self,
        manifest: ModelManifest,
        capability: CapabilityName,
        *,
        request_modality: RequestModality | None = None,
    ) -> dict[str, object] | None:
        host_platform = self.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
        if request_modality == RequestModality.TEXT_ONLY and manifest.text_only_runtime_affinity:
            scoped_capability = f"{capability.value}:{request_modality.value}"
            preference = self.model_registry.metadata_store.get_runtime_preference(
                model_id=manifest.model_id,
                capability=scoped_capability,
                host_platform=host_platform,
            )
            if preference is not None:
                return preference
            if capability == CapabilityName.STREAMING:
                return self.model_registry.metadata_store.get_runtime_preference(
                    model_id=manifest.model_id,
                    capability=f"{CapabilityName.CHAT.value}:{request_modality.value}",
                    host_platform=host_platform,
                )
            return None
        preference = self.model_registry.metadata_store.get_runtime_preference(
            model_id=manifest.model_id,
            capability=capability.value,
            host_platform=host_platform,
        )
        if preference is not None:
            return preference
        if capability == CapabilityName.STREAMING:
            return self.model_registry.metadata_store.get_runtime_preference(
                model_id=manifest.model_id,
                capability=CapabilityName.CHAT.value,
                host_platform=host_platform,
            )
        return None

    def _runtime_preference_assessment(
        self,
        manifest: ModelManifest,
        capability: CapabilityName,
        *,
        request_modality: RequestModality | None = None,
    ) -> RuntimePreferenceAssessment | None:
        return assess_runtime_preference(
            self._runtime_preference_payload(
                manifest,
                capability,
                request_modality=request_modality,
            ),
        )

    def _memory_budget_mb(self) -> int | None:
        host_memory_mb = self._host_memory_mb()
        if host_memory_mb is None:
            return None
        factor = {
            "keep_warm": 0.8,
            "balanced": 0.55,
            "aggressive_unload": 0.35,
        }[self.settings.runtime_policy]
        return max(256, int(host_memory_mb * factor))

    def _host_memory_mb(self) -> int | None:
        return self.runtime_catalog.host_total_memory_mb()

    def _routing_memory_estimate_mb(self, manifest: ModelManifest) -> int | None:
        frontier_plan = build_frontier_serving_plan(manifest=manifest, settings=self.settings)
        if frontier_plan is None:
            return manifest.estimated_memory_mb
        bounded_mode = str(frontier_plan.get("bounded_memory_mode") or "off")
        planned_memory_mb = _coerce_int(frontier_plan.get("planned_memory_mb"))
        full_memory_mb = manifest.estimated_memory_mb
        if (
            manifest.architecture_subtype.value in {"moe", "hybrid_moe"}
            and bounded_mode != "off"
            and planned_memory_mb is not None
            and full_memory_mb is not None
            and planned_memory_mb < full_memory_mb
        ):
            return planned_memory_mb
        return full_memory_mb

    def _modality_routing_details(
        self,
        manifest: ModelManifest,
        runtime: RuntimeContract,
        *,
        request_modality: RequestModality | None,
    ) -> tuple[RoutingModalityPath | None, str | None]:
        if request_modality is None:
            return None, None
        if request_modality == RequestModality.TEXT_ONLY:
            if runtime.affinity in manifest.text_only_runtime_affinity:
                source = manifest.text_only_runtime_source or "same_bundle"
                source_detail = "paired text artifact" if source == "paired_artifact" else "same-bundle text runtime"
                reason = manifest.text_only_runtime_reason or (
                    f"Text-only requests can prefer the declared {source_detail} for this multimodal model."
                )
                return RoutingModalityPath.TEXT_FAST_PATH, reason
            if ModelModality.MULTIMODAL in manifest.modality:
                return (
                    RoutingModalityPath.MULTIMODAL_DEFAULT,
                    "No safe text-only runtime or paired text artifact was declared for this multimodal bundle.",
                )
            return RoutingModalityPath.TEXT_DEFAULT, "The request did not require multimodal conditioning."
        if runtime.affinity in manifest.runtime_affinity and any(
            modality in manifest.modality for modality in (ModelModality.VISION, ModelModality.MULTIMODAL)
        ):
            if request_modality == RequestModality.FRAME_BUNDLE_VIDEO:
                return (
                    RoutingModalityPath.MULTIMODAL_DEFAULT,
                    "Frame-bundle or video-style image attachments require the multimodal runtime path.",
                )
            if request_modality == RequestModality.IMAGE_CONDITIONED:
                return (
                    RoutingModalityPath.MULTIMODAL_DEFAULT,
                    "Image attachments require the multimodal runtime path.",
                )
            return (
                RoutingModalityPath.MULTIMODAL_DEFAULT,
                "Audio-conditioned requests disable the multimodal text-only fast path.",
            )
        return (
            RoutingModalityPath.TEXT_DEFAULT,
            "The request stayed on a text-native path after attachment preprocessing.",
        )

    @staticmethod
    def _chat_request_profile(messages: list[GenerateMessage] | None) -> _ChatRequestProfile:
        required_modalities = [ModelModality.TEXT]
        if messages is None:
            return _ChatRequestProfile(
                request_modality=RequestModality.TEXT_ONLY,
                required_modalities=tuple(required_modalities),
            )
        attachments = [
            attachment
            for message in messages
            for attachment in message.attachments
        ]
        has_image_attachment = any(attachment.attachment_type == "image" for attachment in attachments)
        if has_image_attachment:
            required_modalities.append(ModelModality.VISION)
            request_modality = (
                RequestModality.FRAME_BUNDLE_VIDEO
                if any(ModelRouter._is_frame_bundle_attachment(attachment) for attachment in attachments if attachment.attachment_type == "image")
                else RequestModality.IMAGE_CONDITIONED
            )
            return _ChatRequestProfile(
                request_modality=request_modality,
                required_modalities=tuple(required_modalities),
            )
        if any(attachment.attachment_type == "audio" for attachment in attachments):
            return _ChatRequestProfile(
                request_modality=RequestModality.AUDIO_CONDITIONED,
                required_modalities=tuple(required_modalities),
            )
        return _ChatRequestProfile(
            request_modality=RequestModality.TEXT_ONLY,
            required_modalities=tuple(required_modalities),
        )

    @staticmethod
    def _is_frame_bundle_attachment(attachment) -> bool:
        source_type = attachment.metadata.get("source_type") if isinstance(attachment.metadata, dict) else None
        if source_type == "image_bundle":
            return True
        source_path = attachment.source_path
        if source_path is None:
            return False
        return Path(source_path).expanduser().resolve(strict=False).is_dir()

    @staticmethod
    def _estimate_chat_context_tokens(messages: list[GenerateMessage] | None, max_tokens: int) -> int | None:
        if messages is None:
            return max_tokens
        return sum(_estimate_text_tokens(message.content) for message in messages) + max(0, max_tokens)

    @staticmethod
    def _estimate_embedding_context_tokens(inputs: list[str] | None) -> int | None:
        if not inputs:
            return None
        return sum(_estimate_text_tokens(text) for text in inputs)

    @staticmethod
    def _estimate_rerank_context_tokens(query: str | None, documents: list[str] | None) -> int | None:
        if not query and not documents:
            return None
        query_tokens = _estimate_text_tokens(query or "")
        if not documents:
            return query_tokens
        return max(query_tokens + _estimate_text_tokens(document) for document in documents)

    def _capabilities_for_manifest(self, manifest: ModelManifest) -> tuple[CapabilityName, ...]:
        capabilities: list[CapabilityName] = []
        if any(modality in manifest.modality for modality in (ModelModality.TEXT, ModelModality.MULTIMODAL, ModelModality.VISION)):
            capabilities.extend((CapabilityName.CHAT, CapabilityName.STREAMING))
        if any(modality in manifest.modality for modality in (ModelModality.VISION, ModelModality.MULTIMODAL)):
            capabilities.append(CapabilityName.VISION)
        if ModelModality.EMBEDDING in manifest.modality:
            capabilities.append(CapabilityName.EMBEDDINGS)
        if ModelModality.RERANK in manifest.modality:
            capabilities.append(CapabilityName.RERANK)
        if ModelModality.AUDIO in manifest.modality:
            capabilities.extend((CapabilityName.AUDIO_TRANSCRIPTION, CapabilityName.AUDIO_SPEECH))
        ordered = [capability for capability in self._capability_priority if capability in capabilities]
        return tuple(ordered)


def _estimate_text_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, ceil(len(normalized) / 4))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _unique_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered[:8]
