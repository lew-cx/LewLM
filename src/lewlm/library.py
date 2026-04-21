"""Public embeddable Python API for LewLM."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi import FastAPI

from lewlm.api.app import create_app
from lewlm.config.settings import LewLMSettings
from lewlm.conversion.backend import ConversionBackend
from lewlm.conversion.models import ConversionJobRequest, ConversionPolicy, JobRecord, JobStatus
from lewlm.core.bootstrap import LewLMServices, bootstrap_services
from lewlm.core.chat import ChatExecution, ChatStreamSession
from lewlm.core.citations import CitationContextPackage
from lewlm.core.contracts import (
    GenerateMessage,
    ModelCapabilityReport,
    ModelInventory,
    ModelManifest,
    ModelScanSummary,
    QuantizationProfile,
    ReasoningVisibility,
    RoutingDecision,
    RuntimeAffinity,
    RuntimeContract,
    utc_now,
)
from lewlm.documents.ingest.models import DocumentIngestResult
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.documents.render.service import GeneratedDocumentArtifact
from lewlm.documents.skills.models import BuiltInSkillDescriptor
from lewlm.documents.skills.models import DocumentTransformRequest
from lewlm.events.bus import EventSubscription
from lewlm.history.models import SessionContextPolicy, SessionDetail, SessionExportBundle, SessionRecord
from lewlm.install_profiles import summarize_install_profiles
from lewlm.pack_registry import PackRegistry
from lewlm.prompting import PromptCompilationRequest
from lewlm.telemetry.stats import CacheStats, RuntimeStats
from lewlm.tools.models import LocalToolDescriptor, ToolExecutionEnvelope, ToolExecutionRequest

if TYPE_CHECKING:
    from lewlm.app_helpers import LewLMAppClient


SyncResultT = TypeVar("SyncResultT")


class LewLM:
    """Embeddable facade for LewLM's scan, runtime, document, and operator workflows."""

    def __init__(
        self,
        settings: LewLMSettings | None = None,
        *,
        services: LewLMServices | None = None,
        runtime_overrides: Mapping[RuntimeAffinity, RuntimeContract] | None = None,
        conversion_backend: ConversionBackend | None = None,
    ) -> None:
        if services is not None and (settings is not None or runtime_overrides is not None or conversion_backend is not None):
            raise ValueError("services cannot be combined with settings, runtime_overrides, or conversion_backend.")
        self._owns_services = services is None
        self.services = services or bootstrap_services(
            settings,
            runtime_overrides=runtime_overrides,
            conversion_backend=conversion_backend,
        )
        self.settings = self.services.settings
        self._closed = False

    def __enter__(self) -> "LewLM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def __aenter__(self) -> "LewLM":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def create_app(self) -> FastAPI:
        """Create a FastAPI app bound to this facade's service container."""

        return create_app(services=self.services)

    def config_snapshot(self) -> dict[str, Any]:
        """Return the current redacted configuration snapshot."""

        return self.settings.redacted_snapshot()

    def app_client(self) -> "LewLMAppClient":
        """Return the thin typed helper surface bound to this facade."""

        from lewlm.app_helpers import LewLMAppClient

        return LewLMAppClient.from_lewlm(self)

    def health(self) -> dict[str, Any]:
        """Return an in-process health snapshot aligned with the API contract."""

        storage = self.services.metadata_store.snapshot()
        model_registry = getattr(self.services, "model_registry", None)
        model_count = model_registry.inventory().count if model_registry is not None else storage["model_count"]
        pack_registry = getattr(self.services, "pack_registry", None) or PackRegistry.from_settings(self.settings)
        return {
            "status": "ok",
            "service": self.settings.app_name,
            "version": self.settings.version,
            "time": utc_now(),
            "install_profiles": summarize_install_profiles().model_dump(mode="json"),
            "readiness": self.services.model_router.capability_readiness_summary().model_dump(mode="json"),
            "storage": {
                "healthy": True,
                "database_path": storage["database_path"],
                "schema_version": storage["schema_version"],
                "model_count": model_count,
            },
            "configuration": {
                "data_dir": str(self.settings.data_dir),
                "models_dir": [str(path) for path in self.settings.models_dir],
                "runtime_packs": [report.model_dump(mode="json") for report in pack_registry.runtime_pack_reports()],
                "feature_packs": [report.model_dump(mode="json") for report in pack_registry.feature_pack_reports()],
                "privacy_mode": self.settings.privacy_mode,
                "telemetry_enabled": self.settings.telemetry_enabled,
                "allow_outbound_network": self.settings.allow_outbound_network,
                "audit_log_enabled": self.settings.audit_log_enabled,
                "persistence_encryption_enabled": self.settings.persistence_encryption_enabled,
                "tool_authorization_required": self.settings.tool_authorization_required,
                "parser_sandbox_enabled": self.settings.parser_sandbox_enabled,
                "tool_sandbox_enabled": self.settings.tool_sandbox_enabled,
                "conversion_sandbox_enabled": self.settings.conversion_sandbox_enabled,
            },
            "cluster": self.services.cluster_service.status().model_dump(mode="json"),
        }

    def scan_models(self, roots: Sequence[Path | str] | Path | str | None = None) -> ModelScanSummary:
        """Scan configured or explicit model roots and refresh the local registry."""

        return self.services.model_registry.scan(roots=_normalize_optional_paths(roots))

    def inventory(self) -> ModelInventory:
        """Return the full model inventory."""

        return self.services.model_registry.inventory()

    def list_models(self) -> list[ModelManifest]:
        """Return discovered model manifests."""

        return self.services.model_registry.list_manifests()

    def get_model(self, model_id: str) -> ModelManifest:
        """Return one model manifest from the local registry."""

        return self.services.model_registry.get_manifest(model_id)

    def model_capabilities(self, model_id: str) -> ModelCapabilityReport:
        """Return runtime and target-platform compatibility for one model."""

        return self.services.model_router.model_capability_report(model_id)

    def list_skills(self) -> list[BuiltInSkillDescriptor]:
        """Return the built-in document skill catalog."""

        return self.services.skill_catalog_service.list_skills()

    def get_skill(self, skill_name: str) -> BuiltInSkillDescriptor:
        """Return one built-in document skill descriptor."""

        return self.services.skill_catalog_service.get_skill(skill_name)

    def list_tools(self) -> list[LocalToolDescriptor]:
        """Return the registered local tool catalog."""

        return self.services.tool_catalog_service.list_tools()

    def get_tool(self, tool_name: str) -> LocalToolDescriptor:
        """Return one registered local tool descriptor."""

        return self.services.tool_catalog_service.get_tool(tool_name)

    def create_session(
        self,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        context_policy: SessionContextPolicy = "full_history",
    ) -> SessionRecord:
        """Create a persisted local session when privacy mode allows it."""

        return self.services.session_history_service.create_session(
            title=title,
            metadata=metadata,
            context_policy=context_policy,
        )

    def list_sessions(self) -> list[SessionRecord]:
        """List persisted local sessions."""

        return self.services.session_history_service.list_sessions()

    def get_session(self, session_id: str) -> SessionRecord:
        """Return one persisted session record."""

        return self.services.session_history_service.get_session(session_id)

    def get_session_detail(self, session_id: str) -> SessionDetail:
        """Return one persisted session plus its turns."""

        return self.services.session_history_service.get_session_detail(session_id)

    def session_messages(self, session_id: str) -> list[GenerateMessage]:
        """Return flattened chronological messages for one persisted session."""

        return self.services.session_history_service.list_messages(session_id)

    def export_session(self, session_id: str) -> SessionExportBundle:
        """Export one persisted session as a portable bundle."""

        return self.services.session_history_service.export_session(session_id)

    def import_session(self, bundle: SessionExportBundle, *, title: str | None = None) -> SessionDetail:
        """Import a portable session bundle into local persistence."""

        return self.services.session_history_service.import_session(bundle, title=title)

    def delete_session(self, session_id: str) -> SessionRecord:
        """Delete one persisted local session."""

        return self.services.session_history_service.delete_session(session_id)

    def submit_conversion(
        self,
        request: ConversionJobRequest | None = None,
        *,
        model_id: str | None = None,
        policy: ConversionPolicy = ConversionPolicy.BALANCED,
        custom_bits: int | None = None,
        quantization_profile: QuantizationProfile | None = None,
        force: bool = False,
        idempotency_key: str | None = None,
        authorized_actions: Sequence[str] | None = None,
    ) -> JobRecord:
        """Queue or reuse a conversion job."""

        resolved_request = request
        if resolved_request is None:
            if model_id is None:
                raise ValueError("model_id is required when request is not provided.")
            resolved_request = ConversionJobRequest(
                model_id=model_id,
                policy=policy,
                custom_bits=custom_bits,
                quantization_profile=quantization_profile,
                force=force,
                idempotency_key=idempotency_key,
                authorized_actions=list(authorized_actions or ()),
            )
        return self.services.conversion_service.submit(resolved_request)

    def get_job(self, job_id: str) -> JobRecord:
        """Return a conversion job record."""

        return self.services.conversion_service.get_job(job_id)

    async def wait_for_job_async(
        self,
        job_id: str,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.05,
    ) -> JobRecord:
        """Poll until a conversion job reaches a terminal state."""

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero.")
        deadline = time.monotonic() + timeout_seconds
        while True:
            job = self.get_job(job_id)
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out while waiting for conversion job {job_id}.")
            await asyncio.sleep(poll_interval_seconds)

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.05,
    ) -> JobRecord:
        """Poll until a conversion job reaches a terminal state from synchronous code."""

        return _run_sync(
            lambda: self.wait_for_job_async(
                job_id,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            ),
            helper_name="LewLM.wait_for_job",
            async_name="LewLM.wait_for_job_async",
        )

    async def chat(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[GenerateMessage] | None = None,
        model_id: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        reasoning_visibility: ReasoningVisibility | None = None,
        apply_serving_profile: bool = True,
        citation_context: CitationContextPackage | None = None,
        prompt_request: PromptCompilationRequest | None = None,
        allowed_prompt_file_roots: Sequence[Path | str] | None = None,
    ) -> ChatExecution:
        """Run one non-streaming chat completion."""

        return await self.services.chat_orchestrator.complete(
            model_id=model_id,
            messages=_normalize_messages(prompt=prompt, messages=messages),
            citation_context=citation_context,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_visibility=reasoning_visibility or self.settings.reasoning_visibility,
            apply_serving_profile=apply_serving_profile,
            prompt_request=prompt_request,
            allowed_prompt_file_roots=allowed_prompt_file_roots,
        )

    def chat_sync(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[GenerateMessage] | None = None,
        model_id: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        reasoning_visibility: ReasoningVisibility | None = None,
        apply_serving_profile: bool = True,
        citation_context: CitationContextPackage | None = None,
        prompt_request: PromptCompilationRequest | None = None,
        allowed_prompt_file_roots: Sequence[Path | str] | None = None,
    ) -> ChatExecution:
        """Run one non-streaming chat completion from synchronous code."""

        return _run_sync(
            lambda: self.chat(
                prompt,
                messages=messages,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_visibility=reasoning_visibility,
                apply_serving_profile=apply_serving_profile,
                citation_context=citation_context,
                prompt_request=prompt_request,
                allowed_prompt_file_roots=allowed_prompt_file_roots,
            ),
            helper_name="LewLM.chat_sync",
            async_name="LewLM.chat",
        )

    async def stream_chat(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[GenerateMessage] | None = None,
        model_id: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        reasoning_visibility: ReasoningVisibility | None = None,
        apply_serving_profile: bool = True,
        citation_context: CitationContextPackage | None = None,
        prompt_request: PromptCompilationRequest | None = None,
        allowed_prompt_file_roots: Sequence[Path | str] | None = None,
    ) -> ChatStreamSession:
        """Start a streaming chat completion."""

        return await self.services.chat_orchestrator.stream(
            model_id=model_id,
            messages=_normalize_messages(prompt=prompt, messages=messages),
            citation_context=citation_context,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_visibility=reasoning_visibility or self.settings.reasoning_visibility,
            apply_serving_profile=apply_serving_profile,
            prompt_request=prompt_request,
            allowed_prompt_file_roots=allowed_prompt_file_roots,
        )

    async def warm_model(self, model_id: str) -> RoutingDecision:
        """Load and warm one model in its selected runtime."""

        return await self.services.model_router.warm_model(model_id)

    def warm_model_sync(self, model_id: str) -> RoutingDecision:
        """Load and warm one model in its selected runtime from synchronous code."""

        return _run_sync(
            lambda: self.warm_model(model_id),
            helper_name="LewLM.warm_model_sync",
            async_name="LewLM.warm_model",
        )

    async def unload_model(self, model_id: str) -> RoutingDecision:
        """Unload one model from its selected runtime."""

        return await self.services.model_router.unload_model(model_id)

    def unload_model_sync(self, model_id: str) -> RoutingDecision:
        """Unload one model from synchronous code."""

        return _run_sync(
            lambda: self.unload_model(model_id),
            helper_name="LewLM.unload_model_sync",
            async_name="LewLM.unload_model",
        )

    def generate_document(
        self,
        document: DocumentIR,
        *,
        output_format: DocumentOutputFormat,
        file_name: str | None = None,
        allowed_file_roots: Sequence[Path | str] | None = None,
        base_dir: Path | str | None = None,
        request_id: str | None = None,
    ) -> GeneratedDocumentArtifact:
        """Render a deterministic document artifact."""

        return self.services.document_generation_service.generate(
            document,
            output_format=output_format,
            file_name=file_name,
            allowed_file_roots=allowed_file_roots,
            base_dir=base_dir,
            request_id=request_id,
        )

    def ingest_documents(
        self,
        paths: Sequence[Path | str] | Path | str,
        *,
        title: str | None = None,
        allowed_file_roots: Sequence[Path | str] | None = None,
        base_dir: Path | str | None = None,
        request_id: str | None = None,
    ) -> DocumentIngestResult:
        """Ingest one or more local files into `DocumentIR`."""

        return self.services.document_ingest_service.ingest(
            _normalize_required_paths(paths),
            title=title,
            allowed_file_roots=allowed_file_roots,
            base_dir=base_dir,
            request_id=request_id,
        )

    def transform_document(
        self,
        request: DocumentTransformRequest,
        *,
        allowed_file_roots: Sequence[Path | str] | None = None,
        base_dir: Path | str | None = None,
        request_id: str | None = None,
    ) -> GeneratedDocumentArtifact:
        """Run one built-in document transform workflow."""

        return self.services.document_transform_service.transform(
            request,
            allowed_file_roots=allowed_file_roots,
            base_dir=base_dir,
            request_id=request_id,
        )

    def execute_tool(
        self,
        request: ToolExecutionRequest,
        *,
        actor: str = "cli",
        allowed_file_roots: Sequence[Path | str] | None = None,
        base_dir: Path | str | None = None,
        request_id: str | None = None,
        emit_tool_events: bool = True,
    ) -> ToolExecutionEnvelope:
        """Execute one registered local tool with the shared LewLM tool contract."""

        return self.services.tool_execution_service.execute(
            request,
            actor=actor,
            allowed_file_roots=allowed_file_roots,
            base_dir=base_dir,
            request_id=request_id,
            emit_tool_events=emit_tool_events,
        )

    def cache_stats(self) -> CacheStats:
        """Return conversion-cache and runtime-response-cache statistics."""

        return self.services.telemetry_service.cache_stats()

    async def runtime_stats(self) -> RuntimeStats:
        """Return runtime, scheduler, and target-platform diagnostics."""

        return await self.services.telemetry_service.runtime_stats()

    def runtime_stats_sync(self) -> RuntimeStats:
        """Return runtime diagnostics from synchronous code."""

        return _run_sync(
            self.runtime_stats,
            helper_name="LewLM.runtime_stats_sync",
            async_name="LewLM.runtime_stats",
        )

    def subscribe_events(self) -> EventSubscription:
        """Subscribe to the in-process event bus from an active asyncio loop."""

        return self.services.event_bus.subscribe()

    def close(self) -> None:
        """Release long-lived worker resources owned by this facade."""

        if self._closed:
            return
        if self._owns_services:
            _run_sync(
                self.aclose,
                helper_name="LewLM.close",
                async_name="LewLM.aclose",
            )
            return
        self._closed = True

    async def aclose(self) -> None:
        """Release long-lived resources owned by this facade from async code."""

        if self._closed:
            return
        if self._owns_services:
            await self.services.aclose()
        self._closed = True


def _normalize_messages(
    *,
    prompt: str | None,
    messages: Sequence[GenerateMessage] | None,
) -> list[GenerateMessage]:
    if prompt is not None and messages is not None:
        raise ValueError("Pass either prompt or messages, not both.")
    if messages is not None:
        return list(messages)
    if prompt is not None:
        return [GenerateMessage(role="user", content=prompt)]
    raise ValueError("Either prompt or messages is required.")


def _run_sync(
    awaitable_factory: Callable[[], Awaitable[SyncResultT]],
    *,
    helper_name: str,
    async_name: str,
) -> SyncResultT:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable_factory())
    raise RuntimeError(
        f"{helper_name} cannot run inside an active asyncio event loop. Use `await {async_name}(...)` instead.",
    )


def _normalize_optional_paths(
    paths: Sequence[Path | str] | Path | str | None,
) -> list[Path | str] | None:
    if paths is None:
        return None
    return _normalize_required_paths(paths)


def _normalize_required_paths(
    paths: Sequence[Path | str] | Path | str,
) -> list[Path | str]:
    if isinstance(paths, (str, Path)):
        return [paths]
    return list(paths)
