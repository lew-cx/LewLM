"""Experimental frontier-architecture planning runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import GenerateRequest, GenerateResponse, ModelFormat, ModelManifest, ModelModality, RuntimeAffinity, RuntimeEstimate
from lewlm.runtime.base import ManagedTextRuntime
from lewlm.runtime.experimental.architectures import build_frontier_serving_plan, frontier_plan_notes, is_frontier_architecture


class FrontierExperimentalRuntime(ManagedTextRuntime):
    """Planning-only runtime surface for frontier architecture diagnostics."""

    name = "frontier_experimental"
    affinity = RuntimeAffinity.EXPERIMENTAL
    supported_formats = (ModelFormat.GGUF, ModelFormat.MLX, ModelFormat.HUGGINGFACE)
    supported_modalities = (ModelModality.TEXT,)
    supported_capabilities = frozenset()
    platform_guidance = "This runtime only exposes frontier architecture planning and diagnostics today."

    def __init__(self, *, settings: LewLMSettings) -> None:
        super().__init__()
        self.settings = settings

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, "Frontier architecture planning is available; native execution remains backend-specific."

    def supports_manifest(self, manifest: ModelManifest) -> bool:
        return super().supports_manifest(manifest) and is_frontier_architecture(manifest)

    async def _load_model(self, manifest: ModelManifest) -> None:
        return None

    async def _unload_model(self, model_id: str) -> None:
        return None

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        raise NotImplementedError("FrontierExperimentalRuntime is a planning-only runtime and does not execute generation.")

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        if False:
            yield ""
        raise NotImplementedError("FrontierExperimentalRuntime is a planning-only runtime and does not execute streaming.")

    def _tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens: Sequence[int]) -> str:
        return bytes(tokens).decode("utf-8")

    def estimate_resources(self, manifest: ModelManifest) -> RuntimeEstimate:
        plan = build_frontier_serving_plan(manifest=manifest, settings=self.settings)
        if plan is None:
            return super().estimate_resources(manifest)
        estimated_memory_mb = plan.get("planned_memory_mb")
        return RuntimeEstimate(
            estimated_memory_mb=(
                int(estimated_memory_mb)
                if isinstance(estimated_memory_mb, (int, float))
                else manifest.estimated_memory_mb
            ),
            notes=frontier_plan_notes(plan),
        )
