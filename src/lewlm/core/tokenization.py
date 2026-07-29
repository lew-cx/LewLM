"""Model-accurate token counting and deterministic truncation.

Callers that need to fit content into a model's context otherwise estimate token
counts from byte length. That is safe and reproducible but not model-accurate,
so it wastes context on every request. LewLM already owns a tokenizer for each
loaded model, so it can answer the question exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from lewlm.core.contracts import CapabilityName, ModelManifest, RoutingDecision, RuntimeContract, utc_now
from lewlm.core.errors import UnsupportedCapabilityError
from lewlm.core.execution_metadata import ExecutionMetadata, build_routed_execution_metadata
from lewlm.core.provenance import ComponentKind, component
from lewlm.routing.service import ModelRouter
from lewlm.runtime.residency import ModelResidencyManager

TOKENIZER_COMPONENT_VERSION = "1.0.0"


@dataclass(slots=True)
class TokenCountExecution:
    """Result of counting, and optionally truncating, text for one model."""

    request_id: str
    created_at: int
    model_id: str
    token_count: int
    character_count: int
    truncated: bool
    truncated_text: str | None
    truncated_token_count: int | None
    routing: RoutingDecision
    metadata: ExecutionMetadata


class TokenizationService:
    """Count tokens with the tokenizer of the model that will actually run."""

    def __init__(
        self,
        *,
        model_router: ModelRouter,
        model_residency_manager: ModelResidencyManager | None = None,
    ) -> None:
        self.model_router = model_router
        self.model_residency_manager = model_residency_manager

    async def count_tokens(
        self,
        *,
        model_id: str | None,
        text: str,
        max_tokens: int | None = None,
        correlation_id: str | None = None,
    ) -> TokenCountExecution:
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero when provided.")

        manifest, runtime, routing = self.model_router.route_capability(
            capability=CapabilityName.CHAT,
            requested_model_id=model_id,
        )
        request_id = str(uuid4())
        created_at = int(utc_now().timestamp())

        async with self._model_lease(runtime, manifest, request_id=request_id):
            tokens = await self._tokenize(runtime, manifest, text)
            truncated_text: str | None = None
            truncated_token_count: int | None = None
            if max_tokens is not None and len(tokens) > max_tokens:
                # Detokenizing the kept prefix gives a boundary that is exact for
                # this model, rather than a character estimate the caller has to
                # leave headroom around.
                truncated_text = await self._detokenize(runtime, manifest, tokens[:max_tokens])
                truncated_token_count = max_tokens

        return TokenCountExecution(
            request_id=request_id,
            created_at=created_at,
            model_id=manifest.model_id,
            token_count=len(tokens),
            character_count=len(text),
            truncated=truncated_text is not None,
            truncated_text=truncated_text,
            truncated_token_count=truncated_token_count,
            routing=routing,
            metadata=build_routed_execution_metadata(
                request_id=request_id,
                created=created_at,
                requested_model_id=model_id,
                routing=routing,
                correlation_id=correlation_id,
                components=[
                    component(
                        ComponentKind.TOKENIZER,
                        name=f"{runtime.name}_tokenizer",
                        version=TOKENIZER_COMPONENT_VERSION,
                        implementation=manifest.architecture_family,
                    ),
                ],
            ),
        )

    def _model_lease(self, runtime: RuntimeContract, manifest: ModelManifest, *, request_id: str):
        if self.model_residency_manager is None:
            return _NullLease(runtime, manifest)
        return self.model_residency_manager.acquire(
            runtime,
            manifest,
            request_id=request_id,
            capability="tokenize",
        )

    async def _tokenize(self, runtime: RuntimeContract, manifest: ModelManifest, text: str) -> list[int]:
        try:
            return list(runtime.tokenize(text))
        except UnsupportedCapabilityError:
            raise
        except NotImplementedError as exc:
            raise UnsupportedCapabilityError(
                f"Runtime `{runtime.name}` does not expose a tokenizer.",
                details={"runtime": runtime.name, "model_id": manifest.model_id},
            ) from exc

    async def _detokenize(self, runtime: RuntimeContract, manifest: ModelManifest, tokens: list[int]) -> str:
        try:
            return runtime.detokenize(tokens)
        except UnsupportedCapabilityError:
            raise
        except NotImplementedError as exc:
            raise UnsupportedCapabilityError(
                f"Runtime `{runtime.name}` cannot detokenize, so truncation boundaries are unavailable.",
                details={"runtime": runtime.name, "model_id": manifest.model_id},
            ) from exc


class _NullLease:
    """Load the model directly when no residency manager owns lifecycle."""

    def __init__(self, runtime: RuntimeContract, manifest: ModelManifest) -> None:
        self._runtime = runtime
        self._manifest = manifest

    async def __aenter__(self) -> None:
        await self._runtime.load_model(self._manifest)

    async def __aexit__(self, *exc_info: object) -> None:
        return None
