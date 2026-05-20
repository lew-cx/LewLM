"""First-pass llama.cpp runtime adapter."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
from collections.abc import AsyncIterator, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    CapabilityName,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    ModelModality,
    PerformanceFeatureOwnership,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RuntimeAffinity,
    SpeculationMode,
    runtime_performance_feature_report,
)
from lewlm.core.errors import ConfigurationError
from lewlm.runtime.base import ManagedTextRuntime
from lewlm.runtime.introspection import invoke_with_signature, resolve_backend_callable
from lewlm.runtime.prefix_cache import longest_token_prefix
from lewlm.structured_output import (
    GrammarResponseFormat,
    JSONSchemaResponseFormat,
    StructuredOutputRequest,
    StructuredOutputRuntimeStatus,
)

_LLAMA_PREFILL_BATCH_PARAMETERS = ("n_batch", "batch_size", "prompt_batch_size")
_LLAMA_PREFILL_UBATCH_PARAMETERS = ("n_ubatch", "ubatch_size", "prompt_ubatch_size")


class LlamaCppRuntime(ManagedTextRuntime):
    """Adapter for GGUF-backed inference through llama.cpp."""

    name = "llamacpp"
    affinity = RuntimeAffinity.LLAMACPP
    supported_formats = (ModelFormat.GGUF,)
    supported_modalities = (
        ModelModality.TEXT,
        ModelModality.EMBEDDING,
        ModelModality.RERANK,
        ModelModality.MULTIMODAL,
    )
    supported_capabilities = frozenset(
        {
            CapabilityName.CHAT,
            CapabilityName.STREAMING,
            CapabilityName.EMBEDDINGS,
            CapabilityName.RERANK,
        },
    )
    supported_systems = ("Darwin", "Linux", "Windows")
    platform_guidance = "Install the `llamacpp` extra or another compatible llama-cpp-python build on the target host."

    def __init__(self, *, settings: LewLMSettings | None = None) -> None:
        super().__init__()
        self._settings = settings or LewLMSettings()
        self._clients: dict[str, Any] = {}
        self._prompt_lookup_enabled_model_ids: set[str] = set()
        self._prompt_lookup_request_count = 0
        self._prefix_cache_hits = 0
        self._prefix_cache_misses = 0
        self._prefix_cache_saves = 0
        self._saved_prefill_tokens = 0
        self._max_saved_prefill_tokens = 0
        self._prefilled_uncached_tokens = 0
        self._total_prompt_tokens = 0
        self._prefill_request_count = 0
        self._prefill_prompt_tokens = 0
        self._prefill_batch_count = 0
        self._model_performance_controls: dict[str, dict[str, dict[str, Any]]] = {}
        self._model_load_reports: dict[str, dict[str, Any]] = {}

    def supports_manifest(self, manifest: ModelManifest) -> bool:
        if not super().supports_manifest(manifest):
            return False
        if any(modality in manifest.modality for modality in (ModelModality.EMBEDDING, ModelModality.RERANK)):
            return bool(self._semantic_embedding_surface().get("supported"))
        return True

    def supports_capability(self, capability: CapabilityName) -> bool:
        if capability in {CapabilityName.EMBEDDINGS, CapabilityName.RERANK}:
            return self.is_available() and bool(self._semantic_embedding_surface().get("supported"))
        return super().supports_capability(capability)

    def manifest_capability_reason(self, manifest: ModelManifest, capability: CapabilityName) -> str | None:
        if capability not in {CapabilityName.EMBEDDINGS, CapabilityName.RERANK}:
            return super().manifest_capability_reason(manifest, capability)
        if not any(modality in manifest.modality for modality in (ModelModality.EMBEDDING, ModelModality.RERANK)):
            return "manifest unsupported"
        if not self.is_available():
            return self.availability_reason() or f"{self.name} does not support `{capability.value}`."
        semantic_surface = self._semantic_embedding_surface()
        if not semantic_surface.get("supported"):
            return str(semantic_surface.get("reason"))
        return None

    def _check_environment(self) -> tuple[bool, str | None]:
        try:
            import_module("llama_cpp")
        except ImportError:
            return False, "llama-cpp-python is not installed"
        return True, None

    async def _load_model(self, manifest: ModelManifest) -> None:
        llama_cpp = import_module("llama_cpp")
        llama_class = getattr(llama_cpp, "Llama")
        load_options, control_snapshot = self._load_performance_controls(llama_class)
        effective_model_path = _normalized_model_path(manifest.source_path)
        prompt_lookup_surface = self._prompt_lookup_surface(llama_class)
        semantic_surface = self._semantic_embedding_surface(llama_class=llama_class)
        semantic_embedding_requested = any(
            modality in manifest.modality for modality in (ModelModality.EMBEDDING, ModelModality.RERANK)
        )
        kwargs: dict[str, Any] = {
            "model_path": effective_model_path,
            "n_ctx": manifest.context_length or 4096,
            "verbose": False,
            **load_options,
        }
        semantic_load_parameter = semantic_surface.get("load_parameter")
        if (
            semantic_embedding_requested
            and isinstance(semantic_load_parameter, str)
            and semantic_load_parameter
        ):
            kwargs[semantic_load_parameter] = True
        prompt_lookup_helper = self._build_prompt_lookup_helper(prompt_lookup_surface=prompt_lookup_surface)
        if prompt_lookup_helper is not None:
            kwargs["draft_model"] = prompt_lookup_helper
            self._prompt_lookup_enabled_model_ids.add(manifest.model_id)
        client = llama_class(**kwargs)
        prefix_cache_surface = self._prefix_cache_surface(llama_cpp=llama_cpp, client=client)
        prefix_cache = self._build_prefix_cache_wrapper(
            llama_cpp=llama_cpp,
            prefix_cache_supported=bool(prefix_cache_surface.get("supported")),
        )
        if prefix_cache is not None:
            if prefix_cache_surface.get("attachment_method") == "set_cache":
                client.set_cache(prefix_cache)
            elif prefix_cache_surface.get("attachment_method") == "cache_attribute":
                setattr(client, "cache", prefix_cache)
            setattr(client, "_lewlm_prefix_cache", prefix_cache)
        self._clients[manifest.model_id] = client
        self._model_performance_controls[manifest.model_id] = control_snapshot
        self._model_load_reports[manifest.model_id] = self._build_model_load_report(
            manifest=manifest,
            effective_model_path=effective_model_path,
            load_options=load_options,
            control_snapshot=control_snapshot,
            prompt_lookup_surface=prompt_lookup_surface,
            prefix_cache_surface=prefix_cache_surface,
            semantic_surface=semantic_surface,
            semantic_embedding_requested=semantic_embedding_requested,
            semantic_rerank_requested=ModelModality.RERANK in manifest.modality,
        )

    async def _unload_model(self, model_id: str) -> None:
        self._clients.pop(model_id, None)
        self._prompt_lookup_enabled_model_ids.discard(model_id)
        self._model_performance_controls.pop(model_id, None)
        self._model_load_reports.pop(model_id, None)

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        self._validate_speculation_request(request)
        client = self._clients[request.model_id]
        prompt_tokens = self._tokenize_request_messages(request)
        self._set_speculation_execution_metadata(request)
        request.metadata["performance_controls"] = self._request_performance_controls(model_id=request.model_id)
        request.metadata["runtime_load"] = self._request_runtime_load_report(model_id=request.model_id)
        prefix_cache_before = self._prefix_cache_snapshot_for_client(client)
        structured_output_options = self._structured_output_options(request=request, client=client)
        response = client.create_chat_completion(
            messages=[{"role": message.role, "content": message.content} for message in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=False,
            **structured_output_options,
        )
        self._record_prefix_cache_request(
            request=request,
            client=client,
            before=prefix_cache_before,
            prompt_token_count=len(prompt_tokens),
        )
        self._record_prefill_request(model_id=request.model_id, prompt_token_count=len(prompt_tokens))
        message = response["choices"][0]["message"]
        usage = response.get("usage", {})
        if self._prompt_lookup_active_for_request(request):
            usage = {
                **usage,
                "prompt_lookup_requests": 1,
                "prompt_lookup_max_ngram_size": self._settings.prompt_lookup_max_ngram_size,
                "prompt_lookup_num_pred_tokens": self._settings.prompt_lookup_num_pred_tokens,
            }
            self._prompt_lookup_request_count += 1
        return GenerateResponse(
            model_id=request.model_id,
            output_text=str(message.get("content", "")),
            finish_reason=str(response["choices"][0].get("finish_reason", "stop")),
            usage={str(key): int(value) for key, value in usage.items()},
        )

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        self._validate_speculation_request(request)
        client = self._clients[request.model_id]
        prompt_tokens = self._tokenize_request_messages(request)
        self._set_speculation_execution_metadata(request)
        request.metadata["performance_controls"] = self._request_performance_controls(model_id=request.model_id)
        request.metadata["runtime_load"] = self._request_runtime_load_report(model_id=request.model_id)
        prefix_cache_before = self._prefix_cache_snapshot_for_client(client)
        structured_output_options = self._structured_output_options(request=request, client=client)
        chunks = client.create_chat_completion(
            messages=[{"role": message.role, "content": message.content} for message in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
            **structured_output_options,
        )
        for chunk in chunks:
            content = _llama_stream_chunk_text(chunk)
            if content:
                yield content
            await asyncio.sleep(0)
        self._record_prefix_cache_request(
            request=request,
            client=client,
            before=prefix_cache_before,
            prompt_token_count=len(prompt_tokens),
        )
        self._record_prefill_request(model_id=request.model_id, prompt_token_count=len(prompt_tokens))
        if self._prompt_lookup_active_for_request(request):
            self._prompt_lookup_request_count += 1

    def supports_chunked_prefill(self, capability: CapabilityName) -> bool:
        return capability in {CapabilityName.CHAT, CapabilityName.STREAMING} and self._prefill_control_supported()

    def _tokenize(self, text: str) -> list[int]:
        client = self._clients[next(iter(self._clients))]
        return list(client.tokenize(text.encode("utf-8")))

    def _detokenize(self, tokens: Sequence[int]) -> str:
        client = self._clients[next(iter(self._clients))]
        return client.detokenize(list(tokens)).decode("utf-8")

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        client = self._clients[request.model_id]
        request.metadata["runtime_load"] = self._request_runtime_load_report(model_id=request.model_id)
        request.metadata["semantic_runtime"] = {
            "capability": CapabilityName.EMBEDDINGS.value,
            "execution_mode": "packaged_embedding",
            "reason": "llama.cpp embeddings run through the packaged GGUF runtime on this host.",
        }
        result = self._embedding_result(client=client, inputs=request.inputs)
        return _embedding_response_from_result(result, request)

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        self._ensure_available()
        self._ensure_loaded(request.model_id)
        self._touch_model(request.model_id)
        request.metadata["runtime_load"] = self._request_runtime_load_report(model_id=request.model_id)
        request.metadata["semantic_runtime"] = {
            "capability": CapabilityName.RERANK.value,
            "execution_mode": "embedding_similarity_fallback",
            "native_rerank": False,
            "reason": (
                "llama.cpp does not expose a native packaged rerank API here, so LewLM computes cosine-similarity "
                "scores over packaged GGUF embeddings."
            ),
        }
        if not request.documents:
            return RerankResponse(model_id=request.model_id, results=[])
        client = self._clients[request.model_id]
        result = self._embedding_result(client=client, inputs=[request.query, *request.documents])
        vectors = _embedding_vectors_from_result(result)
        if not vectors:
            return RerankResponse(model_id=request.model_id, results=[])
        query_vector = vectors[0]
        document_vectors = vectors[1:]
        results = [
            RerankResult(
                index=index,
                relevance_score=_cosine_similarity(query_vector, document_vector),
                document=request.documents[index],
            )
            for index, document_vector in enumerate(document_vectors[: len(request.documents)])
        ]
        results.sort(key=lambda item: (-item.relevance_score, item.index))
        if request.top_n is not None:
            results = results[: request.top_n]
        return RerankResponse(model_id=request.model_id, results=results)

    def performance_feature_snapshot(self) -> dict[str, Any]:
        supported = self._supports_prompt_lookup_speculation()
        structured_output_capability = self._runtime_structured_output_capability()
        prefix_cache_surface = self._prefix_cache_surface()
        prefix_cache_supported = bool(prefix_cache_surface.get("supported"))
        prefix_cache_metrics = self._aggregate_prefix_cache_metrics()
        prefill_probe = self._prefill_control_payload(self._llama_parameter_names())
        prefill_controls = self._aggregate_control_entries("prefill_optimization") or [prefill_probe]
        paged_controls = self._aggregate_control_entries("paged_kv_cache")
        quantization_controls = self._aggregate_control_entries("kv_cache_quantization")
        prefill_supported = any(bool(entry.get("supported")) for entry in prefill_controls)
        paged_supported = any(bool(entry.get("supported")) for entry in paged_controls)
        quantization_supported = any(bool(entry.get("supported")) for entry in quantization_controls)
        return {
            "continuous_batching": runtime_performance_feature_report(
                ownership=PerformanceFeatureOwnership.UNSUPPORTED,
                active=False,
                reason="The current llama.cpp adapter does not expose a native batched chat or streaming surface to LewLM.",
                notes=[
                    "LewLM only enables backend-native continuous batching when the selected runtime exposes explicit batched chat or streaming entrypoints.",
                ],
            ),
            "prefix_cache": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.BACKEND_NATIVE
                    if prefix_cache_supported
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=prefix_cache_supported and (
                    prefix_cache_metrics["cache_entries"] > 0 or prefix_cache_metrics["cache_hits"] > 0
                ),
                reason=(
                    "llama.cpp exposes an in-memory longest-prefix KV cache through `LlamaRAMCache`."
                    if prefix_cache_supported
                    else str(prefix_cache_surface.get("reason"))
                ),
                notes=(
                    ["LewLM attaches a runtime-local RAM cache to each loaded GGUF client and records longest-prefix reuse."]
                    if prefix_cache_supported
                    else []
                ),
                metrics=prefix_cache_metrics,
            ),
            "paged_kv_cache": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.BACKEND_NATIVE
                    if paged_supported
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=False,
                reason=(
                    "Installed llama.cpp bindings expose an operator-visible paged KV-cache control."
                    if paged_supported
                    else "LewLM does not currently see a stable paged KV-cache page-size control in the installed llama.cpp bindings."
                ),
                metrics={
                    "requested_page_size_tokens": self._settings.kv_cache_page_size,
                    "requested_max_pages": self._settings.kv_cache_max_pages or 0,
                },
                notes=self._control_notes("paged_kv_cache"),
            ),
            "kv_cache_quantization": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.BACKEND_NATIVE
                    if quantization_supported
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=quantization_supported and self._prefill_request_count > 0,
                reason=(
                    "Installed llama.cpp bindings expose a stable KV-cache quantization surface that LewLM can apply."
                    if quantization_supported
                    else "LewLM does not currently apply KV-cache quantization through llama.cpp because the installed bindings do not expose a stable LewLM-supported contract."
                ),
                metrics={
                    "requested_quantization_bits": self._settings.kv_cache_quantization_bits or 0,
                },
                notes=self._control_notes("kv_cache_quantization"),
            ),
            "prefill_optimization": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.BACKEND_NATIVE
                    if prefill_supported
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=prefill_supported and self._prefill_request_count > 0,
                reason=(
                    "LewLM maps llama.cpp prefill batching onto constructor batch-size controls when the installed bindings expose them."
                    if prefill_supported
                    else "Installed llama.cpp bindings do not expose a LewLM-supported prefill batch-size control."
                ),
                metrics={
                    "requested_prefill_token_batch_size": self._settings.prefill_token_batch_size,
                    "optimized_requests": self._prefill_request_count,
                    "optimized_prompt_tokens": self._prefill_prompt_tokens,
                    "prefill_batches_planned": self._prefill_batch_count,
                },
                notes=self._control_notes("prefill_optimization", fallback_payloads=(prefill_probe,)),
            ),
            "prompt_lookup_speculation": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.BACKEND_NATIVE
                    if supported
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=self._prompt_lookup_request_count > 0,
                modes=["prompt_lookup"] if supported else [],
                reason=(
                    "llama.cpp prompt-lookup speculation is available through `LlamaPromptLookupDecoding`."
                    if supported
                    else str(self._prompt_lookup_surface().get("reason"))
                ),
                metrics={
                    "request_count": self._prompt_lookup_request_count,
                    "configured_max_ngram_size": self._settings.prompt_lookup_max_ngram_size,
                    "configured_num_pred_tokens": self._settings.prompt_lookup_num_pred_tokens,
                },
                notes=(
                    []
                    if self._settings.prompt_lookup_speculation_enabled
                    else [
                        "Enable `LEWLM_PROMPT_LOOKUP_SPECULATION_ENABLED=true` to activate prompt-lookup speculation."
                    ]
                ),
            ),
            "constrained_decoding": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.BACKEND_NATIVE
                    if structured_output_capability["grammar"]
                    else PerformanceFeatureOwnership.PARTIAL
                ),
                active=False,
                modes=(
                    [
                        *(
                            ["json_schema"]
                            if structured_output_capability["json_schema"]
                            else []
                        ),
                        *(
                            ["grammar"]
                            if structured_output_capability["grammar"]
                            else []
                        ),
                    ]
                    if structured_output_capability["grammar"]
                    else ["prompt_guided"]
                ),
                reason=(
                    "llama.cpp exposes decode-time grammar enforcement and optional JSON-schema compilation through "
                    "the installed bindings."
                    if structured_output_capability["grammar"]
                    else str(structured_output_capability.get("grammar_reason"))
                ),
                metrics={
                    "decoder_enforced": structured_output_capability["grammar"],
                    "json_schema_supported": structured_output_capability["json_schema"],
                    "fallback_used": not structured_output_capability["grammar"],
                },
                notes=_structured_output_capability_notes(structured_output_capability),
            ),
        }

    def _supports_prefix_cache(self) -> bool:
        return bool(self._prefix_cache_surface().get("supported"))

    def _runtime_structured_output_capability(self) -> dict[str, bool | str | None]:
        if self._clients:
            return self._structured_output_capability(next(iter(self._clients.values())))
        if not self.is_available():
            return {
                "grammar": False,
                "json_schema": False,
                "grammar_reason": "llama-cpp-python is not installed or unavailable on this host.",
                "json_schema_reason": "llama-cpp-python is not installed or unavailable on this host.",
            }
        try:
            llama_cpp = import_module("llama_cpp")
        except ImportError:
            return {
                "grammar": False,
                "json_schema": False,
                "grammar_reason": "llama-cpp-python is not installed or unavailable on this host.",
                "json_schema_reason": "llama-cpp-python is not installed or unavailable on this host.",
            }
        llama_class = getattr(llama_cpp, "Llama", None)
        if llama_class is None:
            return {
                "grammar": False,
                "json_schema": False,
                "grammar_reason": "Installed llama.cpp bindings do not expose a `Llama` client.",
                "json_schema_reason": "Installed llama.cpp bindings do not expose a `Llama` client.",
            }
        return self._structured_output_capability(llama_class)

    def _structured_output_options(self, *, request: GenerateRequest, client: Any) -> dict[str, Any]:
        contract = request.structured_output
        if contract is None or contract.type == "text":
            return {}
        capability = self._structured_output_capability(client)
        status = self._structured_output_status(contract=contract, capability=capability)
        request.metadata["structured_output_runtime"] = status.model_dump(mode="json")
        if not status.decoder_enforced:
            return {}
        if isinstance(contract, JSONSchemaResponseFormat):
            grammar, fallback_reason = self._json_schema_grammar(contract)
            if grammar is None:
                request.metadata["structured_output_runtime"] = StructuredOutputRuntimeStatus(
                    runtime=self.name,
                    mode="json_schema",
                    enforcement="prompt_guided",
                    decoder_enforced=False,
                    fallback_used=True,
                    fallback_reason=fallback_reason,
                ).model_dump(mode="json")
                return {}
            return {"grammar": grammar}
        if capability["grammar"]:
            grammar, fallback_reason = self._grammar_from_string(contract.grammar)
            if grammar is None:
                request.metadata["structured_output_runtime"] = StructuredOutputRuntimeStatus(
                    runtime=self.name,
                    mode="grammar",
                    enforcement="prompt_guided",
                    decoder_enforced=False,
                    fallback_used=True,
                    fallback_reason=fallback_reason,
                ).model_dump(mode="json")
                return {}
            request.metadata["structured_output_runtime"] = StructuredOutputRuntimeStatus(
                runtime=self.name,
                mode="grammar",
                enforcement="decode_time",
                decoder_enforced=True,
                fallback_used=False,
            ).model_dump(mode="json")
            return {"grammar": grammar}
        return {}

    def structured_output_runtime_status(
        self,
        contract: StructuredOutputRequest | None,
    ) -> StructuredOutputRuntimeStatus | None:
        if contract is None or contract.type == "text":
            return None
        return self._structured_output_status(
            contract=contract,
            capability=self._runtime_structured_output_capability(),
        )

    def _structured_output_status(
        self,
        *,
        contract: JSONSchemaResponseFormat | GrammarResponseFormat,
        capability: dict[str, bool | str | None],
    ) -> StructuredOutputRuntimeStatus:
        if isinstance(contract, JSONSchemaResponseFormat):
            if capability["json_schema"]:
                return StructuredOutputRuntimeStatus(
                    runtime=self.name,
                    mode="json_schema",
                    enforcement="decode_time",
                    decoder_enforced=True,
                    fallback_used=False,
                )
            return StructuredOutputRuntimeStatus(
                runtime=self.name,
                mode="json_schema",
                enforcement="prompt_guided",
                decoder_enforced=False,
                fallback_used=True,
                fallback_reason=str(capability.get("json_schema_reason")),
            )
        if contract.syntax.casefold() not in {"ebnf", "gbnf"}:
            return StructuredOutputRuntimeStatus(
                runtime=self.name,
                mode="grammar",
                enforcement="prompt_guided",
                decoder_enforced=False,
                fallback_used=True,
                fallback_reason=(
                    f"llama.cpp constrained decoding expects `ebnf`/`gbnf`-style grammars; received `{contract.syntax}`."
                ),
            )
        if capability["grammar"]:
            return StructuredOutputRuntimeStatus(
                runtime=self.name,
                mode="grammar",
                enforcement="decode_time",
                decoder_enforced=True,
                fallback_used=False,
            )
        return StructuredOutputRuntimeStatus(
            runtime=self.name,
            mode="grammar",
            enforcement="prompt_guided",
            decoder_enforced=False,
            fallback_used=True,
            fallback_reason=str(capability.get("grammar_reason")),
        )

    @staticmethod
    def _structured_output_capability(client: Any) -> dict[str, bool | str | None]:
        create_chat_completion = getattr(client, "create_chat_completion", None)
        supports_grammar_parameter = _callable_accepts_parameter(create_chat_completion, "grammar")
        if not supports_grammar_parameter:
            reason = "Installed llama.cpp chat completions do not expose a `grammar` parameter for decode-time constrained decoding."
            return {"grammar": False, "json_schema": False, "grammar_reason": reason, "json_schema_reason": reason}
        try:
            llama_cpp = import_module("llama_cpp")
        except ImportError:
            reason = "llama-cpp-python is not installed or unavailable on this host."
            return {"grammar": False, "json_schema": False, "grammar_reason": reason, "json_schema_reason": reason}
        grammar_class = getattr(llama_cpp, "LlamaGrammar", None)
        if grammar_class is None:
            reason = "Installed llama.cpp bindings do not expose `LlamaGrammar` for decode-time constrained decoding."
            return {"grammar": False, "json_schema": False, "grammar_reason": reason, "json_schema_reason": reason}
        supports_grammar = callable(getattr(grammar_class, "from_string", None))
        supports_json_schema = callable(getattr(grammar_class, "from_json_schema", None))
        return {
            "grammar": supports_grammar,
            "json_schema": supports_grammar and supports_json_schema,
            "grammar_reason": (
                None
                if supports_grammar
                else "Installed llama.cpp bindings do not expose `LlamaGrammar.from_string` for decode-time constrained decoding."
            ),
            "json_schema_reason": (
                None
                if supports_grammar and supports_json_schema
                else "Installed llama.cpp bindings do not expose `LlamaGrammar.from_json_schema` for decode-time JSON-schema compilation."
                if supports_grammar
                else "Installed llama.cpp bindings do not expose grammar-based decode-time constrained decoding."
            ),
        }

    @staticmethod
    def _grammar_class() -> Any | None:
        try:
            llama_cpp = import_module("llama_cpp")
        except ImportError:
            return None
        return getattr(llama_cpp, "LlamaGrammar", None)

    @classmethod
    def _json_schema_grammar(cls, contract: JSONSchemaResponseFormat) -> tuple[Any | None, str]:
        grammar_class = cls._grammar_class()
        factory = getattr(grammar_class, "from_json_schema", None)
        if not callable(factory):
            return (
                None,
                "Installed llama.cpp bindings do not expose JSON-schema grammar compilation for decode-time constrained decoding.",
            )
        try:
            return factory(json.dumps(contract.schema_payload, sort_keys=True), verbose=False), ""
        except (TypeError, ValueError) as exc:
            return (
                None,
                "Installed llama.cpp bindings rejected the requested JSON schema for decode-time constrained decoding: "
                f"{exc}",
            )

    @classmethod
    def _grammar_from_string(cls, grammar: str) -> tuple[Any | None, str]:
        grammar_class = cls._grammar_class()
        factory = getattr(grammar_class, "from_string", None)
        if not callable(factory):
            return (
                None,
                "Installed llama.cpp bindings do not expose grammar-based decode-time constrained decoding.",
            )
        try:
            return factory(grammar, verbose=False), ""
        except (TypeError, ValueError) as exc:
            return (
                None,
                "Installed llama.cpp bindings rejected the requested grammar for decode-time constrained decoding: "
                f"{exc}",
            )

    def _build_prefix_cache_wrapper(
        self,
        *,
        llama_cpp: Any,
        prefix_cache_supported: bool,
    ) -> "_InstrumentedLlamaRamCache | None":
        if not prefix_cache_supported:
            return None
        cache_class = getattr(llama_cpp, "LlamaRAMCache", None)
        if not callable(cache_class):
            return None
        return _InstrumentedLlamaRamCache(cache_class=cache_class)

    def _load_performance_controls(self, llama_class: type[Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        parameter_names = _callable_parameter_names(llama_class)
        options, prefill_payload = self._prefill_control_configuration(parameter_names)
        return options, {
            "prefill_optimization": prefill_payload,
            "paged_kv_cache": _performance_control_payload(
                requested=True,
                supported=False,
                effective="rejected",
                reason="LewLM does not currently see a stable paged KV-cache page-size contract in the installed llama.cpp bindings.",
                rejected_parameters=("kv_cache_page_size", "kv_cache_max_pages"),
                requested_page_size_tokens=self._settings.kv_cache_page_size,
                requested_max_pages=self._settings.kv_cache_max_pages,
            ),
            "kv_cache_quantization": _performance_control_payload(
                requested=self._settings.kv_cache_quantization_bits is not None,
                supported=False,
                effective="rejected" if self._settings.kv_cache_quantization_bits is not None else "disabled",
                reason=(
                    "LewLM does not currently apply KV-cache quantization through llama.cpp because the installed bindings do not expose a stable LewLM-supported type contract."
                ),
                rejected_parameters=(("kv_cache_quantization_bits",) if self._settings.kv_cache_quantization_bits is not None else ()),
                requested_quantization_bits=self._settings.kv_cache_quantization_bits,
            ),
        }

    def _prefill_control_supported(self) -> bool:
        return bool(self._prefill_control_payload(self._llama_parameter_names()).get("supported"))

    def _llama_parameter_names(self) -> set[str]:
        if self._clients:
            return _callable_parameter_names(type(next(iter(self._clients.values()))))
        if not self.is_available():
            return set()
        llama_class = getattr(import_module("llama_cpp"), "Llama", None)
        if not callable(llama_class):
            return set()
        return _callable_parameter_names(llama_class)

    def _prefill_control_configuration(
        self,
        parameter_names: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        options: dict[str, Any] = {}
        applied_prefill_parameters: list[str] = []
        batch_parameter = _first_matching_parameter(parameter_names, _LLAMA_PREFILL_BATCH_PARAMETERS)
        if batch_parameter is not None:
            options[batch_parameter] = self._settings.prefill_token_batch_size
            applied_prefill_parameters.append(batch_parameter)
        ubatch_parameter = _first_matching_parameter(parameter_names, _LLAMA_PREFILL_UBATCH_PARAMETERS)
        if ubatch_parameter is not None:
            options[ubatch_parameter] = self._settings.prefill_token_batch_size
            applied_prefill_parameters.append(ubatch_parameter)
        return options, _performance_control_payload(
            requested=True,
            supported=bool(applied_prefill_parameters),
            effective="enabled" if applied_prefill_parameters else "unsupported",
            reason=(
                "LewLM can apply llama.cpp prefill batch-size controls during model load."
                if applied_prefill_parameters
                else "Installed llama.cpp bindings do not expose a LewLM-supported prefill batch-size parameter."
            ),
            applied_parameters=tuple(applied_prefill_parameters),
            rejected_parameters=(() if applied_prefill_parameters else ("prefill_token_batch_size",)),
            requested_prefill_token_batch_size=self._settings.prefill_token_batch_size,
            effective_prefill_token_batch_size=(
                self._settings.prefill_token_batch_size if applied_prefill_parameters else None
            ),
        )

    def _prefill_control_payload(self, parameter_names: set[str]) -> dict[str, Any]:
        _, payload = self._prefill_control_configuration(parameter_names)
        return payload

    def _request_performance_controls(self, *, model_id: str) -> dict[str, dict[str, dict[str, Any]]]:
        controls = self._model_performance_controls.get(model_id, {})
        return {"load": copy.deepcopy(controls)} if controls else {}

    def _request_runtime_load_report(self, *, model_id: str) -> dict[str, Any]:
        report = self._model_load_reports.get(model_id)
        return copy.deepcopy(report) if isinstance(report, dict) else {}

    def _tokenize_request_messages(self, request: GenerateRequest) -> list[int]:
        prompt = "\n".join(f"{message.role}: {message.content}" for message in request.messages)
        return self._tokenize(prompt)

    def _record_prefill_request(self, *, model_id: str, prompt_token_count: int) -> None:
        controls = self._model_performance_controls.get(model_id, {})
        prefill_control = controls.get("prefill_optimization", {})
        if not isinstance(prefill_control, dict) or prefill_control.get("effective") != "enabled":
            return
        normalized_prompt_tokens = max(prompt_token_count, 0)
        self._prefill_request_count += 1
        self._prefill_prompt_tokens += normalized_prompt_tokens
        self._prefill_batch_count += max(
            1,
            math.ceil(normalized_prompt_tokens / self._settings.prefill_token_batch_size),
        )

    def _aggregate_control_entries(self, control_name: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for control_snapshot in self._model_performance_controls.values():
            payload = control_snapshot.get(control_name)
            if isinstance(payload, dict):
                entries.append(payload)
        return entries

    def _control_notes(
        self,
        control_name: str,
        *,
        fallback_payloads: tuple[dict[str, Any], ...] = (),
    ) -> list[str]:
        notes: list[str] = []
        seen: set[str] = set()
        payloads = self._aggregate_control_entries(control_name)
        if not payloads and fallback_payloads:
            payloads = [payload for payload in fallback_payloads if isinstance(payload, dict)]
        for payload in payloads:
            reason = payload.get("reason")
            if isinstance(reason, str) and reason and reason not in seen:
                seen.add(reason)
                notes.append(reason)
            rejected_parameters = payload.get("rejected_parameters")
            if not isinstance(rejected_parameters, list) or not rejected_parameters:
                continue
            rejected_note = "Rejected requested setting(s): " + ", ".join(str(item) for item in rejected_parameters)
            if rejected_note not in seen:
                seen.add(rejected_note)
                notes.append(rejected_note)
        return notes

    def _prefix_cache_snapshot_for_client(self, client: Any) -> dict[str, int]:
        cache = getattr(client, "_lewlm_prefix_cache", None)
        if cache is None:
            return {"cache_hits": 0, "cache_misses": 0, "cache_saves": 0, "saved_prefill_tokens": 0}
        return cache.snapshot()

    def _record_prefix_cache_request(
        self,
        *,
        request: GenerateRequest,
        client: Any,
        before: dict[str, int],
        prompt_token_count: int,
    ) -> None:
        cache = getattr(client, "_lewlm_prefix_cache", None)
        normalized_prompt_tokens = max(prompt_token_count, 0)
        if cache is None:
            request.metadata["prefix_cache"] = {
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_saves": 0,
                "saved_prefill_tokens": 0,
                "max_saved_prefill_tokens": 0,
                "prefilled_uncached_tokens": normalized_prompt_tokens,
                "total_prompt_tokens": normalized_prompt_tokens,
                "effective_prefill_tokens": normalized_prompt_tokens,
            }
            self._prefilled_uncached_tokens += normalized_prompt_tokens
            self._total_prompt_tokens += normalized_prompt_tokens
            return
        after = cache.snapshot()
        saved_prefill_tokens = max(after["saved_prefill_tokens"] - before["saved_prefill_tokens"], 0)
        effective_prefill_tokens = max(normalized_prompt_tokens - saved_prefill_tokens, 0)
        metrics = {
            "cache_hits": max(after["cache_hits"] - before["cache_hits"], 0),
            "cache_misses": max(after["cache_misses"] - before["cache_misses"], 0),
            "cache_saves": max(after["cache_saves"] - before["cache_saves"], 0),
            "saved_prefill_tokens": saved_prefill_tokens,
            "max_saved_prefill_tokens": max(after["max_saved_prefill_tokens"], before["max_saved_prefill_tokens"]),
            "prefilled_uncached_tokens": effective_prefill_tokens,
            "total_prompt_tokens": normalized_prompt_tokens,
            "effective_prefill_tokens": effective_prefill_tokens,
        }
        self._prefix_cache_hits += metrics["cache_hits"]
        self._prefix_cache_misses += metrics["cache_misses"]
        self._prefix_cache_saves += metrics["cache_saves"]
        self._saved_prefill_tokens += metrics["saved_prefill_tokens"]
        self._max_saved_prefill_tokens = max(self._max_saved_prefill_tokens, metrics["max_saved_prefill_tokens"])
        self._prefilled_uncached_tokens += metrics["prefilled_uncached_tokens"]
        self._total_prompt_tokens += metrics["total_prompt_tokens"]
        request.metadata["prefix_cache"] = metrics

    def _aggregate_prefix_cache_metrics(self) -> dict[str, int]:
        entry_count = 0
        cache_size_bytes = 0
        for client in self._clients.values():
            cache = getattr(client, "_lewlm_prefix_cache", None)
            if cache is None:
                continue
            snapshot = cache.snapshot()
            entry_count += snapshot["cache_entries"]
            cache_size_bytes += snapshot["cache_size_bytes"]
        return {
            "cache_entries": entry_count,
            "cache_size_bytes": cache_size_bytes,
            "cache_hits": self._prefix_cache_hits,
            "cache_misses": self._prefix_cache_misses,
            "cache_saves": self._prefix_cache_saves,
            "saved_prefill_tokens": self._saved_prefill_tokens,
            "max_saved_prefill_tokens": self._max_saved_prefill_tokens,
            "prefilled_uncached_tokens": self._prefilled_uncached_tokens,
            "total_prompt_tokens": self._total_prompt_tokens,
        }

    def _supports_prompt_lookup_speculation(self) -> bool:
        return bool(self._prompt_lookup_surface().get("supported"))

    def _build_prompt_lookup_helper(self, *, prompt_lookup_surface: dict[str, Any] | None = None) -> Any | None:
        if not self._settings.prompt_lookup_speculation_enabled:
            return None
        surface = prompt_lookup_surface or self._prompt_lookup_surface()
        if not surface.get("supported"):
            return None
        prompt_lookup_class = self._prompt_lookup_class()
        if prompt_lookup_class is None:
            return None
        return prompt_lookup_class(
            max_ngram_size=self._settings.prompt_lookup_max_ngram_size,
            num_pred_tokens=self._settings.prompt_lookup_num_pred_tokens,
        )

    def _prompt_lookup_class(self):
        try:
            speculative_module = import_module("llama_cpp.llama_speculative")
        except ImportError:
            return None
        candidate = getattr(speculative_module, "LlamaPromptLookupDecoding", None)
        return candidate if callable(candidate) else None

    def _prompt_lookup_surface(self, llama_class: type[Any] | None = None) -> dict[str, Any]:
        if llama_class is None:
            if not self.is_available():
                return {
                    "supported": False,
                    "reason": "llama-cpp-python is not installed or unavailable on this host.",
                }
            llama_class = getattr(import_module("llama_cpp"), "Llama", None)
        if not callable(llama_class):
            return {
                "supported": False,
                "reason": "Installed llama.cpp bindings do not expose a `Llama` client.",
            }
        if self._prompt_lookup_class() is None:
            return {
                "supported": False,
                "reason": "Installed llama-cpp-python does not expose `LlamaPromptLookupDecoding`.",
            }
        if not _callable_accepts_parameter(llama_class, "draft_model"):
            return {
                "supported": False,
                "reason": "Installed llama.cpp bindings do not accept a `draft_model` load option for prompt-lookup speculation.",
            }
        return {
            "supported": True,
            "reason": "llama.cpp prompt-lookup speculation is available through `LlamaPromptLookupDecoding` and the `draft_model` load option.",
        }

    def _prefix_cache_surface(
        self,
        *,
        llama_cpp: Any | None = None,
        client: Any | None = None,
    ) -> dict[str, Any]:
        if llama_cpp is None:
            if not self.is_available():
                return {
                    "supported": False,
                    "attachment_method": None,
                    "reason": "llama-cpp-python is not installed or unavailable on this host.",
                }
            llama_cpp = import_module("llama_cpp")
        cache_class = getattr(llama_cpp, "LlamaRAMCache", None)
        if not callable(cache_class):
            return {
                "supported": False,
                "attachment_method": None,
                "reason": "Installed llama-cpp-python does not expose `LlamaRAMCache`.",
            }
        candidate = client if client is not None else getattr(llama_cpp, "Llama", None)
        attachment_method = _prefix_cache_attachment_method(candidate)
        if attachment_method is None:
            return {
                "supported": False,
                "attachment_method": None,
                "reason": (
                    "Installed llama.cpp bindings expose `LlamaRAMCache`, but the active client does not expose "
                    "`set_cache()` or a `cache` attribute for prefix-cache attachment."
                ),
            }
        return {
            "supported": True,
            "attachment_method": attachment_method,
            "reason": "llama.cpp exposes a runtime-local prefix cache surface that LewLM can attach to the active client.",
        }

    def _build_model_load_report(
        self,
        *,
        manifest: ModelManifest,
        effective_model_path: str,
        load_options: dict[str, Any],
        control_snapshot: dict[str, dict[str, Any]],
        prompt_lookup_surface: dict[str, Any],
        prefix_cache_surface: dict[str, Any],
        semantic_surface: dict[str, Any],
        semantic_embedding_requested: bool,
        semantic_rerank_requested: bool,
    ) -> dict[str, Any]:
        requested_model_path = str(manifest.source_path)
        return {
            "requested_model_path": requested_model_path,
            "effective_model_path": effective_model_path,
            "path_normalized": requested_model_path != effective_model_path,
            "load_option_names": sorted(
                option_name
                for option_name in {"model_path", "n_ctx", "verbose", *load_options}
            ),
            "performance_controls": {
                control_name: str(payload.get("effective"))
                for control_name, payload in control_snapshot.items()
                if isinstance(payload, dict) and payload.get("effective") is not None
            },
            "prompt_lookup": {
                "supported": bool(prompt_lookup_surface.get("supported")),
                "enabled": bool(prompt_lookup_surface.get("supported"))
                and self._settings.prompt_lookup_speculation_enabled,
                "reason": prompt_lookup_surface.get("reason"),
            },
            "prefix_cache": {
                "supported": bool(prefix_cache_surface.get("supported")),
                "attachment_method": prefix_cache_surface.get("attachment_method"),
                "reason": prefix_cache_surface.get("reason"),
            },
            "semantic_text": {
                "requested": semantic_embedding_requested or semantic_rerank_requested,
                "embedding_mode_enabled": semantic_embedding_requested and bool(semantic_surface.get("load_parameter")),
                "embedding_callable": semantic_surface.get("callable_name"),
                "embedding_load_parameter": semantic_surface.get("load_parameter"),
                "rerank_mode": "embedding_similarity_fallback" if semantic_rerank_requested else None,
                "reason": semantic_surface.get("reason"),
            },
        }

    def _embedding_result(self, *, client: Any, inputs: list[str]) -> Any:
        embed = self._embedding_callable(client)
        return invoke_with_signature(
            embed,
            {
                "input": inputs,
                "inputs": inputs,
                "texts": inputs,
                "sentences": inputs,
                "documents": inputs,
            },
            capability=CapabilityName.EMBEDDINGS.value,
        )

    def _embedding_callable(self, client: Any) -> Any:
        return resolve_backend_callable(
            client,
            ("create_embedding", "create_embeddings", "embed"),
        )

    def _semantic_embedding_surface(
        self,
        llama_class: type[Any] | None = None,
        *,
        client: Any | None = None,
    ) -> dict[str, Any]:
        if client is None and llama_class is None:
            if not self.is_available():
                return {
                    "supported": False,
                    "callable_name": None,
                    "load_parameter": None,
                    "reason": "llama-cpp-python is not installed or unavailable on this host.",
                }
            llama_class = getattr(import_module("llama_cpp"), "Llama", None)
        candidate = client if client is not None else llama_class
        if not callable(candidate):
            return {
                "supported": False,
                "callable_name": None,
                "load_parameter": None,
                "reason": "Installed llama.cpp bindings do not expose a `Llama` client.",
            }
        callable_name = _first_callable_name(candidate, ("create_embedding", "create_embeddings", "embed"))
        if callable_name is None:
            return {
                "supported": False,
                "callable_name": None,
                "load_parameter": None,
                "reason": (
                    "Installed llama.cpp bindings do not expose a packaged embedding callable such as "
                    "`create_embedding`."
                ),
            }
        load_parameter = None
        if callable(llama_class) and _callable_accepts_parameter(llama_class, "embedding"):
            load_parameter = "embedding"
        return {
            "supported": True,
            "callable_name": callable_name,
            "load_parameter": load_parameter,
            "reason": (
                "llama.cpp exposes packaged embedding calls, and LewLM uses those embeddings directly plus an "
                "embedding-similarity rerank fallback when needed."
            ),
        }

    def _prompt_lookup_active_for_request(self, request: GenerateRequest) -> bool:
        return (
            request.speculation is not None
            and request.speculation.mode == SpeculationMode.PROMPT_LOOKUP
            and request.model_id in self._prompt_lookup_enabled_model_ids
        )

    def _set_speculation_execution_metadata(self, request: GenerateRequest) -> None:
        speculation = request.speculation
        if speculation is None:
            return
        request.metadata["speculation_execution_path"] = "backend_passthrough"
        request.metadata["speculation_fallback_count"] = 0
        request.metadata["speculation_runtime"] = {
            "ownership": "backend_passthrough",
            "execution_path": "backend_passthrough",
            "controller": "prompt_lookup_backend",
            "drafted_tokens": 0,
            "accepted_tokens": 0,
            "verified_tokens": 0,
            "rejected_tokens": 0,
            "rollback_tokens": 0,
            "fallback_count": 0,
        }

    def _validate_speculation_request(self, request: GenerateRequest) -> None:
        speculation = request.speculation
        if speculation is None:
            return
        if speculation.mode != SpeculationMode.PROMPT_LOOKUP:
            raise ConfigurationError(
                f"llama.cpp runtime does not support `{speculation.mode.value}` speculation.",
                details={"supported_modes": ["prompt_lookup"]},
            )
        if not self._supports_prompt_lookup_speculation():
            raise ConfigurationError(
                "Installed llama.cpp bindings do not expose prompt-lookup speculation support.",
            )
        if request.model_id not in self._prompt_lookup_enabled_model_ids:
            raise ConfigurationError(
                "Prompt-lookup speculation was requested for a llama.cpp model that was loaded without the prompt-lookup helper.",
            )


class _InstrumentedLlamaRamCache:
    def __init__(self, *, cache_class: type[Any]) -> None:
        self._cache = cache_class()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_saves = 0
        self._saved_prefill_tokens = 0
        self._max_saved_prefill_tokens = 0

    def __contains__(self, key: Sequence[int]) -> bool:
        normalized_key = tuple(int(token) for token in key)
        return self._find_longest_prefix_key(normalized_key) is not None

    def __getitem__(self, key: Sequence[int]) -> Any:
        normalized_key = tuple(int(token) for token in key)
        matched_key = self._find_longest_prefix_key(normalized_key)
        if matched_key is None:
            self._cache_misses += 1
            raise KeyError("Key not found")
        prefix_length = longest_token_prefix(matched_key, normalized_key)
        self._cache_hits += 1
        self._saved_prefill_tokens += prefix_length
        self._max_saved_prefill_tokens = max(self._max_saved_prefill_tokens, prefix_length)
        return self._cache[matched_key]

    def __setitem__(self, key: Sequence[int], value: Any) -> None:
        self._cache[key] = value
        self._cache_saves += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "cache_entries": len(getattr(self._cache, "cache_state", {})),
            "cache_size_bytes": int(getattr(self._cache, "cache_size", 0)),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_saves": self._cache_saves,
            "saved_prefill_tokens": self._saved_prefill_tokens,
            "max_saved_prefill_tokens": self._max_saved_prefill_tokens,
        }

    def _find_longest_prefix_key(self, key: tuple[int, ...]) -> tuple[int, ...] | None:
        finder = getattr(self._cache, "_find_longest_prefix_key", None)
        if callable(finder):
            return finder(key)
        cache_state = getattr(self._cache, "cache_state", {})
        best_key: tuple[int, ...] | None = None
        best_prefix_length = 0
        for candidate in cache_state:
            prefix_length = longest_token_prefix(candidate, key)
            if prefix_length > best_prefix_length:
                best_key = candidate
                best_prefix_length = prefix_length
        return best_key


def _callable_parameter_names(callable_obj: Any) -> set[str]:
    try:
        return set(inspect.signature(callable_obj).parameters)
    except (TypeError, ValueError):
        return set()


def _callable_accepts_parameter(callable_obj: Any, parameter_name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters.values()
    return any(
        parameter.name == parameter_name or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _first_callable_name(candidate: Any, names: tuple[str, ...]) -> str | None:
    for name in names:
        if callable(getattr(candidate, name, None)):
            return name
    return None


def _first_matching_parameter(parameter_names: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in parameter_names:
            return candidate
    return None


def _performance_control_payload(
    *,
    requested: bool,
    supported: bool,
    effective: str,
    reason: str,
    applied_parameters: tuple[str, ...] = (),
    rejected_parameters: tuple[str, ...] = (),
    **details: int | float | str | bool | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requested": requested,
        "supported": supported,
        "effective": effective,
        "reason": reason,
        "applied_parameters": list(applied_parameters),
        "rejected_parameters": list(rejected_parameters),
    }
    for key, value in details.items():
        if value is not None:
            payload[key] = value
    return payload


def _normalized_model_path(source_path: str | Path) -> str:
    return str(Path(source_path).expanduser())


def _llama_stream_chunk_text(chunk: Any) -> str:
    if not isinstance(chunk, dict):
        return ""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _prefix_cache_attachment_method(candidate: Any) -> str | None:
    if candidate is None:
        return None
    setter = getattr(candidate, "set_cache", None)
    if callable(setter):
        return "set_cache"
    if hasattr(candidate, "cache") or hasattr(type(candidate), "cache"):
        return "cache_attribute"
    return None


def _structured_output_capability_notes(capability: dict[str, bool | str | None]) -> list[str]:
    notes: list[str] = []
    for key in ("grammar_reason", "json_schema_reason"):
        reason = capability.get(key)
        if isinstance(reason, str) and reason and reason not in notes:
            notes.append(reason)
    return notes


def _embedding_response_from_result(result: Any, request: EmbeddingRequest) -> EmbeddingResponse:
    if isinstance(result, EmbeddingResponse):
        return result
    usage: dict[str, int] = {}
    payload = result
    if isinstance(result, dict):
        usage = _normalize_usage(result.get("usage"))
        payload = result.get("data", result.get("embeddings", result.get("vectors", result.get("results", result))))
    vectors = _normalize_embedding_vectors(payload)
    prompt_tokens = usage.get("prompt_tokens", sum(max(1, len(text.split())) for text in request.inputs))
    normalized_usage = {
        "prompt_tokens": prompt_tokens,
        "total_tokens": usage.get("total_tokens", prompt_tokens),
    }
    return EmbeddingResponse(
        model_id=request.model_id,
        data=[EmbeddingVector(index=index, embedding=vector) for index, vector in enumerate(vectors)],
        usage=normalized_usage,
    )


def _embedding_vectors_from_result(result: Any) -> list[list[float]]:
    payload = result
    if isinstance(result, EmbeddingResponse):
        return [[float(value) for value in item.embedding] for item in result.data]
    if isinstance(result, dict):
        payload = result.get("data", result.get("embeddings", result.get("vectors", result.get("results", result))))
    return _normalize_embedding_vectors(payload)


def _normalize_embedding_vectors(payload: Any) -> list[list[float]]:
    if payload is None:
        return []
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    if _is_numeric_sequence(payload):
        return [[float(value) for value in payload]]
    vectors: list[list[float]] = []
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            if isinstance(item, EmbeddingVector):
                vectors.append([float(value) for value in item.embedding])
                continue
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, dict):
                vector_payload = item.get("embedding", item.get("vector", item.get("values", [])))
                if hasattr(vector_payload, "tolist"):
                    vector_payload = vector_payload.tolist()
                if _is_numeric_sequence(vector_payload):
                    vectors.append([float(value) for value in vector_payload])
                continue
            if _is_numeric_sequence(item):
                vectors.append([float(value) for value in item])
    return vectors


def _normalize_usage(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            normalized[key] = int(value)
    return normalized


def _is_numeric_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and all(
        isinstance(item, (int, float)) for item in value
    )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    shared_size = min(len(left), len(right))
    if shared_size == 0:
        return 0.0
    left_values = [float(left[index]) for index in range(shared_size)]
    right_values = [float(right[index]) for index in range(shared_size)]
    numerator = sum(left_value * right_value for left_value, right_value in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
