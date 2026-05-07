"""First-pass llama.cpp runtime adapter."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
from collections.abc import AsyncIterator, Sequence
from importlib import import_module
from typing import Any

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    PerformanceFeatureOwnership,
    RuntimeAffinity,
    SpeculationMode,
    runtime_performance_feature_report,
)
from lewlm.core.errors import ConfigurationError
from lewlm.runtime.base import ManagedTextRuntime
from lewlm.runtime.prefix_cache import longest_token_prefix
from lewlm.structured_output import GrammarResponseFormat, JSONSchemaResponseFormat, StructuredOutputRuntimeStatus

_LLAMA_PREFILL_BATCH_PARAMETERS = ("n_batch", "batch_size", "prompt_batch_size")
_LLAMA_PREFILL_UBATCH_PARAMETERS = ("n_ubatch", "ubatch_size", "prompt_ubatch_size")


class LlamaCppRuntime(ManagedTextRuntime):
    """Adapter for GGUF-backed inference through llama.cpp."""

    name = "llamacpp"
    affinity = RuntimeAffinity.LLAMACPP
    supported_formats = (ModelFormat.GGUF,)
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
        self._prefill_request_count = 0
        self._prefill_prompt_tokens = 0
        self._prefill_batch_count = 0
        self._model_performance_controls: dict[str, dict[str, dict[str, Any]]] = {}

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
        kwargs: dict[str, Any] = {
            "model_path": manifest.source_path,
            "n_ctx": manifest.context_length or 4096,
            "verbose": False,
            **load_options,
        }
        prompt_lookup_helper = self._build_prompt_lookup_helper()
        if prompt_lookup_helper is not None:
            kwargs["draft_model"] = prompt_lookup_helper
            self._prompt_lookup_enabled_model_ids.add(manifest.model_id)
        client = llama_class(**kwargs)
        prefix_cache = self._build_prefix_cache_wrapper(llama_cpp=llama_cpp)
        if prefix_cache is not None:
            if hasattr(client, "set_cache") and callable(client.set_cache):
                client.set_cache(prefix_cache)
            else:
                setattr(client, "cache", prefix_cache)
            setattr(client, "_lewlm_prefix_cache", prefix_cache)
        self._clients[manifest.model_id] = client
        self._model_performance_controls[manifest.model_id] = control_snapshot

    async def _unload_model(self, model_id: str) -> None:
        self._clients.pop(model_id, None)
        self._prompt_lookup_enabled_model_ids.discard(model_id)
        self._model_performance_controls.pop(model_id, None)

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        self._validate_speculation_request(request)
        client = self._clients[request.model_id]
        prompt_tokens = self._tokenize_request_messages(request)
        self._set_speculation_execution_metadata(request)
        request.metadata["performance_controls"] = self._request_performance_controls(model_id=request.model_id)
        prefix_cache_before = self._prefix_cache_snapshot_for_client(client)
        structured_output_options = self._structured_output_options(request=request, client=client)
        response = client.create_chat_completion(
            messages=[{"role": message.role, "content": message.content} for message in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=False,
            **structured_output_options,
        )
        self._record_prefix_cache_request(request=request, client=client, before=prefix_cache_before)
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
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if isinstance(content, str) and content:
                yield content
            await asyncio.sleep(0)
        self._record_prefix_cache_request(request=request, client=client, before=prefix_cache_before)
        self._record_prefill_request(model_id=request.model_id, prompt_token_count=len(prompt_tokens))
        if self._prompt_lookup_active_for_request(request):
            self._prompt_lookup_request_count += 1

    def _tokenize(self, text: str) -> list[int]:
        client = self._clients[next(iter(self._clients))]
        return list(client.tokenize(text.encode("utf-8")))

    def _detokenize(self, tokens: Sequence[int]) -> str:
        client = self._clients[next(iter(self._clients))]
        return client.detokenize(list(tokens)).decode("utf-8")

    def performance_feature_snapshot(self) -> dict[str, Any]:
        supported = self._supports_prompt_lookup_speculation()
        prefix_cache_supported = self._supports_prefix_cache()
        prefix_cache_metrics = self._aggregate_prefix_cache_metrics()
        prefill_controls = self._aggregate_control_entries("prefill_optimization")
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
                    else "Installed llama-cpp-python does not expose `LlamaRAMCache` or a compatible cache setter."
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
                notes=self._control_notes("prefill_optimization"),
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
                    else "Installed llama-cpp-python does not expose `LlamaPromptLookupDecoding`."
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
        }

    def _supports_prefix_cache(self) -> bool:
        if not self.is_available():
            return False
        llama_cpp = import_module("llama_cpp")
        return callable(getattr(llama_cpp, "LlamaRAMCache", None))

    def _structured_output_options(self, *, request: GenerateRequest, client: Any) -> dict[str, Any]:
        contract = request.structured_output
        if contract is None or contract.type == "text":
            return {}
        capability = self._structured_output_capability(client)
        if isinstance(contract, JSONSchemaResponseFormat):
            if capability["json_schema"]:
                grammar = self._json_schema_grammar(contract)
                if grammar is None:
                    request.metadata["structured_output_runtime"] = StructuredOutputRuntimeStatus(
                        runtime=self.name,
                        mode="json_schema",
                        enforcement="prompt_guided",
                        decoder_enforced=False,
                        fallback_used=True,
                        fallback_reason=(
                            "Installed llama.cpp bindings do not expose JSON-schema grammar compilation for "
                            "decode-time constrained decoding."
                        ),
                    ).model_dump(mode="json")
                    return {}
                request.metadata["structured_output_runtime"] = StructuredOutputRuntimeStatus(
                    runtime=self.name,
                    mode="json_schema",
                    enforcement="decode_time",
                    decoder_enforced=True,
                    fallback_used=False,
                ).model_dump(mode="json")
                return {"grammar": grammar}
            request.metadata["structured_output_runtime"] = StructuredOutputRuntimeStatus(
                runtime=self.name,
                mode="json_schema",
                enforcement="prompt_guided",
                decoder_enforced=False,
                fallback_used=True,
                fallback_reason=(
                    "Installed llama.cpp bindings do not expose JSON-schema grammar compilation for "
                    "decode-time constrained decoding."
                ),
            ).model_dump(mode="json")
            return {}
        if contract.syntax.casefold() not in {"ebnf", "gbnf"}:
            request.metadata["structured_output_runtime"] = StructuredOutputRuntimeStatus(
                runtime=self.name,
                mode="grammar",
                enforcement="prompt_guided",
                decoder_enforced=False,
                fallback_used=True,
                fallback_reason=(
                    f"llama.cpp constrained decoding expects `ebnf`/`gbnf`-style grammars; received `{contract.syntax}`."
                ),
            ).model_dump(mode="json")
            return {}
        if capability["grammar"]:
            grammar = self._grammar_from_string(contract.grammar)
            if grammar is None:
                request.metadata["structured_output_runtime"] = StructuredOutputRuntimeStatus(
                    runtime=self.name,
                    mode="grammar",
                    enforcement="prompt_guided",
                    decoder_enforced=False,
                    fallback_used=True,
                    fallback_reason="Installed llama.cpp bindings do not expose grammar-based decode-time constrained decoding.",
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
        request.metadata["structured_output_runtime"] = StructuredOutputRuntimeStatus(
            runtime=self.name,
            mode="grammar",
            enforcement="prompt_guided",
            decoder_enforced=False,
            fallback_used=True,
            fallback_reason="Installed llama.cpp bindings do not expose grammar-based decode-time constrained decoding.",
        ).model_dump(mode="json")
        return {}

    @staticmethod
    def _structured_output_capability(client: Any) -> dict[str, bool]:
        create_chat_completion = getattr(client, "create_chat_completion", None)
        supports_grammar_parameter = False
        if callable(create_chat_completion):
            try:
                signature = inspect.signature(create_chat_completion)
            except (TypeError, ValueError):
                signature = None
            supports_grammar_parameter = signature is not None and "grammar" in signature.parameters
        if not supports_grammar_parameter:
            return {"grammar": False, "json_schema": False}
        try:
            llama_cpp = import_module("llama_cpp")
        except ImportError:
            return {"grammar": False, "json_schema": False}
        grammar_class = getattr(llama_cpp, "LlamaGrammar", None)
        supports_grammar = callable(getattr(grammar_class, "from_string", None))
        supports_json_schema = callable(getattr(grammar_class, "from_json_schema", None))
        return {
            "grammar": supports_grammar,
            "json_schema": supports_grammar and supports_json_schema,
        }

    @staticmethod
    def _grammar_class() -> Any | None:
        try:
            llama_cpp = import_module("llama_cpp")
        except ImportError:
            return None
        return getattr(llama_cpp, "LlamaGrammar", None)

    @classmethod
    def _json_schema_grammar(cls, contract: JSONSchemaResponseFormat) -> Any | None:
        grammar_class = cls._grammar_class()
        factory = getattr(grammar_class, "from_json_schema", None)
        if not callable(factory):
            return None
        return factory(json.dumps(contract.schema_payload, sort_keys=True), verbose=False)

    @classmethod
    def _grammar_from_string(cls, grammar: str) -> Any | None:
        grammar_class = cls._grammar_class()
        factory = getattr(grammar_class, "from_string", None)
        if not callable(factory):
            return None
        return factory(grammar, verbose=False)

    def _build_prefix_cache_wrapper(self, *, llama_cpp: Any) -> "_InstrumentedLlamaRamCache | None":
        cache_class = getattr(llama_cpp, "LlamaRAMCache", None)
        if not callable(cache_class):
            return None
        return _InstrumentedLlamaRamCache(cache_class=cache_class)

    def _load_performance_controls(self, llama_class: type[Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        parameter_names = _callable_parameter_names(llama_class)
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
        return options, {
            "prefill_optimization": _performance_control_payload(
                requested=True,
                supported=bool(applied_prefill_parameters),
                effective="enabled" if applied_prefill_parameters else "unsupported",
                reason=(
                    "LewLM applied llama.cpp prefill batch-size controls during model load."
                    if applied_prefill_parameters
                    else "Installed llama.cpp bindings do not expose a LewLM-supported prefill batch-size parameter."
                ),
                applied_parameters=tuple(applied_prefill_parameters),
                rejected_parameters=(() if applied_prefill_parameters else ("prefill_token_batch_size",)),
                requested_prefill_token_batch_size=self._settings.prefill_token_batch_size,
                effective_prefill_token_batch_size=(
                    self._settings.prefill_token_batch_size if applied_prefill_parameters else None
                ),
            ),
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

    def _request_performance_controls(self, *, model_id: str) -> dict[str, dict[str, dict[str, Any]]]:
        controls = self._model_performance_controls.get(model_id, {})
        return {"load": copy.deepcopy(controls)} if controls else {}

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

    def _control_notes(self, control_name: str) -> list[str]:
        notes: list[str] = []
        seen: set[str] = set()
        for payload in self._aggregate_control_entries(control_name):
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

    def _record_prefix_cache_request(self, *, request: GenerateRequest, client: Any, before: dict[str, int]) -> None:
        cache = getattr(client, "_lewlm_prefix_cache", None)
        if cache is None:
            request.metadata["prefix_cache"] = {"cache_hits": 0, "cache_misses": 0, "cache_saves": 0}
            return
        after = cache.snapshot()
        metrics = {
            "cache_hits": max(after["cache_hits"] - before["cache_hits"], 0),
            "cache_misses": max(after["cache_misses"] - before["cache_misses"], 0),
            "cache_saves": max(after["cache_saves"] - before["cache_saves"], 0),
            "saved_prefill_tokens": max(after["saved_prefill_tokens"] - before["saved_prefill_tokens"], 0),
            "max_saved_prefill_tokens": max(after["max_saved_prefill_tokens"], before["max_saved_prefill_tokens"]),
        }
        self._prefix_cache_hits += metrics["cache_hits"]
        self._prefix_cache_misses += metrics["cache_misses"]
        self._prefix_cache_saves += metrics["cache_saves"]
        self._saved_prefill_tokens += metrics["saved_prefill_tokens"]
        self._max_saved_prefill_tokens = max(self._max_saved_prefill_tokens, metrics["max_saved_prefill_tokens"])
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
        }

    def _supports_prompt_lookup_speculation(self) -> bool:
        if not self.is_available():
            return False
        return self._prompt_lookup_class() is not None

    def _build_prompt_lookup_helper(self) -> Any | None:
        if not self._settings.prompt_lookup_speculation_enabled:
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
