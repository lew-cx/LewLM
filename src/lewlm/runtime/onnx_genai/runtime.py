"""ONNX Runtime GenAI runtime adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from importlib import import_module
import threading
from typing import Any

from lewlm.core.contracts import (
    CapabilityName,
    GenerateMessage,
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    ModelModality,
    RuntimeAffinity,
    RuntimeCandidateReport,
    runtime_support_path_for_affinity,
)
from lewlm.core.errors import RuntimeUnavailableError
from lewlm.runtime.base import ManagedTextRuntime


@dataclass(slots=True)
class _ONNXGenAIClient:
    model: Any
    tokenizer: Any


class ONNXGenAIRuntime(ManagedTextRuntime):
    """Adapter for ONNX Runtime GenAI model bundles."""

    name = "onnx_genai"
    affinity = RuntimeAffinity.ONNX_GENAI
    supported_formats = (ModelFormat.ONNX_GENAI,)
    supported_modalities = (ModelModality.TEXT, ModelModality.MULTIMODAL)
    supported_capabilities = frozenset({CapabilityName.CHAT, CapabilityName.STREAMING})
    supported_systems = ("Windows", "Linux", "Darwin")
    platform_guidance = (
        "Install the `onnx_genai` extra to enable ONNX Runtime GenAI bundle probing. "
        "On Windows, DirectML provider support depends on the installed ONNX Runtime GenAI wheel and GPU driver stack."
    )

    def __init__(self) -> None:
        super().__init__()
        self._clients: dict[str, _ONNXGenAIClient] = {}

    def _check_environment(self) -> tuple[bool, str | None]:
        try:
            module = import_module("onnxruntime_genai")
        except ImportError:
            return False, (
                "onnxruntime-genai is not installed. Install the `onnx_genai` extra; on Windows, use a "
                "DirectML-capable ONNX Runtime GenAI build when you want local GPU acceleration."
            )
        missing = [
            name
            for name in ("Model", "Tokenizer", "GeneratorParams", "Generator")
            if not hasattr(module, name)
        ]
        if missing:
            return False, (
                "onnxruntime-genai is installed, but the package does not expose the required Python GenAI "
                f"runtime objects: {', '.join(missing)}."
            )
        return True, None

    def candidate_report(self, manifest: ModelManifest | None = None) -> RuntimeCandidateReport:
        report = super().candidate_report(manifest)
        provider_plan = _provider_plan()
        metadata: dict[str, Any] = {
            "provider_family": "onnxruntime_genai",
            "planned_execution_providers": provider_plan,
            "evidence_policy": "load/generate probes and benchmarks upgrade support beyond package availability",
        }
        return report.model_copy(
            update={
                "support_path": runtime_support_path_for_affinity(self.affinity) or "packaged",
                "metadata": {**report.metadata, **metadata},
            },
        )

    async def _load_model(self, manifest: ModelManifest) -> None:
        module = import_module("onnxruntime_genai")
        try:
            model = module.Model(manifest.source_path)
            tokenizer = module.Tokenizer(model)
        except Exception as exc:  # noqa: BLE001 - backend exceptions vary by package version.
            raise RuntimeUnavailableError(
                "ONNX Runtime GenAI failed to load the model bundle.",
                details={
                    "runtime": self.name,
                    "model_id": manifest.model_id,
                    "source_path": manifest.source_path,
                    "cause_type": type(exc).__name__,
                    "cause": str(exc),
                },
            ) from exc
        self._clients[manifest.model_id] = _ONNXGenAIClient(model=model, tokenizer=tokenizer)

    async def _unload_model(self, model_id: str) -> None:
        self._clients.pop(model_id, None)

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        output_text, usage = await asyncio.to_thread(
            self._generate_text_sync,
            request,
        )
        return GenerateResponse(
            model_id=request.model_id,
            output_text=output_text,
            finish_reason="stop",
            usage=usage,
        )

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()

        def _worker() -> None:
            try:
                for delta in self._generate_deltas_sync(request):
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
            except Exception as exc:  # pragma: no cover - surfaced through queue.
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        threading.Thread(target=_worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield str(item)

    def _tokenize(self, text: str) -> list[int]:
        client = self._first_client()
        if client is not None and hasattr(client.tokenizer, "encode"):
            return _int_list(client.tokenizer.encode(text))
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens: Sequence[int]) -> str:
        client = self._first_client()
        if client is not None and hasattr(client.tokenizer, "decode"):
            try:
                decoded = client.tokenizer.decode(list(tokens))
                if isinstance(decoded, str):
                    return decoded
            except Exception:
                pass
        return bytes(tokens).decode("utf-8")

    def _generate_text_sync(self, request: GenerateRequest) -> tuple[str, dict[str, int]]:
        text_parts: list[str] = []
        completion_tokens = 0
        for delta in self._generate_deltas_sync(request):
            text_parts.append(delta)
            completion_tokens += 1
        prompt_tokens = len(self._encode_prompt(self._prompt_from_messages(request.messages), request.model_id))
        return (
            "".join(text_parts),
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    def _generate_deltas_sync(self, request: GenerateRequest) -> Iterator[str]:
        client = self._clients[request.model_id]
        prompt = self._prompt_from_messages(request.messages)
        input_tokens = self._encode_prompt(prompt, request.model_id)
        generator, tokenizer_stream = self._build_generator(
            client=client,
            input_tokens=input_tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        generated_count = 0
        while not bool(generator.is_done()) and generated_count < request.max_tokens:
            if hasattr(generator, "compute_logits"):
                generator.compute_logits()
            if not hasattr(generator, "generate_next_token"):
                raise RuntimeUnavailableError(
                    "ONNX Runtime GenAI Generator does not expose `generate_next_token`.",
                    details={"runtime": self.name},
                )
            generator.generate_next_token()
            token = _latest_generated_token(generator)
            if token is None:
                continue
            generated_count += 1
            delta = _decode_token(tokenizer=client.tokenizer, tokenizer_stream=tokenizer_stream, token=token)
            if delta:
                yield delta

    def _build_generator(
        self,
        *,
        client: _ONNXGenAIClient,
        input_tokens: list[int],
        max_tokens: int,
        temperature: float,
    ) -> tuple[Any, Any | None]:
        module = import_module("onnxruntime_genai")
        params = module.GeneratorParams(client.model)
        _set_search_options(params, prompt_token_count=len(input_tokens), max_tokens=max_tokens, temperature=temperature)
        tokens_attached_before_generator = _attach_input_tokens_to_params(params, input_tokens)
        generator = module.Generator(client.model, params)
        if not tokens_attached_before_generator and hasattr(generator, "append_tokens"):
            generator.append_tokens(input_tokens)
        tokenizer_stream = client.tokenizer.create_stream() if hasattr(client.tokenizer, "create_stream") else None
        return generator, tokenizer_stream

    def _encode_prompt(self, prompt: str, model_id: str) -> list[int]:
        client = self._clients[model_id]
        if not hasattr(client.tokenizer, "encode"):
            raise RuntimeUnavailableError(
                "ONNX Runtime GenAI tokenizer does not expose `encode`.",
                details={"runtime": self.name, "model_id": model_id},
            )
        return _int_list(client.tokenizer.encode(prompt))

    @staticmethod
    def _prompt_from_messages(messages: Sequence[GenerateMessage]) -> str:
        lines = []
        for message in messages:
            role = message.role.strip() or "user"
            lines.append(f"{role}: {message.content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _first_client(self) -> _ONNXGenAIClient | None:
        return next(iter(self._clients.values()), None)


def _provider_plan() -> list[dict[str, str]]:
    return [
        {
            "provider": "directml",
            "platform": "Windows",
            "status": "requires_model_probe",
        },
        {
            "provider": "cuda",
            "platform": "Windows/Linux",
            "status": "requires_model_probe",
        },
        {
            "provider": "cpu",
            "platform": "Windows/Linux/Darwin",
            "status": "requires_model_probe",
        },
    ]


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                result.extend(_int_list(item))
            else:
                result.append(int(item))
        return result
    return [int(value)]


def _set_search_options(
    params: Any,
    *,
    prompt_token_count: int,
    max_tokens: int,
    temperature: float,
) -> None:
    if not hasattr(params, "set_search_options"):
        return
    options = {
        "max_length": prompt_token_count + max_tokens,
        "temperature": temperature,
    }
    try:
        params.set_search_options(**options)
        return
    except TypeError:
        pass
    try:
        params.set_search_options(max_new_tokens=max_tokens, temperature=temperature)
    except TypeError:
        params.set_search_options(max_length=prompt_token_count + max_tokens)


def _attach_input_tokens_to_params(params: Any, input_tokens: list[int]) -> bool:
    if hasattr(params, "input_ids"):
        try:
            params.input_ids = input_tokens
            return True
        except Exception:
            pass
    if hasattr(params, "set_input_ids"):
        params.set_input_ids(input_tokens)
        return True
    if hasattr(params, "set_inputs"):
        params.set_inputs({"input_ids": input_tokens})
        return True
    return False


def _latest_generated_token(generator: Any) -> int | None:
    for attr_name in ("get_next_tokens", "get_next_token"):
        if not hasattr(generator, attr_name):
            continue
        value = getattr(generator, attr_name)()
        tokens = _int_list(value)
        if tokens:
            return tokens[-1]
    if hasattr(generator, "get_sequence"):
        tokens = _int_list(generator.get_sequence(0))
        if tokens:
            return tokens[-1]
    return None


def _decode_token(*, tokenizer: Any, tokenizer_stream: Any | None, token: int) -> str:
    if tokenizer_stream is not None and hasattr(tokenizer_stream, "decode"):
        decoded = tokenizer_stream.decode(token)
        if decoded is not None:
            return str(decoded)
    if hasattr(tokenizer, "decode"):
        decoded = tokenizer.decode([token])
        if decoded is not None:
            return str(decoded)
    return ""
