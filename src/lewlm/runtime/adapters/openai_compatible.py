"""Local-only adapter runtime for OpenAI-compatible accelerator servers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    CapabilityName,
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    ModelModality,
    RuntimeAffinity,
    RuntimeCandidateReport,
)
from lewlm.core.errors import RuntimeUnavailableError
from lewlm.runtime.base import ManagedTextRuntime

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_PERFORMANCE_FEATURE_ORDER = (
    "continuous_batching",
    "prefix_cache",
    "paged_kv_cache",
    "kv_cache_quantization",
    "prefill_optimization",
    "speculative_decoding",
)
_FEATURE_LABELS = {
    "continuous_batching": "continuous batching",
    "prefix_cache": "prefix cache reuse",
    "paged_kv_cache": "paged KV cache",
    "kv_cache_quantization": "KV cache quantization",
    "prefill_optimization": "prefill optimization",
    "speculative_decoding": "speculative decoding",
}
_PROFILE_FEATURES: dict[str, dict[str, tuple[bool, str, str]]] = {
    "openai_compatible": {
        "continuous_batching": (
            True,
            "partial",
            "Local scheduler overlap can be preserved, but batching visibility depends on the upstream server.",
        ),
        "prefix_cache": (
            False,
            "unsupported",
            "Generic OpenAI-compatible endpoints do not expose prompt-prefix cache state or reuse counters.",
        ),
        "paged_kv_cache": (
            False,
            "unsupported",
            "Paged KV cache behavior is not surfaced through the generic compatibility layer.",
        ),
        "kv_cache_quantization": (
            False,
            "unsupported",
            "Per-request KV cache quantization controls are not surfaced through the compatibility layer.",
        ),
        "prefill_optimization": (
            True,
            "partial",
            "Fast prefill may still happen inside the external engine, but request-level tuning knobs are not preserved.",
        ),
        "speculative_decoding": (
            False,
            "unsupported",
            "Speculative decoding controls are not mapped through the local compatibility contract.",
        ),
    },
    "vmlx": {
        "continuous_batching": (
            True,
            "supported",
            "The external scheduler can preserve continuous batching for compatible local OpenAI-style requests.",
        ),
        "prefix_cache": (
            True,
            "supported",
            "vMLX-class servers preserve prompt reuse internally for repeated compatible prefixes.",
        ),
        "paged_kv_cache": (
            True,
            "supported",
            "Paged KV state remains available inside the external accelerator runtime.",
        ),
        "kv_cache_quantization": (
            True,
            "partial",
            "KV cache quantization may remain active in the external engine, but LewLM cannot tune it per request.",
        ),
        "prefill_optimization": (
            True,
            "supported",
            "Prefill acceleration remains active for compatible requests on the external server.",
        ),
        "speculative_decoding": (
            False,
            "unsupported",
            "Speculative decoding controls are not part of the adapter contract yet.",
        ),
    },
    "omlx": {
        "continuous_batching": (
            True,
            "supported",
            "The external server can keep local request batching active for compatible workloads.",
        ),
        "prefix_cache": (
            True,
            "partial",
            "Prefix reuse may stay active, but the adapter cannot surface detailed hit accounting.",
        ),
        "paged_kv_cache": (
            True,
            "partial",
            "KV residency stays external, but LewLM cannot expose allocator-level paging details.",
        ),
        "kv_cache_quantization": (
            False,
            "unsupported",
            "KV cache quantization settings are not mapped into the adapter path.",
        ),
        "prefill_optimization": (
            True,
            "supported",
            "Prefill acceleration remains available for compatible requests.",
        ),
        "speculative_decoding": (
            False,
            "unsupported",
            "Speculative decoding controls are not preserved through the compatibility layer.",
        ),
    },
    "vllm_mlx": {
        "continuous_batching": (
            True,
            "supported",
            "The external runtime preserves batched scheduling for local compatible requests.",
        ),
        "prefix_cache": (
            True,
            "supported",
            "Automatic prefix reuse remains available inside the external runtime.",
        ),
        "paged_kv_cache": (
            True,
            "supported",
            "Paged KV cache residency remains active on the external accelerator path.",
        ),
        "kv_cache_quantization": (
            True,
            "partial",
            "Quantized KV residency may remain active, but LewLM cannot inspect or tune the policy directly.",
        ),
        "prefill_optimization": (
            True,
            "supported",
            "Prefill acceleration remains active for local compatible requests.",
        ),
        "speculative_decoding": (
            False,
            "unsupported",
            "Speculative decoding remains outside the adapter contract.",
        ),
    },
}


def summarize_feature_preservation(
    *,
    native_features: dict[str, Any],
    external_features: dict[str, Any],
) -> dict[str, Any]:
    preserved: list[str] = []
    degraded: list[str] = []
    rejected: list[str] = []
    details: dict[str, dict[str, Any]] = {}
    for feature_name in _PERFORMANCE_FEATURE_ORDER:
        native_entry = _feature_entry(native_features.get(feature_name))
        external_entry = _feature_entry(external_features.get(feature_name))
        if not native_entry["supported"]:
            continue
        support_level = external_entry["support_level"]
        status = "rejected"
        if external_entry["supported"] and support_level == "supported":
            status = "preserved"
            preserved.append(feature_name)
        elif external_entry["supported"] and support_level == "partial":
            status = "degraded"
            degraded.append(feature_name)
        else:
            rejected.append(feature_name)
        details[feature_name] = {
            "feature": feature_name,
            "label": _FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ")),
            "status": status,
            "native": native_entry,
            "external": external_entry,
        }
    return {
        "preserved": preserved,
        "degraded": degraded,
        "rejected": rejected,
        "details": details,
    }


class LocalOpenAICompatibleAdapterRuntime(ManagedTextRuntime):
    """Route compatible text requests to a loopback-only OpenAI-style local server."""

    name = "local_external_adapter"
    affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR
    supported_formats = (ModelFormat.MLX,)
    supported_modalities = (ModelModality.TEXT,)
    supported_capabilities = frozenset({CapabilityName.CHAT, CapabilityName.STREAMING})
    supported_systems = ("Darwin",)
    supported_machines = ("arm64",)
    platform_guidance = (
        "Enable LEWLM_EXTERNAL_ACCELERATOR_ENABLED with a loopback-only "
        "LEWLM_EXTERNAL_ACCELERATOR_BASE_URL to use a local OpenAI-compatible accelerator server."
    )

    def __init__(self, *, settings: LewLMSettings) -> None:
        super().__init__()
        self._settings = settings
        self._discovered_model_ids: tuple[str, ...] | None = None
        self._discovery_error: str | None = None

    def supports_manifest(self, manifest: ModelManifest) -> bool:
        if not super().supports_manifest(manifest):
            return False
        if not self.is_available():
            return False
        try:
            return self._resolve_remote_model_id(manifest) is not None
        except RuntimeUnavailableError:
            return False

    def candidate_report(self, manifest: ModelManifest | None = None) -> RuntimeCandidateReport:
        report = super().candidate_report(manifest)
        if manifest is None or not report.available:
            return report
        try:
            remote_model_id = self._resolve_remote_model_id(manifest)
        except RuntimeUnavailableError as exc:
            return report.model_copy(
                update={
                    "supports_manifest": False,
                    "availability_reason": str(exc),
                },
            )
        if remote_model_id is not None:
            return report
        available_models = list(self._available_remote_models())
        return report.model_copy(
            update={
                "supports_manifest": False,
                "availability_reason": (
                    "The configured external accelerator endpoint did not advertise a compatible local model id. "
                    f"Available ids: {available_models or ['none discovered']}."
                ),
            },
        )

    def performance_feature_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for feature_name, (supported, support_level, reason) in _profile_feature_map(self._settings).items():
            snapshot[feature_name] = {
                "supported": supported,
                "active": supported,
                "support_level": support_level,
                "reason": reason,
                "metrics": {
                    "adapter_profile": self._settings.external_accelerator_profile,
                    "contract": "openai_compatible_local",
                },
            }
        return snapshot

    async def _load_model(self, manifest: ModelManifest) -> None:
        remote_model_id = self._resolve_remote_model_id(manifest)
        if remote_model_id is None:
            raise RuntimeUnavailableError(
                "The configured external accelerator does not advertise a compatible local model.",
                details={
                    "runtime": self.name,
                    "model_id": manifest.model_id,
                    "advertised_model_ids": list(self._available_remote_models()),
                },
            )

    async def _unload_model(self, model_id: str) -> None:
        return None

    async def _warm_model(self, model_id: str) -> None:
        manifest = self._loaded_manifests[model_id]
        request = GenerateRequest(
            model_id=model_id,
            messages=[{"role": "user", "content": "Warm the local accelerator path."}],
            max_tokens=1,
        )
        await self._generate_with_manifest(manifest, request)

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        manifest = self._loaded_manifests[request.model_id]
        return await self._generate_with_manifest(manifest, request)

    async def _stream_generate(self, request: GenerateRequest):
        manifest = self._loaded_manifests[request.model_id]
        remote_model_id = self._require_remote_model_id(manifest)
        payload = self._chat_payload(remote_model_id=remote_model_id, request=request, stream=True)
        queue: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()

        def _worker() -> None:
            try:
                for delta in self._stream_chat_completion(payload):
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
            except Exception as exc:  # pragma: no cover - surfaced through queue
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
            yield cast(str, item)

    def _tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _detokenize(self, tokens) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")

    def _check_environment(self) -> tuple[bool, str | None]:
        if not self._settings.external_accelerator_enabled:
            return False, "External accelerator adapter is disabled."
        base_url = self._settings.external_accelerator_base_url
        if base_url is None:
            return False, "Set LEWLM_EXTERNAL_ACCELERATOR_BASE_URL to a local loopback endpoint."
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            return False, "External accelerator base URL must use http or https."
        host = parsed.hostname
        if host not in _LOOPBACK_HOSTS:
            return False, "External accelerator base URL must target a loopback-only local host."
        return True, None

    async def _generate_with_manifest(self, manifest: ModelManifest, request: GenerateRequest) -> GenerateResponse:
        remote_model_id = self._require_remote_model_id(manifest)
        payload = self._chat_payload(remote_model_id=remote_model_id, request=request, stream=False)
        response_payload = await asyncio.to_thread(self._request_json, "POST", "/v1/chat/completions", payload)
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeUnavailableError(
                "External accelerator returned an invalid chat completion payload.",
                details={"runtime": self.name, "response_keys": sorted(response_payload)},
            )
        message = choices[0].get("message", {})
        output_text = str(message.get("content", ""))
        usage = response_payload.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        return GenerateResponse(
            model_id=request.model_id,
            output_text=output_text,
            finish_reason=str(choices[0].get("finish_reason", "stop")),
            usage={
                key: int(value)
                for key, value in usage.items()
                if isinstance(key, str) and isinstance(value, int | float)
            },
        )

    def _require_remote_model_id(self, manifest: ModelManifest) -> str:
        remote_model_id = self._resolve_remote_model_id(manifest)
        if remote_model_id is None:
            raise RuntimeUnavailableError(
                "External accelerator adapter could not match the manifest to a local advertised model.",
                details={
                    "runtime": self.name,
                    "model_id": manifest.model_id,
                    "advertised_model_ids": list(self._available_remote_models()),
                },
            )
        return remote_model_id

    def _resolve_remote_model_id(self, manifest: ModelManifest) -> str | None:
        available_ids = {model_id.casefold(): model_id for model_id in self._available_remote_models()}
        for candidate in _remote_model_candidates(manifest):
            resolved = available_ids.get(candidate.casefold())
            if resolved is not None:
                return resolved
        return None

    def _available_remote_models(self) -> tuple[str, ...]:
        if self._discovered_model_ids is not None:
            return self._discovered_model_ids
        payload = self._request_json("GET", "/v1/models", None)
        data = payload.get("data")
        model_ids: list[str] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id:
                    model_ids.append(model_id)
        self._discovered_model_ids = tuple(model_ids)
        self._discovery_error = None
        return self._discovered_model_ids

    def _chat_payload(self, *, remote_model_id: str, request: GenerateRequest, stream: bool) -> dict[str, Any]:
        return {
            "model": remote_model_id,
            "messages": [{"role": message.role, "content": message.content} for message in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }

    def _stream_chat_completion(self, payload: dict[str, Any]):
        with self._request("POST", "/v1/chat/completions", payload=payload) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeUnavailableError(
                        "External accelerator returned malformed streaming JSON.",
                        details={"runtime": self.name, "line": data},
                    ) from exc
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta", {})
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield content

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        with self._request(method, path, payload=payload) as response:
            body = response.read().decode("utf-8", errors="ignore")
        if not body:
            return {}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeUnavailableError(
                "External accelerator returned malformed JSON.",
                details={"runtime": self.name, "path": path},
            ) from exc
        if isinstance(parsed, dict):
            return parsed
        raise RuntimeUnavailableError(
            "External accelerator returned an unexpected JSON payload.",
            details={"runtime": self.name, "path": path, "payload_type": type(parsed).__name__},
        )

    def _request(self, method: str, path: str, *, payload: dict[str, Any] | None = None):
        available, reason = self._check_environment()
        if not available:
            raise RuntimeUnavailableError(
                reason or "External accelerator adapter is unavailable.",
                details={"runtime": self.name},
            )
        base_url = self._settings.external_accelerator_base_url or ""
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
            data=data,
            headers=headers,
            method=method,
        )
        try:
            return urlopen(request, timeout=self._settings.external_accelerator_timeout_seconds)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeUnavailableError(
                f"External accelerator request failed with HTTP {exc.code}.",
                details={"runtime": self.name, "path": path, "body": body},
            ) from exc
        except URLError as exc:
            self._discovered_model_ids = ()
            self._discovery_error = str(exc.reason)
            raise RuntimeUnavailableError(
                "Could not reach the configured external accelerator endpoint.",
                details={"runtime": self.name, "path": path, "reason": str(exc.reason)},
            ) from exc


def _profile_feature_map(settings: LewLMSettings) -> dict[str, tuple[bool, str, str]]:
    return _PROFILE_FEATURES.get(settings.external_accelerator_profile, _PROFILE_FEATURES["openai_compatible"])


def _remote_model_candidates(manifest: ModelManifest) -> tuple[str, ...]:
    candidates: list[str] = []
    explicit_model_id = manifest.metadata.get("external_adapter_model_id")
    if isinstance(explicit_model_id, str) and explicit_model_id:
        candidates.append(explicit_model_id)
    explicit_model_ids = manifest.metadata.get("external_adapter_model_ids")
    if isinstance(explicit_model_ids, list):
        candidates.extend(
            item
            for item in explicit_model_ids
            if isinstance(item, str) and item
        )
    candidates.extend((manifest.model_id, manifest.display_name, Path(manifest.source_path).name))
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def _feature_entry(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"supported": False, "support_level": "unsupported", "reason": None}
    supported = bool(payload.get("supported"))
    support_level = str(payload.get("support_level", "supported" if supported else "unsupported"))
    return {
        "supported": supported,
        "support_level": support_level,
        "reason": payload.get("reason"),
        "metrics": payload.get("metrics", {}),
    }
