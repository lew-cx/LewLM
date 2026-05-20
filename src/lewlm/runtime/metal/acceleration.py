"""Shared MLX graph-compilation and attention-kernel helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    GenerateRequest,
    PerformanceFeatureOwnership,
    runtime_performance_feature_report,
)
from lewlm.runtime.introspection import invoke_with_signature

_ACCELERATION_METADATA_KEY = "mlx_acceleration"
_ATTENTION_PARAMETER_CANDIDATES = (
    "attention_kernel",
    "attention_backend",
    "attention_implementation",
    "sdpa_kernel",
    "sdpa_backend",
)
_VALID_ATTENTION_MODES = frozenset({"stock", "flash_attention", "custom_sdpa"})


@dataclass(slots=True)
class _AttentionSupport:
    parameter_name: str | None
    supported_modes: tuple[str, ...]
    preferred_mode: str | None

    @property
    def supported(self) -> bool:
        return self.parameter_name is not None and bool(self.supported_modes)


class MLXAccelerationTracker:
    """Track MLX acceleration support, request usage, and stock fallbacks."""

    def __init__(
        self,
        *,
        settings: LewLMSettings,
        runtime_name: str,
        import_module_fn=import_module,
    ) -> None:
        self.settings = settings
        self.runtime_name = runtime_name
        self._import_module = import_module_fn
        self._compiled_callables: dict[str, Any] = {}
        self._disabled_compiled_callables: set[str] = set()
        self._disabled_compile_reasons: dict[str, str] = {}
        self._compile_attempt_count = 0
        self._compiled_request_count = 0
        self._compile_fallback_request_count = 0
        self._compile_failure_count = 0
        self._stock_request_count = 0
        self._flash_attention_request_count = 0
        self._custom_sdpa_request_count = 0
        self._kernel_fallback_request_count = 0
        self._last_kernel_path = "stock"
        self._last_requested_kernel_mode = settings.mlx_attention_kernel_mode
        self._last_fallback_reason: str | None = None

    def performance_feature_snapshot(self, *, callables: tuple[Any | None, ...]) -> dict[str, Any]:
        compile_supported, compile_reason = self._compile_support()
        attention_support = self._attention_support(callables)
        graph_notes = (
            []
            if self.settings.mlx_graph_compile_enabled
            else [
                "Enable `LEWLM_MLX_GRAPH_COMPILE_ENABLED=true` or use benchmark request overrides to exercise compiled MLX paths.",
            ]
        )
        attention_notes = (
            []
            if self.settings.mlx_attention_kernel_mode != "stock"
            else [
                "Set `LEWLM_MLX_ATTENTION_KERNEL_MODE=flash_attention` or `custom_sdpa` to request an accelerated attention hook.",
            ]
        )
        if attention_support.preferred_mode is not None:
            attention_notes.append(
                f"Installed MLX callables prefer `{attention_support.preferred_mode}` for accelerated attention requests.",
            )
        if self._last_fallback_reason is not None:
            attention_notes.append(f"Last fallback to stock path: {self._last_fallback_reason}")
        return {
            "graph_compilation": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.BACKEND_NATIVE
                    if compile_supported
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=self._compiled_request_count > 0,
                reason=compile_reason,
                notes=graph_notes,
                metrics=_compact_metrics(
                    configured_enabled=self.settings.mlx_graph_compile_enabled,
                    compile_attempts=self._compile_attempt_count,
                    compiled_requests=self._compiled_request_count,
                    compile_fallback_requests=self._compile_fallback_request_count,
                    compile_failures=self._compile_failure_count,
                    compiled_callable_count=len(self._compiled_callables),
                ),
            ),
            "attention_kernel_acceleration": runtime_performance_feature_report(
                ownership=(
                    PerformanceFeatureOwnership.BACKEND_NATIVE
                    if attention_support.supported
                    else PerformanceFeatureOwnership.UNSUPPORTED
                ),
                active=(self._flash_attention_request_count + self._custom_sdpa_request_count) > 0,
                reason=(
                    "Installed MLX generation entrypoints advertise an accelerated attention-kernel hook."
                    if attention_support.supported
                    else "Installed MLX generation entrypoints do not advertise an accelerated attention-kernel hook."
                ),
                notes=attention_notes,
                metrics=_compact_metrics(
                    configured_mode=self.settings.mlx_attention_kernel_mode,
                    preferred_mode=attention_support.preferred_mode,
                    supported_modes=",".join(attention_support.supported_modes) if attention_support.supported_modes else None,
                    kernel_parameter=attention_support.parameter_name,
                    stock_requests=self._stock_request_count,
                    flash_attention_requests=self._flash_attention_request_count,
                    custom_sdpa_requests=self._custom_sdpa_request_count,
                    kernel_fallback_requests=self._kernel_fallback_request_count,
                    last_kernel_path=self._last_kernel_path,
                ),
            ),
        }

    def invoke(
        self,
        *,
        request: GenerateRequest,
        callable_obj: Any,
        callable_key: str,
        provided_values: dict[str, Any],
        capability: str,
        passthrough_keys: tuple[str, ...] = (),
        phase: str = "decode",
    ) -> Any:
        compile_requested = self._graph_compile_requested(request)
        compile_supported, compile_reason = self._compile_support()
        attention_requested_mode = self._requested_attention_mode(request)
        attention_support = self._attention_support((callable_obj,))
        attention_options = (
            {attention_support.parameter_name: attention_requested_mode}
            if attention_support.parameter_name is not None
            and attention_requested_mode != "stock"
            and attention_requested_mode in attention_support.supported_modes
            else {}
        )
        accelerated_callable = callable_obj
        attempted_compile = False
        fallback_reasons: list[str] = []
        if compile_requested:
            if not compile_supported:
                fallback_reasons.append(compile_reason)
            elif callable_key in self._disabled_compiled_callables:
                fallback_reasons.append(
                    self._disabled_compile_reasons.get(
                        callable_key,
                        "LewLM previously disabled this compiled MLX callable after an earlier acceleration failure.",
                    ),
                )
            else:
                attempted_compile = True
                self._compile_attempt_count += 1
                compiled_callable = self._compiled_callable(callable_obj=callable_obj, callable_key=callable_key)
                if compiled_callable is not None:
                    accelerated_callable = compiled_callable
                else:
                    self._compile_fallback_request_count += 1
                    disabled_reason = self._disabled_compile_reasons.get(callable_key)
                    if disabled_reason is not None:
                        fallback_reasons.append(disabled_reason)
        attention_fallback_reason = self._attention_fallback_reason(
            requested_mode=attention_requested_mode,
            attention_support=attention_support,
        )
        if attention_fallback_reason is not None:
            fallback_reasons.append(attention_fallback_reason)
        attempted_kernel_mode = attention_requested_mode if attention_options else "stock"
        metadata = {
            "phase": phase,
            "requested_graph_compile": compile_requested,
            "graph_compile_supported": compile_supported,
            "effective_graph_compile": False,
            "requested_kernel_mode": attention_requested_mode,
            "effective_kernel_path": "stock",
            "attention_kernel_supported": attention_support.supported,
            "preferred_kernel_mode": attention_support.preferred_mode,
            "kernel_parameter": attention_support.parameter_name,
            "acceleration_fallback": bool(fallback_reasons),
        }
        if fallback_reasons:
            metadata["fallback_reason"] = " | ".join(dict.fromkeys(fallback_reasons))
        invocation_values = {**provided_values, **attention_options}
        try:
            result = invoke_with_signature(
                accelerated_callable,
                invocation_values,
                capability=capability,
                passthrough_keys=passthrough_keys,
            )
            metadata["effective_graph_compile"] = attempted_compile and accelerated_callable is not callable_obj
            metadata["phase_compile_state"] = "compiled" if metadata["effective_graph_compile"] else "stock"
            metadata["effective_kernel_path"] = attempted_kernel_mode
            self._record_effective_path(
                kernel_path=str(metadata["effective_kernel_path"]),
                graph_compile=bool(metadata["effective_graph_compile"]),
            )
            self._store_metadata(request=request, metadata=metadata, phase=phase)
            return result
        except Exception as exc:
            if not attempted_compile and attempted_kernel_mode == "stock":
                raise
            self._last_fallback_reason = f"{type(exc).__name__}: {exc}"
            metadata["acceleration_fallback"] = True
            metadata["fallback_reason"] = self._last_fallback_reason
            metadata["phase_compile_state"] = "stock"
            if attempted_compile:
                self._disable_compiled_callable(callable_key=callable_key, reason=self._last_fallback_reason)
                self._compile_failure_count += 1
                self._compile_fallback_request_count += 1
            if attempted_kernel_mode != "stock":
                self._kernel_fallback_request_count += 1
            result = invoke_with_signature(
                callable_obj,
                dict(provided_values),
                capability=capability,
                passthrough_keys=passthrough_keys,
            )
            metadata["effective_graph_compile"] = False
            metadata["effective_kernel_path"] = "stock"
            self._record_effective_path(kernel_path="stock", graph_compile=False)
            self._store_metadata(request=request, metadata=metadata, phase=phase)
            return result

    def _compiled_callable(self, *, callable_obj: Any, callable_key: str) -> Any | None:
        if callable_key in self._disabled_compiled_callables:
            return None
        cached = self._compiled_callables.get(callable_key)
        if cached is not None:
            return cached
        try:
            mlx_core = self._import_module("mlx.core")
        except ImportError:
            return None
        compile_fn = getattr(mlx_core, "compile", None)
        if not callable(compile_fn):
            return None
        try:
            compiled = compile_fn(callable_obj)
        except Exception as exc:
            self._disable_compiled_callable(callable_key=callable_key, reason=f"{type(exc).__name__}: {exc}")
            self._compile_failure_count += 1
            return None
        signature = _safe_signature(callable_obj)
        if signature is not None:
            try:
                compiled.__signature__ = signature
            except Exception:
                pass
        self._compiled_callables[callable_key] = compiled
        return compiled

    def _disable_compiled_callable(self, *, callable_key: str, reason: str | None = None) -> None:
        self._compiled_callables.pop(callable_key, None)
        self._disabled_compiled_callables.add(callable_key)
        if reason:
            self._disabled_compile_reasons[callable_key] = reason

    def clear_compiled_callable(self, *, callable_key: str) -> None:
        self._compiled_callables.pop(callable_key, None)
        self._disabled_compiled_callables.discard(callable_key)
        self._disabled_compile_reasons.pop(callable_key, None)

    def _compile_support(self) -> tuple[bool, str]:
        try:
            mlx_core = self._import_module("mlx.core")
        except ImportError:
            return False, "Installed MLX runtime does not expose `mlx.core` on this host."
        compile_fn = getattr(mlx_core, "compile", None)
        if not callable(compile_fn):
            return False, "Installed MLX runtime does not expose `mlx.core.compile`."
        return True, "LewLM can request `mlx.core.compile` graph-capture acceleration on the active MLX runtime."

    def _attention_support(self, callables: tuple[Any | None, ...]) -> _AttentionSupport:
        for callable_obj in callables:
            if callable_obj is None:
                continue
            parameter_names = _callable_parameter_names(callable_obj)
            parameter_name = _first_matching_parameter(parameter_names, _ATTENTION_PARAMETER_CANDIDATES)
            if parameter_name is None:
                continue
            if "sdpa" in parameter_name:
                return _AttentionSupport(
                    parameter_name=parameter_name,
                    supported_modes=("custom_sdpa",),
                    preferred_mode="custom_sdpa",
                )
            return _AttentionSupport(
                parameter_name=parameter_name,
                supported_modes=("flash_attention", "custom_sdpa"),
                preferred_mode="flash_attention",
            )
        return _AttentionSupport(parameter_name=None, supported_modes=(), preferred_mode=None)

    def _graph_compile_requested(self, request: GenerateRequest) -> bool:
        overrides = _request_acceleration_overrides(request)
        override_value = overrides.get("graph_compile_enabled")
        if isinstance(override_value, bool):
            return override_value
        return self.settings.mlx_graph_compile_enabled

    def _requested_attention_mode(self, request: GenerateRequest) -> str:
        overrides = _request_acceleration_overrides(request)
        override_value = overrides.get("attention_kernel_mode")
        if isinstance(override_value, str) and override_value in _VALID_ATTENTION_MODES:
            self._last_requested_kernel_mode = override_value
            return override_value
        self._last_requested_kernel_mode = self.settings.mlx_attention_kernel_mode
        return self.settings.mlx_attention_kernel_mode

    def _record_effective_path(self, *, kernel_path: str, graph_compile: bool) -> None:
        self._last_kernel_path = kernel_path
        if graph_compile:
            self._compiled_request_count += 1
        if kernel_path == "flash_attention":
            self._flash_attention_request_count += 1
            return
        if kernel_path == "custom_sdpa":
            self._custom_sdpa_request_count += 1
            return
        self._stock_request_count += 1

    @staticmethod
    def _attention_fallback_reason(
        *,
        requested_mode: str,
        attention_support: _AttentionSupport,
    ) -> str | None:
        if requested_mode == "stock":
            return None
        if not attention_support.supported:
            return "Installed MLX generation entrypoints do not advertise an accelerated attention-kernel hook."
        if requested_mode not in attention_support.supported_modes:
            supported_modes = ", ".join(attention_support.supported_modes)
            return f"Requested attention kernel `{requested_mode}` is unsupported; available modes: {supported_modes}."
        return None

    @staticmethod
    def _store_metadata(*, request: GenerateRequest, metadata: dict[str, Any], phase: str) -> None:
        existing = _request_acceleration_overrides(request)
        phase_details = _request_phase_details(existing)
        phase_details[phase] = {**phase_details.get(phase, {}), **metadata}
        request.metadata[_ACCELERATION_METADATA_KEY] = {
            **existing,
            **metadata,
            "phase_details": phase_details,
            "compile_state": _aggregate_compile_state(phase_details),
        }


def mlx_acceleration_measurements(*, request: GenerateRequest) -> dict[str, int]:
    payload = _request_acceleration_overrides(request)
    if not payload:
        return {}
    kernel_path = payload.get("effective_kernel_path")
    return {
        "graph_compile_requests": 1 if payload.get("effective_graph_compile") else 0,
        "graph_compile_fallbacks": 1 if payload.get("acceleration_fallback") and payload.get("requested_graph_compile") else 0,
        "stock_kernel_requests": 1 if kernel_path == "stock" else 0,
        "flash_attention_requests": 1 if kernel_path == "flash_attention" else 0,
        "custom_sdpa_requests": 1 if kernel_path == "custom_sdpa" else 0,
        "kernel_fallback_requests": 1 if payload.get("acceleration_fallback") and payload.get("requested_kernel_mode") != "stock" else 0,
    }


def _request_acceleration_overrides(request: GenerateRequest) -> dict[str, Any]:
    payload = request.metadata.get(_ACCELERATION_METADATA_KEY)
    return payload if isinstance(payload, dict) else {}


def _request_phase_details(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_phase_details = payload.get("phase_details")
    if not isinstance(raw_phase_details, dict):
        return {}
    return {
        str(phase): dict(details)
        for phase, details in raw_phase_details.items()
        if isinstance(phase, str) and isinstance(details, dict)
    }


def _aggregate_compile_state(phase_details: dict[str, dict[str, Any]]) -> str:
    ordered_phases = (
        phase_name
        for phase_name in ("prefill", "decode", "stream")
        if bool(phase_details.get(phase_name, {}).get("effective_graph_compile"))
    )
    compiled_phases = list(ordered_phases)
    return "+".join(compiled_phases) if compiled_phases else "stock"


def _callable_parameter_names(callable_obj: Any) -> set[str]:
    signature = _safe_signature(callable_obj)
    if signature is None:
        return set()
    return {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind is not inspect.Parameter.VAR_POSITIONAL
    }


def _first_matching_parameter(parameter_names: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in parameter_names:
            return candidate
    return None


def _safe_signature(callable_obj: Any) -> inspect.Signature | None:
    try:
        return inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None


def _compact_metrics(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
