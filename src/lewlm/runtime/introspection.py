"""Shared backend introspection helpers for optional runtime packages."""

from __future__ import annotations

import inspect
from typing import Any

from lewlm.core.errors import NotImplementedLewLMError


def resolve_backend_callable(module: Any, names: tuple[str, ...], *, required: bool = True):
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    if required:
        raise NotImplementedLewLMError(
            "LewLM could not find a supported backend entrypoint in the installed runtime package.",
            details={"expected_any_of": list(names)},
        )
    return None


def invoke_with_signature(
    callable_obj: Any,
    provided_values: dict[str, Any],
    *,
    capability: str,
    passthrough_keys: tuple[str, ...] = (),
) -> Any:
    signature = inspect.signature(callable_obj)
    kwargs: dict[str, Any] = {}
    unsupported_required: list[str] = []
    accepts_var_keyword = False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_keyword = True
            continue
        if parameter.name in provided_values and provided_values[parameter.name] is not None:
            kwargs[parameter.name] = provided_values[parameter.name]
            continue
        if parameter.default is inspect.Signature.empty:
            unsupported_required.append(parameter.name)
    if unsupported_required:
        raise NotImplementedLewLMError(
            "LewLM does not yet understand the installed runtime callable signature.",
            details={"capability": capability, "unsupported_required_parameters": unsupported_required},
        )
    if accepts_var_keyword:
        for key in passthrough_keys:
            if key in kwargs:
                continue
            value = provided_values.get(key)
            if value is not None:
                kwargs[key] = value
    return callable_obj(**kwargs)
