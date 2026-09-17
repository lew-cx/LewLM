"""Map caller sampling controls onto what a backend actually supports.

LewLM accepts one sampling vocabulary across every runtime, but backends differ
in which knobs they expose. Silently dropping an unsupported control is the
failure mode this module exists to prevent: a caller that asked for `seed` and
got no determinism has no way to discover that from the response.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lewlm.core.contracts import SamplingControls, SamplingControlReport

#: Control name -> backend parameter name, per runtime family. A control absent
#: from a family's map is reported as unsupported for that backend.
_LLAMACPP_PARAMETERS: dict[str, str] = {
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repeat_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
    "stop": "stop",
}

#: mlx_lm applies most controls through a constructed sampler rather than
#: keyword arguments, so the mapping is resolved by the runtime adapter.
_MLX_TEXT_PARAMETERS: dict[str, str] = {
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    "seed": "seed",
}

_ONNX_GENAI_PARAMETERS: dict[str, str] = {
    "top_p": "top_p",
    "top_k": "top_k",
    "repetition_penalty": "repetition_penalty",
}

#: An OpenAI-compatible bridge accepts the OpenAI sampling vocabulary.
_EXTERNAL_BRIDGE_PARAMETERS: dict[str, str] = {
    "top_p": "top_p",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
    "stop": "stop",
}

_EXTENDED_EXTERNAL_BRIDGE_PARAMETERS: dict[str, str] = {
    **_EXTERNAL_BRIDGE_PARAMETERS,
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
}

SUPPORTED_PARAMETERS: dict[str, dict[str, str]] = {
    "llamacpp": _LLAMACPP_PARAMETERS,
    "mlx_text": _MLX_TEXT_PARAMETERS,
    "onnx_genai": _ONNX_GENAI_PARAMETERS,
    "external_bridge": _EXTERNAL_BRIDGE_PARAMETERS,
    "external_bridge_extended": _EXTENDED_EXTERNAL_BRIDGE_PARAMETERS,
}


def resolve_sampling_controls(
    controls: SamplingControls | None,
    *,
    runtime_name: str,
    family: str,
    available_parameters: Mapping[str, Any] | set[str] | None = None,
) -> tuple[dict[str, Any], SamplingControlReport]:
    """Return backend keyword arguments plus a report of what was honored.

    `available_parameters`, when supplied, is the set of parameter names the
    installed backend build actually accepts. A control the family nominally
    supports but this build does not is still reported as unsupported, so the
    report reflects the running system rather than the documented API.
    """

    report = SamplingControlReport(runtime=runtime_name)
    if controls is None or controls.is_empty:
        return {}, report

    requested = controls.requested()
    report.requested = dict(requested)
    parameter_map = SUPPORTED_PARAMETERS.get(family, {})

    options: dict[str, Any] = {}
    unsupported: list[str] = []
    for name, value in requested.items():
        parameter = parameter_map.get(name)
        if parameter is None:
            unsupported.append(name)
            continue
        if available_parameters is not None and parameter not in available_parameters:
            unsupported.append(name)
            continue
        options[parameter] = value
        report.applied[name] = value

    report.unsupported = sorted(unsupported)
    # Determinism is only claimed when the seed genuinely reached the backend.
    report.deterministic = "seed" in report.applied
    return options, report


def attach_sampling_report(metadata: dict[str, Any], report: SamplingControlReport) -> None:
    """Record the report on a generate request's metadata for the API layer."""

    metadata["sampling_controls"] = report.model_dump(mode="json")
