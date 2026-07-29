"""Import-cheap backend feature-presence probes for inventory reporting.

These probes answer one narrow question per feature: does the *installed*
backend expose the API surface at all. Presence is inventory evidence only;
capability claims still require runtime load/generate probes and benchmark
records. Every probe degrades to an explicit reason instead of a guess.
"""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any

from pydantic import BaseModel

_MLX_DRAFT_PARAMETERS = ("draft_model", "draft", "draft_client")


class BackendFeatureProbe(BaseModel):
    """Presence report for one backend inference feature surface.

    ``present`` is ``True``/``False`` when the installed backend could be
    inspected, and ``None`` when the backend itself is not importable on this
    host.
    """

    profile: str
    backend: str
    feature: str
    present: bool | None = None
    detail: str


def probe_backend_features(
    *,
    disabled_runtime_packs: tuple[str, ...] = (),
    enabled: bool = True,
) -> list[BackendFeatureProbe]:
    """Probe installed backends for the inference feature surfaces LewLM maps."""

    disabled = {name.casefold() for name in disabled_runtime_packs}
    if not enabled:
        disabled.update({"llamacpp", "mlx", "onnx_genai"})
    probes: list[BackendFeatureProbe] = []
    probes.extend(
        _disabled_probes(
            "gguf_fallback_backend",
            "llama_cpp",
            (
                "ngram_draft_speculation",
                "kv_quantization_controls",
                "kv_offload_controls",
                "decode_time_grammar_enforcement",
            ),
        )
        if "llamacpp" in disabled
        else _llamacpp_feature_probes()
    )
    probes.extend(
        _disabled_probes("mlx_local_backend", "mlx_lm", ("draft_model_speculation", "batched_generation"))
        if "mlx" in disabled
        else _mlx_feature_probes()
    )
    probes.extend(
        _disabled_probes("onnx_genai_backend", "onnxruntime_genai", ("generation_api", "multimodal_processor"))
        if "onnx_genai" in disabled
        else _onnx_genai_feature_probes()
    )
    return probes


def _disabled_probes(profile: str, backend: str, features: tuple[str, ...]) -> list[BackendFeatureProbe]:
    return [
        BackendFeatureProbe(
            profile=profile,
            backend=backend,
            feature=feature,
            present=None,
            detail="The matching runtime pack is disabled; LewLM did not import or probe this backend.",
        )
        for feature in features
    ]


def _not_importable(profile: str, backend: str, features: tuple[str, ...]) -> list[BackendFeatureProbe]:
    return [
        BackendFeatureProbe(
            profile=profile,
            backend=backend,
            feature=feature,
            present=None,
            detail=(
                f"`{backend}` is not importable on this host; "
                "feature presence stays unprobed until the matching extra is installed."
            ),
        )
        for feature in features
    ]


def _accepts_parameters(callable_obj: Any, parameter_names: tuple[str, ...]) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    parameters = set(signature.parameters)
    return all(name in parameters for name in parameter_names)


def _accepts_any_parameter(callable_obj: Any, parameter_names: tuple[str, ...]) -> str | None:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None
    parameters = set(signature.parameters)
    for name in parameter_names:
        if name in parameters:
            return name
    return None


def _llamacpp_feature_probes() -> list[BackendFeatureProbe]:
    profile = "gguf_fallback_backend"
    backend = "llama_cpp"
    features = (
        "ngram_draft_speculation",
        "kv_quantization_controls",
        "kv_offload_controls",
        "decode_time_grammar_enforcement",
    )
    try:
        llama_cpp = import_module("llama_cpp")
    except ImportError:
        return _not_importable(profile, backend, features)

    probes: list[BackendFeatureProbe] = []
    llama_class = getattr(llama_cpp, "Llama", None)
    llama_inspectable = callable(llama_class)

    prompt_lookup_present = False
    prompt_lookup_detail = "Installed llama-cpp-python does not expose `LlamaPromptLookupDecoding`."
    try:
        speculative_module = import_module("llama_cpp.llama_speculative")
    except ImportError:
        speculative_module = None
    if speculative_module is not None and callable(getattr(speculative_module, "LlamaPromptLookupDecoding", None)):
        if llama_inspectable and _accepts_parameters(llama_class, ("draft_model",)):
            prompt_lookup_present = True
            prompt_lookup_detail = (
                "`LlamaPromptLookupDecoding` and the `draft_model` load option are exposed; "
                "acceptance-rate benchmarks still decide defaults."
            )
        else:
            prompt_lookup_detail = (
                "`LlamaPromptLookupDecoding` is exposed, but the `Llama` client does not accept a `draft_model` load option."
            )
    probes.append(
        BackendFeatureProbe(
            profile=profile,
            backend=backend,
            feature="ngram_draft_speculation",
            present=prompt_lookup_present,
            detail=prompt_lookup_detail,
        ),
    )

    if llama_inspectable:
        kv_quant_present = _accepts_parameters(llama_class, ("type_k", "type_v"))
        kv_quant_detail = (
            "`Llama` accepts `type_k`/`type_v` KV cache-type controls."
            if kv_quant_present
            else "Installed `Llama` client does not accept `type_k`/`type_v` KV cache-type controls."
        )
        kv_offload_present = _accepts_parameters(llama_class, ("offload_kqv",))
        kv_offload_detail = (
            "`Llama` accepts the `offload_kqv` KV-offload control."
            if kv_offload_present
            else "Installed `Llama` client does not accept the `offload_kqv` KV-offload control."
        )
    else:
        kv_quant_present = False
        kv_quant_detail = "Installed llama.cpp bindings do not expose an inspectable `Llama` client."
        kv_offload_present = False
        kv_offload_detail = kv_quant_detail
    probes.append(
        BackendFeatureProbe(
            profile=profile,
            backend=backend,
            feature="kv_quantization_controls",
            present=kv_quant_present,
            detail=kv_quant_detail,
        ),
    )
    probes.append(
        BackendFeatureProbe(
            profile=profile,
            backend=backend,
            feature="kv_offload_controls",
            present=kv_offload_present,
            detail=kv_offload_detail,
        ),
    )

    grammar_class = getattr(llama_cpp, "LlamaGrammar", None)
    grammar_present = grammar_class is not None and callable(
        getattr(grammar_class, "from_string", None),
    ) and callable(getattr(grammar_class, "from_json_schema", None))
    probes.append(
        BackendFeatureProbe(
            profile=profile,
            backend=backend,
            feature="decode_time_grammar_enforcement",
            present=bool(grammar_present),
            detail=(
                "`LlamaGrammar.from_string` and `.from_json_schema` are exposed for decode-time enforcement."
                if grammar_present
                else "Installed llama.cpp bindings do not expose a complete `LlamaGrammar` surface for decode-time enforcement."
            ),
        ),
    )
    return probes


def _mlx_feature_probes() -> list[BackendFeatureProbe]:
    profile = "mlx_local_backend"
    backend = "mlx_lm"
    features = ("draft_model_speculation", "batched_generation")
    try:
        mlx_lm = import_module("mlx_lm")
    except ImportError:
        return _not_importable(profile, backend, features)

    probes: list[BackendFeatureProbe] = []
    stream_generate = getattr(mlx_lm, "stream_generate", None)
    draft_parameter = (
        _accepts_any_parameter(stream_generate, _MLX_DRAFT_PARAMETERS) if callable(stream_generate) else None
    )
    probes.append(
        BackendFeatureProbe(
            profile=profile,
            backend=backend,
            feature="draft_model_speculation",
            present=draft_parameter is not None,
            detail=(
                f"`mlx_lm.stream_generate` accepts the `{draft_parameter}` draft-model parameter."
                if draft_parameter is not None
                else "Installed mlx-lm does not expose a draft-model parameter on `stream_generate`."
            ),
        ),
    )
    batch_surface = next(
        (name for name in ("BatchGenerator", "batch_generate") if getattr(mlx_lm, name, None) is not None),
        None,
    )
    probes.append(
        BackendFeatureProbe(
            profile=profile,
            backend=backend,
            feature="batched_generation",
            present=batch_surface is not None,
            detail=(
                f"`mlx_lm.{batch_surface}` is exposed for batched generation."
                if batch_surface is not None
                else "Installed mlx-lm does not expose a batched-generation surface."
            ),
        ),
    )
    return probes


def _onnx_genai_feature_probes() -> list[BackendFeatureProbe]:
    profile = "onnx_genai_backend"
    backend = "onnxruntime_genai"
    features = ("generation_api", "multimodal_processor")
    try:
        onnx_genai = import_module("onnxruntime_genai")
    except ImportError:
        return _not_importable(profile, backend, features)

    probes: list[BackendFeatureProbe] = []
    generation_present = all(getattr(onnx_genai, name, None) is not None for name in ("Model", "Generator"))
    probes.append(
        BackendFeatureProbe(
            profile=profile,
            backend=backend,
            feature="generation_api",
            present=generation_present,
            detail=(
                "`Model` and `Generator` are exposed for prepared-bundle generation."
                if generation_present
                else "Installed onnxruntime-genai does not expose the expected `Model`/`Generator` API."
            ),
        ),
    )
    multimodal_surface = next(
        (name for name in ("MultiModalProcessor", "Images") if getattr(onnx_genai, name, None) is not None),
        None,
    )
    probes.append(
        BackendFeatureProbe(
            profile=profile,
            backend=backend,
            feature="multimodal_processor",
            present=multimodal_surface is not None,
            detail=(
                f"`onnxruntime_genai.{multimodal_surface}` is exposed for multimodal bundle inputs."
                if multimodal_surface is not None
                else "Installed onnxruntime-genai does not expose a multimodal processor surface."
            ),
        ),
    )
    return probes
