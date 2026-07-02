from __future__ import annotations

from types import SimpleNamespace

from lewlm.runtime.feature_probes import probe_backend_features


class _FakeLlama:
    def __init__(
        self,
        model_path: str,
        *,
        draft_model=None,
        type_k: int | None = None,
        type_v: int | None = None,
        offload_kqv: bool = True,
    ) -> None:
        raise NotImplementedError


class _FakeLegacyLlama:
    def __init__(self, model_path: str) -> None:
        raise NotImplementedError


def _module_map(monkeypatch, modules: dict[str, object]) -> None:
    def fake_import(name: str):
        if name in modules:
            return modules[name]
        raise ImportError(name)

    monkeypatch.setattr("lewlm.runtime.feature_probes.import_module", fake_import)


def _probe_map(probes) -> dict[tuple[str, str], object]:
    return {(probe.backend, probe.feature): probe for probe in probes}


def test_feature_probes_report_unprobed_backends_with_reasons(monkeypatch) -> None:
    _module_map(monkeypatch, {})

    probes = _probe_map(probe_backend_features())

    assert {
        ("llama_cpp", "ngram_draft_speculation"),
        ("llama_cpp", "kv_quantization_controls"),
        ("llama_cpp", "kv_offload_controls"),
        ("llama_cpp", "decode_time_grammar_enforcement"),
        ("mlx_lm", "draft_model_speculation"),
        ("mlx_lm", "batched_generation"),
        ("onnxruntime_genai", "generation_api"),
        ("onnxruntime_genai", "multimodal_processor"),
    } == set(probes)
    assert all(probe.present is None for probe in probes.values())
    assert all("not importable" in probe.detail for probe in probes.values())


def test_feature_probes_detect_modern_llamacpp_surfaces(monkeypatch) -> None:
    fake_llama_cpp = SimpleNamespace(
        Llama=_FakeLlama,
        LlamaGrammar=SimpleNamespace(from_string=lambda text: None, from_json_schema=lambda schema: None),
    )
    fake_speculative = SimpleNamespace(LlamaPromptLookupDecoding=lambda **kwargs: None)
    _module_map(
        monkeypatch,
        {"llama_cpp": fake_llama_cpp, "llama_cpp.llama_speculative": fake_speculative},
    )

    probes = _probe_map(probe_backend_features())

    assert probes[("llama_cpp", "ngram_draft_speculation")].present is True
    assert "acceptance-rate benchmarks still decide defaults" in probes[("llama_cpp", "ngram_draft_speculation")].detail
    assert probes[("llama_cpp", "kv_quantization_controls")].present is True
    assert probes[("llama_cpp", "kv_offload_controls")].present is True
    assert probes[("llama_cpp", "decode_time_grammar_enforcement")].present is True


def test_feature_probes_report_missing_llamacpp_surfaces_honestly(monkeypatch) -> None:
    fake_llama_cpp = SimpleNamespace(Llama=_FakeLegacyLlama)
    _module_map(monkeypatch, {"llama_cpp": fake_llama_cpp})

    probes = _probe_map(probe_backend_features())

    assert probes[("llama_cpp", "ngram_draft_speculation")].present is False
    assert "does not expose `LlamaPromptLookupDecoding`" in probes[("llama_cpp", "ngram_draft_speculation")].detail
    assert probes[("llama_cpp", "kv_quantization_controls")].present is False
    assert probes[("llama_cpp", "kv_offload_controls")].present is False
    assert probes[("llama_cpp", "decode_time_grammar_enforcement")].present is False


def test_feature_probes_detect_mlx_draft_and_batch_surfaces(monkeypatch) -> None:
    def stream_generate(model, tokenizer, prompt, *, draft_model=None):
        raise NotImplementedError

    fake_mlx_lm = SimpleNamespace(stream_generate=stream_generate, BatchGenerator=object())
    _module_map(monkeypatch, {"mlx_lm": fake_mlx_lm})

    probes = _probe_map(probe_backend_features())

    assert probes[("mlx_lm", "draft_model_speculation")].present is True
    assert "`draft_model`" in probes[("mlx_lm", "draft_model_speculation")].detail
    assert probes[("mlx_lm", "batched_generation")].present is True
    assert "BatchGenerator" in probes[("mlx_lm", "batched_generation")].detail


def test_feature_probes_detect_onnx_genai_surfaces(monkeypatch) -> None:
    fake_onnx = SimpleNamespace(Model=object(), Generator=object(), MultiModalProcessor=object())
    _module_map(monkeypatch, {"onnxruntime_genai": fake_onnx})

    probes = _probe_map(probe_backend_features())

    assert probes[("onnxruntime_genai", "generation_api")].present is True
    assert probes[("onnxruntime_genai", "multimodal_processor")].present is True


def test_feature_probes_report_incomplete_onnx_genai_api(monkeypatch) -> None:
    fake_onnx = SimpleNamespace(Model=object())
    _module_map(monkeypatch, {"onnxruntime_genai": fake_onnx})

    probes = _probe_map(probe_backend_features())

    assert probes[("onnxruntime_genai", "generation_api")].present is False
    assert probes[("onnxruntime_genai", "multimodal_processor")].present is False
