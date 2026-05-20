from __future__ import annotations

from lewlm.core.contracts import build_portable_performance_core_evidence, runtime_performance_feature_report


def _evidence_map(features: dict[str, dict[str, object]]):
    return {
        item.family.value: item
        for item in build_portable_performance_core_evidence(
            performance_features=features,
            runtime_names=["runtime-under-test"],
        )
    }


def test_portable_performance_core_evidence_preserves_owned_backend_and_fallback_modes() -> None:
    evidence = _evidence_map(
        {
            "continuous_batching": runtime_performance_feature_report(ownership="lewlm_owned", reason="owned"),
            "paged_kv_cache": runtime_performance_feature_report(ownership="lewlm_owned", reason="owned"),
            "kv_cache_quantization": runtime_performance_feature_report(ownership="backend_native", reason="backend"),
            "prefix_cache": runtime_performance_feature_report(ownership="lewlm_owned", reason="owned"),
            "prefill_optimization": runtime_performance_feature_report(ownership="backend_native", reason="prefill"),
            "chunked_prefill": runtime_performance_feature_report(ownership="lewlm_owned", reason="chunked"),
            "speculative_decoding": runtime_performance_feature_report(ownership="lewlm_owned", reason="spec"),
            "constrained_decoding": runtime_performance_feature_report(ownership="partial", reason="fallback"),
            "graph_compilation": runtime_performance_feature_report(ownership="backend_native", reason="graph"),
            "attention_kernel_acceleration": runtime_performance_feature_report(
                ownership="backend_native",
                reason="kernel",
            ),
        },
    )

    assert evidence["continuous_batching"].mode.value == "lewlm_owned"
    assert evidence["tiered_kv"].mode.value == "lewlm_owned"
    assert evidence["prefix_reuse"].mode.value == "lewlm_owned"
    assert evidence["prefill_isolation"].mode.value == "fallback"
    assert evidence["speculation"].mode.value == "lewlm_owned"
    assert evidence["constrained_decoding"].mode.value == "fallback"
    assert evidence["kernel_acceleration"].mode.value == "backend_native"


def test_portable_performance_core_evidence_maps_partial_component_support_to_fallback() -> None:
    evidence = _evidence_map(
        {
            "continuous_batching": runtime_performance_feature_report(ownership="backend_native", reason="batched"),
            "prefix_cache": runtime_performance_feature_report(ownership="partial", reason="bridge cache"),
            "paged_kv_cache": runtime_performance_feature_report(ownership="partial", reason="bridge kv"),
            "prefill_optimization": runtime_performance_feature_report(ownership="backend_native", reason="prefill"),
            "speculative_decoding": runtime_performance_feature_report(ownership="unsupported", reason="none"),
            "constrained_decoding": runtime_performance_feature_report(ownership="partial", reason="fallback"),
        },
    )

    assert evidence["continuous_batching"].mode.value == "backend_native"
    assert evidence["prefix_reuse"].mode.value == "fallback"
    assert evidence["tiered_kv"].mode.value == "fallback"
    assert evidence["prefill_isolation"].mode.value == "fallback"
    assert evidence["speculation"].mode.value == "unsupported"
    assert evidence["constrained_decoding"].mode.value == "fallback"
