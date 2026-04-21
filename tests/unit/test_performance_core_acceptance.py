from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_performance_core_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "performance_core_acceptance.py"
    spec = spec_from_file_location("lewlm_performance_core_acceptance", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_performance_core_acceptance_distinguishes_covered_and_missing_pillars() -> None:
    module = _load_performance_core_module()

    runtime_stats = {
        "platform": {"system": "Darwin", "machine": "arm64"},
        "serving_core": {
            "recent_sequence_count": 2,
            "total_sequences_started": 4,
            "total_sequences_completed": 4,
        },
        "request_scheduler": {"peak_active_requests": 2},
        "load_scheduler": {"peak_active_requests": 1},
        "performance_features": [
            {"feature": "serving_core", "supported": True, "runtime_names": ["fake_mlx_semantic"]},
            {"feature": "continuous_batching", "supported": True, "runtime_names": ["fake_mlx_semantic"]},
            {"feature": "prefix_cache", "supported": True, "runtime_names": ["fake_mlx_semantic"]},
            {"feature": "persistent_multi_context_cache", "supported": True, "runtime_names": ["fake_mlx_semantic"]},
            {"feature": "paged_kv_cache", "supported": True, "runtime_names": ["fake_mlx_semantic"]},
            {"feature": "speculative_decoding", "supported": True, "runtime_names": ["fake_mlx_semantic"]},
        ],
    }
    benchmark_artifacts = [
        {
            "artifact_id": "artifact-cache",
            "artifact_path": "/tmp/cache.json",
            "model_id": "demo-model",
            "runtime": "fake_mlx_semantic",
            "performance_features": [
                {"feature": "continuous_batching", "supported": True, "active": True},
                {"feature": "prefix_cache", "supported": True, "active": True},
                {"feature": "persistent_multi_context_cache", "supported": True, "active": True},
                {"feature": "paged_kv_cache", "supported": True, "active": True},
            ],
            "scenarios": [
                {"scenario": "continuous_batching", "status": "observed"},
                {"scenario": "repeated_prefix", "status": "observed"},
                {"scenario": "warm_chat_cache", "status": "observed"},
            ],
        },
        {
            "artifact_id": "artifact-speculation",
            "artifact_path": "/tmp/speculation.json",
            "model_id": "demo-model",
            "runtime": "fake_mlx_semantic",
            "performance_features": [
                {"feature": "speculative_decoding", "supported": True, "active": True},
            ],
            "scenarios": [
                {"scenario": "speculation_selection", "status": "observed"},
            ],
        },
    ]
    serving_profiles = [
        {
            "profile_id": "profile-cache",
            "model_id": "demo-model",
            "capability": "chat",
            "runtime": "fake_mlx_semantic",
            "workload_class": "text_only",
            "artifact": {
                "artifact_id": "artifact-cache",
                "artifact_path": "/tmp/cache.json",
            },
        },
        {
            "profile_id": "profile-speculation",
            "model_id": "demo-model",
            "capability": "chat",
            "runtime": "fake_mlx_semantic",
            "workload_class": "text_only",
            "selected_speculation_mode": "draft_model",
            "artifact": {
                "artifact_id": "artifact-speculation",
                "artifact_path": "/tmp/speculation.json",
            },
        },
    ]
    optimization_defaults = {
        "complete": True,
        "resolved_classes": [
            "runtime_selection",
            "continuous_batching",
            "tiered_kv_cache",
            "speculation",
        ],
        "benchmark_backed_classes": [
            "continuous_batching",
            "tiered_kv_cache",
            "speculation",
        ],
        "model_count": 1,
        "resolved_model_count": 1,
    }

    summary = module.build_performance_core_acceptance_summary(
        runtime_stats=runtime_stats,
        benchmark_artifacts=benchmark_artifacts,
        serving_profiles=serving_profiles,
        optimization_defaults=optimization_defaults,
        capability_reports=[],
    )

    assert summary["format"] == "lewlm-performance-core-acceptance-v1"
    assert summary["covered_pillars"] == [
        "continuous_batching",
        "measured_registry_defaults",
        "prefix_reuse",
        "serving_core",
        "speculation",
        "tiered_kv_cache",
    ]
    assert summary["missing_pillars"] == ["constrained_decoding"]
    assert summary["supported_unproved_pillars"] == []
    assert summary["complete"] is False
    assert summary["pillars"]["serving_core"]["status"] == "covered"
    assert summary["pillars"]["continuous_batching"]["benchmark_backed"] is True
    assert summary["pillars"]["prefix_reuse"]["artifact_ids"] == ["artifact-cache"]
    assert summary["pillars"]["speculation"]["profile_ids"] == ["profile-speculation"]
    assert summary["pillars"]["constrained_decoding"]["status"] == "missing"
    assert summary["pillars"]["constrained_decoding"]["fallback_guidance"]


def test_performance_core_acceptance_marks_constrained_decoding_supported_unproved_from_capability_reports() -> None:
    module = _load_performance_core_module()

    summary = module.build_performance_core_acceptance_summary(
        runtime_stats={
            "platform": {"system": "Darwin", "machine": "arm64"},
            "serving_core": {},
            "request_scheduler": {},
            "load_scheduler": {},
            "performance_features": [],
        },
        benchmark_artifacts=[],
        serving_profiles=[],
        optimization_defaults=None,
        capability_reports=[
            {
                "model_id": "demo-model",
                "measured_capabilities": [
                    {
                        "category": "constrained_decoding",
                        "status": "supported",
                        "runtime_names": ["fake_llamacpp"],
                    },
                ],
            },
        ],
    )

    assert summary["supported_unproved_pillars"] == ["constrained_decoding"]
    assert summary["pillars"]["constrained_decoding"]["status"] == "supported_unproved"
    assert summary["pillars"]["constrained_decoding"]["runtime_names"] == ["fake_llamacpp"]
    assert summary["pillars"]["constrained_decoding"]["metrics"]["measured_statuses"] == "supported"
