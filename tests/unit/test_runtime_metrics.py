from __future__ import annotations

from lewlm.telemetry.runtime_metrics import RuntimeMetricsRecorder


def test_runtime_metrics_recorder_tracks_capability_measurements() -> None:
    recorder = RuntimeMetricsRecorder()

    recorder.record_success(
        model_id="embed-model",
        runtime="fake_mlx_semantic",
        capability="embeddings",
        load_seconds=0.1,
        execution_seconds=0.2,
        usage={"prompt_tokens": 8, "completion_tokens": 0},
        measurements={"input_count": 2, "vector_count": 2, "vector_dimensions": 384},
    )
    recorder.record_failure(
        model_id="embed-model",
        runtime="fake_mlx_semantic",
        capability="embeddings",
        load_seconds=0.05,
        execution_seconds=0.01,
        measurements={"input_count": 1},
    )

    snapshot = recorder.snapshot()
    capability_metrics = next(
        item for item in snapshot["capabilities"] if item["capability"] == "embeddings"
    )

    assert capability_metrics["request_count"] == 2
    assert capability_metrics["failure_count"] == 1
    assert capability_metrics["metric_totals"]["input_count"] == 3
    assert capability_metrics["metric_totals"]["vector_count"] == 2
    assert capability_metrics["metric_totals"]["vector_dimensions"] == 384
    assert capability_metrics["metric_averages"]["input_count"] == 1.5
