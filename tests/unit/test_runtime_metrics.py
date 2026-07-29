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


def test_runtime_metrics_recorder_bounds_and_attributes_application_usage() -> None:
    recorder = RuntimeMetricsRecorder(max_application_entries=2)

    recorder.record_application_request(application_id="rag-chat")
    recorder.record_lease_acquired(
        application_id="rag-chat",
        model_id="shared-model",
        capability="chat",
        residency_wait_seconds=0.25,
        contended=True,
    )
    recorder.record_lease_released(application_id="rag-chat")
    recorder.record_application_result(application_id="rag-chat", failed=False)
    recorder.record_application_request(application_id="document-generator")
    recorder.record_application_result(application_id="document-generator", failed=True)
    recorder.record_application_request(application_id="third-application")

    applications = {
        item["application_id"]: item
        for item in recorder.snapshot()["applications"]
    }

    assert set(applications) == {"rag-chat", "document-generator", "__other__"}
    assert applications["rag-chat"]["request_count"] == 1
    assert applications["rag-chat"]["success_count"] == 1
    assert applications["rag-chat"]["lease_acquisition_count"] == 1
    assert applications["rag-chat"]["active_lease_count"] == 0
    assert applications["rag-chat"]["load_contention_count"] == 1
    assert applications["rag-chat"]["model_usage_counts"] == {"shared-model": 1}
    assert applications["document-generator"]["failure_count"] == 1
    assert applications["__other__"]["request_count"] == 1
