from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from conftest import FakeLlamaCppRuntime, FakeMLXAudioRuntime, FakeMLXSemanticRuntime, FakeMLXVisionRuntime
from lewlm.api.app import create_app
from lewlm.cli.main import main
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import GenerateMessage, GenerateRequest, RuntimeAffinity
from lewlm.core.errors import RuntimeUnavailableError
from lewlm.runtime.experimental import ClusterEnrollWorkerResponse


class InMemoryClusterTransport:
    def __init__(self) -> None:
        self._clients: dict[str, TestClient] = {}
        self._fail_once: dict[str, str] = {}

    def register(self, base_url: str, client: TestClient) -> None:
        self._clients[base_url.rstrip("/")] = client

    def fail_next(self, base_url: str, message: str) -> None:
        self._fail_once[base_url.rstrip("/")] = message

    async def request_json(
        self,
        *,
        method: str,
        base_url: str,
        path: str,
        payload: dict[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        del timeout_seconds
        normalized = base_url.rstrip("/")
        if normalized in self._fail_once:
            message = self._fail_once.pop(normalized)
            raise RuntimeUnavailableError(message, details={"base_url": normalized})
        client = self._clients[normalized]
        response = client.request(method, path, json=payload, headers=dict(headers or {}))
        if response.status_code >= 400:
            raise RuntimeUnavailableError(
                "In-memory cluster request failed.",
                details={"status_code": response.status_code, "body": response.json()},
            )
        return response.json()


def _runtime_overrides() -> dict[RuntimeAffinity, object]:
    return {
        RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
        RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
        RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        RuntimeAffinity.MLX_VISION: FakeMLXVisionRuntime(),
    }


def _write_distributed_bundle(models_root: Path, *, required_workers: int = 2) -> Path:
    bundle = models_root / "distributed-proof-mlx"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "config.json").write_text(
        json.dumps({"model_type": "llama", "max_position_embeddings": 8192}),
        encoding="utf-8",
    )
    (bundle / "weights.safetensors").write_bytes(b"mlx-weights")
    (bundle / "tokenizer.json").write_text("{}", encoding="utf-8")
    (bundle / "distributed_pipeline.json").write_text(
        json.dumps(
            {
                "required_workers": required_workers,
                "layer_count": required_workers * 24,
                "stage_names": ["prefill", "decode", "finalize"][:required_workers],
                "batch_tokens": 256,
                "prefetch_ratio": 0.25,
                "overlap_ratio": 0.3,
                "queue_delay_ms": 1.0,
                "base_compute_ms_per_layer": 1.4,
                "response_template": (
                    "Distributed proof for {model_id} via {worker_trace}. "
                    "Prompt={prompt} layers={layer_trace} stages={stage_count}"
                ),
            },
        ),
        encoding="utf-8",
    )
    return bundle


def _settings_for_node(temp_settings, tmp_path: Path, name: str, *, role: str, models_root: Path | None = None):
    return temp_settings.with_updates(
        data_dir=tmp_path / name,
        models_dir=(models_root,) if models_root is not None else (),
        cluster_role=role,
        cluster_name="integration-cluster",
        cluster_node_name=name,
        cluster_public_base_url=f"http://{name}",
        cluster_enrollment_secret=SecretStr("cluster-secret") if role == "coordinator" else None,
    )


def _enroll_worker(
    coordinator_client: TestClient,
    worker_services,
    *,
    worker_name: str,
    endpoint: str,
    metadata: dict[str, object] | None = None,
) -> None:
    token = coordinator_client.post(
        "/v1/cluster/tokens",
        json={"worker_name": worker_name, "capabilities": ["chat"]},
    ).json()["token"]
    enrollment_payload = coordinator_client.post(
        "/v1/cluster/workers/enroll",
        json={
            "token": token,
            "worker_name": worker_name,
            "endpoint": endpoint,
            "capabilities": ["chat"],
            "metadata": metadata or {},
        },
    ).json()
    worker_services.cluster_service.complete_worker_enrollment(ClusterEnrollWorkerResponse.model_validate(enrollment_payload))


def test_cluster_routes_and_distributed_benchmark(temp_settings, tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    _write_distributed_bundle(models_root)
    coordinator_settings = _settings_for_node(temp_settings, tmp_path, "coordinator", role="coordinator", models_root=models_root)
    worker_a_settings = _settings_for_node(temp_settings, tmp_path, "worker-a", role="worker")
    worker_b_settings = _settings_for_node(temp_settings, tmp_path, "worker-b", role="worker")

    coordinator_services = bootstrap_services(coordinator_settings, runtime_overrides=_runtime_overrides())
    worker_a_services = bootstrap_services(worker_a_settings, runtime_overrides=_runtime_overrides())
    worker_b_services = bootstrap_services(worker_b_settings, runtime_overrides=_runtime_overrides())

    coordinator_app = create_app(coordinator_settings, services=coordinator_services)
    worker_a_app = create_app(worker_a_settings, services=worker_a_services)
    worker_b_app = create_app(worker_b_settings, services=worker_b_services)
    transport = InMemoryClusterTransport()
    try:
        with (
            TestClient(coordinator_app) as coordinator_client,
            TestClient(worker_a_app) as worker_a_client,
            TestClient(worker_b_app) as worker_b_client,
        ):
            transport.register("http://coordinator", coordinator_client)
            transport.register("http://worker-a", worker_a_client)
            transport.register("http://worker-b", worker_b_client)
            coordinator_services.cluster_service.set_transport(transport)

            manifests = coordinator_client.post("/v1/models/scan", json={}).json()["manifests"]
            model_id = next(manifest["model_id"] for manifest in manifests if manifest["display_name"] == "distributed-proof-mlx")

            _enroll_worker(
                coordinator_client,
                worker_a_services,
                worker_name="worker-a",
                endpoint="http://worker-a",
                metadata={
                    "relative_weight": 1.6,
                    "network_latency_ms": 1.0,
                    "network_bandwidth_gbps": 20.0,
                    "max_batch_tokens": 384,
                    "prefetch_tokens": 96,
                    "overlap_ratio": 0.4,
                },
            )
            _enroll_worker(
                coordinator_client,
                worker_b_services,
                worker_name="worker-b",
                endpoint="http://worker-b",
                metadata={
                    "relative_weight": 0.8,
                    "network_latency_ms": 6.0,
                    "network_bandwidth_gbps": 8.0,
                    "max_batch_tokens": 192,
                    "prefetch_tokens": 48,
                    "overlap_ratio": 0.2,
                },
            )

            status_payload = coordinator_client.get("/v1/cluster/status").json()
            assert status_payload["ready_worker_count"] == 2
            assert len(status_payload["workers"]) == 2

            plan_payload = coordinator_client.post("/v1/cluster/plans", json={"model_id": model_id}).json()
            assert plan_payload["stage_count"] == 2
            assert len(plan_payload["assignments"]) == 2
            assert plan_payload["assignments"][0]["worker_name"] == "worker-a"
            assert plan_payload["assignments"][0]["end_layer"] - plan_payload["assignments"][0]["start_layer"] > (
                plan_payload["assignments"][1]["end_layer"] - plan_payload["assignments"][1]["start_layer"]
            )
            assert plan_payload["scheduling"]["selection_mode"] == "heterogeneous_weighted_latency_aware"
            assert plan_payload["scheduling"]["effective_batch_tokens"] >= 224
            assert plan_payload["worker_profiles"][0]["relative_weight"] > plan_payload["worker_profiles"][1]["relative_weight"]

            benchmark = asyncio.run(
                coordinator_services.telemetry_service.benchmark(
                    model_id=model_id,
                    prompt="hello distributed cluster",
                ),
            )
            assert benchmark.runtime == "distributed_experimental"
            assert benchmark.usage["distributed_stage_count"] == 2
            assert benchmark.usage["distributed_worker_count"] == 2
            feature_map = {item.feature.value: item for item in benchmark.performance_features}
            assert feature_map["distributed_pipeline"].supported is True
            scenario = next(item for item in benchmark.scenarios if item.scenario == "distributed_pipeline_scaling")
            assert scenario.status == "observed"
            assert scenario.metrics["throughput_tokens_per_second"] > 0
            assert scenario.metrics["average_stage_utilization"] > 0
            assert scenario.metrics["effective_batch_tokens"] >= 224
            assert scenario.metrics["heterogeneity_ratio"] >= 2.0
            assert len(scenario.samples) == 2
            assert scenario.samples[0].metrics["target_batch_tokens"] > scenario.samples[1].metrics["target_batch_tokens"]
            assert scenario.samples[0].metrics["prefetch_tokens"] > scenario.samples[1].metrics["prefetch_tokens"]
            assert scenario.samples[0].metrics["layer_span"] > scenario.samples[1].metrics["layer_span"]

            runtime_stats = coordinator_client.get("/v1/runtime/stats").json()
            assert runtime_stats["cluster"]["ready_worker_count"] == 2
            assert runtime_stats["cluster"]["latest_execution_metrics"]["throughput_tokens_per_second"] > 0
            assert runtime_stats["cluster"]["latest_execution_metrics"]["bottleneck"] in {
                "balanced",
                "model_execution",
                "network",
                "scheduling",
            }
            runtime_features = {item["feature"]: item for item in runtime_stats["performance_features"]}
            assert runtime_features["distributed_pipeline"]["supported"] is True
            assert runtime_features["distributed_pipeline"]["metrics"]["ready_worker_count"] == 2
            assert runtime_features["distributed_pipeline"]["metrics"]["throughput_tokens_per_second"] > 0

            cluster_stats = coordinator_client.get("/v1/cluster/stats").json()
            assert cluster_stats["plan_count"] >= 1
            assert cluster_stats["latest_execution_metrics"]["critical_path_seconds"] > 0
    finally:
        coordinator_services.close()
        worker_a_services.close()
        worker_b_services.close()


def test_distributed_runtime_recovers_from_worker_failure(temp_settings, tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    _write_distributed_bundle(models_root)
    coordinator_settings = _settings_for_node(temp_settings, tmp_path, "coordinator", role="coordinator", models_root=models_root)
    worker_a_settings = _settings_for_node(temp_settings, tmp_path, "worker-a", role="worker")
    worker_b_settings = _settings_for_node(temp_settings, tmp_path, "worker-b", role="worker")
    worker_c_settings = _settings_for_node(temp_settings, tmp_path, "worker-c", role="worker")

    coordinator_services = bootstrap_services(coordinator_settings, runtime_overrides=_runtime_overrides())
    worker_a_services = bootstrap_services(worker_a_settings, runtime_overrides=_runtime_overrides())
    worker_b_services = bootstrap_services(worker_b_settings, runtime_overrides=_runtime_overrides())
    worker_c_services = bootstrap_services(worker_c_settings, runtime_overrides=_runtime_overrides())

    coordinator_app = create_app(coordinator_settings, services=coordinator_services)
    worker_a_app = create_app(worker_a_settings, services=worker_a_services)
    worker_b_app = create_app(worker_b_settings, services=worker_b_services)
    worker_c_app = create_app(worker_c_settings, services=worker_c_services)
    transport = InMemoryClusterTransport()
    try:
        with (
            TestClient(coordinator_app) as coordinator_client,
            TestClient(worker_a_app) as worker_a_client,
            TestClient(worker_b_app) as worker_b_client,
            TestClient(worker_c_app) as worker_c_client,
        ):
            transport.register("http://coordinator", coordinator_client)
            transport.register("http://worker-a", worker_a_client)
            transport.register("http://worker-b", worker_b_client)
            transport.register("http://worker-c", worker_c_client)
            coordinator_services.cluster_service.set_transport(transport)

            manifests = coordinator_client.post("/v1/models/scan", json={}).json()["manifests"]
            model_id = next(manifest["model_id"] for manifest in manifests if manifest["display_name"] == "distributed-proof-mlx")

            _enroll_worker(coordinator_client, worker_a_services, worker_name="worker-a", endpoint="http://worker-a")
            _enroll_worker(coordinator_client, worker_b_services, worker_name="worker-b", endpoint="http://worker-b")
            _enroll_worker(coordinator_client, worker_c_services, worker_name="worker-c", endpoint="http://worker-c")

            transport.fail_next("http://worker-b", "synthetic worker-b failure")
            manifest = coordinator_services.model_registry.get_manifest(model_id)
            response = asyncio.run(
                coordinator_services.cluster_service.generate(
                    manifest,
                    GenerateRequest(
                        model_id=model_id,
                        messages=[GenerateMessage(role="user", content="recover after failure")],
                        max_tokens=32,
                        temperature=0.0,
                    ),
                ),
            )

            assert response.usage["distributed_recovery_count"] == 1
            plan = coordinator_services.cluster_service.plan_for_model(model_id)
            assert plan is not None
            assert plan.recovery_count == 1
            assert any(assignment.worker_name == "worker-c" for assignment in plan.assignments)
    finally:
        coordinator_services.close()
        worker_a_services.close()
        worker_b_services.close()
        worker_c_services.close()


def test_cli_cluster_benchmark_prints_scaling_breakdown(temp_settings, tmp_path: Path, capsys) -> None:
    models_root = tmp_path / "models"
    _write_distributed_bundle(models_root)
    coordinator_settings = _settings_for_node(temp_settings, tmp_path, "coordinator", role="coordinator", models_root=models_root)
    worker_a_settings = _settings_for_node(temp_settings, tmp_path, "worker-a", role="worker")
    worker_b_settings = _settings_for_node(temp_settings, tmp_path, "worker-b", role="worker")

    coordinator_services = bootstrap_services(coordinator_settings, runtime_overrides=_runtime_overrides())
    worker_a_services = bootstrap_services(worker_a_settings, runtime_overrides=_runtime_overrides())
    worker_b_services = bootstrap_services(worker_b_settings, runtime_overrides=_runtime_overrides())

    coordinator_app = create_app(coordinator_settings, services=coordinator_services)
    worker_a_app = create_app(worker_a_settings, services=worker_a_services)
    worker_b_app = create_app(worker_b_settings, services=worker_b_services)
    transport = InMemoryClusterTransport()
    try:
        with (
            TestClient(coordinator_app) as coordinator_client,
            TestClient(worker_a_app) as worker_a_client,
            TestClient(worker_b_app) as worker_b_client,
        ):
            transport.register("http://coordinator", coordinator_client)
            transport.register("http://worker-a", worker_a_client)
            transport.register("http://worker-b", worker_b_client)
            coordinator_services.cluster_service.set_transport(transport)

            manifests = coordinator_client.post("/v1/models/scan", json={}).json()["manifests"]
            model_id = next(manifest["model_id"] for manifest in manifests if manifest["display_name"] == "distributed-proof-mlx")

            _enroll_worker(
                coordinator_client,
                worker_a_services,
                worker_name="worker-a",
                endpoint="http://worker-a",
                metadata={"relative_weight": 1.4, "max_batch_tokens": 384, "prefetch_tokens": 96},
            )
            _enroll_worker(
                coordinator_client,
                worker_b_services,
                worker_name="worker-b",
                endpoint="http://worker-b",
                metadata={"relative_weight": 0.8, "network_latency_ms": 5.0, "prefetch_tokens": 48},
            )

            exit_code = main(
                ["cluster", "benchmark", "--model", model_id, "--prompt", "cli cluster benchmark"],
                settings=coordinator_settings,
                services=coordinator_services,
            )
            output = capsys.readouterr().out

            assert exit_code == 0
            assert "distributed: stages=2, workers=2" in output
            assert "distributed_pipeline_scaling:" in output
            assert "scaling:" in output
            assert "breakdown:" in output
            assert "stages:" in output
            assert "batch/prefetch" in output
    finally:
        coordinator_services.close()
        worker_a_services.close()
        worker_b_services.close()
