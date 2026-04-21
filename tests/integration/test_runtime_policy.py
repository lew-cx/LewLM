from __future__ import annotations

import asyncio

import pytest

from conftest import FakeLlamaCppRuntime
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import CapabilityName, GenerateMessage, RuntimeAffinity


class TrackingFakeLlamaCppRuntime(FakeLlamaCppRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.warm_calls: list[str] = []
        self.unload_calls: list[str] = []

    async def warm_model(self, model_id: str) -> None:
        self.warm_calls.append(model_id)
        await super().warm_model(model_id)

    async def _unload_model(self, model_id: str) -> None:
        self.unload_calls.append(model_id)
        await super()._unload_model(model_id)


class SlowStreamingFakeLlamaCppRuntime(TrackingFakeLlamaCppRuntime):
    def supports_continuous_batching(self, capability) -> bool:
        return capability != CapabilityName.STREAMING and super().supports_continuous_batching(capability)

    async def _stream_generate(self, request):
        for chunk in ("slow ", "stream ", "reply"):
            assert self.is_model_loaded(request.model_id)
            yield chunk
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_keep_warm_policy_keeps_selected_model_loaded(temp_settings, sample_models_root) -> None:
    runtime = TrackingFakeLlamaCppRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(runtime_policy="keep_warm"),
        runtime_overrides={RuntimeAffinity.LLAMACPP: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        gguf_model_id = next(manifest.model_id for manifest in manifests if manifest.format_type.value == "gguf")

        await services.chat_orchestrator.complete(
            model_id=gguf_model_id,
            messages=[GenerateMessage(role="user", content="keep this warm")],
            max_tokens=64,
            temperature=0.0,
        )

        assert runtime.loaded_model_ids == (gguf_model_id,)
        assert runtime.warm_calls == [gguf_model_id]
        assert runtime.unload_calls == []
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_services_aclose_unloads_loaded_runtime_models(temp_settings, sample_models_root) -> None:
    runtime = TrackingFakeLlamaCppRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(runtime_policy="keep_warm"),
        runtime_overrides={RuntimeAffinity.LLAMACPP: runtime},
    )
    manifests = services.model_registry.scan().manifests
    gguf_model_id = next(manifest.model_id for manifest in manifests if manifest.format_type.value == "gguf")

    await services.model_router.warm_model(gguf_model_id)

    assert runtime.loaded_model_ids == (gguf_model_id,)

    await services.aclose()

    assert runtime.loaded_model_ids == ()
    assert runtime.unload_calls == [gguf_model_id]


@pytest.mark.asyncio
async def test_aggressive_unload_policy_releases_model_after_request(
    temp_settings,
    sample_models_root,
) -> None:
    runtime = TrackingFakeLlamaCppRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(runtime_policy="aggressive_unload"),
        runtime_overrides={RuntimeAffinity.LLAMACPP: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        gguf_model_id = next(manifest.model_id for manifest in manifests if manifest.format_type.value == "gguf")

        await services.chat_orchestrator.complete(
            model_id=gguf_model_id,
            messages=[GenerateMessage(role="user", content="unload after reply")],
            max_tokens=64,
            temperature=0.0,
        )

        assert runtime.loaded_model_ids == ()
        assert runtime.unload_calls == [gguf_model_id]
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_aggressive_unload_waits_for_overlapping_streams_before_unloading(
    temp_settings,
    sample_models_root,
) -> None:
    runtime = SlowStreamingFakeLlamaCppRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(runtime_policy="aggressive_unload"),
        runtime_overrides={RuntimeAffinity.LLAMACPP: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        gguf_model_id = next(manifest.model_id for manifest in manifests if manifest.format_type.value == "gguf")

        async def collect(prompt: str) -> str:
            session = await services.chat_orchestrator.stream(
                model_id=gguf_model_id,
                messages=[GenerateMessage(role="user", content=prompt)],
                max_tokens=32,
                temperature=0.0,
            )
            return "".join([delta async for delta in session.stream])

        first, second = await asyncio.gather(collect("stream one"), collect("stream two"))

        assert first == "slow stream reply"
        assert second == "slow stream reply"
        assert runtime.loaded_model_ids == ()
        assert runtime.unload_calls == [gguf_model_id]
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_balanced_policy_unloads_other_models_after_switching(
    temp_settings,
    sample_models_root,
) -> None:
    second_model_path = sample_models_root / "mistral-7b-instruct-q4_k_m.gguf"
    second_model_path.write_bytes(b"gguf-model-2")

    runtime = TrackingFakeLlamaCppRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(runtime_policy="balanced"),
        runtime_overrides={RuntimeAffinity.LLAMACPP: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        gguf_models = [manifest for manifest in manifests if manifest.format_type.value == "gguf"]
        first_model = next(manifest for manifest in gguf_models if "llama-3.2" in manifest.display_name)
        second_model = next(manifest for manifest in gguf_models if "mistral-7b" in manifest.display_name)

        await services.chat_orchestrator.complete(
            model_id=first_model.model_id,
            messages=[GenerateMessage(role="user", content="use the first model")],
            max_tokens=64,
            temperature=0.0,
        )
        assert set(runtime.loaded_model_ids) == {first_model.model_id}

        await services.chat_orchestrator.complete(
            model_id=second_model.model_id,
            messages=[GenerateMessage(role="user", content="switch models")],
            max_tokens=64,
            temperature=0.0,
        )

        assert runtime.loaded_model_ids == (second_model.model_id,)
        assert runtime.unload_calls == [first_model.model_id]
    finally:
        await services.aclose()


@pytest.mark.asyncio
async def test_runtime_stats_surface_residency_and_switch_telemetry_for_balanced_policy(
    temp_settings,
    sample_models_root,
) -> None:
    second_model_path = sample_models_root / "mistral-7b-instruct-q4_k_m.gguf"
    second_model_path.write_bytes(b"gguf-model-2")

    runtime = TrackingFakeLlamaCppRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(runtime_policy="balanced"),
        runtime_overrides={RuntimeAffinity.LLAMACPP: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        gguf_models = [manifest for manifest in manifests if manifest.format_type.value == "gguf"]
        first_model = next(manifest for manifest in gguf_models if "llama-3.2" in manifest.display_name)
        second_model = next(manifest for manifest in gguf_models if "mistral-7b" in manifest.display_name)

        await services.chat_orchestrator.complete(
            model_id=first_model.model_id,
            messages=[GenerateMessage(role="user", content="use the first model")],
            max_tokens=64,
            temperature=0.0,
        )
        await services.chat_orchestrator.complete(
            model_id=second_model.model_id,
            messages=[GenerateMessage(role="user", content="switch models")],
            max_tokens=64,
            temperature=0.0,
        )

        runtime_stats = await services.telemetry_service.runtime_stats()
        runtime_payload = next(item for item in runtime_stats.runtimes if item["name"] == runtime.name)

        assert runtime_payload["loaded_model_count"] == 1
        assert runtime_payload["estimated_loaded_memory_mb"] == second_model.estimated_memory_mb
        assert runtime_payload["peak_loaded_model_count"] == 2
        assert runtime_payload["peak_estimated_memory_mb"] >= second_model.estimated_memory_mb
        assert runtime_payload["total_load_count"] == 2
        assert runtime_payload["total_unload_count"] == 1
        assert runtime_payload["total_warm_count"] == 0
        assert runtime_payload["total_model_switch_count"] == 1
        assert runtime_payload["loaded_models"][0]["model_id"] == second_model.model_id
        assert runtime_payload["loaded_models"][0]["loaded_at"] is not None
        assert runtime_payload["loaded_models"][0]["last_used_at"] is not None
        assert runtime_payload["loaded_models"][0]["residency_seconds"] is not None
    finally:
        await services.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "expected_loaded_count", "expected_peak_loaded_count", "expected_load_count", "expected_unload_count", "expected_warm_count", "expected_switch_count"),
    [
        ("keep_warm", 3, 3, 3, 0, 6, 2),
        ("balanced", 1, 2, 6, 5, 0, 5),
        ("aggressive_unload", 0, 1, 6, 6, 0, 0),
    ],
)
async def test_benchmark_suite_stress_tracks_mixed_model_residency_by_policy(
    temp_settings,
    sample_models_root,
    policy: str,
    expected_loaded_count: int,
    expected_peak_loaded_count: int,
    expected_load_count: int,
    expected_unload_count: int,
    expected_warm_count: int,
    expected_switch_count: int,
) -> None:
    (sample_models_root / "mistral-7b-instruct-q4_k_m.gguf").write_bytes(b"gguf-model-2")
    (sample_models_root / "qwen2.5-7b-instruct-q4_k_m.gguf").write_bytes(b"gguf-model-3")

    runtime = TrackingFakeLlamaCppRuntime()
    services = bootstrap_services(
        temp_settings.with_updates(runtime_policy=policy),
        runtime_overrides={RuntimeAffinity.LLAMACPP: runtime},
    )
    try:
        manifests = services.model_registry.scan().manifests
        gguf_model_ids = [manifest.model_id for manifest in manifests if manifest.format_type.value == "gguf"]
        assert len(gguf_model_ids) == 3

        suite = await services.telemetry_service.benchmark_suite(
            prompt=f"benchmark residency stress {policy}",
            model_ids=gguf_model_ids,
            repeat_count=2,
        )
        runtime_stats = await services.telemetry_service.runtime_stats()
        runtime_payload = next(item for item in runtime_stats.runtimes if item["name"] == runtime.name)

        assert suite.repeat_count == 2
        assert suite.benchmark_count == 6
        assert suite.model_count == 3
        assert len(suite.results) == 6
        assert len(suite.models) == 3
        assert all(item.run_count == 2 for item in suite.models)
        assert runtime_payload["loaded_model_count"] == expected_loaded_count
        assert runtime_payload["peak_loaded_model_count"] == expected_peak_loaded_count
        if policy == "aggressive_unload":
            assert runtime_payload["total_load_count"] >= expected_load_count
            assert runtime_payload["total_unload_count"] == runtime_payload["total_load_count"]
        else:
            assert runtime_payload["total_load_count"] == expected_load_count
            assert runtime_payload["total_unload_count"] == expected_unload_count
        if policy == "keep_warm":
            assert runtime_payload["total_warm_count"] >= expected_warm_count
        else:
            assert runtime_payload["total_warm_count"] == expected_warm_count
        assert runtime_payload["total_model_switch_count"] == expected_switch_count
        assert len(runtime_payload["loaded_models"]) == expected_loaded_count
        if expected_loaded_count:
            assert all(item["loaded_at"] is not None for item in runtime_payload["loaded_models"])
            assert all(item["residency_seconds"] is not None for item in runtime_payload["loaded_models"])
        else:
            assert runtime_payload["loaded_models"] == []
    finally:
        await services.aclose()
