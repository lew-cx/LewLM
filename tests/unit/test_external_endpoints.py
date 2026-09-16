from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lewlm.config.endpoints import ExternalEndpoint
from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    ConversionStatus,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.registry import ollama_inventory
from lewlm.runtime.adapters import LocalOpenAICompatibleAdapterRuntime
from lewlm.runtime.catalog import RuntimeCatalog
from lewlm.runtime.residency import ModelResidencyManager
from lewlm.runtime.response_cache import RuntimeResponseCache


def _endpoint(endpoint_id: str, port: int, *, profile: str = "openai_compatible", api_key_env: str | None = None):
    return ExternalEndpoint(
        endpoint_id=endpoint_id,
        profile=profile,
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key_env=api_key_env,
    )


def _manifest(endpoint_id: str) -> ModelManifest:
    return ModelManifest(
        model_id="shared-upstream-name",
        display_name="Shared upstream name",
        architecture_family="llama",
        modality=(ModelModality.TEXT,),
        source_path=f"external://{endpoint_id}/shared-upstream-name",
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.EXTERNAL_ACCELERATOR,),
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint=f"fingerprint-{endpoint_id}",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="test"),
        metadata={"external_endpoint_id": endpoint_id, "external_adapter_model_id": "shared-upstream-name"},
    )


@pytest.mark.parametrize("base_url", [
    "https://example.com:8000", "http://user:password@127.0.0.1:8000",
    "http://127.0.0.1:8000/admin", "http://127.0.0.1:8000?v=1",
])
def test_external_endpoint_rejects_nonlocal_or_ambiguous_urls(base_url: str) -> None:
    with pytest.raises(ValidationError):
        ExternalEndpoint(endpoint_id="bad", base_url=base_url)


def test_named_endpoints_are_unique_and_cannot_mix_with_enabled_legacy_settings(tmp_path: Path) -> None:
    endpoint = _endpoint("fast", 8000)
    with pytest.raises(ValidationError, match="cannot be combined"):
        LewLMSettings(data_dir=tmp_path, external_accelerator_enabled=True, external_endpoints=(endpoint,))
    with pytest.raises(ValidationError, match="must be unique"):
        LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint, endpoint))


def test_legacy_settings_resolve_to_one_compatible_endpoint(tmp_path: Path) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path,
        external_accelerator_enabled=True,
        external_accelerator_profile="vllm_local",
        external_accelerator_base_url="http://127.0.0.1:8000",
    )
    endpoint = settings.resolved_external_endpoints()[0]
    runtime = LocalOpenAICompatibleAdapterRuntime(settings=settings)
    assert endpoint.endpoint_id == "legacy-default"
    assert endpoint.profile == "vllm_local"
    assert runtime.name == "local_external_adapter"


def test_endpoint_credentials_are_redacted_from_settings_snapshot(tmp_path: Path) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path,
        external_endpoints=(_endpoint("secured", 8000, api_key_env="SECRET_ENGINE_TOKEN"),),
    )
    snapshot = settings.redacted_snapshot()
    assert snapshot["external_endpoints"][0]["credential_configured"] is True
    assert "SECRET_ENGINE_TOKEN" not in str(snapshot)


def test_endpoint_bound_manifests_route_by_identity_even_with_the_same_model_name(tmp_path: Path, monkeypatch) -> None:
    endpoints = (_endpoint("first", 8001), _endpoint("second", 8002))
    settings = LewLMSettings(data_dir=tmp_path, external_endpoints=endpoints)
    runtimes = {
        endpoint.endpoint_id: LocalOpenAICompatibleAdapterRuntime(settings=settings, endpoint=endpoint)
        for endpoint in endpoints
    }
    for runtime in runtimes.values():
        monkeypatch.setattr(runtime, "_available_remote_models", lambda: ("shared-upstream-name",))
    catalog = RuntimeCatalog({}, endpoint_runtimes=runtimes)

    first = catalog.compatible_runtimes(_manifest("first"), capability=None)[0]
    second = catalog.compatible_runtimes(_manifest("second"), capability=None)[0]

    assert first == [runtimes["first"]]
    assert second == [runtimes["second"]]
    assert runtimes["first"].name != runtimes["second"].name
    assert ModelResidencyManager.key_for(runtimes["first"], _manifest("first")) != ModelResidencyManager.key_for(
        runtimes["second"], _manifest("second")
    )


def test_external_runtime_namespaces_isolate_response_cache_and_coalescing_keys(tmp_path: Path) -> None:
    settings = LewLMSettings(data_dir=tmp_path, external_endpoints=(_endpoint("one", 8001), _endpoint("two", 8002)))
    first = LocalOpenAICompatibleAdapterRuntime(settings=settings, endpoint=settings.external_endpoints[0])
    second = LocalOpenAICompatibleAdapterRuntime(settings=settings, endpoint=settings.external_endpoints[1])
    cache = RuntimeResponseCache(metadata_store=None)  # key construction has no storage side effect
    first_key = cache.for_runtime(first).embedding_cache_key(model_id="same", inputs=["hello"])
    second_key = cache.for_runtime(second).embedding_cache_key(model_id="same", inputs=["hello"])
    assert first_key != second_key


@pytest.mark.asyncio
async def test_lightweight_endpoint_health_uses_only_cached_evidence(tmp_path: Path, monkeypatch) -> None:
    endpoint = _endpoint("passive", 8000)
    runtime = LocalOpenAICompatibleAdapterRuntime(
        settings=LewLMSettings(data_dir=tmp_path, external_endpoints=(endpoint,)), endpoint=endpoint
    )
    monkeypatch.setattr(runtime, "_request_json", lambda *_args, **_kwargs: pytest.fail("health performed a network probe"))
    snapshot = (await RuntimeCatalog({}, endpoint_runtimes={"passive": runtime}).health_snapshot())[0]
    assert snapshot["endpoint"]["endpoint_id"] == "passive"
    assert snapshot["endpoint"]["inventory_state"] == "unknown"
    assert snapshot["endpoint"]["upstream_residency"] == "unknown"


def test_named_ollama_inventory_binds_models_to_the_matching_endpoint(tmp_path: Path, monkeypatch) -> None:
    endpoint = _endpoint("ollama", 11434, profile="ollama_local")
    settings = LewLMSettings(
        data_dir=tmp_path,
        external_endpoints=(endpoint,),
        ollama_discovery_enabled=True,
        ollama_base_url="http://127.0.0.1:11434",
    )
    monkeypatch.setattr(ollama_inventory, "_fetch_tags", lambda *_args, **_kwargs: [{
        "name": "tiny:latest", "model": "tiny:latest", "digest": "a" * 64,
        "details": {"format": "gguf", "family": "llama"}, "capabilities": ["completion"],
    }])
    manifest = ollama_inventory.discover_ollama_models(settings).manifests[0]
    assert manifest.metadata["external_endpoint_id"] == "ollama"
