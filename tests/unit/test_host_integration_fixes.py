"""Regression coverage for host-application integration gaps."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from lewlm.api.openapi import normalize_openapi_schema
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
from lewlm.core.errors import LewLMError, ModelLoadError
from lewlm.runtime.base import ManagedRuntime
from lewlm.runtime.llamacpp.runtime import LlamaCppRuntime


# --- optional settings can be unset from the environment ---------------------


@pytest.mark.parametrize("raw", ["", "null", "none", "None", "NULL", "  ", "~"])
def test_optional_int_setting_accepts_null_env_sentinels(monkeypatch, raw: str) -> None:
    monkeypatch.setenv("LEWLM_KV_CACHE_QUANTIZATION_BITS", raw)
    assert LewLMSettings().kv_cache_quantization_bits is None


def test_optional_setting_still_rejects_a_nonsense_value(monkeypatch) -> None:
    monkeypatch.setenv("LEWLM_KV_CACHE_QUANTIZATION_BITS", "banana")
    with pytest.raises(Exception):
        LewLMSettings()


def test_optional_int_setting_still_accepts_a_real_value(monkeypatch) -> None:
    monkeypatch.setenv("LEWLM_KV_CACHE_QUANTIZATION_BITS", "16")
    assert LewLMSettings().kv_cache_quantization_bits == 16


def test_null_sentinels_apply_to_other_optional_settings(monkeypatch) -> None:
    monkeypatch.setenv("LEWLM_GPU_OFFLOAD_LAYERS", "null")
    monkeypatch.setenv("LEWLM_KV_CACHE_MAX_PAGES", "")
    settings = LewLMSettings()
    assert settings.gpu_offload_layers is None
    assert settings.kv_cache_max_pages is None


def test_required_setting_is_unaffected_by_the_sentinel(monkeypatch) -> None:
    monkeypatch.setenv("LEWLM_KV_CACHE_PAGE_SIZE", "null")
    with pytest.raises(Exception):
        LewLMSettings()


# --- KV cache quantization pairs with flash attention ------------------------


class _FakeGGML:
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q4_0 = 2


def _kv_options(bits: int | None, parameters: set[str]) -> tuple[dict, dict]:
    runtime = LlamaCppRuntime(settings=LewLMSettings(kv_cache_quantization_bits=bits))
    return runtime._kv_cache_quantization_configuration(_FakeGGML(), parameters)


def test_kv_cache_quantization_is_off_by_default() -> None:
    assert LewLMSettings().kv_cache_quantization_bits is None
    options, payload = _kv_options(None, {"type_k", "type_v", "flash_attn"})
    assert options == {}
    assert payload["effective"] == "disabled"


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_kv_cache_enables_flash_attention(bits: int) -> None:
    options, payload = _kv_options(bits, {"type_k", "type_v", "flash_attn"})
    assert options["flash_attn"] is True
    assert options["type_k"] == options["type_v"]
    assert payload["effective"] == "enabled"
    assert "flash_attn" in payload["applied_parameters"]


def test_f16_kv_cache_does_not_require_flash_attention() -> None:
    options, payload = _kv_options(16, {"type_k", "type_v", "flash_attn"})
    assert "flash_attn" not in options
    assert payload["effective"] == "enabled"


def test_quantized_kv_cache_is_refused_without_flash_attention_support() -> None:
    # Emitting a quantized KV cache here would make llama.cpp reject the load,
    # so the control must be refused rather than silently applied.
    options, payload = _kv_options(8, {"type_k", "type_v"})
    assert options == {}
    assert payload["effective"] == "rejected"
    assert "flash attention" in payload["reason"]


# --- backend load failures reach callers as typed envelopes ------------------


class _BrokenRuntime(ManagedRuntime):
    name = "broken"
    affinity = RuntimeAffinity.MLX_TEXT
    supported_formats = (ModelFormat.MLX,)
    supported_modalities = (ModelModality.TEXT,)

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, None

    async def _load_model(self, manifest: ModelManifest) -> None:
        raise ValueError("Model type gemma4 not supported.")

    async def _unload_model(self, model_id: str) -> None:
        return None


def _manifest() -> ModelManifest:
    return ModelManifest(
        model_id="gemma4-preview",
        display_name="gemma4-preview",
        architecture_family="gemma4",
        modality=(ModelModality.TEXT,),
        source_path="/tmp/gemma4-preview",
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )


def test_raw_backend_load_failure_becomes_a_typed_error() -> None:
    with pytest.raises(ModelLoadError) as exc_info:
        asyncio.run(_BrokenRuntime().load_model(_manifest()))

    error = exc_info.value
    assert isinstance(error, LewLMError)
    assert error.code == "model_load_failed"
    assert int(error.status_code) == 503
    # A host application needs enough detail to explain the failure to a user.
    assert error.details["architecture_family"] == "gemma4"
    assert error.details["model_id"] == "gemma4-preview"
    assert error.details["cause_type"] == "ValueError"
    assert "gemma4" in error.details["cause"]


def test_backend_specific_load_errors_are_preserved() -> None:
    class _TypedFailureRuntime(_BrokenRuntime):
        async def _load_model(self, manifest: ModelManifest) -> None:
            raise ModelLoadError("Backend already classified this failure.", details={"origin": "backend"})

    with pytest.raises(ModelLoadError) as exc_info:
        asyncio.run(_TypedFailureRuntime().load_model(_manifest()))
    assert exc_info.value.details == {"origin": "backend"}


def test_cancellation_during_load_is_not_swallowed() -> None:
    class _CancellingRuntime(_BrokenRuntime):
        async def _load_model(self, manifest: ModelManifest) -> None:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_CancellingRuntime().load_model(_manifest()))


def test_load_failure_reaches_http_callers_as_an_error_envelope(app_with_failing_runtime) -> None:
    with TestClient(app_with_failing_runtime) as client:
        manifests = client.post("/v1/models/scan", json={}).json()["manifests"]
        model_id = manifests[0]["model_id"]
        response = client.post(
            "/v1/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()["error"]
    assert error["code"] == "model_load_failed"
    assert error["details"]["cause_type"] == "ValueError"


def test_streaming_load_failure_also_returns_an_envelope(app_with_failing_runtime) -> None:
    with TestClient(app_with_failing_runtime) as client:
        manifests = client.post("/v1/models/scan", json={}).json()["manifests"]
        model_id = manifests[0]["model_id"]
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_load_failed"


# --- the published OpenAPI document resolves as served -----------------------


def _collect_refs(node, found: list[str]) -> None:
    if isinstance(node, dict):
        reference = node.get("$ref")
        if isinstance(reference, str):
            found.append(reference)
        for value in node.values():
            _collect_refs(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, found)


def test_published_openapi_has_no_unresolvable_references(app_with_fake_runtime) -> None:
    schema = app_with_fake_runtime.openapi()
    components = schema["components"]["schemas"]

    refs: list[str] = []
    _collect_refs(schema, refs)
    assert refs, "expected the document to contain references"

    dangling = [
        reference
        for reference in refs
        if not (reference.startswith("#/components/schemas/") and reference.split("/")[-1] in components)
    ]
    assert dangling == []
    assert "$defs" not in json.dumps(schema)


def test_request_bodies_reference_shared_components(app_with_fake_runtime) -> None:
    schema = app_with_fake_runtime.openapi()
    for path in ("/v1/chat/completions", "/v1/responses", "/v1/retrieval/context"):
        body = schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert "$defs" not in body, path


def test_normalizer_keeps_conflicting_definitions_distinct() -> None:
    schema = {
        "components": {"schemas": {"Thing": {"type": "string"}}},
        "paths": {
            "/x": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/$defs/Thing",
                                    "$defs": {"Thing": {"type": "integer"}},
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    normalized = normalize_openapi_schema(schema)
    components = normalized["components"]["schemas"]
    # The pre-existing definition must not be overwritten by the inlined one.
    assert components["Thing"] == {"type": "string"}
    assert components["Thing_2"] == {"type": "integer"}
    body = normalized["paths"]["/x"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert body["$ref"] == "#/components/schemas/Thing_2"


def test_normalizer_rewrites_discriminator_mappings() -> None:
    schema = {
        "paths": {
            "/x": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "oneOf": [{"$ref": "#/$defs/Grammar"}],
                                    "discriminator": {
                                        "propertyName": "type",
                                        "mapping": {"grammar": "#/$defs/Grammar"},
                                    },
                                    "$defs": {"Grammar": {"type": "object"}},
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    normalized = normalize_openapi_schema(schema)
    body = normalized["paths"]["/x"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    # A discriminator mapping is a reference too, and must resolve as published.
    assert body["discriminator"]["mapping"] == {"grammar": "#/components/schemas/Grammar"}
    assert body["oneOf"][0]["$ref"] == "#/components/schemas/Grammar"


def test_normalizer_reuses_an_identical_existing_definition() -> None:
    schema = {
        "components": {"schemas": {"Thing": {"type": "string"}}},
        "paths": {
            "/x": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/$defs/Thing",
                                        "$defs": {"Thing": {"type": "string"}},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    normalized = normalize_openapi_schema(schema)
    assert set(normalized["components"]["schemas"]) == {"Thing"}
    body = normalized["paths"]["/x"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert body["$ref"] == "#/components/schemas/Thing"
