from __future__ import annotations

import json

from fastapi.testclient import TestClient
import pytest

from lewlm.cli.main import main
from lewlm.core.contracts import CapabilityEvidenceState, CapabilityName, RuntimeProvider, RuntimeSupportPath
from lewlm.core.middleware import build_middleware_capabilities_report
from lewlm.core.middleware import _provider_from_runtime_name


def _first_gguf_model_id(scan_payload: dict[str, object]) -> str:
    manifests = scan_payload["manifests"]
    assert isinstance(manifests, list)
    return next(
        str(manifest["model_id"])
        for manifest in manifests
        if isinstance(manifest, dict) and manifest.get("format_type") == "gguf"
    )


def _first_model_with_modality(scan_payload: dict[str, object], modality: str) -> str:
    manifests = scan_payload["manifests"]
    assert isinstance(manifests, list)
    return next(
        str(manifest["model_id"])
        for manifest in manifests
        if isinstance(manifest, dict) and modality in manifest.get("modality", [])
    )


def test_middleware_capability_report_uses_evidence_vocabulary(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    scan_code = main(["models", "scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    assert scan_code == 0
    capsys.readouterr()

    report = build_middleware_capabilities_report(services_with_fake_runtime)
    evidence_by_capability = {
        item.capability.value if hasattr(item.capability, "value") else str(item.capability): item
        for item in report.capability_evidence
    }
    providers = {
        item.runtime_name: item
        for item in report.runtime_providers
    }

    assert report.discovered_model_count >= 1
    assert evidence_by_capability[CapabilityName.CHAT.value].state == CapabilityEvidenceState.DISCOVERED
    assert providers["fake_llamacpp"].provider == RuntimeProvider.LLAMACPP


def test_cli_runtime_probe_and_model_artifacts_emit_json(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    scan_code = main(["models", "scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    model_id = _first_gguf_model_id(scan_payload)
    assert scan_code == 0

    probe_code = main(
        ["runtime", "probe", "--model", model_id, "--capability", "chat", "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    probe_payload = json.loads(capsys.readouterr().out)

    assert probe_code == 0
    assert probe_payload["model_id"] == model_id
    assert probe_payload["mode"] == "routing"
    assert probe_payload["evidence"][0]["capability"] == "chat"
    assert probe_payload["evidence"][0]["state"] == "discovered"

    smoke_code = main(
        [
            "runtime",
            "probe",
            "--model",
            model_id,
            "--capability",
            "chat",
            "--mode",
            "generate",
            "--prompt",
            "Probe this model",
            "--json",
        ],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    smoke_payload = json.loads(capsys.readouterr().out)

    assert smoke_code == 0
    assert smoke_payload["mode"] == "generate"
    assert smoke_payload["persisted"] is True
    assert smoke_payload["generated_text"] == "Echo: Probe this model"
    assert smoke_payload["evidence"][0]["state"] == CapabilityEvidenceState.GENERATE_PASSED.value

    artifacts_code = main(
        ["models", "artifacts", model_id, "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    artifacts_payload = json.loads(capsys.readouterr().out)

    assert artifacts_code == 0
    assert artifacts_payload["model_id"] == model_id
    assert artifacts_payload["runtime_probe_records"][0]["state"] == "generate_passed"
    assert artifacts_payload["capability_evidence"]
    chat_evidence = next(item for item in artifacts_payload["capability_evidence"] if item["capability"] == "chat")
    assert chat_evidence["state"] == "generate_passed"


def test_cli_convert_plan_emits_target_options_without_queueing(
    temp_settings,
    services_with_fake_runtime,
    capsys,
) -> None:
    scan_code = main(["models", "scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    model_id = _first_gguf_model_id(scan_payload)
    assert scan_code == 0

    plan_code = main(
        ["convert", model_id, "--plan", "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    plan_payload = json.loads(capsys.readouterr().out)

    assert plan_code == 0
    assert plan_payload["model_id"] == model_id
    assert plan_payload["default_target_id"] == "gguf_llamacpp"
    assert {target["target_id"] for target in plan_payload["targets"]} >= {"gguf_llamacpp", "onnx_genai"}


def test_cli_runtime_probe_keeps_runtime_success_when_persistence_fails(
    temp_settings,
    services_with_fake_runtime,
    capsys,
    monkeypatch,
) -> None:
    scan_code = main(["models", "scan", "--json"], settings=temp_settings, services=services_with_fake_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    model_id = _first_gguf_model_id(scan_payload)
    assert scan_code == 0

    def fail_persist(**_: object) -> str:
        raise RuntimeError("database locked")

    monkeypatch.setattr(services_with_fake_runtime.metadata_store, "upsert_runtime_probe_record", fail_persist)

    smoke_code = main(
        ["runtime", "probe", "--model", model_id, "--mode", "generate", "--prompt", "Still run", "--json"],
        settings=temp_settings,
        services=services_with_fake_runtime,
    )
    smoke_payload = json.loads(capsys.readouterr().out)

    assert smoke_code == 0
    assert smoke_payload["persisted"] is False
    assert smoke_payload["generated_text"] == "Echo: Still run"
    assert smoke_payload["evidence"][0]["state"] == "generate_passed"


def test_cli_runtime_load_probe_supports_non_chat_capabilities(
    temp_settings,
    services_with_fake_multimodal_runtime,
    capsys,
) -> None:
    scan_code = main(["models", "scan", "--json"], settings=temp_settings, services=services_with_fake_multimodal_runtime)
    scan_payload = json.loads(capsys.readouterr().out)
    model_id = _first_model_with_modality(scan_payload, "embedding")
    assert scan_code == 0

    load_code = main(
        ["runtime", "probe", "--model", model_id, "--capability", "embeddings", "--mode", "load", "--json"],
        settings=temp_settings,
        services=services_with_fake_multimodal_runtime,
    )
    load_payload = json.loads(capsys.readouterr().out)
    generate_code = main(
        ["runtime", "probe", "--model", model_id, "--capability", "embeddings", "--mode", "generate", "--json"],
        settings=temp_settings,
        services=services_with_fake_multimodal_runtime,
    )
    generate_payload = json.loads(capsys.readouterr().out)

    assert load_code == 0
    assert load_payload["persisted"] is True
    assert load_payload["evidence"][0]["state"] == "load_passed"
    assert generate_code != 0
    assert generate_payload["persisted"] is True
    assert generate_payload["evidence"][0]["state"] == "probe_failed"


def test_lewlm_api_namespace_exposes_capabilities_probes_and_artifacts(app_with_fake_runtime) -> None:
    with TestClient(app_with_fake_runtime) as client:
        scan_response = client.post("/v1/models/scan", json={})
        assert scan_response.status_code == 200
        model_id = _first_gguf_model_id(scan_response.json())

        capabilities_response = client.get("/v1/lewlm/capabilities")
        conversion_plan_response = client.post("/v1/lewlm/conversions/plan", json={"model_id": model_id})
        probe_response = client.post(
            "/v1/lewlm/probes",
            json={"model_id": model_id, "capability": "chat"},
        )
        smoke_probe_response = client.post(
            "/v1/lewlm/probes",
            json={"model_id": model_id, "mode": "load"},
        )
        artifacts_response = client.get(f"/v1/lewlm/models/{model_id}/artifacts")

    assert capabilities_response.status_code == 200
    capabilities_payload = capabilities_response.json()
    assert capabilities_payload["capability_evidence"]
    assert capabilities_payload["runtime_providers"]
    assert conversion_plan_response.status_code == 200
    assert conversion_plan_response.json()["default_target_id"] == "gguf_llamacpp"
    assert probe_response.status_code == 200
    assert probe_response.json()["mode"] == "routing"
    assert probe_response.json()["evidence"][0]["state"] == "discovered"
    assert smoke_probe_response.status_code == 200
    assert smoke_probe_response.json()["mode"] == "load"
    assert smoke_probe_response.json()["persisted"] is True
    assert smoke_probe_response.json()["evidence"][0]["state"] == "load_passed"
    assert artifacts_response.status_code == 200
    artifacts_payload = artifacts_response.json()
    assert artifacts_payload["model_id"] == model_id
    assert artifacts_payload["runtime_probe_records"][0]["state"] == "load_passed"
    chat_evidence = next(item for item in artifacts_payload["capability_evidence"] if item["capability"] == "chat")
    assert chat_evidence["state"] == "load_passed"


@pytest.mark.parametrize(
    ("profile", "provider"),
    [
        ("tensorrt_llm_server", RuntimeProvider.TENSORRT_LLM),
        ("openvino_model_server", RuntimeProvider.OPENVINO),
    ],
)
def test_middleware_provider_mapping_keeps_modern_bridges_explicit(profile: str, provider: RuntimeProvider) -> None:
    assert (
        _provider_from_runtime_name(
            "local_external_adapter",
            RuntimeSupportPath.BRIDGE,
            external_profile=profile,
        )
        == provider
    )
