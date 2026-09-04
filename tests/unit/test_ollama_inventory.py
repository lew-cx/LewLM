from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pytest

import lewlm.registry.ollama_inventory as ollama_inventory
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import ModelFormat, ModelModality, RuntimeAffinity
from lewlm.registry.ollama_inventory import (
    HOST_LOCAL,
    OFF_HOST,
    classify_execution_locality,
    discover_ollama_models,
    is_ollama_source,
    ollama_source_path,
)

# Shapes taken from a live `GET /api/tags` on ollama 0.33.2.
_LLAMA_RECORD = {
    "name": "llama3.1:latest",
    "model": "llama3.1:latest",
    "size": 4_920_753_328,
    "digest": "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e",
    "details": {
        "format": "gguf",
        "family": "llama",
        "parameter_size": "8.0B",
        "quantization_level": "Q4_K_M",
        "context_length": 131_072,
        "embedding_length": 4096,
    },
    "capabilities": ["completion", "tools"],
}
_EMBED_RECORD = {
    "name": "nomic-embed-text:latest",
    "model": "nomic-embed-text:latest",
    "size": 274_302_450,
    "digest": "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
    "details": {
        "format": "gguf",
        "family": "nomic-bert",
        "quantization_level": "F16",
        "context_length": 2048,
    },
    "capabilities": ["embedding"],
}
_CLOUD_RECORD = {
    "name": "gpt-oss:120b-cloud",
    "model": "gpt-oss:120b-cloud",
    "size": 0,
    "digest": "cloud1234abcd",
    "details": {"format": "gguf", "family": "gptoss"},
    "capabilities": ["completion", "tools"],
    "remote_host": "https://ollama.com",
    "remote_model": "gpt-oss:120b",
}


def _settings(tmp_path: Path, **overrides) -> LewLMSettings:
    base = {
        "data_dir": tmp_path / "state",
        "models_dir": (tmp_path / "models",),
        "external_accelerator_enabled": True,
        "external_accelerator_base_url": "http://127.0.0.1:11434",
        "external_accelerator_profile": "ollama_local",
        "ollama_discovery_enabled": True,
    }
    base.update(overrides)
    return LewLMSettings(**base)


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc_info) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _serve(monkeypatch, records: list[dict]) -> None:
    payload = json.dumps({"models": records}).encode("utf-8")
    monkeypatch.setattr(
        ollama_inventory,
        "urlopen",
        lambda request, timeout: _FakeResponse(payload),
    )


def _refuse(monkeypatch) -> None:
    monkeypatch.setattr(
        ollama_inventory,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            URLError(ConnectionRefusedError(61, "Connection refused")),
        ),
    )


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"name": "llama3.1:latest"}, HOST_LOCAL),
        ({"name": "gpt-oss:120b-cloud"}, OFF_HOST),
        ({"name": "deepseek-v3.1:cloud"}, OFF_HOST),
        ({"name": "qwen:latest", "remote_host": "https://ollama.com"}, OFF_HOST),
        ({"name": "qwen:latest", "remote_model": "qwen:480b"}, OFF_HOST),
        # An empty structured field is not evidence of anything.
        ({"name": "qwen:latest", "remote_host": "  "}, HOST_LOCAL),
        # `cloud` inside a name is not the suffix convention.
        ({"name": "cloudy-llama:latest"}, HOST_LOCAL),
    ],
)
def test_execution_locality_classification(record: dict, expected: str) -> None:
    assert classify_execution_locality(record) == expected


def test_ollama_manifests_carry_what_routing_and_the_bridge_need(tmp_path: Path, monkeypatch) -> None:
    _serve(monkeypatch, [_LLAMA_RECORD, _EMBED_RECORD])

    result = discover_ollama_models(_settings(tmp_path))

    assert result.succeeded
    by_name = {manifest.display_name: manifest for manifest in result.manifests}
    assert set(by_name) == {"llama3.1:latest", "nomic-embed-text:latest"}

    llama = by_name["llama3.1:latest"]
    assert llama.source_path == "ollama://llama3.1:latest"
    assert is_ollama_source(llama.source_path)
    assert llama.format_type == ModelFormat.GGUF
    assert llama.context_length == 131_072
    assert llama.quantization == "Q4_K_M"
    assert llama.architecture_family == "llama"
    assert llama.fingerprint == _LLAMA_RECORD["digest"]
    # Only the bridge can serve this, so only the bridge is offered. Declaring a
    # packaged affinity would let a local engine win a manifest with no file.
    assert llama.runtime_affinity == (RuntimeAffinity.EXTERNAL_ACCELERATOR,)
    assert llama.metadata["external_adapter_model_id"] == "llama3.1:latest"
    assert llama.metadata["ollama_execution_locality"] == HOST_LOCAL

    # The bridge gates capabilities on manifest modality before probing, so an
    # embedding model labelled `text` would have a working route refused.
    assert by_name["nomic-embed-text:latest"].modality == (ModelModality.EMBEDDING,)
    assert llama.modality == (ModelModality.TEXT,)


def test_cloud_models_stay_out_of_the_registry_until_explicitly_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _serve(monkeypatch, [_LLAMA_RECORD, _CLOUD_RECORD])

    refused = discover_ollama_models(_settings(tmp_path))
    assert [manifest.display_name for manifest in refused.manifests] == ["llama3.1:latest"]
    assert refused.skipped_cloud == ["gpt-oss:120b-cloud"]

    allowed = discover_ollama_models(_settings(tmp_path, ollama_cloud_enabled=True))
    assert {manifest.display_name for manifest in allowed.manifests} == {
        "llama3.1:latest",
        "gpt-oss:120b-cloud",
    }
    assert allowed.skipped_cloud == []
    cloud = next(m for m in allowed.manifests if m.display_name == "gpt-oss:120b-cloud")
    assert cloud.metadata["ollama_execution_locality"] == OFF_HOST
    assert cloud.metadata["ollama_remote_host"] == "https://ollama.com"


def test_unreachable_daemon_degrades_instead_of_raising(tmp_path: Path, monkeypatch) -> None:
    _refuse(monkeypatch)

    result = discover_ollama_models(_settings(tmp_path))

    assert not result.succeeded
    assert result.manifests == []
    assert "Could not reach an Ollama daemon" in (result.error or "")
    # The message must point at the operator's own tooling, since LewLM does not
    # start, supervise, or install Ollama.
    assert "ollama serve" in (result.error or "")


def test_a_record_without_a_digest_still_gets_a_stable_identity(tmp_path: Path, monkeypatch) -> None:
    _serve(monkeypatch, [{"name": "custom:latest", "details": {"family": "llama"}}])

    first = discover_ollama_models(_settings(tmp_path)).manifests[0]
    second = discover_ollama_models(_settings(tmp_path)).manifests[0]

    assert first.fingerprint == second.fingerprint
    assert first.model_id == second.model_id


def test_discovery_never_contacts_the_daemon_while_the_flag_is_off(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The flag is the whole contract: off means LewLM does not look."""

    def _explode(*_args, **_kwargs):
        raise AssertionError("discovery contacted Ollama while ollama_discovery_enabled was false")

    monkeypatch.setattr(ollama_inventory, "urlopen", _explode)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    settings = _settings(tmp_path, ollama_discovery_enabled=False)

    summary = bootstrap_services(settings).model_registry.scan()

    assert summary.discovered_count == 0
    assert summary.notes == []


def test_scan_retires_ollama_models_when_the_flag_is_turned_off(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    _serve(monkeypatch, [_LLAMA_RECORD])

    registry = bootstrap_services(_settings(tmp_path)).model_registry
    assert registry.scan().discovered_count == 1

    off_registry = bootstrap_services(_settings(tmp_path, ollama_discovery_enabled=False)).model_registry
    summary = off_registry.scan()

    assert summary.removed_count == 1
    assert off_registry.list_manifests() == []


def test_an_unreachable_daemon_does_not_retire_registered_models(tmp_path: Path, monkeypatch) -> None:
    """A daemon LewLM does not manage being down is not evidence models are gone."""

    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    _serve(monkeypatch, [_LLAMA_RECORD, _EMBED_RECORD])
    registry = bootstrap_services(_settings(tmp_path)).model_registry
    assert registry.scan().discovered_count == 2

    _refuse(monkeypatch)
    summary = registry.scan()

    assert summary.removed_count == 0
    assert len(registry.list_manifests()) == 2
    assert any("could not read" in note for note in summary.notes)


def test_scan_reports_skipped_cloud_models_and_how_to_include_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    _serve(monkeypatch, [_LLAMA_RECORD, _CLOUD_RECORD])

    summary = bootstrap_services(_settings(tmp_path)).model_registry.scan()

    assert summary.discovered_count == 1
    assert any("gpt-oss:120b-cloud" in note for note in summary.notes)
    assert any("LEWLM_OLLAMA_CLOUD_ENABLED" in note for note in summary.notes)


def test_filesystem_scan_does_not_disturb_the_ollama_namespace(tmp_path: Path, monkeypatch) -> None:
    """`ollama://` sources sit outside every model root and must be reconciled apart."""

    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    _serve(monkeypatch, [_LLAMA_RECORD])
    registry = bootstrap_services(_settings(tmp_path)).model_registry
    registry.scan()

    summary = registry.scan(roots=[models])

    assert summary.removed_count == 0
    assert [m.source_path for m in registry.list_manifests()] == [ollama_source_path("llama3.1:latest")]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"external_accelerator_enabled": False}, "requires external_accelerator_enabled"),
        ({"ollama_base_url": "http://example.com:11434"}, "loopback-only local host"),
        ({"ollama_base_url": "ftp://127.0.0.1:11434"}, "must use http or https"),
        ({"ollama_discovery_timeout_seconds": 0}, "at least 1"),
    ],
)
def test_settings_refuse_an_unusable_ollama_configuration(
    tmp_path: Path,
    overrides: dict,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        _settings(tmp_path, **overrides)


def test_cloud_cannot_be_enabled_without_discovery(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires ollama_discovery_enabled"):
        LewLMSettings(
            data_dir=tmp_path / "state",
            ollama_cloud_enabled=True,
        )
