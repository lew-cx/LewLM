from __future__ import annotations

from lewlm.pack_registry import PackRegistry, PackStatus


def test_pack_registry_reports_cross_platform_pack_descriptions(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.pack_registry.find_spec", lambda module_name: None)

    registry = PackRegistry()
    gguf = registry.report("llamacpp")
    external = registry.report("external_accelerator")
    documents = registry.report("documents")

    assert gguf.status == PackStatus.MISSING_DEPENDENCY
    assert "Cross-platform GGUF runtime pack" in gguf.description
    assert external.status == PackStatus.ACTIVE
    assert "loopback-only OpenAI-compatible local servers" in external.description
    assert documents.status == PackStatus.ACTIVE
    assert "Additive documents pack" in documents.description
