from __future__ import annotations

import asyncio
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_release_manifest_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "generate_release_manifest.py"
    spec = spec_from_file_location("lewlm_release_manifest_performance_core", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_manifest_exposes_benchmark_artifacts_for_performance_core_acceptance(
    temp_settings,
    services_with_fake_attachment_runtime,
    monkeypatch,
) -> None:
    inventory = services_with_fake_attachment_runtime.model_registry.scan()
    text_model_id = next(
        manifest.model_id
        for manifest in inventory.manifests
        if manifest.display_name == "qwen2.5-1.5b-instruct-mlx"
    )
    asyncio.run(
        services_with_fake_attachment_runtime.telemetry_service.benchmark(
            model_id=text_model_id,
            prompt="Release performance-core evidence",
        ),
    )
    monkeypatch.setenv("LEWLM_DATA_DIR", str(temp_settings.data_dir))

    module = _load_release_manifest_module()
    monkeypatch.setattr(module, "bootstrap_services", lambda settings: services_with_fake_attachment_runtime)
    monkeypatch.setattr(
        module,
        "build_dependency_audit",
        lambda *, resolved_packages=None: {
            "format": "lewlm-dependency-audit-v1",
            "resolved_environment": {
                "package_count": len(resolved_packages or []),
                "package_digest": "digest",
            },
            "consistency_check": {
                "tool": "pip check",
                "status": "passed",
                "exit_code": 0,
                "issues": [],
            },
        },
    )
    monkeypatch.setattr(module, "build_sbom", lambda: {"format": "lewlm-sbom-v1"})

    payload = module.build_release_manifest()

    assert payload["benchmark_artifacts"]
    latest_artifact = payload["benchmark_artifacts"][0]
    assert latest_artifact["artifact_id"]
    assert latest_artifact["artifact_path"]
    assert payload["performance_core_acceptance"]["benchmark_artifact_count"] == len(payload["benchmark_artifacts"])
