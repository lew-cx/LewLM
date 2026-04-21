from __future__ import annotations

import asyncio
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_release_manifest_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "generate_release_manifest.py"
    spec = spec_from_file_location("lewlm_release_manifest", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_manifest_includes_registered_model_reports(
    temp_settings,
    services_with_fake_runtime,
    monkeypatch,
) -> None:
    services_with_fake_runtime.model_registry.scan()
    monkeypatch.setenv("LEWLM_DATA_DIR", str(temp_settings.data_dir))

    module = _load_release_manifest_module()
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
    payload = module.build_release_manifest()

    assert payload["format"] == "lewlm-release-manifest-v1"
    assert payload["registered_model_count"] == 3
    assert payload["runtime_stats"]["platform"]["system"]
    assert payload["performance_core_acceptance"]["format"] == "lewlm-performance-core-acceptance-v1"
    assert payload["registered_models"]
    assert payload["registered_models"][0]["validation_key"]
    assert "target_platforms" in payload["registered_models"][0]
    assert "capabilities" in payload["registered_models"][0]
    assert payload["dependency_audit"]["format"] == "lewlm-dependency-audit-v1"
    assert payload["dependency_audit"]["consistency_check"]["status"] == "passed"


def test_release_manifest_includes_frontier_acceptance_evidence(
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
        services_with_fake_attachment_runtime.telemetry_service.autotune(
            model_id=text_model_id,
            prompt="Release frontier proof",
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

    assert payload["frontier_acceptance"]["format"] == "lewlm-frontier-acceptance-v1"
    assert payload["optimization_defaults"]["format"] == "lewlm-optimization-defaults-v1"
    assert payload["performance_core_acceptance"]["format"] == "lewlm-performance-core-acceptance-v1"
    assert payload["optimization_defaults"]["complete"] is True
    assert payload["optimization_defaults"]["model_count"] >= 1
    assert payload["frontier_acceptance"]["recommended_profile_count"] >= 1
    assert payload["frontier_acceptance"]["families"]["dense_text"]["status"] == "covered"
    assert payload["serving_profiles"][0]["profile_id"]


def test_release_manifest_includes_benchmark_backed_multimodal_default_selection(
    temp_settings,
    services_with_fake_attachment_runtime,
    monkeypatch,
) -> None:
    inventory = services_with_fake_attachment_runtime.model_registry.scan()
    vision_model_id = next(
        manifest.model_id
        for manifest in inventory.manifests
        if manifest.display_name == "qwen2-vl-vision-mlx"
    )
    for workload_class, prompt in (
        ("text_only_multimodal", "Release multimodal text defaults"),
        ("single_image", "Release multimodal image defaults"),
        ("repeated_image", "Release multimodal repeated image defaults"),
        ("frame_bundle_video", "Release multimodal bundle defaults"),
        ("audio_conditioned", "Release multimodal audio defaults"),
    ):
        asyncio.run(
            services_with_fake_attachment_runtime.telemetry_service.autotune(
                model_id=vision_model_id,
                prompt=prompt,
                workload_class=workload_class,
            )
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

    assert "multimodal_default_selection" in payload["optimization_defaults"]["optimization_classes"]
    assert "multimodal_default_selection" in payload["optimization_defaults"]["benchmark_backed_classes"]
    model_defaults = next(
        item
        for item in payload["optimization_defaults"]["models"]
        if item["model_id"] == vision_model_id
    )
    assert model_defaults["decisions"]["multimodal_default_selection"]["status"] == "adopted"
    assert model_defaults["decisions"]["multimodal_default_selection"]["benchmark_backed"] is True
