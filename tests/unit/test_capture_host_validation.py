from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load_capture_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "capture_host_validation.py"
    spec = spec_from_file_location("lewlm_capture_host_validation", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_host_validation_captures_core_commands_and_release_bundle(tmp_path: Path) -> None:
    module = _load_capture_module()
    output_dir = tmp_path / "host-validation"

    def fake_runner(command: list[str], *, cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
        assert str((cwd / "src").resolve(strict=False)) in env.get("PYTHONPATH", "")
        if command[1:3] == ["-m", "lewlm"]:
            match command[3]:
                case "config":
                    return 0, json.dumps({"app_name": "LewLM", "validation_manifest_paths": []}), ""
                case "scan":
                    return 0, json.dumps(
                        {
                            "roots_scanned": ["X:\\models"],
                            "discovered_count": 1,
                            "new_count": 1,
                            "updated_count": 0,
                            "unchanged_count": 0,
                            "removed_count": 0,
                        },
                    ), ""
                case "list-models":
                    return 0, json.dumps({"count": 1, "items": [{"model_id": "gguf-model"}]}), ""
                case "doctor":
                    return 0, json.dumps(
                        {
                            "runtime_stats": {
                                "target_platforms": [{"system": "Windows", "machine": "AMD64"}],
                            },
                        },
                    ), ""
                case "capabilities":
                    assert command[4] == "gguf-model"
                    return 0, json.dumps(
                        {
                            "model_id": "gguf-model",
                            "capabilities": [
                                {"capability": "chat", "supported": True},
                                {"capability": "semantic", "supported": True},
                            ],
                        },
                    ), ""
        if command[1].endswith("capture_release_bundle.py"):
            return 0, json.dumps(
                {
                    "format": "lewlm-release-bundle-v1",
                    "validation": {"overall_status": "passed", "checks": {}},
                    "index_path": str(output_dir / "release-bundle" / "release-artifact-index.json"),
                },
            ), ""
        if command[1].endswith("validate_release_candidate.py"):
            return 0, json.dumps(
                {
                    "format": "lewlm-release-candidate-validation-v1",
                    "overall_status": "passed",
                    "checks": {},
                },
            ), ""
        raise AssertionError(f"Unexpected command: {command}")

    payload = module.capture_host_validation(
        output_dir,
        capture_all_capabilities=True,
        required_targets=["Windows:AMD64"],
        minimum_verified_models=1,
        run_command=fake_runner,
    )

    assert payload["overall_status"] == "passed"
    assert [record["name"] for record in payload["commands"]] == ["config", "scan", "list-models", "doctor"]
    assert payload["capabilities"][0]["name"] == "capabilities:gguf-model"
    assert payload["release_bundle"]["capture"]["succeeded"] is True
    assert payload["release_bundle"]["validation"]["succeeded"] is True
    assert (output_dir / "cli" / "config.json").is_file()
    assert (output_dir / "capabilities" / "gguf-model.json").is_file()
    assert (output_dir / "release-bundle" / "capture-release-bundle.json").is_file()
    assert (output_dir / "host-validation-evidence.json").is_file()


def test_capture_host_validation_runs_loopback_http_probes_from_manifest(tmp_path: Path) -> None:
    module = _load_capture_module()
    output_dir = tmp_path / "host-validation"
    probe_manifest = tmp_path / "probes.json"
    probe_manifest.write_text(
        json.dumps(
            {
                "probes": [
                    {
                        "name": "chat-stream",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                        "json_body": {
                            "model": "bridge-model",
                            "stream": True,
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], *, cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
        if command[1:3] == ["-m", "lewlm"]:
            match command[3]:
                case "config":
                    return 0, json.dumps({"app_name": "LewLM"}), ""
                case "scan":
                    return 0, json.dumps({"discovered_count": 0, "new_count": 0, "updated_count": 0, "unchanged_count": 0, "removed_count": 0}), ""
                case "list-models":
                    return 0, json.dumps({"count": 0, "items": []}), ""
                case "doctor":
                    return 0, json.dumps({"runtime_stats": {"target_platforms": []}}), ""
        if command[1].endswith("capture_release_bundle.py"):
            return 0, json.dumps({"format": "lewlm-release-bundle-v1", "validation": {"overall_status": "passed"}}), ""
        if command[1].endswith("validate_release_candidate.py"):
            return 0, json.dumps({"format": "lewlm-release-candidate-validation-v1", "overall_status": "passed", "checks": {}}), ""
        raise AssertionError(f"Unexpected command: {command}")

    def fake_http_runner(**kwargs) -> dict[str, object]:
        path = kwargs["path"]
        if path == "/v1/health":
            return {
                "status_code": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"status": "ok"}).encode("utf-8"),
            }
        if path == "/v1/runtime/stats":
            return {
                "status_code": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"target_platforms": []}).encode("utf-8"),
            }
        if path == "/v1/chat/completions":
            return {
                "status_code": 200,
                "headers": {"Content-Type": "text/event-stream"},
                "body": b"data: {\"delta\":\"hello\"}\n\n",
            }
        raise AssertionError(f"Unexpected HTTP path: {path}")

    payload = module.capture_host_validation(
        output_dir,
        api_base_url="http://127.0.0.1:8080",
        http_probe_manifest_paths=[probe_manifest],
        run_command=fake_runner,
        run_http_request=fake_http_runner,
    )

    assert payload["overall_status"] == "passed"
    assert [record["name"] for record in payload["http_probes"]] == ["health", "runtime-stats", "chat-stream"]
    stream_probe = payload["http_probes"][2]
    assert stream_probe["content_type"] == "text/event-stream"
    assert (output_dir / stream_probe["output_path"]).read_text(encoding="utf-8") == 'data: {"delta":"hello"}\n\n'


def test_capture_host_validation_rejects_non_loopback_api_base_url(tmp_path: Path) -> None:
    module = _load_capture_module()

    def fake_runner(command: list[str], *, cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
        if command[1:3] == ["-m", "lewlm"]:
            match command[3]:
                case "config":
                    return 0, json.dumps({"app_name": "LewLM"}), ""
                case "scan":
                    return 0, json.dumps({"discovered_count": 0, "new_count": 0, "updated_count": 0, "unchanged_count": 0, "removed_count": 0}), ""
                case "list-models":
                    return 0, json.dumps({"count": 0, "items": []}), ""
                case "doctor":
                    return 0, json.dumps({"runtime_stats": {"target_platforms": []}}), ""
        if command[1].endswith("capture_release_bundle.py"):
            return 0, json.dumps({"format": "lewlm-release-bundle-v1", "validation": {"overall_status": "passed"}}), ""
        if command[1].endswith("validate_release_candidate.py"):
            return 0, json.dumps({"format": "lewlm-release-candidate-validation-v1", "overall_status": "passed", "checks": {}}), ""
        raise AssertionError(f"Unexpected command: {command}")

    with pytest.raises(ValueError, match="loopback"):
        module.capture_host_validation(
            tmp_path / "host-validation",
            api_base_url="http://example.com:8000",
            run_command=fake_runner,
        )


def test_capture_host_validation_marks_empty_cli_output_as_failure(tmp_path: Path) -> None:
    module = _load_capture_module()
    output_dir = tmp_path / "host-validation"

    def fake_runner(command: list[str], *, cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
        if command[1:3] == ["-m", "lewlm"]:
            match command[3]:
                case "config":
                    return 0, "", ""
                case "scan":
                    return 0, json.dumps({"discovered_count": 0, "new_count": 0, "updated_count": 0, "unchanged_count": 0, "removed_count": 0}), ""
                case "list-models":
                    return 0, json.dumps({"count": 0, "items": []}), ""
                case "doctor":
                    return 0, json.dumps({"runtime_stats": {"target_platforms": []}}), ""
        if command[1].endswith("capture_release_bundle.py"):
            return 0, json.dumps({"format": "lewlm-release-bundle-v1", "validation": {"overall_status": "passed"}}), ""
        if command[1].endswith("validate_release_candidate.py"):
            return 0, json.dumps({"format": "lewlm-release-candidate-validation-v1", "overall_status": "passed", "checks": {}}), ""
        raise AssertionError(f"Unexpected command: {command}")

    payload = module.capture_host_validation(
        output_dir,
        run_command=fake_runner,
    )

    assert payload["overall_status"] == "failed"
    assert payload["commands"][0]["name"] == "config"
    assert payload["commands"][0]["succeeded"] is False
    assert payload["commands"][0]["parse_error"] == "Expected JSON output, but command produced no stdout."
