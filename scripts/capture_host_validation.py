#!/usr/bin/env python3
"""Capture a host validation workspace around existing LewLM CLI and release scripts."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FORMAT = "lewlm-host-validation-evidence-v1"

CommandResult = tuple[int, str, str]
CommandRunner = Callable[..., CommandResult]
HttpRunner = Callable[..., dict[str, Any]]


def capture_host_validation(
    output_dir: str | Path,
    *,
    capability_models: list[str] | tuple[str, ...] = (),
    capture_all_capabilities: bool = False,
    api_base_url: str | None = None,
    http_probe_manifest_paths: list[str | Path] | tuple[str | Path, ...] = (),
    validation_manifest_paths: list[str | Path] | tuple[str | Path, ...] = (),
    required_systems: list[str] | tuple[str, ...] = (),
    required_targets: list[str] | tuple[str, ...] = (),
    minimum_verified_models: int = 0,
    required_frontier_families: list[str] | tuple[str, ...] = (),
    required_optimization_classes: list[str] | tuple[str, ...] = (),
    required_performance_core_pillars: list[str] | tuple[str, ...] = (),
    run_command: CommandRunner | None = None,
    run_http_request: HttpRunner | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve(strict=False)
    output_path.mkdir(parents=True, exist_ok=True)

    command_runner = run_command or _default_command_runner
    http_runner = run_http_request or _default_http_runner
    command_env = _command_environment()
    cli_command = [sys.executable, "-m", "lewlm"]

    command_records: list[dict[str, Any]] = []
    capability_records: list[dict[str, Any]] = []
    http_records: list[dict[str, Any]] = []

    config_record, _ = _capture_command(
        output_path,
        name="config",
        command=[*cli_command, "config", "--json"],
        relative_output_path=Path("cli") / "config.json",
        cwd=REPO_ROOT,
        env=command_env,
        run_command=command_runner,
    )
    command_records.append(config_record)

    scan_record, _ = _capture_command(
        output_path,
        name="scan",
        command=[*cli_command, "scan", "--json"],
        relative_output_path=Path("cli") / "scan.json",
        cwd=REPO_ROOT,
        env=command_env,
        run_command=command_runner,
    )
    command_records.append(scan_record)

    list_models_record, list_models_payload = _capture_command(
        output_path,
        name="list-models",
        command=[*cli_command, "list-models", "--json"],
        relative_output_path=Path("cli") / "list-models.json",
        cwd=REPO_ROOT,
        env=command_env,
        run_command=command_runner,
    )
    command_records.append(list_models_record)

    doctor_record, _ = _capture_command(
        output_path,
        name="doctor",
        command=[*cli_command, "doctor", "--json"],
        relative_output_path=Path("cli") / "doctor.json",
        cwd=REPO_ROOT,
        env=command_env,
        run_command=command_runner,
    )
    command_records.append(doctor_record)

    requested_models = list(capability_models)
    if capture_all_capabilities and isinstance(list_models_payload, dict):
        requested_models.extend(_model_ids_from_inventory(list_models_payload))

    for model_id in _dedupe_preserve_order(requested_models):
        capability_record, _ = _capture_command(
            output_path,
            name=f"capabilities:{model_id}",
            command=[*cli_command, "capabilities", model_id, "--json"],
            relative_output_path=Path("capabilities") / f"{_safe_name(model_id)}.json",
            cwd=REPO_ROOT,
            env=command_env,
            run_command=command_runner,
        )
        capability_records.append(capability_record)

    normalized_api_base_url: str | None = None
    if api_base_url is not None:
        normalized_api_base_url = _normalize_loopback_base_url(api_base_url)
        http_records.append(
            _capture_http_probe(
                output_path,
                name="health",
                method="GET",
                path="/v1/health",
                api_base_url=normalized_api_base_url,
                headers={},
                timeout_seconds=30.0,
                run_http_request=http_runner,
                relative_output_stem=Path("http") / "health",
            ),
        )
        http_records.append(
            _capture_http_probe(
                output_path,
                name="runtime-stats",
                method="GET",
                path="/v1/runtime/stats",
                api_base_url=normalized_api_base_url,
                headers={},
                timeout_seconds=30.0,
                run_http_request=http_runner,
                relative_output_stem=Path("http") / "runtime-stats",
            ),
        )
        for probe in _load_http_probes(http_probe_manifest_paths):
            http_records.append(
                _capture_http_probe(
                    output_path,
                    name=str(probe["name"]),
                    method=str(probe["method"]),
                    path=str(probe["path"]),
                    api_base_url=normalized_api_base_url,
                    headers=dict(probe.get("headers", {})),
                    timeout_seconds=float(probe.get("timeout_seconds", 30.0)),
                    json_body=probe.get("json_body"),
                    body_text=probe.get("body_text"),
                    body_bytes=probe.get("body_bytes"),
                    content_type=probe.get("content_type"),
                    expected_status=probe.get("expected_status"),
                    run_http_request=http_runner,
                    relative_output_stem=Path(str(probe["output_stem"])),
                ),
            )

    bundle_command = [
        sys.executable,
        str(SCRIPT_DIR / "capture_release_bundle.py"),
        "--output-dir",
        str(output_path / "release-bundle"),
    ]
    for path in validation_manifest_paths:
        bundle_command.extend(["--validation-manifest-path", str(path)])
    for system in required_systems:
        bundle_command.extend(["--require-system", str(system)])
    for target in required_targets:
        bundle_command.extend(["--require-target", str(target)])
    if minimum_verified_models:
        bundle_command.extend(["--minimum-verified-models", str(minimum_verified_models)])
    for family in required_frontier_families:
        bundle_command.extend(["--require-frontier-family", str(family)])
    for optimization_class in required_optimization_classes:
        bundle_command.extend(["--require-optimization-class", str(optimization_class)])

    bundle_record, _ = _capture_command(
        output_path,
        name="capture-release-bundle",
        command=bundle_command,
        relative_output_path=Path("release-bundle") / "capture-release-bundle.json",
        cwd=REPO_ROOT,
        env=command_env,
        run_command=command_runner,
    )

    validate_command = [
        sys.executable,
        str(SCRIPT_DIR / "validate_release_candidate.py"),
        str(output_path / "release-bundle"),
    ]
    for system in required_systems:
        validate_command.extend(["--require-system", str(system)])
    for target in required_targets:
        validate_command.extend(["--require-target", str(target)])
    if minimum_verified_models:
        validate_command.extend(["--minimum-verified-models", str(minimum_verified_models)])
    for family in required_frontier_families:
        validate_command.extend(["--require-frontier-family", str(family)])
    for optimization_class in required_optimization_classes:
        validate_command.extend(["--require-optimization-class", str(optimization_class)])
    for pillar in required_performance_core_pillars:
        validate_command.extend(["--require-performance-core-pillar", str(pillar)])

    explicit_validation_record, _ = _capture_command(
        output_path,
        name="validate-release-candidate",
        command=validate_command,
        relative_output_path=Path("release-bundle") / "validate-release-candidate.json",
        cwd=REPO_ROOT,
        env=command_env,
        run_command=command_runner,
    )

    records = [*command_records, *capability_records, bundle_record, explicit_validation_record]
    overall_passed = all(record["succeeded"] for record in records) and all(record["succeeded"] for record in http_records)
    payload = {
        "format": FORMAT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_path),
        "host_platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cli_command": cli_command,
        "api_base_url": normalized_api_base_url,
        "commands": command_records,
        "capabilities": capability_records,
        "http_probes": http_records,
        "release_bundle": {
            "capture": bundle_record,
            "validation": explicit_validation_record,
        },
        "overall_status": "passed" if overall_passed else "failed",
    }
    _write_json(output_path / "host-validation-evidence.json", payload)
    return payload


def _command_environment() -> dict[str, str]:
    env = dict(os.environ)
    src_path = str(REPO_ROOT / "src")
    existing_python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing_python_path else os.pathsep.join((src_path, existing_python_path))
    return env


def _capture_command(
    output_root: Path,
    *,
    name: str,
    command: list[str],
    relative_output_path: Path,
    cwd: Path,
    env: dict[str, str],
    run_command: CommandRunner,
) -> tuple[dict[str, Any], Any | None]:
    exit_code, stdout_text, stderr_text = run_command(command, cwd=cwd, env=env)
    output_path = output_root / relative_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stdout_text, encoding="utf-8")

    stderr_path: Path | None = None
    if stderr_text:
        stderr_path = output_path.with_suffix(f"{output_path.suffix}.stderr.txt")
        stderr_path.write_text(stderr_text, encoding="utf-8")

    parsed_payload: Any | None = None
    parse_error: str | None = None
    if stdout_text.strip():
        try:
            parsed_payload = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
    else:
        parse_error = "Expected JSON output, but command produced no stdout."

    succeeded = exit_code == 0 and parse_error is None
    record = {
        "name": name,
        "command": command,
        "output_path": _relative_path(output_root, output_path),
        "stderr_path": _relative_path(output_root, stderr_path) if stderr_path is not None else None,
        "exit_code": exit_code,
        "succeeded": succeeded,
        "summary": _summarize_payload(parsed_payload),
    }
    if parse_error is not None:
        record["parse_error"] = parse_error
    return record, parsed_payload


def _capture_http_probe(
    output_root: Path,
    *,
    name: str,
    method: str,
    path: str,
    api_base_url: str,
    headers: dict[str, str],
    timeout_seconds: float,
    run_http_request: HttpRunner,
    relative_output_stem: Path,
    json_body: Any | None = None,
    body_text: str | None = None,
    body_bytes: bytes | None = None,
    content_type: str | None = None,
    expected_status: int | list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    response = run_http_request(
        method=method,
        api_base_url=api_base_url,
        path=path,
        headers=headers,
        timeout_seconds=timeout_seconds,
        json_body=json_body,
        body_text=body_text,
        body_bytes=body_bytes,
        content_type=content_type,
    )
    status_code = int(response["status_code"])
    response_headers = {str(key): str(value) for key, value in dict(response["headers"]).items()}
    body = bytes(response["body"])
    content_type_value = response_headers.get("Content-Type", "")
    output_suffix = _content_type_suffix(content_type_value)
    output_path = output_root / relative_output_stem.with_suffix(output_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_http_response(output_path, body=body, content_type=content_type_value)

    allowed_statuses = _expected_statuses(expected_status)
    succeeded = status_code in allowed_statuses
    return {
        "name": name,
        "method": method,
        "path": path,
        "status_code": status_code,
        "expected_statuses": allowed_statuses,
        "succeeded": succeeded,
        "output_path": _relative_path(output_root, output_path),
        "content_type": content_type_value,
        "summary": _summarize_http_response(body=body, content_type=content_type_value),
    }


def _load_http_probes(paths: list[str | Path] | tuple[str | Path, ...]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for raw_path in paths:
        manifest_path = Path(raw_path).expanduser().resolve(strict=True)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_probes = payload.get("probes")
        else:
            raw_probes = payload
        if not isinstance(raw_probes, list):
            raise ValueError(f"HTTP probe manifest must contain a list of probes: {manifest_path}")
        for index, probe in enumerate(raw_probes, start=1):
            if not isinstance(probe, dict):
                raise ValueError(f"HTTP probe entry #{index} must be an object: {manifest_path}")
            normalized = _normalize_http_probe(probe, manifest_path=manifest_path, index=index)
            probes.append(normalized)
    return probes


def _normalize_http_probe(probe: dict[str, Any], *, manifest_path: Path, index: int) -> dict[str, Any]:
    name = str(probe.get("name") or f"probe-{index}")
    method = str(probe.get("method", "GET")).upper()
    path = str(probe.get("path") or "")
    if not path.startswith("/v1/"):
        raise ValueError(f"HTTP probe '{name}' must target a LewLM /v1/ path: {manifest_path}")
    body_fields = [field for field in ("json_body", "body_text", "body_path") if probe.get(field) is not None]
    if len(body_fields) > 1:
        raise ValueError(f"HTTP probe '{name}' can define only one of json_body, body_text, or body_path.")
    if not isinstance(probe.get("headers", {}), dict):
        raise ValueError(f"HTTP probe '{name}' headers must be an object.")
    output_name = str(probe.get("output_file") or (Path("http") / "probes" / _safe_name(name)))
    normalized: dict[str, Any] = {
        "name": name,
        "method": method,
        "path": path,
        "headers": {str(key): str(value) for key, value in dict(probe.get("headers", {})).items()},
        "timeout_seconds": float(probe.get("timeout_seconds", 30.0)),
        "output_stem": output_name,
        "expected_status": probe.get("expected_status"),
    }
    if probe.get("json_body") is not None:
        normalized["json_body"] = probe["json_body"]
    if probe.get("body_text") is not None:
        normalized["body_text"] = str(probe["body_text"])
    if probe.get("body_path") is not None:
        body_path = (manifest_path.parent / str(probe["body_path"])).resolve(strict=True)
        normalized["body_bytes"] = body_path.read_bytes()
        if probe.get("content_type") is not None:
            normalized["content_type"] = str(probe["content_type"])
    return normalized


def _expected_statuses(expected_status: int | list[int] | tuple[int, ...] | None) -> list[int]:
    if expected_status is None:
        return [200]
    if isinstance(expected_status, int):
        return [expected_status]
    if not isinstance(expected_status, (list, tuple)):
        raise ValueError("expected_status must be an int or list of ints.")
    values = [int(status) for status in expected_status]
    if not values:
        raise ValueError("expected_status cannot be empty.")
    return values


def _write_http_response(path: Path, *, body: bytes, content_type: str) -> None:
    if _is_json_content_type(content_type):
        decoded = body.decode("utf-8")
        parsed = json.loads(decoded)
        _write_json(path, parsed)
        return
    if _is_text_content_type(content_type):
        path.write_text(body.decode("utf-8"), encoding="utf-8")
        return
    path.write_bytes(body)


def _content_type_suffix(content_type: str) -> str:
    if _is_json_content_type(content_type):
        return ".json"
    if _is_text_content_type(content_type):
        return ".txt"
    return ".bin"


def _is_json_content_type(content_type: str) -> bool:
    return "json" in content_type.casefold()


def _is_text_content_type(content_type: str) -> bool:
    lowered = content_type.casefold()
    return lowered.startswith("text/") or "event-stream" in lowered


def _summarize_http_response(*, body: bytes, content_type: str) -> dict[str, Any]:
    if _is_json_content_type(content_type):
        parsed = json.loads(body.decode("utf-8"))
        return _summarize_payload(parsed)
    if _is_text_content_type(content_type):
        text = body.decode("utf-8")
        return {"line_count": len(text.splitlines()), "byte_count": len(body)}
    return {"byte_count": len(body)}


def _summarize_payload(payload: Any | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    summary: dict[str, Any] = {}
    for key in (
        "count",
        "discovered_count",
        "new_count",
        "updated_count",
        "unchanged_count",
        "removed_count",
        "registered_model_count",
        "manifest_count",
        "overall_status",
    ):
        if key in payload:
            summary[key] = payload[key]
    items = payload.get("items")
    if isinstance(items, list):
        summary["item_count"] = len(items)
    capabilities = payload.get("capabilities")
    if isinstance(capabilities, list):
        summary["supported_capabilities"] = sorted(
            str(entry.get("capability"))
            for entry in capabilities
            if isinstance(entry, dict) and entry.get("supported")
        )
    runtime_stats = payload.get("runtime_stats")
    if isinstance(runtime_stats, dict):
        target_platforms = runtime_stats.get("target_platforms")
        if isinstance(target_platforms, list):
            summary["target_platform_count"] = len(target_platforms)
    checks = payload.get("checks")
    if isinstance(checks, dict):
        summary["failed_checks"] = sorted(
            name
            for name, entry in checks.items()
            if isinstance(entry, dict) and entry.get("passed") is False
        )
    return summary or None


def _model_ids_from_inventory(payload: dict[str, Any]) -> list[str]:
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    model_ids: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model_id")
        if model_id:
            model_ids.append(str(model_id))
    return model_ids


def _dedupe_preserve_order(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_loopback_base_url(api_base_url: str) -> str:
    parsed = urllib_parse.urlparse(api_base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("API base URL must use http or https.")
    if parsed.username or parsed.password:
        raise ValueError("API base URL must not include credentials.")
    host = parsed.hostname
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("API base URL must target a loopback host.")
    base_path = parsed.path.rstrip("/")
    normalized = urllib_parse.urlunparse((parsed.scheme, parsed.netloc, base_path, "", "", ""))
    return normalized.rstrip("/")


def _default_command_runner(command: list[str], *, cwd: Path, env: dict[str, str]) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _default_http_runner(
    *,
    method: str,
    api_base_url: str,
    path: str,
    headers: dict[str, str],
    timeout_seconds: float,
    json_body: Any | None = None,
    body_text: str | None = None,
    body_bytes: bytes | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    if json_body is not None:
        payload = json.dumps(json_body).encode("utf-8")
        resolved_headers = {"Content-Type": "application/json", **headers}
    elif body_text is not None:
        payload = body_text.encode("utf-8")
        resolved_headers = dict(headers)
        if content_type is not None:
            resolved_headers.setdefault("Content-Type", content_type)
    elif body_bytes is not None:
        payload = body_bytes
        resolved_headers = dict(headers)
        if content_type is not None:
            resolved_headers.setdefault("Content-Type", content_type)
    else:
        payload = None
        resolved_headers = dict(headers)

    request_url = urllib_parse.urljoin(f"{api_base_url}/", path.lstrip("/"))
    request = urllib_request.Request(
        request_url,
        data=payload,
        headers=resolved_headers,
        method=method,
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            response_headers = dict(response.headers.items())
            return {
                "status_code": response.status,
                "headers": response_headers,
                "body": body,
            }
    except urllib_error.HTTPError as exc:
        body = exc.read()
        return {
            "status_code": exc.code,
            "headers": dict(exc.headers.items()),
            "body": body,
        }


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return normalized or "item"


def _relative_path(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return path.relative_to(root).as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory where host validation artifacts should be written.")
    parser.add_argument(
        "--capabilities-model",
        action="append",
        default=[],
        dest="capability_models",
        help="Capture `lewlm capabilities --json` for the specified registered model id.",
    )
    parser.add_argument(
        "--capture-all-capabilities",
        action="store_true",
        help="After scanning, capture capability reports for every model listed by `lewlm list-models --json`.",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="Optional loopback LewLM API base URL used for /v1/health, /v1/runtime/stats, and probe-manifest requests.",
    )
    parser.add_argument(
        "--http-probe-manifest",
        action="append",
        default=[],
        dest="http_probe_manifest_paths",
        help="JSON file containing additional loopback /v1/ HTTP probes to capture.",
    )
    parser.add_argument(
        "--validation-manifest-path",
        action="append",
        default=[],
        dest="validation_manifest_paths",
        help="Additional file or directory containing host release manifests to include in bundle validation.",
    )
    parser.add_argument(
        "--require-system",
        action="append",
        default=[],
        dest="required_systems",
        help="Require at least one host-verified manifest for the named platform system.",
    )
    parser.add_argument(
        "--require-target",
        action="append",
        default=[],
        dest="required_targets",
        help="Require at least one host-verified manifest for the exact SYSTEM:MACHINE target pair.",
    )
    parser.add_argument(
        "--minimum-verified-models",
        type=int,
        default=0,
        help="Require each enforced target to include at least this many host-verified registered models.",
    )
    parser.add_argument(
        "--require-frontier-family",
        action="append",
        default=[],
        dest="required_frontier_families",
        help="Require host-backed evidence for the named frontier family on each enforced target.",
    )
    parser.add_argument(
        "--require-optimization-class",
        action="append",
        default=[],
        dest="required_optimization_classes",
        help="Require resolved optimization-default coverage for the named optimization class on each enforced target.",
    )
    parser.add_argument(
        "--require-performance-core-pillar",
        action="append",
        default=[],
        dest="required_performance_core_pillars",
        help="Require covered performance-core proof for the named pillar on each enforced target.",
    )
    args = parser.parse_args(argv)

    try:
        payload = capture_host_validation(
            args.output_dir,
            capability_models=args.capability_models,
            capture_all_capabilities=args.capture_all_capabilities,
            api_base_url=args.api_base_url,
            http_probe_manifest_paths=args.http_probe_manifest_paths,
            validation_manifest_paths=args.validation_manifest_paths,
            required_systems=args.required_systems,
            required_targets=args.required_targets,
            minimum_verified_models=args.minimum_verified_models,
            required_frontier_families=args.required_frontier_families,
            required_optimization_classes=args.required_optimization_classes,
            required_performance_core_pillars=args.required_performance_core_pillars,
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
