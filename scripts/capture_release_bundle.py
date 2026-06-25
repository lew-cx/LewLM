#!/usr/bin/env python3
"""Capture a complete LewLM release bundle in one output directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_dependency_audit import build_dependency_audit
from generate_release_manifest import build_release_manifest
from generate_sbom import build_sbom
from validate_release_candidate import build_release_candidate_validation


def capture_release_bundle(
    output_dir: str | Path,
    *,
    validation_manifest_paths: list[str | Path] | tuple[str | Path, ...] = (),
    required_systems: list[str] | tuple[str, ...] = (),
    required_targets: list[str] | tuple[str, ...] = (),
    minimum_verified_models: int = 0,
    required_frontier_families: list[str] | tuple[str, ...] = (),
    required_optimization_classes: list[str] | tuple[str, ...] = (),
) -> dict[str, object]:
    output_path = Path(output_dir).expanduser().resolve(strict=False)
    output_path.mkdir(parents=True, exist_ok=True)
    expanded_validation_paths = _expanded_input_paths(validation_manifest_paths)

    sbom = build_sbom()
    dependency_audit = build_dependency_audit()
    release_manifest = build_release_manifest()

    artifacts = [
        _write_json_artifact(output_path / "sbom.json", sbom),
        _write_json_artifact(output_path / "dependency-audit.json", dependency_audit),
        _write_json_artifact(output_path / "release-manifest.json", release_manifest),
    ]

    bundled_validation_inputs = _bundle_validation_inputs(output_path, expanded_validation_paths)
    validation_payload = build_release_candidate_validation(
        [output_path, *(output_path / str(item["bundled_path"]) for item in bundled_validation_inputs)],
        required_systems=required_systems,
        required_targets=required_targets,
        minimum_verified_models=minimum_verified_models,
        required_frontier_families=required_frontier_families,
        required_optimization_classes=required_optimization_classes,
    )
    artifacts.append(
        _write_json_artifact(output_path / "release-candidate-validation.json", validation_payload),
    )

    index_payload = {
        "format": "lewlm-release-artifact-index-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_path),
        "artifacts": artifacts,
        "validation_inputs": bundled_validation_inputs,
    }
    _write_json(output_path / "release-artifact-index.json", index_payload)

    return {
        "format": "lewlm-release-bundle-v1",
        "generated_at": index_payload["generated_at"],
        "output_dir": str(output_path),
        "artifacts": artifacts,
        "validation_inputs": bundled_validation_inputs,
        "validation": {
            "overall_status": validation_payload["overall_status"],
            "checks": validation_payload["checks"],
        },
        "index_path": str(output_path / "release-artifact-index.json"),
    }


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> dict[str, object]:
    _write_json(path, payload)
    return {
        "path": str(path.name),
        "format": payload.get("format"),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _path_entry(path: Path) -> dict[str, object]:
    payload_format = None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        payload_format = payload.get("format")
    return {
        "path": str(path),
        "format": payload_format,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _bundle_validation_inputs(
    output_path: Path,
    paths: list[str | Path] | tuple[str | Path, ...],
) -> list[dict[str, object]]:
    bundled_dir = output_path / "validation-manifests"
    bundled_inputs: list[dict[str, object]] = []
    for index, source_path in enumerate(_expanded_input_paths(paths), start=1):
        bundled_dir.mkdir(parents=True, exist_ok=True)
        copy_name = _bundled_input_name(source_path, index=index)
        bundled_path = bundled_dir / copy_name
        shutil.copy2(source_path, bundled_path)
        entry = _path_entry(bundled_path)
        bundled_inputs.append(
            {
                "source_path": str(source_path),
                "bundled_path": bundled_path.relative_to(output_path).as_posix(),
                "format": entry["format"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            },
        )
    return bundled_inputs


def _expanded_input_paths(paths: list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve(strict=False)
        if not path.exists():
            raise FileNotFoundError(f"Validation manifest path does not exist: {path}")
        if path.is_dir():
            candidates = sorted(item for item in path.rglob("*.json") if item.is_file())
        else:
            candidates = [path]
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            expanded.append(candidate)
    return expanded


def _bundled_input_name(source_path: Path, *, index: int) -> str:
    payload_format = None
    system = None
    machine = None
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        payload_format = payload.get("format")
        platform_payload = payload.get("platform", {})
        if isinstance(platform_payload, dict):
            system = platform_payload.get("system")
            machine = platform_payload.get("machine")
    if payload_format == "lewlm-release-manifest-v1" and system and machine:
        safe_system = str(system).lower().replace(" ", "-")
        safe_machine = str(machine).lower().replace(" ", "-")
        return f"{safe_system}-{safe_machine}-release-manifest.json"
    return f"{index:02d}-{source_path.name}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory where release artifacts should be written.")
    parser.add_argument(
        "--validation-manifest-path",
        action="append",
        default=[],
        dest="validation_manifest_paths",
        help="Additional file or directory containing host release manifests to include in validation.",
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
    args = parser.parse_args(argv)
    payload = capture_release_bundle(
        args.output_dir,
        validation_manifest_paths=args.validation_manifest_paths,
        required_systems=args.required_systems,
        required_targets=args.required_targets,
        minimum_verified_models=args.minimum_verified_models,
        required_frontier_families=args.required_frontier_families,
        required_optimization_classes=args.required_optimization_classes,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["validation"]["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
