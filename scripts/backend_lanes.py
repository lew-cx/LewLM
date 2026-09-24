#!/usr/bin/env python3
"""Hardware acceptance lanes: detect, run, and summarize, with honest deferrals.

    python scripts/backend_lanes.py detect                       # which lanes this host can run, and why not the others
    python scripts/backend_lanes.py run --lane linux_nvidia --recipe vllm \
        --base-url http://127.0.0.1:8080 --model <id> --output-dir evidence/lanes
    python scripts/backend_lanes.py summary [--records evidence/lanes]   # per-lane status for the release bundle

A lane names a platform (Apple Silicon macOS, Linux CPU, Linux NVIDIA, native
Windows, Windows + WSL2, any development OS, and the Chap UI) and the proof it
requires. `run` executes the common real-engine acceptance suite
(`scripts/backend_acceptance.py`) and the prefix benchmark against a LewLM
server the operator already started for a pinned recipe, and writes one lane
record. When this host cannot run the requested lane, the record is
`deferred` with the exact missing prerequisite and the next command — that is
a valid outcome, not a failure. A failed acceptance case makes the record
`failed` and the command exit 1: a contract failure blocks promotion.
`summary` merges lane records with `examples/backends/compatibility.json` so
deferred hardware stays visible in release evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
COMPATIBILITY_PATH = ROOT / "examples" / "backends" / "compatibility.json"

Runner = Callable[[Sequence[str]], tuple[int, str, str]]


@dataclass(frozen=True)
class Lane:
    lane: str
    title: str
    system: str | tuple[str, ...] | None  # platform.system() required (any of a tuple), or None for any
    machine: str | None           # platform.machine() required (arm64 for Apple Silicon), or None
    needs_nvidia: bool
    needs_wsl: bool
    recipes: tuple[str, ...]
    required_proof: str
    deferral_rule: str
    runnable_by_script: bool = True


LANES: dict[str, Lane] = {
    "any_os": Lane(
        "any_os", "Any development OS", None, None, False, False, (),
        "Fake-server translation, tools/JSON/SSE, cancellation, migration, routing, fallback, schema/client checks",
        "No hardware-based exemption",
    ),
    "linux_cpu": Lane(
        "linux_cpu", "Linux CPU", "Linux", None, False, False, ("llamacpp-cpu",),
        "Lean install, portable GGUF generation, image boot, conversion image, cold/warm rebuild evidence",
        "Defer if Linux/container runtime unavailable",
    ),
    "apple_silicon": Lane(
        "apple_silicon", "Apple Silicon macOS", "Darwin", "arm64", False, False, ("omlx",),
        "oMLX suite and native MLX/llama.cpp/Ollama coexistence",
        "Defer on non-Apple hardware",
    ),
    "linux_nvidia": Lane(
        "linux_nvidia", "Linux NVIDIA", "Linux", None, True, False, ("vllm", "sglang", "exllamav3-tabby", "llamacpp-cuda"),
        "vLLM, SGLang, ExLlamaV3/TabbyAPI suites; llama.cpp CUDA offload/build/cache validation",
        "Defer without supported NVIDIA hardware/driver",
    ),
    "native_windows": Lane(
        "native_windows", "Native Windows", "Windows", None, False, False, ("llamacpp-cpu", "ollama"),
        "Core install, llama.cpp import/generation, Ollama bridge, shutdown/cancellation; ExLlamaV3 only if separately promoted",
        "Linux or WSL results do not satisfy this lane",
    ),
    # Runnable from either side of the boundary: inside a WSL2 distribution, or
    # from Windows (where Chap and a host-run LewLM live) while a WSL2 VM runs
    # the engine -- Docker Desktop's WSL2 backend included.
    "wsl2": Lane(
        "wsl2", "Windows + WSL2", ("Linux", "Windows"), None, False, True, ("vllm", "sglang", "exllamav3-tabby"),
        "Documented engine recipes and Windows-Chap-to-LewLM connectivity",
        "A Linux pass alone does not prove WSL networking",
    ),
    "chap_ui": Lane(
        "chap_ui", "Chap UI, later", None, None, False, False, (),
        "End-user checklist against the same public contracts",
        "Mark UI acceptance pending, never silently complete",
        runnable_by_script=False,
    ),
}

# Which compatibility-manifest profile a recipe promotes, when it has one.
RECIPE_PROFILES = {"omlx": "omlx", "vllm": "vllm_local", "sglang": "sglang_local", "exllamav3-tabby": "exllamav3_tabby"}
RECIPE_ENDPOINTS = {"omlx": "omlx", "vllm": "vllm", "sglang": "sglang", "exllamav3-tabby": "tabby"}


@dataclass
class HostFacts:
    system: str
    machine: str
    nvidia: bool
    nvidia_detail: str
    wsl: bool

    @classmethod
    def detect(cls, runner: Runner | None = None) -> "HostFacts":
        runner = runner or _default_runner
        code, out, err = runner(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        nvidia = code == 0 and bool(out.strip())
        detail = out.strip().splitlines()[0] if nvidia else (err.strip() or "nvidia-smi unavailable")[:200]
        system = platform.system()
        if system == "Windows":
            wsl = _windows_wsl2_running()
        else:
            try:
                wsl = "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
            except OSError:
                wsl = False
        return cls(system=system, machine=platform.machine(), nvidia=nvidia, nvidia_detail=detail, wsl=wsl)


def _windows_wsl2_running() -> bool:
    """Whether `wsl.exe -l -v` lists a running version-2 distribution (Docker Desktop's included)."""

    if shutil.which("wsl.exe") is None:
        return False
    try:
        completed = subprocess.run(["wsl.exe", "-l", "-v"], capture_output=True, timeout=30,
                                   env={**os.environ, "WSL_UTF8": "1"})
    except (OSError, subprocess.TimeoutExpired):
        return False
    # Older wsl.exe ignores WSL_UTF8 and writes UTF-16LE.
    text = completed.stdout.decode("utf-8", errors="replace").replace("\x00", "")
    for line in text.splitlines()[1:]:
        fields = line.replace("*", " ").split()
        if len(fields) >= 3 and fields[-2].lower() == "running" and fields[-1] == "2":
            return True
    return False


def _default_runner(command: Sequence[str]) -> tuple[int, str, str]:
    if shutil.which(command[0]) is None:
        return 127, "", f"{command[0]}: not found on PATH"
    completed = subprocess.run(list(command), capture_output=True, text=True, timeout=60)
    return completed.returncode, completed.stdout, completed.stderr


def lane_blockers(lane: Lane, host: HostFacts) -> list[str]:
    """Why this host cannot run the lane; empty when it can."""

    blockers: list[str] = []
    if not lane.runnable_by_script:
        blockers.append("this lane is a manual checklist; the script never marks it complete")
    systems = (lane.system,) if isinstance(lane.system, str) else lane.system
    if systems is not None and host.system not in systems:
        blockers.append(f"needs {' or '.join(systems)}, host is {host.system}")
    if lane.machine is not None and host.machine != lane.machine:
        blockers.append(f"needs {lane.machine}, host is {host.machine}")
    if lane.needs_nvidia and not host.nvidia:
        blockers.append(f"needs a working NVIDIA driver ({host.nvidia_detail})")
    if lane.needs_wsl and not host.wsl:
        if host.system == "Windows":
            blockers.append("needs a running WSL2 distribution (`wsl.exe -l -v` lists none running at version 2)")
        else:
            blockers.append("needs a WSL2 kernel (no 'microsoft' in /proc/version)")
    return blockers


def detect(host: HostFacts | None = None) -> dict[str, Any]:
    host = host or HostFacts.detect()
    lanes = []
    for lane in LANES.values():
        blockers = lane_blockers(lane, host)
        lanes.append({"lane": lane.lane, "title": lane.title, "runnable_here": not blockers, "blockers": blockers,
                      "recipes": list(lane.recipes), "required_proof": lane.required_proof, "deferral_rule": lane.deferral_rule})
    return {"host": asdict(host), "lanes": lanes}


def _load_script(name: str):
    spec = spec_from_file_location(f"lewlm_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def next_command(lane: Lane, recipe: str | None) -> str:
    recipe_part = f" --recipe {recipe}" if recipe else ""
    return (f"On a {lane.title} host: follow examples/backends/{recipe}/README.md, then "
            f"python scripts/backend_lanes.py run --lane {lane.lane}{recipe_part} --base-url http://127.0.0.1:8080 --model <id> --output-dir <evidence>")


def run_lane(
    *,
    lane_name: str,
    recipe: str | None,
    base_url: str | None,
    model: str | None,
    output_dir: Path,
    host: HostFacts | None = None,
    acceptance_runner: Callable[..., dict[str, Any]] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Run one lane for one recipe, or record why it could not run here."""

    lane = LANES[lane_name]
    host = host or HostFacts.detect()
    output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "format": "lewlm-backend-lane-v1",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane": lane.lane,
        "title": lane.title,
        "recipe": recipe,
        "profile": RECIPE_PROFILES.get(recipe or ""),
        "host": asdict(host),
        "label": label,
        "required_proof": lane.required_proof,
    }
    blockers = lane_blockers(lane, host)
    if blockers:
        record.update({"status": "deferred", "reason": "; ".join(blockers), "next_command": next_command(lane, recipe), "evidence": []})
    elif recipe is not None and recipe not in lane.recipes:
        record.update({"status": "deferred", "reason": f"recipe {recipe!r} is not part of lane {lane.lane}; lanes cover {list(lane.recipes)}", "next_command": next_command(lane, None), "evidence": []})
    elif base_url is None or model is None:
        record.update({"status": "deferred", "reason": "this host can run the lane, but no --base-url/--model for an operator-started LewLM+engine was given",
                       "next_command": next_command(lane, recipe), "evidence": []})
    else:
        runner = acceptance_runner or _run_acceptance
        result = runner(base_url=base_url, model=model, endpoint_id=RECIPE_ENDPOINTS.get(recipe or ""), output_dir=output_dir, label=label)
        failed = [case for case in result.get("cases", []) if case.get("status") == "failed"]
        record.update({
            "status": "failed" if failed else "passed",
            "reason": (f"{len(failed)} acceptance case(s) failed: " + ", ".join(c["case"] for c in failed)) if failed else None,
            "evidence": result.get("evidence", []),
            "acceptance_summary": result.get("summary"),
            "next_command": (None if not failed else "fix the failing cases, then re-run; a failed lane never promotes its recipe"),
        })
    path = output_dir / f"lane-{lane.lane}{('-' + recipe) if recipe else ''}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["record_path"] = str(path)
    return record


def _run_acceptance(*, base_url: str, model: str, endpoint_id: str | None, output_dir: Path, label: str | None) -> dict[str, Any]:
    """The common suite plus the prefix benchmark, as subprocesses, writing evidence files."""

    evidence: list[str] = []
    acceptance_path = output_dir / "acceptance.json"
    command = [sys.executable, str(ROOT / "scripts" / "backend_acceptance.py"), "--base-url", base_url, "--model", model, "--output", str(acceptance_path)]
    if endpoint_id:
        command += ["--endpoint-id", endpoint_id]
    if label:
        command += ["--label", label]
    subprocess.run(command, cwd=ROOT, check=False, timeout=3600)
    cases: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    if acceptance_path.exists():
        payload = json.loads(acceptance_path.read_text(encoding="utf-8"))
        cases = payload.get("results") or payload.get("cases") or []
        summary = payload.get("summary")
        evidence.append(str(acceptance_path))
    else:
        cases = [{"case": "acceptance", "status": "failed", "reason": "backend_acceptance.py produced no record"}]
    benchmark_path = output_dir / "prefix-benchmark-c1.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "bridge_prefix_benchmark.py"), "--lewlm-url", base_url, "--model", model, "--requests", "12", "--output", str(benchmark_path)], cwd=ROOT, check=False, timeout=3600)
    if benchmark_path.exists():
        evidence.append(str(benchmark_path))
    return {"cases": cases, "summary": summary, "evidence": evidence}


def build_lane_summary(records_dir: Path | None = None, compatibility_path: Path = COMPATIBILITY_PATH) -> dict[str, Any]:
    """Per-lane status for release evidence: lane records where they exist, else the manifest, else deferred."""

    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8")) if compatibility_path.exists() else {"recipes": []}
    by_profile = {recipe["profile"]: recipe for recipe in compatibility.get("recipes", [])}
    records: dict[tuple[str, str | None], dict[str, Any]] = {}
    if records_dir is not None and records_dir.exists():
        for path in sorted(records_dir.glob("lane-*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("format") == "lewlm-backend-lane-v1":
                records[(payload["lane"], payload.get("recipe"))] = {**payload, "record_path": str(path)}
    lanes: list[dict[str, Any]] = []
    for lane in LANES.values():
        entries: list[dict[str, Any]] = []
        for recipe in lane.recipes or (None,):
            record = records.get((lane.lane, recipe))
            manifest_entry = by_profile.get(RECIPE_PROFILES.get(recipe or "", ""))
            # Precedence: a real run (passed/failed) beats the manifest; the
            # manifest beats a "nothing was started right now" deferral, so a
            # validated recipe is never hidden by an idle host.
            if record is not None and record.get("status") in {"passed", "failed"}:
                status, source, detail = record["status"], "lane_record", record.get("reason") or record.get("next_command")
            elif manifest_entry is not None:
                status, source = manifest_entry["status"], "compatibility_manifest"
                detail = manifest_entry.get("reason") or manifest_entry.get("next_step") or "; ".join(manifest_entry.get("notes", [])[:1])
                if record is not None:
                    detail = f"{detail} (latest lane run here: {record.get('status')} — {record.get('reason')})"
            elif record is not None:
                status, source, detail = record["status"], "lane_record", record.get("reason") or record.get("next_command")
            elif lane.lane == "any_os":
                status, source, detail = "ci", "ci_matrix", "fixture-only checks run in the CI matrix on Linux, macOS, and Windows"
            elif lane.lane == "chap_ui":
                status, source, detail = "pending", "manual_checklist", "docs/guides/chap-validation.md"
            else:
                status, source, detail = "deferred", "none", lane.deferral_rule
            entries.append({"recipe": recipe, "status": status, "source": source, "detail": detail,
                            **({"record_path": record["record_path"]} if record else {}),
                            **({"evidence_path": manifest_entry.get("evidence_path")} if manifest_entry and manifest_entry.get("evidence_path") else {})})
        lanes.append({"lane": lane.lane, "title": lane.title, "required_proof": lane.required_proof, "deferral_rule": lane.deferral_rule, "entries": entries})
    counts: dict[str, int] = {}
    for lane in lanes:
        for entry in lane["entries"]:
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return {"format": "lewlm-backend-lanes-summary-v1", "lanes": lanes, "counts": counts,
            "note": "Only `validated`/`passed` entries are proof; `deferred` and `pending` remain visible on purpose."}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("detect", help="Which lanes this host can run")
    run = sub.add_parser("run", help="Run one lane for one recipe, or record a deferral")
    run.add_argument("--lane", choices=sorted(LANES), required=True)
    run.add_argument("--recipe", default=None, choices=sorted({r for lane in LANES.values() for r in lane.recipes}))
    run.add_argument("--base-url", default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--label", default=None)
    run.add_argument("--require", action="store_true", help="Exit 1 on a deferral too (for a lane that must run here)")
    summary = sub.add_parser("summary", help="Per-lane status for release evidence")
    summary.add_argument("--records", default=None, help="Directory of lane-*.json records")
    summary.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if args.command == "detect":
        print(json.dumps(detect(), indent=2))
        return 0
    if args.command == "run":
        record = run_lane(lane_name=args.lane, recipe=args.recipe, base_url=args.base_url, model=args.model,
                          output_dir=Path(args.output_dir), label=args.label)
        print(json.dumps({k: record.get(k) for k in ("lane", "recipe", "status", "reason", "next_command", "record_path")}, indent=2))
        if record["status"] == "failed":
            return 1
        if record["status"] == "deferred" and args.require:
            return 1
        return 0
    payload = build_lane_summary(Path(args.records) if args.records else None)
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
