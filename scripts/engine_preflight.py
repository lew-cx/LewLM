#!/usr/bin/env python3
"""Host compatibility preflight for the pinned external-engine recipes.

Runs *before* an image is pulled or a model downloaded, and answers one
question per check: can this host run the recipe as pinned? It reads
``nvidia-smi``, ``docker info``, and a loopback port; it never installs,
pulls, starts, or stops anything. Every requirement it checks comes from the
recipe's pinned image (CUDA version -> minimum driver; the image's compiled
architecture list; a free host-loopback port for LewLM's endpoint rule).

    python scripts/engine_preflight.py --recipe vllm
    python scripts/engine_preflight.py --recipe sglang --json
    python scripts/engine_preflight.py --min-driver 580 --port 8000 --image repo@sha256:...

Exit status: 0 when every check passed or was skipped with a stated reason,
1 when any check failed. The JSON report is the artifact to keep with the
validation record.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence


Runner = Callable[[Sequence[str]], tuple[int, str, str]]
PortProbe = Callable[[str, int], bool]


@dataclass(frozen=True)
class RecipeRequirements:
    """What a pinned recipe needs from the host. Values are recorded, not guessed."""

    recipe: str
    image: str | None
    # Minimum NVIDIA driver major version for the image's CUDA toolkit. CUDA 13.x
    # images need R580+ when run normally (vLLM/SGLang install docs at their pins).
    min_driver_major: int | None
    # Compute capabilities the image's kernels were compiled for. ``None`` means
    # the recipe does not publish a list and the check is skipped, not passed.
    supported_compute_capabilities: tuple[str, ...] | None
    min_free_vram_mib: int
    host_port: int | None
    notes: tuple[str, ...] = ()


# Every value below was read from the pinned image or its upstream docs at the
# recipe's commit; see the recipe README and the step validation record.
RECIPES: dict[str, RecipeRequirements] = {
    "vllm": RecipeRequirements(
        recipe="vllm",
        image="vllm/vllm-openai@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1",
        min_driver_major=580,
        supported_compute_capabilities=("7.5", "8.0", "8.6", "8.9", "9.0", "10.0", "12.0"),
        min_free_vram_mib=4096,
        host_port=8000,
        notes=(
            "vllm/vllm-openai:v0.29.0 (build commit 98dff2a8) is a CUDA 13.0.2 image; TORCH_CUDA_ARCH_LIST from the image config.",
            "R535/R570 hosts can only use the image with VLLM_ENABLE_CUDA_COMPATIBILITY=1 on select datacenter GPUs; not part of this recipe.",
        ),
    ),
    "sglang": RecipeRequirements(
        recipe="sglang",
        image="lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9",
        min_driver_major=580,
        supported_compute_capabilities=None,
        min_free_vram_mib=4096,
        host_port=30000,
        notes=(
            "lmsysorg/sglang:v0.5.19-cu130 (build commit 0bcd8223) is a CUDA 13.0.3 image.",
            "SGLang does not publish a compiled-architecture list for its kernel wheels at this pin; that check is skipped, not passed.",
        ),
    ),
    "exllamav3-tabby": RecipeRequirements(
        recipe="exllamav3-tabby",
        image="ghcr.io/theroyallab/tabbyapi@sha256:10bfcf9d27d1b3a5c7ada786f814b9da43fadad35643362f2fe0816082893172",
        min_driver_major=570,
        supported_compute_capabilities=None,
        min_free_vram_mib=4096,
        host_port=5000,
        notes=(
            "TabbyAPI image built from commit 53da7919 on CUDA 12.8.1; CUDA 12.8 needs an R570+ driver on Linux.",
        ),
    ),
}


@dataclass
class CheckResult:
    name: str
    status: str  # pass | fail | skip
    detail: str
    observed: dict[str, object] = field(default_factory=dict)


def _default_runner(command: Sequence[str]) -> tuple[int, str, str]:
    if shutil.which(command[0]) is None:
        return 127, "", f"{command[0]}: not found on PATH"
    completed = subprocess.run(list(command), capture_output=True, text=True, timeout=60)
    return completed.returncode, completed.stdout, completed.stderr


def _default_port_probe(host: str, port: int) -> bool:
    """True when the port is free on ``host`` (bind succeeds)."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _driver_major(driver_version: str) -> int | None:
    head = driver_version.strip().split(".")[0]
    return int(head) if head.isdigit() else None


def check_image_pin(requirements: RecipeRequirements) -> CheckResult:
    image = requirements.image
    if image is None:
        return CheckResult("image_pin", "skip", "No image declared for this recipe.")
    if "@sha256:" in image and len(image.rsplit("@sha256:", 1)[1]) == 64:
        return CheckResult("image_pin", "pass", "Image is pinned by digest.", {"image": image})
    return CheckResult("image_pin", "fail", "Image is not pinned by digest; a tag can move under the recipe.", {"image": image})


def check_container_runtime(runner: Runner) -> CheckResult:
    code, out, err = runner(["docker", "info", "--format", "{{json .Runtimes}}"])
    if code != 0:
        return CheckResult(
            "container_runtime", "fail",
            "Docker daemon is not reachable; the recipe runs the engine as a container.",
            {"stderr": err.strip()[:300]},
        )
    try:
        runtimes = json.loads(out or "{}")
    except json.JSONDecodeError:
        runtimes = {}
    names = sorted(runtimes) if isinstance(runtimes, dict) else []
    if "nvidia" in names:
        return CheckResult("container_runtime", "pass", "Docker reachable; NVIDIA container runtime registered.", {"runtimes": names})
    return CheckResult(
        "container_runtime", "fail",
        "Docker is reachable but no `nvidia` runtime is registered; install the NVIDIA Container Toolkit and run `nvidia-ctk runtime configure --runtime=docker`.",
        {"runtimes": names},
    )


def check_gpu(requirements: RecipeRequirements, runner: Runner) -> list[CheckResult]:
    code, out, err = runner([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,compute_cap,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ])
    if code != 0:
        detail = "nvidia-smi is unavailable or failed; no NVIDIA GPU/driver is usable on this host."
        return [CheckResult("gpu_present", "fail", detail, {"stderr": err.strip()[:300]})]
    gpus: list[dict[str, object]] = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        index, name, driver, compute_cap, total, used = parts[:6]
        gpus.append({
            "index": index, "name": name, "driver_version": driver, "compute_capability": compute_cap,
            "memory_total_mib": int(total) if total.isdigit() else None,
            "memory_used_mib": int(used) if used.isdigit() else None,
        })
    if not gpus:
        return [CheckResult("gpu_present", "fail", "nvidia-smi returned no GPU rows.", {"stdout": out.strip()[:300]})]
    results = [CheckResult("gpu_present", "pass", f"{len(gpus)} GPU(s) visible.", {"gpus": gpus})]
    gpu = gpus[0]

    driver_major = _driver_major(str(gpu["driver_version"]))
    if requirements.min_driver_major is None:
        results.append(CheckResult("driver_version", "skip", "Recipe declares no minimum driver.", {"driver_version": gpu["driver_version"]}))
    elif driver_major is None:
        results.append(CheckResult("driver_version", "fail", "Could not parse the driver version.", {"driver_version": gpu["driver_version"]}))
    elif driver_major >= requirements.min_driver_major:
        results.append(CheckResult(
            "driver_version", "pass",
            f"Driver {gpu['driver_version']} satisfies R{requirements.min_driver_major}+.",
            {"driver_version": gpu["driver_version"], "min_driver_major": requirements.min_driver_major},
        ))
    else:
        results.append(CheckResult(
            "driver_version", "fail",
            f"Driver {gpu['driver_version']} is older than R{requirements.min_driver_major}; the pinned image's CUDA toolkit needs a newer driver.",
            {"driver_version": gpu["driver_version"], "min_driver_major": requirements.min_driver_major},
        ))

    compute_cap = str(gpu["compute_capability"])
    if requirements.supported_compute_capabilities is None:
        results.append(CheckResult(
            "compute_capability", "skip",
            "Recipe does not publish a compiled-architecture list; verify kernel support on first start.",
            {"compute_capability": compute_cap},
        ))
    elif compute_cap in requirements.supported_compute_capabilities:
        results.append(CheckResult(
            "compute_capability", "pass", f"SM {compute_cap} is in the image's compiled list.",
            {"compute_capability": compute_cap, "supported": list(requirements.supported_compute_capabilities)},
        ))
    else:
        results.append(CheckResult(
            "compute_capability", "fail",
            f"SM {compute_cap} is not in the image's compiled list; kernels would be missing or JIT-compiled.",
            {"compute_capability": compute_cap, "supported": list(requirements.supported_compute_capabilities)},
        ))

    total = gpu["memory_total_mib"]
    used = gpu["memory_used_mib"]
    if isinstance(total, int) and isinstance(used, int):
        free = total - used
        status = "pass" if free >= requirements.min_free_vram_mib else "fail"
        results.append(CheckResult(
            "free_vram", status,
            f"{free} MiB free of {total} MiB; recipe wants at least {requirements.min_free_vram_mib} MiB.",
            {"free_mib": free, "total_mib": total, "min_free_vram_mib": requirements.min_free_vram_mib},
        ))
    else:
        results.append(CheckResult("free_vram", "fail", "nvidia-smi did not report memory totals.", {}))
    return results


def check_host_port(requirements: RecipeRequirements, port_probe: PortProbe) -> CheckResult:
    port = requirements.host_port
    if port is None:
        return CheckResult("host_port", "skip", "Recipe declares no host port.")
    if port_probe("127.0.0.1", port):
        return CheckResult("host_port", "pass", f"127.0.0.1:{port} is free for the engine's loopback publish.", {"port": port})
    return CheckResult(
        "host_port", "fail",
        f"127.0.0.1:{port} is already bound; stop the other server or change the recipe's published port and LewLM's endpoint URL together.",
        {"port": port},
    )


def run_preflight(
    requirements: RecipeRequirements,
    *,
    runner: Runner | None = None,
    port_probe: PortProbe | None = None,
) -> dict[str, object]:
    # Resolved at call time so a test can substitute the host probes.
    runner = runner or _default_runner
    port_probe = port_probe or _default_port_probe
    checks: list[CheckResult] = [check_image_pin(requirements), check_container_runtime(runner)]
    checks.extend(check_gpu(requirements, runner))
    checks.append(check_host_port(requirements, port_probe))
    failed = [check.name for check in checks if check.status == "fail"]
    skipped = [check.name for check in checks if check.status == "skip"]
    return {
        "recipe": requirements.recipe,
        "image": requirements.image,
        "result": "fail" if failed else "pass",
        "failed": failed,
        "skipped": skipped,
        "checks": [asdict(check) for check in checks],
        "notes": list(requirements.notes),
    }


def _requirements_from_args(args: argparse.Namespace) -> RecipeRequirements:
    base = RECIPES.get(args.recipe) if args.recipe else None
    if base is None and args.recipe:
        raise SystemExit(f"Unknown recipe {args.recipe!r}; known: {', '.join(sorted(RECIPES))}")
    caps = tuple(part.strip() for part in args.supported_compute_capabilities.split(",") if part.strip()) if args.supported_compute_capabilities else None
    return RecipeRequirements(
        recipe=(base.recipe if base else "custom"),
        image=args.image if args.image is not None else (base.image if base else None),
        min_driver_major=args.min_driver if args.min_driver is not None else (base.min_driver_major if base else None),
        supported_compute_capabilities=caps if caps is not None else (base.supported_compute_capabilities if base else None),
        min_free_vram_mib=args.min_free_vram_mib if args.min_free_vram_mib is not None else (base.min_free_vram_mib if base else 0),
        host_port=args.port if args.port is not None else (base.host_port if base else None),
        notes=base.notes if base else (),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", choices=sorted(RECIPES), default=None, help="Use a pinned recipe's requirements.")
    parser.add_argument("--image", default=None, help="Override the image reference to check for a digest pin.")
    parser.add_argument("--min-driver", type=int, default=None, help="Minimum NVIDIA driver major version.")
    parser.add_argument("--supported-compute-capabilities", default=None, help="Comma-separated SM list, e.g. 7.5,8.0,8.6.")
    parser.add_argument("--min-free-vram-mib", type=int, default=None)
    parser.add_argument("--port", type=int, default=None, help="Host loopback port the recipe publishes.")
    parser.add_argument("--json", action="store_true", help="Print the JSON report instead of a table.")
    parser.add_argument("--output", default=None, help="Also write the JSON report here.")
    args = parser.parse_args(argv)
    if args.recipe is None and args.min_driver is None and args.image is None and args.port is None:
        parser.error("pass --recipe or explicit requirements")

    report = run_preflight(_requirements_from_args(args))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"recipe: {report['recipe']}   result: {report['result']}")
        for check in report["checks"]:
            print(f"  [{check['status']:4}] {check['name']}: {check['detail']}")
        for note in report["notes"]:
            print(f"  note: {note}")
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
