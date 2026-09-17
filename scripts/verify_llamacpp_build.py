#!/usr/bin/env python3
"""Verify that the installed llama-cpp-python build matches the intended flavor.

A successful ``pip install`` proves that a wheel was written, not that it is
the right one: a prebuilt CPU wheel installs cleanly into a CUDA image, and a
source build silently drops an accelerator when its toolkit is missing. This
script asks the installed backend what it actually exposes (through LewLM's
existing build-flavor detection) and fails when that disagrees with the
expected flavor. Both container images run it after installing the
application wheel; native installs can run it too.

Exit codes: 0 the build matches, 1 it does not, 2 the backend is missing or
cannot be loaded on this host.

Usage:
    python scripts/verify_llamacpp_build.py --expect cpu
    python scripts/verify_llamacpp_build.py --expect gpu --hint cuda
    python scripts/verify_llamacpp_build.py --expect any --json
"""

from __future__ import annotations

import argparse
import json
import sys


def evaluate(flavor, *, expect: str, hint: str | None) -> tuple[int, str]:
    """Return ``(exit_code, message)`` for a detected build flavor."""

    if not flavor.installed or flavor.detection_state == "unavailable":
        return 2, f"llama-cpp-python is not usable on this host: {flavor.reason}"

    offload = flavor.gpu_offload_supported
    hints = ", ".join(flavor.accelerator_hints) or "none"
    described = (
        "GPU-offload-capable" if offload is True else "CPU-only" if offload is False else "unknown-offload"
    )
    summary = f"installed llama.cpp build is {described} (accelerator hints: {hints})"

    if expect == "cpu" and offload is not False:
        return 1, f"expected a CPU-only build but {summary}"
    if expect == "gpu":
        if offload is not True:
            return 1, f"expected a GPU-offload-capable build but {summary}"
        if hint and hint not in flavor.accelerator_hints:
            return 1, f"expected accelerator hint {hint!r} but {summary}"
    return 0, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--expect",
        choices=("cpu", "gpu", "any"),
        default="any",
        help="Flavor the installation was supposed to produce (default: any, only checks loadability).",
    )
    parser.add_argument(
        "--hint",
        default=None,
        help="With --expect gpu, additionally require this accelerator hint (e.g. cuda, vulkan, metal).",
    )
    parser.add_argument("--json", action="store_true", help="Print the full detected flavor as JSON.")
    args = parser.parse_args(argv)

    from lewlm.runtime.llamacpp.build_flavor import detect_llamacpp_build_flavor

    flavor = detect_llamacpp_build_flavor()
    code, message = evaluate(flavor, expect=args.expect, hint=args.hint)
    if args.json:
        payload = flavor.model_dump(mode="json")
        payload["verdict"] = {"expect": args.expect, "hint": args.hint, "exit_code": code, "message": message}
        print(json.dumps(payload, indent=2))
    else:
        print(("OK: " if code == 0 else "FAIL: ") + message, file=sys.stdout if code == 0 else sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
