#!/usr/bin/env python3
"""Validate backend recipe evidence or export its JSON Schema; no engine imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lewlm.utils.backend_compatibility import compatibility_schema, load_backend_compatibility


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("examples/backends/compatibility.json"))
    parser.add_argument("--schema-output", type=Path)
    args = parser.parse_args()
    manifest = load_backend_compatibility(args.manifest)
    if args.schema_output:
        args.schema_output.write_text(json.dumps(compatibility_schema(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"recipes": len(manifest.recipes), "statuses": {r.profile: r.status for r in manifest.recipes}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
