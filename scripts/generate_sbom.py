#!/usr/bin/env python3
"""Generate a lightweight JSON software bill of materials for the current environment."""

from __future__ import annotations

import json
from importlib.metadata import distributions


def build_sbom() -> dict[str, object]:
    packages = []
    for distribution in sorted(distributions(), key=lambda item: item.metadata.get("Name", "").casefold()):
        packages.append(
            {
                "name": distribution.metadata.get("Name"),
                "version": distribution.version,
                "summary": distribution.metadata.get("Summary"),
            },
        )
    return {
        "format": "lewlm-sbom-v1",
        "package_count": len(packages),
        "packages": packages,
    }


if __name__ == "__main__":
    print(json.dumps(build_sbom(), indent=2))
