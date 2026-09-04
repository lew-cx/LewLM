#!/usr/bin/env python3
"""Mirror an Ollama install's model list into a LewLM-readable model shelf.

SUPERSEDED. LewLM now discovers Ollama models natively: set
`LEWLM_OLLAMA_DISCOVERY_ENABLED=true` and run `lewlm scan`. Built-in discovery
needs no placeholder files, no sync step, and no `_converted` suffix on model
ids, and it reconciles the `ollama://` namespace on every scan. See
`docs/guides/models-and-routing.md`. This script is kept for reference and as a
worked example of building manifests from outside LewLM.

LewLM discovers models by walking model roots looking for artifacts on disk.
Ollama keeps its models as content-addressed blobs, so a LewLM scan of
`~/.ollama` finds nothing, and the models an Ollama user already has stay
invisible even though LewLM's external accelerator bridge can serve them
perfectly well over `http://127.0.0.1:11434`.

This script closes that gap from the outside. For each model Ollama reports it
writes a *shelf entry*: a small directory holding a minimal but valid GGUF
header carrying the model's architecture and context window, plus a
`lewlm.conversion_output.json` sidecar that records the modality Ollama
advertises and pins `external_adapter_model_id` to the exact Ollama tag. LewLM's
existing discovery reads both and produces a correct manifest; the bridge then
resolves that manifest back to the Ollama tag and forwards the request.

The shelf entry is a card in a catalogue, not the book. It holds no weights.
Requests are executed by the Ollama daemon, and nothing here changes what LewLM
does at run time.

This is a worked example rather than a supported architecture. It leans on two
things that are not public API — the sidecar shape in
`lewlm/conversion/models.py` and the `external_adapter_model_id` metadata key
read by `lewlm/runtime/adapters/openai_compatible.py` — so pin your LewLM
version if you depend on it.

Usage
-----
    python examples/ollama_bridge_shelf.py sync --out ~/.lewlm/ollama-shelf
    python examples/ollama_bridge_shelf.py sync --out ~/.lewlm/ollama-shelf --dry-run
    python examples/ollama_bridge_shelf.py list

Then point LewLM at both the shelf and the daemon:

    export LEWLM_MODELS_DIR='["/home/you/.lewlm/ollama-shelf"]'   # JSON array of roots
    export LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true
    export LEWLM_EXTERNAL_ACCELERATOR_PROFILE=ollama_local
    export LEWLM_EXTERNAL_ACCELERATOR_BASE_URL=http://127.0.0.1:11434
    export LEWLM_EXTERNAL_ACCELERATOR_TIMEOUT_SECONDS=90
    export LEWLM_RUNTIME_PACKS='["external_accelerator"]'
    lewlm scan && lewlm list-models

Standard library only, so it runs anywhere LewLM does without extra installs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import struct
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_SHELF = Path.home() / ".lewlm" / "ollama-shelf"
SIDECAR_FILENAME = "lewlm.conversion_output.json"
STUB_FILENAME = "model.gguf"
# Marks a directory as ours, so pruning can never touch a real model a user
# happens to keep under the same root.
SHELF_MARKER = "ollama_bridge_shelf"

# GGUF metadata value type tags, from the GGUF spec. Only the two this script
# writes are named here.
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_UINT64 = 10
_GGUF_VERSION = 3


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


def fetch_models(base_url: str, *, timeout: float) -> list[dict[str, Any]]:
    """Return the daemon's model list from the native `/api/tags` endpoint.

    The native endpoint is used rather than the OpenAI-compatible `/v1/models`
    because only the former carries the architecture, context window, and
    capability list a useful manifest needs; `/v1/models` returns bare ids.
    """

    url = base_url.rstrip("/") + "/api/tags"
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise SystemExit(f"ollama returned HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise SystemExit(
            f"could not reach ollama at {url} ({exc.reason}).\n"
            "Is the daemon running? Try: ollama list",
        ) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ollama returned malformed JSON from {url}") from exc

    models = payload.get("models")
    if not isinstance(models, list):
        raise SystemExit(f"unexpected response shape from {url}: no 'models' list")
    return [model for model in models if isinstance(model, dict)]


def modality_for(model: dict[str, Any]) -> str:
    """Map Ollama's advertised capabilities onto a LewLM modality.

    This matters more than it looks: LewLM gates bridge capabilities on manifest
    modality before it probes the endpoint, so an embedding model that arrives
    labelled `text` has its working `/v1/embeddings` route refused.
    """

    capabilities = {
        str(item).casefold()
        for item in model.get("capabilities", [])
        if isinstance(item, str)
    }
    if "embedding" in capabilities:
        return "embedding"
    if "vision" in capabilities:
        return "multimodal"
    return "text"


def architecture_for(model: dict[str, Any]) -> str:
    details = model.get("details")
    details = details if isinstance(details, dict) else {}
    family = details.get("family")
    if isinstance(family, str) and family.strip():
        # GGUF architecture names are bare identifiers; Ollama's family strings
        # already follow that shape (`llama`, `nomic-bert`, `gemma3`).
        return family.strip()
    return "llama"


def context_length_for(model: dict[str, Any]) -> int | None:
    details = model.get("details")
    details = details if isinstance(details, dict) else {}
    value = details.get("context_length")
    if isinstance(value, int) and value > 0:
        return value
    return None


# --------------------------------------------------------------------------- #
# Shelf entries
# --------------------------------------------------------------------------- #


def gguf_header(*, architecture: str, context_length: int | None) -> bytes:
    """Build a minimal valid GGUF v3 header declaring no tensors.

    LewLM reads `general.architecture` and `<arch>.context_length` straight off
    the header, so those two keys are the whole payload. The result is around a
    hundred bytes and describes no weights, which is the honest shape for a card
    that only points at a model the daemon owns.
    """

    pairs: list[tuple[bytes, int, bytes]] = [
        (b"general.architecture", _GGUF_TYPE_STRING, _gguf_string(architecture)),
    ]
    if context_length is not None:
        pairs.append(
            (
                f"{architecture}.context_length".encode("utf-8"),
                _GGUF_TYPE_UINT64,
                struct.pack("<Q", context_length),
            ),
        )

    out = bytearray(b"GGUF")
    out += struct.pack("<I", _GGUF_VERSION)
    out += struct.pack("<Q", 0)  # tensor count
    out += struct.pack("<Q", len(pairs))
    for key, value_type, value in pairs:
        out += struct.pack("<Q", len(key))
        out += key
        out += struct.pack("<I", value_type)
        out += value
    return bytes(out)


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def sidecar_for(model: dict[str, Any], tag: str) -> dict[str, Any]:
    """Build the discovery sidecar that pins this entry to its Ollama tag."""

    details = model.get("details")
    details = details if isinstance(details, dict) else {}
    return {
        "source_display_name": tag,
        "source_model_id": tag,
        "display_name": tag,
        "artifact_role": "standalone",
        "modality": [modality_for(model)],
        "metadata": {
            # The key the bridge reads to resolve this manifest back to a model
            # the daemon advertises. Exact-match, so it carries the full tag.
            "external_adapter_model_id": tag,
            SHELF_MARKER: True,
            "ollama_digest": model.get("digest"),
            "ollama_capabilities": model.get("capabilities", []),
            "ollama_parameter_size": details.get("parameter_size"),
            "ollama_quantization_level": details.get("quantization_level"),
            "ollama_size_bytes": model.get("size"),
            "shelf_note": (
                "Card written by examples/ollama_bridge_shelf.py. Holds no weights; "
                "the Ollama daemon executes this model."
            ),
        },
    }


def entry_dirname(tag: str) -> str:
    """Turn an Ollama tag into a filesystem-safe directory name."""

    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-")
    return safe or "model"


def write_entry(shelf: Path, model: dict[str, Any], *, dry_run: bool) -> tuple[Path, bool]:
    """Write one shelf entry. Returns its path and whether anything changed."""

    tag = str(model.get("name") or model.get("model") or "").strip()
    if not tag:
        raise ValueError("ollama returned a model with no name")

    entry = shelf / entry_dirname(tag)
    header = gguf_header(
        architecture=architecture_for(model),
        context_length=context_length_for(model),
    )
    sidecar = json.dumps(sidecar_for(model, tag), indent=2, sort_keys=True) + "\n"

    stub_path = entry / STUB_FILENAME
    sidecar_path = entry / SIDECAR_FILENAME
    unchanged = (
        stub_path.is_file()
        and sidecar_path.is_file()
        and stub_path.read_bytes() == header
        and sidecar_path.read_text(encoding="utf-8") == sidecar
    )
    if unchanged or dry_run:
        return entry, not unchanged

    entry.mkdir(parents=True, exist_ok=True)
    stub_path.write_bytes(header)
    sidecar_path.write_text(sidecar, encoding="utf-8")
    return entry, True


def prune(shelf: Path, keep: set[str], *, dry_run: bool) -> list[Path]:
    """Remove shelf entries for models the daemon no longer lists.

    Only directories this script wrote are considered, so a real model kept
    under the same root is never touched.
    """

    removed: list[Path] = []
    if not shelf.is_dir():
        return removed
    for child in sorted(shelf.iterdir()):
        if not child.is_dir() or child.name in keep:
            continue
        if not _is_shelf_entry(child):
            continue
        removed.append(child)
        if not dry_run:
            shutil.rmtree(child)
    return removed


def _is_shelf_entry(path: Path) -> bool:
    sidecar = path / SIDECAR_FILENAME
    if not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata = payload.get("metadata")
    return isinstance(metadata, dict) and metadata.get(SHELF_MARKER) is True


def predicted_model_id(tag: str) -> str:
    """Mirror LewLM's converted-artifact id so `sync` can print usable ids.

    Kept in step with `_build_converted_model_id` in `lewlm/registry/discovery.py`.
    """

    slug = re.sub(r"[^a-z0-9]+", "-", tag.casefold()).strip("-") or "model"
    return f"{slug}_converted"


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def command_sync(args: argparse.Namespace) -> int:
    shelf = Path(args.out).expanduser()
    models = fetch_models(args.base_url, timeout=args.timeout)
    if not models:
        print(f"ollama at {args.base_url} lists no models; nothing to shelve")
        return 0

    written = 0
    kept: set[str] = set()
    rows: list[tuple[str, str, str]] = []
    for model in sorted(models, key=lambda item: str(item.get("name", ""))):
        tag = str(model.get("name") or model.get("model") or "").strip()
        entry, changed = write_entry(shelf, model, dry_run=args.dry_run)
        kept.add(entry.name)
        written += int(changed)
        context_length = context_length_for(model)
        rows.append(
            (
                predicted_model_id(tag),
                modality_for(model),
                str(context_length) if context_length else "unknown",
            ),
        )

    removed = prune(shelf, kept, dry_run=args.dry_run)
    prefix = "would write" if args.dry_run else "wrote"
    print(f"{prefix} {written} shelf entr{'y' if written == 1 else 'ies'} under {shelf}")
    if removed:
        gone = "would remove" if args.dry_run else "removed"
        print(f"{gone} {len(removed)} stale entr{'y' if len(removed) == 1 else 'ies'}: "
              + ", ".join(path.name for path in removed))

    width = max((len(row[0]) for row in rows), default=8)
    print()
    print(f"{'model id'.ljust(width)}  modality    context")
    for model_id, modality, context in rows:
        print(f"{model_id.ljust(width)}  {modality.ljust(10)}  {context}")
    print()
    print("Next:")
    print(f'  export LEWLM_MODELS_DIR=\'["{shelf}"]\'')
    print(f"  export LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true")
    print(f"  export LEWLM_EXTERNAL_ACCELERATOR_PROFILE=ollama_local")
    print(f"  export LEWLM_EXTERNAL_ACCELERATOR_BASE_URL={args.base_url}")
    print(f"  export LEWLM_EXTERNAL_ACCELERATOR_TIMEOUT_SECONDS=90")
    print('  export LEWLM_RUNTIME_PACKS=\'["external_accelerator"]\'')
    print("  lewlm scan && lewlm list-models")
    return 0


def command_list(args: argparse.Namespace) -> int:
    models = fetch_models(args.base_url, timeout=args.timeout)
    for model in sorted(models, key=lambda item: str(item.get("name", ""))):
        tag = str(model.get("name", ""))
        context_length = context_length_for(model)
        print(
            f"{tag}\t{architecture_for(model)}\t{modality_for(model)}\t"
            f"{context_length if context_length else 'unknown'}",
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama_bridge_shelf",
        description="Mirror an Ollama model list into a LewLM-readable model shelf.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Ollama endpoint (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="seconds to wait for the daemon (default: 10)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Write shelf entries for every Ollama model.")
    sync_parser.add_argument(
        "--out",
        default=str(DEFAULT_SHELF),
        help=f"shelf directory (default: {DEFAULT_SHELF})",
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without touching the filesystem",
    )
    sync_parser.set_defaults(handler=command_sync)

    list_parser = subparsers.add_parser("list", help="Show what Ollama advertises, without writing.")
    list_parser.set_defaults(handler=command_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
