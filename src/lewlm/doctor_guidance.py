"""Doctor guidance for external engines: installed, reachable, working, and what to run next.

`lewlm doctor` is an explicit operator action, so it may read each configured
endpoint's `/v1/models` once. It still never installs, starts, stops, or
reconfigures an engine; the guidance names the command the operator runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lewlm.core.errors import LewLMError

# Bridge profile -> the pinned recipe directory that sets that engine up.
RECIPE_FOR_PROFILE = {
    "omlx": "omlx",
    "vllm_local": "vllm",
    "sglang_local": "sglang",
    "exllamav3_tabby": "exllamav3-tabby",
}


def _compatibility_recipes(repo_root: Path | None) -> dict[str, dict[str, Any]]:
    if repo_root is None:
        return {}
    path = repo_root / "examples" / "backends" / "compatibility.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {recipe.get("profile"): recipe for recipe in payload.get("recipes", []) if isinstance(recipe, dict)}


def engine_guidance(services, *, repo_root: Path | None = None, probe: bool = True) -> dict[str, Any]:
    """Per-endpoint state plus the next command; ``probe=False`` uses cached inventory only."""

    recipes = _compatibility_recipes(repo_root)
    registered = services.model_registry.list_manifests()
    registered_by_endpoint: dict[str, int] = {}
    for manifest in registered:
        endpoint_id = manifest.metadata.get("external_endpoint_id")
        if isinstance(endpoint_id, str):
            registered_by_endpoint[endpoint_id] = registered_by_endpoint.get(endpoint_id, 0) + 1

    engines: list[dict[str, Any]] = []
    next_steps: list[str] = []
    for endpoint_id, runtime in sorted(services.runtime_catalog.endpoint_runtimes().items()):
        snapshot_method = getattr(runtime, "endpoint_snapshot", None)
        if not callable(snapshot_method):
            continue
        endpoint = getattr(runtime, "endpoint", None)
        if endpoint_id == "legacy-default" and not getattr(endpoint, "enabled", True):
            # The synthesized legacy endpoint when LEWLM_EXTERNAL_ACCELERATOR_*
            # is not enabled: nothing was configured, so nothing to report.
            continue
        profile = str(getattr(endpoint, "profile", snapshot_method().get("profile")))
        recipe_dir = RECIPE_FOR_PROFILE.get(profile)
        recipe = recipes.get(profile)
        entry: dict[str, Any] = {
            "endpoint_id": endpoint_id,
            "profile": profile,
            "enabled": bool(getattr(endpoint, "enabled", True)),
            "base_url": getattr(endpoint, "base_url", None),
            "credential_env": getattr(endpoint, "api_key_env", None),
            "recipe": f"examples/backends/{recipe_dir}/" if recipe_dir else None,
            "recipe_status": recipe.get("status") if recipe else None,
            "registered_model_count": registered_by_endpoint.get(endpoint_id, 0),
        }
        if not entry["enabled"]:
            entry.update({"reachable": None, "advertised_model_count": 0, "state": "disabled",
                          "next_command": f"set \"enabled\": true for endpoint `{endpoint_id}` in LEWLM_EXTERNAL_ENDPOINTS and restart, or remove the entry to roll it back"})
            engines.append(entry)
            next_steps.append(f"endpoint `{endpoint_id}` is disabled (rolled back): {entry['next_command']}")
            continue
        if not runtime.is_available():
            entry.update({"reachable": False, "advertised_model_count": 0, "state": "blocked",
                          "error": runtime.availability_reason(),
                          "next_command": (f"export {entry['credential_env']}=<the engine's inference key>" if entry["credential_env"]
                                           and "environment" in str(runtime.availability_reason() or "") else "fix the configuration named in `error`")})
            engines.append(entry)
            next_steps.append(f"endpoint `{endpoint_id}`: {entry['next_command']}")
            continue
        error = None
        if probe:
            try:
                records = runtime.advertised_model_records(refresh=True)
                advertised = len(records)
                reachable = True
            except LewLMError as exc:
                advertised, reachable, error = 0, False, str(exc)
        else:
            snapshot = snapshot_method()
            advertised = len(snapshot.get("advertised_model_ids") or ())
            reachable = snapshot.get("inventory_state") == "advertised"
            error = snapshot.get("inventory_error")
        if reachable and advertised and entry["registered_model_count"] < advertised:
            state, command = "advertised_not_registered", "lewlm scan"
        elif reachable and advertised:
            state, command = "ready", "lewlm serve   (then chat with the registered model id from `lewlm list-models`)"
        elif reachable:
            state, command = "no_models", (f"load a model in the engine (see {entry['recipe']}README.md), then `lewlm scan`" if entry["recipe"] else "load a model in the engine, then `lewlm scan`")
        else:
            state = "unreachable"
            command = (f"start the engine: see {entry['recipe']}README.md (`docker compose up -d` or the run script), then `lewlm scan`"
                       if entry["recipe"] else f"start the server at {entry['base_url']}, then `lewlm scan`")
        entry.update({"reachable": reachable, "advertised_model_count": advertised, "state": state, "error": error, "next_command": command})
        engines.append(entry)
        if state != "ready":
            next_steps.append(f"endpoint `{endpoint_id}`: {command}")

    if not engines:
        next_steps.append("no external engines configured — optional: pick a recipe under examples/backends/, follow its README, set LEWLM_EXTERNAL_ENDPOINTS, then `lewlm doctor`")
    elif not next_steps:
        next_steps.append("every configured engine is reachable and registered — `lewlm serve`")

    return {
        "engines": engines,
        "recipes": [
            {"profile": profile, "status": recipe.get("status"), "recipe": f"examples/backends/{RECIPE_FOR_PROFILE.get(profile, profile)}/"}
            for profile, recipe in sorted(recipes.items())
        ],
        "next_steps": next_steps,
        "rollback": "to disable an engine: set its entry's \"enabled\": false (or remove it) in LEWLM_EXTERNAL_ENDPOINTS and restart; other models, caches, and Ollama state are untouched and the disabled endpoint's advertised entries leave the registry at the next scan; see docs/operations/backends/rollout-and-rollback.md",
    }
