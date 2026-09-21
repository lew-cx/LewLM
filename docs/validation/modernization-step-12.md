# Modernization step 12 validation — rollout with an easy rollback

Implemented on 2026-09-18 on the baseline Apple Silicon host. The rollout
path is: pick a pinned recipe, follow it, configure one endpoint entry, run
`lewlm doctor`; the rollback is one entry. Both are documented, both are
proven over the public HTTP API against fake engines, and doctor says what to
run next at every state. The remaining hardware and Chap UI work is listed at
the end, unchanged in status.

## What changed

| Area | Change |
| --- | --- |
| Doctor | `lewlm doctor` gains `external_engines`: per configured endpoint — enabled, reachable (one explicit model-list read; `--no-probe` for cached state), advertised vs registered counts, recipe and its manifest status, and the single next command (`start the engine: see examples/backends/<recipe>/README.md …`, `lewlm scan`, `export <KEY>=…`, `lewlm serve`, or re-enable/remove for a rolled-back endpoint); plus `next_steps` and a `rollback` line. The synthesized, disabled `legacy-default` entry is not reported on a fresh install. |
| Routing | A stale endpoint (an earlier successful read, the latest failed, retry window not yet passed) is no longer a routing candidate: routing sees the outage before submitting, so an explicit alias applies and the caller gets the same `503 runtime_unavailable` naming the endpoint whether the failure was found by scan or by the retry. A never-successful read keeps its empty list and existing behaviour. |
| Docs | `docs/operations/backends/rollout-and-rollback.md`: recipe table with statuses, endpoint configuration, doctor reading, the legacy → named-endpoint migration (with the id change called out), the rollback and exactly what it keeps, the outage behaviour, and rolling back the modernization itself. Linked from the README, the configuration reference, the CLI reference, and the nav. |
| Fixture | `FakeOpenAIEngine.die_after_frames` (deterministic mid-stream death for in-process clients); a stopped or dying engine now shuts the socket down so clients see EOF immediately instead of at their read timeout. |

## Portable results on this host

| Command | Result |
| --- | --- |
| `tests/integration/test_rollout_rollback.py::test_enable_serve_lose_fall_back_and_disable` — enable and serve through the new endpoint; engine stopped + rescan → `503` naming the endpoint, the other path still serves, health `ok` with the engine `failed`/`stale`, model still listed; with `explicit_alias` an active stream that loses its engine ends with `finish_reason: "error"` + `partial_output: true` + `[DONE]`, is never replayed and never substituted mid-stream, and the *next* request is served by the alias with `fallback_from_model_id`; disabling the endpoint keeps the fallback path, the other models, and the caches (artifact count unchanged), the disabled endpoint's ids return `404 model_not_found`, doctor reports `disabled` with the re-enable command and the fallback as `ready` | **passed** |
| `…::test_legacy_singular_settings_still_work_and_migrate_to_one_endpoint` — the singular variables synthesize `legacy-default`; the named form registers the same upstream models under new ids; both forms at once are rejected | **passed** |
| Clean lean install: `python3.11 -m venv && pip install /Users/lew/LewLM` ([evidence](evidence/modernization-step-12/lean-clean-install.json)) | 6.4 s, **25 packages, no torch/transformers/llama-cpp/engine packages**; `lewlm doctor --json` from that venv in 0.96 s with `active_profile_ids: [core_only]` and the "no external engines configured" next step |
| Adapter, inventory, endpoints, routing, three profiles, latency contract, inventory availability, Chap contract, CLI suites after the routing change | 111 + 52 passed |

## Deferred

| Item | Missing prerequisite | Command / expected observation |
| --- | --- | --- |
| llama.cpp-only quickstart from a clean environment | a clean environment able to install `llama-cpp-python` (a prebuilt wheel for the OS/Python, or cmake + a compiler) and a small GGUF; on this Mac that is a Metal source build measured in minutes and not the CPU lane | `python -m venv q && q/bin/pip install "lewlm[llamacpp_runtime] @ file://$PWD"`, drop a GGUF into `~/.lewlm/models`, `lewlm scan`, `lewlm serve`, one chat; record install time and that no torch/transformers were pulled |
| Old full/conversion path from a clean environment | same plus torch (CPU lock) | `pip install -e ".[llamacpp]"` in a clean venv, convert and quantize the small fixture (`docker-full` CI job does this in a container on every push) |
| Ollama-only quickstart from a clean environment | a running Ollama daemon with one pulled model (none on this host) | `LEWLM_OLLAMA_DISCOVERY_ENABLED=true lewlm scan`, `lewlm serve`, one chat through `ollama://<model>` |
| Enable/serve/stop/fallback on a real engine | any hardware lane (Apple Silicon: re-install the oMLX recipe) | the same sequence as the portable test, by hand: recipe up → chat → `docker compose stop` / kill → 503 → alias → `docker compose start` → `lewlm scan` → chat |

## Remaining work after step 12

| Lane / item | Status | Where |
| --- | --- | --- |
| Apple Silicon (oMLX) | **validated** for oMLX `b45fb7e` + Qwen2.5-0.5B-Instruct-4bit on the step-05 host; tools inconclusive on that model; vision/embeddings/rerank not probed | step 05 record, `examples/backends/compatibility.json` |
| Linux NVIDIA: vLLM, SGLang, ExLlamaV3/TabbyAPI, llama.cpp CUDA | **deferred** — complete, digest-pinned recipes with every argument verified at the pinned commit; nothing executed | recipe READMEs, `scripts/backend_lanes.py run --lane linux_nvidia --recipe <name>` |
| Linux CPU: lean install, portable GGUF, image boot, conversion image, rebuild evidence | **deferred** here; the `docker-bridge` and `docker-full` CI jobs cover the container parts on every push | step 03 record, `scripts/docker/measure_rebuild.sh` |
| Native Windows, Windows + WSL2 | **deferred** — the CI matrix runs the engine-free contract on Windows; native engine and WSL connectivity proof is separate | `scripts/backend_lanes.py detect` on the target |
| Step 09 tuning measurements (CPU threads, CUDA offload, graph capture) | **deferred** to their lanes; no tuning result is claimed | step 09 record |
| Chap UI checklist (12 items) | **pending** — belongs to Chap's repository against a tagged LewLM | `docs/guides/chap-validation.md` |

Nothing above is labelled universally supported. An adapter with a `deferred`
recipe is shipped as experimental with that status visible in `lewlm doctor`,
the compatibility manifest, and the release manifest's `backend_lanes`.

## Rollback

Revert the step commit. Doctor's `external_engines`, the fixture controls,
and the documentation are additive. The routing change affects only an
endpoint whose last inventory read failed: it now fails before submission
(and can fall back) instead of at the transport; the status code and error
code are the same.
