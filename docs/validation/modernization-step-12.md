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

## Clean-environment quickstarts and the real-engine sequence (2026-09-21)

Three of the four deferrals below were executed on the same host. Each ran in
a fresh `python3.11` venv installed from this checkout, with `LEWLM_DATA_DIR`
isolated so the user's registry and caches were never touched.

| Quickstart | Evidence | Result |
| --- | --- | --- |
| **Ollama-only**, exactly as `docs/guides/models-and-routing.md` documents it (singular `LEWLM_EXTERNAL_ACCELERATOR_*` + `LEWLM_OLLAMA_DISCOVERY_ENABLED`), against Ollama 0.33.2 with ten pulled models | [ollama-quickstart.json](evidence/modernization-step-12/ollama-quickstart.json) | **passed** — lean install 5.6 s / 25 packages / no engine or conversion packages; `lewlm scan` 0.9 s → 10 manifests (`ollama://…`, text/multimodal/embedding modalities from the daemon); one chat through `llama3-latest-365c0bd3c000` in 2.77 s, `ok`, `finish_reason: stop`, `engine_profile: ollama_local`, `execution_locality: host_local` |
| **llama.cpp-only**, `pip install "lewlm[llamacpp_runtime] @ file://…"` | [llamacpp-quickstart.json](evidence/modernization-step-12/llamacpp-quickstart.json) | **passed** — 50.0 s wall including a Metal source build of `llama-cpp-python 0.3.35` (PyPI ships no macOS wheel; cmake 4.3 + Xcode CLT on the host); 30 packages, **no torch/transformers**; doctor `[core_only, gguf_fallback_backend]`; `lewlm scan` 0.5 s over the user's models folder → the two Gemma-4 GGUFs `runnable` on `llamacpp` (MLX/HF folders discovered but not runnable here, as expected without those extras); `lewlm serve` healthy in 0.63 s with `chat` ready; first chat 3.3 s including the 5 GB Q8 load on Metal (`ggml_metal_device_init … Apple M2 Max`), `ok`, measured usage; warm chat 0.2 s |
| **Enable → serve → outage → alias → restart → disable on a real engine** (oMLX `b45fb7e` re-installed from the recipe lock in 87.4 s, model Qwen2.5-0.5B-Instruct-4bit @ `a5339a4`, `run-omlx.sh`; LewLM in the llama.cpp venv with `explicit_alias` → the Gemma GGUF) | [omlx-rollout-rollback.json](evidence/modernization-step-12/omlx-rollout-rollback.json) | **passed** — oMLX ready 5.69 s cold / 2.48 s restart; scan registered `qwen2-5-0-5b-instruct-4bit-omlx-6fed8f48` (the step-05 id); doctor `state: ready`, next command `lewlm serve`; chat via `omlx` 200 in 1.7 s; **SIGKILL after the first content chunk** → the stream ended in 0.21 s with `finish_reason: error`, `error.code: runtime_unavailable`, `partial_output: true`, `[DONE]`, no replay, no mid-stream substitution; rescan kept the model `stale` (note names the endpoint), health `ok` with the engine `stale`, the model still listed; the next request was served by the alias on `llamacpp` with `fallback_from_model_id` set; restart + rescan → `advertised`, chat back on `omlx` with no fallback; `"enabled": false` + restart → scan removed exactly one manifest, the oMLX id is `404 model_not_found`, the alias path still serves, the data directory's entry count is unchanged (caches kept), doctor reports `disabled` with the re-enable command |

Two things the runs surfaced:

- **`GET /v1/health` was generating on the bridge.** Readiness asked the
  bridge whether each advertised model supported vision/embeddings/etc., and
  the bridge answered by sending a real request — against Ollama that loaded
  every advertised model before health could respond (21.7 s cold with ten
  models; unbounded with large ones). Fixed in `a12d23d`: readiness paths are
  passive (advertised = candidate, evidence `discovered`; a request or the
  new explicit `probe_manifest_capability` observes and is then reported).
  Re-measured on the same quickstart: serve → healthy **0.82 s**, Ollama
  received one `GET /v1/models` and loaded nothing; the first embeddings
  request still verified and served in 0.34 s. A bootstrap-level regression
  now fails on the old code.
- Observations, not changed: a symlinked model directory inside a models
  root is not scanned (`os.walk` without `followlinks`); `lewlm doctor` from a
  wheel install shows `recipe_status: null` because recipe statuses are read
  from a source checkout's `examples/backends/compatibility.json`.

## Deferred

| Item | Missing prerequisite | Command / expected observation |
| --- | --- | --- |
| Old full/conversion path from a clean environment | torch (CPU lock) in a clean venv | `pip install -e ".[llamacpp]"` in a clean venv, convert and quantize the small fixture (`docker-full` CI job does this in a container on every push) |

## Remaining work after step 12

| Lane / item | Status | Where |
| --- | --- | --- |
| Apple Silicon (oMLX) | **validated** for oMLX `b45fb7e` + Qwen2.5-0.5B-Instruct-4bit on the step-05 host; tools inconclusive on that model; vision/embeddings/rerank not probed; the enable/outage/alias/restart/disable sequence re-run for real on 2026-09-21 (above) | step 05 record, `examples/backends/compatibility.json` |
| llama.cpp-only and Ollama-only clean-environment quickstarts | **passed** on this host on 2026-09-21 (above) | `evidence/modernization-step-12/` |
| Linux NVIDIA: vLLM, SGLang, ExLlamaV3/TabbyAPI, llama.cpp CUDA | **deferred** — complete, digest-pinned recipes with every argument verified at the pinned commit; nothing executed | recipe READMEs, `scripts/backend_lanes.py run --lane linux_nvidia --recipe <name>` |
| Linux CPU: lean install, portable GGUF, image boot, conversion image, rebuild evidence | **deferred** here; the `docker-bridge` and `docker-full` CI jobs cover the container parts on every push | step 03 record, `scripts/docker/measure_rebuild.sh` |
| Native Windows, Windows + WSL2 | **deferred** — the CI matrix runs the engine-free contract on Windows; native engine and WSL connectivity proof is separate | `scripts/backend_lanes.py detect` on the target |
| Step 09 tuning measurements (CPU threads, CUDA offload, graph capture) | **deferred** to their lanes; no tuning result is claimed | step 09 record |
| Chap UI checklist (13 items) | **pending** — belongs to Chap's repository against a tagged LewLM; item 13 (events reconnect) was added with the G13 fix | `docs/guides/chap-validation.md` |

Nothing above is labelled universally supported. An adapter with a `deferred`
recipe is shipped as experimental with that status visible in `lewlm doctor`,
the compatibility manifest, and the release manifest's `backend_lanes`.

## Rollback

Revert the step commit. Doctor's `external_engines`, the fixture controls,
and the documentation are additive. The routing change affects only an
endpoint whose last inventory read failed: it now fails before submission
(and can fall back) instead of at the transport; the status code and error
code are the same.
