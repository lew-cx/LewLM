# Modernization step 09 validation — runtime latency contract

Implemented on 2026-09-18 on the baseline Apple Silicon host. Rule 9 of the
roadmap allows tuning only engines with passing hardware evidence; on this
host that is oMLX (step 05) and the native Apple paths. The portable contract
that makes tuning usable — measured presets with a fingerprint, no middleware
waits in front of engines, explicit-only warming, startup phases — is
implemented and tested. The CPU and CUDA tuning measurements themselves are
**deferred** with exact commands; nothing below claims a tuning result.

## What changed

| Item | Change |
| --- | --- |
| 09.1 Startup phases | `RuntimeInstanceMetadata.ready_at` is set when bootstrap finishes; bridges record `first_advertised_at` on their first successful inventory read; `GET /v1/runtime` gains `startup` with `lewlm_ready_seconds`, per-endpoint engine `state`/`inventory_age_seconds`/`advertised_model_count`/`first_advertised_at` (cached inventory, no probe), and process-local `warm_models`/`loading_models` (residency, never upstream). Health and model listing keep answering while an engine is down. |
| 09.2 Presets | `SERVING_PROFILE_PRESETS = ("interactive", "throughput")`; `LEWLM_SERVING_PROFILE_PRESET` picks which one a deployment applies; `lewlm autotune --preset` / `POST /v1/benchmarks/autotune.preset` select the objective (`latency_first` or the new `throughput_first` sort key over the concurrent-throughput measurement autotune already takes). Recommendations are stored per preset (the default keeps the pre-preset key so old profiles stay addressable; `throughput` is a separate record and never falls back). |
| 09.2 Fingerprint | `serving_profile_fingerprint()` = host, runtime name, engine profile and server (bridges), model revision (manifest fingerprint), precision, workload class, preset. Stored on every new `ServingProfileRecommendation`; at resolve time a differing known input yields `status: stale` with `stale_inputs` named and overrides rejected. Profiles without a fingerprint keep applying. |
| 09.3 Admission | Verified rather than changed: the aggregate admission cap is the one `RuntimeRequestScheduler`, so N aliases on one endpoint cannot exceed `LEWLM_MAX_CONCURRENT_RUNTIME_REQUESTS` upstream; step 07 already removed any LewLM microbatch window in front of bridges and proved two requests overlap. |
| 09.7 Warm | Verified rather than changed: scan, `/v1/health`, `/v1/models`, and `/v1/runtime` make no generation request to a bridge; warm remains an explicit lifecycle action (step 05 showed a bridge warm is a one-token probe and an unload releases only LewLM's lease). |
| 09.5 / 09.8 | No change needed: every recipe already persists engine caches on their own volumes (weights separate from compiled artifacts), and no new CUDA dependency was added to startup; the portable CPU route and ONNX paths are untouched. |

## Portable results on this host

| Command | Result |
| --- | --- |
| `tests/unit/test_latency_contract.py` — presets normalize strictly; fingerprint captures the seven inputs and is stable; a changed artifact revision makes the profile `stale` with the input named and overrides rejected while a `throughput` request never adopts an `interactive` profile, the settings-level preset applies, and fingerprint-less legacy profiles still select; **six aliases on one endpoint with a cap of 2 never exceeded 2 in flight at the fake engine** (peak observed upstream, queue depth ≥ 4, admission released); scan/health/models/runtime make zero generation calls and `startup` reports the three phases from cached state even after the engine dies; `first_advertised_at` is recorded once | **6 passed** |
| llama.cpp runtime, metadata store, introspection, CLI, operations, benchmark-diagnostics, host-integration, library, and API suites (document/multimodal cases deselected — step-00 `pyexpat`) | **225 passed** |
| Isolated process start × 3 (`external_accelerator` pack, empty models dir, no endpoints) → `GET /v1/runtime` | [startup-phases.json](evidence/modernization-step-09/startup-phases.json): process-to-answer 0.675 / 0.724 / 0.731 s; `lewlm_ready_seconds` (bootstrap alone) 0.054 / 0.057 / 0.057 s. Bootstrap is ~8 % of the wall time; the rest is interpreter import and server start. One observation set, not a benchmark. |

## Deferred measurements and exact prerequisites

| Measurement | Missing prerequisite | Follow-up command / expected observation |
| --- | --- | --- |
| 09.4 llama.cpp CPU tuning (thread count vs physical cores, batch/microbatch, context, mmap) | A CPU-lane host (Linux x86-64 or the CPU image from step 03); this Mac's llama.cpp build is Metal and the only local GGUF is a 5 GB Q8 model — not the target lane. LewLM exposes no `n_threads` control today; add one only if the measurement identifies a useful setting | `lewlm autotune --model <gguf id> --preset interactive` then `--preset throughput`; compare `candidate_summaries`; for threads, run `lewlm benchmark` under `OMP_NUM_THREADS`/`GGML` thread env variations and record p50/p95 TTFT and tokens/s per the benchmark protocol (3 warmups, ≥ 30 requests, concurrency 1/2/4) |
| 09.4 accelerator offload | Linux/NVIDIA host with the CUDA image | same protocol with `LEWLM_GPU_OFFLOAD_LAYERS` at 0 / partial / -1 |
| 09.6 eager vs graph capture / compile | Linux/NVIDIA host with the vLLM or SGLang recipe running | vLLM: `--enforce-eager` vs default (and `--compilation-config`); SGLang: default CUDA graphs vs `--cuda-graph-backend-decode=disabled`; measure interactive startup and steady-state throughput with `scripts/bridge_prefix_benchmark.py --requests 30 --concurrency 1/2/4`; adoption gate: ≥ 10 % on the preset's target metric, ≤ 5 % regression on the other |
| 09.1 engine-ready and model-warm timings on real engines | Any running engine recipe | read `first_advertised_at - lewlm_ready_at` and `warm_models[].load_seconds` from `GET /v1/runtime` after `lewlm scan` and an explicit warm |
| Long prefill not blocking a short request at the LewLM layer | Any engine with two concurrent requests | acceptance `concurrency` case with one long-prefill prompt; the short request's first content must arrive before the long one completes (`bridge_prefix_benchmark.py --concurrency 2` reports both) |
| Before/after comparison per the benchmark protocol | Hardware lanes above | keep `docs/validation/evidence/modernization-step-05/prefix-benchmark-c*.json` as the oMLX baseline; any tuning change on that lane re-runs the same command |

## Rollback

Revert the step commit. `ready_at`, `first_advertised_at`, `startup`,
`preset`, `fingerprint`, `stale_inputs`, and the `stale` status are additive
fields with `None`/empty defaults; `serving_profile_preset` defaults to the
previous behaviour; stored profiles are not migrated (old keys still resolve).
