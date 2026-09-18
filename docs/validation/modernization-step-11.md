# Modernization step 11 validation — CI and hardware acceptance lanes

Implemented on 2026-09-18 on the baseline Apple Silicon host. The lanes,
their runner, their markers, and their place in release evidence are in; the
CI changes take effect on the next push. This step adds no engine execution
of its own: the roadmap's three required demonstrations — a passing
fixture-only lane, a deliberate hardware deferral, and a deliberate contract
failure that blocks acceptance — are all reproduced below.

## What changed

| Area | Change |
| --- | --- |
| Lane runner | `scripts/backend_lanes.py`: `detect` (which lanes this host can run and why not the others), `run` (acceptance suite + prefix benchmark against an operator-started LewLM + engine → one `lane-<lane>-<recipe>.json` record: `passed`, `failed` with exit 1, or `deferred` with the exact prerequisite and next command), `summary` (merges records with `examples/backends/compatibility.json`; only `validated`/`passed` is proof; an idle-host deferral never hides a validated recipe; a real failed run overrides it). |
| pytest markers | `real_engine`, `apple_silicon`, `linux_cpu`, `linux_nvidia`, `native_windows`, `wsl2` registered in `pyproject.toml` before use. |
| Hardware tests | `tests/hardware/test_backend_lanes_real.py`: one test per lane; runs the lane through the runner when `LEWLM_LANE_BASE_URL`/`LEWLM_LANE_MODEL`/`LEWLM_LANE_RECIPE` name an engine on a host that can run it; otherwise **skips** with the missing prerequisite in the raw output. |
| Release evidence | `generate_release_manifest.py` embeds `backend_lanes` (from `build_lane_summary`, records dir via `LEWLM_LANE_RECORDS_DIR`), so deferred hardware and the pending Chap UI lane are visible in every release bundle. |
| CI matrix (Linux/macOS/Windows) | New step running the endpoint/migration/transport/inventory/routing/profile/latency/preflight/lane/compatibility/residency/cancellation/failure-envelope/host-integration/bundle/Chap-contract tests with no engine installed; `pytest -rs tests/hardware` so every lane's skip reason is in the log; shared-runtime subprocess tests on Linux and macOS (native Windows waits for its own lane proof); pydantic pinned to the version the bundle was generated with. |
| CI full suite | The integration-bundle schema snapshot is back in the gate (it is generated now, with `generated_with` recorded in the bundle and pinned in CI). |
| Bundle export | `--bundle` path argument; `--check` explains a pydantic version mismatch instead of leaving drift ambiguous. |
| Docs | Lanes table and commands in the release-and-validation reference. |

## The three demonstrations

| Demonstration | Where | Result |
| --- | --- | --- |
| Passing fixture-only lane | `test_chap_contract.py::test_chap_smoke_passes_in_fixture_mode` (subprocess of `examples/chap_backend_smoke.py --fixture`), plus the whole "any development OS" matrix step | passes here; runs on Linux/macOS/Windows in CI |
| Deliberate hardware deferral | `python scripts/backend_lanes.py run --lane linux_nvidia --recipe vllm` on this Mac → [lane-linux_nvidia-vllm.json](evidence/modernization-step-11/lane-linux_nvidia-vllm.json): `deferred`, reason "needs Linux, host is Darwin; needs a working NVIDIA driver (nvidia-smi: not found on PATH)", next command recorded; `pytest -rs tests/hardware` → 5 skips, each with its prerequisite; `test_backend_lanes.py` covers the same with injected host facts | deferral is visible and exit 0 (exit 1 only with `--require`) |
| Deliberate contract failure that blocks acceptance | `test_backend_lanes.py::test_a_failed_acceptance_case_makes_the_lane_fail_and_blocks_promotion` (a failing acceptance case → lane `failed`, CLI exit 1, "never promotes"); `test_chap_contract.py::test_a_drifted_bundle_is_a_contract_failure_that_blocks_the_gate` (one removed schema property → `export_integration_bundle.py --check` exit 1); `test_backend_compatibility.py` (a `validated` recipe without a real lock/digest/evidence is rejected) | all block |

## Portable results on this host

| Command | Result |
| --- | --- |
| `tests/unit/test_backend_lanes.py` | 4 passed |
| `tests/unit/test_release_manifest.py` (+ `backend_lanes` assertions), `tests/unit/test_backend_lanes.py` | 7 passed |
| `pytest -rs tests/hardware` | 5 skipped with actionable reasons ([detect.json](evidence/modernization-step-11/detect.json)) |
| `tests/integration/test_shared_runtime_baseline.py`, `test_shared_runtime_multiprocess.py` on macOS | 8 passed (basis for adding them to the macOS/Linux matrix) |
| `scripts/backend_lanes.py summary` | [lanes-summary.json](evidence/modernization-step-11/lanes-summary.json): `validated 1` (oMLX from step 05), `ci 1`, `deferred 10`, `pending 1` |
| Full unit + integration suites | see the handoff record |

## Deferred

| Lane | Status | Next |
| --- | --- | --- |
| Apple Silicon (oMLX) | validated in step 05; runnable here but not re-executed because the step-05 oMLX environment (`~/.lewlm-engines/omlx`) was not retained on this host | `examples/backends/omlx/README.md`, then `LEWLM_LANE_BASE_URL=… LEWLM_LANE_MODEL=… LEWLM_LANE_RECIPE=omlx pytest tests/hardware -m apple_silicon` |
| Linux CPU, Linux NVIDIA (vLLM, SGLang, TabbyAPI, llama.cpp CUDA), native Windows, WSL2 | deferred | `python scripts/backend_lanes.py detect` on the target host, then `run` per recipe; each recipe README has the exact serve/prove steps |
| CI jobs on the next push | the matrix additions, the pinned pydantic, and the restored bundle gate are exercised by the next push | inspect the `core-tests` logs for the hardware skip reasons and the `chap-smoke-*` artifacts |
| Chap UI | pending | `docs/guides/chap-validation.md` |

## Rollback

Revert the step commit. The runner, markers, hardware tests, and manifest
section are additive; CI returns to the previous matrix and the bundle
snapshot exclusion.
