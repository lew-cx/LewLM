# Modernization baseline and implementation record

Captured 2026-09-16 before runtime changes. Baseline runtime revision: `caae222`.
Approved roadmap committed as `f6c9174`; the initial dirty tree contained only that roadmap and its two documentation links.

## Step 00 — complete, hardware validation deferred

The portable baseline and compatibility-evidence contract are complete. Engine commits and model artifacts are candidate pins, not validated installation recipes. Dependency locks/image digests will be captured in the isolated engine environments in steps 05–08. Nothing is labeled validated.

Host: Apple M2 Max, 12 physical cores, 96 GiB unified memory, Darwin 25.2.0 arm64, Python 3.14.7. Installed-package versions are preserved in [host.json](evidence/modernization-baseline/host.json). No NVIDIA GPU/toolkit is available. Docker client 28.1.1 is installed, but its daemon is unavailable. No oMLX, ExLlamaV3, vLLM, or SGLang distribution is installed in the LewLM environment. Ollama's executable is present; no daemon or model was started for this capture.

The existing MLX Qwen2.5-0.5B-Instruct 4-bit snapshot was hashed locally. The HF BF16 equivalent was pinned through upstream metadata without downloading weights. Both model cards identify Apache-2.0 licensing. Model/tokenizer revisions, weight hashes, and workload limits are in [compatibility.json](../../examples/backends/compatibility.json). TabbyAPI's BF16 model is a candidate; this does not claim that an EXL3 conversion has been produced or tested. Upstream source-commit capture is preserved in [upstream.json](evidence/modernization-baseline/upstream.json).

### Results

| Command / observation | Result |
| --- | --- |
| Roadmap's focused pytest command, using `.venv/bin/python` | **154 passed**, 11.16 seconds; [full output](evidence/modernization-baseline/focused-unrestricted.log) |
| `.venv/bin/python -m pytest -q -p no:cacheprovider -m 'not long_running'` | **1038 passed, 17 failed, 1 skipped, 5 deselected**, 199.37 seconds; [full output](evidence/modernization-baseline/full-unrestricted.log) |
| `.venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_backend_compatibility.py` | Six initial contract tests passed; additional validation coverage is recorded in the commit |
| `.venv/bin/python scripts/validate_backend_compatibility.py --schema-output examples/backends/compatibility.schema.json` | All four recipes validate as `deferred`; schema generated from the same typed contract |
| One isolated process start to `/v1/health` | Measured in [startup.json](evidence/modernization-baseline/startup.json); one observation, not a performance benchmark |

The first sandboxed focused run had two loopback socket permission failures. The same tests passed with socket access. The sandboxed full run was interrupted and replaced with the unrestricted baseline run above. These sandbox failures are not product regressions.

All 17 full-suite failures involve document rendering/ingestion or attachment paths. The output includes a host Python `pyexpat`/system `libexpat` symbol mismatch (`_XML_SetAllocTrackerActivationThreshold`), which also affects XML-dependent document libraries. Preserve these as pre-existing failures; do not fix or suppress them as part of the bridge changes without separate evidence. The environment also lacks Torch and is not a full production dependency lock.

The process-start observation used a new temporary `LEWLM_DATA_DIR`, empty `LEWLM_MODELS_DIR`, `LEWLM_RUNTIME_PACKS=["external_accelerator"]`, a free loopback port, and `.venv/bin/python -m lewlm serve --host 127.0.0.1 --port <port>`. It polled `/v1/health` every 50 ms and terminated the child cleanly. Current pack defaults still caused a native llama.cpp/Metal diagnostic import during health; the captured result must not be called a pure bridge-only startup. No model generation occurred.

### Deferred measurements and exact prerequisites

| Measurement | Missing prerequisite | Follow-up command / expected observation |
| --- | --- | --- |
| Clean CPU image build | Running Docker daemon/Linux builder | `docker build --progress=plain -t lewlm:baseline .`; retain time and full build log |
| No-change/app-only CPU rebuild | Successful initial build | Repeat the build, then repeat after a temporary app-only edit in a disposable checkout; count native compilation commands and elapsed time |
| CUDA compile/offload | Linux NVIDIA host, driver, compatible toolkit/image | `docker build --progress=plain -f Dockerfile.cuda -t lewlm:baseline-cuda .`, then `docker run --rm --gpus all lewlm:baseline-cuda doctor --json`; follow with real GGUF generation and offload evidence |
| Cold dependency install | Disposable environment with a selected Python/platform lock | `python -m pip install -e '.[dev,llamacpp,documents]'` in that environment; record download and native compile phases separately; do not replace the current environment |
| Engine ready/model load/TTFT/throughput/peak memory | Isolated pinned engine environment, validated model, real host | Complete roadmap steps 05–08 and run the common real-engine suite; record all timing and memory fields without fabricated defaults |
| oMLX inference | Compatible isolated Python, oMLX installation/configuration | Step 05 on this Apple Silicon host; do not reuse the unvalidated Python 3.14 stack |
| vLLM/SGLang/ExLlamaV3 inference | Linux/NVIDIA host and isolated dependency stacks | Steps 06–08; compare the same pinned BF16 model where supported |
| Native Windows / WSL2 | Separate target OS | Run the corresponding roadmap step 11 lane; macOS results cannot satisfy it |

### Evidence contract and rollback

`src/lewlm/utils/backend_compatibility.py` and its exported JSON Schema distinguish `deferred`, `failed`, and `validated`. Promotion to `validated` requires exact Python/engine versions, host/accelerator/driver/toolkit identity, an immutable image digest or hashed dependency lock, a model snapshot with artifact hashes, and an evidence path. Source pins alone cannot pass the validated contract. The schema validates evidence structure, not the truth of a benchmark; an actual hardware acceptance run is still required.

Reproduce contract validation with `python scripts/validate_backend_compatibility.py`. No engine imports, installation, network access, or downloads occur. Rollback is a revert of the step-00 commit; no runtime behavior or operator state has changed.

Next eligible step: 01, endpoint identity and capability evidence. Step 03 can also proceed from this baseline, but will retain the Docker/CUDA hardware deferrals above.

## Step 01 — complete

Commit: this step's commit, immediately after baseline commit `4226e97`.

Behavior implemented: named loopback endpoints coexist under stable IDs while
the singular settings resolve to `legacy-default`. Explicit endpoint bindings
select the correct runtime; runtime names, residency keys, deterministic
response/coalescing keys, health evidence, provider evidence, and Ollama
inventory carry the endpoint boundary. Adapter health reads cached evidence and
reports upstream residency/cancellation as unknown. Profile performance flags
are inactive until observed rather than activated by their labels.

Compatibility: old settings, runtime name, and Ollama model IDs remain valid.
An explicit endpoint collection plus the enabled legacy endpoint is rejected
with migration guidance. Named Ollama discovery requires one matching
`ollama_local` endpoint. Credential reference names and values are absent from
redacted settings output.

Verification: `tests/unit/test_external_endpoints.py` covers URL/ID validation,
legacy migration, secret redaction, equal-name endpoint routing, residency and
cache isolation, passive health, and Ollama binding. Focused settings, catalog,
Ollama, middleware, install-profile, runtime-metrics, adapter, and host API
regressions pass. The shared-runtime multiprocess test is run with the final
step gate. No real engine is required or claimed for this configuration step.

Rollback: revert the step commit. The setting is additive and no stored data is
migrated. Next eligible step: 02.

## Step 03 — implemented; container/CUDA hardware lanes deferred

Commit: this step's commit, immediately after step 02 commit `93bce1c`.
Full record: [modernization-step-03.md](modernization-step-03.md).

Behavior implemented: a serving-only `llamacpp_runtime` extra; image
dependency inputs exported from `pyproject.toml` (`requirements/`); both
Dockerfiles restructured so the dependency layer is keyed on that input and
the LewLM wheel is installed `--no-deps` last; `IMAGE_FLAVOR`
(`bridge|serving|full`, default `full`); bounded `BUILD_JOBS`; `ccache` and
BuildKit cache mounts; prebuilt-wheel or keyed source build for
`llama-cpp-python`; explicit torch indexes; CUDA SM validation before compile;
build-time flavor verification (`scripts/verify_llamacpp_build.py`);
`image_flavor` and a real `storage_access` write probe in doctor/health; CI
bridge boot smoke plus a full-image rebuild-contract gate
(`scripts/docker/measure_rebuild.sh`); lock generator
(`scripts/docker/lock_dependencies.sh`). `MTL` added as a Metal marker in
build-flavor detection after observing current llama.cpp output.

Compatibility: existing build/compose commands and `EXTRAS` keep working; the
old default equals the `full` flavor. Additive doctor/health fields only.

Commands run and results: roadmap focused regression **160 passed**; unit
suite **786 passed, 37 deselected** (document tests blocked by the host
`pyexpat` mismatch recorded in step 00); step-specific set 117 passed;
`export_dependency_inputs.py --check` in sync; `docker compose config` valid;
clean-venv simulations of the app stage, the `bridge` flavor, and the
`serving` source-build command line all pass. The verifier rejected a
would-be CPU-only macOS build that llama-cpp-python forced to Metal — the
intended failure mode.

Real-engine evidence: none claimed. Deferred: Docker clean/no-change/app-only
timings and logs, `BUILD_JOBS` timing comparison, lean-image contents in a
built image, fixture conversion, pinned locks, Linux ISA report, CUDA
compile/offload/generation, prebuilt-wheel path, native Windows wheel safety,
and the two CI docker jobs on the next push — each with its exact command in
the step record.

Rollback: revert the step commit; nothing stored changes. Next eligible step:
04, external discovery and routing.

## Step 04 — complete

Commit: this step's commit, immediately after step 03 commit `18f4123`.
Full record: [modernization-step-04.md](modernization-step-04.md).

Behavior implemented: `lewlm scan` inventories every enabled named endpoint's
`/v1/models` (bounded concurrency, per-endpoint failures) and registers
endpoint-qualified manifests with `external://<endpoint_id>/<upstream_id>`
sources, exact upstream ids, evidence-only formats (`ModelFormat.EXL3` added),
and `loopback_unverified` locality. Successful inventory has a 30 s TTL;
failures keep the last-known list as `stale`. URI-backed manifests never enter
conversion or packaged runtimes. Ollama manifests bind explicitly to the
Ollama endpoint. Routing decisions and execution metadata record
`endpoint_id`, `engine_profile`, `execution_locality`, and any preflight
fallback; `external_fallback_policy=explicit_alias` with
`external_fallback_aliases` is the only automatic substitution and never
happens after submission. Passive bridge health now carries the static
performance-feature snapshot, repairing two `test_operations` regressions
introduced in step 01.

Compatibility: additive settings and fields; Ollama ids/metadata preserved;
default routing order unchanged.

Commands run and results: step suite 13 passed; focused + registry/Ollama/
routing/residency/cache set 314 passed; unit+integration 1058 passed with only
the 6 step-00 document/`pyexpat` failures remaining. The integration-bundle
schema snapshot was already failing before this step and is owned by step 10.

Real-engine evidence: none claimed; deferred to steps 05–08 with commands in
the step record. Rollback: revert the step commit. Next eligible step: 05,
oMLX (Apple Silicon host available), or 06–08 pending Linux/NVIDIA.

## Step 05 — complete; Apple Silicon lane passed

Commit: this step's commit, immediately after step 04 commit `24aafb4`.
Full record: [modernization-step-05.md](modernization-step-05.md).

Behavior implemented: `examples/backends/omlx/` recipe (pinned commit,
isolated install, loopback launch with explicit limits, LewLM env, lock);
`docs/operations/backends/omlx.md`; `scripts/backend_acceptance.py` (the
common real-engine suite over LewLM's HTTP API) and
`scripts/bridge_prefix_benchmark.py`; `omlx` promoted to `validated` with
environment pins, lock sha256, evidence path, and scoped notes (`notes`
added to the validated contract). Fixes the run surfaced: structured-output
prediction/outcome now agree with a new `enforcement_evidence` field and no
false decoder claim for bridges; streams that fail after output started end
with a terminal `StreamErrorEnvelope` chunk instead of a reset socket;
bridge lifecycle results say the unload released a lease only.

Real-engine evidence: oMLX 0.7.0.dev3 @ b45fb7e, mlx 0.32.2, Python 3.11.15,
Apple M2 Max / macOS Darwin 25.2.0, Qwen2.5-0.5B-Instruct-4bit @ a5339a4 —
acceptance 10 passed / 0 failed / tools inconclusive / fallback done by hand;
engine stop, stale inventory, restart, and mid-stream kill all observed;
LewLM first-content overhead +4.8 ms p50 at concurrency 1, none at 2.

Deferred: tools on a larger model, vision/embeddings/rerank probes, custom
kernels, the full step-09 benchmark protocol, upstream abort verification
from oMLX's log. Rollback: revert the step commit. Next eligible steps: 06–08
need Linux/NVIDIA; 09–10 can proceed on this host.

## Implementation handoff

Append each completed step or reviewable substep here with its commit, behavior, commands/results, deferred tests, and rollback. Never mark the full roadmap complete while hardware or Chap UI acceptance is pending.
