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

## Implementation handoff

Append each completed step or reviewable substep here with its commit, behavior, commands/results, deferred tests, and rollback. Never mark the full roadmap complete while hardware or Chap UI acceptance is pending.
