# Modernization on Windows and Linux: validation record

Run on 2026-09-24 against the modernization (steps 00–12) as reviewed on
2026-09-22, which had been developed and validated on Apple Silicon only. This
record covers the platform lanes that review deferred: native Windows, Windows
+ WSL2, Linux CPU, and Linux with an NVIDIA GPU. Every lane was run for real;
defects found on the way were fixed on the branch and the lane re-run.

| Host | |
| --- | --- |
| OS | Windows 11 Home 10.0.26200, Git for Windows (`core.autocrlf=true`) |
| CPU / RAM | Intel Core Ultra 9 285HX (24 cores, Arrow Lake: AVX2/AVX-VNNI, **no AVX-512**), 63.4 GB |
| GPU | NVIDIA GeForce RTX 5090 Laptop, 24 GB, compute capability 12.0 (Blackwell), driver 610.47 |
| Python | 3.11.9 (native Windows); 3.11 / 3.12 in the Linux images |
| Containers | Docker Desktop 4.74.0, engine 29.4.3, WSL2 kernel 6.6.87.2-microsoft-standard; NVIDIA passthrough working |

Linux here means Linux userspace in Docker Desktop's VM, which runs the WSL2
kernel; the lane records carry `host.wsl: true`. No bare-metal Linux host was
available, and nothing below claims one.

## Results

| Lane (recipe) | Topology | Result | Evidence |
| --- | --- | --- | --- |
| Native Windows (llamacpp-cpu) | fresh venv, `.[llamacpp_runtime]` from the prebuilt CPU wheel index; Gemma-4-E2B Q8_K_P | **passed** 11/11 applicable; health 1.3 s, first chat incl. 5 GB load 2.0 s, warm 0.1 s; CTRL_BREAK shutdown graceful in 0.6 s, port released | `native-windows-llamacpp-cpu/` |
| Native Windows (ollama) | fresh **lean** install (25 packages, 8 s), the documented singular settings; Ollama 0.34.4 in Docker on `127.0.0.1:11434`, qwen2.5:0.5b | **passed** 11/11; discovery `ollama://qwen2.5:0.5b`, locality `host_local` | `native-windows-ollama/` |
| Windows + WSL2 (vllm) | LewLM native on Windows → vLLM v0.29.0 (digest-pinned) in Docker Desktop | **passed** 11/11 on a cold engine; plus the real-engine rollout sequence and the Chap smoke (12/12 real-model checks) | `wsl2-vllm/` |
| Windows + WSL2 (sglang) | LewLM native → SGLang v0.5.19-cu130 (digest-pinned) | **passed** 11/11; up to 3 requests running upstream (`#running-req`) | `wsl2-sglang/` |
| Windows + WSL2 (exllamav3-tabby) | LewLM native → TabbyAPI 53da7919 / ExLlamaV3 1.5.0; EXL3 artifact converted here | **passed** 10/10 + tools inconclusive (the 0.5B model answered in text) | `wsl2-exllamav3-tabby/` |
| Linux CPU (llamacpp-cpu) | `full` CPU image, lane run inside the container | **passed** 11/11 | `linux-cpu-container/` |
| Linux NVIDIA (llamacpp-cuda) | `lewlm:cuda` built for SM 120 on CUDA 12.8, `LEWLM_GPU_OFFLOAD_LAYERS=-1`, lane inside the container | **passed** 11/11; `CUDA : ARCHS = 1200`, offload on the 5090 | `linux-nvidia-container-llamacpp-cuda/` |

"Applicable" excludes `fallback`, which the suite cannot exercise without
stopping the engine; for vLLM it was exercised on the real engine (below).
`lanes-summary.json` is `scripts/backend_lanes.py summary` over these seven
records: 7 passed, oMLX still validated from its own host, the three GPU
engines still deferred for bare-metal Linux/NVIDIA, Chap UI pending.

Repeated-prefix time to first content through LewLM, concurrency 1, p50 (an
observation, not a comparison — no direct-engine baseline was taken): vLLM
46 ms, TabbyAPI 44 ms, Ollama 48 ms, SGLang 59 ms, llama.cpp CUDA 58 ms,
llama.cpp CPU 231 ms (Windows) / 629 ms (container).

### Real-engine rollout and rollback (vLLM, Windows + WSL2)

`wsl2-vllm/rollout-rollback.json` repeats the step-12 sequence on this
platform with a llama.cpp alias (`LEWLM_EXTERNAL_FALLBACK_POLICY=explicit_alias`
→ the Gemma GGUF): chat through `vllm` / `vllm_local`; two LewLM streams
overlap upstream (vLLM's own `vllm:num_requests_running` peaked at 2); the
engine killed after the first content chunk ends the stream in 0.5 s with
`finish_reason: error`, `runtime_unavailable`, `partial_output: true`,
`[DONE]`, no replay; rescan keeps the model listed as `stale` with health
`ok`; the next request is served by the alias with `fallback_from_model_id`;
restart (25 s to healthy with the compile cache) returns it to vLLM with no
fallback; `"enabled": false` removes exactly the vLLM model (404), the alias
still serves, and the data directory is untouched.

### Containers and rebuilds

| Check | Result |
| --- | --- |
| `bridge` image (CI `docker` job equivalent) | built in 50 s with no native compile; doctor: data dir writable as uid 10001; `/v1/health` ok |
| `full` image: clean / no-change / app-only (`measure_rebuild.sh --reset`) | 300 s (250 native compile steps) / 33 s / **67 s with 0** — `docker-rebuild-full/` |
| App-only rebuild after the fixes on this branch | 83 s, 0 native compile steps |
| `serving` image | 33 s (reused the keyed llama.cpp wheel from `full`), 116 MB vs 481 MB, no torch/transformers/weasyprint, generation probe passed |
| `full` image converts and quantizes | pinned Qwen2.5-0.5B-Instruct safetensors → GGUF Q4_K_M (398 MB) in 12 s; the converted model generated |
| CUDA image for Blackwell (`docker compose --profile gpu build` with this host's `.env`: CUDA 12.8.0, SM 120) | built (427 CUDA/C++ objects in 200 s at `BUILD_JOBS=12`); build-time proof `the cuda backend is compiled in`; lane above passed with full offload |
| Portable CPU ISA of both images | `SSE3 SSSE3 AVX AVX2 F16C FMA BMI2`: the floor is x86-64-v3 (see the Docker guide) |

### Test suites

| Suite | Result |
| --- | --- |
| Full non-long-running suite, native Windows (fresh Python 3.11.9 venv, `.[dev,documents]`, pinned pydantic), at `bfc345b` | **1,201 passed, 58 skipped, 0 failed** in 370 s (before the fixes it could not finish: the first timeout-guarded run was killed by a 90 s test timeout) |
| Full non-long-running suite, Linux (`python:3.11-slim`, the CI `full-suite` install incl. `.[llamacpp]` + CPU torch), at `4f7ff06` | **1,202 passed, 57 skipped, 0 failed** in 254 s |
| Shared-runtime subprocess tests on Windows | 8 passed, three consecutive runs (now enabled in the CI Windows matrix) |
| llama.cpp-installed tests on Windows (grammar, runtime, opt-in real Gemma GGUF smoke) | passed |
| Release gates on Windows at `bfc345b`: `export_integration_bundle.py --check`, `export_dependency_inputs.py --check`, `validate_backend_compatibility.py`, `pip check`, `git diff --check`, `pytest -rs tests/hardware` (5 skips, each with its reason) | all pass |
| Chap smoke, fixture mode: from the checkout, and from a wheel installed into a clean venv (import path outside the checkout) | 14/14 and 14/14 |

Skips are MLX (macOS-only), the hardware lanes when no engine is named, and
the opt-in real-model smoke; none is a Windows or Linux exemption.

## Defects found and fixed

Each is its own commit on the branch, with a regression test that fails
without the fix.

| Defect | Where it showed | Fix |
| --- | --- | --- |
| A Windows checkout (`core.autocrlf=true`) gave the container scripts and `requirements/` CRLF: `sh` rejected the CUDA arch validator and the native requirement kept a trailing `\r` pip rejects | first Docker build from Windows | `.gitattributes` pins LF for those inputs; the extraction strips `\r`; the sh-based contract tests run wherever `sh` exists (they were skipped on Windows) |
| `measure_rebuild.sh` aborted on its first build on every platform (`local` expanded `$label` before assigning it) | the CI `docker-full` job's script, never run before | split the assignment |
| The suite could not finish on Windows: 60k SQLite commits per heavy test (cluster status rewrote an unchanged list on every snapshot), a per-statement schema commit, and a CA-bundle load per bridge transport | full suite on NTFS | write only on change, one schema transaction, one shared SSL context; the slowest test went from 131 s to 32 s, and ext4 pays the same fsyncs |
| Seeded llama.cpp replies were not reproducible (KV reuse re-evaluates only the prompt's tail), and **streaming dropped all sampling controls** | native Windows lane, `sampling` | seeded requests evaluate from an empty KV state; streaming applies and reports sampling |
| vLLM: seeded replies depend on prefix-cache state | WSL2 lane, cold engine | a fresh `cache_salt` for seeded requests (5/5 fresh pairs identical with prefix caching off; intermittent divergence with it on) |
| SGLang ignores `seed` unless the server runs with `--enable-deterministic-inference` | WSL2 lane | `seed` reported unsupported for `sglang_local` |
| TabbyAPI has no `seed` at all, yet implements `top_k`/`min_p`/`repetition_penalty`, which LewLM reported unsupported | WSL2 lane | a control map from what the pinned server implements |
| Ollama applies the seed but an identical cached prompt samples differently, with no per-request bypass | native Windows Ollama lane | seed forwarded, `deterministic: false` |
| `lewlm chat > file` crashed with `UnicodeEncodeError` after generating (cp1252 on redirected Windows output) | CLI with CJK/emoji output | redirected Windows output is UTF-8 |
| vLLM would not start under WSL2 (`UVA is not available`) | WSL2 lane | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` in the recipe (ignored outside WSL2) |
| SGLang crash-looped (22 restarts): its JIT's staging rename fails on a Windows bind mount | WSL2 lane | kernel cache on a named volume |
| **The CUDA image could not build anywhere**: its verify step loads `libllama.so`, which needs the driver's `libcuda.so.1`, absent in `docker build`; and its llama.cpp was built `-march=native` (AVX-VNNI from this builder) | first CUDA build | driver stub for that one command and a `gpu-build` proof; `GGML_NATIVE=OFF` default; a CI job now builds the CUDA image |
| A CUDA llama.cpp wheel on Windows could not find pip-installed cuBLAS | native CUDA attempt | pip-installed NVIDIA DLL folders are registered before import; the load error names the fix |
| `verify_llamacpp_build.py` said OK for a wheel that dies at the first model load with an illegal instruction (AVX-512 build on a CPU without it) | native CUDA attempt | build CPU features are compared with the host (`missing_cpu_features`); the verifier exits 2 and doctor explains |
| The opt-in real GGUF smoke scanned stub weights and could only fail | llama.cpp-installed tests | points at the real weights |
| The WSL2 lane could only be recorded from inside Linux, although its proof (Windows-to-LewLM connectivity) is observed from Windows | lane runner | a Windows host with a running WSL2 VM qualifies |

## Deferred, with the exact prerequisite

| Item | Why not here | Next step / expected observation |
| --- | --- | --- |
| Bare-metal Linux NVIDIA for vLLM, SGLang, TabbyAPI | only WSL2-kernel Linux on this host | on a Linux host: each recipe README, then `scripts/backend_lanes.py run --lane linux_nvidia --recipe <name>`; the three stay `deferred` in the compatibility manifest until then |
| Native Windows llama.cpp with CUDA | the prebuilt 0.3.35 Windows CUDA wheels need AVX-512 (this CPU has none) and ship SMs up to 9.0; a source build needs MSVC Build Tools + CUDA Toolkit ≥ 12.8, absent here | install both, `CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF" pip install --no-binary llama-cpp-python ...` with long paths enabled; expect `verify_llamacpp_build.py --expect gpu --hint cuda` OK and the lane to pass. The CUDA container is the working GPU path meanwhile |
| Engines running natively on Windows (Ollama for Windows, ExLlamaV3 with `triton-windows`) | not installed; Ollama ran in Docker for the bridge lane | install Ollama for Windows and re-run `--lane native_windows --recipe ollama` |
| The WSL2 lane from inside a WSL distribution | only Docker Desktop's VM is installed | `wsl --install Ubuntu`, run LewLM there, `detect` then `run --lane wsl2` |
| Pinned dependency locks, `BUILD_JOBS` comparison, CUDA rebuild contract | not required to prove behaviour; time | `scripts/docker/lock_dependencies.sh …`; two `measure_rebuild.sh --reset` runs with `BUILD_JOBS=1/4`; `measure_rebuild.sh --dockerfile Dockerfile.cuda` |
| Step 09 tuning measurements | out of scope for a platform pass | the step 09 record's commands |
| Chap UI checklist | belongs to Chap | `docs/guides/chap-validation.md` |

## Rollback

Every change is a separate commit and additive. Reverting a fix commit
restores the previous behaviour; no setting, route, response field, or model
id changed. The integration bundle is unchanged (`export_integration_bundle.py
--check` passes). Sampling reports became more accurate for `sglang_local`,
`exllamav3_tabby`, and `ollama_local` (seed unsupported or not
deterministic); a client that relied on `deterministic: true` from those
engines was relying on a claim the engines did not honour.
