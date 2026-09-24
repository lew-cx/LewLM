# LewLM modernization roadmap

Status: implemented through step 12 on 2026-09-18; each step's portable acceptance passed and its hardware lanes are recorded as validated or deferred in the per-step validation records under `docs/validation/`. The step sections below keep their original wording plus an implementation-status line.
Research date: 2026-09-16. Repository baseline: `caae222`, package version `0.4.2`.

Post-implementation review: [2026-09-22 release review](../validation/modernization-review.md) records corrective fixes, independent portable regression results, installed-wheel acceptance, and remaining platform deferrals.

Windows and Linux: [2026-09-24 validation record](../validation/modernization-windows-linux.md) runs the native Windows, Windows + WSL2, Linux CPU, and Linux NVIDIA (llama.cpp CUDA) lanes for real on an RTX 5090 host — all passed after the fixes it lists — and states what stays deferred (bare-metal Linux for vLLM/SGLang/TabbyAPI, native Windows CUDA).

The objective is faster installation, startup, and interactive inference while keeping LewLM a small, easy-to-integrate middleware layer. Complete the existing external-server integration, add ExLlamaV3 through TabbyAPI, and improve the portable CPU/CUDA paths. Keep native MLX, llama.cpp, llama.cpp-server, and Ollama available.

## Execution rules

1. Execute steps 00–12 in order. Each step is a reviewable change; split large steps into smaller PRs without changing their dependencies. Step 03 may be developed independently after 00, but routing and engine work must wait for their prerequisite contracts.
2. Before editing, inspect the named files and their current tests. Paths below identify ownership, not permission to replace whole modules. Reuse existing contracts, errors, telemetry, schedulers, and client methods.
3. Finish each step's portable tests before advancing. Record real-engine validation separately as `passed`, `failed`, or `deferred`, with the exact missing OS/hardware, command, and expected result. A mock pass never counts as a hardware pass. Deferred hardware work does not block unrelated implementation; it does block promotion of the affected configuration to supported/default.
4. Preserve existing configuration, public model IDs, Python entry points, HTTP routes, and response fields. Add fields with backward-compatible defaults. Any unavoidable contract break requires a separate migration design before implementation.
5. New engines stay optional and operator-managed. Do not install, download, launch, update, or stop an engine on ordinary LewLM startup, discovery, health checks, or chat requests. Provide explicit setup instructions and opt-in deployment recipes.
6. Keep the current default routing order until measured evidence justifies a separately reviewed change. New profile names do not make a runtime faster, available, or capability-complete.
7. Keep one model-residency authority per service container, following [ADR-001](adr-001-shared-runtime-residency.md). Backend-native batching and KV caches remain backend-owned. Do not create another model loader or GPU scheduler inside LewLM.
8. Ship one stable Chap-facing API. Chap must not need engine names, engine SDKs, CUDA settings, or engine-specific SSE parsers to send a request or cancel it.

All new filenames, configuration keys, and profile IDs explicitly labeled **proposed** below must be implemented before examples can claim they work. Existing commands are identified separately. This roadmap does not select untested engine versions.

## What exists and what needs work

| Existing implementation | Consequence for this plan |
| --- | --- |
| `runtime/adapters/openai_compatible.py` already includes `omlx`, `vllm_local`, `sglang_local`, `vllm_mlx`, `ollama_local`, and `llamacpp_server` profiles. | Extend and verify these profiles. `vllm_mlx` and `vllm_local` remain distinct. Add only the missing ExLlamaV3/TabbyAPI profile. |
| Settings expose one external accelerator URL/profile; `RuntimeCatalog` is keyed by runtime affinity. | Introduce endpoint identity so Ollama and an accelerator can coexist without overwriting one another. |
| The bridge sends model/messages/max_tokens/temperature/stream. Its streaming parser yields content strings, and its worker uses an unbounded queue. | Preserve sampling, structured output, tool calls, reasoning policy, usage, cancellation, and backpressure across the boundary. |
| Bridge-supported formats omit `HUGGINGFACE`; the model contract already has that enum. Ollama inventory already creates manifests without local files. | Reuse those concepts for endpoint-advertised models; do not disguise all external artifacts as MLX or GGUF. |
| Successful external discovery is cached until invalidation. Bridge unload is currently a no-op. | Refresh inventory predictably and distinguish a LewLM lease from actual upstream model residency. |
| `pyproject.toml`'s `llamacpp` extra includes Torch/Transformers and other conversion dependencies. | Add a serving-only install path without silently removing conversion from the legacy extra. |
| Both Dockerfiles set `PIP_NO_CACHE_DIR=1`, force CMake, and copy the full repository before installing. CUDA defaults target `75;80;86;89`. | Separate dependencies from app code, cache native builds, and provide explicit device-targeted builds. |
| CPU Docker already uses `GGML_NATIVE=OFF`; CUDA Docker already enables offload with `LEWLM_GPU_OFFLOAD_LAYERS=-1`. | Preserve these intentions and verify effective behavior; these are not new features. |
| Shared residency, cancellation, typed clients, OpenAPI, integration fixtures, benchmarks, and host validation already exist. | Extend these surfaces rather than building a parallel integration layer. |

Paths in this table are relative to `src/lewlm/` unless they name a root file.

## Research-backed integration choices

| Engine | Integration decision | First real validation target | Source |
| --- | --- | --- | --- |
| oMLX | Complete `omlx` over its local OpenAI-compatible server. Its upstream batching and tiered prefix/KV caching are reasons to benchmark it, not reasons to replace native MLX. | Apple Silicon macOS; select the OS/Python minimum from the pinned release. | [oMLX upstream](https://github.com/jundot/omlx) |
| ExLlamaV3 | Add **proposed** `exllamav3_tabby` using TabbyAPI, the recommended server. Use a verified EXL3 or other explicitly supported artifact; do not assume EXL2/GGUF interchangeability. | Linux + NVIDIA first. Native Windows is a separate compatibility gate. | [ExLlamaV3 upstream](https://github.com/turboderp-org/exllamav3/blob/master/README.md), [TabbyAPI upstream](https://github.com/theroyallab/tabbyAPI) |
| vLLM | Complete `vllm_local`; keep its dependency stack in a separate environment/container. Start with one GPU and one supported model. | Linux + supported NVIDIA GPU first; WSL2 and AMD are separately validated recipes. | [vLLM upstream](https://github.com/vllm-project/vllm), [GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) |
| SGLang | Complete `sglang_local` using its OpenAI-compatible endpoints. Start with one GPU; preserve backend scheduling and prefix reuse. | Linux + supported NVIDIA GPU first; other accelerators require separate validation. | [API documentation](https://docs.sglang.io/docs/basic_usage/openai_api), [installation](https://docs.sglang.io/docs/get-started/install) |

These are LewLM's proposed initial support targets, not claims that upstream engines only work on those platforms. HTTP bridge compatibility is independent of where an engine can execute. Current upstream requirements move quickly: never copy one engine's CUDA/PyTorch pins into another engine's environment or assume LewLM's present CUDA 12.6 image fits them all.

## Step 00 — Capture the baseline and choose reproducible validation inputs

**Touch:** existing tests, `benchmarking/`, `scripts/capture_host_validation.py`, `scripts/generate_release_manifest.py`; **proposed** `docs/validation/modernization-baseline.md` and `examples/backends/compatibility.json`.

1. Record the working revision, dirty-tree state, host OS/architecture, Python, installed optional extras, and available engines. Record CPU/GPU, RAM/VRAM, driver, toolkit, and container versions when present. Do not import every engine to discover it.
2. Run the focused regression command below and the full non-long-running suite when its declared dependencies are installed. Separate pre-existing failures from new ones; preserve full failure output.
3. Pick a small licensed chat model per backend that fits the validation machine. Record immutable model revision, tokenizer/chat-template revision, quantization, context/output limits, and artifact hash. Use the same model family and workload for cross-engine comparisons; label differing precision/artifacts explicitly.
4. For each recipe, resolve and record an exact engine release or commit, Python version, dependency lock, wheel hashes or image digest, supported GPU architectures, and driver/toolkit compatibility. TabbyAPI describes itself as a rolling release, so pin its commit/image rather than a moving tag. [TabbyAPI upstream](https://github.com/theroyallab/tabbyAPI)
5. Capture cold dependency installation, clean image build, no-change rebuild, app-only rebuild, process-to-health, engine-to-ready, model cold load, first-token latency, warm throughput, and peak RAM/VRAM. Keep downloads and network time separate from compile/load time.

**Test and exit:** the baseline report contains commands and actual results, or explicit deferrals. `compatibility.json` is schema-validated and rejects missing versions/digests for recipes labeled validated. No benchmark number may be invented to fill a missing hardware run.

Existing portable regression command, from the repository root in a development environment:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/unit/test_settings.py \
  tests/unit/test_external_adapter_runtime.py \
  tests/unit/test_ollama_inventory.py \
  tests/unit/test_runtime_catalog.py \
  tests/unit/test_routing.py \
  tests/unit/test_install_profiles.py \
  tests/unit/test_model_residency.py \
  tests/unit/test_stream_abandonment.py \
  tests/integration/test_host_integration_api.py \
  tests/integration/test_stream_cancellation.py
```

Do not treat this list as the only tests required for later changes. Each step adds the behavior-specific cases below.

## Step 01 — Define endpoint identity and capability evidence

**Depends on:** 00. **Touch:** `config/settings.py`, `core/contracts.py`, `runtime/catalog.py`, `runtime/identity.py`, `runtime/residency.py`, `core/bootstrap.py`, `install_profiles.py`, `registry/ollama_inventory.py`.

1. Add a typed endpoint configuration collection, **proposed** `external_endpoints`, with stable `endpoint_id`, existing profile ID, enabled flag, loopback base URL, optional secret-reference name, and separate connect/read/pool timeout values. Validate duplicate IDs and invalid URLs at configuration time.
2. When the collection is absent, synthesize `legacy-default` from the existing `LEWLM_EXTERNAL_ACCELERATOR_*` settings. If an explicit collection and an enabled legacy endpoint are both supplied, reject the ambiguous configuration with migration guidance. Preserve Ollama's existing enablement/locality checks.
3. Preserve `RuntimeAffinity.EXTERNAL_ACCELERATOR`. Add an endpoint-instance lookup within the catalog; do not overload one affinity entry with whichever endpoint happened to register last. Resolve endpoint-bound manifests explicitly; retain `get_runtime(affinity)` for legacy/default callers.
4. Give each endpoint adapter a stable unique runtime name. Keep legacy identity unchanged in single-endpoint mode. Include endpoint identity in residency, discovery, capability, response-cache, coalescing, benchmark, and measured-preference keys. Do not merge different servers merely because they advertise the same model name.
5. Reuse `BridgeProfile` and capability-evidence types. Distinguish advertised capability, validated request support, enforcement mode, and observed runtime behavior. Unknown stays unknown. A profile label alone must not assert active prefix caching, constrained decoding, or cancellation support.
6. Add endpoint/profile/model evidence to existing health, doctor, and runtime responses. Fast health paths use cached state; active generation probes remain explicit operations.

**Test and exit:** cover legacy migration, ambiguous settings, two endpoints advertising identical IDs, isolated discovery failures, secret redaction, and two-client shared residency. Run shared-runtime baseline/multiprocess tests where the local OS supports the harness. Old configurations and old model IDs must continue to work.

## Step 02 — Make the shared HTTP bridge complete and cancellable

**Depends on:** 01. **Touch:** `runtime/adapters/openai_compatible.py`, `runtime/base.py`, `core/contracts.py`, `core/chat.py`, `core/chat_streams.py`, `runtime/cancellation.py`, `runtime/sampling.py`, `tool_call_contract.py`, `structured_output.py`, `pyproject.toml`.

1. Extract a shared asynchronous HTTP transport with a service-owned, reusable connection pool and bounded connection limits. Promote the already-used lightweight `httpx` dependency to an appropriate runtime dependency; do not add upstream engine SDKs. Close the pool through the existing service lifecycle. Replace the thread-per-stream/unbounded-queue path; use direct backpressured iteration or an explicitly bounded queue.
2. Normalize server-root and `/v1` base URLs once so requests never become `/v1/v1/...`. Keep loopback-only validation, disable ambient HTTP proxies for this transport, and reject redirects rather than allowing them to escape the configured endpoint. Resolve backend credentials independently of caller authentication and redact them everywhere.
3. Map `SamplingControls`, stop sequences, tool declarations/choices/results, and structured-output requests through a typed translation layer. Preserve unset-versus-explicit values. Reject or explicitly report unsupported controls using existing policy; never silently claim they were applied. Audit prompt compilation so native tool/JSON support does not receive duplicate instructions or double chat templates.
4. Add an optional structured runtime-stream contract for content, reasoning, tool-call fragments, usage, finish, and errors. Adapt legacy `AsyncIterator[str]` runtimes into it; do not require a simultaneous rewrite of native MLX/llama.cpp. Add typed tool-call data to internal request/response contracts where it is missing.
5. Parse SSE as events: handle partial UTF-8/network chunks, CRLF, comments, multiline data, empty choices, usage-only terminal events, and `[DONE]`. Accumulate tool arguments by choice/tool index with stable IDs. Preserve finish reasons. Unexpected EOF before a valid completion is an error, not a fabricated success.
6. Map the structured stream into existing chat and responses-style surfaces. Preserve the existing terminal usage/prompt-trace conventions and reasoning-visibility policy. A native upstream tool call passes through LewLM's existing validation/authorization path; do not execute it both in the engine and in LewLM. Keep upstream automatic tool execution disabled in the recipes.
7. Propagate task cancellation, request-handle cancellation, client disconnect, and deadline expiry to closure of the upstream request. Release scheduler admission and residency leases exactly once. Describe GPU abort as unverified until a real server demonstrates it; closing a socket alone proves only transport cancellation.
8. Preserve upstream status distinctions through existing error types: authentication, invalid request/context, rate limiting, unavailable model, transport timeout, and malformed response. Do not automatically retry generation after an ambiguous upstream failure, or after any output/tool-call fragment has reached the caller.

**Test and exit:** extend `test_external_adapter_runtime.py` with real loopback fake-server tests for fragmented SSE, tool-only replies, structured output, sampling, secrets, redirects, timeouts, 401/429/5xx, and malformed/truncated streams. A slow consumer must not cause unbounded buffered output. Cancellation must close the fake upstream connection and restore lease/admission counts within a bounded test deadline; a second stream must continue. Run chat/responses, tool-call, async-client, and cancellation regression tests. No GPU is required for this step.

**Implementation status:** completed on 2026-09-17. See the [bridge contract](backend-bridge-contract.md) and [validation record](../validation/modernization-step-02.md). Engine-specific GPU abort remains unverified until steps 05–08 run on their required hardware.

## Step 03 — Make CPU/CUDA installation and rebuilding inexpensive

**Depends on:** 00; integrate with 01–02 before release. **Touch:** `pyproject.toml`, `Dockerfile`, `Dockerfile.cuda`, `.dockerignore`, `docker-compose.yml`, `.env.example`, `.github/workflows/ci.yml`, `install_profiles.py`, `runtime/llamacpp/build_flavor.py`, installation/Docker docs.

1. Add **proposed** `llamacpp_runtime` containing only dependencies needed for GGUF inference. Keep legacy `llamacpp` behavior intact for this release and retain `gguf_conversion`. Offer explicit serving-only and full/conversion image targets; keep existing full-image commands working. Documents/OCR and Torch must not be installed into the lean bridge/serving image merely because a user might convert a model later.
2. Generate platform-specific locked dependency inputs. Install/build dependencies before copying changing application source; install the final LewLM wheel with `--no-deps` after the dependency layer. Include the metadata files required by setuptools in the wheel stage. App-only edits must not rerun native dependency compilation.
3. Use BuildKit cache mounts for package downloads and compiler caches. Remove `PIP_NO_CACHE_DIR=1` from cached builder operations; caches remain outside the final image. Keep dependency locks, not moving package resolution, as the dependency-layer input. [Docker cache guidance](https://docs.docker.com/build/cache/optimize/)
4. Prefer a verified, pinned llama-cpp-python wheel when the selected OS/Python/accelerator combination has one. Verify wheel flavor and run an inference probe; a CPU wheel is not a successful CUDA installation. Keep an explicit source-build fallback and restrict `FORCE_CMAKE=1` to that path. Available wheel combinations are release-specific. [llama-cpp-python installation](https://github.com/abetlen/llama-cpp-python)
5. In source builds use Ninja, Release mode, `ccache`, and explicit C/C++/CUDA compiler launchers where supported. Add **proposed** `BUILD_JOBS`, default 4, overridable for constrained hosts, and wire it into CMake/native extension compilation. Validate that the tools actually honor it. Avoid unbounded `nproc` on memory-limited runners. [llama.cpp build guidance](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
6. Keep the broad release architecture list as an explicit compatibility profile. Add a documented per-device recipe using the existing `CUDA_ARCHITECTURES` argument (for example `89` only on a verified Ada target). Validate the chosen SM against the selected toolkit before a long build. Do not use `native` for distributable artifacts or depend on a GPU being visible during Docker build. [CMake CUDA architecture selection](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html)
7. Key native wheel/compiler caches by source revision, OS/CPU architecture, Python ABI, compiler, build flags, CUDA version, and target SMs. Separate CPU and CUDA wheels even when their Python package version is identical. Changing an architecture or backend flag must invalidate the correct artifact.
8. Keep portable CPU artifacts conservative. `GGML_NATIVE=OFF` is already present, but inspect effective compiler flags and test the minimum supported CPU ISA; that flag alone is not evidence that every old CPU can execute every enabled instruction. Offer host-tuned CPU builds as an explicit local option.
9. Pin compatible CUDA builder/runtime images and retain required runtime libraries. Keep quantization-tool builds CPU-only and separately cacheable. Resolve Torch from a verified CPU or CUDA lock only in images that need it. Do not infer a compatible Torch CUDA build from the default package index.
10. Make fast CI use `CONVERSION_TOOLS=disabled` when only boot/health is being tested. Retain a separate gate for the full conversion image. Provide persisted model/cache volumes and verify write permissions as the runtime user.

**Test and exit:** clean-build, cached-build, and app-only-change rebuild each selected image; retain BuildKit logs and elapsed times. An app-only edit must perform zero llama.cpp/CUDA compilation. Verify lean installation has no Torch/Transformers/engine packages, `doctor` works, and the full image still converts and quantizes a small fixture. Linux CPU/container tests can run in a suitable local container runtime; real CUDA execution requires NVIDIA hardware. Test CUDA offload and model generation there, not just package import. Test native Windows wheel/import safety on Windows and preserve the current container fallback. Report missing environments explicitly.

**Implementation status:** implemented on 2026-09-17; portable acceptance passed. See the [validation record](../validation/modernization-step-03.md). Docker build/rebuild timings, Linux ISA evidence, CUDA compile/offload, pinned locks, and native Windows wheel safety are deferred there with exact commands; the `docker-full` CI job produces the rebuild evidence on the next push.

## Step 04 — Discover external models and route without losing fallbacks

**Depends on:** 01–02. **Touch:** `registry/service.py`, `registry/discovery.py`, `registry/ollama_inventory.py`, `utils/model_identity.py`, `runtime/catalog.py`, `routing/service.py`, `routing/measured_preferences.py`, `runtime/support_strategy.py`.

1. Add endpoint inventory using `/v1/models`, with bounded concurrency and per-endpoint failures. Give successful results a configurable TTL (proposed default 30 seconds), retain explicit refresh, and preserve the short failure retry behavior. An unreachable endpoint is not evidence that its models were deleted.
2. Represent a new advertised model with a stable endpoint-qualified ID and **proposed** `external://<endpoint_id>/<encoded-upstream-id>` source URI. Store the exact upstream ID separately. Extend path guards so URI-backed models never enter file hashing, conversion, or local weight loading. Preserve existing Ollama IDs and `ollama://` handling.
3. Reuse `ModelFormat.HUGGINGFACE` for verified HF artifacts and add **proposed** `ModelFormat.EXL3` only where artifact evidence supports it. If inventory does not reveal the weight format, report unknown and allow endpoint-bound serving through verified capabilities. Do not require a local file or automatic conversion just to use an advertised model.
4. Bind Ollama manifests to the Ollama endpoint, not the currently selected accelerator. Preserve cloud opt-in and execution-locality metadata. Generic loopback transport alone does not prove local inference.
5. Route explicit model/endpoint selections exactly. Default routing retains current behavior; configured preferences apply only to compatible, ready candidates. A same-name model on another endpoint is not automatically the same artifact.
6. Preserve native llama.cpp GGUF, llama.cpp-server, and Ollama paths as selectable candidates. Automatic fallback may happen only before upstream generation is submitted, only under an explicit fallback policy, and only to a registered compatible artifact. Cross-format model substitution needs an explicit verified alias mapping. Never attempt to load EXL3 directly with llama.cpp.
7. Record chosen endpoint, engine profile, locality, and fallback reason in existing execution metadata. After generation submission or partial output, surface failure and let the caller decide whether to retry; do not splice generations or repeat tool work.

**Test and exit:** two live fake endpoints and Ollama can coexist; identical upstream names remain distinct; inventory refresh adds/removes only the correct records; failure preserves the last known inventory with stale status. Test explicit routing, preflight fallback, unsupported-format rejection, cloud-disabled behavior, and no mid-stream retry. Run registry, Ollama, routing, model-inventory, residency, and cache-isolation tests.

**Implementation status:** completed on 2026-09-17; portable acceptance passed. See the [validation record](../validation/modernization-step-04.md). Real-engine inventory and EXL3 format evidence are validated in steps 05–08.

## Step 05 — Validate and document oMLX

**Depends on:** 02 and 04. **Touch:** existing `omlx` profile, install/readiness reporting; **proposed** `examples/backends/omlx/` and `docs/operations/backends/omlx.md`.

1. Use the existing `omlx` identity. Supply an operator-run recipe pinned by step 00, with an existing model directory, loopback binding, credential reference, and explicit memory/cache limits. Do not install another oMLX copy if the operator already has a running server.
2. Validate text chat and streaming first. Enable tools, JSON-schema enforcement, vision, embeddings, or rerank only for the exact model/configuration that passes their probes. oMLX documents these APIs and tiered caching, but endpoint availability is not a universal model capability. [oMLX upstream](https://github.com/jundot/omlx)
3. Let oMLX manage physical model loading and cache eviction. Report bridge leases separately from upstream residency. A LewLM unload may release local bookkeeping but must not claim it freed upstream RAM; destructive upstream lifecycle control remains outside this milestone.
4. Benchmark repeated-prefix conversations with LewLM response caching disabled. Preserve engine-native cache behavior and return unknown for hit counters the engine does not expose. Keep native MLX, llama.cpp, and Ollama routing unchanged.

**Test and exit:** portable profile/translation tests run everywhere. On Apple Silicon, execute the common real-engine acceptance suite below, repeated-prefix measurements, server restart/refresh, and two simultaneous chats. Defer that hardware lane on other hosts. Publish the exact passing model/configuration, not a blanket oMLX support claim.

**Implementation status:** completed on 2026-09-17 with the Apple Silicon lane executed for real. `omlx` is `validated` in `examples/backends/compatibility.json` for oMLX `b45fb7e` + `mlx-community/Qwen2.5-0.5B-Instruct-4bit` on this host; see the [validation record](../validation/modernization-step-05.md), the [recipe](../../examples/backends/omlx/README.md), and the [operator doc](../operations/backends/omlx.md). Tools are inconclusive on the 0.5B model; vision/embeddings/rerank are not probed. The common suite lives in `scripts/backend_acceptance.py` and the prefix measurement in `scripts/bridge_prefix_benchmark.py`, both HTTP-only, for reuse by steps 06–08 and Chap.

## Step 06 — Add ExLlamaV3 through TabbyAPI

**Depends on:** 02 and 04. **Touch:** shared adapter profile registry/settings, format/evidence reporting; **proposed** `examples/backends/exllamav3-tabby/` and matching operator documentation.

1. Add `exllamav3_tabby` to settings validation, profile/evidence reporting, and install guidance. Reuse the shared transport; do not embed ExLlamaV3 in the LewLM process. TabbyAPI is upstream's recommended OpenAI-compatible server. [ExLlamaV3 README](https://github.com/turboderp-org/exllamav3/blob/master/README.md)
2. Provide one pinned Linux/NVIDIA setup recipe with a small verified EXL3 artifact, model ID mapping, loopback exposure, and inference credential reference. Disable automatic tool execution and keep administrative model-loading credentials out of Chap. Verify selected TabbyAPI configuration keys against the pinned commit.
3. Prefer matching precompiled artifacts; if the selected ExLlamaV3 installation uses first-import JIT compilation, expose that cold-start cost and persist its extension cache. Bound `MAX_JOBS`. Do not describe moving compilation from installation to first request as a speed improvement. [ExLlamaV3 installation/build notes](https://github.com/turboderp-org/exllamav3/blob/master/README.md)
4. Validate model formats and feature support from the running server. Do not treat ExLlamaV2/EXL2 instructions as V3 evidence. Keep model conversion an explicit offline action rather than extending chat to perform it.
5. Include the selected server's shared-memory and memory-budget requirements in container configuration. TabbyAPI documents shared-memory needs for tensor parallelism/CPU MoE offload; the initial LewLM recipe should remain single-GPU and small-model. [TabbyAPI deployment](https://github.com/theroyallab/tabbyAPI)

**Test and exit:** portable tests cover profile selection, format gating, auth, and error translation. Linux/NVIDIA runs the common real-engine suite, first-start versus cached restart, and memory observations. Native Windows needs its own install/generation/cancellation proof; a Linux-container pass does not certify native Windows. An unavailable Tabby endpoint must leave llama.cpp and Ollama usable.

**Implementation status:** portable part completed on 2026-09-17 (`exllamav3_tabby` profile, provider evidence, digest-pinned recipe with config keys verified at the pinned commit, docs, tests). The Linux/NVIDIA lane is deferred with exact commands in the [validation record](../validation/modernization-step-06.md); `exllamav3_tabby` stays `deferred` in the compatibility manifest until it passes.

## Step 07 — Complete vLLM integration

**Depends on:** 02 and 04. **Touch:** existing `vllm_local` profile; **proposed** `examples/backends/vllm/` and corresponding operator docs.

1. Ship an isolated pinned image or virtual-environment recipe. Derive its Python/CUDA/PyTorch constraints from the selected vLLM release and run a compatibility preflight before download/build. Prefer upstream binary distributions where compatible. [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
2. Validate the recipe's `vllm serve` arguments against that release's CLI. Set explicit served model name, loopback exposure, context limit, GPU memory budget, and bounded request concurrency; avoid consuming all VRAM on a desktop. [vLLM serve reference](https://docs.vllm.ai/en/latest/cli/serve/)
3. Start with text chat and streaming. Capability-gate tool calling and structured output. Select and record the model's required tool parser/chat template; an OpenAI-compatible endpoint does not by itself guarantee correct automatic tool choice. [vLLM tool-calling documentation](https://docs.vllm.ai/en/latest/features/tool_calling/)
4. Preserve upstream continuous batching and prefix caching. Disable redundant LewLM microbatch delay/coalescing for this endpoint while retaining bounded admission and per-request cancellation. Measure through step 09 rather than assigning a universal throughput preset.
5. Persist the versioned vLLM compilation cache and measure a restart with compatible artifacts. Record actual compile/cache-hit evidence; keep model download cache and compiled-code cache separate. [vLLM compile-cache design](https://docs.vllm.ai/en/latest/design/torch_compile/)

**Test and exit:** portable profile/translation tests plus the common suite on Linux/NVIDIA. Verify two concurrent requests actually reach upstream concurrently, warm-restart timings, memory bounds, and parser behavior. WSL2/ROCm are separate lanes; keep them unvalidated until measured. Distinguish this profile from `vllm_mlx` in UI-facing evidence and all docs.

**Implementation status:** portable part completed on 2026-09-18 (digest-pinned `v0.29.0` recipe with every `vllm serve` argument checked at the commit, `scripts/engine_preflight.py`, explicit backend-native batching ownership on the bridge, `vllm_local`/`vllm_mlx` separation, tests, docs). The Linux/NVIDIA lane is deferred with exact commands in the [validation record](../validation/modernization-step-07.md); `vllm_local` stays `deferred` in the compatibility manifest until it passes.

## Step 08 — Complete SGLang integration

**Depends on:** 02 and 04. **Touch:** existing `sglang_local` profile; **proposed** `examples/backends/sglang/` and corresponding operator docs.

1. Ship an isolated pinned setup recipe. Use that release's install matrix; do not freeze a CUDA requirement from a moving documentation page into LewLM's generic package dependencies. SGLang publishes platform-specific installation guidance. [SGLang installation](https://docs.sglang.io/docs/get-started/install)
2. Validate a `python -m sglang.launch_server` recipe with explicit model path, served identity, loopback exposure, context, and memory/concurrency limits. Set `--mem-fraction-static`, chunked prefill, and graph-capture sizes only from measured profiles for the selected release. Do not globally enable `--enable-torch-compile`; current upstream documentation flags that path as out of maintenance. [SGLang server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
3. Exercise the shared OpenAI request/stream translation. Capability-gate native structured output, tools, and reasoning parsing per model. Preserve backend prefix reuse without reconstructing KV state in LewLM. [SGLang API documentation](https://docs.sglang.io/docs/basic_usage/openai_api)
4. Persist only supported compilation/kernel caches for the pinned stack. Record paths and cache keys in the recipe. Prefer matching prebuilt kernels when available; do not prescribe speculative FlashInfer wheel names or share incompatible engine caches.

**Test and exit:** portable profile/translation tests plus the common suite on Linux/NVIDIA. Measure repeated-prefix requests and concurrent streams with LewLM response caching disabled, memory bounds, and cold/warm startup. Run fake-server inventory/auth/error tests on other OSes; defer GPU execution explicitly.

**Implementation status:** portable part completed on 2026-09-18 (digest-pinned `v0.5.19` recipe with every `launch_server` argument checked at the commit, `qwen25`/`xgrammar` recorded, torch.compile deliberately off, preflight preset, observable `usage.cached_tokens` for `--enable-cache-report`, fake-server inventory/auth/error tests, docs). The Linux/NVIDIA lane is deferred with exact commands in the [validation record](../validation/modernization-step-08.md); `sglang_local` stays `deferred` in the compatibility manifest until it passes.

## Step 09 — Tune non-MLX runtime latency without inflating middleware

**Depends on:** 03–08; only tune engines with passing hardware evidence. **Touch:** `runtime/scheduler.py`, `runtime/request_coalescer.py`, `runtime/llamacpp/runtime.py`, `serving_profiles.py`, `benchmarking/`, `routing/measured_preferences.py`, telemetry.

1. Measure three startup phases separately: LewLM ready, engine ready, and model warm. Keep health/model listing responsive while an engine is unavailable or warming. Reuse bounded discovery caches; do not synchronously run full capability probes on each UI refresh.
2. Add two measured workload presets, **proposed** `interactive` and `throughput`, through existing serving profiles. Interactive favors first-token/p95 latency at concurrency 1–2; throughput tests higher concurrent load. Store host, engine/configuration, model revision, precision, and workload in the profile fingerprint; reject stale recommendations after those inputs change.
3. For external engines, retain an admission cap but remove duplicate batching waits and global serialization. Keep backend-native token scheduling inside the engine. Verify that many model aliases on one endpoint cannot bypass its aggregate admission cap.
4. For llama.cpp, benchmark CPU thread counts against physical cores, batch/microbatch sizes, context size, mmap behavior, and accelerator offload. Avoid simultaneously saturating CPU threads with several requests. Evaluate BLAS for prefill separately from decode. Add controls only when evidence identifies a useful setting; reuse current settings first.
5. Persist compiler/kernel caches per compatible stack and writable runtime identity. Use exact supported paths for the selected engine. Account for cache size and explicit pruning; never clear model weights as part of compilation-cache cleanup.
6. Compare eager/reduced graph capture against compiled/graph execution for interactive startup and steady-state throughput. Changes must improve the chosen workload without hiding warmup costs. Do not disable CUDA graphs universally or enable expensive compilation universally.
7. Warm a model only on an explicit user/operator warm action or a configured startup policy. Reuse the residency manager for native single-flight loading. For bridges, report warmup as a probe of upstream state; do not pretend LewLM owns its eviction policy.
8. Preserve the portable CPU route on non-NVIDIA machines. Keep existing ONNX paths intact. Vulkan/HIP and alternative vLLM/SGLang accelerator recipes are opt-in follow-ups, each requiring its own build and inference evidence; do not make CUDA detection a prerequisite for LewLM startup.

**Test and exit:** compare before/after using the benchmark protocol below. Run scheduler fairness/backpressure, response-cache, residency, and cancellation regressions. Demonstrate that a long prefill does not indefinitely block a short request at the LewLM layer. Validate CPU tuning on CPU hardware and CUDA tuning on NVIDIA hardware. Unsupported hardware produces guidance and a usable fallback, not a failing core import.

**Implementation status:** portable contract completed on 2026-09-18 (startup phases on `GET /v1/runtime`, `interactive`/`throughput` presets with a measured-input fingerprint and `stale` rejection, aggregate admission cap proven across aliases, explicit-only warming verified). CPU/CUDA tuning measurements are deferred with exact commands in the [validation record](../validation/modernization-step-09.md); no tuning result is claimed.

## Step 10 — Package the stable Chap integration contract

**Depends on:** 01–02 and 04; extend fixtures after 05–08. **Touch:** `api/schemas/`, `api/openapi.py`, existing health/models/chat/responses/events/runtime routes, `app_helpers.py`, `examples/integration-bundle.json`, `docs/guides/host-app-integration.md`.

1. Keep Chap pointed at one LewLM base URL. Reuse `GET /v1/health`, `GET /v1/models`, model-capabilities routes, `GET /v1/runtime`, `POST /v1/chat/completions`, `POST /v1/responses`, and existing cancellation/events surfaces. Add fields to these contracts instead of creating an engine-specific API for Chap.
2. Expose stable model ID, selected endpoint/profile, capability/enforcement evidence, unavailable reason, and configured fallback result in appropriate existing schemas. Separate service health, engine availability, and model warmth so Chap can show useful state without guessing from an HTTP 200.
3. Publish exact request/response/SSE examples for text, native tool calls, JSON output, terminal usage, unavailable engine, and cancellation. Define which fields can be absent and how an incomplete stream is represented. Keep raw backend payloads and credentials out of public errors.
4. Preserve `x-request-id`, `x-lewlm-correlation-id`, and application identity semantics. Chap may supply a cancellation handle and `application_id=chap`; application identity still grants no authorization. Keep cancellation's process-local ownership visible.
5. Update synchronous/asynchronous typed clients and integration schemas in the same change as API fields. Build a reproducible schema-export command if one is missing; do not hand-edit generated schemas. Validate the bundle under pinned Pydantic versions so known schema drift cannot excuse a contract failure.
6. Add **proposed** `examples/chap_backend_smoke.py`, operating entirely through HTTP with a fixture mode and an optional real-model mode. It must list models/capabilities, stream a reply, cancel a second request, exercise a tool round trip and JSON output where advertised, and report structured failures. Keep it useful before Chap UI code exists.
7. Add **proposed** `docs/guides/chap-validation.md` with a later UI checklist: model picker, capability-driven controls, first-token rendering, stop button, tool deltas/results, JSON display, engine restart, and fallback explanation. Record UI results separately from backend contract tests.

**Test and exit:** run `test_integration_bundle.py` including its schema snapshot in the pinned environment, OpenAPI `$ref` checks, host-integration, async-client, and request-cancellation tests. Run the smoke script against the fake HTTP backend on every OS. Chap UI testing may remain pending; backend/API correctness must not wait for it. If Chap runs as a separate browser origin, test a narrowly configured allowed origin and streaming headers without introducing a wildcard policy.

**Implementation status:** completed on 2026-09-18. `lewlm.testing.fake_backend` (a shippable LewLM + fake engine for UI development with no model), `examples/chap_backend_smoke.py` (13 HTTP checks, fixture and real-model modes, run by CI on Linux/macOS/Windows), `scripts/export_integration_bundle.py` (generated `schemas`/`errors` with `--check`, plus a captured `chap` section of exact payloads and field notes), additive `engines[]` on health, endpoint/engine state on `capability_availability[]`, a `cancelled` terminal chunk, a consistent 503 for an engine outage, a residency race fix, and `docs/guides/chap-validation.md` with the pending UI checklist. See the [validation record](../validation/modernization-step-10.md).

## Step 11 — Add CI and hardware acceptance lanes

**Depends on:** all implementation steps. **Touch:** `.github/workflows/ci.yml`, tests/support, existing acceptance and release scripts, **proposed** backend acceptance runner.

1. Keep the lightweight Linux/macOS/Windows matrix. Add endpoint/migration/stream contract tests there without installing model engines. Include shared-runtime subprocess tests once harness behavior is validated on each OS.
2. Keep full Linux regression coverage and add lean/full image build checks. Cache by the exact dependency/build inputs. Verify app-only rebuild behavior with build logs, not a brittle shared-runner timing assertion.
3. Register explicit pytest markers for new OS/hardware tests before using them. Run real engines only in opt-in/local or appropriately provisioned CI lanes. A missing GPU/engine should produce an actionable skip reason in the raw test output and a `deferred` result in release evidence.
4. Feed per-engine results into `capture_host_validation.py` and release manifests. Store commands, versions, artifact identity, measured results, logs, and observed capability evidence. Only passing exact configurations may be labeled validated in the compatibility manifest.
5. Require the following platform separation. Add AMD/Vulkan/HIP or multi-GPU lanes only when their recipes are actually being promoted.

| Lane | Required proof | Deferral rule |
| --- | --- | --- |
| Any development OS | Fake-server translation, tools/JSON/SSE, cancellation, migration, routing, fallback, schema/client checks | No hardware-based exemption |
| Linux CPU | Lean install, portable GGUF generation, image boot, conversion image, cold/warm rebuild evidence | Defer if Linux/container runtime unavailable |
| Apple Silicon macOS | oMLX suite and native MLX/llama.cpp/Ollama coexistence | Defer on non-Apple hardware |
| Linux NVIDIA | vLLM, SGLang, ExLlamaV3/TabbyAPI suites; llama.cpp CUDA offload/build/cache validation | Defer without supported NVIDIA hardware/driver |
| Native Windows | Core install, llama.cpp import/generation, Ollama bridge, shutdown/cancellation; ExLlamaV3 only if separately promoted | Linux or WSL results do not satisfy this lane |
| Windows + WSL2 | Documented engine recipes and Windows-Chap-to-LewLM connectivity | A Linux pass alone does not prove WSL networking |
| Chap UI, later | End-user checklist against the same public contracts | Mark UI acceptance pending, never silently complete |

**Test and exit:** demonstrate one passing fixture-only CI lane, one deliberate hardware deferral, and one deliberate contract failure that blocks acceptance. Deferred hardware must remain visible in the release bundle.

**Implementation status:** completed on 2026-09-18. `scripts/backend_lanes.py` (detect/run/summary with honest deferrals and blocking failures), registered hardware markers with `tests/hardware` lanes that skip with actionable reasons, `backend_lanes` in the release manifest, the CI matrix's engine-free contract lane on Linux/macOS/Windows with pinned pydantic and the bundle gate restored. All three required demonstrations are recorded in the [validation record](../validation/modernization-step-11.md); hardware lanes other than Apple Silicon remain deferred.

## Step 12 — Roll out with an easy rollback

**Depends on:** 11 and the relevant real-engine lane passing.

1. Publish version-pinned setup recipes and the compatibility manifest. Add concise doctor guidance: what is installed, what is reachable, what works, and the next command needed. Keep normal setup to choose an optional backend, follow its recipe, configure its endpoint, and run doctor.
2. Keep accelerators opt-in. Promote a recommended path only for a validated host/model/workload tuple. Do not replace broad default routing with an unmeasured engine ranking.
3. Publish the legacy-settings migration example and an endpoint-disable rollback example. Disabling a new endpoint restores the previously configured routing candidates without deleting models, caches, or Ollama state.
4. Run the old llama.cpp-only and Ollama-only quickstarts from clean environments. Verify the old full/conversion installation path as well as the new lean path.
5. Record remaining deferred OS/hardware and Chap UI work. A release may ship an experimental adapter with clear validation status; it must not label that adapter universally supported.

**Test and exit:** enable a new endpoint, serve a request, stop/disable it, and serve through an explicitly compatible existing path. Test the same operation with an active stream: fail it transparently without replay, then allow a new request on the selected fallback. Docs and doctor must agree with actual behavior.

**Implementation status:** completed on 2026-09-18. `lewlm doctor` reports each engine's state and the next command; the enable → serve → outage → alias fallback → interrupted stream (no replay) → disable sequence is proven over HTTP in `tests/integration/test_rollout_rollback.py`; a stale endpoint is no longer a routing candidate so the fallback engages before submission; the legacy migration and the one-entry rollback are documented in `docs/operations/backends/rollout-and-rollback.md`; the lean install was verified from a clean environment. The llama.cpp-only and Ollama-only clean-environment quickstarts and the real-engine enable/outage/alias/restart/disable sequence (oMLX + llama.cpp alias) passed on the Apple Silicon host on 2026-09-21; every hardware lane except Apple Silicon remains deferred with its commands in the [validation record](../validation/modernization-step-12.md), which also lists all remaining work.

## Common real-engine acceptance suite

Run this for each engine/model/configuration selected for promotion. Provide one reusable HTTP harness with engine-specific setup fixtures; avoid four copies of the same API test code.

| Case | Required observation |
| --- | --- |
| Discovery and identity | Correct advertised model, stable LewLM ID, accurate endpoint binding, refreshed inventory after restart |
| Chat and responses | Nonstreaming and streaming complete; one valid terminal outcome, finish reason, and honest usage |
| Sampling | Supported values arrive upstream; unsupported controls are rejected or explicitly reported |
| Structured output | Valid JSON/schema for an advertised native path; enforcement metadata distinguishes decoder enforcement from prompt guidance |
| Tools | Tool-only response and fragmented arguments reconstruct correctly; tool result continuation works; execution occurs once under existing authorization |
| Reasoning | Hidden reasoning is not leaked into visible text; configured visibility remains consistent |
| Cancellation | Client disconnect and named cancel stop middleware work, close transport, release leases; verify upstream stop separately where observable |
| Concurrency | At least two requests overlap upstream; cancelling one does not cancel the other |
| Failure | Bad key, missing model, timeout, overload, server stop, malformed/truncated stream produce stable errors; no post-submission automatic replay |
| Fallback | Disabled/unavailable endpoint does not remove native/Ollama candidates; only an explicitly compatible preflight fallback is selected |
| Lifetime | Warm/restart/cache behavior is recorded; local lease release is not presented as physical upstream unload |
| Optional modalities | Only advertised vision/embedding/rerank/audio capabilities are probed; failures narrow evidence rather than breaking plain text chat |

## Benchmark protocol and acceptance thresholds

These are proposed engineering gates, not claims about current performance.

1. Record five cold process starts and five warm-cache restarts separately. Keep model downloads outside startup timing. Cold compilation-cache tests must isolate all relevant caches in dedicated test directories; never wipe a user's caches.
2. For steady state, use three warmups followed by at least 30 measured requests per workload at concurrency 1, 2, and 4 where memory permits. Include short chat, long prefill, repeated conversation prefix, tool output, and constrained JSON. Record failures and excluded runs rather than discarding them silently.
3. Compare direct upstream requests against requests through LewLM with equivalent messages, template, sampling, and output limits. Report p50/p95 time to first token, inter-token latency, total latency, completed requests/sec, output tokens/sec, and peak memory. Report token counts as unknown when authoritative counts are unavailable.
4. Disable LewLM response caching/coalescing for engine-performance comparisons. Test response-cache benefits separately. Report prefix-cache hits only when observable; repeated-prefix speedups alone are inference, not a measured hit counter.
5. Required deterministic build gate: no native dependency recompilation after an app-only edit; an unchanged build reuses dependency layers. Initial performance target: at least 50% lower app-only rebuild wall time versus the baseline on the same machine. If unmet, retain the measured result and investigate before claiming the target achieved.
6. Initial middleware target: warmed p95 first-token overhead versus direct upstream no greater than `max(20 ms, 10% of direct p95)`, and warmed throughput regression no greater than 5% for equivalent requests. Benchmark cancellation/metadata overhead as part of the middleware. If the target fails, keep the feature opt-in and attach a profile explaining the bottleneck.
7. Suggested optimization adoption gate: at least 10% improvement in the preset's target metric, no more than 5% regression in its other primary latency/throughput metric, and no correctness or memory-budget regression. Repeat noisy comparisons before changing defaults. Do not put these timing thresholds into shared-runner unit tests.

## Deployment boundary and intentionally deferred work

Loopback means the network namespace where LewLM runs. A Compose service name or `host.docker.internal` is not currently a valid loopback endpoint. Initial recipes should run LewLM on the host with engine ports published only to host loopback, or use an explicitly tested shared network namespace. A Windows/WSL recipe must demonstrate its actual connectivity. Do not weaken loopback checks just to make a sidecar example work. Supporting private remote endpoints requires a separate network-policy design.

Keep this modernization bounded: no custom CUDA kernels, automatic model conversion/downloads during serving, universal engine installer, distributed scheduler, multi-node serving, or speculative-decoding framework rewrite. Existing capabilities remain intact. Chap-specific presentation/workflow stays in Chap; the engine-independent contracts and proof harness stay in LewLM.

## Agent handoff record

At the end of every step, append the following to the implementation tracking document created in step 00:

```text
Step:
Revision / changed paths:
Behavior implemented:
Compatibility / migration effect:
Commands run and results:
Real-engine evidence (engine + model + OS/GPU + pinned versions):
Deferred tests (exact missing prerequisite + command + expected result):
Benchmark artifacts and baseline comparison:
Rollback procedure:
Next eligible step:
```

Mark a step implementation-complete only after its portable acceptance passes. Mark its backend/platform validated only after that exact hardware lane passes. Keep Chap UI acceptance as its own status.
