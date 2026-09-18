# Modernization step 08 validation — SGLang

Implemented on 2026-09-18 on the baseline Apple Silicon host. The portable
part of this step is complete; the real-engine lane needs Linux + NVIDIA and
is **deferred**. Nothing about SGLang is labelled validated.

## What changed

| Area | Change |
| --- | --- |
| Recipe | `examples/backends/sglang/`: `docker-compose.yml` (image pinned by digest, host-loopback port only, `shm_size: 8g`, single GPU, `/health` healthcheck — always open in SGLang's auth middleware — key on `--api-key` because the server reads it from nowhere else at the pin, separate weight and kernel-cache volumes, `--tool-call-parser qwen25`, `--grammar-backend xgrammar`, bounded context/memory/concurrency, chunked-prefill and CUDA-graph sizes left at defaults on purpose, `--enable-torch-compile` deliberately absent), `lewlm.env.example`, README with the preflight, serve, wire, prove, and promote procedure and the exact measurements the step-08 exit criteria ask for. |
| Preflight | `scripts/engine_preflight.py --recipe sglang` (added in step 07) carries the image digest, R580 minimum, port 30000, and an explicit *skip* — not a pass — for the compute-capability check, because SGLang publishes no compiled-architecture list for its kernel wheel at the pin. |
| Bridge | `_normalize_usage` keeps `prompt_tokens_details.cached_tokens` as `cached_tokens` and is now used on the non-streaming path too. Public `CompletionUsage` gains an additive `cached_tokens: int \| None` (default `None`), filled only when the backend reported the counter. This is the one observable prefix-cache hit counter the OpenAI contract carries (SGLang `--enable-cache-report`, vLLM `--enable-prompt-tokens-details`); absence means unknown, never zero. |
| Docs | `docs/operations/backends/sglang.md` (nav), configuration reference, capability-matrix row, HTTP API reference (`usage.cached_tokens`). |
| Manifest | `sglang_local` stays `deferred`; its candidate source is now the release commit and the reason/next step name the image digest and the exact promotion procedure. |

## Pins verified against upstream

| Item | Evidence |
| --- | --- |
| SGLang release `v0.5.19` = commit `0bcd822377da7b5718e674eaf9c870d349424dd1` | GitHub releases API (`latest`, published 2026-09-05); annotated tag `59f20bff…` dereferenced to the commit on 2026-09-18 |
| Image `lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9` | Docker Hub tag `v0.5.19-cu130` (same digest as `v0.5.19`); amd64 image `sha256:37bbbd34…`; config labels `ai.sglang.build.commit` / `org.opencontainers.image.revision` = the commit, `SGLANG_IMAGE_TAG=lmsysorg/sglang:v0.5.19`, `CUDA_VERSION=13.0.3`, cuDNN `9.14.0.64-1`, entrypoint `/opt/nvidia/nvidia_entrypoint.sh` |
| Dependency stack | `docker/Dockerfile` at the commit: base `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, `torch==2.13.0`, `sglang-kernel==0.4.6.post1` installed as a prebuilt `cp310-abi3` wheel for CUDA 13 (PyPI listing confirmed); `docs/docs/get-started/install.mdx`: Python ≥ 3.10, CUDA 13 by default, `-cu129` images for CUDA 12 hosts, `--shm-size`/`--ipc=host` in the documented `docker run` |
| Driver requirement R580+ | Image `NVIDIA_REQUIRE_CUDA=cuda>=13.0`; CUDA 13 needs an R580 driver run normally (same rule vLLM's docs state for its CUDA 13 image) |
| `launch_server` arguments | `python/sglang/srt/server_args.py` at the commit: `--model-path`, `--revision`, `--served-model-name`, `--host`, `--port`, `--api-key`, `--dtype` (choices include `bfloat16`), `--random-seed`, `--context-length`, `--mem-fraction-static`, `--max-running-requests`, `--tool-call-parser`, `--grammar-backend` (`xgrammar`, `outlines`, `llguidance`, `none`), `--enable-cache-report`; `--cuda-graph-max-bs` and `--disable-cuda-graph` are deprecated aliases at this commit and are not used |
| Auth shape | `python/sglang/srt/utils/auth.py` at the commit: `/health*` and `/metrics*` always allowed; other routes need `Authorization: Bearer <api_key>`; `--admin-api-key` gates management endpoints and is not configured |
| `qwen25` parser for Qwen2.5 | `python/sglang/srt/function_call/function_call_parser.py` (`ToolCallParserEnum`) and `docs/docs/advanced_features/tool_parser.mdx` at the commit |
| `--enable-torch-compile` off | `docs/docs/advanced_features/server_arguments.mdx` at the commit: "This feature is out of maintenance and might cause error"; the tuning page still lists it as an option, which is why the recipe states the choice explicitly |
| Tuning knobs left at defaults | `docs/docs/advanced_features/hyperparameter_tuning.mdx` at the commit describes `--mem-fraction-static`, `--chunked-prefill-size`, `--max-running-requests`, `--cuda-graph-max-bs-decode` as measured adjustments (`available_gpu_mem` log line); the roadmap tunes them in step 09 |
| `/v1/models` record | `python/sglang/srt/entrypoints/http_server.py` at the commit returns `id=served_model_name` and `max_model_len=context_len`, which LewLM reads as the context length |
| Upstream-overlap observation | `Decode batch. #running-req: N` log line, documented on the tuning page at the commit; the recipe keeps the default `info` log level so it is emitted |

## Portable results on this host

| Command | Result |
| --- | --- |
| `tests/unit/test_sglang_local_profile.py` — profile → `RuntimeProvider.SGLANG`, backend-native batching, `partial` prefix cache, nothing active; `/health` open while `/v1/models` needs the Bearer key (wrong key → structured 401, `inventory_state: failed`, recovery with the right key, `max_model_len` read); `cached_tokens` survives both the non-streaming and streaming bridge paths through the real chat orchestrator when the fake server reports it and is **absent** (not zero) when it does not, boolean values rejected, `CompletionUsage.cached_tokens` filled only from the counter; a stopped SGLang is a `RoutingError` naming `endpoint_id: sglang` / `engine_profile: sglang_local` / `fallback_policy: none`, the model stays registered as stale after a rescan | **4 passed** |
| `tests/unit/test_vllm_local_profile.py`, `tests/unit/test_exllamav3_tabby_profile.py` | 10 passed alongside |
| Adapter, transport, host-integration, async-client, cancellation, failure-envelope, tool-call wiring, library, contract-gap suites (public `CompletionUsage` change) | **244 passed** |
| `tests/integration/test_integration_bundle.py` | 1 failed, 8 passed — the same schema-snapshot failure before and after this change (`git stash` comparison); owned by step 10 |
| `scripts/validate_backend_compatibility.py` | 4 recipes; `sglang_local` deferred |

## Deferred measurements and exact prerequisites

| Measurement | Missing prerequisite | Follow-up command / expected observation |
| --- | --- | --- |
| Preflight pass | Linux host, NVIDIA driver ≥ R580, NVIDIA Container Toolkit | `python scripts/engine_preflight.py --recipe sglang --output preflight.json` → `result: pass` with `compute_capability` skipped |
| Kernel support for the host GPU | Same | first `docker compose up`: the server reaches `/health` = 200 and logs no missing-kernel error for the GPU's SM; record the SM in the validation record |
| Serve + common suite | Same | `lewlm scan`, `scripts/backend_acceptance.py --endpoint-id sglang`; expected: 0 failed; tools recorded per what the 0.5B model actually emits |
| Repeated-prefix requests | Same | `bridge_prefix_benchmark.py` at concurrency 1 and 2 with LewLM response caching untouched; report the speed-up as an observation; a separate run with `--enable-cache-report` gives `usage.cached_tokens` through LewLM |
| Two requests overlap upstream | Same | during the `concurrency` case, `Decode batch. #running-req: 2` in `docker compose logs sglang` |
| Cold start vs warm restart | Same | time `up` → `/health` 200 on first start (download + kernel JIT), then `docker compose restart sglang`; list `/root/.cache` and record which directories were reused |
| Memory bound | Same | the `available_gpu_mem=… GB` line before ready and `nvidia-smi --query-gpu=memory.used` while serving |
| Coexistence | Same | `docker compose stop sglang`, `lewlm scan` (model kept, stale), route a llama.cpp and an Ollama model, `start`, rescan |
| CUDA 12 / ROCm / Ascend / multi-GPU | Separate lanes | each needs its own image digest and proof |

## Rollback

Revert the step commit. The recipe, docs, and manifest wording are additive.
`CompletionUsage.cached_tokens` is an additive optional field with a `None`
default; `_normalize_usage` keeping `cached_tokens` only adds a key when the
backend supplied it.
