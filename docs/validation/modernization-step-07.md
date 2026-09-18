# Modernization step 07 validation — vLLM

Implemented on 2026-09-18 on the baseline Apple Silicon host. The portable
part of this step is complete; the real-engine lane needs Linux + NVIDIA and
is **deferred**. Nothing about vLLM is labelled validated.

## What changed

| Area | Change |
| --- | --- |
| Preflight | `scripts/engine_preflight.py`: a host check to run *before* pulling an image — digest pin, Docker + `nvidia` runtime, driver ≥ the image's CUDA minimum, GPU SM against the image's compiled list, free VRAM, free host-loopback port. Presets for `vllm`, `sglang`, and `exllamav3-tabby` carry values read from each pinned image. JSON report for the evidence folder. It installs, pulls, starts, and stops nothing. |
| Recipe | `examples/backends/vllm/`: `docker-compose.yml` (image pinned by digest, host-loopback port only, `shm_size: 8g`, single GPU, unauthenticated `/health` healthcheck, key from `VLLM_API_KEY` so it is not in `ps`, separate weight and compile-cache volumes, no usage telemetry), `lewlm.env.example`, README with the preflight, serve, wire, prove, and promote procedure, plus the exact measurements the step-07 exit criteria ask for. |
| Bridge | `LocalOpenAICompatibleAdapterRuntime.continuous_batching_ownership()` reports `backend_native` for profiles whose feature map says the server batches (vLLM, SGLang, TabbyAPI, oMLX, vMLX, vLLM-MLX, TensorRT-LLM) and `unsupported` otherwise. Reporting only: `supports_continuous_batching` stays False, so LewLM's frontier microbatch window is never opened in front of a bridge; the serving snapshot now says `backend_native_batch` instead of `request_scoped` for these endpoints. |
| Docs | `docs/operations/backends/vllm.md` (nav), configuration reference (`vllm_local` vs `vllm_mlx`), capability-matrix row. |
| Manifest | `vllm_local` stays `deferred`; its candidate source is now the release commit and the reason/next step name the image digest and the exact promotion procedure. |

## Pins verified against upstream

| Item | Evidence |
| --- | --- |
| vLLM release `v0.29.0` = commit `98dff2a81d747d1dba01a47f939f48c3526d4206` | GitHub releases API (`latest`, published 2026-09-09) and `git/ref/tags/v0.29.0` on 2026-09-18 |
| Image `vllm/vllm-openai@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` | Docker Hub manifest list for tag `v0.29.0`; amd64 image `sha256:082ca6f0…`; config labels `ai.vllm.build.commit` / `org.opencontainers.image.revision` = the commit, `VLLM_IMAGE_TAG=vllm/vllm-openai:v0.29.0`, `CUDA_VERSION=13.0.2`, `TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 12.0`, `VLLM_ENABLE_CUDA_COMPATIBILITY=0`, entrypoint `vllm serve` |
| Dependency stack | `requirements/cuda.txt` at the commit: `torch==2.13.0`; `pyproject.toml`: Python `>=3.10,<3.15`; `docker/Dockerfile`: Python 3.12 |
| Driver requirement R580+ | `docs/getting_started/installation/gpu.cuda.inc.md` at the commit: CUDA 13 images need an R580 driver run normally; R535/R570 only via `VLLM_ENABLE_CUDA_COMPATIBILITY=1` on select professional/datacenter GPUs |
| `vllm serve` arguments | `vllm/engine/arg_utils.py` (`--revision`, `--tokenizer-revision`, `--served-model-name`, `--dtype`, `--seed`, `--max-model-len`, `--gpu-memory-utilization` default 0.92, `--max-num-seqs`, `--max-num-batched-tokens`, `--enable-prefix-caching` default on) and `vllm/entrypoints/launchers/cli_args.py` (`--host`, `--port`, `--enable-auto-tool-choice`, `--tool-call-parser`, `--enable-force-include-usage`, `--uvicorn-log-level`) at the commit |
| `VLLM_API_KEY` | `vllm/entrypoints/serve/middleware/register.py` at the commit: CLI `--api-key` else `envs.VLLM_API_KEY`; `authenticate.py`: guarded prefixes `/v1`, `/v2`, `/inference`, `/cohere` — `/health` is open |
| `hermes` parser for Qwen2.5 | `docs/features/tool_calling.md` at the commit: "For Qwen2.5 … use the `hermes` parser" |
| Compile cache | `docs/design/torch_compile.md` at the commit: `~/.cache/vllm/torch_compile_cache/<hash>/rank_0_0/`, copyable between starts; `VLLM_CACHE_ROOT` in `vllm/envs.py` relocates it |
| Upstream-overlap observation | `vllm/v1/metrics/loggers.py` at the commit logs `Running: %d reqs` periodically; `--disable-log-stats` was therefore left off |
| `shm_size` | vLLM's docker docs at the commit: `--ipc=host` *or* `--shm-size`; PyTorch shares tensors through `/dev/shm` |

## Portable results on this host

| Command | Result |
| --- | --- |
| `tests/unit/test_engine_preflight.py` — matching host passes; old driver, unlisted SM, low VRAM, and bound port fail independently with actionable text; missing GPU/Docker/toolkit fail; SGLang preset skips (not passes) the architecture check; every preset is digest-pinned and a tag is rejected; CLI report/exit code; explicit requirements without a preset | **7 passed** |
| `tests/unit/test_vllm_local_profile.py` — `vllm_local` vs `vllm_mlx` distinct in provider/profile/snapshot/name/feature metrics; `json_schema` forwarded as `decode_time` + `upstream_native` with `decoder_enforced: false`; bridge never opens a LewLM microbatch window yet reports `backend_native_batch` (generic profile stays `unsupported`); **two streams through the real chat orchestrator reach the fake engine concurrently** (each stream is gated on a barrier that only releases when both requests are inside the fake server), peak admission 2, cancelling one closes exactly its upstream connection while the other completes with `finish_reason: stop`, admission released; upstream 400 for `tool_choice` without the parser flags → structured `invalid_request`, one upstream call, key not leaked | **5 passed** |
| Focused regression + adapter/endpoint/inventory/serving-core/operations/library/stream suites | **264 passed** |
| `tests/unit tests/integration -m 'not long_running'` (integration-bundle snapshot deselected; owned by step 10) | **1102 passed, 11 failed** — all 11 are document/ingest/attachment tests failing with the step-00 host `pyexpat`/`libexpat` symbol mismatch; the same tests fail identically on a clean `HEAD` worktree |
| `scripts/engine_preflight.py --recipe vllm` on this Mac | `fail` — `container_runtime` (no daemon) and `gpu_present` (no `nvidia-smi`); the intended deferral signal |
| `scripts/validate_backend_compatibility.py` | 4 recipes; `vllm_local` deferred |

## Deferred measurements and exact prerequisites

| Measurement | Missing prerequisite | Follow-up command / expected observation |
| --- | --- | --- |
| Preflight pass | Linux host, NVIDIA driver ≥ R580, NVIDIA Container Toolkit, GPU with SM in `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | `python scripts/engine_preflight.py --recipe vllm --output preflight.json` → `result: pass` |
| Serve + common suite | Same host | `docker compose up -d` with the digest-pinned image, `lewlm scan`, `scripts/backend_acceptance.py --endpoint-id vllm`; expected: 0 failed; tools recorded per what the 0.5B model actually emits |
| Two requests overlap upstream | Same | during the `concurrency` case, vLLM's stats line shows `Running: 2 reqs`; the LewLM-side overlap proven here by the fake server is not a substitute |
| Cold start vs warm restart | Same | time `up` → `/health` 200 on first start (download + compile), then `docker compose restart vllm`; the `Using cache directory …torch_compile_cache…` line and whether graphs were loaded from cache |
| Memory bound | Same | `nvidia-smi --query-gpu=memory.used` while serving ≈ `0.30 × total` + activations |
| Middleware overhead | Same | `bridge_prefix_benchmark.py` at concurrency 1 and 2 against the roadmap target `max(20 ms, 10 % of direct p95)` |
| Coexistence | Same | `docker compose stop vllm`, `lewlm scan` (model kept, stale), route a llama.cpp and an Ollama model, `start`, rescan |
| WSL2 / ROCm | Separate lanes | not covered by this recipe; each needs its own connectivity/kernel proof |

## Rollback

Revert the step commit. The preflight script, recipe, docs, and manifest
wording are additive. `continuous_batching_ownership` on the bridge changes
only the reported `runtime_adapter.kind` (`backend_native_batch` instead of
`request_scoped`) for backend-batching profiles; no scheduling behaviour
depends on it.
