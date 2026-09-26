# vLLM (Linux + NVIDIA)

[vLLM](https://github.com/vllm-project/vllm) serves a model behind its own
OpenAI-compatible server with continuous batching and automatic prefix
caching. LewLM fronts it through the `vllm_local` bridge profile: discovery,
routing by served model name, sampling, `json_schema` output, tools, and
cancellation all cross the loopback boundary the same way they do for every
other endpoint. vLLM is never imported into LewLM, and LewLM never loads a
model on the server's behalf.

The pinned recipe — digest-pinned image, `docker compose` file whose every
argument was checked against `vllm serve` at the pinned commit, the LewLM
environment, the preflight, and the proof commands — is in
[`examples/backends/vllm/`](https://github.com/lew-cx/LewLM/tree/main/examples/backends/vllm).

## Status

**Passed on Windows + WSL2; deferred on bare-metal Linux.** On 2026-09-24 the
recipe ran unchanged, apart from `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, which is now
in the compose file. It ran in Docker Desktop on an RTX 5090 Laptop (SM 12.0),
with LewLM native on Windows. It passed the common acceptance suite (11/11),
the real-engine rollout and rollback sequence, and the Chap smoke in
real-model mode (12/12). See the
[Windows/Linux validation record](../../validation/modernization-windows-linux.md).
A WSL2 pass is its own lane. `examples/backends/compatibility.json` keeps
`vllm_local` as `deferred` until the suite passes on a bare-metal Linux/NVIDIA
host. Do not read this profile's presence in `LEWLM_EXTERNAL_ENDPOINTS` as
support elsewhere.

## `vllm_local` is not `vllm_mlx`

`vllm_local` names the Linux/NVIDIA vLLM server from upstream. `vllm_mlx` is
the profile for the Apple Silicon fork and carries its own evidence. They share
`RuntimeProvider.VLLM` but differ in `profile` everywhere Chap can read it:
`/v1/health` and `/v1/runtime` endpoint snapshots, `bridge_profile.profile_id`,
manifest metadata `external_profile`, and execution metadata `engine_profile`.
Never reuse one profile's recipe or results for the other.

## What LewLM does with this profile

- **Preflight, not autodetection.** `scripts/engine_preflight.py --recipe
  vllm` checks the digest pin, container runtime, driver ≥ R580 (the image is
  CUDA 13.0), the GPU's SM against the image's compiled list, free VRAM, and a
  free loopback port — before anything is pulled. LewLM itself never runs it.
- **Batching stays upstream.** The bridge reports
  `runtime_adapter.kind: backend_native_batch`: vLLM schedules its own
  batches, so LewLM opens no microbatch window and does not serialize
  requests; it keeps the bounded admission cap
  (`LEWLM_MAX_CONCURRENT_RUNTIME_REQUESTS`) and per-request cancellation, and
  two concurrent requests reach the engine concurrently. Prefix caching is a
  server setting (on by default at the pin); LewLM's own response cache never
  applies to chat.
- **Structured output.** `json_schema` is forwarded natively
  (`enforcement_evidence: upstream_native`) and validated after generation;
  LewLM does not claim vLLM's decoder enforced it.
- **Tools.** The recipe starts vLLM with `--enable-auto-tool-choice
  --tool-call-parser hermes` (what upstream prescribes for Qwen2.5). Without
  those flags vLLM rejects `tool_choice: auto` with a 400, which LewLM
  surfaces as a structured `invalid_request` error and never retries. LewLM
  still gates tools per model by probing.
- **Keys.** The endpoint's `api_key_env` names the key the container reads
  from `VLLM_API_KEY`. vLLM guards `/v1` with it; `/health` stays open for the
  container healthcheck.
- **Caches.** Model weights (`HF_HOME`) and compiled artifacts
  (`VLLM_CACHE_ROOT`) are separate volumes; a warm restart reuses the compile
  cache, and the cold-versus-warm timing is a recorded measurement, not a
  claim.
- **Failure.** An unreachable vLLM produces a 503 naming `endpoint_id: vllm`;
  `lewlm scan` keeps its models as stale; native llama.cpp and Ollama routes
  are unaffected.

## Proof, when hardware is available

Run the recipe's section 2: `scripts/backend_acceptance.py` (the same common
suite oMLX passed), `scripts/bridge_prefix_benchmark.py` at concurrency 1
and 2, the upstream-overlap observation from vLLM's own stats log, cold start
versus warm restart with the compile cache mounted, `nvidia-smi` memory, and
the stop/rescan/coexistence check. WSL2 and ROCm are separate lanes.
