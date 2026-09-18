# SGLang (Linux + NVIDIA)

[SGLang](https://github.com/sgl-project/sglang) serves a model behind its own
OpenAI-compatible server with RadixAttention prefix reuse and continuous
batching. LewLM fronts it through the `sglang_local` bridge profile:
discovery, routing by served model name, sampling, `json_schema` output,
tools, and cancellation all cross the loopback boundary the same way they do
for every other endpoint. SGLang is never imported into LewLM, and LewLM never
loads a model on the server's behalf or reconstructs any KV state.

The pinned recipe — digest-pinned image, `docker compose` file whose every
argument was checked against `server_args.py` at the pinned commit, the LewLM
environment, the preflight, and the proof commands — is in
[`examples/backends/sglang/`](https://github.com/lew-cx/LewLM/tree/main/examples/backends/sglang).

## Status

**Deferred.** No Linux/NVIDIA host was available when the recipe was written
(2026-09-18); every pin was verified against upstream, nothing was executed.
`examples/backends/compatibility.json` keeps `sglang_local` as `deferred`
until the acceptance suite passes on real hardware. Do not read this profile's
presence in `LEWLM_EXTERNAL_ENDPOINTS` as support.

## What LewLM does with this profile

- **Preflight, not autodetection.** `scripts/engine_preflight.py --recipe
  sglang` checks the digest pin, container runtime, driver ≥ R580 (the image
  is CUDA 13.0), free VRAM, and a free loopback port before anything is
  pulled. SGLang publishes no compiled-architecture list for its kernel wheel
  at the pin, so that check is reported as skipped, never as passed.
- **Batching and prefix reuse stay upstream.** The bridge reports
  `runtime_adapter.kind: backend_native_batch`: SGLang schedules its own
  batches and radix-cache reuse, so LewLM opens no microbatch window and
  does not serialize requests; it keeps the bounded admission cap
  (`LEWLM_MAX_CONCURRENT_RUNTIME_REQUESTS`) and per-request cancellation.
  Prefix-cache hits are reported as `partial` ownership: a repeated-prefix
  speed-up is an observation, and a hit count exists only if the server is
  run with `--enable-cache-report` and its usage detail is captured.
- **Structured output.** `json_schema` is forwarded natively
  (`enforcement_evidence: upstream_native`) and validated after generation;
  the recipe names `--grammar-backend xgrammar` so the evidence says which
  backend was in use. LewLM does not claim the decoder enforced it.
- **Tools and reasoning.** The recipe sets `--tool-call-parser qwen25`
  (upstream's parser for Qwen2.5). `--reasoning-parser` is not set for this
  model; LewLM's reasoning-visibility policy still applies to whatever the
  server returns. Both are capability-gated per model by probing.
- **Keys.** The endpoint's `api_key_env` names the key the container was
  started with (`--api-key`). SGLang's middleware guards every route with it
  except `/health*` and `/metrics*`; the admin key for management endpoints
  is never configured.
- **Failure.** An unreachable SGLang produces a 503 naming
  `endpoint_id: sglang`; `lewlm scan` keeps its models as stale; native
  llama.cpp and Ollama routes are unaffected.

## Proof, when hardware is available

Run the recipe's section 2: `scripts/backend_acceptance.py` (the same common
suite oMLX passed), `scripts/bridge_prefix_benchmark.py` at concurrency 1
and 2 with LewLM response caching untouched, the upstream-overlap
observation from SGLang's `Decode batch. #running-req` log line, cold start
versus warm restart with both cache volumes mounted, the
`available_gpu_mem` line and `nvidia-smi` memory, and the
stop/rescan/coexistence check. CUDA 12 images, ROCm, Ascend, and multi-GPU
are separate lanes.
