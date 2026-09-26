# ExLlamaV3 via TabbyAPI (Linux + NVIDIA)

[TabbyAPI](https://github.com/theroyallab/tabbyAPI) is ExLlamaV3's official
OpenAI-compatible server. LewLM fronts it through the `exllamav3_tabby`
bridge profile: discovery, routing by model id, sampling, `json_schema`
output, tools, and cancellation all cross the loopback boundary the same way
they do for every other endpoint. ExLlamaV3 itself is never imported into
LewLM, and LewLM never loads or converts a model on the server's behalf.

The pinned recipe — digest-pinned image, `config.yml` with keys verified at
the pinned commit, key split, compose file, LewLM environment, and the proof
commands — is in
[`examples/backends/exllamav3-tabby/`](https://github.com/lew-cx/LewLM/tree/main/examples/backends/exllamav3-tabby).

## Status

**Passed on Windows + WSL2; deferred on bare-metal Linux.** On 2026-09-24 the
recipe ran in Docker Desktop on an RTX 5090 Laptop (SM 12.0), with LewLM native
on Windows. It used an EXL3 4.0 bpw artifact converted with the image's own
ExLlamaV3 1.5.0. The common acceptance suite passed 10 checks; tools were
inconclusive because the 0.5B model answered in text. TabbyAPI has no `seed` at
this pin, so LewLM reports it as unsupported. LewLM forwards `top_k`, `min_p`,
and `repetition_penalty`, which TabbyAPI does implement. See the
[Windows/Linux validation record](../../validation/modernization-windows-linux.md).
`examples/backends/compatibility.json` keeps `exllamav3_tabby` as `deferred`
until the suite passes on a bare-metal Linux/NVIDIA host. Do not read this
profile's presence in `LEWLM_EXTERNAL_ENDPOINTS` as support elsewhere.

## What LewLM does with this profile

- **Format.** A model advertised by TabbyAPI is registered with
  `format_type: unknown` unless its `/v1/models` record names a format;
  `exl3` is assigned only from evidence. Either way the model is
  endpoint-bound: llama.cpp is never a candidate for it, and conversion
  refuses it as URI-backed.
- **Keys.** The endpoint's `api_key_env` names the *inference* key. TabbyAPI's
  `admin_key` (model load/unload) is not configured anywhere in LewLM.
- **Structured output.** `json_schema` is forwarded natively
  (`enforcement_evidence: upstream_native`) and validated after generation;
  LewLM does not claim TabbyAPI enforced it for the loaded model until the
  acceptance run shows it.
- **Performance features.** Reported as backend-native or partial with
  `active: false` until observed: batching, paged KV cache, `cache_mode`
  quantization, and chunked prefill are TabbyAPI settings, not LewLM
  controls.
- **Failure.** An unreachable TabbyAPI produces a 503 naming
  `endpoint_id: tabby`; `lewlm scan` keeps its models as stale; native
  llama.cpp and Ollama routes are unaffected.

## Proof, when hardware is available

Run the recipe's section 4: `scripts/backend_acceptance.py` (the same common
suite oMLX passed), `scripts/bridge_prefix_benchmark.py`, first-start versus
cached-restart timings, `nvidia-smi` memory, and the stop/rescan/coexistence
check. Native Windows is a separate lane with its own proof.
