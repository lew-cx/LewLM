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

**Deferred.** No Linux/NVIDIA host was available when the recipe was written
(2026-09-17); every pin was verified against upstream, nothing was executed.
`examples/backends/compatibility.json` keeps `exllamav3_tabby` as `deferred`
until the acceptance suite passes on real hardware. Do not read this profile's
presence in `LEWLM_EXTERNAL_ENDPOINTS` as support.

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
