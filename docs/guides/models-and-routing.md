# Models and routing

LewLM separates **model discovery** from **runtime selection**.

## Discovery and registry

Use the registry flow to discover local bundles and store normalized manifests:

```bash
lewlm scan
lewlm list-models
lewlm capabilities <model-id>
```

Each manifest records:

- format (`gguf`, `mlx`, `huggingface`, `audio_folder`, `adapter_bundle`)
- modality (`text`, `vision`, `audio`, `embedding`, `rerank`, `multimodal`)
- runtime affinities
- tokenizer and processor paths when known
- quantization metadata
- estimated memory and context length when known
- conversion status (`runnable`, `requires_conversion`, `not_supported`, `unknown`)

## Routing behavior

LewLM routes by capability and request shape:

- chat / responses
- embeddings
- rerank
- audio transcription
- audio speech

For chat-like requests it also classifies the request modality, such as:

- text only
- text-only multimodal bundle
- single image
- repeated image
- frame-bundle/video
- audio-conditioned

## Capability reports

`lewlm capabilities <model-id>` and `GET /v1/models/{model_id}/capabilities` expose:

- which capabilities are supported today
- a machine-readable `readiness_state` for each capability and runtime candidate
- which runtime LewLM prefers
- why a capability is blocked or downgraded
- which capability claims now have measured host evidence vs still remain unmeasured
- estimated memory notes
- target-platform guidance and fallbacks

## Serving profiles

Serving profiles are persisted host/model/runtime/workload recommendations. They tune settings such as:

- runtime policy
- native batch window and max batch size
- KV cache page sizing and quantization
- prefill token batch size
- MLX graph compilation
- MLX attention kernel mode

Requests can opt out per call with `apply_serving_profile=false`.

## Conversion-aware routing

If a discovered model is not runnable yet, LewLM can:

- mark the bundle as `requires_conversion`
- queue a conversion job
- expose fallback guidance where a different runtime or target path is possible

## Operator workflow

1. Scan roots.
2. Inspect `conversion_status` and capability reports.
3. Convert incompatible source bundles when needed.
4. Warm a target model.
5. Benchmark or autotune if multiple profiles are viable.

See [Runtime and capability matrix](../reference/runtime-capability-matrix.md) for the current backend table.

## Fronting an Ollama install you already run

LewLM can publish the models a local Ollama daemon already holds, so they appear in
`lewlm list-models` alongside your local artifacts and are selected the same way — by model id.

Ollama is a **model source here, not a LewLM runtime**. LewLM does not install, launch, configure,
update, or supervise it; getting Ollama is yours to do, via the desktop client or the CLI. Discovery
reads the daemon's inventory, and the existing external accelerator bridge executes the requests.
No packaged Ollama runtime exists, and none is planned.

```bash
# You install and run Ollama yourself.
ollama serve        # or just run the desktop client

export LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true
export LEWLM_EXTERNAL_ACCELERATOR_PROFILE=ollama_local
export LEWLM_EXTERNAL_ACCELERATOR_BASE_URL=http://127.0.0.1:11434
export LEWLM_EXTERNAL_ACCELERATOR_TIMEOUT_SECONDS=90
export LEWLM_OLLAMA_DISCOVERY_ENABLED=true

lewlm scan && lewlm list-models
```

`lewlm scan` now reads `GET /api/tags` and registers one manifest per model, carrying the real
context window, quantization, parameter size, and the capability-derived modality — so an embedding
model arrives as an embedding model rather than as text. Each manifest declares only
`external_accelerator` affinity, so routing sends it to the bridge with no ambiguity even on a host
where MLX or llama.cpp is installed.

After that, swapping between a local artifact and an Ollama-resident one is just changing the model
id in the request. There is no separate switch, and nothing to restart.

Pull a new model and rescan to pick it up:

```bash
ollama pull qwen3:8b
lewlm scan
```

Turning `LEWLM_OLLAMA_DISCOVERY_ENABLED` off and rescanning retires every `ollama://` model. A daemon
that is merely unreachable retires nothing — the scan says so in its `notes` and leaves the registry
alone, because a component LewLM does not manage being down is not evidence your models are gone.

Ollama models bind to the endpoint that *is* the daemon: with named endpoints, the one `ollama_local`
entry whose URL matches `LEWLM_OLLAMA_BASE_URL`; with the singular settings, `legacy-default` only
when its profile is `ollama_local` and its URL is the daemon's. Otherwise the models are still
registered but stay unroutable and the scan says why — they are never sent to some other accelerator
that happens to be configured.

## Models advertised by named endpoints

Any other enabled entry in `LEWLM_EXTERNAL_ENDPOINTS` (vLLM, SGLang, oMLX, llama.cpp-server, a
generic OpenAI-compatible server) is inventoried the same way: `lewlm scan` reads its `/v1/models`
and registers each advertised model with an endpoint-qualified id and an
`external://<endpoint_id>/<upstream id>` source. Two endpoints serving `shared-name` give you two
models with two ids; pick the one on the engine you mean and routing goes there exactly. The
[configuration reference](../reference/configuration.md#endpoint-model-inventory) describes the
identity, format evidence, staleness, and refresh rules.

```bash
export LEWLM_EXTERNAL_ENDPOINTS='[{"endpoint_id":"gpu","profile":"vllm_local","base_url":"http://127.0.0.1:8001"}]'
lewlm scan && lewlm list-models          # e.g. qwen2-5-0-5b-instruct-gpu-3f9a1c2b
lewlm chat --model qwen2-5-0-5b-instruct-gpu-3f9a1c2b --prompt "Hello"
```

Every response's execution metadata names the endpoint, engine profile, and execution locality
(`host_local`, `off_host`, or `loopback_unverified`) that served it, so a host app can display where a
reply came from without parsing runtime names.

### When an endpoint is down

By default an explicitly requested model on an unreachable endpoint fails with a routing error that
names the endpoint and its inventory state; nothing is substituted, and the other endpoints, native
llama.cpp, and Ollama keep serving their own models. If you want an automatic stand-in, opt in per
model:

```bash
export LEWLM_EXTERNAL_FALLBACK_POLICY=explicit_alias
export LEWLM_EXTERNAL_FALLBACK_ALIASES='{"qwen2-5-0-5b-instruct-gpu-3f9a1c2b":"qwen2.5-0.5b-instruct-q4_k_m"}'
```

The alias must be a registered, runnable model that satisfies the same request; the substitution is
decided before anything is sent upstream and is recorded in the routing decision
(`fallback_from_model_id`, `fallback_reason`). A stream that fails after it has started is reported
as a failure — LewLM does not replay it against another engine.

Testing this without an engine: `tests/unit/test_external_inventory.py` runs two fake loopback
`/v1/models` servers and a stubbed Ollama daemon through the real bootstrap. To try it by hand,
point one entry at any OpenAI-compatible server on loopback, `lewlm scan`, stop the server,
`lewlm scan` again (the models stay, marked stale), then `lewlm chat` against one of them and read
the error's `endpoint_id`.

### Ollama Cloud

An Ollama daemon serves both models it runs here and models it relays to Ollama's cloud, over the
same loopback address. LewLM classifies each model at discovery time and **leaves cloud-backed models
out of the registry** unless you set `LEWLM_OLLAMA_CLOUD_ENABLED=true`; a scan reports what it
skipped. Enabling it means prompts for those models leave the machine, which is why it is a separate,
explicit decision rather than a consequence of enabling discovery.

LewLM holds no cloud credentials and never talks to `ollama.com`. Signing in is `ollama signin`, and
the daemon owns that relationship — exactly as it owns the runtime.

See [Ollama model discovery](../reference/configuration.md#ollama-model-discovery) for the full
settings table and the scan-reconciliation rules.

### The `examples/ollama_bridge_shelf.py` script

The shelf script predates built-in discovery and wrote placeholder files on disk so that a filesystem
scan would find something. Built-in discovery supersedes it: it needs no files, no sync step, and no
`_converted` suffix on model ids. The script remains for reference only.
