# Configuration reference

All settings live on `LewLMSettings` and use the `LEWLM_` prefix.

## Core service settings

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LEWLM_ENVIRONMENT` | `development` | environment label |
| `LEWLM_HOST` | `127.0.0.1` | API bind host |
| `LEWLM_PORT` | `8080` | API bind port |
| `LEWLM_LOG_LEVEL` | `INFO` | runtime log level |
| `LEWLM_DATA_DIR` | `~/.lewlm` | state root |
| `LEWLM_MODELS_DIR` | derived from `data_dir/models` | scan roots |
| `LEWLM_PRIVACY_MODE` | `false` | privacy-oriented behaviors |
| `LEWLM_TELEMETRY_ENABLED` | `false` | telemetry toggle |
| `LEWLM_BACKEND_FEATURE_PROBES_ENABLED` | `false` | opt-in import-based installed-backend inventory probes for diagnostic hosts; leave disabled in constrained or model-serving processes |
| `LEWLM_ALLOW_OUTBOUND_NETWORK` | `false` | outbound network policy |

## Pack selection

These settings let an operator load only the built-in runtime and feature packs they actually want. When unset, LewLM keeps the default built-in pack set and only disables packs explicitly listed in the denylist fields.

Use JSON arrays for the environment variables, for example:

```bash
export LEWLM_DISABLED_RUNTIME_PACKS='["llamacpp"]'
export LEWLM_DISABLED_FEATURE_PACKS='["documents"]'
```

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LEWLM_RUNTIME_PACKS` | empty | optional runtime-pack allowlist |
| `LEWLM_DISABLED_RUNTIME_PACKS` | empty | runtime-pack denylist |
| `LEWLM_FEATURE_PACKS` | empty | optional feature-pack allowlist |
| `LEWLM_DISABLED_FEATURE_PACKS` | empty | feature-pack denylist |

Built-in runtime pack names: `mlx`, `llamacpp`, `external_accelerator`, `experimental`, `distributed_experimental`.

Built-in feature pack names: `documents`.

## Request guards

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LEWLM_API_KEY_REQUIRED` | `false` | require API keys for guarded routes |
| `LEWLM_API_KEYS` | empty | accepted API keys |
| `LEWLM_LIFECYCLE_OPERATOR_API_KEYS` | empty | scoped keys for residency inspection, warm, and drain |
| `LEWLM_LIFECYCLE_ADMINISTRATOR_API_KEYS` | empty | scoped keys for unload and disruptive diagnostics, including all operator permissions |
| `LEWLM_REQUEST_MAX_BYTES` | `52428800` | request size limit |
| `LEWLM_RATE_LIMIT_REQUESTS` | `120` | requests per window |
| `LEWLM_RATE_LIMIT_WINDOW_SECONDS` | `60` | rate-limit window |

## Scheduling and concurrency

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LEWLM_MAX_CONCURRENT_RUNTIME_REQUESTS` | `4` | runtime request concurrency |
| `LEWLM_MAX_CONCURRENT_MODEL_LOADS` | `1` | concurrent warm/load control |
| `LEWLM_MAX_APPLICATION_METRIC_ENTRIES` | `64` | bounded named application metric entries before overflow aggregation |
| `LEWLM_RUNTIME_REQUEST_QUEUE_LIMIT` | `16` | queue depth limit |
| `LEWLM_RUNTIME_REQUEST_QUEUE_TIMEOUT_SECONDS` | `15` | queue wait timeout |
| `LEWLM_REQUEST_CANCELLATION_INTENT_TTL_SECONDS` | `300` | how long a cancellation for a request that has not arrived yet is honoured |
| `LEWLM_REQUEST_CANCELLATION_MAX_TRACKED_REQUESTS` | `512` | bound on remembered request handles |
| `LEWLM_CONTINUOUS_BATCH_WINDOW_MILLISECONDS` | `8` | native batch join window |
| `LEWLM_CONTINUOUS_BATCH_MAX_BATCH_SIZE` | `4` | native batch size cap |
| `LEWLM_DECODE_PRIORITY_SCHEDULING_ENABLED` | `true` | decode-priority scheduling |
| `LEWLM_LONG_PREFILL_TOKEN_THRESHOLD` | `1024` | long-prefill cutoff |
| `LEWLM_PREFILL_ISOLATION_ENABLED` | `false` | separate prefill lane |
| `LEWLM_PREFILL_ISOLATION_MAX_CONCURRENT_REQUESTS` | `1` | prefill lane capacity |
| `LEWLM_PREFILL_ISOLATION_DECODE_RESERVE` | `1` | reserved decode slots |
| `LEWLM_PREFILL_TOKEN_BATCH_SIZE` | `512` | chunked prefill sizing |

## Serving and residency

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LEWLM_RUNTIME_POLICY` | `balanced` | keep-warm vs unload policy |
| `LEWLM_LLAMACPP_MAX_CONTEXT_TOKENS` | `16384` | largest context window llama.cpp is asked to reserve, whatever a model advertises; llama.cpp allocates its KV cache for the whole of `n_ctx` at load, so a 131k-token model would cost gigabytes before the first prompt. Unset to serve each model's full advertised window. Routing admits only what this leaves servable |
| `LEWLM_UNKNOWN_CONTEXT_TOKEN_LIMIT` | `4096` | largest estimated request routed to a model whose context length LewLM never recorded; the estimate is the prompt plus `max_tokens`, and the bound is reported as `unknown_context_token_limit` on the routing error |
| `LEWLM_MODEL_DRAIN_TIMEOUT_SECONDS` | `30` | synchronous and default asynchronous drain timeout |
| `LEWLM_KV_CACHE_PAGE_SIZE` | `256` | paged KV sizing |
| `LEWLM_KV_CACHE_MAX_PAGES` | `64` | maximum KV pages |
| `LEWLM_KV_CACHE_QUANTIZATION_BITS` | unset | KV quantization (`16`/`8`/`4`); a quantized cache also enables llama.cpp flash attention, and is refused when the installed build cannot accept it |
| `LEWLM_GPU_OFFLOAD_LAYERS` | unset | GPU layer offload for GGUF models (`-1` offloads all layers); applied only when the installed llama.cpp build reports GPU offload support |
| `LEWLM_MLX_GRAPH_COMPILE_ENABLED` | `false` | MLX graph compile toggle |
| `LEWLM_MLX_ATTENTION_KERNEL_MODE` | `stock` | MLX attention kernel mode |
| `LEWLM_REASONING_VISIBILITY` | `hidden` | default reasoning surface |

## Speculation and advanced runtime flags

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LEWLM_SPECULATIVE_DECODING_ENABLED` | `false` | enable speculation |
| `LEWLM_SPECULATIVE_DECODING_DRAFT_MODEL_ID` | unset | explicit draft model |
| `LEWLM_SPECULATIVE_DECODING_NUM_DRAFT_TOKENS` | `3` | draft token count |
| `LEWLM_PROMPT_LOOKUP_SPECULATION_ENABLED` | `false` | prompt lookup speculation |
| `LEWLM_PROMPT_LOOKUP_MAX_NGRAM_SIZE` | `2` | lookup n-gram size |
| `LEWLM_PROMPT_LOOKUP_NUM_PRED_TOKENS` | `10` | predicted token count |
| `LEWLM_MOE_BOUNDED_MEMORY_MODE` | `off` | MoE bounded-memory mode |
| `LEWLM_MOE_RESIDENT_EXPERT_COUNT` | `4` | resident experts target |

## External accelerators

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LEWLM_EXTERNAL_ACCELERATOR_ENABLED` | `false` | enable local adapter runtime |
| `LEWLM_EXTERNAL_ACCELERATOR_PROFILE` | `openai_compatible` | adapter profile |
| `LEWLM_EXTERNAL_ACCELERATOR_BASE_URL` | unset | adapter endpoint |
| `LEWLM_EXTERNAL_ACCELERATOR_TIMEOUT_SECONDS` | `10` | adapter timeout |
| `LEWLM_EXTERNAL_ENDPOINTS` | unset | JSON array of named local adapter endpoints |

The four singular `LEWLM_EXTERNAL_ACCELERATOR_*` settings remain the compatible
single-endpoint form. LewLM resolves them as the internal endpoint ID
`legacy-default`. For multiple engines, use the named collection instead:

```text
LEWLM_EXTERNAL_ENDPOINTS=[{"endpoint_id":"mlx","profile":"omlx","base_url":"http://127.0.0.1:8000/v1","read_timeout_seconds":30},{"endpoint_id":"gpu","profile":"vllm_local","base_url":"http://127.0.0.1:8001","api_key_env":"VLLM_API_KEY"}]
```

Each entry accepts `endpoint_id`, `profile`, `enabled`, `base_url`, optional
`api_key_env`, and positive `connect_timeout_seconds`, `read_timeout_seconds`,
and `pool_timeout_seconds`. IDs must be unique. URLs may name the server root or
end in `/v1`; LewLM normalizes those spellings to one endpoint identity. URLs
cannot contain credentials, query parameters, fragments, or non-loopback hosts.
The credential environment-variable name is omitted from redacted configuration
output. Transport authentication is implemented in the shared-transport step;
until then, leave `api_key_env` unset.

Do not combine an explicit collection with
`LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true`. To migrate, move the old profile and
URL into one named entry, then remove or disable the singular enable flag. An
explicit empty array means no named endpoints; it does not restore the legacy
configuration.

Supported `LEWLM_EXTERNAL_ACCELERATOR_PROFILE` values are:
`openai_compatible`, `vmlx`, `omlx`, `vllm_mlx`, `vllm_local`, `sglang_local`, `tensorrt_llm_server`, `openvino_model_server`, `ollama_local`, and `llamacpp_server`.

`tensorrt_llm_server` and `openvino_model_server` are bridge profiles for compatible local servers; `ollama_local` and `llamacpp_server` keep the generic OpenAI-compatible bridge contract explicit for local servers that present themselves through those loopback shapes. None of these profiles promote backend-native behavior to LewLM-owned packaged parity.

`LEWLM_EXTERNAL_ACCELERATOR_BASE_URL` must point to a loopback-only local server such as
`http://127.0.0.1:8000`. LewLM rejects a base URL whose host is not `127.0.0.1`, `localhost`, or `::1`.

That check validates the **first hop only**. It cannot tell whether the loopback server executes the
request locally or relays it somewhere else, and several common setups do relay: an Ollama daemon
serving a cloud-hosted model, an SSH tunnel bound to a loopback port, or a gateway process listening
on loopback. Where the request is ultimately executed is a property of the server you configure, not
of the URL, so treat this setting as "LewLM will not dial a remote host itself" rather than as a
guarantee that prompts stay on the machine.

Use this path when LewLM should front a loopback-only OpenAI-compatible local server instead of importing a runtime package directly.

## Ollama model discovery

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LEWLM_OLLAMA_DISCOVERY_ENABLED` | `false` | publish a local Ollama daemon's models as LewLM manifests |
| `LEWLM_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint to read |
| `LEWLM_OLLAMA_DISCOVERY_TIMEOUT_SECONDS` | `5` | seconds to wait for the daemon |
| `LEWLM_OLLAMA_CLOUD_ENABLED` | `false` | include Ollama models that execute off-host |

Ollama is a **model source**, not a LewLM-owned runtime. LewLM does not install, launch, configure,
update, or supervise it, and no packaged Ollama runtime exists. Obtaining Ollama is entirely yours —
the desktop client or the CLI, whichever you prefer. When discovery is enabled, `lewlm scan` asks the
daemon what it holds and registers each model as a manifest whose `source_path` is `ollama://<tag>`;
the **external accelerator bridge** executes those requests, which is why
With singular settings, `LEWLM_OLLAMA_DISCOVERY_ENABLED` requires
`LEWLM_EXTERNAL_ACCELERATOR_ENABLED`. With named endpoints, it requires exactly
one enabled `ollama_local` entry whose normalized URL matches
`LEWLM_OLLAMA_BASE_URL`. This binds every discovered Ollama manifest to the
correct endpoint while preserving its existing public model ID.

While `LEWLM_OLLAMA_DISCOVERY_ENABLED` is false — the default — LewLM never contacts the daemon.

Scan behavior is deliberately asymmetric so that a component LewLM does not manage cannot cost you
your registry:

- turning the flag **off** retires the whole `ollama://` namespace on the next scan
- a daemon that is **unreachable** retires nothing, and the scan reports the failure in its `notes`
- a filesystem scan never touches `ollama://` manifests, and vice versa

### Ollama Cloud

An Ollama daemon can serve models it runs on this host and models it relays to Ollama's cloud. Both
arrive over the same loopback endpoint, so LewLM classifies each model at discovery time from the
native `/api/tags` record — a `remote_host`/`remote_model` field, or the `-cloud` naming convention.

Cloud-backed models are **left out of the registry** unless `LEWLM_OLLAMA_CLOUD_ENABLED=true`, and a
scan reports how many it skipped. Enabling it means prompts for those models leave the machine.
LewLM holds no cloud credentials and never contacts `ollama.com`: signing in is `ollama signin`, and
the daemon owns the account relationship entirely.

Whether a given cloud model is reachable at all depends on the daemon advertising it, which is
Ollama's decision rather than LewLM's.

On Linux and Windows, including NVIDIA-backed local servers, this is the intended bridge path when you already run a compatible local endpoint. LewLM does not bundle that server, and this path remains bridge-only even when benchmarks are favorable, so keep the bridge/runtime distinction explicit in operator docs and deployments.

## File access, sandboxing, and persistence

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LEWLM_FILE_ACCESS_ROOTS` | derived from `data_dir` | allowed API file roots |
| `LEWLM_VALIDATION_MANIFEST_PATHS` | empty | external validation manifests |
| `LEWLM_TOOL_AUTHORIZATION_REQUIRED` | `false` | require explicit action grants |
| `LEWLM_PARSER_SANDBOX_ENABLED` | `true` | isolate document parsers |
| `LEWLM_PARSER_SANDBOX_TIMEOUT_SECONDS` | `30` | parser timeout |
| `LEWLM_PARSER_SANDBOX_CLEAR_ENVIRONMENT` | `true` | clear parser env |
| `LEWLM_TOOL_SANDBOX_ENABLED` | `true` | isolate local tools |
| `LEWLM_TOOL_SANDBOX_TIMEOUT_SECONDS` | `120` | tool timeout |
| `LEWLM_TOOL_SANDBOX_CLEAR_ENVIRONMENT` | `true` | clear tool env |
| `LEWLM_CONVERSION_SANDBOX_ENABLED` | `true` | isolate conversions |
| `LEWLM_CONVERSION_SANDBOX_TIMEOUT_SECONDS` | `1800` | conversion timeout |
| `LEWLM_CONVERSION_SANDBOX_CLEAR_ENVIRONMENT` | `true` | clear conversion env |
| `LEWLM_CONVERSION_WORKER_COUNT` | `1` | conversion worker pool size |
| `LEWLM_AUDIT_LOG_ENABLED` | `false` | JSONL audit log |
| `LEWLM_PERSISTENCE_ENCRYPTION_ENABLED` | `false` | enable encrypted persistence |
| `LEWLM_PERSISTENCE_ENCRYPTION_PASSPHRASE` | unset | encryption passphrase |
| `LEWLM_PERSISTENCE_ENCRYPTION_KDF_ITERATIONS` | `600000` | KDF cost |

## Browser access (CORS)

LewLM serves no CORS headers by default: it is local-first, and a permissive default would let any page an operator visits reach a loopback model server.

| Setting | Default | Purpose |
| --- | --- | --- |
| `LEWLM_CORS_ENABLED` | `false` | Turn CORS on. Requires `cors_allow_origins`. |
| `LEWLM_CORS_ALLOW_ORIGINS` | `()` | Explicit origin allowlist. |
| `LEWLM_CORS_ALLOW_CREDENTIALS` | `false` | Cannot be combined with a `*` origin; startup refuses that pairing. |
| `LEWLM_CORS_ALLOW_METHODS` | `GET, POST, PATCH, DELETE, OPTIONS` | Permitted methods. |
| `LEWLM_CORS_ALLOW_HEADERS` | LewLM request headers | Includes `x-api-key`, `x-lewlm-*`, and `x-request-id`. |
| `LEWLM_CORS_EXPOSE_HEADERS` | `x-request-id, x-lewlm-correlation-id` | Headers a browser caller can read. |
| `LEWLM_CORS_MAX_AGE_SECONDS` | `600` | Preflight cache duration. |

Two misconfigurations are refused at startup rather than at request time: enabling CORS with no origins, and combining credentials with a wildcard origin.

## Cluster

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LEWLM_CLUSTER_ROLE` | `standalone` | node role |
| `LEWLM_CLUSTER_NAME` | `default` | cluster namespace |
| `LEWLM_CLUSTER_NODE_NAME` | `node` | local node name |
| `LEWLM_CLUSTER_PUBLIC_BASE_URL` | unset | coordinator-visible URL |
| `LEWLM_CLUSTER_COORDINATOR_URL` | unset | worker coordinator URL |
| `LEWLM_CLUSTER_ENROLLMENT_SECRET` | unset | enrollment secret |
| `LEWLM_CLUSTER_TOKEN_TTL_SECONDS` | `900` | token lifetime |
| `LEWLM_CLUSTER_WORKER_HEARTBEAT_TIMEOUT_SECONDS` | `30` | worker expiry |
| `LEWLM_CLUSTER_STAGE_TIMEOUT_SECONDS` | `15` | pipeline-stage timeout |
