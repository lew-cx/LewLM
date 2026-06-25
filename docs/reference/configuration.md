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
| `LEWLM_REQUEST_MAX_BYTES` | `52428800` | request size limit |
| `LEWLM_RATE_LIMIT_REQUESTS` | `120` | requests per window |
| `LEWLM_RATE_LIMIT_WINDOW_SECONDS` | `60` | rate-limit window |

## Scheduling and concurrency

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LEWLM_MAX_CONCURRENT_RUNTIME_REQUESTS` | `4` | runtime request concurrency |
| `LEWLM_MAX_CONCURRENT_MODEL_LOADS` | `1` | concurrent warm/load control |
| `LEWLM_RUNTIME_REQUEST_QUEUE_LIMIT` | `16` | queue depth limit |
| `LEWLM_RUNTIME_REQUEST_QUEUE_TIMEOUT_SECONDS` | `15` | queue wait timeout |
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
| `LEWLM_KV_CACHE_PAGE_SIZE` | `256` | paged KV sizing |
| `LEWLM_KV_CACHE_MAX_PAGES` | `64` | maximum KV pages |
| `LEWLM_KV_CACHE_QUANTIZATION_BITS` | `8` | KV quantization |
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

Supported `LEWLM_EXTERNAL_ACCELERATOR_PROFILE` values are:
`openai_compatible`, `vmlx`, `omlx`, `vllm_mlx`, `vllm_local`, `sglang_local`, `tensorrt_llm_server`, `openvino_model_server`, `ollama_local`, and `llamacpp_server`.

`tensorrt_llm_server` and `openvino_model_server` are bridge profiles for compatible local servers; `ollama_local` and `llamacpp_server` keep the generic OpenAI-compatible bridge contract explicit for local servers that present themselves through those loopback shapes. None of these profiles promote backend-native behavior to LewLM-owned packaged parity.

`LEWLM_EXTERNAL_ACCELERATOR_BASE_URL` must point to a loopback-only local server such as
`http://127.0.0.1:8000`; remote/cloud endpoints are intentionally rejected.

Use this path when LewLM should front a loopback-only OpenAI-compatible local server instead of importing a runtime package directly.

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
