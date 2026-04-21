# Configuration

LewLM configuration is environment-driven through `LewLMSettings`, with the prefix `LEWLM_`.

## Core paths

The most important path settings are:

- `LEWLM_DATA_DIR`
- `LEWLM_MODELS_DIR`
- `LEWLM_FILE_ACCESS_ROOTS`
- `LEWLM_VALIDATION_MANIFEST_PATHS`

By default, LewLM normalizes `data_dir` to `~/.lewlm` and derives `models_dir` as `~/.lewlm/models`.

## Common environment examples

### Relocate all state

```bash
export LEWLM_DATA_DIR=/srv/lewlm
```

### Scan custom model roots

```bash
export LEWLM_MODELS_DIR='["/models/mlx","/models/gguf"]'
```

### Require API keys for the local server

```bash
export LEWLM_API_KEY_REQUIRED=true
export LEWLM_API_KEYS='["dev-key-1"]'
```

### Tighten file access

```bash
export LEWLM_FILE_ACCESS_ROOTS='["/srv/lewlm/templates","/srv/lewlm/uploads"]'
```

### Enable audit logging and encrypted persistence

```bash
export LEWLM_AUDIT_LOG_ENABLED=true
export LEWLM_PERSISTENCE_ENCRYPTION_ENABLED=true
export LEWLM_PERSISTENCE_ENCRYPTION_PASSPHRASE='change-me'
```

## Tuning themes

| Theme | Main settings |
| --- | --- |
| Request guards | `API_KEY_REQUIRED`, `REQUEST_MAX_BYTES`, `RATE_LIMIT_*` |
| Concurrency and scheduling | `MAX_CONCURRENT_RUNTIME_REQUESTS`, `RUNTIME_REQUEST_QUEUE_*`, `DECODE_PRIORITY_*`, `PREFILL_*` |
| Runtime residency | `RUNTIME_POLICY` |
| MLX serving | `MLX_GRAPH_COMPILE_ENABLED`, `MLX_ATTENTION_KERNEL_MODE`, `KV_CACHE_*` |
| Tool/parser/conversion isolation | `*_SANDBOX_ENABLED`, `*_SANDBOX_TIMEOUT_SECONDS`, `*_SANDBOX_CLEAR_ENVIRONMENT` |
| Cluster | `CLUSTER_ROLE`, `CLUSTER_*` |

## Request-time overrides vs global configuration

Some behaviors are controlled per request instead of globally:

- `apply_serving_profile`
- `reasoning_visibility`
- prompt overrides such as `system_prompt`, `developer_prompt`, `tools`, `mcp_tools`, and output schema settings
- `authorized_actions`
- `idempotency_key`

## What to read next

- [Configuration reference](../reference/configuration.md) for the full setting catalog
- [Security](../security.md) for request guards, file scoping, and persistence rules
