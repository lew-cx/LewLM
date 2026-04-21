# Storage and state

By default LewLM stores state under:

```text
~/.lewlm
```

## Default directory layout

| Path | Purpose |
| --- | --- |
| `models/` | default model scan root |
| `logs/` | audit log and related logs |
| `cache/` | conversion cache, response cache, block cache, materialized cache data |
| `benchmarks/` | benchmark artifacts |
| `tmp/` | request workspaces and sandbox roots |
| `keys/` | persistence salt and key material |
| `metadata.sqlite3` | primary metadata store |

## Temporary workspaces

LewLM creates isolated work areas under `tmp/`, including:

- `parser-sandbox`
- `tool-sandbox`
- `conversion-sandbox`
- `materialized-cache`

Multipart uploads are copied into secure per-request workspaces and cleaned up when the request completes or fails.

## Persisted state examples

LewLM persists data such as:

- model manifests and registry metadata
- conversion jobs and artifacts
- serving-profile recommendations
- runtime metrics and benchmark metadata
- session history
- cached deterministic responses

## Security-sensitive state

When encrypted persistence is enabled, LewLM encrypts selected structured metadata and cache artifacts. See [Security](../security.md) for details.

## State isolation

If you want multiple applications or environments to use separate LewLM state, set a different `LEWLM_DATA_DIR` per environment.
