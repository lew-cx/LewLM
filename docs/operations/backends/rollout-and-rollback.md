# Rolling an engine out — and back

Every accelerator behind LewLM is opt-in, operator-run, and disabled by
removing one configuration entry. Normal setup is four steps: choose a
backend, follow its recipe, configure its endpoint, run `lewlm doctor`.
Nothing here is required to use LewLM with its packaged runtimes or Ollama.

## 1. Choose a backend and follow its recipe

| Backend | Recipe | Platform lane | Status in `examples/backends/compatibility.json` |
| --- | --- | --- | --- |
| oMLX | `examples/backends/omlx/` | Apple Silicon macOS | `validated` (oMLX `b45fb7e`, Qwen2.5-0.5B-Instruct-4bit, the step-05 host) |
| vLLM | `examples/backends/vllm/` | Linux + NVIDIA | `deferred` until its lane passes |
| SGLang | `examples/backends/sglang/` | Linux + NVIDIA | `deferred` until its lane passes |
| ExLlamaV3 via TabbyAPI | `examples/backends/exllamav3-tabby/` | Linux + NVIDIA | `deferred` until its lane passes |

Each recipe pins an exact release (commit, image digest or dependency lock,
model revision and hashes), publishes its port to host loopback only, and
tells you what to measure before promoting it. A `deferred` recipe is
complete and pinned but has not been executed on its hardware; the
`validated` one names the exact engine + model + host it passed on, and
nothing broader. The engine runs where the recipe says; LewLM never installs,
starts, updates, or stops it. Run `python scripts/engine_preflight.py --recipe
<name>` before pulling a CUDA image.

## 2. Configure the endpoint

```bash
export VLLM_API_KEY=...        # the engine's inference key; the endpoint names the variable, never the value
export LEWLM_EXTERNAL_ENDPOINTS='[{"endpoint_id":"vllm","profile":"vllm_local","base_url":"http://127.0.0.1:8000/v1","api_key_env":"VLLM_API_KEY","read_timeout_seconds":120}]'
lewlm doctor
lewlm scan && lewlm list-models
lewlm serve
```

Several endpoints coexist in the same array (an oMLX server and an Ollama
daemon, say). The endpoint id becomes part of every model id and source URI
(`external://<endpoint_id>/<upstream id>`), and of residency, cache, and
benchmark identity, so two servers advertising the same model name are never
merged.

## 3. Read `lewlm doctor`

Doctor reports, per configured engine: whether it is enabled, reachable
(doctor reads its model list once — an explicit action, not something health
or chat ever do), how many models it advertises and how many LewLM has
registered, the recipe and its manifest status, and the one command to run
next. Typical lines:

```text
engine vllm (vllm_local): unreachable, 0 advertised, 1 registered, recipe deferred
  error: The configured external accelerator endpoint `http://127.0.0.1:8000/v1/models` refused the connection.
  next: start the engine: see examples/backends/vllm/README.md (`docker compose up -d` or the run script), then `lewlm scan`
next step: endpoint `vllm`: start the engine: ...
```

`ready` means "reachable and every advertised model is registered — `lewlm
serve`". `advertised_not_registered` means `lewlm scan`. `blocked` means the
configuration itself (usually a missing credential variable) stops LewLM
before any request. `disabled` means you rolled it back.

## 4. Migrating from the single-server settings

The pre-modernization form still works and is resolved to one endpoint named
`legacy-default` with the old runtime name, so nothing changes until you
migrate:

```bash
# old (still valid)
LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true
LEWLM_EXTERNAL_ACCELERATOR_BASE_URL=http://127.0.0.1:8000
LEWLM_EXTERNAL_ACCELERATOR_PROFILE=vllm_local

# new (choose your own id; `legacy-default` is reserved for the synthesized entry)
LEWLM_EXTERNAL_ACCELERATOR_ENABLED=false
LEWLM_EXTERNAL_ENDPOINTS='[{"endpoint_id":"gpu","profile":"vllm_local","base_url":"http://127.0.0.1:8000/v1","api_key_env":"VLLM_API_KEY"}]'
```

Setting both an enabled legacy endpoint and an explicit array is rejected at
startup with this guidance. Because the endpoint id is part of the LewLM model
id, migrating changes the ids of that server's models (`lewlm scan` registers
the new ones; the same upstream models are advertised). Update clients that
pinned an id; nothing else moves.

## 5. Rolling back

Disable the endpoint and restart:

```bash
# either flip the flag ...
LEWLM_EXTERNAL_ENDPOINTS='[{"endpoint_id":"gpu","profile":"vllm_local","base_url":"http://127.0.0.1:8000/v1","enabled":false}]'
# ... or delete the entry / unset the variable
```

What happens, all of it verified by `tests/integration/test_rollout_rollback.py`:

- Routing candidates return to exactly what they were before the endpoint
  existed: packaged runtimes, Ollama, other endpoints. Their models, the
  response and block caches, benchmark artifacts, and Ollama's own state are
  untouched.
- The disabled endpoint's advertised models leave the registry at the next
  scan, so requests for their ids return `404 model_not_found`. Point clients
  at the model they should use instead. An explicit fallback alias for a
  removed id no longer applies.
- `lewlm doctor` lists the endpoint as `disabled` with the command to
  re-enable it; `GET /v1/health.engines[]` shows `enabled: false`.
- Nothing is uninstalled. The engine's container, its caches, and its model
  directory are yours; stop or remove them with the recipe's own
  `docker compose down` (add `-v` only if you want the cache volumes gone).

## While the engine is merely down

An engine that stops while its endpoint stays enabled is not a rollback:

- `GET /v1/health` stays `ok` and reports the engine as `failed` or `stale`;
  its models stay registered (an unreachable engine is not evidence they were
  deleted).
- A request for one of its models is `503 runtime_unavailable` naming
  `details.endpoint_id`, before any generation is submitted — unless
  `LEWLM_EXTERNAL_FALLBACK_POLICY=explicit_alias` maps that model id to a
  registered compatible model in `LEWLM_EXTERNAL_FALLBACK_ALIASES`, in which
  case the alias serves it and `metadata.routing.fallback_from_model_id`
  says so. LewLM never substitutes a model on its own.
- A stream that loses its engine mid-way ends with a terminal chunk
  (`finish_reason: "error"`, an `error` envelope with `partial_output: true`)
  and `[DONE]`; the delivered text stands and nothing is replayed. The next
  request routes normally (to the alias, if configured).
- When the engine returns, `lewlm scan` (or the automatic retry a few seconds
  later) puts it back in service.

## Rolling back the modernization itself

Every step is one commit and additive: new settings default to the old
behaviour, new response fields default to `null`/absent, stored profiles are
not migrated, old model ids and routes are unchanged. Reverting a step
commit restores the previous state; the only user-visible changes a client
can notice are listed in each step's validation record under "Rollback".
