# oMLX (Apple Silicon)

[oMLX](https://github.com/jundot/omlx) is an operator-managed MLX inference
server with continuous batching and a tiered (RAM + SSD) KV cache. LewLM
fronts it through the `omlx` bridge profile: LewLM discovers the models it
advertises, routes requests to it by model id, and forwards sampling,
structured output, tools, and cancellation over its OpenAI-compatible API.
LewLM never installs, starts, stops, or updates the server.

The complete pinned recipe — install, model directory, launch script, LewLM
environment, and verification commands — lives in
[`examples/backends/omlx/`](https://github.com/lew-cx/LewLM/tree/main/examples/backends/omlx).
This page records what was validated and how to read the results.

## Validated configuration

| | |
| --- | --- |
| Engine | oMLX `0.7.0.dev3` at commit `b45fb7e`, `mlx 0.32.2`, `mlx-lm 0.31.3`, `transformers 5.17.0` |
| Python / host | 3.11.15 on Apple M2 Max, macOS Darwin 25.2.0 |
| Model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a4` |
| Status in `examples/backends/compatibility.json` | `validated` for this exact configuration |
| Evidence | `docs/validation/evidence/modernization-step-05/` (`acceptance.json`, `prefix-benchmark-c1.json`, `prefix-benchmark-c2.json`, `engine-control.json`) |

Passed: chat, streaming, sampling with observed seeded determinism, native
`json_schema` structured output with post-generation validation, hidden
reasoning, cancellation, two concurrent streams, error mapping, lifecycle,
engine stop/restart, and mid-stream engine loss. **Inconclusive:** tool
calling (the 0.5B model emitted a malformed `<tool_call>` block). **Not
probed:** vision, embeddings, rerank. Nothing outside the table is claimed;
a different model or oMLX commit needs its own run.

## Numbers

Repeated-prefix requests (600-token shared prefix, 32 output tokens, oMLX
cache in its normal configuration, LewLM response cache not applicable to
chat):

| Concurrency | Time to first content via LewLM (p50 / p95) | Direct to oMLX (p50 / p95) | LewLM overhead |
| --- | --- | --- | --- |
| 1 | 141.4 / 142.2 ms | 136.6 / 137.3 ms | +4.8 / +4.9 ms |
| 2 | 267.2 / 267.4 ms | 267.7 / 272.6 ms | −0.5 / −5.2 ms (noise) |

The roadmap's middleware target is `max(20 ms, 10 % of direct p95)`; both
runs are inside it. oMLX's prefix cache shows up as a cold-first-request
difference (164 ms vs ~141 ms warm via LewLM); LewLM reports no hit counter
because the bridge cannot observe one.

## What the evidence means

- **Structured output.** `GET /v1/models/{id}/capabilities` predicts
  `decode_time_modes: []` and the response reports `enforcement: decode_time`,
  `decoder_enforced: false`, `enforcement_evidence: upstream_native`,
  `validation.state: valid`. Read together: the contract went to oMLX
  natively, oMLX's decoder (llguidance) enforced it, and LewLM verified the
  output instead of asserting what it could not see. A client that needs a
  guarantee reads `validation`.
- **Lifecycle.** `warm` sends a one-token probe; `unload` releases LewLM's
  lease and reports `backend_operation_performed: false` with a reason that
  says the server's memory was not freed. `upstream_residency` stays
  `unknown`.
- **Engine loss.** With oMLX stopped, a chat returns 503 `runtime_unavailable`
  naming `endpoint_id: omlx`; `lewlm scan` keeps the model registered and
  marks the endpoint `stale`; native MLX, llama.cpp, and Ollama runtimes stay
  available. When oMLX dies mid-stream the client receives a terminal chunk
  with `finish_reason: "error"`, `error.code: runtime_unavailable`,
  `error.partial_output: true`, then `[DONE]`; the request is counted once and
  never replayed.
- **Cancellation.** Cancelling by `x-request-id` after the first chunk moves
  the handle to `cancelling` then `cancelled`, the LewLM stream stops, and a
  follow-up request succeeds. Whether oMLX aborted its own decode is not
  observable through the bridge; read oMLX's log for that.

## Testing this yourself

```bash
python scripts/backend_acceptance.py --base-url http://127.0.0.1:8080 --model <id> --endpoint-id omlx --output acceptance.json
python scripts/bridge_prefix_benchmark.py --lewlm-url http://127.0.0.1:8080 --model <id> --direct-url http://127.0.0.1:8000/v1 --direct-model <upstream id> --direct-api-key-env OMLX_API_KEY
```

Then stop, rescan, chat, restart, and rescan as described in the recipe.
Both scripts speak only LewLM's public HTTP API, so a host app such as Chap
can reuse the same flow against the same endpoints.
