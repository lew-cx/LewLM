# Modernization step 05 validation — oMLX

Executed on 2026-09-17 on the baseline Apple Silicon host (Apple M2 Max,
96 GiB, macOS Darwin 25.2.0). This is the one engine lane the host can run,
and it was run for real: oMLX at the step-00 pin was installed in an isolated
Python 3.11 environment, served the step-00 model on loopback, and LewLM in
front of it passed the common real-engine acceptance suite. The recipe is
promoted to `validated` in `examples/backends/compatibility.json` for exactly
this engine commit, dependency lock, model snapshot, and host.

## Real-engine evidence

| Item | Value |
| --- | --- |
| Engine | `jundot/omlx` @ `b45fb7e5a127355c0769d0b7c828849db17492e7` (`omlx 0.7.0.dev3`), `mlx 0.32.2`, `mlx-lm 0.31.3`, `transformers 5.17.0`, `llguidance 1.8.0`; 109-package lock at `examples/backends/omlx/requirements.lock.txt` (sha256 recorded in the manifest) |
| Install | `python3.11 -m venv && pip install ./src`: 116.7 s wall; no Xcode, no custom kernels |
| Model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a4…`; `model.safetensors` sha256 verified against the step-00 pin before use |
| Server | loopback only, API key required, `--memory-guard safe --memory-guard-gb 16 --max-concurrent-requests 4 --hot-cache-max-size 2GB` + 2 GB SSD cache; sandboxed `HOME`; ready in 2 s cold / 1.7 s restart |
| LewLM | `LEWLM_EXTERNAL_ENDPOINTS=[{"endpoint_id":"omlx","profile":"omlx","api_key_env":"OMLX_API_KEY",…}]`; `lewlm scan` registered `qwen2-5-0-5b-instruct-4bit-omlx-6fed8f48` (`external://omlx/Qwen2.5-0.5B-Instruct-4bit`, format `unknown`, context 32768 from `max_model_len`, locality `loopback_unverified`) |

### Common acceptance suite (`scripts/backend_acceptance.py`)

[acceptance.json](evidence/modernization-step-05/acceptance.json): **10 passed, 0 failed, 1 inconclusive, 1 not exercised**.

| Case | Result | Observation |
| --- | --- | --- |
| discovery_identity | passed | bound to `omlx`; endpoint snapshot `advertised`, TTL 30 s, `upstream_residency: unknown` |
| chat_nonstreaming | passed | `finish_reason: stop`, measured usage 36/2/38, `endpoint_id/engine_profile/execution_locality` in metadata, 108 ms execute |
| chat_streaming | passed | one terminal chunk with usage, `[DONE]`, no malformed frames |
| sampling | passed | `top_p`, `seed`, `stop` applied, none unsupported; **two seeded runs at temperature 0.9 produced identical output** (determinism observed, not assumed) |
| structured_output | passed | prediction `decode_time_modes: []` and outcome `decoder_enforced: false` / `enforcement_evidence: upstream_native` agree; output validated `full_json_schema` |
| tools | inconclusive | the 0.5B model emitted `<tool_call>{{…}}</tool_call>` with doubled braces — malformed JSON that neither oMLX nor LewLM parsed; not a bridge failure, not a pass |
| reasoning | passed | no `<think>` leakage with `reasoning_visibility: hidden` |
| cancellation | passed | `cancelling` → `cancelled` after the first chunk, stream stopped at 1 chunk / 288 ms, follow-up request 200 |
| concurrency | passed | two streams overlapped at LewLM; cancelling one left the other to complete (667 chars) |
| failure | passed | unknown model → 404 JSON error; health still `ok` |
| fallback | not exercised by the harness | done by hand below |
| lifetime | passed | warm = one-token probe; unload = lease release with `backend_operation_performed: false` and an explicit "did not free upstream memory" reason |

### Engine control by hand ([engine-control.json](evidence/modernization-step-05/engine-control.json))

| Step | Observation |
| --- | --- |
| oMLX stopped, chat | 503 `runtime_unavailable`, details name `endpoint_id: omlx` |
| scan while stopped | note "could not be read … keeping 1 previously registered model(s) as stale"; model still registered; endpoint `inventory_state: stale`; `mlx_text`, `mlx_vision`, `mlx_audio`, `llamacpp`, `frontier_experimental` still available |
| restart | ready in 1.7 s; chat 200; rescan back to `advertised` |
| SIGKILL mid-stream | stream stayed HTTP 200, delivered the partial chunk, then a terminal chunk `finish_reason: error`, `error.code: runtime_unavailable`, `error.partial_output: true`, then `[DONE]`; LewLM counted 1 request / 1 failure — no replay |

### Repeated-prefix measurement (`scripts/bridge_prefix_benchmark.py`)

600-token shared prefix, 32 output tokens, 3 warmups + 12 measured, oMLX cache in normal configuration. LewLM's response cache never applies to chat.

| Concurrency | LewLM TTFC p50 / p95 | Direct p50 / p95 | Overhead | Target |
| --- | --- | --- | --- | --- |
| 1 ([c1](evidence/modernization-step-05/prefix-benchmark-c1.json)) | 141.4 / 142.2 ms | 136.6 / 137.3 ms | +4.8 / +4.9 ms | ≤ 20 ms ✔ |
| 2 ([c2](evidence/modernization-step-05/prefix-benchmark-c2.json)) | 267.2 / 267.4 ms | 267.7 / 272.6 ms | −0.5 / −5.2 ms | ≤ 27.3 ms ✔ |

Cold first request via LewLM 163.8 ms vs ~141 ms warm: oMLX's prefix cache, reported as an observation; no hit counter is exposed through the bridge. A first pass of this benchmark reported a "134 ms overhead" because it timed oMLX's empty role-only first chunk as first output; both harnesses now time the first *content* chunk, and a 12-sample A/B across direct, orchestrator, and HTTP layers put LewLM's first-content overhead at 5.7 ms median and total at 6.4 ms.

## Findings the run produced, and what changed

1. **Structured-output prediction contradicted the outcome.** The bridge's capability report said `prompt_guided` while the generation result claimed `decode_time` with `decoder_enforced: true` — a claim LewLM cannot observe. Both now report the native path honestly: `enforcement: decode_time`, `decoder_enforced: false`, new `enforcement_evidence: upstream_native` on `StructuredOutputRuntimeStatus` and `StructuredOutputResult` (additive; `decoder` / `prompt` for the other cases), and validation runs after generation. The validation message no longer says "LewLM enforced … at decode time" for a bridge.
2. **A stream that failed after output had started reset the socket.** The transport raised correctly and nothing was replayed, but the SSE route let the exception escape after the response had begun (Starlette "response already started"). Both stream routes now end with a terminal chunk carrying a redacted `StreamErrorEnvelope` (`code`, `message`, `details` without upstream bodies, `partial_output`) and `finish_reason: "error"` (chat) / `done: true` (responses), then `[DONE]`. Covered by `tests/integration/test_stream_failure_envelope.py` and observed live.
3. **Bridge lifecycle wording implied freeing upstream memory.** Runtimes may now supply `lifecycle_note`/`lifecycle_backend_operation_performed`; the bridge's unload says it released LewLM's lease only.
4. **`deterministic: true` was a claim.** The acceptance harness now runs two seeded requests and fails the case if they differ.
5. Harness fixes from the first run: httpx streaming context handling, `/v1/models/{id}` envelope (`model`), endpoint evidence read from `/v1/runtime/stats`, first-content timing.

## Portable results

| Command | Result |
| --- | --- |
| `tests/integration/test_stream_failure_envelope.py` | 2 passed |
| `tests/unit/test_backend_compatibility.py` (validated recipe must point at a real lock whose sha256 matches, and at evidence with zero failures) | 9 passed |
| Focused streaming/cancellation/adapter/structured-output/routing suites | see the handoff record |

## Deferred

| Measurement | Missing prerequisite | Follow-up |
| --- | --- | --- |
| Tool calling on oMLX | a model that reliably emits well-formed tool calls (e.g. Qwen2.5-7B-Instruct-4bit) | rerun `backend_acceptance.py --only tools`; promote tools only for the model that passes |
| Vision / embeddings / rerank | matching oMLX-served models | probe per model; oMLX documents the endpoints, availability is per model |
| oMLX native custom kernels | full Xcode | only relevant to the model families oMLX lists; not part of this recipe |
| Full benchmark protocol (30+ requests × concurrency 1/2/4, long prefill, tool output, constrained JSON) | step 09 | `bridge_prefix_benchmark.py --requests 30 --concurrency 4` on a quiet host |
| Verified upstream abort on cancel | oMLX log inspection | read oMLX's request log for the cancelled request id |

## Rollback

Revert the step commit. `examples/backends/omlx/` and the evidence are
additive; the compatibility manifest returns to `deferred` for `omlx`. The
stream-error envelope, `enforcement_evidence`, and lifecycle notes are
additive response fields with `None`/absent defaults.
