# Modernization step 10 validation — the Chap integration contract

Implemented on 2026-09-18 on the baseline Apple Silicon host. Everything in
this step is portable: it is proven over HTTP against a fake engine, on this
host and in CI on Linux, macOS, and Windows. Chap UI acceptance is a separate,
**pending** status recorded in the [Chap validation guide](../guides/chap-validation.md).

## What changed

| Area | Change |
| --- | --- |
| Fixture (new, shipped) | `lewlm.testing.fake_backend`: `FakeOpenAIEngine` (rule-based loopback `/v1/models` + `/v1/chat/completions`, streaming, native tool calls, schema-shaped JSON, long replies, key gating, stop/restart, die-mid-stream) and `FakeBackendFixture` (LewLM served by uvicorn on a loopback port in front of it, temp data dir, optional API key and CORS origins). `python -m lewlm.testing.fake_backend --port 8080` gives Chap a complete backend with no model. |
| Smoke (new) | `examples/chap_backend_smoke.py`: HTTP-only, `--fixture` or `--base-url`; 13 checks — service health, runtime startup, model picker, request identity, streaming, non-streaming, structured output, tools, reasoning, cancellation, failure, engine outage, interrupted stream — with structured observations and a JSON report. Reuses `scripts/backend_acceptance.py`'s harness. |
| Bundle (generated) | `scripts/export_integration_bundle.py` regenerates `schemas` and `errors` from the models (`--check` refuses drift) and captures the `chap` section — field notes plus exact, normalized payloads for every UI state — from the fixture. The previously failing schema snapshot is regenerated, not hand-edited. Byte-stable across runs. |
| Schemas (additive) | `ModelCapabilityAvailability` += `endpoint_id`, `engine_profile`, `execution_locality`, `engine_state` (`packaged` or the endpoint's cached inventory state). `HealthResponse` += `engines[]` (cached, no probe). `ResponseChunk` += `finish_reason`. `ModelResidencySnapshot` += `pending_lease_count`. |
| Streams | A named cancel now ends the stream with a terminal chunk (`finish_reason: "cancelled"` on chat; `done: true, finish_reason: "cancelled"` on responses) followed by `[DONE]`, instead of a bare end of body indistinguishable from a dropped connection. The upstream stream is still closed. |
| Errors | An explicitly requested endpoint-bound model whose endpoint is disabled or whose last inventory read failed is now `503 runtime_unavailable` with `details.endpoint_id` (it was a `400 routing_error` when the cache had been invalidated, and a 503 when it had not). |
| Residency fix | Callers waiting on a load are counted as `pending_lease_count` and converted to the lease under one lock. Before, the `balanced` policy's after-request cleanup could unload a *different* model on the same runtime in the instant between its load finishing and its waiter taking the lease ("Residency disappeared during loading"), which the six-alias test reproduced 3 times in 8 runs. Regression test added; 0 in 8 after the fix. |
| CI | The OS matrix now runs `export_integration_bundle.py --check` and the smoke in fixture mode, uploading `chap-smoke.json` per OS. |
| Docs | `docs/guides/chap-validation.md` (contract, three-state model, UI checklist with 12 pending items), host-app guide section, bundle/HTTP-API/configuration references, README pointer. |

## Portable results on this host

| Command | Result |
| --- | --- |
| `python examples/chap_backend_smoke.py --fixture` | **13 passed, 0 failed** ([report](evidence/modernization-step-10/chap-smoke.json)) |
| `python scripts/export_integration_bundle.py --check` | passes; two consecutive regenerations produce identical files |
| `tests/integration/test_chap_contract.py` — smoke via subprocess, bundle check, chap examples validate against `HealthResponse`/`RuntimeInfo`/`ModelCapabilityAvailability`/`ChatCompletionChunk`/`ChatCompletionResponse`/error rehydration, cancelled terminal chunk on both stream routes with the upstream connection closed, narrow CORS origin on JSON and SSE with `x-request-id` exposed and nothing for another origin | **7 passed** |
| `tests/integration/test_integration_bundle.py` (previously failing snapshot) | **9 passed** |
| `tests/unit/test_model_residency.py` (+ pending-lease regression) | passed |
| Residency race probe (six aliases, cap 2, 8 runs) | 3/8 failed before the fix; 0/8 after |
| Unit + integration suites (document/`pyexpat` cases deselected) | see the handoff record |

## Deferred

| Item | Status |
| --- | --- |
| Chap UI checklist (12 items) | **pending** — belongs to Chap's repository against a tagged LewLM; the backend side of each item is covered above |
| Smoke in real-model mode | run `chap_backend_smoke.py --base-url … --model …` against each engine recipe when its hardware lane runs; oMLX on this host is the first candidate |
| Bundle capture on other OSes | the capture is normalized and OS-independent; CI checks `schemas`/`errors` on every OS and runs the smoke, but only regenerates on request |

## Rollback

Revert the step commit. All schema fields are additive with `None`/empty
defaults; the cancelled terminal chunk and the 503 for an endpoint outage are
behaviour changes a client sees only in those two situations; the residency
fix changes no public shape beyond the additive `pending_lease_count`.
