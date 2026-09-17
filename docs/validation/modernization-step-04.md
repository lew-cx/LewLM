# Modernization step 04 validation

Implemented on 2026-09-17 on the baseline Apple Silicon host. This step is
portable registry/routing work: every acceptance case runs against two live
fake loopback `/v1/models` servers and a stubbed Ollama daemon through the
real service bootstrap. No engine, GPU, or Docker daemon is required or
claimed.

## What changed

| Area | Change |
| --- | --- |
| Identity | `utils/model_identity.py` gains the URI-backed source namespace: `external://<endpoint_id>/<percent-encoded upstream id>`, `is_uri_source` / `is_external_source` / `parse_external_source`, and `external_model_id` (`<upstream-slug>-<endpoint-id>-<8 hex>`; stable, endpoint-qualified, collision-safe). `ollama://` is unchanged. |
| Inventory | New `registry/external_inventory.py` reads every enabled non-`ollama_local` endpoint's `/v1/models` through the endpoint's own runtime (shared credentials, timeouts, loopback rule) with bounded concurrency (`external_inventory_concurrency`, default 4) and per-endpoint failure isolation. Manifests carry `external_endpoint_id`, `external_profile`, the exact `external_upstream_model_id`, `execution_locality=loopback_unverified`, and an identity fingerprint (not an artifact hash). `ModelRegistry.scan` merges these with filesystem and Ollama results; the registry is bound to the catalog's endpoint runtimes by bootstrap. |
| Adapter | Successful `/v1/models` reads honor `external_inventory_ttl_seconds` (default 30; `0` = process lifetime). A failed refresh after a success keeps the last-known list and reports `inventory_state: "stale"` (with `inventory_error`, `inventory_age_seconds`, `inventory_ttl_seconds` in health); the 5 s failure retry is preserved; `advertised_model_records(refresh=True)` is the explicit-refresh path. Endpoint-bound manifests bypass the adapter's format gate (an unknown weight format is not a reason to refuse the owning endpoint). A `failed` inventory (never read) is a runtime-unavailable verdict in `candidate_report`. Passive health now attaches the adapter's static performance-feature snapshot (no probes). |
| Format evidence | `ModelFormat.EXL3` added. Format is assigned only from evidence: explicit `format` on the record, a `.gguf` artifact name, or a GGUF-only server (`llamacpp_server`, `ollama_local`); otherwise `unknown`. |
| Path guards | `ManagedRuntime.supports_manifest` refuses any URI-backed manifest for non-bridge runtimes, so llama.cpp/MLX/ONNX never open a URI as weights regardless of declared format. `ConversionService.plan_targets` and `submit` raise `ConversionError` for URI-backed sources. Registry stale-path logic never treats a URI as a filesystem path. |
| Ollama binding | Ollama manifests bind explicitly to the endpoint that *is* the daemon (named `ollama_local` entry, or `legacy-default` when its profile is `ollama_local` and its URL matches). When no endpoint is the daemon, models are still registered, stay unroutable, and the scan note says how to fix it. Ollama manifests also write the shared `execution_locality` key. |
| Registry reconciliation | External sources retire only when their endpoint is disabled/removed or a *successful* read no longer lists them; a failed read retires nothing and the note reports how many models are kept as stale. |
| Routing | `RoutingDecision` and execution metadata gain `endpoint_id`, `engine_profile`, `execution_locality`, `fallback_from_model_id`, `fallback_reason` (additive). Explicit model selection resolves exactly to its bound endpoint; an unroutable bound model's error carries `endpoint_id`, `engine_profile`, `fallback_policy`, and the endpoint's inventory snapshot. New `external_fallback_policy` (`none` default / `explicit_alias`) with `external_fallback_aliases`: substitution only for a URI-backed requested model, only to the configured registered alias, only if it is runnable and satisfies the same request, only before submission, always recorded. Default routing order is unchanged. |
| Settings | `external_inventory_ttl_seconds`, `external_inventory_concurrency`, `external_fallback_policy`, `external_fallback_aliases`, all validated. |
| Docs | Configuration reference (new settings, endpoint inventory rules) and models/routing guide (named-endpoint models, down-endpoint behavior, fallback, how to test). |

Compatibility: no existing setting, route, response field, or model id changed.
`ollama://` ids and `ollama_*` metadata keys are preserved. New fields default
to `None`/empty. A registry with no bound endpoint runtimes (tests, tools that
construct `ModelRegistry` directly) performs no external inventory and retires
no external rows.

## Portable results on this host

| Command / observation | Result |
| --- | --- |
| `tests/unit/test_external_inventory.py` (13 cases: URI round-trip, stable ids, format evidence, two live endpoints + Ollama coexisting with distinct identities, refresh add/remove only the changed endpoint, unreachable endpoint kept as stale with health evidence, disabled endpoint retired, conversion/packaged-runtime guards, adapter TTL and explicit refresh, exact explicit routing with evidence, unavailable endpoint surfaces failure while other paths keep serving, explicit-alias fallback recorded / unregistered alias refused, mid-stream failure not retried — fake server saw exactly one request) | **13 passed** |
| Roadmap focused regression command plus registry, Ollama, external endpoints, conversion, prefix-cache, discovery, and CLI suites | **314 passed** |
| `tests/unit` + `tests/integration`, `-m "not long_running"`, excluding the host-blocked document tests and the integration-bundle snapshot | **1058 passed, 6 failed, 42 deselected**; all 6 failures are the step-00 baseline `pyexpat`/document failures |
| `tests/integration/test_operations.py::test_runtime_stats_include_benchmark_history_and_target_platforms` and `::test_chat_orchestrator_batches_chat_requests_with_backend_native_batching` | Both **failed at every revision since step 01** (`e6b5fa5`, verified by worktree bisect; passing at `4226e97`) because step 01 routed the bridge through passive health without its static feature snapshot, dropping `partial` from the continuous-batching ownership aggregate. Fixed here by attaching the static snapshot to passive health (no probes; `test_lightweight_endpoint_health_uses_only_cached_evidence` still passes). Both now pass. |
| `tests/integration/test_integration_bundle.py::test_integration_bundle_schemas_match_current_models` | **Pre-existing failure** at `93bce1c` before this step (chat schema fields added in step 02 plus Pydantic drift). The bundle is hand-maintained and no export command exists; roadmap step 10 item 5 owns building it. Not hand-edited here. |

## Deferred measurements and exact prerequisites

| Measurement | Missing prerequisite | Follow-up command / expected observation |
| --- | --- | --- |
| Inventory against real engines (`vllm serve`, SGLang, oMLX, TabbyAPI, llama.cpp-server) | Steps 05–08 environments | `lewlm scan` with the engine's entry in `LEWLM_EXTERNAL_ENDPOINTS`; expected: one manifest per advertised id, `external_profile` matching, `format_type` per the evidence rules (`unknown` for vLLM/SGLang unless the record says otherwise), then `lewlm chat --model <id>`. |
| EXL3 format evidence from a running TabbyAPI | Step 06 | Confirm what TabbyAPI's `/v1/models` record exposes and, if it names the format, that the manifest reports `exl3`; otherwise `unknown` with endpoint-bound serving. |
| Stale-inventory behavior across an engine restart | Any real endpoint | Scan, stop the engine, scan (models kept, note says stale), start the engine, scan (state returns to `advertised`, removals applied only now). |
| Fallback across engines with a real stream | Two real endpoints | Configure `explicit_alias`, stop the primary, send a request: decision carries `fallback_from_model_id`; then kill the engine mid-stream on the alias: the stream errors and the engine log shows one request. |

## Rollback

Revert the step commit. New settings are additive with safe defaults. Any
`external://` rows already in the registry are left in place by the old scan
logic (they are under no model root), unroutable but harmless; re-applying the
step retires or refreshes them on the next scan. No stored data is migrated.
