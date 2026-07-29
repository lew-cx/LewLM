# Shared runtime hardening action plan

This turns `07-21-devguide.md` into implementation-sized work. Completed items describe the baseline in this change; remaining items are ordered so later work builds on one lifecycle authority.

## Phase 0 — ownership audit and decision record (complete)

- Record service, lifespan, facade, app-client, runtime-state, scheduler, lifecycle-route, worker, and test ownership.
- Preserve the registry/runtime/document package boundaries.
- Establish explicit compatibility rules for embedded and HTTP-backed operation.

Acceptance: ADR-001 exists and identifies direct lifecycle paths and intentional candidate-container isolation.

## Phase 1 — identity and residency primitives (complete)

- Create stable process metadata per service container.
- Add `(runtime_name, model_id)` residency keys and public snapshots.
- Implement same-key single-flight loads, joined-waiter counters, failure state, retry, and waiter-cancellation shielding.
- Keep aggregate load admission in `RuntimeRequestScheduler`.
- Add usage leases, safe 409 unload, bounded drain, idempotent unload, and shutdown coordination.
- Make concurrent drain/unload callers join one backend transition so the adapter unload hook runs once.
- Reconcile backend state when legacy paths load or unload outside the manager.

Acceptance: focused concurrency tests cover one backend load, cancellation, retry, lease cleanup, unload conflict, and drain.

## Phase 2 — primary orchestration and policy convergence (complete)

- Inject the service-owned manager into chat and multimodal orchestrators.
- Hold leases across non-streaming generation, native batches, stream-task lifetime, embeddings, rerank, retrieval stages, transcription, and speech.
- Hold speculative companion-model leases for the same lifetime as their parent request.
- Route balanced/aggressive cleanup through safe unload checks.
- Route primary telemetry benchmarks and smoke probes through the container's residency authority.
- Run external-adapter comparisons in a distinct service container with strict runtime cloning and refuse unsafe fallback to primary objects.

- Native batch-stream wrappers now stop queueing abandoned output, close native iteration when every consumer disconnects, preserve unaffected batch members, and release KV reservations and residency leases exactly once.

## Phase 3 — public contracts and authorization (complete)

- Add `GET /v1/runtime`, residency list/detail routes, drain, and enriched warm/unload responses.
- Resolve lifecycle operations by manifest/runtime compatibility rather than requiring chat capability.
- Extend `LewLMAppClient` without changing existing constructors or capability methods.
- Add optional application/client identity headers.
- Reuse explicit action authorization for warm, drain, and unload when that policy is enabled.
- Make the synchronous drain timeout configurable with `model_drain_timeout_seconds`.
- Prove lifecycle authorization audits exclude unrelated prompt and document headers.

- Add scoped operator and administrator API credentials. Operators can load, drain, and inspect; administrators can additionally unload and run disruptive diagnostics. Application identity remains untrusted context in every authorization audit.
- Preserve synchronous drain and add a `202` operation flow with pending/running/succeeded/failed/cancelled states, idempotent submission, polling, cancellation, bounded timeout, errors, events, and audits.
- Finalize even pre-start operation cancellation during explicit cancel or shutdown, and reject idempotency replays whose model or timeout differs.

## Phase 4 — observability (complete for this release)

- Add runtime identity to health, compact runtime info, runtime stats, lifecycle results, and residency lifecycle events.
- Report state, use count, load attempts, joined waiters, timestamps, failure, and estimated memory.
- Emit load-requested, load-joined, usage-acquired/released, drain-requested, and unload lifecycle events with stable operation IDs.

- Attribute request outcomes, residency leases, residency wait, model/capability use, failures, and same-load contention to bounded application summaries. Client-instance IDs remain in logs/traces/events and are deliberately excluded from metric dimensions.
- Capture identity before scheduler handoff so every member of native chat and embedding batches keeps its own application attribution and lease accounting.
- Keep idle native MLX package discovery import-free by default; explicit feature probing remains opt-in and loaded runtimes continue to report live feature metrics.

## Phase 5 — process-boundary validation (complete baseline; CI matrix remains operational work)

- Two independent HTTP clients against one app verify one residency and one load attempt.
- A subprocess fixture starts one Uvicorn server with a fake runtime and two separate application processes.
- A load barrier proves the second process joins an in-flight load. An active-lease barrier proves HTTP 409 during use, client disconnect safety, continued use by the second process, one backend load, and successful post-use unload.
- The subprocess harness now closes its caller-owned service container on graceful process shutdown and asserts that the backend unload is not duplicated.

Follow-up: run the subprocess test on Linux, macOS, and Windows CI and add platform-specific diagnostics for process startup failures.

## Phase 6 — deployment and ecosystem examples (baseline complete)

- Document one worker per intended model replica and explicit horizontal replicas.
- Show RAG Chat and document-generation clients verifying the same runtime ID.
- Keep `DocumentIR`, ingestion, validation, and deterministic rendering in LewLM; keep document types, templates, workflow, and artifact lifecycle in the document application.

Follow-up:

- Add reverse-proxy examples for deliberate replicas with affinity/stickiness guidance.
- Add a real external document-application contract test when a public integration repository exists.

## Release gates

- Run unit, integration, process-boundary, and existing document suites.
- Run real-model smoke tests only on capable opt-in hosts.
- Record backend cancellation/thread-safety limitations per adapter.
- Reject multi-worker model-owning deployment examples during documentation review.
