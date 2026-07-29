# ADR-001: Process-local shared runtime identity and model residency

- Status: accepted baseline
- Date: 2026-07-21
- Scope: incremental hardening; no package reorganization

## Context and repository audit

`LewLMServices` is the long-lived ownership boundary. `bootstrap_services()` creates the registry, runtime catalog and runtime adapter instances, schedulers, caches, orchestrators, telemetry, storage, and document services. Candidate serving-profile and benchmark paths call a `service_factory`; those calls intentionally create isolated containers and runtime instances.

The FastAPI lifespan stores one container on `app.state.services`, attaches the event bus to its event loop, and closes only containers created by the app factory. A caller-provided container remains caller-owned. `LewLM()` follows the same rule: it closes services it bootstraps, but not a container supplied through `services=`. `LewLMAppClient.from_lewlm()` calls the facade in process; `from_http()` serializes the same typed contracts to a long-running server and is the production multi-application path.

Before this ADR, live model state was held independently inside each `ManagedRuntime` in `_loaded_manifests`, timestamps, and counters. `load_model()` used an idempotent loaded check but had no per-model synchronization. Two callers could both pass that check and enter `_load_model()`. `unload_model()` was idempotent but had no active-use information.

Chat and multimodal orchestration each acquired runtime-request admission, separately acquired global model-load admission for cold loads, called the runtime load method, executed the capability, and applied the configured runtime policy. The global load scheduler bounded aggregate load work but was not a same-key correctness boundary. Warm and unload routes called `ModelRouter`, which called runtime methods directly. Runtime stats inferred loaded IDs from backend health snapshots. Existing coverage was strongest around backend lifecycle counters, request/load scheduling, runtime policy, chat streaming, and API warm/unload behavior; it did not prove same-model single-flight loading or lease-aware unload.

Each Uvicorn worker constructs a separate service container and therefore owns separate Python runtime objects and model memory. No in-memory component in this ADR is shared across operating-system processes.

## Decision

One `LewLMServices` container receives:

- immutable `RuntimeInstanceMetadata`, created once at bootstrap;
- one `ModelResidencyManager`, shared by the router and primary orchestrators;
- a residency key of `(runtime.name, manifest.model_id)`;
- per-key single-flight load tasks protected by a short async manager lock;
- model usage leases that cover backend execution, streaming tasks, and speculative companion models;
- lease-aware unload and bounded drain operations;
- capability-neutral lifecycle routing for chat, semantic, vision, and audio-only manifests;
- explicit load-owner, joined-waiter, usage, drain, and unload events with shared operation IDs;
- additive runtime identity and residency HTTP/client surfaces.

The persisted `ModelRegistry` remains responsible only for manifests. Backend adapters still own actual model objects. The existing global model-load scheduler remains responsible for aggregate admission; the residency manager owns same-key deduplication.

Waiters use `asyncio.shield()` so cancellation of one request does not cancel a shared load. Failures transition to `failed`, clear the in-flight task, retain a safe error summary, and allow retry. Normal unload returns HTTP 409 while a lease is active. Drain transitions the record to `draining`, refuses new leases, waits for the count to reach zero, and unloads. Force unload is not part of this baseline.

The `balanced` and `aggressive_unload` policy cleanup path now delegates to the residency manager when available. Balanced cleanup skips models leased by another request. Primary telemetry benchmarks and smoke probes use the same residency authority. External-adapter comparisons create a distinct service container, strictly clone the two compared runtimes, report the candidate runtime ID, and refuse to run if cloning would reuse a primary object. Shutdown retains a final backend sweep for legacy or deliberately isolated paths.

Application and client IDs are untrusted observability metadata. HTTP clients send `x-lewlm-application-id` and `x-lewlm-client-instance-id`; these values are never residency keys or credentials.

## Request flows

Non-streaming text and multimodal requests follow:

```text
route -> request admission -> residency acquire
      -> join/create load task -> global load admission -> backend load once
      -> increment model use -> backend capability -> decrement model use
      -> safe runtime-policy cleanup -> release request admission
```

Streaming acquires the lease before returning the stream session and releases it in the stream task's `finally` path. Explicit warm uses `ensure_loaded`, takes a short lease for the backend warm hook, and leaves the model resident. Explicit unload checks usage before calling the backend. Runtime shutdown stops new residency acquisitions, waits for shared loads, drains tracked use, unloads tracked models once, and performs the compatibility sweep.

## Consequences and limitations

Production applications can verify they reached the same process and can share one loaded residency. Closing a client has no model-lifecycle side effect. Embedded mode remains supported but creates its own runtime unless the caller deliberately shares a `LewLMServices` container.

This baseline does not share objects across workers or hosts, cancel arbitrary backend execution, add automatic memory-pressure eviction, or implement product-specific document workflows.

Lifecycle authorization supports scoped operator and administrator API credentials. Operators may inspect residency, warm, and drain; administrators may also unload and run disruptive diagnostics. The older explicit-action header remains only as a compatibility mode when scoped credentials are not configured. Application identity participates in decisions and audits but is never authentication.

Synchronous drain remains compatible. The additive asynchronous drain surface creates one service-owned operation with a stable ID and pending, running, succeeded, failed, or cancelled status. Callers can poll or cancel it; idempotency keys are scoped to application identity. Cancelling a drain restores a still-loaded residency to `ready` rather than leaving it stuck in `draining`.

Runtime request metrics maintain a bounded application table. They count requests, outcomes, leases, residency wait, model/capability use, and joined-load contention. Only the configured application ID is a metric dimension; client instance IDs and request IDs remain event/log/trace metadata.
