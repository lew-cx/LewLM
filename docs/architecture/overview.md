# Architecture overview

LewLM is structured around one core service container that powers every public surface.

## Entry points

LewLM currently exposes:

- a CLI (`lewlm`)
- a FastAPI app (`create_app()` and `lewlm serve`)
- a Python facade (`LewLM`)
- event subscriptions over SSE, WebSocket, and in-process async queues

## Scope-aware subsystem layers

At bootstrap time LewLM now wires services in four explicit layers:

- **core**: settings, audit/authorization, metadata, registry, runtime contracts/catalog selection, routing, session state, conversion, telemetry, and the public CLI/API/Python surfaces
- **performance core**: serving-control state, schedulers, cache surfaces, speculation control, serving profiles, and benchmark-backed acceptance helpers
- **optional modules**: documents, document-oriented local tools, and install-selectable runtime packs
- **experimental**: frontier diagnostics plus distributed cluster proof surfaces

See the [scope matrix](../reference/scope-matrix.md) for the complete labeling and dependency map.

## High-level data flow

1. **Registry** discovers and normalizes local models.
2. **Router** picks a compatible model/runtime pair for the requested capability.
3. **Orchestrators** manage chat or multimodal request execution.
4. **Schedulers and caches** apply serving controls and reuse where available.
5. **Telemetry** records runtime, cache, and benchmark data.
6. **Events** publish lifecycle signals to API and in-process subscribers.

## Design shape

LewLM is intentionally not a GUI product. It acts as a middleware-first backend under other applications, with a stable local interface even when the concrete runtime differs by model family or platform.

On the primary Apple Silicon MLX text path, LewLM now owns the serving-control layers that create the most product value: serving-core state, continuous batching, tiered KV accounting, scheduler-aware prefix reuse, and selective speculation control. Other runtimes can still remain adapter-backed or backend-native as long as LewLM reports that boundary honestly through capability reports, benchmark artifacts, serving-profile defaults, and release-manifest acceptance summaries.
