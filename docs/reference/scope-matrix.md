# Scope matrix

LewLM now keeps an explicit scope matrix in the codebase so the project can distinguish **core**, **performance core**, **optional modules**, **experimental surfaces**, and **out-of-scope ideas** without relying on roadmap memory alone.

## Scope labels

| Label | Meaning |
| --- | --- |
| `core` | Middleware-first backend behavior LewLM should keep stable across CLI, API, events, and Python embedding. |
| `performance_core` | LewLM-owned serving-control and benchmark/acceptance layers where selective performance ownership creates clear product value. |
| `optional_module` | Install-selectable modules that plug into the core package but should not be mistaken for always-on baseline scope. |
| `experimental` | Proof, diagnostics, or research-oriented surfaces that remain opt-in and truthfully bounded. |
| `out_of_scope` | Product directions LewLM should explicitly avoid. |

## Subsystem matrix

| Subsystem | Scope | Code paths | Install extras | Notes |
| --- | --- | --- | --- | --- |
| Configuration and security | `core` | `config/`, `security/` | base install | Filesystem scope, authorization, sandboxing, and local audit controls. |
| Public surfaces | `core` | `cli/`, `api/`, `events/`, `library.py`, `app_helpers.py` | base install | LewLM's stable middleware-facing entry points. |
| Registry and routing | `core` | `registry/`, `routing/`, `runtime/catalog.py`, `core/contracts.py`, chat/multimodal orchestrators | base install | Discovery, manifests, runtime contracts, and routing decisions. |
| Session and local state | `core` | `storage/`, `history/` | base install | Local persistence and session/export state. |
| Conversion pipeline | `core` | `conversion/` | base install | Compatibility checks, jobs, and conversion metadata. |
| Telemetry and capability reporting | `core` | `telemetry/`, `core/middleware.py`, `utils/validation_manifests.py` | base install | Runtime stats, readiness, middleware evidence states, and measured capability truth. |
| Serving-control layers | `performance_core` | `core/serving_core.py`, `core/speculation.py`, scheduler/cache modules, `serving_profiles.py` | base install | The selectively owned optimization layer LewLM is prepared to stand behind. |
| Benchmark and acceptance artifacts | `performance_core` | `benchmarking/` | base install | Evidence for defaults, claims, and prove-out work. |
| Concrete runtime packs | `optional_module` | `runtime/mlx_*`, `runtime/llamacpp`, `runtime/onnx_genai`, `runtime/adapters`, `runtime/metal` | base install + config, `mlx`, `llamacpp`, `onnx_genai` | Runtime contracts are core; concrete backend packs stay install-selectable, including the loopback external-accelerator bridge and ONNX Runtime GenAI prepared-bundle path. |
| Documents and local tooling | `optional_module` | `documents/`, `tools/` | `documents` | Deterministic ingest/render/skills and document-oriented local tools. |
| Frontier architecture diagnostics | `experimental` | `runtime/experimental/architectures.py`, `runtime/experimental/frontier.py` | none | Planning and diagnostics only. |
| Distributed cluster workflows | `experimental` | `runtime/experimental/distributed.py`, `api/routes/cluster.py` | none | Pipeline-parallel proof surface, not a production tensor-parallel engine. |
| Workflow engine | `out_of_scope` | none | n/a | LewLM should stay app-agnostic rather than becoming a workflow orchestrator. |
| Vector database and collection control plane | `out_of_scope` | none | n/a | Retrieval helpers may package context, but LewLM should not become a vector database product. |
| GUI or consumer chat app | `out_of_scope` | none | n/a | LewLM remains a backend package, not a desktop app. |

## Dependency shape

The intended dependency direction is:

1. **Core** establishes configuration, security, registry, routing, state, and public surfaces.
2. **Performance core** layers serving control and acceptance evidence on top of that base.
3. **Optional modules** plug into the core package without redefining the product boundary.
4. **Experimental** surfaces stay clearly separated from default-path claims.

That separation is visible in `core/bootstrap.py`, where service assembly is now grouped into core foundation, performance core, optional modules, and experimental helpers instead of one flat bootstrap block.
