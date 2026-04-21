# LewLM

LewLM is a **local-first middleware backend** for local AI applications. It exposes one core system through a CLI, a local HTTP API, SSE/WebSocket event streams, and an embeddable Python facade.

This documentation set is organized around the surfaces LewLM actually ships today:

- **Getting started** for install paths and a fast first run
- **Guides** for day-to-day operator and application workflows
- **Reference** for CLI, API, Python, configuration, runtime, and document details
- **Architecture** for how the registry, router, runtimes, storage, and telemetry fit together
- **Operations** for security and troubleshooting

## What LewLM provides today

- local model discovery and registry updates
- runtime routing across MLX, llama.cpp, audio, vision, and optional loopback adapters
- selectively owned serving-control layers on the primary MLX text path, with benchmark-backed default adoption
- chat and responses-style generation
- embeddings, rerank, audio transcription, and speech synthesis
- document ingest, deterministic document rendering, and built-in transform skills
- local tool execution with explicit authorization gates
- session persistence, export/import, and event streaming
- benchmarking, serving-profile autotune, and benchmark artifact capture
- release-manifest proof surfaces for performance-core and default-path acceptance
- experimental cluster coordination and distributed planning surfaces

## Public surfaces

| Surface | Best entry point |
| --- | --- |
| CLI | [CLI reference](reference/cli.md) |
| HTTP API | [HTTP API reference](reference/http-api.md) |
| Python embedding | [Python API reference](reference/python-api.md) |
| Host-app adoption | [Host-app integration](guides/host-app-integration.md) |
| Runtime behavior | [Runtime and capability matrix](reference/runtime-capability-matrix.md) |
| Product scope | [Scope matrix](reference/scope-matrix.md) |
| Documents | [Documents guide](guides/documents.md) |
| Security posture | [Security](security.md) |

## Suggested reading order

1. Start with [Getting started](getting-started/index.md).
2. Read [Models and routing](guides/models-and-routing.md), [Chat and responses](guides/chat-and-responses.md), and [Host-app integration](guides/host-app-integration.md).
3. Add [Documents](guides/documents.md), [Tools and skills](guides/tools-and-skills.md), and [Benchmarking and autotune](guides/benchmarking-and-autotune.md) as needed.
4. Use the Reference section when you need exact commands, endpoints, settings, or type surfaces.
