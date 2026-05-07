---
name: local-ai-runtime-review
description: "Review LewLM local AI runtime, routing, serving, MLX, llama.cpp, OpenAI-compatible adapter, benchmark, and fallback changes. Use for local-first runtime correctness and capability reporting."
argument-hint: "Describe the runtime or routing change"
---

# Local AI Runtime Review

Use this skill for changes that affect model discovery, routing, serving profiles, runtime adapters, benchmarks, or capability/fallback reporting.

## Procedure

1. Identify which runtime path is affected: MLX text, MLX vision, MLX audio, llama.cpp/GGUF, external OpenAI-compatible adapter, documents, or core routing.
2. Check whether the change affects public behavior through the CLI, API, events, Python facade, or app helpers.
3. Verify capability claims are honest:
   - return explicit fallback metadata when a runtime cannot support a constraint
   - avoid universal performance claims across all backends
   - keep optional dependencies profile-gated
4. Map to tests under `tests\unit`, `tests\integration`, or `tests\e2e`.
5. Avoid requiring local model weights unless the user explicitly requested real-model validation.

## Output

Return:

- affected runtime path
- capability/fallback impact
- public surface impact
- targeted tests
