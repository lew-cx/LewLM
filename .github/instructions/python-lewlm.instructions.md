---
description: "Use when: editing LewLM Python source, tests, CLI commands, API routes, runtimes, registry, documents, security, local model workflows, or pytest coverage."
applyTo: ["src/**/*.py", "tests/**/*.py"]
---

# LewLM Python Guidelines

- Target Python 3.11 and existing project dependencies from `pyproject.toml`.
- Keep runtime capability and fallback behavior explicit, typed, and testable.
- Do not use broad `except Exception` blocks unless the surrounding code already has a deliberate error-boundary pattern.
- Prefer deterministic unit tests with fixtures over tests that require real model weights, network access, or host-specific accelerators.
- For API changes, update schemas, routes, tests, and public docs together when the user-facing contract changes.
- For runtime changes, preserve local-first behavior and avoid forcing optional MLX, llama.cpp, document, or adapter dependencies onto core installs.
