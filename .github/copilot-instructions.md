# LewLM AI Agent Instructions

LewLM is a Python 3.11 local-first middleware backend for model discovery, routing, serving, CLI/API surfaces, document workflows, and runtime capability reporting.

## Project priorities

- Keep LewLM local-first. Do not introduce cloud dependencies, telemetry, hosted services, model downloads, or MCP/plugin integrations unless the user explicitly asks for that external system.
- Preserve honest capability reporting. Runtime-dependent behavior must return explicit fallback metadata instead of pretending every backend has parity.
- Avoid committing local model weights, machine-specific paths, secrets, tokens, personal data, benchmark artifacts, or generated runtime state.
- Keep optional dependencies optional. MLX, llama.cpp, document tooling, and external adapters should remain profile-gated when possible.
- Prefer narrow, well-tested changes over broad rewrites. Follow existing module boundaries under `src\lewlm`.

## Build and test

- Install for development with `python -m pip install -e ".[dev,documents]"`.
- Run targeted tests first with `python -m pytest <test-path> -q`.
- Run the full suite with `python -m pytest -q` before broad or behavior-changing changes.
- Treat `long_running` tests as opt-in unless the user explicitly requests real-model or benchmark validation.

## AI tooling

- Prefer instructions, skills, and custom agents for workflow improvements before adding MCP servers or plugins.
- Use hooks only for deterministic guardrails or context injection, not for long-running automation.
- Keep custom agents focused and minimally tooled. Use GPT-5.x models from the picker when available, but do not hardcode model names unless the environment has verified support.
