---
name: lewlm-validation
description: "Plan and run validation for LewLM changes. Use for pytest selection, targeted test mapping, Python hook/script syntax checks, docs-only checks, long_running test decisions, and pre-PR confidence."
argument-hint: "Describe changed files or behavior"
---

# LewLM Validation

Use this skill to choose the smallest useful validation for a LewLM change.

## Procedure

1. Identify changed areas:
   - `src\lewlm\api` -> API route/schema tests.
   - `src\lewlm\runtime` or `src\lewlm\routing` -> runtime, scheduler, routing, and fallback tests.
   - `src\lewlm\documents` or `src\lewlm\conversion` -> document and conversion tests.
   - `.github\agents`, `.github\skills`, `.github\instructions`, `.github\hooks` -> customization validation and script syntax checks.
2. Run the narrowest command first, for example:
   - `python -m pytest tests\unit\test_registry.py -q`
   - `python -m pytest tests\integration\test_runtime_policy.py -q`
   - `python -m py_compile .github\hooks\scripts\safety_guard.py`
3. Use `python -m pytest -q` for broad behavior changes or before PR handoff.
4. Treat tests marked `long_running` as opt-in unless the user asks for real-model benchmark coverage.

## Output

Return:

- validation scope
- command
- result
- next command only if needed
