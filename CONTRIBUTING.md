# Contributing to LewLM

Thanks for contributing.

## Getting started

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev,documents]"
```

## Development workflow

1. Keep changes focused on a single problem.
2. Add or update tests when behavior changes.
3. Update public docs when the user-facing surface changes.
4. Avoid committing local model weights, machine-specific paths, secrets, or other personal data.

Run the test suite before opening a pull request:

```bash
pytest -q
```

## Documentation

- `README.md` is the public GitHub landing page.
- `docs/` contains the public documentation set.
- Example payloads and integration snippets live under `examples/`.

## Pull requests

Pull requests should describe the change clearly and note any behavior, docs, or compatibility impact.
