# Contributing to LewLM

Thanks for contributing.

## Getting started

**macOS / Linux**

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev,documents]"
```

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,documents]"
```

If you prefer not to activate the environment, install with `.venv\Scripts\python -m pip install -e ".[dev,documents]"` on Windows or `.venv/bin/python -m pip install -e ".[dev,documents]"` on macOS/Linux.

## Development workflow

1. Keep changes focused on a single problem.
2. Add or update tests when behavior changes.
3. Update public docs when the user-facing surface changes.
4. Avoid committing local model weights, machine-specific paths, secrets, or other personal data.

Run the test suite before opening a pull request:

```bash
python -m pytest -q
```

## Documentation

- `README.md` is the public GitHub landing page.
- `docs/` contains the public documentation set.
- Example payloads and integration snippets live under `examples/`.

## Pull requests

Pull requests should describe the change clearly and note any behavior, docs, or compatibility impact.
