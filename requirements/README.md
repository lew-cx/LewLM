# Dependency inputs for the container images

`bridge.txt`, `serving.txt`, and `full.txt` are **generated** from
`pyproject.toml` by `scripts/export_dependency_inputs.py` and checked by
`tests/unit/test_dependency_inputs.py`. Do not edit them by hand; change
`pyproject.toml` and rerun the script.

They exist so the Docker dependency layer is keyed on a small file that only
changes when the dependency set changes, never on application source. They are
deliberately unpinned inputs.

`locks/` holds pinned, hashed resolutions produced on the image's own platform
by `scripts/docker/lock_dependencies.sh`. Select one with
`--build-arg DEPENDENCY_INPUT=requirements/locks/<name>.txt`. Nothing under
`locks/` is generated automatically; a lock is only as current as the last
time someone ran the script.
