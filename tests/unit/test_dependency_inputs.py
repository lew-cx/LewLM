"""The image dependency inputs stay derived from pyproject.toml and lean."""

from __future__ import annotations

import tomllib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_CONVERSION_ONLY = ("torch", "transformers", "sentencepiece", "protobuf", "safetensors")


def _load_exporter():
    spec = spec_from_file_location("lewlm_export_dependency_inputs", REPO_ROOT / "scripts" / "export_dependency_inputs.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _names(requirements: list[str]) -> set[str]:
    return {req.split(";")[0].split(">")[0].split("<")[0].split("=")[0].strip().casefold() for req in requirements}


def test_committed_dependency_inputs_match_pyproject() -> None:
    exporter = _load_exporter()
    assert exporter.export(check=True) == 0, "run `python scripts/export_dependency_inputs.py`"


def test_every_flavor_file_exists_and_names_its_extras() -> None:
    exporter = _load_exporter()
    for flavor, extras in exporter.IMAGE_FLAVORS.items():
        text = exporter.output_path(flavor).read_text(encoding="utf-8")
        assert f"# Flavor: {flavor}" in text
        assert (", ".join(extras) or "none") in text


def test_llamacpp_runtime_extra_has_no_conversion_dependencies() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    optional = project["optional-dependencies"]
    runtime = _names(optional["llamacpp_runtime"])
    legacy = _names(optional["llamacpp"])

    assert "llama-cpp-python" in runtime
    assert not runtime & set(_CONVERSION_ONLY), runtime
    # The legacy extra keeps conversion for this release; it must remain a
    # superset so existing `.[llamacpp]` installs are unchanged.
    assert runtime <= legacy
    assert set(_CONVERSION_ONLY) <= legacy
    assert _names(optional["gguf_conversion"]) <= legacy


def test_flavors_nest_from_lean_to_full() -> None:
    exporter = _load_exporter()
    project = exporter.load_project()
    bridge = _names(exporter.flavor_requirements("bridge", project))
    serving = _names(exporter.flavor_requirements("serving", project))
    full = _names(exporter.flavor_requirements("full", project))

    assert bridge == _names(project["dependencies"])
    assert bridge < serving < full
    assert "llama-cpp-python" in serving
    assert not serving & set(_CONVERSION_ONLY)
    assert "weasyprint" not in serving
    assert {"torch", "transformers", "weasyprint"} <= full


def test_exporter_rejects_unknown_flavor() -> None:
    exporter = _load_exporter()
    try:
        exporter.flavor_requirements("gpu")
    except KeyError as exc:
        assert "gpu" in str(exc)
    else:
        raise AssertionError("unknown flavor must be rejected")
