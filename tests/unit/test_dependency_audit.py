from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_dependency_audit_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "generate_dependency_audit.py"
    spec = spec_from_file_location("lewlm_dependency_audit", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_audit_captures_reproducibility_inputs(monkeypatch) -> None:
    module = _load_dependency_audit_module()
    monkeypatch.setattr(
        module,
        "_pip_check",
        lambda: {
            "tool": "pip check",
            "status": "passed",
            "exit_code": 0,
            "issues": [],
        },
    )

    payload = module.build_dependency_audit(
        resolved_packages=[
            "uvicorn==0.32.0",
            "fastapi==0.115.0",
            "uvicorn==0.32.0",
        ],
    )

    assert payload["format"] == "lewlm-dependency-audit-v1"
    assert payload["project_file"]["path"] == "pyproject.toml"
    assert len(payload["project_file"]["sha256"]) == 64
    assert payload["dependency_spec"]["dependencies"]
    assert "dev" in payload["dependency_spec"]["optional_groups"]
    assert len(payload["dependency_spec"]["digest"]) == 64
    assert payload["compatibility_gates"]["format"] == "lewlm-dependency-compatibility-gates-v1"
    assert payload["compatibility_gates"]["gates"]["llama_cpp_python_bindings"]["classification"] == "optional"
    assert any(
        requirement.startswith("cmake>=")
        for requirement in payload["compatibility_gates"]["gates"]["llama_cpp_python_bindings"]["requirements"]
    )
    assert payload["compatibility_gates"]["gates"]["optional_bridge_clients"]["classification"] == "bridge_owned"
    assert payload["compatibility_gates"]["gates"]["mlx_031_plus"]["classification"] == "watchlisted"
    assert payload["resolved_environment"]["package_count"] == 3
    assert len(payload["resolved_environment"]["package_digest"]) == 64
    assert payload["consistency_check"]["status"] == "passed"
