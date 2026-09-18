"""Portable checks for `scripts/engine_preflight.py` (modernization step 07).

The script's inputs are `nvidia-smi`, `docker info`, and a loopback bind, all
injected here, so the pass/fail logic is proven on any OS without a GPU. It
never claims a GPU exists on the test host.
"""

from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load():
    root = Path(__file__).resolve().parents[2]
    spec = spec_from_file_location("lewlm_engine_preflight", root / "scripts" / "engine_preflight.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules under
    # `from __future__ import annotations`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_runner(*, docker_ok: bool = True, nvidia_runtime: bool = True, smi_rows: list[str] | None = None, smi_code: int = 0):
    def runner(command):
        if command[0] == "docker":
            if not docker_ok:
                return 1, "", "Cannot connect to the Docker daemon"
            runtimes = {"runc": {}, **({"nvidia": {}} if nvidia_runtime else {})}
            return 0, json.dumps(runtimes), ""
        if command[0] == "nvidia-smi":
            if smi_code:
                return smi_code, "", "NVIDIA-SMI has failed"
            return 0, "\n".join(smi_rows or []) + "\n", ""
        raise AssertionError(f"unexpected command {command}")
    return runner


def test_vllm_recipe_passes_on_a_matching_host() -> None:
    module = _load()
    runner = _fake_runner(smi_rows=["0, NVIDIA GeForce RTX 4090, 580.65.06, 8.9, 24564, 1200"])
    report = module.run_preflight(module.RECIPES["vllm"], runner=runner, port_probe=lambda host, port: True)

    assert report["result"] == "pass" and report["failed"] == []
    by_name = {check["name"]: check for check in report["checks"]}
    assert by_name["image_pin"]["status"] == "pass"
    assert by_name["container_runtime"]["status"] == "pass"
    assert by_name["driver_version"]["observed"] == {"driver_version": "580.65.06", "min_driver_major": 580}
    assert by_name["compute_capability"]["status"] == "pass"
    assert by_name["free_vram"]["observed"]["free_mib"] == 24564 - 1200
    assert by_name["host_port"]["observed"] == {"port": 8000}


def test_old_driver_unlisted_sm_low_vram_and_bound_port_each_fail_independently() -> None:
    module = _load()
    runner = _fake_runner(smi_rows=["0, Tesla T4, 535.104.05, 7.0, 15360, 14000"])
    report = module.run_preflight(module.RECIPES["vllm"], runner=runner, port_probe=lambda host, port: False)

    assert report["result"] == "fail"
    assert report["failed"] == ["driver_version", "compute_capability", "free_vram", "host_port"]
    by_name = {check["name"]: check for check in report["checks"]}
    assert "R580" in by_name["driver_version"]["detail"]
    assert "7.0" in by_name["compute_capability"]["detail"]
    assert by_name["host_port"]["detail"].startswith("127.0.0.1:8000 is already bound")


def test_missing_gpu_or_docker_is_a_failure_with_an_actionable_reason() -> None:
    module = _load()
    report = module.run_preflight(
        module.RECIPES["vllm"],
        runner=_fake_runner(docker_ok=False, smi_code=127),
        port_probe=lambda host, port: True,
    )
    assert report["result"] == "fail"
    assert set(report["failed"]) == {"container_runtime", "gpu_present"}

    no_toolkit = module.run_preflight(
        module.RECIPES["vllm"],
        runner=_fake_runner(nvidia_runtime=False, smi_rows=["0, NVIDIA L4, 580.65.06, 8.9, 23034, 0"]),
        port_probe=lambda host, port: True,
    )
    runtime_check = next(check for check in no_toolkit["checks"] if check["name"] == "container_runtime")
    assert runtime_check["status"] == "fail" and "nvidia-ctk runtime configure" in runtime_check["detail"]


def test_sglang_recipe_skips_the_architecture_check_rather_than_passing_it() -> None:
    module = _load()
    runner = _fake_runner(smi_rows=["0, NVIDIA A10, 580.65.06, 8.6, 23028, 0"])
    report = module.run_preflight(module.RECIPES["sglang"], runner=runner, port_probe=lambda host, port: True)

    assert report["result"] == "pass"
    assert report["skipped"] == ["compute_capability"]
    assert report["image"].startswith("lmsysorg/sglang@sha256:")
    assert report["checks"][-1]["observed"] == {"port": 30000}


def test_every_recipe_is_digest_pinned_and_a_tag_is_rejected() -> None:
    module = _load()
    for recipe in module.RECIPES.values():
        assert module.check_image_pin(recipe).status == "pass", recipe.recipe
    tagged = module.RecipeRequirements(
        recipe="custom", image="vllm/vllm-openai:latest", min_driver_major=None,
        supported_compute_capabilities=None, min_free_vram_mib=0, host_port=None,
    )
    assert module.check_image_pin(tagged).status == "fail"


def test_cli_writes_a_report_and_exits_nonzero_on_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load()
    monkeypatch.setattr(module, "_default_runner", _fake_runner(smi_code=1))
    monkeypatch.setattr(module, "_default_port_probe", lambda host, port: True)
    output = tmp_path / "preflight.json"

    assert module.main(["--recipe", "vllm", "--json", "--output", str(output)]) == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["result"] == "fail" and "gpu_present" in report["failed"]
    assert json.loads(capsys.readouterr().out)["recipe"] == "vllm"


def test_explicit_requirements_work_without_a_recipe(monkeypatch, capsys) -> None:
    module = _load()
    monkeypatch.setattr(module, "_default_runner", _fake_runner(smi_rows=["0, NVIDIA RTX A6000, 575.57.08, 8.6, 49140, 0"]))
    monkeypatch.setattr(module, "_default_port_probe", lambda host, port: True)

    assert module.main(["--min-driver", "570", "--supported-compute-capabilities", "8.6,8.9", "--port", "5000"]) == 0
    assert "result: pass" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        module.main([])
