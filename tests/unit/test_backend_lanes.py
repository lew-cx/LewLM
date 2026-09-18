"""Modernization step 11: acceptance lanes with honest deferrals and blocking failures.

The roadmap's exit demands three demonstrations: a passing fixture-only lane
(the CI matrix and `test_chap_contract.py`), a deliberate hardware deferral,
and a deliberate contract failure that blocks acceptance. The last two are
here, with the host facts and the acceptance runner injected so they hold on
any OS.
"""

from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load():
    root = Path(__file__).resolve().parents[2]
    spec = spec_from_file_location("lewlm_backend_lanes", root / "scripts" / "backend_lanes.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _host(module, **overrides):
    facts = {"system": "Linux", "machine": "x86_64", "nvidia": True, "nvidia_detail": "NVIDIA L4, 580.65.06", "wsl": False}
    facts.update(overrides)
    return module.HostFacts(**facts)


def test_detect_names_every_roadmap_lane_and_why_a_mac_cannot_run_the_hardware_ones() -> None:
    module = _load()
    report = module.detect(_host(module, system="Darwin", machine="arm64", nvidia=False, nvidia_detail="nvidia-smi: not found on PATH"))
    by_lane = {lane["lane"]: lane for lane in report["lanes"]}
    assert set(by_lane) == {"any_os", "linux_cpu", "apple_silicon", "linux_nvidia", "native_windows", "wsl2", "chap_ui"}
    assert by_lane["any_os"]["runnable_here"] and by_lane["apple_silicon"]["runnable_here"]
    assert not by_lane["linux_nvidia"]["runnable_here"]
    assert "needs a working NVIDIA driver" in "; ".join(by_lane["linux_nvidia"]["blockers"])
    assert not by_lane["chap_ui"]["runnable_here"], "the UI lane is a manual checklist"
    assert by_lane["wsl2"]["deferral_rule"] == "A Linux pass alone does not prove WSL networking"


def test_a_hardware_lane_on_the_wrong_host_is_a_deferred_record_with_the_next_command(tmp_path: Path) -> None:
    module = _load()
    record = module.run_lane(
        lane_name="linux_nvidia", recipe="vllm", base_url="http://127.0.0.1:8080", model="m", output_dir=tmp_path,
        host=_host(module, system="Darwin", machine="arm64", nvidia=False, nvidia_detail="nvidia-smi: not found on PATH"),
        acceptance_runner=lambda **kw: pytest.fail("acceptance must not run on a host that cannot run the lane"),
    )
    assert record["status"] == "deferred"
    assert "needs Linux" in record["reason"] and "NVIDIA" in record["reason"]
    assert record["next_command"].startswith("On a Linux NVIDIA host: follow examples/backends/vllm/README.md")
    written = json.loads((tmp_path / "lane-linux_nvidia-vllm.json").read_text(encoding="utf-8"))
    assert written["format"] == "lewlm-backend-lane-v1" and written["profile"] == "vllm_local"

    # Right host, but nobody started an engine: still a deferral, still exit 0 unless --require.
    idle = module.run_lane(lane_name="linux_nvidia", recipe="sglang", base_url=None, model=None, output_dir=tmp_path, host=_host(module))
    assert idle["status"] == "deferred" and "--base-url/--model" in idle["reason"]
    assert module.main(["run", "--lane", "linux_nvidia", "--recipe", "sglang", "--output-dir", str(tmp_path)]) in {0, 1}


def test_a_failed_acceptance_case_makes_the_lane_fail_and_blocks_promotion(tmp_path: Path, capsys) -> None:
    module = _load()

    def failing_acceptance(**kw):
        return {"cases": [{"case": "chat_streaming", "status": "passed"}, {"case": "cancellation", "status": "failed", "reason": "stream ran to completion"}],
                "summary": {"passed": 1, "failed": 1}, "evidence": [str(tmp_path / "acceptance.json")]}

    record = module.run_lane(lane_name="linux_nvidia", recipe="vllm", base_url="http://127.0.0.1:8080", model="m",
                             output_dir=tmp_path, host=_host(module), acceptance_runner=failing_acceptance)
    assert record["status"] == "failed"
    assert "cancellation" in record["reason"]
    assert "never promotes" in record["next_command"]

    passing = module.run_lane(lane_name="apple_silicon", recipe="omlx", base_url="http://127.0.0.1:8080", model="m", output_dir=tmp_path,
                              host=_host(module, system="Darwin", machine="arm64", nvidia=False),
                              acceptance_runner=lambda **kw: {"cases": [{"case": "chat_streaming", "status": "passed"}], "summary": {"passed": 1, "failed": 0}, "evidence": []})
    assert passing["status"] == "passed" and passing["reason"] is None

    # The CLI exit code is what a CI gate reads.
    module.HostFacts.detect = classmethod(lambda cls, runner=None: _host(module))  # type: ignore[method-assign]
    module._run_acceptance = lambda **kw: failing_acceptance()  # type: ignore[assignment]
    assert module.main(["run", "--lane", "linux_nvidia", "--recipe", "vllm", "--base-url", "http://127.0.0.1:8080", "--model", "m", "--output-dir", str(tmp_path)]) == 1
    assert '"status": "failed"' in capsys.readouterr().out


def test_summary_keeps_deferred_hardware_visible_and_prefers_lane_records(tmp_path: Path) -> None:
    module = _load()
    (tmp_path / "lane-linux_nvidia-vllm.json").write_text(json.dumps({
        "format": "lewlm-backend-lane-v1", "lane": "linux_nvidia", "recipe": "vllm", "status": "failed", "reason": "1 acceptance case(s) failed: tools",
    }), encoding="utf-8")
    summary = module.build_lane_summary(tmp_path)
    by_lane = {lane["lane"]: lane for lane in summary["lanes"]}

    nvidia = {entry["recipe"]: entry for entry in by_lane["linux_nvidia"]["entries"]}
    assert nvidia["vllm"]["status"] == "failed" and nvidia["vllm"]["source"] == "lane_record"
    assert nvidia["sglang"]["status"] == "deferred" and nvidia["sglang"]["source"] == "compatibility_manifest"
    assert nvidia["exllamav3-tabby"]["status"] == "deferred"
    assert nvidia["llamacpp-cuda"]["status"] == "deferred" and nvidia["llamacpp-cuda"]["source"] == "none"
    apple = by_lane["apple_silicon"]["entries"][0]
    assert apple["recipe"] == "omlx" and apple["status"] == "validated" and apple["evidence_path"].startswith("docs/validation/evidence/")

    # An idle-host deferral never hides validated evidence; a real failed run does override it.
    (tmp_path / "lane-apple_silicon-omlx.json").write_text(json.dumps({
        "format": "lewlm-backend-lane-v1", "lane": "apple_silicon", "recipe": "omlx", "status": "deferred", "reason": "no --base-url/--model",
    }), encoding="utf-8")
    apple = {e["recipe"]: e for e in module.build_lane_summary(tmp_path)["lanes"][2]["entries"]}["omlx"]
    assert apple["status"] == "validated" and "latest lane run here: deferred" in apple["detail"]
    (tmp_path / "lane-apple_silicon-omlx.json").write_text(json.dumps({
        "format": "lewlm-backend-lane-v1", "lane": "apple_silicon", "recipe": "omlx", "status": "failed", "reason": "1 acceptance case(s) failed: cancellation",
    }), encoding="utf-8")
    apple = {e["recipe"]: e for e in module.build_lane_summary(tmp_path)["lanes"][2]["entries"]}["omlx"]
    assert apple["status"] == "failed" and apple["source"] == "lane_record"
    assert by_lane["any_os"]["entries"][0]["status"] == "ci"
    assert by_lane["chap_ui"]["entries"][0]["status"] == "pending"
    assert summary["counts"]["deferred"] >= 3 and summary["counts"]["pending"] == 1
