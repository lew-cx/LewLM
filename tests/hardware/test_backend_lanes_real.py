"""Real-engine hardware lanes (modernization step 11).

Each test is one lane. It runs only when an operator has started LewLM in
front of a pinned engine recipe and named it:

    LEWLM_LANE_BASE_URL=http://127.0.0.1:8080 LEWLM_LANE_MODEL=<id> LEWLM_LANE_RECIPE=vllm \\
        python -m pytest -q -p no:cacheprovider tests/hardware -m linux_nvidia

Otherwise every lane *skips* with the exact missing prerequisite in the raw
test output — never a silent pass — and the release evidence keeps it
`deferred`. A failed acceptance case fails the lane. Nothing here installs,
starts, or stops an engine.
"""

from __future__ import annotations

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _lanes():
    spec = spec_from_file_location("lewlm_backend_lanes", ROOT / "scripts" / "backend_lanes.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_lane_or_skip(lane_name: str, tmp_path: Path) -> None:
    lanes = _lanes()
    lane = lanes.LANES[lane_name]
    host = lanes.HostFacts.detect()
    blockers = lanes.lane_blockers(lane, host)
    if blockers:
        pytest.skip(f"lane {lane_name} deferred on this host: {'; '.join(blockers)}. {lane.deferral_rule}.")
    base_url = os.environ.get("LEWLM_LANE_BASE_URL")
    model = os.environ.get("LEWLM_LANE_MODEL")
    recipe = os.environ.get("LEWLM_LANE_RECIPE")
    if not base_url or not model or not recipe:
        pytest.skip(
            f"lane {lane_name} can run on this host but no engine was named: set LEWLM_LANE_BASE_URL, LEWLM_LANE_MODEL, "
            f"and LEWLM_LANE_RECIPE (one of {list(lane.recipes)}) after following examples/backends/<recipe>/README.md."
        )
    if recipe not in lane.recipes:
        pytest.skip(f"recipe {recipe!r} is not part of lane {lane_name}; this lane covers {list(lane.recipes)}.")
    output_dir = Path(os.environ.get("LEWLM_LANE_RECORDS_DIR") or tmp_path)
    record = lanes.run_lane(lane_name=lane_name, recipe=recipe, base_url=base_url, model=model, output_dir=output_dir,
                            label=os.environ.get("LEWLM_LANE_LABEL"))
    assert record["status"] == "passed", f"{record['reason']} (record: {record['record_path']})"


@pytest.mark.real_engine
@pytest.mark.apple_silicon
def test_apple_silicon_lane(tmp_path: Path) -> None:
    _run_lane_or_skip("apple_silicon", tmp_path)


@pytest.mark.real_engine
@pytest.mark.linux_nvidia
def test_linux_nvidia_lane(tmp_path: Path) -> None:
    _run_lane_or_skip("linux_nvidia", tmp_path)


@pytest.mark.real_engine
@pytest.mark.linux_cpu
def test_linux_cpu_lane(tmp_path: Path) -> None:
    _run_lane_or_skip("linux_cpu", tmp_path)


@pytest.mark.real_engine
@pytest.mark.native_windows
def test_native_windows_lane(tmp_path: Path) -> None:
    _run_lane_or_skip("native_windows", tmp_path)


@pytest.mark.real_engine
@pytest.mark.wsl2
def test_wsl2_lane(tmp_path: Path) -> None:
    _run_lane_or_skip("wsl2", tmp_path)
