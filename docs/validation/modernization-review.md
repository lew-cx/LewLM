# Modernization steps 00–12: release review

Reviewed 2026-09-21–22 against `141d591569c06cdabf05c290f877bf0cbd383c38`, plus the working-tree fixes recorded here. Host: Apple Silicon macOS; verification: isolated Python 3.11.15, Pydantic 2.13.1 / pydantic-core 2.46.1.

**Disposition: approved for the portable, opt-in experimental release scope after these fixes.** This is a local code-review recommendation, not a GitHub approval, deployment, or promotion of untested engine/platform combinations. Existing hardware restrictions and Chap UI acceptance remain in force.

The review used the [roadmap](../architecture/modernization-roadmap.md), per-step evidence, changes since `caae222`, and executable regression checks. Detailed code review concentrated on endpoint identity/discovery, HTTP transport, streaming and tool translation, routing/fallback, residency, install/build inputs, and release contracts. The full suite exercised the remaining portable surfaces.

## Corrected defects

| Finding | Fix and verification |
| --- | --- |
| Streaming 307/400/401/404/429/503 responses raised `httpx.ResponseNotRead` instead of the expected typed error. | Classify using status without reading the streaming body; close through the response context. All six new cases failed before the fix. Raw upstream error bodies are also omitted from public errors. |
| Synchronous discovery/probes could follow redirects and use ambient proxies; legacy `/v1` URLs produced doubled paths. | Dedicated no-proxy, no-redirect opener and shared root normalization. Live loopback regressions verify no redirected request and correct discovery with a bad ambient proxy. |
| Malformed inventory could look like a successful empty list and remove registered models. | Reject malformed inventory, retain last-known records, and mark the endpoint stale. Five invalid-payload regressions also verify recovery to an intentionally empty valid list. |
| In-band SSE error objects could be ignored; malformed choices/deltas could crash or disappear; a completed iterator did not deterministically close its upstream iterator. | Typed terminal errors, shape validation, and explicit async iterator closure. Tests cover errors followed by `[DONE]`, malformed non-streaming responses, and immediate close on completion. |
| Non-streaming replies containing both text and native tool calls lost the calls. | Add an optional typed native-call field to the internal generation response and pass it through the existing declared-tool validator for single and batched execution. HTTP tests preserve content, call ID, and arguments, and reject undeclared tools and null arguments. Public schema snapshots remain unchanged. |
| Both Dockerfiles passed a pip-compile continuation backslash as part of the native package requirement. | Strip the continuation before invoking pip. Execute each Dockerfile's actual extraction command against plain, hashed, marker-bearing, and absent native requirements. The two hashed-lock cases failed before the fix; all 17 image-contract tests pass. |
| Mocked MLX tests depended on real MLX packages being installed. | Complete package-discovery fixtures in the text/audio/vision unit modules and the installed-but-unsupported conversion case. No production availability checks were relaxed and no tests were skipped to obtain a pass. All 80 related tests pass without MLX installed. |

## Verification

Evidence is in [evidence/modernization-review](evidence/modernization-review/).

- Full non-long-running suite: **1,231 passed, 7 skipped, 5 deselected**, 218.44 seconds. The last null-argument preservation refinement and expanded mixed-reply parameterization were subsequently checked separately: **3 passed**. Two warnings concern upstream test-client deprecations.
- Wheel built and installed to an isolated target; its import path was verified outside the checkout. The installed-wheel Chap HTTP smoke passed **14/14** checks, including cancellation, outage, interrupted streams, tools, JSON, and reasoning.
- Integration-bundle schemas/errors match the checkout under pinned Pydantic.
- Generated dependency inputs match `pyproject.toml`; `pip check` reports no broken requirements.
- Backend compatibility manifest validates: oMLX retains its existing validated tuple; vLLM, SGLang, and ExLlamaV3/TabbyAPI remain deferred.
- `git diff --check` passes.

Core reproduction commands (activate an isolated Python 3.11 environment first):

```sh
python -m pip install -e '.[dev,documents]' numpy safetensors 'pydantic==2.13.1' 'pydantic-core==2.46.1'
python -m pytest -q -p no:cacheprovider -m 'not long_running'
python scripts/export_integration_bundle.py --check
python scripts/export_dependency_inputs.py --check
python scripts/validate_backend_compatibility.py
python -m pip check
python -m pip wheel --no-deps --wheel-dir /tmp/lewlm-review-dist .
python examples/chap_backend_smoke.py --fixture --stream-delay-ms 5 --output /tmp/chap-smoke.json
python -m pytest -q -p no:cacheprovider -rs tests/hardware
```

The repository's existing Python 3.14 environment failed to import `pyexpat` because its extension referenced an unavailable system libexpat symbol; even `pip check` failed there. An initial sandbox run also denied local socket binding. Those runs are not the release result. The passing run used a fresh Python 3.11 environment with loopback/subprocess access. Dependency versions are retained with the evidence.

## Deferred acceptance

| Work | Remaining prerequisite / command |
| --- | --- |
| Actual CPU/CUDA image builds, lean/full installs, native compilation and rebuild caching | Docker CLI exists but its daemon is not running. Run the repository's `docker` and `docker-full` CI jobs and `scripts/docker/measure_rebuild.sh` on a Docker host. Portable shell tests do not establish image-build success. |
| Linux/NVIDIA engines and CUDA offload | Supported Linux/NVIDIA host; follow pinned recipe, then `python scripts/backend_lanes.py run --lane linux_nvidia --recipe <recipe>`. |
| Native Windows and WSL2 | Run the documented lanes on their respective hosts; macOS tests do not establish Windows networking or native runtime compatibility. |
| Fresh real-engine Apple Silicon acceptance for this patch | No operator-started engine was named through `LEWLM_LANE_BASE_URL`, `LEWLM_LANE_MODEL`, and `LEWLM_LANE_RECIPE`. Prior step-05/12 oMLX evidence is retained, not claimed as a new run. |
| Chap UI | Execute [Chap's checklist](../guides/chap-validation.md) in the Chap repository against the selected release. The backend smoke is not UI acceptance. |
| Long-running model benchmarks | Explicitly excluded from this review run; no new performance or hardware claims. |

Changes are local and uncommitted. No engines were installed or launched for ordinary discovery, no production deployment was performed, and no external approval was requested. Rollback is the review patch itself; public model IDs, routing defaults, and the existing compatibility statuses were preserved.
