# Release and validation

LewLM includes release-helper scripts for artifact capture and host validation.

## SBOM

```bash
python scripts/generate_sbom.py > out/sbom.json
```

## Dependency audit

```bash
python scripts/generate_dependency_audit.py > out/dependency-audit.json
```

## Release manifest

```bash
python scripts/generate_release_manifest.py > out/release-manifest.json
```

The release manifest now includes:

- `install_profiles` for current-host packaged-versus-bridge guidance
- `dependency_audit.compatibility_gates` for the 2026 dependency baseline states
- `frontier_acceptance` for frontier-family proof coverage
- `optimization_defaults` for benchmark-backed default adoption state
- `performance_core_acceptance` for Milestone 81-style serving-core proof coverage across batching, prefix reuse, tiered KV, speculation, constrained decoding, and measured default adoption
- `standards_refresh_acceptance` for the completed Milestones 121-132 matrix plus the operator summary of current, bridge-backed, optional, experimental, unsupported, and unverified states

## Bundle capture

```bash
python scripts/capture_release_bundle.py --output-dir out --require-target Darwin:arm64 --minimum-verified-models 1
```

The bundle capture writes artifacts such as:

- `sbom.json`
- `dependency-audit.json`
- `release-manifest.json`
- `release-candidate-validation.json`
- `release-artifact-index.json`

## Host validation workspace

```bash
python scripts/capture_host_validation.py --output-dir out/host-validation \
  --capture-all-capabilities \
  --require-target Darwin:arm64 \
  --require-target Linux:x86_64 \
  --require-target Windows:AMD64 \
  --minimum-verified-models 1
```

This workspace capture wraps the existing CLI and release scripts into one evidence directory. It records:

- `cli/config.json`, `cli/scan.json`, `cli/list-models.json`, and `cli/doctor.json`
- `capabilities/*.json` for explicitly requested models or every discovered model when `--capture-all-capabilities` is set
- `release-bundle/` outputs from `capture_release_bundle.py` plus an explicit `validate-release-candidate.json`
- `host-validation-evidence.json`, a machine-readable index with command summaries, exit codes, and artifact locations

When you already have a local LewLM API running on loopback, add `--api-base-url http://127.0.0.1:8000` to capture `/v1/health`, `/v1/runtime/stats`, and any extra `/v1/` probes listed in `--http-probe-manifest`. This is the intended Milestone 118 workflow for real-host chat, streaming, semantic, vision, audio, and document evidence without checking local artifacts into the repository.

Example probe manifest:

```json
{
  "probes": [
    {
      "name": "chat-stream",
      "method": "POST",
      "path": "/v1/chat/completions",
      "json_body": {
        "model": "bridge-model",
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}]
      }
    }
  ]
}
```

## Multi-host validation

```bash
python scripts/validate_release_candidate.py out \
  --require-target Darwin:arm64 \
  --require-target Linux:x86_64 \
  --require-target Windows:AMD64 \
  --minimum-verified-models 1 \
  --require-performance-core-pillar serving_core \
  --require-performance-core-pillar continuous_batching \
  --require-performance-core-pillar measured_registry_defaults \
  > out/release-candidate-validation.json
```

Release candidate validation now also checks `standards_refresh_milestones_completed`, which requires each enforced target to carry the completed 2026 standards-refresh summary from the release manifest. Missing standards proof remains explicit even when the host, dependency, frontier, or performance-core checks pass.

## Hardware acceptance lanes

Engine recipes are promoted per platform lane, never in general. The lanes,
their required proof, and their deferral rules are encoded in
`scripts/backend_lanes.py` and mirror the roadmap:

| Lane | Required proof | Deferral rule |
| --- | --- | --- |
| Any development OS | fake-server translation, tools/JSON/SSE, cancellation, migration, routing, fallback, schema/client checks | no hardware-based exemption (the CI matrix on Linux, macOS, Windows) |
| Linux CPU | lean install, portable GGUF generation, image boot, conversion image, cold/warm rebuild evidence | defer if Linux/container runtime unavailable |
| Apple Silicon macOS | oMLX suite and native MLX/llama.cpp/Ollama coexistence | defer on non-Apple hardware |
| Linux NVIDIA | vLLM, SGLang, ExLlamaV3/TabbyAPI suites; llama.cpp CUDA offload/build/cache | defer without supported NVIDIA hardware/driver |
| Native Windows | core install, llama.cpp import/generation, Ollama bridge, shutdown/cancellation | Linux or WSL results do not satisfy it |
| Windows + WSL2 | documented engine recipes and Windows-Chap-to-LewLM connectivity | a Linux pass alone does not prove WSL networking |
| Chap UI | the end-user checklist in the Chap validation guide | pending until run, never silently complete |

```bash
python scripts/backend_lanes.py detect                                   # which lanes this host can run, and why not the others
python scripts/backend_lanes.py run --lane linux_nvidia --recipe vllm \
    --base-url http://127.0.0.1:8080 --model <id> --output-dir evidence/lanes
python scripts/backend_lanes.py summary --records evidence/lanes         # what the release manifest embeds as `backend_lanes`
LEWLM_LANE_BASE_URL=http://127.0.0.1:8080 LEWLM_LANE_MODEL=<id> LEWLM_LANE_RECIPE=vllm \
    python -m pytest -q -p no:cacheprovider tests/hardware -m linux_nvidia   # the same lane as a pytest gate
```

`run` executes the common acceptance suite and the prefix benchmark against an
operator-started LewLM + engine and writes a `lane-<lane>-<recipe>.json`
record: `passed`, `failed` (exit 1 — a contract failure blocks promotion), or
`deferred` with the exact missing prerequisite and the next command. The
`tests/hardware` lanes skip with the same reasons when they cannot run, so
raw CI output always says why. The release manifest's `backend_lanes` section
merges lane records with `examples/backends/compatibility.json`: only
`validated`/`passed` entries are proof; `deferred` and `pending` entries stay
visible on purpose, and an idle-host deferral never hides a validated recipe.

## What these scripts are for

They help capture:

- the resolved dependency environment
- host and runtime readiness details
- git commit consistency
- verified-model coverage across required targets
- performance-core proof coverage across required targets
- portable validation-manifest handoffs

## Useful companion pages

- [Benchmarking and autotune](../guides/benchmarking-and-autotune.md)
- [Runtime and capability matrix](runtime-capability-matrix.md)
