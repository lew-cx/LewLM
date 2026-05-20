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

- `frontier_acceptance` for frontier-family proof coverage
- `optimization_defaults` for benchmark-backed default adoption state
- `performance_core_acceptance` for Milestone 81-style serving-core proof coverage across batching, prefix reuse, tiered KV, speculation, constrained decoding, and measured default adoption

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
