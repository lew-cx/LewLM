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
