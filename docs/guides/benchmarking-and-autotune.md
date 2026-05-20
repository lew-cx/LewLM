# Benchmarking and autotune

LewLM includes operator-facing benchmarking and serving-profile selection surfaces.

## Benchmarking

The CLI benchmark flow can exercise multiple model/runtime combinations and persist artifacts under `data_dir/benchmarks`.

Key use cases:

- compare runtime choices
- evaluate serving-profile settings
- inspect optimization attribution
- capture benchmark artifacts for later review

## Autotune

Autotune benchmarks serving-profile candidates and persists the recommended profile for a specific host/model/runtime/workload tuple.

CLI:

```bash
lewlm autotune --model <model-id>
```

HTTP:

```http
POST /v1/benchmarks/autotune
```

Request fields:

- `model_id`
- `prompt`
- `capability`
- `workload_class`

## Safe default adoption

LewLM treats autotune output as a **measured default-adoption input**, not just an operator convenience. Persisted serving profiles give LewLM a host/model/runtime/workload recommendation that can be reused as the safe default path when the same request shape appears again.

That keeps default-path adoption honest:

- the chosen settings are tied to a specific host/model/runtime/workload tuple
- profile application can still be rejected when the routed runtime no longer matches
- request surfaces can opt out with `apply_serving_profile=false`
- optimization-default summaries can report which classes are benchmark-backed vs merely resolved

## What LewLM measures

The telemetry surface records more than just total runtime:

- load time
- execution time
- prompt and completion token counts
- completion tokens per second
- phase breakdowns
- serving-profile application details
- performance-feature attribution

## Performance features surfaced in stats

Examples of feature-level reporting include:

- continuous batching
- prefix and persistent cache behaviors
- paged KV and KV quantization
- speculative decoding
- prompt lookup speculation
- graph compilation and attention kernel acceleration
- request scheduling and prefill isolation
- multimodal feature and encoder caching

For the portable performance-core features, benchmark results and runtime stats also carry `ownership_modes`. That lets LewLM distinguish LewLM-owned behavior from backend-native preservation, partial adapter preservation, and outright unsupported paths without flattening everything into an MLX-shaped claim.

## Performance-core prove-out

Milestone 81 proof lives in the release-artifact flow rather than a separate hidden checklist. `scripts/generate_release_manifest.py` now captures:

- raw `benchmark_artifacts`
- persisted `serving_profiles`
- `optimization_defaults`
- `performance_core_acceptance`

`performance_core_acceptance` summarizes whether the current host has benchmark-backed proof for LewLM's selectively owned serving pillars:

- serving core
- continuous batching
- prefix reuse
- tiered KV cache
- speculation
- constrained decoding
- measured registry/default adoption

Use that summary when you need to answer "what does LewLM truly own on this host today?" without overclaiming parity across every backend.

Milestone 103 now chooses **GGUF via llama.cpp** as LewLM's first-class non-Apple path. On that path, benchmark-backed defaults come from serving-profile/autotune artifacts plus runtime-local evidence LewLM can package and report directly.

`--compare-external-adapter` artifacts still matter, but they now stay in the bridge-evidence bucket: LewLM uses them for fallback honesty and preservation reporting instead of promoting the external adapter over a first-class packaged runtime.

`runtime_support_strategy.paths[].performance_core_evidence` now reports those non-Apple defaults per portable serving behavior instead of flattening the whole path to one blanket benchmark flag. That means operators can distinguish:

- behaviors that are benchmark-backed on the current host
- behaviors that remain backend-native but not yet benchmark-backed
- fallback or unsupported behaviors that still keep their explicit boundary

## Related operator endpoints

- `GET /v1/cache/stats`
- `GET /v1/runtime/stats`
- `GET /v1/jobs/{job_id}`

See [Release and validation](../reference/release-and-validation.md) for release-bundle capture and validation scripts.
