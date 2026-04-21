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

## Related operator endpoints

- `GET /v1/cache/stats`
- `GET /v1/runtime/stats`
- `GET /v1/jobs/{job_id}`

See [Release and validation](../reference/release-and-validation.md) for release-bundle capture and validation scripts.
