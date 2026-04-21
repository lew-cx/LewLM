# Runtime routing and serving

## Registry to runtime selection

Routing begins with discovered `ModelManifest` records. The router uses manifest metadata plus request intent to choose:

- a model
- a runtime
- a routing decision with explanations and alternatives

## Main routing steps

1. Filter candidate manifests by requested model, modality, and context hints.
2. Ask the runtime catalog for compatible runtimes.
3. Reject runtimes that fail platform or environment checks.
4. Score remaining candidates.
5. Apply serving-profile preferences when available.
6. return the chosen model/runtime pair plus alternatives.

## Serving layers

Serving behavior combines:

- runtime lifecycle management (`load`, `warm`, `unload`)
- request scheduling and backpressure
- decode-priority and prefill-aware admission control
- scheduler-integrated prefix/prefill cache reuse for repeated prompt state
- response caching and in-flight coalescing for deterministic operations
- block and multimodal encoder cache surfaces
- serving-profile materialization for request-specific settings

## Capability reporting

LewLM's capability reports are more than a boolean matrix. They include:

- supported vs blocked capabilities
- preferred runtime
- memory estimates
- target-platform notes
- fallback guidance for conversion or alternate runtimes

## Experimental layers

LewLM also carries experimental routing-adjacent surfaces:

- frontier architecture planning
- distributed planning and pipeline-stage coordination

Those are intentionally separated from the default runtime path.
