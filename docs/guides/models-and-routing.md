# Models and routing

LewLM separates **model discovery** from **runtime selection**.

## Discovery and registry

Use the registry flow to discover local bundles and store normalized manifests:

```bash
lewlm scan
lewlm list-models
lewlm capabilities <model-id>
```

Each manifest records:

- format (`gguf`, `mlx`, `huggingface`, `audio_folder`, `adapter_bundle`)
- modality (`text`, `vision`, `audio`, `embedding`, `rerank`, `multimodal`)
- runtime affinities
- tokenizer and processor paths when known
- quantization metadata
- estimated memory and context length when known
- conversion status (`runnable`, `requires_conversion`, `not_supported`, `unknown`)

## Routing behavior

LewLM routes by capability and request shape:

- chat / responses
- embeddings
- rerank
- audio transcription
- audio speech

For chat-like requests it also classifies the request modality, such as:

- text only
- text-only multimodal bundle
- single image
- repeated image
- frame-bundle/video
- audio-conditioned

## Capability reports

`lewlm capabilities <model-id>` and `GET /v1/models/{model_id}/capabilities` expose:

- which capabilities are supported today
- a machine-readable `readiness_state` for each capability and runtime candidate
- which runtime LewLM prefers
- why a capability is blocked or downgraded
- which capability claims now have measured host evidence vs still remain unmeasured
- estimated memory notes
- target-platform guidance and fallbacks

## Serving profiles

Serving profiles are persisted host/model/runtime/workload recommendations. They tune settings such as:

- runtime policy
- native batch window and max batch size
- KV cache page sizing and quantization
- prefill token batch size
- MLX graph compilation
- MLX attention kernel mode

Requests can opt out per call with `apply_serving_profile=false`.

## Conversion-aware routing

If a discovered model is not runnable yet, LewLM can:

- mark the bundle as `requires_conversion`
- queue a conversion job
- expose fallback guidance where a different runtime or target path is possible

## Operator workflow

1. Scan roots.
2. Inspect `conversion_status` and capability reports.
3. Convert incompatible source bundles when needed.
4. Warm a target model.
5. Benchmark or autotune if multiple profiles are viable.

See [Runtime and capability matrix](../reference/runtime-capability-matrix.md) for the current backend table.
