# Runtime and capability matrix

LewLM ships multiple runtime backends plus a few experimental surfaces.

## Runtime matrix

| Runtime | Pack | Install profile | Formats | Modalities | Capabilities | Platforms today | Operator notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mlx_text` | `mlx` | Apple MLX local backend | `mlx` | text, multimodal-adjacent text paths | chat, streaming, embeddings, rerank | Darwin arm64/aarch64 | First-class packaged path |
| `mlx_vision` | `mlx` | Apple MLX local backend | `mlx` | vision, multimodal | chat, streaming, vision | Darwin arm64/aarch64 | First-class packaged path |
| `mlx_audio` | `mlx` | Apple MLX local backend | `mlx`, `audio_folder` | audio | audio transcription, audio speech | Darwin arm64/aarch64 | First-class packaged path |
| `llamacpp` | `llamacpp` | Cross-platform GGUF backend | `gguf` | text | chat, streaming | Darwin, Linux, Windows | First-class non-Apple packaged runtime family. LewLM productizes install/readiness guidance, benchmark-backed defaults, and runtime-local control mapping here without claiming MLX-level ownership parity. |
| local OpenAI-compatible adapter | `external_accelerator` | Cross-platform external accelerator bridge | `mlx`, `gguf`, `audio_folder` | text, vision, audio, embedding, rerank, multimodal | chat, streaming, vision, audio transcription, audio speech, embeddings, rerank | Darwin, Linux, Windows | Bridge to a loopback-only OpenAI-compatible local server. Vision uses OpenAI-style image content blocks on `/v1/chat/completions`; audio uses `/v1/audio/transcriptions` and `/v1/audio/speech`; embeddings and rerank remain adapter-backed through compatible local semantic endpoints. LewLM does not claim MLX-level multimodal optimization or telemetry parity on this path, and bridge wins do not replace the first-class non-Apple packaged default. |
| frontier experimental | `experimental` | n/a | `gguf`, `mlx`, `huggingface` | text | planning/diagnostics only | diagnostic surface | Experimental only |
| distributed experimental | `distributed_experimental` | n/a | multiple | text, multimodal | chat, streaming | experimental cluster mode | Experimental only |

## Capability names

LewLM capability reporting uses:

- `chat`
- `streaming`
- `vision`
- `audio_transcription`
- `audio_speech`
- `embeddings`
- `rerank`
- `conversion`

## Local external adapter bridge

The `external_accelerator` runtime is a loopback-only local adapter bridge, not a general remote OpenAI integration. It is intended for:

- LewLM in front of another local OpenAI-compatible server on the same machine
- LewLM manifests that can be matched to an advertised remote model id for text, vision, audio, embeddings, or rerank
- chat and streaming through standard OpenAI-compatible routes
- vision through `/v1/chat/completions` with OpenAI-style image content blocks
- audio transcription through a compatible local `/v1/audio/transcriptions` route
- audio speech through a compatible local `/v1/audio/speech` route
- embeddings through a compatible local `/v1/embeddings` route
- rerank through a compatible local `/v1/rerank` route or equivalent extension
- bridge-oriented operator setups where LewLM does not own the low-level server process

The adapter runtime can report ready on Darwin, Linux, and Windows when LewLM is configured against a loopback-only local server that exposes the needed routes. Linux/Windows guidance, including NVIDIA-oriented local-server setups, is still bridge guidance rather than a claim that LewLM bundles or owns the upstream semantic backend.

The adapter can match advertised model ids through:

- explicit manifest metadata: `external_adapter_model_id` / `external_adapter_model_ids`
- the LewLM `model_id` and `display_name`
- the local source path name or file stem
- converted-bundle source metadata such as `source_model_id` and `source_display_name`

### External adapter profiles

These profiles are performance-preservation hints only. They do **not** imply LewLM owns the upstream execution backend, and LewLM only widens the reported capability contract when the local server actually exposes the corresponding adapter routes.

For portable performance-core reporting, runtime snapshots now tag the major text-path features with an explicit ownership mode:

- `lewlm_owned` for behavior LewLM directly implements and measures
- `backend_native` for behavior the backend owns but LewLM can truthfully detect/preserve
- `partial` for behavior that may remain active upstream but is only partially preserved or observable through LewLM
- `unsupported` when LewLM cannot claim the feature on that path

| Profile | Intended local server shape | Notes |
| --- | --- | --- |
| `openai_compatible` | generic local OpenAI-compatible server | safest default when LewLM cannot assume richer backend behavior |
| `vmlx` | Apple-oriented vMLX-class server | richer Apple-local preservation profile |
| `omlx` | Apple-oriented OMLX-class server | Apple-local text profile with partial cache reporting |
| `vllm_mlx` | vLLM-style compatible local server | useful bridge hint when the loopback server preserves paged-KV and prefix-cache behavior |
| `vllm_local` | vLLM-style local server | cross-platform bridge hint for local servers that preserve semantic and scheduler behavior |
| `sglang_local` | SGLang-style local server | cross-platform bridge hint for local servers that preserve compatible loopback semantic routes |

## Routing considerations

Routing combines:

- manifest format and modality
- capability needs
- host-platform compatibility
- runtime availability
- request modality for chat-like workloads
- persisted serving-profile preferences

## Workload classes for serving profiles

| Workload class | Meaning |
| --- | --- |
| `text_only` | plain text request |
| `text_only_multimodal` | text request against multimodal-capable model |
| `single_image` | one image attachment |
| `repeated_image` | multiple or repeated image contexts |
| `frame_bundle_video` | video/frame-bundle input |
| `audio_conditioned` | audio-conditioned request |

## Performance features surfaced in telemetry

Examples of runtime and cache features exposed through runtime/cache stats:

- continuous batching
- prefix cache
- disk-backed cache and block cache
- paged KV cache and KV cache quantization
- graph compilation
- attention kernel acceleration
- speculative decoding
- prompt lookup speculation
- request scheduling and backpressure
- decode-priority scheduling
- prefill optimization and isolation
- multimodal feature and encoder caching

For `continuous_batching`, runtime stats also publish aggregate ownership through `ownership_modes`, `chat_streaming_ownership_mode`, `lewlm_owned_runtime_count`, `backend_native_runtime_count`, and `partial_runtime_count`. On the primary MLX text path, LewLM owns the persistent per-model scheduler while MLX `BatchGenerator` remains the decode primitive underneath; non-MLX or adapter-backed paths can now report backend-native or partial preservation without pretending LewLM owns the same core.

Milestone 103 now selects **GGUF via llama.cpp** as the single first-class non-Apple path. That choice is evidence-backed because LewLM can package the runtime, attach benchmark-backed serving defaults to it directly, and report runtime-local control boundaries honestly without turning every external server family into an equal product promise.

## Important boundaries

- The MLX runtimes are intentionally Apple Silicon-first.
- `llamacpp` is the main packaged cross-platform runtime path today and the first-class non-Apple runtime family.
- `external_accelerator` remains loopback-only and adapter-backed in this milestone.
- `external_accelerator` is a bridge to another local server, not proof that LewLM owns or bundles that server.
- `external_accelerator` only claims vision, audio, embeddings, or rerank when the configured local server satisfies the matching compatibility probe.
- `external_accelerator` does not currently claim MLX-owned encoder caching, MLX-level multimodal telemetry parity, or adapter-contract speculation controls.
- NVIDIA-oriented Linux/Windows operators should think of the external accelerator path as loopback bridge guidance first and packaged parity second.
- The frontier and distributed runtimes should be treated as experimental surfaces, not default production backends.
