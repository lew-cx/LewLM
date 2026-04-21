# Runtime and capability matrix

LewLM ships multiple runtime backends plus a few experimental surfaces.

## Runtime matrix

| Runtime | Affinity | Formats | Modalities | Capabilities | Platforms |
| --- | --- | --- | --- | --- | --- |
| `mlx_text` | `mlx_text` | `mlx` | text, multimodal-adjacent text paths | chat, streaming, embeddings, rerank | Darwin arm64/aarch64 |
| `mlx_vision` | `mlx_vision` | `mlx` | vision, multimodal | chat, streaming, vision | Darwin arm64/aarch64 |
| `mlx_audio` | `mlx_audio` | `mlx`, `audio_folder` | audio | audio transcription, audio speech | Darwin arm64/aarch64 |
| `llamacpp` | `llamacpp` | `gguf` | text | chat, streaming | Darwin, Linux, Windows |
| local OpenAI-compatible adapter | `external_accelerator` | local adapter-backed text path | text | chat, streaming | local loopback-oriented adapter configuration |
| frontier experimental | `experimental` | `gguf`, `mlx`, `huggingface` | text | planning/diagnostics only | diagnostic surface |
| distributed experimental | `distributed_experimental` | multiple | text, multimodal | chat, streaming | experimental cluster mode |

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

For `continuous_batching`, runtime stats also report whether chat/streaming support is currently **LewLM-owned** or still **backend-native** via `chat_streaming_ownership_mode`, `lewlm_owned_runtime_count`, and `backend_native_runtime_count`. On the primary MLX text path, LewLM now owns the persistent per-model scheduler while MLX `BatchGenerator` remains the decode primitive underneath.

## Important boundaries

- The MLX runtimes are intentionally Apple Silicon-first.
- `llamacpp` is the main cross-platform fallback path.
- The frontier and distributed runtimes should be treated as experimental surfaces, not default production backends.
