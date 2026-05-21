# Runtime and capability matrix

LewLM's parity contract is about a stable **local-first middleware backend** surface plus honest support-path reporting. Full parity does **not** mean LewLM becomes a universal serving engine, GUI, hosted service, vector database, or workflow framework, and it does **not** mean every backend becomes equally packaged or equally LewLM-owned.

## Acceptance state legend

LewLM keeps two separate axes in the 2026 standards contract:

- support-path labels explain how a request reaches execution on the current host or target host
- acceptance states explain who owns or validates a named Milestone 120 vocabulary term

### 2026 acceptance states

| State | Meaning | Machine-readable contract |
| --- | --- | --- |
| `lewlm_owned` | LewLM directly implements, controls, and validates the behavior on that path | `standards_acceptance_contract.acceptance_states[].state`; later milestones can also surface it through `performance_features[].ownership_modes[]` or `runtime_support_strategy.paths[].performance_core_evidence[].mode` |
| `backend_native` | the backend owns the behavior and LewLM only detects, preserves, or reports it honestly | `standards_acceptance_contract.acceptance_states[].state`; stronger per-path evidence can appear under runtime support strategy or performance-core evidence |
| `partial` | only part of the intended behavior is preserved, observable, or exposed through LewLM | `standards_acceptance_contract.acceptance_states[].state`; adapter-preservation details can also surface through `performance_features[].ownership_modes[]` |
| `fallback` | LewLM keeps the public request contract but downgrades to a narrower or translated execution path | `standards_acceptance_contract.acceptance_states[].state`; runtime-specific fallback detail still lives in `fallback_used`, `fallback_reason`, `readiness_state`, or performance-core evidence `mode = "fallback"` |
| `unsupported` | LewLM does not claim the behavior on that path or host | `standards_acceptance_contract.acceptance_states[].state`; runtime-specific rejection detail still lives in `supported = false` and readiness fields |
| `unverified` | the term is reserved in the contract, but LewLM does not yet have host proof or probe-backed evidence for a stronger claim | `standards_acceptance_contract.acceptance_states[].state`; target validation still uses fields such as `verification_method` separately |

### Support-path and evidence signals

| Signal | Meaning | Machine-readable contract |
| --- | --- | --- |
| Packaged | LewLM imports or ships the runtime path directly on the relevant host | `support_path = "packaged"` in install-profile recommendations and runtime candidates |
| Bridge-backed | LewLM keeps the public contract but delegates execution to a loopback-only local server | `support_path = "bridge"` plus the `external_accelerator` runtime path |
| Fallback | LewLM accepts the request shape but reports a narrower execution or conversion-guided path | structured-output `fallback_used` / `fallback_reason`; target-platform `readiness_state = "fallback_guided"`; performance-core evidence `mode = "fallback"` |
| Unsupported | the requested runtime path is outside the documented promise on that host or feature class | `supported = false` with readiness such as `blocked`, `unsupported`, or `runtime_unavailable` |
| Benchmark-backed | LewLM has persisted host/model/runtime evidence for a default or performance behavior | `benchmark_backed = true` or `benchmark_backed_defaults = true` |
| Host-probed | the current host was verified directly instead of inheriting a declared cross-platform contract | `verification_method = "host_probe"` |

## Full parity acceptance matrix

| Feature class | Public-surface contract | Apple packaged path | Non-Apple packaged path | Bridge path | Fallback / unsupported boundary | Evidence and readiness |
| --- | --- | --- | --- | --- | --- | --- |
| Chat | CLI, HTTP API, events, and Python keep one local chat contract | `mlx_text` on Apple Silicon | `llamacpp` is the first-class packaged non-Apple text path on Darwin, Linux, and Windows | `external_accelerator` when another loopback-only local server already owns execution | non-runnable or conversion-required bundles stay explicit through blocked or fallback-guided readiness instead of widening product scope | `support_path`, install-profile recommendations, and target-platform `verification_method` distinguish packaged from bridge and `host_probe` from `runtime_contract` |
| Streaming | shared streaming contract across CLI, SSE/events, HTTP, and Python helpers | `mlx_text` streaming on Apple Silicon | `llamacpp` streaming on Darwin, Linux, and Windows | external bridge-backed streaming from a compatible loopback server | LewLM does not silently collapse streaming parity into non-streaming success on unsupported paths | same packaged-versus-bridge support-path reporting as chat |
| Semantic text | embeddings and rerank stay one public contract | `mlx_text` on Apple Silicon | `llamacpp` on compatible semantic GGUF models, with packaged embedding-similarity fallback for rerank when the backend lacks a native rerank API | external bridge with compatible local `/v1/embeddings` and `/v1/rerank` endpoints | without compatible semantic GGUF models or adapter routes LewLM reports blocked or runtime-unavailable instead of claiming universal semantic parity | `feature_class = "semantic_text"` recommendations report `support_path = "packaged"` on non-Apple hosts while bridge alternatives stay visible separately |
| Vision | image-conditioned chat stays shared across public surfaces | `mlx_vision` on Apple Silicon | unsupported as a first-class packaged non-Apple promise today | external bridge with OpenAI-style image content blocks on `/v1/chat/completions` | no packaged non-Apple vision parity claim; missing adapter compatibility stays explicit | current-host and target-platform readiness still separate `host_probe` from declared bridge guidance |
| Audio | transcription and speech stay one public contract | `mlx_audio` on Apple Silicon | unsupported as a packaged non-Apple promise today | external bridge with compatible local `/v1/audio/transcriptions` and `/v1/audio/speech`; this is the current bridge-only non-Apple public audio path | missing loopback routes stay blocked or runtime-unavailable; LewLM does not bundle the upstream server | install-profile recommendations, endpoint probes, and target-platform reports keep the bridge boundary explicit |
| Structured output | LewLM accepts structured-output requests through the same public surfaces | MLX keeps the contract but uses prompt-guided fallback when decode-time enforcement is unavailable | `llamacpp` is the first-class packaged decode-time JSON-schema and grammar enforcement path | external bridge may preserve the request shape but is not the portable decode-time enforcement default | `fallback_used` and `fallback_reason` remain explicit instead of overclaiming equal runtime parity | structured-output runtime metadata plus `constrained_decoding` performance-core evidence show fallback versus stronger packaged support |
| Documents | document ingest, render, and transform stay additive and stable | `.[documents]` installs LewLM-side packages on supported hosts | same additive documents extra on Darwin, Linux, and Windows | unsupported as a bridge-backed runtime class | OCR-style flows still need a local OCR engine; this does not redefine LewLM as a vector database or workflow engine | `documents_enabled_backend` plus OCR notes keep install readiness explicit |
| Performance-core evidence | runtime stats, health, and telemetry keep one reporting contract even when ownership differs | strongest on the Apple MLX text path, where LewLM can truthfully report `lewlm_owned` behavior | selective GGUF evidence is accepted when reported as `backend_native`, `fallback`, or `unsupported`, with benchmark-backed defaults where measured | bridge paths can report preserved backend-native or partial behavior without claiming LewLM-owned execution | no universal serving-core parity claim; unsupported and fallback modes stay public | `performance_core_evidence[].mode`, runtime `ownership_modes`, and `benchmark_backed` flags are the source of truth |

## Runtime inventory by backend

| Runtime | Pack | Install profile | Formats | Modalities | Capabilities | Platforms today | Operator notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mlx_text` | `mlx` | Apple MLX local backend | `mlx` | text, multimodal-adjacent text paths | chat, streaming, embeddings, rerank | Darwin arm64/aarch64 | First-class packaged path. Structured output remains prompt-guided fallback on this path rather than a decode-time enforcement claim. |
| `mlx_vision` | `mlx` | Apple MLX local backend | `mlx` | vision, multimodal | chat, streaming, vision | Darwin arm64/aarch64 | First-class packaged path |
| `mlx_audio` | `mlx` | Apple MLX local backend | `mlx`, `audio_folder` | audio | audio transcription, audio speech | Darwin arm64/aarch64 | First-class packaged path |
| `llamacpp` | `llamacpp` | Cross-platform GGUF backend | `gguf` | text, embedding, rerank | chat, streaming, embeddings, rerank | Darwin, Linux, Windows | First-class non-Apple packaged runtime family. LewLM productizes install/readiness guidance, benchmark-backed defaults, runtime-local control mapping, packaged embeddings, and packaged structured-output enforcement here without claiming MLX-level ownership parity. When llama.cpp lacks a native rerank API, LewLM keeps rerank honest through packaged embedding-similarity fallback instead of pretending backend parity. Non-Apple audio parity is not packaged on this path today. |
| local OpenAI-compatible adapter | `external_accelerator` | Cross-platform external accelerator bridge | `mlx`, `gguf`, `audio_folder` | text, vision, audio, embedding, rerank, multimodal | chat, streaming, vision, audio transcription, audio speech, embeddings, rerank | Darwin, Linux, Windows | Bridge to a loopback-only OpenAI-compatible local server. Vision uses OpenAI-style image content blocks on `/v1/chat/completions`; audio uses `/v1/audio/transcriptions` and `/v1/audio/speech`, which LewLM probes separately; embeddings and rerank remain adapter-backed through compatible local semantic endpoints. This is LewLM's current bridge-only non-Apple public audio parity path. LewLM does not claim MLX-level multimodal optimization or telemetry parity on this path, and structured-output requests stay bridge-backed with explicit fallback metadata rather than packaged decode-time parity. Bridge wins do not replace the first-class non-Apple packaged default. |
| frontier experimental | `experimental` | n/a | `gguf`, `mlx`, `huggingface` | text | planning/diagnostics only | diagnostic surface | Experimental only |
| distributed experimental | `distributed_experimental` | n/a | multiple | text, multimodal | chat, streaming | experimental cluster mode | Experimental only |

## 2026 standards vocabulary

The Milestone 120 vocabulary ships in `standards_acceptance_contract.vocabulary[].name` on `GET /v1/health.install_profiles`, `GET /v1/runtime/stats`, and `GET /v1/models/{model_id}/capabilities`. These names are normative reporting keys, not automatic support claims.

- `memory and context`: `kv_offload`, `kv_quantization`, `hybrid_memory`, `pd_disaggregation`, `distributed_kv_transfer`
- `structured output and reasoning`: `strict_tool_parser`, `reasoning_tags`, `parallel_tool_calls`, `streaming_tool_calls`, `responses_api_events`
- `speculation`: `mtp_speculation`, `eagle_speculation`, `dflash_speculation`, `ngram_draft_speculation`, `reasoning_budget_speculation`
- `dependency baselines`: `transformers_v5_ready`, `cuda13_ready`, `pytorch211_ready`, `cxx20_ready`
- `multimodal, document, and semantic`: `multimodal_omni`, `document_ocr_transformer`, `long_context_embedding`
- `agent interoperability`: `local_agent_sandbox`

Later milestones should attach per-term states through `runtime_support_strategy`, `performance_core_evidence`, `measured_capabilities`, target-platform verification data, install-profile guidance, and bridge probes without inventing new term names.

The release prove-out now lifts the same matrix into `release-manifest.json.standards_refresh_acceptance` so `validate_release_candidate.py` can verify Milestones 121-132 from one artifact bundle without flattening unsupported or unverified states into success claims.

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

Acceptance-matrix rows such as **semantic text**, **audio**, **structured output**, **documents**, and **performance-core evidence** intentionally group multiple machine-readable fields. `semantic text` means embeddings plus rerank, `audio` means transcription plus speech, and structured-output enforcement is reported through runtime metadata rather than the basic capability enum list.

## Recommended operator path by feature class

| Platform | Chat | Semantic text | Vision | Audio | Structured output |
| --- | --- | --- | --- | --- | --- |
| macOS | Apple MLX on Apple Silicon; GGUF on non-MLX Macs | Apple MLX on Apple Silicon; external bridge on non-MLX Macs | Apple MLX vision on Apple Silicon; external bridge on non-MLX Macs | Apple MLX audio on Apple Silicon; external bridge on non-MLX Macs | GGUF/llama.cpp for decode-time enforcement; MLX stays prompt-guided fallback |
| Linux | GGUF/llama.cpp packaged default | GGUF/llama.cpp packaged default for compatible semantic GGUF models; bridge remains optional | external accelerator bridge | external accelerator bridge | GGUF/llama.cpp packaged default |
| Windows | GGUF/llama.cpp packaged default | GGUF/llama.cpp packaged default for compatible semantic GGUF models; bridge remains optional | external accelerator bridge | external accelerator bridge | GGUF/llama.cpp packaged default |

`semantic text` covers embeddings and rerank, and `structured output` here means decode-time JSON-schema or grammar enforcement. On non-Apple hosts, compatible semantic GGUF models stay on the packaged path while `/v1/runtime/stats.runtime_support_strategy` and `install_profiles.recommended_feature_paths` keep the packaged-versus-bridge distinction explicit.

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

### Public vision request surfaces

The same bridge-backed vision path is available through LewLM's shared public surfaces:

- CLI: `lewlm chat --attach-image ...`
- HTTP API: `/v1/chat/completions` and `/v1/responses` with `input_image` parts
- Python facade: `LewLM.chat()` / `LewLM.chat_sync()` with `GenerateAttachment(attachment_type="image", ...)`
- Typed helpers: `LewLMAppClient.chat_completion()` / `.responses()` with `InputImagePart`

On the external adapter path, LewLM forwards OpenAI-style image content blocks to `/v1/chat/completions`, including the public `detail` hint when provided. This remains a **bridge** boundary: LewLM does not claim MLX-owned multimodal optimization, encoder-cache ownership, or full telemetry parity on Linux/Windows for that path.

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
| `ollama_local` | Ollama-compatible local server | explicit alias for the generic OpenAI-compatible bridge contract when the loopback endpoint follows Ollama-style local deployment patterns |
| `llamacpp_server` | llama.cpp-server-compatible local server | explicit alias for the generic OpenAI-compatible bridge contract when the loopback endpoint follows llama.cpp-server local deployment patterns |

## Routing considerations

Routing combines:

- manifest format and modality
- capability needs
- host-platform compatibility
- runtime availability
- request modality for chat-like workloads
- structured-output decode-time availability when the request asks for a non-text contract
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

For that path, `runtime_support_strategy.paths[].performance_core_evidence` is the behavior-level source of truth:

- `benchmark_backed: true` means LewLM has persisted host/model/runtime evidence for that serving behavior
- `mode: backend_native` means the backend still owns the primitive even when LewLM adopts it as the measured default
- `mode: fallback` or `mode: unsupported` keeps the non-claiming boundary explicit when the packaged GGUF path cannot expose the same primitive

## Important boundaries

- The MLX runtimes are intentionally Apple Silicon-first.
- `llamacpp` is the main packaged cross-platform runtime path today and the first-class non-Apple runtime family.
- Non-Apple `audio_transcription` and `audio_speech` are currently bridge-backed through `external_accelerator`, not packaged through `llamacpp`, and LewLM keeps that bridge-only audio boundary explicit.
- `external_accelerator` remains loopback-only and adapter-backed in this milestone.
- `external_accelerator` is a bridge to another local server, not proof that LewLM owns or bundles that server.
- `external_accelerator` only claims vision, audio, embeddings, or rerank when the configured local server satisfies the matching compatibility probe.
- `external_accelerator` does not currently claim MLX-owned encoder caching, MLX-level multimodal telemetry parity, or adapter-contract speculation controls.
- NVIDIA-oriented Linux/Windows operators should think of the external accelerator path as loopback bridge guidance first and packaged parity second.
- The frontier and distributed runtimes should be treated as experimental surfaces, not default production backends.
