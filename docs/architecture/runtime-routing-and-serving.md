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

For runtime-strategy reporting, LewLM now keeps the first-class non-Apple path behavior-specific: `runtime_support_strategy.paths[].performance_core_evidence` records whether continuous batching, prefix reuse, tiered KV, speculation, and related serving behaviors are benchmark-backed on the current host, merely backend-native, or still fallback/unsupported.

## Standards acceptance contract

Milestone 120 adds a shared `standards_acceptance_contract` to install-profile guidance, runtime stats, and per-model capability reports. That contract is a vocabulary registry, not a blanket feature-claim table.

It fixes one common state legend for later milestones: `lewlm_owned`, `backend_native`, `partial`, `fallback`, `unsupported`, and `unverified`.

It also reserves the 2026 reporting keys that later runtime, bridge, and validation work must reuse:

- `memory and context`: `kv_offload`, `kv_quantization`, `hybrid_memory`, `pd_disaggregation`, `distributed_kv_transfer`
- `structured output and reasoning`: `strict_tool_parser`, `reasoning_tags`, `parallel_tool_calls`, `streaming_tool_calls`, `responses_api_events`
- `speculation`: `mtp_speculation`, `eagle_speculation`, `dflash_speculation`, `ngram_draft_speculation`, `reasoning_budget_speculation`
- `dependency baselines`: `transformers_v5_ready`, `cuda13_ready`, `pytorch211_ready`, `cxx20_ready`
- `multimodal, document, and semantic`: `multimodal_omni`, `document_ocr_transformer`, `long_context_embedding`
- `agent interoperability`: `local_agent_sandbox`

Support-path labels such as `packaged` and `bridge` stay separate from these acceptance states. Later milestones should keep using runtime-specific fields like `runtime_support_strategy`, `performance_core_evidence`, `measured_capabilities`, and `verification_method` when they attach stronger per-path claims.

## Experimental layers

LewLM also carries experimental routing-adjacent surfaces:

- frontier architecture planning
- distributed planning and pipeline-stage coordination

Those are intentionally separated from the default runtime path.
