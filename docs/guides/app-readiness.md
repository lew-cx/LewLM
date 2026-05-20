# App readiness

Host applications can treat LewLM's readiness story as a small three-step flow:

1. call `GET /v1/health` for service health plus the current capability-readiness summary
2. call `GET /v1/runtime/stats` when you need runtime-level detail for fallback, diagnostics, or startup telemetry
3. call `GET /v1/models/{model_id}/capabilities` when the host app wants to pin a specific model and inspect its exact readiness state

## Readiness states

LewLM now reports one consistent `readiness_state` for capability checks:

| State | Meaning |
| --- | --- |
| `ready` | LewLM can serve that capability on the current host right now |
| `no_models` | no discovered model currently advertises that capability |
| `conversion_required` | LewLM found candidates, but they must be converted first |
| `runtime_unavailable` | candidate models exist, but no compatible local runtime is ready on this host |
| `blocked` | candidate models exist, but LewLM still cannot serve the capability on this host |

## `/v1/health`

`/v1/health` stays focused on fast startup checks. In addition to storage and configuration status, it now includes:

- `install_profiles.active_profile_ids` so host apps can tell which documented install profile is actually present
- `install_profiles.profiles[]` with per-profile readiness and notes
- `readiness.status` for an overall host-app summary (`ready`, `partial`, or `blocked`)
- `readiness.ready_capability_count` and `readiness.capability_count`
- `readiness.capabilities[]` with per-capability readiness details, candidate model counts, and support-path labels such as `packaged` vs `bridge`

Use this route when your app needs a low-cost answer to "what did this LewLM install include, and can it do chat, embeddings, rerank, or audio work here?"

## `/v1/runtime/stats`

`/v1/runtime/stats` includes the same `readiness` block plus deeper runtime detail:

- per-runtime `readiness_state`
- per-capability `available_support_paths`, `packaged_runtime_names`, `bridge_runtime_names`, and `bridge_only`
- loaded-model and scheduler state
- benchmark, cache, and target-platform diagnostics
- `runtime_support_strategy` so host apps can see which runtime family is first-class on the current product line, which non-Apple path LewLM now productizes, and which paths remain bridge-only

For cross-platform audio specifically, `bridge_only=true` means LewLM can serve the public audio request class on this host, but only through the loopback external-accelerator bridge rather than a packaged runtime.

Use this route when your app needs to choose a degraded mode, show a diagnostics screen, or capture startup telemetry.

## Per-model capability reports

`GET /v1/models/{model_id}/capabilities` now exposes:

- `runtime_candidates[].readiness_state`
- `runtime_candidates[].support_path`
- `capabilities[].readiness_state`
- `capabilities[].support_path`

That lets a host app answer "is this exact model ready for chat or audio on this machine?" without translating several booleans and fallback notes by hand.

## Failure envelopes

When a request fails because the current host or selected runtime path cannot serve it, LewLM keeps the same machine-readable diagnostics across host-app surfaces:

- HTTP API responses return a top-level `error` object with `code`, `message`, and `details`
- typed local-server helpers raise `LewLMAppClientHTTPError` with the same `code`, `status_code`, and `details`
- `details` can carry parity-specific fields such as `support_path`, `feature_class`, `available_support_paths`, `bridge_only`, and `fallback_guidance`

That means readiness checks and request-time failures use the same vocabulary when a host app needs to downgrade, explain a bridge-only path, or show operator guidance.

## Event envelopes

Event streams over SSE, WebSocket, and the Python event bus now expose shared top-level fields:

- `request_id`
- `model_id`
- `runtime`
- `capability`
- `operation`
- `stage`
- `status`

Those fields are also mirrored into `payload` when LewLM can derive them safely, so consumers can treat lifecycle events consistently across chat, embeddings, rerank, audio, and document workflows.

Example:

```json
{
  "event_id": "...",
  "type": "request.completed",
  "scope": "request",
  "request_id": "req_123",
  "model_id": "qwen2-embed",
  "runtime": "fake_mlx_semantic",
  "capability": "embeddings",
  "operation": "embeddings",
  "status": "completed",
  "payload": {
    "request_id": "req_123",
    "model_id": "qwen2-embed",
    "runtime": "fake_mlx_semantic",
    "capability": "embeddings",
    "operation": "embeddings",
    "status": "completed"
  }
}
```
