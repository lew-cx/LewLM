# Modernization step 06 validation — ExLlamaV3 via TabbyAPI

Implemented on 2026-09-17 on the baseline Apple Silicon host. The portable
part of this step is complete; the real-engine lane needs Linux + NVIDIA and
is **deferred**. Nothing about ExLlamaV3 or TabbyAPI is labelled validated.

## What changed

| Area | Change |
| --- | --- |
| Profile | `exllamav3_tabby` added to `ExternalProfile`, `RuntimeProvider.EXLLAMAV3` added and mapped from the profile (`bridge_profile()` and middleware provider evidence), install-profile note, and a conservative performance-feature map (batching / paged KV / `cache_mode` quantization / chunked prefill as backend-native or partial, speculative decoding unsupported, constrained decoding partial) — all `active: false` until observed. |
| Format gating | Reuses step 04: a TabbyAPI listing registers `format_type: unknown`; `exl3` only when the record names it. The base-runtime URI guard and llama.cpp's GGUF-only `supported_formats` mean an EXL3 model can never reach llama.cpp; conversion refuses it as URI-backed. |
| Recipe | `examples/backends/exllamav3-tabby/`: `docker-compose.yml` (image pinned by digest, host-loopback port only, `shm_size: 8g`, single GPU, healthcheck), `config.yml` (keys verified against `config_sample.yml` at commit `53da791`: one model at startup, `inline_model_loading: false`, `disable_fetch_requests: true`, bounded `max_seq_len`/`cache_size`, `max_batch_size: 4`, `cache_mode: FP16`), `api_tokens.yml` (inference key vs admin key split), `lewlm.env.example`, README with the offline EXL3 conversion step, JIT/`MAX_JOBS` note, proof commands, and promotion checklist. |
| Docs | `docs/operations/backends/exllamav3-tabby.md` (nav), configuration reference profile list, capability-matrix row. |
| Manifest | `exllamav3_tabby` stays `deferred`; its reason and next step now name the pinned image digest and the exact promotion procedure. |

## Pins verified against upstream

| Item | Evidence |
| --- | --- |
| TabbyAPI commit `53da7919d4e45c63f4acbcbbc00cbe0f60a1ce65` | step-00 pin; `config_sample.yml`, `README.md`, `docker/docker-compose.yml`, `api_tokens_sample.yml` fetched at that revision and used for every key in the recipe |
| Image `ghcr.io/theroyallab/tabbyapi@sha256:10bfcf9d27d1b3a5c7ada786f814b9da43fadad35643362f2fe0816082893172` | OCI index resolved from ghcr.io on 2026-09-17; amd64 manifest `sha256:c4d24298…`; image config label `org.opencontainers.image.revision` = the pinned commit; `CUDA_VERSION=12.8.1`, `NVIDIA_REQUIRE_CUDA=cuda>=12.8` |
| ExLlamaV3 commit `02aef45cd681b960a00afcd0749a4ab99e6c1bfe` | README at that revision: prebuilt release wheels recommended; PyPI/source installs JIT-compile at first import; `MAX_JOBS` bounds compilation; `convert.py -i -o -w -b` is the conversion entry point; Windows needs `triton-windows` |
| `--shm-size` / `shm_size: 8g` | TabbyAPI README and compose at the pin: ExLlamaV3 uses POSIX shared memory; Docker's 64 MiB default is too small |
| Candidate model | `Qwen/Qwen2.5-0.5B-Instruct` BF16 @ `7ae5576…` (step-00 pin) as the conversion input; no EXL3 artifact exists yet |

## Portable results on this host

| Command | Result |
| --- | --- |
| `tests/unit/test_exllamav3_tabby_profile.py` — profile selection and provider mapping, format evidence (`unknown` by default, `exl3` only from the record, never accepted by llama.cpp), missing inference key blocks the endpoint before any request, wrong key → structured 401 `RuntimeUnavailableError` with `inventory_state: failed` then recovery with the right key, and a stopped Tabby endpoint leaves Ollama routable while conversion of the Tabby model is refused | **5 passed** |
| Middleware provider, adapter, install-profile, and settings suites | 73 passed alongside |
| `scripts/validate_backend_compatibility.py` | 4 recipes; `exllamav3_tabby` deferred |

## Deferred measurements and exact prerequisites

| Measurement | Missing prerequisite | Follow-up command / expected observation |
| --- | --- | --- |
| EXL3 conversion | Linux/NVIDIA host with CUDA 12.8 torch | recipe section 1: `python convert.py -i … -o … -w … -b 4.0`; record output `*.safetensors` sha256 |
| Serve + common suite | Same host, NVIDIA Container Toolkit | `docker compose up -d` with the digest-pinned image, `lewlm scan`, `scripts/backend_acceptance.py --endpoint-id tabby`; expected: 0 failed, tools/json_schema recorded per the loaded model |
| First start vs cached restart, memory | Same | time from `up` to `/health` on first start and on restart; `nvidia-smi --query-gpu=memory.used` while serving |
| Format evidence | Same | inspect `/v1/models` record; if it names `exl3`, the manifest reports `exl3`; otherwise it stays `unknown` and serving is still endpoint-bound |
| Coexistence | Same | stop the container, `lewlm scan` (model kept, stale), route a llama.cpp and an Ollama model |
| Native Windows | Windows host with `triton-windows` | separate install/generation/cancellation proof; a Linux pass does not certify it |

## Rollback

Revert the step commit. The profile value, provider enum member, feature map,
recipe, and docs are additive; an endpoint configured with `exllamav3_tabby`
would fail settings validation after a revert, which is the correct signal.
