# vLLM behind LewLM (Linux + NVIDIA)

Status: **passed on Windows 11 + Docker Desktop (WSL2)** on 2026-09-24 —
RTX 5090 Laptop (SM 12.0, driver 610.47), LewLM native on Windows, this
compose file unchanged except for `VLLM_WSL2_ENABLE_PIN_MEMORY` (below): the
common acceptance suite (11 passed, fallback exercised separately), upstream
concurrency from vLLM's own gauge, kill mid-stream, alias fallback, restart,
and disable. See the [Windows/Linux validation record](../../../docs/validation/modernization-windows-linux.md).
Bare-metal Linux/NVIDIA remains **deferred** in
`examples/backends/compatibility.json`: a WSL2 pass is its own lane.

| Input | Pin |
| --- | --- |
| vLLM | release `v0.29.0` = commit `98dff2a81d747d1dba01a47f939f48c3526d4206`; image `vllm/vllm-openai@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` (Docker Hub manifest list for `v0.29.0`; amd64 image `sha256:082ca6f0…`; config labels `ai.vllm.build.commit` = the commit, `VLLM_IMAGE_TAG=vllm/vllm-openai:v0.29.0`) |
| Inside the image | CUDA `13.0.2`, Python 3.12, `torch==2.13.0` (`requirements/cuda.txt` at the commit), `TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 12.0`, `VLLM_ENABLE_CUDA_COMPATIBILITY=0` |
| Host requirements | NVIDIA driver **R580 or newer** (CUDA 13 run normally), NVIDIA Container Toolkit, one GPU with compute capability in the list above and ≥ 4 GiB free. R535/R570 hosts can only use this image through `VLLM_ENABLE_CUDA_COMPATIBILITY=1`, which upstream limits to select professional/datacenter GPUs; that mode is *not* part of this recipe |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` BF16 @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0; step-00 pin, `model.safetensors` sha256 in the manifest), served as `qwen2.5-0.5b-instruct` |
| Tool parser | `hermes` — vLLM's `docs/features/tool_calling.md` at the commit names it for `Qwen/Qwen2.5-*` (the chat template is Hermes-style) |
| LewLM profile | `vllm_local`, endpoint id `vllm`. **Not** `vllm_mlx`, which is the Apple Silicon fork's profile |

## 0. Preflight — before pulling anything

```bash
python scripts/engine_preflight.py --recipe vllm --output preflight.json
```

It checks the digest pin, Docker + the `nvidia` container runtime, driver
≥ R580, the GPU's SM against the image's compiled list, free VRAM, and that
`127.0.0.1:8000` is free. It pulls, installs, and starts nothing. A failing
check names the fix; keep `preflight.json` with the evidence.

## 1. Serve on host loopback

```bash
cd examples/backends/vllm
export VLLM_API_KEY="$(openssl rand -hex 24)"     # inference key; LewLM sends the same value
docker compose up -d
docker compose logs -f vllm                        # first start downloads the model, then compiles
curl -s -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8000/v1/models
```

What the compose file fixes, and why:

- **Loopback only.** Port 8000 is published to `127.0.0.1`; LewLM must run on
  the host, not in another container (`127.0.0.1` is the endpoint rule).
- **One model, immutable.** `--revision`/`--tokenizer-revision` pin the
  snapshot; `--served-model-name qwen2.5-0.5b-instruct` is the id LewLM
  discovers. `HF_HUB_OFFLINE=1` after the first start prevents any re-download.
- **Bounded memory and concurrency.** `--max-model-len 8192`,
  `--gpu-memory-utilization 0.30`, `--max-num-seqs 4`,
  `--max-num-batched-tokens 4096`. vLLM's defaults (0.92 of VRAM, hundreds of
  sequences) assume a dedicated server; raise these only from measurements.
- **Key from the environment.** `VLLM_API_KEY` is read by the server at this
  commit (`--api-key` on the CLI would override it) so the key is not in `ps`.
  `/health` is outside the guard; `/v1/*` requires the key.
- **Tools.** `--enable-auto-tool-choice --tool-call-parser hermes` let a
  `tool_choice: auto` request be accepted. LewLM still gates tools per model
  by probing; the flags do not make the 0.5B model a good tool caller.
- **Caches, separate.** `HF_HOME=/cache/huggingface` (weights) and
  `VLLM_CACHE_ROOT=/cache/vllm` (compiled artifacts:
  `torch_compile_cache/<hash>/rank_0_0/…`) are two volumes. vLLM's
  `torch_compile` design doc says the compile cache directory can be carried
  between starts; keeping it lets a restart skip compilation, and either cache
  can be pruned without touching the other.
- **No telemetry.** `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`.
- **Honest usage.** `--enable-force-include-usage` puts usage on streamed
  replies so LewLM's terminal chunk carries real counts.

## 2. Wire LewLM and prove it

```bash
source examples/backends/vllm/lewlm.env.example      # VLLM_API_KEY must be exported first
lewlm scan && lewlm list-models                      # e.g. qwen2-5-0-5b-instruct-vllm-<hex>
lewlm serve --port 8080 &
python scripts/backend_acceptance.py --base-url http://127.0.0.1:8080 \
    --model <id> --endpoint-id vllm --output acceptance.json
python scripts/bridge_prefix_benchmark.py --lewlm-url http://127.0.0.1:8080 \
    --model <id> --direct-url http://127.0.0.1:8000/v1 --direct-model qwen2.5-0.5b-instruct \
    --direct-api-key-env VLLM_API_KEY --requests 12 --concurrency 1 --output prefix-c1.json
python scripts/bridge_prefix_benchmark.py --lewlm-url http://127.0.0.1:8080 \
    --model <id> --direct-url http://127.0.0.1:8000/v1 --direct-model qwen2.5-0.5b-instruct \
    --direct-api-key-env VLLM_API_KEY --requests 12 --concurrency 2 --output prefix-c2.json
```

Also record, exactly as the step-07 exit criteria ask:

| Measurement | How |
| --- | --- |
| Two requests overlap *upstream* | `docker compose logs vllm` during the acceptance `concurrency` case: both request ids should be running at once (`Running: 2 reqs` in the engine stats line, or enable `--enable-log-requests` for one run). The LewLM-side overlap alone is not proof |
| Cold start vs warm restart | time from `docker compose up` to `/health` = 200 on the first start (download + compile), then `docker compose restart vllm` with the caches mounted; note the `Using cache directory …torch_compile_cache…` log line and whether it says the graph was loaded from cache |
| Memory bound | `nvidia-smi --query-gpu=memory.used --format=csv` while serving; it must stay near `0.30 × total` plus activations, not the whole card |
| Parser behaviour | the acceptance `tools` case; if the 0.5B model emits malformed calls, record `inconclusive` and do not claim tools for it |
| Coexistence | `docker compose stop vllm`, `lewlm scan` (model kept as stale), route a llama.cpp and an Ollama model, then `start` and rescan |

To promote: fill the `vllm_local` recipe in `compatibility.json` with the
environment (Python inside the image, driver, toolkit, engine versions), the
image digest above as the `installation`, and the evidence path;
`tests/unit/test_backend_compatibility.py` checks the shape.

## Not covered by this recipe

- **ROCm** is a separate lane with its own kernels proof.
- **Windows / WSL2** (Docker Desktop) is covered: run LewLM natively on
  Windows against `127.0.0.1:8000`. vLLM keeps pinned memory off under a
  WSL2 kernel unless `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (set in the compose
  file; ignored elsewhere), and its default V2 model runner will not start
  without it (`RuntimeError: UVA is not available`). Cold start 191 s
  (compile + CUDA graphs), warm restart 25–27 s with the compile cache.
- **Seeded requests** carry a fresh `cache_salt`, so vLLM evaluates the whole
  prompt and the seed reproduces; with a prefix-cache hit it otherwise may not.
- **`vllm_mlx`** (Apple Silicon fork) is a different profile with different
  evidence; do not reuse this recipe or its results for it.
- **Multi-GPU / tensor parallel** — the recipe is single-GPU by design; TP
  needs `--tensor-parallel-size` and a larger `shm_size`.
