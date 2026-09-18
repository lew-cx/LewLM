# SGLang behind LewLM (Linux + NVIDIA)

Status: **deferred** in `examples/backends/compatibility.json`. Every pin
below was verified against upstream at the pinned revisions on 2026-09-18,
but no Linux/NVIDIA host was available to run it, so nothing here is labelled
validated. The recipe is complete enough to run and prove; do that before
promoting it.

| Input | Pin |
| --- | --- |
| SGLang | release `v0.5.19` = commit `0bcd822377da7b5718e674eaf9c870d349424dd1`; image `lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9` (Docker Hub tag `v0.5.19-cu130`, same digest as `v0.5.19`; amd64 image `sha256:37bbbd34…`; config labels `ai.sglang.build.commit` = the commit, `SGLANG_IMAGE_TAG=lmsysorg/sglang:v0.5.19`) |
| Inside the image | CUDA `13.0.3`, cuDNN 9.14, `torch==2.13.0`, `sglang-kernel==0.4.6.post1` (prebuilt `cp310-abi3` wheel from PyPI for CUDA 13 — `docker/Dockerfile` at the commit), entrypoint `/opt/nvidia/nvidia_entrypoint.sh` |
| Host requirements | NVIDIA driver **R580 or newer** (CUDA 13), NVIDIA Container Toolkit, one GPU with ≥ 4 GiB free. SGLang does not publish a compiled-architecture list for its kernel wheel at this pin, so the preflight *skips* that check; kernel support is confirmed on first start, not assumed. A `-cu129` image variant exists for CUDA 12 hosts but is a different digest and is not this recipe |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` BF16 @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0; step-00 pin, `model.safetensors` sha256 in the manifest), served as `qwen2.5-0.5b-instruct` |
| Tool parser | `qwen25` — SGLang's `docs/docs/advanced_features/tool_parser.mdx` at the commit names it for Qwen2.5; it is in `FunctionCallParser.ToolCallParserEnum` at the commit |
| Grammar backend | `xgrammar` — one of `GRAMMAR_BACKEND_CHOICES` at the commit (`xgrammar`, `outlines`, `llguidance`, `none`) |
| LewLM profile | `sglang_local`, endpoint id `sglang` |

## 0. Preflight — before pulling anything

```bash
python scripts/engine_preflight.py --recipe sglang --output preflight.json
```

It checks the digest pin, Docker + the `nvidia` container runtime, driver
≥ R580, free VRAM, and that `127.0.0.1:30000` is free. It pulls, installs,
and starts nothing. The compute-capability check is reported as `skip` for
this recipe (see above); keep `preflight.json` with the evidence.

## 1. Serve on host loopback

```bash
cd examples/backends/sglang
export SGLANG_API_KEY="$(openssl rand -hex 24)"   # inference key; LewLM sends the same value
docker compose up -d
docker compose logs -f sglang                     # first start downloads the model, then compiles kernels
curl -s -H "Authorization: Bearer $SGLANG_API_KEY" http://127.0.0.1:30000/v1/models
```

What the compose file fixes, and why:

- **Loopback only.** Port 30000 is published to `127.0.0.1`; LewLM must run
  on the host, not in another container (`127.0.0.1` is the endpoint rule).
- **One model, immutable.** `--revision` pins the snapshot;
  `--served-model-name qwen2.5-0.5b-instruct` is the id LewLM discovers, and
  SGLang's `/v1/models` record carries `max_model_len`, which LewLM reads as
  the context length. `HF_HUB_OFFLINE=1` after the first start prevents any
  re-download.
- **Bounded memory and concurrency.** `--context-length 8192`,
  `--mem-fraction-static 0.30` (share of VRAM for weights + KV pool; the
  heuristic default takes most of the card), `--max-running-requests 4`.
  `--chunked-prefill-size` and the CUDA-graph batch sizes are left at
  SGLang's defaults: the roadmap tunes them only from measured profiles
  (step 09). `--enable-torch-compile` is not set; upstream's own
  server-arguments page at this commit marks it out of maintenance.
- **Key on the command line.** SGLang reads the inference key only from
  `--api-key` at this commit, so it is visible in `ps` inside the container
  namespace. Its middleware requires `Authorization: Bearer <key>` on every
  route except `/health*` and `/metrics*`. `--admin-api-key` (management
  endpoints) is deliberately not configured; LewLM never manages the server.
- **Tools and structured output.** `--tool-call-parser qwen25` lets a tool
  request be parsed upstream; `--grammar-backend xgrammar` names the
  structured-output backend explicitly. LewLM still gates both per model by
  probing; the flags do not make the 0.5B model a good tool caller.
- **Caches, separate.** `HF_HOME=/cache/huggingface` (weights) and
  `/root/.cache` (whatever the pinned stack writes: FlashInfer JIT kernels,
  and inductor artifacts only if torch.compile were ever enabled) are two
  volumes. The recipe does not prescribe cache paths it has not observed:
  list `/root/.cache` after the first real start and record the directories
  and their sizes in the validation record.

## 2. Wire LewLM and prove it

```bash
source examples/backends/sglang/lewlm.env.example    # SGLANG_API_KEY must be exported first
lewlm scan && lewlm list-models                      # e.g. qwen2-5-0-5b-instruct-sglang-<hex>
lewlm serve --port 8080 &
python scripts/backend_acceptance.py --base-url http://127.0.0.1:8080 \
    --model <id> --endpoint-id sglang --output acceptance.json
python scripts/bridge_prefix_benchmark.py --lewlm-url http://127.0.0.1:8080 \
    --model <id> --direct-url http://127.0.0.1:30000/v1 --direct-model qwen2.5-0.5b-instruct \
    --direct-api-key-env SGLANG_API_KEY --requests 12 --concurrency 1 --output prefix-c1.json
python scripts/bridge_prefix_benchmark.py --lewlm-url http://127.0.0.1:8080 \
    --model <id> --direct-url http://127.0.0.1:30000/v1 --direct-model qwen2.5-0.5b-instruct \
    --direct-api-key-env SGLANG_API_KEY --requests 12 --concurrency 2 --output prefix-c2.json
```

Also record, exactly as the step-08 exit criteria ask:

| Measurement | How |
| --- | --- |
| Repeated-prefix requests | the two `bridge_prefix_benchmark.py` runs above, with LewLM response caching untouched (it never applies to chat); SGLang's radix cache is on by default — report the speed-up as an observation, not a hit counter, unless `--enable-cache-report` is turned on for a separate run and its `usage.prompt_tokens_details` cached-token count is captured |
| Two streams overlap *upstream* | `docker compose logs sglang` during the acceptance `concurrency` case: the `Decode batch. #running-req: 2` line. The LewLM-side overlap alone is not proof |
| Memory bound | `nvidia-smi --query-gpu=memory.used --format=csv` while serving; also the `available_gpu_mem=… GB` line SGLang logs just before ready |
| Cold start vs warm restart | time from `docker compose up` to `/health` = 200 on the first start (download + kernel JIT), then `docker compose restart sglang` with both caches mounted; list `/root/.cache` and note which directories were reused |
| Coexistence | `docker compose stop sglang`, `lewlm scan` (model kept as stale), route a llama.cpp and an Ollama model, then `start` and rescan |

To promote: fill the `sglang_local` recipe in `compatibility.json` with the
environment (Python inside the image, driver, toolkit, engine versions), the
image digest above as the `installation`, and the evidence path;
`tests/unit/test_backend_compatibility.py` checks the shape.

## Not covered by this recipe

- **CUDA 12 hosts** — use the `-cu129` image only as a separately pinned and
  separately validated recipe.
- **Other accelerators** (ROCm, Ascend, Xeon variants on Docker Hub) — each
  is its own lane with its own image digest and proof.
- **Multi-GPU / tensor parallel** — the recipe is single-GPU by design.
