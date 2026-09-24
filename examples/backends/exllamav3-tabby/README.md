# ExLlamaV3 via TabbyAPI behind LewLM (Linux + NVIDIA)

Status: **passed on Windows 11 + Docker Desktop (WSL2)** on 2026-09-24 —
RTX 5090 Laptop (SM 12.0, driver 610.47), LewLM native on Windows, the EXL3
artifact below: the common acceptance suite (10 passed; tools inconclusive
because the 0.5B model answered in text). TabbyAPI has no `seed` at this pin,
so LewLM reports it unsupported and forwards `top_k`/`min_p`/
`repetition_penalty`, which it does implement. See the
[Windows/Linux validation record](../../../docs/validation/modernization-windows-linux.md).
Bare-metal Linux/NVIDIA remains **deferred** in
`examples/backends/compatibility.json`.

| Input | Pin |
| --- | --- |
| TabbyAPI | commit `53da7919d4e45c63f4acbcbbc00cbe0f60a1ce65`; image `ghcr.io/theroyallab/tabbyapi@sha256:10bfcf9d27d1b3a5c7ada786f814b9da43fadad35643362f2fe0816082893172` (built from that commit, label `org.opencontainers.image.revision`; CUDA 12.8.1; amd64 manifest `sha256:c4d24298…`) |
| ExLlamaV3 | commit `02aef45cd681b960a00afcd0749a4ab99e6c1bfe` (used by the image; needed on the host only to convert a model) |
| Requirements | NVIDIA driver for CUDA 12.8, NVIDIA Container Toolkit, one GPU. TabbyAPI is a rolling release: never re-pin to a moving tag |
| Candidate model | `Qwen/Qwen2.5-0.5B-Instruct` BF16 @ `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0), converted offline to EXL3 (below). Produced 2026-09-24 at 4.0 bpw with the image's own ExLlamaV3 1.5.0+cu128 (142 s on an RTX 5090 Laptop): `model.safetensors` sha256 `6cb53f9992cb1f21823b1e739038553b1c4af8f86194817b44806676450a32b8` |
| LewLM profile | `exllamav3_tabby`, endpoint id `tabby` |

## 1. Model: EXL3 is produced offline, never by LewLM

TabbyAPI serves ExLlamaV3, which loads EXL3-quantized models (and can load
unquantized HF safetensors). It does **not** load GGUF, and LewLM never hands
an EXL3 artifact to llama.cpp. Conversion is an explicit operator step with
ExLlamaV3's converter on a CUDA host (a couple of minutes for a 0.5B model):

```bash
# On the CUDA host, in a venv with a CUDA 12.8 torch (>= 2.6):
git clone https://github.com/turboderp-org/exllamav3.git && cd exllamav3
git checkout 02aef45cd681b960a00afcd0749a4ab99e6c1bfe
pip install torch --index-url https://download.pytorch.org/whl/cu128
# Prefer a release wheel matching your torch/CUDA/Python from
# https://github.com/turboderp-org/exllamav3/releases ; otherwise the PyPI/source
# install compiles the extension at first import (JIT, minutes, once per torch
# version). Bound that with MAX_JOBS=4 and keep the torch extension cache
# (~/.cache/torch_extensions) on a persistent volume so a restart does not pay it again.
MAX_JOBS=4 pip install --no-build-isolation .
python convert.py -i /models/Qwen2.5-0.5B-Instruct -o /models/Qwen2.5-0.5B-Instruct-exl3-4.0bpw -w /tmp/exl3-work -b 4.0
sha256sum /models/Qwen2.5-0.5B-Instruct-exl3-4.0bpw/*.safetensors   # record these before promotion
```

JIT compilation moved from install time to first request is a cold-start
cost, not a speed-up; the recipe uses the published image (extension already
built) for serving, and the note above only applies to the conversion venv.

Simpler, and the way the 2026-09-24 artifact was made: convert with the
pinned serving image itself, so the converter is exactly the ExLlamaV3 that
will load the result. It works from Windows too (Docker Desktop); the work and
output directories stay inside the container and only the finished artifact
is copied to the bind mount:

```bash
docker run --rm --gpus all --shm-size 8g -v "$PWD/models:/models" --entrypoint sh \
  ghcr.io/theroyallab/tabbyapi@sha256:10bfcf9d27d1b3a5c7ada786f814b9da43fadad35643362f2fe0816082893172 -c '
  python3 -m exllamav3.conversion.convert_model -i /models/Qwen2.5-0.5B-Instruct -o /tmp/out -w /tmp/work -b 4.0 &&
  mkdir -p /models/Qwen2.5-0.5B-Instruct-exl3-4.0bpw && cp -r /tmp/out/. /models/Qwen2.5-0.5B-Instruct-exl3-4.0bpw/'
```

## 2. Keys

Edit `api_tokens.yml` with two distinct random values. `api_key` is the
inference key LewLM uses; `admin_key` loads and unloads models and stays out
of LewLM, Chap, and every client. LewLM routes to the model the server
already advertises and never asks TabbyAPI to load one.

## 3. Serve on host loopback

```bash
cd examples/backends/exllamav3-tabby
TABBY_MODELS_DIR=/models docker compose up -d
curl -s -H "Authorization: Bearer $TABBY_API_KEY" http://127.0.0.1:5000/v1/models
```

`config.yml` loads exactly one model at startup (`model_name`), keeps
`inline_model_loading: false`, disables API-triggered downloads, bounds
context/cache to 8192 tokens, sets `max_batch_size: 4`, and keeps
`cache_mode: FP16`. `docker-compose.yml` publishes port 5000 to host loopback
only, sets `shm_size: 8g` (ExLlamaV3 uses POSIX shared memory; the Docker
default is too small), and pins the image by digest. LewLM must run on the
host, not in another container: `127.0.0.1` is the endpoint rule.

## 4. Wire LewLM and prove it

```bash
source examples/backends/exllamav3-tabby/lewlm.env.example    # set TABBY_API_KEY first
lewlm scan && lewlm list-models       # e.g. qwen2-5-0-5b-instruct-exl3-4-0bpw-tabby-<hex>
lewlm serve --port 8080 &
python scripts/backend_acceptance.py --base-url http://127.0.0.1:8080 \
    --model <id> --endpoint-id tabby --output acceptance.json
TABBY_API_KEY=... python scripts/bridge_prefix_benchmark.py --lewlm-url http://127.0.0.1:8080 \
    --model <id> --direct-url http://127.0.0.1:5000/v1 --direct-model Qwen2.5-0.5B-Instruct-exl3-4.0bpw \
    --direct-api-key-env TABBY_API_KEY --requests 12
```

Also record: first start (extension load + model load) versus a cached
restart, `nvidia-smi` memory while serving, and what `/v1/models` says about
the model so the manifest's `format_type` (currently `unknown` unless the
record names a format) can be made `exl3` from evidence. Then stop the
container, `lewlm scan`, and confirm llama.cpp and Ollama models still route.

To promote: fill the `exllamav3_tabby` recipe in `compatibility.json` with
the environment (Python, driver, toolkit, engine versions), the image digest
above as the `installation`, the EXL3 artifact hashes, and the evidence path;
`tests/unit/test_backend_compatibility.py` checks the shape.

## Native Windows

The engine running natively on Windows is not covered by this recipe; a
container pass does not certify it (ExLlamaV3 on Windows needs
`triton-windows` and its own install/generation/cancellation proof). The
container on Docker Desktop with LewLM native on Windows **is** covered — see
the status above. Keep your real keys out of the tracked `api_tokens.yml`: put
them in a file outside the repository and mount it over `/app/api_tokens.yml`
with a compose override.
