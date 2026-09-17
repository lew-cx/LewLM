# oMLX behind LewLM (Apple Silicon)

Validated configuration (modernization step 05, 2026-09-17):

| Input | Value |
| --- | --- |
| Engine | [jundot/omlx](https://github.com/jundot/omlx) @ `b45fb7e5a127355c0769d0b7c828849db17492e7` (`omlx 0.7.0.dev3`) |
| Runtime stack | `mlx 0.32.2`, `mlx-lm 0.31.3 @ ab1806e`, `transformers 5.17.0`, `llguidance 1.8.0` — full resolution in [requirements.lock.txt](requirements.lock.txt) |
| Python | 3.11.15 (oMLX requires 3.11–3.13; LewLM's own environment can be any supported version) |
| Host | Apple M2 Max, 96 GiB unified memory, macOS Darwin 25.2.0 (oMLX requires macOS 15+) |
| Model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, hashes in [`compatibility.json`](../compatibility.json) |
| LewLM model id produced | `qwen2-5-0-5b-instruct-4bit-omlx-6fed8f48` (`external://omlx/Qwen2.5-0.5B-Instruct-4bit`) |
| Evidence | [`docs/validation/evidence/modernization-step-05/`](../../../docs/validation/evidence/modernization-step-05/) |

What passed: text chat and streaming, sampling (`top_p`/`seed`/`stop` applied, seeded
determinism observed), `json_schema` structured output forwarded natively and validated after
generation, hidden reasoning, cancellation, two concurrent streams, error mapping, lifecycle lease
semantics, engine stop/restart with stale inventory, and mid-stream engine loss (terminal error
event, no replay). Tool calling was **inconclusive** on this 0.5B model; vision, embeddings, and
rerank were **not probed**. This recipe claims exactly the table above, not oMLX in general.

## 1. Install oMLX in its own environment

oMLX is not on PyPI and pins its own `mlx`. Keep it out of LewLM's environment.
If you already run oMLX (menu-bar app, Homebrew service), skip this section and
use that server; do not install a second copy.

```bash
mkdir -p ~/.lewlm-engines/omlx && cd ~/.lewlm-engines/omlx
git clone https://github.com/jundot/omlx.git src
git -C src checkout b45fb7e5a127355c0769d0b7c828849db17492e7
python3.11 -m venv venv && venv/bin/pip install --upgrade pip
venv/bin/pip install ./src            # ~2 minutes on an M2 Max, no Xcode needed
venv/bin/python -c "import omlx, mlx.core as mx; print('omlx ok, mlx', mx.__version__)"
```

Native custom kernels (`OMLX_WITH_CUSTOM_KERNEL=1`) need full Xcode and only
matter for the model families oMLX lists; the validated run did not build them.

## 2. Point it at MLX models

`--model-dir` takes a directory of MLX-format model folders. A Hugging Face
cache snapshot can be linked in place:

```bash
mkdir -p ~/.lewlm-engines/omlx/models
ln -s ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
      ~/.lewlm-engines/omlx/models/Qwen2.5-0.5B-Instruct-4bit
```

## 3. Run it on loopback with explicit limits

```bash
OMLX_API_KEY=$(openssl rand -hex 16) examples/backends/omlx/run-omlx.sh
```

[`run-omlx.sh`](run-omlx.sh) binds `127.0.0.1` only, requires the API key, and
sets the memory guard, concurrency cap, hot-cache size, and SSD-cache directory
and size explicitly. `HOME` is redirected so oMLX's `~/.omlx/settings.json`
lands under the engine directory rather than your real home. Check it:

```bash
curl -s -H "Authorization: Bearer $OMLX_API_KEY" http://127.0.0.1:8000/v1/models
```

## 4. Wire LewLM

```bash
source examples/backends/omlx/lewlm.env.example   # edit OMLX_API_KEY first
lewlm scan            # registers one model per advertised id, e.g. qwen2-5-0-5b-instruct-4bit-omlx-6fed8f48
lewlm list-models
lewlm chat --model qwen2-5-0-5b-instruct-4bit-omlx-6fed8f48 --prompt "Hello"
lewlm doctor          # install_profiles.external_endpoints shows the endpoint
```

The model's `format_type` is `unknown` on purpose — `/v1/models` does not
reveal weights — and its `execution_locality` is `loopback_unverified`.
Native MLX, llama.cpp, and Ollama models keep routing exactly as before; the
oMLX model is one more selectable id.

## 5. Prove it, then read the evidence

```bash
lewlm serve --port 8080 &
python scripts/backend_acceptance.py --base-url http://127.0.0.1:8080 \
    --model qwen2-5-0-5b-instruct-4bit-omlx-6fed8f48 --endpoint-id omlx \
    --output acceptance.json
OMLX_API_KEY=... python scripts/bridge_prefix_benchmark.py --lewlm-url http://127.0.0.1:8080 \
    --model qwen2-5-0-5b-instruct-4bit-omlx-6fed8f48 \
    --direct-url http://127.0.0.1:8000/v1 --direct-model Qwen2.5-0.5B-Instruct-4bit \
    --direct-api-key-env OMLX_API_KEY --requests 12 --output prefix.json
```

The acceptance harness runs the roadmap's common suite over LewLM's public
API and records `passed` / `failed` / `inconclusive` / `not_exercised` per
case. To exercise the engine-control lanes by hand: stop oMLX, `lewlm scan`
(the model stays, marked stale), send a chat (a 503 naming `endpoint_id`
`omlx`), start oMLX, `lewlm scan` again (`advertised`). Killing oMLX during a
stream ends that stream with a terminal chunk whose `finish_reason` is
`error` and `error.partial_output` is `true`; LewLM does not replay it.

## What LewLM does and does not own here

- oMLX owns model residency, its hot/SSD KV cache, batching, and eviction.
  `POST /v1/models/{id}/unload` releases LewLM's bridge lease only and says so;
  `upstream_residency` is reported as `unknown`.
- LewLM's response cache never applies to chat, so repeated-prefix speedups
  in the benchmark are oMLX's prefix cache. Hit counters are not exposed
  through the bridge and are reported as unknown.
- Structured output is forwarded as a native `response_format`; LewLM reports
  `enforcement: decode_time` with `enforcement_evidence: upstream_native` and
  validates the output after generation instead of claiming to have
  constrained oMLX's decoder.
