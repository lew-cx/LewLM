# Installation

LewLM is packaged as a Python project. Start from a source checkout unless you specifically want to pin a Git reference from another project.

## Choose a path first

| Host | Path |
| --- | --- |
| macOS on Apple Silicon | native install with `.[mlx]` — MLX needs Metal, which containers cannot reach |
| Linux / Windows | **Docker** — see [Running LewLM in Docker](../operations/docker.md) |

On non-Apple hosts the container image is the promoted deployment. It already
contains a llama.cpp build compiled for your GPU (CUDA, via `Dockerfile.cuda`)
and llama.cpp's `convert_hf_to_gguf.py` plus `llama-quantize`, which
`lewlm convert` shells out to. The `llama-cpp-python` wheel ships **neither**
converter tool — it installs only the shared libraries — so a native install
satisfies the converter's Python dependencies while leaving the converter
itself absent, and `lewlm convert` reports `requires_install` until you supply
it.

The native instructions below remain fully supported; `lewlm doctor` reports
which of the two shapes it is running in and tailors its guidance accordingly.

## Base setup

```bash
git clone https://github.com/lew-cx/LewLM.git
cd LewLM
```

**macOS / Linux**

```bash
python3 -m venv .venv
. .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If you prefer not to activate the environment, use `.venv/bin/python -m pip ...` on macOS/Linux or `.venv\Scripts\python -m pip ...` on Windows for the install commands below.

## Install profiles

Choose the profile that matches what you want from the current environment:

| Profile | Install command | Best for | What it gives you | Notes |
| --- | --- | --- | --- | --- |
| Core only | `python -m pip install -e .` | registry, health, config, and contract-first integration work | CLI, local API, model registry, routing, readiness surfaces | No inference runtime or document packages |
| Apple MLX local backend | `python -m pip install -e ".[mlx]"` | Apple Silicon text, vision, and audio serving | MLX runtime adapters | First-class packaged runtime; requires macOS on arm64/aarch64 |
| Cross-platform GGUF backend | `python -m pip install -e ".[llamacpp]"` | packaged local serving on macOS, Linux, and Windows | llama.cpp runtime for LewLM's first-class non-Apple text path, embeddings, and packaged rerank fallback, plus the HF→GGUF conversion dependencies | LewLM's first-class non-Apple runtime family; non-Apple audio still uses the bridge path below |
| Cross-platform GGUF backend, serving only | `python -m pip install -e ".[llamacpp_runtime]"` | the same llama.cpp serving path when you never convert models | `llama-cpp-python` and nothing else — no Torch, Transformers, or tokenizer packages | Fastest GGUF install; the input for the lean `serving` container flavor. Add `.[gguf_conversion]` later if you need `lewlm convert` |
| ONNX Runtime GenAI backend | `python -m pip install -e ".[onnx_genai]"` | Windows-native prepared ONNX bundles, provider probing, and HF-to-ONNX conversion | ONNX GenAI bundle discovery plus CPU, DirectML, and CUDA provider metadata, and executable HF-to-ONNX conversion via the onnxruntime-genai model builder | Load/generate adapter for compatible ONNX bundles; HF-to-ONNX conversion is executable when this extra is installed |
| Cross-platform external accelerator bridge | `python -m pip install -e .` | LewLM in front of another local loopback server | base package plus the built-in bridge runtime pack | Requires `LEWLM_EXTERNAL_ACCELERATOR_ENABLED=true` and `LEWLM_EXTERNAL_ACCELERATOR_BASE_URL`; text and streaming use `/v1/chat/completions`, vision uses OpenAI-style image content blocks on that route, and audio requires compatible local `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints. LewLM does not install the external server, and this is the current bridge-only non-Apple audio parity path |
| Documents add-on | `python -m pip install -e ".[documents]"` | ingest, render, and transform workflows | PDF, DOCX, XLSX, OCR-oriented, and deterministic artifact packages | Additive; combine with a runtime profile when you also want inference |

**You need at least one runtime profile** such as Apple MLX, cross-platform GGUF, or the external accelerator bridge before expecting chat, vision, embeddings, rerank, or audio execution. The documents profile is separate on purpose so LewLM can stay modular for middleware hosts that do not need local parsing/rendering.

Common combinations:

- `python -m pip install -e ".[mlx,documents]"` for Apple Silicon plus document workflows
- `python -m pip install -e ".[llamacpp,documents]"` for cross-platform GGUF plus document workflows
- `python -m pip install -e ".[dev,documents]"` for repository development and tests

Operator guidance by host:

- **Apple Silicon macOS:** use `.[mlx]` or `.[mlx,documents]` when you want LewLM's first-class local runtime path.
- **Linux / Windows / non-MLX Macs:** use `.[llamacpp]` for the documented packaged runtime path. This is the first-class non-Apple runtime family LewLM now productizes. If you install `.[mlx]` there, `lewlm doctor` reports it as installed but host-blocked. Use `.[llamacpp_runtime]` when the host only serves GGUF models: it skips the multi-hundred-megabyte Torch/Transformers conversion stack entirely, and `.[llamacpp]` keeps its batteries-included meaning.
- **Native Windows llama.cpp:** PyPI ships `llama-cpp-python` only as source, which needs the Microsoft C++ Build Tools and, for 0.3.35, fails even earlier on a default Windows install because its vendored web UI exceeds the 260-character path limit. Install the prebuilt CPU wheel instead:
  `python -m pip install -e ".[llamacpp_runtime]" --only-binary llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`
  (validated on Windows 11 with the full acceptance suite, 2026-09-24). Prebuilt CUDA wheels (`/whl/cu124`, `/whl/cu125`, `/whl/cu130`) also need NVIDIA's cuBLAS DLL — `pip install "nvidia-cublas>=13,<14"` for cu130; LewLM puts pip-installed NVIDIA DLL folders on the Windows search path itself — and must match your CPU: the 0.3.35 cu130 Windows wheel is compiled with AVX-512 and cannot run on CPUs without it (most consumer Intel CPUs since Alder Lake). `scripts/verify_llamacpp_build.py` and `lewlm doctor` report that mismatch (`missing_cpu_features`). For NVIDIA GPUs on Windows the CUDA container image is the validated path.
- **Verify a llama.cpp install, not just the pip exit code:** `python scripts/verify_llamacpp_build.py --expect cpu` (or `--expect gpu --hint cuda` / `--hint vulkan` / `--hint metal`) asks the installed backend what it actually reports and fails when a prebuilt wheel or a source build did not produce the intended flavor. Exit code 2 means the native library does not load on this host (the Windows illegal-instruction case), which is the signal to switch to the container image.
- **Windows ONNX / DirectML bundles:** use `.[onnx_genai]` when you want LewLM to discover, load, and probe compatible ONNX GenAI bundles while reporting CPU, DirectML, and CUDA provider plans. With this extra installed, HF-to-ONNX conversion is also executable via the official `onnxruntime-genai` model builder (`lewlm convert <model-id> --target onnx_genai`): precision follows the conversion policy (`max_quality` → fp16, otherwise int4) and the execution provider follows `LEWLM_ONNX_GENAI_CONVERSION_EXECUTION_PROVIDER` (`cpu` default, or `dml`/`cuda`). GGUF/llama.cpp remains the first-class non-Apple default; ONNX is the Windows-native alternative.
- **Linux / Windows with another local server:** the external accelerator bridge is the intended path when you already run a loopback-only OpenAI-compatible server, including NVIDIA-backed local servers. LewLM does not bundle that server; text and streaming use `/v1/chat/completions`, image-conditioned chat requires a compatible server that accepts OpenAI-style image content blocks there, audio requires compatible local `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints that LewLM probes separately, and bridge-backed semantic routes still depend on compatible local `/v1/embeddings` and `/v1/rerank` support. This bridge is the current bridge-only non-Apple audio parity path. Treat it as an adapter-backed bridge path rather than a packaged-parity claim, and do not treat bridge benchmark wins as a replacement for the first-class packaged GGUF default, which now also covers compatible semantic GGUF models.
- **Documents profile:** `.[documents]` installs the Python-side packages only. OCR still depends on a working local engine such as `tesseract`.

## Recommended default feature routes

| Platform | Chat | Semantic text | Vision | Audio | Structured output |
| --- | --- | --- | --- | --- | --- |
| macOS | Apple MLX on Apple Silicon; GGUF on non-MLX Macs | Apple MLX on Apple Silicon; external bridge on non-MLX Macs | Apple MLX vision on Apple Silicon; external bridge on non-MLX Macs | Apple MLX audio on Apple Silicon; external bridge on non-MLX Macs | GGUF/llama.cpp when you need decode-time enforcement; MLX remains prompt-guided fallback |
| Linux | `.[llamacpp]` | `.[llamacpp]` for embedding-capable semantic GGUF models; bridge remains optional | external accelerator bridge with OpenAI-style image content blocks | external accelerator bridge with `/v1/audio/transcriptions` and `/v1/audio/speech`; bridge-only audio parity | `.[llamacpp]` |
| Windows | `.[llamacpp]`; `.[onnx_genai]` is the probe-gated DirectML-native candidate | `.[llamacpp]` for embedding-capable semantic GGUF models; bridge remains optional | external accelerator bridge with OpenAI-style image content blocks | external accelerator bridge with `/v1/audio/transcriptions` and `/v1/audio/speech`; bridge-only audio parity | `.[llamacpp]` |

`semantic text` covers embeddings and rerank, and `structured output` here means decode-time JSON-schema or grammar enforcement. On non-Apple hosts, LewLM keeps compatible semantic GGUF models on the packaged path and uses packaged embedding-similarity fallback for rerank when llama.cpp lacks a native rerank API. `lewlm doctor` and `GET /v1/health` expose the current-host view of this table under `install_profiles.recommended_feature_paths`.

## Reading support states

LewLM keeps install-profile guidance machine-readable so host apps and release validation can distinguish packaged, bridge-backed, fallback, unsupported, benchmark-backed, and host-probed states without inferring them from prose.

`install_profiles.standards_acceptance_contract` now adds the Milestone 120 acceptance-state legend (`lewlm_owned`, `backend_native`, `partial`, `fallback`, `unsupported`, `unverified`) plus the reserved 2026 vocabulary keys. Terms such as `kv_offload`, `responses_api_events`, `transformers_v5_ready`, and `local_agent_sandbox` live there now so later milestones can report them without inventing new field names.

| State | Meaning in install-profile and runtime docs | Machine-readable signal |
| --- | --- | --- |
| Packaged | LewLM ships or imports the runtime path directly on the current host or target host | `install_profiles.recommended_feature_paths[].support_path = "packaged"` |
| Bridge-backed | LewLM keeps the public contract but delegates execution to a loopback-only local adapter runtime | `install_profiles.recommended_feature_paths[].support_path = "bridge"` |
| Fallback | LewLM accepts the request shape but reports a narrower execution path or conversion-guided route | target-platform `readiness_state = "fallback_guided"`; structured-output `fallback_used`; performance-core evidence `mode = "fallback"` |
| Unsupported | the current host, target host, or runtime path is outside the documented product promise | `supported = false` plus readiness such as `blocked`, `unsupported`, or `runtime_unavailable` |
| Benchmark-backed | LewLM has persisted host/model/runtime evidence for the documented default or performance behavior | `benchmark_backed = true` or `benchmark_backed_defaults = true` |
| Host-probed | the current host has been checked directly instead of inheriting a declared target-platform contract | `verification_method = "host_probe"` |

| Acceptance state | Meaning in the 2026 vocabulary contract | Machine-readable signal |
| --- | --- | --- |
| `lewlm_owned` | LewLM implements and validates the named behavior directly | `install_profiles.standards_acceptance_contract.acceptance_states[]` |
| `backend_native` | the backend owns the behavior and LewLM only detects or preserves it | `install_profiles.standards_acceptance_contract.acceptance_states[]` |
| `partial` | only part of the behavior is preserved or observable | `install_profiles.standards_acceptance_contract.acceptance_states[]` |
| `fallback` | LewLM keeps the request contract but downgrades execution | `install_profiles.standards_acceptance_contract.acceptance_states[]` |
| `unsupported` | LewLM does not claim the named behavior on that path | `install_profiles.standards_acceptance_contract.acceptance_states[]` |
| `unverified` | the term is reserved but not yet backed by host proof or a stronger probe | `install_profiles.standards_acceptance_contract.acceptance_states[]` |

See the [Runtime and capability matrix](../reference/runtime-capability-matrix.md) for the per-feature acceptance matrix that uses this vocabulary.

## Git install

If you want another project to install a pinned LewLM version directly from Git:

```bash
python -m pip install "lewlm @ git+https://github.com/lew-cx/LewLM.git@<tag-or-commit>"
```

Profile-oriented examples:

```bash
python -m pip install "lewlm @ git+https://github.com/lew-cx/LewLM.git@<tag-or-commit>"
python -m pip install "lewlm[llamacpp] @ git+https://github.com/lew-cx/LewLM.git@<tag-or-commit>"
python -m pip install "lewlm[mlx,documents] @ git+https://github.com/lew-cx/LewLM.git@<tag-or-commit>"
```

## Confirm the active profile

After install, use either surface below to confirm what the current host can actually use:

```bash
lewlm doctor
curl http://127.0.0.1:8080/v1/health
```

Both now include an `install_profiles` summary with the active profile ids, per-profile readiness notes, and the current-host `recommended_feature_paths` guidance.

`lewlm doctor` also prints the detected host memory when the probe succeeds. If it cannot determine total memory on the current host, it reports that explicitly so routing/readiness output stays honest.

## Install behavior and local state

Installing LewLM does **not** overwrite `~/.lewlm`. LewLM creates missing directories on first use and reuses the existing state for the current OS user unless you override `LEWLM_DATA_DIR`. On Windows, the default location is `%USERPROFILE%\.lewlm`.

## Next step

Run the [Quickstart](quickstart.md) flow that matches the profile you selected.
