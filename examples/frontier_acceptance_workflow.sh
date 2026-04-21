#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-out/frontier-acceptance}"

: "${TEXT_MODEL:?set TEXT_MODEL to a runnable dense-text model id}"
: "${VLM_MODEL:?set VLM_MODEL to a runnable vision-capable model id}"

lewlm scan --json

lewlm benchmark --model "$TEXT_MODEL"
lewlm autotune --model "$TEXT_MODEL"

lewlm benchmark --model "$VLM_MODEL"
lewlm autotune --model "$VLM_MODEL"

python scripts/capture_release_bundle.py \
  --output-dir "$OUT_DIR" \
  --require-target Darwin:arm64 \
  --require-frontier-family dense_text \
  --require-frontier-family vlm \
  --require-frontier-family repeated_multimodal \
  --require-frontier-family speculative_family \
  --minimum-verified-models 1

python scripts/validate_release_candidate.py \
  "$OUT_DIR" \
  --require-target Darwin:arm64 \
  --require-frontier-family dense_text \
  --require-frontier-family vlm \
  --require-frontier-family repeated_multimodal \
  --require-frontier-family speculative_family \
  --minimum-verified-models 1

# Add these when you have host-backed evidence for them:
#   --require-frontier-family mixed_precision_conversion
#   --require-frontier-family frontier_architecture
#   --require-frontier-family distributed_multi_host
