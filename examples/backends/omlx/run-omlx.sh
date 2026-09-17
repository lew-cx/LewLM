#!/usr/bin/env bash
# Start oMLX on loopback with the settings LewLM's step-05 validation used.
#
# Operator-run: LewLM never starts, stops, installs, or updates this server.
# Requires the isolated environment from README.md (OMLX_VENV) and a directory
# of MLX-format model folders (OMLX_MODEL_DIR). Every limit below is explicit
# so the server's memory footprint is a decision, not a default.
#
#   OMLX_API_KEY=... examples/backends/omlx/run-omlx.sh
#
# The same key must be exported for LewLM (see lewlm.env.example): the
# endpoint's `api_key_env` names the variable, never the value.

set -euo pipefail

: "${OMLX_VENV:=$HOME/.lewlm-engines/omlx/venv}"
: "${OMLX_MODEL_DIR:=$HOME/.lewlm-engines/omlx/models}"
: "${OMLX_HOME:=$HOME/.lewlm-engines/omlx/home}"     # oMLX writes ~/.omlx/settings.json; keep it out of your real home
: "${OMLX_CACHE_DIR:=$HOME/.lewlm-engines/omlx/cache}"
: "${OMLX_PORT:=8000}"
: "${OMLX_MEMORY_GUARD_GB:=16}"                        # process ceiling in GB; size it to the models you pin
: "${OMLX_MAX_CONCURRENT_REQUESTS:=4}"
: "${OMLX_HOT_CACHE_MAX_SIZE:=2GB}"
: "${OMLX_SSD_CACHE_MAX_SIZE:=2GB}"

if [ -z "${OMLX_API_KEY:-}" ]; then
    echo "OMLX_API_KEY is required so LewLM can authenticate to the endpoint" >&2
    exit 2
fi
mkdir -p "$OMLX_HOME" "$OMLX_CACHE_DIR"

exec env HOME="$OMLX_HOME" "$OMLX_VENV/bin/omlx" serve \
    --model-dir "$OMLX_MODEL_DIR" \
    --host 127.0.0.1 --port "$OMLX_PORT" \
    --api-key "$OMLX_API_KEY" \
    --memory-guard safe --memory-guard-gb "$OMLX_MEMORY_GUARD_GB" \
    --max-concurrent-requests "$OMLX_MAX_CONCURRENT_REQUESTS" \
    --hot-cache-max-size "$OMLX_HOT_CACHE_MAX_SIZE" \
    --paged-ssd-cache-dir "$OMLX_CACHE_DIR" --paged-ssd-cache-max-size "$OMLX_SSD_CACHE_MAX_SIZE" \
    --log-level info
