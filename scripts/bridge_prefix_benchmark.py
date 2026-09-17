#!/usr/bin/env python3
"""Repeated-prefix latency through LewLM versus direct to the upstream server.

Sends the same conversation -- one long shared prefix followed by a short,
changing final turn -- first through LewLM's `/v1/chat/completions` and then
straight to the engine's OpenAI-compatible endpoint, streaming both, and
reports time-to-first-chunk and total latency per request. The first request
is reported separately from the warm ones so an engine's prefix cache (which
LewLM neither owns nor can count) shows up as a first-vs-rest difference
rather than being claimed as a hit counter.

LewLM's response cache never applies to chat, so nothing here needs
disabling on the LewLM side; run it with the engine's own cache in its
normal configuration and say so in the evidence.

Usage:
    python scripts/bridge_prefix_benchmark.py --lewlm-url http://127.0.0.1:8080 \
        --model qwen2-5-0-5b-instruct-4bit-omlx-6fed8f48 \
        --direct-url http://127.0.0.1:8000/v1 --direct-model Qwen2.5-0.5B-Instruct-4bit \
        --direct-api-key-env OMLX_API_KEY --requests 10 --warmups 3 --prefix-tokens 600 \
        --output docs/validation/evidence/<step>/prefix-benchmark.json

This is an observation harness, not the roadmap's full benchmark protocol
(30+ requests at several concurrencies); use --requests/--concurrency to
scale it up on a quiet machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

_PREFIX_SENTENCE = (
    "The committee reviewed the quarterly logistics report, noting that warehouse throughput rose while "
    "transit delays fell, and recommended continued investment in route planning tools. "
)


def _prefix(tokens: int) -> str:
    # ~28 tokens per sentence for typical BPE vocabularies; over-approximate slightly.
    repeats = max(1, tokens // 24)
    return _PREFIX_SENTENCE * repeats


def _messages(prefix: str, index: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prefix},
        {"role": "assistant", "content": "Understood. I have read the report."},
        {"role": "user", "content": f"Question {index}: In one sentence, what did the committee recommend?"},
    ]


def _stream_once(client: httpx.Client, path: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    first: float | None = None
    chunks = 0
    status = None
    with client.stream("POST", path, json=payload, headers=headers) as response:
        status = response.status_code
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunks += 1
            # Time to first *content*: servers often send an empty role-only
            # chunk immediately, which says nothing about prefill.
            if first is None:
                try:
                    delta = json.loads(data)["choices"][0].get("delta") or {}
                except (ValueError, KeyError, IndexError, TypeError):
                    delta = {}
                if delta.get("content"):
                    first = time.perf_counter()
    finished = time.perf_counter()
    return {
        "status": status,
        "ttfc_ms": round((first - started) * 1000, 1) if first is not None else None,
        "total_ms": round((finished - started) * 1000, 1),
        "chunks": chunks,
    }


def _run_series(*, client: httpx.Client, path: str, model: str, prefix: str, requests: int, warmups: int,
                max_tokens: int, headers: dict[str, str], concurrency: int, extra: dict[str, Any]) -> dict[str, Any]:
    def one(index: int) -> dict[str, Any]:
        payload = {"model": model, "messages": _messages(prefix, index), "max_tokens": max_tokens,
                   "temperature": 0.0, "stream": True, **extra}
        return _stream_once(client, path, payload, headers)

    cold = one(0)
    warm_runs: list[dict[str, Any]] = [one(index) for index in range(1, warmups + 1)]
    measured: list[dict[str, Any]]
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            measured = list(pool.map(one, range(100, 100 + requests)))
    else:
        measured = [one(index) for index in range(100, 100 + requests)]
    ok = [run for run in measured if run["status"] == 200 and run["ttfc_ms"] is not None]
    ttfc = sorted(run["ttfc_ms"] for run in ok)
    total = sorted(run["total_ms"] for run in ok)

    def pct(values: list[float], p: float) -> float | None:
        if not values:
            return None
        k = max(0, min(len(values) - 1, round((p / 100) * (len(values) - 1))))
        return values[k]

    return {
        "cold_first_request": cold,
        "warmups": warm_runs,
        "measured_count": len(measured),
        "succeeded": len(ok),
        "ttfc_ms": {"p50": pct(ttfc, 50), "p95": pct(ttfc, 95), "mean": round(statistics.fmean(ttfc), 1) if ttfc else None},
        "total_ms": {"p50": pct(total, 50), "p95": pct(total, 95), "mean": round(statistics.fmean(total), 1) if total else None},
        "runs": measured,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lewlm-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True, help="LewLM model id")
    parser.add_argument("--lewlm-api-key", default=None)
    parser.add_argument("--direct-url", default=None, help="Upstream OpenAI-compatible base URL ending in /v1 (optional)")
    parser.add_argument("--direct-model", default=None, help="Upstream model id (defaults to --model)")
    parser.add_argument("--direct-api-key-env", default=None, help="Env var holding the upstream API key")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prefix-tokens", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    prefix = _prefix(args.prefix_tokens)
    record: dict[str, Any] = {
        "format": "lewlm-bridge-prefix-benchmark-v1",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "settings": {"requests": args.requests, "warmups": args.warmups, "concurrency": args.concurrency,
                     "prefix_tokens_approx": args.prefix_tokens, "prefix_chars": len(prefix), "max_tokens": args.max_tokens,
                     "lewlm_response_cache": "not applicable to chat", "engine_prefix_cache_hits": "unknown (not exposed through the bridge)"},
    }

    lewlm_headers = {"Authorization": f"Bearer {args.lewlm_api_key}"} if args.lewlm_api_key else {}
    with httpx.Client(base_url=args.lewlm_url, timeout=args.timeout) as client:
        record["via_lewlm"] = _run_series(
            client=client, path="/v1/chat/completions", model=args.model, prefix=prefix, requests=args.requests,
            warmups=args.warmups, max_tokens=args.max_tokens, headers=lewlm_headers, concurrency=args.concurrency, extra={},
        )
    if args.direct_url:
        direct_headers: dict[str, str] = {}
        if args.direct_api_key_env and os.environ.get(args.direct_api_key_env):
            direct_headers["Authorization"] = f"Bearer {os.environ[args.direct_api_key_env]}"
        with httpx.Client(base_url=args.direct_url.rstrip("/"), timeout=args.timeout) as client:
            record["direct"] = _run_series(
                client=client, path="/chat/completions", model=args.direct_model or args.model, prefix=prefix,
                requests=args.requests, warmups=args.warmups, max_tokens=args.max_tokens, headers=direct_headers,
                concurrency=args.concurrency, extra={"stream_options": {"include_usage": True}},
            )
        via, direct = record["via_lewlm"]["ttfc_ms"], record["direct"]["ttfc_ms"]
        if via["p95"] is not None and direct["p95"] is not None:
            record["middleware_overhead"] = {
                "ttfc_p95_delta_ms": round(via["p95"] - direct["p95"], 1),
                "ttfc_p50_delta_ms": round(via["p50"] - direct["p50"], 1),
                "roadmap_target_ms": round(max(20.0, 0.10 * direct["p95"]), 1),
                "within_target": (via["p95"] - direct["p95"]) <= max(20.0, 0.10 * direct["p95"]),
            }

    summary = {k: v for k, v in record.items() if k not in ("via_lewlm", "direct")}
    print(json.dumps({**summary, "via_lewlm": {k: v for k, v in record["via_lewlm"].items() if k != "runs"},
                      **({"direct": {k: v for k, v in record["direct"].items() if k != "runs"}} if "direct" in record else {})},
                     indent=2, default=str))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, default=str)
            handle.write("\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
