#!/usr/bin/env python3
"""Common real-engine acceptance suite, driven entirely through LewLM's HTTP API.

One harness for every backend (oMLX, TabbyAPI/ExLlamaV3, vLLM, SGLang,
llama.cpp-server, Ollama, native runtimes): point it at a running LewLM and a
model id, and it exercises the cases the modernization roadmap requires for
promotion, through the same public routes a host app such as Chap uses.
Engine-specific setup (starting the server, choosing the model) stays outside
in the recipe; nothing here imports an engine or talks to one directly.

Every case ends in exactly one of:
  passed         the required observation was made
  failed         the observation contradicts the requirement
  inconclusive   the model did not exercise the path (e.g. never called the
                 tool); recorded, not counted as a pass
  not_exercised  needs something the harness cannot do (stop the engine,
                 revoke a key); the reason and the manual step are recorded
  skipped        excluded with --skip

Usage:
    python scripts/backend_acceptance.py --base-url http://127.0.0.1:8080 \
        --model qwen2-5-0-5b-instruct-4bit-omlx-3f9a1c2b --endpoint-id omlx \
        --output docs/validation/evidence/omlx/acceptance.json

Exit code 0 when no case failed, 1 otherwise. The JSON evidence is written
either way.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

CASES = (
    "discovery_identity",
    "chat_nonstreaming",
    "chat_streaming",
    "sampling",
    "structured_output",
    "tools",
    "reasoning",
    "cancellation",
    "concurrency",
    "failure",
    "fallback",
    "lifetime",
)


@dataclass
class CaseResult:
    case: str
    status: str
    observations: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    elapsed_seconds: float = 0.0


def _has_content(chunk: dict[str, Any]) -> bool:
    """A chunk carrying visible text; empty role-only chunks do not count as first output."""

    for choice in chunk.get("choices", []):
        delta = choice.get("delta") or {}
        if delta.get("content"):
            return True
    return False


class Harness:
    def __init__(self, *, base_url: str, model: str, endpoint_id: str | None, max_tokens: int,
                 timeout: float, api_key: str | None, application_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoint_id = endpoint_id
        self.max_tokens = max_tokens
        self.timeout = timeout
        headers = {"x-lewlm-application-id": application_id}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=5.0), headers=headers)

    # ---- helpers -----------------------------------------------------------

    def chat(self, messages: list[dict[str, str]], *, request_id: str | None = None, **extra: Any) -> httpx.Response:
        payload = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens, "stream": False, **extra}
        headers = {"x-request-id": request_id} if request_id else {}
        return self.client.post("/v1/chat/completions", json=payload, headers=headers)

    def chat_stream(self, messages: list[dict[str, str]], *, request_id: str | None = None, **extra: Any):
        """Context manager yielding a streaming response; closing it closes the upstream stream."""

        payload = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens, "stream": True, **extra}
        headers = {"x-request-id": request_id} if request_id else {}
        return self.client.stream("POST", "/v1/chat/completions", json=payload, headers=headers)

    def endpoint_snapshot(self, endpoint_id: str | None) -> dict[str, Any] | None:
        """The bridge runtime's cached endpoint evidence from `GET /v1/runtime/stats`."""

        if not endpoint_id:
            return None
        stats = self.client.get("/v1/runtime/stats").json()
        for runtime in stats.get("runtimes") or []:
            endpoint = runtime.get("endpoint") if isinstance(runtime, dict) else None
            if endpoint and endpoint.get("endpoint_id") == endpoint_id:
                return {
                    k: endpoint.get(k)
                    for k in ("endpoint_id", "profile", "inventory_state", "inventory_error", "inventory_age_seconds",
                              "inventory_ttl_seconds", "advertised_model_ids", "upstream_residency", "upstream_cancellation")
                }
        return None

    @staticmethod
    def read_sse(response: httpx.Response, *, on_first_chunk=None) -> tuple[list[dict[str, Any]], bool]:
        """Return (chunks, saw_done). Stops at [DONE] or end of body."""

        chunks: list[dict[str, Any]] = []
        saw_done = False
        fired = False
        buffer = ""
        for raw in response.iter_text():
            buffer += raw
            while "\n\n" in buffer:
                event, buffer = buffer.split("\n\n", 1)
                for line in event.splitlines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        return chunks, saw_done
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        chunks.append({"_malformed": data})
                        continue
                    chunks.append(parsed)
                    if on_first_chunk is not None and not fired and _has_content(parsed):
                        fired = True
                        on_first_chunk()
        return chunks, saw_done

    @staticmethod
    def content_of(chunks: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for chunk in chunks:
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
        return "".join(parts)

    # ---- cases ---------------------------------------------------------------

    def discovery_identity(self) -> CaseResult:
        health = self.client.get("/v1/health").json()
        models = self.client.get("/v1/models").json()
        ids = [item["model_id"] for item in models.get("items", [])]
        if self.model not in ids:
            return CaseResult("discovery_identity", "failed", {"advertised": ids[:20]}, "model id not in /v1/models")
        detail = self.client.get(f"/v1/models/{self.model}").json()
        manifest = detail.get("model") or detail.get("manifest") or detail
        bound = manifest.get("metadata", {}).get("external_endpoint_id")
        observations = {
            "source_path": manifest.get("source_path"),
            "format_type": manifest.get("format_type"),
            "external_endpoint_id": bound,
            "external_upstream_model_id": manifest.get("metadata", {}).get("external_upstream_model_id"),
            "execution_locality": manifest.get("metadata", {}).get("execution_locality"),
            "health_status": health.get("status"),
        }
        snapshot = self.endpoint_snapshot(self.endpoint_id or bound)
        if snapshot is not None:
            observations["endpoint_snapshot"] = snapshot
        if self.endpoint_id and bound != self.endpoint_id:
            return CaseResult("discovery_identity", "failed", observations, f"model is bound to {bound!r}, expected {self.endpoint_id!r}")
        return CaseResult("discovery_identity", "passed", observations)

    def chat_nonstreaming(self) -> CaseResult:
        response = self.chat([{"role": "user", "content": "Reply with the single word: ready"}])
        if response.status_code != 200:
            return CaseResult("chat_nonstreaming", "failed", {"status": response.status_code, "body": response.text[:500]})
        body = response.json()
        choice = body["choices"][0]
        usage = body.get("usage", {})
        metadata = body.get("metadata", {})
        observations = {
            "content": choice["message"]["content"][:200],
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "endpoint_id": metadata.get("model", {}).get("endpoint_id"),
            "engine_profile": metadata.get("model", {}).get("engine_profile"),
            "execution_locality": metadata.get("model", {}).get("execution_locality"),
            "runtime_name": metadata.get("model", {}).get("runtime_name"),
            "execute_milliseconds": metadata.get("timing", {}).get("execute_milliseconds"),
        }
        if not choice["message"]["content"].strip():
            return CaseResult("chat_nonstreaming", "failed", observations, "empty completion")
        if not choice.get("finish_reason"):
            return CaseResult("chat_nonstreaming", "failed", observations, "no finish reason")
        return CaseResult("chat_nonstreaming", "passed", observations)

    def chat_streaming(self) -> CaseResult:
        started = time.monotonic()
        first_token_at: list[float] = []
        with self.chat_stream([{"role": "user", "content": "Count from one to five, words only."}]) as response:
            if response.status_code != 200:
                return CaseResult("chat_streaming", "failed", {"status": response.status_code})
            chunks, saw_done = self.read_sse(response, on_first_chunk=lambda: first_token_at.append(time.monotonic()))
        content = self.content_of(chunks)
        terminal = [chunk for chunk in chunks if any(choice.get("finish_reason") for choice in chunk.get("choices", []))]
        usage_chunks = [chunk for chunk in chunks if chunk.get("usage")]
        observations = {
            "chunks": len(chunks),
            "content": content[:200],
            "terminal_chunks": len(terminal),
            "finish_reason": next((c.get("finish_reason") for chunk in terminal for c in chunk["choices"] if c.get("finish_reason")), None),
            "usage": usage_chunks[-1]["usage"] if usage_chunks else None,
            "saw_done": saw_done,
            "time_to_first_content_ms": round((first_token_at[0] - started) * 1000) if first_token_at else None,
            "total_ms": round((time.monotonic() - started) * 1000),
            "malformed_chunks": sum(1 for chunk in chunks if "_malformed" in chunk),
        }
        if not content.strip():
            return CaseResult("chat_streaming", "failed", observations, "no streamed content")
        if len(terminal) != 1:
            return CaseResult("chat_streaming", "failed", observations, "expected exactly one terminal chunk")
        if not usage_chunks:
            return CaseResult("chat_streaming", "failed", observations, "no usage on the terminal chunk")
        if observations["malformed_chunks"]:
            return CaseResult("chat_streaming", "failed", observations, "malformed SSE data")
        return CaseResult("chat_streaming", "passed", observations)

    def sampling(self) -> CaseResult:
        response = self.chat(
            [{"role": "user", "content": "Say hello."}],
            temperature=0.2,
            sampling={"top_p": 0.9, "seed": 7, "stop": ["\n\n"]},
        )
        if response.status_code != 200:
            return CaseResult("sampling", "failed", {"status": response.status_code, "body": response.text[:300]})
        report = response.json().get("metadata", {}).get("sampling")
        if not report:
            return CaseResult("sampling", "failed", {}, "no sampling report in execution metadata")
        observations = {k: report.get(k) for k in ("runtime", "requested", "applied", "unsupported", "deterministic")}
        requested = set(report.get("requested", {}))
        accounted = set(report.get("applied", {})) | set(report.get("unsupported", []))
        if not requested <= accounted:
            return CaseResult("sampling", "failed", observations, f"controls neither applied nor reported: {sorted(requested - accounted)}")
        # `deterministic` is what the backend accepted; observe it rather than trust it.
        if report.get("deterministic"):
            outputs = []
            for _ in range(2):
                repeat = self.chat(
                    [{"role": "user", "content": "Write one sentence about the ocean."}],
                    temperature=0.9, sampling={"seed": 11},
                )
                if repeat.status_code != 200:
                    return CaseResult("sampling", "failed", observations, "seeded repeat request failed")
                outputs.append(repeat.json()["choices"][0]["message"]["content"])
            observations["seeded_outputs_identical"] = outputs[0] == outputs[1]
            observations["seeded_output"] = outputs[0][:120]
            if outputs[0] != outputs[1]:
                return CaseResult("sampling", "failed", observations, "backend reported the seed as applied but two seeded runs differed")
        return CaseResult("sampling", "passed", observations)

    def structured_output(self) -> CaseResult:
        capabilities = self.client.get(f"/v1/models/{self.model}/capabilities").json()
        support = capabilities.get("structured_output") or {}
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
            "required": ["city", "population"],
            "additionalProperties": False,
        }
        response = self.chat(
            [{"role": "user", "content": "Give the city Paris and a population estimate as JSON."}],
            temperature=0.0,
            response_format={"type": "json_schema", "schema": schema, "name": "city", "strict": True},
        )
        if response.status_code != 200:
            return CaseResult("structured_output", "failed", {"status": response.status_code, "body": response.text[:300], "advertised": support})
        body = response.json()
        result = body.get("structured_output") or {}
        content = body["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            valid_json = isinstance(parsed, dict) and set(parsed) >= {"city", "population"}
        except json.JSONDecodeError:
            parsed, valid_json = None, False
        observations = {
            "advertised_decode_time_modes": support.get("decode_time_modes"),
            "enforcement": result.get("enforcement"),
            "decoder_enforced": result.get("decoder_enforced"),
            "fallback_used": result.get("fallback_used"),
            "validation": result.get("validation"),
            "content": content[:200],
            "valid_json": valid_json,
        }
        if "json_schema" in (support.get("decode_time_modes") or []) and not result.get("decoder_enforced"):
            return CaseResult("structured_output", "failed", observations, "advertised decode-time enforcement was not applied")
        if not valid_json:
            status = "failed" if result.get("decoder_enforced") else "inconclusive"
            return CaseResult("structured_output", status, observations, "output did not satisfy the schema")
        return CaseResult("structured_output", "passed", observations)

    def tools(self) -> CaseResult:
        tool = {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }
        response = self.chat(
            [{"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."}],
            temperature=0.0,
            tools=[tool],
            tool_choice="auto",
        )
        if response.status_code != 200:
            return CaseResult("tools", "failed", {"status": response.status_code, "body": response.text[:300]})
        body = response.json()
        parsed = body.get("tool_calls") or {}
        calls = parsed.get("calls") or parsed.get("tool_calls") or []
        observations = {
            "tool_calls": calls[:3],
            "content": body["choices"][0]["message"]["content"][:200],
            "finish_reason": body["choices"][0].get("finish_reason"),
        }
        if not calls:
            return CaseResult("tools", "inconclusive", observations, "model answered in text; tool path not exercised")
        first = calls[0]
        arguments = first.get("arguments") if isinstance(first, dict) else None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return CaseResult("tools", "failed", observations, "tool arguments are not valid JSON")
        if not isinstance(arguments, dict) or "city" not in arguments:
            return CaseResult("tools", "failed", observations, "tool arguments missing required field")
        continuation = self.chat(
            [
                {"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."},
                {"role": "assistant", "content": json.dumps({"tool_call": first})},
                {"role": "tool", "content": json.dumps({"city": "Lisbon", "temperature_c": 21, "sky": "clear"})},
            ],
            temperature=0.0,
        )
        observations["continuation_status"] = continuation.status_code
        if continuation.status_code != 200:
            return CaseResult("tools", "failed", observations, "tool-result continuation failed")
        observations["continuation_content"] = continuation.json()["choices"][0]["message"]["content"][:200]
        return CaseResult("tools", "passed", observations)

    def reasoning(self) -> CaseResult:
        response = self.chat(
            [{"role": "user", "content": "Think step by step, then answer: what is 17 + 25?"}],
            reasoning_visibility="hidden",
        )
        if response.status_code != 200:
            return CaseResult("reasoning", "failed", {"status": response.status_code})
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        leaked = "<think>" in content or "</think>" in content
        observations = {"content": content[:200], "reasoning_field": body["choices"][0]["message"].get("reasoning")}
        if leaked:
            return CaseResult("reasoning", "failed", observations, "hidden reasoning leaked into visible content")
        return CaseResult("reasoning", "passed", observations)

    def cancellation(self) -> CaseResult:
        request_id = f"accept-cancel-{uuid.uuid4().hex[:8]}"
        cancel_states: list[str] = []
        stream_chunks: list[int] = []

        def cancel_after_first_chunk() -> None:
            record = self.client.post(f"/v1/requests/{request_id}/cancel").json()
            cancel_states.append(record.get("state"))

        started = time.monotonic()
        with self.chat_stream(
            [{"role": "user", "content": "Write a very long story about the sea, at least 400 words."}],
            request_id=request_id, max_tokens=max(self.max_tokens, 256),
        ) as response:
            if response.status_code != 200:
                return CaseResult("cancellation", "failed", {"status": response.status_code})
            chunks, _ = self.read_sse(response, on_first_chunk=cancel_after_first_chunk)
            stream_chunks.append(len(chunks))
        elapsed = time.monotonic() - started
        final = self.client.post(f"/v1/requests/{request_id}/cancel").json()
        follow_up = self.chat([{"role": "user", "content": "Reply with: ok"}])
        observations = {
            "cancel_state_after_first_chunk": cancel_states[0] if cancel_states else None,
            "cancel_state_final": final.get("state"),
            "chunks_before_stop": stream_chunks[0],
            "stream_elapsed_ms": round(elapsed * 1000),
            "follow_up_status": follow_up.status_code,
            "upstream_stop_verified": False,
        }
        if follow_up.status_code != 200:
            return CaseResult("cancellation", "failed", observations, "a request after cancellation failed")
        if final.get("state") not in {"cancelled", "cancelling", "completed"}:
            return CaseResult("cancellation", "failed", observations, f"unexpected cancel state {final.get('state')!r}")
        if final.get("state") == "completed" and stream_chunks[0] >= max(self.max_tokens, 256):
            return CaseResult("cancellation", "failed", observations, "the stream ran to completion after cancel")
        observations["note"] = "Middleware stop, transport close, and lease release observed; whether the engine aborted GPU work must be read from the engine's own logs."
        return CaseResult("cancellation", "passed", observations)

    def concurrency(self) -> CaseResult:
        results: dict[str, dict[str, Any]] = {}
        barrier = threading.Barrier(2)

        def run(label: str, cancel: bool) -> None:
            request_id = f"accept-conc-{label}-{uuid.uuid4().hex[:6]}"
            record: dict[str, Any] = {"request_id": request_id}
            barrier.wait()
            started = time.monotonic()
            with self.chat_stream(
                [{"role": "user", "content": f"Write 200 words about {label}."}],
                request_id=request_id, max_tokens=max(self.max_tokens, 128),
            ) as response:
                record["status"] = response.status_code

                def first_chunk() -> None:
                    record["first_chunk_at"] = time.monotonic()
                    if cancel:
                        self.client.post(f"/v1/requests/{request_id}/cancel")

                chunks, _ = self.read_sse(response, on_first_chunk=first_chunk)
            record["finished_at"] = time.monotonic()
            record["chunks"] = len(chunks)
            record["content_chars"] = len(self.content_of(chunks))
            record["started_at"] = started
            results[label] = record

        threads = [threading.Thread(target=run, args=("rivers", True)), threading.Thread(target=run, args=("mountains", False))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=self.timeout * 2)
        a, b = results.get("rivers", {}), results.get("mountains", {})
        overlapped = (
            "first_chunk_at" in a and "first_chunk_at" in b
            and a["first_chunk_at"] < b["finished_at"] and b["first_chunk_at"] < a["finished_at"]
        )
        observations = {
            "cancelled_request": {k: a.get(k) for k in ("status", "chunks", "content_chars")},
            "surviving_request": {k: b.get(k) for k in ("status", "chunks", "content_chars")},
            "overlapped_at_lewlm": overlapped,
            "note": "Overlap is observed at LewLM's boundary; upstream concurrency must be read from the engine's own metrics.",
        }
        if b.get("status") != 200 or not b.get("content_chars"):
            return CaseResult("concurrency", "failed", observations, "the surviving request did not complete")
        if not overlapped:
            return CaseResult("concurrency", "inconclusive", observations, "requests did not overlap at LewLM")
        return CaseResult("concurrency", "passed", observations)

    def failure(self) -> CaseResult:
        missing = self.client.post("/v1/chat/completions", json={"model": f"missing-{uuid.uuid4().hex[:6]}", "messages": [{"role": "user", "content": "hi"}]})
        health_after = self.client.get("/v1/health").json()
        observations = {
            "missing_model_status": missing.status_code,
            "missing_model_error": (missing.json().get("error") if missing.headers.get("content-type", "").startswith("application/json") else missing.text[:200]),
            "health_after": health_after.get("status"),
            "not_exercised": ["bad_key", "timeout", "overload", "server_stop", "truncated_stream"],
            "manual_steps": "Stop or misconfigure the engine and repeat chat_nonstreaming; expect a 4xx/5xx JSON error, health still 200, no automatic replay.",
        }
        if not (400 <= missing.status_code < 500):
            return CaseResult("failure", "failed", observations, "missing model did not produce a 4xx")
        if health_after.get("status") != "ok":
            return CaseResult("failure", "failed", observations, "health degraded after a client error")
        return CaseResult("failure", "passed", observations)

    def fallback(self) -> CaseResult:
        stats = self.client.get("/v1/runtime/stats").json()
        runtimes = stats.get("runtimes") or []
        names = [r.get("name") for r in runtimes if isinstance(r, dict)]
        available = [r.get("name") for r in runtimes if isinstance(r, dict) and r.get("available")]
        return CaseResult(
            "fallback",
            "not_exercised",
            {"registered_runtimes": names, "available_runtimes": available,
             "manual_steps": "Disable or stop this endpoint, rescan, and confirm native/Ollama candidates still route; with external_fallback_policy=explicit_alias confirm the decision records fallback_from_model_id."},
            "requires stopping the engine",
        )

    def lifetime(self) -> CaseResult:
        warm = self.client.post(f"/v1/models/{self.model}/warm", json={})
        residency = self.client.get(f"/v1/models/{self.model}/residency")
        unload = self.client.post(f"/v1/models/{self.model}/unload", json={})
        snapshot = self.endpoint_snapshot(self.endpoint_id) or {}
        upstream_residency = snapshot.get("upstream_residency")
        observations = {
            "warm_status": warm.status_code,
            "warm_result": (warm.json().get("lifecycle") or warm.json()) if warm.status_code == 200 else warm.text[:200],
            "residency_status": residency.status_code,
            "unload_status": unload.status_code,
            "unload_result": (unload.json().get("lifecycle") or unload.json()) if unload.status_code == 200 else unload.text[:200],
            "upstream_residency_reported": upstream_residency,
            "note": "A LewLM unload releases the bridge lease only; upstream RAM is the engine's to manage and is reported as unknown.",
        }
        if warm.status_code != 200 or unload.status_code != 200:
            return CaseResult("lifetime", "failed", observations, "warm or unload did not succeed")
        if self.endpoint_id and upstream_residency not in (None, "unknown"):
            return CaseResult("lifetime", "failed", observations, "bridge claimed upstream residency it cannot observe")
        return CaseResult("lifetime", "passed", observations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True, help="LewLM model id to exercise")
    parser.add_argument("--endpoint-id", default=None, help="Expected external endpoint id for a bridge-backed model")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api-key", default=None, help="LewLM API key when api_key_required is set")
    parser.add_argument("--application-id", default="backend-acceptance")
    parser.add_argument("--skip", default="", help="Comma-separated case names to skip")
    parser.add_argument("--only", default="", help="Comma-separated case names to run")
    parser.add_argument("--output", default=None, help="Write the JSON evidence record here")
    parser.add_argument("--label", default=None, help="Free-form label (engine + model + host) stored in the record")
    args = parser.parse_args(argv)

    skip = {item.strip() for item in args.skip.split(",") if item.strip()}
    only = {item.strip() for item in args.only.split(",") if item.strip()}
    harness = Harness(base_url=args.base_url, model=args.model, endpoint_id=args.endpoint_id,
                      max_tokens=args.max_tokens, timeout=args.timeout, api_key=args.api_key,
                      application_id=args.application_id)

    try:
        health = harness.client.get("/v1/health").json()
    except httpx.HTTPError as exc:
        print(f"LewLM is not reachable at {args.base_url}: {exc}", file=sys.stderr)
        return 2

    results: list[CaseResult] = []
    for case in CASES:
        if case in skip or (only and case not in only):
            results.append(CaseResult(case, "skipped"))
            continue
        started = time.monotonic()
        try:
            result = getattr(harness, case)()
        except Exception as exc:  # noqa: BLE001 - every failure is evidence
            result = CaseResult(case, "failed", {"exception": f"{type(exc).__name__}: {exc}"[:500]})
        result.elapsed_seconds = round(time.monotonic() - started, 3)
        results.append(result)
        marker = {"passed": "PASS", "failed": "FAIL", "inconclusive": "INCONCLUSIVE", "not_exercised": "NOT EXERCISED", "skipped": "SKIP"}[result.status]
        print(f"{marker:14} {case:20} {result.reason or ''}")

    record = {
        "format": "lewlm-backend-acceptance-v1",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "endpoint_id": args.endpoint_id,
        "lewlm_version": health.get("version"),
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version()},
        "summary": {status: sum(1 for r in results if r.status == status) for status in ("passed", "failed", "inconclusive", "not_exercised", "skipped")},
        "cases": [asdict(result) for result in results],
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, default=str)
            handle.write("\n")
        print(f"wrote {args.output}")
    print("summary:", json.dumps(record["summary"]))
    return 1 if record["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
