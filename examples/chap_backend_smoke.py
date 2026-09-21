#!/usr/bin/env python3
"""Chap backend smoke: prove LewLM's public HTTP contract before any UI exists.

Everything goes over HTTP against one base URL — exactly what Chap will do.
Two modes:

    python examples/chap_backend_smoke.py --fixture                 # no model, no engine: LewLM + a fake engine, in-process
    python examples/chap_backend_smoke.py --base-url http://127.0.0.1:8080 --model <id> [--endpoint-id vllm]

Fixture mode runs on any OS and is what CI uses. Real-model mode runs the
same checks against a server you already started (oMLX, vLLM, SGLang,
TabbyAPI, llama.cpp, Ollama ...); the engine-outage and mid-stream-death
checks only run in fixture mode because they need to kill the engine.

Each check records what it observed; a failure is structured, never a
traceback. Exit status 1 when any check failed. Write the JSON report with
--output and keep it with the validation record.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import httpx

_ROOT = Path(__file__).resolve().parents[1]


def _load_acceptance_harness():
    spec = spec_from_file_location("lewlm_backend_acceptance", _ROOT / "scripts" / "backend_acceptance.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Check:
    name: str
    status: str  # passed | failed | inconclusive | skipped
    observations: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    elapsed_ms: int = 0


class ChapSmoke:
    """Chap-shaped checks layered on the common acceptance harness."""

    def __init__(self, *, base_url: str, model: str | None, endpoint_id: str | None, api_key: str | None, timeout: float, fixture=None) -> None:
        self.acceptance = _load_acceptance_harness()
        self.fixture = fixture
        self.endpoint_id = endpoint_id
        headers = {"x-lewlm-application-id": "chap"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout, connect=5.0), headers=headers)
        self.model = model or self._pick_model()
        self.harness = self.acceptance.Harness(base_url=base_url, model=self.model, endpoint_id=endpoint_id, max_tokens=48,
                                               timeout=timeout, api_key=api_key, application_id="chap")

    def _pick_model(self) -> str:
        models = self.client.get("/v1/models").json()
        for availability in models.get("capability_availability") or []:
            if availability.get("chat_ready"):
                return availability["model_id"]
        raise SystemExit("no chat-ready model advertised; pass --model")

    # ---- Chap-specific checks --------------------------------------------------

    def service_health(self) -> Check:
        response = self.client.get("/v1/health")
        body = response.json()
        engines = body.get("engines") or []
        observations = {
            "status": body.get("status"),
            "readiness": (body.get("readiness") or {}).get("status"),
            "engines": [{k: engine.get(k) for k in ("endpoint_id", "profile", "enabled", "state", "advertised_model_count")} for engine in engines],
            "x_request_id_echoed": bool(response.headers.get("x-request-id")),
        }
        if response.status_code != 200 or body.get("status") != "ok":
            return Check("service_health", "failed", observations, "health is not ok")
        if "engines" not in body:
            return Check("service_health", "failed", observations, "health does not separate engine state from service state")
        return Check("service_health", "passed", observations)

    def runtime_startup(self) -> Check:
        body = self.client.get("/v1/runtime").json()
        startup = body.get("startup") or {}
        observations = {
            "lewlm_ready_seconds": startup.get("lewlm_ready_seconds"),
            "engines": [{k: engine.get(k) for k in ("endpoint_id", "state", "first_advertised_at")} for engine in startup.get("engines") or []],
            "warm_models": [item.get("model_id") for item in startup.get("warm_models") or []],
            "loading_models": [item.get("model_id") for item in startup.get("loading_models") or []],
        }
        if startup.get("lewlm_ready_at") is None:
            return Check("runtime_startup", "failed", observations, "no lewlm_ready_at on /v1/runtime.startup")
        return Check("runtime_startup", "passed", observations)

    def model_picker(self) -> Check:
        models = self.client.get("/v1/models").json()
        availability = next((item for item in models.get("capability_availability") or [] if item.get("model_id") == self.model), None)
        capabilities = self.client.get(f"/v1/models/{self.model}/capabilities").json()
        structured = capabilities.get("structured_output") or {}
        observations = {
            "count": models.get("count"),
            "chat_ready_count": models.get("chat_ready_count"),
            "availability": availability,
            "structured_output_prediction": {k: structured.get(k) for k in ("decode_time_modes", "prompt_guided_modes")} if isinstance(structured, dict) else structured,
        }
        if availability is None:
            return Check("model_picker", "failed", observations, "selected model is missing from capability_availability")
        for key in ("endpoint_id", "engine_profile", "engine_state", "reason"):
            if key not in availability:
                return Check("model_picker", "failed", observations, f"capability_availability lacks `{key}`")
        if self.endpoint_id and availability.get("endpoint_id") != self.endpoint_id:
            return Check("model_picker", "failed", observations, f"model is bound to {availability.get('endpoint_id')!r}, expected {self.endpoint_id!r}")
        return Check("model_picker", "passed", observations)

    def request_identity(self) -> Check:
        request_id = f"chap-{uuid.uuid4().hex[:10]}"
        correlation_id = f"conv-{uuid.uuid4().hex[:8]}"
        response = self.client.post(
            "/v1/chat/completions",
            json={"model": self.model, "messages": [{"role": "user", "content": "Reply with: ok"}], "max_tokens": 8},
            headers={"x-request-id": request_id, "x-lewlm-correlation-id": correlation_id},
        )
        metadata = response.json().get("metadata", {}) if response.status_code == 200 else {}
        observations = {
            "status": response.status_code,
            "x_request_id": response.headers.get("x-request-id"),
            "x_lewlm_correlation_id": response.headers.get("x-lewlm-correlation-id"),
            "metadata_request_id": metadata.get("request_id"),
            "metadata_correlation_id": metadata.get("correlation_id"),
        }
        if response.status_code != 200:
            return Check("request_identity", "failed", observations, "chat failed")
        if response.headers.get("x-request-id") != request_id or metadata.get("request_id") != request_id:
            return Check("request_identity", "failed", observations, "x-request-id was not preserved")
        if response.headers.get("x-lewlm-correlation-id") != correlation_id or metadata.get("correlation_id") != correlation_id:
            return Check("request_identity", "failed", observations, "x-lewlm-correlation-id was not echoed in the header and metadata")
        return Check("request_identity", "passed", observations)

    def responses_finish_reason(self) -> Check:
        """Both surfaces say why generation stopped, so truncation is visible on either."""

        vocabulary = {"stop", "length", "tool_calls"}
        finished = self.client.post("/v1/responses", json={"model": self.model, "input": "Reply with: ok", "max_output_tokens": 16})
        truncated = self.client.post("/v1/responses", json={"model": self.model, "input": "Write a long story.", "max_output_tokens": 4 if self.fixture is None else 256})
        chat = self.harness.chat([{"role": "user", "content": "Reply with: ok"}])
        finished_body = finished.json() if finished.status_code == 200 else {}
        truncated_body = truncated.json() if truncated.status_code == 200 else {}
        chat_body = chat.json() if chat.status_code == 200 else {}
        observations = {
            "finished_status": finished.status_code,
            "finished_finish_reason": finished_body.get("finish_reason"),
            "truncated_status": truncated.status_code,
            "truncated_finish_reason": truncated_body.get("finish_reason"),
            "chat_finish_reason": ((chat_body.get("choices") or [{}])[0]).get("finish_reason"),
        }
        if finished.status_code != 200 or truncated.status_code != 200 or chat.status_code != 200:
            return Check("responses_finish_reason", "failed", observations, "a responses or chat request failed")
        for label, value in (("finished", observations["finished_finish_reason"]), ("truncated", observations["truncated_finish_reason"])):
            if value not in vocabulary:
                return Check("responses_finish_reason", "failed", observations, f"the {label} reply's finish_reason is not in {sorted(vocabulary)}")
        if observations["chat_finish_reason"] not in vocabulary:
            return Check("responses_finish_reason", "failed", observations, "the chat surface's finish_reason is not in the published vocabulary")
        if self.fixture is not None and observations["truncated_finish_reason"] != "length":
            return Check("responses_finish_reason", "failed", observations, "the fixture's truncated reply must report `length`")
        return Check("responses_finish_reason", "passed", observations)

    def unavailable_engine(self) -> Check:
        if self.fixture is None:
            return Check("unavailable_engine", "skipped", {}, "needs fixture mode to stop the engine")
        with self.fixture.engine_stopped():
            self.client.post("/v1/models/scan", json={})
            response = self.harness.chat([{"role": "user", "content": "hi"}])
            health = self.client.get("/v1/health").json()
            body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            error = body.get("error") or {}
            observations = {
                "status": response.status_code,
                "error_code": error.get("code"),
                "error_endpoint_id": (error.get("details") or {}).get("endpoint_id"),
                "health_status": health.get("status"),
                "engine_state_in_health": next((e.get("state") for e in health.get("engines") or [] if e.get("endpoint_id") == self.fixture.endpoint_id), None),
                "model_still_listed": any(item.get("model_id") == self.model for item in self.client.get("/v1/models").json().get("items") or []),
            }
        self.client.post("/v1/models/scan", json={})
        recovered = self.harness.chat([{"role": "user", "content": "hi"}])
        observations["recovered_status"] = recovered.status_code
        if response.status_code != 503 or error.get("code") != "runtime_unavailable":
            return Check("unavailable_engine", "failed", observations, "an engine outage must be a 503 runtime_unavailable")
        if observations["error_endpoint_id"] != self.fixture.endpoint_id:
            return Check("unavailable_engine", "failed", observations, "the error does not name the endpoint")
        if health.get("status") != "ok" or observations["engine_state_in_health"] not in {"failed", "stale"}:
            return Check("unavailable_engine", "failed", observations, "health must stay ok while reporting the engine as failed/stale")
        if not observations["model_still_listed"]:
            return Check("unavailable_engine", "failed", observations, "an unreachable engine is not evidence its model was deleted")
        if recovered.status_code != 200:
            return Check("unavailable_engine", "failed", observations, "chat did not recover after the engine returned and a rescan")
        return Check("unavailable_engine", "passed", observations)

    def stream_interrupted(self) -> Check:
        if self.fixture is None:
            return Check("stream_interrupted", "skipped", {}, "needs fixture mode to kill the engine mid-stream")
        engine = self.fixture.engine
        chunks: list[dict[str, Any]] = []

        def kill_after_first() -> None:
            engine.die_mid_stream.set()

        try:
            with self.harness.chat_stream([{"role": "user", "content": "Write a long story."}], max_tokens=256) as response:
                status = response.status_code
                chunks, saw_done = self.harness.read_sse(response, on_first_chunk=kill_after_first)
        finally:
            engine.die_mid_stream.clear()
        terminal = next((chunk for chunk in reversed(chunks) if any(choice.get("finish_reason") for choice in chunk.get("choices", []))), None)
        error = (terminal or {}).get("error") or {}
        observations = {
            "status": status,
            "chunks": len(chunks),
            "saw_done": saw_done,
            "terminal_finish_reason": next((c.get("finish_reason") for c in (terminal or {}).get("choices", []) if c.get("finish_reason")), None),
            "error_code": error.get("code"),
            "partial_output": error.get("partial_output"),
        }
        follow_up = self.harness.chat([{"role": "user", "content": "hi"}])
        observations["follow_up_status"] = follow_up.status_code
        if status != 200 or observations["terminal_finish_reason"] != "error" or not saw_done:
            return Check("stream_interrupted", "failed", observations, "an interrupted stream must end with a finish_reason=error chunk then [DONE]")
        if not error.get("code") or error.get("partial_output") is not True:
            return Check("stream_interrupted", "failed", observations, "the terminal error envelope must carry a code and partial_output")
        if follow_up.status_code != 200:
            return Check("stream_interrupted", "failed", observations, "a new request after the interruption failed")
        return Check("stream_interrupted", "passed", observations)

    # ---- runner ----------------------------------------------------------------

    CHAP_CHECKS = ("service_health", "runtime_startup", "model_picker", "request_identity", "responses_finish_reason")
    HARNESS_CASES = ("chat_streaming", "chat_nonstreaming", "structured_output", "tools", "reasoning", "cancellation", "failure")
    FIXTURE_ONLY = ("unavailable_engine", "stream_interrupted")

    def run(self, *, only: set[str]) -> list[Check]:
        results: list[Check] = []
        for name in (*self.CHAP_CHECKS, *self.HARNESS_CASES, *self.FIXTURE_ONLY):
            if only and name not in only:
                continue
            started = time.monotonic()
            try:
                if name in self.HARNESS_CASES:
                    case = getattr(self.harness, name)()
                    check = Check(name, case.status, case.observations, case.reason)
                else:
                    check = getattr(self, name)()
            except Exception as exc:  # noqa: BLE001 - every failure is evidence
                check = Check(name, "failed", {"exception": f"{type(exc).__name__}: {exc}"[:500]})
            check.elapsed_ms = round((time.monotonic() - started) * 1000)
            results.append(check)
            marker = {"passed": "PASS", "failed": "FAIL", "inconclusive": "INCONCLUSIVE", "not_exercised": "NOT EXERCISED", "skipped": "SKIP"}.get(check.status, check.status.upper())
            print(f"{marker:14} {name:20} {check.reason or ''}")
        return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=None, help="A running LewLM server")
    parser.add_argument("--fixture", action="store_true", help="Start LewLM + a fake engine in-process and test that")
    parser.add_argument("--model", default=None, help="LewLM model id; default: first chat-ready model")
    parser.add_argument("--endpoint-id", default=None, help="Expected endpoint id for the model (fixture: `fixture`)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--only", default="", help="Comma-separated check names")
    parser.add_argument("--output", default=None, help="Write the JSON report here")
    parser.add_argument("--stream-delay-ms", type=int, default=20, help="Fixture engine: delay between streamed chunks")
    args = parser.parse_args(argv)
    if bool(args.base_url) == bool(args.fixture):
        parser.error("pass exactly one of --base-url or --fixture")

    fixture = None
    if args.fixture:
        sys.path.insert(0, str(_ROOT / "src"))
        from lewlm.testing import FakeBackendFixture, FakeOpenAIEngine

        fixture = FakeBackendFixture(engine=FakeOpenAIEngine(stream_delay_seconds=args.stream_delay_ms / 1000.0), api_key=args.api_key)
        fixture.start()
        base_url = fixture.base_url
        endpoint_id = args.endpoint_id or fixture.endpoint_id
    else:
        base_url = args.base_url
        endpoint_id = args.endpoint_id
    try:
        smoke = ChapSmoke(base_url=base_url, model=args.model, endpoint_id=endpoint_id, api_key=args.api_key, timeout=args.timeout, fixture=fixture)
        print(f"LewLM: {base_url}   model: {smoke.model}   mode: {'fixture' if fixture else 'real'}")
        results = smoke.run(only={item.strip() for item in args.only.split(",") if item.strip()})
    finally:
        if fixture is not None:
            fixture.stop()

    summary = {status: sum(1 for check in results if check.status == status) for status in ("passed", "failed", "inconclusive", "skipped")}
    record = {
        "format": "lewlm-chap-smoke-v1",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": base_url,
        "mode": "fixture" if fixture else "real",
        "model": smoke.model,
        "endpoint_id": endpoint_id,
        "summary": summary,
        "checks": [asdict(check) for check in results],
    }
    if args.output:
        Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {summary}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
