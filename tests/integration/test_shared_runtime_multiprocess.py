"""One server process and two application processes share one model residency."""

from __future__ import annotations

from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "support" / "shared_runtime_server.py"
_CLIENT_CODE = r"""
import json, sys, time
from pathlib import Path
from lewlm import LewLMAppClient
from lewlm.api.schemas import ChatMessage
client = LewLMAppClient.from_http(sys.argv[1], application_id=sys.argv[3], timeout_seconds=15)
runtime = client.runtime_info()
response = client.chat_completion(model=sys.argv[2], messages=[ChatMessage(role="user", content=sys.argv[4])])
outputs = [response.choices[0].message.content]
if sys.argv[3] == "document-generator":
    coordination_dir = Path(sys.argv[5])
    (coordination_dir / "client-b-first-complete").touch()
    while not (coordination_dir / "client-a-disconnected").exists():
        time.sleep(0.01)
    continued = client.chat_completion(
        model=sys.argv[2],
        messages=[ChatMessage(role="user", content="continue after rag disconnect")],
    )
    outputs.append(continued.choices[0].message.content)
print(json.dumps({"runtime_instance_id": runtime.runtime_instance_id, "outputs": outputs}))
"""


def _json_request(base_url: str, method: str, path: str, payload: dict | None = None) -> tuple[int, object]:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"accept": "application/json"}
    if body is not None:
        headers["content-type"] = "application/json"
    request = Request(f"{base_url}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _wait_for(predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for subprocess integration state.")


def test_two_client_processes_share_one_server_residency(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    coordination_dir = tmp_path / "coordination"
    models_dir.mkdir()
    coordination_dir.mkdir()
    (models_dir / "shared-model.gguf").write_bytes(b"fake-gguf")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, str(_SERVER_SCRIPT), str(data_dir), str(models_dir), str(coordination_dir), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    client_a = None
    client_b = None
    try:
        def server_ready() -> bool:
            if server.poll() is not None:
                raise AssertionError(server.stderr.read())
            try:
                return _json_request(base_url, "GET", "/v1/runtime")[0] == 200
            except (URLError, ConnectionError):
                return False

        _wait_for(server_ready)
        scan_status, scan = _json_request(base_url, "POST", "/v1/models/scan", {})
        assert scan_status == 200
        model_id = scan["manifests"][0]["model_id"]

        client_a = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CLIENT_CODE,
                base_url,
                model_id,
                "rag-chat",
                "hold-active-lease",
                str(coordination_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        def load_started() -> bool:
            if client_a.poll() is not None:
                server.terminate()
                _, server_stderr = server.communicate(timeout=5)
                raise AssertionError(f"{client_a.stderr.read()}\nSERVER:\n{server_stderr}")
            return (coordination_dir / "load-started").exists()

        _wait_for(load_started)
        client_b = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CLIENT_CODE,
                base_url,
                model_id,
                "document-generator",
                "build status report",
                str(coordination_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.1)
        (coordination_dir / "allow-load").touch()
        _wait_for(lambda: (coordination_dir / "lease-active").exists())

        unload_status, unload_error = _json_request(base_url, "POST", f"/v1/models/{model_id}/unload")
        assert unload_status == 409
        assert unload_error["error"]["details"]["active_usage_count"] >= 1

        _wait_for(lambda: (coordination_dir / "client-b-first-complete").exists())
        (coordination_dir / "release-lease").touch()
        a_stdout, a_stderr = client_a.communicate(timeout=15)
        assert client_a.returncode == 0, a_stderr
        client_a_payload = json.loads(a_stdout)
        (coordination_dir / "client-a-disconnected").touch()
        b_stdout, b_stderr = client_b.communicate(timeout=15)
        assert client_b.returncode == 0, b_stderr
        client_b_payload = json.loads(b_stdout)

        assert client_a_payload["runtime_instance_id"] == client_b_payload["runtime_instance_id"]
        assert "process echo" in client_a_payload["outputs"][0]
        assert len(client_b_payload["outputs"]) == 2
        assert all("process echo" in output for output in client_b_payload["outputs"])

        stats_status, stats = _json_request(base_url, "GET", "/v1/runtime/stats")
        assert stats_status == 200
        assert stats["runtime_instance_id"] == client_a_payload["runtime_instance_id"]
        assert len(stats["residencies"]) == 1
        assert stats["residencies"][0]["load_attempt_count"] == 1
        runtime_stats = next(item for item in stats["runtimes"] if item["name"] == "process_test")
        assert runtime_stats["total_load_count"] == 1

        final_unload_status, final_unload = _json_request(base_url, "POST", f"/v1/models/{model_id}/unload")
        assert final_unload_status == 200
        assert final_unload["backend_operation_performed"] is True
        _wait_for(lambda: (coordination_dir / "unload-count").exists())
        assert (coordination_dir / "unload-count").read_text(encoding="utf-8") == "1"
        if os.name == "nt":
            server.terminate()
        else:
            server.send_signal(signal.SIGINT)
        server.wait(timeout=10)
        if os.name != "nt":
            assert server.returncode == 0
        assert (coordination_dir / "unload-count").read_text(encoding="utf-8") == "1"
    finally:
        for process in (client_a, client_b):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
