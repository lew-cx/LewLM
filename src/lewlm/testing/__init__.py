"""Test doubles LewLM ships for host apps: a fake engine and a fixture server.

Nothing here is used by the runtime. It exists so a host app (Chap, a CI job,
a notebook) can exercise LewLM's public HTTP contract on any machine with no
model, no GPU, and no engine installed, and see the same shapes a real engine
produces: models, capabilities, streaming, native tool calls, JSON output,
usage, cancellation, an engine that disappears, and structured errors.
"""

from lewlm.testing.fake_backend import FakeBackendFixture, FakeOpenAIEngine

__all__ = ["FakeBackendFixture", "FakeOpenAIEngine"]
