# Modernization step 02 validation

Validated on 2026-09-17 on the baseline Apple Silicon host. This step is
portable bridge work and does not require an engine installation or GPU.

The focused bridge/chat/tool/cancellation regression command completed with
`103 passed`. It includes real temporary loopback-server coverage and
in-memory fragmented transport coverage for SSE framing, native tool calls,
usage, finish reasons, cancellation, redirects, credentials, status mapping,
timeouts, and malformed/truncated responses.

The full unit suite completed with `786 passed, 5 failed`. Every remaining
failure is in document/XLSX/PDF ingestion or rendering and has the same host
Python 3.14 `pyexpat` versus system `libexpat` symbol mismatch recorded in the
step 00 baseline. `pip check` is also blocked at import time by that host
interpreter mismatch. No bridge, chat, response, tool-call, lifecycle, or
cancellation test failed.

Real upstream GPU cancellation remains unverified. These tests prove that
closing a LewLM stream closes the upstream HTTP response and that a subsequent
stream can use the same pool; they do not prove that a specific engine aborts
already-submitted GPU work.

