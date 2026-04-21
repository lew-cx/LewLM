"""Benchmark helpers for CLI and test entrypoints."""

from lewlm.benchmarking.direct import benchmark_direct_chat_manifest
from lewlm.benchmarking.external import benchmark_runtime_chat_manifest

__all__ = ["benchmark_direct_chat_manifest", "benchmark_runtime_chat_manifest"]
