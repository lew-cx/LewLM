"""Subprocess server used by the process-boundary residency integration test."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from collections.abc import AsyncIterator, Sequence

import uvicorn

from lewlm.api.app import create_app
from lewlm.config.settings import LewLMSettings
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import (
    CapabilityName,
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    ModelModality,
    RuntimeAffinity,
)
from lewlm.runtime.base import ManagedTextRuntime


class ProcessTestRuntime(ManagedTextRuntime):
    name = "process_test"
    affinity = RuntimeAffinity.LLAMACPP
    supported_formats = (ModelFormat.GGUF,)
    supported_modalities = (ModelModality.TEXT,)
    supported_capabilities = frozenset({CapabilityName.CHAT, CapabilityName.STREAMING})

    def __init__(self, *, coordination_dir: Path) -> None:
        super().__init__()
        self.coordination_dir = coordination_dir

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, None

    async def _load_model(self, manifest: ModelManifest) -> None:
        (self.coordination_dir / "load-started").touch()
        while not (self.coordination_dir / "allow-load").exists():
            await asyncio.sleep(0.01)

    async def _unload_model(self, model_id: str) -> None:
        counter_path = self.coordination_dir / "unload-count"
        count = int(counter_path.read_text(encoding="utf-8")) if counter_path.exists() else 0
        counter_path.write_text(str(count + 1), encoding="utf-8")

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        prompt = " ".join(message.content for message in request.messages)
        if "hold-active-lease" in prompt:
            (self.coordination_dir / "lease-active").touch()
            while not (self.coordination_dir / "release-lease").exists():
                await asyncio.sleep(0.01)
        return GenerateResponse(
            model_id=request.model_id,
            output_text=f"process echo: {prompt}",
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        response = await self._generate(request)
        yield response.output_text

    def _tokenize(self, text: str) -> list[int]:
        return list(text.encode())

    def _detokenize(self, tokens: Sequence[int]) -> str:
        return bytes(tokens).decode()


def main() -> None:
    data_dir = Path(sys.argv[1])
    models_dir = Path(sys.argv[2])
    coordination_dir = Path(sys.argv[3])
    port = int(sys.argv[4])
    settings = LewLMSettings(
        data_dir=data_dir,
        models_dir=[models_dir],
        host="127.0.0.1",
        port=port,
        runtime_policy="keep_warm",
    )
    services = bootstrap_services(
        settings,
        runtime_overrides={RuntimeAffinity.LLAMACPP: ProcessTestRuntime(coordination_dir=coordination_dir)},
    )
    try:
        uvicorn.run(create_app(services=services), host="127.0.0.1", port=port, log_level="error")
    finally:
        services.close()


if __name__ == "__main__":
    main()
