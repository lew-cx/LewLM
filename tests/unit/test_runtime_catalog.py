from __future__ import annotations

from collections.abc import AsyncIterator

from lewlm.core.contracts import (
    CapabilityName,
    ConversionStatus,
    GenerateRequest,
    GenerateResponse,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RequestModality,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.runtime.base import ManagedTextRuntime
from lewlm.runtime.catalog import RuntimeCatalog


def test_host_total_memory_snapshot_uses_windows_probe(monkeypatch) -> None:
    class FakeKernel32:
        def GlobalMemoryStatusEx(self, status_pointer) -> int:
            status_pointer._obj.ullTotalPhys = 16 * 1024 * 1024 * 1024
            return 1

    monkeypatch.setattr("lewlm.runtime.catalog.platform.system", lambda: "Windows")
    monkeypatch.setattr("lewlm.runtime.catalog.ctypes.WinDLL", lambda *args, **kwargs: FakeKernel32(), raising=False)

    total_memory_mb, source, reason = RuntimeCatalog.host_total_memory_snapshot()

    assert total_memory_mb == 16 * 1024
    assert source == "windows_globalmemorystatusex"
    assert reason is None


def test_host_total_memory_snapshot_falls_back_to_linux_proc_meminfo(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.runtime.catalog.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        RuntimeCatalog,
        "_posix_total_memory_mb",
        staticmethod(lambda: (None, "POSIX sysconf did not return usable physical-memory values.")),
    )
    monkeypatch.setattr(
        RuntimeCatalog,
        "_linux_proc_meminfo_total_memory_mb",
        staticmethod(lambda: (32 * 1024, None)),
    )

    total_memory_mb, source, reason = RuntimeCatalog.host_total_memory_snapshot()

    assert total_memory_mb == 32 * 1024
    assert source == "linux_proc_meminfo"
    assert reason is None


def test_host_platform_snapshot_preserves_memory_probe_reason(monkeypatch) -> None:
    monkeypatch.setattr("lewlm.runtime.catalog.platform.system", lambda: "Windows")
    monkeypatch.setattr("lewlm.runtime.catalog.platform.release", lambda: "11")
    monkeypatch.setattr("lewlm.runtime.catalog.platform.machine", lambda: "AMD64")
    monkeypatch.setattr("lewlm.runtime.catalog.platform.python_version", lambda: "3.11.9")
    monkeypatch.setattr(
        RuntimeCatalog,
        "host_total_memory_snapshot",
        staticmethod(lambda: (None, None, "Windows GlobalMemoryStatusEx failed.")),
    )

    snapshot = RuntimeCatalog.host_platform_snapshot()

    assert snapshot.system == "Windows"
    assert snapshot.machine == "AMD64"
    assert snapshot.total_memory_mb is None
    assert snapshot.total_memory_source is None
    assert snapshot.total_memory_reason == "Windows GlobalMemoryStatusEx failed."


class _VisionBlockedBridgeRuntime(ManagedTextRuntime):
    name = "vision_blocked_bridge"
    affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR
    supported_formats = (ModelFormat.MLX,)
    supported_modalities = (ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL)
    supported_capabilities = frozenset({CapabilityName.CHAT, CapabilityName.STREAMING, CapabilityName.VISION})

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, None

    def supports_manifest_capability(self, manifest: ModelManifest, capability: CapabilityName) -> bool:
        if capability == CapabilityName.VISION:
            return False
        return super().supports_manifest_capability(manifest, capability)

    def manifest_capability_reason(self, manifest: ModelManifest, capability: CapabilityName) -> str | None:
        if capability == CapabilityName.VISION:
            return "Vision is not available through this bridge runtime."
        return super().manifest_capability_reason(manifest, capability)

    async def _load_model(self, manifest: ModelManifest) -> None:
        return None

    async def _unload_model(self, model_id: str) -> None:
        return None

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(model_id=request.model_id, output_text="ok", finish_reason="stop")

    async def _stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        if False:
            yield ""

    def _tokenize(self, text: str) -> list[int]:
        return []

    def _detokenize(self, tokens) -> str:
        return ""


def test_catalog_rejects_image_conditioned_chat_when_runtime_lacks_vision_capability() -> None:
    manifest = ModelManifest(
        model_id="vision-model",
        display_name="Vision Model",
        architecture_family="qwen2_vl",
        modality=(ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL),
        source_path="X:\\models\\vision-model",
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.EXTERNAL_ACCELERATOR,),
        estimated_memory_mb=1024,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="vision-model-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )
    catalog = RuntimeCatalog({RuntimeAffinity.EXTERNAL_ACCELERATOR: _VisionBlockedBridgeRuntime()})

    compatible, alternatives = catalog.compatible_runtimes(
        manifest,
        capability=CapabilityName.CHAT,
        request_modality=RequestModality.IMAGE_CONDITIONED,
    )

    assert compatible == []
    assert any("Vision is not available through this bridge runtime." in item for item in alternatives)
