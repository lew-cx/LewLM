from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from conftest import set_host_platform
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


class _DeterministicMLXTextRuntime(ManagedTextRuntime):
    name = "fake_mlx_text"
    affinity = RuntimeAffinity.MLX_TEXT
    supported_formats = (ModelFormat.MLX,)
    supported_modalities = (ModelModality.TEXT,)
    supported_capabilities = frozenset({CapabilityName.CHAT, CapabilityName.STREAMING})
    supported_systems = ("Darwin",)
    supported_machines = ("arm64", "aarch64")
    platform_guidance = "Install the `mlx` extra on Apple Silicon macOS to enable MLX-native text generation."

    def __init__(self, *, available: bool = True) -> None:
        super().__init__()
        self._available = available

    def _check_environment(self) -> tuple[bool, str | None]:
        return self._available, (None if self._available else "Disabled for deterministic test coverage.")

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


class _DeterministicGGUFRuntime(_DeterministicMLXTextRuntime):
    name = "fake_llamacpp"
    affinity = RuntimeAffinity.LLAMACPP
    supported_formats = (ModelFormat.GGUF,)
    supported_systems = ("Darwin", "Linux", "Windows")
    supported_machines = ()
    platform_guidance = "Install the `llamacpp` extra or another compatible llama-cpp-python build on the target host."


class _DeterministicMLXVisionRuntime(_DeterministicMLXTextRuntime):
    name = "fake_mlx_vision"
    affinity = RuntimeAffinity.MLX_VISION
    supported_formats = (ModelFormat.MLX,)
    supported_modalities = (ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL)
    supported_capabilities = frozenset({CapabilityName.CHAT, CapabilityName.STREAMING, CapabilityName.VISION})
    platform_guidance = "Install the `mlx` extra on Apple Silicon macOS to enable MLX-native vision generation."


class _DeterministicVisionBridgeRuntime(ManagedTextRuntime):
    name = "local_external_adapter"
    affinity = RuntimeAffinity.EXTERNAL_ACCELERATOR
    supported_formats = (ModelFormat.MLX, ModelFormat.GGUF)
    supported_modalities = (ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL)
    supported_capabilities = frozenset({CapabilityName.CHAT, CapabilityName.STREAMING, CapabilityName.VISION})
    platform_guidance = (
        "Enable LEWLM_EXTERNAL_ACCELERATOR_ENABLED and point LEWLM_EXTERNAL_ACCELERATOR_BASE_URL "
        "at a loopback-only local OpenAI-compatible server on this host."
    )

    def _check_environment(self) -> tuple[bool, str | None]:
        return True, None

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


def _text_manifest(
    *,
    model_id: str,
    format_type: ModelFormat,
    runtime_affinity: tuple[RuntimeAffinity, ...],
    source_path: str,
) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        display_name=model_id,
        architecture_family="llama",
        modality=(ModelModality.TEXT,),
        source_path=source_path,
        format_type=format_type,
        runtime_affinity=runtime_affinity,
        estimated_memory_mb=512,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint=f"{model_id}-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )


def _vision_manifest(*, model_id: str, runtime_affinity: tuple[RuntimeAffinity, ...], source_path: str) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        display_name=model_id,
        architecture_family="qwen2_vl",
        modality=(ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL),
        source_path=source_path,
        format_type=ModelFormat.MLX,
        runtime_affinity=runtime_affinity,
        estimated_memory_mb=1024,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint=f"{model_id}-fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )


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


@pytest.mark.parametrize(
    ("system", "machine", "expected_host_fallback_count"),
    [
        ("Darwin", "arm64", 0),
        ("Linux", "x86_64", 1),
        ("Windows", "AMD64", 1),
    ],
)
def test_target_platform_matrix_reports_host_probed_platform_rows(
    monkeypatch,
    system: str,
    machine: str,
    expected_host_fallback_count: int,
) -> None:
    set_host_platform(monkeypatch, system=system, machine=machine)
    monkeypatch.setattr(
        RuntimeCatalog,
        "host_total_memory_snapshot",
        staticmethod(lambda: (24 * 1024, "synthetic", None)),
    )
    catalog = RuntimeCatalog(
        {
            RuntimeAffinity.MLX_TEXT: _DeterministicMLXTextRuntime(),
            RuntimeAffinity.LLAMACPP: _DeterministicGGUFRuntime(),
        },
    )
    manifests = [
        _text_manifest(
            model_id="mlx-model",
            format_type=ModelFormat.MLX,
            runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
            source_path="X:\\models\\mlx-model",
        ),
        _text_manifest(
            model_id="gguf-model",
            format_type=ModelFormat.GGUF,
            runtime_affinity=(RuntimeAffinity.LLAMACPP,),
            source_path="X:\\models\\gguf-model.gguf",
        ),
    ]

    rows = {
        (row["system"], row["machine"]): row
        for row in catalog.target_platform_matrix(manifests)
    }
    host_row = rows[(system, machine)]
    host_runtimes = {item["runtime_affinity"]: item for item in host_row["runtimes"]}

    assert host_row["verification_method"] == "host_probe"
    assert host_row["readiness_state"] == "verified"
    assert host_row["fallback_model_count"] == expected_host_fallback_count
    assert "gguf-model" in host_row["compatible_models"]
    assert host_runtimes["llamacpp"]["readiness_state"] == "verified"
    if system == "Darwin":
        assert "mlx-model" in host_row["compatible_models"]
        assert host_runtimes["mlx_text"]["readiness_state"] == "verified"
    else:
        assert "mlx-model" in host_row["fallback_models"]
        assert host_runtimes["mlx_text"]["readiness_state"] == "unsupported"

    declared_row = rows[("Darwin", "arm64") if (system, machine) != ("Darwin", "arm64") else ("Linux", "x86_64")]
    assert declared_row["verification_method"] == "runtime_contract"
    assert declared_row["readiness_state"] == "declared"


@pytest.mark.parametrize(
    ("system", "machine"),
    [
        ("Linux", "x86_64"),
        ("Windows", "AMD64"),
    ],
)
def test_describe_manifest_targets_reports_fallback_guidance_for_non_apple_text_mlx_hosts(
    monkeypatch,
    system: str,
    machine: str,
) -> None:
    set_host_platform(monkeypatch, system=system, machine=machine)
    monkeypatch.setattr(
        RuntimeCatalog,
        "host_total_memory_snapshot",
        staticmethod(lambda: (24 * 1024, "synthetic", None)),
    )
    catalog = RuntimeCatalog(
        {
            RuntimeAffinity.MLX_TEXT: _DeterministicMLXTextRuntime(),
            RuntimeAffinity.LLAMACPP: _DeterministicGGUFRuntime(),
        },
    )
    manifest = _text_manifest(
        model_id="mlx-model",
        format_type=ModelFormat.MLX,
        runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
        source_path="X:\\models\\mlx-model",
    )

    reports = {
        (report.system, report.machine): report
        for report in catalog.describe_manifest_targets(manifest)
    }
    host_report = reports[(system, machine)]
    darwin_report = reports[("Darwin", "arm64")]

    assert host_report.supported is False
    assert host_report.readiness_state == "fallback_guided"
    assert host_report.verification_method == "runtime_contract"
    assert host_report.fallback_available is True
    assert host_report.fallback_reason is not None
    assert "GGUF build" in host_report.fallback_reason
    assert _DeterministicGGUFRuntime.platform_guidance in host_report.install_hints
    assert darwin_report.supported is True
    assert darwin_report.readiness_state == "declared"


@pytest.mark.parametrize(
    ("system", "machine"),
    [
        ("Linux", "x86_64"),
        ("Windows", "AMD64"),
    ],
)
def test_describe_manifest_targets_reports_bridge_guidance_for_non_apple_vision_hosts(
    monkeypatch,
    system: str,
    machine: str,
) -> None:
    set_host_platform(monkeypatch, system=system, machine=machine)
    monkeypatch.setattr(
        RuntimeCatalog,
        "host_total_memory_snapshot",
        staticmethod(lambda: (24 * 1024, "synthetic", None)),
    )
    catalog = RuntimeCatalog(
        {
            RuntimeAffinity.MLX_VISION: _DeterministicMLXVisionRuntime(),
            RuntimeAffinity.EXTERNAL_ACCELERATOR: _DeterministicVisionBridgeRuntime(),
        },
    )
    manifest = _vision_manifest(
        model_id="vision-model",
        runtime_affinity=(RuntimeAffinity.MLX_VISION,),
        source_path="X:\\models\\vision-model",
    )

    reports = {
        (report.system, report.machine): report
        for report in catalog.describe_manifest_targets(manifest)
    }
    host_report = reports[(system, machine)]
    darwin_report = reports[("Darwin", "arm64")]

    assert host_report.supported is False
    assert host_report.readiness_state == "fallback_guided"
    assert host_report.verification_method == "runtime_contract"
    assert host_report.fallback_available is True
    assert host_report.fallback_reason is not None
    assert "bridge-backed via local_external_adapter" in host_report.fallback_reason
    assert "OpenAI-style image content blocks" in host_report.fallback_reason
    assert _DeterministicVisionBridgeRuntime.platform_guidance in host_report.install_hints
    assert darwin_report.supported is True
    assert darwin_report.readiness_state == "declared"


@pytest.mark.parametrize(
    ("system", "machine"),
    [
        ("Darwin", "arm64"),
        ("Linux", "x86_64"),
        ("Windows", "AMD64"),
    ],
)
def test_describe_manifest_targets_marks_host_gguf_as_verified(
    monkeypatch,
    system: str,
    machine: str,
) -> None:
    set_host_platform(monkeypatch, system=system, machine=machine)
    monkeypatch.setattr(
        RuntimeCatalog,
        "host_total_memory_snapshot",
        staticmethod(lambda: (24 * 1024, "synthetic", None)),
    )
    catalog = RuntimeCatalog({RuntimeAffinity.LLAMACPP: _DeterministicGGUFRuntime()})
    manifest = _text_manifest(
        model_id="gguf-model",
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
        source_path="X:\\models\\gguf-model.gguf",
    )

    reports = {
        (report.system, report.machine): report
        for report in catalog.describe_manifest_targets(manifest)
    }
    host_report = reports[(system, machine)]

    assert host_report.supported is True
    assert host_report.readiness_state == "verified"
    assert host_report.verification_method == "host_probe"
    assert host_report.runtime_affinities == [RuntimeAffinity.LLAMACPP]
