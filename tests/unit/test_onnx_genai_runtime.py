from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lewlm.core.contracts import GenerateMessage, GenerateRequest, RuntimeReadinessState
from lewlm.registry.discovery import discover_models
from lewlm.runtime.onnx_genai.runtime import ONNXGenAIRuntime


def test_onnx_genai_runtime_reports_planned_provider_evidence(monkeypatch, tmp_path: Path) -> None:
    bundle_dir = tmp_path / "phi-onnx"
    bundle_dir.mkdir()
    (bundle_dir / "genai_config.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "model.onnx").write_bytes(b"onnx")
    (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest = discover_models([tmp_path])[0]

    def fake_import_module(module_name: str):
        raise ImportError(module_name)

    monkeypatch.setattr("lewlm.runtime.onnx_genai.runtime.import_module", fake_import_module)

    report = ONNXGenAIRuntime().candidate_report(manifest)

    assert report.runtime_name == "onnx_genai"
    assert report.readiness_state == RuntimeReadinessState.RUNTIME_UNAVAILABLE
    assert report.available is False
    assert report.supports_manifest is True
    assert "onnxruntime-genai is not installed" in str(report.availability_reason)
    assert report.metadata["provider_family"] == "onnxruntime_genai"
    assert {item["provider"] for item in report.metadata["planned_execution_providers"]} == {
        "cpu",
        "cuda",
        "directml",
    }


def test_onnx_genai_runtime_reports_ready_when_python_surface_is_available(monkeypatch, tmp_path: Path) -> None:
    manifest = _onnx_manifest(tmp_path)
    monkeypatch.setattr("lewlm.runtime.onnx_genai.runtime.import_module", lambda module_name: _fake_onnx_module())

    report = ONNXGenAIRuntime().candidate_report(manifest)

    assert report.available is True
    assert report.readiness_state == RuntimeReadinessState.READY
    assert report.supports_manifest is True


async def test_onnx_genai_runtime_loads_generates_and_streams(monkeypatch, tmp_path: Path) -> None:
    manifest = _onnx_manifest(tmp_path)
    monkeypatch.setattr("lewlm.runtime.onnx_genai.runtime.import_module", lambda module_name: _fake_onnx_module())
    runtime = ONNXGenAIRuntime()

    await runtime.load_model(manifest)
    request = GenerateRequest(
        model_id=manifest.model_id,
        messages=[GenerateMessage(role="user", content="Say hi")],
        max_tokens=2,
        temperature=0.2,
    )

    response = await runtime.generate(request)
    chunks = []
    async for chunk in runtime.stream_generate(request):
        chunks.append(chunk)

    assert response.output_text == "Hello world"
    assert response.usage["completion_tokens"] == 2
    assert "".join(chunks) == "Hello world"


def _onnx_manifest(tmp_path: Path):
    bundle_dir = tmp_path / "phi-onnx"
    bundle_dir.mkdir()
    (bundle_dir / "genai_config.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "model.onnx").write_bytes(b"onnx")
    (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return discover_models([tmp_path])[0]


def _fake_onnx_module():
    class FakeModel:
        def __init__(self, path: str) -> None:
            self.path = path

    class FakeTokenizer:
        def __init__(self, model: FakeModel) -> None:
            self.model = model

        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(token) for token in tokens)

        def create_stream(self):
            return FakeTokenizerStream()

    class FakeTokenizerStream:
        def decode(self, token: int) -> str:
            return {101: "Hello", 102: " world"}.get(token, "")

    class FakeGeneratorParams:
        def __init__(self, model: FakeModel) -> None:
            self.model = model
            self.input_ids = []
            self.search_options = {}

        def set_search_options(self, **kwargs) -> None:
            self.search_options.update(kwargs)

    class FakeGenerator:
        def __init__(self, model: FakeModel, params: FakeGeneratorParams) -> None:
            self.model = model
            self.params = params
            self.tokens = [101, 102]
            self.index = 0
            self.current_token = None

        def is_done(self) -> bool:
            return self.index >= len(self.tokens)

        def compute_logits(self) -> None:
            return None

        def generate_next_token(self) -> None:
            self.current_token = self.tokens[self.index]
            self.index += 1

        def get_next_tokens(self) -> list[int]:
            return [self.current_token]

    return SimpleNamespace(
        Model=FakeModel,
        Tokenizer=FakeTokenizer,
        GeneratorParams=FakeGeneratorParams,
        Generator=FakeGenerator,
    )
