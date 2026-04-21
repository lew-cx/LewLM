from __future__ import annotations

from types import SimpleNamespace

import pytest

from lewlm.config.settings import LewLMSettings
from lewlm.core.contracts import (
    ConversionStatus,
    GenerateMessage,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    SpeculationMode,
    ValidationState,
)
from lewlm.core.errors import ConfigurationError
from lewlm.core.speculation import plan_chat_speculation


class FakeRegistry:
    def __init__(self, manifests: list[ModelManifest]) -> None:
        self._manifests = {manifest.model_id: manifest for manifest in manifests}

    def list_manifests(self) -> list[ModelManifest]:
        return list(self._manifests.values())

    def get_manifest(self, model_id: str) -> ModelManifest:
        return self._manifests[model_id]


def test_plan_chat_speculation_auto_selects_smaller_mlx_draft_model(tmp_path) -> None:
    primary = _manifest("primary", ModelFormat.MLX, RuntimeAffinity.MLX_TEXT, estimated_memory_mb=2048)
    draft = _manifest("draft", ModelFormat.MLX, RuntimeAffinity.MLX_TEXT, estimated_memory_mb=512)
    planner_result = plan_chat_speculation(
        model_registry=FakeRegistry([primary, draft]),
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            speculative_decoding_enabled=True,
            speculative_decoding_num_draft_tokens=5,
        ),
        primary_manifest=primary,
        runtime=SimpleNamespace(affinity=RuntimeAffinity.MLX_TEXT),
        messages=[GenerateMessage(role="user", content="hello world")],
        max_tokens=32,
    )

    assert planner_result is not None
    assert planner_result.draft_manifest is not None
    assert planner_result.draft_manifest.model_id == "draft"
    assert planner_result.request.mode == SpeculationMode.DRAFT_MODEL
    assert planner_result.request.auto_selected is True
    assert planner_result.request.num_draft_tokens == 5


def test_plan_chat_speculation_builds_prompt_lookup_request(tmp_path) -> None:
    primary = _manifest("gguf", ModelFormat.GGUF, RuntimeAffinity.LLAMACPP, estimated_memory_mb=1024)
    planner_result = plan_chat_speculation(
        model_registry=FakeRegistry([primary]),
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            prompt_lookup_speculation_enabled=True,
            prompt_lookup_max_ngram_size=4,
            prompt_lookup_num_pred_tokens=12,
        ),
        primary_manifest=primary,
        runtime=SimpleNamespace(affinity=RuntimeAffinity.LLAMACPP),
        messages=[GenerateMessage(role="user", content="lookup prompt")],
        max_tokens=32,
    )

    assert planner_result is not None
    assert planner_result.draft_manifest is None
    assert planner_result.request.mode == SpeculationMode.PROMPT_LOOKUP
    assert planner_result.request.prompt_lookup_max_ngram_size == 4
    assert planner_result.request.prompt_lookup_num_pred_tokens == 12


def test_plan_chat_speculation_rejects_primary_model_as_explicit_draft(tmp_path) -> None:
    primary = _manifest("primary", ModelFormat.MLX, RuntimeAffinity.MLX_TEXT, estimated_memory_mb=2048)
    with pytest.raises(ConfigurationError):
        plan_chat_speculation(
            model_registry=FakeRegistry([primary]),
            settings=LewLMSettings(
                data_dir=tmp_path / "state",
                speculative_decoding_enabled=True,
                speculative_decoding_draft_model_id="primary",
            ),
            primary_manifest=primary,
            runtime=SimpleNamespace(affinity=RuntimeAffinity.MLX_TEXT),
            messages=[GenerateMessage(role="user", content="hello world")],
            max_tokens=32,
        )


def test_plan_chat_speculation_can_prefer_frontier_metadata_adapter(tmp_path) -> None:
    primary = _manifest("primary", ModelFormat.MLX, RuntimeAffinity.MLX_TEXT, estimated_memory_mb=2048)
    primary = primary.model_copy(
        update={
            "metadata": {
                "speculation_modes": [
                    {
                        "mode": "medusa",
                        "companion_model_id": "medusa-helper",
                        "parameters": {"backend_parameter": "medusa_model"},
                    },
                ],
            },
        },
    )
    draft = _manifest("draft", ModelFormat.MLX, RuntimeAffinity.MLX_TEXT, estimated_memory_mb=512)
    medusa = _manifest("medusa-helper", ModelFormat.MLX, RuntimeAffinity.MLX_TEXT, estimated_memory_mb=768)

    planner_result = plan_chat_speculation(
        model_registry=FakeRegistry([primary, draft, medusa]),
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            speculative_decoding_enabled=True,
            speculative_decoding_num_draft_tokens=5,
        ),
        primary_manifest=primary,
        runtime=SimpleNamespace(affinity=RuntimeAffinity.MLX_TEXT),
        messages=[GenerateMessage(role="user", content="hello world")],
        max_tokens=32,
        preferred_mode=SpeculationMode.MEDUSA,
    )

    assert planner_result is not None
    assert planner_result.request.mode == SpeculationMode.MEDUSA
    assert planner_result.request.companion_model_id == "medusa-helper"
    assert planner_result.request.parameters["backend_parameter"] == "medusa_model"
    assert planner_result.companion_manifests[0].model_id == "medusa-helper"
    assert planner_result.benchmark_preferred is True


def test_plan_chat_speculation_accepts_swift_alias_for_heterogeneous_vocab(tmp_path) -> None:
    primary = _manifest("primary", ModelFormat.MLX, RuntimeAffinity.MLX_TEXT, estimated_memory_mb=2048)
    primary = primary.model_copy(
        update={
            "metadata": {
                "speculation_modes": [
                    {
                        "mode": "swift",
                        "parameters": {"backend_parameter": "swift"},
                    },
                ],
            },
        },
    )

    planner_result = plan_chat_speculation(
        model_registry=FakeRegistry([primary]),
        settings=LewLMSettings(
            data_dir=tmp_path / "state",
            speculative_decoding_enabled=True,
        ),
        primary_manifest=primary,
        runtime=SimpleNamespace(affinity=RuntimeAffinity.MLX_TEXT),
        messages=[GenerateMessage(role="user", content="return code")],
        max_tokens=32,
    )

    assert planner_result is not None
    assert planner_result.request.mode == SpeculationMode.HETEROGENEOUS_VOCAB
    assert planner_result.request.parameters["backend_parameter"] == "swift"


def _manifest(model_id: str, format_type: ModelFormat, affinity: RuntimeAffinity, *, estimated_memory_mb: int) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        display_name=model_id,
        architecture_family="qwen2",
        modality=(ModelModality.TEXT,),
        source_path=f"/tmp/{model_id}",
        format_type=format_type,
        runtime_affinity=(affinity,),
        estimated_memory_mb=estimated_memory_mb,
        context_length=8192,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint=f"fingerprint-{model_id}",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )
