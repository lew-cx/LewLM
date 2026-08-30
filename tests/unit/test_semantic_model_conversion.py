"""LWL-M25-005: embedding and reranking bundles must convert on every supported path.

Sentence-transformers publishes these models as a bare backbone wrapped around a
decoder, which historically failed every exporter: the tensors are missing their
`model.` prefix, the config promises an `lm_head` that was never saved, and the
modality gates refused anything that was not text or vision.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from lewlm.config.settings import LewLMSettings
from lewlm.conversion.backend import (
    LlamaCppConversionBackend,
    MLXConversionBackend,
    OnnxGenAIConversionBackend,
)
from lewlm.conversion.checkpoint import (
    SAFETENSORS_INDEX_FILENAME,
    inspect_checkpoint_layout,
    is_backbone_checkpoint,
    normalize_checkpoint_bundle,
)
from lewlm.conversion.models import (
    CONVERSION_OUTPUT_METADATA_FILENAME,
    QUANTIZATION_PROFILE_METADATA_FILENAME,
    ConversionPolicy,
)
from lewlm.core.contracts import (
    ConversionStatus,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.core.errors import ConversionError
from lewlm.registry.discovery import discover_models


def _write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    """Write a minimal float32 safetensors shard holding the given raw payloads."""

    header: dict[str, object] = {}
    payload = bytearray()
    for name, raw in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": "F32", "shape": [len(raw) // 4], "data_offsets": [start, len(payload)]}
    header_bytes = json.dumps(header).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)


def _read_safetensors(path: Path) -> dict[str, bytes]:
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
        payload = handle.read()
    return {
        name: payload[entry["data_offsets"][0] : entry["data_offsets"][1]]
        for name, entry in header.items()
        if name != "__metadata__"
    }


def _backbone_bundle(root: Path, *, tie_word_embeddings: bool = True) -> Path:
    """A sentence-transformers export: decoder weights with no `model.` prefix."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "tie_word_embeddings": tie_word_embeddings,
                "hidden_size": 4,
            },
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_safetensors(
        root / "model.safetensors",
        {
            "embed_tokens.weight": b"\x01\x02\x03\x04" * 4,
            "layers.0.self_attn.q_proj.weight": b"\x05\x06\x07\x08" * 4,
            "norm.weight": b"\x09\x0a\x0b\x0c" * 2,
        },
    )
    return root


def _sentence_transformers_metadata(root: Path, *, rerank: bool) -> None:
    modules = (
        [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.base.modules.transformer.Transformer"},
            {
                "idx": 1,
                "name": "1",
                "path": "1_LogitScore",
                "type": "sentence_transformers.cross_encoder.modules.logit_score.LogitScore",
            },
        ]
        if rerank
        else [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
        ]
    )
    (root / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    (root / "config_sentence_transformers.json").write_text(
        json.dumps({"model_type": "CrossEncoder"} if rerank else {"similarity_fn_name": "cosine"}),
        encoding="utf-8",
    )


def _manifest(path: Path, *, modality: tuple[ModelModality, ...]) -> ModelManifest:
    return ModelManifest(
        model_id="semantic-model",
        display_name=path.name,
        architecture_family="qwen3",
        modality=modality,
        source_path=str(path),
        format_type=ModelFormat.HUGGINGFACE,
        runtime_affinity=(RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_TEXT),
        conversion_status=ConversionStatus.REQUIRES_CONVERSION,
        fingerprint="fingerprint",
        last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
    )


def _report(backend, manifest: ModelManifest, output_path: Path):
    return backend.compatibility_report(
        manifest,
        settings=LewLMSettings(),
        policy=ConversionPolicy.BALANCED,
        custom_bits=None,
        quantization_profile=None,
        cache_key="cache-key",
        output_path=output_path,
    )


class TestCheckpointLayoutDetection:
    def test_backbone_export_is_flagged_for_normalization(self, tmp_path: Path) -> None:
        layout = inspect_checkpoint_layout(_backbone_bundle(tmp_path / "embedding"))

        assert layout.needs_tensor_prefix is True
        assert layout.needs_normalization is True
        assert layout.is_decoder is True

    def test_config_promising_an_absent_lm_head_is_reconciled(self, tmp_path: Path) -> None:
        layout = inspect_checkpoint_layout(
            _backbone_bundle(tmp_path / "embedding", tie_word_embeddings=False),
        )

        assert layout.needs_tie_reconciliation is True

    def test_classifier_head_does_not_substitute_for_an_lm_head(self, tmp_path: Path) -> None:
        source = _backbone_bundle(tmp_path / "reranker", tie_word_embeddings=False)
        _write_safetensors(
            source / "model.safetensors",
            {
                "embed_tokens.weight": b"\x01" * 8,
                "layers.0.self_attn.q_proj.weight": b"\x02" * 8,
                "score.weight": b"\x03" * 8,
            },
        )

        layout = inspect_checkpoint_layout(source)

        assert layout.has_lm_head is False
        assert layout.needs_tie_reconciliation is True

    def test_canonical_causal_lm_bundle_is_left_alone(self, tmp_path: Path) -> None:
        root = tmp_path / "canonical"
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps({"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": False}),
            encoding="utf-8",
        )
        _write_safetensors(
            root / "model.safetensors",
            {"model.embed_tokens.weight": b"\x01\x02\x03\x04", "lm_head.weight": b"\x05\x06\x07\x08"},
        )

        assert is_backbone_checkpoint(root) is False

    def test_placeholder_weight_file_does_not_allocate_a_bogus_header(self, tmp_path: Path) -> None:
        # A stub file's leading bytes decode as an enormous header length; inspection
        # must reject it on the declared size rather than trying to allocate it.
        root = tmp_path / "stub"
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}), encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"weights-1")

        layout = inspect_checkpoint_layout(root)

        assert layout.tensor_names == ()
        assert layout.needs_normalization is False

    def test_encoder_only_bundle_is_not_a_decoder(self, tmp_path: Path) -> None:
        root = tmp_path / "bert"
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"architectures": ["BertModel"]}), encoding="utf-8")
        _write_safetensors(
            root / "model.safetensors",
            {"embeddings.word_embeddings.weight": b"\x01\x02\x03\x04", "encoder.layer.0.output.dense.weight": b"\x00" * 8},
        )

        assert inspect_checkpoint_layout(root).is_decoder is False


class TestCheckpointNormalization:
    def test_tensors_are_reprefixed_without_altering_their_payloads(self, tmp_path: Path) -> None:
        source = _backbone_bundle(tmp_path / "embedding")
        original = _read_safetensors(source / "model.safetensors")

        result = normalize_checkpoint_bundle(source, tmp_path / "normalized")

        assert result.normalized is True
        normalized = _read_safetensors(result.source_path / "model.safetensors")
        assert set(normalized) == {f"model.{name}" for name in original}
        # Renaming a tensor must never disturb the bytes it points at.
        for name, raw in original.items():
            assert normalized[f"model.{name}"] == raw

    def test_absent_lm_head_ties_the_output_projection(self, tmp_path: Path) -> None:
        source = _backbone_bundle(tmp_path / "embedding", tie_word_embeddings=False)

        result = normalize_checkpoint_bundle(source, tmp_path / "normalized")

        config = json.loads((result.source_path / "config.json").read_text(encoding="utf-8"))
        assert config["tie_word_embeddings"] is True
        assert result.metadata["tie_word_embeddings_forced"] is True

    def test_top_level_lm_head_is_not_moved_into_the_backbone(self, tmp_path: Path) -> None:
        source = _backbone_bundle(tmp_path / "embedding", tie_word_embeddings=False)
        _write_safetensors(
            source / "model.safetensors",
            {
                "embed_tokens.weight": b"\x01" * 8,
                "layers.0.self_attn.q_proj.weight": b"\x02" * 8,
                "norm.weight": b"\x03" * 8,
                "lm_head.weight": b"\x04" * 8,
            },
        )

        result = normalize_checkpoint_bundle(source, tmp_path / "normalized")

        tensors = _read_safetensors(result.source_path / "model.safetensors")
        assert set(tensors) == {
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        }
        config = json.loads((result.source_path / "config.json").read_text(encoding="utf-8"))
        assert config["tie_word_embeddings"] is False

    @pytest.mark.parametrize("shard_reference", ["absolute", "parent"])
    def test_index_shards_cannot_escape_the_bundle(
        self,
        tmp_path: Path,
        shard_reference: str,
    ) -> None:
        outside = tmp_path / "outside.safetensors"
        _write_safetensors(outside, {"embed_tokens.weight": b"\x01" * 8})
        original = outside.read_bytes()
        source = tmp_path / "bundle"
        source.mkdir()
        (source / "config.json").write_text(
            json.dumps({"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": True}),
            encoding="utf-8",
        )
        shard_name = str(outside) if shard_reference == "absolute" else "../outside.safetensors"
        (source / SAFETENSORS_INDEX_FILENAME).write_text(
            json.dumps({"weight_map": {"embed_tokens.weight": shard_name}}),
            encoding="utf-8",
        )

        with pytest.raises(ConversionError, match="safe relative filename"):
            normalize_checkpoint_bundle(source, tmp_path / "normalized")

        assert outside.read_bytes() == original
        assert not (tmp_path / "normalized").exists()

    def test_sharded_bundle_rewrites_its_weight_map(self, tmp_path: Path) -> None:
        source = tmp_path / "sharded"
        source.mkdir()
        (source / "config.json").write_text(
            json.dumps({"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": True}),
            encoding="utf-8",
        )
        _write_safetensors(source / "model-00001-of-00002.safetensors", {"embed_tokens.weight": b"\x01" * 8})
        _write_safetensors(source / "model-00002-of-00002.safetensors", {"layers.0.mlp.up_proj.weight": b"\x02" * 8})
        (source / SAFETENSORS_INDEX_FILENAME).write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 16},
                    "weight_map": {
                        "embed_tokens.weight": "model-00001-of-00002.safetensors",
                        "layers.0.mlp.up_proj.weight": "model-00002-of-00002.safetensors",
                    },
                },
            ),
            encoding="utf-8",
        )

        result = normalize_checkpoint_bundle(source, tmp_path / "normalized")

        index = json.loads((result.source_path / SAFETENSORS_INDEX_FILENAME).read_text(encoding="utf-8"))
        assert set(index["weight_map"]) == {"model.embed_tokens.weight", "model.layers.0.mlp.up_proj.weight"}
        assert _read_safetensors(result.source_path / "model-00002-of-00002.safetensors") == {
            "model.layers.0.mlp.up_proj.weight": b"\x02" * 8,
        }

    def test_support_files_travel_with_the_normalized_bundle(self, tmp_path: Path) -> None:
        source = _backbone_bundle(tmp_path / "embedding")
        (source / "chat_template.jinja").write_text("template", encoding="utf-8")

        result = normalize_checkpoint_bundle(source, tmp_path / "normalized")

        assert (result.source_path / "tokenizer.json").exists()
        assert (result.source_path / "chat_template.jinja").read_text(encoding="utf-8") == "template"

    def test_output_cannot_be_created_inside_the_source_bundle(self, tmp_path: Path) -> None:
        source = _backbone_bundle(tmp_path / "embedding")

        with pytest.raises(ConversionError, match="outside the source bundle"):
            normalize_checkpoint_bundle(source, source / "normalized")

        assert not (source / "normalized").exists()

    def test_canonical_bundle_is_returned_untouched(self, tmp_path: Path) -> None:
        root = tmp_path / "canonical"
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps({"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": True}),
            encoding="utf-8",
        )
        _write_safetensors(root / "model.safetensors", {"model.embed_tokens.weight": b"\x01\x02\x03\x04"})

        result = normalize_checkpoint_bundle(root, tmp_path / "normalized")

        assert result.normalized is False
        assert result.source_path == root


class TestSemanticRoleDiscovery:
    def test_pooling_module_graph_marks_an_embedding_model(self, tmp_path: Path) -> None:
        # The directory name says nothing, so only the module graph can classify it.
        bundle = _backbone_bundle(tmp_path / "models" / "internal-retriever-v2")
        _sentence_transformers_metadata(bundle, rerank=False)

        manifests = {m.display_name: m for m in discover_models([tmp_path / "models"])}

        assert manifests["internal-retriever-v2"].modality == (ModelModality.EMBEDDING,)

    def test_cross_encoder_module_graph_marks_a_reranker(self, tmp_path: Path) -> None:
        bundle = _backbone_bundle(tmp_path / "models" / "internal-scorer-v2")
        _sentence_transformers_metadata(bundle, rerank=True)

        manifests = {m.display_name: m for m in discover_models([tmp_path / "models"])}

        assert manifests["internal-scorer-v2"].modality == (ModelModality.RERANK,)

    def test_single_label_sequence_classifier_is_a_reranker(self, tmp_path: Path) -> None:
        bundle = tmp_path / "models" / "pair-scorer"
        bundle.mkdir(parents=True)
        (bundle / "config.json").write_text(
            json.dumps({"architectures": ["XLMRobertaForSequenceClassification"], "num_labels": 1}),
            encoding="utf-8",
        )
        (bundle / "tokenizer.json").write_text("{}", encoding="utf-8")
        _write_safetensors(bundle / "model.safetensors", {"encoder.layer.0.output.dense.weight": b"\x00" * 8})

        manifests = {m.display_name: m for m in discover_models([tmp_path / "models"])}

        assert manifests["pair-scorer"].modality == (ModelModality.RERANK,)


class TestConversionGates:
    _MLX_REFUSAL = "supports text- or vision-capable models"

    @pytest.mark.parametrize("modality", [(ModelModality.EMBEDDING,), (ModelModality.RERANK,)])
    def test_mlx_admits_semantic_models_built_on_a_decoder(
        self,
        tmp_path: Path,
        modality: tuple[ModelModality, ...],
    ) -> None:
        manifest = _manifest(_backbone_bundle(tmp_path / "embedding"), modality=modality)

        report = _report(MLXConversionBackend(), manifest, tmp_path / "out")

        # mlx-lm may be absent on this host, so assert the modality gate specifically.
        assert self._MLX_REFUSAL not in report.reason

    def test_mlx_still_refuses_an_encoder_only_semantic_model(self, tmp_path: Path) -> None:
        root = tmp_path / "bert-embedder"
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"architectures": ["BertModel"]}), encoding="utf-8")
        _write_safetensors(root / "model.safetensors", {"embeddings.word_embeddings.weight": b"\x00" * 8})
        manifest = _manifest(root, modality=(ModelModality.EMBEDDING,))

        report = _report(MLXConversionBackend(), manifest, tmp_path / "out")

        assert report.can_convert is False
        assert "decoder backbone" in report.reason

    def test_onnx_admits_semantic_models_built_on_a_decoder(self, tmp_path: Path) -> None:
        manifest = _manifest(_backbone_bundle(tmp_path / "embedding"), modality=(ModelModality.EMBEDDING,))

        report = _report(OnnxGenAIConversionBackend(), manifest, tmp_path / "out")

        assert "does not" not in report.reason

    def test_llamacpp_reports_the_normalization_it_will_perform(self, tmp_path: Path) -> None:
        manifest = _manifest(
            _backbone_bundle(tmp_path / "embedding", tie_word_embeddings=False),
            modality=(ModelModality.EMBEDDING,),
        )

        report = _report(LlamaCppConversionBackend(), manifest, tmp_path / "out")

        assert any("bare transformer backbone" in warning for warning in report.warnings)
        assert any("tie_word_embeddings" in warning for warning in report.warnings)


class TestConvertedArtifactModality:
    """A converted artifact must keep serving the capability it was converted for.

    Conversion output holds none of the sentence-transformers files discovery reads,
    and its directory is named by cache key, so nothing in it says "embedding".
    """

    def _converted_bundle(self, root: Path, *, modality: list[str] | None) -> Path:
        bundle = root / "3f9c2a8d14b7"
        bundle.mkdir(parents=True)
        (bundle / "config.json").write_text(
            json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
            encoding="utf-8",
        )
        (bundle / "tokenizer.json").write_text("{}", encoding="utf-8")
        _write_safetensors(bundle / "model.safetensors", {"model.embed_tokens.weight": b"\x00" * 8})
        (bundle / QUANTIZATION_PROFILE_METADATA_FILENAME).write_text(
            json.dumps({"strategy": "weight_only", "precision": "int4"}),
            encoding="utf-8",
        )
        payload = {
            "source_display_name": "Qwen3-Embedding_0.6B",
            "source_model_id": "source-id",
            "display_name": "Qwen3-Embedding_0.6B (converted)",
            "artifact_role": "standalone",
            "artifact_family_id": "family",
            "metadata": {"backend_name": "mlx_lm", "target_format": "mlx"},
        }
        if modality is not None:
            payload["modality"] = modality
        (bundle / CONVERSION_OUTPUT_METADATA_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
        return bundle

    def test_recorded_modality_survives_rediscovery(self, tmp_path: Path) -> None:
        self._converted_bundle(tmp_path / "conversions", modality=["embedding"])

        manifest = discover_models([tmp_path / "conversions"])[0]

        assert manifest.modality == (ModelModality.EMBEDDING,)
        assert RuntimeAffinity.MLX_TEXT in manifest.runtime_affinity

    def test_rerank_artifact_is_not_rediscovered_as_chat(self, tmp_path: Path) -> None:
        self._converted_bundle(tmp_path / "conversions", modality=["rerank"])

        manifest = discover_models([tmp_path / "conversions"])[0]

        assert manifest.modality == (ModelModality.RERANK,)

    def test_metadata_without_modality_falls_back_to_inference(self, tmp_path: Path) -> None:
        # Artifacts converted by earlier versions carry no modality field.
        self._converted_bundle(tmp_path / "conversions", modality=None)

        manifest = discover_models([tmp_path / "conversions"])[0]

        assert manifest.modality == (ModelModality.TEXT,)


class TestPairedArtifactPlanning:
    """mlx-vlm and mlx-lm support different architectures; planning must respect that."""

    def _multimodal_manifest(self, tmp_path: Path, *, model_type: str) -> ModelManifest:
        root = tmp_path / f"{model_type}-vl"
        root.mkdir(parents=True)
        (root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": model_type,
                    "text_config": {"hidden_size": 2048},
                    "vision_config": {"image_size": 448},
                },
            ),
            encoding="utf-8",
        )
        _write_safetensors(root / "model.safetensors", {"model.embed_tokens.weight": b"\x00" * 8})
        return ModelManifest(
            model_id=f"{model_type}-vl",
            display_name=f"{model_type}-vl",
            architecture_family=model_type,
            modality=(ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL),
            source_path=str(root),
            format_type=ModelFormat.HUGGINGFACE,
            runtime_affinity=(RuntimeAffinity.CONVERSION, RuntimeAffinity.MLX_VISION),
            conversion_status=ConversionStatus.REQUIRES_CONVERSION,
            fingerprint="fingerprint",
            last_validation_result=ModelValidationResult(status=ValidationState.VALID, message="ok"),
        )

    def test_text_artifact_is_dropped_when_mlx_lm_lacks_the_architecture(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = MLXConversionBackend()
        monkeypatch.setattr(
            "lewlm.conversion.backend._mlx_lm_supports_model_type",
            lambda model_type: model_type != "gemma4",
        )
        manifest = self._multimodal_manifest(tmp_path, model_type="gemma4")

        report = _report(backend, manifest, tmp_path / "out")

        # The multimodal artifact still converts and serves both text and vision.
        assert [plan.artifact_key for plan in report.artifact_plans] == ["standalone"]
        assert report.layered_output is False
        assert any("no builder for `gemma4`" in warning for warning in report.warnings)

    def test_paired_plan_is_kept_when_mlx_lm_has_the_architecture(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = MLXConversionBackend()
        monkeypatch.setattr("lewlm.conversion.backend._mlx_lm_supports_model_type", lambda model_type: True)
        monkeypatch.setattr(backend, "availability_reason", lambda: None)
        monkeypatch.setattr(backend, "_conversion_backend_available", lambda _backend: True)
        manifest = self._multimodal_manifest(tmp_path, model_type="gemma4")

        report = _report(backend, manifest, tmp_path / "out")

        assert [plan.artifact_key for plan in report.artifact_plans] == ["multimodal", "text"]
        assert report.layered_output is True
        assert not any("no builder" in warning for warning in report.warnings)


class TestAppleDoubleSidecars:
    """Bundles copied off macOS carry `._*` sidecars beside every real file.

    They match the `.safetensors` suffix but hold resource-fork metadata, so
    reading one as a shard fails on its declared header length. Non-Apple hosts
    are exactly the ones receiving such copies, so discovery has to skip them.
    """

    @staticmethod
    def _add_appledouble_sidecars(root: Path) -> None:
        for entry in list(root.iterdir()):
            # AppleDouble headers start with the magic 0x00051607; as a
            # little-endian u64 that is a nonsense safetensors header length.
            (root / f"._{entry.name}").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 4092)

    def test_layout_inspection_ignores_sidecar_shards(self, tmp_path: Path) -> None:
        root = _backbone_bundle(tmp_path / "embedding")
        self._add_appledouble_sidecars(root)

        layout = inspect_checkpoint_layout(root)

        assert layout.shard_names == ("model.safetensors",)
        assert "embed_tokens.weight" in layout.tensor_names

    def test_normalization_skips_sidecar_shards(self, tmp_path: Path) -> None:
        root = _backbone_bundle(tmp_path / "embedding")
        _sentence_transformers_metadata(root, rerank=False)
        self._add_appledouble_sidecars(root)

        result = normalize_checkpoint_bundle(root, tmp_path / "normalized")

        assert result.normalized is True
        normalized_shards = sorted(
            path.name for path in result.source_path.iterdir() if path.name.endswith(".safetensors")
        )
        assert normalized_shards == ["model.safetensors"]
        tensors = _read_safetensors(result.source_path / "model.safetensors")
        assert all(name.startswith("model.") for name in tensors)
