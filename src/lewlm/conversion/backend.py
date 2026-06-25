"""Conversion backends for model format adaptation."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from lewlm.config.settings import LewLMSettings
from lewlm.conversion.jang import is_jang_bundle, missing_jang_dependencies, normalize_jang_bundle
from lewlm.conversion.models import (
    ConversionCompatibilityReport,
    ConversionPolicy,
    ConversionProfileSupport,
    LayeredConversionArtifact,
    quantization_mode_from_profile,
    resolve_quantization_profile,
)
from lewlm.core.contracts import (
    ModelFormat,
    ModelArtifactRole,
    ModelManifest,
    ModelModality,
    QuantizationPrecision,
    QuantizationProfile,
    QuantizationStrategy,
    RuntimeAffinity,
)
from lewlm.core.errors import ConversionError


@dataclass(slots=True)
class ConversionExecutionArtifact:
    artifact_key: str
    role: ModelArtifactRole
    display_name: str
    output_path: Path
    format_type: ModelFormat
    modality: tuple[ModelModality, ...]
    runtime_affinity: tuple[RuntimeAffinity, ...]
    derived_from: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConversionExecutionResult:
    output_path: Path
    logs: list[str]
    artifacts: tuple[ConversionExecutionArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class _ResolvedTool:
    command: tuple[str, ...]
    display_name: str
    cwd: Path | None = None
    env: dict[str, str] | None = None


class ConversionBackend(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def availability_reason(self) -> str | None: ...

    def compatibility_report(
        self,
        manifest: ModelManifest,
        *,
        settings: LewLMSettings,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile: QuantizationProfile | None,
        cache_key: str,
        output_path: Path,
    ) -> ConversionCompatibilityReport: ...

    def convert(
        self,
        manifest: ModelManifest,
        *,
        settings: LewLMSettings,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile: QuantizationProfile | None,
        output_path: Path,
        work_dir: Path,
    ) -> ConversionExecutionResult: ...


class LlamaCppConversionBackend:
    """Export compatible Hugging Face bundles to GGUF with llama.cpp tools."""

    name = "llamacpp_gguf"

    _CONVERTER_CANDIDATES = ("convert_hf_to_gguf.py", "llama-convert-hf-to-gguf", "convert-hf-to-gguf")
    _QUANTIZER_CANDIDATES = ("llama-quantize", "quantize")

    def is_available(self) -> bool:
        return self._converter_tool(None) is not None

    def availability_reason(self) -> str | None:
        if self.is_available():
            return None
        return self._converter_missing_reason(None)

    def compatibility_report(
        self,
        manifest: ModelManifest,
        *,
        settings: LewLMSettings,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile: QuantizationProfile | None,
        cache_key: str,
        output_path: Path,
    ) -> ConversionCompatibilityReport:
        requested_profile = quantization_profile.model_copy(deep=True) if quantization_profile is not None else None
        resolved_profile = resolve_quantization_profile(
            policy=policy,
            custom_bits=custom_bits,
            requested_profile=quantization_profile,
        )
        profile_support = self._profile_support(resolved_profile)
        quantization_mode = self._gguf_output_label(policy=policy, profile=resolved_profile)
        artifact_plans = [
            self._artifact_plan(
                manifest=manifest,
                policy=policy,
                resolved_profile=resolved_profile,
            ),
        ]
        source_path = Path(manifest.source_path)
        jang_source = is_jang_bundle(source_path)
        warnings: list[str] = []
        if jang_source:
            warnings.append(
                "Source bundle uses JANG-packed safetensors; LewLM will normalize it to standard HF safetensors before GGUF export."
            )
        if ModelModality.VISION in manifest.modality or ModelModality.MULTIMODAL in manifest.modality:
            warnings.append(
                "GGUF conversion is reported as a text-capable artifact first; multimodal/mmproj runtime support remains probe-gated."
            )
        if manifest.format_type == ModelFormat.GGUF:
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                target_format=ModelFormat.GGUF.value,
                backend_name=self.name,
                can_convert=False,
                already_runnable=True,
                reason="Model is already in GGUF format and can stay on the llama.cpp runtime path.",
                cache_key=cache_key,
                output_path=manifest.source_path,
                quantization_mode=manifest.quantization,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                artifact_plans=artifact_plans,
            )
        if manifest.format_type != ModelFormat.HUGGINGFACE:
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                target_format=ModelFormat.GGUF.value,
                backend_name=self.name,
                can_convert=False,
                reason="The llama.cpp GGUF exporter supports Hugging Face-style source bundles.",
                cache_key=cache_key,
                output_path=str(output_path),
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                artifact_plans=artifact_plans,
            )
        if (ModelModality.VISION in manifest.modality or ModelModality.MULTIMODAL in manifest.modality) and not jang_source:
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                target_format=ModelFormat.GGUF.value,
                backend_name=self.name,
                can_convert=False,
                reason=(
                    "LewLM's packaged GGUF conversion path currently supports text and semantic Hugging Face "
                    "bundles; vision/multimodal sources still need a model-specific exporter or bridge-backed runtime."
                ),
                cache_key=cache_key,
                output_path=str(output_path),
                quantization_mode=quantization_mode,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                artifact_plans=artifact_plans,
            )
        if (
            ModelModality.TEXT not in manifest.modality
            and ModelModality.EMBEDDING not in manifest.modality
            and ModelModality.RERANK not in manifest.modality
        ):
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                target_format=ModelFormat.GGUF.value,
                backend_name=self.name,
                can_convert=False,
                reason="The llama.cpp GGUF exporter supports text-like Hugging Face bundles only.",
                cache_key=cache_key,
                output_path=str(output_path),
                quantization_mode=quantization_mode,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                artifact_plans=artifact_plans,
            )
        if jang_source and (missing_packages := missing_jang_dependencies()):
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                target_format=ModelFormat.GGUF.value,
                backend_name=self.name,
                can_convert=False,
                reason=(
                    "JANG-packed safetensors require optional conversion dependencies before LewLM can normalize "
                    "the source for llama.cpp."
                ),
                cache_key=cache_key,
                output_path=str(output_path),
                quantization_mode=quantization_mode,
                custom_bits=custom_bits,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                artifact_plans=artifact_plans,
                warnings=[*warnings, f"Missing packages: {', '.join(missing_packages)}."],
            )
        if not profile_support.supported:
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                target_format=ModelFormat.GGUF.value,
                backend_name=self.name,
                can_convert=False,
                reason=profile_support.reason,
                cache_key=cache_key,
                output_path=str(output_path),
                quantization_mode=quantization_mode,
                custom_bits=custom_bits,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                artifact_plans=artifact_plans,
                warnings=[*warnings, *profile_support.warnings],
            )
        if not settings.allow_outbound_network and not Path(manifest.source_path).exists():
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                target_format=ModelFormat.GGUF.value,
                backend_name=self.name,
                can_convert=False,
                reason="GGUF conversion requires a local Hugging Face source path when outbound network access is disabled.",
                cache_key=cache_key,
                output_path=str(output_path),
                quantization_mode=quantization_mode,
                custom_bits=custom_bits,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                artifact_plans=artifact_plans,
                warnings=warnings,
            )
        if self._converter_tool(settings) is None:
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                target_format=ModelFormat.GGUF.value,
                backend_name=self.name,
                can_convert=False,
                reason=self._converter_missing_reason(settings),
                cache_key=cache_key,
                output_path=str(output_path),
                quantization_mode=quantization_mode,
                custom_bits=custom_bits,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                artifact_plans=artifact_plans,
                warnings=warnings,
            )
        if (
            self._gguf_quantization_type(policy=policy, profile=resolved_profile) is not None
            and self._quantizer_tool(settings) is None
        ):
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                target_format=ModelFormat.GGUF.value,
                backend_name=self.name,
                can_convert=False,
                reason=self._quantizer_missing_reason(settings),
                cache_key=cache_key,
                output_path=str(output_path),
                quantization_mode=quantization_mode,
                custom_bits=custom_bits,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                artifact_plans=artifact_plans,
                warnings=warnings,
            )
        return ConversionCompatibilityReport(
            model_id=manifest.model_id,
            source_format=manifest.format_type,
            target_format=ModelFormat.GGUF.value,
            backend_name=self.name,
            can_convert=True,
            reason=(
                "Model can be normalized from JANG-packed HF weights and exported to GGUF with the configured llama.cpp conversion tools."
                if jang_source
                else "Model can be exported to GGUF with the configured llama.cpp conversion tools."
            ),
            cache_key=cache_key,
            output_path=str(output_path),
            quantization_mode=quantization_mode,
            custom_bits=custom_bits,
            requested_profile=requested_profile,
            resolved_profile=resolved_profile,
            profile_support=[profile_support],
            artifact_plans=artifact_plans,
            warnings=warnings,
        )

    def convert(
        self,
        manifest: ModelManifest,
        *,
        settings: LewLMSettings,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile: QuantizationProfile | None,
        output_path: Path,
        work_dir: Path,
    ) -> ConversionExecutionResult:
        resolved_profile = resolve_quantization_profile(
            policy=policy,
            custom_bits=custom_bits,
            requested_profile=quantization_profile,
        )
        profile_support = self._profile_support(resolved_profile)
        if not profile_support.supported:
            raise ConversionError(profile_support.reason)
        converter = self._converter_tool(settings)
        if converter is None:
            raise ConversionError(self._converter_missing_reason(settings))
        quantization_type = self._gguf_quantization_type(policy=policy, profile=resolved_profile)
        quantizer = self._quantizer_tool(settings) if quantization_type is not None else None
        if quantization_type is not None and quantizer is None:
            raise ConversionError(self._quantizer_missing_reason(settings))

        output_path.mkdir(parents=True, exist_ok=True)
        outtype = self._gguf_outtype(resolved_profile)
        final_filename = self._output_filename(manifest=manifest, policy=policy, profile=resolved_profile)
        final_path = output_path / final_filename
        intermediate_path = final_path if quantization_type is None else work_dir / f"{final_path.stem}-{outtype}.gguf"
        conversion_source_path = Path(manifest.source_path)
        logs: list[str] = []
        source_metadata: dict[str, Any] = {}
        if is_jang_bundle(conversion_source_path):
            normalized = normalize_jang_bundle(conversion_source_path, work_dir / "jang-normalized-hf")
            conversion_source_path = normalized.source_path
            logs.extend(normalized.logs)
            source_metadata = {
                "source_preprocessing": "jang_normalization",
                **normalized.metadata,
            }
        converter_command = [
            *converter.command,
            str(conversion_source_path),
            "--outfile",
            str(intermediate_path),
            "--outtype",
            outtype,
        ]
        if source_metadata or (manifest.estimated_memory_mb is not None and manifest.estimated_memory_mb >= 8192):
            converter_command.append("--use-temp-file")
        logs.extend(
            self._run_command(
                converter_command,
                cwd=converter.cwd or work_dir,
                env=converter.env,
                failure_message="llama.cpp HF-to-GGUF export failed.",
            ),
        )
        if not intermediate_path.exists():
            raise ConversionError(
                "llama.cpp conversion completed without producing a GGUF file.",
                details={"expected_output_path": str(intermediate_path)},
            )
        if quantization_type is not None:
            assert quantizer is not None
            logs.extend(
                self._run_command(
                    [*quantizer.command, str(intermediate_path), str(final_path), quantization_type],
                    cwd=quantizer.cwd or work_dir,
                    env=quantizer.env,
                    failure_message="llama.cpp GGUF quantization failed.",
                ),
            )
            if not final_path.exists():
                raise ConversionError(
                    "llama.cpp quantization completed without producing a GGUF file.",
                    details={"expected_output_path": str(final_path), "quantization": quantization_type},
                )
        artifact = self._artifact_plan(manifest=manifest, policy=policy, resolved_profile=resolved_profile)
        return ConversionExecutionResult(
            output_path=output_path,
            logs=logs,
            artifacts=(
                ConversionExecutionArtifact(
                    artifact_key=artifact.artifact_key,
                    role=artifact.role,
                    display_name=artifact.display_name,
                    output_path=final_path,
                    format_type=ModelFormat.GGUF,
                    modality=artifact.modality,
                    runtime_affinity=artifact.runtime_affinity,
                    metadata={**artifact.metadata, **source_metadata},
                ),
            ),
        )

    @classmethod
    def _artifact_plan(
        cls,
        *,
        manifest: ModelManifest,
        policy: ConversionPolicy,
        resolved_profile: QuantizationProfile,
    ) -> LayeredConversionArtifact:
        label = cls._gguf_output_label(policy=policy, profile=resolved_profile) or "unsupported"
        metadata: dict[str, Any] = {"target_format": ModelFormat.GGUF.value}
        source_path = Path(manifest.source_path)
        if is_jang_bundle(source_path):
            metadata["source_preprocessing"] = "jang_normalization"
        if ModelModality.VISION in manifest.modality or ModelModality.MULTIMODAL in manifest.modality:
            metadata["multimodal_support"] = "probe_gated"
        return LayeredConversionArtifact(
            artifact_key="gguf",
            role=ModelArtifactRole.STANDALONE,
            display_name=f"{manifest.display_name} ({label} gguf)",
            relative_path=cls._output_filename(manifest=manifest, policy=policy, profile=resolved_profile),
            format_type=ModelFormat.GGUF,
            modality=cls._gguf_modalities(manifest.modality),
            runtime_affinity=(RuntimeAffinity.LLAMACPP,),
            quantization=label,
            quantization_profile=resolved_profile,
            metadata=metadata,
        )

    @staticmethod
    def _gguf_modalities(modalities: tuple[ModelModality, ...]) -> tuple[ModelModality, ...]:
        retained = [
            modality
            for modality in modalities
            if modality in {ModelModality.TEXT, ModelModality.EMBEDDING, ModelModality.RERANK}
        ]
        return tuple(dict.fromkeys(retained or [ModelModality.TEXT]))

    def _profile_support(self, profile: QuantizationProfile) -> ConversionProfileSupport:
        if profile.strategy != QuantizationStrategy.WEIGHT_ONLY:
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="The packaged llama.cpp GGUF exporter currently supports weight-only conversion profiles.",
            )
        if profile.layer_overrides:
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="llama.cpp GGUF conversion cannot materialize per-layer precision overrides through this backend.",
            )
        if profile.activation_precision is not None:
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="Activation-aware precision requires a calibration-aware quantizer; the packaged GGUF path is weight-only.",
                requires_calibration=True,
            )
        if profile.kv_cache_precision is not None:
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="GGUF export changes model weights; runtime KV-cache quantization is configured separately at inference time.",
            )
        if profile.external_quantizer is not None:
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="External adaptive quantizers need an explicit backend integration before LewLM can materialize them.",
                requires_external_quantizer=True,
            )
        custom_bits = profile.metadata.get("custom_bits")
        if (
            isinstance(custom_bits, int)
            and self._gguf_quantization_type(policy=ConversionPolicy.BALANCED, profile=profile) is None
            and self._gguf_output_label(policy=ConversionPolicy.BALANCED, profile=profile) is None
        ):
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason=(
                    "llama.cpp GGUF conversion does not have a LewLM-supported "
                    f"quantization mapping for {custom_bits}-bit weights."
                ),
            )
        if self._gguf_output_label(policy=ConversionPolicy.BALANCED, profile=profile) is None:
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="The requested weight precision does not map to a supported llama.cpp GGUF output type.",
            )
        return ConversionProfileSupport(
            requested_profile=profile,
            supported=True,
            reason="The requested weight-only profile maps to a llama.cpp GGUF export or quantization type.",
        )

    @classmethod
    def _output_filename(
        cls,
        *,
        manifest: ModelManifest,
        policy: ConversionPolicy,
        profile: QuantizationProfile,
    ) -> str:
        label = cls._gguf_output_label(policy=policy, profile=profile) or "gguf"
        return f"{cls._safe_stem(manifest.display_name)}-{label}.gguf"

    @classmethod
    def _gguf_output_label(cls, *, policy: ConversionPolicy, profile: QuantizationProfile) -> str | None:
        quantization_type = cls._gguf_quantization_type(policy=policy, profile=profile)
        if quantization_type is not None:
            return quantization_type.casefold()
        outtype = cls._gguf_outtype(profile)
        if isinstance(profile.metadata.get("custom_bits"), int):
            return outtype if outtype in {"q8_0"} else None
        return outtype if outtype in {"f16", "bf16", "f32"} else None

    @staticmethod
    def _gguf_outtype(profile: QuantizationProfile) -> str:
        if profile.metadata.get("custom_bits") == 8 or profile.weight_precision == QuantizationPrecision.INT8:
            return "q8_0"
        if profile.weight_precision == QuantizationPrecision.BF16:
            return "bf16"
        if profile.weight_precision == QuantizationPrecision.FP32:
            return "f32"
        return "f16"

    @staticmethod
    def _gguf_quantization_type(*, policy: ConversionPolicy, profile: QuantizationProfile) -> str | None:
        custom_bits = profile.metadata.get("custom_bits")
        if isinstance(custom_bits, int):
            return {
                2: "Q2_K",
                3: "Q3_K_M",
                4: "Q4_K_M",
                5: "Q5_K_M",
                6: "Q6_K",
            }.get(custom_bits)
        if profile.weight_precision in {None, QuantizationPrecision.FP16, QuantizationPrecision.BF16, QuantizationPrecision.FP32}:
            return None
        if profile.weight_precision == QuantizationPrecision.INT2:
            return "Q2_K"
        if profile.weight_precision == QuantizationPrecision.INT3:
            return "Q3_K_M"
        if profile.weight_precision == QuantizationPrecision.INT4:
            return "Q3_K_M" if policy == ConversionPolicy.MAX_FIT else "Q4_K_M"
        if profile.weight_precision == QuantizationPrecision.INT6:
            return "Q6_K"
        if profile.weight_precision == QuantizationPrecision.INT8:
            return None
        return None

    @staticmethod
    def _safe_stem(value: str) -> str:
        stem = "".join(character if character.isalnum() else "-" for character in value.casefold()).strip("-")
        while "--" in stem:
            stem = stem.replace("--", "-")
        return stem or "model"

    @classmethod
    def _converter_tool(cls, settings: LewLMSettings | None) -> "_ResolvedTool | None":
        configured = settings.llamacpp_convert_hf_to_gguf_path if settings is not None else None
        return cls._resolve_tool(configured, cls._CONVERTER_CANDIDATES, settings=settings, tool_kind="converter")

    @classmethod
    def _quantizer_tool(cls, settings: LewLMSettings | None) -> "_ResolvedTool | None":
        configured = settings.llamacpp_quantize_path if settings is not None else None
        return cls._resolve_tool(configured, cls._QUANTIZER_CANDIDATES, settings=settings, tool_kind="quantizer")

    @classmethod
    def _resolve_tool(
        cls,
        configured_path: Path | None,
        candidates: tuple[str, ...],
        *,
        settings: LewLMSettings | None,
        tool_kind: str,
    ) -> "_ResolvedTool | None":
        if configured_path is not None:
            if configured_path.exists():
                return cls._tool_from_path(configured_path)
            return None
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved is not None:
                return cls._tool_from_path(Path(resolved))
        if settings is not None:
            for candidate in cls._local_llamacpp_tool_candidates(settings=settings, tool_kind=tool_kind):
                if candidate.exists():
                    return cls._tool_from_path(candidate)
        return None

    @staticmethod
    def _tool_from_path(path: Path) -> "_ResolvedTool":
        command = (sys.executable, str(path)) if path.suffix.casefold() == ".py" else (str(path),)
        llama_cpp_root = _llamacpp_repo_root_for_tool(path)
        env = None
        cwd = None
        if path.suffix.casefold() == ".py" and llama_cpp_root is not None:
            cwd = llama_cpp_root
            python_paths = [str(llama_cpp_root), str(llama_cpp_root / "gguf-py")]
            existing_pythonpath = os.environ.get("PYTHONPATH")
            if existing_pythonpath:
                python_paths.append(existing_pythonpath)
            env = {"PYTHONPATH": os.pathsep.join(python_paths)}
        return _ResolvedTool(command=command, display_name=str(path), cwd=cwd, env=env)

    @staticmethod
    def _local_llamacpp_tool_candidates(*, settings: LewLMSettings, tool_kind: str) -> tuple[Path, ...]:
        root = settings.data_dir / "tools" / "llama.cpp"
        if tool_kind == "converter":
            return (
                root / "convert_hf_to_gguf.py",
                root / "convert.py",
            )
        executable_names = (
            "llama-quantize.exe",
            "quantize.exe",
            "llama-quantize",
            "quantize",
        )
        search_dirs = (
            root,
            root / "build" / "bin" / "Release",
            root / "build" / "bin",
            root / "build" / "Release",
        )
        return tuple(directory / executable for directory in search_dirs for executable in executable_names)

    @staticmethod
    def _converter_missing_reason(settings: LewLMSettings | None) -> str:
        configured = settings.llamacpp_convert_hf_to_gguf_path if settings is not None else None
        if configured is not None:
            return f"Configured llama.cpp HF-to-GGUF converter was not found at `{configured}`."
        return (
            "Could not find llama.cpp `convert_hf_to_gguf.py`. Install or build llama.cpp and set "
            "`LEWLM_LLAMACPP_CONVERT_HF_TO_GGUF_PATH` to the converter script."
        )

    @staticmethod
    def _quantizer_missing_reason(settings: LewLMSettings | None) -> str:
        configured = settings.llamacpp_quantize_path if settings is not None else None
        if configured is not None:
            return f"Configured llama.cpp quantizer was not found at `{configured}`."
        return (
            "Could not find llama.cpp `llama-quantize`. Install or build llama.cpp and set "
            "`LEWLM_LLAMACPP_QUANTIZE_PATH` to the quantizer executable."
        )

    @staticmethod
    def _run_command(command: list[str], *, cwd: Path, failure_message: str, env: dict[str, str] | None = None) -> list[str]:
        logs = [f"Running llama.cpp conversion command: {' '.join(shlex.quote(part) for part in command)}"]
        subprocess_env = None
        if env is not None:
            subprocess_env = {**os.environ, **env}
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
            **({"env": subprocess_env} if subprocess_env is not None else {}),
        )
        if completed.stdout:
            logs.extend(line for line in completed.stdout.splitlines() if line)
        if completed.stderr:
            logs.extend(line for line in completed.stderr.splitlines() if line)
        if completed.returncode != 0:
            raise ConversionError(
                failure_message,
                details={"returncode": completed.returncode, "logs": logs[-10:]},
            )
        return logs


def _llamacpp_repo_root_for_tool(path: Path) -> Path | None:
    """Return the llama.cpp checkout root for tools inside a local clone."""

    for candidate in (path.parent, *path.parents):
        if (candidate / "gguf-py").exists() and (candidate / "conversion").exists():
            return candidate
    return None


class MLXConversionBackend:
    """Use `mlx_lm.convert` for compatible local Hugging Face model bundles."""

    name = "mlx_lm"

    def is_available(self) -> bool:
        if not self._supports_host_platform():
            return False
        return any(importlib.util.find_spec(module_name) is not None for module_name in ("mlx_lm", "mlx_vlm"))

    def availability_reason(self) -> str | None:
        if not self._supports_host_platform():
            return "MLX conversion is only supported on Apple Silicon macOS hosts."
        if self.is_available():
            return None
        return "Neither mlx-lm nor mlx-vlm is installed"

    def compatibility_report(
        self,
        manifest: ModelManifest,
        *,
        settings: LewLMSettings,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile: QuantizationProfile | None,
        cache_key: str,
        output_path: Path,
    ) -> ConversionCompatibilityReport:
        requested_profile = quantization_profile.model_copy(deep=True) if quantization_profile is not None else None
        resolved_profile = resolve_quantization_profile(
            policy=policy,
            custom_bits=custom_bits,
            requested_profile=quantization_profile,
        )
        conversion_plans = self._conversion_plans(manifest, output_path=output_path, resolved_profile=resolved_profile)
        primary_backend = conversion_plans[0].backend if conversion_plans else self._conversion_backend(manifest)
        artifact_plans = [plan.as_artifact_plan() for plan in conversion_plans]
        warnings: list[str] = []
        profile_support = self._profile_support(resolved_profile)
        if manifest.format_type == ModelFormat.MLX:
            if not self._supports_host_platform():
                return ConversionCompatibilityReport(
                    model_id=manifest.model_id,
                    source_format=manifest.format_type,
                    backend_name=primary_backend.name,
                    can_convert=False,
                    reason="Model is already in MLX format, but MLX execution is only supported on Apple Silicon macOS hosts.",
                    cache_key=cache_key,
                    output_path=str(manifest.source_path),
                    requested_profile=requested_profile,
                    resolved_profile=resolved_profile,
                    profile_support=[profile_support],
                    layered_output=len(artifact_plans) > 1,
                    artifact_plans=artifact_plans,
                )
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                backend_name=primary_backend.name,
                can_convert=False,
                already_runnable=True,
                reason="Model is already in MLX format and does not need conversion.",
                cache_key=cache_key,
                output_path=str(manifest.source_path),
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                layered_output=len(artifact_plans) > 1,
                artifact_plans=artifact_plans,
            )
        if manifest.format_type == ModelFormat.GGUF:
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                backend_name=primary_backend.name,
                can_convert=False,
                reason="GGUF models stay on the llama.cpp runtime path and are not converted to MLX here.",
                cache_key=cache_key,
                output_path=str(output_path),
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                layered_output=len(artifact_plans) > 1,
                artifact_plans=artifact_plans,
            )
        if manifest.format_type != ModelFormat.HUGGINGFACE:
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                backend_name=primary_backend.name,
                can_convert=False,
                reason="This conversion pipeline currently supports Hugging Face-style text bundles only.",
                cache_key=cache_key,
                output_path=str(output_path),
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                layered_output=len(artifact_plans) > 1,
                artifact_plans=artifact_plans,
            )
        if ModelModality.TEXT not in manifest.modality and ModelModality.VISION not in manifest.modality:
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                backend_name=primary_backend.name,
                can_convert=False,
                reason="This conversion pipeline currently supports text- or vision-capable models only.",
                cache_key=cache_key,
                output_path=str(output_path),
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                layered_output=len(artifact_plans) > 1,
                artifact_plans=artifact_plans,
            )
        availability_reason = self.availability_reason()
        if availability_reason is not None:
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                backend_name=primary_backend.name,
                can_convert=False,
                reason=availability_reason,
                cache_key=cache_key,
                output_path=str(output_path),
                quantization_mode=quantization_mode_from_profile(resolved_profile),
                custom_bits=custom_bits,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                layered_output=len(artifact_plans) > 1,
                artifact_plans=artifact_plans,
            )
        if custom_bits is not None and policy != ConversionPolicy.CUSTOM_BITS:
            warnings.append("Custom bits were provided without selecting the `custom_bits` policy.")
        if not profile_support.supported:
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                backend_name=primary_backend.name,
                can_convert=False,
                reason=profile_support.reason,
                cache_key=cache_key,
                output_path=str(output_path),
                quantization_mode=quantization_mode_from_profile(resolved_profile),
                custom_bits=custom_bits,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                layered_output=len(artifact_plans) > 1,
                artifact_plans=artifact_plans,
                warnings=[*warnings, *profile_support.warnings],
            )
        if not settings.allow_outbound_network and not Path(manifest.source_path).exists():
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                backend_name=primary_backend.name,
                can_convert=False,
                reason="Conversion requires a local model path when outbound network access is disabled.",
                cache_key=cache_key,
                output_path=str(output_path),
                quantization_mode=quantization_mode_from_profile(resolved_profile),
                custom_bits=custom_bits,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                layered_output=len(artifact_plans) > 1,
                artifact_plans=artifact_plans,
            )
        missing_backends = [
            plan.backend.package_name
            for plan in conversion_plans
            if not self._conversion_backend_available(plan.backend)
        ]
        if missing_backends:
            missing = ", ".join(dict.fromkeys(missing_backends))
            return ConversionCompatibilityReport(
                model_id=manifest.model_id,
                source_format=manifest.format_type,
                backend_name=primary_backend.name,
                can_convert=False,
                reason=(
                    f"{missing} is not installed"
                    if len(dict.fromkeys(missing_backends)) == 1
                    else f"Paired conversion requires all local MLX backends: missing {missing}"
                ),
                cache_key=cache_key,
                output_path=str(output_path),
                quantization_mode=quantization_mode_from_profile(resolved_profile),
                custom_bits=custom_bits,
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                profile_support=[profile_support],
                layered_output=len(artifact_plans) > 1,
                artifact_plans=artifact_plans,
            )
        target_description = (
            "paired multimodal and text MLX artifacts"
            if len(artifact_plans) > 1
            else f"MLX with the local {primary_backend.package_name} backend"
        )
        return ConversionCompatibilityReport(
            model_id=manifest.model_id,
            source_format=manifest.format_type,
            backend_name=primary_backend.name,
            can_convert=True,
            reason=f"Model can be converted to {target_description}.",
            cache_key=cache_key,
            output_path=str(output_path),
            quantization_mode=quantization_mode_from_profile(resolved_profile),
            custom_bits=custom_bits,
            requested_profile=requested_profile,
            resolved_profile=resolved_profile,
            profile_support=[profile_support],
            layered_output=len(artifact_plans) > 1,
            artifact_plans=artifact_plans,
            warnings=warnings,
        )

    def convert(
        self,
        manifest: ModelManifest,
        *,
        settings: LewLMSettings,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile: QuantizationProfile | None,
        output_path: Path,
        work_dir: Path,
    ) -> ConversionExecutionResult:
        resolved_profile = resolve_quantization_profile(
            policy=policy,
            custom_bits=custom_bits,
            requested_profile=quantization_profile,
        )
        conversion_plans = self._conversion_plans(manifest, output_path=output_path, resolved_profile=resolved_profile)
        if len(conversion_plans) == 1 and conversion_plans[0].relative_output_path in {"", "."}:
            logs = self._run_conversion_command(
                conversion_backend=conversion_plans[0].backend,
                manifest=manifest,
                output_path=output_path,
                work_dir=work_dir,
                resolved_profile=resolved_profile,
            )
            return ConversionExecutionResult(
                output_path=output_path,
                logs=logs,
                artifacts=(
                    ConversionExecutionArtifact(
                        artifact_key=conversion_plans[0].artifact_key,
                        role=conversion_plans[0].role,
                        display_name=conversion_plans[0].display_name,
                        output_path=output_path,
                        format_type=ModelFormat.MLX,
                        modality=conversion_plans[0].modality,
                        runtime_affinity=conversion_plans[0].runtime_affinity,
                        derived_from=conversion_plans[0].derived_from,
                    ),
                ),
            )

        output_path.mkdir(parents=True, exist_ok=True)
        logs: list[str] = []
        artifacts: list[ConversionExecutionArtifact] = []
        for plan in conversion_plans:
            artifact_output_path = output_path / plan.relative_output_path
            artifact_output_path.parent.mkdir(parents=True, exist_ok=True)
            logs.extend(
                self._run_conversion_command(
                    conversion_backend=plan.backend,
                    manifest=manifest,
                    output_path=artifact_output_path,
                    work_dir=work_dir,
                    resolved_profile=resolved_profile,
                ),
            )
            artifacts.append(
                ConversionExecutionArtifact(
                    artifact_key=plan.artifact_key,
                    role=plan.role,
                    display_name=plan.display_name,
                    output_path=artifact_output_path,
                    format_type=ModelFormat.MLX,
                    modality=plan.modality,
                    runtime_affinity=plan.runtime_affinity,
                    derived_from=plan.derived_from,
                ),
            )
        return ConversionExecutionResult(output_path=output_path, logs=logs, artifacts=tuple(artifacts))

    def _conversion_backend(self, manifest: ModelManifest) -> "_MLXConverterTarget":
        if ModelModality.VISION in manifest.modality:
            return _MLXConverterTarget(name="mlx_vlm", package_name="mlx-vlm", module_name="mlx_vlm")
        return _MLXConverterTarget(name="mlx_lm", package_name="mlx-lm", module_name="mlx_lm")

    def _conversion_plans(
        self,
        manifest: ModelManifest,
        *,
        output_path: Path,
        resolved_profile: QuantizationProfile,
    ) -> list["_MLXConversionPlan"]:
        if self._supports_dual_artifacts(manifest):
            quantization = quantization_mode_from_profile(resolved_profile) or manifest.quantization
            return [
                _MLXConversionPlan(
                    artifact_key="multimodal",
                    role=ModelArtifactRole.MULTIMODAL_RUNNABLE,
                    display_name=f"{manifest.display_name} (multimodal mlx)",
                    relative_output_path="multimodal",
                    backend=_MLXConverterTarget(name="mlx_vlm", package_name="mlx-vlm", module_name="mlx_vlm"),
                    modality=manifest.modality,
                    runtime_affinity=(RuntimeAffinity.MLX_VISION,),
                    quantization=quantization,
                    quantization_profile=resolved_profile,
                ),
                _MLXConversionPlan(
                    artifact_key="text",
                    role=ModelArtifactRole.TEXT_RUNNABLE,
                    display_name=f"{manifest.display_name} (text mlx)",
                    relative_output_path="text",
                    backend=_MLXConverterTarget(name="mlx_lm", package_name="mlx-lm", module_name="mlx_lm"),
                    modality=(ModelModality.TEXT,),
                    runtime_affinity=(RuntimeAffinity.MLX_TEXT,),
                    derived_from="multimodal",
                    quantization=quantization,
                    quantization_profile=resolved_profile,
                ),
            ]
        backend = self._conversion_backend(manifest)
        return [
            _MLXConversionPlan(
                artifact_key="standalone",
                role=ModelArtifactRole.STANDALONE,
                display_name=manifest.display_name,
                relative_output_path="." if output_path.suffix == "" else "",
                backend=backend,
                modality=manifest.modality,
                runtime_affinity=(
                    (RuntimeAffinity.MLX_VISION,)
                    if backend.name == "mlx_vlm"
                    else (RuntimeAffinity.MLX_TEXT,)
                ),
                quantization=quantization_mode_from_profile(resolved_profile) or manifest.quantization,
                quantization_profile=resolved_profile,
            ),
        ]

    @staticmethod
    def _supports_dual_artifacts(manifest: ModelManifest) -> bool:
        supported_families = {"gemma", "gemma4", "qwen", "qwen2", "qwen2vl"}
        architecture_family = manifest.architecture_family.casefold().replace("-", "").replace("_", "")
        return (
            manifest.format_type == ModelFormat.HUGGINGFACE
            and ModelModality.TEXT in manifest.modality
            and ModelModality.VISION in manifest.modality
            and architecture_family in supported_families
        )

    def _run_conversion_command(
        self,
        *,
        conversion_backend: "_MLXConverterTarget",
        manifest: ModelManifest,
        output_path: Path,
        work_dir: Path,
        resolved_profile: QuantizationProfile,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            conversion_backend.module_name,
            "convert",
            "--hf-path",
            manifest.source_path,
            "--mlx-path",
            str(output_path),
        ]
        if self._profile_uses_4bit_quantization(resolved_profile):
            command.append("-q")
            command.extend(["--q-bits", "4"])
        logs = [
            f"Running {conversion_backend.name} conversion command: {' '.join(shlex.quote(part) for part in command)}",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=work_dir,
            check=False,
        )
        if completed.stdout:
            logs.extend(line for line in completed.stdout.splitlines() if line)
        if completed.stderr:
            logs.extend(line for line in completed.stderr.splitlines() if line)
        if completed.returncode != 0:
            raise ConversionError(
                "MLX conversion command failed.",
                details={"returncode": completed.returncode, "logs": logs[-10:], "backend": conversion_backend.name},
            )
        if not output_path.exists():
            fallback_path = work_dir / "mlx_model"
            if fallback_path.exists():
                fallback_path.rename(output_path)
            else:
                raise ConversionError(
                    "MLX conversion completed without producing an output directory.",
                    details={"expected_output_path": str(output_path), "backend": conversion_backend.name},
                )
        return logs

    @staticmethod
    def _conversion_backend_available(conversion_backend: "_MLXConverterTarget") -> bool:
        return importlib.util.find_spec(conversion_backend.module_name) is not None

    def _profile_support(self, profile: QuantizationProfile) -> ConversionProfileSupport:
        if profile.strategy == QuantizationStrategy.WEIGHT_ONLY:
            if profile.layer_overrides:
                return ConversionProfileSupport(
                    requested_profile=profile,
                    supported=False,
                    reason="The bundled MLX converter cannot apply per-layer overrides to a weight-only conversion profile.",
                )
            if profile.activation_precision is not None:
                return ConversionProfileSupport(
                    requested_profile=profile,
                    supported=False,
                    reason="Weight-only MLX conversion cannot also record activation precision without an activation-aware backend.",
                )
            if profile.weight_precision in {None, QuantizationPrecision.FP16}:
                return ConversionProfileSupport(
                    requested_profile=profile,
                    supported=True,
                    reason="Unquantized MLX conversion is supported.",
                )
            if profile.weight_precision == QuantizationPrecision.INT4:
                return ConversionProfileSupport(
                    requested_profile=profile,
                    supported=True,
                    reason="4-bit MLX conversion is supported through mlx_lm/mlx_vlm.",
                )
            if profile.name == ConversionPolicy.CUSTOM_BITS.value and profile.metadata.get("custom_bits") is not None:
                return ConversionProfileSupport(
                    requested_profile=profile,
                    supported=False,
                    reason="The current MLX conversion integration only supports 4-bit custom quantization requests.",
                )
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="The bundled MLX converter currently supports only unquantized or 4-bit weight-only conversion.",
            )
        if profile.strategy == QuantizationStrategy.ACTIVATION_AWARE:
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="Activation-aware conversion needs calibration and an adaptive quantizer integration that this MLX backend does not currently expose.",
                requires_calibration=True,
            )
        if profile.strategy == QuantizationStrategy.MIXED_PRECISION:
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="Per-layer mixed-precision conversion metadata can be recorded, but the bundled MLX converter cannot materialize layer overrides during conversion.",
            )
        if profile.strategy == QuantizationStrategy.HYBRID_FP8:
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="The current MLX conversion path does not expose native FP8 or hybrid mixed-precision conversion on Apple Silicon.",
                requires_native_fp8=True,
            )
        external_quantizer = profile.external_quantizer
        if external_quantizer is None:
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="External adaptive profiles require an explicit external quantizer selection.",
                requires_external_quantizer=True,
            )
        missing_packages = [
            package_name
            for package_name in dict.fromkeys(
                [
                    *external_quantizer.required_packages,
                    *([external_quantizer.module] if external_quantizer.module else []),
                ],
            )
            if importlib.util.find_spec(package_name) is None
        ]
        if missing_packages:
            return ConversionProfileSupport(
                requested_profile=profile,
                supported=False,
                reason="The requested external adaptive quantizer is not installed on this host.",
                missing_packages=missing_packages,
                requires_external_quantizer=True,
            )
        return ConversionProfileSupport(
            requested_profile=profile,
            supported=False,
            reason="LewLM records external adaptive quantizer selections and compatibility details, but this build does not bundle an execution adapter for them.",
            requires_external_quantizer=True,
        )

    @staticmethod
    def _profile_uses_4bit_quantization(profile: QuantizationProfile) -> bool:
        return (
            profile.strategy == QuantizationStrategy.WEIGHT_ONLY
            and profile.weight_precision == QuantizationPrecision.INT4
        )

    @staticmethod
    def _supports_host_platform() -> bool:
        return platform.system() == "Darwin" and platform.machine().casefold() == "arm64"


class AutoConversionBackend:
    """Select the best packaged conversion backend for the current host and model."""

    name = "auto"

    def __init__(self, backends: tuple[ConversionBackend, ...] | None = None) -> None:
        self.backends = backends or (LlamaCppConversionBackend(), MLXConversionBackend())

    def is_available(self) -> bool:
        return any(backend.is_available() for backend in self.backends)

    def availability_reason(self) -> str | None:
        if self.is_available():
            return None
        reasons = [backend.availability_reason() for backend in self.backends]
        return "; ".join(reason for reason in reasons if reason) or "No packaged conversion backend is available."

    def compatibility_report(
        self,
        manifest: ModelManifest,
        *,
        settings: LewLMSettings,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile: QuantizationProfile | None,
        cache_key: str,
        output_path: Path,
    ) -> ConversionCompatibilityReport:
        reports = [
            backend.compatibility_report(
                manifest,
                settings=settings,
                policy=policy,
                custom_bits=custom_bits,
                quantization_profile=quantization_profile,
                cache_key=cache_key,
                output_path=output_path,
            )
            for backend in self._ordered_backends(manifest)
        ]
        for report in reports:
            if report.already_runnable or report.can_convert:
                return report
        return self._fallback_report(manifest, reports)

    def convert(
        self,
        manifest: ModelManifest,
        *,
        settings: LewLMSettings,
        policy: ConversionPolicy,
        custom_bits: int | None,
        quantization_profile: QuantizationProfile | None,
        output_path: Path,
        work_dir: Path,
    ) -> ConversionExecutionResult:
        for backend in self._ordered_backends(manifest):
            report = backend.compatibility_report(
                manifest,
                settings=settings,
                policy=policy,
                custom_bits=custom_bits,
                quantization_profile=quantization_profile,
                cache_key="conversion-run",
                output_path=output_path,
            )
            if report.can_convert:
                return backend.convert(
                    manifest,
                    settings=settings,
                    policy=policy,
                    custom_bits=custom_bits,
                    quantization_profile=quantization_profile,
                    output_path=output_path,
                    work_dir=work_dir,
                )
        compatibility = self.compatibility_report(
            manifest,
            settings=settings,
            policy=policy,
            custom_bits=custom_bits,
            quantization_profile=quantization_profile,
            cache_key="conversion-run",
            output_path=output_path,
        )
        raise ConversionError(compatibility.reason)

    def _ordered_backends(self, manifest: ModelManifest) -> tuple[ConversionBackend, ...]:
        if manifest.format_type == ModelFormat.MLX:
            return self._backends_by_name("mlx_lm", "llamacpp_gguf")
        if platform.system() == "Darwin" and platform.machine().casefold() == "arm64":
            return self._backends_by_name("mlx_lm", "llamacpp_gguf")
        return self._backends_by_name("llamacpp_gguf", "mlx_lm")

    def _backends_by_name(self, *names: str) -> tuple[ConversionBackend, ...]:
        by_name = {backend.name: backend for backend in self.backends}
        ordered = [by_name[name] for name in names if name in by_name]
        ordered.extend(backend for backend in self.backends if backend.name not in names)
        return tuple(ordered)

    @staticmethod
    def _fallback_report(manifest: ModelManifest, reports: list[ConversionCompatibilityReport]) -> ConversionCompatibilityReport:
        if manifest.format_type == ModelFormat.MLX:
            for report in reports:
                if report.backend_name in {"mlx_lm", "mlx_vlm"}:
                    return report
        if ModelModality.VISION in manifest.modality or ModelModality.MULTIMODAL in manifest.modality:
            for report in reports:
                if report.backend_name == "llamacpp_gguf":
                    return report
        return reports[0]


@dataclass(frozen=True, slots=True)
class _MLXConversionPlan:
    artifact_key: str
    role: ModelArtifactRole
    display_name: str
    relative_output_path: str
    backend: "_MLXConverterTarget"
    modality: tuple[ModelModality, ...]
    runtime_affinity: tuple[RuntimeAffinity, ...]
    derived_from: str | None = None
    quantization: str | None = None
    quantization_profile: QuantizationProfile | None = None

    def as_artifact_plan(self) -> LayeredConversionArtifact:
        return LayeredConversionArtifact(
            artifact_key=self.artifact_key,
            role=self.role,
            display_name=self.display_name,
            relative_path=self.relative_output_path,
            format_type=ModelFormat.MLX,
            modality=self.modality,
            runtime_affinity=self.runtime_affinity,
            derived_from=self.derived_from,
            quantization=self.quantization,
            quantization_profile=self.quantization_profile,
        )


@dataclass(frozen=True, slots=True)
class _MLXConverterTarget:
    name: str
    package_name: str
    module_name: str


def run_isolated_conversion(
    *,
    backend_module: str,
    backend_qualname: str,
    manifest_payload: dict[str, Any],
    settings_payload: dict[str, Any],
    policy: str,
    custom_bits: int | None,
    quantization_profile_payload: dict[str, Any] | None,
    output_path: str,
    work_dir: str,
) -> ConversionExecutionResult:
    """Reconstruct a backend in a subprocess and run a conversion there."""

    backend_class = _resolve_backend_class(backend_module, backend_qualname)
    backend = backend_class()
    manifest = ModelManifest.model_validate(manifest_payload)
    settings = LewLMSettings.model_validate(settings_payload)
    return backend.convert(
        manifest,
        settings=settings,
        policy=ConversionPolicy(policy),
        custom_bits=custom_bits,
        quantization_profile=(
            QuantizationProfile.model_validate(quantization_profile_payload)
            if quantization_profile_payload is not None
            else None
        ),
        output_path=Path(output_path),
        work_dir=Path(work_dir),
    )


def backend_descriptor(backend: ConversionBackend) -> tuple[str, str]:
    """Return an importable backend descriptor for subprocess reconstruction."""

    backend_class = type(backend)
    module_name = backend_class.__module__
    qualname = backend_class.__qualname__
    if "<locals>" in qualname:
        raise ConversionError(
            "Sandboxed conversion requires an importable backend class.",
            details={"backend_module": module_name, "backend_qualname": qualname},
        )
    return module_name, qualname


def _resolve_backend_class(module_name: str, qualname: str):
    module = importlib.import_module(module_name)
    current = module
    for attribute in qualname.split("."):
        current = getattr(current, attribute)
    return current
