"""Conversion backends for model format adaptation."""

from __future__ import annotations

import importlib
import importlib.util
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from lewlm.config.settings import LewLMSettings
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


class MLXConversionBackend:
    """Use `mlx_lm.convert` for compatible local Hugging Face model bundles."""

    name = "mlx_lm"

    def is_available(self) -> bool:
        return any(importlib.util.find_spec(module_name) is not None for module_name in ("mlx_lm", "mlx_vlm"))

    def availability_reason(self) -> str | None:
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
        supported_families = {"gemma", "qwen"}
        return (
            manifest.format_type == ModelFormat.HUGGINGFACE
            and ModelModality.TEXT in manifest.modality
            and ModelModality.VISION in manifest.modality
            and manifest.architecture_family.casefold() in supported_families
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
