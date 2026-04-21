from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import pytest

from lewlm import LewLM
from lewlm.api.message_normalization import normalize_chat_messages
from lewlm.api.schemas.chat import ChatMessage, InputImagePart, InputTextPart
from lewlm.config.settings import LewLMSettings
from lewlm.conversion.models import ConversionPolicy, JobStatus
from lewlm.core.contracts import GenerateAttachment, GenerateMessage, ModelFormat, ModelManifest, RuntimeAffinity
from lewlm.runtime.introspection import invoke_with_signature, resolve_backend_callable
from lewlm.runtime.mlx_vision.runtime import load_mlx_vlm_backend_client
from lewlm.telemetry.stats import _ensure_benchmark_multimodal_assets

_RUN_ENV = "LEWLM_RUN_REAL_MODEL_BENCHMARK"
_OUTPUT_DIR_ENV = "LEWLM_REAL_MODEL_BENCHMARK_OUTPUT_DIR"
_CONVERSION_TIMEOUT_ENV = "LEWLM_REAL_MODEL_BENCHMARK_CONVERSION_TIMEOUT_SECONDS"
_DEFAULT_CONVERSION_TIMEOUT_SECONDS = 7_200.0


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    score: float
    passed_checks: int
    total_checks: int
    details: dict[str, Any]


class ScoreFunction(Protocol):
    def __call__(self, output_text: str) -> BenchmarkScore: ...


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    description: str
    max_tokens: int
    scorer: ScoreFunction
    raw_messages: tuple[ChatMessage, ...] | None = None
    prepared_messages: tuple[GenerateMessage, ...] | None = None
    required_runtime_affinities: tuple[RuntimeAffinity, ...] = ()
    workload_class: str = "text_only"

    def __post_init__(self) -> None:
        if (self.raw_messages is None) == (self.prepared_messages is None):
            raise ValueError("Benchmark cases require exactly one message source.")

    def applies_to(self, manifest: ModelManifest) -> bool:
        return all(affinity in manifest.runtime_affinity for affinity in self.required_runtime_affinities)


@dataclass(frozen=True, slots=True)
class PreparedBenchmarkCase:
    case_id: str
    description: str
    messages: tuple[GenerateMessage, ...]
    max_tokens: int
    scorer: ScoreFunction
    workload_class: str
    preprocessing_seconds: float
    preparation_mode: str


@dataclass(slots=True)
class _LoadedMLXTextClient:
    model: Any
    tokenizer: Any | None


@dataclass(slots=True)
class _LoadedMLXVisionClient:
    model: Any
    processor: Any | None


class _DirectRunner(Protocol):
    runtime_name: str

    def run_case(self, case: PreparedBenchmarkCase) -> dict[str, Any]: ...

    def close(self) -> None: ...


@pytest.mark.long_running
def test_end_to_end_real_gemma_backend_benchmark(
    external_models_settings: LewLMSettings,
    external_models_root: Path,
    tmp_path: Path,
) -> None:
    _require_real_benchmark_environment()

    output_dir = _resolve_output_dir(tmp_path)
    benchmark_state_dir = output_dir / "runtime-state"
    if benchmark_state_dir.exists():
        shutil.rmtree(benchmark_state_dir)
    benchmark_settings = external_models_settings.with_updates(
        data_dir=benchmark_state_dir,
        models_dir=(external_models_root, benchmark_state_dir / "cache"),
        runtime_policy="keep_warm",
        conversion_sandbox_enabled=False,
    )
    benchmark_assets = _ensure_benchmark_multimodal_assets(benchmark_settings.benchmarks_dir)
    benchmark_cases = _benchmark_cases(benchmark_assets)

    with LewLM(benchmark_settings) as app:
        initial_scan = app.scan_models()
        source_manifests = _external_gemma_manifests(initial_scan.manifests, external_models_root)
        assert len(source_manifests) == 5

        report_models: list[dict[str, Any]] = []
        for source_manifest in source_manifests:
            runnable_manifest, conversion_report = _resolve_runnable_manifest(app, source_manifest)
            try:
                prepared_cases = _prepare_benchmark_cases(
                    app,
                    runnable_manifest,
                    benchmark_cases,
                    file_access_roots=(benchmark_settings.benchmarks_dir,),
                )
                optimized_report = _run_lewlm_benchmark(app, runnable_manifest, prepared_cases)
                direct_report = _run_direct_benchmark(runnable_manifest, prepared_cases)
                _apply_case_comparison_attribution(
                    optimized_report=optimized_report,
                    direct_report=direct_report,
                )
                report_models.append(
                    {
                        "source_manifest": source_manifest.model_dump(mode="json"),
                        "runnable_manifest": runnable_manifest.model_dump(mode="json"),
                        "conversion": conversion_report,
                        "case_ids": [case.case_id for case in prepared_cases],
                        "case_count": len(prepared_cases),
                        "optimized": optimized_report,
                        "direct": direct_report,
                    },
                )
            finally:
                _cleanup_transient_conversion_artifact(
                    conversion_report,
                    benchmark_state_dir=benchmark_state_dir,
                )

        runtime_stats = app.runtime_stats_sync().model_dump(mode="json")

    report_payload = {
        "benchmark": {
            "name": "gemma-real-model-backend-benchmark",
            "enabled_env": _RUN_ENV,
            "case_ids": [case.case_id for case in benchmark_cases],
            "case_count": len(benchmark_cases),
        },
        "summary": _build_summary(report_models),
        "runtime_stats": runtime_stats,
        "models": report_models,
    }
    report_payload["regression"] = _compare_to_previous_report(output_dir, report_payload)
    report_path = _write_report(output_dir, report_payload)

    assert report_path.exists()
    assert report_payload["regression"]["status"] != "failed"
    assert report_payload["summary"]["model_count"] == 5
    assert report_payload["summary"]["converted_model_count"] == 3
    assert report_payload["summary"]["runnable_source_model_count"] == 2
    for model_report in report_models:
        assert model_report["optimized"]["case_count"] == model_report["case_count"]
        assert model_report["direct"]["case_count"] == model_report["case_count"]
        assert model_report["optimized"]["non_empty_output_count"] == model_report["case_count"]
        assert model_report["direct"]["non_empty_output_count"] == model_report["case_count"]
        assert model_report["optimized"]["runtime_name"]
        assert model_report["direct"]["runtime_name"]
        assert model_report["optimized"]["average_score"] >= 0.0
        assert model_report["direct"]["average_score"] >= 0.0


def _require_real_benchmark_environment() -> None:
    if os.environ.get(_RUN_ENV) != "1":
        pytest.skip(
            f"Set {_RUN_ENV}=1 to run the long real-model Gemma backend benchmark suite.",
        )
    pytest.importorskip("llama_cpp")
    pytest.importorskip("mlx_lm")
    pytest.importorskip("mlx_vlm")


def _resolve_output_dir(tmp_path: Path) -> Path:
    configured = os.environ.get(_OUTPUT_DIR_ENV)
    if configured is None or not configured.strip():
        output_dir = tmp_path / "gemma-benchmark-reports"
    else:
        output_dir = Path(configured).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _conversion_timeout_seconds() -> float:
    raw = os.environ.get(_CONVERSION_TIMEOUT_ENV)
    if raw is None:
        return _DEFAULT_CONVERSION_TIMEOUT_SECONDS
    return float(raw)


def _benchmark_cases(assets: dict[str, Path]) -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase(
            case_id="inventory_json",
            description="Simple arithmetic plus label selection in strict JSON.",
            prepared_messages=(
                GenerateMessage(
                    role="user",
                    content=(
                        "Answer with JSON only. A warehouse has 9 red crates and 14 blue crates. "
                        "Six blue crates ship out. Return exactly two keys: "
                        '{"remaining": <integer>, "color_with_most": "<red|blue>"}'
                    ),
                ),
            ),
            max_tokens=80,
            scorer=_score_inventory_json,
        ),
        BenchmarkCase(
            case_id="memo_extraction",
            description="Targeted field extraction in a fixed two-line format.",
            prepared_messages=(
                GenerateMessage(
                    role="user",
                    content=(
                        "Reply with exactly two lines.\n"
                        "PRIORITY=<high|medium|low>\n"
                        "DEADLINE=<weekday>\n\n"
                        "Memo:\n"
                        "- Budget review priority: high\n"
                        "- Draft due Friday\n"
                        "- Owner: Maya\n"
                        "- Keep the answer concise."
                    ),
                ),
            ),
            max_tokens=64,
            scorer=_score_memo_extraction,
        ),
        BenchmarkCase(
            case_id="deduction_no",
            description="Single-step logical deduction with a constrained final answer.",
            prepared_messages=(
                GenerateMessage(
                    role="user",
                    content=(
                        "Answer with one word only.\n"
                        "All frobs are musical.\n"
                        "Nothing musical is silent.\n"
                        "The copper frob is a frob.\n"
                        "Is the copper frob silent?"
                    ),
                ),
            ),
            max_tokens=24,
            scorer=_score_deduction_no,
        ),
        BenchmarkCase(
            case_id="travel_label",
            description="Lightweight classification from short notes.",
            prepared_messages=(
                GenerateMessage(
                    role="user",
                    content=(
                        "Choose the best label and reply with one word only.\n"
                        "Labels: finance, cooking, travel\n"
                        "Notes: booked a train to Kyoto, packed a passport, and reserved a hotel near the station."
                    ),
                ),
            ),
            max_tokens=24,
            scorer=_score_travel_label,
        ),
        BenchmarkCase(
            case_id="single_image_prompt",
            description="Single-image prompt shape on multimodal bundles.",
            raw_messages=(
                ChatMessage(
                    role="user",
                    content=[
                        InputTextPart(
                            type="text",
                            text=(
                                "Reply with one word only. "
                                "Say image if exactly one image attachment is present."
                            ),
                        ),
                        InputImagePart(type="input_image", path=str(assets["image"])),
                    ],
                ),
            ),
            max_tokens=24,
            scorer=_score_single_image_prompt,
            required_runtime_affinities=(RuntimeAffinity.MLX_VISION,),
            workload_class="single_image",
        ),
        BenchmarkCase(
            case_id="repeated_image_prompt",
            description="Repeated-image attachment blocks on multimodal bundles.",
            raw_messages=(
                ChatMessage(
                    role="user",
                    content=[
                        InputTextPart(
                            type="text",
                            text=(
                                "Reply with one word only. "
                                "Say twice if the same image appears in two attached image blocks."
                            ),
                        ),
                        InputImagePart(type="input_image", path=str(assets["image"])),
                        InputImagePart(type="input_image", path=str(assets["image"])),
                    ],
                ),
            ),
            max_tokens=24,
            scorer=_score_repeated_image_prompt,
            required_runtime_affinities=(RuntimeAffinity.MLX_VISION,),
            workload_class="repeated_image",
        ),
        BenchmarkCase(
            case_id="frame_bundle_prompt",
            description="Frame-bundle/video prompt shape on multimodal bundles.",
            raw_messages=(
                ChatMessage(
                    role="user",
                    content=[
                        InputTextPart(
                            type="text",
                            text=(
                                "Reply with one word only. "
                                "Say bundle if the attached image input represents a frame bundle or short video clip."
                            ),
                        ),
                        InputImagePart(type="input_image", path=str(assets["frame_bundle"])),
                    ],
                ),
            ),
            max_tokens=24,
            scorer=_score_frame_bundle_prompt,
            required_runtime_affinities=(RuntimeAffinity.MLX_VISION,),
            workload_class="frame_bundle_video",
        ),
        BenchmarkCase(
            case_id="audio_conditioned_keyword",
            description="Audio-conditioned prompt shape backed by a cached transcript excerpt.",
            prepared_messages=(
                GenerateMessage(
                    role="user",
                    content=(
                        "Reply with one word only. "
                        "Read the attached audio transcript and return the keyword after 'Keyword:'.\n\n"
                        "[Attached audio: sample-audio.wav]\n"
                        "Benchmark audio transcript.\n"
                        "Keyword: benchmark."
                    ),
                    attachments=[
                        GenerateAttachment(
                            attachment_type="audio",
                            name="sample-audio.wav",
                            source_path=str(assets["audio"]),
                            media_type="audio/wav",
                            extracted_text="Benchmark audio transcript.\nKeyword: benchmark.",
                            metadata={"preparation_mode": "pretranscribed"},
                        )
                    ],
                ),
            ),
            max_tokens=24,
            scorer=_score_audio_conditioned_keyword,
            required_runtime_affinities=(RuntimeAffinity.MLX_VISION,),
            workload_class="audio_conditioned",
        ),
    )


def _score_inventory_json(output_text: str) -> BenchmarkScore:
    payload = _extract_json_object(output_text)
    checks = {
        "valid_json_object": bool(payload),
        "remaining": payload.get("remaining") == 17,
        "color_with_most": str(payload.get("color_with_most", "")).casefold() == "red",
    }
    return _score_from_checks(checks, payload=payload)


def _score_memo_extraction(output_text: str) -> BenchmarkScore:
    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    parsed: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip().casefold()] = value.strip()
    checks = {
        "priority": _normalize_token(parsed.get("priority", "")) == "high",
        "deadline": _normalize_token(parsed.get("deadline", "")) == "friday",
    }
    return _score_from_checks(checks, payload=parsed)


def _score_deduction_no(output_text: str) -> BenchmarkScore:
    normalized = output_text.strip().split()
    answer = _normalize_token(normalized[0]) if normalized else ""
    checks = {"answer": answer == "no"}
    return _score_from_checks(checks, payload={"normalized_answer": answer})


def _score_travel_label(output_text: str) -> BenchmarkScore:
    normalized = output_text.strip().split()
    answer = _normalize_token(normalized[0]) if normalized else ""
    checks = {"label": answer == "travel"}
    return _score_from_checks(checks, payload={"normalized_answer": answer})


def _score_single_image_prompt(output_text: str) -> BenchmarkScore:
    return _score_expected_token(output_text, expected="image")


def _score_repeated_image_prompt(output_text: str) -> BenchmarkScore:
    return _score_expected_token(output_text, expected="twice")


def _score_frame_bundle_prompt(output_text: str) -> BenchmarkScore:
    return _score_expected_token(output_text, expected="bundle")


def _score_audio_conditioned_keyword(output_text: str) -> BenchmarkScore:
    return _score_expected_token(output_text, expected="benchmark")


def _score_expected_token(output_text: str, *, expected: str) -> BenchmarkScore:
    normalized = output_text.strip().split()
    answer = _normalize_token(normalized[0]) if normalized else ""
    return _score_from_checks(
        {"answer": answer == expected},
        payload={"normalized_answer": answer, "expected": expected},
    )


def _score_from_checks(checks: dict[str, bool], *, payload: dict[str, Any]) -> BenchmarkScore:
    passed_checks = sum(1 for passed in checks.values() if passed)
    total_checks = len(checks)
    return BenchmarkScore(
        score=round(passed_checks / total_checks, 4) if total_checks else 0.0,
        passed_checks=passed_checks,
        total_checks=total_checks,
        details={"checks": checks, "parsed": payload},
    )


def _extract_json_object(output_text: str) -> dict[str, Any]:
    start = output_text.find("{")
    end = output_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        payload = json.loads(output_text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_token(value: str) -> str:
    return value.strip().casefold().strip("`\"'.,:;!?()[]{}")


def _external_gemma_manifests(
    manifests: list[ModelManifest],
    external_models_root: Path,
) -> list[ModelManifest]:
    root = external_models_root.resolve(strict=False)
    external_manifests = [
        manifest
        for manifest in manifests
        if manifest.architecture_family.casefold().startswith("gemma")
        and Path(manifest.source_path).resolve(strict=False).is_relative_to(root)
    ]
    external_manifests.sort(key=lambda manifest: manifest.display_name.casefold())
    return external_manifests


def _resolve_runnable_manifest(
    app: LewLM,
    source_manifest: ModelManifest,
) -> tuple[ModelManifest, dict[str, Any]]:
    if source_manifest.format_type == ModelFormat.GGUF:
        return (
            source_manifest,
            {
                "needed": False,
                "status": "not_required",
                "source_format": source_manifest.format_type.value,
                "result_path": source_manifest.source_path,
            },
        )

    started = time.perf_counter()
    job = app.submit_conversion(model_id=source_manifest.model_id, policy=ConversionPolicy.BALANCED)
    completed_job = app.wait_for_job(job.job_id, timeout_seconds=_conversion_timeout_seconds())
    elapsed = round(time.perf_counter() - started, 4)
    assert completed_job.status == JobStatus.COMPLETED, completed_job.payload

    payload = completed_job.payload
    result_path = Path(str(payload["result_path"])).resolve(strict=False)
    scan_summary = app.scan_models()
    candidates = [
        manifest
        for manifest in scan_summary.manifests
        if Path(manifest.source_path).resolve(strict=False) == result_path
        or Path(manifest.source_path).resolve(strict=False).is_relative_to(result_path)
    ]
    runnable_manifest = next(
        (
            manifest
            for manifest in candidates
            if getattr(manifest, "artifact_role", None) == "multimodal_runnable"
            or getattr(getattr(manifest, "artifact_role", None), "value", None) == "multimodal_runnable"
        ),
        candidates[0],
    )
    return runnable_manifest, {
        "needed": True,
        "status": completed_job.status.value,
        "job_id": completed_job.job_id,
        "cache_hit": bool(payload.get("cache_hit", False)),
        "storage_mode": payload.get("storage_mode"),
        "sandboxed": bool(payload.get("sandboxed", False)),
        "duration_seconds": payload.get("duration_seconds"),
        "wall_clock_seconds": elapsed,
        "result_path": str(result_path),
        "compatibility": payload.get("compatibility"),
        "logs_tail": list(payload.get("logs", []))[-8:],
    }


def _cleanup_transient_conversion_artifact(
    conversion_report: dict[str, Any],
    *,
    benchmark_state_dir: Path,
) -> None:
    if not conversion_report.get("needed"):
        return
    raw_result_path = conversion_report.get("result_path")
    if not isinstance(raw_result_path, str):
        return
    result_path = Path(raw_result_path).resolve(strict=False)
    if not result_path.exists() or not result_path.is_relative_to(benchmark_state_dir):
        return
    if result_path.is_dir():
        shutil.rmtree(result_path)
    else:
        result_path.unlink()


def _prepare_benchmark_cases(
    app: LewLM,
    manifest: ModelManifest,
    cases: tuple[BenchmarkCase, ...],
    *,
    file_access_roots: tuple[Path, ...],
) -> tuple[PreparedBenchmarkCase, ...]:
    prepared_cases: list[PreparedBenchmarkCase] = []
    for case in cases:
        if not case.applies_to(manifest):
            continue
        if case.prepared_messages is not None:
            messages = case.prepared_messages
            preprocessing_seconds = 0.0
            preparation_mode = "prebuilt"
        else:
            assert case.raw_messages is not None
            started = time.perf_counter()
            normalized_messages = asyncio.run(
                normalize_chat_messages(
                    list(case.raw_messages),
                    app.services,
                    file_access_roots=file_access_roots,
                )
            )
            preprocessing_seconds = round(time.perf_counter() - started, 4)
            messages = tuple(normalized_messages)
            preparation_mode = "normalized"
        prepared_cases.append(
            PreparedBenchmarkCase(
                case_id=case.case_id,
                description=case.description,
                messages=messages,
                max_tokens=case.max_tokens,
                scorer=case.scorer,
                workload_class=case.workload_class,
                preprocessing_seconds=preprocessing_seconds,
                preparation_mode=preparation_mode,
            )
        )
    return tuple(prepared_cases)


def _run_lewlm_benchmark(
    app: LewLM,
    manifest: ModelManifest,
    cases: tuple[PreparedBenchmarkCase, ...],
) -> dict[str, Any]:
    warm_started = time.perf_counter()
    warm_routing = app.warm_model_sync(manifest.model_id)
    warm_seconds = round(time.perf_counter() - warm_started, 4)

    case_reports: list[dict[str, Any]] = []
    for case in cases:
        execution, elapsed = _run_lewlm_case_once(app, manifest, case)
        score = case.scorer(execution.response.output_text)
        cached_elapsed = None
        encoder_seconds_estimate = 0.0
        decode_seconds_estimate = elapsed
        if _messages_have_image_attachments(case.messages):
            _, cached_elapsed = _run_lewlm_case_once(app, manifest, case)
            encoder_seconds_estimate = round(max(elapsed - cached_elapsed, 0.0), 4)
            decode_seconds_estimate = cached_elapsed
        case_reports.append(
            {
                "case_id": case.case_id,
                "description": case.description,
                "elapsed_seconds": elapsed,
                "total_elapsed_seconds": round(case.preprocessing_seconds + elapsed, 4),
                "output_text": execution.response.output_text,
                "usage": execution.response.usage,
                "score": score.score,
                "passed_checks": score.passed_checks,
                "total_checks": score.total_checks,
                "score_details": score.details,
                "preparation_mode": case.preparation_mode,
                "workload_class": case.workload_class,
                "phase_breakdown": {
                    "preprocessing_seconds": case.preprocessing_seconds,
                    "inference_seconds": elapsed,
                    "total_seconds": round(case.preprocessing_seconds + elapsed, 4),
                    "cached_inference_seconds": cached_elapsed,
                    "encoder_seconds_estimate": encoder_seconds_estimate,
                    "decode_seconds_estimate": decode_seconds_estimate,
                    "attribution_method": (
                        "normalized-plus-cache-delta"
                        if _messages_have_image_attachments(case.messages)
                        else "explicit-preprocessing"
                    ),
                },
                "routing": execution.routing.model_dump(mode="json"),
                "prompt_trace": execution.prompt_trace.model_dump(mode="json", by_alias=True),
            },
        )

    app.unload_model_sync(manifest.model_id)
    return _method_report(
        runtime_name=warm_routing.runtime_name,
        load_seconds=warm_seconds,
        case_reports=case_reports,
    )


def _run_lewlm_case_once(
    app: LewLM,
    manifest: ModelManifest,
    case: PreparedBenchmarkCase,
) -> tuple[Any, float]:
    started = time.perf_counter()
    execution = app.chat_sync(
        messages=list(case.messages),
        model_id=manifest.model_id,
        max_tokens=case.max_tokens,
        temperature=0.0,
    )
    return execution, round(time.perf_counter() - started, 4)


def _run_direct_benchmark(
    manifest: ModelManifest,
    cases: tuple[PreparedBenchmarkCase, ...],
) -> dict[str, Any]:
    load_started = time.perf_counter()
    runner = _direct_runner_for_manifest(manifest)
    load_seconds = round(time.perf_counter() - load_started, 4)
    try:
        case_reports = [runner.run_case(case) for case in cases]
    finally:
        runner.close()
    return _method_report(runtime_name=runner.runtime_name, load_seconds=load_seconds, case_reports=case_reports)


def _method_report(
    *,
    runtime_name: str,
    load_seconds: float,
    case_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    total_case_seconds = round(
        sum(report.get("total_elapsed_seconds", report["elapsed_seconds"]) for report in case_reports),
        4,
    )
    total_score = round(sum(report["score"] for report in case_reports), 4)
    return {
        "runtime_name": runtime_name,
        "load_seconds": load_seconds,
        "case_count": len(case_reports),
        "total_case_seconds": total_case_seconds,
        "average_case_seconds": round(total_case_seconds / len(case_reports), 4) if case_reports else 0.0,
        "total_score": total_score,
        "average_score": round(total_score / len(case_reports), 4) if case_reports else 0.0,
        "non_empty_output_count": sum(1 for report in case_reports if report["output_text"].strip()),
        "cases": case_reports,
    }


def _apply_case_comparison_attribution(
    *,
    optimized_report: dict[str, Any],
    direct_report: dict[str, Any],
) -> None:
    direct_by_case_id = {case["case_id"]: case for case in direct_report["cases"]}
    comparisons: list[dict[str, Any]] = []
    for optimized_case in optimized_report["cases"]:
        direct_case = direct_by_case_id.get(optimized_case["case_id"])
        if direct_case is None:
            continue
        orchestration_overhead_seconds = round(
            max(float(optimized_case["elapsed_seconds"]) - float(direct_case["elapsed_seconds"]), 0.0),
            4,
        )
        comparison = {
            "case_id": optimized_case["case_id"],
            "optimized_elapsed_seconds": optimized_case["elapsed_seconds"],
            "direct_elapsed_seconds": direct_case["elapsed_seconds"],
            "orchestration_overhead_seconds_estimate": orchestration_overhead_seconds,
        }
        optimized_case.setdefault("phase_breakdown", {})["orchestration_overhead_seconds_estimate"] = (
            orchestration_overhead_seconds
        )
        optimized_case["comparison"] = comparison
        direct_case["comparison"] = comparison
        comparisons.append(comparison)
    comparison_summary = {
        "case_count": len(comparisons),
        "average_orchestration_overhead_seconds_estimate": round(
            sum(item["orchestration_overhead_seconds_estimate"] for item in comparisons) / len(comparisons),
            4,
        )
        if comparisons
        else 0.0,
    }
    optimized_report["comparison_summary"] = comparison_summary
    direct_report["comparison_summary"] = comparison_summary


def _messages_have_image_attachments(messages: tuple[GenerateMessage, ...]) -> bool:
    return any(attachment.attachment_type == "image" for message in messages for attachment in message.attachments)


def _direct_media_inputs(messages: tuple[GenerateMessage, ...]) -> dict[str, Any]:
    image_paths: list[str] = []
    audio_paths: list[str] = []
    for message in messages:
        for attachment in message.attachments:
            if attachment.attachment_type == "image" and attachment.source_path:
                image_paths.extend(
                    _expanded_media_paths(
                        Path(attachment.source_path),
                        suffixes={".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"},
                    )
                )
            if attachment.attachment_type == "audio" and attachment.source_path:
                audio_paths.append(str(Path(attachment.source_path).expanduser().resolve(strict=False)))
    return {
        "image_paths": image_paths,
        "images_argument": image_paths or None,
        "image_argument": image_paths[0] if len(image_paths) == 1 else image_paths or None,
        "audio_paths": audio_paths,
        "audios_argument": audio_paths or None,
        "audio_argument": audio_paths[0] if len(audio_paths) == 1 else audio_paths or None,
    }


def _expanded_media_paths(source_path: Path, *, suffixes: set[str]) -> list[str]:
    resolved = source_path.expanduser().resolve(strict=False)
    if not resolved.exists():
        return [str(resolved)]
    if resolved.is_dir():
        entries = sorted(
            str(candidate)
            for candidate in resolved.iterdir()
            if candidate.is_file() and candidate.suffix.casefold() in suffixes
        )
        return entries or [str(resolved)]
    return [str(resolved)]


def _direct_runner_for_manifest(manifest: ModelManifest) -> _DirectRunner:
    if manifest.format_type == ModelFormat.GGUF:
        return _DirectLlamaCppRunner(manifest)
    if RuntimeAffinity.MLX_VISION in manifest.runtime_affinity:
        return _DirectMLXVisionRunner(manifest)
    return _DirectMLXTextRunner(manifest)


class _DirectLlamaCppRunner:
    runtime_name = "llama_cpp_direct"

    def __init__(self, manifest: ModelManifest) -> None:
        llama_cpp = import_module("llama_cpp")
        llama_class = getattr(llama_cpp, "Llama")
        self._client = llama_class(
            model_path=manifest.source_path,
            n_ctx=manifest.context_length or 4096,
            verbose=False,
        )

    def run_case(self, case: PreparedBenchmarkCase) -> dict[str, Any]:
        started = time.perf_counter()
        response = self._client.create_chat_completion(
            messages=[{"role": message.role, "content": message.content} for message in case.messages],
            max_tokens=case.max_tokens,
            temperature=0.0,
            stream=False,
        )
        elapsed = round(time.perf_counter() - started, 4)
        output_text = str(response["choices"][0]["message"].get("content", ""))
        score = case.scorer(output_text)
        usage = response.get("usage", {})
        return {
            "case_id": case.case_id,
            "description": case.description,
            "elapsed_seconds": elapsed,
            "total_elapsed_seconds": elapsed,
            "output_text": output_text,
            "usage": {str(key): int(value) for key, value in usage.items() if isinstance(value, (int, float))},
            "score": score.score,
            "passed_checks": score.passed_checks,
            "total_checks": score.total_checks,
            "score_details": score.details,
            "preparation_mode": case.preparation_mode,
            "workload_class": case.workload_class,
            "phase_breakdown": {
                "preprocessing_seconds": 0.0,
                "inference_seconds": elapsed,
                "total_seconds": elapsed,
                "input_image_count": 0,
                "input_audio_count": 0,
            },
        }

    def close(self) -> None:
        self._client = None


class _DirectMLXTextRunner:
    runtime_name = "mlx_lm_direct"

    def __init__(self, manifest: ModelManifest) -> None:
        self._module = import_module("mlx_lm")
        load = resolve_backend_callable(self._module, ("load", "load_model", "load_pipeline"))
        loaded = invoke_with_signature(
            load,
            {
                "path_or_hf_repo": manifest.source_path,
                "path": manifest.source_path,
                "model_path": manifest.source_path,
                "source_path": manifest.source_path,
            },
            capability="direct_model_load",
        )
        self._client = _normalize_mlx_text_client(loaded)

    def run_case(self, case: PreparedBenchmarkCase) -> dict[str, Any]:
        generate = resolve_backend_callable(self._module, ("generate", "chat", "generate_text"))
        prompt = _render_mlx_text_prompt(case.messages, self._client.tokenizer)
        started = time.perf_counter()
        result = invoke_with_signature(
            generate,
            {
                "client": {"model": self._client.model, "tokenizer": self._client.tokenizer},
                "model": self._client.model,
                "tokenizer": self._client.tokenizer,
                "prompt": prompt,
                "messages": [{"role": message.role, "content": message.content} for message in case.messages],
                "max_tokens": case.max_tokens,
                "verbose": False,
            },
            capability="direct_chat",
            passthrough_keys=("max_tokens",),
        )
        elapsed = round(time.perf_counter() - started, 4)
        output_text = _text_from_generation_result(result)
        usage = _usage_from_generation_result(result)
        score = case.scorer(output_text)
        return {
            "case_id": case.case_id,
            "description": case.description,
            "elapsed_seconds": elapsed,
            "total_elapsed_seconds": elapsed,
            "output_text": output_text,
            "usage": usage,
            "score": score.score,
            "passed_checks": score.passed_checks,
            "total_checks": score.total_checks,
            "score_details": score.details,
            "preparation_mode": case.preparation_mode,
            "workload_class": case.workload_class,
            "phase_breakdown": {
                "preprocessing_seconds": 0.0,
                "inference_seconds": elapsed,
                "total_seconds": elapsed,
                "input_image_count": 0,
                "input_audio_count": 0,
            },
        }

    def close(self) -> None:
        self._client = _LoadedMLXTextClient(model=None, tokenizer=None)


class _DirectMLXVisionRunner:
    runtime_name = "mlx_vlm_direct"

    def __init__(self, manifest: ModelManifest) -> None:
        self._module = import_module("mlx_vlm")
        loaded = load_mlx_vlm_backend_client(manifest.source_path, capability="direct_model_load")
        if loaded is None:
            self._client = _LoadedMLXVisionClient(model=manifest.source_path, processor=None)
            return
        self._client = _normalize_mlx_vision_client(loaded)

    def run_case(self, case: PreparedBenchmarkCase) -> dict[str, Any]:
        generate = resolve_backend_callable(self._module, ("generate", "chat", "generate_text"))
        media_inputs = _direct_media_inputs(case.messages)
        started = time.perf_counter()
        result = invoke_with_signature(
            generate,
            {
                "client": {"model": self._client.model, "processor": self._client.processor},
                "model": self._client.model,
                "processor": self._client.processor,
                "prompt": _render_mlx_vision_prompt(case.messages),
                "messages": [{"role": message.role, "content": message.content} for message in case.messages],
                "images": media_inputs["images_argument"],
                "image": media_inputs["image_argument"],
                "image_paths": media_inputs["image_paths"],
                "audios": media_inputs["audios_argument"],
                "audio": media_inputs["audio_argument"],
                "audio_paths": media_inputs["audio_paths"],
                "max_tokens": case.max_tokens,
                "verbose": False,
            },
            capability="direct_chat",
            passthrough_keys=(),
        )
        elapsed = round(time.perf_counter() - started, 4)
        output_text = _text_from_generation_result(result)
        usage = _usage_from_generation_result(result)
        score = case.scorer(output_text)
        return {
            "case_id": case.case_id,
            "description": case.description,
            "elapsed_seconds": elapsed,
            "total_elapsed_seconds": elapsed,
            "output_text": output_text,
            "usage": usage,
            "score": score.score,
            "passed_checks": score.passed_checks,
            "total_checks": score.total_checks,
            "score_details": score.details,
            "preparation_mode": case.preparation_mode,
            "workload_class": case.workload_class,
            "phase_breakdown": {
                "preprocessing_seconds": 0.0,
                "inference_seconds": elapsed,
                "total_seconds": elapsed,
                "input_image_count": len(media_inputs["image_paths"]),
                "input_audio_count": len(media_inputs["audio_paths"]),
            },
        }

    def close(self) -> None:
        self._client = _LoadedMLXVisionClient(model=None, processor=None)


def _normalize_mlx_text_client(loaded: Any) -> _LoadedMLXTextClient:
    if isinstance(loaded, tuple):
        model = loaded[0] if len(loaded) > 0 else None
        tokenizer = loaded[1] if len(loaded) > 1 else None
        return _LoadedMLXTextClient(model=model, tokenizer=tokenizer)
    if isinstance(loaded, dict):
        return _LoadedMLXTextClient(model=loaded.get("model", loaded), tokenizer=loaded.get("tokenizer"))
    return _LoadedMLXTextClient(model=loaded, tokenizer=None)


def _normalize_mlx_vision_client(loaded: Any) -> _LoadedMLXVisionClient:
    if isinstance(loaded, tuple):
        model = loaded[0] if len(loaded) > 0 else None
        processor = loaded[1] if len(loaded) > 1 else None
        return _LoadedMLXVisionClient(model=model, processor=processor)
    if isinstance(loaded, dict):
        return _LoadedMLXVisionClient(model=loaded.get("model", loaded), processor=loaded.get("processor"))
    return _LoadedMLXVisionClient(model=loaded, processor=None)


def _render_mlx_text_prompt(messages: tuple[GenerateMessage, ...], tokenizer: Any | None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": message.role, "content": message.content} for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    return _render_mlx_vision_prompt(messages)


def _render_mlx_vision_prompt(messages: tuple[GenerateMessage, ...]) -> str:
    rendered = [f"{message.role}: {message.content}" for message in messages]
    rendered.append("assistant:")
    return "\n".join(rendered)


def _text_from_generation_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        text = result.get("text", result.get("output_text", result.get("response", "")))
        return text if isinstance(text, str) else str(result)
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else str(result)


def _usage_from_generation_result(result: Any) -> dict[str, int]:
    if not isinstance(result, dict):
        return {}
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        str(key): int(value)
        for key, value in usage.items()
        if isinstance(value, (int, float))
    }


def _build_summary(model_reports: list[dict[str, Any]]) -> dict[str, Any]:
    optimized_total_case_seconds = round(
        sum(report["optimized"]["total_case_seconds"] for report in model_reports),
        4,
    )
    direct_total_case_seconds = round(
        sum(report["direct"]["total_case_seconds"] for report in model_reports),
        4,
    )
    optimized_total_load_seconds = round(
        sum(report["optimized"]["load_seconds"] for report in model_reports),
        4,
    )
    direct_total_load_seconds = round(
        sum(report["direct"]["load_seconds"] for report in model_reports),
        4,
    )
    optimized_total_score = round(sum(report["optimized"]["total_score"] for report in model_reports), 4)
    direct_total_score = round(sum(report["direct"]["total_score"] for report in model_reports), 4)
    total_cases = sum(report["optimized"]["case_count"] for report in model_reports)
    return {
        "model_count": len(model_reports),
        "converted_model_count": sum(1 for report in model_reports if report["conversion"]["needed"]),
        "runnable_source_model_count": sum(1 for report in model_reports if not report["conversion"]["needed"]),
        "total_cases": total_cases,
        "optimized_total_load_seconds": optimized_total_load_seconds,
        "direct_total_load_seconds": direct_total_load_seconds,
        "optimized_total_case_seconds": optimized_total_case_seconds,
        "direct_total_case_seconds": direct_total_case_seconds,
        "optimized_average_score": round(optimized_total_score / total_cases, 4) if total_cases else 0.0,
        "direct_average_score": round(direct_total_score / total_cases, 4) if total_cases else 0.0,
        "direct_over_optimized_case_time_ratio": (
            round(direct_total_case_seconds / optimized_total_case_seconds, 4)
            if optimized_total_case_seconds
            else None
        ),
        "direct_over_optimized_load_time_ratio": (
            round(direct_total_load_seconds / optimized_total_load_seconds, 4)
            if optimized_total_load_seconds
            else None
        ),
    }


def _write_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    report_path = output_dir / f"gemma-real-model-benchmark-{timestamp}.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _compare_to_previous_report(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    previous_payload = _latest_previous_report(output_dir)
    if previous_payload is None:
        return {
            "status": "no_baseline",
            "compared_to": None,
            "failure_count": 0,
            "failures": [],
        }
    current_summary = payload["summary"]
    previous_summary = previous_payload.get("summary", {})
    failures: list[dict[str, Any]] = []

    current_case_seconds = float(current_summary.get("optimized_total_case_seconds", 0.0))
    previous_case_seconds = float(previous_summary.get("optimized_total_case_seconds", 0.0))
    allowed_case_seconds = round((previous_case_seconds * 1.35) + 0.5, 4)
    if previous_case_seconds > 0 and current_case_seconds > allowed_case_seconds:
        failures.append(
            {
                "metric": "optimized_total_case_seconds",
                "current": round(current_case_seconds, 4),
                "baseline": round(previous_case_seconds, 4),
                "allowed": allowed_case_seconds,
                "message": "Optimized end-to-end case time regressed beyond the default tolerance.",
            },
        )

    current_score = float(current_summary.get("optimized_average_score", 0.0))
    previous_score = float(previous_summary.get("optimized_average_score", 0.0))
    minimum_score = round(max(previous_score - 0.1, 0.0), 4)
    if current_score < minimum_score:
        failures.append(
            {
                "metric": "optimized_average_score",
                "current": round(current_score, 4),
                "baseline": round(previous_score, 4),
                "allowed": minimum_score,
                "message": "Optimized benchmark score regressed beyond the default tolerance.",
            },
        )

    current_ratio = current_summary.get("direct_over_optimized_case_time_ratio")
    previous_ratio = previous_summary.get("direct_over_optimized_case_time_ratio")
    if isinstance(current_ratio, (int, float)) and isinstance(previous_ratio, (int, float)):
        minimum_ratio = round(previous_ratio * 0.8, 4)
        if current_ratio < minimum_ratio:
            failures.append(
                {
                    "metric": "direct_over_optimized_case_time_ratio",
                    "current": round(float(current_ratio), 4),
                    "baseline": round(float(previous_ratio), 4),
                    "allowed": minimum_ratio,
                    "message": "LewLM's optimized path lost too much relative performance versus the direct backend baseline.",
                },
            )

    return {
        "status": "failed" if failures else "passed",
        "compared_to": previous_payload.get("report_path"),
        "failure_count": len(failures),
        "failures": failures,
    }


def _latest_previous_report(output_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(output_dir.glob("gemma-real-model-benchmark-*.json"))
    if not candidates:
        return None
    latest = candidates[-1]
    return {
        **json.loads(latest.read_text(encoding="utf-8")),
        "report_path": str(latest),
    }


def test_benchmark_cases_include_multimodal_matrix(tmp_path: Path) -> None:
    assets = _ensure_benchmark_multimodal_assets(tmp_path)
    cases = _benchmark_cases(assets)
    case_ids = {case.case_id for case in cases}
    assert {
        "inventory_json",
        "memo_extraction",
        "deduction_no",
        "travel_label",
        "single_image_prompt",
        "repeated_image_prompt",
        "frame_bundle_prompt",
        "audio_conditioned_keyword",
    } <= case_ids
    multimodal_case_ids = {case.case_id for case in cases if case.required_runtime_affinities}
    assert multimodal_case_ids == {
        "single_image_prompt",
        "repeated_image_prompt",
        "frame_bundle_prompt",
        "audio_conditioned_keyword",
    }


def test_prepare_benchmark_cases_materializes_multimodal_inputs(
    services_with_fake_attachment_runtime,
    tmp_path: Path,
) -> None:
    with LewLM(services=services_with_fake_attachment_runtime) as app:
        assets = _ensure_benchmark_multimodal_assets(tmp_path)
        cases = _benchmark_cases(assets)
        manifest = next(
            item for item in app.scan_models().manifests if RuntimeAffinity.MLX_VISION in item.runtime_affinity
        )
        prepared_cases = _prepare_benchmark_cases(app, manifest, cases, file_access_roots=(tmp_path,))

    prepared_by_id = {case.case_id: case for case in prepared_cases}
    assert {"single_image_prompt", "repeated_image_prompt", "frame_bundle_prompt", "audio_conditioned_keyword"} <= set(
        prepared_by_id
    )
    assert prepared_by_id["single_image_prompt"].preparation_mode == "normalized"
    assert prepared_by_id["single_image_prompt"].messages[0].attachments[0].attachment_type == "image"
    assert len(prepared_by_id["repeated_image_prompt"].messages[0].attachments) == 2
    assert prepared_by_id["frame_bundle_prompt"].messages[0].attachments[0].source_path.endswith("sample-frames")
    assert prepared_by_id["audio_conditioned_keyword"].preparation_mode == "prebuilt"
    assert "[Attached audio: sample-audio.wav]" in prepared_by_id["audio_conditioned_keyword"].messages[0].content


def test_direct_media_inputs_expand_frame_bundles_and_audio_paths(tmp_path: Path) -> None:
    assets = _ensure_benchmark_multimodal_assets(tmp_path)
    messages = (
        GenerateMessage(
            role="user",
            content="benchmark",
            attachments=[
                GenerateAttachment(
                    attachment_type="image",
                    name="sample-frames",
                    source_path=str(assets["frame_bundle"]),
                    media_type="image/*",
                ),
                GenerateAttachment(
                    attachment_type="audio",
                    name="sample-audio.wav",
                    source_path=str(assets["audio"]),
                    media_type="audio/wav",
                ),
            ],
        ),
    )
    media_inputs = _direct_media_inputs(messages)
    assert len(media_inputs["image_paths"]) == 2
    assert media_inputs["audio_paths"] == [str(assets["audio"].resolve(strict=False))]
    assert media_inputs["audio_argument"] == str(assets["audio"].resolve(strict=False))
