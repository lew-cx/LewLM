"""LewLM command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import inspect
import io
import json
import re
import sys
import time
import warnings
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Sequence

import uvicorn

from lewlm.api.app import create_app
from lewlm.api.message_normalization import normalize_chat_messages
from lewlm.api.schemas.chat import ChatMessage, InputAudioPart, InputFilePart, InputImagePart, InputTextPart
from lewlm.benchmarking import benchmark_direct_chat_manifest, benchmark_runtime_chat_manifest
from lewlm.cli.display import (
    coerce_float as _coerce_float,
    coerce_int as _coerce_int,
    format_delta as _format_delta,
    format_percent as _format_percent,
    format_rate as _format_rate,
    format_seconds as _format_seconds,
    inventory_display_lines as _inventory_display_lines,
    print_benchmark_table as _print_benchmark_table,
    style as _style,
)
from lewlm.config.settings import LewLMSettings, get_settings
from lewlm.conversion.models import ConversionJobRequest, ConversionPolicy, JobRecord, JobStatus
from lewlm.core.bootstrap import LewLMServices, bootstrap_services
from lewlm.core.contracts import (
    CapabilityName,
    ConversionStatus,
    ExternalQuantizerReference,
    GenerateMessage,
    LayerQuantizationOverride,
    MeasuredCapabilityCategory,
    MeasuredCapabilityEvidenceSource,
    MeasuredCapabilityStatus,
    ModelArtifactRole,
    ModelCapabilityReport,
    ModelInventory,
    ModelManifest,
    ModelModality,
    ModelScanSummary,
    QuantizationPrecision,
    QuantizationProfile,
    QuantizationStrategy,
    ReasoningOutput,
    ReasoningVisibility,
    RuntimeAffinity,
    quantization_profile_label,
)
from lewlm.core.errors import ConfigurationError, LewLMError, NotImplementedLewLMError
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.documents.skills.models import BuiltInSkillDescriptor, parse_document_transform_request
from lewlm.history.models import SESSION_CONTEXT_POLICIES, SessionDetail, SessionExportBundle, SessionRecord
from lewlm.install_profiles import summarize_install_profiles
from lewlm.prompting import PromptCompilationRequest
from lewlm.registry.discovery import discover_models
from lewlm.routing.measured_preferences import assess_runtime_preference
from lewlm.runtime.experimental import ClusterEnrollWorkerResponse, ClusterStatus
from lewlm.runtime.adapters import summarize_feature_preservation
from lewlm.security.authorization import ToolAction
from lewlm.security.files import read_scoped_text_file
from lewlm.serving_profiles import SERVING_PROFILE_WORKLOAD_CLASS_CHOICES, is_attachment_workload_class
from lewlm.tools.models import (
    DocumentGenerateToolRequest,
    DocumentTransformToolRequest,
    GenerateDocumentToolInput,
    LocalToolDescriptor,
    parse_tool_execution_request,
)


class ExitCode(IntEnum):
    OK = 0
    ERROR = 1
    USAGE = 2
    NOT_IMPLEMENTED = 3


_CLI_CHAT_BENCHMARK_MAX_TOKENS = 128
_BENIGN_BENCHMARK_STDERR_PATTERNS = (
    re.compile(r"^llama_context: n_ctx_seq \(\d+\) < n_ctx_train \(\d+\) -- the full capacity of the model will not be utilized$"),
    re.compile(r"^llama_kv_cache_iswa: using full-size SWA cache .*"),
    re.compile(
        r"^llama_kv_cache: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to \d+$",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lewlm", description="Local-first multimodal runtime orchestration.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Start the local LewLM API server.")
    serve_parser.add_argument("--host", default=None, help="Bind address override.")
    serve_parser.add_argument("--port", default=None, type=int, help="Bind port override.")
    serve_parser.set_defaults(handler=handle_serve)

    doctor_parser = subparsers.add_parser("doctor", help="Inspect local LewLM health and configuration.")
    doctor_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    doctor_parser.set_defaults(handler=handle_doctor)

    config_parser = subparsers.add_parser("config", help="Show resolved LewLM configuration.")
    config_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    config_parser.set_defaults(handler=handle_config)

    scan_parser = subparsers.add_parser("scan", help="Scan model directories and update the registry.")
    scan_parser.add_argument("paths", nargs="*", help="Optional model roots to scan instead of configured roots.")
    scan_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    scan_parser.set_defaults(handler=handle_scan)

    list_models_parser = subparsers.add_parser("list-models", help="List models from the local registry.")
    list_models_parser.add_argument("--all", action="store_true", help="Show every registered artifact instead of the grouped default view.")
    list_models_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    list_models_parser.set_defaults(handler=handle_list_models)

    list_skills_parser = subparsers.add_parser("list-skills", help="List built-in LewLM skills.")
    list_skills_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    list_skills_parser.set_defaults(handler=handle_list_skills)

    show_skill_parser = subparsers.add_parser("show-skill", help="Show details for a built-in LewLM skill.")
    show_skill_parser.add_argument("skill_name", help="Built-in skill identifier to inspect.")
    show_skill_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    show_skill_parser.set_defaults(handler=handle_show_skill)

    list_tools_parser = subparsers.add_parser("list-tools", help="List registered local LewLM tools.")
    list_tools_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    list_tools_parser.set_defaults(handler=handle_list_tools)

    show_tool_parser = subparsers.add_parser("show-tool", help="Show details for a registered local LewLM tool.")
    show_tool_parser.add_argument("tool_name", help="Registered tool identifier to inspect.")
    show_tool_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    show_tool_parser.set_defaults(handler=handle_show_tool)

    list_sessions_parser = subparsers.add_parser("list-sessions", help="List persisted local chat sessions.")
    list_sessions_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    list_sessions_parser.set_defaults(handler=handle_list_sessions)

    show_session_parser = subparsers.add_parser("show-session", help="Show a persisted local chat session.")
    show_session_parser.add_argument("session_id", help="Persisted session identifier to inspect.")
    show_session_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    show_session_parser.set_defaults(handler=handle_show_session)

    export_session_parser = subparsers.add_parser("export-session", help="Export a persisted session to a JSON bundle.")
    export_session_parser.add_argument("session_id", help="Persisted session identifier to export.")
    export_session_parser.add_argument("--output", required=True, help="Destination JSON file path.")
    export_session_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    export_session_parser.set_defaults(handler=handle_export_session)

    import_session_parser = subparsers.add_parser("import-session", help="Import a session bundle from a JSON file.")
    import_session_parser.add_argument("--input", required=True, help="Path to a JSON session bundle.")
    import_session_parser.add_argument("--title", default=None, help="Optional title override for the imported session.")
    import_session_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    import_session_parser.set_defaults(handler=handle_import_session)

    delete_session_parser = subparsers.add_parser("delete-session", help="Delete a persisted local chat session.")
    delete_session_parser.add_argument("session_id", help="Persisted session identifier to delete.")
    delete_session_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    delete_session_parser.set_defaults(handler=handle_delete_session)

    capabilities_parser = subparsers.add_parser(
        "capabilities",
        help="Show capability support and runtime compatibility for a registered model.",
    )
    capabilities_parser.add_argument("model", help="Registered model identifier to inspect.")
    capabilities_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    capabilities_parser.set_defaults(handler=handle_capabilities)

    convert_parser = subparsers.add_parser("convert", help="Queue or run a local model conversion job.")
    convert_parser.add_argument("model", help="Registered model identifier to convert.")
    convert_parser.add_argument(
        "--policy",
        default=ConversionPolicy.BALANCED.value,
        choices=[policy.value for policy in ConversionPolicy],
        help="Conversion policy to apply.",
    )
    convert_parser.add_argument("--custom-bits", type=int, default=None, help="Custom quantization bits for custom_bits policy.")
    convert_parser.add_argument(
        "--profile",
        choices=[strategy.value for strategy in QuantizationStrategy],
        help="Optional advanced quantization strategy overlay for conversion compatibility reporting.",
    )
    convert_parser.add_argument("--profile-name", default=None, help="Optional human-readable label for the quantization profile.")
    convert_parser.add_argument(
        "--weight-precision",
        choices=[precision.value for precision in QuantizationPrecision],
        help="Requested weight precision for an advanced quantization profile.",
    )
    convert_parser.add_argument(
        "--activation-precision",
        choices=[precision.value for precision in QuantizationPrecision],
        help="Requested activation precision for activation-aware profiles.",
    )
    convert_parser.add_argument(
        "--compute-precision",
        choices=[precision.value for precision in QuantizationPrecision],
        help="Requested compute precision for mixed or hybrid profiles.",
    )
    convert_parser.add_argument(
        "--kv-cache-precision",
        choices=[precision.value for precision in QuantizationPrecision],
        help="Optional KV-cache precision metadata recorded with the profile.",
    )
    convert_parser.add_argument(
        "--calibration-samples",
        type=int,
        default=None,
        help="Calibration sample count metadata for activation-aware profiles.",
    )
    convert_parser.add_argument(
        "--layer-override",
        action="append",
        default=[],
        help="Per-layer override in the form layer_pattern:weight_precision[:activation_precision[:compute_precision]].",
    )
    convert_parser.add_argument("--external-quantizer", default=None, help="Optional external adaptive quantizer name.")
    convert_parser.add_argument("--external-profile", default=None, help="Optional external quantizer profile name.")
    convert_parser.add_argument("--external-module", default=None, help="Optional Python module expected for the external quantizer.")
    convert_parser.add_argument("--force", action="store_true", help="Ignore existing cached artifacts and reconvert.")
    convert_parser.add_argument("--no-wait", action="store_true", help="Return immediately after the job is queued.")
    _add_authorize_argument(convert_parser)
    _add_idempotency_key_argument(convert_parser)
    convert_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    convert_parser.set_defaults(handler=handle_convert)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run reproducible runtime benchmarks and regression checks against one or more models.",
    )
    benchmark_parser.add_argument("--model", default=None, help="Optional explicit model id.")
    benchmark_parser.add_argument("--all", action="store_true", help="Benchmark all runnable models for the selected capability.")
    benchmark_parser.add_argument(
        "--capability",
        default=CapabilityName.CHAT.value,
        choices=[
            CapabilityName.CHAT.value,
            CapabilityName.EMBEDDINGS.value,
            CapabilityName.RERANK.value,
            CapabilityName.AUDIO_TRANSCRIPTION.value,
        ],
        help="Capability workload to benchmark.",
    )
    benchmark_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat a benchmark suite this many times; only valid with --all.",
    )
    benchmark_parser.add_argument("--prompt", default="Benchmark ping", help="Primary input text to use during benchmarking.")
    benchmark_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Run this many warmup requests before recording warm-path TTFT and steady-state decode metrics.",
    )
    benchmark_parser.add_argument(
        "--compare-direct",
        action="store_true",
        help="For chat benchmarks, run a direct backend baseline first and compare it against LewLM-managed inference.",
    )
    benchmark_parser.add_argument(
        "--compare-external-adapter",
        action="store_true",
        help="For chat benchmarks, compare LewLM's native runtime against the configured local external accelerator adapter.",
    )
    benchmark_parser.add_argument(
        "--convert-missing",
        action="store_true",
        help="When using --compare-direct, convert eligible non-runnable models first and include conversion timing in the results.",
    )
    benchmark_parser.add_argument(
        "--convert-policy",
        action="append",
        dest="convert_policies",
        choices=[policy.value for policy in ConversionPolicy],
        default=[],
        help="Repeat to compare multiple conversion policies when --compare-direct and --convert-missing are enabled.",
    )
    benchmark_parser.add_argument(
        "--convert-profile",
        action="append",
        dest="convert_profiles",
        choices=[
            strategy.value
            for strategy in QuantizationStrategy
            if strategy != QuantizationStrategy.WEIGHT_ONLY
        ],
        default=[],
        help="Repeat to compare advanced quantization profile presets when --compare-direct and --convert-missing are enabled.",
    )
    benchmark_parser.add_argument(
        "--compare-metric",
        default="cold_total_seconds",
        choices=(
            "cold_total_seconds",
            "cold_load_seconds",
            "warm_total_seconds",
            "ttft_seconds",
            "steady_state_decode_seconds",
        ),
        help="Primary metric to use for LewLM-versus-direct comparison summaries.",
    )
    benchmark_parser.add_argument(
        "--disable-serving-profile",
        action="store_true",
        help="Ignore persisted autotuned serving profiles for this benchmark run.",
    )
    benchmark_parser.add_argument(
        "--workload-class",
        choices=list(SERVING_PROFILE_WORKLOAD_CLASS_CHOICES),
        default=None,
        help="Optional chat workload class to benchmark, including multimodal attachment patterns.",
    )
    benchmark_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    benchmark_parser.set_defaults(handler=handle_benchmark)

    autotune_parser = subparsers.add_parser(
        "autotune",
        help="Benchmark serving-profile candidates and persist a recommended configuration for a chat model.",
    )
    autotune_parser.add_argument("--model", default=None, help="Optional explicit model id.")
    autotune_parser.add_argument("--prompt", default="Benchmark ping", help="Primary input text to use during autotuning.")
    autotune_parser.add_argument(
        "--capability",
        default=CapabilityName.CHAT.value,
        choices=[CapabilityName.CHAT.value],
        help="Autotuning currently supports chat workloads only.",
    )
    autotune_parser.add_argument(
        "--workload-class",
        choices=list(SERVING_PROFILE_WORKLOAD_CLASS_CHOICES),
        default=None,
        help="Optional chat workload class to autotune, including multimodal attachment patterns.",
    )
    autotune_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    autotune_parser.set_defaults(handler=handle_autotune)

    cluster_parser = subparsers.add_parser("cluster", help="Manage experimental multi-host cluster workflows.")
    cluster_subparsers = cluster_parser.add_subparsers(dest="cluster_command", required=True)

    cluster_status_parser = cluster_subparsers.add_parser("status", help="Show local cluster status.")
    cluster_status_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    cluster_status_parser.set_defaults(handler=handle_cluster_status)

    cluster_token_parser = cluster_subparsers.add_parser("issue-token", help="Issue a signed cluster worker token.")
    cluster_token_parser.add_argument("--worker-name", default=None, help="Optional expected worker name.")
    cluster_token_parser.add_argument(
        "--capability",
        action="append",
        dest="capabilities",
        default=[],
        help="Capability claim to embed in the token. Repeatable.",
    )
    cluster_token_parser.add_argument("--ttl-seconds", type=int, default=None, help="Optional token lifetime override.")
    cluster_token_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    cluster_token_parser.set_defaults(handler=handle_cluster_issue_token)

    cluster_join_parser = cluster_subparsers.add_parser("join", help="Enroll this worker with a coordinator.")
    cluster_join_parser.add_argument("--coordinator", required=True, help="Coordinator base URL.")
    cluster_join_parser.add_argument("--token", required=True, help="Signed worker enrollment token.")
    cluster_join_parser.add_argument("--worker-name", default=None, help="Optional worker name override.")
    cluster_join_parser.add_argument("--endpoint", default=None, help="Optional worker public base URL override.")
    cluster_join_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    cluster_join_parser.set_defaults(handler=handle_cluster_join)

    cluster_heartbeat_parser = cluster_subparsers.add_parser("heartbeat", help="Send a worker heartbeat to the coordinator.")
    cluster_heartbeat_parser.add_argument("--coordinator", default=None, help="Optional coordinator base URL override.")
    cluster_heartbeat_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    cluster_heartbeat_parser.set_defaults(handler=handle_cluster_heartbeat)

    cluster_plan_parser = cluster_subparsers.add_parser("plan", help="Build a distributed plan for a model.")
    cluster_plan_parser.add_argument("--model", required=True, help="Model identifier to plan.")
    cluster_plan_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    cluster_plan_parser.set_defaults(handler=handle_cluster_plan)

    cluster_benchmark_parser = cluster_subparsers.add_parser(
        "benchmark",
        help="Run a benchmark through the experimental distributed runtime.",
    )
    cluster_benchmark_parser.add_argument("--model", required=True, help="Distributed model identifier.")
    cluster_benchmark_parser.add_argument("--prompt", default="Benchmark ping", help="Prompt to benchmark.")
    cluster_benchmark_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    cluster_benchmark_parser.set_defaults(handler=handle_cluster_benchmark)

    warm_parser = subparsers.add_parser("warm", help="Warm a specific model in its selected runtime.")
    warm_parser.add_argument("model", help="Model identifier to warm.")
    warm_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    warm_parser.set_defaults(handler=handle_warm)

    unload_parser = subparsers.add_parser("unload", help="Unload a specific model from its selected runtime.")
    unload_parser.add_argument("model", help="Model identifier to unload.")
    unload_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    unload_parser.set_defaults(handler=handle_unload)

    chat_parser = subparsers.add_parser("chat", help="Run a local chat completion.")
    chat_parser.add_argument("prompt", nargs="?", help="User prompt to send.")
    chat_parser.add_argument("--message", default=None, help="User prompt override.")
    chat_parser.add_argument("--model", default=None, help="Optional explicit model id.")
    chat_parser.add_argument(
        "--session-id",
        default=None,
        help="Persist this turn into an existing local session and include its history as context.",
    )
    chat_parser.add_argument(
        "--save-session",
        action="store_true",
        help="Create a new local session and persist this chat turn into it.",
    )
    chat_parser.add_argument(
        "--session-title",
        default=None,
        help="Optional title when creating a new persisted session.",
    )
    chat_parser.add_argument(
        "--session-context-policy",
        default="full_history",
        choices=list(SESSION_CONTEXT_POLICIES),
        help="Context policy to apply when creating a new persisted session.",
    )
    chat_parser.add_argument("--system-prompt", default=None, help="Inline system prompt override.")
    chat_parser.add_argument("--developer-prompt", default=None, help="Inline developer prompt override.")
    chat_parser.add_argument(
        "--pretext-file",
        default=None,
        help="Path to a local text file containing prompt pretext instructions.",
    )
    chat_parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Path to a local text file to prepend as a system prompt.",
    )
    chat_parser.add_argument(
        "--skills-file",
        default=None,
        help="Path to a local JSON file containing a prompt skill override.",
    )
    output_contract_group = chat_parser.add_mutually_exclusive_group()
    output_contract_group.add_argument(
        "--response-format-file",
        default=None,
        help="Path to a local JSON file containing a response_format contract.",
    )
    output_contract_group.add_argument(
        "--output-schema-file",
        default=None,
        help="Legacy path to a local JSON file describing a raw JSON-schema output contract.",
    )
    chat_parser.add_argument(
        "--tools-file",
        default=None,
        help="Path to a local JSON file containing tool declarations.",
    )
    chat_parser.add_argument(
        "--mcp-tools-file",
        default=None,
        help="Path to a local JSON file containing prompt-only local MCP tool listings.",
    )
    chat_parser.add_argument(
        "--reasoning-visibility",
        default=None,
        choices=[visibility.value for visibility in ReasoningVisibility],
        help="Expose explicit model-emitted reasoning as hidden, summarized, or raw when the model emits designated reasoning blocks.",
    )
    chat_parser.add_argument(
        "--attach-file",
        action="append",
        default=[],
        help="Attach a local document or text file to the user message. Repeatable.",
    )
    chat_parser.add_argument(
        "--attach-image",
        action="append",
        default=[],
        help="Attach a local image to the user message. Repeatable.",
    )
    chat_parser.add_argument(
        "--attach-audio",
        action="append",
        default=[],
        help="Attach a local audio file to the user message. Repeatable.",
    )
    chat_parser.add_argument(
        "--disable-serving-profile",
        action="store_true",
        help="Ignore persisted autotuned serving profiles for this chat request.",
    )
    chat_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    chat_parser.add_argument("--stream", action="store_true", help="Print streamed deltas as they arrive.")
    chat_parser.set_defaults(handler=handle_chat)

    generate_doc_parser = subparsers.add_parser("generate-doc", help="Render a document artifact from a JSON IR file.")
    generate_doc_parser.add_argument("--input", required=True, help="Path to a JSON file containing a DocumentIR payload.")
    generate_doc_parser.add_argument(
        "--format",
        required=True,
        choices=[output_format.value for output_format in DocumentOutputFormat],
        help="Output format to render.",
    )
    generate_doc_parser.add_argument("--output", required=True, help="Destination file path for the rendered artifact.")
    _add_authorize_argument(generate_doc_parser)
    _add_idempotency_key_argument(generate_doc_parser)
    generate_doc_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    generate_doc_parser.set_defaults(handler=handle_generate_doc)

    transform_parser = subparsers.add_parser("transform", help="Run a built-in document skill from a JSON request file.")
    transform_parser.add_argument("--input", required=True, help="Path to a JSON file containing a transform request.")
    transform_parser.add_argument("--output", required=True, help="Destination file path for the rendered artifact.")
    _add_authorize_argument(transform_parser)
    _add_idempotency_key_argument(transform_parser)
    transform_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    transform_parser.set_defaults(handler=handle_transform)

    run_tool_parser = subparsers.add_parser("run-tool", help="Execute a registered local tool from a JSON request file.")
    run_tool_parser.add_argument("--input", required=True, help="Path to a JSON file containing a tool execution request.")
    run_tool_parser.add_argument("--output", default=None, help="Optional destination path for artifact-producing tools.")
    _add_authorize_argument(run_tool_parser)
    _add_idempotency_key_argument(run_tool_parser)
    run_tool_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    run_tool_parser.set_defaults(handler=handle_run_tool)

    cache_parser = subparsers.add_parser("cache", help="Inspect and manage the managed LewLM cache.")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command")
    cache_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    cache_parser.set_defaults(handler=handle_cache)
    cache_clear_conversions_parser = cache_subparsers.add_parser(
        "clear-conversions",
        help="Remove cached conversion artifacts and rescan configured model roots.",
    )
    cache_clear_conversions_parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    cache_clear_conversions_parser.set_defaults(handler=handle_cache_clear_conversions)

    return parser


@contextlib.contextmanager
def _filtered_benchmark_noise():
    stderr_buffer = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stderr(stderr_buffer):
        warnings.filterwarnings(
            "ignore",
            message=r"At least one mel filter has all zero values\..*",
            category=UserWarning,
            module=r"transformers\.audio_utils",
        )
        try:
            yield
        finally:
            filtered_stderr = _filter_benchmark_stderr(stderr_buffer.getvalue())
            if filtered_stderr:
                sys.stderr.write(filtered_stderr)
                if not filtered_stderr.endswith("\n"):
                    sys.stderr.write("\n")


def _filter_benchmark_stderr(stderr_text: str) -> str:
    filtered_lines = [
        line
        for line in stderr_text.splitlines(keepends=True)
        if not any(pattern.match(line.strip()) for pattern in _BENIGN_BENCHMARK_STDERR_PATTERNS)
    ]
    return "".join(filtered_lines)


def _add_authorize_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--authorize",
        action="append",
        default=[],
        choices=[action.value for action in ToolAction],
        help="Explicitly authorize a tool-like action when the policy requires it.",
    )


def _add_idempotency_key_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--idempotency-key",
        default=None,
        help="Optional idempotency key to safely retry the same operation without re-executing it.",
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: LewLMSettings | None = None,
    services: LewLMServices | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    resolved_settings = settings or get_settings()
    handler = getattr(args, "handler")
    owned_services: LewLMServices | None = None
    if services is None and handler not in {handle_config, handle_serve}:
        owned_services = bootstrap_services(resolved_settings)
        services = owned_services
    try:
        return int(handler(args, resolved_settings, services))
    except NotImplementedLewLMError as exc:
        print(str(exc), file=sys.stderr)
        return int(ExitCode.NOT_IMPLEMENTED)
    except LewLMError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": exc.to_dict()}, indent=2))
        else:
            _print_cli_error(exc)
        return int(ExitCode.ERROR)
    finally:
        if owned_services is not None:
            owned_services.close()


def handle_serve(args: argparse.Namespace, settings: LewLMSettings, services: LewLMServices | None = None) -> ExitCode:
    server_settings = settings.with_updates(
        host=args.host or settings.host,
        port=args.port or settings.port,
    )
    uvicorn.run(
        create_app(server_settings, services=services),
        host=server_settings.host,
        port=server_settings.port,
        log_level=server_settings.log_level.lower(),
    )
    return ExitCode.OK


def handle_doctor(args: argparse.Namespace, settings: LewLMSettings, services: LewLMServices | None = None) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    runtime_stats = asyncio.run(resolved_services.telemetry_service.runtime_stats())
    cache_stats = resolved_services.telemetry_service.cache_stats()
    payload = {
        "service": settings.app_name,
        "version": settings.version,
        "configuration": settings.redacted_snapshot(),
        "install_profiles": summarize_install_profiles().model_dump(mode="json"),
        "storage": resolved_services.metadata_store.snapshot(),
        "event_bus": {"subscriber_count": resolved_services.event_bus.subscriber_count},
        "runtime_stats": runtime_stats.model_dump(mode="json"),
        "cache_stats": cache_stats.model_dump(mode="json"),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        platform_payload = payload["runtime_stats"]["platform"]
        host_supported_runtimes = sum(
            1
            for runtime in payload["runtime_stats"]["runtimes"]
            if runtime.get("host_platform_supported", True)
        )
        print(f"{payload['service']} {payload['version']}")
        print(
            "platform: "
            f"{platform_payload['system']} {platform_payload['machine']} "
            f"(release {platform_payload['release']}, Python {platform_payload['python_version']})",
        )
        print(f"database: {payload['storage']['database_path']}")
        print(f"models tracked: {payload['storage']['model_count']}")
        print(f"privacy mode: {'on' if settings.privacy_mode else 'off'}")
        print(f"outbound network: {'enabled' if settings.allow_outbound_network else 'disabled'}")
        print(f"rate limit: {settings.rate_limit_requests} requests / {settings.rate_limit_window_seconds}s")
        print(f"audit log: {'enabled' if settings.audit_log_enabled else 'disabled'}")
        print(f"encrypted persistence: {'enabled' if settings.persistence_encryption_enabled else 'disabled'}")
        print(f"tool authorization: {'required' if settings.tool_authorization_required else 'not required'}")
        print(f"parser sandbox: {'enabled' if settings.parser_sandbox_enabled else 'disabled'}")
        print(f"conversion sandbox: {'enabled' if settings.conversion_sandbox_enabled else 'disabled'}")
        install_profiles = payload["install_profiles"]
        print("active install profiles: " + ", ".join(install_profiles["active_profile_ids"]))
        recommended_profile = install_profiles.get("recommended_profile_id")
        if isinstance(recommended_profile, str) and recommended_profile:
            print(f"recommended runtime profile: {recommended_profile}")
        for profile in install_profiles.get("profiles", []):
            if not isinstance(profile, dict):
                continue
            state = "active" if profile.get("installed") else "inactive"
            if profile.get("installed") and not profile.get("ready"):
                state = "installed but host-blocked"
            print(f"install profile {profile.get('profile')}: {state}")
            notes = profile.get("notes")
            if isinstance(notes, list):
                for note in notes[:2]:
                    if isinstance(note, str) and note:
                        print(f"  note: {note}")
        print(f"file roots: {', '.join(str(root) for root in settings.file_access_roots)}")
        print(f"cache artifacts: {payload['cache_stats']['artifact_count']}")
        print(f"runtime response cache: {payload['cache_stats']['runtime_response_count']} entries")
        print(f"runtime policy: {payload['runtime_stats']['runtime_policy']}")
        cluster_status = payload["runtime_stats"].get("cluster")
        if isinstance(cluster_status, dict):
            print(
                "cluster: "
                f"role={cluster_status.get('role')}, "
                f"ready_workers={cluster_status.get('ready_worker_count')}, "
                f"plans={cluster_status.get('plan_count')}",
            )
        print(f"validation manifests: {payload['runtime_stats']['validation_manifest_count']}")
        print(f"host-supported runtimes: {host_supported_runtimes}/{len(payload['runtime_stats']['runtimes'])}")
        for pack in payload["runtime_stats"].get("runtime_packs", []):
            if not isinstance(pack, dict):
                continue
            print(f"runtime pack {pack.get('name')}: {pack.get('status')}")
        for pack in payload["runtime_stats"].get("feature_packs", []):
            if not isinstance(pack, dict):
                continue
            print(f"feature pack {pack.get('name')}: {pack.get('status')}")
        print(
            "scheduler: "
            f"active={payload['runtime_stats']['request_scheduler']['active_requests']}, "
            f"queued={payload['runtime_stats']['request_scheduler']['queued_requests']}, "
            f"rejected={payload['runtime_stats']['request_scheduler']['rejected_requests']}",
        )
        print(
            "load scheduler: "
            f"active={payload['runtime_stats']['load_scheduler']['active_requests']}, "
            f"queued={payload['runtime_stats']['load_scheduler']['queued_requests']}, "
            f"rejected={payload['runtime_stats']['load_scheduler']['rejected_requests']}",
        )
        print(f"runtime requests: {payload['runtime_stats']['request_metrics']['total_requests']}")
        print(f"runtime failures: {payload['runtime_stats']['request_metrics']['failure_count']}")
        average_execution_seconds = payload["runtime_stats"]["request_metrics"]["average_execution_seconds"]
        if average_execution_seconds is not None:
            print(f"avg execution: {average_execution_seconds}s")
        for capability_metrics in payload["runtime_stats"]["request_metrics"]["capabilities"]:
            print(
                f"capability {capability_metrics['capability']}: "
                f"requests={capability_metrics['request_count']}, "
                f"avg execution={capability_metrics['average_execution_seconds']}s",
            )
        benchmark_summary = payload["runtime_stats"]["benchmark_summary"]
        print(f"benchmarks recorded: {benchmark_summary['total_runs']}")
        if benchmark_summary["last_run_at"] is not None:
            print(f"last benchmark: {benchmark_summary['last_run_at']}")
        artifact_summary = benchmark_summary.get("artifact_summary", {})
        latest_artifact = artifact_summary.get("latest_artifact")
        if latest_artifact is not None:
            print(f"latest benchmark artifact: {latest_artifact['artifact_path']}")
            print(f"latest regression status: {latest_artifact['regression_status']}")
        measured_registry = payload["runtime_stats"].get("measured_capability_registry")
        if isinstance(measured_registry, dict):
            print(f"measured probes: {measured_registry.get('total_records', 0)}")
            for category in measured_registry.get("categories", [])[:6]:
                if not isinstance(category, dict):
                    continue
                print(
                    f"  measured {category.get('category')}: "
                    f"{category.get('status')} ({category.get('record_count', 0)} record(s))",
                )
        optimization_defaults = payload["runtime_stats"].get("optimization_defaults")
        if isinstance(optimization_defaults, dict):
            print(
                "optimization defaults: "
                f"complete={'yes' if optimization_defaults.get('complete') else 'no'}, "
                f"models={optimization_defaults.get('resolved_model_count', 0)}/{optimization_defaults.get('model_count', 0)}, "
                f"classes={len(optimization_defaults.get('resolved_classes', []))}/{len(optimization_defaults.get('optimization_classes', []))}",
            )
            for item in optimization_defaults.get("models", [])[:10]:
                if not isinstance(item, dict):
                    continue
                unresolved = item.get("unresolved_classes") or []
                print(
                    f"  default {item.get('display_name', item.get('model_id', 'model'))}: "
                    f"runtime={item.get('runtime')}, "
                    f"profile={item.get('profile_id') or 'none'}, "
                    f"{'resolved' if not unresolved else 'unresolved=' + ','.join(unresolved)}",
                )
                workload_defaults = item.get("workload_defaults")
                if isinstance(workload_defaults, list) and workload_defaults:
                    print(f"    workloads: {_format_workload_defaults_summary(workload_defaults)}")
        for target in payload["runtime_stats"]["target_platforms"]:
            print(
                f"target {target['system']} {target['machine']}: "
                f"models={target['compatible_model_count']} compatible, "
                f"verified={target['verified_model_count']}, "
                f"fallbacks={target['fallback_model_count']}, "
                f"runtimes={target['supported_runtime_count']} supported, "
                f"readiness={target['readiness_state']}",
            )
            if target["verified_hosts"]:
                print(f"  verified hosts: {', '.join(target['verified_hosts'])}")
        print("performance features:")
        _print_performance_features(payload["runtime_stats"]["performance_features"])
    return ExitCode.OK


def _format_workload_defaults_summary(workload_defaults: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for item in workload_defaults[:5]:
        if not isinstance(item, dict):
            continue
        workload_class = str(item.get("workload_class", "workload"))
        runtime = str(item.get("runtime") or "unroutable")
        details: list[str] = []
        modality_path = item.get("modality_path")
        if isinstance(modality_path, str) and modality_path:
            details.append(modality_path)
        profile_id = item.get("profile_id")
        if isinstance(profile_id, str) and profile_id:
            details.append(f"profile={profile_id}")
        else:
            details.append(str(item.get("profile_status") or "no-profile"))
        if item.get("benchmark_backed"):
            details.append("bench")
        rendered.append(f"{workload_class}->{runtime}({','.join(details)})")
    if len(workload_defaults) > 5:
        rendered.append(f"+{len(workload_defaults) - 5} more")
    return ", ".join(rendered)


def handle_config(args: argparse.Namespace, settings: LewLMSettings, services: LewLMServices | None = None) -> ExitCode:
    payload = settings.redacted_snapshot()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_config_summary(settings)
    return ExitCode.OK


def handle_cluster_status(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    status = resolved_services.cluster_service.status()
    payload = status.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_cluster_status(status)
    return ExitCode.OK


def handle_cluster_issue_token(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    payload = resolved_services.cluster_service.issue_enrollment_token(
        worker_name=args.worker_name,
        capabilities=args.capabilities or [CapabilityName.CHAT.value],
        ttl_seconds=args.ttl_seconds,
    ).model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"cluster: {payload['cluster_name']}")
        print(f"expires: {payload['expires_at']}")
        print(f"token: {payload['token']}")
    return ExitCode.OK


def handle_cluster_join(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    endpoint = args.endpoint or resolved_services.cluster_service.public_base_url()
    response_payload = asyncio.run(
        resolved_services.cluster_service.transport.request_json(
            method="POST",
            base_url=args.coordinator,
            path="/v1/cluster/workers/enroll",
            payload={
                "token": args.token,
                "worker_name": args.worker_name,
                "endpoint": endpoint,
                "capabilities": [CapabilityName.CHAT.value],
                "metadata": {"node_name": settings.cluster_node_name},
            },
        ),
    )
    response = ClusterEnrollWorkerResponse.model_validate(response_payload)
    session = resolved_services.cluster_service.complete_worker_enrollment(response)
    payload = {
        "worker": response.worker.model_dump(mode="json"),
        "session": {**session.model_dump(mode="json"), "session_token": "<redacted>"},
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"enrolled worker {response.worker.worker_name} ({response.worker.worker_id})")
        print(f"coordinator: {session.coordinator_url}")
    return ExitCode.OK


def handle_cluster_heartbeat(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    session = resolved_services.cluster_service.worker_session()
    if session is None:
        raise ConfigurationError("Enroll the worker first with `lewlm cluster join`.")
    coordinator = args.coordinator or session.coordinator_url
    payload = asyncio.run(
        resolved_services.cluster_service.transport.request_json(
            method="POST",
            base_url=coordinator,
            path="/v1/cluster/workers/heartbeat",
            payload={"worker_id": session.worker_id, "session_token": session.session_token},
        ),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"heartbeat recorded for {session.worker_name}")
    return ExitCode.OK


def handle_cluster_plan(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    manifest = resolved_services.model_registry.get_manifest(args.model)
    plan = resolved_services.cluster_service.plan_manifest(manifest)
    payload = plan.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"model: {payload['model_id']}")
        print(f"stages: {payload['stage_count']}")
        scheduling = payload.get("scheduling", {})
        if isinstance(scheduling, dict):
            print(
                "tuning: "
                f"mode={scheduling.get('selection_mode', 'n/a')} "
                f"batch={scheduling.get('effective_batch_tokens', 'n/a')} "
                f"prefetch={scheduling.get('average_prefetch_tokens', 'n/a')} "
                f"heterogeneity={scheduling.get('heterogeneity_ratio', 'n/a')}",
            )
        for assignment in payload["assignments"]:
            print(
                f"- stage {assignment['stage_index']}: {assignment['worker_name']} "
                f"[layers {assignment['start_layer']}:{assignment['end_layer']}, "
                f"weight={assignment.get('relative_weight', 'n/a')}, "
                f"batch={assignment.get('target_batch_tokens', 'n/a')}, "
                f"prefetch={assignment.get('prefetch_tokens', 'n/a')}]",
            )
    return ExitCode.OK


def handle_cluster_benchmark(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    result = asyncio.run(
        resolved_services.telemetry_service.benchmark(
            model_id=args.model,
            prompt=args.prompt,
            capability=CapabilityName.CHAT.value,
        ),
    )
    payload = result.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"model: {payload['model_id']}")
        print(f"runtime: {payload['runtime']}")
        print(f"load: {payload['load_seconds']}s")
        print(f"generate: {payload['generate_seconds']}s")
        print(f"total: {payload['total_seconds']}s")
        if isinstance(payload.get("usage"), dict):
            print(
                "distributed: "
                f"stages={payload['usage'].get('distributed_stage_count')}, "
                f"workers={payload['usage'].get('distributed_worker_count')}, "
                f"recoveries={payload['usage'].get('distributed_recovery_count')}",
            )
        _print_benchmark_scenario_highlights(payload)
        _print_cluster_benchmark_details(payload)
    return ExitCode.OK


def handle_scan(args: argparse.Namespace, settings: LewLMSettings, services: LewLMServices | None = None) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    roots = [Path(path) for path in args.paths] if args.paths else None
    summary = resolved_services.model_registry.scan(roots=roots)
    if args.json:
        print(summary.model_dump_json(indent=2))
    else:
        _print_scan_summary(summary)
    return ExitCode.OK


def handle_list_models(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    inventory = resolved_services.model_registry.inventory()
    if args.json or args.all:
        raw_manifests = resolved_services.model_registry.list_manifests()
        raw_inventory = ModelInventory(count=len(raw_manifests), items=raw_manifests)
        if args.json:
            print(raw_inventory.model_dump_json(indent=2))
        else:
            _print_inventory_raw(raw_inventory)
    else:
        _print_inventory(inventory)
    return ExitCode.OK


def handle_list_skills(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    skills = resolved_services.skill_catalog_service.list_skills()
    payload = {
        "count": len(skills),
        "items": [skill.model_dump(mode="json") for skill in skills],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_skills(skills)
    return ExitCode.OK


def handle_show_skill(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    skill = resolved_services.skill_catalog_service.get_skill(args.skill_name)
    if args.json:
        print(json.dumps(skill.model_dump(mode="json"), indent=2))
    else:
        _print_skill_detail(skill)
    return ExitCode.OK


def handle_list_tools(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    tools = resolved_services.tool_catalog_service.list_tools()
    payload = {
        "count": len(tools),
        "items": [tool.model_dump(mode="json") for tool in tools],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_tools(tools)
    return ExitCode.OK


def handle_show_tool(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    tool = resolved_services.tool_catalog_service.get_tool(args.tool_name)
    if args.json:
        print(json.dumps(tool.model_dump(mode="json"), indent=2))
    else:
        _print_tool_detail(tool)
    return ExitCode.OK


def handle_list_sessions(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    sessions = resolved_services.session_history_service.list_sessions()
    payload = {
        "count": len(sessions),
        "items": [session.model_dump(mode="json") for session in sessions],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_sessions(sessions)
    return ExitCode.OK


def handle_show_session(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    session = resolved_services.session_history_service.get_session_detail(args.session_id)
    if args.json:
        print(json.dumps(session.model_dump(mode="json"), indent=2))
    else:
        _print_session_detail(session)
    return ExitCode.OK


def handle_export_session(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    output_path = Path(args.output).expanduser()
    bundle = resolved_services.session_history_service.export_session(args.session_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8")
    resolved_services.audit_logger.record(
        action="session_export",
        outcome="success",
        actor="cli",
        details={"session_id": args.session_id, "output_path": str(output_path), "turn_count": len(bundle.turns)},
    )
    payload = {
        "session_id": args.session_id,
        "title": bundle.session.title,
        "turn_count": len(bundle.turns),
        "output_path": str(output_path),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"exported {args.session_id} to {output_path}")
    return ExitCode.OK


def handle_import_session(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    input_path = Path(args.input).expanduser().resolve(strict=False)
    _, bundle_text = read_scoped_text_file(
        input_path,
        allowed_roots=(input_path.parent,),
        purpose="Session bundle",
        media_type="application/json",
    )
    bundle = SessionExportBundle.model_validate_json(bundle_text)
    session = resolved_services.session_history_service.import_session(bundle, title=args.title)
    resolved_services.audit_logger.record(
        action="session_import",
        outcome="success",
        actor="cli",
        details={"session_id": session.session_id, "input_path": str(input_path), "turn_count": len(session.turns)},
    )
    payload = {
        "session_id": session.session_id,
        "title": session.title,
        "turn_count": session.turn_count,
        "message_count": session.message_count,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"imported session {session.session_id} ({session.turn_count} turns)")
    return ExitCode.OK


def handle_delete_session(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    session = resolved_services.session_history_service.delete_session(args.session_id)
    resolved_services.audit_logger.record(
        action="session_delete",
        outcome="success",
        actor="cli",
        details={"session_id": session.session_id},
    )
    payload = {"status": "deleted", "session_id": session.session_id}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"deleted session {session.session_id}")
    return ExitCode.OK


def handle_capabilities(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    report = resolved_services.model_router.model_capability_report(args.model)
    payload = report.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_capability_report(report)
    return ExitCode.OK


def handle_convert(args: argparse.Namespace, settings: LewLMSettings, services: LewLMServices | None = None) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    resolved_services.tool_authorizer.require(
        ToolAction.MODEL_CONVERSION,
        authorizations=args.authorize,
        actor="cli",
        details={"model_id": args.model, "policy": args.policy},
    )
    request = ConversionJobRequest(
        model_id=args.model,
        policy=ConversionPolicy(args.policy),
        custom_bits=args.custom_bits,
        quantization_profile=_conversion_quantization_profile_from_args(args),
        force=args.force,
        idempotency_key=args.idempotency_key,
        authorized_actions=list(args.authorize),
    )
    job = resolved_services.conversion_service.submit(request)
    if not args.no_wait and job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
        job = _wait_for_job(resolved_services, job.job_id)
    payload = job.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        suffix = " [replayed]" if payload.get("idempotent_replay") else ""
        print(f"{payload['job_id']} [{payload['status']}] {suffix}".rstrip())
        if payload["payload"].get("result_path"):
            print(f"result: {payload['payload']['result_path']}")
    return ExitCode.OK if job.status != JobStatus.FAILED else ExitCode.ERROR


def _conversion_quantization_profile_from_args(args: argparse.Namespace) -> QuantizationProfile | None:
    profile_strategy = getattr(args, "profile", None)
    profile_name = getattr(args, "profile_name", None)
    weight_precision = getattr(args, "weight_precision", None)
    activation_precision = getattr(args, "activation_precision", None)
    compute_precision = getattr(args, "compute_precision", None)
    kv_cache_precision = getattr(args, "kv_cache_precision", None)
    calibration_samples = getattr(args, "calibration_samples", None)
    layer_override_specs = list(getattr(args, "layer_override", []) or [])
    external_quantizer = getattr(args, "external_quantizer", None)
    external_profile = getattr(args, "external_profile", None)
    external_module = getattr(args, "external_module", None)
    if not any(
        (
            profile_strategy,
            profile_name,
            weight_precision,
            activation_precision,
            compute_precision,
            kv_cache_precision,
            calibration_samples is not None,
            layer_override_specs,
            external_quantizer,
            external_profile,
            external_module,
        ),
    ):
        return None
    return QuantizationProfile(
        name=profile_name,
        strategy=QuantizationStrategy(profile_strategy or QuantizationStrategy.WEIGHT_ONLY.value),
        weight_precision=QuantizationPrecision(weight_precision) if weight_precision is not None else None,
        activation_precision=(
            QuantizationPrecision(activation_precision) if activation_precision is not None else None
        ),
        compute_precision=QuantizationPrecision(compute_precision) if compute_precision is not None else None,
        kv_cache_precision=QuantizationPrecision(kv_cache_precision) if kv_cache_precision is not None else None,
        calibration_samples=calibration_samples,
        layer_overrides=[_parse_layer_override_spec(spec) for spec in layer_override_specs],
        external_quantizer=(
            ExternalQuantizerReference(
                name=external_quantizer,
                profile=external_profile,
                module=external_module,
                required_packages=[external_module] if external_module else [],
            )
            if external_quantizer is not None
            else None
        ),
    )


def _parse_layer_override_spec(spec: str) -> LayerQuantizationOverride:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ConfigurationError(
            "Layer overrides must use layer_pattern:weight_precision[:activation_precision[:compute_precision]].",
        )
    return LayerQuantizationOverride(
        layer_pattern=parts[0],
        weight_precision=QuantizationPrecision(parts[1]),
        activation_precision=QuantizationPrecision(parts[2]) if len(parts) > 2 and parts[2] else None,
        compute_precision=QuantizationPrecision(parts[3]) if len(parts) > 3 and parts[3] else None,
    )


def handle_benchmark(args: argparse.Namespace, settings: LewLMSettings, services: LewLMServices | None = None) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    if args.workload_class is not None and args.capability != CapabilityName.CHAT.value:
        raise ConfigurationError("`--workload-class` is only supported for chat benchmarks.")
    if args.all and args.model is not None:
        raise ConfigurationError("Use either `--model` or `--all` for `lewlm benchmark`, not both.")
    if args.all and is_attachment_workload_class(args.workload_class):
        raise ConfigurationError("Use `--model` with attachment-bearing benchmark workload classes.")
    if not args.all and args.repeat != 1:
        raise ConfigurationError("Use `--repeat` only with `--all` for `lewlm benchmark`.")
    if args.warmup_runs < 0:
        raise ConfigurationError("`--warmup-runs` must be zero or greater.")
    if args.compare_direct and args.compare_external_adapter:
        raise ConfigurationError("Choose either `--compare-direct` or `--compare-external-adapter`, not both.")
    if args.workload_class is not None and (args.compare_direct or args.compare_external_adapter):
        raise ConfigurationError("`--workload-class` is only supported for managed benchmark runs.")
    if args.compare_direct:
        resolved_services.model_registry.scan()
        if args.capability != CapabilityName.CHAT.value:
            raise ConfigurationError("`--compare-direct` currently supports chat benchmarks only.")
        if args.convert_policies and not args.convert_missing:
            raise ConfigurationError("Use `--convert-policy` only together with `--convert-missing`.")
        if args.convert_profiles and not args.convert_missing:
            raise ConfigurationError("Use `--convert-profile` only together with `--convert-missing`.")
        with _filtered_benchmark_noise():
            payload = _run_direct_benchmark_suite(
                args,
                settings,
                resolved_services,
                progress=_emit_benchmark_progress if not args.json else None,
            )
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            _print_direct_benchmark_suite(payload)
        return ExitCode.ERROR if payload["summary"]["failed_run_count"] else ExitCode.OK
    if args.compare_external_adapter:
        resolved_services.model_registry.scan()
        if args.capability != CapabilityName.CHAT.value:
            raise ConfigurationError("`--compare-external-adapter` currently supports chat benchmarks only.")
        if args.all:
            raise ConfigurationError("`--compare-external-adapter` currently requires a single `--model` benchmark target.")
        if args.model is None:
            raise ConfigurationError("Use `--model` with `--compare-external-adapter`.")
        with _filtered_benchmark_noise():
            payload = _run_external_adapter_benchmark(args, settings, resolved_services)
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            _print_external_adapter_benchmark(payload)
        return ExitCode.ERROR if payload.get("status") == "failed" else ExitCode.OK
    if args.all:
        result = asyncio.run(
            resolved_services.telemetry_service.benchmark_suite_lightweight(
                prompt=args.prompt,
                capability=args.capability,
                repeat_count=args.repeat,
                warmup_run_count=args.warmup_runs,
                apply_serving_profile=not args.disable_serving_profile,
                workload_class=args.workload_class,
            ),
        )
    else:
        result = asyncio.run(
            resolved_services.telemetry_service.benchmark_lightweight(
                model_id=args.model,
                prompt=args.prompt,
                capability=args.capability,
                warmup_run_count=args.warmup_runs,
                apply_serving_profile=not args.disable_serving_profile,
                workload_class=args.workload_class,
            ),
        )
    payload = result.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.all:
        print(
            f"benchmarked {payload['benchmark_count']} {payload['capability']} run(s) across "
            f"{payload['model_count']} model(s)",
        )
        if payload.get("workload_class") is not None:
            print(f"workload class: {payload['workload_class']}")
        print(f"repeats: {payload['repeat_count']}")
        print(f"load: {payload['total_load_seconds']}s")
        print(f"generate: {payload['total_generate_seconds']}s")
        print(f"elapsed: {payload['total_elapsed_seconds']}s")
        if payload["average_total_seconds"] is not None:
            print(f"average total: {payload['average_total_seconds']}s")
        suite_ttft = payload.get("metric_summaries", {}).get("ttft_seconds", {})
        suite_warm = payload.get("metric_summaries", {}).get("warm_total_seconds", {})
        if suite_warm.get("status") == "completed":
            print(
                f"average warm total: {_format_seconds(_coerce_float(suite_warm.get('optimized_average')))}",
            )
        if suite_ttft.get("status") == "completed":
            print(f"average ttft: {_format_seconds(_coerce_float(suite_ttft.get('optimized_average')))}")
        for model_summary in payload["models"]:
            print(
                f"* {model_summary['model_id']}: runs={model_summary['run_count']} "
                f"avg={model_summary['average_total_seconds']}s fastest={model_summary['fastest_total_seconds']}s",
            )
        for item in payload["results"]:
            speculation_suffix = _benchmark_speculation_suffix(item)
            print(
                f"- {item['model_id']} via {item['runtime']}: "
                f"load={item['load_seconds']}s generate={item['generate_seconds']}s total={item['total_seconds']}s"
                f"{speculation_suffix}",
            )
            _print_serving_profile_summary(item.get("serving_profile"), prefix="  ")
        print("performance features:")
        _print_performance_features(payload["performance_features"])
        _print_benchmark_scenario_highlights(payload)
        if payload.get("artifact") is not None:
            print(f"artifact: {payload['artifact']['artifact_path']}")
        if payload.get("regression") is not None:
            print(f"regression: {payload['regression']['status']}")
    else:
        print(f"model: {payload['model_id']}")
        print(f"runtime: {payload['runtime']}")
        print(f"capability: {payload['capability']}")
        if payload.get("workload_class") is not None:
            print(f"workload class: {payload['workload_class']}")
        print(f"load: {payload['load_seconds']}s")
        print(f"generate: {payload['generate_seconds']}s")
        print(f"total: {payload['total_seconds']}s")
        print(f"recorded: {payload['created_at']}")
        if payload["completion_tokens_per_second"] is not None:
            print(f"throughput: {payload['completion_tokens_per_second']} tokens/s")
        phase_breakdown = payload.get("phase_breakdown") or {}
        warm_total_seconds = _coerce_float(phase_breakdown.get("warm_total_seconds"))
        ttft_seconds = _coerce_float(phase_breakdown.get("ttft_seconds"))
        steady_decode_seconds = _coerce_float(phase_breakdown.get("steady_state_decode_seconds"))
        steady_decode_rate = _coerce_float(phase_breakdown.get("steady_state_decode_tokens_per_second"))
        if warm_total_seconds is not None:
            print(f"warm total: {warm_total_seconds}s")
        if ttft_seconds is not None:
            print(f"ttft: {ttft_seconds}s")
        if steady_decode_seconds is not None:
            print(f"steady decode: {steady_decode_seconds}s")
        if steady_decode_rate is not None:
            print(f"steady decode throughput: {steady_decode_rate} tokens/s")
        selected_speculation_mode = _selected_benchmark_speculation_mode(payload)
        if selected_speculation_mode is not None:
            print(f"speculation: {selected_speculation_mode}")
        _print_serving_profile_summary(payload.get("serving_profile"))
        print("performance features:")
        _print_performance_features(payload["performance_features"])
        _print_benchmark_scenario_highlights(payload)
        if payload.get("artifact") is not None:
            print(f"artifact: {payload['artifact']['artifact_path']}")
        if payload.get("regression") is not None:
            print(f"regression: {payload['regression']['status']}")
    regression = payload.get("regression")
    if isinstance(regression, dict) and regression.get("status") == "failed":
        return ExitCode.ERROR
    return ExitCode.OK


def handle_autotune(args: argparse.Namespace, settings: LewLMSettings, services: LewLMServices | None = None) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    result = asyncio.run(
        resolved_services.telemetry_service.autotune(
            model_id=args.model,
            prompt=args.prompt,
            capability=args.capability,
            workload_class=args.workload_class,
        ),
    )
    payload = result.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2))
        return ExitCode.OK
    print(f"model: {payload['model_id']}")
    print(f"runtime: {payload['runtime']}")
    print(f"capability: {payload['capability']}")
    print(f"workload class: {payload['workload_class']}")
    print(f"profile id: {payload['profile_id']}")
    print(f"recorded: {payload['recommended_at']}")
    print(f"objective: {payload['selection_objective']}")
    print(f"reason: {payload['reason']}")
    metrics = payload.get("metrics") or {}
    if metrics:
        print(
            "metrics: "
            + ", ".join(
                f"{key}={value}"
                for key, value in metrics.items()
                if value is not None
            ),
        )
    if payload.get("selected_speculation_mode") is not None:
        print(f"speculation: {payload['selected_speculation_mode']}")
    if payload.get("quantization_profile") is not None:
        print(f"quantization: {payload['quantization_profile']}")
    if payload.get("active_kernel_path") is not None:
        print(f"kernel path: {payload['active_kernel_path']}")
    if payload.get("active_cache_features"):
        print("cache features: " + ", ".join(payload["active_cache_features"]))
    print("recommended settings:")
    for key, value in payload["effective_settings"].items():
        print(f"- {key}={value}")
    print("candidate summary:")
    for candidate in payload["candidate_summaries"]:
        candidate_metrics = [f"total={candidate['total_seconds']}s"]
        if candidate.get("continuous_batching_throughput") is not None:
            candidate_metrics.append(f"batch_rps={candidate['continuous_batching_throughput']}")
        if candidate.get("warm_cache_ttft_ratio") is not None:
            candidate_metrics.append(f"warm_ratio={candidate['warm_cache_ttft_ratio']}")
        print(f"- {candidate['name']} ({', '.join(candidate_metrics)})")
    if payload.get("artifact") is not None:
        print(f"artifact: {payload['artifact']['artifact_path']}")
    return ExitCode.OK


def _run_direct_benchmark_suite(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    prompt = args.prompt
    benchmark_targets = _compare_direct_benchmark_targets(args, services, progress=progress)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results: list[dict[str, Any]] = []
    total_runs = len(benchmark_targets) * (args.repeat if args.all else 1)
    if progress is not None:
        progress(f"running compare-direct benchmark suite across {total_runs} target run(s)...")
    try:
        for run_index in range(args.repeat if args.all else 1):
            for target in benchmark_targets:
                source_manifest = target["source_manifest"]
                benchmark_manifest = target["benchmark_manifest"]
                conversion_payload = dict(target["conversion"])
                run_position = len(results) + 1
                if benchmark_manifest is None:
                    if progress is not None:
                        progress(
                            f"[{run_position}/{total_runs}] skipping {source_manifest.display_name}: "
                            "no runnable benchmark artifact was available.",
                        )
                    skipped_status = "failed" if conversion_payload.get("status") == "failed" else "skipped"
                    results.append(
                        {
                            "run_index": run_index + 1,
                            "model_id": source_manifest.model_id,
                            "display_name": source_manifest.display_name,
                            "benchmark_model_id": None,
                            "benchmark_display_name": None,
                            "format_type": source_manifest.format_type.value,
                            "source_path": source_manifest.source_path,
                            "conversion": conversion_payload,
                            "direct": {
                                "status": skipped_status,
                                "reason": "No runnable benchmark artifact was available for this model.",
                            },
                            "optimized": {
                                "status": skipped_status,
                                "reason": "No runnable benchmark artifact was available for this model.",
                            },
                            "comparison": _benchmark_comparison_summary(
                                {"status": "skipped"},
                                {"status": "skipped"},
                                primary_metric=args.compare_metric,
                            ),
                        },
                    )
                    continue
                if progress is not None:
                    progress(f"[{run_position}/{total_runs}] running direct baseline: {benchmark_manifest.display_name}")
                direct_payload = _run_direct_benchmark_once(
                    manifest=benchmark_manifest,
                    prompt=prompt,
                    warmup_run_count=args.warmup_runs,
                )
                if progress is not None:
                    progress(f"[{run_position}/{total_runs}] running LewLM-managed benchmark: {benchmark_manifest.display_name}")
                optimized_payload = _invoke_managed_benchmark_once(
                    services=services,
                    model_id=benchmark_manifest.model_id,
                    prompt=prompt,
                    warmup_run_count=args.warmup_runs,
                    apply_serving_profile=not args.disable_serving_profile,
                )
                results.append(
                    {
                        "run_index": run_index + 1,
                        "model_id": source_manifest.model_id,
                        "display_name": source_manifest.display_name,
                        "benchmark_model_id": benchmark_manifest.model_id,
                        "benchmark_display_name": benchmark_manifest.display_name,
                        "format_type": source_manifest.format_type.value,
                        "source_path": source_manifest.source_path,
                        "conversion": conversion_payload,
                        "direct": direct_payload,
                        "optimized": optimized_payload,
                        "comparison": _benchmark_comparison_summary(
                            direct_payload,
                            optimized_payload,
                            primary_metric=args.compare_metric,
                        ),
                    },
                )
                asyncio.run(services.runtime_catalog.unload_all_models())
    finally:
        _cleanup_transient_benchmark_manifests(services, benchmark_targets)
    _annotate_profile_metrics(results)
    models = _benchmark_model_summaries(results, primary_metric=args.compare_metric)
    payload = {
        "benchmark_type": "direct_chat_comparison",
        "capability": CapabilityName.CHAT.value,
        "prompt": prompt,
        "host_platform": services.runtime_catalog.host_platform_snapshot().model_dump(mode="json"),
        "comparison_controls": _benchmark_comparison_controls(args),
        "repeat_count": args.repeat if args.all else 1,
        "benchmark_count": len(results),
        "model_count": len(
            {str(target["source_manifest"].model_id) for target in benchmark_targets if isinstance(target["source_manifest"], ModelManifest)}
        ),
        "created_at": created_at,
        "models": models,
        "results": results,
        "summary": _benchmark_suite_summary(
            results,
            primary_metric=args.compare_metric,
            model_summaries=models,
        ),
    }
    artifact_path = _write_direct_benchmark_artifact(settings=settings, payload=payload)
    payload["artifact"] = {"artifact_path": str(artifact_path)}
    return payload


def _run_external_adapter_benchmark(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices,
) -> dict[str, Any]:
    manifest = services.model_registry.get_manifest(args.model)
    if manifest.conversion_status != ConversionStatus.RUNNABLE:
        raise ConfigurationError("`--compare-external-adapter` requires a runnable local model.")
    native_runtime = services.runtime_catalog.select_runtime(manifest, capability=CapabilityName.CHAT)
    external_runtime = services.runtime_catalog.get_runtime(RuntimeAffinity.EXTERNAL_ACCELERATOR)
    if external_runtime is None:
        raise ConfigurationError("No external accelerator runtime is registered in the current catalog.")
    external_candidate_report = getattr(external_runtime, "candidate_report", None)
    if callable(external_candidate_report):
        candidate_report = external_candidate_report(manifest)
        if not candidate_report.available or not candidate_report.supports_manifest:
            raise ConfigurationError(
                candidate_report.availability_reason
                or "The configured external accelerator is not ready for the selected model."
            )
    elif not external_runtime.supports_manifest(manifest):
        raise ConfigurationError("The configured external accelerator is not compatible with the selected model.")

    native_payload = _safe_runtime_benchmark(
        runtime=native_runtime,
        manifest=manifest,
        prompt=args.prompt,
        max_tokens=_CLI_CHAT_BENCHMARK_MAX_TOKENS,
        warmup_run_count=args.warmup_runs,
    )
    external_payload = _safe_runtime_benchmark(
        runtime=external_runtime,
        manifest=manifest,
        prompt=args.prompt,
        max_tokens=_CLI_CHAT_BENCHMARK_MAX_TOKENS,
        warmup_run_count=args.warmup_runs,
    )
    comparison = _external_adapter_comparison_summary(
        native_payload=native_payload,
        external_payload=external_payload,
        primary_metric=args.compare_metric,
    )
    feature_preservation = summarize_feature_preservation(
        native_features=_performance_features_payload(native_payload),
        external_features=_performance_features_payload(external_payload),
    )
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _persist_external_adapter_probe_records(
        services=services,
        manifest=manifest,
        external_runtime=external_runtime,
        feature_preservation=feature_preservation,
        created_at=created_at,
    )
    routing_preference = _persist_external_adapter_runtime_preference(
        services=services,
        manifest=manifest,
        native_runtime=native_runtime,
        external_runtime=external_runtime,
        comparison=comparison,
        feature_preservation=feature_preservation,
        created_at=created_at,
    )
    payload = {
        "status": "completed"
        if native_payload.get("status") == "completed" and external_payload.get("status") == "completed"
        else "failed",
        "benchmark_type": "external_adapter_comparison",
        "created_at": created_at,
        "model_id": manifest.model_id,
        "display_name": manifest.display_name,
        "prompt": args.prompt,
        "comparison_controls": {
            "primary_metric": args.compare_metric,
            "warmup_run_count": args.warmup_runs,
            "adapter_profile": settings.external_accelerator_profile,
            "adapter_base_url": settings.external_accelerator_base_url,
        },
        "native": native_payload,
        "external_adapter": external_payload,
        "comparison": comparison,
        "feature_preservation": feature_preservation,
        "routing_preference": routing_preference,
    }
    artifact_path = _write_external_adapter_benchmark_artifact(settings=settings, payload=payload)
    payload["artifact"] = {"artifact_path": str(artifact_path)}
    return payload


def _safe_runtime_benchmark(
    *,
    runtime,
    manifest: ModelManifest,
    prompt: str,
    max_tokens: int,
    warmup_run_count: int,
) -> dict[str, Any]:
    try:
        return benchmark_runtime_chat_manifest(
            runtime,
            manifest,
            prompt=prompt,
            max_tokens=max_tokens,
            warmup_run_count=warmup_run_count,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "model_id": manifest.model_id,
            "display_name": manifest.display_name,
            "runtime": runtime.name,
            "runtime_affinity": runtime.affinity.value,
            "error": str(exc),
            "phase_breakdown": {},
            "performance_features": runtime.performance_feature_snapshot(),
        }


def _persist_external_adapter_probe_records(
    *,
    services: LewLMServices,
    manifest: ModelManifest,
    external_runtime,
    feature_preservation: dict[str, Any],
    created_at: str,
) -> None:
    host_platform = services.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
    details = feature_preservation.get("details", {})
    if not isinstance(details, dict):
        return
    status_map = {
        "preserved": MeasuredCapabilityStatus.SUPPORTED.value,
        "degraded": MeasuredCapabilityStatus.DEGRADED.value,
        "rejected": MeasuredCapabilityStatus.REJECTED.value,
    }
    for feature_name, entry in details.items():
        if not isinstance(entry, dict):
            continue
        status = status_map.get(entry.get("status"))
        if status is None:
            continue
        external_details = entry.get("external", {})
        reason = (
            str(external_details.get("reason"))
            if isinstance(external_details, dict) and external_details.get("reason") is not None
            else f"External adapter preservation measured as `{entry.get('status')}` for `{feature_name}`."
        )
        services.metadata_store.upsert_capability_probe_record(
            category=MeasuredCapabilityCategory.ADAPTER_PRESERVATION.value,
            probe_name=str(feature_name),
            host_platform=host_platform,
            status=status,
            source=MeasuredCapabilityEvidenceSource.EXTERNAL_ADAPTER_COMPARISON.value,
            reason=reason,
            runtime_name=external_runtime.name,
            runtime_affinity=external_runtime.affinity.value,
            model_id=manifest.model_id,
            recorded_at=created_at,
            details={
                "comparison_status": entry.get("status"),
                "feature_detail": entry,
            },
        )


def _persist_external_adapter_runtime_preference(
    *,
    services: LewLMServices,
    manifest: ModelManifest,
    native_runtime,
    external_runtime,
    comparison: dict[str, Any],
    feature_preservation: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    if comparison.get("status") != "completed":
        return {
            "applied": False,
            "reason": "Comparison metrics were unavailable, so no routing preference was persisted.",
        }
    winner = comparison.get("winner")
    if winner not in {"native", "external"}:
        return {
            "applied": False,
            "reason": "No decisive benchmark winner was found.",
        }
    preferred_runtime = native_runtime if winner == "native" else external_runtime
    baseline_runtime = external_runtime if winner == "native" else native_runtime
    selected_metric_value = comparison.get("selected_metric_value")
    baseline_metric_value = comparison.get("baseline_metric_value")
    host_platform = services.runtime_catalog.host_platform_snapshot().model_dump(mode="json")
    persisted_payload = {
        "model_id": manifest.model_id,
        "capability": CapabilityName.CHAT.value,
        "host_platform": host_platform,
        "selected_runtime_affinity": preferred_runtime.affinity.value,
        "selected_runtime_name": preferred_runtime.name,
        "baseline_runtime_affinity": baseline_runtime.affinity.value,
        "baseline_runtime_name": baseline_runtime.name,
        "primary_metric": comparison.get("primary_metric"),
        "selected_metric_value": selected_metric_value,
        "baseline_metric_value": baseline_metric_value,
        "winner": winner,
        "feature_preservation": feature_preservation,
        "recorded_at": created_at,
        "source": "compare_external_adapter",
    }
    assessment = assess_runtime_preference(persisted_payload)
    if assessment is not None:
        persisted_payload["preference_status"] = "adopted" if assessment.adopted else "downgraded"
        persisted_payload["effective_runtime_affinity"] = assessment.effective_runtime_affinity
        persisted_payload["effective_runtime_name"] = assessment.effective_runtime_name
        if assessment.downgrade_reason is not None:
            persisted_payload["downgrade_reason"] = assessment.downgrade_reason
    services.model_registry.metadata_store.upsert_runtime_preference(
        model_id=manifest.model_id,
        capability=CapabilityName.CHAT.value,
        host_platform=host_platform,
        payload=persisted_payload,
    )
    if assessment is not None and not assessment.adopted:
        effective_runtime = assessment.effective_runtime_name or assessment.effective_runtime_affinity or "the managed runtime"
        return {
            "applied": False,
            "persisted": True,
            "selected_runtime_affinity": preferred_runtime.affinity.value,
            "selected_runtime_name": preferred_runtime.name,
            "effective_runtime_affinity": assessment.effective_runtime_affinity,
            "effective_runtime_name": assessment.effective_runtime_name,
            "primary_metric": comparison.get("primary_metric"),
            "selected_metric_value": selected_metric_value,
            "baseline_metric_value": baseline_metric_value,
            "reason": f"Persisted benchmark evidence, but LewLM keeps `{effective_runtime}` because {assessment.downgrade_reason}.",
        }
    return {
        "applied": True,
        "persisted": True,
        "selected_runtime_affinity": preferred_runtime.affinity.value,
        "selected_runtime_name": preferred_runtime.name,
        "baseline_runtime_affinity": baseline_runtime.affinity.value,
        "baseline_runtime_name": baseline_runtime.name,
        "effective_runtime_affinity": assessment.effective_runtime_affinity if assessment is not None else preferred_runtime.affinity.value,
        "effective_runtime_name": assessment.effective_runtime_name if assessment is not None else preferred_runtime.name,
        "primary_metric": comparison.get("primary_metric"),
        "selected_metric_value": selected_metric_value,
        "baseline_metric_value": baseline_metric_value,
    }


def _performance_features_payload(payload: dict[str, Any]) -> dict[str, Any]:
    performance_features = payload.get("performance_features")
    return performance_features if isinstance(performance_features, dict) else {}


def _compare_direct_benchmark_targets(
    args: argparse.Namespace,
    services: LewLMServices,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, ModelManifest | dict[str, Any] | None]]:
    prompt = args.prompt
    if args.model is not None:
        candidate_manifests = [services.model_registry.get_manifest(args.model)]
    elif args.all:
        candidate_manifests = [
            manifest
            for manifest in services.model_registry.list_manifests()
            if _manifest_supports_chat_benchmark(manifest)
            and not _is_conversion_cache_manifest(services.settings, manifest)
        ]
    else:
        manifest, _, _ = services.model_router.route_chat(
            None,
            messages=[GenerateMessage(role="user", content=prompt)],
            max_tokens=_CLI_CHAT_BENCHMARK_MAX_TOKENS,
        )
        candidate_manifests = [manifest]

    runnable_candidate_count = sum(
        1 for manifest in candidate_manifests if manifest.conversion_status == ConversionStatus.RUNNABLE
    )
    conversion_candidate_count = len(candidate_manifests) - runnable_candidate_count
    if progress is not None:
        progress(
            "preparing direct benchmark targets: "
            f"{len(candidate_manifests)} candidate model(s), "
            f"{runnable_candidate_count} runnable, "
            f"{conversion_candidate_count} requiring conversion.",
        )

    targets: list[dict[str, ModelManifest | dict[str, Any] | None]] = []
    for manifest_index, manifest in enumerate(candidate_manifests, start=1):
        if manifest.conversion_status == ConversionStatus.RUNNABLE:
            targets.append(
                {
                    "source_manifest": manifest,
                    "benchmark_manifest": manifest,
                    "conversion": _no_conversion_payload(),
                },
            )
            continue
        if not args.convert_missing:
            if args.model is not None:
                raise ConfigurationError(
                    "The requested model is not runnable yet. Re-run with `--convert-missing` or convert it first.",
                )
            continue
        for request in _benchmark_conversion_requests(args, manifest):
            if progress is not None:
                progress(
                    f"[prepare {manifest_index}/{len(candidate_manifests)}] converting "
                    f"{manifest.display_name} ({_conversion_profile_label_from_request(request)})...",
                )
            conversion_payload, benchmark_manifest = _convert_manifest_for_benchmark(services, manifest, request)
            if progress is not None:
                cache_suffix = " cached" if conversion_payload.get("cache_hit") else ""
                progress(
                    f"[prepare {manifest_index}/{len(candidate_manifests)}] conversion "
                    f"{conversion_payload.get('status', 'unknown')}{cache_suffix}: {manifest.display_name}",
                )
            targets.append(
                {
                    "source_manifest": manifest,
                    "benchmark_manifest": benchmark_manifest,
                    "conversion": conversion_payload,
                },
            )
    if not targets:
        raise ConfigurationError("No runnable chat benchmark targets were available.")
    return targets


def _manifest_supports_chat_benchmark(manifest: ModelManifest) -> bool:
    return any(
        modality in manifest.modality
        for modality in (ModelModality.TEXT, ModelModality.VISION, ModelModality.MULTIMODAL)
    )


def _is_conversion_cache_manifest(settings: LewLMSettings, manifest: ModelManifest) -> bool:
    cache_root = settings.cache_dir / "conversions"
    try:
        return Path(manifest.source_path).expanduser().resolve(strict=False).is_relative_to(cache_root)
    except ValueError:
        return False


def _no_conversion_payload() -> dict[str, Any]:
    return {
        "needed": False,
        "status": "not_required",
        "request": None,
        "profile_label": None,
        "cache_hit": False,
        "duration_seconds": None,
        "result_path": None,
        "job_id": None,
        "logs_tail": [],
    }


def _benchmark_conversion_requests(args: argparse.Namespace, manifest: ModelManifest) -> list[ConversionJobRequest]:
    requested_policies = list(args.convert_policies or []) or [ConversionPolicy.BALANCED.value]
    requests = [
        ConversionJobRequest(
            model_id=manifest.model_id,
            policy=ConversionPolicy(policy_value),
        )
        for policy_value in requested_policies
    ]
    requests.extend(
        ConversionJobRequest(
            model_id=manifest.model_id,
            policy=ConversionPolicy.BALANCED,
            quantization_profile=_benchmark_quantization_profile_preset(profile_name),
        )
        for profile_name in list(args.convert_profiles or [])
    )
    deduped_requests: list[ConversionJobRequest] = []
    seen_signatures: set[str] = set()
    for request in requests:
        signature = json.dumps(
            request.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped_requests.append(request)
    return deduped_requests


def _benchmark_quantization_profile_preset(profile_name: str) -> QuantizationProfile:
    strategy = QuantizationStrategy(profile_name)
    if strategy == QuantizationStrategy.ACTIVATION_AWARE:
        return QuantizationProfile(
            name=profile_name,
            strategy=strategy,
            weight_precision=QuantizationPrecision.INT4,
            activation_precision=QuantizationPrecision.INT8,
            calibration_samples=128,
        )
    if strategy == QuantizationStrategy.MIXED_PRECISION:
        return QuantizationProfile(
            name=profile_name,
            strategy=strategy,
            weight_precision=QuantizationPrecision.INT4,
            compute_precision=QuantizationPrecision.BF16,
        )
    if strategy == QuantizationStrategy.HYBRID_FP8:
        return QuantizationProfile(
            name=profile_name,
            strategy=strategy,
            weight_precision=QuantizationPrecision.INT4,
            compute_precision=QuantizationPrecision.FP8_E4M3,
        )
    if strategy == QuantizationStrategy.EXTERNAL_ADAPTIVE:
        return QuantizationProfile(
            name=profile_name,
            strategy=strategy,
            external_quantizer=ExternalQuantizerReference(name="external_adaptive"),
        )
    return QuantizationProfile(name=profile_name, strategy=strategy)


def _convert_manifest_for_benchmark(
    services: LewLMServices,
    manifest: ModelManifest,
    request: ConversionJobRequest,
) -> tuple[dict[str, Any], ModelManifest | None]:
    job = services.conversion_service.submit(request)
    if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
        job = _wait_for_job(services, job.job_id)
    payload = dict(job.payload)
    conversion_payload = {
        "needed": True,
        "status": job.status.value,
        "request": request.model_dump(mode="json"),
        "profile_label": _conversion_profile_label_from_payload(payload),
        "cache_hit": bool(payload.get("cache_hit", False)),
        "duration_seconds": _coerce_float(payload.get("duration_seconds")),
        "result_path": payload.get("result_path"),
        "job_id": job.job_id,
        "logs_tail": list(payload.get("logs", []))[-8:],
        "compatibility": payload.get("compatibility"),
    }
    if job.status != JobStatus.COMPLETED:
        conversion_payload["reason"] = payload.get("error", "Conversion did not complete successfully.")
        return conversion_payload, None
    result_path = payload.get("result_path")
    if not isinstance(result_path, str):
        conversion_payload["status"] = "failed"
        conversion_payload["reason"] = "Conversion completed without a discoverable result path."
        return conversion_payload, None
    discovered = discover_models((Path(result_path).expanduser().resolve(strict=False),))
    if not discovered:
        conversion_payload["status"] = "failed"
        conversion_payload["reason"] = "Converted artifact was not discoverable as a runnable model."
        return conversion_payload, None
    benchmark_manifest = _select_benchmark_manifest(source_manifest=manifest, discovered=discovered)
    conversion_payload["artifacts"] = [item.model_dump(mode="json") for item in discovered]
    conversion_payload["registered_transiently"] = _register_benchmark_manifest(services, benchmark_manifest)
    return conversion_payload, benchmark_manifest


def _select_benchmark_manifest(*, source_manifest: ModelManifest, discovered: list[ModelManifest]) -> ModelManifest:
    if len(discovered) == 1:
        return discovered[0]
    if ModelModality.VISION in source_manifest.modality or ModelModality.MULTIMODAL in source_manifest.modality:
        for manifest in discovered:
            if manifest.artifact_role == ModelArtifactRole.MULTIMODAL_RUNNABLE:
                return manifest
    for manifest in discovered:
        if manifest.artifact_role in {ModelArtifactRole.TEXT_RUNNABLE, ModelArtifactRole.STANDALONE}:
            return manifest
    return discovered[0]


def _register_benchmark_manifest(services: LewLMServices, manifest: ModelManifest) -> bool:
    current_manifests = services.model_registry.list_manifests()
    if any(existing.source_path == manifest.source_path for existing in current_manifests):
        return False
    services.metadata_store.replace_model_manifests(
        [*current_manifests, manifest],
        stale_source_paths=(),
    )
    return True


def _cleanup_transient_benchmark_manifests(
    services: LewLMServices,
    benchmark_targets: list[dict[str, ModelManifest | dict[str, Any] | None]],
) -> None:
    transient_source_paths = {
        str(benchmark_manifest.source_path)
        for target in benchmark_targets
        if bool(target["conversion"].get("registered_transiently"))
        and isinstance(benchmark_manifest := target["benchmark_manifest"], ModelManifest)
    }
    if not transient_source_paths:
        return
    retained_manifests = [
        manifest
        for manifest in services.model_registry.list_manifests()
        if manifest.source_path not in transient_source_paths
    ]
    services.metadata_store.replace_model_manifests(retained_manifests, stale_source_paths=tuple(transient_source_paths))


def _benchmark_target_model_ids(args: argparse.Namespace, services: LewLMServices) -> list[str]:
    if args.model is not None:
        services.model_router.route_chat(
            args.model,
            messages=[GenerateMessage(role="user", content=args.prompt)],
            max_tokens=_CLI_CHAT_BENCHMARK_MAX_TOKENS,
        )
        return [args.model]
    if args.all:
        candidate_model_ids: list[str] = []
        for manifest in services.model_registry.list_manifests():
            report = services.model_router.model_capability_report(manifest.model_id)
            if any(item.capability == CapabilityName.CHAT and item.supported for item in report.capabilities):
                candidate_model_ids.append(manifest.model_id)
        if not candidate_model_ids:
            raise ConfigurationError("No runnable chat models are available for benchmarking.")
        return candidate_model_ids
    manifest, _, _ = services.model_router.route_chat(
        None,
        messages=[GenerateMessage(role="user", content=args.prompt)],
        max_tokens=_CLI_CHAT_BENCHMARK_MAX_TOKENS,
    )
    return [manifest.model_id]


def _run_direct_benchmark_once(
    *,
    manifest: Any,
    prompt: str,
    warmup_run_count: int,
) -> dict[str, Any]:
    try:
        result = benchmark_direct_chat_manifest(
            manifest,
            prompt=prompt,
            max_tokens=_CLI_CHAT_BENCHMARK_MAX_TOKENS,
            warmup_run_count=warmup_run_count,
        )
    except ModuleNotFoundError as exc:
        return {
            "status": "unsupported",
            "reason": f"Missing optional direct-backend dependency: {exc.name}.",
            "error_type": type(exc).__name__,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "error_type": type(exc).__name__,
        }
    payload = {"status": "completed", **result}
    payload["phase_breakdown"] = _normalize_benchmark_phase_breakdown(payload)
    payload["optimization_attribution"] = _direct_benchmark_optimization_attribution(manifest=manifest, payload=payload)
    payload["ttft_seconds"] = _coerce_float(payload["phase_breakdown"].get("ttft_seconds"))
    payload["decode_tokens_per_second"] = _coerce_float(
        payload["phase_breakdown"].get("steady_state_decode_tokens_per_second"),
    ) or _decode_tokens_per_second(payload)
    return payload


def _run_managed_benchmark_once(
    *,
    services: LewLMServices,
    model_id: str,
    prompt: str,
    warmup_run_count: int,
    apply_serving_profile: bool,
) -> dict[str, Any]:
    try:
        result = asyncio.run(
            services.telemetry_service.benchmark_lightweight(
                model_id=model_id,
                prompt=prompt,
                capability=CapabilityName.CHAT.value,
                warmup_run_count=warmup_run_count,
                apply_serving_profile=apply_serving_profile,
            ),
        )
    except Exception as exc:
        return {
            "status": "failed",
            "model_id": model_id,
            "reason": str(exc),
            "error_type": type(exc).__name__,
        }
    payload = {"status": "completed", **result.model_dump(mode="json")}
    payload["phase_breakdown"] = _normalize_benchmark_phase_breakdown(payload)
    payload["ttft_seconds"] = _coerce_float(payload["phase_breakdown"].get("ttft_seconds"))
    payload["decode_tokens_per_second"] = _coerce_float(
        payload["phase_breakdown"].get("steady_state_decode_tokens_per_second"),
    ) or _decode_tokens_per_second(payload)
    return payload


def _invoke_managed_benchmark_once(
    *,
    services: LewLMServices,
    model_id: str,
    prompt: str,
    warmup_run_count: int,
    apply_serving_profile: bool,
) -> dict[str, Any]:
    parameters = inspect.signature(_run_managed_benchmark_once).parameters
    kwargs: dict[str, Any] = {
        "services": services,
        "model_id": model_id,
        "prompt": prompt,
        "warmup_run_count": warmup_run_count,
    }
    if "apply_serving_profile" in parameters:
        kwargs["apply_serving_profile"] = apply_serving_profile
    return _run_managed_benchmark_once(**kwargs)


def _emit_benchmark_progress(message: str) -> None:
    print(message, flush=True)


def _conversion_profile_label_from_request(request: ConversionJobRequest) -> str:
    if request.quantization_profile is not None:
        return quantization_profile_label(request.quantization_profile) or request.quantization_profile.name
    return request.policy.value


def _conversion_profile_label_from_payload(payload: dict[str, Any]) -> str | None:
    compatibility = payload.get("compatibility")
    if isinstance(compatibility, dict):
        quantization_mode = compatibility.get("quantization_mode")
        if isinstance(quantization_mode, str) and quantization_mode:
            return quantization_mode
    request = payload.get("request")
    if isinstance(request, dict):
        policy = request.get("policy")
        if isinstance(policy, str) and policy:
            return policy
    return None


def _benchmark_comparison_controls(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "primary_metric": args.compare_metric,
        "warmup_run_count": args.warmup_runs,
        "notes": [
            "Cold-load, warm-path, TTFT, and steady-state decode phases are recorded separately.",
            "Warmup runs are excluded from the reported comparison metrics.",
            "LewLM only claims scheduler or residency wins when the direct comparison agrees with the matching managed proof scenario.",
        ],
    }


def _normalize_benchmark_phase_breakdown(payload: dict[str, Any]) -> dict[str, Any]:
    phase_breakdown = payload.get("phase_breakdown")
    if isinstance(phase_breakdown, dict):
        normalized = dict(phase_breakdown)
    else:
        normalized = {}
    fallback_values = {
        "cold_load_seconds": _coerce_float(payload.get("load_seconds")),
        "cold_generate_seconds": _coerce_float(payload.get("generate_seconds")),
        "cold_total_seconds": _coerce_float(payload.get("total_seconds")),
        "warm_load_seconds": _coerce_float(payload.get("warm_load_seconds")),
        "warm_generate_seconds": _coerce_float(payload.get("warm_generate_seconds")),
        "warm_total_seconds": _coerce_float(payload.get("warm_total_seconds", payload.get("total_seconds"))),
        "ttft_seconds": _coerce_float(payload.get("ttft_seconds")),
        "steady_state_decode_seconds": _coerce_float(payload.get("steady_state_decode_seconds")),
        "steady_state_decode_tokens_per_second": _coerce_float(
            payload.get("steady_state_decode_tokens_per_second", payload.get("decode_tokens_per_second")),
        ),
    }
    for key, value in fallback_values.items():
        normalized.setdefault(key, value)
    warmup_run_count = payload.get("warmup_run_count")
    if isinstance(warmup_run_count, bool):
        normalized.setdefault("warmup_run_count", int(warmup_run_count))
    elif isinstance(warmup_run_count, int):
        normalized.setdefault("warmup_run_count", warmup_run_count)
    elif isinstance(warmup_run_count, float):
        normalized.setdefault("warmup_run_count", int(warmup_run_count))
    else:
        normalized.setdefault("warmup_run_count", None)
    return normalized


def _merge_benchmark_scenarios(
    *,
    existing: Any,
    additional: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in existing if isinstance(existing, list) else []:
        if not isinstance(item, dict):
            continue
        scenario_name = item.get("scenario")
        if isinstance(scenario_name, str):
            if scenario_name in seen:
                continue
            seen.add(scenario_name)
        merged.append(dict(item))
    for item in additional:
        scenario_name = item.get("scenario")
        if isinstance(scenario_name, str):
            if scenario_name in seen:
                continue
            seen.add(scenario_name)
        merged.append(dict(item))
    return merged


def _direct_benchmark_optimization_attribution(*, manifest: Any, payload: dict[str, Any]) -> dict[str, Any]:
    quantization_label = quantization_profile_label(getattr(manifest, "quantization_profile", None)) or getattr(
        manifest,
        "quantization",
        None,
    )
    return {
        "cache_reuse": {
            "status": "not_applicable",
            "detail": "Direct backend baselines bypass LewLM-managed cache orchestration and only report raw backend timings.",
            "metrics": {},
        },
        "batching": {
            "status": "not_applicable",
            "detail": "Direct backend baselines do not run through LewLM's continuous-batching scheduler.",
            "metrics": {},
        },
        "speculation": {
            "status": "not_applicable",
            "detail": "Direct backend baselines are measured without LewLM-managed speculation selection.",
            "metrics": {},
        },
        "kernel_acceleration": {
            "status": "unknown",
            "detail": (
                "The direct backend baseline does not expose LewLM-level kernel attribution, so only elapsed phase timings "
                "are recorded here."
            ),
            "metrics": {
                "runtime": payload.get("runtime"),
                "ttft_seconds": _coerce_float(_normalize_benchmark_phase_breakdown(payload).get("ttft_seconds")),
            },
        },
        "quantization_profile": {
            "status": "active" if quantization_label is not None else "inactive",
            "detail": (
                f"Benchmarked the artifact quantization profile `{quantization_label}`."
                if quantization_label is not None
                else "No explicit artifact quantization profile was recorded for the direct baseline."
            ),
            "metrics": {
                "artifact_quantization": getattr(manifest, "quantization", None),
                "artifact_quantization_profile": quantization_label,
            },
        },
        "serving_profile_defaults": {
            "status": "not_applicable",
            "detail": "Serving-profile defaults are LewLM orchestration settings and do not apply to direct backend baselines.",
            "metrics": {},
        },
    }


def _annotate_profile_metrics(results: list[dict[str, Any]]) -> None:
    reference_outputs = _reference_quality_outputs(results)
    for item in results:
        reference = reference_outputs.get(str(item["model_id"]))
        optimized = item.get("optimized", {})
        item["profile_metrics"] = {
            "profile_label": item.get("conversion", {}).get("profile_label") or "runnable",
            "model_size_bytes": _benchmark_result_size_bytes(item),
            "cold_load_seconds": _benchmark_metric_value(optimized, "cold_load_seconds"),
            "warm_total_seconds": _benchmark_metric_value(optimized, "warm_total_seconds"),
            "ttft_seconds": _coerce_float(optimized.get("ttft_seconds")),
            "decode_tokens_per_second": _coerce_float(optimized.get("decode_tokens_per_second")),
            "quality_proxy": _quality_proxy(
                optimized.get("output_text"),
                reference_output=reference.get("output_text") if reference is not None else None,
                reference_profile=reference.get("profile_label") if reference is not None else None,
            ),
            "serving_profile_compatibility": _serving_profile_compatibility(optimized),
        }


def _reference_quality_outputs(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(str(item["model_id"]), []).append(item)
    references: dict[str, dict[str, Any]] = {}
    for model_id, items in grouped.items():
        preferred = next(
            (
                item
                for item in items
                if item.get("optimized", {}).get("status") == "completed"
                and isinstance(item.get("conversion", {}).get("request"), dict)
                and item.get("conversion", {}).get("request", {}).get("policy") == ConversionPolicy.MAX_QUALITY.value
            ),
            None,
        )
        if preferred is None:
            preferred = next((item for item in items if item.get("optimized", {}).get("status") == "completed"), None)
        if preferred is None:
            continue
        references[model_id] = {
            "output_text": preferred.get("optimized", {}).get("output_text"),
            "profile_label": preferred.get("conversion", {}).get("profile_label") or "runnable",
        }
    return references


def _benchmark_result_size_bytes(item: dict[str, Any]) -> int | None:
    conversion = item.get("conversion", {})
    result_path = conversion.get("result_path")
    if isinstance(result_path, str) and result_path:
        return _path_size_bytes(Path(result_path).expanduser().resolve(strict=False))
    source_path = item.get("source_path")
    if isinstance(source_path, str) and source_path:
        return _path_size_bytes(Path(source_path).expanduser().resolve(strict=False))
    return None


def _path_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _format_profile_size_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    units = ("KiB", "MiB", "GiB", "TiB")
    value = float(size_bytes)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if value < 1024.0 or candidate == units[-1]:
            break
        value /= 1024.0
    return f"{value:.1f} {unit}"


def _serving_profile_compatibility(payload: dict[str, Any]) -> dict[str, Any]:
    serving_profile = payload.get("serving_profile")
    if not isinstance(serving_profile, dict):
        return {
            "status": "unknown",
            "classification": "unknown",
            "profile_id": None,
            "accepted_setting_count": 0,
            "rejected_setting_count": 0,
            "effective_settings": None,
            "reason": "No serving-profile application payload was recorded for this benchmarked run.",
        }
    accepted_settings = serving_profile.get("accepted_settings")
    rejected_settings = serving_profile.get("rejected_settings")
    accepted_setting_count = len(accepted_settings) if isinstance(accepted_settings, dict) else 0
    rejected_setting_count = len(rejected_settings) if isinstance(rejected_settings, dict) else 0
    status = str(serving_profile.get("status", "unknown"))
    classification = "not_profiled"
    if status == "selected" and rejected_setting_count == 0:
        classification = "fully_supported"
    elif status == "selected":
        classification = "partially_supported"
    elif status == "runtime_mismatch":
        classification = "incompatible"
    elif status in {"disabled", "not_found", "unavailable"}:
        classification = "not_profiled"
    return {
        "status": status,
        "classification": classification,
        "profile_id": serving_profile.get("profile_id"),
        "accepted_setting_count": accepted_setting_count,
        "rejected_setting_count": rejected_setting_count,
        "effective_settings": serving_profile.get("effective_settings"),
        "reason": serving_profile.get("reason"),
    }


def _quality_proxy(output_text: Any, *, reference_output: Any, reference_profile: str | None) -> dict[str, Any]:
    if not isinstance(output_text, str) or not isinstance(reference_output, str):
        return {
            "reference_profile": reference_profile,
            "exact_match": None,
            "token_overlap_ratio": None,
        }
    output_tokens = output_text.split()
    reference_tokens = reference_output.split()
    if not reference_tokens:
        overlap_ratio = 1.0 if not output_tokens else 0.0
    else:
        overlap = sum(1 for index, token in enumerate(output_tokens) if index < len(reference_tokens) and token == reference_tokens[index])
        overlap_ratio = round(overlap / len(reference_tokens), 4)
    return {
        "reference_profile": reference_profile,
        "exact_match": output_text == reference_output,
        "token_overlap_ratio": overlap_ratio,
    }


def _average_benchmark_value(values: Sequence[float | None]) -> float | None:
    realized = [value for value in values if value is not None]
    return round(sum(realized) / len(realized), 4) if realized else None


def _quality_proxy_summary(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    quality_items = [
        item.get("quality_proxy")
        for item in items
        if isinstance(item.get("quality_proxy"), dict)
    ]
    reference_profile = next(
        (
            str(quality_item.get("reference_profile"))
            for quality_item in quality_items
            if isinstance(quality_item.get("reference_profile"), str)
        ),
        None,
    )
    exact_values = [
        bool(value)
        for quality_item in quality_items
        if isinstance((value := quality_item.get("exact_match")), bool)
    ]
    overlap_ratio = _average_benchmark_value(
        [
            _coerce_float(quality_item.get("token_overlap_ratio"))
            for quality_item in quality_items
        ],
    )
    return {
        "reference_profile": reference_profile,
        "exact_match": all(exact_values) if exact_values else None,
        "token_overlap_ratio": overlap_ratio,
    }


def _serving_profile_compatibility_summary(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    compatibility_items = [
        item.get("serving_profile_compatibility")
        for item in items
        if isinstance(item.get("serving_profile_compatibility"), dict)
    ]
    if not compatibility_items:
        return {
            "status": "unknown",
            "classification": "unknown",
            "profile_id": None,
            "accepted_setting_count": 0,
            "rejected_setting_count": 0,
            "effective_settings": None,
            "reason": "No serving-profile compatibility data was recorded for this profile.",
        }
    classification_rank = {
        "fully_supported": 0,
        "partially_supported": 1,
        "not_profiled": 2,
        "incompatible": 3,
        "unknown": 4,
    }
    selected_item = max(
        compatibility_items,
        key=lambda item: classification_rank.get(str(item.get("classification", "unknown")), 4),
    )
    return {
        "status": selected_item.get("status"),
        "classification": selected_item.get("classification"),
        "profile_id": selected_item.get("profile_id"),
        "accepted_setting_count": int(
            round(
                _average_benchmark_value(
                    [
                        _coerce_float(item.get("accepted_setting_count"))
                        for item in compatibility_items
                    ],
                )
                or 0.0,
            ),
        ),
        "rejected_setting_count": int(
            round(
                _average_benchmark_value(
                    [
                        _coerce_float(item.get("rejected_setting_count"))
                        for item in compatibility_items
                    ],
                )
                or 0.0,
            ),
        ),
        "effective_settings": selected_item.get("effective_settings"),
        "reason": selected_item.get("reason"),
    }


def _profile_summary_status(items: Sequence[dict[str, Any]]) -> tuple[str, str | None]:
    reasons = [
        str(reason)
        for item in items
        for reason in (
            item.get("optimized", {}).get("reason"),
            item.get("conversion", {}).get("reason"),
        )
        if isinstance(reason, str) and reason
    ]
    optimized_statuses = {str(item.get("optimized", {}).get("status", "")) for item in items}
    conversion_statuses = {str(item.get("conversion", {}).get("status", "")) for item in items}
    if "completed" in optimized_statuses:
        return "completed", reasons[0] if reasons else None
    if "failed" in optimized_statuses or "failed" in conversion_statuses:
        return "failed", reasons[0] if reasons else None
    if "unsupported" in optimized_statuses:
        return "unsupported", reasons[0] if reasons else None
    return "skipped", reasons[0] if reasons else None


def _profile_summaries(items: Sequence[dict[str, Any]], *, primary_metric: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        profile_key = str(item.get("benchmark_model_id") or item.get("model_id"))
        grouped.setdefault(profile_key, []).append(item)
    summaries: list[dict[str, Any]] = []
    for profile_key, profile_items in sorted(grouped.items()):
        sample = profile_items[0]
        metric_items = [
            metric_payload
            for item in profile_items
            if isinstance((metric_payload := item.get("profile_metrics")), dict)
        ]
        status, reason = _profile_summary_status(profile_items)
        metrics = {
            "model_size_bytes": next(
                (
                    value
                    for item in metric_items
                    if isinstance((value := item.get("model_size_bytes")), int)
                ),
                None,
            ),
            "cold_load_seconds": _average_benchmark_value(
                [_coerce_float(item.get("cold_load_seconds")) for item in metric_items],
            ),
            "warm_total_seconds": _average_benchmark_value(
                [_coerce_float(item.get("warm_total_seconds")) for item in metric_items],
            ),
            "ttft_seconds": _average_benchmark_value(
                [_coerce_float(item.get("ttft_seconds")) for item in metric_items],
            ),
            "decode_tokens_per_second": _average_benchmark_value(
                [_coerce_float(item.get("decode_tokens_per_second")) for item in metric_items],
            ),
            "quality_proxy": _quality_proxy_summary(metric_items),
            "serving_profile_compatibility": _serving_profile_compatibility_summary(metric_items),
        }
        summaries.append(
            {
                "benchmark_model_id": sample.get("benchmark_model_id"),
                "profile_key": profile_key,
                "profile_label": sample.get("conversion", {}).get("profile_label")
                or sample.get("profile_metrics", {}).get("profile_label")
                or "runnable",
                "run_count": len(profile_items),
                "status": status,
                "reason": reason,
                "conversion": dict(sample.get("conversion", _no_conversion_payload())),
                "comparison_metric": primary_metric,
                "comparison_metric_value": _average_benchmark_value(
                    [
                        _benchmark_metric_value(item.get("optimized", {}), primary_metric)
                        for item in profile_items
                        if isinstance(item.get("optimized"), dict)
                    ],
                ),
                "metrics": metrics,
            },
        )
    return summaries


def _profile_metric_sort_value(metric_name: str, value: float | None) -> float:
    if value is None:
        return float("inf")
    if _benchmark_metric_direction(metric_name) == "higher_better":
        return -value
    return value


def _profile_recommendation(profile_summaries: Sequence[dict[str, Any]], *, primary_metric: str) -> dict[str, Any]:
    completed_candidates = [
        item
        for item in profile_summaries
        if item.get("status") == "completed"
    ]
    if not completed_candidates:
        return {
            "status": "unavailable",
            "selection_objective": "serving_latency_first_with_quality_proxy",
            "profile_label": None,
            "benchmark_model_id": None,
            "reason": "No conversion profile completed the conversion-to-serving benchmark loop successfully.",
            "supporting_metrics": {},
            "tradeoff_notes": [],
        }
    compatibility_rank = {
        "fully_supported": 0,
        "partially_supported": 1,
        "not_profiled": 2,
        "incompatible": 3,
        "unknown": 4,
    }
    ranked_candidates = sorted(
        completed_candidates,
        key=lambda item: (
            compatibility_rank.get(
                str(item.get("metrics", {}).get("serving_profile_compatibility", {}).get("classification", "unknown")),
                4,
            ),
            _profile_metric_sort_value(primary_metric, _coerce_float(item.get("comparison_metric_value"))),
            _profile_metric_sort_value(
                "warm_total_seconds",
                _coerce_float(item.get("metrics", {}).get("warm_total_seconds")),
            ),
            _profile_metric_sort_value(
                "ttft_seconds",
                _coerce_float(item.get("metrics", {}).get("ttft_seconds")),
            ),
            _profile_metric_sort_value(
                "steady_state_decode_tokens_per_second",
                _coerce_float(item.get("metrics", {}).get("decode_tokens_per_second")),
            ),
            _coerce_float(item.get("metrics", {}).get("model_size_bytes")) or float("inf"),
            0
            if item.get("metrics", {}).get("quality_proxy", {}).get("exact_match") is True
            else 1,
            -(
                _coerce_float(item.get("metrics", {}).get("quality_proxy", {}).get("token_overlap_ratio"))
                or 0.0
            ),
            str(item.get("profile_label", "")),
        ),
    )
    winner = ranked_candidates[0]
    winner_metrics = winner.get("metrics", {})
    quality_proxy = winner_metrics.get("quality_proxy", {})
    serving_compatibility = winner_metrics.get("serving_profile_compatibility", {})
    tradeoff_notes: list[str] = []
    reference_profile = quality_proxy.get("reference_profile")
    if quality_proxy.get("exact_match") is True and isinstance(reference_profile, str):
        tradeoff_notes.append(f"Output matched the `{reference_profile}` quality reference exactly.")
    elif quality_proxy.get("exact_match") is False and isinstance(reference_profile, str):
        overlap_ratio = _coerce_float(quality_proxy.get("token_overlap_ratio"))
        tradeoff_notes.append(
            f"Output diverged from the `{reference_profile}` quality reference"
            + (
                f" (token overlap {overlap_ratio:.2f})."
                if overlap_ratio is not None
                else "."
            ),
        )
    compatibility_classification = str(serving_compatibility.get("classification", "unknown"))
    if compatibility_classification == "fully_supported":
        tradeoff_notes.append("Serving-profile application stayed fully compatible for the selected runtime.")
    elif compatibility_classification == "partially_supported":
        tradeoff_notes.append(
            "Serving-profile application stayed only partially compatible and rejected at least one runtime-unsupported override."
        )
    elif compatibility_classification == "incompatible":
        tradeoff_notes.append("Serving-profile application was incompatible with the active runtime selection.")
    else:
        tradeoff_notes.append("No persisted serving-profile overrides were applied for this converted artifact.")
    size_bytes = winner_metrics.get("model_size_bytes")
    if isinstance(size_bytes, int):
        tradeoff_notes.append(f"Converted bundle size: {_format_profile_size_bytes(size_bytes)}.")
    reason = (
        f"Selected `{winner.get('profile_label')}` because it produced the strongest "
        f"{_benchmark_metric_label(primary_metric)} for this host/workload after conversion."
    )
    return {
        "status": "recommended",
        "selection_objective": "serving_latency_first_with_quality_proxy",
        "profile_label": winner.get("profile_label"),
        "benchmark_model_id": winner.get("benchmark_model_id"),
        "reason": reason,
        "supporting_metrics": {
            "comparison_metric": primary_metric,
            "comparison_metric_value": winner.get("comparison_metric_value"),
            "cold_load_seconds": winner_metrics.get("cold_load_seconds"),
            "warm_total_seconds": winner_metrics.get("warm_total_seconds"),
            "ttft_seconds": winner_metrics.get("ttft_seconds"),
            "decode_tokens_per_second": winner_metrics.get("decode_tokens_per_second"),
            "model_size_bytes": winner_metrics.get("model_size_bytes"),
            "quality_proxy": quality_proxy,
            "serving_profile_compatibility": serving_compatibility,
        },
        "tradeoff_notes": tradeoff_notes,
    }


def _decode_tokens_per_second(payload: dict[str, Any]) -> float | None:
    explicit_value = _coerce_float(payload.get("completion_tokens_per_second"))
    if explicit_value is not None:
        return explicit_value
    usage = payload.get("usage")
    generate_seconds = _coerce_float(payload.get("generate_seconds"))
    if isinstance(usage, dict) and generate_seconds not in (None, 0.0):
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, (int, float)):
            return round(float(completion_tokens) / generate_seconds, 4)
    return None


def _benchmark_metric_label(metric_name: str) -> str:
    return {
        "cold_load_seconds": "cold load",
        "cold_total_seconds": "cold total",
        "warm_total_seconds": "warm total",
        "ttft_seconds": "TTFT",
        "steady_state_decode_seconds": "steady decode",
        "steady_state_decode_tokens_per_second": "steady decode rate",
    }.get(metric_name, metric_name)


def _benchmark_metric_unit(metric_name: str) -> str:
    if metric_name == "steady_state_decode_tokens_per_second":
        return "tokens/s"
    return "seconds"


def _benchmark_metric_direction(metric_name: str) -> str:
    if metric_name == "steady_state_decode_tokens_per_second":
        return "higher_better"
    return "lower_better"


def _benchmark_metric_value(payload: dict[str, Any], metric_name: str) -> float | None:
    phase_breakdown = _normalize_benchmark_phase_breakdown(payload)
    return _coerce_float(phase_breakdown.get(metric_name))


def _benchmark_metric_summary(
    metric_name: str,
    direct_payload: dict[str, Any],
    optimized_payload: dict[str, Any],
) -> dict[str, Any]:
    direct_value = _benchmark_metric_value(direct_payload, metric_name)
    optimized_value = _benchmark_metric_value(optimized_payload, metric_name)
    direction = _benchmark_metric_direction(metric_name)
    if (
        direct_payload.get("status") != "completed"
        or optimized_payload.get("status") != "completed"
        or direct_value is None
        or optimized_value is None
    ):
        return {
            "status": "unavailable",
            "metric": metric_name,
            "label": _benchmark_metric_label(metric_name),
            "unit": _benchmark_metric_unit(metric_name),
            "direction": direction,
            "direct": direct_value,
            "optimized": optimized_value,
            "lewlm_advantage": None,
            "lewlm_advantage_percent": None,
            "direct_over_optimized_ratio": None,
            "winner": None,
        }
    lewlm_advantage = (
        round(optimized_value - direct_value, 4)
        if direction == "higher_better"
        else round(direct_value - optimized_value, 4)
    )
    baseline_for_percent = abs(direct_value)
    advantage_percent = (
        round((lewlm_advantage / baseline_for_percent) * 100.0, 2)
        if baseline_for_percent > 0
        else 0.0
    )
    ratio = (
        round(direct_value / optimized_value, 4)
        if direction == "lower_better" and optimized_value != 0
        else round(optimized_value / direct_value, 4)
        if direction == "higher_better" and direct_value != 0
        else None
    )
    winner = "tie"
    if lewlm_advantage > 0:
        winner = "lewlm"
    elif lewlm_advantage < 0:
        winner = "direct"
    return {
        "status": "completed",
        "metric": metric_name,
        "label": _benchmark_metric_label(metric_name),
        "unit": _benchmark_metric_unit(metric_name),
        "direction": direction,
        "direct": direct_value,
        "optimized": optimized_value,
        "lewlm_advantage": lewlm_advantage,
        "lewlm_advantage_percent": advantage_percent,
        "direct_over_optimized_ratio": ratio,
        "winner": winner,
    }


def _benchmark_comparison_summary(
    direct_payload: dict[str, Any],
    optimized_payload: dict[str, Any],
    *,
    primary_metric: str,
) -> dict[str, Any]:
    metric_summaries = {
        metric_name: _benchmark_metric_summary(metric_name, direct_payload, optimized_payload)
        for metric_name in (
            "cold_load_seconds",
            "cold_total_seconds",
            "warm_total_seconds",
            "ttft_seconds",
            "steady_state_decode_seconds",
            "steady_state_decode_tokens_per_second",
        )
    }
    primary_summary = metric_summaries[primary_metric]
    if primary_summary["status"] != "completed":
        return {
            "status": "unavailable",
            "primary_metric": primary_metric,
            "time_saved_seconds": None,
            "time_saved_percent": None,
            "direct_over_optimized_ratio": None,
            "metric_summaries": metric_summaries,
            "evidence": _benchmark_comparison_evidence(
                direct_payload=direct_payload,
                optimized_payload=optimized_payload,
                metric_summaries=metric_summaries,
            ),
        }
    return {
        "status": "completed",
        "primary_metric": primary_metric,
        "time_saved_seconds": (
            primary_summary["lewlm_advantage"] if primary_summary["unit"] == "seconds" else None
        ),
        "time_saved_percent": (
            primary_summary["lewlm_advantage_percent"] if primary_summary["unit"] == "seconds" else None
        ),
        "direct_over_optimized_ratio": primary_summary["direct_over_optimized_ratio"],
        "metric_summaries": metric_summaries,
        "evidence": _benchmark_comparison_evidence(
            direct_payload=direct_payload,
            optimized_payload=optimized_payload,
            metric_summaries=metric_summaries,
        ),
        }


def _external_adapter_metric_summary(
    metric_name: str,
    native_payload: dict[str, Any],
    external_payload: dict[str, Any],
) -> dict[str, Any]:
    native_value = _benchmark_metric_value(native_payload, metric_name)
    external_value = _benchmark_metric_value(external_payload, metric_name)
    direction = _benchmark_metric_direction(metric_name)
    if (
        native_payload.get("status") != "completed"
        or external_payload.get("status") != "completed"
        or native_value is None
        or external_value is None
    ):
        return {
            "status": "unavailable",
            "metric": metric_name,
            "label": _benchmark_metric_label(metric_name),
            "unit": _benchmark_metric_unit(metric_name),
            "direction": direction,
            "native": native_value,
            "external": external_value,
            "external_advantage": None,
            "external_advantage_percent": None,
            "winner": None,
            "native_over_external_ratio": None,
        }
    external_advantage = (
        round(external_value - native_value, 4)
        if direction == "higher_better"
        else round(native_value - external_value, 4)
    )
    baseline_for_percent = abs(native_value)
    external_advantage_percent = (
        round((external_advantage / baseline_for_percent) * 100.0, 2)
        if baseline_for_percent > 0
        else 0.0
    )
    ratio = (
        round(native_value / external_value, 4)
        if direction == "lower_better" and external_value != 0
        else round(external_value / native_value, 4)
        if direction == "higher_better" and native_value != 0
        else None
    )
    winner = "tie"
    if external_advantage > 0:
        winner = "external"
    elif external_advantage < 0:
        winner = "native"
    return {
        "status": "completed",
        "metric": metric_name,
        "label": _benchmark_metric_label(metric_name),
        "unit": _benchmark_metric_unit(metric_name),
        "direction": direction,
        "native": native_value,
        "external": external_value,
        "external_advantage": external_advantage,
        "external_advantage_percent": external_advantage_percent,
        "winner": winner,
        "native_over_external_ratio": ratio,
    }


def _external_adapter_comparison_summary(
    *,
    native_payload: dict[str, Any],
    external_payload: dict[str, Any],
    primary_metric: str,
) -> dict[str, Any]:
    metric_summaries = {
        metric_name: _external_adapter_metric_summary(metric_name, native_payload, external_payload)
        for metric_name in (
            "cold_load_seconds",
            "cold_total_seconds",
            "warm_total_seconds",
            "ttft_seconds",
            "steady_state_decode_seconds",
            "steady_state_decode_tokens_per_second",
        )
    }
    primary_summary = metric_summaries[primary_metric]
    if primary_summary["status"] != "completed":
        return {
            "status": "unavailable",
            "primary_metric": primary_metric,
            "winner": None,
            "selected_metric_value": None,
            "baseline_metric_value": None,
            "metric_summaries": metric_summaries,
        }
    winner = primary_summary["winner"]
    selected_metric_value = (
        primary_summary["external"]
        if winner == "external"
        else primary_summary["native"]
        if winner == "native"
        else None
    )
    baseline_metric_value = (
        primary_summary["native"]
        if winner == "external"
        else primary_summary["external"]
        if winner == "native"
        else None
    )
    return {
        "status": "completed",
        "primary_metric": primary_metric,
        "winner": winner,
        "selected_metric_value": selected_metric_value,
        "baseline_metric_value": baseline_metric_value,
        "native_over_external_ratio": primary_summary["native_over_external_ratio"],
        "metric_summaries": metric_summaries,
    }


def _benchmark_comparison_evidence(
    *,
    direct_payload: dict[str, Any],
    optimized_payload: dict[str, Any],
    metric_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scheduler = _benchmark_scheduler_evidence(
        direct_payload=direct_payload,
        optimized_payload=optimized_payload,
    )
    residency = _benchmark_residency_evidence(
        optimized_payload=optimized_payload,
        metric_summaries=metric_summaries,
    )
    winning_dimensions = [
        name
        for name, payload in (("scheduler", scheduler), ("residency", residency))
        if payload["status"] == "proved"
    ]
    overall_status = "unavailable"
    if winning_dimensions:
        overall_status = "proved"
    elif any(payload["status"] in {"inconclusive", "unsupported"} for payload in (scheduler, residency)):
        overall_status = "inconclusive"
    return {
        "overall_status": overall_status,
        "winning_dimensions": winning_dimensions,
        "scheduler": scheduler,
        "residency": residency,
    }


def _benchmark_scheduler_evidence(
    *,
    direct_payload: dict[str, Any],
    optimized_payload: dict[str, Any],
) -> dict[str, Any]:
    if direct_payload.get("status") != "completed" or optimized_payload.get("status") != "completed":
        return {
            "status": "unavailable",
            "reason": "LewLM scheduler evidence is only available when both the direct and managed benchmark passes complete.",
            "metrics": {},
        }
    scenario = _benchmark_scenario(optimized_payload, "continuous_batching")
    if scenario is None:
        return {
            "status": "unavailable",
            "reason": "The managed benchmark payload did not include a continuous batching proof scenario.",
            "metrics": {},
        }
    scenario_status = str(scenario.get("status", "unavailable"))
    if scenario_status == "unsupported":
        return {
            "status": "unsupported",
            "reason": str(scenario.get("reason", "The selected runtime did not advertise native continuous batching.")),
            "metrics": {},
        }
    if scenario_status != "observed":
        return {
            "status": "unavailable",
            "reason": str(scenario.get("reason", "Continuous batching evidence was not observed for this managed run.")),
            "metrics": {},
        }
    scenario_metrics = scenario.get("metrics", {})
    if not isinstance(scenario_metrics, dict):
        scenario_metrics = {}
    direct_warm_total_seconds = _benchmark_metric_value(direct_payload, "warm_total_seconds")
    throughput_requests_per_second = _coerce_float(scenario_metrics.get("throughput_requests_per_second"))
    direct_equivalent_requests_per_second = (
        round(1.0 / direct_warm_total_seconds, 4)
        if direct_warm_total_seconds not in (None, 0.0)
        else None
    )
    throughput_advantage_requests_per_second = (
        round(throughput_requests_per_second - direct_equivalent_requests_per_second, 4)
        if throughput_requests_per_second is not None and direct_equivalent_requests_per_second is not None
        else None
    )
    throughput_advantage_percent = (
        round((throughput_advantage_requests_per_second / direct_equivalent_requests_per_second) * 100.0, 2)
        if throughput_advantage_requests_per_second is not None
        and direct_equivalent_requests_per_second not in (None, 0.0)
        else None
    )
    single_request_elapsed_seconds = _coerce_float(scenario_metrics.get("single_request_elapsed_seconds"))
    single_request_penalty_seconds = (
        round(single_request_elapsed_seconds - direct_warm_total_seconds, 4)
        if single_request_elapsed_seconds is not None and direct_warm_total_seconds is not None
        else None
    )
    single_request_penalty_percent = (
        round((single_request_penalty_seconds / direct_warm_total_seconds) * 100.0, 2)
        if single_request_penalty_seconds is not None and direct_warm_total_seconds not in (None, 0.0)
        else None
    )
    single_request_penalty_budget_seconds = (
        round(max(direct_warm_total_seconds * 0.25, 0.05), 4)
        if direct_warm_total_seconds is not None
        else None
    )
    native_batch_count_delta = _coerce_int(scenario_metrics.get("native_batch_count_delta"))
    native_batched_request_delta = _coerce_int(scenario_metrics.get("native_batched_request_delta"))
    frontier_batch_count_delta = _coerce_int(scenario_metrics.get("frontier_batch_count_delta"))
    frontier_batched_request_delta = _coerce_int(scenario_metrics.get("frontier_batched_request_delta"))
    batched_request_delta = native_batched_request_delta
    if batched_request_delta is None:
        batched_request_delta = frontier_batched_request_delta or 0
    average_batch_size = _coerce_float(scenario_metrics.get("average_batch_size"))
    average_batch_utilization = _coerce_float(scenario_metrics.get("average_batch_utilization"))
    has_batch_evidence = batched_request_delta > 0 and (
        (average_batch_size is not None and average_batch_size > 1.0)
        or (average_batch_utilization is not None and average_batch_utilization > 0.0)
    )
    bounded_single_request_penalty = (
        single_request_penalty_seconds is None
        or single_request_penalty_budget_seconds is None
        or single_request_penalty_seconds <= single_request_penalty_budget_seconds
    )
    status = "inconclusive"
    reason = (
        "LewLM recorded backend-native batching activity, but the throughput or penalty evidence did not clear "
        "the proof threshold."
    )
    if (
        throughput_advantage_requests_per_second is not None
        and throughput_advantage_requests_per_second > 0
        and has_batch_evidence
        and bounded_single_request_penalty
    ):
        status = "proved"
        reason = (
            "LewLM's backend-native batched execution path delivered higher throughput than the direct warm-path "
            "baseline with bounded single-request cost."
        )
    metrics = {
        "concurrency": _coerce_int(scenario_metrics.get("concurrency")),
        "throughput_requests_per_second": throughput_requests_per_second,
        "direct_equivalent_requests_per_second": direct_equivalent_requests_per_second,
        "throughput_advantage_requests_per_second": throughput_advantage_requests_per_second,
        "throughput_advantage_percent": throughput_advantage_percent,
        "native_batch_count_delta": native_batch_count_delta,
        "native_batched_request_delta": native_batched_request_delta,
        "frontier_batch_count_delta": frontier_batch_count_delta,
        "frontier_batched_request_delta": frontier_batched_request_delta,
        "average_batch_size": average_batch_size,
        "average_batch_utilization": average_batch_utilization,
        "average_queue_delay_seconds": _coerce_float(scenario_metrics.get("average_queue_delay_seconds")),
        "single_request_elapsed_seconds": single_request_elapsed_seconds,
        "direct_warm_total_seconds": direct_warm_total_seconds,
        "single_request_penalty_seconds": single_request_penalty_seconds,
        "single_request_penalty_percent": single_request_penalty_percent,
        "single_request_penalty_budget_seconds": single_request_penalty_budget_seconds,
        "bounded_single_request_penalty": bounded_single_request_penalty,
    }
    return {
        "status": status,
        "reason": reason,
        "metrics": {key: value for key, value in metrics.items() if value is not None},
    }


def _benchmark_residency_evidence(
    *,
    optimized_payload: dict[str, Any],
    metric_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if optimized_payload.get("status") != "completed":
        return {
            "status": "unavailable",
            "reason": "LewLM residency evidence is only available when the managed benchmark pass completes.",
            "metrics": {},
        }
    scenario = _benchmark_scenario(optimized_payload, "warm_chat_cache")
    if scenario is None:
        return {
            "status": "unavailable",
            "reason": "The managed benchmark payload did not include a warm-cache proof scenario.",
            "metrics": {},
        }
    scenario_status = str(scenario.get("status", "unavailable"))
    if scenario_status != "observed":
        return {
            "status": "unavailable" if scenario_status != "unsupported" else "unsupported",
            "reason": str(scenario.get("reason", "Warm-cache evidence was not observed for this managed run.")),
            "metrics": {},
        }
    scenario_metrics = scenario.get("metrics", {})
    if not isinstance(scenario_metrics, dict):
        scenario_metrics = {}
    ttft_summary = metric_summaries.get("ttft_seconds", {})
    warm_total_summary = metric_summaries.get("warm_total_seconds", {})
    winning_metric = next(
        (
            metric_name
            for metric_name, summary in (
                ("ttft_seconds", ttft_summary),
                ("warm_total_seconds", warm_total_summary),
            )
            if summary.get("status") == "completed" and summary.get("winner") == "lewlm"
        ),
        None,
    )
    average_warm_over_cold_ttft_ratio = _coerce_float(scenario_metrics.get("average_warm_over_cold_ttft_ratio"))
    total_cache_restores = _coerce_int(scenario_metrics.get("total_cache_restores")) or 0
    total_persistent_cache_hits = _coerce_int(scenario_metrics.get("total_persistent_cache_hits")) or 0
    total_warm_cached_tokens = _coerce_int(scenario_metrics.get("total_warm_cached_tokens")) or 0
    total_warm_saved_prefill_tokens = _coerce_int(scenario_metrics.get("total_warm_saved_prefill_tokens")) or 0
    cache_activity_observed = any(
        metric > 0
        for metric in (
            total_cache_restores,
            total_persistent_cache_hits,
            total_warm_cached_tokens,
            total_warm_saved_prefill_tokens,
        )
    )
    status = "inconclusive"
    reason = "LewLM recorded warm-cache activity, but the direct comparison did not also show a warm-path latency win."
    if winning_metric is not None and average_warm_over_cold_ttft_ratio is not None and average_warm_over_cold_ttft_ratio < 1.0 and cache_activity_observed:
        status = "proved"
        reason = "LewLM's residency path reduced warm latency with explicit cache reuse and also beat the direct baseline on a warm-path metric."
    metrics = {
        "winning_metric": winning_metric,
        "average_cold_ttft_seconds": _coerce_float(scenario_metrics.get("average_cold_ttft_seconds")),
        "average_warm_ttft_seconds": _coerce_float(scenario_metrics.get("average_warm_ttft_seconds")),
        "average_warm_over_cold_ttft_ratio": average_warm_over_cold_ttft_ratio,
        "average_cold_elapsed_seconds": _coerce_float(scenario_metrics.get("average_cold_elapsed_seconds")),
        "average_warm_elapsed_seconds": _coerce_float(scenario_metrics.get("average_warm_elapsed_seconds")),
        "total_cache_restores": total_cache_restores,
        "total_persistent_cache_hits": total_persistent_cache_hits,
        "total_warm_cached_tokens": total_warm_cached_tokens,
        "total_warm_saved_prefill_tokens": total_warm_saved_prefill_tokens,
    }
    return {
        "status": status,
        "reason": reason,
        "metrics": {key: value for key, value in metrics.items() if value is not None},
    }


def _benchmark_scenario(payload: dict[str, Any], scenario_name: str) -> dict[str, Any] | None:
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return None
    return next(
        (
            scenario
            for scenario in scenarios
            if isinstance(scenario, dict) and scenario.get("scenario") == scenario_name
        ),
        None,
    )


def _aggregate_benchmark_evidence(evidence_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    scheduler = _aggregate_benchmark_evidence_dimension(
        evidence_payloads=evidence_payloads,
        dimension="scheduler",
        numeric_metric_names=(
            "throughput_requests_per_second",
            "direct_equivalent_requests_per_second",
            "throughput_advantage_requests_per_second",
            "throughput_advantage_percent",
            "native_batched_request_delta",
            "frontier_batched_request_delta",
            "average_batch_size",
            "average_batch_utilization",
            "average_queue_delay_seconds",
            "single_request_penalty_seconds",
            "single_request_penalty_percent",
            "single_request_penalty_budget_seconds",
        ),
    )
    residency = _aggregate_benchmark_evidence_dimension(
        evidence_payloads=evidence_payloads,
        dimension="residency",
        numeric_metric_names=(
            "average_cold_ttft_seconds",
            "average_warm_ttft_seconds",
            "average_warm_over_cold_ttft_ratio",
            "average_cold_elapsed_seconds",
            "average_warm_elapsed_seconds",
            "total_cache_restores",
            "total_persistent_cache_hits",
            "total_warm_cached_tokens",
            "total_warm_saved_prefill_tokens",
        ),
    )
    winning_dimensions = [
        name
        for name, payload in (("scheduler", scheduler), ("residency", residency))
        if payload["status"] == "proved"
    ]
    overall_status = "unavailable"
    if winning_dimensions:
        overall_status = "proved"
    elif any(payload["status"] in {"inconclusive", "unsupported"} for payload in (scheduler, residency)):
        overall_status = "inconclusive"
    return {
        "overall_status": overall_status,
        "winning_dimensions": winning_dimensions,
        "scheduler": scheduler,
        "residency": residency,
    }


def _aggregate_benchmark_evidence_dimension(
    *,
    evidence_payloads: list[dict[str, Any]],
    dimension: str,
    numeric_metric_names: Sequence[str],
) -> dict[str, Any]:
    items = [
        payload.get(dimension)
        for payload in evidence_payloads
        if isinstance(payload, dict) and isinstance(payload.get(dimension), dict)
    ]
    if not items:
        return {
            "status": "unavailable",
            "proved_run_count": 0,
            "observed_run_count": 0,
            "metrics": {},
        }
    status_counts = {
        status: sum(1 for item in items if item.get("status") == status)
        for status in ("proved", "inconclusive", "unsupported", "unavailable")
    }
    status = "unavailable"
    if status_counts["proved"]:
        status = "proved"
    elif status_counts["inconclusive"] or status_counts["unsupported"]:
        status = "inconclusive"
    metric_averages: dict[str, float] = {}
    for metric_name in numeric_metric_names:
        values = [
            _coerce_float(item.get("metrics", {}).get(metric_name))
            for item in items
            if isinstance(item.get("metrics"), dict)
            and _coerce_float(item.get("metrics", {}).get(metric_name)) is not None
        ]
        if values:
            metric_averages[metric_name] = round(sum(values) / len(values), 4)
    return {
        "status": status,
        "proved_run_count": status_counts["proved"],
        "observed_run_count": status_counts["proved"] + status_counts["inconclusive"],
        "unsupported_run_count": status_counts["unsupported"],
        "metrics": metric_averages,
    }


def _benchmark_model_summaries(results: list[dict[str, Any]], *, primary_metric: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(item["model_id"], []).append(item)
    summaries: list[dict[str, Any]] = []
    for model_id, items in sorted(grouped.items()):
        display_name = str(items[0]["display_name"])
        direct_totals = [
            _benchmark_metric_value(item["direct"], primary_metric)
            for item in items
            if item["comparison"].get("status") == "completed"
        ]
        optimized_totals = [
            _benchmark_metric_value(item["optimized"], primary_metric)
            for item in items
            if item["comparison"].get("status") == "completed"
        ]
        saved_seconds = [
            _coerce_float(item["comparison"].get("time_saved_seconds"))
            for item in items
            if item["comparison"].get("status") == "completed"
        ]
        direct_values = [value for value in direct_totals if value is not None]
        optimized_values = [value for value in optimized_totals if value is not None]
        saved_values = [value for value in saved_seconds if value is not None]
        metric_summaries: dict[str, dict[str, Any]] = {}
        for metric_name in (
            "cold_load_seconds",
            "cold_total_seconds",
            "warm_total_seconds",
            "ttft_seconds",
            "steady_state_decode_seconds",
            "steady_state_decode_tokens_per_second",
        ):
            comparisons = [
                item["comparison"]["metric_summaries"][metric_name]
                for item in items
                if item.get("comparison", {}).get("metric_summaries", {}).get(metric_name) is not None
                and item["comparison"]["metric_summaries"][metric_name].get("status") == "completed"
            ]
            if not comparisons:
                metric_summaries[metric_name] = {
                    "status": "unavailable",
                    "metric": metric_name,
                    "label": _benchmark_metric_label(metric_name),
                    "unit": _benchmark_metric_unit(metric_name),
                    "direction": _benchmark_metric_direction(metric_name),
                    "direct_average": None,
                    "optimized_average": None,
                    "lewlm_advantage_average": None,
                    "lewlm_advantage_percent_average": None,
                }
                continue
            metric_summaries[metric_name] = {
                "status": "completed",
                "metric": metric_name,
                "label": _benchmark_metric_label(metric_name),
                "unit": _benchmark_metric_unit(metric_name),
                "direction": _benchmark_metric_direction(metric_name),
                "direct_average": round(sum(item["direct"] for item in comparisons) / len(comparisons), 4),
                "optimized_average": round(sum(item["optimized"] for item in comparisons) / len(comparisons), 4),
                "lewlm_advantage_average": round(
                    sum(item["lewlm_advantage"] for item in comparisons) / len(comparisons),
                    4,
                ),
                "lewlm_advantage_percent_average": round(
                    sum(item["lewlm_advantage_percent"] for item in comparisons) / len(comparisons),
                    2,
                ),
            }
        ttft_values = [
            _coerce_float(item.get("profile_metrics", {}).get("ttft_seconds"))
            for item in items
            if _coerce_float(item.get("profile_metrics", {}).get("ttft_seconds")) is not None
        ]
        decode_values = [
            _coerce_float(item.get("profile_metrics", {}).get("decode_tokens_per_second"))
            for item in items
            if _coerce_float(item.get("profile_metrics", {}).get("decode_tokens_per_second")) is not None
        ]
        average_direct = round(sum(direct_values) / len(direct_values), 4) if direct_values else None
        average_optimized = round(sum(optimized_values) / len(optimized_values), 4) if optimized_values else None
        average_saved = round(sum(saved_values) / len(saved_values), 4) if saved_values else None
        conversion = dict(items[0].get("conversion", _no_conversion_payload()))
        profile_summaries = _profile_summaries(items, primary_metric=primary_metric)
        profile_recommendation = _profile_recommendation(profile_summaries, primary_metric=primary_metric)
        evidence_summary = _aggregate_benchmark_evidence(
            [
                item.get("comparison", {}).get("evidence", {})
                for item in items
                if isinstance(item.get("comparison"), dict)
            ],
        )
        summaries.append(
            {
                "model_id": model_id,
                "display_name": display_name,
                "run_count": len(items),
                "benchmark_model_id": items[0].get("benchmark_model_id"),
                "direct_runtime": next(
                    (str(item["direct"].get("runtime")) for item in items if item["direct"].get("status") == "completed"),
                    None,
                ),
                "optimized_runtime": next(
                    (str(item["optimized"].get("runtime")) for item in items if item["optimized"].get("status") == "completed"),
                    None,
                ),
                "average_direct_total_seconds": average_direct,
                "average_optimized_total_seconds": average_optimized,
                "average_time_saved_seconds": average_saved,
                "average_time_saved_percent": (
                    round((average_saved / average_direct) * 100.0, 2)
                    if average_saved is not None and average_direct not in (None, 0.0)
                    else None
                ),
                "primary_metric": primary_metric,
                "metric_summaries": metric_summaries,
                "average_ttft_seconds": round(sum(ttft_values) / len(ttft_values), 4) if ttft_values else None,
                "average_decode_tokens_per_second": (
                    round(sum(decode_values) / len(decode_values), 4) if decode_values else None
                ),
                "comparison_available": bool(saved_values),
                "conversion": conversion,
                "evidence_summary": evidence_summary,
                "profile_summaries": profile_summaries,
                "profile_recommendation": profile_recommendation,
                "profiles": [
                    {
                        "run_index": item["run_index"],
                        "benchmark_model_id": item.get("benchmark_model_id"),
                        "conversion": item.get("conversion", _no_conversion_payload()),
                        "profile_metrics": item.get("profile_metrics", {}),
                    }
                    for item in items
                ],
            },
        )
    return summaries


def _benchmark_suite_summary(
    results: list[dict[str, Any]],
    *,
    primary_metric: str,
    model_summaries: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    direct_values = [
        _benchmark_metric_value(item["direct"], primary_metric)
        for item in results
        if item["comparison"].get("status") == "completed"
    ]
    optimized_values = [
        _benchmark_metric_value(item["optimized"], primary_metric)
        for item in results
        if item["comparison"].get("status") == "completed"
    ]
    saved_values = [
        _coerce_float(item["comparison"].get("time_saved_seconds"))
        for item in results
        if item["comparison"].get("status") == "completed"
    ]
    direct_totals = [value for value in direct_values if value is not None]
    optimized_totals = [value for value in optimized_values if value is not None]
    saved_totals = [value for value in saved_values if value is not None]
    conversions = [dict(item.get("conversion", _no_conversion_payload())) for item in results]
    converted_items = [item for item in conversions if item.get("needed")]
    conversion_durations = [
        _coerce_float(item.get("duration_seconds"))
        for item in converted_items
        if not item.get("cache_hit", False)
    ]
    realized_conversion_durations = [value for value in conversion_durations if value is not None]
    metric_summaries: dict[str, dict[str, Any]] = {}
    for metric_name in (
        "cold_load_seconds",
        "cold_total_seconds",
        "warm_total_seconds",
        "ttft_seconds",
        "steady_state_decode_seconds",
        "steady_state_decode_tokens_per_second",
    ):
        comparisons = [
            item["comparison"]["metric_summaries"][metric_name]
            for item in results
            if item.get("comparison", {}).get("metric_summaries", {}).get(metric_name) is not None
            and item["comparison"]["metric_summaries"][metric_name].get("status") == "completed"
        ]
        if not comparisons:
            metric_summaries[metric_name] = {
                "status": "unavailable",
                "metric": metric_name,
                "label": _benchmark_metric_label(metric_name),
                "unit": _benchmark_metric_unit(metric_name),
                "direction": _benchmark_metric_direction(metric_name),
                "direct_average": None,
                "optimized_average": None,
                "lewlm_advantage_average": None,
                "lewlm_advantage_percent_average": None,
            }
            continue
        metric_summaries[metric_name] = {
            "status": "completed",
            "metric": metric_name,
            "label": _benchmark_metric_label(metric_name),
            "unit": _benchmark_metric_unit(metric_name),
            "direction": _benchmark_metric_direction(metric_name),
            "direct_average": round(sum(item["direct"] for item in comparisons) / len(comparisons), 4),
            "optimized_average": round(sum(item["optimized"] for item in comparisons) / len(comparisons), 4),
            "lewlm_advantage_average": round(
                sum(item["lewlm_advantage"] for item in comparisons) / len(comparisons),
                4,
            ),
            "lewlm_advantage_percent_average": round(
                sum(item["lewlm_advantage_percent"] for item in comparisons) / len(comparisons),
                2,
            ),
        }
    ttft_values = [
        _coerce_float(item.get("profile_metrics", {}).get("ttft_seconds"))
        for item in results
        if _coerce_float(item.get("profile_metrics", {}).get("ttft_seconds")) is not None
    ]
    decode_values = [
        _coerce_float(item.get("profile_metrics", {}).get("decode_tokens_per_second"))
        for item in results
        if _coerce_float(item.get("profile_metrics", {}).get("decode_tokens_per_second")) is not None
    ]
    evidence_summary = _aggregate_benchmark_evidence(
        [
            item.get("comparison", {}).get("evidence", {})
            for item in results
            if isinstance(item.get("comparison"), dict)
        ],
    )
    recommendations = [
        recommendation
        for summary in list(model_summaries or [])
        if isinstance((recommendation := summary.get("profile_recommendation")), dict)
    ]
    return {
        "completed_run_count": len(saved_totals),
        "failed_run_count": sum(
            1
            for item in results
            if item["direct"].get("status") == "failed" or item["optimized"].get("status") == "failed"
        ),
        "unsupported_run_count": sum(1 for item in results if item["direct"].get("status") == "unsupported"),
        "direct_total_seconds": round(sum(direct_totals), 4) if direct_totals else None,
        "optimized_total_seconds": round(sum(optimized_totals), 4) if optimized_totals else None,
        "time_saved_seconds": round(sum(saved_totals), 4) if saved_totals else None,
        "converted_model_count": len({item.get("result_path") or item.get("job_id") for item in converted_items}),
        "conversion_cache_hit_count": sum(1 for item in converted_items if item.get("cache_hit", False)),
        "conversion_total_seconds": round(sum(realized_conversion_durations), 4) if realized_conversion_durations else None,
        "primary_metric": primary_metric,
        "metric_summaries": metric_summaries,
        "average_ttft_seconds": round(sum(ttft_values) / len(ttft_values), 4) if ttft_values else None,
        "average_decode_tokens_per_second": round(sum(decode_values) / len(decode_values), 4) if decode_values else None,
        "evidence_summary": evidence_summary,
        "profile_recommendations": recommendations,
        "recommended_profile_count": sum(1 for recommendation in recommendations if recommendation.get("status") == "recommended"),
        "time_saved_percent": (
            round((sum(saved_totals) / sum(direct_totals)) * 100.0, 2)
            if saved_totals and direct_totals and sum(direct_totals) > 0
            else None
        ),
    }


def _write_direct_benchmark_artifact(*, settings: LewLMSettings, payload: dict[str, Any]) -> Path:
    settings.prepare_directories()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    artifact_path = settings.benchmarks_dir / f"cli-direct-chat-benchmark-{timestamp}.json"
    artifact_payload = {**payload, "artifact": {"artifact_path": str(artifact_path)}}
    artifact_path.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_path


def _write_external_adapter_benchmark_artifact(*, settings: LewLMSettings, payload: dict[str, Any]) -> Path:
    settings.prepare_directories()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    artifact_path = settings.benchmarks_dir / f"cli-external-adapter-benchmark-{timestamp}.json"
    artifact_payload = {**payload, "artifact": {"artifact_path": str(artifact_path)}}
    artifact_path.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_path


def _print_direct_benchmark_suite(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    comparison_controls = payload.get("comparison_controls", {})
    primary_metric = str(summary.get("primary_metric", comparison_controls.get("primary_metric", "cold_total_seconds")))
    metric_summaries = summary.get("metric_summaries", {})
    primary_summary = metric_summaries.get(primary_metric, {})
    evidence_summary = summary.get("evidence_summary", {})
    print(_style("LewLM benchmark vs direct inference", "1", "36"))
    print(f"prompt: {payload['prompt']}")
    print(f"runs: {payload['benchmark_count']} across {payload['model_count']} model(s)")
    print(
        "comparison basis: "
        f"{_benchmark_metric_label(primary_metric)} (warmups excluded: {comparison_controls.get('warmup_run_count', 0)})",
    )
    if summary["converted_model_count"]:
        conversion_line = f"conversion: {summary['converted_model_count']} model(s)"
        if summary["conversion_total_seconds"] is not None:
            conversion_line += f" in {_format_seconds(summary['conversion_total_seconds'])}"
        if summary["conversion_cache_hit_count"]:
            conversion_line += f" ({summary['conversion_cache_hit_count']} cached)"
        print(conversion_line)
    if primary_summary.get("status") == "completed":
        print(
            f"{primary_summary['label']}: {_format_benchmark_metric_value(primary_summary['direct_average'], primary_metric)}  "
            f"lewlm: {_format_benchmark_metric_value(primary_summary['optimized_average'], primary_metric)}  "
            f"advantage: {_format_benchmark_advantage(primary_summary['lewlm_advantage_average'], primary_metric)} "
            f"({_format_percent(primary_summary['lewlm_advantage_percent_average'])})",
        )
    warm_summary = metric_summaries.get("warm_total_seconds", {})
    if warm_summary.get("status") == "completed" and primary_metric != "warm_total_seconds":
        print(
            f"warm total: {_format_benchmark_metric_value(warm_summary['direct_average'], 'warm_total_seconds')}  "
            f"lewlm: {_format_benchmark_metric_value(warm_summary['optimized_average'], 'warm_total_seconds')}  "
            f"advantage: {_format_benchmark_advantage(warm_summary['lewlm_advantage_average'], 'warm_total_seconds')}",
        )
    ttft_summary = metric_summaries.get("ttft_seconds", {})
    if ttft_summary.get("status") == "completed":
        print(
            f"avg ttft: {_format_benchmark_metric_value(ttft_summary['direct_average'], 'ttft_seconds')}  "
            f"lewlm: {_format_benchmark_metric_value(ttft_summary['optimized_average'], 'ttft_seconds')}  "
            f"advantage: {_format_benchmark_advantage(ttft_summary['lewlm_advantage_average'], 'ttft_seconds')}",
        )
    decode_rate_summary = metric_summaries.get("steady_state_decode_tokens_per_second", {})
    if decode_rate_summary.get("status") == "completed":
        print(
            f"steady decode: {_format_benchmark_metric_value(decode_rate_summary['direct_average'], 'steady_state_decode_tokens_per_second')}  "
            f"lewlm: {_format_benchmark_metric_value(decode_rate_summary['optimized_average'], 'steady_state_decode_tokens_per_second')}  "
            f"advantage: {_format_benchmark_advantage(decode_rate_summary['lewlm_advantage_average'], 'steady_state_decode_tokens_per_second')}",
        )
    if isinstance(evidence_summary, dict):
        scheduler_evidence = evidence_summary.get("scheduler", {})
        residency_evidence = evidence_summary.get("residency", {})
        if isinstance(scheduler_evidence, dict) and scheduler_evidence.get("status") in {"proved", "inconclusive"}:
            print(_benchmark_scheduler_evidence_line(scheduler_evidence))
        if isinstance(residency_evidence, dict) and residency_evidence.get("status") in {"proved", "inconclusive"}:
            print(_benchmark_residency_evidence_line(residency_evidence))
    if payload.get("artifact") is not None:
        print(f"artifact: {payload['artifact']['artifact_path']}")
    print()
    _print_benchmark_table(
        headers=(
            "Model",
            "Convert",
            f"Direct {_benchmark_metric_label(primary_metric)}",
            "LewLM",
            "Saved",
            "Delta",
        ),
        rows=[
            (
                item["display_name"],
                _benchmark_conversion_value(item.get("conversion", _no_conversion_payload())),
                _format_benchmark_metric_value(
                    _coerce_float(item.get("metric_summaries", {}).get(primary_metric, {}).get("direct_average")),
                    primary_metric,
                ),
                _format_benchmark_metric_value(
                    _coerce_float(item.get("metric_summaries", {}).get(primary_metric, {}).get("optimized_average")),
                    primary_metric,
                ),
                _format_benchmark_advantage(
                    _coerce_float(item.get("metric_summaries", {}).get(primary_metric, {}).get("lewlm_advantage_average")),
                    primary_metric,
                ),
                _format_percent(
                    _coerce_float(item.get("metric_summaries", {}).get(primary_metric, {}).get("lewlm_advantage_percent_average")),
                ),
            )
            for item in payload["models"]
        ],
        alignments=("left", "left", "right", "right", "right", "right"),
    )
    recommendations = [
        (
            str(item.get("display_name", item.get("benchmark_model_id", "model"))),
            recommendation,
        )
        for item in payload.get("models", [])
        if isinstance(item, dict)
        and isinstance((recommendation := item.get("profile_recommendation")), dict)
        and recommendation.get("status") == "recommended"
        and (
            bool(item.get("conversion", {}).get("needed"))
            or len(item.get("profile_summaries", [])) > 1
        )
    ]
    if recommendations:
        print()
        print("Recommended profiles:")
        for label, recommendation in recommendations:
            supporting_metrics = recommendation.get("supporting_metrics", {})
            metric_name = str(supporting_metrics.get("comparison_metric", primary_metric))
            print(
                f"- {label}: {recommendation.get('profile_label')} "
                f"({_benchmark_metric_label(metric_name)}="
                f"{_format_benchmark_metric_value(_coerce_float(supporting_metrics.get('comparison_metric_value')), metric_name)})",
            )
            print(f"  {recommendation.get('reason')}")
            for note in recommendation.get("tradeoff_notes", [])[:3]:
                print(f"  - {note}")
    serving_profiles = [
        (
            str(item.get("display_name", item.get("benchmark_model_id", "model"))),
            optimized.get("serving_profile"),
        )
        for item in payload.get("results", [])
        if isinstance(item, dict)
        and isinstance((optimized := item.get("optimized")), dict)
        and isinstance(optimized.get("serving_profile"), dict)
    ]
    if serving_profiles:
        print()
        print("Serving profiles:")
        seen_models: set[str] = set()
        for label, profile in serving_profiles:
            if label in seen_models:
                continue
            seen_models.add(label)
            print(f"- {label}")
            _print_serving_profile_summary(profile, prefix="  ")


def _print_external_adapter_benchmark(payload: dict[str, Any]) -> None:
    comparison = payload.get("comparison", {})
    routing_preference = payload.get("routing_preference", {})
    feature_preservation = payload.get("feature_preservation", {})
    metric_summaries = comparison.get("metric_summaries", {})
    primary_metric = str(comparison.get("primary_metric", "cold_total_seconds"))
    primary_summary = metric_summaries.get(primary_metric, {})
    print("LewLM native vs local external adapter")
    print(f"model: {payload['model_id']}")
    print(f"native runtime: {payload['native'].get('runtime')}")
    print(f"external runtime: {payload['external_adapter'].get('runtime')}")
    print(f"primary metric: {_benchmark_metric_label(primary_metric)}")
    print(f"winner: {comparison.get('winner', 'unavailable')}")
    if isinstance(primary_summary, dict) and primary_summary.get("status") == "completed":
        print(
            f"native={_format_benchmark_metric_value(_coerce_float(primary_summary.get('native')), primary_metric)} "
            f"external={_format_benchmark_metric_value(_coerce_float(primary_summary.get('external')), primary_metric)}",
        )
    if payload["native"].get("status") == "failed":
        print(f"native error: {payload['native'].get('error')}")
    if payload["external_adapter"].get("status") == "failed":
        print(f"external error: {payload['external_adapter'].get('error')}")
    print(
        "preserved: "
        + (
            ", ".join(feature_preservation.get("preserved", []))
            if feature_preservation.get("preserved")
            else "none"
        ),
    )
    print(
        "degraded: "
        + (
            ", ".join(feature_preservation.get("degraded", []))
            if feature_preservation.get("degraded")
            else "none"
        ),
    )
    print(
        "rejected: "
        + (
            ", ".join(feature_preservation.get("rejected", []))
            if feature_preservation.get("rejected")
            else "none"
        ),
    )
    if isinstance(routing_preference, dict):
        if routing_preference.get("applied"):
            print(
                "routing preference: "
                f"{routing_preference.get('selected_runtime_name')} "
                f"({routing_preference.get('selected_runtime_affinity')})",
            )
        elif routing_preference.get("reason") is not None:
            print(f"routing preference: {routing_preference['reason']}")
    if payload.get("artifact") is not None:
        print(f"artifact: {payload['artifact']['artifact_path']}")


def _benchmark_saved_value(payload: dict[str, Any]) -> str:
    if payload.get("comparison_available") is False:
        return "n/a"
    value = _coerce_float(payload.get("average_time_saved_seconds", payload.get("time_saved_seconds")))
    return _format_delta(value)


def _benchmark_saved_percent(payload: dict[str, Any]) -> str:
    if payload.get("comparison_available") is False:
        return "n/a"
    value = _coerce_float(payload.get("average_time_saved_percent", payload.get("time_saved_percent")))
    return _format_percent(value)


def _benchmark_conversion_value(payload: dict[str, Any]) -> str:
    if not payload.get("needed", False):
        return "-"
    profile_label = payload.get("profile_label")
    status = str(payload.get("status", "unknown"))
    if status != "completed":
        return f"{profile_label}: failed" if isinstance(profile_label, str) and profile_label else "failed"
    if payload.get("cache_hit", False):
        cached_label = "cached"
        if isinstance(profile_label, str) and profile_label:
            cached_label = f"{profile_label} cached"
        return _style(cached_label, "36")
    duration = _coerce_float(payload.get("duration_seconds"))
    if isinstance(profile_label, str) and profile_label:
        return f"{profile_label} {_format_seconds(duration) if duration is not None else 'done'}"
    return _format_seconds(duration) if duration is not None else "done"


def _benchmark_scheduler_evidence_line(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    detail_parts: list[str] = []
    throughput = _coerce_float(metrics.get("throughput_requests_per_second"))
    direct_equivalent = _coerce_float(metrics.get("direct_equivalent_requests_per_second"))
    if throughput is not None and direct_equivalent is not None:
        detail_parts.append(f"{_format_rate(throughput)} vs direct {_format_rate(direct_equivalent)}")
    utilization = _coerce_float(metrics.get("average_batch_utilization"))
    if utilization is not None:
        detail_parts.append(f"util {_format_percent(utilization * 100.0)}")
    penalty = _coerce_float(metrics.get("single_request_penalty_seconds"))
    budget = _coerce_float(metrics.get("single_request_penalty_budget_seconds"))
    if penalty is not None:
        detail_parts.append(
            f"single penalty {_format_delta(penalty)}"
            + (f" / budget {_format_seconds(budget)}" if budget is not None else "")
        )
    return f"scheduler: {payload['status']}  " + "  ".join(detail_parts)


def _benchmark_residency_evidence_line(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    detail_parts: list[str] = []
    ratio = _coerce_float(metrics.get("average_warm_over_cold_ttft_ratio"))
    if ratio is not None:
        detail_parts.append(f"warm/cold TTFT {ratio:.4f}x")
    cache_restores = _coerce_int(metrics.get("total_cache_restores"))
    if cache_restores is not None:
        detail_parts.append(f"restores {cache_restores}")
    cached_tokens = _coerce_int(metrics.get("total_warm_saved_prefill_tokens"))
    if cached_tokens is not None:
        detail_parts.append(f"saved tokens {cached_tokens}")
    return f"residency: {payload['status']}  " + "  ".join(detail_parts)

def _format_benchmark_metric_value(value: float | None, metric_name: str) -> str:
    if _benchmark_metric_unit(metric_name) == "tokens/s":
        return _format_rate(value)
    return _format_seconds(value)


def _format_benchmark_advantage(value: float | None, metric_name: str) -> str:
    if value is None:
        return "n/a"
    if _benchmark_metric_unit(metric_name) == "tokens/s":
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.4f} tok/s"
    return _format_delta(value)


def handle_warm(args: argparse.Namespace, settings: LewLMSettings, services: LewLMServices | None = None) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    decision = asyncio.run(resolved_services.model_router.warm_model(args.model))
    payload = {
        "status": "warmed",
        "model_id": decision.model_id,
        "runtime": decision.runtime_name,
        "reason": decision.reason,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"warmed {payload['model_id']} via {payload['runtime']}")
    return ExitCode.OK


def handle_unload(args: argparse.Namespace, settings: LewLMSettings, services: LewLMServices | None = None) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    decision = asyncio.run(resolved_services.model_router.unload_model(args.model))
    payload = {
        "status": "unloaded",
        "model_id": decision.model_id,
        "runtime": decision.runtime_name,
        "reason": decision.reason,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"unloaded {payload['model_id']} via {payload['runtime']}")
    return ExitCode.OK


def handle_chat(args: argparse.Namespace, settings: LewLMSettings, services: LewLMServices | None = None) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    reasoning_visibility = _cli_reasoning_visibility(args, settings)
    prompt = args.message or args.prompt
    if not prompt:
        raise ConfigurationError("A prompt is required for `lewlm chat`.")
    if args.session_id and args.save_session:
        raise ConfigurationError("Use either `--session-id` or `--save-session`, not both.")
    if args.session_title and not args.save_session:
        raise ConfigurationError("`--session-title` requires `--save-session`.")
    if args.session_context_policy != "full_history" and not args.save_session:
        raise ConfigurationError("`--session-context-policy` requires `--save-session`.")

    messages: list[GenerateMessage] = []
    prompt_request = _build_cli_prompt_request(args)
    allowed_prompt_file_roots = _cli_prompt_file_roots(settings, args)

    attachment_paths = [Path(path).expanduser().resolve(strict=False) for path in args.attach_file + args.attach_image + args.attach_audio]
    if attachment_paths:
        content_parts: list[InputTextPart | InputFilePart | InputImagePart | InputAudioPart] = [
            InputTextPart(type="text", text=prompt)
        ]
        content_parts.extend(
            InputFilePart(type="input_file", path=str(path))
            for path in (Path(value).expanduser().resolve(strict=False) for value in args.attach_file)
        )
        content_parts.extend(
            InputImagePart(type="input_image", path=str(path))
            for path in (Path(value).expanduser().resolve(strict=False) for value in args.attach_image)
        )
        content_parts.extend(
            InputAudioPart(type="input_audio", path=str(path))
            for path in (Path(value).expanduser().resolve(strict=False) for value in args.attach_audio)
        )
        file_access_roots = tuple(sorted({path.parent for path in attachment_paths}, key=str))
        user_message = asyncio.run(
            normalize_chat_messages(
                [ChatMessage(role="user", content=content_parts)],
                resolved_services,
                file_access_roots=file_access_roots,
            ),
        )[0]
        messages.append(user_message)
    else:
        messages.append(GenerateMessage(role="user", content=prompt))

    session_id = args.session_id
    if args.save_session:
        session = resolved_services.session_history_service.create_session(
            title=args.session_title,
            context_policy=args.session_context_policy,
        )
        session_id = session.session_id
    conversation_messages = (
        resolved_services.session_history_service.build_conversation_messages(
            session_id=session_id,
            new_messages=messages,
        )
        if session_id is not None
        else messages
    )

    if args.stream:
        async def run_stream() -> tuple[dict[str, object], ReasoningOutput | None]:
            stream_session = await resolved_services.chat_orchestrator.stream(
                model_id=args.model,
                messages=conversation_messages,
                max_tokens=512,
                temperature=0.7,
                apply_serving_profile=not args.disable_serving_profile,
                reasoning_visibility=reasoning_visibility,
                prompt_request=prompt_request,
                allowed_prompt_file_roots=allowed_prompt_file_roots,
            )
            collected: list[str] = []
            async for delta in stream_session.stream:
                collected.append(delta)
                if not args.json:
                    print(delta, end="", flush=True)
            output_text = "".join(collected)
            if session_id is not None:
                resolved_services.session_history_service.record_turn(
                    session_id=session_id,
                    request_kind="cli.chat",
                    input_messages=messages,
                    response_message=GenerateMessage(role="assistant", content=output_text),
                    requested_model_id=args.model,
                    model_id=stream_session.model_id,
                    max_tokens=512,
                    temperature=0.7,
                    finish_reason="stop",
                    metadata=stream_session.request_metadata,
                )
            payload = {
                "id": stream_session.request_id,
                "model": stream_session.model_id,
                "output_text": output_text,
                "message_count": stream_session.prompt_trace.message_count,
                "session_id": session_id,
                "reasoning": stream_session.reasoning.model_dump(mode="json") if stream_session.reasoning is not None else None,
                "serving_profile": (
                    stream_session.serving_profile.model_dump(mode="json")
                    if stream_session.serving_profile is not None
                    else None
                ),
            }
            if args.json and prompt_request is not None and prompt_request.include_trace:
                payload["prompt_trace"] = stream_session.prompt_trace.model_dump(mode="json", by_alias=True)
            return payload, stream_session.reasoning

        payload, reasoning = asyncio.run(run_stream())
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            if isinstance(payload.get("serving_profile"), dict):
                _print_serving_profile_summary(payload.get("serving_profile"), prefix="")
            _print_cli_reasoning(reasoning)
            if session_id is not None:
                print()
                print(f"session: {session_id}")
        return ExitCode.OK

    execution = asyncio.run(
        resolved_services.chat_orchestrator.complete(
            model_id=args.model,
            messages=conversation_messages,
            max_tokens=512,
            temperature=0.7,
            apply_serving_profile=not args.disable_serving_profile,
            reasoning_visibility=reasoning_visibility,
            prompt_request=prompt_request,
            allowed_prompt_file_roots=allowed_prompt_file_roots,
        ),
    )
    if session_id is not None:
        resolved_services.session_history_service.record_turn(
            session_id=session_id,
            request_kind="cli.chat",
            input_messages=messages,
            response_message=GenerateMessage(role="assistant", content=execution.response.output_text),
            requested_model_id=args.model,
            model_id=execution.response.model_id,
            max_tokens=512,
            temperature=0.7,
            finish_reason=execution.response.finish_reason,
            usage=execution.response.usage,
            metadata=execution.request_metadata,
        )
    payload = {
        "id": execution.request_id,
        "model": execution.response.model_id,
        "output_text": execution.response.output_text,
        "finish_reason": execution.response.finish_reason,
        "message_count": execution.prompt_trace.message_count,
        "session_id": session_id,
        "reasoning": execution.response.reasoning.model_dump(mode="json") if execution.response.reasoning is not None else None,
        "serving_profile": (
            execution.serving_profile.model_dump(mode="json")
            if execution.serving_profile is not None
            else None
        ),
    }
    if args.json and prompt_request is not None and prompt_request.include_trace:
        payload["prompt_trace"] = execution.prompt_trace.model_dump(mode="json", by_alias=True)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(execution.response.output_text)
        if execution.serving_profile is not None and execution.serving_profile.profile_id is not None:
            print(
                f"serving profile: {execution.serving_profile.profile_id} "
                f"({', '.join(f'{key}={value}' for key, value in execution.serving_profile.effective_settings.items())})",
            )
        _print_cli_reasoning(execution.response.reasoning)
        if session_id is not None:
            print(f"session: {session_id}")
    return ExitCode.OK


def _build_cli_prompt_request(args: argparse.Namespace) -> PromptCompilationRequest | None:
    include_trace = bool(
        args.json
        and any(
            (
                args.system_prompt,
                args.developer_prompt,
                args.pretext_file,
                args.system_prompt_file,
                args.skills_file,
                args.response_format_file,
                args.output_schema_file,
                args.tools_file,
                args.mcp_tools_file,
            ),
        )
    )
    prompt_request = PromptCompilationRequest(
        actor="cli",
        system_prompt=args.system_prompt,
        developer_prompt=args.developer_prompt,
        pretext_path=args.pretext_file,
        system_prompt_file_path=args.system_prompt_file,
        skills_path=args.skills_file,
        response_format_path=args.response_format_file,
        output_schema_path=args.output_schema_file,
        tools_path=args.tools_file,
        mcp_tools_path=args.mcp_tools_file,
        include_trace=include_trace,
    )
    if prompt_request.include_trace or prompt_request.has_requested_overrides():
        return prompt_request
    return None


def _cli_reasoning_visibility(args: argparse.Namespace, settings: LewLMSettings) -> ReasoningVisibility:
    if args.reasoning_visibility is None:
        return settings.reasoning_visibility
    return ReasoningVisibility(args.reasoning_visibility)


def _print_cli_reasoning(reasoning: ReasoningOutput | None) -> None:
    if reasoning is None or not reasoning.available:
        return
    print()
    if reasoning.visibility == ReasoningVisibility.SUMMARIZED and reasoning.summary:
        print(f"[reasoning-summary] {reasoning.summary}")
        return
    if reasoning.visibility == ReasoningVisibility.RAW_MODEL_EMITTED and reasoning.content:
        print("[reasoning]")
        print(reasoning.content)
        print("[/reasoning]")


def _cli_prompt_file_roots(
    settings: LewLMSettings,
    args: argparse.Namespace,
) -> tuple[Path, ...] | None:
    file_paths = [
        args.pretext_file,
        args.system_prompt_file,
        args.skills_file,
        args.response_format_file,
        args.output_schema_file,
        args.tools_file,
        args.mcp_tools_file,
    ]
    if not any(file_paths):
        return None
    roots = {Path(root).expanduser().resolve(strict=False) for root in settings.file_access_roots}
    roots.update(
        Path(path).expanduser().resolve(strict=False).parent
        for path in file_paths
        if path is not None
    )
    return tuple(sorted(roots, key=str))


def handle_generate_doc(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    input_path = Path(args.input).expanduser().resolve(strict=False)
    output_path = Path(args.output).expanduser()
    _, document_text = read_scoped_text_file(
        input_path,
        allowed_roots=(input_path.parent,),
        purpose="Document input",
        media_type="application/json",
    )
    document = DocumentIR.model_validate_json(document_text)
    response = resolved_services.tool_execution_service.execute(
        DocumentGenerateToolRequest(
            input=GenerateDocumentToolInput(
                output_format=DocumentOutputFormat(args.format),
                document=document,
                file_name=output_path.name,
                authorized_actions=list(args.authorize),
                idempotency_key=args.idempotency_key,
            ),
        ),
        actor="cli",
        allowed_file_roots=(input_path.parent,),
        base_dir=input_path.parent,
    )
    artifact_bytes = base64.b64decode(str(response.result["content_base64"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(artifact_bytes)
    payload = {
        "request_id": response.request_id,
        "idempotency_key": response.idempotency_key,
        "idempotent_replay": response.idempotent_replay,
        "file_name": response.result["file_name"],
        "output_format": response.result["output_format"],
        "media_type": response.result["media_type"],
        "size_bytes": response.result["size_bytes"],
        "output_path": str(output_path),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        suffix = " [replayed]" if payload["idempotent_replay"] else ""
        print(f"generated {payload['output_path']} ({payload['size_bytes']} bytes){suffix}")
    return ExitCode.OK


def handle_transform(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    input_path = Path(args.input).expanduser().resolve(strict=False)
    output_path = Path(args.output).expanduser()
    _, request_text = read_scoped_text_file(
        input_path,
        allowed_roots=(input_path.parent,),
        purpose="Transform request",
        media_type="application/json",
    )
    request = parse_document_transform_request(request_text)
    request = request.model_copy(
        update={
            "file_name": output_path.name,
            "authorized_actions": _merge_authorizations(request.authorized_actions, args.authorize),
            "idempotency_key": args.idempotency_key or request.idempotency_key,
        },
    )
    response = resolved_services.tool_execution_service.execute(
        DocumentTransformToolRequest(input=request),
        actor="cli",
        allowed_file_roots=(input_path.parent,),
        base_dir=input_path.parent,
    )
    artifact_bytes = base64.b64decode(str(response.result["content_base64"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(artifact_bytes)
    payload = {
        "request_id": response.request_id,
        "idempotency_key": response.idempotency_key,
        "idempotent_replay": response.idempotent_replay,
        "skill": request.skill,
        "file_name": response.result["file_name"],
        "output_format": response.result["output_format"],
        "media_type": response.result["media_type"],
        "size_bytes": response.result["size_bytes"],
        "output_path": str(output_path),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        suffix = " [replayed]" if payload["idempotent_replay"] else ""
        print(f"transformed {payload['output_path']} via {payload['skill']}{suffix}")
    return ExitCode.OK


def handle_run_tool(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    input_path = Path(args.input).expanduser().resolve(strict=False)
    _, request_text = read_scoped_text_file(
        input_path,
        allowed_roots=(input_path.parent,),
        purpose="Tool execution request",
        media_type="application/json",
    )
    request = parse_tool_execution_request(request_text)
    if args.authorize:
        request = _inject_tool_authorizations(request, list(args.authorize))
    if args.idempotency_key is not None:
        request = _inject_tool_idempotency_key(request, args.idempotency_key)
    response = resolved_services.tool_execution_service.execute(
        request,
        actor="cli",
        allowed_file_roots=(input_path.parent,),
        base_dir=input_path.parent,
    )
    payload = response.model_dump(mode="json")
    if args.output is not None:
        content_base64 = payload["result"].get("content_base64")
        if content_base64 is None:
            raise ConfigurationError("`--output` is only supported for artifact-producing tools.")
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(content_base64))
        payload["result"].pop("content_base64", None)
        payload["result"]["output_path"] = str(output_path)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        result = payload["result"]
        suffix = " [replayed]" if payload.get("idempotent_replay") else ""
        if "output_path" in result:
            print(f"ran {payload['tool']} -> {result['output_path']}{suffix}")
        else:
            print(f"ran {payload['tool']} ({payload['trace']['summary']}){suffix}")
    return ExitCode.OK


def handle_cache(args: argparse.Namespace, settings: LewLMSettings, services: LewLMServices | None = None) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    stats = resolved_services.telemetry_service.cache_stats()
    payload = stats.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"cache dir: {payload['cache_dir']}")
        print(f"artifacts: {payload['artifact_count']}")
        print(f"runtime response entries: {payload['runtime_response_count']}")
        print(f"block cache entries: {payload['block_cache_count']}")
        print(f"multimodal feature entries: {payload['multimodal_feature_count']}")
        print(f"files: {payload['file_count']}")
        print(f"size: {payload['total_size_bytes']} bytes")
        print(
            "hits: "
            f"{payload['cache_hits']} (conversion {payload['conversion_cache_hits']}, "
            f"runtime {payload['runtime_cache_hits']}, block {payload['block_cache_hits']}), "
            "misses: "
            f"{payload['cache_misses']} (conversion {payload['conversion_cache_misses']}, "
            f"runtime {payload['runtime_cache_misses']}, block {payload['block_cache_misses']})",
        )
        print("performance features:")
        _print_performance_features(payload["performance_features"])
    return ExitCode.OK


def handle_cache_clear_conversions(
    args: argparse.Namespace,
    settings: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    resolved_services = services or bootstrap_services(settings)
    purge_summary = resolved_services.conversion_service.clear_cache()
    conversion_root = Path(str(purge_summary["cache_root"])).expanduser().resolve(strict=False)
    scan_roots = _dedupe_scan_roots((*settings.models_dir, conversion_root))
    summary = resolved_services.model_registry.scan(roots=scan_roots)
    payload = {
        **purge_summary,
        "scan": summary.model_dump(mode="json"),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"cleared conversion cache: {payload['cache_root']}")
        print(
            f"removed {payload['removed_entries']} cache entr{'y' if payload['removed_entries'] == 1 else 'ies'} "
            f"and {payload['cleared_artifact_records']} artifact record"
            f"{'' if payload['cleared_artifact_records'] == 1 else 's'}",
        )
        if payload["cleared_idempotent_records"]:
            print(
                f"cleared {payload['cleared_idempotent_records']} conversion idempotency record"
                f"{'' if payload['cleared_idempotent_records'] == 1 else 's'}",
            )
        print(
            "rescanned "
            f"{len(summary.roots_scanned)} root(s); {summary.removed_count} stale entr"
            f"{'y' if summary.removed_count == 1 else 'ies'} removed from the registry",
        )
    return ExitCode.OK


def handle_not_implemented(
    args: argparse.Namespace,
    _: LewLMSettings,
    services: LewLMServices | None = None,
) -> ExitCode:
    raise NotImplementedLewLMError(f"`{args.command}` is reserved for a later milestone.")


def _dedupe_scan_roots(roots: tuple[Path, ...]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        normalized = Path(root).expanduser().resolve(strict=False)
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _print_scan_summary(summary: ModelScanSummary) -> None:
    print(
        "scanned "
        f"{len(summary.roots_scanned)} root(s), found {summary.discovered_count} model(s) "
        f"({summary.new_count} new, {summary.updated_count} updated, "
        f"{summary.unchanged_count} unchanged, {summary.removed_count} removed)",
    )
    for manifest in summary.manifests:
        modalities = ",".join(modality.value for modality in manifest.modality)
        print(f"- {manifest.display_name} [{manifest.format_type.value}/{modalities}] {manifest.source_path}")


def _print_inventory(inventory: ModelInventory) -> None:
    for line in _inventory_display_lines(inventory):
        print(line)


def _print_inventory_raw(inventory: ModelInventory) -> None:
    if not inventory.items:
        print("No models registered.")
        return
    for manifest in inventory.items:
        modalities = ",".join(modality.value for modality in manifest.modality)
        affinities = ",".join(affinity.value for affinity in manifest.runtime_affinity)
        print(f"{manifest.model_id}: {manifest.format_type.value} [{modalities}] -> {affinities}")
        print(f"  path: {manifest.source_path}")


def _print_capability_report(report: ModelCapabilityReport) -> None:
    modalities = ",".join(modality.value for modality in report.modality)
    platform = report.host_platform
    print(f"{report.model_id}: {report.format_type.value} [{modalities}]")
    print(f"validation key: {report.validation_key}")
    print(
        "host: "
        f"{platform.system} {platform.machine} "
        f"(release {platform.release}, Python {platform.python_version})",
    )
    print("runtime candidates:")
    for candidate in report.runtime_candidates:
        print(
            f"- {candidate.runtime_name} ({candidate.runtime_affinity.value}): "
            f"registered={'yes' if candidate.registered else 'no'}, "
            f"available={'yes' if candidate.available else 'no'}, "
            f"host_supported={'yes' if candidate.host_platform_supported else 'no'}, "
            f"manifest_supported={'yes' if candidate.supports_manifest else 'no'}",
        )
        if candidate.availability_reason:
            print(f"  reason: {candidate.availability_reason}")
    print("target platforms:")
    if not report.target_platforms:
        print("- none")
    for target in report.target_platforms:
        runtimes = ",".join(affinity.value for affinity in target.runtime_affinities) or "n/a"
        print(
            f"- {target.system} {target.machine}: "
            f"{'supported' if target.supported else 'unsupported'} "
            f"(readiness={target.readiness_state}, verification={target.verification_method}, runtimes={runtimes})",
        )
        print(f"  reason: {target.reason}")
        if target.fallback_reason and target.fallback_reason != target.reason:
            print(f"  fallback: {target.fallback_reason}")
        if target.verified_hosts:
            print(f"  verified hosts: {', '.join(target.verified_hosts)}")
        for hint in target.install_hints:
            print(f"  install hint: {hint}")
        for note in target.notes:
            print(f"  note: {note}")
    print("capabilities:")
    if not report.capabilities:
        print("- none")
    else:
        for capability in report.capabilities:
            runtime_name = capability.runtime_name or "n/a"
            print(
                f"- {capability.capability.value}: "
                f"{'supported' if capability.supported else 'unsupported'} via {runtime_name}",
            )
            print(f"  reason: {capability.reason}")
            if capability.estimated_memory_mb is not None:
                print(f"  estimated memory: {capability.estimated_memory_mb} MB")
            for alternative in capability.alternatives:
                print(f"  alternative: {alternative}")
            for note in capability.notes:
                print(f"  note: {note}")
    print("measured capability registry:")
    if not report.measured_capabilities:
        print("- none")
        return
    for measured in report.measured_capabilities:
        print(f"- {measured.category.value}: {measured.status.value}")
        print(f"  reason: {measured.reason}")
        if measured.latest_recorded_at is not None:
            print(f"  recorded: {measured.latest_recorded_at.isoformat()}")
        if measured.runtime_names:
            print(f"  runtimes: {', '.join(measured.runtime_names)}")
        if measured.sources:
            print(f"  sources: {', '.join(measured.sources)}")


def _print_performance_features(features: list[dict[str, Any]]) -> None:
    if not features:
        print("- none")
        return
    for feature in features:
        state = "unsupported"
        if feature.get("supported"):
            state = "active" if feature.get("active") else "available"
        print(f"- {feature['feature']}: {state}")
        print(f"  reason: {feature['reason']}")
        metrics = feature.get("metrics") or {}
        if metrics:
            print("  metrics: " + ", ".join(f"{key}={value}" for key, value in metrics.items()))
        for note in feature.get("notes", []):
            print(f"  note: {note}")
        for guidance in feature.get("fallback_guidance", []):
            print(f"  fallback: {guidance}")


def _selected_benchmark_speculation_mode(payload: dict[str, Any]) -> str | None:
    selected_mode: str | None = None
    candidate_count: int | None = None
    workload_class: str | None = None
    scenarios = payload.get("scenarios")
    if isinstance(scenarios, list):
        for scenario in scenarios:
            if not isinstance(scenario, dict) or scenario.get("scenario") != "speculation_selection":
                continue
            metrics = scenario.get("metrics")
            if not isinstance(metrics, dict):
                continue
            mode_value = metrics.get("selected_mode")
            if isinstance(mode_value, str) and mode_value:
                selected_mode = mode_value
            count_value = metrics.get("candidate_count")
            if isinstance(count_value, int):
                candidate_count = count_value
            workload_value = metrics.get("workload_class")
            if isinstance(workload_value, str) and workload_value:
                workload_class = workload_value
            break
    if selected_mode is None:
        return None
    suffix_parts: list[str] = []
    if workload_class is not None:
        suffix_parts.append(workload_class)
    if candidate_count is not None and candidate_count > 0:
        suffix_parts.append(f"{candidate_count} candidate{'s' if candidate_count != 1 else ''}")
    if suffix_parts:
        return f"{selected_mode} ({', '.join(suffix_parts)})"
    return selected_mode


def _benchmark_speculation_suffix(payload: dict[str, Any]) -> str:
    selected_mode = _selected_benchmark_speculation_mode(payload)
    if selected_mode is None:
        return ""
    return f" speculation={selected_mode}"


def _print_serving_profile_summary(payload: dict[str, Any] | None, *, prefix: str = "") -> None:
    if not isinstance(payload, dict):
        return
    profile_id = payload.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        return
    effective_settings = payload.get("effective_settings")
    accepted_settings = payload.get("accepted_settings")
    summary_settings = effective_settings if isinstance(effective_settings, dict) and effective_settings else accepted_settings
    summary_suffix = (
        " (" + ", ".join(f"{key}={value}" for key, value in summary_settings.items()) + ")"
        if isinstance(summary_settings, dict) and summary_settings
        else ""
    )
    workload_class = payload.get("workload_class")
    workload_suffix = f" [workload={workload_class}]" if isinstance(workload_class, str) and workload_class else ""
    print(f"{prefix}serving profile: {profile_id}{workload_suffix}{summary_suffix}")
    if isinstance(effective_settings, dict) and effective_settings:
        print(
            f"{prefix}effective settings: "
            + ", ".join(f"{key}={value}" for key, value in effective_settings.items()),
        )
    if isinstance(accepted_settings, dict) and accepted_settings:
        print(
            f"{prefix}accepted: "
            + ", ".join(f"{key}={value}" for key, value in accepted_settings.items()),
        )
    rejected_settings = payload.get("rejected_settings")
    if isinstance(rejected_settings, dict) and rejected_settings:
        rejected_parts = []
        for key, rejection in rejected_settings.items():
            if not isinstance(rejection, dict):
                continue
            rejected_parts.append(f"{key} ({rejection.get('reason', 'rejected')})")
        if rejected_parts:
            print(f"{prefix}rejected: " + ", ".join(rejected_parts))


def _print_benchmark_scenario_highlights(payload: dict[str, Any]) -> None:
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return
    highlight_lines: list[str] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        scenario_name = scenario.get("scenario")
        metrics = scenario.get("metrics")
        if not isinstance(metrics, dict):
            continue
        summary = [str(scenario.get("status", "observed"))]
        if scenario_name == "multimodal_encoder_reuse":
            encoder_hits = metrics.get("multimodal_encoder_cache_hit_delta")
            encoder_misses = metrics.get("multimodal_encoder_cache_miss_delta")
            attachment_hits = metrics.get("multimodal_feature_cache_hit_delta")
            ratio = metrics.get("average_second_over_first_ratio")
            advantage = metrics.get("average_encoder_advantage_seconds")
            if encoder_hits is not None:
                summary.append(f"encoder_hits={encoder_hits}")
            if encoder_misses is not None:
                summary.append(f"encoder_misses={encoder_misses}")
            if attachment_hits is not None:
                summary.append(f"attachment_hits={attachment_hits}")
            if ratio is not None:
                summary.append(f"ratio={ratio}")
            if advantage is not None:
                summary.append(f"saved={advantage}s")
            highlight_lines.append(f"- multimodal_encoder_reuse: {', '.join(summary)}")
            continue
        if scenario_name == "speculation_selection":
            selected_mode = metrics.get("selected_mode")
            workload = metrics.get("workload_class")
            candidate_count = metrics.get("candidate_count")
            safe_count = metrics.get("safe_candidate_count")
            skipped_count = metrics.get("skipped_candidate_count")
            acceptance = _coerce_float(metrics.get("selected_acceptance_rate"))
            verified = metrics.get("selected_verified_tokens")
            rollback = metrics.get("selected_rollback_tokens")
            if workload is not None:
                summary.insert(0, f"workload={workload}")
            if selected_mode is not None:
                summary.append(f"selected={selected_mode}")
            if candidate_count is not None:
                summary.append(f"candidates={candidate_count}")
            if safe_count is not None:
                summary.append(f"safe={safe_count}")
            if skipped_count is not None:
                summary.append(f"skipped={skipped_count}")
            if acceptance is not None:
                summary.append(f"acceptance={acceptance}")
            if verified is not None:
                summary.append(f"verified={verified}")
            if rollback is not None:
                summary.append(f"rollback={rollback}")
            highlight_lines.append(f"- speculation_selection: {', '.join(summary)}")
            samples = scenario.get("samples")
            if isinstance(samples, list):
                for sample in samples:
                    if not isinstance(sample, dict):
                        continue
                    sample_metrics = sample.get("metrics")
                    if not isinstance(sample_metrics, dict):
                        continue
                    status = sample_metrics.get("selection_status")
                    reason = sample_metrics.get("outcome_reason")
                    mode = sample_metrics.get("mode")
                    if status not in {"skipped", "lost"} or not isinstance(reason, str) or not reason:
                        continue
                    highlight_lines.append(f"  - {mode}: {reason}")
            continue
        if scenario_name == "frontier_architecture_modes":
            subtype = metrics.get("architecture_subtype")
            if subtype is None and isinstance(scenario.get("samples"), list) and scenario["samples"]:
                sample_metrics = scenario["samples"][0].get("metrics", {})
                if isinstance(sample_metrics, dict):
                    subtype = sample_metrics.get("architecture_subtype")
                    if sample_metrics.get("effective_loaded_memory_mb") is not None:
                        summary.append(f"effective_mem={sample_metrics['effective_loaded_memory_mb']}MB")
                    if sample_metrics.get("resident_expert_count") is not None:
                        summary.append(f"resident_experts={sample_metrics['resident_expert_count']}")
                    if sample_metrics.get("expert_swap_count") is not None:
                        summary.append(f"swaps={sample_metrics['expert_swap_count']}")
                    if sample_metrics.get("state_cache_bytes") is not None:
                        summary.append(f"state_cache={sample_metrics['state_cache_bytes']}B")
                    if sample_metrics.get("planning_only") is not None:
                        summary.append(f"planning_only={sample_metrics['planning_only']}")
            if subtype is not None:
                summary.insert(0, str(subtype))
            highlight_lines.append(f"- frontier_architecture_modes: {', '.join(summary)}")
            continue
        if scenario_name == "distributed_pipeline_scaling":
            bottleneck = metrics.get("bottleneck")
            if bottleneck is not None:
                summary.insert(0, f"bottleneck={bottleneck}")
            throughput = _coerce_float(metrics.get("throughput_tokens_per_second"))
            if throughput is not None:
                summary.append(f"steady={_format_rate(throughput)}")
            critical = _coerce_float(metrics.get("critical_path_seconds"))
            if critical is not None:
                summary.append(f"critical={_format_seconds(critical)}")
            utilization = _coerce_float(metrics.get("average_stage_utilization"))
            if utilization is not None:
                summary.append(f"util={_format_percent(utilization * 100.0)}")
            speedup = _coerce_float(metrics.get("speedup_vs_single_host_percent"))
            if speedup is not None:
                summary.append(f"speedup={_format_percent(speedup)}")
            highlight_lines.append(f"- distributed_pipeline_scaling: {', '.join(summary)}")
            continue
        if scenario_name != "mlx_acceleration_paths":
            continue
        kernel_paths = metrics.get("kernel_paths")
        if kernel_paths is not None:
            summary.append(f"kernels={kernel_paths}")
        compile_states = metrics.get("compile_states")
        if compile_states is not None:
            summary.append(f"compile_state={compile_states}")
        compiled_sample_count = metrics.get("compiled_sample_count")
        sample_count = metrics.get("sample_count")
        if compiled_sample_count is not None and sample_count is not None:
            summary.append(f"compiled={compiled_sample_count}/{sample_count}")
        decode_shortcut_sample_count = metrics.get("decode_shortcut_sample_count")
        if decode_shortcut_sample_count is not None and sample_count is not None:
            summary.append(f"decode_shortcuts={decode_shortcut_sample_count}/{sample_count}")
        shortcut_prefill_tokens = metrics.get("total_shortcut_prefill_tokens")
        if shortcut_prefill_tokens is not None:
            summary.append(f"shortcut_tokens={shortcut_prefill_tokens}")
        ratio = metrics.get("average_accelerated_over_stock_ratio")
        if ratio is not None:
            summary.append(f"ratio={ratio}")
        payoff = metrics.get("average_time_saved_seconds")
        if payoff is not None:
            summary.append(f"saved={payoff}s")
        fallback_count = metrics.get("fallback_sample_count")
        if fallback_count is not None:
            summary.append(f"fallbacks={fallback_count}")
        samples = scenario.get("samples")
        fallback_reasons: list[str] = []
        if isinstance(samples, list):
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                sample_metrics = sample.get("metrics")
                if not isinstance(sample_metrics, dict):
                    continue
                reason = sample_metrics.get("fallback_reason")
                if isinstance(reason, str) and reason and reason not in fallback_reasons:
                    fallback_reasons.append(reason)
        if fallback_reasons:
            summary.append(f"fallback_reason={' | '.join(fallback_reasons)}")
        highlight_lines.append(f"- mlx_acceleration_paths: {', '.join(summary)}")
    if not highlight_lines:
        return
    print("scenarios:")
    for line in highlight_lines:
        print(line)


def _print_cluster_benchmark_details(payload: dict[str, Any]) -> None:
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return
    scenario = next(
        (
            item
            for item in scenarios
            if isinstance(item, dict) and item.get("scenario") == "distributed_pipeline_scaling"
        ),
        None,
    )
    if not isinstance(scenario, dict):
        return
    metrics = scenario.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    print(
        "scaling: "
        f"critical={_format_seconds(_coerce_float(metrics.get('critical_path_seconds')))} "
        f"steady={_format_rate(_coerce_float(metrics.get('throughput_tokens_per_second')))} "
        f"util={_format_percent((_coerce_float(metrics.get('average_stage_utilization')) or 0.0) * 100.0)} "
        f"batch={_coerce_int(metrics.get('effective_batch_tokens')) or 0} "
        f"prefetch={_coerce_int(metrics.get('average_prefetch_tokens')) or 0} "
        f"bottleneck={metrics.get('bottleneck', 'balanced')}"
    )
    print(
        "breakdown: "
        f"compute={_format_percent(_coerce_float(metrics.get('compute_share_percent')))} "
        f"network={_format_percent(_coerce_float(metrics.get('network_share_percent')))} "
        f"scheduling={_format_percent(_coerce_float(metrics.get('scheduling_share_percent')))} "
        f"speedup={_format_percent(_coerce_float(metrics.get('speedup_vs_single_host_percent')))}"
    )
    samples = scenario.get("samples")
    if not isinstance(samples, list) or not samples:
        return
    rows: list[tuple[str, ...]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_metrics = sample.get("metrics")
        if not isinstance(sample_metrics, dict):
            continue
        rows.append(
            (
                str(sample_metrics.get("stage_name", sample_metrics.get("stage_index", "?"))),
                str(sample_metrics.get("worker_name", "worker")),
                str(sample_metrics.get("layer_span", "?")),
                _format_seconds(_coerce_float(sample_metrics.get("stage_elapsed_seconds"))),
                _format_percent((_coerce_float(sample_metrics.get("utilization")) or 0.0) * 100.0),
                f"{_coerce_int(sample_metrics.get('target_batch_tokens')) or 0}/{_coerce_int(sample_metrics.get('prefetch_tokens')) or 0}",
                str(sample_metrics.get("bottleneck", "balanced")),
            ),
        )
    if not rows:
        return
    print("stages:")
    _print_benchmark_table(
        ("stage", "worker", "layers", "elapsed", "util", "batch/prefetch", "bottleneck"),
        rows,
    )


def _print_skills(skills: list[BuiltInSkillDescriptor]) -> None:
    if not skills:
        print("No built-in skills registered.")
        return
    for skill in skills:
        print(f"{skill.name} v{skill.version} [{skill.tool_name}]")


def _print_skill_detail(skill: BuiltInSkillDescriptor) -> None:
    print(f"{skill.name} v{skill.version}")
    print(f"category: {skill.category}")
    print(f"tool: {skill.tool_name}")
    print(f"authorization: {skill.required_authorization}")
    print(f"description: {skill.description}")
    if skill.supported_input_hints:
        print("inputs: " + ", ".join(skill.supported_input_hints))
    if skill.supported_output_formats:
        print("outputs: " + ", ".join(output.value for output in skill.supported_output_formats))
    if skill.tags:
        print("tags: " + ", ".join(skill.tags))
    if skill.example_path:
        print(f"example: {skill.example_path}")


def _print_tools(tools: list[LocalToolDescriptor]) -> None:
    if not tools:
        print("No local tools registered.")
        return
    for tool in tools:
        print(f"{tool.name} v{tool.version} [{tool.required_authorization}]")


def _print_tool_detail(tool: LocalToolDescriptor) -> None:
    print(f"{tool.name} v{tool.version}")
    print(f"execution mode: {tool.execution_mode}")
    print(f"authorization: {tool.required_authorization}")
    print(f"result type: {tool.result_type}")
    print(f"description: {tool.description}")
    if tool.tags:
        print("tags: " + ", ".join(tool.tags))
    if tool.aliases:
        print("aliases: " + ", ".join(tool.aliases))
    if tool.input_schema:
        print("input schema:")
        print(json.dumps(tool.input_schema, indent=2, sort_keys=True))


def _print_config_summary(settings: LewLMSettings) -> None:
    print(f"{settings.app_name} {settings.version}")
    print(f"data dir: {settings.data_dir}")
    print(f"model roots: {', '.join(str(root) for root in settings.models_dir)}")
    print(f"runtime pack allowlist: {', '.join(settings.runtime_packs) or '(default)'}")
    print(f"runtime pack denylist: {', '.join(settings.disabled_runtime_packs) or '(none)'}")
    print(f"feature pack allowlist: {', '.join(settings.feature_packs) or '(default)'}")
    print(f"feature pack denylist: {', '.join(settings.disabled_feature_packs) or '(none)'}")
    print(f"file access roots: {', '.join(str(root) for root in settings.file_access_roots)}")
    print(f"validation manifests: {len(settings.validation_manifest_paths)}")
    for manifest_path in settings.validation_manifest_paths:
        print(f"  - {manifest_path}")
    print(f"runtime policy: {settings.runtime_policy}")
    print(f"cluster role: {settings.cluster_role}")
    print(f"cluster name: {settings.cluster_name}")
    if settings.cluster_coordinator_url is not None:
        print(f"cluster coordinator: {settings.cluster_coordinator_url}")
    if settings.cluster_public_base_url is not None:
        print(f"cluster public URL: {settings.cluster_public_base_url}")
    print(f"privacy mode: {'on' if settings.privacy_mode else 'off'}")
    print(f"outbound network: {'enabled' if settings.allow_outbound_network else 'disabled'}")
    print(f"audit log: {'enabled' if settings.audit_log_enabled else 'disabled'}")
    print(f"encrypted persistence: {'enabled' if settings.persistence_encryption_enabled else 'disabled'}")
    print(f"tool authorization: {'required' if settings.tool_authorization_required else 'not required'}")
    print(f"parser sandbox: {'enabled' if settings.parser_sandbox_enabled else 'disabled'}")
    print(f"conversion sandbox: {'enabled' if settings.conversion_sandbox_enabled else 'disabled'}")
    print(
        "release bundle command: python scripts/capture_release_bundle.py --output-dir out "
        "--require-target Darwin:arm64 --require-frontier-family dense_text "
        "--require-frontier-family speculative_family --minimum-verified-models 1"
    )
    print(
        "release validation command: python scripts/validate_release_candidate.py out "
        "--require-target Darwin:arm64 --require-frontier-family dense_text "
        "--require-frontier-family speculative_family --minimum-verified-models 1"
    )


def _print_cluster_status(status: ClusterStatus) -> None:
    print(f"cluster: {status.cluster_name}")
    print(f"role: {status.role}")
    print(f"node: {status.node_name}")
    if status.coordinator_url is not None:
        print(f"coordinator: {status.coordinator_url}")
    print(f"ready workers: {status.ready_worker_count}")
    print(f"stale workers: {status.stale_worker_count}")
    print(f"plans: {status.plan_count}")
    if status.worker_session is not None:
        print(f"worker session: {status.worker_session.worker_id}")
    if status.latest_execution_metrics:
        print(
            "last run: "
            f"critical={_format_seconds(_coerce_float(status.latest_execution_metrics.get('critical_path_seconds')))} "
            f"steady={_format_rate(_coerce_float(status.latest_execution_metrics.get('throughput_tokens_per_second')))} "
            f"util={_format_percent((_coerce_float(status.latest_execution_metrics.get('average_stage_utilization')) or 0.0) * 100.0)} "
            f"bottleneck={status.latest_execution_metrics.get('bottleneck', 'balanced')}"
        )
    for worker in status.workers:
        print(f"- {worker.worker_name} [{worker.status}] {worker.endpoint}")


def _print_sessions(sessions: list[SessionRecord]) -> None:
    if not sessions:
        print("No sessions saved.")
        return
    for session in sessions:
        title = session.title or "(untitled)"
        print(
            f"{session.session_id}: {title} "
            f"[policy={session.context_policy}, turns={session.turn_count}, messages={session.message_count}, updated={session.updated_at.isoformat()}]",
        )


def _print_session_detail(session: SessionDetail) -> None:
    title = session.title or "(untitled)"
    print(f"{session.session_id}: {title}")
    print(f"context policy: {session.context_policy}")
    print(f"turns: {session.turn_count}")
    print(f"messages: {session.message_count}")
    print(f"created: {session.created_at.isoformat()}")
    print(f"updated: {session.updated_at.isoformat()}")
    if session.metadata:
        print(f"metadata: {json.dumps(session.metadata, sort_keys=True)}")
    if not session.turns:
        print("No turns recorded.")
        return
    print("turns:")
    for index, turn in enumerate(session.turns, start=1):
        user_excerpt = _summarize_message_sequence(turn.input_messages)
        assistant_excerpt = _truncate_text(turn.response_message.content)
        print(
            f"{index}. {turn.request_kind} -> {turn.model_id} "
            f"[reason={turn.finish_reason}, created={turn.created_at.isoformat()}]",
        )
        print(f"   user: {user_excerpt}")
        print(f"   assistant: {assistant_excerpt}")


def _summarize_message_sequence(messages: list[GenerateMessage]) -> str:
    if not messages:
        return "(no messages)"
    return " | ".join(f"{message.role}: {_truncate_text(message.content)}" for message in messages)


def _truncate_text(value: str, *, limit: int = 120) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _inject_tool_authorizations(request, authorizations: list[str]):
    nested_input = request.input.model_copy(
        update={"authorized_actions": _merge_authorizations(request.input.authorized_actions, authorizations)},
    )
    return request.model_copy(update={"input": nested_input})


def _inject_tool_idempotency_key(request, idempotency_key: str):
    nested_input = request.input.model_copy(update={"idempotency_key": idempotency_key})
    return request.model_copy(update={"input": nested_input})


def _merge_authorizations(existing: Sequence[str], additional: Sequence[str]) -> list[str]:
    return sorted({*existing, *additional})


def _print_cli_error(exc: LewLMError) -> None:
    print(f"Error: {exc}", file=sys.stderr)
    fallback_guidance = _detail_lines(exc.details.get("fallback_guidance"))
    alternatives = _detail_lines(exc.details.get("alternatives"))
    if fallback_guidance:
        print("Guidance:", file=sys.stderr)
        for item in fallback_guidance:
            print(f"- {item}", file=sys.stderr)
    if alternatives:
        print("Diagnostics:", file=sys.stderr)
        for item in alternatives[:4]:
            print(f"- {item}", file=sys.stderr)


def _detail_lines(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _wait_for_job(services: LewLMServices, job_id: str) -> JobRecord:
    while True:
        job = services.conversion_service.get_job(job_id)
        if job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
            return job
        time.sleep(0.05)
