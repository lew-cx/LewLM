from __future__ import annotations

from pathlib import Path

import pytest

from lewlm.core.errors import FileAccessError
from lewlm.core.contracts import (
    ConversionStatus,
    GenerateMessage,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.prompting import PromptCompilationRequest, PromptCompiler
from lewlm.structured_output import GrammarResponseFormat


def _build_manifest(*, model_id: str, architecture_family: str) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        display_name=model_id,
        architecture_family=architecture_family,
        modality=(ModelModality.TEXT,),
        source_path=f"/tmp/{model_id}",
        format_type=ModelFormat.GGUF,
        runtime_affinity=(RuntimeAffinity.LLAMACPP,),
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint="fingerprint",
        last_validation_result=ModelValidationResult(
            status=ValidationState.VALID,
            message="ok",
        ),
    )


def test_prompt_compiler_builds_trace_from_local_overrides(
    temp_settings,
    sample_prompt_assets,
) -> None:
    compiler = PromptCompiler(temp_settings)

    result = compiler.compile(
        messages=[GenerateMessage(role="user", content="Plan milestone eight")],
        request=PromptCompilationRequest(
            actor="api",
            pretext_path=str(sample_prompt_assets["pretext"]),
            developer_prompt="Keep the output terse.",
            skills_path=str(sample_prompt_assets["skill"]),
            output_schema_path=str(sample_prompt_assets["output_schema"]),
            tools_path=str(sample_prompt_assets["tools"]),
            mcp_tools_path=str(sample_prompt_assets["mcp_tools"]),
            include_trace=True,
        ),
        model_manifest=_build_manifest(
            model_id="llama-3.2-3b-instruct",
            architecture_family="llama",
        ),
    )

    assert result.trace.selected_template == "tool_structured_output"
    assert result.trace.model_prompt_template is not None
    assert result.trace.model_prompt_template.id == "llama-instruct-v1"
    assert result.trace.message_count == len(result.messages)
    assert result.trace.output_contract.format == "json_schema"
    assert result.trace.serialized_model_prompt is not None
    assert "[INST]" in result.trace.serialized_model_prompt
    ingest_entry = next(entry for entry in result.trace.tool_plan if entry.name == "documents.ingest")
    assert ingest_entry.registered is True
    assert ingest_entry.execution_mode == "local_tool"
    assert ingest_entry.required_authorization == "document_ingest"
    local_lookup = next(entry for entry in result.trace.tool_plan if entry.name == "local_lookup")
    assert local_lookup.registered is False
    assert local_lookup.execution_mode == "prompt_only"
    mcp_lookup = next(entry for entry in result.trace.tool_plan if entry.name == "search_milestones")
    assert mcp_lookup.source == "mcp_tools_file"
    assert mcp_lookup.mcp_server == "roadmap"
    assert mcp_lookup.registered is False
    assert mcp_lookup.execution_mode == "prompt_only"
    assert mcp_lookup.metadata_trusted is False
    assert [override.source for override in result.trace.overrides] == [
        "pretext_file",
        "developer_prompt",
        "skills_file",
        "tools_file",
        "mcp_tools_file",
        "output_schema_file",
    ]


def test_prompt_compiler_resolves_chatml_template_for_qwen_models(temp_settings) -> None:
    compiler = PromptCompiler(temp_settings)

    result = compiler.compile(
        messages=[GenerateMessage(role="user", content="Return a compact status.")],
        request=PromptCompilationRequest(
            actor="api",
            include_trace=True,
        ),
        model_manifest=_build_manifest(
            model_id="qwen2.5-1.5b-instruct",
            architecture_family="qwen2",
        ),
    )

    assert result.trace.model_prompt_template is not None
    assert result.trace.model_prompt_template.id == "chatml-v1"
    assert result.trace.model_prompt_template.source == "architecture_family"
    assert result.trace.serialized_model_prompt is not None
    assert result.trace.serialized_model_prompt.startswith("<|im_start|>user")


def test_prompt_compiler_accepts_response_format_grammar_contract(temp_settings) -> None:
    compiler = PromptCompiler(temp_settings)

    result = compiler.compile(
        messages=[GenerateMessage(role="user", content="Return only ok")],
        request=PromptCompilationRequest(
            actor="api",
            response_format=GrammarResponseFormat(
                grammar='root ::= "ok"',
                syntax="ebnf",
                name="literal_ok",
            ),
            include_trace=True,
        ),
        model_manifest=_build_manifest(
            model_id="llama-3.2-3b-instruct",
            architecture_family="llama",
        ),
    )

    assert result.trace.selected_template == "structured_output"
    assert result.trace.output_contract.format == "grammar"
    assert result.trace.output_contract.syntax == "ebnf"
    assert result.trace.output_contract.grammar == 'root ::= "ok"'
    assert [override.source for override in result.trace.overrides] == ["response_format"]


def test_prompt_compiler_loads_response_format_file(
    temp_settings,
    sample_prompt_assets,
) -> None:
    compiler = PromptCompiler(temp_settings)

    result = compiler.compile(
        messages=[GenerateMessage(role="user", content="Return a status object")],
        request=PromptCompilationRequest(
            actor="cli",
            response_format_path=str(sample_prompt_assets["response_format"]),
            include_trace=True,
        ),
        model_manifest=_build_manifest(
            model_id="llama-3.2-3b-instruct",
            architecture_family="llama",
        ),
    )

    assert result.trace.selected_template == "structured_output"
    assert result.trace.output_contract.format == "json_schema"
    assert result.trace.output_contract.name == "milestone_summary"
    assert [override.source for override in result.trace.overrides] == ["response_format_file"]


def test_prompt_compiler_rejects_out_of_scope_pretext_file(temp_settings, tmp_path: Path) -> None:
    compiler = PromptCompiler(temp_settings)
    outside_path = tmp_path / "outside-pretext.txt"
    outside_path.write_text("outside scope", encoding="utf-8")

    with pytest.raises(FileAccessError):
        compiler.compile(
            messages=[GenerateMessage(role="user", content="Check scope")],
            request=PromptCompilationRequest(
                actor="api",
                pretext_path=str(outside_path),
            ),
        )


def test_prompt_compiler_rejects_out_of_scope_mcp_tools_file(temp_settings, tmp_path: Path) -> None:
    compiler = PromptCompiler(temp_settings)
    outside_path = tmp_path / "outside-mcp-tools.json"
    outside_path.write_text(
        '[{"name":"escape_scope","description":"outside","server":"rogue"}]',
        encoding="utf-8",
    )

    with pytest.raises(FileAccessError):
        compiler.compile(
            messages=[GenerateMessage(role="user", content="Check MCP scope")],
            request=PromptCompilationRequest(
                actor="api",
                mcp_tools_path=str(outside_path),
            ),
        )
