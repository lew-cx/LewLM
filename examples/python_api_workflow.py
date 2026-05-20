from __future__ import annotations

from pathlib import Path

from lewlm import LewLM, LewLMSettings
from lewlm.conversion.models import ConversionPolicy, JobStatus
from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat, DocumentSection, ParagraphBlock
from lewlm.tools.models import DocumentGenerateToolRequest, GenerateDocumentToolInput


def _feature_path_summary(items: list[dict[str, object]]) -> dict[str, str]:
    return {
        str(item["feature_class"]): f'{item["label"]} [{item["support_path"]}]'
        for item in items
    }


def main() -> None:
    data_dir = Path.home() / ".lewlm"
    output_dir = Path("out")
    repo_models = Path(__file__).resolve().parents[1] / "src" / "lewlm" / "models"
    models_dir = repo_models if repo_models.exists() else data_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = LewLMSettings(
        data_dir=data_dir,
        models_dir=(models_dir,),
    )

    with LewLM(settings) as lewlm:
        health = lewlm.health()
        recommended_feature_paths = _feature_path_summary(health["install_profiles"]["recommended_feature_paths"])
        print(f"scanning models from {models_dir}")
        print(f"recommended feature paths: {recommended_feature_paths}")
        scan_summary = lewlm.scan_models()
        print(f"discovered {scan_summary.discovered_count} model(s)")
        print(f"built-in skills: {len(lewlm.list_skills())}")
        print(f"local tools: {len(lewlm.list_tools())}")

        inventory = lewlm.inventory()
        conversion_candidate = next(
            (manifest for manifest in inventory.items if manifest.conversion_status.value == "requires_conversion"),
            None,
        )
        if conversion_candidate is not None:
            job = lewlm.submit_conversion(
                model_id=conversion_candidate.model_id,
                policy=ConversionPolicy.BALANCED,
            )
            final_job = lewlm.wait_for_job(job.job_id, timeout_seconds=120.0)
            print(f"conversion {final_job.job_id}: {final_job.status.value}")
            if final_job.status == JobStatus.FAILED:
                raise RuntimeError(f"conversion failed for {conversion_candidate.model_id}: {final_job.payload}")

        runnable_model = next(
            (manifest for manifest in lewlm.list_models() if manifest.conversion_status.value == "runnable"),
            None,
        )
        if runnable_model is None:
            raise RuntimeError("No runnable model is available. Scan or convert a compatible local model first.")

        chat = lewlm.chat_sync(
            prompt="Summarize the LewLM package surface in one sentence.",
            model_id=runnable_model.model_id,
        )
        print(chat.response.output_text)

        document = DocumentIR(
            title="LewLM Package Workflow",
            sections=[
                DocumentSection(
                    heading="Summary",
                    blocks=[
                        ParagraphBlock(text="This report was rendered through the embeddable LewLM Python API."),
                    ],
                ),
            ],
        )
        artifact = lewlm.generate_document(
            document,
            output_format=DocumentOutputFormat.MARKDOWN,
            file_name="python-api-workflow.md",
        )
        (output_dir / artifact.file_name).write_bytes(artifact.content)

        tool_result = lewlm.execute_tool(
            DocumentGenerateToolRequest(
                input=GenerateDocumentToolInput(
                    output_format=DocumentOutputFormat.JSON,
                    file_name="python-api-workflow.json",
                    document=document,
                ),
            ),
        )
        (output_dir / "python-api-workflow-tool.json").write_text(
            tool_result.model_dump_json(indent=2),
            encoding="utf-8",
        )

        session = lewlm.create_session(
            title="Python API Workflow",
            context_policy="summary_and_last_turn",
        )
        session_bundle = lewlm.export_session(session.session_id)
        (output_dir / "python-api-session.json").write_text(
            session_bundle.model_dump_json(indent=2),
            encoding="utf-8",
        )
        lewlm.delete_session(session.session_id)

        runtime_stats = lewlm.runtime_stats_sync()
        print(f"runtime policy: {runtime_stats.runtime_policy}")
        print(f"validation manifests: {runtime_stats.validation_manifest_count}")


if __name__ == "__main__":
    main()
