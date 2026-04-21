from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from lewlm.api.app import create_app


EXPECTED_BUNDLED_MODELS = {
    "Gemma-4-26B-A4B-it": {
        "display_name": "Gemma-4-26B-A4B-it",
        "format_type": "huggingface",
        "modality": {"text", "vision", "multimodal"},
        "runtime_affinity": ["conversion", "mlx_vision"],
        "conversion_status": "requires_conversion",
        "context_length": 262144,
        "quantization": None,
    },
    "Gemma-4-31B-it": {
        "display_name": "Gemma-4-31B-it",
        "format_type": "huggingface",
        "modality": {"text", "vision", "multimodal"},
        "runtime_affinity": ["conversion", "mlx_vision"],
        "conversion_status": "requires_conversion",
        "context_length": 262144,
        "quantization": None,
    },
    "Gemma-4-31B-JANG_4M": {
        "display_name": "Gemma-4-31B-JANG_4M",
        "format_type": "mlx",
        "modality": {"text", "vision", "multimodal"},
        "runtime_affinity": ["mlx_vision"],
        "conversion_status": "runnable",
        "context_length": 262144,
        "quantization": "int4",
    },
    "Gemma-4-E2B-Hauhau": {
        "display_name": "Gemma-4-E2B-hauhau_agg.Q8_K_P",
        "format_type": "gguf",
        "modality": {"text"},
        "runtime_affinity": ["llamacpp"],
        "conversion_status": "runnable",
        "context_length": None,
        "quantization": "q8_k_p",
    },
    "Gemma-4-E4B-Hauhau": {
        "display_name": "Gemma-4-E4B-hauhau_agg.Q8_K_P",
        "format_type": "gguf",
        "modality": {"text"},
        "runtime_affinity": ["llamacpp"],
        "conversion_status": "runnable",
        "context_length": None,
        "quantization": "q8_k_p",
    },
}


def _assert_manifest_matches(manifest: dict[str, object], expected: dict[str, object]) -> None:
    assert manifest["display_name"] == expected["display_name"]
    assert manifest["format_type"] == expected["format_type"]
    assert set(manifest["modality"]) == expected["modality"]
    assert manifest["runtime_affinity"] == expected["runtime_affinity"]
    assert manifest["conversion_status"] == expected["conversion_status"]
    assert manifest["context_length"] == expected["context_length"]
    assert manifest["quantization"] == expected["quantization"]


def _wait_for_completed_job(client: TestClient, job_id: str) -> dict[str, object]:
    job_payload: dict[str, object] | None = None
    for _ in range(40):
        job_response = client.get(f"/v1/jobs/{job_id}")
        job_payload = job_response.json()
        if job_payload["status"] == "completed":
            break
        time.sleep(0.05)
    assert job_payload is not None
    assert job_payload["status"] == "completed"
    return job_payload


def test_end_to_end_external_models_scan_detects_external_models_folder_by_folder(
    external_models_settings,
    external_models_root: Path,
) -> None:
    app = create_app(external_models_settings)
    with TestClient(app) as client:
        root_response = client.post("/v1/models/scan", json={})
        assert root_response.status_code == 200
        root_payload = root_response.json()

        assert root_payload["discovered_count"] == len(EXPECTED_BUNDLED_MODELS)
        assert root_payload["new_count"] == len(EXPECTED_BUNDLED_MODELS)

        manifests_by_name = {
            manifest["display_name"]: manifest
            for manifest in root_payload["manifests"]
        }
        assert set(manifests_by_name) == {
            expected["display_name"]
            for expected in EXPECTED_BUNDLED_MODELS.values()
        }
        for expected in EXPECTED_BUNDLED_MODELS.values():
            _assert_manifest_matches(manifests_by_name[expected["display_name"]], expected)

        for folder_name, expected in EXPECTED_BUNDLED_MODELS.items():
            response = client.post(
                "/v1/models/scan",
                json={"paths": [str(external_models_root / folder_name)]},
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["discovered_count"] == 1
            _assert_manifest_matches(payload["manifests"][0], expected)


def test_end_to_end_external_models_exercise_all_expected_model_folders(
    app_with_external_models_runtime_and_conversion,
    external_models_root: Path,
) -> None:
    with TestClient(app_with_external_models_runtime_and_conversion) as client:
        warmed_model_ids: list[str] = []
        converted_result_paths: list[Path] = []

        for folder_name, expected in EXPECTED_BUNDLED_MODELS.items():
            scan_response = client.post(
                "/v1/models/scan",
                json={"paths": [str(external_models_root / folder_name)]},
            )
            assert scan_response.status_code == 200
            manifest = scan_response.json()["manifests"][0]
            _assert_manifest_matches(manifest, expected)

            capabilities_response = client.get(f"/v1/models/{manifest['model_id']}/capabilities")
            assert capabilities_response.status_code == 200
            capabilities = capabilities_response.json()["capabilities"]
            assert capabilities

            if expected["conversion_status"] == "runnable":
                assert any(item["capability"] == "chat" and item["supported"] for item in capabilities)

                warm_response = client.post(f"/v1/models/{manifest['model_id']}/warm")
                assert warm_response.status_code == 200
                assert warm_response.json()["status"] == "warmed"
                warm_runtime = warm_response.json()["runtime"]
                if expected["runtime_affinity"] == ["mlx_vision"] and "multimodal" in expected["modality"]:
                    assert warm_runtime == "fake_mlx_semantic"

                prompt = f"Confirm bundled fallback coverage for {folder_name}."
                chat_response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": manifest["model_id"],
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                assert chat_response.status_code == 200
                assert chat_response.json()["model"] == manifest["model_id"]
                expected_response_prefix = "Vision echo: " if warm_runtime == "fake_mlx_vision" else "Echo: "
                assert chat_response.json()["choices"][0]["message"]["content"] == f"{expected_response_prefix}{prompt}"

                unload_response = client.post(f"/v1/models/{manifest['model_id']}/unload")
                assert unload_response.status_code == 200
                assert unload_response.json()["status"] == "unloaded"
                warmed_model_ids.append(manifest["model_id"])
                continue

            assert all(not item["supported"] for item in capabilities)

            convert_response = client.post(
                "/v1/models/convert",
                json={"model_id": manifest["model_id"], "policy": "balanced"},
            )
            assert convert_response.status_code == 200
            job_payload = _wait_for_completed_job(client, convert_response.json()["job_id"])

            result_path = Path(job_payload["payload"]["result_path"])
            assert result_path.exists()
            converted_result_paths.append(result_path)

            converted_scan = client.post(
                "/v1/models/scan",
                json={"paths": [str(result_path)]},
            )
            assert converted_scan.status_code == 200
            converted_manifests = converted_scan.json()["manifests"]
            assert len(converted_manifests) == 2
            multimodal_manifest = next(
                item for item in converted_manifests if item["artifact_role"] == "multimodal_runnable"
            )
            text_manifest = next(
                item for item in converted_manifests if item["artifact_role"] == "text_runnable"
            )
            assert multimodal_manifest["format_type"] == "mlx"
            assert set(manifest["modality"]).issubset(set(multimodal_manifest["modality"]))
            assert multimodal_manifest["conversion_status"] == "runnable"
            assert multimodal_manifest["runtime_affinity"] == ["mlx_vision"]
            assert text_manifest["format_type"] == "mlx"
            assert set(text_manifest["modality"]) == {"text"}
            assert text_manifest["runtime_affinity"] == ["mlx_text"]
            assert text_manifest["artifact_family_id"] == multimodal_manifest["artifact_family_id"]

    assert len(warmed_model_ids) == 3
    assert len(converted_result_paths) == 2


def test_end_to_end_external_multimodal_model_converts_and_handles_multimodal_io(
    app_with_external_models_runtime_and_conversion,
    sample_attachment_sources,
) -> None:
    with TestClient(app_with_external_models_runtime_and_conversion) as client:
        first_scan = client.post("/v1/models/scan", json={})
        assert first_scan.status_code == 200
        first_payload = first_scan.json()
        assert first_payload["discovered_count"] == 8

        source_model_id = next(
            manifest["model_id"]
            for manifest in first_payload["manifests"]
            if manifest["display_name"] == "Gemma-4-26B-A4B-it"
        )
        convert_response = client.post(
            "/v1/models/convert",
            json={"model_id": source_model_id, "policy": "balanced"},
        )
        assert convert_response.status_code == 200
        job_payload = _wait_for_completed_job(client, convert_response.json()["job_id"])

        result_path = Path(job_payload["payload"]["result_path"])
        compatibility = job_payload["payload"]["compatibility"]
        assert compatibility["can_convert"] is True
        assert compatibility["quantization_mode"] == "4bit"
        assert result_path.exists()
        assert {item["role"] for item in job_payload["payload"]["artifacts"]} == {
            "multimodal_runnable",
            "text_runnable",
        }

        second_scan = client.post("/v1/models/scan", json={})
        assert second_scan.status_code == 200
        second_payload = second_scan.json()
        assert second_payload["discovered_count"] == 10

        converted_multimodal_manifest = next(
            manifest
            for manifest in second_payload["manifests"]
            if manifest["artifact_role"] == "multimodal_runnable"
            and Path(manifest["source_path"]).is_relative_to(result_path)
        )
        converted_text_manifest = next(
            manifest
            for manifest in second_payload["manifests"]
            if manifest["artifact_role"] == "text_runnable"
            and Path(manifest["source_path"]).is_relative_to(result_path)
        )
        assert converted_multimodal_manifest["format_type"] == "mlx"
        assert set(converted_multimodal_manifest["modality"]) == {"text", "vision", "multimodal"}
        assert converted_multimodal_manifest["runtime_affinity"] == ["mlx_vision"]
        assert converted_multimodal_manifest["conversion_status"] == "runnable"
        assert converted_multimodal_manifest["context_length"] == 262144
        assert converted_text_manifest["format_type"] == "mlx"
        assert set(converted_text_manifest["modality"]) == {"text"}
        assert converted_text_manifest["runtime_affinity"] == ["mlx_text"]
        assert converted_text_manifest["artifact_family_id"] == converted_multimodal_manifest["artifact_family_id"]

        chat_response = client.post(
            "/v1/chat/completions",
            json={
                "model": converted_multimodal_manifest["model_id"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Review these converted model inputs."},
                            {"type": "input_image", "path": str(sample_attachment_sources["image_one"])},
                            {"type": "input_file", "path": str(sample_attachment_sources["docx"])},
                            {"type": "input_audio", "path": str(sample_attachment_sources["audio"])},
                        ],
                    },
                ],
            },
        )

    assert chat_response.status_code == 200
    assert chat_response.json()["model"] == converted_multimodal_manifest["model_id"]
    output_text = chat_response.json()["choices"][0]["message"]["content"]
    assert "[Attached image: receipt-front.png]" in output_text
    assert "[Attached document: sample.docx]" in output_text
    assert "Transcribed voice-note.wav" in output_text
    assert "[images: receipt-front.png]" in output_text


def test_end_to_end_real_external_gguf_prompt_smoke_is_opt_in(
    external_models_settings,
    external_models_root: Path,
) -> None:
    if os.environ.get("LEWLM_RUN_REAL_MODEL_SMOKE") != "1":
        pytest.skip("Set LEWLM_RUN_REAL_MODEL_SMOKE=1 to exercise the real Gemma-4-E2B GGUF prompt path.")
    pytest.importorskip("llama_cpp")

    app = create_app(external_models_settings)
    with TestClient(app) as client:
        scan_response = client.post(
            "/v1/models/scan",
            json={"paths": [str(external_models_root / "Gemma-4-E2B-Hauhau")]},
        )
        assert scan_response.status_code == 200
        model_id = scan_response.json()["manifests"][0]["model_id"]

        chat_response = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Reply with the word ready."}],
                "max_tokens": 32,
            },
        )

    assert chat_response.status_code == 200
    assert chat_response.json()["choices"][0]["message"]["content"].strip()
