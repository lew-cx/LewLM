from __future__ import annotations

from pathlib import Path

from lewlm.config.settings import LewLMSettings


def test_settings_prepare_default_directories(tmp_path: Path) -> None:
    settings = LewLMSettings(data_dir=tmp_path / "state")

    settings.prepare_directories()

    assert settings.models_dir == (settings.data_dir / "models",)
    assert settings.database_path == settings.data_dir / "metadata.sqlite3"
    assert settings.logs_dir.exists()
    assert settings.cache_dir.exists()
    assert settings.temp_dir.exists()


def test_redacted_snapshot_omits_api_key_values(temp_settings: LewLMSettings) -> None:
    snapshot = temp_settings.redacted_snapshot()

    assert snapshot["api_key_count"] == 1
    assert snapshot["api_key_required"] is False
    assert snapshot["runtime_policy"] == "balanced"
    assert snapshot["rate_limit_requests"] == 120
    assert snapshot["rate_limit_window_seconds"] == 60
    assert snapshot["audit_log_enabled"] is False
    assert snapshot["persistence_encryption_enabled"] is False
    assert snapshot["persistence_encryption_kdf_iterations"] == 600_000
    assert snapshot["tool_authorization_required"] is False
    assert snapshot["parser_sandbox_enabled"] is True
    assert snapshot["parser_sandbox_timeout_seconds"] == 30
    assert snapshot["parser_sandbox_clear_environment"] is True
    assert snapshot["parser_sandbox_dir"].endswith("parser-sandbox")
    assert snapshot["tool_sandbox_enabled"] is True
    assert snapshot["tool_sandbox_timeout_seconds"] == 120
    assert snapshot["tool_sandbox_clear_environment"] is True
    assert snapshot["tool_sandbox_dir"].endswith("tool-sandbox")
    assert snapshot["conversion_sandbox_enabled"] is True
    assert snapshot["conversion_sandbox_timeout_seconds"] == 1800
    assert snapshot["conversion_sandbox_clear_environment"] is True
    assert snapshot["conversion_sandbox_dir"].endswith("conversion-sandbox")
    assert snapshot["kv_cache_page_size"] == 256
    assert snapshot["kv_cache_max_pages"] == 64
    assert snapshot["kv_cache_quantization_bits"] == 8
    assert snapshot["prefill_token_batch_size"] == 512
    assert snapshot["file_access_roots"] == [str(temp_settings.data_dir)]
    assert snapshot["validation_manifest_paths"] == []
    assert snapshot["reasoning_visibility"] == "hidden"
    assert snapshot["speculative_decoding_enabled"] is False
    assert snapshot["speculative_decoding_draft_model_id"] is None
    assert snapshot["speculative_decoding_num_draft_tokens"] == 3
    assert snapshot["prompt_lookup_speculation_enabled"] is False
    assert snapshot["prompt_lookup_max_ngram_size"] == 2
    assert snapshot["prompt_lookup_num_pred_tokens"] == 10
    assert snapshot["moe_bounded_memory_mode"] == "off"
    assert snapshot["moe_resident_expert_count"] == 4
    assert "test-key" not in str(snapshot)


def test_settings_normalize_pack_configuration(tmp_path: Path) -> None:
    settings = LewLMSettings(
        data_dir=tmp_path / "state",
        runtime_packs=("MLX", "llama-cpp"),
        disabled_feature_packs=("Documents",),
    )

    assert settings.runtime_packs == ("mlx", "llamacpp")
    assert settings.disabled_feature_packs == ("documents",)
