from __future__ import annotations

import asyncio

from conftest import FakeLlamaCppRuntime, FakeMLXAudioRuntime, FakeMLXSemanticRuntime
from lewlm.core.bootstrap import bootstrap_services
from lewlm.core.contracts import GenerateMessage, ModelFormat, RuntimeAffinity


def test_chat_orchestrator_uses_prefix_cache_for_scheduler_admission(
    temp_settings,
    sample_models_root,
) -> None:
    services = bootstrap_services(
        temp_settings,
        runtime_overrides={
            RuntimeAffinity.EXPERIMENTAL: FakeLlamaCppRuntime(),
            RuntimeAffinity.LLAMACPP: FakeLlamaCppRuntime(),
            RuntimeAffinity.MLX_TEXT: FakeMLXSemanticRuntime(),
            RuntimeAffinity.MLX_AUDIO: FakeMLXAudioRuntime(),
        },
    )
    services.model_registry.scan()
    model_id = next(
        manifest.model_id
        for manifest in services.model_registry.list_manifests()
        if manifest.format_type == ModelFormat.MLX
    )
    message = "cached prefix " * 2048
    first = asyncio.run(
        services.chat_orchestrator.complete(
            model_id=model_id,
            messages=[GenerateMessage(role="user", content=message)],
            max_tokens=8,
            temperature=0.0,
        ),
    )
    second = asyncio.run(
        services.chat_orchestrator.complete(
            model_id=model_id,
            messages=[GenerateMessage(role="user", content=message)],
            max_tokens=8,
            temperature=0.0,
        ),
    )

    first_scheduling = first.request_metadata["scheduling"]
    second_scheduling = second.request_metadata["scheduling"]

    assert first_scheduling["prefill_heavy"] is True
    assert int(first_scheduling["prompt_token_estimate"]) >= services.settings.long_prefill_token_threshold
    assert first_scheduling["cached_prefix_tokens"] == 0
    assert second_scheduling["prefix_cache_candidate"] is True
    assert int(second_scheduling["cached_prefix_tokens"]) > 0
    assert int(second_scheduling["prompt_token_estimate"]) < int(first_scheduling["prompt_token_estimate"])
    assert second_scheduling["prefill_heavy"] is False
    assert int(second_scheduling["total_prompt_tokens"]) > int(second_scheduling["prompt_token_estimate"])
