from __future__ import annotations

import pytest

from lewlm.core.errors import NotImplementedLewLMError
from lewlm.runtime.introspection import invoke_with_signature


def test_invoke_with_signature_passes_selected_kwargs_to_var_keyword_callables() -> None:
    captured: dict[str, object] = {}

    def callable_obj(*, model, prompt, **kwargs):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "ok"

    result = invoke_with_signature(
        callable_obj,
        {
            "model": "test-model",
            "prompt": "hello",
            "max_tokens": 64,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": "hello"}],
        },
        capability="chat",
        passthrough_keys=("max_tokens", "temperature"),
    )

    assert result == "ok"
    assert captured == {
        "model": "test-model",
        "prompt": "hello",
        "kwargs": {"max_tokens": 64, "temperature": 0.2},
    }


def test_invoke_with_signature_rejects_unknown_required_parameters() -> None:
    def callable_obj(*, path_or_hf_repo, revision):
        return path_or_hf_repo, revision

    with pytest.raises(NotImplementedLewLMError) as exc_info:
        invoke_with_signature(
            callable_obj,
            {"path_or_hf_repo": "/tmp/model"},
            capability="model_load",
        )

    assert exc_info.value.details == {
        "capability": "model_load",
        "unsupported_required_parameters": ["revision"],
    }
