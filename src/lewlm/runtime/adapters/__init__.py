"""Optional local runtime adapters for external accelerators."""

from lewlm.runtime.adapters.openai_compatible import (
    LocalOpenAICompatibleAdapterRuntime,
    summarize_feature_preservation,
)

__all__ = [
    "LocalOpenAICompatibleAdapterRuntime",
    "summarize_feature_preservation",
]
