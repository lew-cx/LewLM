"""Application settings for LewLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from lewlm._version import __version__
from lewlm.core.contracts import ReasoningVisibility
from lewlm.pack_registry import KNOWN_FEATURE_PACKS, KNOWN_RUNTIME_PACKS, canonicalize_pack_name


def _normalize_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


class LewLMSettings(BaseSettings):
    """Resolved configuration for a LewLM process."""

    model_config = SettingsConfigDict(
        env_prefix="LEWLM_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "LewLM"
    version: str = __version__
    environment: Literal["development", "test", "production"] = "development"
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    data_dir: Path = Path.home() / ".lewlm"
    models_dir: tuple[Path, ...] = ()
    runtime_packs: tuple[str, ...] = ()
    disabled_runtime_packs: tuple[str, ...] = ()
    feature_packs: tuple[str, ...] = ()
    disabled_feature_packs: tuple[str, ...] = ()
    privacy_mode: bool = False
    telemetry_enabled: bool = False
    allow_outbound_network: bool = False
    api_keys: tuple[SecretStr, ...] = ()
    api_key_required: bool = False
    request_max_bytes: int = 50 * 1024 * 1024
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    max_concurrent_runtime_requests: int = 4
    max_concurrent_model_loads: int = 1
    runtime_request_queue_limit: int = 16
    runtime_request_queue_timeout_seconds: int = 15
    continuous_batch_window_milliseconds: int = 8
    continuous_batch_max_batch_size: int = 4
    decode_priority_scheduling_enabled: bool = True
    long_prefill_token_threshold: int = 1024
    prefill_isolation_enabled: bool = False
    prefill_isolation_max_concurrent_requests: int = 1
    prefill_isolation_decode_reserve: int = 1
    cluster_role: Literal["standalone", "coordinator", "worker"] = "standalone"
    cluster_name: str = "default"
    cluster_node_name: str = "node"
    cluster_public_base_url: str | None = None
    cluster_coordinator_url: str | None = None
    cluster_enrollment_secret: SecretStr | None = None
    cluster_token_ttl_seconds: int = 900
    cluster_worker_heartbeat_timeout_seconds: int = 30
    cluster_stage_timeout_seconds: int = 15
    audit_log_enabled: bool = False
    persistence_encryption_enabled: bool = False
    persistence_encryption_passphrase: SecretStr | None = None
    persistence_encryption_kdf_iterations: int = 600_000
    tool_authorization_required: bool = False
    parser_sandbox_enabled: bool = True
    parser_sandbox_timeout_seconds: int = 30
    parser_sandbox_clear_environment: bool = True
    tool_sandbox_enabled: bool = True
    tool_sandbox_timeout_seconds: int = 120
    tool_sandbox_clear_environment: bool = True
    conversion_sandbox_enabled: bool = True
    conversion_sandbox_timeout_seconds: int = 1800
    conversion_sandbox_clear_environment: bool = True
    conversion_worker_count: int = 1
    runtime_policy: Literal["keep_warm", "balanced", "aggressive_unload"] = "balanced"
    kv_cache_page_size: int = 256
    kv_cache_max_pages: int | None = 64
    kv_cache_quantization_bits: int | None = 8
    prefill_token_batch_size: int = 512
    mlx_graph_compile_enabled: bool = False
    mlx_attention_kernel_mode: Literal["stock", "flash_attention", "custom_sdpa"] = "stock"
    reasoning_visibility: ReasoningVisibility = ReasoningVisibility.HIDDEN
    speculative_decoding_enabled: bool = False
    speculative_decoding_draft_model_id: str | None = None
    speculative_decoding_num_draft_tokens: int = 3
    prompt_lookup_speculation_enabled: bool = False
    prompt_lookup_max_ngram_size: int = 2
    prompt_lookup_num_pred_tokens: int = 10
    moe_bounded_memory_mode: Literal["off", "partial_load", "expert_streaming"] = "off"
    moe_resident_expert_count: int = 4
    external_accelerator_enabled: bool = False
    external_accelerator_profile: Literal[
        "openai_compatible",
        "vmlx",
        "omlx",
        "vllm_mlx",
        "vllm_local",
        "sglang_local",
        "ollama_local",
        "llamacpp_server",
    ] = "openai_compatible"
    external_accelerator_base_url: str | None = None
    external_accelerator_timeout_seconds: int = 10
    file_access_roots: tuple[Path, ...] = ()
    validation_manifest_paths: tuple[Path, ...] = ()

    @model_validator(mode="after")
    def normalize_paths(self) -> Self:
        self.data_dir = _normalize_path(self.data_dir)
        model_roots = self.models_dir or (self.data_dir / "models",)
        self.models_dir = tuple(_normalize_path(Path(root)) for root in model_roots)
        self.runtime_packs = _normalize_pack_names(
            self.runtime_packs,
            known_names=KNOWN_RUNTIME_PACKS,
            field_name="runtime_packs",
        )
        self.disabled_runtime_packs = _normalize_pack_names(
            self.disabled_runtime_packs,
            known_names=KNOWN_RUNTIME_PACKS,
            field_name="disabled_runtime_packs",
        )
        self.feature_packs = _normalize_pack_names(
            self.feature_packs,
            known_names=KNOWN_FEATURE_PACKS,
            field_name="feature_packs",
        )
        self.disabled_feature_packs = _normalize_pack_names(
            self.disabled_feature_packs,
            known_names=KNOWN_FEATURE_PACKS,
            field_name="disabled_feature_packs",
        )
        scoped_roots = self.file_access_roots or (self.data_dir,)
        self.file_access_roots = tuple(_normalize_path(Path(root)) for root in scoped_roots)
        self.validation_manifest_paths = tuple(
            _normalize_path(Path(path))
            for path in self.validation_manifest_paths
        )
        if set(self.runtime_packs) & set(self.disabled_runtime_packs):
            raise ValueError("runtime_packs and disabled_runtime_packs cannot contain the same pack.")
        if set(self.feature_packs) & set(self.disabled_feature_packs):
            raise ValueError("feature_packs and disabled_feature_packs cannot contain the same pack.")
        if self.api_key_required and not self.api_keys:
            raise ValueError("api_key_required cannot be enabled without at least one api key.")
        if self.persistence_encryption_enabled and self.persistence_encryption_passphrase is None:
            raise ValueError("persistence_encryption_enabled requires persistence_encryption_passphrase.")
        if self.kv_cache_page_size < 1:
            raise ValueError("kv_cache_page_size must be at least 1.")
        if self.kv_cache_max_pages is not None and self.kv_cache_max_pages < 1:
            raise ValueError("kv_cache_max_pages must be at least 1 when set.")
        if self.kv_cache_quantization_bits is not None and self.kv_cache_quantization_bits < 1:
            raise ValueError("kv_cache_quantization_bits must be at least 1 when set.")
        if self.prefill_token_batch_size < 1:
            raise ValueError("prefill_token_batch_size must be at least 1.")
        if self.continuous_batch_window_milliseconds < 1:
            raise ValueError("continuous_batch_window_milliseconds must be at least 1.")
        if self.continuous_batch_max_batch_size < 1:
            raise ValueError("continuous_batch_max_batch_size must be at least 1.")
        if self.long_prefill_token_threshold < 1:
            raise ValueError("long_prefill_token_threshold must be at least 1.")
        if self.prefill_isolation_max_concurrent_requests < 1:
            raise ValueError("prefill_isolation_max_concurrent_requests must be at least 1.")
        if self.prefill_isolation_decode_reserve < 0:
            raise ValueError("prefill_isolation_decode_reserve must be at least 0.")
        if self.cluster_token_ttl_seconds < 1:
            raise ValueError("cluster_token_ttl_seconds must be at least 1.")
        if self.cluster_worker_heartbeat_timeout_seconds < 1:
            raise ValueError("cluster_worker_heartbeat_timeout_seconds must be at least 1.")
        if self.cluster_stage_timeout_seconds < 1:
            raise ValueError("cluster_stage_timeout_seconds must be at least 1.")
        if self.speculative_decoding_num_draft_tokens < 1:
            raise ValueError("speculative_decoding_num_draft_tokens must be at least 1.")
        if self.prompt_lookup_max_ngram_size < 1:
            raise ValueError("prompt_lookup_max_ngram_size must be at least 1.")
        if self.prompt_lookup_num_pred_tokens < 1:
            raise ValueError("prompt_lookup_num_pred_tokens must be at least 1.")
        if self.moe_resident_expert_count < 1:
            raise ValueError("moe_resident_expert_count must be at least 1.")
        if self.external_accelerator_timeout_seconds < 1:
            raise ValueError("external_accelerator_timeout_seconds must be at least 1.")
        return self

    @computed_field(return_type=Path)
    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @computed_field(return_type=Path)
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @computed_field(return_type=Path)
    @property
    def benchmarks_dir(self) -> Path:
        return self.data_dir / "benchmarks"

    @computed_field(return_type=Path)
    @property
    def audit_log_path(self) -> Path:
        return self.logs_dir / "audit.jsonl"

    @computed_field(return_type=Path)
    @property
    def parser_sandbox_dir(self) -> Path:
        return self.temp_dir / "parser-sandbox"

    @computed_field(return_type=Path)
    @property
    def tool_sandbox_dir(self) -> Path:
        return self.temp_dir / "tool-sandbox"

    @computed_field(return_type=Path)
    @property
    def conversion_sandbox_dir(self) -> Path:
        return self.temp_dir / "conversion-sandbox"

    @computed_field(return_type=Path)
    @property
    def keys_dir(self) -> Path:
        return self.data_dir / "keys"

    @computed_field(return_type=Path)
    @property
    def persistence_salt_path(self) -> Path:
        return self.keys_dir / "persistence.salt"

    @computed_field(return_type=Path)
    @property
    def temp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @computed_field(return_type=Path)
    @property
    def materialized_cache_dir(self) -> Path:
        return self.temp_dir / "materialized-cache"

    @computed_field(return_type=Path)
    @property
    def database_path(self) -> Path:
        return self.data_dir / "metadata.sqlite3"

    def prepare_directories(self) -> None:
        for path in (
            self.data_dir,
            self.logs_dir,
            self.cache_dir,
            self.benchmarks_dir,
            self.temp_dir,
            self.parser_sandbox_dir,
            self.tool_sandbox_dir,
            self.conversion_sandbox_dir,
            self.keys_dir,
            self.materialized_cache_dir,
            *self.models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def with_updates(self, **updates: Any) -> "LewLMSettings":
        payload = self.model_dump(mode="python", exclude_computed_fields=True)
        payload.update(updates)
        return type(self).model_validate(payload)

    def redacted_snapshot(self) -> dict[str, Any]:
        return {
            "app_name": self.app_name,
            "version": self.version,
            "environment": self.environment,
            "host": self.host,
            "port": self.port,
            "log_level": self.log_level,
            "data_dir": str(self.data_dir),
            "models_dir": [str(path) for path in self.models_dir],
            "runtime_packs": list(self.runtime_packs),
            "disabled_runtime_packs": list(self.disabled_runtime_packs),
            "feature_packs": list(self.feature_packs),
            "disabled_feature_packs": list(self.disabled_feature_packs),
            "logs_dir": str(self.logs_dir),
            "cache_dir": str(self.cache_dir),
            "benchmarks_dir": str(self.benchmarks_dir),
            "temp_dir": str(self.temp_dir),
            "database_path": str(self.database_path),
            "privacy_mode": self.privacy_mode,
            "telemetry_enabled": self.telemetry_enabled,
            "allow_outbound_network": self.allow_outbound_network,
            "api_key_count": len(self.api_keys),
            "api_key_required": self.api_key_required,
            "request_max_bytes": self.request_max_bytes,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window_seconds": self.rate_limit_window_seconds,
            "max_concurrent_runtime_requests": self.max_concurrent_runtime_requests,
            "max_concurrent_model_loads": self.max_concurrent_model_loads,
            "runtime_request_queue_limit": self.runtime_request_queue_limit,
            "runtime_request_queue_timeout_seconds": self.runtime_request_queue_timeout_seconds,
            "continuous_batch_window_milliseconds": self.continuous_batch_window_milliseconds,
            "continuous_batch_max_batch_size": self.continuous_batch_max_batch_size,
            "decode_priority_scheduling_enabled": self.decode_priority_scheduling_enabled,
            "long_prefill_token_threshold": self.long_prefill_token_threshold,
            "prefill_isolation_enabled": self.prefill_isolation_enabled,
            "prefill_isolation_max_concurrent_requests": self.prefill_isolation_max_concurrent_requests,
            "prefill_isolation_decode_reserve": self.prefill_isolation_decode_reserve,
            "cluster_role": self.cluster_role,
            "cluster_name": self.cluster_name,
            "cluster_node_name": self.cluster_node_name,
            "cluster_public_base_url": self.cluster_public_base_url,
            "cluster_coordinator_url": self.cluster_coordinator_url,
            "cluster_token_ttl_seconds": self.cluster_token_ttl_seconds,
            "cluster_worker_heartbeat_timeout_seconds": self.cluster_worker_heartbeat_timeout_seconds,
            "cluster_stage_timeout_seconds": self.cluster_stage_timeout_seconds,
            "audit_log_enabled": self.audit_log_enabled,
            "audit_log_path": str(self.audit_log_path),
            "persistence_encryption_enabled": self.persistence_encryption_enabled,
            "persistence_encryption_kdf_iterations": self.persistence_encryption_kdf_iterations,
            "persistence_salt_path": str(self.persistence_salt_path),
            "tool_authorization_required": self.tool_authorization_required,
            "parser_sandbox_enabled": self.parser_sandbox_enabled,
            "parser_sandbox_timeout_seconds": self.parser_sandbox_timeout_seconds,
            "parser_sandbox_clear_environment": self.parser_sandbox_clear_environment,
            "parser_sandbox_dir": str(self.parser_sandbox_dir),
            "tool_sandbox_enabled": self.tool_sandbox_enabled,
            "tool_sandbox_timeout_seconds": self.tool_sandbox_timeout_seconds,
            "tool_sandbox_clear_environment": self.tool_sandbox_clear_environment,
            "tool_sandbox_dir": str(self.tool_sandbox_dir),
            "conversion_sandbox_enabled": self.conversion_sandbox_enabled,
            "conversion_sandbox_timeout_seconds": self.conversion_sandbox_timeout_seconds,
            "conversion_sandbox_clear_environment": self.conversion_sandbox_clear_environment,
            "conversion_sandbox_dir": str(self.conversion_sandbox_dir),
            "conversion_worker_count": self.conversion_worker_count,
            "runtime_policy": self.runtime_policy,
            "kv_cache_page_size": self.kv_cache_page_size,
            "kv_cache_max_pages": self.kv_cache_max_pages,
            "kv_cache_quantization_bits": self.kv_cache_quantization_bits,
            "prefill_token_batch_size": self.prefill_token_batch_size,
            "mlx_graph_compile_enabled": self.mlx_graph_compile_enabled,
            "mlx_attention_kernel_mode": self.mlx_attention_kernel_mode,
            "reasoning_visibility": self.reasoning_visibility.value,
            "speculative_decoding_enabled": self.speculative_decoding_enabled,
            "speculative_decoding_draft_model_id": self.speculative_decoding_draft_model_id,
            "speculative_decoding_num_draft_tokens": self.speculative_decoding_num_draft_tokens,
            "prompt_lookup_speculation_enabled": self.prompt_lookup_speculation_enabled,
            "prompt_lookup_max_ngram_size": self.prompt_lookup_max_ngram_size,
            "prompt_lookup_num_pred_tokens": self.prompt_lookup_num_pred_tokens,
            "moe_bounded_memory_mode": self.moe_bounded_memory_mode,
            "moe_resident_expert_count": self.moe_resident_expert_count,
            "external_accelerator_enabled": self.external_accelerator_enabled,
            "external_accelerator_profile": self.external_accelerator_profile,
            "external_accelerator_base_url": self.external_accelerator_base_url,
            "external_accelerator_timeout_seconds": self.external_accelerator_timeout_seconds,
            "file_access_roots": [str(path) for path in self.file_access_roots],
            "validation_manifest_paths": [str(path) for path in self.validation_manifest_paths],
        }


_SETTINGS_CACHE: LewLMSettings | None = None


def _normalize_pack_names(
    values: tuple[str, ...] | list[str] | str,
    *,
    known_names: frozenset[str],
    field_name: str,
) -> tuple[str, ...]:
    raw_values = (values,) if isinstance(values, str) else tuple(values)
    normalized: list[str] = []
    for value in raw_values:
        pack_name = canonicalize_pack_name(str(value))
        if pack_name not in known_names:
            known = ", ".join(sorted(known_names))
            raise ValueError(f"{field_name} contains unknown pack `{value}`. Known packs: {known}.")
        if pack_name not in normalized:
            normalized.append(pack_name)
    return tuple(normalized)


def get_settings() -> LewLMSettings:
    """Return a cached settings object for the current process."""

    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None:
        _SETTINGS_CACHE = LewLMSettings()
    return _SETTINGS_CACHE


def reset_settings_cache() -> None:
    """Clear the cached settings object. Used by tests."""

    global _SETTINGS_CACHE
    _SETTINGS_CACHE = None
