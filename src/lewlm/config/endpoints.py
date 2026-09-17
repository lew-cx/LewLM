"""Typed, local-only external engine configuration; no network or engine imports."""

from __future__ import annotations

import hashlib
from typing import Literal, Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
ExternalProfile = Literal[
    "openai_compatible", "vmlx", "omlx", "vllm_mlx", "vllm_local",
    "sglang_local", "tensorrt_llm_server", "openvino_model_server",
    "ollama_local", "llamacpp_server", "exllamav3_tabby",
]


def server_root(url: str) -> str:
    """Accept the documented root and /v1 spellings, with one canonical identity."""
    root = url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


class ExternalEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
    profile: ExternalProfile = "openai_compatible"
    enabled: bool = True
    base_url: str
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    connect_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    read_timeout_seconds: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    pool_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOOPBACK_HOSTS:
            raise ValueError("External endpoint base_url must use http or https on a loopback-only local host.")
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("External endpoint URLs cannot contain credentials, query parameters, or fragments; use api_key_env.")
        if parsed.path.rstrip("/") not in {"", "/v1"}:
            raise ValueError("External endpoint base_url must be a server root or end in /v1.")
        if parsed.port is not None and parsed.port < 1:
            raise ValueError("External endpoint port must be between 1 and 65535.")
        return self

    @property
    def cache_namespace(self) -> str:
        # A changed destination/profile must never consume an old server's cache.
        identity = f"{self.endpoint_id}|{self.profile}|{server_root(self.base_url)}"
        return "external:" + hashlib.sha256(identity.encode()).hexdigest()
