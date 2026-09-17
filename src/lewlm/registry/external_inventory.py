"""Publish the models named external endpoints advertise as LewLM manifests.

Like Ollama, an external endpoint is a model *source*: an operator-managed
server that LewLM neither installs nor supervises. `lewlm scan` asks each
enabled named endpoint what it serves (`GET /v1/models`, through the same
runtime and transport that will later execute requests, so credentials,
timeouts, and loopback rules are shared) and registers one manifest per
advertised model.

Identity is endpoint-qualified on purpose. Two endpoints advertising the same
upstream name produce two manifests with different LewLM ids and different
`external://<endpoint_id>/<upstream_id>` source URIs. Nothing here claims they
are the same artifact.

What the endpoint does not say, the manifest does not invent: the weight
format is `unknown` unless the record carries evidence, modality is text until
a capability probe proves otherwise, and execution locality is
`loopback_unverified` because a loopback URL proves only the first hop.

Ollama endpoints (`ollama_local` profile) are deliberately left to
`ollama_inventory`, which reads the richer native `/api/tags` and applies the
cloud opt-in; reading them here as well would double-register the models and
lose the locality classification.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
from typing import Any

from lewlm.config.endpoints import ExternalEndpoint, server_root
from lewlm.core.contracts import (
    ArchitectureSubtype,
    ConversionStatus,
    ModelFormat,
    ModelManifest,
    ModelModality,
    ModelValidationResult,
    RuntimeAffinity,
    ValidationState,
)
from lewlm.core.errors import RuntimeUnavailableError
from lewlm.utils.model_identity import EXTERNAL_SOURCE_SCHEME, external_model_id, external_source_path

EXECUTION_LOCALITY_METADATA_KEY = "execution_locality"
LOOPBACK_UNVERIFIED = "loopback_unverified"

# Profiles whose servers can only serve GGUF artifacts. That is evidence by
# construction, not a guess from the profile label.
_GGUF_ONLY_PROFILES = frozenset({"llamacpp_server", "ollama_local"})
_FORMAT_HINTS: dict[str, ModelFormat] = {
    "gguf": ModelFormat.GGUF,
    "exl3": ModelFormat.EXL3,
    "huggingface": ModelFormat.HUGGINGFACE,
    "safetensors": ModelFormat.HUGGINGFACE,
    "mlx": ModelFormat.MLX,
}


@dataclass(frozen=True)
class ExternalInventoryResult:
    """What one endpoint read produced, including why it produced nothing."""

    endpoint_id: str
    endpoint: str
    profile: str
    manifests: list[ModelManifest] = field(default_factory=list)
    # Set when the endpoint could not be read. A failed read is not evidence
    # that its models are gone, so callers must not prune on it.
    error: str | None = None
    # The runtime still holds a list from an earlier successful read.
    stale: bool = False

    @property
    def succeeded(self) -> bool:
        return self.error is None


def discover_external_models(
    endpoint_runtimes: Mapping[str, Any],
    *,
    max_concurrency: int = 4,
    refresh: bool = True,
) -> list[ExternalInventoryResult]:
    """Read every enabled non-Ollama endpoint and build manifests for its models.

    Never raises. Endpoints are read concurrently, bounded by
    ``max_concurrency``; each failure is confined to its own result.
    """

    targets = [
        (endpoint_id, runtime)
        for endpoint_id, runtime in endpoint_runtimes.items()
        if _inventoried_endpoint(runtime) is not None
    ]
    if not targets:
        return []
    workers = max(1, min(max_concurrency, len(targets)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lewlm-inventory") as pool:
        return list(pool.map(lambda item: _read_endpoint(item[0], item[1], refresh=refresh), targets))


def _inventoried_endpoint(runtime: Any) -> ExternalEndpoint | None:
    endpoint = getattr(runtime, "endpoint", None)
    if not isinstance(endpoint, ExternalEndpoint) or not endpoint.enabled:
        return None
    if endpoint.profile == "ollama_local":
        return None
    if not callable(getattr(runtime, "advertised_model_records", None)):
        return None
    return endpoint


def _read_endpoint(endpoint_id: str, runtime: Any, *, refresh: bool) -> ExternalInventoryResult:
    endpoint = _inventoried_endpoint(runtime)
    assert endpoint is not None  # filtered by the caller
    root = server_root(endpoint.base_url)
    try:
        records = runtime.advertised_model_records(refresh=refresh)
    except RuntimeUnavailableError as exc:
        return ExternalInventoryResult(
            endpoint_id=endpoint_id,
            endpoint=root,
            profile=endpoint.profile,
            error=str(exc),
            stale=getattr(runtime, "inventory_state", None) == "stale",
        )
    manifests: list[ModelManifest] = []
    seen: set[str] = set()
    for record in records:
        upstream_id = _record_id(record)
        if not upstream_id or upstream_id in seen:
            continue
        seen.add(upstream_id)
        manifests.append(build_external_manifest(record, endpoint=endpoint, upstream_id=upstream_id))
    return ExternalInventoryResult(endpoint_id=endpoint_id, endpoint=root, profile=endpoint.profile, manifests=manifests)


def build_external_manifest(record: dict[str, Any], *, endpoint: ExternalEndpoint, upstream_id: str) -> ModelManifest:
    root = server_root(endpoint.base_url)
    format_type = _format_evidence(record, profile=endpoint.profile)
    context_length = _positive_int(record.get("max_model_len")) or _positive_int(record.get("context_length"))
    # Identity, not an artifact hash: `/v1/models` does not expose weights.
    fingerprint = hashlib.sha256(f"{endpoint.endpoint_id}\x00{upstream_id}".encode("utf-8")).hexdigest()
    return ModelManifest(
        model_id=external_model_id(endpoint.endpoint_id, upstream_id),
        display_name=upstream_id,
        architecture_family=_architecture_family(record),
        architecture_subtype=ArchitectureSubtype.UNKNOWN,
        modality=(ModelModality.TEXT,),
        source_path=external_source_path(endpoint.endpoint_id, upstream_id),
        format_type=format_type,
        quantization=_string_or_none(record.get("quantization")),
        # Only the endpoint's own runtime can serve this; no packaged runtime
        # may ever be handed an `external://` source.
        runtime_affinity=(RuntimeAffinity.EXTERNAL_ACCELERATOR,),
        context_length=context_length,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint=fingerprint,
        last_validation_result=ModelValidationResult(
            status=ValidationState.VALID,
            message=f"Advertised by endpoint `{endpoint.endpoint_id}` ({endpoint.profile}) at {root}.",
            details={
                "source_kind": EXTERNAL_SOURCE_SCHEME,
                "fingerprint_kind": "endpoint_identity",
                "format_evidence": "record" if format_type is not ModelFormat.UNKNOWN else "none",
            },
        ),
        metadata={
            "source_kind": EXTERNAL_SOURCE_SCHEME,
            "external_endpoint_id": endpoint.endpoint_id,
            "external_profile": endpoint.profile,
            "external_endpoint_url": root,
            # Exact upstream id, separate from the LewLM id and the encoded URI.
            "external_adapter_model_id": upstream_id,
            "external_upstream_model_id": upstream_id,
            "external_owned_by": _string_or_none(record.get("owned_by")),
            "external_record_created": _positive_int(record.get("created")),
            "external_record_root": _string_or_none(record.get("root")),
            EXECUTION_LOCALITY_METADATA_KEY: LOOPBACK_UNVERIFIED,
            "runtime_ownership_note": (
                "Executed by an operator-managed server behind a loopback endpoint. LewLM discovers and forwards; "
                "it does not install, supervise, or own this runtime, and a loopback URL does not prove local execution."
            ),
        },
    )


def _record_id(record: dict[str, Any]) -> str:
    value = record.get("id")
    return value.strip() if isinstance(value, str) else ""


def _string_or_none(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _architecture_family(record: dict[str, Any]) -> str:
    for container in (record, record.get("metadata")):
        if isinstance(container, dict):
            for key in ("architecture", "family", "model_type"):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return "unknown"


def _format_evidence(record: dict[str, Any], *, profile: str) -> ModelFormat:
    """Weight format only when something actually says so.

    Evidence, in order: an explicit ``format`` field on the record or its
    ``metadata``; a ``.gguf`` artifact name in ``id``/``root``; a server that
    can serve nothing but GGUF. Otherwise ``unknown`` -- an OpenAI-compatible
    listing does not reveal the weights, and guessing MLX or GGUF would send
    conversion and routing down the wrong path.
    """

    for container in (record, record.get("metadata")):
        if isinstance(container, dict):
            value = container.get("format")
            if isinstance(value, str):
                hinted = _FORMAT_HINTS.get(value.strip().casefold())
                if hinted is not None:
                    return hinted
    for key in ("id", "root"):
        value = record.get(key)
        if isinstance(value, str) and value.strip().casefold().endswith(".gguf"):
            return ModelFormat.GGUF
    if profile in _GGUF_ONLY_PROFILES:
        return ModelFormat.GGUF
    return ModelFormat.UNKNOWN
