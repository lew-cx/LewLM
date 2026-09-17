"""Publish a locally installed Ollama daemon's models as LewLM manifests.

Ollama is treated as a model *source*, never as a LewLM-owned runtime. LewLM
does not install, launch, configure, update, or supervise Ollama; obtaining it
is entirely the operator's business, through the desktop client or the CLI.
What LewLM does is ask a daemon the operator is already running what it holds,
and publish the answer as manifests the existing external accelerator bridge
knows how to execute.

That division is why this module lives under `registry/` rather than
`runtime/`. It produces manifests and nothing else. No runtime class is added,
no adapter is specialized, and the serving path is untouched: a manifest built
here declares `EXTERNAL_ACCELERATOR` affinity and carries the Ollama tag in
`external_adapter_model_id`, which is all the existing adapter needs to resolve
and forward the request.

Discovery is off unless `ollama_discovery_enabled` is set. While it is off,
nothing here runs and LewLM never contacts the daemon.

Manifests built here have no file behind them. `source_path` is an
`ollama://<tag>` URI rather than a filesystem path, which keeps the Ollama
namespace distinct from every model root and makes reconciliation independent
of filesystem scans. Callers that need a real file must check the scheme with
`is_ollama_source` first.

Execution locality
------------------
An Ollama daemon can serve models it runs on this host and models it relays to
Ollama's cloud. Both arrive over the same loopback endpoint and are
indistinguishable to an OpenAI-compatible client, so locality is classified
here, at the only point where the evidence exists: the native `/api/tags`
record. Cloud-backed models are left out of the registry unless
`ollama_cloud_enabled` is set, so off-host execution cannot be reached by
accident. LewLM holds no cloud credentials — `ollama signin` is the operator's
step, and the daemon owns the account relationship.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from math import ceil
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lewlm.config.settings import LewLMSettings
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

OLLAMA_SOURCE_SCHEME = "ollama"
OLLAMA_SOURCE_PREFIX = f"{OLLAMA_SOURCE_SCHEME}://"

EXECUTION_LOCALITY_METADATA_KEY = "ollama_execution_locality"
HOST_LOCAL = "host_local"
OFF_HOST = "off_host"

# Ollama names a cloud-backed model with a `cloud` tag or a `-cloud` suffix.
# The convention is the only signal available before a daemon reports the
# structured `remote_host`/`remote_model` fields, so both are checked.
_CLOUD_NAME_PATTERN = re.compile(r"(?::cloud$|-cloud(?::|$))", re.IGNORECASE)
_REMOTE_RECORD_FIELDS = ("remote_host", "remote_model")

_MODALITY_BY_CAPABILITY = {
    "embedding": ModelModality.EMBEDDING,
    "rerank": ModelModality.RERANK,
    "vision": ModelModality.MULTIMODAL,
}


@dataclass(frozen=True)
class OllamaInventoryResult:
    """What one inventory read produced, including why it produced nothing."""

    endpoint: str
    manifests: list[ModelManifest] = field(default_factory=list)
    skipped_cloud: list[str] = field(default_factory=list)
    # Set when the daemon could not be read. A failed read is not evidence that
    # the operator's models are gone, so callers must not prune on it.
    error: str | None = None
    # Set when the models were registered but no configured endpoint is the
    # Ollama daemon, so nothing can route to them until configuration changes.
    binding_note: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def is_ollama_source(source_path: str) -> bool:
    """Whether a manifest's `source_path` names an Ollama-resident model.

    Callers that resolve `source_path` as a file must check this first: an
    Ollama manifest has no file behind it.
    """

    return source_path.startswith(OLLAMA_SOURCE_PREFIX)


def ollama_source_path(tag: str) -> str:
    return f"{OLLAMA_SOURCE_PREFIX}{tag}"


def classify_execution_locality(record: dict[str, Any]) -> str:
    """Decide whether Ollama executes this model on the host or in its cloud.

    Structured evidence wins over the naming convention, because a daemon that
    reports `remote_host` is stating the fact outright.
    """

    for field_name in _REMOTE_RECORD_FIELDS:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return OFF_HOST
    for key in ("name", "model"):
        value = record.get(key)
        if isinstance(value, str) and _CLOUD_NAME_PATTERN.search(value):
            return OFF_HOST
    return HOST_LOCAL


def discover_ollama_models(settings: LewLMSettings) -> OllamaInventoryResult:
    """Read the daemon's model list and build a manifest for each usable model.

    Never raises. A daemon that is not running is an ordinary state for a
    component LewLM does not manage, so it comes back as an error string and an
    empty manifest list rather than failing the caller's scan.
    """

    endpoint = settings.ollama_base_url.rstrip("/")
    try:
        records = _fetch_tags(endpoint, timeout=settings.ollama_discovery_timeout_seconds)
    except _OllamaUnavailable as exc:
        return OllamaInventoryResult(endpoint=endpoint, error=str(exc))

    bound_endpoint_id, binding_note = _ollama_endpoint_binding(settings, endpoint)
    manifests: list[ModelManifest] = []
    skipped_cloud: list[str] = []
    seen_tags: set[str] = set()
    for record in records:
        tag = _record_tag(record)
        if not tag or tag in seen_tags:
            continue
        seen_tags.add(tag)
        locality = classify_execution_locality(record)
        if locality == OFF_HOST and not settings.ollama_cloud_enabled:
            skipped_cloud.append(tag)
            continue
        manifest = _build_manifest(record, tag=tag, locality=locality, endpoint=endpoint)
        if bound_endpoint_id is not None:
            manifest.metadata["external_endpoint_id"] = bound_endpoint_id
        manifests.append(manifest)

    return OllamaInventoryResult(
        endpoint=endpoint,
        manifests=manifests,
        skipped_cloud=sorted(skipped_cloud),
        binding_note=binding_note,
    )


def _ollama_endpoint_binding(settings: LewLMSettings, endpoint: str) -> tuple[str | None, str | None]:
    """Which configured endpoint *is* this daemon, so its models bind to it.

    An Ollama model must execute on the Ollama daemon, never on whichever
    other accelerator happens to be configured. With named endpoints, settings
    validation already guarantees exactly one matching `ollama_local` entry.
    With the legacy singular settings, the one endpoint may or may not be the
    daemon; when it is not, the models are still registered (nothing is
    silently dropped) but stay unbound and unroutable, and the scan says why.
    """

    from lewlm.config.endpoints import server_root

    daemon_root = server_root(endpoint)
    for config in settings.resolved_external_endpoints():
        if config.enabled and config.profile == "ollama_local" and server_root(config.base_url) == daemon_root:
            return config.endpoint_id, None
    return None, (
        f"No enabled external endpoint with profile `ollama_local` targets {daemon_root}; the discovered Ollama "
        "models are registered but cannot be routed. Point LEWLM_EXTERNAL_ACCELERATOR_PROFILE=ollama_local and "
        "LEWLM_EXTERNAL_ACCELERATOR_BASE_URL at the daemon, or add an `ollama_local` entry to LEWLM_EXTERNAL_ENDPOINTS."
    )


class _OllamaUnavailable(Exception):
    """The daemon could not be read. Carries an operator-facing reason."""


def _fetch_tags(endpoint: str, *, timeout: int) -> list[dict[str, Any]]:
    url = f"{endpoint}/api/tags"
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        raise _OllamaUnavailable(f"Ollama returned HTTP {exc.code} for {url}.") from exc
    except URLError as exc:
        raise _OllamaUnavailable(
            f"Could not reach an Ollama daemon at {url} ({exc.reason}). "
            "LewLM does not start Ollama; run the desktop client or `ollama serve`.",
        ) from exc
    except TimeoutError as exc:
        raise _OllamaUnavailable(f"Ollama at {url} did not respond within {timeout}s.") from exc
    except OSError as exc:
        raise _OllamaUnavailable(f"Could not read Ollama at {url} ({exc}).") from exc

    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError as exc:
        raise _OllamaUnavailable(f"Ollama returned malformed JSON from {url}.") from exc
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise _OllamaUnavailable(f"Ollama returned no model list from {url}.")
    return [record for record in models if isinstance(record, dict)]


def _record_tag(record: dict[str, Any]) -> str:
    for key in ("name", "model"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _record_details(record: dict[str, Any]) -> dict[str, Any]:
    details = record.get("details")
    return details if isinstance(details, dict) else {}


def _capabilities(record: dict[str, Any]) -> tuple[str, ...]:
    raw = record.get("capabilities")
    if not isinstance(raw, list):
        return ()
    return tuple(sorted({item.casefold() for item in raw if isinstance(item, str) and item}))


def _modality(capabilities: tuple[str, ...]) -> tuple[ModelModality, ...]:
    """Map Ollama's capability list onto a LewLM modality.

    This is load-bearing rather than cosmetic: the bridge refuses a capability
    whose modality the manifest does not declare, so an embedding model that
    arrived labelled `text` would have its working embeddings route rejected
    before the endpoint was ever probed.
    """

    for capability in capabilities:
        modality = _MODALITY_BY_CAPABILITY.get(capability)
        if modality is not None:
            return (modality,)
    return (ModelModality.TEXT,)


def _fingerprint(record: dict[str, Any], *, tag: str) -> str:
    digest = record.get("digest")
    if isinstance(digest, str) and digest.strip():
        return digest.strip()
    # A daemon that reports no digest still needs a stable identity, and the tag
    # is the only thing guaranteed present.
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _model_id(tag: str, fingerprint: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", tag.casefold()).strip("-") or "model"
    return f"{slug}-{fingerprint[:12]}"


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _build_manifest(
    record: dict[str, Any],
    *,
    tag: str,
    locality: str,
    endpoint: str,
) -> ModelManifest:
    details = _record_details(record)
    capabilities = _capabilities(record)
    fingerprint = _fingerprint(record, tag=tag)
    size_bytes = _positive_int(record.get("size"))
    family = details.get("family")
    architecture_family = family.strip() if isinstance(family, str) and family.strip() else "unknown"
    quantization = details.get("quantization_level")

    return ModelManifest(
        model_id=_model_id(tag, fingerprint),
        display_name=tag,
        architecture_family=architecture_family,
        architecture_subtype=ArchitectureSubtype.UNKNOWN,
        modality=_modality(capabilities),
        source_path=ollama_source_path(tag),
        format_type=ModelFormat.GGUF,
        quantization=quantization if isinstance(quantization, str) and quantization else None,
        # The bridge is the only runtime that can serve this. Declaring it alone
        # keeps routing deterministic instead of letting a packaged runtime win a
        # manifest whose weights it has no way to open.
        runtime_affinity=(RuntimeAffinity.EXTERNAL_ACCELERATOR,),
        context_length=_positive_int(details.get("context_length")),
        estimated_memory_mb=ceil(size_bytes / (1024 * 1024)) if size_bytes else None,
        conversion_status=ConversionStatus.RUNNABLE,
        fingerprint=fingerprint,
        last_validation_result=ModelValidationResult(
            status=ValidationState.VALID,
            message=f"Advertised by an Ollama daemon at {endpoint}.",
            details={"source_kind": OLLAMA_SOURCE_SCHEME},
        ),
        metadata={
            "source_kind": OLLAMA_SOURCE_SCHEME,
            # What the bridge matches against the endpoint's advertised ids.
            "external_adapter_model_id": tag,
            EXECUTION_LOCALITY_METADATA_KEY: locality,
            # Shared key read by routing/execution metadata for every source kind.
            "execution_locality": locality,
            "ollama_endpoint": endpoint,
            "ollama_tag": tag,
            "ollama_digest": record.get("digest"),
            "ollama_capabilities": list(capabilities),
            "ollama_parameter_size": details.get("parameter_size"),
            "ollama_format": details.get("format"),
            **({"size_bytes": size_bytes} if size_bytes else {}),
            **(
                {
                    "ollama_remote_host": record.get("remote_host"),
                    "ollama_remote_model": record.get("remote_model"),
                }
                if locality == OFF_HOST
                else {}
            ),
            "runtime_ownership_note": (
                "Executed by an operator-managed Ollama daemon. LewLM discovers and forwards; "
                "it does not install, supervise, or own this runtime."
            ),
        },
    )
