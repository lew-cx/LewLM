"""Explicit subsystem scope labels for LewLM."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SubsystemScopeLabel(str, Enum):
    """Stable scope labels used across code and docs."""

    CORE = "core"
    PERFORMANCE_CORE = "performance_core"
    OPTIONAL_MODULE = "optional_module"
    EXPERIMENTAL = "experimental"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class SubsystemScope:
    """One subsystem entry in the LewLM scope matrix."""

    key: str
    title: str
    scope: SubsystemScopeLabel
    summary: str
    code_paths: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    install_extras: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


_SCOPE_MATRIX: tuple[SubsystemScope, ...] = (
    SubsystemScope(
        key="configuration_security",
        title="Configuration and security",
        scope=SubsystemScopeLabel.CORE,
        summary="Process settings, filesystem scope, authorization, sandboxing, and local audit controls.",
        code_paths=(
            "src/lewlm/config",
            "src/lewlm/security",
        ),
    ),
    SubsystemScope(
        key="public_surfaces",
        title="Public surfaces",
        scope=SubsystemScopeLabel.CORE,
        summary="The CLI, local HTTP API, event stream transport, embeddable Python facade, and typed app helpers.",
        code_paths=(
            "src/lewlm/cli",
            "src/lewlm/api",
            "src/lewlm/events",
            "src/lewlm/library.py",
            "src/lewlm/app_helpers.py",
        ),
        depends_on=(
            "configuration_security",
            "registry_routing",
            "session_state",
        ),
    ),
    SubsystemScope(
        key="registry_routing",
        title="Registry and routing",
        scope=SubsystemScopeLabel.CORE,
        summary="Model discovery, manifests, runtime contracts, runtime catalog selection, and request routing decisions.",
        code_paths=(
            "src/lewlm/registry",
            "src/lewlm/routing",
            "src/lewlm/runtime/catalog.py",
            "src/lewlm/core/contracts.py",
            "src/lewlm/core/chat.py",
            "src/lewlm/core/multimodal.py",
        ),
        depends_on=("configuration_security",),
    ),
    SubsystemScope(
        key="session_state",
        title="Session and local state",
        scope=SubsystemScopeLabel.CORE,
        summary="Local persistence for metadata, sessions, exports/imports, and operator-visible state.",
        code_paths=(
            "src/lewlm/storage",
            "src/lewlm/history",
        ),
        depends_on=("configuration_security",),
    ),
    SubsystemScope(
        key="conversion_pipeline",
        title="Conversion pipeline",
        scope=SubsystemScopeLabel.CORE,
        summary="Compatibility checks, background conversion jobs, quantization policy metadata, and conversion records.",
        code_paths=("src/lewlm/conversion",),
        depends_on=(
            "configuration_security",
            "registry_routing",
            "session_state",
        ),
    ),
    SubsystemScope(
        key="telemetry_reporting",
        title="Telemetry and capability reporting",
        scope=SubsystemScopeLabel.CORE,
        summary="Runtime stats, readiness reporting, measured capability truth, and operator diagnostics.",
        code_paths=(
            "src/lewlm/telemetry",
            "src/lewlm/utils/validation_manifests.py",
        ),
        depends_on=(
            "registry_routing",
            "session_state",
        ),
    ),
    SubsystemScope(
        key="serving_control",
        title="Serving-control layers",
        scope=SubsystemScopeLabel.PERFORMANCE_CORE,
        summary="LewLM-owned scheduler, batching window, cache surfaces, speculation control, and serving-policy plumbing.",
        code_paths=(
            "src/lewlm/core/serving_core.py",
            "src/lewlm/core/speculation.py",
            "src/lewlm/runtime/scheduler.py",
            "src/lewlm/runtime/paged_kv.py",
            "src/lewlm/runtime/prefix_cache.py",
            "src/lewlm/runtime/request_coalescer.py",
            "src/lewlm/runtime/response_cache.py",
            "src/lewlm/serving_profiles.py",
        ),
        depends_on=(
            "registry_routing",
            "session_state",
            "telemetry_reporting",
        ),
        notes=(
            "This is the selectively owned optimization layer that differentiates LewLM from a thin wrapper.",
        ),
    ),
    SubsystemScope(
        key="benchmark_acceptance",
        title="Benchmark and acceptance artifacts",
        scope=SubsystemScopeLabel.PERFORMANCE_CORE,
        summary="Benchmark capture, direct/runtime comparisons, serving-profile prove-out, and release-manifest acceptance evidence.",
        code_paths=("src/lewlm/benchmarking",),
        depends_on=(
            "serving_control",
            "telemetry_reporting",
        ),
    ),
    SubsystemScope(
        key="runtime_packs",
        title="Concrete runtime packs",
        scope=SubsystemScopeLabel.OPTIONAL_MODULE,
        summary="Install-selectable MLX, llama.cpp, and loopback-adapter backends that plug into the core runtime contract.",
        code_paths=(
            "src/lewlm/runtime/mlx_text",
            "src/lewlm/runtime/mlx_vision",
            "src/lewlm/runtime/mlx_audio",
            "src/lewlm/runtime/llamacpp",
            "src/lewlm/runtime/adapters",
            "src/lewlm/runtime/metal",
        ),
        depends_on=(
            "registry_routing",
            "serving_control",
        ),
        install_extras=(
            "mlx",
            "llamacpp",
        ),
        notes=(
            "LewLM's runtime contracts are core; these concrete backend packs remain install-selectable modules.",
        ),
    ),
    SubsystemScope(
        key="documents_module",
        title="Documents and local tooling",
        scope=SubsystemScopeLabel.OPTIONAL_MODULE,
        summary="Document ingest, deterministic rendering, built-in document skills, and document-oriented local tools.",
        code_paths=(
            "src/lewlm/documents",
            "src/lewlm/tools",
        ),
        depends_on=(
            "configuration_security",
            "public_surfaces",
            "session_state",
        ),
        install_extras=("documents",),
    ),
    SubsystemScope(
        key="frontier_diagnostics",
        title="Frontier architecture diagnostics",
        scope=SubsystemScopeLabel.EXPERIMENTAL,
        summary="Proof-oriented architecture classification and planning surfaces for frontier model shapes.",
        code_paths=(
            "src/lewlm/runtime/experimental/architectures.py",
            "src/lewlm/runtime/experimental/frontier.py",
        ),
        depends_on=("registry_routing",),
    ),
    SubsystemScope(
        key="distributed_cluster",
        title="Distributed cluster workflows",
        scope=SubsystemScopeLabel.EXPERIMENTAL,
        summary="Coordinator/worker enrollment, pipeline-stage planning, and multi-host proof execution surfaces.",
        code_paths=(
            "src/lewlm/runtime/experimental/distributed.py",
            "src/lewlm/api/routes/cluster.py",
        ),
        depends_on=(
            "public_surfaces",
            "registry_routing",
            "session_state",
        ),
        notes=(
            "This remains a proof path rather than a production tensor-parallel serving layer.",
        ),
    ),
    SubsystemScope(
        key="workflow_engine",
        title="Workflow engine",
        scope=SubsystemScopeLabel.OUT_OF_SCOPE,
        summary="Persistent workflow graphs, agent-framework orchestration, or application-specific automations owned by LewLM.",
        notes=(
            "LewLM should stay middleware-first and app-agnostic rather than becoming a workflow engine.",
        ),
    ),
    SubsystemScope(
        key="vector_database_control_plane",
        title="Vector database and collection control plane",
        scope=SubsystemScopeLabel.OUT_OF_SCOPE,
        summary="Persistent collection CRUD, vector database ownership, or retrieval control-plane services.",
        notes=(
            "Retrieval helpers can package context for apps, but LewLM should not become a vector database product.",
        ),
    ),
    SubsystemScope(
        key="gui_consumer_app",
        title="GUI or consumer chat application",
        scope=SubsystemScopeLabel.OUT_OF_SCOPE,
        summary="Desktop UI, end-user chat product features, or other GUI-first application layers.",
        notes=(
            "LewLM is a backend package that sits under other apps rather than replacing them.",
        ),
    ),
)


def scope_matrix() -> tuple[SubsystemScope, ...]:
    """Return the immutable LewLM subsystem scope matrix."""

    return _SCOPE_MATRIX


def scope_entry(key: str) -> SubsystemScope:
    """Return one scope entry by key."""

    for entry in _SCOPE_MATRIX:
        if entry.key == key:
            return entry
    raise KeyError(key)


def scope_entries_for(scope: SubsystemScopeLabel) -> tuple[SubsystemScope, ...]:
    """Return entries for one scope label."""

    return tuple(entry for entry in _SCOPE_MATRIX if entry.scope == scope)


def scope_dependency_map() -> dict[str, tuple[str, ...]]:
    """Return the explicit subsystem dependency map."""

    return {entry.key: entry.depends_on for entry in _SCOPE_MATRIX}


__all__ = [
    "SubsystemScope",
    "SubsystemScopeLabel",
    "scope_dependency_map",
    "scope_entries_for",
    "scope_entry",
    "scope_matrix",
]
