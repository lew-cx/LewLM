from __future__ import annotations

from lewlm.core.bootstrap import bootstrap_services
from lewlm.scope_matrix import (
    SubsystemScopeLabel,
    scope_dependency_map,
    scope_entries_for,
    scope_entry,
    scope_matrix,
)


def test_scope_matrix_labels_expected_boundaries() -> None:
    assert scope_entry("serving_control").scope == SubsystemScopeLabel.PERFORMANCE_CORE
    assert scope_entry("runtime_packs").scope == SubsystemScopeLabel.OPTIONAL_MODULE
    assert scope_entry("documents_module").install_extras == ("documents",)
    assert scope_entry("distributed_cluster").scope == SubsystemScopeLabel.EXPERIMENTAL
    assert scope_entry("vector_database_control_plane").scope == SubsystemScopeLabel.OUT_OF_SCOPE


def test_scope_matrix_dependency_map_references_known_entries() -> None:
    entries = scope_matrix()
    known_keys = {entry.key for entry in entries}

    assert len(entries) == len(known_keys)
    for dependency_keys in scope_dependency_map().values():
        assert set(dependency_keys) <= known_keys

    assert [entry.key for entry in scope_entries_for(SubsystemScopeLabel.CORE)]


def test_bootstrap_services_preserves_scope_layer_outputs(temp_settings) -> None:
    services = bootstrap_services(temp_settings)
    try:
        assert services.prompt_compiler.tool_catalog is services.tool_catalog_service
        assert services.tool_catalog_service.list_tools()
        assert services.skill_catalog_service.list_skills()
        assert services.cluster_service is not None
    finally:
        services.close()
