"""OpenAPI document normalization.

Routes that declare a request body through `openapi_extra` embed a standalone
Pydantic JSON Schema, which carries its own `$defs` block and `#/$defs/...`
references. Those references resolve against a schema root that does not exist
in the published OpenAPI document, so standard tooling cannot load the contract.

This module hoists every inlined `$defs` entry into `components/schemas` and
repoints the references at it, leaving a document that resolves as published.

It also publishes models that no route binds as a body or `response_model` —
the streaming chunks and the hand-declared request bodies — under their own
names in `components/schemas`. FastAPI only registers what it binds, so those
would otherwise exist in the document solely as anonymous inline blobs, leaving
a code generator with no named type to emit.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

_DEFS_KEY = "$defs"
_REF_KEY = "$ref"
_LOCAL_DEFS_PREFIX = "#/$defs/"
_COMPONENTS_PREFIX = "#/components/schemas/"


def register_named_schemas(schema: dict[str, Any], models: tuple[type[BaseModel], ...]) -> dict[str, Any]:
    """Publish `models` into `components/schemas` under their class names.

    Each model's nested definitions are hoisted alongside it and its references
    rewritten to point at components, so the published entry resolves without
    the `$defs` root Pydantic assumes. The input document is modified in place.
    """

    components = schema.setdefault("components", {}).setdefault("schemas", {})
    for model in models:
        model_schema = model.model_json_schema(ref_template=f"{_COMPONENTS_PREFIX}{{model}}")
        definitions = model_schema.pop(_DEFS_KEY, None)
        if isinstance(definitions, dict):
            for name, definition in definitions.items():
                components.setdefault(name, definition)
        components[model.__name__] = model_schema
    return schema


def component_ref(model: type[BaseModel]) -> dict[str, str]:
    """Reference a model published by `register_named_schemas`."""

    return {_REF_KEY: f"{_COMPONENTS_PREFIX}{model.__name__}"}


def normalize_openapi_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return `schema` with all inlined `$defs` hoisted into components.

    The input document is modified in place and returned. Definitions that
    already exist in `components/schemas` with identical content are reused
    rather than duplicated; a genuine name conflict is given a suffixed name so
    no definition is silently overwritten.
    """

    components = schema.setdefault("components", {}).setdefault("schemas", {})
    _hoist_node(schema, components=components)
    return schema


def _hoist_node(node: Any, *, components: dict[str, Any]) -> Any:
    """Recursively hoist `$defs` blocks found at or beneath `node`."""

    if isinstance(node, list):
        return [_hoist_node(item, components=components) for item in node]
    if not isinstance(node, dict):
        return node

    definitions = node.pop(_DEFS_KEY, None)
    if isinstance(definitions, dict) and definitions:
        renames = _register_definitions(definitions, components=components)
        _rewrite_refs(node, renames=renames)

    for key, value in node.items():
        node[key] = _hoist_node(value, components=components)
    return node


def _register_definitions(
    definitions: dict[str, Any],
    *,
    components: dict[str, Any],
) -> dict[str, str]:
    """Move `definitions` into `components`, returning local-name -> final-name."""

    # Nested definitions reference their siblings by local name, so resolve the
    # naming for the whole block before rewriting any of them.
    renames = {name: name for name in definitions}
    prepared: dict[str, Any] = {}
    for name, definition in definitions.items():
        hoisted = _hoist_node(definition, components=components)
        _rewrite_refs(hoisted, renames=renames)
        prepared[name] = hoisted

    for name, definition in prepared.items():
        existing = components.get(name)
        if existing is None or existing == definition:
            components[name] = definition
            continue
        # Same name, different shape: keep both under distinct names.
        suffix = 2
        while f"{name}_{suffix}" in components and components[f"{name}_{suffix}"] != definition:
            suffix += 1
        final_name = f"{name}_{suffix}"
        components[final_name] = definition
        renames[name] = final_name

    if any(local != final for local, final in renames.items()):
        # A conflict renamed at least one definition; repoint sibling references.
        for name in prepared:
            _rewrite_refs(components[renames[name]], renames=renames)
    return renames


def _rewrite_refs(node: Any, *, renames: dict[str, str]) -> None:
    """Rewrite `#/$defs/X` references in place to their component location."""

    if isinstance(node, list):
        for item in node:
            _rewrite_refs(item, renames=renames)
        return
    if not isinstance(node, dict):
        return
    reference = node.get(_REF_KEY)
    if isinstance(reference, str):
        node[_REF_KEY] = _rewritten_reference(reference, renames=renames)
    # A discriminator maps property values to schema references, which are not
    # spelled with `$ref` but must resolve just the same.
    discriminator = node.get("discriminator")
    if isinstance(discriminator, dict):
        mapping = discriminator.get("mapping")
        if isinstance(mapping, dict):
            discriminator["mapping"] = {
                key: _rewritten_reference(value, renames=renames) if isinstance(value, str) else value
                for key, value in mapping.items()
            }
    for value in node.values():
        _rewrite_refs(value, renames=renames)


def _rewritten_reference(reference: str, *, renames: dict[str, str]) -> str:
    if not reference.startswith(_LOCAL_DEFS_PREFIX):
        return reference
    local_name = reference[len(_LOCAL_DEFS_PREFIX) :]
    return f"{_COMPONENTS_PREFIX}{renames.get(local_name, local_name)}"


__all__ = ["component_ref", "normalize_openapi_schema", "register_named_schemas"]
