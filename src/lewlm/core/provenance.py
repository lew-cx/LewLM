"""Explicit provenance for the components that produced a result.

`ExecutionMetadata.version` identifies the envelope schema, which is not enough
for a caller that needs to know *which* renderer, parser, chunker, OCR engine,
or scoring policy actually produced an artifact. These records name those
components and pin their versions so a host application can reproduce, cache,
or invalidate a result without guessing from the envelope version.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as distribution_version
from typing import Literal

from pydantic import BaseModel, Field


class ComponentKind(str, Enum):
    """The role a component played in producing a result."""

    PARSER = "parser"
    RENDERER = "renderer"
    CHUNKER = "chunker"
    OCR = "ocr"
    SCORING_POLICY = "scoring_policy"
    TOKENIZER = "tokenizer"


class ComponentProvenance(BaseModel):
    """One named, versioned component that contributed to a result."""

    kind: ComponentKind
    name: str = Field(description="Stable LewLM-owned component name, not a package name.")
    version: str = Field(description="Version of the LewLM-owned component behaviour.")
    implementation: str | None = Field(
        default=None,
        description="Third-party distribution that performs the work, when one does.",
    )
    implementation_version: str | None = Field(
        default=None,
        description="Installed version of `implementation`, or null when it cannot be determined.",
    )
    deterministic: bool = Field(
        default=True,
        description="Whether the same input reproduces the same output on this component.",
    )


@lru_cache(maxsize=None)
def installed_version(distribution: str) -> str | None:
    """Return the installed version of a distribution, or `None` when absent.

    Resolution is cached because installed versions cannot change inside a
    running process, and provenance is attached to every document result.
    """

    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - defensive against odd metadata backends
        return None


def component(
    kind: ComponentKind,
    *,
    name: str,
    version: str,
    implementation: str | None = None,
    deterministic: bool = True,
) -> ComponentProvenance:
    """Build a provenance record, resolving the implementation version if present."""

    return ComponentProvenance(
        kind=kind,
        name=name,
        version=version,
        implementation=implementation,
        implementation_version=installed_version(implementation) if implementation else None,
        deterministic=deterministic,
    )


# --- LewLM-owned component versions ------------------------------------------
#
# Bump a version here when the component's *observable behaviour* changes, so a
# caller can tell a cached result apart from one produced by newer logic.

CHUNKER_NAME = "section_block_semantic_segmentation"
CHUNKER_VERSION = "1.0.0"

RETRIEVAL_SCORING_POLICY_NAME = "rerank_primary_embedding_tiebreak"
RETRIEVAL_SCORING_POLICY_VERSION = "1.0.0"

#: Parser component names keyed by the file suffix family LewLM routes to them.
_PARSER_IMPLEMENTATIONS: dict[str, tuple[str, str | None]] = {
    "text": ("text_line_paragraph_parser", None),
    "markdown": ("markdown_block_parser", None),
    "csv": ("csv_table_parser", None),
    "xlsx": ("xlsx_sheet_parser", "openpyxl"),
    "docx": ("docx_body_parser", "python-docx"),
    "pdf": ("pdf_page_parser", "pypdf"),
    "image": ("image_metadata_parser", "pillow"),
    "image_bundle": ("image_bundle_parser", "pillow"),
}

_RENDERER_IMPLEMENTATIONS: dict[str, str | None] = {
    "text": None,
    "markdown": None,
    "json": None,
    "csv": None,
    "docx": "python-docx",
    "pdf": "reportlab",
    "xlsx": "openpyxl",
}

PARSER_VERSION = "1.0.0"
RENDERER_VERSION = "1.0.0"


def parser_provenance(source_type: str) -> ComponentProvenance:
    """Provenance for the parser LewLM used on a given source type."""

    name, implementation = _PARSER_IMPLEMENTATIONS.get(
        source_type,
        (f"{source_type}_parser", None),
    )
    return component(
        ComponentKind.PARSER,
        name=name,
        version=PARSER_VERSION,
        implementation=implementation,
    )


def renderer_provenance(output_format: str) -> ComponentProvenance:
    """Provenance for the renderer that produced an artifact."""

    return component(
        ComponentKind.RENDERER,
        name=f"{output_format}_renderer",
        version=RENDERER_VERSION,
        implementation=_RENDERER_IMPLEMENTATIONS.get(output_format),
    )


def chunker_provenance() -> ComponentProvenance:
    """Provenance for the chunking strategy applied to an ingested document."""

    return component(ComponentKind.CHUNKER, name=CHUNKER_NAME, version=CHUNKER_VERSION)


def ocr_provenance(*, backend_name: str | None, available: bool) -> ComponentProvenance:
    """Provenance for the OCR engine, including when no engine was available."""

    return component(
        ComponentKind.OCR,
        name=backend_name or "unavailable",
        version=PARSER_VERSION,
        implementation=backend_name if available else None,
        # OCR output depends on the engine build and model weights, so LewLM
        # does not claim reproducibility for it.
        deterministic=False,
    )


def retrieval_scoring_provenance() -> ComponentProvenance:
    """Provenance for the retrieval ranking policy."""

    return component(
        ComponentKind.SCORING_POLICY,
        name=RETRIEVAL_SCORING_POLICY_NAME,
        version=RETRIEVAL_SCORING_POLICY_VERSION,
    )


class RetrievalScoringPolicy(BaseModel):
    """The named, versioned ranking rules a retrieval response actually used.

    LewLM's ranking is rerank-primary with embedding tie-breaking and a stable
    original-input-order fallback. That is a reasonable policy, but it is only
    reproducible for a caller if it is named and versioned.
    """

    name: str = RETRIEVAL_SCORING_POLICY_NAME
    version: str = RETRIEVAL_SCORING_POLICY_VERSION
    primary_signal: Literal["rerank", "embedding", "none"]
    tie_break_signal: Literal["embedding", "original_order"]
    final_tie_break: Literal["original_order"] = "original_order"
    normalization: Literal["none", "cosine"] = "none"
    missing_score_behaviour: Literal["rejected", "ranked_last"] = "rejected"
    deduplication: Literal["none", "chunk_id"] = "chunk_id"
    embeddings_used: bool
    rerank_used: bool
