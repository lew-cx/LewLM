"""Reserved style-token vocabulary and deterministic style resolution.

`DocumentIR.style_tokens` declares document-level brand tokens. Sections,
blocks, headers, footers, and citations reference those tokens by name through
their own `style_tokens` lists.

Only the reserved names in :class:`StyleTokenRole` change rendered output, and
only with values that satisfy the role's grammar. Every other declared token is
preserved as lineage and never reaches a renderer, so ingestion provenance
tokens such as ``code``, ``ocr``, or a source DOCX paragraph style name stay
inert. LewLM never accepts raw CSS or executable styling: a token value is
either a validated colour, a validated font family, a bounded point size, or a
member of a closed keyword set.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import re

MIN_FONT_SIZE_PT = 4.0
MAX_FONT_SIZE_PT = 96.0

_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
_FONT_FAMILY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")


class StyleElement(str, Enum):
    """Element classes a reserved style token can be scoped to."""

    DOCUMENT = "document"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CALLOUT = "callout"
    IMAGE = "image"
    CITATION = "citation"
    HEADER = "header"
    FOOTER = "footer"


BODY_ELEMENTS = frozenset(
    {
        StyleElement.PARAGRAPH,
        StyleElement.LIST,
        StyleElement.TABLE,
        StyleElement.CALLOUT,
        StyleElement.IMAGE,
        StyleElement.CITATION,
        StyleElement.HEADER,
        StyleElement.FOOTER,
    },
)


class StyleTokenRole(str, Enum):
    """Reserved style-token names that renderers honour."""

    BODY_COLOR = "body_color"
    BODY_FONT = "body_font"
    BODY_FONT_SIZE_PT = "body_font_size_pt"
    HEADING_COLOR = "heading_color"
    HEADING_FONT = "heading_font"
    HEADING_FONT_SIZE_PT = "heading_font_size_pt"
    SURFACE_COLOR = "surface_color"
    ACCENT_COLOR = "accent_color"
    EMPHASIS = "emphasis"


class StyleValueKind(str, Enum):
    COLOR = "color"
    FONT_FAMILY = "font_family"
    FONT_SIZE_PT = "font_size_pt"
    EMPHASIS = "emphasis"


EMPHASIS_VALUES = ("normal", "bold", "italic", "bold_italic")


@dataclass(frozen=True, slots=True)
class StyleRoleSpec:
    role: StyleTokenRole
    value_kind: StyleValueKind
    default_elements: frozenset[StyleElement]
    description: str


_ROLE_SPECS: dict[StyleTokenRole, StyleRoleSpec] = {
    StyleTokenRole.BODY_COLOR: StyleRoleSpec(
        role=StyleTokenRole.BODY_COLOR,
        value_kind=StyleValueKind.COLOR,
        default_elements=BODY_ELEMENTS,
        description="Text colour for body content.",
    ),
    StyleTokenRole.BODY_FONT: StyleRoleSpec(
        role=StyleTokenRole.BODY_FONT,
        value_kind=StyleValueKind.FONT_FAMILY,
        default_elements=BODY_ELEMENTS,
        description="Font family for body content.",
    ),
    StyleTokenRole.BODY_FONT_SIZE_PT: StyleRoleSpec(
        role=StyleTokenRole.BODY_FONT_SIZE_PT,
        value_kind=StyleValueKind.FONT_SIZE_PT,
        default_elements=BODY_ELEMENTS,
        description="Point size for body content.",
    ),
    StyleTokenRole.HEADING_COLOR: StyleRoleSpec(
        role=StyleTokenRole.HEADING_COLOR,
        value_kind=StyleValueKind.COLOR,
        default_elements=frozenset({StyleElement.HEADING}),
        description="Text colour for section headings.",
    ),
    StyleTokenRole.HEADING_FONT: StyleRoleSpec(
        role=StyleTokenRole.HEADING_FONT,
        value_kind=StyleValueKind.FONT_FAMILY,
        default_elements=frozenset({StyleElement.HEADING}),
        description="Font family for section headings.",
    ),
    StyleTokenRole.HEADING_FONT_SIZE_PT: StyleRoleSpec(
        role=StyleTokenRole.HEADING_FONT_SIZE_PT,
        value_kind=StyleValueKind.FONT_SIZE_PT,
        default_elements=frozenset({StyleElement.HEADING}),
        description="Point size for section headings.",
    ),
    StyleTokenRole.SURFACE_COLOR: StyleRoleSpec(
        role=StyleTokenRole.SURFACE_COLOR,
        value_kind=StyleValueKind.COLOR,
        default_elements=frozenset({StyleElement.CALLOUT}),
        description="Background fill for callout surfaces.",
    ),
    StyleTokenRole.ACCENT_COLOR: StyleRoleSpec(
        role=StyleTokenRole.ACCENT_COLOR,
        value_kind=StyleValueKind.COLOR,
        default_elements=frozenset(),
        description="Accent colour for callout rules and table header fills.",
    ),
    StyleTokenRole.EMPHASIS: StyleRoleSpec(
        role=StyleTokenRole.EMPHASIS,
        value_kind=StyleValueKind.EMPHASIS,
        default_elements=frozenset(),
        description="Bold/italic emphasis, applied only where referenced.",
    ),
}

# Deterministic application order. Body roles resolve first so that heading and
# surface roles win when an element opts into both.
_ROLE_ORDER: tuple[StyleTokenRole, ...] = (
    StyleTokenRole.BODY_COLOR,
    StyleTokenRole.BODY_FONT,
    StyleTokenRole.BODY_FONT_SIZE_PT,
    StyleTokenRole.HEADING_COLOR,
    StyleTokenRole.HEADING_FONT,
    StyleTokenRole.HEADING_FONT_SIZE_PT,
    StyleTokenRole.SURFACE_COLOR,
    StyleTokenRole.ACCENT_COLOR,
    StyleTokenRole.EMPHASIS,
)


def style_role_specs() -> tuple[StyleRoleSpec, ...]:
    """Return the reserved role vocabulary in application order."""

    return tuple(_ROLE_SPECS[role] for role in _ROLE_ORDER)


def resolve_style_role(name: str) -> StyleTokenRole | None:
    """Return the reserved role for a token name, or `None` for lineage tokens."""

    try:
        return StyleTokenRole(name)
    except ValueError:
        return None


def normalize_style_value(role: StyleTokenRole, value: str) -> str:
    """Validate and normalize a reserved token value.

    Raises:
        ValueError: when the value does not satisfy the role's grammar.
    """

    spec = _ROLE_SPECS[role]
    candidate = value.strip()
    if spec.value_kind is StyleValueKind.COLOR:
        if not _COLOR_PATTERN.match(candidate):
            raise ValueError(f"Style token '{role.value}' requires a #RRGGBB colour value.")
        return "#" + candidate[1:].upper()
    if spec.value_kind is StyleValueKind.FONT_FAMILY:
        if not _FONT_FAMILY_PATTERN.match(candidate):
            raise ValueError(
                f"Style token '{role.value}' requires a font family of letters, digits, spaces, '.', '_', or '-'.",
            )
        return candidate
    if spec.value_kind is StyleValueKind.FONT_SIZE_PT:
        try:
            size = float(candidate)
        except ValueError as exc:
            raise ValueError(f"Style token '{role.value}' requires a numeric point size.") from exc
        if not MIN_FONT_SIZE_PT <= size <= MAX_FONT_SIZE_PT:
            raise ValueError(
                f"Style token '{role.value}' requires a point size between {MIN_FONT_SIZE_PT:g} and {MAX_FONT_SIZE_PT:g}.",
            )
        return f"{size:g}"
    if candidate not in EMPHASIS_VALUES:
        allowed = ", ".join(EMPHASIS_VALUES)
        raise ValueError(f"Style token '{role.value}' requires one of: {allowed}.")
    return candidate


def normalize_style_scope(role: StyleTokenRole, applies_to: str | None) -> frozenset[StyleElement]:
    """Return the element classes a declared token applies to by default.

    An `applies_to` value outside the element vocabulary yields an empty scope:
    the token stays declarable and inspectable but never renders on its own.
    """

    spec = _ROLE_SPECS[role]
    if applies_to is None:
        return spec.default_elements
    candidate = applies_to.strip().lower()
    if candidate in {"", StyleElement.DOCUMENT.value}:
        return spec.default_elements
    try:
        element = StyleElement(candidate)
    except ValueError:
        return frozenset()
    return frozenset({element})


def is_known_style_scope(applies_to: str | None) -> bool:
    """Return whether `applies_to` names a known element class."""

    if applies_to is None:
        return True
    candidate = applies_to.strip().lower()
    if candidate == "":
        return True
    return candidate in {element.value for element in StyleElement}


@dataclass(frozen=True, slots=True)
class ResolvedStyle:
    """Concrete styling resolved for one rendered element."""

    text_color: str | None = None
    background_color: str | None = None
    font_family: str | None = None
    font_size_pt: float | None = None
    bold: bool = False
    italic: bool = False

    @property
    def is_empty(self) -> bool:
        return (
            self.text_color is None
            and self.background_color is None
            and self.font_family is None
            and self.font_size_pt is None
            and not self.bold
            and not self.italic
        )


EMPTY_STYLE = ResolvedStyle()


@dataclass(frozen=True, slots=True)
class _ActiveToken:
    role: StyleTokenRole
    value: str
    elements: frozenset[StyleElement]


@dataclass(frozen=True, slots=True)
class DocumentStyleSheet:
    """Resolved view of a document's reserved style tokens."""

    tokens: dict[StyleTokenRole, _ActiveToken]

    @classmethod
    def from_tokens(cls, style_tokens: Iterable) -> DocumentStyleSheet:
        active: dict[StyleTokenRole, _ActiveToken] = {}
        for token in style_tokens:
            role = resolve_style_role(token.name)
            if role is None:
                continue
            try:
                value = normalize_style_value(role, token.value)
            except ValueError:
                continue
            active[role] = _ActiveToken(
                role=role,
                value=value,
                elements=normalize_style_scope(role, token.applies_to),
            )
        return cls(tokens=active)

    @classmethod
    def from_document(cls, document) -> DocumentStyleSheet:
        return cls.from_tokens(document.style_tokens)

    @property
    def is_empty(self) -> bool:
        return not self.tokens

    def accent_color_for(
        self,
        element: StyleElement,
        *,
        token_names: Sequence[str] = (),
    ) -> str | None:
        """Resolve the accent used by a table header or callout rule."""

        token = self.tokens.get(StyleTokenRole.ACCENT_COLOR)
        if token is None:
            return None
        if token.role.value in set(token_names):
            return token.value
        if token.elements:
            return token.value if element in token.elements else None
        # Accent has renderer-specific defaults instead of a generic text or
        # background property: table headers and callout rules only.
        if element in {StyleElement.TABLE, StyleElement.CALLOUT}:
            return token.value
        return None

    def for_element(
        self,
        element: StyleElement,
        *,
        token_names: Sequence[str] = (),
    ) -> ResolvedStyle:
        """Resolve styling for one element class and its explicit references."""

        referenced = set(token_names)
        style = EMPTY_STYLE
        for role in _ROLE_ORDER:
            token = self.tokens.get(role)
            if token is None:
                continue
            if element not in token.elements and role.value not in referenced:
                continue
            style = _apply_role(style, token)
        return style


def _apply_role(style: ResolvedStyle, token: _ActiveToken) -> ResolvedStyle:
    role = token.role
    if role in {StyleTokenRole.BODY_COLOR, StyleTokenRole.HEADING_COLOR}:
        return replace(style, text_color=token.value)
    if role in {StyleTokenRole.SURFACE_COLOR, StyleTokenRole.ACCENT_COLOR}:
        return replace(style, background_color=token.value)
    if role in {StyleTokenRole.BODY_FONT, StyleTokenRole.HEADING_FONT}:
        return replace(style, font_family=token.value)
    if role in {StyleTokenRole.BODY_FONT_SIZE_PT, StyleTokenRole.HEADING_FONT_SIZE_PT}:
        return replace(style, font_size_pt=float(token.value))
    return replace(
        style,
        bold=token.value in {"bold", "bold_italic"},
        italic=token.value in {"italic", "bold_italic"},
    )


_BLOCK_ELEMENTS: dict[str, StyleElement] = {
    "paragraph": StyleElement.PARAGRAPH,
    "list": StyleElement.LIST,
    "table": StyleElement.TABLE,
    "callout": StyleElement.CALLOUT,
    "image": StyleElement.IMAGE,
}


def element_for_block(block) -> StyleElement:
    """Return the element class of a document block."""

    return _BLOCK_ELEMENTS.get(block.type, StyleElement.PARAGRAPH)


def style_css(style: ResolvedStyle) -> str:
    """Serialize a resolved style into CSS declarations.

    Every value has already passed its role grammar, so no escaping beyond the
    quoting below is required and no caller-supplied CSS can reach the output.
    """

    declarations: list[str] = []
    if style.text_color is not None:
        declarations.append(f"color: {style.text_color}")
    if style.background_color is not None:
        declarations.append(f"background-color: {style.background_color}")
    if style.font_family is not None:
        declarations.append(f"font-family: '{style.font_family}'")
    if style.font_size_pt is not None:
        declarations.append(f"font-size: {style.font_size_pt:g}pt")
    if style.bold:
        declarations.append("font-weight: bold")
    if style.italic:
        declarations.append("font-style: italic")
    return "; ".join(declarations)


def hex_digits(color: str) -> str:
    """Return the bare `RRGGBB` digits of a validated colour value."""

    return color.lstrip("#").upper()
