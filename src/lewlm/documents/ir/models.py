"""Structured document intermediate representation."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class DocumentOutputFormat(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    DOCX = "docx"
    PDF = "pdf"
    XLSX = "xlsx"

    @property
    def default_extension(self) -> str:
        if self == type(self).TEXT:
            return ".txt"
        if self == type(self).MARKDOWN:
            return ".md"
        return f".{self.value}"


class StyleToken(BaseModel):
    name: str
    value: str
    applies_to: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HeaderFooterContent(BaseModel):
    left: str | None = None
    center: str | None = None
    right: str | None = None
    style_tokens: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    label: str
    text: str
    url: str | None = None
    title: str | None = None
    style_tokens: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParagraphBlock(BaseModel):
    type: Literal["paragraph"] = "paragraph"
    text: str
    style_tokens: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: str | None = None
    style_tokens: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ListBlock(BaseModel):
    type: Literal["list"] = "list"
    ordered: bool = False
    items: list[str] = Field(default_factory=list)
    style_tokens: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalloutBlock(BaseModel):
    type: Literal["callout"] = "callout"
    kind: Literal["info", "warning", "success", "note"] = "info"
    title: str | None = None
    body: str
    style_tokens: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    alt_text: str
    path: str | None = None
    caption: str | None = None
    role: Literal["image", "logo"] = "image"
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    style_tokens: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


DocumentBlock = Annotated[
    ParagraphBlock | TableBlock | ListBlock | CalloutBlock | ImageBlock,
    Field(discriminator="type"),
]


class DocumentSection(BaseModel):
    heading: str | None = None
    level: int = 1
    style_tokens: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    blocks: list[DocumentBlock] = Field(default_factory=list)


class DocumentIR(BaseModel):
    title: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    style_tokens: list[StyleToken] = Field(default_factory=list)
    header: HeaderFooterContent | None = None
    footer: HeaderFooterContent | None = None
    sections: list[DocumentSection] = Field(default_factory=list)
    references_title: str = "References"
    citations: list[Citation] = Field(default_factory=list)
