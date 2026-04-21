"""Validation helpers for document IR payloads."""

from __future__ import annotations

from pathlib import Path

from lewlm.core.errors import DocumentValidationError
from lewlm.documents.ir.models import DocumentIR, ImageBlock, TableBlock


class DocumentIRValidator:
    """Validate document IR objects before rendering."""

    def validate(self, document: DocumentIR) -> DocumentIR:
        if not document.title.strip():
            raise DocumentValidationError("Document title cannot be empty.")
        if not document.sections:
            raise DocumentValidationError("Document must contain at least one section.")

        for section_index, section in enumerate(document.sections):
            if not section.blocks:
                raise DocumentValidationError(
                    "Document sections must contain at least one block.",
                    details={"section_index": section_index},
                )
            for block_index, block in enumerate(section.blocks):
                if isinstance(block, TableBlock):
                    self._validate_table(section_index, block_index, block)
                if isinstance(block, ImageBlock):
                    self._validate_image(section_index, block_index, block)
        return document

    def validate_csv_document(self, document: DocumentIR) -> TableBlock:
        self.validate(document)
        for section in document.sections:
            for block in section.blocks:
                if isinstance(block, TableBlock):
                    return block
        raise DocumentValidationError("CSV rendering requires at least one table block.")

    def _validate_table(self, section_index: int, block_index: int, block: TableBlock) -> None:
        if not block.rows:
            raise DocumentValidationError(
                "Table blocks must contain at least one row.",
                details={"section_index": section_index, "block_index": block_index},
            )
        column_count = len(block.headers) if block.headers else len(block.rows[0])
        if column_count == 0:
            raise DocumentValidationError(
                "Table blocks must contain at least one column.",
                details={"section_index": section_index, "block_index": block_index},
            )
        for row_index, row in enumerate(block.rows):
            if len(row) != column_count:
                raise DocumentValidationError(
                    "Table rows must all have the same number of columns.",
                    details={
                        "section_index": section_index,
                        "block_index": block_index,
                        "row_index": row_index,
                        "expected_columns": column_count,
                        "actual_columns": len(row),
                    },
                )

    def _validate_image(self, section_index: int, block_index: int, block: ImageBlock) -> None:
        if block.path is None:
            return
        image_path = Path(block.path).expanduser()
        if not image_path.exists():
            raise DocumentValidationError(
                "Image block references a missing file.",
                details={
                    "section_index": section_index,
                    "block_index": block_index,
                    "path": str(image_path),
                },
            )
