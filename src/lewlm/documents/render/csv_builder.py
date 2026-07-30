"""CSV document renderer."""

from __future__ import annotations

import csv
from io import StringIO

from lewlm.documents.ir.models import DocumentIR, DocumentOutputFormat
from lewlm.documents.render.base import DocumentRenderer
from lewlm.documents.validators.ir import DocumentIRValidator


class CsvDocumentRenderer(DocumentRenderer):
    """Render CSV.

    CSV carries tabular values only, so every style token is inert in this
    format by contract.
    """

    output_format = DocumentOutputFormat.CSV
    media_type = "text/csv"
    file_extension = ".csv"

    def __init__(self, validator: DocumentIRValidator) -> None:
        self.validator = validator

    def render(self, document: DocumentIR) -> bytes:
        table = self.validator.validate_csv_document(document)
        headers = table.headers or [f"column_{index + 1}" for index in range(len(table.rows[0]))]
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(headers)
        writer.writerows(table.rows)
        return buffer.getvalue().encode("utf-8")
