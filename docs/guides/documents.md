# Documents

LewLM includes three document-facing workflows:

1. **ingest** local files into structured `DocumentIR`
2. **generate** deterministic output artifacts from `DocumentIR`
3. **transform** structured inputs through built-in document skills

## Supported inputs

Document ingest can work with:

- TXT / Markdown
- PDF
- DOCX
- CSV
- XLSX
- images used in OCR-style flows

## Document IR

LewLM's `DocumentIR` model supports:

- sections
- paragraphs
- tables
- lists
- callouts
- images
- style tokens
- headers and footers
- citations

## Output formats

Generated artifacts can target:

- text
- markdown
- json
- csv
- docx
- pdf
- xlsx

## CLI workflows

### Render a deterministic artifact

```bash
lewlm generate-doc request.json --output out/report.pdf
```

### Run a built-in transform

```bash
lewlm transform examples/meeting-transcript-notes.json --output out/notes.md
```

## API workflows

- `POST /v1/documents/ingest`
- `POST /v1/documents/generate`
- `POST /v1/documents/transform`

The document APIs support `authorized_actions` and `idempotency_key` fields where relevant.

## Ingest packaging for apps

`POST /v1/documents/ingest` and `LewLM.ingest_documents()` return the parsed `document` plus two app-facing packaging lists:

- `sources[]` for one record per ingested file or bundle
- `chunks[]` for retrieval-ready text slices

Each source now includes:

- `source_id` for a stable machine key derived from the source path
- `source_label` and `source_name` for display and citation packaging
- `media_type` when LewLM can determine it
- parser-specific provenance in `metadata` such as page counts, OCR usage, or image counts

Each chunk now includes:

- `chunk_id` and `section_id` so downstream storage does not need to invent IDs
- `source_id`, `source_label`, and `section_label` so UI and citation layers can render useful labels directly
- `section_heading`, `section_level`, `source_path`, and `source_type`
- provenance in `metadata` such as `page_number`, `sheet_title`, `source_media_type`, `chunk_index`, and `char_count` when available

Document sections are also normalized with shared metadata keys such as `source_id`, `section_id`, `section_label`, `source_label`, and `section_index`, so chunk packaging and raw `DocumentIR` sections line up.

## OCR and file scoping

OCR-like extraction is part of the document pipeline, but file reads remain scoped:

- API file access is validated against configured file access roots
- CLI requests derive tighter per-request scopes from the explicit files you pass in

## Useful example files

| Example | Workflow |
| --- | --- |
| `examples/document.json` | raw `DocumentIR` rendering |
| `examples/contract-transform.json` | contract text replacement |
| `examples/branded-document-template.json` | styled template with images |
| `examples/meeting-transcript-notes.json` | transcript-to-notes |
| `examples/ocr-assisted-extraction.json` | OCR-style structured extraction |

See [Document formats and skills](../reference/document-formats-and-skills.md) for the exact block and skill tables.
