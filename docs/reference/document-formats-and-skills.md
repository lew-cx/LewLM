# Document formats and skills

## Document output formats

LewLM can render deterministic artifacts in:

| Format | Extension |
| --- | --- |
| `text` | `.txt` |
| `markdown` | `.md` |
| `json` | `.json` |
| `csv` | `.csv` |
| `docx` | `.docx` |
| `pdf` | `.pdf` |
| `xlsx` | `.xlsx` |

## Document IR blocks

| Block type | Purpose |
| --- | --- |
| `paragraph` | plain text paragraphs |
| `table` | structured headers and rows |
| `list` | ordered or unordered list content |
| `callout` | info/warning/success/note callouts |
| `image` | local image or logo placement |

Supporting structures include:

- style tokens
- headers and footers
- citations
- per-block metadata

## Built-in skill catalog

| Skill | Primary use | Example file |
| --- | --- | --- |
| `contract_text_replacement` | placeholder substitution in contract-like text | `examples/contract-transform.json` |
| `receipt_extraction` | normalize receipts to structured tables | `examples/receipt-transform.json` |
| `branded_document_template` | branded reports with optional logo and hero image | `examples/branded-document-template.json` |
| `file_template` | render a reusable `DocumentIR` template file | `examples/file-template-transform.json` |
| `document_comparison` | summarize shared and unique sections | `examples/document-compare-transform.json` |
| `ocr_assisted_extraction` | OCR text into structured fields | `examples/ocr-assisted-extraction.json` |
| `meeting_transcript_notes` | notes, decisions, action items | `examples/meeting-transcript-notes.json` |
| `long_document_memo` | highlights, questions, and outline from long text | `examples/long-document-memo.json` |
| `speech_transcript_cleanup` | clean speaker-attributed transcript text | `examples/speech-transcript-cleanup.json` |

## Local document tools

| Tool | Result type | Required authorization |
| --- | --- | --- |
| `documents.generate` | artifact | `document_generate` |
| `documents.ingest` | `document_ir` | `document_ingest` |
| `documents.transform` | artifact | `document_transform` |

## Validation expectations

Document validation enforces:

- non-empty titles
- at least one section
- at least one block per section
- table row widths that match the header width
- image paths that resolve inside allowed roots
