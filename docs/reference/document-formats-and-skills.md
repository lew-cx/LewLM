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

## Style tokens

`DocumentIR.style_tokens` declares document-level tokens. Sections, blocks,
headers, footers, and citations reference them by name through their own
`style_tokens` lists, and a section's references cascade to its blocks.

Only the reserved names below change rendered output. Any other declared token
is accepted, validated for nothing beyond a non-empty unique name, echoed in
JSON, and never reaches a renderer — that keeps ingestion provenance tokens such
as `code`, `ocr`, and source DOCX paragraph style names inert. LewLM does not
accept raw CSS or executable styling: a reserved token value is a validated
colour, a validated font family, a bounded point size, or a closed keyword.

| Reserved name | Value grammar | Default element scope |
| --- | --- | --- |
| `body_color` | `#RRGGBB` | body content |
| `body_font` | font family of letters, digits, spaces, `.`, `_`, `-` (≤64 chars) | body content |
| `body_font_size_pt` | number between 4 and 96 | body content |
| `heading_color` | `#RRGGBB` | headings |
| `heading_font` | font family | headings |
| `heading_font_size_pt` | number between 4 and 96 | headings |
| `surface_color` | `#RRGGBB` | callouts |
| `accent_color` | `#RRGGBB` | callout rules and table header fills |
| `emphasis` | `normal`, `bold`, `italic`, `bold_italic` | referenced elements only |

`applies_to` restricts a token to one element class: `document` (or omitted)
keeps the default scope, and `heading`, `paragraph`, `list`, `table`, `callout`,
`image`, `citation`, `header`, or `footer` narrows it to that class. An element
that references a token by name applies it regardless of scope. Body roles
resolve before heading roles, so a heading wins when an element opts into both.

### Per-format behaviour

| Format | Colour | Font family | Point size | Emphasis | Accent |
| --- | --- | --- | --- | --- | --- |
| `text` | inert | inert | inert | inert | inert |
| `markdown` | inert | inert | inert | `**`/`*` wrapping | inert |
| `json` | echoed verbatim, not interpreted | echoed | echoed | echoed | echoed |
| `csv` | inert | inert | inert | inert | inert |
| `docx` | run font colour and `w:shd` shading | run font name | run font size | run bold/italic | table header cell shading |
| `pdf` | CSS colour and background | CSS font family | CSS font size | CSS weight/style | callout rule and table header fill |
| `xlsx` | cell font colour and solid fill | cell font name | cell font size | cell bold/italic | table header cell fill |

PDF style support above describes the WeasyPrint path. The ReportLab fallback
honours colour, point size, emphasis, and accent fills but ignores font family
tokens because it can only lay out its built-in font set.

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
- unique, non-empty style-token names
- reserved style-token values and scopes that match the published vocabulary
