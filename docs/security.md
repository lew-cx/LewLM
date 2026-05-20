# LewLM Security Notes

LewLM is built with local-first defaults and currently enforces the following guardrails:

- outbound network access is disabled by default
- server API keys can be required with `LEWLM_API_KEY_REQUIRED=true`
- request bodies are limited by `LEWLM_REQUEST_MAX_BYTES`
- HTTP traffic is rate-limited with `LEWLM_RATE_LIMIT_REQUESTS` over `LEWLM_RATE_LIMIT_WINDOW_SECONDS`
- request content types are restricted per endpoint, with multipart enabled only on the multimodal upload paths
- document template and image paths are scoped to configured local roots
- model scan overrides through the API are scoped to configured model roots
- audio transcription requests accept either base64-encoded JSON or multipart file uploads and validate audio bytes against supported signatures before runtime dispatch
- chat and responses attachment parts can reference scoped local files or secure per-request multipart uploads before document ingest or audio transcription runs
- conversion work runs in per-job temporary workspaces under `data_dir/tmp`
- document ingest parsers run in spawned worker processes by default with `LEWLM_PARSER_SANDBOX_TIMEOUT_SECONDS` controlling the timeout, `LEWLM_PARSER_SANDBOX_CLEAR_ENVIRONMENT` controlling inherited environment clearing, and isolated worker roots under `data_dir/tmp/parser-sandbox`
- local tool execution runs in spawned worker processes by default with `LEWLM_TOOL_SANDBOX_TIMEOUT_SECONDS` controlling the timeout, `LEWLM_TOOL_SANDBOX_CLEAR_ENVIRONMENT` controlling inherited environment clearing, and isolated worker roots under `data_dir/tmp/tool-sandbox`
- conversion backends run in spawned worker processes by default with `LEWLM_CONVERSION_SANDBOX_TIMEOUT_SECONDS` controlling the timeout, `LEWLM_CONVERSION_SANDBOX_CLEAR_ENVIRONMENT` controlling inherited environment clearing, and isolated worker roots under `data_dir/tmp/conversion-sandbox`
- audit logging can be enabled with `LEWLM_AUDIT_LOG_ENABLED=true`
- explicit authorization gates for document and conversion operations can be enabled with `LEWLM_TOOL_AUTHORIZATION_REQUIRED=true`
- metadata-store payloads, audit-log lines, and conversion cache artifacts can be encrypted with `LEWLM_PERSISTENCE_ENCRYPTION_ENABLED=true` and `LEWLM_PERSISTENCE_ENCRYPTION_PASSPHRASE`

## File access roots

By default, API-driven file access is scoped to `data_dir`. Override it with `LEWLM_FILE_ACCESS_ROOTS` when the server must read templates or document assets from other local directories.

CLI flows use stricter per-request scopes derived from the files you pass in:

- `generate-doc` reads the JSON IR file you specify and scopes document asset paths to that file's directory tree
- `transform` reads the request JSON you specify and scopes template access to that file's directory tree
- `chat --system-prompt-file` only reads the explicit prompt file you pass in
- `chat --attach-file`, `--attach-image`, and `--attach-audio` scope attachment reads to the explicit files you pass in

API chat and responses requests can now reference local files through JSON content-part arrays or upload files through multipart requests. Local paths are validated against `LEWLM_FILE_ACCESS_ROOTS`, while multipart uploads are copied into secure per-request workspaces under `data_dir/tmp` and removed when the request finishes.

When `LEWLM_TOOL_AUTHORIZATION_REQUIRED=true` is enabled, the following flows also require an explicit action grant:

- API payloads must include `authorized_actions`, such as `["document_generate"]`, `["document_transform"]`, `["document_ingest"]`, or `["model_conversion"]`
- CLI commands must include `--authorize <action>` for `generate-doc`, `transform`, and `convert`

## Multipart upload handling

`POST /v1/chat/completions` and `POST /v1/responses` accept multipart requests with a `payload_json` form field plus uniquely named file parts. Content parts can then reference those uploads by `upload_name`.

LewLM writes each uploaded file into a secure per-request workspace before attachment normalization, image/document ingest, or audio transcription runs. Those workspaces are deleted when the request completes, including streamed chat/response requests after the stream closes, and they are also cleaned up when multipart parsing or downstream attachment validation fails.

## Audio request handling

`/v1/audio/transcriptions` accepts either JSON with `audio_base64` or multipart form uploads with a `file` part. LewLM currently accepts WAV, FLAC, OGG, and MP3 signatures before routing the request to a compatible audio runtime.

## Encrypted persistence

When `LEWLM_PERSISTENCE_ENCRYPTION_ENABLED=true` is enabled, LewLM derives a local encryption key from `LEWLM_PERSISTENCE_ENCRYPTION_PASSPHRASE` and a salt stored at `data_dir/keys/persistence.salt`.

The current implementation encrypts:

- `app_kv` values in the metadata database
- conversion job payloads
- conversion artifact output-path and metadata payloads
- conversion cache artifacts stored on disk as encrypted `.lewlmcache` archives
- model manifest payload JSON
- stored registry source paths, with keyed digests used in lookup columns
- audit-log lines when audit logging is enabled

When encryption is enabled, LewLM also migrates existing plain conversion cache entries into encrypted archives the next time the service starts.

This covers LewLM's current structured metadata and cached conversion outputs, but it is not whole-database SQLite encryption and any future history store will need its own encrypted persistence path.

## Audit log mode

When audit logging is enabled, LewLM writes structured JSONL records to `data_dir/logs/audit.jsonl` with restrictive file permissions. The current implementation records:

- denied and authorized tool-gate checks
- failed HTTP requests that return structured LewLM errors
- document generate / ingest / transform successes and local-tool execution failures
- model scan completion
- conversion queue, cache-hit, completion, and failure events

Audit-log details are redacted before they hit disk:

- secrets such as API keys, tokens, passphrases, and authorization values are replaced with deterministic redaction markers
- prompt-like freeform text is hashed and length-tagged instead of stored verbatim
- filesystem paths and roots are reduced to basename-plus-digest markers so operators can correlate records without leaking directory layouts

## Release-readiness helpers

Generate a lightweight SBOM for the current environment with:

```bash
python scripts/generate_sbom.py > out/sbom.json
```

Generate a dependency-consistency and reproducibility audit with:

```bash
python scripts/generate_dependency_audit.py > out/dependency-audit.json
```

Generate a richer release manifest for host validation and reproducible packaging with:

```bash
python scripts/generate_release_manifest.py > out/release-manifest.json
```

The dependency audit records `pip check` results plus normalized digests for the declared dependency spec in `pyproject.toml` and the resolved environment from `pip freeze`.

The release manifest captures the current git commit (when available), `pip freeze`, the dependency-audit summary, redacted runtime configuration, runtime readiness diagnostics, `frontier_acceptance`, `optimization_defaults`, `performance_core_acceptance`, and the generated SBOM in one JSON artifact. Release builds should capture the resolved environment alongside the generated SBOM and dependency audit.

Capture the full local release bundle in one step with:

```bash
python scripts/capture_release_bundle.py --output-dir out --require-target Darwin:arm64 --minimum-verified-models 1
```

The bundle capture writes `sbom.json`, `dependency-audit.json`, `release-manifest.json`, `release-candidate-validation.json`, and `release-artifact-index.json`. The artifact index records SHA256 digests and sizes for the generated artifacts, plus any external validation-manifest inputs supplied to the bundle workflow.

When you pass `--validation-manifest-path`, the bundle capture copies those external release manifests into `validation-manifests/` inside the output bundle before validation so the resulting bundle remains portable across machines and handoffs.

For Milestone 118-style host proof capture, use `python scripts/capture_host_validation.py --output-dir out/host-validation --capture-all-capabilities ...` to collect config, scan/list, doctor, capability, release-bundle, and optional loopback `/v1/` probe evidence in one machine-readable workspace without committing local artifacts.

Validate a multi-host release candidate set with:

```bash
python scripts/validate_release_candidate.py out \
  --require-target Darwin:arm64 \
  --require-target Linux:x86_64 \
  --require-target Windows:AMD64 \
  --minimum-verified-models 1 \
  --require-performance-core-pillar serving_core \
  --require-performance-core-pillar continuous_batching \
  > out/release-candidate-validation.json
```

The validator loads LewLM release manifests from the provided files or directories, ignores non-release JSON artifacts such as SBOM and dependency-audit outputs, and exits non-zero when the manifest set shows git-commit drift, dependency-audit inconsistency, missing host-probe verification, missing required systems or exact `SYSTEM:MACHINE` targets, insufficient verified-model coverage, missing required frontier families, missing required optimization-default classes, or missing required performance-core proof pillars for the enforced targets.
