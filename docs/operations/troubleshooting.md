# Troubleshooting

## First commands to run

When something looks wrong, start with:

```bash
lewlm doctor
lewlm config
lewlm cache
```

And, if the API is running:

```bash
curl http://127.0.0.1:8080/v1/health
curl http://127.0.0.1:8080/v1/runtime/stats
```

## Common issues

### No models are found

Check:

- `LEWLM_MODELS_DIR`
- whether the files are in a supported local layout
- whether you scanned after moving files

### A runtime is unavailable

This usually means one of:

- the wrong install profile is active
- the host platform does not match the runtime
- the adapter endpoint is not reachable

Use `lewlm doctor`, `GET /v1/health`, and `lewlm capabilities <model-id>` to confirm.

### A model requires conversion

Look for `conversion_status: requires_conversion` and queue a conversion job before retrying the request.

### File access is denied

Check:

- `LEWLM_FILE_ACCESS_ROOTS`
- the exact files passed to the CLI
- whether multipart uploads or prompt files are inside allowed roots

### Tool execution is denied

If `LEWLM_TOOL_AUTHORIZATION_REQUIRED=true`, supply the correct authorization:

```bash
--authorize document_transform
```

Or add `authorized_actions` to the API payload.

### Requests fail with size or rate-limit errors

Review:

- `LEWLM_REQUEST_MAX_BYTES`
- `LEWLM_RATE_LIMIT_REQUESTS`
- `LEWLM_RATE_LIMIT_WINDOW_SECONDS`

## Best diagnostics surfaces

| Surface | Best for |
| --- | --- |
| `lewlm doctor` | install profile plus overall readiness |
| `lewlm config` | resolved settings |
| `GET /v1/health` | service, storage, install-profile, and readiness health |
| `GET /v1/cache/stats` | cache state and feature-level metrics |
| `GET /v1/runtime/stats` | runtime availability, schedulers, residency |
| `GET /v1/jobs/{job_id}` | conversion job state |
