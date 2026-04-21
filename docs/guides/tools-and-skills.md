# Tools and skills

LewLM exposes both **built-in skills** and **local executable tools**.

## Built-in skills

Built-in skills are deterministic document transforms. The current catalog includes:

- contract text replacement
- receipt extraction
- branded document template
- file template rendering
- document comparison
- OCR-assisted extraction
- meeting transcript notes
- long document memo
- speech transcript cleanup

Use:

```bash
lewlm list-skills
lewlm show-skill <skill-name>
```

Or:

- `GET /v1/skills`
- `GET /v1/skills/{skill_name}`

## Local tools

The current local tool catalog is intentionally small and explicit:

- `documents.generate`
- `documents.ingest`
- `documents.transform`

Use:

```bash
lewlm list-tools
lewlm show-tool <tool-name>
lewlm run-tool request.json
```

Or:

- `GET /v1/tools`
- `GET /v1/tools/{tool_name}`
- `POST /v1/tools/execute`

Python surfaces:

- `LewLM.list_tools()` / `LewLM.get_tool()` / `LewLM.execute_tool()`
- `LewLMAppClient.list_tools()` / `LewLMAppClient.get_tool()` / `LewLMAppClient.execute_tool()`

For a minimal app-shaped prove-out, see `examples/app_starter_proofs.py local-tool-app`.

## Authorization model

Local tool-like operations can require explicit authorization when:

```bash
LEWLM_TOOL_AUTHORIZATION_REQUIRED=true
```

Then:

- API payloads must include `authorized_actions`
- CLI flows must pass `--authorize <action>`

Relevant actions today:

- `document_generate`
- `document_ingest`
- `document_transform`
- `model_conversion`

## Sandbox behavior

Tool execution can run in spawned worker processes under:

- `data_dir/tmp/tool-sandbox`

Controls:

- `LEWLM_TOOL_SANDBOX_ENABLED`
- `LEWLM_TOOL_SANDBOX_TIMEOUT_SECONDS`
- `LEWLM_TOOL_SANDBOX_CLEAR_ENVIRONMENT`

## Idempotency

Many tool and document request shapes also support `idempotency_key`, which lets LewLM distinguish a new execution from a replay of an equivalent request.

## Related pages

- [Prompting and MCP-style tools](prompting-and-mcp-tools.md)
- [Document formats and skills](../reference/document-formats-and-skills.md)
- [Security](../security.md)
