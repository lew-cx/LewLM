# Prompting and MCP-style tools

LewLM has an explicit prompt-compilation layer. It can combine model messages with structured overrides and return an inspectable prompt trace.

## Prompt override inputs

Prompt compilation accepts:

- `system_prompt`
- `developer_prompt`
- `pretext_path`
- `system_prompt_file_path` on the Python side
- `skills_path`
- `response_format` or `response_format_path`
- `output_schema` or `output_schema_path`
- `tools` or `tools_path`
- `mcp_tools` or `mcp_tools_path`
- `include_trace`

## Prompt tool definitions

LewLM distinguishes two related concepts:

| Type | Purpose |
| --- | --- |
| `PromptToolDefinition` | prompt-only tool metadata folded into the compiled prompt |
| `PromptMCPToolDefinition` | prompt-only MCP-style tool metadata that records a `server` plus the tool schema |

These are not the same as executing a real LewLM local tool. Prompt tool metadata influences prompt construction; local tool execution uses the tool catalog and authorization flow documented in [Tools and skills](tools-and-skills.md).

## Prompt trace

If `include_prompt_trace=true`, LewLM can return a trace with:

- selected prompt template
- message roles and attachment plan
- tool plan, including whether a tool is prompt-only or backed by a local tool
- output contract
- applied overrides and their sources

## MCP-style metadata file

The repository includes `examples/local-mcp-tools.json`, which shows the expected shape:

```json
{
  "mcp_tools": [
    {
      "server": "roadmap",
      "name": "search_milestones",
      "description": "Searches locally indexed milestone notes from an MCP-compatible catalog.",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"}
        },
        "required": ["query"]
      }
    }
  ]
}
```

## Structured outputs

LewLM can attach an output contract to a prompt via:

- inline `response_format`
- file-backed `response_format_path`
- inline `output_schema`
- file-backed `output_schema_path`

Prefer `response_format` / `response_format_path` for the stable shared contract. `output_schema` / `output_schema_path` remain available as legacy JSON-schema-only aliases.

## When to use what

| Need | Use |
| --- | --- |
| Add system/developer framing | prompt overrides |
| Attach reusable skill metadata to prompt construction | `skills_path` |
| Describe callable tools inside the prompt | `tools` / `tools_path` |
| Add MCP-style prompt metadata | `mcp_tools` / `mcp_tools_path` |
| Execute a local deterministic tool | tool catalog and `run-tool` / `/v1/tools/execute` |
