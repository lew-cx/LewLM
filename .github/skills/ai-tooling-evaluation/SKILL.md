---
name: ai-tooling-evaluation
description: "Evaluate or design AI tooling for VS Code Copilot: custom agents, skills, instructions, hooks, MCP servers, plugins, prompts, model routing, and workflow automation. Use when setting up or improving an AI-assisted development environment."
argument-hint: "Describe the workflow or AI tooling problem"
---

# AI Tooling Evaluation

Use this skill when deciding how to improve the AI development environment for this workspace.

## Procedure

1. Define the workflow pain point in one sentence.
2. Check existing `.github\agents`, `.github\skills`, `.github\instructions`, `.github\hooks`, and `.github\prompts` before adding anything.
3. Pick the smallest primitive that solves the problem:
   - **Instruction**: guidance that should be loaded automatically.
   - **Skill**: repeatable multi-step workflow with optional references or scripts.
   - **Custom agent**: focused role, context isolation, or restricted tools.
   - **Hook**: deterministic lifecycle enforcement or context injection.
   - **MCP server/plugin**: only when the workflow needs live external data, APIs, tools, or state that files cannot provide.
4. Keep descriptions keyword-rich and include trigger phrases such as "Use when: ...".
5. Validate new files:
   - JSON hooks parse cleanly.
   - Markdown customization files have valid frontmatter when required.
   - Skill folder name matches the `name` field.

## MCP and plugin threshold

Do not add an MCP server or plugin just because it is possible. Recommend one only when at least one of these is true:

- The agent needs authenticated access to a system outside the repo.
- The workflow needs live state that changes independently of files.
- Existing CLI/file workflows are too slow, brittle, or unsafe.
- The integration can be configured without committing secrets.

## Output

Return a short recommendation with:

- chosen primitive
- files to create or update
- why MCP/plugin is or is not justified
- validation command or check
