# Gemini — Tool-Specific Instructions

> **All project context lives in `AGENTS.md`.** Read it FIRST — architecture,
> config, source-of-truth (`config.py`), key technical decisions, and workflows
> are all there. This file carries only Gemini-specific notes and must NEVER
> duplicate project facts. (Full copies here are exactly what drifted out of sync
> with `AGENTS.md` before; keep this a pointer, not a copy.)

## Conventions

- **NO AI watermarks.** Never add `Co-Authored-By`, "Generated with AI", or any
  mention of AI in commits, code, or docs. (Full user-preferences list in `AGENTS.md`.)
- Gemini does not use the dual-graph MCP retrieval protocol — rely on `AGENTS.md`,
  `config.py`, and `CONTEXT.md` for context.
