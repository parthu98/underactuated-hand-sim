# Claude Code — Tool-Specific Instructions

> **All project context lives in `AGENTS.md`.** This file contains ONLY
> Claude-specific dual-graph MCP mechanics. Never duplicate project info here.

## Dual-Graph retrieval order (MANDATORY)

1. **Call `graph_continue` first** — before any file exploration, grep, or reading.
2. If it returns `needs_project=true` → call `graph_scan` with `pwd`. Don't ask.
3. If it returns `skip=true` (<5 files) → no broad/recursive exploration; read named
   files only, or ask what to work on.
4. **Read `recommended_files` with `graph_read` — one call per file.** Accepts a
   single `file` string (not an array). `file::symbol` entries (e.g.
   `analytical_model.py::solve`) read only that symbol's lines — pass verbatim.
5. **Obey `confidence` caps strictly:**
   - `high` → stop; no grep/exploration.
   - `medium` → if recommended files insufficient, `fallback_rg` ≤ `max_supplementary_greps`,
     then `graph_read` ≤ `max_supplementary_files`. Then stop.
   - `low` → `fallback_rg` ≤ caps, then `graph_read` ≤ caps. Then stop.

## Rules

- No `rg`/`grep`/bash file exploration before `graph_continue`.
- No broad/recursive exploration at any confidence level.
- `max_supplementary_*` are hard caps — never exceed.
- Don't dump full chat history. Don't call `graph_retrieve` more than once per turn.
- After edits, call `graph_register_edit` with changed files (use `file::symbol`).

## Memory (see AGENTS.md for the policy)

- Log decisions/tasks/blockers immediately via
  `graph_add_memory(type=..., content="<15 words", tags=[...], files=[...])`.
- On session end, (re)write `CONTEXT.md` per the format in `AGENTS.md`.
