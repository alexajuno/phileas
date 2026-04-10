# README Rewrite — Design Spec

## Context

The current README is functional but robotic — it reads like a technical manual with feature lists, architecture tables, CLI command tables, and setup mode explanations. For a project that wants to be a *companion*, the first thing people see should feel human. This rewrite shifts the README from "what it does" to "what it wants to be", moves technical depth to docs, and adds a clear install path.

## Audience

AI power users first (people who use Claude/GPT daily and want continuity), developers second (they'll find the architecture in docs). Lead with the human story, let the technical quality show through brevity and confidence.

## Approach: Story-first

Open with the problem emotionally, flow into vision, land on install + connect. No tables, no CLI lists, no architecture diagrams in the README itself.

## Structure

### 1. Title + tagline
```
# Phileas — long-term memory for AI companions
```

### 2. Opening (the problem + what Phileas is)
3-4 short paragraphs borrowed from vision.md, adapted for README:
- Your AI forgets you every session
- The models are good enough — what's missing is the memory layer
- Phileas is that layer — local, connected via MCP
- Named after Phileas Fogg — a companion for the journey

No feature bullets. No jargon. Just the story.

### 3. Get started
```
pip install phileas-memory
phileas init
```
One sentence: "The setup wizard walks you through connecting to your AI and choosing a storage location."

### 4. Connect to your AI
- Claude Code: `phileas init` handles it automatically
- Other MCP clients: `phileas serve` + link to docs/mcp-integration.md
- No JSON config block in README

### 5. What it believes (design principles)
Four one-line bullets from vision.md:
- Local-first
- Model-agnostic
- Human-like memory (not perfect recall)
- Open

### 6. Learn more (docs links)
Clean list:
- Quick Start
- CLI Reference
- Configuration
- LLM Setup
- MCP Integration

### 7. Footer
- Requirements: Python 3.14+
- License: MIT

## What's removed from README
- CLI commands table (17 commands) → already in docs/cli-reference.md
- Architecture table (SQLite/ChromaDB/KuzuDB) → stays in docs/design.md
- Setup modes explanation (Claude Code / Standalone / Both) → covered by `phileas init` wizard and docs
- "How it works" section with MemoryEngine details
- "Performance" section about the daemon
- Manual MCP JSON config block
- Features bullet list

## What's new
- Emotional opening drawn from vision.md
- "What it believes" section (design principles, reframed)
- Cleaner "Get started" that trusts the wizard

## Files to modify
- `README.md` — full rewrite
- No new files needed (docs/cli-reference.md already exists and is comprehensive)

## Verification
1. Read the new README and confirm it flows naturally
2. Verify all doc links point to existing files
3. Check that `pip install phileas-memory` and `phileas init` are the correct commands (confirmed: pyproject.toml has `phileas-memory` as package name, CLI entry point is `phileas`)
4. Confirm docs/cli-reference.md covers all commands that were removed from README
