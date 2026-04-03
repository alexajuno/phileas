# Phileas Public Release — Design Spec

**Date:** 2026-04-03
**Goal:** Make Phileas installable, configurable, and usable by anyone — not just its creator.

---

## Context

Phileas is a local-first long-term memory system for AI companions. It runs as an MCP server, storing memories in a triple-store backend (SQLite + ChromaDB + KuzuDB) with sophisticated multi-stage retrieval.

Today it works, but only for one person. Paths are hardcoded, configuration requires code edits, LLM intelligence depends on Claude Code doing the thinking, and there's no way to try it without understanding MCP internals.

This design reshapes Phileas into a standalone product that anyone can install, configure, and use — with or without an AI client.

---

## Design Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target audience | Broad — developers, AI users, any MCP client | Not just Claude Code power users |
| Front door | CLI + MCP | CLI for the aha moment, MCP for real AI integration |
| LLM support | Optional from day one, via litellm | Supports 100+ providers including Ollama for local models |
| LLM operations | All five: extraction, importance, recall rewriting, consolidation, contradiction detection | Full autonomy — Phileas thinks for itself |
| LLM provider abstraction | litellm | Battle-tested, zero provider-switching code |
| Packaging | PyPI + `phileas init` wizard | Smooth install, guided first-run |
| Migration | Auto-detect existing data, zero-transform | Existing memories preserved, trust matters for a memory product |
| Approach | Ship everything at once | First impression must show the full vision |

---

## 1. CLI & User Experience

### Command surface

**Setup:**
```bash
pip install phileas-memory
phileas init              # Interactive wizard
phileas status            # Health check
```

**Core memory operations:**
```bash
phileas remember "text"          # Store memory (LLM auto-scores importance)
phileas recall "query"           # Smart retrieval (LLM rewrites query)
phileas forget <memory-id>       # Soft delete (archive)
phileas update <memory-id> "new" # Edit in place
```

**Intelligence operations:**
```bash
phileas ingest <file-or-text>    # Extract memories from freeform text
phileas consolidate              # Find and merge similar memories
phileas contradictions           # Scan for conflicting memories
```

**Inspection & data management:**
```bash
phileas list                     # Browse memories (paginated, filterable)
phileas show <memory-id>         # Full detail of one memory
phileas export --format json     # Backup
phileas import <file>            # Restore
phileas graph                    # Entity relationship summary
```

**MCP server:**
```bash
phileas serve                    # Start MCP server
```

### The first 5 minutes

```
$ pip install phileas-memory
$ phileas init

  Welcome to Phileas — your long-term memory companion.

  Where should Phileas store data? [~/.phileas]:

  Configure an LLM provider? This enables smart features like
  auto-extraction, importance scoring, and contradiction detection.

  Provider (anthropic/openai/ollama/skip): anthropic
  API key: sk-ant-***
  Model [claude-haiku-4-5-20251001]:

  Downloading embedding model (all-MiniLM-L6-v2)... done (90MB)
  Downloading reranker model (ms-marco-MiniLM-L-6-v2)... done (91MB)
  Testing LLM connection... done

  Phileas is ready. Try:
    phileas remember "something about yourself"
    phileas recall "what do you know about me"

$ phileas remember "I'm a software engineer who loves building AI tools"
  Stored (importance: 7/10, type: profile)

$ phileas recall "what do I do"
  [1] I'm a software engineer who loves building AI tools
      importance: 7 | type: profile | stored: just now
```

---

## 2. Configuration System

### Config file: `~/.phileas/config.toml`

Created by `phileas init`. Editable by hand. Every hardcoded constant becomes configurable.

```toml
[storage]
home = "~/.phileas"              # Override with PHILEAS_HOME env var

[llm]
provider = "anthropic"           # Any litellm-supported provider
model = "claude-haiku-4-5-20251001"
api_key_env = "ANTHROPIC_API_KEY"    # Reads key from this env var (never stored in config)

[llm.operations]
# Override model per operation (optional — defaults to [llm].model)
# extraction = "claude-sonnet-4-6"
# consolidation = "claude-haiku-4-5-20251001"
# contradiction = "claude-haiku-4-5-20251001"
# importance = "claude-haiku-4-5-20251001"
# query_rewrite = "claude-haiku-4-5-20251001"

[embeddings]
model = "all-MiniLM-L6-v2"

[reranker]
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

[recall]
similarity_floor = 0.5
relevance_floor = 0.15
graph_boost = 0.5
mmr_lambda = 0.7
default_top_k = 10

[scoring]
relevance_weight = 0.55
importance_weight = 0.2
recency_weight = 0.15
access_weight = 0.1

[logging]
level = "INFO"
file_max_bytes = 5242880         # 5MB
file_backup_count = 3
```

### Config loading priority

env vars > config file > code defaults

### Key rules

- API keys are **never** stored in the config file — only the env var name
- Per-operation model overrides let users run cheap models for importance, strong models for extraction
- `PHILEAS_HOME` env var overrides everything — set it and all data lives under that directory
- Zero config works: Phileas runs with sensible defaults, minus LLM features which need a key

---

## 3. LLM Integration Layer

Five autonomous operations, all powered by litellm. Each has a prompt, structured I/O, and graceful fallback when no LLM is configured.

### 3.1 Memory Extraction

**Trigger:** `phileas ingest`, `phileas remember` with long text, MCP `ingest_session`

**Input:** Freeform text (conversation, notes, anything)

**Output:** Array of discrete memories with metadata:
```json
[
  {
    "summary": "Met with Sarah, CTO of Acme Corp",
    "type": "event",
    "importance": 5,
    "entities": ["Sarah", "Acme Corp"],
    "relationships": [["Sarah", "CTO_OF", "Acme Corp"]]
  }
]
```

**Fallback:** Store raw text as a single memory. No extraction, no auto-entities.

### 3.2 Auto-Importance Scoring

**Trigger:** Every `phileas remember` call

**Input:** Memory summary + type

**Output:** Importance score 1-10

**Prompt context:** "Rate this memory's long-term importance (1=trivial, 10=life-defining). Consider: is this a core identity fact? A key decision? A fleeting detail?"

**Fallback:** Default importance of 5. User can override with `--importance`.

### 3.3 Smart Recall (Query Rewriting)

**Trigger:** Every `phileas recall` call

**Input:** User's natural language query

**Output:** Array of rewritten search terms that fan out across keyword + semantic + graph search

**Example:**
```
Input:  "what tech stack am I using"
Output: ["technology stack", "programming languages", "frameworks",
         "database choices", "tools and infrastructure"]
```

**Fallback:** Use the raw query as-is (current behavior).

### 3.4 Consolidation

**Trigger:** `phileas consolidate` (manual) or automatically when active tier-2 memory count exceeds 100 (configurable via `[consolidation].auto_threshold` in config, set to 0 to disable)

**Input:** Cluster of similar tier-2 memories

**Output:** Single tier-3 memory summarizing the cluster

**Behavior:** Original tier-2 memories get `CONSOLIDATED_INTO` graph edges and are archived.

**Fallback:** Shows clusters but can't merge. Prints message explaining LLM is needed.

### 3.5 Contradiction Detection

**Trigger:** `phileas contradictions` or optionally on every `remember` call

**Input:** New memory + related existing memories (retrieved by recall)

**Output:**
```json
{
  "contradicts": true,
  "memory_ids": ["abc123"],
  "explanation": "Database choice changed from MongoDB to Postgres"
}
```

**Behavior (CLI):** User is prompted: keep both, archive old, or update old.
**Behavior (MCP):** Contradiction returned as metadata on the `memorize` response.

**Fallback:** No detection. Silent pass-through.

### LLM module architecture

```
src/phileas/llm/
  __init__.py          # LLMClient: litellm wrapper, retries, cost tracking
  extraction.py        # Memory extraction prompt + parsing
  importance.py        # Importance scoring prompt + parsing
  query_rewrite.py     # Query expansion prompt + parsing
  consolidation.py     # Cluster merging prompt + parsing
  contradiction.py     # Conflict detection prompt + parsing
  prompts/             # Prompt templates (separate files for easy tuning)
    extraction.txt
    importance.txt
    query_rewrite.txt
    consolidation.txt
    contradiction.txt
```

Each module: one prompt template, one `async def run(...)` function, one Pydantic model for structured output.

---

## 4. Architecture

### Current structure
```
src/phileas/
  server.py    engine.py    db.py    vector.py    graph.py
  scoring.py   models.py    logging.py   ingest.py
```

### New structure
```
src/phileas/
  cli/
    __init__.py        # Click app entry point
    commands.py        # All CLI commands
    formatter.py       # Pretty terminal output (Rich)
    wizard.py          # `phileas init` interactive setup

  llm/
    __init__.py        # LLMClient (litellm wrapper)
    extraction.py
    importance.py
    query_rewrite.py
    consolidation.py
    contradiction.py
    prompts/
      extraction.txt
      importance.txt
      query_rewrite.txt
      consolidation.txt
      contradiction.txt

  config.py            # Config loading: env -> toml -> defaults
  server.py            # MCP server (calls engine, same as CLI)
  engine.py            # Core orchestrator (gains LLM calls, reads config)
  db.py                # SQLite (unchanged)
  vector.py            # ChromaDB (unchanged)
  graph.py             # KuzuDB (unchanged)
  scoring.py           # Scoring (weights from config)
  models.py            # Pydantic models (extended for LLM outputs)
  logging.py           # Logging (config-driven)
  ingest.py            # Session parsing (uses llm/extraction)
  migrate.py           # Data migration utilities
```

### Architectural principle

**Engine is the single orchestrator.** Both CLI and MCP server call `engine.py`. The engine calls into `llm/` when configured, falls back gracefully when not. No business logic in CLI or server.

```
CLI commands --> engine.py --> db.py / vector.py / graph.py
MCP tools ----> engine.py --> llm/ (optional)
                   ^
              config.py
```

---

## 5. Migration & Backward Compatibility

### Existing data

- `phileas init` detects existing `memory.db`, `chroma/`, `graph/` in `~/.phileas/`
- Prompts to migrate (creates `config.toml` with defaults matching current hardcoded values)
- **Zero data transformation** — SQLite schema, ChromaDB collections, KuzuDB graph are unchanged
- Migration is purely about adding the config layer on top

### MCP server

- `phileas serve` replaces `mcp run src/phileas/server.py`
- All 12 existing MCP tools continue working identically
- Contradiction detection results returned as metadata on `memorize` responses
- Claude Code plugin/skill continues working — just update the MCP command

### Zero breaking changes

- If no `config.toml` exists, Phileas runs with hardcoded defaults (identical to today)
- Existing users who don't run `phileas init` see zero behavior change

---

## 6. Packaging & Distribution

### PyPI package

```toml
[project]
name = "phileas-memory"
version = "0.1.0"
description = "Local-first long-term memory for AI companions"
requires-python = ">=3.11"
dependencies = [
    "mcp[cli]",
    "sentence-transformers>=5.3.0",
    "chromadb>=1.0.0",
    "kuzu>=0.8.0",
    "litellm>=1.0.0",
    "click>=8.0",
    "rich>=13.0",
]

[project.scripts]
phileas = "phileas.cli:app"
```

### Python version

Drop from 3.14+ to **3.11+** to widen audience. Audit for 3.14-specific syntax.

### Model management

- `phileas init` downloads embedding + reranker models upfront with Rich progress bars
- Models cached in HuggingFace default cache (`~/.cache/huggingface/`)
- `phileas status` shows model availability and sizes

---

## 7. Documentation

### README.md — complete rewrite

Short, punchy, shows the value in 30 seconds:

```
# Phileas — Long-term memory for AI companions

Your AI forgets you every session. Phileas fixes that.

## Quick start
  pip install phileas-memory
  phileas init
  phileas remember "I'm a backend engineer working on distributed systems"
  phileas recall "what do I work on"

## Features
- 100% local — your memories never leave your machine
- Smart — auto-extracts, scores, and organizes memories
- Connected — knowledge graph links people, projects, and concepts
- Works with any AI — Claude, GPT, Ollama, or any MCP-compatible client
```

### Additional docs

| File | Purpose |
|------|---------|
| `docs/quickstart.md` | 5-minute guided tutorial |
| `docs/configuration.md` | Full config reference with examples |
| `docs/architecture.md` | How it works (cleaned up from existing design.md) |
| `docs/llm-setup.md` | Provider-specific setup guides (Anthropic, OpenAI, Ollama) |
| `docs/cli-reference.md` | All commands with examples |
| `docs/mcp-integration.md` | How to wire into AI clients |
| `docs/faq.md` | Common questions, troubleshooting |

---

## Component Summary

| Component | What | Status |
|-----------|------|--------|
| CLI | `phileas remember/recall/forget/ingest/consolidate/contradictions/list/show/export/import/graph/serve/init/status` | New |
| Config | `~/.phileas/config.toml` + `PHILEAS_HOME` env var | New |
| LLM layer | litellm, 5 autonomous operations, per-operation model config | New |
| Engine | Same interface, gains LLM calls, reads config | Modified |
| Storage | SQLite + ChromaDB + KuzuDB | Unchanged |
| MCP server | `phileas serve`, all 12 tools preserved | Modified |
| Migration | Auto-detect existing data, zero-transform, add config | New |
| Packaging | PyPI `phileas-memory`, Python 3.11+, Click + Rich | New |
| Docs | README, quickstart, config ref, CLI ref, FAQ, architecture, LLM setup, MCP guide | New |
