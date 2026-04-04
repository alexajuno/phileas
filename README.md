# Phileas -- Long-term memory for AI companions

Your AI forgets you every session. Phileas fixes that.

Phileas is a local-first memory system that gives AI companions persistent, intelligent memory. It runs on your machine, stores everything locally, and connects to any AI via [MCP](https://modelcontextprotocol.io/).

Named after Phileas Fogg -- a steadfast partner for the journey.

## Quick start

```bash
pip install phileas-memory
phileas init
phileas remember "I'm a backend engineer who loves distributed systems"
phileas recall "what do I work on"
```

## Features

- **100% local** -- your memories never leave your machine
- **Smart** -- auto-extracts facts, scores importance, and detects contradictions (with optional LLM)
- **Connected** -- knowledge graph links people, projects, and concepts
- **Works with any AI** -- Claude, GPT, Ollama, or any MCP-compatible client
- **Fast** -- semantic search + graph traversal + cross-encoder reranking + MMR diversity

## How it works

Phileas uses a triple-store architecture -- each store handles what it's good at:

| Store | Tech | Role |
|-------|------|------|
| **Relational** | SQLite | Memories, metadata, importance scoring, tiers, session tracking |
| **Vector** | ChromaDB | Semantic search via sentence-transformers embeddings |
| **Graph** | KuzuDB | Entity relationships and knowledge graph traversal |

The `MemoryEngine` orchestrates all three. When you store a memory, it gets persisted to SQLite, embedded in ChromaDB, and linked into the knowledge graph in KuzuDB. When you recall, candidates are gathered from all three paths, reranked by a cross-encoder, and diversity-selected via MMR.

## CLI commands

| Command | Description |
|---------|-------------|
| `phileas init` | Interactive setup wizard |
| `phileas remember "text"` | Store a memory |
| `phileas recall "query"` | Search memories |
| `phileas forget <id>` | Archive a memory |
| `phileas update <id> "text"` | Update a memory's content |
| `phileas list` | Browse all memories |
| `phileas show <id>` | Show full detail of a memory |
| `phileas ingest <source>` | Extract memories from text or a file (requires LLM) |
| `phileas consolidate` | Merge similar memories into summaries (requires LLM) |
| `phileas contradictions` | Find conflicting memories (requires LLM) |
| `phileas export` | Export memories as JSON |
| `phileas serve` | Start MCP server |
| `phileas start` | Start background daemon (keeps models loaded for fast CLI) |
| `phileas stop` | Stop the daemon |
| `phileas usage` | Show LLM token usage, cost, and request breakdown |
| `phileas status` | Show system health and stats |

## Setup modes

`phileas init` offers three modes:

1. **Claude Code** -- Claude is the brain, Phileas stores memories. Auto-configures `~/.claude/.mcp.json`.
2. **Standalone CLI** -- Phileas uses an LLM API (OpenAI, Anthropic, Ollama) for smart features.
3. **Both** -- Claude Code integration + standalone CLI access.

## Performance

For faster CLI commands, start the daemon to keep models loaded in memory:

```bash
phileas start    # Models load once, CLI commands become instant
phileas stop     # When you're done
```

Without the daemon, each CLI command loads models from scratch (~1-2s overhead).

## Connect to an AI

`phileas init` (mode 1 or 3) auto-configures Claude Code. Or manually:

```bash
phileas serve
```

Add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "phileas": {
      "command": "phileas",
      "args": ["serve"]
    }
  }
}
```

See [MCP Integration](docs/mcp-integration.md) for other clients.

## Documentation

- [Quick Start](docs/quickstart.md) -- 5-minute guided tutorial
- [Configuration](docs/configuration.md) -- Full config.toml reference
- [CLI Reference](docs/cli-reference.md) -- All commands with options and examples
- [LLM Setup](docs/llm-setup.md) -- Provider guides for Anthropic, OpenAI, Ollama
- [MCP Integration](docs/mcp-integration.md) -- Connecting Phileas to AI clients

## Requirements

Python 3.14+

## License

MIT
