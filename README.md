# Phileas

Local-first long-term memory for AI companions. Named after Phileas Fogg — a steadfast partner for the journey.

Phileas is the persistent layer that lets an AI actually *know you* over time — remembering, forgetting naturally, and adapting as you change. It runs as an [MCP server](https://modelcontextprotocol.io/) that any compatible AI client (Claude Code, etc.) can connect to.

## Architecture

Triple-store backend — each store handles what it's good at:

| Store | Tech | Role |
|-------|------|------|
| **Relational** | SQLite | Memories, metadata, importance scoring, tiers, session tracking |
| **Vector** | ChromaDB | Semantic search via sentence-transformers embeddings |
| **Graph** | KuzuDB | Entity relationships and knowledge graph traversal |

The `MemoryEngine` orchestrates all three, so tools interact with a single interface.

## Setup

Requires Python 3.14+.

```bash
# Install
pip install -e .

# Or with uv
uv pip install -e .
```

Add to your MCP client config (e.g. `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "phileas": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/phileas", "mcp", "run", "src/phileas/server.py"]
    }
  }
}
```

## Status

Core engine is built and running. Active areas:

- Memory consolidation (tier-2 clustering into tier-3 summaries)
- Decay and natural forgetting
- Richer graph queries

## License

MIT
