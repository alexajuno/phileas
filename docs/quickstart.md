# Quick Start

Get Phileas running and store your first memories in under 5 minutes.

## 1. Install

```bash
pip install phileas-memory
```

Or with uv:

```bash
uv pip install phileas-memory
```

Requires Python 3.14+.

## 2. Initialize

Run the setup wizard:

```bash
phileas init
```

The wizard asks how you'll use Phileas and configures the rest accordingly:

1. Usage mode: with Claude Code, standalone CLI, or both
2. Data directory (default: `~/.phileas`)
3. LLM provider (standalone only — `anthropic`, `openai`, or `ollama`)
4. Claude Code wiring: writes the MCP entry to `~/.claude/.mcp.json`, installs the recall skill, and syncs hook state
5. Downloads the embedding model (`all-MiniLM-L6-v2`) and reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

API keys are read from environment variables — never written to disk.

## 3. Store some memories

```bash
phileas remember "I'm a backend engineer who loves distributed systems"
```

Try different memory types:

```bash
phileas remember "My name is Alex" --type profile --importance 9
phileas remember "Started new job at Acme Corp in March 2026" --type event --importance 7
phileas remember "I prefer tabs over spaces" --type behavior
```

Memory types: `profile`, `event`, `knowledge`, `behavior`, `reflection`.

## 4. Recall memories

```bash
phileas recall "what do I work on"
```

Options: `--top-k N` (max results) and `--type <type>` (filter by memory type).

## 5. Browse and inspect

```bash
phileas list             # list memories
phileas show <id>        # full detail of one memory
phileas status           # system health and counts
phileas stats            # LLM usage, memory, graph, hook stats
```

## 6. Connect to an AI

If you picked "with Claude Code" in step 2, the wizard already wired everything up — restart Claude Code and Phileas will recall and memorize automatically.

For other MCP clients, start the server and point your client at it:

```bash
phileas serve
```

See [MCP Integration](mcp-integration.md) for client-specific setup.

## Next steps

- [Configuration](configuration.md) -- Customize scoring weights, retrieval thresholds, and LLM settings
- [LLM Setup](llm-setup.md) -- Configure Anthropic, OpenAI, or Ollama for smart features

For the command reference, run `phileas --help` or `phileas COMMAND --help`.
