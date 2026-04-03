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

The wizard walks you through:
1. Choosing a data directory (default: `~/.phileas`)
2. Picking an LLM provider (optional -- you can skip and add one later)
3. Downloading the embedding model (`all-MiniLM-L6-v2`)
4. Downloading the reranker model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
5. Testing the LLM connection (if configured)

Expected output:

```
Welcome to Phileas -- long-term memory for AI companions.

Where should Phileas store data? [~/.phileas]:
LLM provider (used for extraction, consolidation, contradiction detection):
  anthropic  -- Claude models via Anthropic API
  openai     -- GPT models via OpenAI API
  ollama     -- Local models via Ollama
  skip       -- Configure later

LLM provider [skip]:

Wrote /home/you/.phileas/config.toml

Downloading models ...
  Downloading embedding model all-MiniLM-L6-v2 ...
  Downloading reranker model cross-encoder/ms-marco-MiniLM-L-6-v2 ...

Phileas is ready.

Suggested next steps:
  phileas remember "I prefer Python over JavaScript"
  phileas recall "programming languages"
  phileas status
```

## 3. Store some memories

```bash
phileas remember "I'm a backend engineer who loves distributed systems"
```

```
Stored [a1b2c3d4] [knowledge] I'm a backend engineer who loves distributed systems
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

```
Results for 'what do I work on'
  [a1b2c3d4] [knowledge] I'm a backend engineer who loves distributed systems  (score=0.82)
  [e5f6g7h8] [event] Started new job at Acme Corp in March 2026  (score=0.71)
```

Filter by type:

```bash
phileas recall "who am I" --type profile
```

## 5. Browse and inspect

List all memories:

```bash
phileas list
```

Show full detail:

```bash
phileas show a1b2c3d4
```

Check system health:

```bash
phileas status
```

```
Phileas Status
  Total memories:     4
  Active tier-2:      4
  Active tier-3:      0
  Archived:           0
  Vector embeddings:  4
  Graph nodes:        0
  Graph edges:        0
```

## 6. Connect to an AI

Start the MCP server:

```bash
phileas serve
```

Add Phileas to Claude Code by editing `~/.claude/settings.json`:

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

Now Claude Code can store and recall memories about you across sessions. See [MCP Integration](mcp-integration.md) for more details and other AI clients.

## Next steps

- [Configuration](configuration.md) -- Customize scoring weights, retrieval thresholds, and LLM settings
- [CLI Reference](cli-reference.md) -- All commands with full options
- [LLM Setup](llm-setup.md) -- Configure Anthropic, OpenAI, or Ollama for smart features
