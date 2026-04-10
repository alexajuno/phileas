# Phileas — long-term memory for AI companions

Your AI forgets you every session. You can talk to the most capable model in the world, but it doesn't *know* you. No continuity. No relationship that deepens over time.

The models are good enough. What's missing is the layer around them — the memory, the context, the sense of who you are and where you've been.

Phileas is that layer. It runs on your machine, stores everything locally, and connects to any AI through [MCP](https://modelcontextprotocol.io/).

Named after Phileas Fogg — a companion for the journey.

## Get started

```bash
pip install phileas-memory
phileas init
```

The setup wizard walks you through connecting to your AI (Claude, GPT, Ollama, or any MCP client) and choosing where to store your memories. That's it.

## Connect to your AI

If you use Claude Code, `phileas init` handles this automatically.

For other MCP clients, start the server and point your client at it:

```bash
phileas serve
```

See [MCP Integration](docs/mcp-integration.md) for client-specific setup.

## What it believes

- **Local-first** — your memories stay on your machine
- **Model-agnostic** — works with any LLM
- **Human-like, not perfect** — remembers what matters, lets the rest fade
- **Open** — run it yourself, see how it works

## Learn more

- [Quick Start](docs/quickstart.md) — guided tutorial
- [CLI Reference](docs/cli-reference.md) — all commands
- [Configuration](docs/configuration.md) — config.toml reference
- [LLM Setup](docs/llm-setup.md) — provider guides
- [MCP Integration](docs/mcp-integration.md) — connecting to AI clients

## Requirements

Python 3.14+

## License

MIT
