# Phileas — persistent memory for AI

AI conversations reset every session. Phileas is a local memory layer that any LLM can read and write through [MCP](https://modelcontextprotocol.io/), so context survives across sessions and tools.

## Get started

```bash
pip install phileas-memory
phileas init
```

The setup wizard connects Phileas to your MCP client (Claude, GPT, Ollama, or any other) and chooses where to store memories.

## Connect to your AI

If you use Claude Code, `phileas init` handles this automatically.

For other MCP clients, start the server and point your client at it:

```bash
phileas serve
```

## Principles

- **Local-first** — memories stay on your machine
- **Model-agnostic** — works with any LLM via MCP
- **Natural forgetting** — irrelevant detail decays; recall favors what stays useful
- **Open** — run it yourself, read the code

For the command reference, run `phileas --help` or `phileas COMMAND --help`.

## Requirements

Python 3.14+

## License

MIT
