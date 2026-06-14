# Phileas — persistent memory for AI

AI conversations reset every session. Phileas is a local memory layer that any LLM can read and write through [MCP](https://modelcontextprotocol.io/), so context survives across sessions and tools.

## Get started

```bash
pip install phileas-memory
phileas init
```

The setup wizard connects Phileas to your MCP client (Claude, GPT, Ollama, or any other) and chooses where to store memories.

### First run

On first run, `phileas init` downloads two small models that run locally — an
embedding model (`all-MiniLM-L6-v2`) and a reranker
(`ms-marco-MiniLM-L-6-v2`), about 150 MB together — from
[Hugging Face](https://huggingface.co/). Expect a one-time wait on a slow
connection; they're cached afterward, so later runs work offline.

No external LLM API key is needed to try Phileas: your MCP client's model does
the reasoning, and the embedding and reranking run on your machine.

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

Python 3.11+

## License

MIT
