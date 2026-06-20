# Phileas: persistent memory for AI

AI conversations reset every session. Phileas is a local memory layer that any LLM can read and write through [MCP](https://modelcontextprotocol.io/), so context survives across sessions and tools.

## Requirements

- **Python 3.11 or newer.**
- **An MCP client.** Phileas exposes its memory over MCP. The setup wizard wires it into [Claude Code](https://docs.claude.com/en/docs/claude-code) automatically; any other MCP client (a GPT or Ollama front-end, and so on) connects to `phileas serve`.
- **A few hundred MB of disk and one download.** Phileas runs two small models locally (about 150 MB) and depends on PyTorch. The steps below keep that download lean.

## Installation

```bash
python3 -m venv ~/.venvs/phileas # an isolated environment
source ~/.venvs/phileas/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install phileas-memory

phileas init
```

Phileas uses PyTorch only to run two small models (an embedding model and a reranker) on the CPU, so the first install line fetches PyTorch's lean CPU build. That keeps the download to a few hundred MB; without it, `pip` pulls the multi-gigabyte CUDA build on Linux. On macOS and Windows the CPU build is already the default, so that line is harmless there too.

`phileas init` is the setup wizard: it chooses where memories live, connects Phileas to Claude Code, downloads the models, and starts the background daemon. Restart Claude Code afterward so it picks up the memory tools.

The MCP server is launched by its full path, so Claude Code finds it whether or not the venv is active. To run `phileas` commands yourself (such as `phileas status`), activate the venv first.

### First run

The first run downloads two models from [Hugging Face](https://huggingface.co/) that then run locally: an embedding model (`all-MiniLM-L6-v2`) and a reranker (`ms-marco-MiniLM-L-6-v2`), about 150 MB together. They are cached after the first download, so later runs work offline. No external LLM API key is needed: your MCP client's model does the reasoning, while embedding and reranking run on your machine.

## Connect other MCP clients

Claude Code is configured by `phileas init`. For any other MCP client, start the server and point the client at it:

```bash
phileas serve
```

## Principles

- **Local-first:** memories stay on your machine.
- **Model-agnostic:** works with any LLM via MCP.
- **Natural forgetting:** irrelevant detail decays; recall favors what stays useful.
- **Open:** run it yourself, read the code.

For the full command list, run `phileas --help` or `phileas COMMAND --help`.

## License

MIT
