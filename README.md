# Phileas: persistent memory for AI

AI conversations reset every session. Phileas is a local memory layer that any LLM can read and write through [MCP](https://modelcontextprotocol.io/), so context survives across sessions and tools.

## Requirements

- **Python 3.11 or newer.**
- **An MCP client.** Phileas exposes its memory over MCP. The setup wizard wires it into [Claude Code](https://docs.claude.com/en/docs/claude-code) automatically; any other MCP client (a GPT or Ollama front-end, and so on) connects to `phileas serve`.
- **A few hundred MB of disk and one download.** Phileas runs two small models locally (about 150 MB) and depends on PyTorch. The steps below keep that download lean.

## Installation

Install with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv tool install phileas-memory --torch-backend cpu

phileas init
```

`--torch-backend cpu` fetches PyTorch's lean CPU build (a few hundred MB) instead of the multi-gigabyte CUDA build uv would otherwise pull on Linux. Phileas runs two small models (an embedding model and a reranker) on the CPU and needs nothing more; on macOS and Windows the CPU build is the default anyway. `uv tool install` puts `phileas` on your PATH, so `phileas status` and the other commands just work.

Prefer pip? Create a virtual environment and install into it:

```bash
python3 -m venv ~/.venvs/phileas # an isolated environment
source ~/.venvs/phileas/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install phileas-memory

phileas init
```

With the pip route, activate that venv before running `phileas` commands yourself.

`phileas init` is the setup wizard: it chooses where memories live, connects Phileas to Claude Code, downloads the models, and starts the background daemon. Restart Claude Code afterward so it picks up the memory tools.

The MCP server is launched by its full path, so Claude Code finds it regardless of your shell environment.

### First run

The first run downloads three models from [Hugging Face](https://huggingface.co/) that then run locally: an embedding model (`all-MiniLM-L6-v2`), a reranker (`ms-marco-MiniLM-L-6-v2`), and an NLI model (`nli-deberta-v3-small`) for contradiction detection, about 300 MB together. They are cached after the first download, so later runs work offline. No API key is needed: the three models run locally on your machine. In Claude Code, capture hooks store every turn verbatim as it happens (the raw floor under memory), and Claude records the conclusions you endorse with `memorize`.

## Connect other MCP clients

`phileas init` wires up Claude Code. Any other MCP client connects to the same server: `phileas serve` speaks MCP over stdio, so register it the way that client registers a stdio server.

Most clients (Cursor, Antigravity, and others) read a JSON config with the same shape. Add Phileas to its `mcpServers` map:

```json
{
  "mcpServers": {
    "phileas": {
      "type": "stdio",
      "command": "/absolute/path/to/phileas",
      "args": ["serve"]
    }
  }
}
```

Use the absolute path to the `phileas` executable (run `command -v phileas` to find it), since the client launches it without your shell's PATH or an active venv. To point a client at a named profile, add `"env": {"PHILEAS_PROFILE": "<name>"}` to the entry.

### Proactive recall and memorize

The tools work in any client, and the server ships usage guidance that every MCP client receives on connect. The extra layer that makes memory feel automatic (recall before answering, memorize when something worth keeping comes up, the query shapes that retrieve well) lives in a skill file. `phileas init` installs it for Claude Code at `~/.claude/skills/phileas/SKILL.md`; its text uses bare tool names so it carries to any client. If your client has a rules or skills mechanism, put that file's content into it.

## Monitor with the web dashboard

Phileas also has an optional local web dashboard for browsing your memory, read-only.

The dashboard is a small web app, so it needs [Node.js](https://nodejs.org) 20 or newer (which bundles the `pnpm` package manager through `corepack`). With that in place:

```bash
phileas web install   # one-time: fetches and builds the dashboard
phileas web           # serves it at http://127.0.0.1:3000 and opens your browser
```

`phileas web install` clones the dashboard into `~/.local/share/phileas/web` and builds it. `phileas web` then starts the server; press Ctrl-C to stop. To watch a named profile, pass it through the top-level flag: `phileas --profile work web`. Use `phileas web update` to pull a newer dashboard, `phileas web status` to see where it lives and whether it's ready, and `phileas web --help` for the rest.

## Principles

- **Local-first:** memories stay on your machine.
- **Model-agnostic:** works with any LLM via MCP.
- **Natural forgetting:** irrelevant detail decays; recall favors what stays useful.
- **Open:** run it yourself, read the code.

For the full command list, run `phileas --help` or `phileas COMMAND --help`.

## License

MIT
