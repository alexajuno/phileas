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

## Profiles: separate memories, and switching between them

A profile is an isolated Phileas instance with its own data directory, daemon, and memories, so a `work` brain and a `personal` brain never mix. Each lives at `~/.config/phileas/profiles/<name>`. Create one with `phileas init --profile <name>`, and run `phileas profile list` to see the profiles on this machine and which one is active.

Which profile is in use is decided independently for two surfaces.

The `phileas` CLI follows an active profile you set once: `phileas profile use <name>` makes flag-less commands (`phileas memorize`, `phileas recall`, `phileas status`) target that brain until you switch again. A `--profile <name>` flag or the `PHILEAS_PROFILE` env var overrides it for a single invocation.

An MCP client (Claude Code, the phone app) follows the profile its own config pins, not the CLI's active profile. To bind a client per directory, run `phileas profile pin <name>` there: it writes a `phileas` server override carrying `PHILEAS_PROFILE=<name>`, so a client launched in that directory connects to that brain regardless of your global default. Reopen the client and approve the server for it to take effect; `phileas profile unpin` removes the pin. The pin is private to you by default; `--scope project` writes a shared `.mcp.json` instead.

## Principles

- **Local-first:** memories stay on your machine.
- **Model-agnostic:** works with any LLM via MCP.
- **Natural forgetting:** irrelevant detail decays; recall favors what stays useful.
- **Open:** run it yourself, read the code.

For the full command list, run `phileas --help` or `phileas COMMAND --help`.

## License

MIT
