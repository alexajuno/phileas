# MCP Integration

Phileas runs as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server, allowing any compatible AI client to store and recall memories.

## Starting the server

```bash
phileas serve
```

This starts the Phileas MCP server using stdio transport. The server exposes the following tools to connected AI clients:

| Tool | Description |
|------|-------------|
| `memorize` | Store a memory with type, importance, entities, and relationships |
| `recall` | Search memories by natural language query |
| `update` | Update a memory's content (preserves history) |
| `forget` | Archive a memory |
| `relate` | Create relationship edges in the knowledge graph |
| `about` | Get all memories connected to an entity |
| `timeline` | Get memories in a date range |
| `ingest_session` | Parse a Claude Code JSONL session file for memory extraction |
| `mark_session_done` | Mark a session as processed |
| `consolidate` | Find clusters of similar memories for summarization |
| `status` | Get system health and statistics |

## Claude Code

Add Phileas to `~/.claude/settings.json`:

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

If you installed Phileas in a virtual environment or with uv, use the full path or run through uv:

```json
{
  "mcpServers": {
    "phileas": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/phileas", "phileas", "serve"]
    }
  }
}
```

Or with an explicit Python path:

```json
{
  "mcpServers": {
    "phileas": {
      "command": "/path/to/venv/bin/phileas",
      "args": ["serve"]
    }
  }
}
```

After adding the config, restart Claude Code. Phileas tools will appear in the tool list.

## Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent path on your OS:

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

## Other MCP clients

Any MCP-compatible client can connect to Phileas. The server uses **stdio transport** -- the client launches `phileas serve` as a subprocess and communicates over stdin/stdout.

General pattern for client configuration:

```json
{
  "command": "phileas",
  "args": ["serve"]
}
```

## MCP tool details

### memorize

Store a memory about the user.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `summary` | string | yes | -- | What to remember (1-2 sentences) |
| `memory_type` | string | no | `"knowledge"` | `profile`, `event`, `knowledge`, `behavior`, `reflection` |
| `importance` | int | no | `5` | Importance 1-10 |
| `daily_ref` | string | no | today | Date in YYYY-MM-DD format |
| `entities` | list/JSON | no | -- | Entities to link: `[{"name": "...", "type": "..."}]` |
| `relationships` | list/JSON | no | -- | Relationships: `[{"from_name", "from_type", "edge", "to_name", "to_type"}]` |

### recall

Search memories by query.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Natural language query |
| `top_k` | int | no | `5` | Max results |
| `memory_type` | string | no | *all* | Filter by type |
| `min_importance` | int | no | *none* | Minimum importance threshold |

### update

Update a memory's content in place.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `memory_id` | string | yes | UUID of the memory |
| `summary` | string | yes | New summary text |

### forget

Archive a memory.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | yes | -- | UUID of the memory |
| `reason` | string | no | *none* | Reason for archiving |

### relate

Create a relationship edge in the knowledge graph.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `from_name` | string | yes | Source entity name |
| `from_type` | string | yes | Source entity type (e.g., `Person`) |
| `edge_type` | string | yes | Relationship (e.g., `WORKS_AT`, `KNOWS`, `LIKES`) |
| `to_name` | string | yes | Target entity name |
| `to_type` | string | yes | Target entity type (e.g., `Company`) |
| `memory_id` | string | no | Memory to link to the source entity |

### about

Get memories connected to an entity.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | yes | -- | Entity name |
| `entity_type` | string | no | *any* | Entity type filter |

### timeline

Get memories in a date range.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `start_date` | string | yes | -- | Start date (YYYY-MM-DD) |
| `end_date` | string | no | *same as start* | End date (YYYY-MM-DD) |

### ingest_session

Parse a Claude Code JSONL session file and return conversation text for memory extraction.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_path` | string | yes | Absolute path to the `.jsonl` file |

### mark_session_done

Mark a session as processed after memory extraction.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_path` | string | yes | Same path passed to `ingest_session` |

### consolidate

Find clusters of similar tier-2 memories for summarization.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_cluster_size` | int | no | `3` | Minimum memories per cluster |
| `max_clusters` | int | no | `10` | Maximum clusters to return |

### status

Get system health and memory statistics. No parameters.

## Environment variables

Make sure any required API keys are available in the environment where the MCP server runs. For Claude Code, the server inherits the shell environment, so keys set in `~/.bashrc` or `~/.zshrc` will be available.

If you need to pass environment variables explicitly:

```json
{
  "mcpServers": {
    "phileas": {
      "command": "phileas",
      "args": ["serve"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "PHILEAS_HOME": "/custom/path"
      }
    }
  }
}
```

## Troubleshooting

**Server not starting:** Make sure `phileas serve` works from your terminal first. If it fails, run `phileas status` to check the setup.

**Tools not appearing:** Restart your AI client after updating the MCP config. Check that the command path is correct -- use `which phileas` to find the full path.

**Permission errors:** If using a virtual environment, make sure the MCP config points to the correct Python/phileas binary inside the venv.

**Logs:** Check `~/.phileas/phileas.log` for server-side errors.
