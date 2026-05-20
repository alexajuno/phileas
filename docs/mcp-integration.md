# MCP Integration

Phileas runs as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server, allowing any compatible AI client to store and recall memories.

## Starting the server

```bash
phileas serve
```

This starts the Phileas MCP server using stdio transport. Tools exposed to the client:

| Tool | Description |
|------|-------------|
| `memorize` | Store a memory (summary, type, importance, entities, relationships) |
| `memorize_batch` | Store multiple memories in one call |
| `context` | Return the user's core / hot-set memories (identity, preferences, key facts) |
| `recall` | Graph-first semantic recall |
| `recall_recent` | Top memories per day for the last N days — time-relative queries |
| `thread` | Verbatim text of an ingested event plus every memory extracted from it |
| `update` | Update a memory's summary and/or add entities |
| `forget` | Archive a memory |
| `relate` | Create a relationship edge in the knowledge graph |
| `about` | Get memories connected to an entity, optionally expanding to neighbors |
| `timeline` | Memories anchored to a date or date range |
| `list_day_memories` | Every active memory anchored to a date (input for reflection) |
| `reflect` | Synthesize daily reflection memories |
| `ingest_session` | Parse a Claude Code JSONL session file for memory extraction |
| `mark_session_done` | Mark a session as processed |
| `merge_entities` | Fold duplicate entity rows into a canonical one |
| `status` | System health and statistics |

## Claude Code

`phileas init` wires this up automatically. It writes to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "phileas": {
      "type": "stdio",
      "command": "phileas",
      "args": ["serve"]
    }
  }
}
```

If you installed Phileas in a virtual environment or with uv, the wizard picks the resolved binary path. To configure manually, use either the binary on `PATH` or the full path:

```json
{
  "mcpServers": {
    "phileas": {
      "type": "stdio",
      "command": "/path/to/venv/bin/phileas",
      "args": ["serve"]
    }
  }
}
```

After updating `.mcp.json`, restart Claude Code. Phileas tools will appear in the tool list, and the recall skill (installed by the wizard at `~/.claude/skills/phileas/SKILL.md`) will fire on memory-relevant prompts.

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

Any MCP-compatible client can connect to Phileas. The server uses **stdio transport** — the client launches `phileas serve` as a subprocess and communicates over stdin/stdout.

General pattern:

```json
{
  "command": "phileas",
  "args": ["serve"]
}
```

## MCP tool details

### memorize

Store a memory about the user.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `summary` | string | required | What to remember (1-2 sentences, in your own words) |
| `memory_type` | string | `"knowledge"` | `profile`, `event`, `knowledge`, `behavior`, `reflection` |
| `importance` | int | `5` | Importance 1-10 |
| `daily_ref` | string | today | Date in YYYY-MM-DD format |
| `entities` | list/JSON | — | `[{"name": str, "type": str, "description"?: str}]` |
| `relationships` | list/JSON | — | `[{"from_name", "from_type", "edge", "to_name", "to_type"}]` |
| `source_event_id` | string | — | Event id this memory was extracted from |

### memorize_batch

Store multiple memories in one call. Takes a `memories` list/JSON; each item accepts the same fields as `memorize`.

### context

Return the hot-set memories — identity, preferences, key facts — without the full recall pipeline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | `10` | Max core memories |
| `memory_type` | string | *all* | Filter by type |

### recall

Graph-first retrieval. Entity lookup → memory pivot → semantic supplement.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Natural language query |
| `memory_type` | string | *all* | Filter by type |
| `min_importance` | int | — | Minimum importance threshold |
| `top_k` | int | `30` | Max results |

### recall_recent

Top memories per day for the last N days, grouped newest-day first. Use for time-relative queries ("recently", "yesterday", "last chat").

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | int | `7` | How many days back |
| `top_per_day` | int | `10` | Max per day, sorted by importance |
| `min_importance` | int | `5` | Floor — relaxed per-day if no items pass |

### thread

Return the verbatim text of an ingested event plus every memory extracted from it. Use as a follow-up when a memory's `source_event_id` points somewhere interesting.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `event_id` | string | yes | Event UUID (from a memory's `source_event_id`) |

### update

Update a memory's summary in place (old version snapshotted) and/or add entities/relationships (additive).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `memory_id` | string | yes | UUID of the memory |
| `summary` | string | no | New summary (omit to keep existing) |
| `entities` | list/JSON | no | Entities to link |
| `relationships` | list/JSON | no | Relationships to add |

### forget

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `memory_id` | string | yes | UUID of the memory |
| `reason` | string | no | Reason for archiving |

### relate

Create a relationship edge in the knowledge graph.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `from_name` | string | yes | Source entity name |
| `from_type` | string | yes | Source entity type (e.g. `Person`) |
| `edge_type` | string | yes | Relationship (e.g. `WORKS_AT`, `KNOWS`) |
| `to_name` | string | yes | Target entity name |
| `to_type` | string | yes | Target entity type |
| `memory_id` | string | no | Memory to link to the source entity |

### about

Memories connected to an entity. Optionally expand one hop via REL edges.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | required | Entity name |
| `entity_type` | string | *any* | Type filter |
| `expand` | bool | `false` | If true, include neighbors via REL edges |
| `memory_type` | string \| list | *all* | Filter memories by type — useful for hub entities |

### timeline

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | string | required | YYYY-MM-DD |
| `end_date` | string | start_date | YYYY-MM-DD |
| `window` | int | `1` | Days to expand search in both directions |

### list_day_memories

Every active memory anchored to a date, no window. Input for `reflect`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date` | string | today | YYYY-MM-DD |

### reflect

Synthesize 1–5 reflection memories from a day's activity.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date` | string | today | YYYY-MM-DD |

### ingest_session

Parse a Claude Code JSONL session file and return its conversation text. The client (Claude Code) extracts memories and calls `memorize` for each, then `mark_session_done`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_path` | string | yes | Absolute path to the `.jsonl` file |

### mark_session_done

Mark a session as processed so it won't be re-ingested.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_path` | string | yes | Same path passed to `ingest_session` |

### merge_entities

Fold duplicate entity rows into a canonical one. Cleanup primitive for entity-aliasing drift — when the same person/place was minted under multiple ids because the linker didn't recognize a name variant.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `canonical_id` | string | yes | Entity uuid that should survive |
| `duplicate_ids` | list[string] | yes | Entity uuids to fold in |

### status

System health and memory statistics. No parameters.

## Environment variables

Make sure any required API keys are available in the environment where the MCP server runs. For Claude Code, the server inherits the shell environment, so keys set in `~/.bashrc` or `~/.zshrc` will be available.

To pass environment variables explicitly:

```json
{
  "mcpServers": {
    "phileas": {
      "type": "stdio",
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

**Tools not appearing:** Restart your AI client after updating `.mcp.json`. Check that the command path is correct — use `which phileas` to find the full path.

**Permission errors:** If using a virtual environment, make sure the MCP config points to the correct binary inside the venv.

**Logs:** Check `~/.phileas/phileas.log` for server-side errors.
