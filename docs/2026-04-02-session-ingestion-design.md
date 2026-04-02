# Phileas Session Ingestion

**Date:** 2026-04-02
**Status:** Approved

## Problem

Phileas only stores memories when Claude explicitly calls `memorize` during a conversation. This is selective and misses most personal signal — thoughts, feelings, decisions, reflections, relationship dynamics — that surface in conversation but aren't flagged as "worth memorizing" in the moment.

Claude Code already stores full conversation transcripts locally as JSONL files. We can process these after the fact to extract personal memories.

## Design

A scheduled job runs every 6 hours, scans for unprocessed Claude Code session files, and invokes Claude Code to extract personal memories into Phileas.

### Data Flow

```
~/.claude/projects/*/*.jsonl  (raw sessions, already on disk)
        |
        v
  Tracker (~/.phileas/ingested-sessions.json)
  -- which files have been processed
        |
        v
  For each new session:
    claude -p "<extraction prompt>" with session file path
        |
        v
  Phileas memories (SQLite + ChromaDB + KuzuDB)
```

### Components

#### 1. Ingestion script (`~/.phileas/ingest.sh`)

Shell script that:

- Reads tracker file to know what's been processed
- Scans `~/.claude/projects/*/*.jsonl` for unprocessed files
- Skips files < 5KB (empty/aborted sessions)
- Skips files modified within the last 30 minutes (likely still active)
- Acquires a lockfile (`~/.phileas/ingest.lock`) via `flock` to prevent overlapping runs
- For each new session: pre-filters the JSONL to extract only user/assistant message lines (drops tool_result, system, permission, attachment lines), pipes the filtered content to `claude -p` with the extraction prompt
- Updates tracker after each successful ingestion
- Logs to `~/.phileas/ingest.log`

**MCP access:** The `claude -p` invocation inherits the user's `~/.claude/.mcp.json` config, which already includes the Phileas MCP server. No special setup needed.

#### 2. Tracker (`~/.phileas/ingested-sessions.json`)

```json
{
  "processed": {
    "/home/ajuno/.claude/projects/-home-ajuno/abc123.jsonl": {
      "mtime": 1775127375,
      "ingested_at": "2026-04-02T18:00:00Z"
    }
  }
}
```

Keyed by file path. Stores mtime at time of processing so we can re-process if a session file was appended to (resumed session).

#### 3. Extraction prompt

The `claude -p` call receives a prompt that instructs Claude to:

- Read the session JSONL file
- Extract ONLY personal information: feelings, decisions, reflections, relationship dynamics, life events, self-assessments, preferences, patterns
- Ignore technical content: code, tool results, file contents, debugging steps, system prompts
- For each personal fact, call `mcp__phileas__memorize` with appropriate `memory_type` and `importance`
- If the session has no personal content, do nothing

#### 4. Schedule

Every 6 hours via system crontab:

```
0 */6 * * * ~/.phileas/ingest.sh
```

### What gets extracted

| Extract | Example |
|---------|---------|
| Feelings | "frustrated about salary review" |
| Decisions + reasoning | "chose to leave the badminton group because..." |
| Self-reflections | "realized I project emotions onto situations" |
| Relationship observations | "conversation with @phuongtq felt distant" |
| Life events | "started debugging Phileas graph layer" |
| Preferences discovered | "prefers file-based processing over hooks" |

### What gets ignored

- Code, diffs, file contents
- Tool call results (bash output, grep results, etc.)
- System prompts, permission modes
- Technical debugging steps
- Anything already captured in git history

### Edge cases

- **Resumed sessions:** Same JSONL file gets appended. Tracker stores mtime — if mtime changed since last ingestion, re-process. The extraction prompt should handle seeing some already-memorized content gracefully (Phileas recall can deduplicate).
- **Very large sessions:** Session files can be huge. The extraction prompt should instruct Claude to focus on user messages and assistant text, skip tool_result blocks.
- **No personal content:** Many sessions are purely technical. The prompt explicitly says "if no personal content, do nothing." No empty memories.
- **Concurrent access:** Uses `flock` on `~/.phileas/ingest.lock`. If another run is active, the new run exits immediately.

### Cost estimate

- ~430 existing sessions, but most are technical-only
- Ongoing: ~5-10 sessions/day, 4 cron runs/day
- Each ingestion is one `claude -p` call per session file
- Initial backfill of 430 sessions will be expensive — consider processing only sessions from the last 30 days for the first run, then expanding.

### Not in scope

- Real-time / in-session ingestion (existing phileas skill handles this)
- Technical fact extraction (git handles this)
- Raw conversation storage (files already on disk)
- Duplicate detection at write time (Phileas recall handles this downstream)
