# Phileas Stats — Design

**Date:** 2026-04-17
**Status:** Approved, pending implementation plan
**Replaces scope of:** static `phileas usage` command (kept as alias)

## Problem

`phileas usage` is a static snapshot of LLM spend from `usage.db`. It says nothing about memory lifecycle, recall quality, graph health, consolidation runs, ingest dedup, or daemon health. There is no time axis, no rates, no drill-down. Given known quality issues (entity-tagged recall gap, MMR perf fix, KuzuDB lock contention) the lack of observability is a real blocker to judging whether Phileas is working.

## Goal

A richer CLI — **only** a CLI — that covers both a cost/health watchdog and a behavioral dashboard, scoped across every layer of Phileas:

- LLM cost/latency
- Memory lifecycle (memorize/forget/update rates)
- Recall quality (top-1 score, empty-rate, latency, hot-set hit rate)
- Graph health (nodes, edges, orphan rate, entity-tag coverage)
- Consolidation / reflection runs
- Ingest + dedup
- Daemon health (uptime, errors, lock contention)

"Everything" in v1. Full instrumentation accepted.

## Non-goals

- No TUI, no web dashboard, no push notifications.
- No per-user / multi-tenant — Phileas is single-user.
- No historical backfill for new instrumentation — counters start at deploy time.
- No alerting (thresholds, pages). Just surface numbers; human reads them.

## CLI surface

Root is a Click group `phileas stats`. Legacy `phileas usage` is kept as an alias for `phileas stats llm`.

```
phileas stats                    # alias for `overview`
phileas stats overview           # one-screen dashboard: headline numbers + sparklines
phileas stats llm                # tokens, cost, by-op, failures (today's `usage` output)
phileas stats memory             # memorize/forget/update rates, by type, distributions
phileas stats recall             # query volume, top-1 score, empty-rate, latency p50/p95, hot-hit rate
phileas stats graph              # node/edge counts, orphan rate, entity-tag coverage
phileas stats consolidation      # last run, clusters, new memories, LLM cost per run
phileas stats ingest             # sessions ingested, memories per session, dedup rate
phileas stats daemon             # uptime, error counts, lock contention events
```

**Shared flags on every subcommand:**

| Flag | Default | Notes |
|---|---|---|
| `--since` | `7d` | Accepts `24h`, `7d`, `30d`, `all` |
| `--bucket` | auto | `hour`, `day`, `week`; auto picks based on `--since` |
| `--json` | off | Emits raw dict; skips render |
| `--watch` | off | Re-runs every N seconds; no TUI, just a re-render loop |

**Output style:** Rich tables + ASCII sparklines using unicode blocks (`▁▂▃▄▅▆▇█`). Each subcommand prints a headline row (3–5 key numbers) above drill-down tables. No new deps.

## Data model

New SQLite DB at `~/.phileas/metrics.db`. Separate from `usage.db` and `phileas.db` so it can be pruned independently. WAL mode. Single-writer — all inserts go through one `MetricsWriter` instance per process.

```sql
-- recall quality + latency, one row per recall() call
CREATE TABLE recall_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    query_len INTEGER,         -- char count only, not the text (privacy)
    top_k INTEGER,
    returned INTEGER,          -- how many results actually returned
    top1_score REAL,           -- null if empty
    mean_score REAL,
    empty INTEGER NOT NULL,    -- 1 if returned=0
    hot_hit INTEGER,           -- 1 if served from hot-memory cache
    latency_ms REAL,
    stage_timings_json TEXT    -- optional: {keyword, semantic, mmr}
);
CREATE INDEX idx_recall_created ON recall_events(created_at);

-- ingest/dedup signals, one row per memorize() call
CREATE TABLE ingest_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    memory_type TEXT,
    importance INTEGER,
    entity_count INTEGER,      -- feeds entity-population gap metric
    deduped INTEGER NOT NULL,  -- 1 if merged into existing memory
    source TEXT                -- mcp|cli|session_ingest
);
CREATE INDEX idx_ingest_created ON ingest_events(created_at);

-- daemon lifecycle + error counters
CREATE TABLE daemon_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,        -- start|stop|error|lock_contention|consolidate_run|reflect_run
    payload_json TEXT
);
CREATE INDEX idx_daemon_created_kind ON daemon_events(created_at, kind);
```

**Derived from existing sources (no new table):**

- Memory lifecycle → `phileas.db` (`memories` table: `created_at`, `type`, soft-delete flag).
- Graph health → kuzu `MATCH` counts at query time. Use the same snapshot-copy trick as `scripts/export_phileas.py` (copy `~/.phileas/graph` + `graph.wal` to a tempdir, open read-only) to avoid conflicting with the daemon's exclusive lock.
- LLM + consolidation cost → `usage.db`, filtered by `operation` (`consolidate_*`, `reflect_*`).

**Retention:** none by default. Add `phileas stats prune --older-than 90d` later if volume warrants it.

## Instrumentation

One thin writer, best-effort semantics (inserts must never raise into the calling code path).

```
src/phileas/stats/writer.py
  class MetricsWriter:
      def __init__(self, db_path: Path): ...
      def record_recall(...): ...
      def record_ingest(...): ...
      def record_daemon(kind: str, payload: dict | None = None): ...
```

All public methods swallow exceptions with a debug-level log. The writer owns one `sqlite3.Connection` opened in WAL mode, reused for the process lifetime.

**Hook points:**

| Site | Event |
|---|---|
| `engine.recall()` — end of function | `recall_events` row. `hot_hit` already known in current code path. `stage_timings_json` filled if timings are already collected, else null. |
| `engine.memorize()` — end of function | `ingest_events` row. `deduped=1` when the merge path returns an existing id. |
| Daemon `main()` — boot / shutdown | `daemon_events(kind=start|stop)`. `error` on top-level exception handler. |
| Kuzu writer — exception handler | `daemon_events(kind=lock_contention)` when IOError/lock errors are caught. |
| `engine.consolidate()` / `engine.reflect()` — end of run | `daemon_events(kind=consolidate_run|reflect_run, payload={duration_ms, clusters, new_memories, cost_usd})`. |

## Module layout

```
src/phileas/stats/
  __init__.py
  cli.py           # Click group `stats`, subcommands registered here
  writer.py        # MetricsWriter (imported by engine + daemon)
  queries.py       # Pure-SQL query functions against metrics.db + phileas.db + usage.db
  graph_probe.py   # Kuzu snapshot-read for node/edge counts + entity-tag coverage
  render.py        # Shared rich.table + sparkline helpers
  time.py          # parse_since(), bucket_auto(), bucketize()
```

Each subcommand in `cli.py` follows the same shape:

```
data = queries.<subcommand>(since, bucket)
if json_flag:
    click.echo(json.dumps(data, default=str))
    return
render.<subcommand>(data)
```

`--watch` wraps the body in a `while True: sleep(N); clear(); run()` loop.

## Rollout phases

Three phases, each independently shippable.

1. **Phase 1 — Foundation + free data.**
   - Create `src/phileas/stats/` skeleton, register `stats` group in existing CLI.
   - Implement `time.py`, `render.py` (incl. `spark()`), `graph_probe.py`.
   - Ship subcommands that need no new instrumentation: `overview`, `llm`, `memory`, `graph`, `consolidation`.
   - Keep `phileas usage` as an alias for `stats llm`.

2. **Phase 2 — Recall + ingest instrumentation.**
   - Add `MetricsWriter`, create `metrics.db` schema.
   - Wire `recall_events` and `ingest_events` into `engine.recall()` and `engine.memorize()`.
   - Ship `stats recall` and `stats ingest`.

3. **Phase 3 — Daemon health + UX polish.**
   - Wire `daemon_events` for start/stop/error/lock_contention/consolidate_run/reflect_run.
   - Ship `stats daemon`.
   - Add `--watch` loop, finalize `--json` contract (stable shape per subcommand).

## Testing

- `queries.py` — unit tests with fixture SQLite DBs seeded via the schema files. One test per subcommand's query.
- `time.py` — unit tests for `parse_since()` (valid/invalid strings, `all`), `bucket_auto()` (range → bucket), `bucketize()` (events into buckets incl. empty buckets).
- `render.spark()` — unit test for output length + extremes (all-zero, single value, NaN guard).
- `writer.py` — unit test that exceptions inside `record_*` are swallowed (inject a bad path).
- No integration tests in phase 1; verify overview manually against live data.

## Open questions

None at spec time. Defer until implementation surfaces them.
