# Phileas Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static `phileas usage` snapshot with a `phileas stats` subcommand tree covering LLM cost, memory lifecycle, recall quality, graph health, consolidation, ingest dedup, and daemon health — backed by a new `metrics.db` for event-level instrumentation.

**Architecture:** A new `phileas.stats` package owns a Click group, a best-effort SQLite writer, query helpers against three DBs (`metrics.db`, `phileas.db`, `usage.db`), a Kuzu snapshot probe, and a shared Rich-based renderer. Instrumentation is hooked at `engine.recall()`, `engine.memorize()`, daemon lifecycle, and Kuzu exception paths. Rolled out in three phases: free-data first, then recall/ingest instrumentation, then daemon health + UX polish.

**Tech Stack:** Python 3.14, Click, Rich, SQLite (stdlib), Kuzu, pytest.

**Spec:** `docs/superpowers/specs/2026-04-17-phileas-stats-design.md`

---

## File Structure

**Create:**
- `src/phileas/stats/__init__.py` — package marker.
- `src/phileas/stats/time.py` — `parse_since`, `bucket_auto`, `bucketize`.
- `src/phileas/stats/render.py` — `spark`, `headline`, shared table helpers.
- `src/phileas/stats/writer.py` — `MetricsWriter` with schema + best-effort inserts.
- `src/phileas/stats/queries.py` — pure SQL query functions.
- `src/phileas/stats/graph_probe.py` — Kuzu snapshot-read helpers.
- `src/phileas/stats/cli.py` — Click group `stats` with subcommands.
- `tests/test_stats_time.py`
- `tests/test_stats_render.py`
- `tests/test_stats_writer.py`
- `tests/test_stats_queries.py`

**Modify:**
- `src/phileas/cli/__init__.py` — register new `stats` group; keep `usage` alias.
- `src/phileas/engine.py` — instrument `recall()` and `memorize()` to emit metric events.
- `src/phileas/daemon.py` — emit start/stop/error/lock/consolidate/reflect events.
- `src/phileas/graph.py` (or wherever Kuzu writes happen) — emit `lock_contention` event on IOError.

---

# Phase 1 — Foundation + Free Data

All tasks in this phase ship usable commands derived from data Phileas already has (`phileas.db`, `usage.db`, Kuzu). No instrumentation yet.

## Task 1: Package skeleton and time helpers

**Files:**
- Create: `src/phileas/stats/__init__.py`
- Create: `src/phileas/stats/time.py`
- Create: `tests/test_stats_time.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stats_time.py
from datetime import datetime, timezone, timedelta

import pytest

from phileas.stats.time import parse_since, bucket_auto, bucketize


def test_parse_since_hours():
    now = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
    assert parse_since("24h", now) == now - timedelta(hours=24)


def test_parse_since_days():
    now = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
    assert parse_since("7d", now) == now - timedelta(days=7)


def test_parse_since_all():
    now = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
    assert parse_since("all", now) is None


def test_parse_since_invalid():
    with pytest.raises(ValueError):
        parse_since("banana", datetime.now(timezone.utc))


def test_bucket_auto_ranges():
    assert bucket_auto(timedelta(hours=24)) == "hour"
    assert bucket_auto(timedelta(days=7)) == "day"
    assert bucket_auto(timedelta(days=60)) == "week"
    assert bucket_auto(None) == "week"  # "all"


def test_bucketize_day():
    events = [
        {"created_at": "2026-04-15T10:00:00+00:00", "v": 1},
        {"created_at": "2026-04-15T22:30:00+00:00", "v": 2},
        {"created_at": "2026-04-16T01:00:00+00:00", "v": 4},
    ]
    out = bucketize(events, "day", field="v")
    assert out == [("2026-04-15", 3), ("2026-04-16", 4)]


def test_bucketize_fills_empty_buckets():
    events = [
        {"created_at": "2026-04-14T10:00:00+00:00", "v": 1},
        {"created_at": "2026-04-16T10:00:00+00:00", "v": 2},
    ]
    start = datetime(2026, 4, 14, tzinfo=timezone.utc)
    end = datetime(2026, 4, 16, tzinfo=timezone.utc)
    out = bucketize(events, "day", field="v", start=start, end=end)
    assert out == [("2026-04-14", 1), ("2026-04-15", 0), ("2026-04-16", 2)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stats_time.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phileas.stats'`.

- [ ] **Step 3: Create package and implement time helpers**

```python
# src/phileas/stats/__init__.py
"""Phileas stats — observability CLI and instrumentation."""
```

```python
# src/phileas/stats/time.py
"""Time parsing, bucket selection, and event bucketization for stats."""

from __future__ import annotations

import re
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Iterable

_SINCE_RE = re.compile(r"^(\d+)([hdw])$")


def parse_since(expr: str, now: datetime) -> datetime | None:
    """Parse expressions like '24h', '7d', '4w', or 'all'.

    Returns the cutoff datetime, or None for 'all'.
    Raises ValueError for unrecognized input.
    """
    if expr == "all":
        return None
    m = _SINCE_RE.match(expr)
    if not m:
        raise ValueError(f"invalid --since value: {expr!r}")
    n, unit = int(m.group(1)), m.group(2)
    delta = {"h": timedelta(hours=n), "d": timedelta(days=n), "w": timedelta(weeks=n)}[unit]
    return now - delta


def bucket_auto(window: timedelta | None) -> str:
    """Pick a bucket size based on window length."""
    if window is None:
        return "week"
    if window <= timedelta(hours=48):
        return "hour"
    if window <= timedelta(days=31):
        return "day"
    return "week"


def _key(dt: datetime, bucket: str) -> str:
    if bucket == "hour":
        return dt.strftime("%Y-%m-%d %H:00")
    if bucket == "day":
        return dt.strftime("%Y-%m-%d")
    if bucket == "week":
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    raise ValueError(f"unknown bucket: {bucket}")


def _step(bucket: str) -> timedelta:
    return {"hour": timedelta(hours=1), "day": timedelta(days=1), "week": timedelta(weeks=1)}[bucket]


def bucketize(
    events: Iterable[dict],
    bucket: str,
    field: str = "count",
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[tuple[str, float]]:
    """Group events by bucket and sum `field`.

    If `start`/`end` given, fills empty buckets with 0 so sparklines are aligned.
    Each event dict must contain 'created_at' (ISO-8601 with tz).
    """
    sums: "OrderedDict[str, float]" = OrderedDict()
    if start is not None and end is not None:
        cursor = start
        while cursor <= end:
            sums[_key(cursor, bucket)] = 0.0
            cursor += _step(bucket)
    for ev in events:
        dt = datetime.fromisoformat(ev["created_at"])
        key = _key(dt, bucket)
        sums[key] = sums.get(key, 0.0) + float(ev.get(field, 0) or 0)
    return [(k, v) for k, v in sums.items()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stats_time.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phileas/stats/__init__.py src/phileas/stats/time.py tests/test_stats_time.py
git commit -m "feat(stats): add time parsing and bucketization helpers"
```

## Task 2: Render helpers (spark + headline)

**Files:**
- Create: `src/phileas/stats/render.py`
- Create: `tests/test_stats_render.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stats_render.py
from phileas.stats.render import spark


def test_spark_empty():
    assert spark([]) == ""


def test_spark_single_value():
    assert spark([5.0]) == "█"


def test_spark_all_zero():
    assert spark([0, 0, 0, 0]) == "▁▁▁▁"


def test_spark_monotone():
    # Increasing values should produce non-decreasing block heights
    out = spark([1, 2, 3, 4, 5, 6, 7, 8])
    assert len(out) == 8
    # first should be lowest, last should be highest
    blocks = "▁▂▃▄▅▆▇█"
    assert blocks.index(out[0]) <= blocks.index(out[-1])


def test_spark_handles_nan_as_zero():
    out = spark([float("nan"), 1.0, 2.0])
    assert len(out) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stats_render.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement render.py**

```python
# src/phileas/stats/render.py
"""Shared rendering helpers for `phileas stats` subcommands."""

from __future__ import annotations

import math
from typing import Iterable

from rich.console import Console
from rich.table import Table

_BLOCKS = "▁▂▃▄▅▆▇█"

console = Console()


def spark(values: Iterable[float]) -> str:
    """Render a list of values as a unicode sparkline.

    NaN and None are treated as 0. Empty input returns "".
    """
    nums = [0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in values]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    if hi <= lo:
        # All equal — render as full blocks if non-zero, minimum blocks otherwise
        block = _BLOCKS[-1] if hi > 0 else _BLOCKS[0]
        return block * len(nums)
    span = hi - lo
    out = []
    for v in nums:
        idx = int((v - lo) / span * (len(_BLOCKS) - 1))
        out.append(_BLOCKS[idx])
    return "".join(out)


def headline(title: str, pairs: list[tuple[str, str]]) -> Table:
    """Render a 2-column key/value headline table."""
    table = Table(title=title, show_header=False, box=None, pad_edge=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in pairs:
        table.add_row(k, v)
    return table
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stats_render.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phileas/stats/render.py tests/test_stats_render.py
git commit -m "feat(stats): add sparkline and headline render helpers"
```

## Task 3: Query layer (scaffold) + LLM/memory queries

**Files:**
- Create: `src/phileas/stats/queries.py`
- Create: `tests/test_stats_queries.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stats_queries.py
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from phileas.stats.queries import llm_summary, memory_lifecycle


@pytest.fixture
def usage_db(tmp_path: Path) -> Path:
    p = tmp_path / "usage.db"
    conn = sqlite3.connect(p)
    conn.executescript(
        """
        CREATE TABLE llm_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation TEXT NOT NULL,
            model TEXT NOT NULL,
            provider TEXT,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0.0,
            latency_ms REAL NOT NULL DEFAULT 0.0,
            success INTEGER NOT NULL DEFAULT 1,
            error TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    now = datetime(2026, 4, 17, tzinfo=timezone.utc)
    rows = [
        ("remember", "claude", "anthropic", 100, 50, 150, 0.01, 500, 1, None, (now - timedelta(days=1)).isoformat()),
        ("recall", "claude", "anthropic", 200, 30, 230, 0.02, 800, 1, None, (now - timedelta(days=2)).isoformat()),
        ("recall", "claude", "anthropic", 100, 0, 100, 0.00, 100, 0, "err", (now - timedelta(days=10)).isoformat()),
    ]
    conn.executemany(
        """INSERT INTO llm_usage
        (operation, model, provider, prompt_tokens, completion_tokens, total_tokens,
         cost_usd, latency_ms, success, error, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    conn.close()
    return p


def test_llm_summary_respects_since(usage_db: Path):
    since = datetime(2026, 4, 15, tzinfo=timezone.utc)
    out = llm_summary(usage_db, since=since)
    # Only 2 rows are within window
    assert out["total_requests"] == 2
    assert out["total_cost_usd"] == pytest.approx(0.03)
    assert out["failed"] == 0


def test_llm_summary_all(usage_db: Path):
    out = llm_summary(usage_db, since=None)
    assert out["total_requests"] == 3
    assert out["failed"] == 1


@pytest.fixture
def phileas_db(tmp_path: Path) -> Path:
    p = tmp_path / "memory.db"
    conn = sqlite3.connect(p)
    conn.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL
        );
        """
    )
    now = datetime(2026, 4, 17, tzinfo=timezone.utc)
    rows = [
        ("a", "event", "active", (now - timedelta(days=1)).isoformat()),
        ("b", "event", "active", (now - timedelta(days=1)).isoformat()),
        ("c", "knowledge", "active", (now - timedelta(days=3)).isoformat()),
        ("d", "event", "archived", (now - timedelta(days=5)).isoformat()),
    ]
    conn.executemany(
        "INSERT INTO memories (id, type, status, created_at) VALUES (?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return p


def test_memory_lifecycle_by_type(phileas_db: Path):
    out = memory_lifecycle(phileas_db, since=None)
    by_type = {row["type"]: row for row in out["by_type"]}
    assert by_type["event"]["created"] == 3
    assert by_type["event"]["active"] == 2
    assert by_type["knowledge"]["created"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stats_queries.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement queries.py (llm_summary + memory_lifecycle)**

```python
# src/phileas/stats/queries.py
"""SQL query helpers for phileas stats.

Each function takes explicit DB paths and a `since` datetime (or None for all-time)
and returns a plain dict/list-of-dicts.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _since_clause(since: datetime | None, col: str = "created_at") -> tuple[str, tuple]:
    if since is None:
        return "", ()
    return f" WHERE {col} >= ?", (since.isoformat(),)


def llm_summary(usage_db: Path, since: datetime | None) -> dict:
    """Aggregate stats from usage.db."""
    where, params = _since_clause(since)
    with _connect(usage_db) as conn:
        row = conn.execute(
            f"""SELECT
                COUNT(*) AS total_requests,
                COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0) AS successful,
                COALESCE(SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), 0) AS failed,
                COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms
            FROM llm_usage{where}""",
            params,
        ).fetchone()
    return {k: row[k] for k in row.keys()}


def llm_by_operation(usage_db: Path, since: datetime | None) -> list[dict]:
    where, params = _since_clause(since)
    with _connect(usage_db) as conn:
        rows = conn.execute(
            f"""SELECT operation,
                       COUNT(*) AS requests,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COALESCE(SUM(cost_usd), 0.0) AS cost_usd,
                       COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
                       COALESCE(SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), 0) AS failures
                FROM llm_usage{where}
                GROUP BY operation
                ORDER BY requests DESC""",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def llm_timeseries(usage_db: Path, since: datetime | None) -> list[dict]:
    """Raw rows for client-side bucketize()."""
    where, params = _since_clause(since)
    with _connect(usage_db) as conn:
        rows = conn.execute(
            f"SELECT created_at, total_tokens, cost_usd FROM llm_usage{where}",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def memory_lifecycle(phileas_db: Path, since: datetime | None) -> dict:
    """Memorize rate by type, plus active/archived counts."""
    where, params = _since_clause(since)
    with _connect(phileas_db) as conn:
        rows = conn.execute(
            f"""SELECT type,
                       COUNT(*) AS created,
                       SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) AS active,
                       SUM(CASE WHEN status='archived' THEN 1 ELSE 0 END) AS archived
                FROM memories{where}
                GROUP BY type
                ORDER BY created DESC""",
            params,
        ).fetchall()
        total = conn.execute(f"SELECT COUNT(*) AS c FROM memories{where}", params).fetchone()["c"]
    return {"total_created": total, "by_type": [dict(r) for r in rows]}


def memory_timeseries(phileas_db: Path, since: datetime | None) -> list[dict]:
    where, params = _since_clause(since)
    with _connect(phileas_db) as conn:
        rows = conn.execute(
            f"SELECT created_at, type FROM memories{where}",
            params,
        ).fetchall()
    return [dict(r) | {"count": 1} for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stats_queries.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phileas/stats/queries.py tests/test_stats_queries.py
git commit -m "feat(stats): add LLM and memory lifecycle queries"
```

## Task 4: Consolidation query + graph probe

**Files:**
- Create: `src/phileas/stats/graph_probe.py`
- Modify: `src/phileas/stats/queries.py` (add `consolidation_runs`)
- Modify: `tests/test_stats_queries.py` (add consolidation test)

- [ ] **Step 1: Add failing test for `consolidation_runs`**

Append to `tests/test_stats_queries.py`:

```python
from phileas.stats.queries import consolidation_runs


def test_consolidation_runs_from_usage(usage_db: Path):
    # Seed a consolidate_* row
    conn = sqlite3.connect(usage_db)
    conn.execute(
        """INSERT INTO llm_usage
        (operation, model, provider, prompt_tokens, completion_tokens, total_tokens,
         cost_usd, latency_ms, success, error, created_at)
        VALUES ('consolidate_cluster','claude','anthropic',500,100,600,0.05,1200,1,NULL,?)""",
        (datetime(2026, 4, 17, tzinfo=timezone.utc).isoformat(),),
    )
    conn.commit()
    conn.close()
    out = consolidation_runs(usage_db, since=None)
    assert out["runs"] == 1
    assert out["total_cost_usd"] == pytest.approx(0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stats_queries.py::test_consolidation_runs_from_usage -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `consolidation_runs` in queries.py**

Append to `src/phileas/stats/queries.py`:

```python
def consolidation_runs(usage_db: Path, since: datetime | None) -> dict:
    """Count consolidation / reflection LLM calls and cost."""
    clause, params = _since_clause(since)
    where = (clause + " AND" if clause else " WHERE") + " operation LIKE 'consolidate%' OR operation LIKE 'reflect%'"
    sql = f"""SELECT COUNT(*) AS runs,
                     COALESCE(SUM(cost_usd),0.0) AS total_cost_usd,
                     COALESCE(SUM(total_tokens),0) AS total_tokens,
                     MAX(created_at) AS last_run
              FROM llm_usage{where}"""
    with _connect(usage_db) as conn:
        row = conn.execute(sql, params).fetchone()
    return {k: row[k] for k in row.keys()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stats_queries.py -v`
Expected: PASS (all query tests).

- [ ] **Step 5: Implement graph_probe.py (no unit tests — integration only)**

```python
# src/phileas/stats/graph_probe.py
"""Read-only Kuzu probes for graph health.

The daemon holds an exclusive lock on ~/.phileas/graph. We snapshot-copy the
graph files to a tempdir and open a read-only kuzu connection — same trick
used by scripts/export_phileas.py.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def _snapshot(graph_path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="phileas-stats-"))
    dst = tmp / "graph"
    shutil.copytree(graph_path, dst)
    # Also copy WAL if present
    wal = graph_path.with_name(graph_path.name + ".wal")
    if wal.exists():
        shutil.copy2(wal, dst.with_name(dst.name + ".wal"))
    return dst


def node_edge_counts(graph_path: Path) -> dict:
    """Return {'nodes': int, 'edges': int, 'by_node_type': {...}, 'by_edge_type': {...}}.

    Returns empty result on any failure (best-effort)."""
    import kuzu  # local import — keeps import cost off the CLI hot path

    snap = _snapshot(graph_path)
    try:
        db = kuzu.Database(str(snap), read_only=True)
        conn = kuzu.Connection(db)
        by_node: dict[str, int] = {}
        for tbl in ("Memory", "Entity", "Day"):
            try:
                r = conn.execute(f"MATCH (n:{tbl}) RETURN count(n) AS c")
                by_node[tbl] = r.get_next()[0] if r.has_next() else 0
            except Exception:
                by_node[tbl] = 0
        by_edge: dict[str, int] = {}
        for rel in ("MENTIONS", "OCCURRED_ON", "RELATES_TO", "SUPERSEDES"):
            try:
                r = conn.execute(f"MATCH ()-[e:{rel}]->() RETURN count(e) AS c")
                by_edge[rel] = r.get_next()[0] if r.has_next() else 0
            except Exception:
                by_edge[rel] = 0
        return {
            "nodes": sum(by_node.values()),
            "edges": sum(by_edge.values()),
            "by_node_type": by_node,
            "by_edge_type": by_edge,
        }
    finally:
        shutil.rmtree(snap.parent, ignore_errors=True)
```

- [ ] **Step 6: Commit**

```bash
git add src/phileas/stats/queries.py src/phileas/stats/graph_probe.py tests/test_stats_queries.py
git commit -m "feat(stats): add consolidation query and kuzu graph probe"
```

## Task 5: CLI group — `stats llm`, `stats memory`, `stats graph`, `stats consolidation`

**Files:**
- Create: `src/phileas/stats/cli.py`
- Modify: `src/phileas/cli/__init__.py`

- [ ] **Step 1: Implement stats/cli.py**

```python
# src/phileas/stats/cli.py
"""`phileas stats` Click group."""

from __future__ import annotations

import json as json_mod
from datetime import datetime, timezone

import click
from rich.table import Table

from phileas.config import load_config
from phileas.stats import queries, render
from phileas.stats.time import parse_since, bucket_auto, bucketize


def _shared_flags(fn):
    fn = click.option("--since", default="7d", show_default=True, help="Time window: 24h, 7d, 30d, all.")(fn)
    fn = click.option("--bucket", default="auto", help="Bucket size: hour, day, week, auto.")(fn)
    fn = click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of tables.")(fn)
    return fn


def _resolve_window(since_expr: str) -> tuple[datetime | None, datetime, str]:
    now = datetime.now(timezone.utc)
    since = parse_since(since_expr, now)
    window = None if since is None else now - since
    bucket = bucket_auto(window)
    return since, now, bucket


@click.group("stats")
def stats():
    """Observability for Phileas — LLM, memory, recall, graph, daemon."""


@stats.command("llm")
@_shared_flags
def stats_llm(since: str, bucket: str, as_json: bool):
    """LLM usage — tokens, cost, by-op, failures."""
    cfg = load_config()
    since_dt, _, auto_bucket = _resolve_window(since)
    bucket_used = auto_bucket if bucket == "auto" else bucket
    summary = queries.llm_summary(cfg.home / "usage.db", since_dt)
    by_op = queries.llm_by_operation(cfg.home / "usage.db", since_dt)
    ts = queries.llm_timeseries(cfg.home / "usage.db", since_dt)
    cost_series = [v for _, v in bucketize(ts, bucket_used, field="cost_usd")]
    if as_json:
        click.echo(json_mod.dumps(
            {"summary": summary, "by_operation": by_op, "cost_series": cost_series, "bucket": bucket_used},
            default=str,
        ))
        return
    render.console.print(render.headline(f"LLM Usage ({since})", [
        ("Requests", str(summary["total_requests"])),
        ("Failures", str(summary["failed"])),
        ("Tokens", f"{summary['total_tokens']:,}"),
        ("Cost", f"${summary['total_cost_usd']:.4f}"),
        ("Avg latency", f"{summary['avg_latency_ms']:.0f}ms"),
        (f"Cost {bucket_used}", render.spark(cost_series)),
    ]))
    if by_op:
        t = Table(title="By Operation")
        for col in ("Operation", "Requests", "Tokens", "Cost", "Avg ms", "Fails"):
            t.add_column(col)
        for r in by_op:
            t.add_row(r["operation"], str(r["requests"]), f"{r['total_tokens']:,}",
                      f"${r['cost_usd']:.4f}", f"{r['avg_latency_ms']:.0f}",
                      str(r["failures"]) if r["failures"] else "")
        render.console.print(t)


@stats.command("memory")
@_shared_flags
def stats_memory(since: str, bucket: str, as_json: bool):
    """Memory lifecycle — memorize rate by type."""
    cfg = load_config()
    since_dt, _, auto_bucket = _resolve_window(since)
    bucket_used = auto_bucket if bucket == "auto" else bucket
    data = queries.memory_lifecycle(cfg.db_path, since_dt)
    ts = queries.memory_timeseries(cfg.db_path, since_dt)
    rate_series = [v for _, v in bucketize(ts, bucket_used, field="count")]
    if as_json:
        click.echo(json_mod.dumps(
            {"summary": data, "rate_series": rate_series, "bucket": bucket_used},
            default=str,
        ))
        return
    render.console.print(render.headline(f"Memory Lifecycle ({since})", [
        ("Created", str(data["total_created"])),
        (f"Rate {bucket_used}", render.spark(rate_series)),
    ]))
    t = Table(title="By Type")
    for col in ("Type", "Created", "Active", "Archived"):
        t.add_column(col)
    for r in data["by_type"]:
        t.add_row(r["type"], str(r["created"]), str(r["active"] or 0), str(r["archived"] or 0))
    render.console.print(t)


@stats.command("graph")
@click.option("--json", "as_json", is_flag=True)
def stats_graph(as_json: bool):
    """Graph health — node/edge counts by type."""
    from phileas.stats import graph_probe

    cfg = load_config()
    try:
        data = graph_probe.node_edge_counts(cfg.graph_path)
    except Exception as e:
        click.echo(f"graph probe failed: {e}", err=True)
        raise SystemExit(1)
    if as_json:
        click.echo(json_mod.dumps(data))
        return
    render.console.print(render.headline("Graph", [
        ("Nodes", f"{data['nodes']:,}"),
        ("Edges", f"{data['edges']:,}"),
    ]))
    t = Table(title="By Type")
    t.add_column("Kind")
    t.add_column("Label")
    t.add_column("Count")
    for k, v in data["by_node_type"].items():
        t.add_row("node", k, f"{v:,}")
    for k, v in data["by_edge_type"].items():
        t.add_row("edge", k, f"{v:,}")
    render.console.print(t)


@stats.command("consolidation")
@_shared_flags
def stats_consolidation(since: str, bucket: str, as_json: bool):
    """Consolidation / reflection runs — count, cost, last run."""
    cfg = load_config()
    since_dt, _, _ = _resolve_window(since)
    data = queries.consolidation_runs(cfg.home / "usage.db", since_dt)
    if as_json:
        click.echo(json_mod.dumps(data, default=str))
        return
    render.console.print(render.headline(f"Consolidation ({since})", [
        ("Runs", str(data["runs"])),
        ("Cost", f"${data['total_cost_usd']:.4f}"),
        ("Tokens", f"{data['total_tokens']:,}"),
        ("Last run", data["last_run"] or "never"),
    ]))


@stats.command("overview")
@_shared_flags
@click.pass_context
def stats_overview(ctx, since: str, bucket: str, as_json: bool):
    """Everything at a glance."""
    ctx.invoke(stats_llm, since=since, bucket=bucket, as_json=as_json)
    ctx.invoke(stats_memory, since=since, bucket=bucket, as_json=as_json)
    ctx.invoke(stats_graph, as_json=as_json)
    ctx.invoke(stats_consolidation, since=since, bucket=bucket, as_json=as_json)
```

- [ ] **Step 2: Register group in `src/phileas/cli/__init__.py`**

Add import and registration. Modify `src/phileas/cli/__init__.py`:

```python
# Add near other imports
from phileas.stats.cli import stats

# Add near bottom, with other app.add_command lines
app.add_command(stats)
```

- [ ] **Step 3: Smoke-test manually**

Run: `uv run phileas stats llm --since 7d` — expect the same numbers as the old `phileas usage` (when no --since applied) but scoped to the window.
Run: `uv run phileas stats overview --json` — expect JSON lines per section.
Run: `uv run phileas stats graph` — expect node/edge counts.

- [ ] **Step 4: Commit**

```bash
git add src/phileas/stats/cli.py src/phileas/cli/__init__.py
git commit -m "feat(stats): add `phileas stats` group with llm/memory/graph/consolidation/overview"
```

---

# Phase 2 — Recall + Ingest Instrumentation

## Task 6: MetricsWriter + schema

**Files:**
- Create: `src/phileas/stats/writer.py`
- Create: `tests/test_stats_writer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stats_writer.py
import sqlite3
from pathlib import Path

from phileas.stats.writer import MetricsWriter


def test_writer_creates_schema(tmp_path: Path):
    w = MetricsWriter(tmp_path / "metrics.db")
    w.record_recall(query_len=10, top_k=10, returned=5, top1_score=0.9,
                    mean_score=0.7, empty=False, hot_hit=True, latency_ms=12.3)
    w.close()
    conn = sqlite3.connect(tmp_path / "metrics.db")
    rows = conn.execute("SELECT top1_score, hot_hit, empty FROM recall_events").fetchall()
    assert rows == [(0.9, 1, 0)]


def test_writer_records_ingest(tmp_path: Path):
    w = MetricsWriter(tmp_path / "metrics.db")
    w.record_ingest(memory_type="event", importance=7, entity_count=2, deduped=False, source="cli")
    w.close()
    conn = sqlite3.connect(tmp_path / "metrics.db")
    row = conn.execute("SELECT memory_type, deduped, entity_count FROM ingest_events").fetchone()
    assert row == ("event", 0, 2)


def test_writer_records_daemon(tmp_path: Path):
    w = MetricsWriter(tmp_path / "metrics.db")
    w.record_daemon("start")
    w.record_daemon("lock_contention", payload={"path": "graph"})
    w.close()
    conn = sqlite3.connect(tmp_path / "metrics.db")
    kinds = [r[0] for r in conn.execute("SELECT kind FROM daemon_events ORDER BY id")]
    assert kinds == ["start", "lock_contention"]


def test_writer_swallows_bad_path(tmp_path: Path):
    # Pointing at a path that's a directory should not raise on record_*
    bad = tmp_path / "not-a-db"
    bad.mkdir()
    w = MetricsWriter(bad / "impossible" / "metrics.db")  # parent creation should still work
    # but force a failure by monkey-patching the conn to None after init
    w._conn = None  # noqa: SLF001
    # Must not raise
    w.record_recall(query_len=1, top_k=1, returned=0, top1_score=None,
                    mean_score=None, empty=True, hot_hit=False, latency_ms=1.0)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_stats_writer.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement writer.py**

```python
# src/phileas/stats/writer.py
"""Best-effort SQLite writer for phileas metrics.

Writes to ~/.phileas/metrics.db. All public methods swallow exceptions and
log at debug — metrics must never break user operations.
"""

from __future__ import annotations

import json as json_mod
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS recall_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    query_len INTEGER,
    top_k INTEGER,
    returned INTEGER,
    top1_score REAL,
    mean_score REAL,
    empty INTEGER NOT NULL,
    hot_hit INTEGER,
    latency_ms REAL,
    stage_timings_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_recall_created ON recall_events(created_at);

CREATE TABLE IF NOT EXISTS ingest_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    memory_type TEXT,
    importance INTEGER,
    entity_count INTEGER,
    deduped INTEGER NOT NULL,
    source TEXT
);
CREATE INDEX IF NOT EXISTS idx_ingest_created ON ingest_events(created_at);

CREATE TABLE IF NOT EXISTS daemon_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_daemon_created_kind ON daemon_events(created_at, kind);
"""


class MetricsWriter:
    """Single-writer SQLite sink for phileas metrics."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._conn: sqlite3.Connection | None = None
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
        except Exception as e:
            log.debug("metrics writer init failed", extra={"err": str(e)})
            self._conn = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def record_recall(
        self,
        query_len: int,
        top_k: int,
        returned: int,
        top1_score: float | None,
        mean_score: float | None,
        empty: bool,
        hot_hit: bool,
        latency_ms: float,
        stage_timings: dict | None = None,
    ) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                """INSERT INTO recall_events
                   (created_at, query_len, top_k, returned, top1_score, mean_score,
                    empty, hot_hit, latency_ms, stage_timings_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    self._now(), query_len, top_k, returned, top1_score, mean_score,
                    int(empty), int(hot_hit), latency_ms,
                    json_mod.dumps(stage_timings) if stage_timings else None,
                ),
            )
        except Exception as e:
            log.debug("record_recall failed", extra={"err": str(e)})

    def record_ingest(
        self,
        memory_type: str | None,
        importance: int | None,
        entity_count: int,
        deduped: bool,
        source: str,
    ) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                """INSERT INTO ingest_events
                   (created_at, memory_type, importance, entity_count, deduped, source)
                   VALUES (?,?,?,?,?,?)""",
                (self._now(), memory_type, importance, entity_count, int(deduped), source),
            )
        except Exception as e:
            log.debug("record_ingest failed", extra={"err": str(e)})

    def record_daemon(self, kind: str, payload: dict | None = None) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO daemon_events (created_at, kind, payload_json) VALUES (?,?,?)",
                (self._now(), kind, json_mod.dumps(payload) if payload else None),
            )
        except Exception as e:
            log.debug("record_daemon failed", extra={"err": str(e)})

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_stats_writer.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phileas/stats/writer.py tests/test_stats_writer.py
git commit -m "feat(stats): add MetricsWriter with best-effort SQLite sink"
```

## Task 7: Wire MetricsWriter into `engine.recall()`

**Files:**
- Modify: `src/phileas/engine.py` (`__init__` + `recall()`)
- Modify: `tests/test_engine.py` (add assertion on recall_events)

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_engine.py` (adapt fixture name to the existing engine fixture in that file):

```python
def test_recall_emits_metric_event(engine, tmp_path):
    # Seed one memory
    engine.memorize("integration test seed", memory_type="knowledge")
    _ = engine.recall("integration test", top_k=5, _skip_llm=True)

    import sqlite3
    metrics_db = engine.config.home / "metrics.db"
    assert metrics_db.exists()
    conn = sqlite3.connect(metrics_db)
    row = conn.execute(
        "SELECT top_k, returned, empty FROM recall_events ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[0] == 5
    assert row[2] in (0, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine.py::test_recall_emits_metric_event -v`
Expected: FAIL (no metrics.db).

- [ ] **Step 3: Instrument `engine.recall()`**

In `src/phileas/engine.py`:

1. At the top with other imports:

```python
from phileas.stats.writer import MetricsWriter
```

2. In `MemoryEngine.__init__`, after `self._hot = HotMemorySet.build(...)`:

```python
self._metrics = MetricsWriter(self.config.home / "metrics.db")
```

3. In `recall()`, replace the final `return results` block so it also records the event. Find the `return results` at end of the `with OpTimer(...)` block and wrap:

```python
# (immediately before `return results` inside the OpTimer with-block)
try:
    top1 = results[0]["score"] if results else None
    mean = sum(r.get("score", 0.0) for r in results) / len(results) if results else None
    self._metrics.record_recall(
        query_len=len(query),
        top_k=top_k,
        returned=len(results),
        top1_score=top1,
        mean_score=mean,
        empty=not results,
        hot_hit=bool(getattr(timer, "extra", {}).get("hot_hit", False)),
        latency_ms=timer.elapsed_ms if hasattr(timer, "elapsed_ms") else 0.0,
    )
except Exception:
    pass
return results
```

Note: `timer.elapsed_ms` — confirm the attribute name used by `OpTimer` in `src/phileas/logging.py`; adjust if it's called `duration_ms` or similar. If the timer hasn't computed elapsed yet inside the with-block, compute it manually with `time.perf_counter` captured at function entry.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_engine.py::test_recall_emits_metric_event -v`
Expected: PASS.

- [ ] **Step 5: Run full engine tests to catch regressions**

Run: `uv run pytest tests/test_engine.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phileas/engine.py tests/test_engine.py
git commit -m "feat(stats): instrument engine.recall to emit metric events"
```

## Task 8: Wire MetricsWriter into `engine.memorize()`

**Files:**
- Modify: `src/phileas/engine.py` (`memorize()`)
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Write failing test**

```python
def test_memorize_emits_metric_event(engine):
    engine.memorize("ingest metric test", memory_type="knowledge", importance=5)
    import sqlite3
    conn = sqlite3.connect(engine.config.home / "metrics.db")
    row = conn.execute(
        "SELECT memory_type, importance, deduped, source FROM ingest_events ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row[0] == "knowledge"
    assert row[1] == 5
    assert row[2] in (0, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine.py::test_memorize_emits_metric_event -v`
Expected: FAIL (no ingest_events row).

- [ ] **Step 3: Instrument `engine.memorize()`**

At the point where `memorize()` returns after a successful add, capture the dedup flag (if the existing code has a merge path that returns an existing id, that's `deduped=True`; otherwise `deduped=False`). Example:

```python
# inside memorize(), after determining `result` and whether dedup happened
try:
    self._metrics.record_ingest(
        memory_type=memory_type,
        importance=importance,
        entity_count=len(entities) if entities is not None else 0,
        deduped=bool(result.get("deduped", False)),
        source=source or "cli",
    )
except Exception:
    pass
```

If `memorize()` does not currently accept `source`, add it as an optional kwarg with default `"cli"`; thread it through from MCP call sites as `"mcp"` in a later commit if desired.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_engine.py::test_memorize_emits_metric_event -v`
Expected: PASS.

- [ ] **Step 5: Run full engine tests**

Run: `uv run pytest tests/test_engine.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phileas/engine.py tests/test_engine.py
git commit -m "feat(stats): instrument engine.memorize to emit ingest events"
```

## Task 9: `stats recall` and `stats ingest` subcommands

**Files:**
- Modify: `src/phileas/stats/queries.py` (add `recall_summary`, `ingest_summary`)
- Modify: `src/phileas/stats/cli.py` (add subcommands)
- Modify: `tests/test_stats_queries.py` (add metrics_db fixture + tests)

- [ ] **Step 1: Add failing tests for recall_summary and ingest_summary**

Append to `tests/test_stats_queries.py`:

```python
@pytest.fixture
def metrics_db(tmp_path: Path) -> Path:
    p = tmp_path / "metrics.db"
    conn = sqlite3.connect(p)
    conn.executescript(
        """
        CREATE TABLE recall_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            query_len INTEGER, top_k INTEGER, returned INTEGER,
            top1_score REAL, mean_score REAL,
            empty INTEGER NOT NULL, hot_hit INTEGER,
            latency_ms REAL, stage_timings_json TEXT
        );
        CREATE TABLE ingest_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            memory_type TEXT, importance INTEGER,
            entity_count INTEGER, deduped INTEGER NOT NULL, source TEXT
        );
        """
    )
    now = datetime(2026, 4, 17, tzinfo=timezone.utc).isoformat()
    conn.executemany(
        """INSERT INTO recall_events
        (created_at, query_len, top_k, returned, top1_score, mean_score, empty, hot_hit, latency_ms)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        [
            (now, 20, 10, 5, 0.9, 0.7, 0, 1, 40.0),
            (now, 15, 10, 0, None, None, 1, 0, 200.0),
            (now, 30, 10, 8, 0.8, 0.6, 0, 0, 60.0),
        ],
    )
    conn.executemany(
        """INSERT INTO ingest_events
        (created_at, memory_type, importance, entity_count, deduped, source)
        VALUES (?,?,?,?,?,?)""",
        [
            (now, "event", 7, 2, 0, "mcp"),
            (now, "event", 5, 0, 1, "mcp"),
            (now, "knowledge", 8, 3, 0, "cli"),
        ],
    )
    conn.commit()
    conn.close()
    return p


def test_recall_summary(metrics_db: Path):
    from phileas.stats.queries import recall_summary
    out = recall_summary(metrics_db, since=None)
    assert out["total_recalls"] == 3
    assert out["empty_rate"] == pytest.approx(1 / 3)
    assert out["hot_hit_rate"] == pytest.approx(1 / 3)
    assert out["avg_top1"] == pytest.approx((0.9 + 0.8) / 2)


def test_ingest_summary(metrics_db: Path):
    from phileas.stats.queries import ingest_summary
    out = ingest_summary(metrics_db, since=None)
    assert out["total_ingests"] == 3
    assert out["dedup_rate"] == pytest.approx(1 / 3)
    by_type = {r["memory_type"]: r for r in out["by_type"]}
    assert by_type["event"]["count"] == 2
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_stats_queries.py -v`
Expected: FAIL on the two new tests.

- [ ] **Step 3: Add `recall_summary` and `ingest_summary`**

Append to `src/phileas/stats/queries.py`:

```python
def recall_summary(metrics_db: Path, since: datetime | None) -> dict:
    where, params = _since_clause(since)
    with _connect(metrics_db) as conn:
        row = conn.execute(
            f"""SELECT
                COUNT(*) AS total_recalls,
                COALESCE(AVG(top1_score), 0.0) AS avg_top1,
                COALESCE(AVG(mean_score), 0.0) AS avg_mean,
                COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
                SUM(empty) * 1.0 / NULLIF(COUNT(*), 0) AS empty_rate,
                SUM(hot_hit) * 1.0 / NULLIF(COUNT(*), 0) AS hot_hit_rate
            FROM recall_events{where}""",
            params,
        ).fetchone()
        lat = conn.execute(
            f"SELECT latency_ms FROM recall_events{where} ORDER BY latency_ms",
            params,
        ).fetchall()
    latencies = [r["latency_ms"] for r in lat if r["latency_ms"] is not None]
    def _p(q: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(q * len(latencies)))
        return float(latencies[idx])
    return {
        "total_recalls": row["total_recalls"],
        "avg_top1": row["avg_top1"] or 0.0,
        "avg_mean": row["avg_mean"] or 0.0,
        "avg_latency_ms": row["avg_latency_ms"] or 0.0,
        "empty_rate": row["empty_rate"] or 0.0,
        "hot_hit_rate": row["hot_hit_rate"] or 0.0,
        "p50_latency_ms": _p(0.5),
        "p95_latency_ms": _p(0.95),
    }


def ingest_summary(metrics_db: Path, since: datetime | None) -> dict:
    where, params = _since_clause(since)
    with _connect(metrics_db) as conn:
        row = conn.execute(
            f"""SELECT
                COUNT(*) AS total_ingests,
                COALESCE(AVG(entity_count), 0.0) AS avg_entities,
                SUM(deduped) * 1.0 / NULLIF(COUNT(*), 0) AS dedup_rate,
                SUM(CASE WHEN entity_count = 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0)
                    AS zero_entity_rate
            FROM ingest_events{where}""",
            params,
        ).fetchone()
        by_type = conn.execute(
            f"""SELECT memory_type, COUNT(*) AS count,
                       AVG(entity_count) AS avg_entities,
                       SUM(deduped) * 1.0 / COUNT(*) AS dedup_rate
                FROM ingest_events{where}
                GROUP BY memory_type
                ORDER BY count DESC""",
            params,
        ).fetchall()
    return {
        "total_ingests": row["total_ingests"],
        "avg_entities": row["avg_entities"] or 0.0,
        "dedup_rate": row["dedup_rate"] or 0.0,
        "zero_entity_rate": row["zero_entity_rate"] or 0.0,
        "by_type": [dict(r) for r in by_type],
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_stats_queries.py -v`
Expected: PASS.

- [ ] **Step 5: Add CLI subcommands**

Append to `src/phileas/stats/cli.py`:

```python
@stats.command("recall")
@_shared_flags
def stats_recall(since: str, bucket: str, as_json: bool):
    """Recall quality — top1, empty rate, latency, hot-hit rate."""
    cfg = load_config()
    since_dt, _, _ = _resolve_window(since)
    metrics_db = cfg.home / "metrics.db"
    if not metrics_db.exists():
        click.echo("No recall events yet. Run some recalls first.", err=True)
        raise SystemExit(1)
    data = queries.recall_summary(metrics_db, since_dt)
    if as_json:
        click.echo(json_mod.dumps(data, default=str))
        return
    render.console.print(render.headline(f"Recall ({since})", [
        ("Queries", str(data["total_recalls"])),
        ("Empty rate", f"{data['empty_rate']:.1%}"),
        ("Hot-hit rate", f"{data['hot_hit_rate']:.1%}"),
        ("Avg top-1", f"{data['avg_top1']:.3f}"),
        ("Latency p50", f"{data['p50_latency_ms']:.0f}ms"),
        ("Latency p95", f"{data['p95_latency_ms']:.0f}ms"),
    ]))


@stats.command("ingest")
@_shared_flags
def stats_ingest(since: str, bucket: str, as_json: bool):
    """Ingest rate, dedup rate, entity coverage by type."""
    cfg = load_config()
    since_dt, _, _ = _resolve_window(since)
    metrics_db = cfg.home / "metrics.db"
    if not metrics_db.exists():
        click.echo("No ingest events yet.", err=True)
        raise SystemExit(1)
    data = queries.ingest_summary(metrics_db, since_dt)
    if as_json:
        click.echo(json_mod.dumps(data, default=str))
        return
    render.console.print(render.headline(f"Ingest ({since})", [
        ("Memorize calls", str(data["total_ingests"])),
        ("Dedup rate", f"{data['dedup_rate']:.1%}"),
        ("Zero-entity rate", f"{data['zero_entity_rate']:.1%}"),
        ("Avg entities", f"{data['avg_entities']:.2f}"),
    ]))
    t = Table(title="By Type")
    for col in ("Type", "Count", "Avg entities", "Dedup rate"):
        t.add_column(col)
    for r in data["by_type"]:
        t.add_row(r["memory_type"] or "(null)", str(r["count"]),
                  f"{(r['avg_entities'] or 0):.2f}", f"{(r['dedup_rate'] or 0):.1%}")
    render.console.print(t)
```

Also add `ctx.invoke(stats_recall, ...)` and `ctx.invoke(stats_ingest, ...)` to `stats_overview` (guarded by `metrics_db.exists()` to stay friendly on fresh installs).

- [ ] **Step 6: Smoke-test**

Run: `uv run phileas stats recall --since 7d`
Run: `uv run phileas stats ingest --since 7d --json`

- [ ] **Step 7: Commit**

```bash
git add src/phileas/stats/queries.py src/phileas/stats/cli.py tests/test_stats_queries.py
git commit -m "feat(stats): add recall and ingest subcommands backed by metrics.db"
```

---

# Phase 3 — Daemon Health + UX Polish

## Task 10: Daemon instrumentation

**Files:**
- Modify: `src/phileas/daemon.py`
- Modify: `src/phileas/graph.py` (lock exception path)
- Modify: `src/phileas/engine.py` (`consolidate`/`reflect` exits)

- [ ] **Step 1: Emit `start` / `stop` / `error`**

In `src/phileas/daemon.py`, construct one `MetricsWriter` at daemon boot:

```python
from phileas.stats.writer import MetricsWriter

metrics = MetricsWriter(cfg.home / "metrics.db")
metrics.record_daemon("start", payload={"pid": os.getpid()})
try:
    run_forever(...)
except Exception as e:
    metrics.record_daemon("error", payload={"err": str(e)[:500]})
    raise
finally:
    metrics.record_daemon("stop")
    metrics.close()
```

- [ ] **Step 2: Emit `lock_contention`**

In `src/phileas/graph.py` (wherever Kuzu writes catch IOError/lock errors), add `self._metrics.record_daemon("lock_contention", payload={"op": op_name})`. Thread a `MetricsWriter` reference through `GraphStore.__init__` (optional kwarg, default None — gracefully skip if missing).

- [ ] **Step 3: Emit `consolidate_run` / `reflect_run`**

In `engine.consolidate()` and `engine.reflect()`, at successful return:

```python
self._metrics.record_daemon(
    "consolidate_run",
    payload={"clusters": n_clusters, "new_memories": n_new, "duration_ms": duration_ms, "cost_usd": cost_usd},
)
```

Use the same `self._metrics` created in Task 7.

- [ ] **Step 4: Manual verification**

Start the daemon, trigger a consolidation, stop the daemon. Check:

```bash
sqlite3 ~/.phileas/metrics.db 'SELECT created_at, kind FROM daemon_events ORDER BY id DESC LIMIT 10'
```

Expect `start`, `consolidate_run`, `stop` entries.

- [ ] **Step 5: Commit**

```bash
git add src/phileas/daemon.py src/phileas/graph.py src/phileas/engine.py
git commit -m "feat(stats): instrument daemon lifecycle, lock contention, consolidation/reflection"
```

## Task 11: `stats daemon` subcommand

**Files:**
- Modify: `src/phileas/stats/queries.py` (add `daemon_summary`)
- Modify: `src/phileas/stats/cli.py` (add `stats daemon`)
- Modify: `tests/test_stats_queries.py`

- [ ] **Step 1: Failing test**

Append:

```python
def test_daemon_summary(metrics_db: Path):
    # Seed daemon_events table
    conn = sqlite3.connect(metrics_db)
    conn.executescript(
        """CREATE TABLE daemon_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL, kind TEXT NOT NULL, payload_json TEXT
        );"""
    )
    now = datetime(2026, 4, 17, tzinfo=timezone.utc)
    rows = [
        ((now - timedelta(hours=2)).isoformat(), "start", None),
        ((now - timedelta(hours=1)).isoformat(), "lock_contention", None),
        ((now - timedelta(hours=1)).isoformat(), "error", '{"err":"x"}'),
        (now.isoformat(), "consolidate_run", '{"clusters": 4}'),
    ]
    conn.executemany("INSERT INTO daemon_events (created_at, kind, payload_json) VALUES (?,?,?)", rows)
    conn.commit()
    conn.close()
    from phileas.stats.queries import daemon_summary
    out = daemon_summary(metrics_db, since=None)
    assert out["errors"] == 1
    assert out["lock_contentions"] == 1
    assert out["last_start"] is not None
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_stats_queries.py::test_daemon_summary -v`
Expected: FAIL.

- [ ] **Step 3: Implement `daemon_summary`**

Append to `src/phileas/stats/queries.py`:

```python
def daemon_summary(metrics_db: Path, since: datetime | None) -> dict:
    where, params = _since_clause(since)
    with _connect(metrics_db) as conn:
        counts = conn.execute(
            f"SELECT kind, COUNT(*) AS c FROM daemon_events{where} GROUP BY kind",
            params,
        ).fetchall()
        last_start = conn.execute(
            "SELECT MAX(created_at) AS ts FROM daemon_events WHERE kind='start'"
        ).fetchone()["ts"]
        last_stop = conn.execute(
            "SELECT MAX(created_at) AS ts FROM daemon_events WHERE kind='stop'"
        ).fetchone()["ts"]
    by_kind = {r["kind"]: r["c"] for r in counts}
    return {
        "by_kind": by_kind,
        "errors": by_kind.get("error", 0),
        "lock_contentions": by_kind.get("lock_contention", 0),
        "consolidate_runs": by_kind.get("consolidate_run", 0),
        "reflect_runs": by_kind.get("reflect_run", 0),
        "last_start": last_start,
        "last_stop": last_stop,
    }
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_stats_queries.py::test_daemon_summary -v`
Expected: PASS.

- [ ] **Step 5: Add `stats daemon` subcommand**

Append to `src/phileas/stats/cli.py`:

```python
@stats.command("daemon")
@_shared_flags
def stats_daemon(since: str, bucket: str, as_json: bool):
    """Daemon health — uptime, errors, lock contention, consolidation runs."""
    cfg = load_config()
    since_dt, _, _ = _resolve_window(since)
    metrics_db = cfg.home / "metrics.db"
    if not metrics_db.exists():
        click.echo("No daemon events yet.", err=True)
        raise SystemExit(1)
    data = queries.daemon_summary(metrics_db, since_dt)
    if as_json:
        click.echo(json_mod.dumps(data, default=str))
        return
    render.console.print(render.headline(f"Daemon ({since})", [
        ("Errors", str(data["errors"])),
        ("Lock contentions", str(data["lock_contentions"])),
        ("Consolidate runs", str(data["consolidate_runs"])),
        ("Reflect runs", str(data["reflect_runs"])),
        ("Last start", data["last_start"] or "—"),
        ("Last stop", data["last_stop"] or "—"),
    ]))
```

Add to overview as well.

- [ ] **Step 6: Commit**

```bash
git add src/phileas/stats/queries.py src/phileas/stats/cli.py tests/test_stats_queries.py
git commit -m "feat(stats): add `stats daemon` subcommand"
```

## Task 12: `--watch` loop

**Files:**
- Modify: `src/phileas/stats/cli.py`

- [ ] **Step 1: Add `--watch` option to the group**

Replace `_shared_flags` with a wrapper that also supports `--watch N`:

```python
def _shared_flags(fn):
    fn = click.option("--since", default="7d", show_default=True)(fn)
    fn = click.option("--bucket", default="auto")(fn)
    fn = click.option("--json", "as_json", is_flag=True)(fn)
    fn = click.option("--watch", default=0, type=int, metavar="SECS",
                      help="Re-run every N seconds (0 = one-shot).")(fn)
    return fn
```

- [ ] **Step 2: Factor a `_run_with_watch` helper**

```python
import time

def _run_with_watch(fn, watch: int, *args, **kwargs):
    if watch <= 0:
        return fn(*args, **kwargs)
    try:
        while True:
            render.console.clear()
            fn(*args, **kwargs)
            time.sleep(watch)
    except KeyboardInterrupt:
        pass
```

Wrap each subcommand's body by passing `watch` through. Example for `stats_llm`:

```python
@stats.command("llm")
@_shared_flags
def stats_llm(since, bucket, as_json, watch):
    def _body():
        # (existing body, unchanged)
        ...
    _run_with_watch(_body, watch)
```

Apply to all subcommands.

- [ ] **Step 3: Smoke-test**

Run: `uv run phileas stats overview --watch 3` — screen should redraw every 3s; Ctrl-C exits cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/phileas/stats/cli.py
git commit -m "feat(stats): add --watch loop to all subcommands"
```

## Task 13: Keep `phileas usage` as alias + migration note

**Files:**
- Modify: `src/phileas/cli/commands.py`
- Modify: `src/phileas/cli/__init__.py`

- [ ] **Step 1: Make `usage` delegate to `stats llm`**

In `src/phileas/cli/commands.py`, replace the body of the existing `usage` command:

```python
@click.command()
@click.option("--recent", default=0, type=int, help="Deprecated — shown for compatibility.")
@click.pass_context
def usage(ctx, recent: int):
    """Alias for `phileas stats llm` (kept for backward compatibility)."""
    from phileas.stats.cli import stats_llm
    ctx.invoke(stats_llm, since="all", bucket="auto", as_json=False, watch=0)
```

- [ ] **Step 2: Manual verification**

Run: `uv run phileas usage` — expect the new `stats llm` output.

- [ ] **Step 3: Commit**

```bash
git add src/phileas/cli/commands.py src/phileas/cli/__init__.py
git commit -m "refactor(stats): retarget `phileas usage` to `stats llm`"
```

---

## Self-Review

**Spec coverage:**

| Spec item | Task |
|---|---|
| `stats overview` | Task 5 (+ extended in 9, 11) |
| `stats llm` | Task 5 |
| `stats memory` | Task 5 |
| `stats recall` | Task 9 |
| `stats graph` | Task 5 (+ Task 4 probe) |
| `stats consolidation` | Task 5 (+ Task 4 query) |
| `stats ingest` | Task 9 |
| `stats daemon` | Task 11 |
| `metrics.db` schema (recall/ingest/daemon) | Task 6 |
| `MetricsWriter` best-effort + WAL | Task 6 |
| Instrument `recall()` | Task 7 |
| Instrument `memorize()` | Task 8 |
| Instrument daemon lifecycle + lock + consolidate/reflect | Task 10 |
| `--since`, `--bucket`, `--json` | Task 5 (shared flags) |
| `--watch` | Task 12 |
| Sparklines (no new dep) | Task 2 |
| Kuzu snapshot probe | Task 4 |
| `usage` alias for `stats llm` | Task 13 |
| Unit tests: time/render/writer/queries | Tasks 1, 2, 3, 4, 6, 9, 11 |

All spec items have a task.

**Placeholder scan:** No `TBD`/`TODO`. One reference in Task 7 asks the implementer to confirm `OpTimer`'s elapsed-ms attribute name and provides a concrete fallback (measure via `time.perf_counter` at function entry) — this is guidance, not a gap.

**Type consistency:** `MetricsWriter` method names (`record_recall`, `record_ingest`, `record_daemon`) are consistent across Tasks 6–11. `queries.*` return dicts with the same keys referenced in `cli.py` (`total_recalls`, `empty_rate`, `hot_hit_rate`, `by_type`, `dedup_rate`, `errors`, `lock_contentions`). Flags (`since`, `bucket`, `as_json`, `watch`) are consistent across subcommands.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-17-phileas-stats.md`.
