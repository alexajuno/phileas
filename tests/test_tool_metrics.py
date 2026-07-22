"""Instrumentation tests (AA-106): output_chars capture + the stats summary.

`output_chars` is the realized context cost a tool dumps into the agent — the
before/after surface for the pointer split. These also cover the guarded
migration that backfills the column onto an already-created metrics.db.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from phileas.stats.queries import tool_calls_summary
from phileas.stats.writer import MetricsWriter


def test_migration_backfills_output_chars_on_legacy_db(tmp_path: Path):
    db = tmp_path / "metrics.db"
    # Simulate a pre-AA-106 tool_calls table (no output_chars column).
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE tool_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, "
        "tool TEXT NOT NULL, latency_ms REAL, ok INTEGER NOT NULL, error TEXT)"
    )
    con.commit()
    con.close()

    writer = MetricsWriter(db)
    cols = {row[1] for row in writer._conn.execute("PRAGMA table_info(tool_calls)")}
    assert "output_chars" in cols


def test_record_tool_call_persists_output_chars(tmp_path: Path):
    writer = MetricsWriter(tmp_path / "metrics.db")
    writer.record_tool_call(tool="recall", latency_ms=12.0, ok=True, output_chars=1234)
    row = writer._conn.execute("SELECT tool, output_chars FROM tool_calls").fetchone()
    assert row == ("recall", 1234)


def test_tool_calls_summary_percentiles_and_drill_in_rate(tmp_path: Path):
    writer = MetricsWriter(tmp_path / "metrics.db")
    for chars in (1000, 2000, 3000, 4000):
        writer.record_tool_call(tool="recall", latency_ms=10.0, ok=True, output_chars=chars)
    writer.record_tool_call(tool="source", latency_ms=4.0, ok=True, output_chars=300)
    writer.record_tool_call(tool="get_source_memories", latency_ms=8.0, ok=True, output_chars=900)
    writer.record_tool_call(tool="recall", latency_ms=10.0, ok=False, error="ValueError", output_chars=None)

    summary = tool_calls_summary(tmp_path / "metrics.db", None)

    by_tool = {t["tool"]: t for t in summary["by_tool"]}
    assert by_tool["recall"]["calls"] == 5
    assert by_tool["recall"]["errors"] == 1
    # 4 char samples (the failed call had None) -> p50 picks an upper-middle sample
    assert by_tool["recall"]["p50_chars"] in (2000, 3000)
    assert by_tool["recall"]["p95_chars"] == 4000
    # drill-in rate = (source + get_source_memories) / (recall + recall_recent) = 2 / 5
    assert summary["drill_in_rate"] == 2 / 5
    assert summary["total_calls"] == 7


def test_tool_calls_summary_empty_db(tmp_path: Path):
    writer = MetricsWriter(tmp_path / "metrics.db")  # noqa: F841 — creates schema
    summary = tool_calls_summary(tmp_path / "metrics.db", None)
    assert summary == {"total_calls": 0, "drill_in_rate": 0.0, "by_tool": []}
