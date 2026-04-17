import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from phileas.stats.queries import consolidation_runs, llm_summary, memory_lifecycle


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
        CREATE TABLE memory_items (
            id TEXT PRIMARY KEY,
            memory_type TEXT NOT NULL,
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
        "INSERT INTO memory_items (id, memory_type, status, created_at) VALUES (?,?,?,?)",
        rows,
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


def test_consolidation_runs_from_usage(usage_db: Path):
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


def test_daemon_summary(tmp_path: Path):
    p = tmp_path / "daemon_metrics.db"
    conn = sqlite3.connect(p)
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
        (now.isoformat(), "reflect_run", '{"insights": 4}'),
    ]
    conn.executemany(
        "INSERT INTO daemon_events (created_at, kind, payload_json) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    from phileas.stats.queries import daemon_summary

    out = daemon_summary(p, since=None)
    assert out["errors"] == 1
    assert out["lock_contentions"] == 1
    assert out["reflect_runs"] == 1
    assert out["last_start"] is not None
