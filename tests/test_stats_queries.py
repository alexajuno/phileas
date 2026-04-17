import sqlite3
from datetime import datetime, timedelta, timezone
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
    conn.executemany("INSERT INTO memories (id, type, status, created_at) VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return p


def test_memory_lifecycle_by_type(phileas_db: Path):
    out = memory_lifecycle(phileas_db, since=None)
    by_type = {row["type"]: row for row in out["by_type"]}
    assert by_type["event"]["created"] == 3
    assert by_type["event"]["active"] == 2
    assert by_type["knowledge"]["created"] == 1
