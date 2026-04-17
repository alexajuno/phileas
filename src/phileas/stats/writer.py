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
                    self._now(),
                    query_len,
                    top_k,
                    returned,
                    top1_score,
                    mean_score,
                    int(empty),
                    int(hot_hit),
                    latency_ms,
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
