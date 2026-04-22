"""SQLite storage backend for Phileas.

Canonical data store. ChromaDB and KuzuDB are derived indexes
that can be rebuilt from this database.
"""

import functools
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from phileas.models import Event, MemoryItem


def _locked(method):
    """Serialize Database access across threads via self._lock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


DEFAULT_DB_PATH = Path.home() / ".phileas" / "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 5,
    tier INTEGER NOT NULL DEFAULT 2,
    status TEXT NOT NULL DEFAULT 'active',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    daily_ref TEXT,
    consolidated_into TEXT REFERENCES memory_items(id),
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    last_reinforced TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_sessions (
    session_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    received_at TEXT NOT NULL,
    extraction_status TEXT NOT NULL DEFAULT 'pending',
    extraction_error TEXT,
    memory_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_items_status ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_items_tier ON memory_items(tier);
CREATE INDEX IF NOT EXISTS idx_items_type ON memory_items(memory_type);
CREATE INDEX IF NOT EXISTS idx_items_daily_ref ON memory_items(daily_ref);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(extraction_status);
CREATE INDEX IF NOT EXISTS idx_events_received ON events(received_at);
"""


MIGRATIONS = [
    "ALTER TABLE memory_items ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE memory_items ADD COLUMN last_reinforced TEXT",
    "ALTER TABLE memory_items ADD COLUMN raw_text TEXT",
    "ALTER TABLE memory_items ADD COLUMN source_event_id TEXT REFERENCES events(id)",
]


class Database:
    def __init__(self, path: Path = DEFAULT_DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self):
        """Apply schema migrations idempotently."""
        for sql in MIGRATIONS:
            try:
                self.conn.execute(sql)
                self.conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

    def close(self):
        self.conn.close()

    # --- Memory Items ---

    @_locked
    def save_item(self, item: MemoryItem) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO memory_items
               (id, summary, memory_type, importance, tier, status,
                access_count, last_accessed, daily_ref,
                consolidated_into, reinforcement_count, last_reinforced,
                raw_text, source_event_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.id,
                item.summary,
                item.memory_type,
                item.importance,
                item.tier,
                item.status,
                item.access_count,
                item.last_accessed.isoformat() if item.last_accessed else None,
                item.daily_ref,
                item.consolidated_into,
                item.reinforcement_count,
                item.last_reinforced.isoformat() if item.last_reinforced else None,
                item.raw_text,
                item.source_event_id,
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
            ),
        )
        self.conn.commit()

    @_locked
    def get_item(self, item_id: str) -> MemoryItem | None:
        row = self.conn.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    @_locked
    def get_active_items(self) -> list[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memory_items WHERE status = 'active' ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    @_locked
    def get_items_by_type(self, memory_type: str) -> list[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memory_items WHERE memory_type = ? AND status = 'active' ORDER BY created_at DESC",
            (memory_type,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    @_locked
    def get_items_by_tier(self, tier: int) -> list[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memory_items WHERE tier = ? AND status = 'active' ORDER BY created_at DESC",
            (tier,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    @_locked
    def search_by_keyword(self, query: str, top_k: int = 10) -> list[MemoryItem]:
        """Keyword search using SQLite LIKE. Splits query into words, scores by match count."""
        words = query.lower().split()
        if not words:
            return self.get_active_items()[:top_k]

        conditions = " OR ".join(["LOWER(summary) LIKE ?" for _ in words])
        score_expr = " + ".join(["(CASE WHEN LOWER(summary) LIKE ? THEN 1 ELSE 0 END)" for _ in words])
        params = [f"%{w}%" for w in words]

        rows = self.conn.execute(
            f"""SELECT *, ({score_expr}) as match_count
            FROM memory_items
            WHERE status = 'active' AND ({conditions})
            ORDER BY match_count DESC, created_at DESC
            LIMIT ?""",
            params + params + [top_k],
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    @_locked
    def archive_item(self, item_id: str, reason: str | None = None) -> None:
        self.conn.execute(
            "UPDATE memory_items SET status = 'archived', updated_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), item_id),
        )
        self.conn.commit()

    @_locked
    def update_item(self, item_id: str, summary: str) -> MemoryItem | None:
        """Update a memory's summary in place, preserving created_at and daily_ref."""
        item = self.get_item(item_id)
        if not item:
            return None
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE memory_items SET summary = ?, updated_at = ? WHERE id = ?",
            (summary, now, item_id),
        )
        self.conn.commit()
        return self.get_item(item_id)

    @_locked
    def snapshot_item(self, item: MemoryItem) -> str:
        """Create an archived copy of a memory, returning the snapshot's ID."""
        snapshot = MemoryItem(
            summary=item.summary,
            memory_type=item.memory_type,
            importance=item.importance,
            tier=item.tier,
            status="archived",
            access_count=item.access_count,
            last_accessed=item.last_accessed,
            daily_ref=item.daily_ref,
            consolidated_into=item.consolidated_into,
            created_at=item.created_at,
        )
        self.save_item(snapshot)
        return snapshot.id

    @_locked
    def bump_access(self, item_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE memory_items SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, item_id),
        )
        self.conn.commit()

    @_locked
    def get_counts(self) -> dict:
        row = self.conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN tier = 2 AND status = 'active' THEN 1 ELSE 0 END) as tier2,
                SUM(CASE WHEN tier = 3 AND status = 'active' THEN 1 ELSE 0 END) as tier3,
                SUM(CASE WHEN status = 'archived' THEN 1 ELSE 0 END) as archived
            FROM memory_items"""
        ).fetchone()
        return {"total": row["total"], "tier2": row["tier2"], "tier3": row["tier3"], "archived": row["archived"]}

    # --- Processed Sessions ---

    @_locked
    def is_session_processed(self, session_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM processed_sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row is not None

    @_locked
    def mark_session_processed(self, session_id: str, file_path: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO processed_sessions (session_id, file_path, processed_at) VALUES (?, ?, ?)",
            (session_id, file_path, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    @_locked
    def get_processed_session_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM processed_sessions").fetchone()
        return row["cnt"]

    # --- Timeline ---

    @_locked
    def get_items_by_date_range(self, start_date: str, end_date: str | None = None) -> list[MemoryItem]:
        if end_date:
            rows = self.conn.execute(
                """SELECT * FROM memory_items
                WHERE status = 'active' AND daily_ref >= ? AND daily_ref <= ?
                ORDER BY daily_ref ASC, created_at ASC""",
                (start_date, end_date),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM memory_items
                WHERE status = 'active' AND daily_ref = ?
                ORDER BY created_at ASC""",
                (start_date,),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    @_locked
    def get_items_since(self, since_iso: str, limit: int = 100) -> list[MemoryItem]:
        """Get active memories created after a given ISO timestamp."""
        rows = self.conn.execute(
            """SELECT * FROM memory_items
            WHERE status = 'active' AND created_at > ?
            ORDER BY created_at ASC LIMIT ?""",
            (since_iso, limit),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    # --- Hot Set ---

    @_locked
    def get_hot_items(
        self,
        profile_behavior_floor: int = 7,
        identity_floor: int = 9,
        reinforcement_floor: int = 3,
        access_floor: int = 20,
        max_size: int = 100,
    ) -> list[MemoryItem]:
        """Return memories that qualify for the hot set (always-relevant cache)."""
        rows = self.conn.execute(
            """SELECT * FROM memory_items
            WHERE status = 'active'
            AND (
                (memory_type IN ('profile', 'behavior') AND importance >= ?)
                OR importance >= ?
                OR (reinforcement_count >= ? AND importance >= 6)
                OR (access_count >= ? AND importance >= 6)
            )
            ORDER BY importance DESC, access_count DESC
            LIMIT ?""",
            (profile_behavior_floor, identity_floor, reinforcement_floor, access_floor, max_size),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    # --- Internal ---

    @_locked
    def reinforce_item(self, item_id: str) -> None:
        """Increment reinforcement_count and update last_reinforced timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE memory_items SET reinforcement_count = reinforcement_count + 1, last_reinforced = ? WHERE id = ?",
            (now, item_id),
        )
        self.conn.commit()

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        last_accessed = None
        if row["last_accessed"]:
            last_accessed = datetime.fromisoformat(row["last_accessed"])
        last_reinforced = None
        if row["last_reinforced"]:
            last_reinforced = datetime.fromisoformat(row["last_reinforced"])
        return MemoryItem(
            id=row["id"],
            summary=row["summary"],
            memory_type=row["memory_type"],
            importance=row["importance"],
            tier=row["tier"],
            status=row["status"],
            access_count=row["access_count"],
            last_accessed=last_accessed,
            daily_ref=row["daily_ref"],
            consolidated_into=row["consolidated_into"],
            reinforcement_count=row["reinforcement_count"],
            last_reinforced=last_reinforced,
            raw_text=row["raw_text"] if "raw_text" in row.keys() else None,
            source_event_id=row["source_event_id"] if "source_event_id" in row.keys() else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # --- Events (raw ingested turns) ---

    @_locked
    def save_event(self, event: Event) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO events
               (id, text, received_at, extraction_status, extraction_error, memory_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event.id,
                event.text,
                event.received_at.isoformat(),
                event.extraction_status,
                event.extraction_error,
                event.memory_count,
            ),
        )
        self.conn.commit()

    @_locked
    def get_event(self, event_id: str) -> Event | None:
        row = self.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if not row:
            return None
        return Event(
            id=row["id"],
            text=row["text"],
            received_at=datetime.fromisoformat(row["received_at"]),
            extraction_status=row["extraction_status"],
            extraction_error=row["extraction_error"],
            memory_count=row["memory_count"],
        )

    @_locked
    def mark_event_extracted(self, event_id: str, memory_count: int) -> None:
        self.conn.execute(
            "UPDATE events SET extraction_status = 'extracted', memory_count = ?, extraction_error = NULL WHERE id = ?",
            (memory_count, event_id),
        )
        self.conn.commit()

    @_locked
    def mark_event_failed(self, event_id: str, error: str) -> None:
        self.conn.execute(
            "UPDATE events SET extraction_status = 'failed', extraction_error = ? WHERE id = ?",
            (error, event_id),
        )
        self.conn.commit()

    @_locked
    def get_pending_events(self, limit: int = 100) -> list[Event]:
        rows = self.conn.execute(
            """SELECT * FROM events
               WHERE extraction_status IN ('pending', 'failed')
               ORDER BY received_at ASC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            Event(
                id=row["id"],
                text=row["text"],
                received_at=datetime.fromisoformat(row["received_at"]),
                extraction_status=row["extraction_status"],
                extraction_error=row["extraction_error"],
                memory_count=row["memory_count"],
            )
            for row in rows
        ]

    @_locked
    def get_event_counts(self) -> dict:
        rows = self.conn.execute(
            """SELECT extraction_status, COUNT(*) as n FROM events GROUP BY extraction_status"""
        ).fetchall()
        return {row["extraction_status"]: row["n"] for row in rows}
