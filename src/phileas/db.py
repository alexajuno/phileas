"""SQLite storage backend for Phileas.

Canonical data store. ChromaDB and KuzuDB are derived indexes
that can be rebuilt from this database.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from phileas.models import MemoryItem

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
    source_session_id TEXT,
    consolidated_into TEXT REFERENCES memory_items(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_sessions (
    session_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_status ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_items_tier ON memory_items(tier);
CREATE INDEX IF NOT EXISTS idx_items_type ON memory_items(memory_type);
CREATE INDEX IF NOT EXISTS idx_items_daily_ref ON memory_items(daily_ref);
"""


class Database:
    def __init__(self, path: Path = DEFAULT_DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    # --- Memory Items ---

    def save_item(self, item: MemoryItem) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO memory_items
               (id, summary, memory_type, importance, tier, status,
                access_count, last_accessed, daily_ref, source_session_id,
                consolidated_into, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                item.source_session_id,
                item.consolidated_into,
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
            ),
        )
        self.conn.commit()

    def get_item(self, item_id: str) -> MemoryItem | None:
        row = self.conn.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    def get_active_items(self) -> list[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memory_items WHERE status = 'active' ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_items_by_type(self, memory_type: str) -> list[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memory_items WHERE memory_type = ? AND status = 'active' ORDER BY created_at DESC",
            (memory_type,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_items_by_tier(self, tier: int) -> list[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memory_items WHERE tier = ? AND status = 'active' ORDER BY created_at DESC",
            (tier,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

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

    def archive_item(self, item_id: str, reason: str | None = None) -> None:
        self.conn.execute(
            "UPDATE memory_items SET status = 'archived', updated_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), item_id),
        )
        self.conn.commit()

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
            source_session_id=item.source_session_id,
            consolidated_into=item.consolidated_into,
            created_at=item.created_at,
        )
        self.save_item(snapshot)
        return snapshot.id

    def bump_access(self, item_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE memory_items SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, item_id),
        )
        self.conn.commit()

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

    def is_session_processed(self, session_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM processed_sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row is not None

    def mark_session_processed(self, session_id: str, file_path: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO processed_sessions (session_id, file_path, processed_at) VALUES (?, ?, ?)",
            (session_id, file_path, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def get_processed_session_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM processed_sessions").fetchone()
        return row["cnt"]

    # --- Timeline ---

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

    # --- Internal ---

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        last_accessed = None
        if row["last_accessed"]:
            last_accessed = datetime.fromisoformat(row["last_accessed"])
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
            source_session_id=row["source_session_id"],
            consolidated_into=row["consolidated_into"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
