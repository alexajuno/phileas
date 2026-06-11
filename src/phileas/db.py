"""SQLite storage backend for Phileas.

Canonical data store. ChromaDB and KuzuDB are derived indexes
that can be rebuilt from this database.
"""

import functools
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from phileas.models import Event, MemoryItem


def _locked(method):
    """Serialize Database access across threads via self._lock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


_PREVIEW_CHARS = 240


def _preview(text: str) -> str:
    """Mirror web/src/lib/phileas-db.ts:preview — first 240 chars, ellipsis if longer."""
    return text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "…"


DEFAULT_DB_PATH = Path.home() / ".phileas" / "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 5,
    status TEXT NOT NULL DEFAULT 'active',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    daily_ref TEXT,
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
    received_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_status ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_items_type ON memory_items(memory_type);
CREATE INDEX IF NOT EXISTS idx_items_daily_ref ON memory_items(daily_ref);
CREATE INDEX IF NOT EXISTS idx_events_received ON events(received_at);
"""


MIGRATIONS = [
    "ALTER TABLE memory_items ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE memory_items ADD COLUMN last_reinforced TEXT",
    "ALTER TABLE memory_items ADD COLUMN raw_text TEXT",
    "ALTER TABLE memory_items ADD COLUMN source_event_id TEXT REFERENCES events(id)",
    "DROP INDEX IF EXISTS idx_items_tier",
    "ALTER TABLE memory_items DROP COLUMN tier",
    "DROP INDEX IF EXISTS idx_events_status",
    "ALTER TABLE events DROP COLUMN extraction_status",
    "ALTER TABLE events DROP COLUMN extraction_error",
    "ALTER TABLE events DROP COLUMN memory_count",
    "ALTER TABLE memory_items DROP COLUMN consolidated_into",
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
               (id, summary, memory_type, importance, status,
                access_count, last_accessed, daily_ref,
                reinforcement_count, last_reinforced,
                raw_text, source_event_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.id,
                item.summary,
                item.memory_type,
                item.importance,
                item.status,
                item.access_count,
                item.last_accessed.isoformat() if item.last_accessed else None,
                item.daily_ref,
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
    def get_items_by_id_prefix(self, prefix: str, limit: int = 10) -> list[MemoryItem]:
        """Resolve a memory by an id prefix (e.g. the 8-char pointer id).

        Returns every active-or-archived match (capped at ``limit``) so the
        caller can distinguish the no-match, unique-match, and ambiguous-prefix
        cases. A full uuid resolves to at most one row. Used by `hydrate`
        (AA-106) to turn a cheap pointer id8 back into a full record.
        """
        clean = (prefix or "").strip()
        if not clean:
            return []
        rows = self.conn.execute(
            "SELECT * FROM memory_items WHERE id LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"{clean}%", limit),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

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
    def search_by_keyword(self, query: str, top_k: int | None = None) -> list[MemoryItem]:
        """Keyword search using SQLite LIKE — AND-match across whitespace tokens.

        The agent is expected to pass a focused term query (one concept, 1–4
        words). All tokens must appear in the summary; an empty/whitespace-only
        query returns nothing. No stopword stripping — that's the agent's
        concern. Use multiple recall() calls for compound questions.
        """
        words = query.lower().split()
        if not words:
            return []

        conditions = " AND ".join(["LOWER(summary) LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words]

        sql = f"SELECT * FROM memory_items WHERE status = 'active' AND ({conditions}) ORDER BY created_at DESC"
        sql_params: list = list(params)
        if top_k is not None:
            sql += " LIMIT ?"
            sql_params.append(top_k)

        rows = self.conn.execute(sql, sql_params).fetchall()
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
            status="archived",
            access_count=item.access_count,
            last_accessed=item.last_accessed,
            daily_ref=item.daily_ref,
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
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN status = 'archived' THEN 1 ELSE 0 END) as archived
            FROM memory_items"""
        ).fetchone()
        return {"total": row["total"], "active": row["active"] or 0, "archived": row["archived"] or 0}

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

    # --- Web dashboard reads ---
    #
    # Rows shaped for web/src/lib/types.ts:MemoryItem — the daemon's read
    # contract for the dashboard. We serialize straight from the raw sqlite row
    # (not via MemoryItem) so created_at/updated_at stay the exact stored ISO
    # strings the day-window comparison relies on, and the exposed column set is
    # owned here rather than inherited from the model. Only columns the base
    # schema actually creates are listed — the legacy `tags` / `source_session_id`
    # columns exist on older DBs but not on a freshly built one, so reading them
    # here would raise `no such column` on a rebuilt store.

    _WEB_COLS = (
        "id, summary, memory_type, importance, status, "
        "access_count, reinforcement_count, last_reinforced, "
        "raw_text, daily_ref, created_at, updated_at"
    )

    @staticmethod
    def _row_to_web_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "summary": row["summary"],
            "memory_type": row["memory_type"],
            "importance": row["importance"],
            "status": row["status"],
            "access_count": row["access_count"],
            "reinforcement_count": row["reinforcement_count"],
            "last_reinforced": row["last_reinforced"],
            "raw_text": row["raw_text"],
            "daily_ref": row["daily_ref"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @_locked
    def web_memories_for_day(self, start_iso: str, end_iso: str) -> list[dict]:
        """Active memories created within [start_iso, end_iso).

        The bounds are the UTC-ISO day window the client computes from its local
        day (day.ts). We do not recompute them here so the boundary string
        comparison stays byte-identical to the current direct-DB read.
        """
        rows = self.conn.execute(
            f"""SELECT {self._WEB_COLS} FROM memory_items
                WHERE status = 'active'
                  AND created_at >= ? AND created_at < ?
                ORDER BY created_at DESC""",
            (start_iso, end_iso),
        ).fetchall()
        return [self._row_to_web_dict(r) for r in rows]

    @_locked
    def web_search(self, query: str, limit: int = 100) -> list[dict]:
        """Keyword search over summary/raw_text — up to 8 whitespace terms,
        LIKE-AND, backslash-escaped."""
        terms = (query or "").split()[:8]
        if not terms:
            return []
        clauses: list[str] = []
        params: list[str | int] = []
        for term in terms:
            clauses.append("(summary LIKE ? ESCAPE '\\' OR raw_text LIKE ? ESCAPE '\\')")
            like = "%" + re.sub(r"([\\%_])", r"\\\1", term) + "%"
            params.extend([like, like])
        params.append(limit)
        rows = self.conn.execute(
            f"""SELECT {self._WEB_COLS} FROM memory_items
                WHERE status = 'active' AND {" AND ".join(clauses)}
                ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()
        return [self._row_to_web_dict(r) for r in rows]

    @_locked
    def web_export(
        self,
        start_iso: str | None = None,
        end_iso: str | None = None,
        memory_type: str | None = None,
        min_importance: int | None = None,
    ) -> list[dict]:
        """Filtered export — mirrors queries.ts:fetchMemoriesForExport. Bounds are
        client-computed UTC ISO; min_importance only filters when > 1."""
        clauses = ["status = 'active'"]
        params: list[str | int] = []
        if start_iso:
            clauses.append("created_at >= ?")
            params.append(start_iso)
        if end_iso:
            clauses.append("created_at < ?")
            params.append(end_iso)
        if memory_type:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        if min_importance and min_importance > 1:
            clauses.append("importance >= ?")
            params.append(min_importance)
        rows = self.conn.execute(
            f"""SELECT {self._WEB_COLS} FROM memory_items
                WHERE {" AND ".join(clauses)}
                ORDER BY created_at DESC""",
            params,
        ).fetchall()
        return [self._row_to_web_dict(r) for r in rows]

    @_locked
    def web_memories_by_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""SELECT {self._WEB_COLS} FROM memory_items
                WHERE status = 'active' AND id IN ({placeholders})
                ORDER BY created_at DESC""",
            ids,
        ).fetchall()
        return [self._row_to_web_dict(r) for r in rows]

    @_locked
    def web_days_with_counts(self, limit: int = 60, tz_offset_minutes: int | None = None) -> list[dict]:
        """Active-memory counts bucketed by LOCAL day — mirrors queries.ts:fetchDaysWithCounts.

        created_at is stored UTC; the bucket key is the viewer's *local* day.
        ``tz_offset_minutes`` pins the bucketing timezone for a remote client;
        None falls back to the daemon's own local timezone (correct when the
        daemon and the viewer share a machine — Phase 1's single-host case).
        """
        rows = self.conn.execute("SELECT created_at FROM memory_items WHERE status = 'active'").fetchall()
        tz = timezone(timedelta(minutes=tz_offset_minutes)) if tz_offset_minutes is not None else None
        buckets: dict[str, int] = {}
        for r in rows:
            local = datetime.fromisoformat(r["created_at"]).astimezone(tz)
            key = local.strftime("%Y-%m-%d")
            buckets[key] = buckets.get(key, 0) + 1
        ordered = sorted(buckets.items(), key=lambda kv: kv[0], reverse=True)
        return [{"day": day, "count": count} for day, count in ordered[:limit]]

    @_locked
    def web_memories_brief(self, ids: list[str]) -> list[dict]:
        """Minimal columns for a set of ids, ANY status — mirrors the resolveMemories
        helper in monitoring/traces/[id]. Ordering and placeholders for missing ids
        are the caller's concern (the trace view reconstructs input order)."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""SELECT id, summary, memory_type, importance, created_at
                FROM memory_items WHERE id IN ({placeholders})""",
            ids,
        ).fetchall()
        return [
            {
                "id": r["id"],
                "summary": r["summary"],
                "memory_type": r["memory_type"],
                "importance": r["importance"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    @_locked
    def web_ingestion_health(self) -> dict:
        """Event-ingestion counts (1h / 24h / total) — mirrors phileas-db.ts:fetchIngestionHealth."""
        now = datetime.now(timezone.utc)
        h1 = (now - timedelta(hours=1)).isoformat()
        d1 = (now - timedelta(days=1)).isoformat()
        return {
            "events_received_1h": self.conn.execute(
                "SELECT COUNT(*) FROM events WHERE received_at >= ?", (h1,)
            ).fetchone()[0],
            "events_received_24h": self.conn.execute(
                "SELECT COUNT(*) FROM events WHERE received_at >= ?", (d1,)
            ).fetchone()[0],
            "events_total": self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        }

    @_locked
    def web_ingestion_events(self, limit: int = 50) -> list[dict]:
        """Recent ingested events with truncated text — mirrors phileas-db.ts:listIngestionEvents."""
        limit = min(max(50 if limit is None else limit, 1), 500)
        rows = self.conn.execute(
            "SELECT id, text, received_at FROM events ORDER BY received_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": r["id"], "received_at": r["received_at"], "text_preview": _preview(r["text"])} for r in rows]

    @_locked
    def web_ingestion_event(self, event_id: str) -> dict | None:
        """One event plus its linked active memories — mirrors phileas-db.ts:fetchIngestionEvent."""
        ev = self.conn.execute("SELECT id, text, received_at FROM events WHERE id = ?", (event_id,)).fetchone()
        if not ev:
            return None
        mems = self.conn.execute(
            """SELECT id, summary, memory_type, importance, created_at
               FROM memory_items
               WHERE source_event_id = ? AND status = 'active'
               ORDER BY created_at ASC""",
            (event_id,),
        ).fetchall()
        return {
            "event": {"id": ev["id"], "text": ev["text"], "received_at": ev["received_at"]},
            "memories": [
                {
                    "id": m["id"],
                    "summary": m["summary"],
                    "memory_type": m["memory_type"],
                    "importance": m["importance"],
                    "created_at": m["created_at"],
                }
                for m in mems
            ],
        }

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
            status=row["status"],
            access_count=row["access_count"],
            last_accessed=last_accessed,
            daily_ref=row["daily_ref"],
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
            """INSERT OR REPLACE INTO events (id, text, received_at)
               VALUES (?, ?, ?)""",
            (event.id, event.text, event.received_at.isoformat()),
        )
        self.conn.commit()

    @_locked
    def get_memories_for_event(self, event_id: str) -> list[MemoryItem]:
        """Active memories whose source_event_id == event_id, oldest first."""
        rows = self.conn.execute(
            """SELECT * FROM memory_items
               WHERE source_event_id = ? AND status = 'active'
               ORDER BY created_at ASC""",
            (event_id,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    @_locked
    def get_all_events(self, limit: int | None = None) -> list[Event]:
        """All events in insertion order — used by the embed-backfill script."""
        sql = "SELECT id, text, received_at FROM events ORDER BY received_at ASC"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self.conn.execute(sql, params).fetchall()
        return [
            Event(
                id=row["id"],
                text=row["text"],
                received_at=datetime.fromisoformat(row["received_at"]),
            )
            for row in rows
        ]

    @_locked
    def get_event(self, event_id: str) -> Event | None:
        row = self.conn.execute(
            "SELECT id, text, received_at FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if not row:
            return None
        return Event(
            id=row["id"],
            text=row["text"],
            received_at=datetime.fromisoformat(row["received_at"]),
        )
