"""SQLite storage backend for Phileas.

Canonical data store. ChromaDB and KuzuDB are derived indexes
that can be rebuilt from this database.
"""

import functools
import json
import logging
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from phileas.config import resolve_home
from phileas.models import Event, MemoryItem, Thread
from phileas.scoring import RECALL_GAIN, RESTUDY_GAIN, delta_storage, retrieval_strength

log = logging.getLogger(__name__)


def _days_since_iso(ts: str | None) -> float:
    """Days since an ISO-8601 timestamp string (0.0 if missing)."""
    if not ts:
        return 0.0
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)


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


def _fts_match_query(query: str) -> str | None:
    """Build a safe FTS5 MATCH expression from free-form query text.

    Each whitespace token becomes a quoted prefix term (``"tok"*``) and the
    terms are OR-ed together, so a memory is a candidate if its content contains *any*
    query token (full multi-token overlap then ranks higher under BM25). Quoting
    neutralises FTS5 query operators in user input; the trailing ``*`` outside
    the quote is the prefix operator, so ``swed`` reaches "sweden". unicode61
    case-folds, so no lowering is needed.

    Returns ``None`` when the query has no usable token (empty, whitespace, or
    pure punctuation) — the caller treats that as "no results".
    """
    terms = []
    for tok in query.split():
        if not any(c.isalnum() for c in tok):
            continue  # a pure-punctuation token tokenizes to nothing
        terms.append('"' + tok.replace('"', '""') + '"*')
    return " OR ".join(terms) if terms else None


DEFAULT_DB_PATH = resolve_home() / "memory.db"

# Sentinel thread id. A turn whose conversation was never recorded points here,
# so thread_id is never null and a thread -> event drill-down resolves to a row.
# "unknown" reads as exactly that: origin not recorded.
UNKNOWN_EVENT_ID = "unknown"
UNKNOWN_THREAD_ID = "unknown"


def clean_source_event_id(value: str | None) -> str | None:
    """Normalize a memory's source-event reference for storage.

    A memory either traces to one captured turn (a real event id) or has no single
    source, which is NULL: a reflection or rollup derived from other memories, or a
    legacy row from before turns were tracked. Empty strings and the legacy
    ``UNKNOWN_EVENT_ID`` sentinel collapse to NULL too, so the placeholder never
    lands on a memory's provenance again.
    """
    sid = (value or "").strip()
    if not sid or sid == UNKNOWN_EVENT_ID:
        return None
    return sid


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    daily_ref TEXT,
    source_event_id TEXT REFERENCES events(id),
    storage_strength REAL NOT NULL DEFAULT 0.5,
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    last_reinforced TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- A raw turn. ``source_kind`` records the surface that captured it, so health
-- can track per-source recency. ``thread_id`` is the conversation it belongs
-- to: a thread is the ordered run of turns sharing this id. ``attribution`` is
-- whose words the segment is (self/other/source); ``extraction_status`` is the
-- distillation queue state, 'extracted' unless ingest marks a turn 'pending'.
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    received_at TEXT NOT NULL,
    source_kind TEXT,
    thread_id TEXT NOT NULL,
    attribution TEXT,
    extraction_status TEXT NOT NULL DEFAULT 'extracted'
);

CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source_kind TEXT,
    label TEXT,
    client_key TEXT
);

-- A client's stable conversation key maps to at most one thread, so a resumed
-- or compacted session re-attaches to the thread it already opened. Partial so
-- threads opened without a key (no continuity) don't collide on NULL.
CREATE UNIQUE INDEX IF NOT EXISTS idx_threads_client_key
    ON threads(client_key) WHERE client_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_items_status ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_items_type ON memory_items(memory_type);
CREATE INDEX IF NOT EXISTS idx_items_daily_ref ON memory_items(daily_ref);
CREATE INDEX IF NOT EXISTS idx_events_received ON events(received_at);
CREATE INDEX IF NOT EXISTS idx_events_thread ON events(thread_id, received_at);

-- Reconciliation judgments that stuck: entity pairs a judge ruled distinct.
-- ``reconcile`` filters these out so every run surfaces only unjudged pairs,
-- instead of re-litigating the whole roster forever. Keyed by the sorted id
-- pair; rows referencing a since-merged (deleted) entity are simply inert.
CREATE TABLE IF NOT EXISTS reconcile_dismissals (
    pair_key TEXT PRIMARY KEY,
    a_id TEXT NOT NULL,
    b_id TEXT NOT NULL,
    judged_at TEXT NOT NULL
);

-- Loose clusters awaiting roll-up, detected during recall and drained by the
-- ``consolidate`` command. Holds refs (member ids), not bodies, so it always
-- reflects current memories at drain time. Keyed by an ``anchor`` theme so the
-- same cluster re-surfaced by another query refreshes one row instead of
-- stacking. ``presented_at`` gives an untouched cluster a resurface cooldown.
CREATE TABLE IF NOT EXISTS consolidation_queue (
    id TEXT PRIMARY KEY,
    anchor TEXT NOT NULL,
    member_ids TEXT NOT NULL,
    loose_count INTEGER NOT NULL,
    span_start TEXT,
    span_end TEXT,
    detected_at TEXT NOT NULL,
    presented_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_consqueue_status ON consolidation_queue(status);

-- Inverted index over memory content, powering the keyword (sparse) leg of
-- recall via FTS5 + BM25. Standalone (not external-content): it stores mem_id
-- plus a copy of the content, so it stays decoupled from memory_items' integer
-- rowid and is kept in sync from the write paths in this module. Mirrors the
-- active set only, so BM25 corpus statistics reflect live memories.
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    mem_id UNINDEXED,
    content,
    tokenize = 'unicode61'
);

-- Sentinel thread and event for turns whose conversation was never recorded: an
-- event with an unrecorded thread points at the 'unknown' thread, so a NOT NULL
-- thread_id always resolves to a row and a thread -> event drill-down holds.
INSERT OR IGNORE INTO threads (id, created_at, source_kind, label)
    VALUES ('unknown', '1970-01-01T00:00:00+00:00', 'unknown', 'unknown provenance');
INSERT OR IGNORE INTO events (id, text, received_at, source_kind, thread_id)
    VALUES ('unknown', '', '1970-01-01T00:00:00+00:00', 'unknown', 'unknown');
"""


class Database:
    def __init__(self, path: Path = DEFAULT_DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.conn.executescript(SCHEMA)
        self._migrate()
        self._reconcile_fts()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created.

        The schema is CREATE TABLE IF NOT EXISTS, which never alters an existing
        table, and ``extraction_status`` has to be filterable (the worker selects
        ``WHERE extraction_status='pending'``), so a pre-existing events table must
        actually gain the column. Each ALTER is column-guarded, so this is
        idempotent; existing rows backfill to 'extracted', leaving turns captured
        before the observer pipeline out of the worker's queue.
        """
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(events)")}
        if "attribution" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN attribution TEXT")
        if "extraction_status" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN extraction_status TEXT NOT NULL DEFAULT 'extracted'")
        # The AI's attribution was renamed 'other' -> 'assistant'; carry old rows
        # forward so the value set stays consistent. Idempotent: a no-op once done.
        self.conn.execute("UPDATE events SET attribution = 'assistant' WHERE attribution = 'other'")

        # ``memory_items.summary`` was renamed ``content`` — a memory's text is its
        # content, not a summary of a longer body. Guarded on the old column so it
        # runs once. FTS5 has no RENAME COLUMN, so the index is dropped and recreated
        # with the new column; ``_reconcile_fts`` (next in __init__) repopulates it.
        item_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(memory_items)")}
        if "summary" in item_cols and "content" not in item_cols:
            self.conn.execute("ALTER TABLE memory_items RENAME COLUMN summary TO content")
        fts_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(memory_fts)")}
        if "content" not in fts_cols:
            self.conn.execute("DROP TABLE IF EXISTS memory_fts")
            self.conn.execute(
                "CREATE VIRTUAL TABLE memory_fts USING fts5(mem_id UNINDEXED, content, tokenize = 'unicode61')"
            )
        self.conn.commit()

    def _reconcile_fts(self) -> None:
        """Seed the FTS index for any active memory it doesn't yet hold.

        Self-healing and idempotent: it inserts only the active memories missing
        from ``memory_fts``, so it costs nothing once the index is current and it
        rebuilds an index that drifted out from under the database. Cheap for a
        personal-size corpus, so it runs on every open.
        """
        self.conn.execute(
            "INSERT INTO memory_fts(mem_id, content) "
            "SELECT id, content FROM memory_items "
            "WHERE status = 'active' AND id NOT IN (SELECT mem_id FROM memory_fts)"
        )
        self.conn.commit()

    def _fts_upsert(self, mem_id: str, content: str) -> None:
        """Refresh a memory's row in the FTS index (delete-then-insert)."""
        self.conn.execute("DELETE FROM memory_fts WHERE mem_id = ?", (mem_id,))
        self.conn.execute("INSERT INTO memory_fts(mem_id, content) VALUES (?, ?)", (mem_id, content))

    def _fts_delete(self, mem_id: str) -> None:
        """Drop a memory from the FTS index (archive / soft delete)."""
        self.conn.execute("DELETE FROM memory_fts WHERE mem_id = ?", (mem_id,))

    def close(self):
        self.conn.close()

    # --- Memory Items ---

    @_locked
    def save_item(self, item: MemoryItem) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO memory_items
               (id, content, memory_type, status,
                access_count, last_accessed, daily_ref,
                storage_strength, reinforcement_count, last_reinforced,
                source_event_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.id,
                item.content,
                item.memory_type,
                item.status,
                item.access_count,
                item.last_accessed.isoformat() if item.last_accessed else None,
                item.daily_ref,
                item.storage_strength,
                item.reinforcement_count,
                item.last_reinforced.isoformat() if item.last_reinforced else None,
                clean_source_event_id(item.source_event_id),
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
            ),
        )
        self._fts_upsert(item.id, item.content)
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
    def get_items_by_status(self, status: str | None = "active") -> list[MemoryItem]:
        """Memories with the given status, newest first. ``status=None`` returns every item."""
        if status is None:
            rows = self.conn.execute("SELECT * FROM memory_items ORDER BY created_at DESC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memory_items WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    @_locked
    def search_by_keyword_scored(self, query: str, top_k: int | None = None) -> list[tuple[MemoryItem, float]]:
        """Keyword search over the FTS5 index, ranked by BM25.

        Each whitespace token becomes a prefix term and the terms are OR-ed
        together (see ``_fts_match_query``): a memory is a candidate if its
        content contains *any* query token, and BM25 ranks the candidates — a
        memory covering more of the query, or matching rarer terms, scores higher, so a
        focused query whose tokens co-occur lands on top while a clumsy query
        whose tokens are spread across memories still surfaces each contributor
        instead of collapsing to nothing. BM25 supplies term weighting (rarity,
        term-frequency saturation, document-length normalization) directly.

        Returns ``(item, bm25)`` pairs in best-first order. SQLite's ``bm25()``
        is negative, more-negative meaning a better match, so the raw value sorts
        ascending; callers that want a magnitude negate it. An empty,
        whitespace-only, or pure-punctuation query returns nothing. No stopword
        stripping — that's the agent's concern; the downstream rerank and
        distributional cut decide what's actually worth keeping.
        """
        match = _fts_match_query(query)
        if match is None:
            return []

        sql = (
            "SELECT m.*, bm25(memory_fts) AS rank "
            "FROM memory_fts JOIN memory_items m ON m.id = memory_fts.mem_id "
            "WHERE memory_fts MATCH ? AND m.status = 'active' "
            "ORDER BY rank"
        )
        params: list = [match]
        if top_k is not None:
            sql += " LIMIT ?"
            params.append(top_k)

        rows = self.conn.execute(sql, params).fetchall()
        return [(self._row_to_item(row), row["rank"]) for row in rows]

    def search_by_keyword(self, query: str, top_k: int | None = None) -> list[MemoryItem]:
        """BM25-ranked keyword hits without their scores (see search_by_keyword_scored)."""
        return [item for item, _ in self.search_by_keyword_scored(query, top_k=top_k)]

    @_locked
    def archive_item(self, item_id: str, reason: str | None = None) -> None:
        self.conn.execute(
            "UPDATE memory_items SET status = 'archived', updated_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), item_id),
        )
        self._fts_delete(item_id)
        self.conn.commit()

    @_locked
    def update_item(self, item_id: str, content: str) -> MemoryItem | None:
        """Update a memory's content in place, preserving created_at and daily_ref."""
        item = self.get_item(item_id)
        if not item:
            return None
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE memory_items SET content = ?, updated_at = ? WHERE id = ?",
            (content, now, item_id),
        )
        self._fts_upsert(item_id, content)
        self.conn.commit()
        return self.get_item(item_id)

    @_locked
    def snapshot_item(self, item: MemoryItem) -> str:
        """Create an archived copy of a memory, returning the snapshot's ID."""
        snapshot = MemoryItem(
            content=item.content,
            memory_type=item.memory_type,
            status="archived",
            access_count=item.access_count,
            last_accessed=item.last_accessed,
            daily_ref=item.daily_ref,
            storage_strength=item.storage_strength,
            created_at=item.created_at,
        )
        self.save_item(snapshot)
        return snapshot.id

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
        "id, content, memory_type, status, "
        "access_count, storage_strength, reinforcement_count, last_reinforced, "
        "daily_ref, created_at, updated_at"
    )

    @staticmethod
    def _row_to_web_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "content": row["content"],
            "memory_type": row["memory_type"],
            "status": row["status"],
            "access_count": row["access_count"],
            "storage_strength": row["storage_strength"],
            "reinforcement_count": row["reinforcement_count"],
            "last_reinforced": row["last_reinforced"],
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
        """Keyword search over content — up to 8 whitespace terms, LIKE-AND,
        backslash-escaped."""
        terms = (query or "").split()[:8]
        if not terms:
            return []
        clauses: list[str] = []
        params: list[str | int] = []
        for term in terms:
            clauses.append("content LIKE ? ESCAPE '\\'")
            like = "%" + re.sub(r"([\\%_])", r"\\\1", term) + "%"
            params.append(like)
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
    ) -> list[dict]:
        """Filtered export — mirrors queries.ts:fetchMemoriesForExport. Bounds are
        client-computed UTC ISO."""
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
            f"""SELECT id, content, memory_type, created_at
                FROM memory_items WHERE id IN ({placeholders})""",
            ids,
        ).fetchall()
        return [
            {
                "id": r["id"],
                "content": r["content"],
                "memory_type": r["memory_type"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    @_locked
    def event_status_counts(self) -> dict[str, int]:
        """Event counts grouped by extraction_status — the worker's queue depth."""
        return {
            row[0]: row[1]
            for row in self.conn.execute("SELECT extraction_status, COUNT(*) FROM events GROUP BY extraction_status")
        }

    @_locked
    def web_ingestion_health(self) -> dict:
        """Event-ingestion counts (1h / 24h / total), plus the extraction queue depth."""
        now = datetime.now(timezone.utc)
        h1 = (now - timedelta(hours=1)).isoformat()
        d1 = (now - timedelta(days=1)).isoformat()
        status_counts = self.event_status_counts()
        return {
            "events_received_1h": self.conn.execute(
                "SELECT COUNT(*) FROM events WHERE received_at >= ?", (h1,)
            ).fetchone()[0],
            "events_received_24h": self.conn.execute(
                "SELECT COUNT(*) FROM events WHERE received_at >= ?", (d1,)
            ).fetchone()[0],
            "events_total": self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "events_pending": status_counts.get("pending", 0),
            "events_failed": status_counts.get("failed", 0),
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
            """SELECT id, content, memory_type, created_at
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
                    "content": m["content"],
                    "memory_type": m["memory_type"],
                    "created_at": m["created_at"],
                }
                for m in mems
            ],
        }

    # --- Internal ---

    @_locked
    def record_retrieval(self, item_id: str, retrieval_before: float, relevance: float) -> float:
        """Record a successful recall; return the storage strength gained.

        Grows storage strength — difficulty-weighted by how decayed the memory
        was (``retrieval_before``) and gated by ``relevance`` — counts the
        access, and refreshes accessibility by resetting last_accessed so
        retrieval strength recovers toward 1. The caller computes
        ``retrieval_before`` before this reset; the returned delta lets it report
        per-recall growth without recomputing.
        """
        now = datetime.now(timezone.utc).isoformat()
        delta = delta_storage(relevance, retrieval_before, gain=RECALL_GAIN)
        self.conn.execute(
            "UPDATE memory_items SET storage_strength = storage_strength + ?, "
            "access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (delta, now, item_id),
        )
        self.conn.commit()
        return delta

    @_locked
    def storage_health(self, fading_threshold: float = 0.25) -> dict:
        """Snapshot of the active store's strength distribution and guardrails.

        - storage_p50/p90/max — is durability accumulating, or flat at the seed?
        - fading_count — active memories whose retrieval strength has decayed
          below ``fading_threshold`` (candidates the model is letting go).
        - recalls_top5pct_share — fraction of all recalls held by the busiest 5%
          of memories; the ossification guardrail (rises toward 1 if a handful
          dominate).
        - reinforced_24h — strengthening events (recall + re-study) in the last day.
        """
        rows = self.conn.execute(
            "SELECT storage_strength, access_count, last_accessed, created_at, last_reinforced "
            "FROM memory_items WHERE status = 'active'"
        ).fetchall()
        if not rows:
            return {
                "active": 0,
                "storage_p50": 0.0,
                "storage_p90": 0.0,
                "storage_max": 0.0,
                "fading_count": 0,
                "fading_threshold": fading_threshold,
                "recalls_top5pct_share": 0.0,
                "reinforced_24h": 0,
            }

        strengths = sorted(r["storage_strength"] for r in rows)

        def _pct(p: float) -> float:
            return round(strengths[min(len(strengths) - 1, int(p * len(strengths)))], 3)

        fading = sum(
            1
            for r in rows
            if retrieval_strength(_days_since_iso(r["last_accessed"] or r["created_at"]), r["storage_strength"])
            < fading_threshold
        )
        accesses = sorted((r["access_count"] for r in rows), reverse=True)
        total_acc = sum(accesses)
        top_k = max(1, len(accesses) // 20)
        top_share = round(sum(accesses[:top_k]) / total_acc, 3) if total_acc else 0.0
        reinforced_24h = sum(1 for r in rows if r["last_reinforced"] and _days_since_iso(r["last_reinforced"]) < 1.0)

        return {
            "active": len(rows),
            "storage_p50": _pct(0.5),
            "storage_p90": _pct(0.9),
            "storage_max": round(strengths[-1], 3),
            "fading_count": fading,
            "fading_threshold": fading_threshold,
            "recalls_top5pct_share": top_share,
            "reinforced_24h": reinforced_24h,
        }

    @_locked
    def reinforce_item(self, item_id: str) -> None:
        """Record a re-study event: a similar memory arrived.

        Grows storage strength by the smaller re-study gain (the testing effect
        favours retrieving over re-encountering), counts the reinforcement, and
        refreshes accessibility. Difficulty weighting comes from the item's
        current retrieval strength, derived here from last_accessed + storage.
        """
        row = self.conn.execute(
            "SELECT storage_strength, last_accessed, created_at FROM memory_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        if row is None:
            return
        days = _days_since_iso(row["last_accessed"] or row["created_at"])
        rs_before = retrieval_strength(days, row["storage_strength"])
        delta = delta_storage(1.0, rs_before, gain=RESTUDY_GAIN)
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE memory_items SET storage_strength = storage_strength + ?, "
            "reinforcement_count = reinforcement_count + 1, last_reinforced = ?, "
            "last_accessed = ? WHERE id = ?",
            (delta, now, now, item_id),
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
            content=row["content"],
            memory_type=row["memory_type"],
            status=row["status"],
            access_count=row["access_count"],
            last_accessed=last_accessed,
            daily_ref=row["daily_ref"],
            storage_strength=row["storage_strength"] if "storage_strength" in row.keys() else 0.5,
            reinforcement_count=row["reinforcement_count"],
            last_reinforced=last_reinforced,
            source_event_id=row["source_event_id"] if "source_event_id" in row.keys() else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # --- Events (raw ingested turns) ---

    @_locked
    def save_event(self, event: Event) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO events
               (id, text, received_at, source_kind, thread_id, attribution, extraction_status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event.id,
                event.text,
                event.received_at.isoformat(),
                event.source_kind,
                event.thread_id or event.id,
                event.attribution,
                event.extraction_status,
            ),
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

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        keys = row.keys()
        return Event(
            id=row["id"],
            text=row["text"],
            received_at=datetime.fromisoformat(row["received_at"]),
            source_kind=row["source_kind"] or "unknown",
            thread_id=(row["thread_id"] if "thread_id" in keys else None) or row["id"],
            attribution=row["attribution"] if "attribution" in keys else None,
            extraction_status=row["extraction_status"] if "extraction_status" in keys else "extracted",
        )

    @_locked
    def get_all_events(self, limit: int | None = None) -> list[Event]:
        """All events in insertion order — used by the embed-backfill script."""
        sql = (
            "SELECT id, text, received_at, source_kind, thread_id, attribution, extraction_status "
            "FROM events ORDER BY received_at ASC"
        )
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    @_locked
    def get_event(self, event_id: str) -> Event | None:
        row = self.conn.execute(
            "SELECT id, text, received_at, source_kind, thread_id, attribution, extraction_status "
            "FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_event(row)

    @_locked
    def get_events_for_thread(self, thread_id: str) -> list[Event]:
        """All raw turns in a thread, oldest first — the conversation in order."""
        rows = self.conn.execute(
            "SELECT id, text, received_at, source_kind, thread_id, attribution, extraction_status FROM events "
            "WHERE thread_id = ? ORDER BY received_at ASC",
            (thread_id,),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    @_locked
    def get_thread_ids_for_events(self, event_ids: list[str]) -> dict[str, str]:
        """Map each event id to its thread id, in one batched lookup.

        The recall_recent snapshot groups recent memories by conversation, and
        each memory carries only its ``source_event_id``; this resolves a whole
        gather window's events to their threads without a query per memory. An
        event with a null ``thread_id`` stands as its own thread.
        """
        ids = [e for e in dict.fromkeys(event_ids) if e]
        out: dict[str, str] = {}
        chunk = 500
        for i in range(0, len(ids), chunk):
            part = ids[i : i + chunk]
            q = f"SELECT id, thread_id FROM events WHERE id IN ({','.join('?' * len(part))})"
            for row in self.conn.execute(q, part):
                out[row["id"]] = row["thread_id"] or row["id"]
        return out

    @_locked
    def get_pending_events_for_thread(self, thread_id: str) -> list[Event]:
        """A thread's turns still awaiting distillation, oldest first.

        The extraction worker's per-thread work list: the window it builds a
        transcript from, then marks extracted (or failed).
        """
        rows = self.conn.execute(
            "SELECT id, text, received_at, source_kind, thread_id, attribution, extraction_status "
            "FROM events WHERE thread_id = ? AND extraction_status = 'pending' "
            "ORDER BY received_at ASC",
            (thread_id,),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    @_locked
    def pending_thread_ids(self) -> list[str]:
        """Distinct threads holding at least one pending turn — the worker's recovery seed.

        On daemon start this reseeds the dirty map so turns buffered before a
        restart still flush instead of stalling unseen.
        """
        rows = self.conn.execute("SELECT DISTINCT thread_id FROM events WHERE extraction_status = 'pending'").fetchall()
        return [row["thread_id"] for row in rows]

    @_locked
    def mark_events_extracted(self, event_ids: list[str]) -> None:
        """Flip a flushed window from 'pending' to 'extracted'."""
        self._set_extraction_status(event_ids, "extracted")

    @_locked
    def mark_events_failed(self, event_ids: list[str]) -> None:
        """Flip a window the worker gave up on to 'failed' after its retries."""
        self._set_extraction_status(event_ids, "failed")

    # --- Reconciliation dismissals (judged-distinct entity pairs) ---

    @staticmethod
    def reconcile_pair_key(a_id: str, b_id: str) -> str:
        """Order-insensitive key for an entity pair."""
        return "|".join(sorted((a_id, b_id)))

    @_locked
    def add_reconcile_dismissal(self, a_id: str, b_id: str) -> None:
        """Record that a judge ruled this entity pair distinct, so reconcile stops surfacing it."""
        self.conn.execute(
            "INSERT OR IGNORE INTO reconcile_dismissals (pair_key, a_id, b_id, judged_at) VALUES (?, ?, ?, ?)",
            (
                self.reconcile_pair_key(a_id, b_id),
                a_id,
                b_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    @_locked
    def reconcile_dismissal_keys(self) -> set[str]:
        """Every judged-distinct pair key."""
        rows = self.conn.execute("SELECT pair_key FROM reconcile_dismissals").fetchall()
        return {row["pair_key"] for row in rows}

    # --- Consolidation queue (loose clusters awaiting roll-up) ---

    @_locked
    def enqueue_consolidation(self, anchor: str, member_ids: list[str], span: tuple[str, str] | None) -> None:
        """Queue a loose cluster for roll-up, upserting by ``anchor``.

        A theme re-surfaced by another recall refreshes its pending row rather
        than stacking a duplicate; an unchanged member set is a no-op so a
        repeated query doesn't reset the resurface cooldown.
        """
        if not member_ids:
            return
        members_json = json.dumps(sorted(member_ids))
        span_start, span_end = span or (None, None)
        now = datetime.now(timezone.utc).isoformat()
        existing = self.conn.execute(
            "SELECT id, member_ids FROM consolidation_queue WHERE anchor = ? AND status = 'pending'",
            (anchor,),
        ).fetchone()
        if existing:
            if existing["member_ids"] == members_json:
                return
            self.conn.execute(
                "UPDATE consolidation_queue SET member_ids = ?, loose_count = ?, "
                "span_start = ?, span_end = ?, detected_at = ? WHERE id = ?",
                (members_json, len(member_ids), span_start, span_end, now, existing["id"]),
            )
        else:
            self.conn.execute(
                "INSERT INTO consolidation_queue (id, anchor, member_ids, loose_count, "
                "span_start, span_end, detected_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')",
                (uuid.uuid4().hex, anchor, members_json, len(member_ids), span_start, span_end, now),
            )
        self.conn.commit()

    @_locked
    def list_pending_consolidations(self) -> list[dict]:
        """Pending clusters, newest detection first; ``member_ids`` parsed to a list."""
        rows = self.conn.execute(
            "SELECT id, anchor, member_ids, loose_count, span_start, span_end, detected_at, presented_at "
            "FROM consolidation_queue WHERE status = 'pending' ORDER BY detected_at DESC"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "anchor": r["anchor"],
                "member_ids": json.loads(r["member_ids"]),
                "loose_count": r["loose_count"],
                "span": (r["span_start"], r["span_end"]) if r["span_start"] else None,
                "detected_at": r["detected_at"],
                "presented_at": r["presented_at"],
            }
            for r in rows
        ]

    @_locked
    def mark_consolidation(self, queue_id: str, status: str) -> None:
        """Set a queue row's status (``done`` | ``dismissed`` | ``pending``)."""
        self.conn.execute("UPDATE consolidation_queue SET status = ? WHERE id = ?", (status, queue_id))
        self.conn.commit()

    @_locked
    def touch_consolidations_presented(self, queue_ids: list[str]) -> None:
        """Stamp ``presented_at`` on the rows just shown, for the resurface cooldown."""
        if not queue_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" * len(queue_ids))
        self.conn.execute(
            f"UPDATE consolidation_queue SET presented_at = ? WHERE id IN ({placeholders})",
            [now, *queue_ids],
        )
        self.conn.commit()

    @_locked
    def drop_consolidation(self, queue_id: str) -> None:
        """Delete a queue row outright — used when its cluster is fully rolled up."""
        self.conn.execute("DELETE FROM consolidation_queue WHERE id = ?", (queue_id,))
        self.conn.commit()

    def _set_extraction_status(self, event_ids: list[str], status: str) -> None:
        ids = [e for e in dict.fromkeys(event_ids) if e]
        if not ids:
            return
        chunk = 500
        for i in range(0, len(ids), chunk):
            part = ids[i : i + chunk]
            placeholders = ",".join("?" * len(part))
            self.conn.execute(
                f"UPDATE events SET extraction_status = ? WHERE id IN ({placeholders})",
                [status, *part],
            )
        self.conn.commit()

    # --- Threads (conversations: ordered runs of raw turns) ---

    def _row_to_thread(self, row: sqlite3.Row) -> Thread:
        return Thread(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            source_kind=row["source_kind"] or "agent",
            label=row["label"],
            client_key=row["client_key"] if "client_key" in row.keys() else None,
        )

    @_locked
    def save_thread(self, thread: Thread) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO threads (id, created_at, source_kind, label, client_key) VALUES (?, ?, ?, ?, ?)",
            (thread.id, thread.created_at.isoformat(), thread.source_kind, thread.label, thread.client_key),
        )
        self.conn.commit()

    @_locked
    def get_thread(self, thread_id: str) -> Thread | None:
        row = self.conn.execute(
            "SELECT id, created_at, source_kind, label, client_key FROM threads WHERE id = ?",
            (thread_id,),
        ).fetchone()
        return self._row_to_thread(row) if row else None

    @_locked
    def get_thread_by_client_key(self, client_key: str) -> Thread | None:
        row = self.conn.execute(
            "SELECT id, created_at, source_kind, label, client_key FROM threads WHERE client_key = ?",
            (client_key,),
        ).fetchone()
        return self._row_to_thread(row) if row else None
