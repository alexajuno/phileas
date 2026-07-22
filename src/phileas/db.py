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
from phileas.models import MemoryItem, Source
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

# Sentinel source id. A memory whose originating session was never recorded points
# here, so source_id is never a dangling reference and a source -> memories
# drill-down resolves to a row. "unknown" reads as exactly that: origin not recorded.
UNKNOWN_SOURCE_ID = "unknown"


def clean_source_id(value: str | None) -> str | None:
    """Normalize a memory's source reference for storage.

    A memory either traces to one captured session (a real source id) or has no
    single source, which is NULL: a reflection or rollup derived from other
    memories, or a legacy row from before sessions were tracked. Empty strings and
    the ``UNKNOWN_SOURCE_ID`` sentinel collapse to NULL too, so the placeholder
    never lands on a memory's provenance again.
    """
    sid = (value or "").strip()
    if not sid or sid == UNKNOWN_SOURCE_ID:
        return None
    return sid


def _attr_to_role(attribution: str | None) -> str:
    """Map a legacy event attribution to a unified-format turn role.

    ``self`` was the user's own words, so it becomes ``user``; ``assistant`` and
    ``source`` carry over. An untagged legacy turn defaults to ``user``.
    """
    return {"self": "user", "assistant": "assistant", "source": "source"}.get(attribution or "", "user")


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    daily_ref TEXT,
    source_id TEXT REFERENCES sources(id),
    storage_strength REAL NOT NULL DEFAULT 1.0,
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    last_reinforced TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- A whole ingested session, distilled into memories as a unit. ``payload`` is the
-- unified-format transcript (``{client_key, kind, cwd, started_at, ended_at,
-- turns: [...]}``) held as one JSON blob, since a session is always read and
-- written whole. ``client_key`` is the producing session's stable identity, so an
-- upsert is get-or-create on it and a resumed session updates the row it already
-- opened. ``extraction_status`` is the distillation lifecycle (open/ready/
-- extracting/extracted/failed); ``extracted_through`` is the high-water mark of
-- turns already distilled, so a resume re-extracts only its new turns.
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    client_key TEXT,
    kind TEXT,
    cwd TEXT,
    label TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    turn_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    last_activity_at TEXT,
    ended_at TEXT,
    extraction_status TEXT NOT NULL DEFAULT 'open',
    extracted_through INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

-- A client's stable session key maps to at most one source, so a resumed or
-- compacted session re-attaches to the source it already opened. Partial so
-- sources opened without a key (no continuity) don't collide on NULL.
CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_client_key
    ON sources(client_key) WHERE client_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(extraction_status);

-- A memory's provenance is the SET of sessions it was distilled from — usually
-- one, but a rollup can span several. This join holds that set (many-to-many
-- memory <-> source). ``memory_items.source_id`` is the degenerate one-element
-- case, the memory's primary source, mirrored into this table.
CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id TEXT NOT NULL REFERENCES memory_items(id),
    source_id TEXT NOT NULL REFERENCES sources(id),
    PRIMARY KEY (memory_id, source_id)
);

CREATE INDEX IF NOT EXISTS idx_items_status ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_items_type ON memory_items(memory_type);
CREATE INDEX IF NOT EXISTS idx_items_daily_ref ON memory_items(daily_ref);

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

-- Candidate memories awaiting the user's review before they become real
-- memories. The live model enqueues here (via propose_memory); the user approves
-- / edits / rejects through `phileas memory queue` or the web, and an approved
-- row is materialized through engine.memorize. Holds the full memory payload
-- (entities / relationships as JSON) plus the source it was distilled from, so
-- approval attaches that source as the memory's provenance.
CREATE TABLE IF NOT EXISTS memory_proposals (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_text TEXT,
    memory_type TEXT NOT NULL DEFAULT 'knowledge',
    entities TEXT NOT NULL DEFAULT '[]',
    relationships TEXT NOT NULL DEFAULT '[]',
    source_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON memory_proposals(status);

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

-- Sentinel source for memories whose originating session was never recorded, so
-- a memory.source_id always resolves to a row and a source -> memories drill-down
-- holds.
INSERT OR IGNORE INTO sources (id, kind, label, payload, turn_count, extraction_status, created_at)
    VALUES ('unknown', 'unknown', 'unknown provenance', '{}', 0, 'extracted', '1970-01-01T00:00:00+00:00');
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
        """Bring a database created under an older schema forward, idempotently.

        The schema is CREATE TABLE IF NOT EXISTS, which never alters an existing
        table, so a column added later has to be ALTERed in by hand and a renamed
        model migrated. Each step is guarded on the old shape, so running this on
        an already-current database is a no-op.
        """
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

        # ``memory_items.source_event_id`` (FK to the retired events table) became
        # ``source_id`` (FK to sources). Add the new column so writes land; the
        # legacy fold below backfills it from the old one.
        if "source_id" not in item_cols:
            self.conn.execute("ALTER TABLE memory_items ADD COLUMN source_id TEXT")
        self.conn.commit()

        self._rebase_storage_strength()
        self._migrate_legacy_sources()

        # The join's source_id index lives here rather than in SCHEMA: on a legacy
        # database the table only gains that column during the fold above, so
        # indexing it in the up-front schema script would fail.
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_sources_source ON memory_sources(source_id)")
        self.conn.commit()

    def _rebase_storage_strength(self) -> None:
        """Move existing rows onto the uniform 1.0 starting strength, once.

        Storage strength used to start at a per-type value (event 0.4, knowledge
        and reflection 0.5, behavior 0.6, profile and decision 0.7); it now starts
        at 1.0 for every memory. Subtracting the start a row was written with and
        adding the new one shifts the whole store onto the new floor while keeping
        each memory's earned growth intact. Stamped in ``user_version`` so a row
        can't be shifted twice.
        """
        if self.conn.execute("PRAGMA user_version").fetchone()[0] >= 1:
            return
        self.conn.execute(
            """UPDATE memory_items SET storage_strength = storage_strength + 1.0 - (
                   CASE memory_type
                       WHEN 'profile' THEN 0.7
                       WHEN 'decision' THEN 0.7
                       WHEN 'behavior' THEN 0.6
                       WHEN 'event' THEN 0.4
                       ELSE 0.5
                   END)"""
        )
        self.conn.execute("PRAGMA user_version = 1")
        self.conn.commit()

    def _table_exists(self, name: str) -> bool:
        return (
            self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
            is not None
        )

    def _migrate_legacy_sources(self) -> None:
        """Fold a legacy threads+events database into the sources model, once.

        Each thread becomes a source whose payload is its turns in order; a
        memory's ``source_event_id`` is rewired to the source of the event's
        thread, and the ``memory_sources`` join is remapped from event ids to
        source ids. The old tables are left in place (readable, inert) so the fold
        stays reversible; a later cleanup drops them. Guarded on the events table,
        so it never runs on a database born under the sources schema, and on a
        sentinel-only sources table, so it runs at most once.
        """
        if not self._table_exists("events"):
            return
        if self.conn.execute("SELECT 1 FROM sources WHERE id != 'unknown' LIMIT 1").fetchone():
            return

        # 1. Every thread an event belongs to becomes a source (DISTINCT covers
        #    one-turn threads that never got a threads row).
        thread_ids = [
            r[0] for r in self.conn.execute("SELECT DISTINCT thread_id FROM events WHERE thread_id != 'unknown'")
        ]
        for tid in thread_ids:
            evs = self.conn.execute(
                "SELECT id, text, received_at, source_kind, attribution FROM events "
                "WHERE thread_id = ? ORDER BY received_at ASC",
                (tid,),
            ).fetchall()
            if not evs:
                continue
            meta = self.conn.execute(
                "SELECT created_at, source_kind, label, client_key FROM threads WHERE id = ?", (tid,)
            ).fetchone()
            turns = [
                {"i": i, "role": _attr_to_role(e["attribution"]), "text": e["text"], "ts": e["received_at"]}
                for i, e in enumerate(evs)
            ]
            kind = (meta["source_kind"] if meta else evs[0]["source_kind"]) or "claude_code_session"
            client_key = meta["client_key"] if meta else None
            started, ended = evs[0]["received_at"], evs[-1]["received_at"]
            payload = json.dumps(
                {"client_key": client_key, "kind": kind, "started_at": started, "ended_at": ended, "turns": turns}
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO sources (id, client_key, kind, label, payload, turn_count, started_at, "
                "last_activity_at, ended_at, extraction_status, extracted_through, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'extracted', ?, ?)",
                (
                    tid,
                    client_key,
                    kind,
                    meta["label"] if meta else None,
                    payload,
                    len(turns),
                    started,
                    ended,
                    ended,
                    len(turns),
                    meta["created_at"] if meta else started,
                ),
            )

        # 2. Rewire memory_items.source_id from the old source_event_id via its event's thread.
        if "source_event_id" in {r["name"] for r in self.conn.execute("PRAGMA table_info(memory_items)")}:
            self.conn.execute(
                "UPDATE memory_items SET source_id = ("
                "  SELECT e.thread_id FROM events e WHERE e.id = memory_items.source_event_id"
                ") WHERE source_id IS NULL AND source_event_id IS NOT NULL AND source_event_id != 'unknown'"
            )

        # 3. Remap the memory_sources join from event ids to source (thread) ids.
        if "event_id" in {r["name"] for r in self.conn.execute("PRAGMA table_info(memory_sources)")}:
            old = self.conn.execute("SELECT memory_id, event_id FROM memory_sources").fetchall()
            emap = {r[0]: r[1] for r in self.conn.execute("SELECT id, thread_id FROM events")}
            self.conn.execute("DROP TABLE memory_sources")
            self.conn.execute(
                "CREATE TABLE memory_sources (memory_id TEXT NOT NULL REFERENCES memory_items(id), "
                "source_id TEXT NOT NULL REFERENCES sources(id), PRIMARY KEY (memory_id, source_id))"
            )
            rows = [
                (m["memory_id"], emap[m["event_id"]])
                for m in old
                if m["event_id"] in emap and emap[m["event_id"]] != "unknown"
            ]
            if rows:
                self.conn.executemany("INSERT OR IGNORE INTO memory_sources (memory_id, source_id) VALUES (?, ?)", rows)

        # 4. Proposals anchored to a thread now anchor to that source (same id).
        if "source_id" not in {r["name"] for r in self.conn.execute("PRAGMA table_info(memory_proposals)")}:
            self.conn.execute("ALTER TABLE memory_proposals ADD COLUMN source_id TEXT")
        if "thread_id" in {r["name"] for r in self.conn.execute("PRAGMA table_info(memory_proposals)")}:
            self.conn.execute("UPDATE memory_proposals SET source_id = thread_id WHERE source_id IS NULL")

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
                source_id, created_at, updated_at)
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
                clean_source_id(item.source_id),
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
            ),
        )
        self._fts_upsert(item.id, item.content)
        self.conn.commit()

    @_locked
    def add_memory_sources(self, memory_id: str, source_ids: list[str]) -> None:
        """Record a memory's provenance: the set of sessions it was distilled from.

        Idempotent per (memory_id, source_id). Empty / 'unknown' / missing ids are
        dropped by ``clean_source_id``, so only real sessions become sources; a
        memory with no resolvable source (a reflection, a legacy row) simply has no
        rows here.
        """
        rows = [(memory_id, sid) for raw in source_ids if (sid := clean_source_id(raw))]
        if not rows:
            return
        self.conn.executemany("INSERT OR IGNORE INTO memory_sources (memory_id, source_id) VALUES (?, ?)", rows)
        self.conn.commit()

    @_locked
    def get_source_ids_for_memory(self, memory_id: str) -> list[str]:
        """The session ids a memory was distilled from, oldest source first."""
        rows = self.conn.execute(
            "SELECT ms.source_id FROM memory_sources ms JOIN sources s ON s.id = ms.source_id "
            "WHERE ms.memory_id = ? ORDER BY s.started_at ASC",
            (memory_id,),
        ).fetchall()
        return [row["source_id"] for row in rows]

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
        cases. A full uuid resolves to at most one row, so a caller holding the
        id8 printed on a recall line can name that memory unambiguously.
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
    def source_status_counts(self) -> dict[str, int]:
        """Source counts grouped by extraction_status — the worker's queue depth."""
        return {
            row[0]: row[1]
            for row in self.conn.execute("SELECT extraction_status, COUNT(*) FROM sources GROUP BY extraction_status")
        }

    @_locked
    def web_ingestion_health(self) -> dict:
        """Source-ingestion counts (1h / 24h / total), plus the extraction queue depth."""
        now = datetime.now(timezone.utc)
        h1 = (now - timedelta(hours=1)).isoformat()
        d1 = (now - timedelta(days=1)).isoformat()
        status_counts = self.source_status_counts()
        return {
            "sources_ingested_1h": self.conn.execute(
                "SELECT COUNT(*) FROM sources WHERE created_at >= ?", (h1,)
            ).fetchone()[0],
            "sources_ingested_24h": self.conn.execute(
                "SELECT COUNT(*) FROM sources WHERE created_at >= ?", (d1,)
            ).fetchone()[0],
            "sources_total": self.conn.execute("SELECT COUNT(*) FROM sources WHERE id != 'unknown'").fetchone()[0],
            "sources_ready": status_counts.get("ready", 0),
            "sources_failed": status_counts.get("failed", 0),
        }

    @_locked
    def web_ingestion_sources(self, limit: int = 50) -> list[dict]:
        """Recent ingested sessions with turn count and extraction status."""
        limit = min(max(50 if limit is None else limit, 1), 500)
        rows = self.conn.execute(
            "SELECT id, client_key, kind, label, turn_count, extraction_status, last_activity_at "
            "FROM sources WHERE id != 'unknown' ORDER BY last_activity_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "client_key": r["client_key"],
                "kind": r["kind"],
                "label": r["label"],
                "turn_count": r["turn_count"],
                "extraction_status": r["extraction_status"],
                "last_activity_at": r["last_activity_at"],
            }
            for r in rows
        ]

    @_locked
    def web_ingestion_source(self, source_id: str) -> dict | None:
        """One session plus its linked active memories."""
        src = self.conn.execute(
            "SELECT id, client_key, kind, label, payload, turn_count, extraction_status, "
            "started_at, ended_at, last_activity_at FROM sources WHERE id = ?",
            (source_id,),
        ).fetchone()
        if not src:
            return None
        mems = self.conn.execute(
            """SELECT id, content, memory_type, created_at
               FROM memory_items
               WHERE status = 'active'
                 AND (source_id = ?
                      OR id IN (SELECT memory_id FROM memory_sources WHERE source_id = ?))
               ORDER BY created_at ASC""",
            (source_id, source_id),
        ).fetchall()
        return {
            "source": {
                "id": src["id"],
                "client_key": src["client_key"],
                "kind": src["kind"],
                "label": src["label"],
                "payload": json.loads(src["payload"] or "{}"),
                "turn_count": src["turn_count"],
                "extraction_status": src["extraction_status"],
                "started_at": src["started_at"],
                "ended_at": src["ended_at"],
                "last_activity_at": src["last_activity_at"],
            },
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
            storage_strength=row["storage_strength"] if "storage_strength" in row.keys() else 1.0,
            reinforcement_count=row["reinforcement_count"],
            last_reinforced=last_reinforced,
            source_id=row["source_id"] if "source_id" in row.keys() else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # --- Sources (whole ingested sessions) ---

    def _row_to_source(self, row: sqlite3.Row) -> Source:
        keys = row.keys()

        def _dt(key: str) -> datetime | None:
            v = row[key] if key in keys else None
            return datetime.fromisoformat(v) if v else None

        return Source(
            id=row["id"],
            client_key=row["client_key"] if "client_key" in keys else None,
            kind=(row["kind"] if "kind" in keys else None) or "claude_code_session",
            cwd=row["cwd"] if "cwd" in keys else None,
            label=row["label"] if "label" in keys else None,
            payload=json.loads((row["payload"] if "payload" in keys else None) or "{}"),
            turn_count=(row["turn_count"] if "turn_count" in keys else 0) or 0,
            started_at=_dt("started_at"),
            last_activity_at=_dt("last_activity_at") or datetime.now(timezone.utc),
            ended_at=_dt("ended_at"),
            extraction_status=(row["extraction_status"] if "extraction_status" in keys else None) or "open",
            extracted_through=(row["extracted_through"] if "extracted_through" in keys else 0) or 0,
            created_at=_dt("created_at") or datetime.now(timezone.utc),
        )

    @_locked
    def save_source(self, source: Source) -> None:
        """Insert or replace a source by id (the id is stable per client_key)."""
        self.conn.execute(
            """INSERT OR REPLACE INTO sources
               (id, client_key, kind, cwd, label, payload, turn_count,
                started_at, last_activity_at, ended_at, extraction_status,
                extracted_through, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source.id,
                source.client_key,
                source.kind,
                source.cwd,
                source.label,
                json.dumps(source.payload or {}),
                source.turn_count,
                source.started_at.isoformat() if source.started_at else None,
                source.last_activity_at.isoformat() if source.last_activity_at else None,
                source.ended_at.isoformat() if source.ended_at else None,
                source.extraction_status,
                source.extracted_through,
                source.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    @_locked
    def get_source(self, source_id: str) -> Source | None:
        row = self.conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        return self._row_to_source(row) if row else None

    @_locked
    def get_source_by_client_key(self, client_key: str) -> Source | None:
        """The source a client's stable session key already opened, or None."""
        row = self.conn.execute("SELECT * FROM sources WHERE client_key = ?", (client_key,)).fetchone()
        return self._row_to_source(row) if row else None

    @_locked
    def get_all_sources(self, limit: int | None = None) -> list[Source]:
        """Every real (non-sentinel) source, oldest first."""
        sql = "SELECT * FROM sources WHERE id != 'unknown' ORDER BY created_at ASC"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return [self._row_to_source(row) for row in self.conn.execute(sql, params).fetchall()]

    @_locked
    def get_memories_for_source(self, source_id: str) -> list[MemoryItem]:
        """Active memories distilled from this session, oldest first.

        Reads the many-to-many ``memory_sources`` join, unioned with the single
        ``source_id`` column so a memory's primary source is always found.
        """
        rows = self.conn.execute(
            """SELECT * FROM memory_items
               WHERE status = 'active'
                 AND (source_id = ?
                      OR id IN (SELECT memory_id FROM memory_sources WHERE source_id = ?))
               ORDER BY created_at ASC""",
            (source_id, source_id),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    @_locked
    def get_ready_sources(self, limit: int = 20) -> list[Source]:
        """Sessions marked 'ready' (done, awaiting distillation), oldest activity first."""
        rows = self.conn.execute(
            "SELECT * FROM sources WHERE extraction_status = 'ready' ORDER BY last_activity_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_source(row) for row in rows]

    @_locked
    def set_source_status(self, source_id: str, status: str) -> None:
        """Move a source through its distillation lifecycle."""
        self.conn.execute("UPDATE sources SET extraction_status = ? WHERE id = ?", (status, source_id))
        self.conn.commit()

    @_locked
    def mark_source_extracted(self, source_id: str, extracted_through: int) -> None:
        """Record a source distilled through ``extracted_through`` of its turns."""
        self.conn.execute(
            "UPDATE sources SET extraction_status = 'extracted', extracted_through = ? WHERE id = ?",
            (extracted_through, source_id),
        )
        self.conn.commit()

    @_locked
    def reset_extracting_sources(self) -> list[str]:
        """Return any source stuck 'extracting' (a crash mid-distillation) to 'ready'.

        The worker's recovery seed on daemon start, so a session interrupted
        mid-distillation is retried instead of stalling in a held state.
        """
        ids = [r[0] for r in self.conn.execute("SELECT id FROM sources WHERE extraction_status = 'extracting'")]
        if ids:
            self.conn.execute("UPDATE sources SET extraction_status = 'ready' WHERE extraction_status = 'extracting'")
            self.conn.commit()
        return ids

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

    # --- Memory proposals (the review queue) ---

    @staticmethod
    def _row_to_proposal(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "content": row["content"],
            "source_text": row["source_text"],
            "memory_type": row["memory_type"],
            "entities": json.loads(row["entities"] or "[]"),
            "relationships": json.loads(row["relationships"] or "[]"),
            "source_id": row["source_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
        }

    @_locked
    def save_proposal(
        self,
        proposal_id: str,
        content: str,
        memory_type: str = "knowledge",
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
        source_text: str | None = None,
        source_id: str | None = None,
    ) -> None:
        """Enqueue one candidate memory for review (status 'pending')."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO memory_proposals (id, content, source_text, memory_type, entities, "
            "relationships, source_id, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (
                proposal_id,
                content,
                source_text,
                memory_type,
                json.dumps(entities or []),
                json.dumps(relationships or []),
                source_id,
                now,
            ),
        )
        self.conn.commit()

    @_locked
    def get_proposal(self, proposal_id: str) -> dict | None:
        """One proposal by full id or 8-char prefix, or None (prefix picks the newest match)."""
        row = self.conn.execute(
            "SELECT * FROM memory_proposals WHERE id = ? OR id LIKE ? ORDER BY created_at DESC LIMIT 1",
            (proposal_id, f"{proposal_id}%"),
        ).fetchone()
        return self._row_to_proposal(row) if row else None

    @_locked
    def list_proposals(self, status: str | None = "pending", source_id: str | None = None) -> list[dict]:
        """Proposals, newest first; filter by status (default 'pending') and/or source.

        Pass ``status=None`` for every status.
        """
        clauses: list[str] = []
        args: list[str] = []
        if status is not None:
            clauses.append("status = ?")
            args.append(status)
        if source_id is not None:
            clauses.append("source_id = ?")
            args.append(source_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM memory_proposals {where} ORDER BY created_at DESC", args).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    @_locked
    def update_proposal(self, proposal_id: str, fields: dict) -> None:
        """Edit a pending proposal's memory payload before approval.

        Only content, source_text, memory_type, entities, relationships are
        editable; unknown keys are ignored. entities/relationships are JSON-encoded.
        """
        editable = {"content", "source_text", "memory_type", "entities", "relationships"}
        sets: list[str] = []
        args: list = []
        for key, value in fields.items():
            if key not in editable:
                continue
            if key in ("entities", "relationships"):
                value = json.dumps(value or [])
            sets.append(f"{key} = ?")
            args.append(value)
        if not sets:
            return
        args.append(proposal_id)
        self.conn.execute(f"UPDATE memory_proposals SET {', '.join(sets)} WHERE id = ?", args)
        self.conn.commit()

    @_locked
    def mark_proposal_resolved(self, proposal_id: str, status: str) -> None:
        """Set a proposal's terminal status ('approved' | 'rejected') and stamp resolved_at."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE memory_proposals SET status = ?, resolved_at = ? WHERE id = ?",
            (status, now, proposal_id),
        )
        self.conn.commit()

    @_locked
    def drop_consolidation(self, queue_id: str) -> None:
        """Delete a queue row outright — used when its cluster is fully rolled up."""
        self.conn.execute("DELETE FROM consolidation_queue WHERE id = ?", (queue_id,))
        self.conn.commit()
