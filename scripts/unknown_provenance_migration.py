"""One-off: repoint unrecorded provenance to the 'unknown' sentinel and tighten
source_event_id / thread_id to NOT NULL on the live database.

Going-forward writes are already safe (db.SCHEMA is NOT NULL, save_item floors a
missing source to the sentinel, and the MCP boundary requires a real event). This
script brings the one existing database, ~/.phileas/memory.db, to that same shape:

  1. memory_items.source_event_id IS NULL  -> 'unknown'   (memories from before
     the provenance contract existed)
  2. events that the thread backfill stamped with their own id  -> 'unknown'
     (turns captured before the conversation layer; their thread was never real)
  3. rebuild both tables so the two columns are NOT NULL, preserving every
     existing column (including the vestigial source_session_id / tags)

Idempotent and reversible: it skips the rebuild if the columns are already NOT
NULL, and it writes a timestamped backup of the db file before touching anything.
Run with the daemon stopped.
"""

import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / ".phileas" / "memory.db"

# The conversation-thread feature (commit bc9a43d) landed 2026-06-16. Every turn
# captured before that date had its thread_id stamped to its own id by the
# one-time backfill, i.e. its real conversation was never recorded.
THREADING_CUTOFF = "2026-06-16"


def _col_not_null(conn: sqlite3.Connection, table: str, col: str) -> bool:
    for row in conn.execute(f"PRAGMA table_info({table})"):
        if row[1] == col:
            return bool(row[3])
    raise SystemExit(f"{table}.{col} not found")


def main() -> None:
    # Optional path arg lets the migration run against a throwaway copy first.
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DB_PATH
    if not db_path.exists():
        raise SystemExit(f"no database at {db_path}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_suffix(f".db.unknown-prov-{stamp}.bak")
    shutil.copy2(db_path, backup)
    print(f"backup: {backup}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")  # ride out any transient lock
    try:
        # Sentinels first, so they are part of the baseline rather than a delta
        # the row-count verify would have to special-case (no-op if the app
        # already seeded them since the schema change).
        conn.execute(
            "INSERT OR IGNORE INTO threads (id, created_at, source_kind, label) "
            "VALUES ('unknown', '1970-01-01T00:00:00+00:00', 'unknown', 'unknown provenance')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO events (id, text, received_at, source_kind, thread_id) "
            "VALUES ('unknown', '', '1970-01-01T00:00:00+00:00', 'unknown', 'unknown')"
        )
        conn.commit()

        before_items = conn.execute("SELECT count(*) FROM memory_items").fetchone()[0]
        before_events = conn.execute("SELECT count(*) FROM events").fetchone()[0]

        n_mem = conn.execute(
            "UPDATE memory_items SET source_event_id = 'unknown' WHERE source_event_id IS NULL"
        ).rowcount
        n_ev = conn.execute(
            "UPDATE events SET thread_id = 'unknown' WHERE thread_id = id AND received_at < ? AND id <> 'unknown'",
            (THREADING_CUTOFF,),
        ).rowcount
        conn.commit()
        print(f"repointed memories -> unknown event: {n_mem}")
        print(f"repointed turns    -> unknown thread: {n_ev}")

        already = _col_not_null(conn, "memory_items", "source_event_id") and _col_not_null(conn, "events", "thread_id")
        if already:
            print("columns already NOT NULL; skipping rebuild")
        else:
            _rebuild(conn)
            print("rebuilt memory_items + events as NOT NULL")

        # Verify before reporting success.
        null_src = conn.execute("SELECT count(*) FROM memory_items WHERE source_event_id IS NULL").fetchone()[0]
        null_thr = conn.execute("SELECT count(*) FROM events WHERE thread_id IS NULL").fetchone()[0]
        after_items = conn.execute("SELECT count(*) FROM memory_items").fetchone()[0]
        after_events = conn.execute("SELECT count(*) FROM events").fetchone()[0]

        ok = (
            null_src == 0
            and null_thr == 0
            and after_items == before_items
            and after_events == before_events
            and _col_not_null(conn, "memory_items", "source_event_id")
            and _col_not_null(conn, "events", "thread_id")
        )
        print(
            f"verify: items {before_items}->{after_items}, events {before_events}->{after_events}, "
            f"null source_event_id={null_src}, null thread_id={null_thr}"
        )
        if not ok:
            print("VERIFY FAILED — restore from backup", file=sys.stderr)
            raise SystemExit(1)
        print("done")
    finally:
        conn.close()


def _rebuild(conn: sqlite3.Connection) -> None:
    """Rebuild both tables with NOT NULL on the provenance columns.

    Column lists are explicit and match the live shape (which still carries the
    vestigial source_session_id / tags) so nothing is dropped. SQLite cannot
    tighten a column in place, hence the create-copy-swap.
    """
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("BEGIN")

    conn.execute(
        """
        CREATE TABLE events_new (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            received_at TEXT NOT NULL,
            source_kind TEXT,
            thread_id TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO events_new (id, text, received_at, source_kind, thread_id) "
        "SELECT id, text, received_at, source_kind, thread_id FROM events"
    )
    conn.execute("DROP TABLE events")
    conn.execute("ALTER TABLE events_new RENAME TO events")
    conn.execute("CREATE INDEX idx_events_received ON events(received_at)")
    conn.execute("CREATE INDEX idx_events_thread ON events(thread_id, received_at)")

    conn.execute(
        """
        CREATE TABLE memory_items_new (
            id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed TEXT,
            daily_ref TEXT,
            source_session_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            reinforcement_count INTEGER NOT NULL DEFAULT 0,
            last_reinforced TEXT,
            tags TEXT NOT NULL DEFAULT '[]',
            source_event_id TEXT NOT NULL DEFAULT 'unknown' REFERENCES events(id),
            storage_strength REAL NOT NULL DEFAULT -1.0
        )
        """
    )
    conn.execute(
        "INSERT INTO memory_items_new (id, summary, memory_type, status, access_count, "
        "last_accessed, daily_ref, source_session_id, created_at, updated_at, "
        "reinforcement_count, last_reinforced, tags, source_event_id, storage_strength) "
        "SELECT id, summary, memory_type, status, access_count, last_accessed, daily_ref, "
        "source_session_id, created_at, updated_at, reinforcement_count, last_reinforced, "
        "tags, source_event_id, storage_strength FROM memory_items"
    )
    conn.execute("DROP TABLE memory_items")
    conn.execute("ALTER TABLE memory_items_new RENAME TO memory_items")
    conn.execute("CREATE INDEX idx_items_status ON memory_items(status)")
    conn.execute("CREATE INDEX idx_items_type ON memory_items(memory_type)")
    conn.execute("CREATE INDEX idx_items_daily_ref ON memory_items(daily_ref)")

    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")


if __name__ == "__main__":
    main()
