"""One-off: seed the memory_sources join from the legacy memory_items.source_event_id.

Provenance became a set (many-to-many memory_sources) rather than a single
source_event_id column. Going-forward writes populate the join in engine.memorize;
this brings the one existing database, ~/.phileas/memory.db, to the same shape by
copying every real (non-NULL, non-'unknown') source_event_id into memory_sources
as a one-element set.

Idempotent: INSERT OR IGNORE on the (memory_id, event_id) primary key, so a second
run is a no-op. It writes a timestamped backup before touching anything. The
memory_sources table itself is created by the app on startup; run the app (or the
daemon) once first, or run with the daemon stopped so the schema is current.
"""

import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / ".phileas" / "memory.db"


def main() -> None:
    # Optional path arg lets the backfill run against a throwaway copy first.
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DB_PATH
    if not db_path.exists():
        raise SystemExit(f"no database at {db_path}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_suffix(f".db.memsrc-{stamp}.bak")
    shutil.copy2(db_path, backup)
    print(f"backup: {backup}")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 5000")  # ride out any transient lock
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "memory_sources" not in tables:
            raise SystemExit("memory_sources table missing; open the store once so the schema is current, then re-run")

        cur = conn.execute(
            """INSERT OR IGNORE INTO memory_sources (memory_id, event_id)
               SELECT id, source_event_id FROM memory_items
               WHERE source_event_id IS NOT NULL AND source_event_id != 'unknown'"""
        )
        conn.commit()
        print(f"linked {cur.rowcount} legacy source(s) into memory_sources")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
