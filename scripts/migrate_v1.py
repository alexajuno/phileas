"""One-time migration from v1 schema to new schema.

Reads old memory_items, re-inserts with new fields, embeds into ChromaDB.
Backs up old DB first.
"""

import shutil
import sqlite3
from pathlib import Path

from phileas.db import Database
from phileas.models import MemoryItem
from phileas.vector import VectorStore

OLD_DB = Path.home() / ".phileas" / "memory.db"
BACKUP_DB = Path.home() / ".phileas" / "memory.db.v1.bak"

IMPORTANCE_DEFAULTS = {
    "profile": 9,
    "event": 6,
    "knowledge": 5,
    "behavior": 5,
    "reflection": 7,
}


def migrate():
    if not OLD_DB.exists():
        print("No existing database found. Nothing to migrate.")
        return

    # Backup first
    shutil.copy(OLD_DB, BACKUP_DB)
    print(f"Backed up to {BACKUP_DB}")

    # Read old data
    old_conn = sqlite3.connect(str(OLD_DB))
    old_conn.row_factory = sqlite3.Row
    old_items = old_conn.execute("SELECT * FROM memory_items").fetchall()
    print(f"Found {len(old_items)} old memories")
    old_conn.close()

    # Delete old DB so new schema can be created fresh
    OLD_DB.unlink()

    # Initialize new backends
    db = Database(path=OLD_DB)
    vs = VectorStore()

    migrated = 0
    for row in old_items:
        memory_type = row["memory_type"]
        importance = IMPORTANCE_DEFAULTS.get(memory_type, 5)
        daily_ref = row["daily_ref"] if "daily_ref" in row.keys() else None

        item = MemoryItem(
            id=row["id"],
            summary=row["summary"],
            memory_type=memory_type,
            importance=importance,
            daily_ref=daily_ref,
        )
        db.save_item(item)
        vs.add(item.id, item.summary)
        migrated += 1
        if migrated % 20 == 0:
            print(f"  ...migrated {migrated}/{len(old_items)}")

    db.close()
    print(f"\nDone. Migrated {migrated} memories to new schema + ChromaDB.")
    print(f"Old DB backed up at: {BACKUP_DB}")


if __name__ == "__main__":
    migrate()
