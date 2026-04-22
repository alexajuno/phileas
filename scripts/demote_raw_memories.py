"""Demote legacy memories that are raw user turns to the events table.

Heuristic: summary LIKE 'User: %' among active memories. These rows slipped
past the earlier archive_raw_dumps.py pass because they're *user-only* turns
(no 'Assistant:' / 'A:' section), but they're still verbatim prompts — not
AI-written facts. Home for raw turns is the events table, not memory_items.

For each match:
  1. Create an Event(text=summary, received_at=created_at,
     extraction_status='skipped', extraction_error='legacy raw turn …')
  2. Archive the memory row.
  3. Remove it from Chroma (memories + raw collections).

Status 'skipped' keeps the retry loop from re-extracting them. If any ever
turn out to be worth extracting, flip them back with:

    UPDATE events SET extraction_status='pending' WHERE extraction_status='skipped' AND id IN (...);
    # then: phileas retry-events

Run with --dry-run (default) to preview, then --confirm to act.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from phileas.config import load_config
from phileas.db import Database
from phileas.models import Event, MemoryItem
from phileas.vector import VectorStore


def _candidates(db: Database) -> list[MemoryItem]:
    rows = db.conn.execute(
        """
        SELECT * FROM memory_items
        WHERE status = 'active'
          AND summary LIKE 'User: %'
        ORDER BY created_at ASC
        """
    ).fetchall()
    return [db._row_to_item(r) for r in rows]


def _demote_one(db: Database, vector: VectorStore, item: MemoryItem) -> str:
    received_at: datetime = item.created_at or datetime.utcnow()
    event = Event(
        text=item.summary,
        received_at=received_at,
        extraction_status="skipped",
        extraction_error="legacy raw turn converted from memory row",
        memory_count=0,
    )
    db.save_event(event)
    db.archive_item(item.id)
    try:
        vector.delete(item.id)
    except Exception:
        pass
    try:
        vector.delete_raw(item.id)
    except Exception:
        pass
    return event.id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Actually demote (default is dry-run).")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of rows affected.")
    args = parser.parse_args()

    cfg = load_config()
    db = Database(path=cfg.db_path)
    items = _candidates(db)
    if args.limit:
        items = items[: args.limit]

    print(f"Candidates: {len(items)} raw user-turn memories")
    for item in items[:5]:
        preview = item.summary[:120].replace("\n", " ")
        print(f"  [{item.id[:8]}] {item.memory_type}/imp={item.importance} {preview}...")
    if len(items) > 5:
        print(f"  ... and {len(items) - 5} more")

    if not args.confirm:
        print("\nDRY RUN — pass --confirm to demote.")
        return

    vector = VectorStore(path=cfg.chroma_path)
    demoted = 0
    for item in items:
        _demote_one(db, vector, item)
        demoted += 1

    print(f"\nDemoted {demoted} memory rows → events (status='skipped').")
    print("Undo: UPDATE memory_items SET status='active' WHERE status='archived' AND summary LIKE 'User: %';")
    print("      (the skipped event rows can also be DELETEd from events if the memory row is restored)")


if __name__ == "__main__":
    main()
