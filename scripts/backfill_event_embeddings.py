"""Backfill event-text embeddings into the new ChromaDB `events` collection.

One-time migration for the thread/provenance recall feature (AA-50). Idempotent:
ChromaDB upsert means re-running is safe.

Run after deploying the code change so existing events become searchable via
Path 6 / `thread()`. New events embed automatically through `engine.save_event`.

Usage:
    uv run python scripts/backfill_event_embeddings.py
"""

from __future__ import annotations

from phileas.config import load_config
from phileas.db import Database
from phileas.vector import VectorStore


def main() -> int:
    config = load_config()
    db = Database(path=config.db_path)
    vector = VectorStore(path=config.chroma_path)

    events = db.get_all_events()
    if not events:
        print("No events to backfill.")
        return 0

    embedded = 0
    skipped = 0
    for event in events:
        if not event.text:
            skipped += 1
            continue
        vector.add_event(event.id, event.text)
        embedded += 1
        if embedded % 100 == 0:
            print(f"  embedded {embedded}/{len(events)}")

    print(f"Done. Embedded {embedded} events ({skipped} skipped). Collection size: {vector.event_count()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
