"""Archive legacy raw-dump memories and remove them from Chroma.

Background: prior to the events-backed ingest refactor, a silent fallback path
in `llm/extraction.py` wrote raw conversation turns ("User: ...\\n\\nAssistant: ...")
directly into memory_items as `knowledge`/`importance=5` rows. Recall surfaces
them as noise because their summaries aren't AI-written facts — they're chat
transcripts. The raw turn can always be rebuilt from transcripts or from
future events, so losing them is fine; what matters is keeping recall clean.

This script is archive-only (reversible). To undo, run a SQL UPDATE that
flips status back to 'active' for the rows it touched.

Run with --dry-run (default) to preview, then --confirm to actually act.
"""

from __future__ import annotations

import argparse
from typing import Iterable

from phileas.config import load_config
from phileas.db import Database
from phileas.models import MemoryItem
from phileas.vector import VectorStore


def _candidates(db: Database) -> list[MemoryItem]:
    """Tight heuristic: legacy knowledge/imp=5 rows whose summary is a raw
    ``User: ...`` + ``Assistant:``/``A:`` turn dump."""
    rows = db.conn.execute(
        """
        SELECT * FROM memory_items
        WHERE status = 'active'
          AND memory_type = 'knowledge'
          AND importance = 5
          AND summary LIKE 'User: %'
          AND (summary LIKE '%Assistant:%' OR summary LIKE '%' || char(10) || char(10) || 'A:%')
        ORDER BY created_at ASC
        """
    ).fetchall()
    return [db._row_to_item(r) for r in rows]


def _archive_all(db: Database, vector: VectorStore, items: Iterable[MemoryItem]) -> tuple[int, int]:
    archived = 0
    vector_removed = 0
    for item in items:
        db.archive_item(item.id)
        archived += 1
        try:
            vector.delete(item.id)
            vector_removed += 1
        except Exception:
            pass
        try:
            vector.delete_raw(item.id)
        except Exception:
            pass
    return archived, vector_removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Actually archive (default is dry-run).")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of rows affected.")
    args = parser.parse_args()

    cfg = load_config()
    db = Database(path=cfg.db_path)
    items = _candidates(db)
    if args.limit:
        items = items[: args.limit]

    print(f"Candidates: {len(items)} raw-dump memories")
    for item in items[:5]:
        preview = item.summary[:120].replace("\n", " ")
        print(f"  [{item.id[:8]}] imp={item.importance} {preview}...")
    if len(items) > 5:
        print(f"  ... and {len(items) - 5} more")

    if not args.confirm:
        print("\nDRY RUN — pass --confirm to archive.")
        return

    vector = VectorStore(path=cfg.chroma_path)
    archived, vector_removed = _archive_all(db, vector, items)
    print(f"\nArchived {archived} memory rows; removed {vector_removed} from Chroma.")
    print("Undo: UPDATE memory_items SET status='active' WHERE status='archived' AND summary LIKE 'User: %';")


if __name__ == "__main__":
    main()
