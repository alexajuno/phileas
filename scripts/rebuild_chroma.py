#!/usr/bin/env python3
"""Rebuild ChromaDB index from SQLite data.

ChromaDB is a derived index — SQLite is the source of truth.
This script deletes and recreates the ChromaDB collections,
re-embedding all active memories.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path.home() / ".phileas" / "memory.db"
CHROMA_PATH = Path.home() / ".phileas" / "chroma"
BATCH_SIZE = 100


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    # Read all active memories from SQLite
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, summary, memory_type, raw_text FROM memory_items WHERE status = 'active'"
    ).fetchall()
    conn.close()

    print(f"Found {len(rows)} active memories in SQLite.")
    raw_count = sum(1 for r in rows if r["raw_text"])
    print(f"  {raw_count} have raw_text.")

    # Delete and recreate ChromaDB
    if CHROMA_PATH.exists():
        print(f"Deleting ChromaDB at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    import chromadb

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_or_create_collection(name="memories", metadata={"hnsw:space": "cosine"})
    raw_col = client.get_or_create_collection(name="raw_memories", metadata={"hnsw:space": "cosine"})

    # Re-add summaries in batches
    print("Re-embedding summaries...")
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        ids = [r["id"] for r in batch]
        docs = [r["summary"] for r in batch]
        metas = [{"memory_type": r["memory_type"]} for r in batch]
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        print(f"  Summaries: {i + len(batch)}/{len(rows)}")

    # Re-add raw_texts in batches
    raw_rows = [r for r in rows if r["raw_text"]]
    if raw_rows:
        print("Re-embedding raw_texts...")
        for i in range(0, len(raw_rows), BATCH_SIZE):
            batch = raw_rows[i : i + BATCH_SIZE]
            ids = [r["id"] for r in batch]
            docs = [r["raw_text"] for r in batch]
            metas = [{"memory_type": r["memory_type"]} for r in batch]
            raw_col.upsert(ids=ids, documents=docs, metadatas=metas)
            print(f"  Raw texts: {i + len(batch)}/{len(raw_rows)}")

    print(f"\nDone. Collections: memories={col.count()}, raw_memories={raw_col.count()}")


if __name__ == "__main__":
    main()
