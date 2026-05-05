#!/usr/bin/env python3
"""Migrate Vietnamese memories to English summaries + raw_text.

Reads all active memories, detects Vietnamese text, translates via claude CLI,
updates SQLite + re-embeds in ChromaDB.
"""

import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

# Vietnamese-specific character pattern
VI_PATTERN = re.compile(r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", re.IGNORECASE)

DB_PATH = Path.home() / ".phileas" / "memory.db"
CHROMA_PATH = Path.home() / ".phileas" / "chroma"

BATCH_SIZE = 5  # translate N at a time to reduce claude CLI calls


def is_vietnamese(text: str) -> bool:
    """Check if text contains Vietnamese diacritics."""
    if not text:
        return False
    return bool(VI_PATTERN.search(text))


def translate_batch(texts: list[str]) -> list[str]:
    """Translate a batch of Vietnamese texts to English via claude CLI."""
    # Build a numbered list for batch translation
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    prompt = f"""Translate each numbered Vietnamese text below to English.
Keep the same meaning and tone. Keep @mentions, dates, and proper nouns as-is.
Return ONLY a JSON array of strings, one per input, in the same order.
Do not add any explanation.

{numbered}"""

    cmd = ["claude", "-p", "--output-format", "json", "--model", "haiku"]
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=120)

    if result.returncode != 0 and not result.stdout.strip():
        raise RuntimeError(f"claude-cli failed: {result.stderr[:300]}")

    try:
        data = json.loads(result.stdout)
        response_text = data.get("result", result.stdout)
    except json.JSONDecodeError, ValueError:
        response_text = result.stdout.strip()

    # Parse JSON array from response
    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", response_text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    try:
        translations = json.loads(cleaned)
        if isinstance(translations, list) and len(translations) == len(texts):
            return translations
    except json.JSONDecodeError, ValueError:
        pass

    # Fallback: translate one by one
    print("  Batch parse failed, falling back to individual translation...")
    return [translate_single(t) for t in texts]


def translate_single(text: str) -> str:
    """Translate a single Vietnamese text to English."""
    prompt = f"""Translate this Vietnamese text to English. Keep @mentions, dates, and proper nouns as-is.
Return ONLY the translation, nothing else.

{text}"""

    cmd = ["claude", "-p", "--output-format", "json", "--model", "haiku"]
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=60)

    if result.returncode != 0 and not result.stdout.strip():
        raise RuntimeError(f"claude-cli failed: {result.stderr[:300]}")

    try:
        data = json.loads(result.stdout)
        return data.get("result", result.stdout).strip()
    except json.JSONDecodeError, ValueError:
        return result.stdout.strip()


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Fetch all active memories
    rows = conn.execute("SELECT id, summary, raw_text FROM memory_items WHERE status = 'active'").fetchall()

    # Filter to Vietnamese
    to_migrate = []
    for row in rows:
        summary_vi = is_vietnamese(row["summary"])
        raw_vi = is_vietnamese(row["raw_text"] or "")
        if summary_vi or raw_vi:
            to_migrate.append(
                {
                    "id": row["id"],
                    "summary": row["summary"],
                    "raw_text": row["raw_text"],
                    "summary_vi": summary_vi,
                    "raw_vi": raw_vi,
                }
            )

    print(f"Found {len(to_migrate)} Vietnamese memories to migrate out of {len(rows)} total.")

    if not to_migrate:
        print("Nothing to migrate.")
        return

    # Import ChromaDB for re-embedding
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_or_create_collection(name="memories", metadata={"hnsw:space": "cosine"})
    raw_col = client.get_or_create_collection(name="raw_memories", metadata={"hnsw:space": "cosine"})

    migrated = 0
    errors = 0

    # Process in batches
    for batch_start in range(0, len(to_migrate), BATCH_SIZE):
        batch = to_migrate[batch_start : batch_start + BATCH_SIZE]
        print(
            f"\nBatch {batch_start // BATCH_SIZE + 1} ({batch_start + 1}-{batch_start + len(batch)}/{len(to_migrate)})"
        )

        # Collect summaries to translate
        summary_texts = []
        summary_indices = []
        for i, mem in enumerate(batch):
            if mem["summary_vi"]:
                summary_texts.append(mem["summary"])
                summary_indices.append(i)

        # Collect raw_texts to translate
        raw_texts = []
        raw_indices = []
        for i, mem in enumerate(batch):
            if mem["raw_vi"] and mem["raw_text"]:
                raw_texts.append(mem["raw_text"])
                raw_indices.append(i)

        try:
            # Translate summaries
            if summary_texts:
                translated_summaries = translate_batch(summary_texts)
                for idx, translation in zip(summary_indices, translated_summaries):
                    batch[idx]["new_summary"] = translation

            # Translate raw_texts
            if raw_texts:
                translated_raws = translate_batch(raw_texts)
                for idx, translation in zip(raw_indices, translated_raws):
                    batch[idx]["new_raw_text"] = translation

        except Exception as e:
            print(f"  Translation error: {e}")
            errors += len(batch)
            continue

        # Apply updates
        for mem in batch:
            mem_id = mem["id"]
            new_summary = mem.get("new_summary", mem["summary"])
            new_raw = mem.get("new_raw_text", mem["raw_text"])

            # Update SQLite
            conn.execute(
                "UPDATE memory_items SET summary = ?, raw_text = ? WHERE id = ?",
                (new_summary, new_raw, mem_id),
            )

            # Re-embed summary in ChromaDB
            if mem.get("new_summary"):
                col.upsert(ids=[mem_id], documents=[new_summary])

            # Re-embed raw_text in ChromaDB
            if mem.get("new_raw_text"):
                raw_col.upsert(ids=[mem_id], documents=[new_raw])

            short = new_summary[:70]
            print(f"  [{mem_id[:8]}] {short}...")
            migrated += 1

        conn.commit()

    conn.close()
    print(f"\nDone. Migrated: {migrated}, Errors: {errors}")


if __name__ == "__main__":
    main()
