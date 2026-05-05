# Vietnamese → English Memory Migration

**Date:** 2026-04-09
**Status:** In progress

## Problem

ChromaDB uses `all-MiniLM-L6-v2` for embeddings — an English-centric model. Vietnamese text produces poor embeddings, causing recall failures:

- Vietnamese query vs Vietnamese document: similarity ~0.40-0.49
- `similarity_floor = 0.5` filters out all Vietnamese memories
- Memories are stored successfully (SQLite + ChromaDB) but invisible to recall

## Evidence

```
English query "day off work April 9 2026":
  - English memory (TEST): sim = 0.64 ✓ (found)
  - Vietnamese memories: not in top 10

Vietnamese query "xin nghỉ làm April 9":
  - Vietnamese memories: sim = 0.40-0.49 ✗ (below floor)
```

## Solution

Migrate all Vietnamese summaries and raw_text to English. This is the correct fix because:

1. Embedding model works well with English
2. Keyword search becomes language-consistent
3. Queries (often mixed English/Vietnamese) match better against English summaries

## Migration Script

`scripts/migrate_vi_to_en.py`

- Detects Vietnamese text via diacritics regex
- Translates via `claude --model haiku` in batches of 5
- Updates both SQLite (summary + raw_text) and ChromaDB (re-embeds)
- Preserves @mentions, dates, proper nouns

### Usage

```bash
cd ~/phileas
.venv/bin/python scripts/migrate_vi_to_en.py
```

### Scope

- 104 Vietnamese memories out of 593 active (as of 2026-04-09)
- Affects: `memory_items.summary`, `memory_items.raw_text`, ChromaDB `memories` + `raw_memories` collections

## Future Policy

All new memories should be stored in English. Update the Phileas skill instructions to enforce this.
