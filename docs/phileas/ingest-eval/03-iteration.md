# 03 — Iteration: fix ingest, measure each step

**Blocked by:** [02 — Eval harness](./02-eval-harness.md)
**Created:** 2026-04-21

## Goal

Make targeted changes to the extraction pipeline. Re-run the eval harness after each change. Merge only the changes that move the metrics.

## Iteration backlog

Each item below is a candidate sub-issue. Sequence matters — earlier items unblock later ones or reduce confounds.

### Iter 1 — Drop the fallback-as-memory path

**Problem:** `src/phileas/llm/extraction.py:22-31` — when the LLM is unavailable or the JSON parse fails, the fallback wraps the *entire transcript* as one `knowledge` memory. That's where most `User: continue smoke test` entries come from.

**Change:** Make `_fallback()` return `[]` (zero memories is a valid outcome). Log the failure + transcript length so we can alert on it if the LLM starts failing silently.

**Expected signal:** noise rate drops sharply in the eval (the zero-expected stratum stops producing output). Precision improves because the unprocessed chunks were always counted as false positives.

**Files:** `src/phileas/llm/extraction.py`, `src/phileas/daemon.py` (logging)

### Iter 2 — Enforce English in the extraction prompt

**Problem:** `src/phileas/llm/prompts/extraction.txt` has no language rule, so Vietnamese summaries pass through at ingest. The rule exists only in `fact_derivation.txt:35`, which runs during consolidation — too late.

**Change:** Add "Always write summaries in English. Translate if the source is in another language." to `extraction.txt`. Add a similar rule to `entity_extraction.txt` for entity names where translation is appropriate (but preserve proper nouns).

**Expected signal:** `non-English rate` metric goes to 0.

**Files:** `src/phileas/llm/prompts/extraction.txt`, `src/phileas/llm/prompts/entity_extraction.txt`

### Iter 3 — Add a cheap gating stage

**Problem:** The pipeline always calls the full extraction prompt even on obvious non-memories (continuation prompts, tool echoes, system-reminders). That's expensive and invites false positives.

**Change:** Two-stage:
1. **Gate** — a small prompt asking "Does this text contain anything memory-worthy about the user's life, preferences, goals, or domain knowledge? Answer yes/no with one sentence of justification." Fast model, ≤50 tokens out.
2. **Extract** — run the existing extraction only when the gate returns yes.

Tune the gate to err toward skipping: false negatives on the gate are cheaper than false positives through the full pipeline.

**Expected signal:** noise rate drops further; latency per transcript drops on low-signal inputs; token cost drops.

**Files:** new `src/phileas/llm/gate.py`, update `src/phileas/daemon.py:295`.

### Iter 4 — Migrate raw text from Chroma to SQLite FTS5

**Problem:** `src/phileas/daemon.py:302` passes `raw_text=text` (full transcript) to **every** memory extracted from that transcript. N memories → N copies in the `raw_memories` ChromaDB collection (`src/phileas/vector.py:147`). This duplicates storage and multiplies recall hits.

**Change:**
- Create `raw_sessions` SQLite table keyed by `session_id` (+ `created_at`, `text`), with an FTS5 virtual table over `text`.
- Update the Stop hook / daemon `ingest` endpoint to write the transcript **once** per session into `raw_sessions` before kicking extraction.
- Drop the `raw_memories` ChromaDB collection and the `add_raw` / `search_raw` code paths in `vector.py`.
- Update recall Path 5 (`src/phileas/engine.py:506-514`) to query `raw_sessions` via FTS5 keyword search instead of vector search. Keyword fits the raw use case better anyway (verbatim recall of names, phrases).
- Remove the `raw_text` column from `memory_items` (migration), since raw now lives session-side. Drop `raw_text` from `engine.memorize()` args.

**Expected signal:** smaller `~/.phileas/chroma/` directory, faster recall Path 5, no more "5 copies of the same transcript surfacing in results."

**Files:** `src/phileas/db.py` (schema + migration), `src/phileas/vector.py` (remove raw collection), `src/phileas/engine.py` (Path 5 rewrite), `src/phileas/hooks/memorize.py`, `src/phileas/daemon.py`.

**Migration note:** existing `memory_items.raw_text` data can be discarded — it's duplicated junk. But confirm with user before dropping the column.

### Iter 5 (optional) — Importance floor at ingest

**Problem:** Even after the above, the pipeline stores `importance=5` memories that rarely surface. They pile up and dilute recall over time.

**Change:** In `daemon.py` ingest, drop memories with `importance < 5` on write. Leave `update()` / `reinforce` unaffected (those paths might legitimately lower importance).

**Expected signal:** memory count stops growing linearly with session count; recall hit quality rises. This is the most invasive item — run it last and lean on the eval to decide.

**Files:** `src/phileas/daemon.py`.

## Definition of done

- Each merged iteration has an eval run recorded in `tests/eval/runs/` showing the metric delta against baseline.
- Noise rate < 10%, precision ≥ 70% on the gold set.
- No Vietnamese summaries in memories created after the English-enforcement change.
- `~/.phileas/chroma/raw_memories` collection is gone.
- `memory_items` table either has no `raw_text` column or only has it populated for legacy rows.

## What we're explicitly not doing

- Changing the 5 memory types or their meanings.
- Redesigning recall scoring.
- Rewriting `reflect()` or `consolidate()`.
- Moving off SQLite or ChromaDB — the issue is what we write, not where.
