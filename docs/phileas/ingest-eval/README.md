# Ingest noise reduction: eval-driven redesign

**Status:** planning
**Created:** 2026-04-21
**Owner:** Giao

## Problem

Since the Stop-hook → daemon ingest pipeline shipped (commits `c146e76`, `cd62286`), the memory store is dominated by low-value `knowledge` entries — raw `User:` prompt echoes, Vietnamese chitchat, system-reminder dumps, skill file bodies. Sample from `~/.phileas/memory.db` on 2026-04-20:

```
knowledge = 160 (36% of today's ~444 memories)
```

Top-accessed entries with `imp=7` are genuinely useful (`"ModelProvider.php has USD-based cost fields..."`). The bulk at `imp=5, access=1-3` are noise — most dilute recall rather than help it.

## Root causes (traced in code, not speculated)

1. **Fallback-as-memory** — `src/phileas/llm/extraction.py:22-31`: when the LLM is unavailable or returns malformed JSON, the fallback wraps the **entire transcript** as one `knowledge` memory with `summary == raw_text == text`. Zero-memories is not a valid outcome.
2. **No gating** — extraction always returns ≥1 memory. There's no "is this text memory-worthy at all?" gate before extraction.
3. **Raw duplication** — `src/phileas/daemon.py:302` passes `raw_text=text` (the full transcript) to **every** memory extracted from it. One transcript producing 5 memories creates 5 copies of the full transcript in the `raw_memories` ChromaDB collection — all retrievable by Path 5 in recall (`src/phileas/engine.py:506-514`).
4. **English rule missing from primary extraction** — `src/phileas/llm/prompts/fact_derivation.txt:35` says "Always write in English," but `extraction.txt` has no such rule, so Vietnamese raw text bleeds through at ingest.

## Why we can't just refactor

Every design discussion without measurement is vibes. We need to know the current precision/recall/noise rate on a fixed input set so changes are comparable. Research papers (mem0, MemGPT, etc.) inform priors; they don't drop-in solve for our input distribution (Claude Code transcripts, mixed VN/EN, tool output, system reminders).

## Method: eval-driven iteration

Three phases, each a separate Linear issue and doc:

1. **[01 — Gold set](./01-gold-set.md)** — hand-label ~40 real transcripts sampled from `~/.phileas/ingested-sessions.json`. For each, write the expected memory outputs (including zero). This is the ground truth.
2. **[02 — Eval harness](./02-eval-harness.md)** — replay the gold set through the extraction pipeline, compute precision/recall/noise rate per memory type. Baseline the current system before changing anything.
3. **[03 — Iteration](./03-iteration.md)** — targeted changes driven by the eval: drop the fallback-as-memory path, add a cheap gating stage, enforce English in the extraction prompt, migrate raw text from `raw_memories` Chroma collection to a SQLite FTS5 table.

## Non-goals

- Redesigning recall (separate concern — we're fixing *what gets stored*).
- Changing the 5 memory types or their hot-set weighting.
- Removing the Stop hook or moving to sync ingest.

## Success signal

After iteration, against the same gold set:
- Noise rate (memories produced where gold says zero) drops from current baseline (TBD) to <10%.
- Precision (memories matching gold) ≥ 70%.
- No Vietnamese summary strings in new memories.
- Full transcript stored once per session (SQLite FTS5), not N-times (Chroma).
