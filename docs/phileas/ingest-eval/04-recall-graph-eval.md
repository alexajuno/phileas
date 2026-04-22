# 04 — Recall / graph eval (follow-up)

**Created:** 2026-04-21
**Parent project:** [Phileas: ingest noise reduction](https://linear.app/alexajuno/project/phileas-ingest-noise-reduction-08c5b9782afe)
**Status:** planning (not started)

## Why this exists

The extraction eval (PHI-7 / `02-eval-harness.md`) measures whether `extract_memories(client, text)` produces the right summaries, types, importances, and graph annotations *per transcript*. It does **not** measure whether recall finds the right memory when asked.

A concrete failure surfaced during gold-set labeling:

> User says (Vietnamese): *"tự dưng thi thoảng cứ nhìn trúng mặt chị"* — "sometimes I suddenly catch sight of her face"
> User asks: *"do you know who I'm mentioning?"*
> Claude hedges; proposes a wrong person; does not resolve to `phuongtq`.

Why it failed, verified against the live graph on 2026-04-21:

1. `phuongtq.aliases = []` — no nickname/kinship terms populated, so `chị` → `phuongtq` is unreachable.
2. `phuongtq` has only 3 REL edges (`KNOWS Giao`, `WORKS_AT Ownego`, reverse `KNOWS`). No emotional-context edges.
3. Zero memories link `phuongtq` to any `love/reject/sad` concept entity — those entities don't exist in the graph.
4. The recall path that would have helped (alias → entity → memories about that entity) has no data to traverse.

## What this eval measures

Given:
- A **recall query** (text + optional prior-conversation context).
- A **graph state snapshot** (the entities and relationships Phileas knows about at query time).

Measure: does the right memory / entity surface in the top-K results?

### Dimensions

1. **Alias resolution** — nicknames, kinship terms (`chị`, `em`), first-name variants, typos. Does the query's surface form map to the right entity?
2. **Cross-lingual referents** — Vietnamese pronouns + emotional context → English-named entity (and vice versa).
3. **Entity-type canonicalization** — same entity known as `Project` in some memories and `Tool` in others should still resolve.
4. **Contextual disambiguation** — when the query is ambiguous (e.g., "she"), does prior conversation context anchor the right referent?
5. **Emotion/concept linking** — if the query mentions "rejection" or "love songs," do memories that mention the *same event* (but phrased differently) surface?

## Pipeline under test

`src/phileas/engine.recall()` — specifically Paths 2–4 (entity-expand, graph-traversal) since Path 1 (embedding search) is orthogonal.

## Proposed structure

```
tests/eval/gold-recall/
  queries/<id>.yaml          # query text + optional context + expected top-K
  snapshots/<id>.graph.json  # frozen graph state for the query's timeline
  runs/<ts>-<slug>/
    per_query.jsonl
    summary.json
    summary.md
```

Each query YAML:

```yaml
id: chi-to-phuongtq-01
query: "do you know who I'm mentioning?"
prior_context: |
  User mentioned catching sight of 'chị' while hearing love songs; felt sad.
expected_top_k:
  - memory_id: <any memory about phuongtq>
  - entity_name: phuongtq
tolerance: 5   # must appear in top-5
```

## Metrics (informational for baseline, gated after)

- **Top-K entity recall**: expected entity appears in top-K nearest.
- **Alias hit rate**: queries with alias terms where resolution succeeds.
- **Cross-lingual hit rate**: VN-query → EN-named-entity resolution rate.
- **Type-canonicalization success**: resolution rate when the same entity exists with conflicting types in the graph.
- **Zero-result rate**: queries that return nothing when they should return something (critical failure).

## Deliverables

- `tests/eval/gold-recall/` — query + snapshot directory
- `tests/eval/run_recall.py` — runner invoking `engine.recall()` directly
- `tests/eval/match_recall.py` — scoring (top-K, alias hit, etc.)
- `tests/eval/compare_recall.py` — two-run diff
- First baseline run with the `chị → phuongtq` query as case #1
- Follow-up doc `05-recall-iteration.md` for fixes (alias backfill, cross-lingual entity normalization, etc.)

## Out of scope

- Changing the graph schema (keep current `Entity` / `Memory` / `ABOUT` / `REL`).
- Full natural-language query understanding (we're scoring specific probes, not a chat agent).
- Reranking; that's a separate concern downstream of retrieval.

## First case (not yet added)

**`chị → phuongtq`** — the 2026-04-21 failure dissected above. Requires a redacted / synthesized version of the original transcript since the verbatim content is deeply personal. Hold until the harness exists.
