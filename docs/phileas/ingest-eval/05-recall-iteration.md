# 05 — Recall iteration (follow-up)

**Created:** 2026-04-21
**Parent project:** [Phileas: ingest noise reduction](https://linear.app/alexajuno/project/phileas-ingest-noise-reduction-08c5b9782afe)
**Blocked by:** [04 — Recall / graph eval harness](./04-recall-graph-eval.md) (PHI-11 — now landed)
**Status:** planning

## Baseline run

`tests/eval/gold-recall/runs/20260421T120234Z-baseline/` — 2 synthetic cases:

| case | tags | result | note |
| -- | -- | -- | -- |
| `alias-gap-chi-01` | alias-resolution, cross-lingual | HIT @ rank 1 | leaks through semantic search on a small decoy corpus |
| `alias-present-01` | alias-resolution, cross-lingual, positive-control | MISS (0 results) | aliases populated but not reachable — see finding #1 |

## Baseline findings

### 1. Non-ASCII aliases are unreachable via graph search (real bug)

`phileas.graph.GraphStore.set_aliases` serialises aliases with plain
`json.dumps(aliases)`, which defaults to `ensure_ascii=True`. Vietnamese
characters become `\uXXXX` escape sequences inside Kuzu, so the
`CONTAINS $q` match in `search_nodes` can never find them.

Concretely, `set_aliases("Person", "lan-vo", ["chị"])` stores
`'["ch\\u1ecb"]'` on the node. A query word `"chị"` then CONTAINS-matches
nothing. This alone explains the 2026-04-21 `chị → phuongtq` failure
regardless of whether `phuongtq.aliases` was empty: even a populated VN
alias wouldn't have helped.

**Fix:** `json.dumps(aliases, ensure_ascii=False)` in
`src/phileas/graph.py:382` (and any other JSON-serialised-then-CONTAINS'd
column — audit `props` at line 339 too). Add a regression test in
`tests/` exercising `search_nodes` for a VN alias round-trip. Then re-run
the eval; `alias-present-01` should flip to HIT.

### 2. Tokeniser keeps trailing punctuation, defeating graph CONTAINS (real bug)

`src/phileas/engine.py:438` does `words = query.split()`, so a query
like `"who is chị?"` splits to `["who", "is", "chị?"]`. `search_nodes`
then looks for `"chị?"` literally in names/aliases — no match. Found
during initial harness bring-up when both cases returned 0 results until
the queries were rewritten without `?`.

**Fix:** strip common punctuation before word-level graph lookup, or
tokenise with `re.findall(r"\w+", query)` in path 3. Low risk,
self-contained. Add a gold-recall case with punctuation afterwards to
lock the fix in.

### 3. Semantic path is too permissive on small corpora (eval-side concern)

`alias-gap-chi-01` hit at rank 1 despite aliases being empty and the
query term `"chị"` not appearing anywhere in memory summaries. With only
5 memories, ChromaDB's default embedder drifted the expected `nga-nguyen`
birthday memory into the top-K on low-signal cosine alone.

This is noise from the synthetic gold set, not a production bug. Real
graphs have hundreds to thousands of memories; the leakage attenuates
naturally. Two mitigations:

- Grow each snapshot's decoy pool to ~30+ memories so `top_k=5` has room
  to fail.
- Lower `tolerance` for alias-resolution-tagged cases to `1` — a true
  alias hit should land at rank 1 or be considered a miss.

## Planned iterations (roughly in this order)

1. **VN alias encoding fix** — ships finding #1 above. Unblocks every
   cross-lingual alias case.
2. **Query tokeniser cleanup** — finding #2. Strip punctuation before
   graph CONTAINS.
3. **Grow gold-recall corpus** — add the real redacted `chị → phuongtq`
   case (see PHI-13), plus decoy padding per finding #3. Target ~10
   cases spanning the five dimensions from `04-recall-graph-eval.md`.
4. **Alias backfill pass** — one-shot script that re-runs entity
   extraction over existing memories with a prompt explicitly asking for
   kinship / nickname aliases, and merges results into
   `Entity.aliases`. Gate by post-fix eval.
5. **Cross-lingual entity normalization** — when a VN-named memory
   references an EN-named entity (or vice versa), propose merge via a
   deterministic rule (shared ABOUT neighbours + name-edit-distance).
   Exploratory; may drop if alias coverage is enough.

## Out of scope

- Rerank changes. Retrieval comes first; reranking is downstream.
- Query rewrite / LLM-mediated understanding. The harness runs with
  `_skip_llm=True` so rewrites don't pollute signal; any LLM-layer work
  gets its own harness.
- Schema changes to `Entity` / `ABOUT` / `REL`. Keep structural fixes
  in data, not in shape.
