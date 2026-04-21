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

## Iteration 1 — fixes landed 2026-04-21

Commits after the baseline ran (same day):

- `src/phileas/graph.py` — `json.dumps(..., ensure_ascii=False)` on both
  `set_aliases` and `upsert_node` props. Regression test
  `tests/test_graph.py::test_non_ascii_alias_roundtrip`.
- `src/phileas/engine.py` — `re.findall(r"\w+", query)` in path 3 instead
  of `query.split()`. New gold-recall case `alias-present-punct-01` locks
  the `?`-trailing query in.
- `tests/eval/gold-recall/snapshots/alias-gap-chi-01.graph.json` —
  decoys grown from 2 → 24, expected-memory summaries rewritten to drop
  the entity name (so only alias → graph can bridge the query).

Post-fix run
`tests/eval/gold-recall/runs/20260421T122015Z-iso-alias-signal/`:

| case | result | rank |
| -- | -- | --: |
| `alias-gap-chi-01` | HIT (surprising) | 1 |
| `alias-present-01` | HIT (expected now) | 1 |
| `alias-present-punct-01` | HIT (expected now) | 1 |

Finding #1 is closed — `alias-present-01` flipped from MISS → HIT,
confirming VN-alias graph lookup works end-to-end once the JSON
escape stops. Finding #2 is closed — punctuation-tolerant gold case hits.

Finding #3 re-opened as a new observation: even with 24 unrelated
decoys and zero mention of `nga-nguyen` in the expected summaries,
`alias-gap-chi-01` still retrieves those memories semantically. Root
cause is the embedder's tone/topic bias (personal-narrative queries
map closer to personal-narrative summaries than to
technical/diary summaries), not anything VN-specific. See open
question below.

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

1. ~~**VN alias encoding fix**~~ — shipped 2026-04-21 (finding #1).
2. ~~**Query tokeniser cleanup**~~ — shipped 2026-04-21 (finding #2).
3. **Grow gold-recall corpus** — add the real redacted `chị → phuongtq`
   case (see PHI-13), plus cases that isolate each of the five PHI-11
   dimensions in its own signal. Target ~10 cases.
4. **Alias backfill pass** — one-shot script that re-runs entity
   extraction over existing memories with a prompt explicitly asking for
   kinship / nickname aliases, and merges results into
   `Entity.aliases`. Gate by post-fix eval.
5. **Cross-lingual entity normalization** — when a VN-named memory
   references an EN-named entity (or vice versa), propose merge via a
   deterministic rule (shared ABOUT neighbours + name-edit-distance).
   Exploratory; may drop if alias coverage is enough.

## Closed — VN input deprecated on 2026-04-21 19:26

Giao committed to English-only for Claude Code sessions. Forward ingest
is guaranteed English, so neither the translate-on-ingest option nor a
multilingual embedder swap is needed. `all-MiniLM-L6-v2` stays.

Scope implications:

- The cross-lingual dimension from `04-recall-graph-eval.md` moves from
  *primary metric* to *regression coverage for historical VN memories*.
- The `chị → phuongtq` case (PHI-13) lands once as a read-path regression
  for data already in the graph, then stops accumulating siblings.
- Iteration priorities 4 (alias backfill) and 5 (cross-lingual
  normalization) collapse into a single "backfill kinship aliases on
  existing VN-era entities" task, gated only by the cost of being
  wrong on historical retrieval.

The two fixes from Iteration 1 (VN alias JSON escape, query tokenizer)
still land — both affect any non-ASCII or punctuated input, not just VN.

## Out of scope

- Rerank changes. Retrieval comes first; reranking is downstream.
- Query rewrite / LLM-mediated understanding. The harness runs with
  `_skip_llm=True` so rewrites don't pollute signal; any LLM-layer work
  gets its own harness.
- Schema changes to `Entity` / `ABOUT` / `REL`. Keep structural fixes
  in data, not in shape.
