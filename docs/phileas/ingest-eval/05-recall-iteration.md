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

## Iteration 2 — expanded gold set + casing fix (2026-04-21, afternoon)

Added 4 gold-recall cases targeting the non-cross-lingual PHI-11
dimensions:

- `casing-drift-01` — query uses lowercase `phileas`, entity stored as
  `Phileas`. Kuzu's `CONTAINS` is case-sensitive; pre-fix the case
  **missed** with 0 results. Fix in `graph.search_nodes` (wrap both
  sides in `lower()`) lands HIT at rank 1.
- `type-confusion-01` — same entity stored under two types
  (`Project` + `Tool`). Each has an ABOUT memory. `search_nodes` finds
  both; recall returns the union. Hits at rank 1 out of the box.
- `rel-bridge-01` — query names entity B, memories are on entity A,
  A↔B via REL edge. Path 3 entity traversal bridges. HIT at rank 1.
- `nickname-alias-en-01` — English alias positive control. HIT.

Run `expansion-fixed` → 6/7 HIT at rank 1. Only `alias-gap-chi-01`
"hits" by accident (embedder tone-bias, known).

## Iteration 3 — pronoun / referent disambiguation (2026-04-21, evening)

Trace of the live `7c2c08fd-d019-4212-a00e-233d90a62112.jsonl` session
showed the real `chị → phuongtq` failure survives all iteration-1/2
fixes. Root cause: the query `"đố biết chị ở trên mình nhắc đến là ai"`
contains no literal string that matches any entity name or alias, and
semantic search drifts to *tone-similar* memories (Chiennv rejection
at score 0.86, not phuongtq).

### Architectural change

`engine.recall` stage 0 now runs a single `analyze_query` LLM call that
returns `(queries, needs_referent_resolution, pronoun_hints)`. When the
query is flagged ambiguous, a second `resolve_referents` call picks
the likely Person entity from the top ~15 candidates (ranked by
ABOUT-edge count + recency + most-recent-summary as vibe proxy).

Resolved entity names flow into path 3's existing
`_add_memories_for_entity` loop, so scoring/MMR/rerank stay unchanged.
Runs under ~2 LLM calls per recall when ambiguous; 1 call otherwise;
0 calls when `_skip_llm=True` (eval harness, tests).

### New artefacts

- `src/phileas/llm/query_rewrite.py::analyze_query` (richer replacement
  for `rewrite_query`; legacy function kept as thin wrapper).
- `src/phileas/llm/referent_resolve.py` — `resolve_referents` +
  `build_person_candidates`.
- `src/phileas/llm/prompts/query_analyze.txt`,
  `src/phileas/llm/prompts/referent_resolve.txt`.
- `src/phileas/graph.py::get_top_entities_by_type` — type-filtered
  top-N by edge count.
- `tests/test_referent_resolve.py` — 9 unit tests (mocked LLM).
- `tests/test_recall_referent_integration.py` — 3 integration tests
  exercising the full recall path with mocked LLM, proves phuongtq
  surfaces for an ambiguous `chị` query against a seeded graph.

### Follow-up — MCP recall was hard-wired to skip the LLM

After shipping the resolver, testing the real transcript through MCP
still failed. Root cause: `src/phileas/server.py:175` passed
`_skip_llm=True` on every MCP recall call (commit `53cc816` —
"MCP server skips LLM operations (Claude Code is the brain)"). That
pre-dated the new pipeline and silently disabled both query-rewrite
*and* the new referent resolver for every live recall. Dropped the
flag; `engine.recall` still gates internally on `self.llm.available`,
so keyless MCP environments are unaffected.

### Follow-up — probe script + scoring fix (2026-04-21 late evening)

Giao reported the MCP-routed query still failed after the _skip_llm fix.
Built `scripts/probe_recall.py` to bypass MCP entirely: snapshots
~/.phileas to a tempdir, builds a fresh MemoryEngine against the copy
with the LLM enabled, and prints every stage's decision. Ran against
the real query `đố biết chị ở trên mình nhắc đến là ai`.

Three issues surfaced in sequence, each fixed in this iteration:

1. **Resolver picked wrong entity** — with one summary per candidate,
   the LLM chose `Phương` + `Ngan` over `phuongtq` because "phuongtq"
   reads as an opaque handle next to the Vietnamese-looking alternatives.
   Fix: `recent_summary_per_entity=1 → 3`, reformatted the candidate
   block as multi-line bullets, and added an explicit prompt note that
   handle-shaped names encode first name + initials.
2. **Referent boost lost to CE normalisation** — cross-encoder scores
   are min-max normalised to [0, 1] so the best CE hit always reaches
   1.0 regardless of absolute quality. Raised `referent_ids` relevance
   floor from 0.85 to 0.95 and removed the `is_referent` flag from
   REL-edge-traversed neighbours so resolving "chị → anhnq" doesn't
   pull every coworker.
3. **Importance/reinforcement weight still buried the right memory** —
   compute_score blends five signals, and a high-importance unrelated
   memory at CE=1.0 can still beat a referent memory at relevance=0.95.
   Final sort now prioritises referent hits by resolver rank
   (1-indexed) before falling back to compute_score — so the LLM's
   first pick always tops the list.

After all three: the query surfaces a phuongtq memory at rank 1 on the
live graph. Another phuongtq hit lands at rank 5 for context.

### Open items

- The redacted real-transcript gold case (PHI-13) still hasn't landed;
  integration tests cover the code path but don't pin the real case.
- `build_person_candidates` is O(15 × memory_count_per_person) SQLite
  round-trips. Cheap for now, worth batching if we expand beyond
  Person entity types.
- Claude Code session still needs to relaunch so its MCP subprocess
  picks up the new server.py. A mere daemon restart isn't enough —
  the MCP stdio process is owned by the Claude Code session.
- Duplicate entity problem — `phuongtq` (54 memories) and `Phương`
  (9 memories) are the same real person split across two graph nodes.
  Not fixed; an entity-dedup backfill is the right follow-up.

## Planned iterations beyond #3

1. **Alias backfill pass** — one-shot script that re-runs entity
   extraction over existing memories asking for nicknames/kinship
   aliases explicitly. Useful fallback even with referent resolution
   in place.
2. **Grow gold-recall corpus** — add contextual-pronoun case (prior
   context anchoring a bare "she"), emotion/concept case, and the
   redacted `chị → phuongtq` real-transcript regression.
3. **Instrument the resolver** — log analyze_query + resolve_referents
   outcomes so we can see over time what fraction of recalls get
   flagged ambiguous and how often resolution lands the right entity.

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
