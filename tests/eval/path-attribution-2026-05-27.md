# Recall paths: when each one earns its keep

Investigation 2026-05-27 into why `engine.recall` averages ~26s on hot
queries despite path-3 cosine work being vectorized. Adds per-sub-path
attribution to `recall_traces.extra` (commit `0cf0a1d`) and replays a
curated query set through the live daemon to measure unique-contribution
per path.

## TL;DR

| regime (n=10 each side) | avg latency | path3 cost | path3b cost | path4 cost | path3 unique | path3b unique | path4 unique |
|---|---|---|---|---|---|---|---|
| Entity-rich (Path 3 fires) | **25.7s** | <0.2s | 9.5s | 15.3s | 6.1 | 4.2 | **0.6** |
| Entity-less (Path 3 empty) | **1.6s** | <0.1s | 0s | 0.3s | 0.0 | 0.0 | **11.6** |

The two regimes are doing different jobs.

- **Entity-less queries lean entirely on Path 4.** It's the dominant
  unique-result producer (~12 of the top-30) and it's cheap (~350ms)
  because Path 3 produced no `graph_ids` for Path 3b to chew through.
- **Entity-rich queries pay 25s for what Path 3 already covered.**
  Path 3b and Path 4 grind through 60–1200-candidate pools producing
  ~4 and ~0.6 unique results respectively, while Path 3 alone gave
  ~6 unique.

The structural problem isn't that any path is useless — it's that
Path 3b and Path 4 scale O(pool × entities × neighbours) without
checking whether Path 3 already saturated the pool.

## Methodology

`tests/eval/path_attribution.py` replays a query list through the daemon
and reads back the new attribution columns:

- `path3_count`, `path3b_count`, `path4_count` from `recall_traces.extra`
  — how many candidates each sub-path contributed (before scoring).
- `result_gather_histogram`, `result_unique_path_counts` from same
  — how many made the final top-K, and how many came in via that path
  *alone*.
- `stage_timings_json.graph_path3 / graph_path3b_pivot / graph_path4_bridge`
  from `recall_events` — ms spent in each block.

19-query default mix: 6 entity-rich (`Hanoi heat`, `anhnq dental braces`,
`Ownego boss minhnt staging`, `phileas recall performance`, `phuongtq`,
`Genshin Impact`) + 13 entity-less (`loneliness`, `feeling stuck at work`,
`remembering childhood`, `recurring patterns`, `nervous system reset`,
`exit arc`, `career ceiling`, `wound frame`, `conditional love`, plus
3 Vietnamese feeling-shape phrases).

"Entity-rich" is defined post-hoc as `path3_count > 0` — i.e. Path 3
found at least one entity name in the query. No external classifier.

## What each path does

```
keyword     SQLite LIKE on the summary text (AND-match on tokens)
semantic    ChromaDB cosine search on memory summaries
path3       Per-token graph.lookup_nodes (exact normalised name match)
              + 1-hop entity↔entity neighbours
path3b      For every memory found via path3, walk its entities and
              fetch all of *their* memories, then their related entities'
              memories. Memory-pivot expansion.
path3c      LLM-resolved referents (pronoun/kinship → entity). Currently
              ~0ms because referent_names is empty in the daemon path.
path4       For every memory in the candidate pool (keyword ∪ semantic
              ∪ path3* ∪ raw_text), walk its entities and fetch
              connected memories. Semantic-to-graph bridge.
raw_text    ChromaDB cosine on a separate "raw verbatim" collection.
event_thread Event-text cosine search → sibling memories extracted
              from the same event.
```

## Why Path 4 earns its keep on entity-less queries

Path 4's premise: query is purely conceptual, has no entity-namable
tokens, so Path 3 finds nothing. Semantic search catches a few
"feeling-shaped" memories. Each of those memories is linked to one or
two entities. Walking those entities surfaces memories that aren't
semantically close to the query but *share context* with the seed.

Worked example from the data:

- Query `"khi nào mình cảm thấy bình yên"` — no proper nouns.
- Path 3 finds nothing.
- Path 4 takes the semantic seeds → walks their entities → returns 26 of
  30 final results via that bridge alone, in 168ms.

Without Path 4 the answer is "8 raw_text hits and that's it."

## Why Path 3b and Path 4 are wasteful on entity-rich queries

Two compounding mechanisms:

1. **Seed-set overlap.** Path 3b seeds from `graph_ids`. Path 4 seeds
   from `candidates.keys()`, which is a superset. For every seed that's
   in both — which is every graph-sourced memory — Path 4 produces the
   exact same bridged neighbours Path 3b already produced.

2. **Path 3 already covered the pool.** When a query names an entity
   (`phuongtq`, `anhnq`, `Ownego`), Path 3 directly pulls every memory
   tagged with that entity *and* their 1-hop neighbours. The pool is
   already 60–1200 memories with high entity density. Bridging further
   from those seeds finds entities Path 3 has already pulled — the new
   neighbours score below `relevance_floor` against the original query
   anyway, and MMR drops them as redundant.

Quantitatively, on the 6 entity-rich queries with `path3_count > 100`
(anhnq, Ownego, phileas perf, phuongtq, loneliness, feeling stuck):

- Path 3b cost: 9.5s avg, 4.2 unique results (only 2 of 6 queries
  produced any path3b-unique results at all).
- Path 4 cost: 15.3s avg, 0.6 unique results (1 of 6 produced any).

On entity-rich queries with small Path 3 (≤60 candidates): `Hanoi heat`,
`Genshin Impact`, `career ceiling`. There Path 3b/4 are valuable —
6, 13, 16 unique respectively — and the cost is bearable (Genshin Impact
ran 14s total; career ceiling 19s).

So the threshold isn't entity-rich vs entity-less, it's
**pool-already-saturated vs not**.

## Recommendation

Cap iteration in Path 3b and Path 4 by candidate-pool size. Default
proposal:

```toml
[recall]
path3b_max_seeds = 30   # iterate at most this many graph_ids memories
path4_max_seeds = 30    # iterate at most this many candidates
```

Predicted effect:

- Entity-less queries: no change. Path 4 already had <30 seeds; the cap
  doesn't fire. Latency stays ~1.6s.
- Entity-rich queries with small Path 3: minimal change. Cap doesn't fire
  on `Genshin Impact` (8 seeds), barely on `Hanoi heat` (60 seeds → cap
  to 30; lose ~half of path3b's 6 unique → ~3 unique).
- Entity-rich queries with big Path 3: dramatic. `anhnq` etc. drop from
  ~25s to ~3s (the cosine_full + rerank + filter stages remain), with
  near-zero unique loss because path3b/4 contributed ~0 there anyway.

For Path 4 specifically, ordering the candidate pool so non-`graph_ids`
seeds (keyword, semantic, raw_text) come first preserves Path 4's
designed value — those are exactly the seeds that produce *new* bridges
not covered by Path 3b.

## Pitfalls for future review

- The `result_unique_path_counts` metric under-counts when a memory
  legitimately enters via multiple paths. E.g. `phuongtq` returned 30
  results but every path shows 0 unique — all 30 came in via 2+ paths
  simultaneously. The `result_gather_histogram` is the more honest
  "did this path contribute at all" signal. Cross-reference both.
- The pre-cap 5-query probe (May 27 ~14:00 UTC, traces 5357–5361) gave
  the *wrong* answer because it tested only entity-rich queries. Don't
  measure a path on cases outside its design surface.
- Path 3c (referent resolution) is dormant in the daemon path —
  `referent_names` is never populated. The instrumentation captures
  cost as ~0ms because the loop body never runs.
