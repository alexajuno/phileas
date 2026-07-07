# Recall eval

A repeatable loop for telling whether a recall change helped: a labeled gold set
over a fixed Mara corpus, run through two configs against the real model, scored
on recall@k / MRR / nDCG and resource cost, with the gate that cut any missing
gold memory named in the diff. Numbers, not vibes.

Like the rest of `evals/`, this is local and not committed. The only production
code it relies on is the recall trace seam (`phileas.recall_trace`, exercised by
`tests/test_recall_trace.py`).

## Pieces

- `seed.py` — seeds the grown Mara corpus (the sibling `coldstart` eval's sessions + extractions) into the isolated `mara-eval` profile at `~/.phileas-mara-eval`. Run once, and again whenever the corpus changes (`--reset` rebuilds).
- `goldset.json` — hand-curated labeled queries across the query taxonomy (entity, entity_nonmerge, event, temporal, opinion, stance_evolution, decay_noise). Each relevant/excluded memory is a distinctive **content substring**; the runner resolves it to one id, so a re-seed (which mints fresh ids) never rots the gold set.
- `configs.json` — named recall configs = the `PHILEAS_*` knobs `recall()` reads per call. `baseline` is the production default (rrf / rank / ratio / index).
- `metrics.py` — recall@k, MRR, nDCG@k (graded), hit@k, intrusion@1, and cost summaries (mean / p50 / p90). Surface-agnostic: every function takes `(results, gold...)`.
- `ab.py` — the runner. Loads the fixture and gold set, asserts the real reranker, freezes the store, runs every query through config A and B reading each trace via `recall_trace.record()`, and prints a per-query table, a per-`query_type` scorecard, and an A/B diff.
- `_engine.py` — isolated `mara-eval` engine builder; refuses to touch the real `~/.phileas`.
- `PROVENANCE.md` — what the fixture is and how to rebuild it.

## Run

Via the project venv python:

1. Seed once: run `seed.py` (it prints the built store's shape).
2. A/B: run `ab.py --a baseline --b <other>` (e.g. `no_rerank`, `floor_fusion`, `gap_cut`, `legacy_path3`). Flags: `--k` (top_k for @k metrics, default 10), `--out <dir>` (write the full results JSON).

The runner prints `RERANKER: loaded …` (proof it ran the real model, not a stub)
and `NOISE FLOOR 0.000000` (config A run twice is identical, so any A/B delta is
signal, not variance). A non-zero noise floor means the store was not frozen or
env bled across configs — treat the comparison as confounded.

## Reading the output

- **Per-query table:** `r@k mrr ndcg`, candidate pool size, returned count, output chars, latency, and notes. `miss:<gate>` names the gate that cut a gold memory the config failed to surface (`cosine_entry` / `graph_entity` / `relevance_cut` / `not_gathered`). `LEAK×n` flags an adversarial `excluded` memory that wrongly surfaced (a non-merge or wrong-sense failure).
- **Scorecard:** means overall and by `query_type`, plus latency and output-chars cost. "On which query types is it better or worse" reads straight off this.
- **A/B diff:** per-metric and per-type deltas (B − A), then the regressions — each gold memory A surfaced that B dropped, annotated with the gate B cut it at. That points straight at what to tune.

## Extending

- New query: add an entry to `goldset.json` (relevant/excluded as content substrings); the runner validates uniqueness.
- New config: add a named block to `configs.json`.
- Re-curate after corpus changes: `fixture_version` in `goldset.json` is the corpus fingerprint; the runner warns if it drifts.
