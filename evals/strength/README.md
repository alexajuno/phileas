# Strength influence eval

Answers one question with a number: **how much does the two-strength prior (storage / retrieval / access) actually move what a recall returns?**

This is an influence measurement, not a quality comparison. It needs no gold set and no ground truth, because it asks how far the output moves rather than whether it moved in the right direction.

## What it measures

`recall()` computes the blended score at the very end of the pipeline, after the relevance cut and after MMR has already chosen the result set. So the prior can only permute a set that relevance and diversity already fixed. The eval quantifies that permutation:

- **topN set change** — how often the first N results differ from a relevance-only ranking. `auto_recall` truncates to `PER_QUERY_TOP_K` (6) per query and `MAX_POINTERS` (12) overall, so a reordering past those depths is what turns into a membership change at the surface.
- **any order** — how often the returned order differs at all.
- **tau** — Kendall rank correlation against the relevance-only ordering; 1.0 means the prior changed nothing.
- **per-term marginals** — the same numbers with `storage`, `retrieval`, or `access` zeroed one at a time, so each term's share of the movement is visible.
- **full/uniform** — every memory rebased onto the uniform 1.0 starting strength, to show what dropping the per-type seed does to the prior's reach.
- **adjacent relevance gap** — how far apart neighbouring candidates sit on relevance. The prior can only overturn a pair when its own gap exceeds theirs, so this is what bounds the whole effect.

## Method

Real queries are replayed from a `metrics.db` `recall_traces` table against a frozen copy of a real store. Each query runs through the engine **once**; `score_components` is spied on to capture the `(relevance, storage_strength, days, access_count)` tuple for every selected memory. Every arm is then re-ranked offline from those tuples, so ablating a term costs no extra retrieval.

Replay is read-only: `reinforce=False` keeps the store from mutating under the eval.

## Run

```
.venv/bin/python evals/strength/influence.py \
    --home <frozen-store-copy> \
    --metrics <path/to/metrics.db> \
    --limit 400 \
    --cache runs.json
```

`--home` must point at a **copy**, never a live profile; the builder refuses any home that does not resolve to the path given. `--cache` reuses a previous capture so the analysis can be re-run without touching the engine.

Freeze the copy's `storage_strength` at its as-recorded values before replaying, or opening the store will migrate them:

```
sqlite3 <copy>/memory.db 'PRAGMA user_version = 1'
```

## Reading the result

A high `any order` with a low `top1` means the prior is shuffling the tail while leaving the lead alone. `topN` at or above the typical returned-set size is necessarily 0, since the prior cannot change set membership; only depths below that size carry information.
