# Roll-up retrieval benchmark

A deterministic test set for the roll-up consolidation layer (retrieval-at-scale):
does coverage-gated collapse return one covering summary for a broad query without
cannibalizing the specific episode behind a narrow query?

## Run

```sh
.venv/bin/python evals/rollup/bench.py             # full scale sweep
.venv/bin/python evals/rollup/bench.py diagnose 30 # per-topic narrow-query breakdown at E=30
```

## What it measures

A controlled corpus: 6 topics, each with E episodes (swept 5 / 15 / 30 / 60) plus
one gist that rolls them up, plus 10 distractor topics for noise. Two query
classes with known answers:

- **Aggregation** (query = the topic): the answer is the gist. Metric: gist hit@1,
  hit@3, MRR. Higher is better.
- **Specific** (query = one facet's content words): the answer is a concrete
  episode; the gist is a wrong competitor. Metric: intrusion@1, the share of
  narrow queries where the gist outranks its own episodes. Lower is better.

`recall()` is deterministic at a fixed store, and the harness freezes recall's
write side-effect (`record_retrieval`), so the baseline-vs-baseline noise floor is
exactly zero. Any off-vs-on delta is the mechanism, not variance (the harness
prints the floor to prove it).

## Result (2026-06-18, gate 0.50)

Coverage-gated collapse recovers broad recall at every pile size (agg hit@1 = 1.00,
versus the gist buried at MRR 0.04 with collapse off) while adding zero
cannibalization (intrusion cost +0.00 at E = 15 / 30 / 60). The aggregation benefit
grows with the pile (+0.32 at E=5 up to +0.96 at E=60), the retrieval-at-scale
signature.

This replaced an earlier fixed-weight ranking lift, which drove intrusion to 1.00:
it boosted the gist on every query, broad or narrow.

## Known blind spot

The corpus has lexically clean facets: each facet's content words appear only in
its own episodes, so a narrow content-word query gathers only its facet (coverage
~0.20, well under the gate). Real memories may share vocabulary across sub-themes,
where a narrow query could pull in siblings, raise coverage, and trip collapse.
The next iteration should add a corpus with lexically-overlapping facets to test
that case.
