# Temporal-deixis eval

Does the deixis retrieval path actually work? A dateful question ("what did I fix yesterday", "dentist tomorrow") should be answered from the day it names, even when a better-worded memory on another day would outrank it on topic. This eval settles it with numbers: a synthetic day-anchored corpus with engineered off-day decoys, run under `PHILEAS_DEIXIS=scope` and `=off` against the real reranker, scored on recall@k, day-precision, and decoy leakage.

The path under test is engine Path 3d (`resolve_temporal` seeds the resolved day's `Day` node into the gather pool) plus the Stage-2 scope cut (restricts the candidate set to that day before the rerank, engine.py:1334).

## Pieces

- `corpus.json` — the synthetic fixture. Every memory carries an explicit `daily_ref` (so `memorize` links it to a known `Day` node) and a `role`: `target` (the on-day memory a query should surface, phrased in ordinary words), `decoy` (an off-day memory written to be the *stronger* topical match, which scope must beat), `filler` (same-day context), `noise` (unrelated). `ref_date` is the pinned "now" (2026-07-04, a Saturday).
- `goldset.json` — labeled deictic queries. `relevant`/`excluded` are distinctive summary **substrings** resolved to one id against the seeded store, so a re-seed never rots the set. `phrase`/`dates` are the deictic span and the ISO days it must resolve to — a resolver-level check independent of retrieval. `control` queries carry no deixis (dates empty; scope must behave identically to off).
- `seed.py` — memorizes the corpus into the isolated `temporal-eval` profile at `~/.phileas-temporal-eval`. Run once, and again whenever the corpus changes (`--reset` rebuilds).
- `run.py` — the runner. Freezes `engine.date.today()` to `ref_date` around each recall (the one clock read Path 3d makes), so resolution is deterministic whatever day the eval runs on. Runs every query under scope and off, prints a per-query table and a scope-vs-off summary.
- `metrics.py` — recall@k, MRR, plus the two deixis measures: **day-precision** (share of results filed under the resolved day) and **decoy-surfaced** (did the engineered off-day match leak in).
- `_engine.py` — isolated `temporal-eval` engine builder; refuses to touch the real `~/.phileas`. Daemon-free direct GraphStore, so the running daemon's Kuzu lock never blocks it.

## Run

Via the project venv python:

1. Seed once: `python evals/temporal/seed.py` (`--reset` to rebuild).
2. Run: `python evals/temporal/run.py` (`--k`, default 10; `--out <dir>` writes the results JSON).

The runner prints `RERANKER: loaded …` (proof it ran the real model) and `NOISE FLOOR 0` (scope run twice is identical, so any scope/off delta is signal, not variance).

## Reading the output

Two rows per query, `scope` then `off`:

- **ret** — how many memories came back.
- **r@k** — recall@k of the day's relevant memories.
- **dayP** — day-precision. Scope should hold this near 1.0 (the answer is that day's page); off drops as off-day matches interleave. `-` for control (no resolved day).
- **leak** — 1 if the off-day decoy surfaced. Scope should be 0; off is expected to leak.

Notes flag a **resolver mismatch** (`temporal.py` disagreed with the gold's expected dates — a resolver bug or a stale gold entry) or a **control break** (a no-deixis query returned different results under scope vs off, which should never happen).

The summary reads straight off: scope should show high day-precision and zero leaks; off should leak the decoys and score lower day-precision — the measured separation that says the path earns its place.

## Extending

- New scenario: add the target plus a topically-stronger off-day decoy to `corpus.json`, then a query to `goldset.json` (relevant/excluded as unique substrings). Re-seed.
- Moving `ref_date`: pick days relative to the new reference and update both files' `ref_date` in lockstep.
