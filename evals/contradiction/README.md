# Contradiction detection eval

Does the contradiction probe's similarity band actually separate "genuine conflict" from "unrelated" from "near-duplicate restatement"? That gate is the only empirical part of the feature (resolution and read-back are deterministic graph writes, covered by `tests/test_contradiction.py`), and whether it's calibrated depends entirely on the embedding model's geometry. This measures it on a labeled set instead of guessing.

Like the rest of `evals/`, it's local and not committed. It drives the real `VectorStore` + `GraphStore` + `MemoryEngine` — the same path `memorize()` takes in production.

## Pieces

- `goldset.json` — hand-curated `(first, second)` pairs across four categories: `conflict` (same subject, incompatible value), `unrelated`, `duplicate` (near-verbatim restatement), and `related_compatible` (same topic, both true, no conflict). `expect_flag` is what a perfect "genuine conflict only" detector should do; `category` drives the per-category breakdown. Each case also carries `*_entities` and `*_rel` annotations (the structured input a competent agent would attach on memorize) for the structured and co-subject detectors.
- `bench.py` — the band-calibration runner. For each pair: seed `first`, measure the raw nearest-neighbour cosine similarity of `second` (band-independent, via `vector.search`), then run the real `memorize()` probe and record whether it flagged. Each pair runs in its own throwaway temp store so they can't cross-contaminate.
- `detectors.py` — the detection approaches as comparable binary detectors: `cosine_band` (today's probe), `cosine_widened`, `cosubject` (any / topic), `structured` (functional-edge object swap), and `nli` (an NLI cross-encoder, same pattern as `reranker.py`).
- `benchmark.py` — runs every detector and the composites over the goldset and prints precision / recall / F1, a per-category flag breakdown, and the notable cases that separate the approaches. This answers "which approach is best."

## Run

Via the project venv python, run `bench.py` for band calibration or `benchmark.py` for the approach comparison (both take an optional `--json out.json`). First run loads the embedding model; `benchmark.py` also loads an NLI model. The runners read the live `CONTRADICTION_SIM_FLOOR` / `CONTRADICTION_SIM_CEILING` from `phileas.engine`, so the band in the report is always the real one.

## Reading the output

- **Per-case table** — each pair's measured similarity, which band it lands in (`below` / `in-band` / `above`), whether it flagged, and whether that matches `expect_flag`. Sorted by similarity so the floor/ceiling boundaries are easy to see.
- **Per-category flag rate** — flagged/total and similarity range per category. The boundary between the top `related_compatible` and the bottom flagged `conflict` shows how much margin the floor actually has.
- **Confusion matrix** — precision/recall against the ideal detector. False negatives are conflicts the floor missed; false positives are non-conflicts that wrongly flagged.

## Extending

Add a case to `goldset.json` (pick a `category`, set `expect_flag`). To probe a threshold change, edit `CONTRADICTION_SIM_FLOOR` / `CONTRADICTION_SIM_CEILING` in `src/phileas/engine.py` and re-run — the report tracks the live values.
