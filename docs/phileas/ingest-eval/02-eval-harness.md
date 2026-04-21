# 02 — Extraction eval harness

**Blocked by:** [01 — Gold set](./01-gold-set.md)
**Blocks:** [03 — Iteration](./03-iteration.md)
**Created:** 2026-04-21

## Goal

Replay the gold set through the extraction pipeline and produce a reproducible benchmark report. Every candidate change in phase 3 is evaluated by re-running this harness.

## Pipeline under test

`src/phileas/llm/extraction.extract_memories(client, text)` — the async function invoked by the daemon's ingest loop (`src/phileas/daemon.py:295`). The harness replaces the daemon's call site with a direct function call; no HTTP or queue involved.

We do **not** test `engine.memorize()` — that's SQLite/Chroma/Kuzu plumbing. The eval is about the *decision layer*: given text, what memories should we produce?

## Inputs

- `tests/eval/gold/transcripts/*.txt` — real transcripts
- `tests/eval/gold/labels/*.yaml` — expected memories per transcript
- An `LLMClient` instance (real LLM, not mocked — we're evaluating the actual extraction prompt in production config)

## Outputs

Per run, write to `tests/eval/runs/<timestamp>-<slug>/`:

- `per_case.jsonl` — one line per transcript: `{id, expected_count, predicted_count, matches, misses, false_positives, predicted_memories, elapsed_ms}`
- `summary.json` — aggregate metrics
- `summary.md` — human-readable report (top false positives, top misses, per-stratum breakdown)

The `<slug>` identifies the experiment (e.g., `baseline`, `no-fallback`, `with-gating`, `english-enforced`).

## Metrics

Compute per-transcript, then aggregate.

### Matching rule

A predicted memory is a **match** against an expected memory when:
1. **All `required_substrings`** from the expected entry appear (case-insensitive) in the predicted `summary`, AND
2. The predicted `memory_type` matches expected, AND
3. Predicted `importance` is within ±2 of expected.

Each expected memory can match at most one predicted memory (greedy assignment by order).

### Aggregate signals

| Metric | Definition | Target |
|---|---|---|
| **Precision** | matches / predicted_count (across all transcripts) | ≥ 0.70 |
| **Recall** | matches / expected_count | ≥ 0.70 |
| **Noise rate** | transcripts where `expected_count == 0` but `predicted_count > 0`, divided by zero-expected transcripts | < 0.10 |
| **Over-extraction rate** | transcripts where `predicted_count > expected_count`, divided by non-zero-expected transcripts | < 0.30 |
| **Miss rate** | transcripts where `expected_count > 0` but `predicted_count == 0` | < 0.10 |
| **Non-English rate** | predicted summaries containing Vietnamese diacritics or common VN function words, divided by predicted_count | 0.00 |
| **Median p95 latency** | per-transcript wall time | Report only; not a gate |

Break each down by stratum (see gold set spec) so regressions in one input class don't hide behind aggregate wins.

## CLI shape

```bash
# Baseline run (current extraction with no changes)
uv run python -m tests.eval.run_extraction \
  --gold tests/eval/gold \
  --out tests/eval/runs \
  --slug baseline

# Compare two runs
uv run python -m tests.eval.compare \
  tests/eval/runs/<timestamp>-baseline \
  tests/eval/runs/<timestamp>-no-fallback
```

The comparator prints a per-metric delta table (baseline → candidate, with sign), and lists transcripts where match-set changed.

## Implementation sketch

File: `tests/eval/run_extraction.py`

```python
# Pseudocode — real implementation lives in this file after 01 ships.
async def run_case(case_id, transcript_text, expected, llm) -> dict:
    t0 = time.monotonic()
    predicted = await extract_memories(llm, transcript_text)
    elapsed_ms = (time.monotonic() - t0) * 1000
    matches, misses, false_positives = match(predicted, expected)
    return {
        "id": case_id,
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "matches": [...],
        "misses": [...],
        "false_positives": [...],
        "predicted_memories": predicted,
        "elapsed_ms": elapsed_ms,
    }
```

The `match()` function implements the matching rule above. Keep it pure (no I/O) so it's trivially unit-testable.

## Determinism

LLM responses are non-deterministic. Two mitigations:
1. Run each transcript **N=3** times per eval. Report per-metric mean + stddev. A candidate "wins" a metric only if `mean_new - stddev_new > mean_baseline + stddev_baseline`.
2. Set temperature to 0 in the extraction LLM client for eval runs, if configurable. Document which setting was used in `summary.json`.

## Baseline first

Before any phase-3 change, run the harness against the current `main` extraction code and record the baseline. The README's success criteria (noise < 10%, precision ≥ 70%) are defined *relative to this baseline being worse*. If the baseline already meets targets, re-examine the problem statement.

## Deliverables

- `tests/eval/run_extraction.py` — runner
- `tests/eval/match.py` — matching logic (pure function)
- `tests/eval/compare.py` — two-run diff
- `tests/eval/runs/<timestamp>-baseline/` — first real run, committed
- README addendum in `tests/eval/README.md` explaining how to run it

## Out of scope

- UI for browsing runs (inspect JSON/markdown files directly).
- Testing the full daemon loop (extraction is the decision point).
- Testing `reflect()` or `consolidate()` (separate evals, separate day).
