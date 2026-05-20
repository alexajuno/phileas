# Extraction eval harness

Measures precision / recall / noise rate of `phileas.llm.extraction.extract_memories` against a fixed gold set of real Claude Code transcripts.

## Layout

```
tests/eval/
  gold/
    transcripts/<id>.txt     # real transcript text, one per file
    labels/<id>.yaml         # expected memories per transcript
    sampling.md              # how transcripts were sampled, redactions, change log
  runs/<timestamp>-<slug>/   # one directory per eval run
    per_case.jsonl
    summary.json
    summary.md
  sampler.py                 # stratified sampler over ~/.claude/projects JSONL files
  match.py                   # pure matching logic (unit-tested)
  run_extraction.py          # runner: replay gold set through extract_memories()
  compare.py                 # two-run diff with significance gating
```

## Sampling new transcripts

```bash
uv run python -m tests.eval.sampler \
  --projects-dir ~/.claude/projects \
  --out tests/eval/gold \
  --count 40 \
  --seed 42
```

The sampler reconstructs the last user+assistant turn of each transcript using `gather_last_exchange()` (inlined from the historical Stop-hook implementation) and writes it byte-for-byte to `gold/transcripts/<id>.txt`.

It also writes skeleton label YAML files with `expected_memories: []` for hand-labeling.

## Running the harness

```bash
# Baseline (current main)
uv run python -m tests.eval.run_extraction \
  --gold tests/eval/gold \
  --out tests/eval/runs \
  --slug baseline \
  --repeats 3

# Compare two runs
uv run python -m tests.eval.compare \
  tests/eval/runs/<timestamp>-baseline \
  tests/eval/runs/<timestamp>-candidate
```

## Metrics

A predicted memory matches an expected one when every `required_substrings[i]` appears (case-insensitive) in the predicted `summary`, the `memory_type` matches, and predicted `importance` is within ±2 of expected. See `match.py` for the full rule, including graph-track matching for entities and relationships.

Targets:

| Metric | Target |
| -- | -- |
| Precision | ≥ 0.70 |
| Recall | ≥ 0.70 |
| Noise rate (expected=0 but predicted>0) | < 0.10 |
| Over-extraction rate | < 0.30 |
| Miss rate | < 0.10 |
| Non-English rate in predicted summaries | 0.00 |

Aggregates are reported with per-stratum breakdowns so regressions don't hide.
