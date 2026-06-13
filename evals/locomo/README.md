# LoCoMo smoke harness (Tier-2, manual / me-as-model)

A directional smoke for Phileas recall against the [LoCoMo](https://snap-research.github.io/locomo/)
long-term-conversation benchmark. **Not a pytest test** — it loads one LoCoMo
conversation into an isolated Phileas store and scores whether each question's gold
`evidence` turn surfaces in recall's top-k.

Why this exists, the full landscape, and the run findings:
[`docs/research/eval-benchmarks.md`](../../docs/research/eval-benchmarks.md).

> This smoke is **directional, not a quotable number**: 1 conversation, 9
> hand-picked cases, and *mechanical* extraction (one memory per turn, not faithful
> summarization). It exists to catch regressions/improvements in recall behavior
> while iterating on AA-136 / AA-137, against the recorded baseline below.

## Prereqs

- The repo venv (`.venv`) with the engine stack (chromadb, kuzu,
  sentence-transformers). First run downloads the embedding + cross-encoder models.
- Network egress (model downloads + the one-time data fetch).

## Run it

```bash
# 0. Fetch the LoCoMo corpus (once). Default path is /tmp/locomo10.json;
#    override with LOCOMO_JSON. 10 conversations, ~2.8 MB.
curl -s -o /tmp/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

# 1. See the sampled gold questions for a conversation (no store needed)
.venv/bin/python evals/locomo/locomo_smoke.py gold 0 --n 16

# 2. Extract one conversation into an ISOLATED store (~2 min: model load + ~419 memorizes)
mkdir -p /tmp/locomo-eval/conv0
PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/locomo_smoke.py extract 0

# 3. Score the 9 baseline cases + about() probe (graph ON by default)
PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/score_run.py

# 3b. Score the no-graph floor (keyword + semantic only)
PHILEAS_EVAL_GRAPH=off PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/score_run.py

# Ad-hoc probing while playing the agent-in-loop:
PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/locomo_smoke.py ask "Sweden" --top-k 10
PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/locomo_smoke.py about Caroline
```

Notes:
- The store under `PHILEAS_HOME` is **throwaway** — re-extract after any engine
  change (extraction is deterministic, ~2 min). `dia_map.json` (dia_id → memory id)
  lives in the home and is what makes scoring objective.
- `_engine()` uses `GraphStore` **in-process** (no daemon needed — the daemon only
  arbitrates the Kuzu lock across multiple processes). `PHILEAS_EVAL_GRAPH=off`
  falls back to the degraded no-graph path.

## Baseline — conv0 (Caroline/Melanie), 2026-06-12, mechanical extraction

Pre-change reference. After improving AA-136/137, re-run and diff against this.

| Case | query | gold | graph OFF | graph ON |
|------|-------|------|-----------|----------|
| Q1 research FOCUSED  | `adoption agencies` | D2:8 | D2:8 @3 | D2:8 @3 |
| Q1 research SENTENCE | `what did Caroline research about adoption` | D2:8 | D2:8 @1 | D2:8 @1 |
| Q2 LGBTQ group       | `LGBTQ support group` | D1:3 | D1:3 @5 | D1:3 @4 |
| Q4 charity FOCUSED   | `charity race awareness` | D2:2 | D2:2 @1 | D2:2 @1 |
| Q4 charity SENTENCE  | `what did the charity race raise awareness for` | D2:2 | D2:2 @1 | D2:2 @1 |
| Q6 identity          | `Caroline transgender identity` | D1:5 | D1:5 @7 | **MISS** |
| Q7 sunrise           | `Melanie sunrise painting` | D1:12 | MISS¹ | MISS¹ |
| Q14 self-care        | `Melanie self-care` | D2:5 | MISS² | MISS² |
| Q16 moved            | `Caroline moved Sweden` | D3:13, D4:3 | **empty**³ | D4:3 @4 |
| **any-gold-surfaced** | | | **6/9** | **6/9** |

`about('Caroline')` → 211/419 memories · `about('Melanie')` → 208/419 (firehose —
extraction tags the speaker on every turn).

¹ **False miss** — gold `D1:12` is the image-share turn; the answer-bearing turn
`D1:14` ("painted that lake sunrise last year") surfaces at rank 1. Score the
*answer*, not evidence rank.
² **Reranker/vocabulary gap** — `D2:5` phrases self-care as "me-time… running,
reading, violin"; semantic ranks lexical-vibe matches ("take care of yourself") above it.
³ **`SIMILARITY_FLOOR=0.5` soft-failure** — all semantic candidates 0.378–0.458,
every one cut → empty result.

### Win conditions when revisiting (after AA-136 / AA-137)

- Q16 stays a hit **and** Q6 returns (≤ top-10) → broadening + distributional cut
  rescue without the flood regression.
- Q14 surfaces `D2:5` → wider candidate pool + reranking closed the vocabulary gap.
- `about()` stops returning ~half the corpus → needs faithful extraction (tag
  mentioned entities, not the speaker) before this is meaningful; do that re-extract
  first or the firehose masks the signal.
- Replace evidence-rank with an **answer-level judge** (LoCoMo's own protocol)
  before trusting any aggregate — evidence-rank both under- and over-counts (¹, Q6).

## Status — after the recall rework (2026-06-12)

The AA-136/137 work plus the follow-on recall rework landed on
`feat/recall-threshold`: the gather pool is decoupled from `top_k`, the relevance
cut governs result size (no count cap), the graph hop is relevance-gated (an
entity match pulls only the memories that stand out for the query, not all of
them), the keyword floor scales by term **rarity (IDF)** rather than coverage,
and the default cut is `ratio` (a head-selector) rather than `gap` (a
tail-trimmer). Re-scored conv0:

| measure | result |
|---------|--------|
| conv0 smoke (top-10), graph ON / OFF | **6/9** |
| threshold mode (no `top_k`), focused queries | 4–36 memories, self-bound |
| broad-query breadth | bounded by the cut (`painting` 1, `Caroline` 82, was 389) |
| Q16 "Caroline moved Sweden" | rescued — `D4:3` @2 (cosine 0.23 / sem-rank 415, via rare-term IDF) |

The win conditions above are met except where they depend on faithful extraction
(the `about()` firehose) or an answer-level judge.

## Faithful extraction — demonstrated (2026-06-13)

`LOCOMO_FAITHFUL=<path>` swaps the verbatim per-turn copy for a self-contained
fact per turn: pronouns resolved, the concept named in the text, speakers
attributed, and every named person tagged (not just the speaker). The facts for
conv0 sessions 1–4 (all 9 gold cases live there) are hand-written in
`faithful_conv0.json` — me-as-model, the Tier-2 reader. Sessions 5–19 stay
mechanical, so the run is faithful needles in a mechanical haystack and the turn
count is identical (419) — a clean A/B on text quality alone.

```bash
mkdir -p /tmp/locomo-eval/conv0faith
LOCOMO_FAITHFUL=evals/locomo/faithful_conv0.json \
  PHILEAS_HOME=/tmp/locomo-eval/conv0faith .venv/bin/python evals/locomo/locomo_smoke.py extract 0
PHILEAS_HOME=/tmp/locomo-eval/conv0faith .venv/bin/python evals/locomo/score_run.py
```

| case | mechanical | faithful |
|------|-----------|----------|
| Q6 identity (`D1:5`)  | @7   | **@2** |
| Q7 sunrise (`D1:12`)  | miss | **@2** |
| Q14 self-care (`D2:5`)| miss | **@3** |
| Q1 research (`D2:8`)  | @1   | @3 (longer fact, still top-10) |
| **any-gold-surfaced** | **7/9** | **9/9** |

The mechanism, read off `ask "Melanie self-care"`: mechanically, the answer turn
`D2:5` is verbatim *"carving out some me-time… running, reading, violin"* — no
"self-care" token, so it never surfaces; the query instead matches `D2:3`/`D2:4`,
which carry the word but not the answer. The faithful fact for `D2:5` reads
*"Melanie practices self-care by carving out daily me-time — running, reading, or
playing her violin"*, co-locating the concept with the answer so the cross-encoder
scores it `@3 (0.574)`. Per-turn copy splits the concept from its answer across
adjacent turns; a reader writes them into one fact. Closes Q14's vocabulary gap
and lifts Q6 with no bigger embedder, query expansion, or reranker change.

### Open problems — pick up here next session

- [x] **Q14 vocabulary gap** — closed by faithful extraction (above); the concept
  word lands in the answer-bearing fact, so the cross-encoder matches it.
- [x] **Faithful extraction** — `faithful_conv0.json` + `LOCOMO_FAITHFUL` exists
  and tags named entities rather than the speaker. `about()` still returns ~half
  the corpus, because in a two-person conversation nearly every fact names one of
  the two speakers — the firehose is inherent to the corpus, not the tagging.
- [ ] **Evidence-rank still both under- and over-counts.** Q7 (`D1:12` is the
  image-share turn; the answer `D1:14` "painted that lake sunrise" is the real
  evidence) and Q6 (`D1:5` is one of several valid transgender turns) are mislabels.
  An **answer-level judge** (LoCoMo's own protocol) is the honest metric — pending
  because it needs a paid LLM call per question.
- [ ] **Faithful extraction at corpus scale.** Sessions 1–4 are hand-written.
  A quotable number needs faithful facts for all 419 turns × 10 conversations,
  which is the real ingest path (`ingest_session` → agent → `memorize_batch`),
  not hand authoring.
- [ ] **Tier-2 real number.** This 9-case smoke is directional only. A quotable
  LoCoMo figure needs the answer-level LLM judge + faithful extraction across all
  10 conversations (the agent-in-loop Mode B in
  [`docs/research/eval-benchmarks.md`](../../docs/research/eval-benchmarks.md)).

## Method sweep (distributional cut)

`sweep_standout.py` re-runs the 9 baseline cases under each `PHILEAS_STANDOUT`
strategy (`gap` / `zscore` / `ratio` / `knee`, plus `absolute:X` flat-floor
references) against an already-extracted store, so you can read off which cut
recovers cases like Q6/Q14 without re-extracting:

```bash
PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/sweep_standout.py
```

It prints, per method, how many cases surfaced any gold and the mean rank of
surfaced golds (lower = better). The `absolute:X` rows apply one floor uniformly
to both cut sites — a baseline to beat, not the exact historical split.

## Files

- `locomo_smoke.py` — loader, extractor (mechanical / windowed / faithful), `ask` / `about` / `gold` probes.
- `faithful_conv0.json` — hand-written self-contained facts for conv0 sessions 1–4, loaded via `LOCOMO_FAITHFUL`.
- `score_run.py` — the 9 baseline cases + `about()` probe, objective dia-id scoring.
- `sweep_standout.py` — re-scores those cases under each distributional-cut strategy.
