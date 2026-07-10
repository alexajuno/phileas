# LongMemEval eval

Measuring phileas on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025): 500 hand-curated questions, each with its evidence sessions hidden among ~40-50 distractor sessions, across six memory abilities. The corpus is a stranger's chat history authored by someone else, so a number here measures the mechanism, not a gold set we wrote to our own strengths.

**Read [STATUS.md](STATUS.md) first** — it records where this stands (proven end-to-end, blocked on an extraction rate cap) and exactly how to resume.

Two approaches live here:

- **`faithful.py` — the real test (WIP).** Runs each haystack through phileas's actual capture pipeline: an LLM extracts durable facts per session, `engine.memorize` stores them, `recall` retrieves, a reader answers, and LongMemEval's own judge grades it. This exercises the whole system and yields the per-type accuracy the field reports (comparable to Mem0 / Zep / Letta). The one piece to settle is the extraction model (see STATUS).
- **`run.py` / `qa.py` — earlier retrieval-only exploration.** These index *raw* sessions (no extraction) and score whether recall surfaces the evidence. Cheap and $0, but they do not faithfully test phileas: they measure the vector + reranker ranking over raw chat blobs, not the capture pipeline. Kept for reference; the rest of this README describes them.

## Data

The dataset is not committed (the `s` file is ~277 MB). It lives in a sibling [LongMemEval](https://github.com/xiaowu0162/LongMemEval) checkout, and the runner defaults to `../../LongMemEval/data/longmemeval_s_cleaned.json`. Use the **cleaned** release, which corrects the answer-key errors an [independent audit](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer) found in this benchmark family.

```bash
mkdir -p LongMemEval/data && cd LongMemEval/data
curl -sSL -O https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

Point elsewhere with `--data <path>`.

## Pieces

- `_engine.py` — builds an isolated engine over an ephemeral temp dir (one store per question), refuses any path inside the real `~/.phileas`, and freezes the store's retrieval-strength write. The cross-encoder reranker is a process-global singleton, so it loads once and is shared across every per-question store.
- `run.py` — the runner. For each answerable instance: ingest every haystack session as one memory (tagged with its session id and date), `recall(question, top_k=k)`, map the returned memories back to their session ids, and score session-level recall@k / hit@k / MRR / nDCG@k. Aggregates overall and per `question_type`. Reuses the recall eval's `metrics.py` verbatim.

## Run

Via the project venv python, from the `core/` root:

```bash
.venv/bin/python evals/longmemeval/run.py --limit 20        # quick dev slice
.venv/bin/python evals/longmemeval/run.py --k 5 --out _runs # full 500, write JSON
```

Flags: `--k` (top_k for the @k metrics, default 5), `--limit N` (score only the first N answerable instances), `--data <path>`, `--out <dir>`. The runner prints `RERANKER: loaded …` as proof it ran the real model, and a running `hit@k` every 10 instances.

## Reading the output

- **recall@k** — of a question's gold evidence sessions, the fraction that landed in the top-k.
- **hit@k** — did any gold evidence session land in the top-k (the "did recall find it at all" signal).
- **mrr / ndcg@k** — how high the evidence ranked.
- **by question_type** — `temporal-reasoning` grades the temporal path, `knowledge-update` grades supersession, `multi-session` grades cross-session synthesis, the `single-session-*` types grade plain lookup. This is where phileas's retrieval is strong or weak, read straight off the scorecard.

## Method and its edges

- **Session granularity.** Each haystack session is stored as one memory (its turns rendered `role: content`), so scoring is session-level recall, one of LongMemEval's two retrieval granularities. Turn-level (one memory per turn, scored against the `has_answer` turn flags) is the finer-grained variant, left for a later pass.
- **Retrieval over raw sessions.** Phase 1 skips phileas's extraction step (distilling sessions into facts needs an LLM), so it indexes raw session text directly. That is an honest floor for "phileas retrieving without extraction"; the extraction-then-retrieve fidelity is a Phase 2 concern. One consequence: a long session is truncated by the embedder's input window, which turn granularity would avoid.
- **Per-question isolation.** Every question has its own haystack, so pooling all 500 into one store would let one question's distractors leak into another's. Each question gets a fresh store, built and torn down in turn. Cost is wall-clock (a full run is tens of minutes), not money.
- **Abstention skipped.** Questions whose id ends `_abs` are unanswerable by design, so retrieval recall does not apply; they are skipped and counted, matching LongMemEval's retrieval protocol. They return in Phase 2, where the reader is graded on correctly refusing.
