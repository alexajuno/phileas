# LongMemEval eval

Measuring phileas on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025): 500 hand-curated questions, each with its evidence sessions hidden among ~40-50 distractor sessions, across six memory abilities. The corpus is a stranger's chat history authored by someone else, so a number here measures the mechanism, not a gold set we wrote to our own strengths.

**Read [STATUS.md](STATUS.md) first** — it records where this stands (proven end-to-end) and the command to produce the next number.

`faithful.py` runs each haystack through phileas's actual capture pipeline, so the score reflects what phileas would remember rather than a reranker over raw chat blobs:

    extract (an LLM reads each session, emits memory JSON)
      -> engine.memorize (the real capture path)
      -> recall(question)
      -> reader (an LLM answers from what recall surfaced)
      -> judge  (LongMemEval's own per-type prompt grades the answer)

This exercises the whole system and yields the per-type accuracy the field reports (comparable to Mem0 / Zep / Letta). All three LLM roles (extract, read, judge) run on Claude Haiku through headless `claude -p`, drawing on the Claude Code subscription rather than a metered API key. It reuses phileas's own `PhileasClaudeCodeChat` adapter, so the subprocess isolations that adapter carries apply here too. The model is one knob: point `MODEL` at another Claude Code alias (`haiku`, `sonnet`, `opus`) to swap it.

## Data

The dataset is not committed (the `s` file is ~277 MB). It lives in a sibling [LongMemEval](https://github.com/xiaowu0162/LongMemEval) checkout, and the harness defaults to `../../LongMemEval/data/longmemeval_s_cleaned.json`. Use the **cleaned** release, which corrects the answer-key errors an [independent audit](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer) found in this benchmark family.

```bash
mkdir -p LongMemEval/data && cd LongMemEval/data
curl -sSL -O https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

## Run

From the `core/` root, with a logged-in `claude` CLI on PATH (no key to source):

```bash
.venv/bin/python evals/longmemeval/faithful.py 1                  # 1/type = 6 instances (smoke)
.venv/bin/python evals/longmemeval/faithful.py 10 faithful_s.json # 10/type = 60 instances
```

Args are positional: `PER_TYPE` (instances per question type, default 1) and `OUT_NAME` (results file, default `faithful_s.json`). Results checkpoint to `OUT_NAME` after every instance, so a run is safe to interrupt and inspect mid-flight. Each line prints `[i/N] <type> <sessions>s -> <memories>mem | OK|MISS|EXTRACT-FAIL`. The cost is subscription rate limits rather than dollars: extraction dominates (~50 headless calls per instance), and the retry/backoff loop absorbs the 429s that pacing would otherwise avoid.

## Reading the output

- **overall accuracy** — fraction of scored instances the judge marked correct.
- **by question_type** — `temporal-reasoning` grades the temporal path, `knowledge-update` grades supersession, `multi-session` grades cross-session synthesis, `single-session-*` grade plain lookup, `single-session-preference` grades personalized use of a stated preference. This is where the pipeline is strong or weak, read straight off the scorecard.
- **EXTRACT-FAIL** — an instance that extracted 0 memories (every session's extraction call errored) is an infrastructure failure, not a wrong answer; it is excluded from the accuracy denominator and reported separately.

## Method and its edges

- **Session granularity.** Extraction reads one haystack session at a time (its turns rendered `role: content`) and emits durable facts, which `engine.memorize` stores through the real capture path. A very long session can be truncated by the extractor's input window.
- **Per-question isolation.** Every question has its own haystack, so each instance gets a fresh throwaway store (profile `longmemeval-eval`), built and torn down in turn, to keep one question's distractors from leaking into another's.
- **Abstention skipped.** Questions whose id ends `_abs` are unanswerable by design; they are filtered out so the accuracy number is over answerable instances only.
