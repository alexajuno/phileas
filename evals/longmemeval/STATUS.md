# LongMemEval eval — status & how to resume

Snapshot of an in-progress effort to measure phileas on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025). Read this first when picking the work back up.

## Goal

Get a real, comparable accuracy number for phileas on LongMemEval `s` (500 questions, evidence hidden among ~48 distractor sessions each), broken down by the six question types.

## What's proven

- **The faithful pipeline works end-to-end.** `faithful.py` runs each haystack through phileas's actual capture path: an LLM extracts durable facts per session -> `engine.memorize` -> `recall(question)` -> an LLM answers from what recall surfaced -> LongMemEval's own judge prompt grades it.
- **Oracle instance 0** (a car question, 3 evidence sessions): phileas answered **correctly** ("GPS system failing" vs gold "GPS system not functioning"). The answer-bearing fact was extracted, recalled at rank 2, and answered.
- **First `s` instance** (single-session-user, 53 sessions incl. distractors): **179 memories extracted, judged OK**. One real `s` data point.

## The blocker (why there's no multi-instance number yet)

The extractor is **Claude Code headless** (`claude -p`, haiku) so it runs under a subscription with no API key. But subscription headless haiku is **rate/usage-capped at ~50 calls per window**. A real run needs hundreds-to-thousands of extraction calls, so after the first instance (~50 sessions) every later instance extraction-fails with 0 memories:

```
[1/18] single-session-user 53s -> 179mem | OK
[2/18] single-session-user 45s -> 0mem   | EXTRACT-FAIL
[3/18] single-session-user 50s -> 0mem   | EXTRACT-FAIL
```

Lowering concurrency (5 -> 3) and adding retries did not help. A throttled `claude -p` returns the usage-limit notice as a **successful** result (exit 0, non-empty text), so retry-on-empty does not catch it and it records as 0 memories rather than an error.

## Open decisions (pick up here)

1. **Extraction model.** Either (A) stay haiku-only and pace far under the cap (~1 instance per usage window, impractical for a real number), or (B) swap `extract_session`'s LLM for an API model — **gpt-4o-mini** via the OpenAI key is reliable and ~$0.55 for a 60-instance run. The pipeline is model-agnostic; only `claude()`/`extract_session` in `faithful.py` change. B is the practical path.
2. **Capture the exact rate-limit message** and make the extractor detect it (treat as failure, back off, or stop) instead of silently recording 0 memories. Needs re-triggering the cap (~50 haiku calls).
3. **Then**: run `faithful.py 10 faithful_s.json` (10/type = 60 instances) for the first real per-type number.

## Files

- `faithful.py` — the real harness (extract -> memorize -> recall -> answer -> judge). Self-contained; extractor is the piece to swap for option B. **This is the one that matters.**
- `run.py`, `qa.py`, `_engine.py` — an earlier **retrieval-only** exploration that indexes *raw* sessions and skips extraction. Kept for reference, but it does not faithfully test phileas (it measures vector+reranker ranking over raw chat blobs, not the capture pipeline). See README.
- `README.md` — suite overview and data download.

## Resume checklist

1. Data (not committed, ~277 MB): download `longmemeval_s_cleaned.json` into a sibling `LongMemEval/data/` checkout (see README "Data").
2. Decide extraction model (A vs B above). For B, point `extract_session` at gpt-4o-mini (`OPENAI_API_KEY` from `~/.secrets/openai.env`).
3. Smoke: `.venv/bin/python evals/longmemeval/faithful.py 1` (1/type = 6 instances). Confirm no EXTRACT-FAIL.
4. Real run: `.venv/bin/python evals/longmemeval/faithful.py 10 faithful_s.json`. Results checkpoint to `faithful_s.json` after every instance.
