# LongMemEval eval — status & how to resume

Snapshot of an in-progress effort to measure phileas on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025). Read this first when picking the work back up.

## Goal

Get a real, comparable accuracy number for phileas on LongMemEval `s` (500 questions, evidence hidden among ~48 distractor sessions each), broken down by the six question types.

## What's proven

- **The faithful pipeline works end-to-end.** `faithful.py` runs each haystack through phileas's actual capture path: an LLM extracts durable facts per session -> `engine.memorize` -> `recall(question)` -> an LLM answers from what recall surfaced -> LongMemEval's own judge prompt grades it.
- **Oracle instance 0** (a car question, 3 evidence sessions): phileas answered **correctly** ("GPS system failing" vs gold "GPS system not functioning"). The answer-bearing fact was extracted, recalled at rank 2, and answered.
- **First `s` instance** (single-session-user, 53 sessions incl. distractors): **179 memories extracted, judged OK**. One real `s` data point.

## Harness state

All three LLM roles (extract, read, judge) run on **Claude Haiku** through headless `claude -p`, drawing on the Claude Code subscription (no API key, no per-call fee). It reuses phileas's own `PhileasClaudeCodeChat` adapter, so the subprocess isolations that adapter carries apply here too. The model is a single knob: `MODEL` at the top of `faithful.py` (a Claude Code alias — `haiku`, `sonnet`, `opus`). The cost is subscription rate limits, not dollars: extraction dominates (~50 calls per instance), so a 60-instance run is ~3k headless calls — the retry/backoff loop absorbs the 429s that pacing would otherwise avoid.

## Resume checklist

1. Data (not committed, ~277 MB): download `longmemeval_s_cleaned.json` into a sibling `LongMemEval/data/` checkout (see README "Data").
2. Auth: a logged-in `claude` CLI on PATH (the harness exits early if the binary is missing). No key to source.
3. Smoke: `.venv/bin/python evals/longmemeval/faithful.py 1` (1/type = 6 instances). Confirm no EXTRACT-FAIL.
4. Real run: `.venv/bin/python evals/longmemeval/faithful.py 10 faithful_s.json` (10/type = 60 instances) for the first real per-type number. Results checkpoint to `faithful_s.json` after every instance.

## Then

- Scale up (`faithful.py 25 …`, or the full set) once the 60-instance number looks sane.
- Compare per-type accuracy against the field's published Mem0 / Zep / Letta numbers.
