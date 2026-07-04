# Decision-memory eval — Harborbase

**Question:** does a recorded decision come back when you are working in the area it governs, stay out of areas it does not, and step aside when it is reversed?

**Why a fictional project:** following the `coldstart` precedent, the corpus is a stranger (a made-up SaaS backend, `bible.json`), not phileas's own decisions, so a pass measures the mechanism rather than overfitting to this repo. Dogfooding phileas's real decisions is the qualitative companion to this; a real third-party repo (e.g. a Laravel app) is the later breadth/scale stress test.

## Method

`run.py` loads every decision in `bible.json` through the real `tool_runner.memorize` path (so `source_text` becomes the body event, `entities` become the locus, `memory_type="decision"`) into an isolated throwaway store under `_store`, then grades deterministically — no model judgment, no network. Each decision carries probes:

- `{"about": <entity>}` — the locus lookup a pre-edit hook would run (`about(<file>, memory_type="decision")`).
- `{"recall": <query>}` — the topical path.

## What it grades

- **Layer 1a, retrieval:** each decision surfaces for its own `should_hit` probes (its file, its subsystem, its topic).
- **Layer 1b, isolation:** each decision stays out of its `should_miss` probes — an unrelated file/topic does not surface it. This is the relevance-scoping claim: recall spends no context on decisions that do not govern the area in hand.
- **Layer 3, evolution:** a superseded decision (LISTEN/NOTIFY) drops out of active recall while the decision that replaced it (SKIP LOCKED) takes its place on the same loci.

## Result

63/63 checks pass. Locus-scoped retrieval, clean cross-subsystem isolation, and supersession-aware recall all hold on a project phileas has never seen.

## Not yet covered (next layers)

- **Layer 2, capture (model-in-loop):** blind subagents read dated session transcripts and decide whether/how to call `memorize` — grading trigger rate, locus-tag correctness, and content fidelity against the bible. This tests the instruction layer, not just the engine. Needs an agent harness; the `coldstart` blind-extraction setup is the template.
- **Layer 4, guardrail (end-to-end):** a fresh session about to violate a decision; does it surface and change the output. Needs the recall trigger (the deferred hook integration); can be simulated now by injecting a recalled decision and checking obedience.
