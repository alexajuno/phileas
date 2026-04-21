# 01 — Gold set of labeled Claude Code transcripts

**Blocks:** [02 — Eval harness](./02-eval-harness.md)
**Created:** 2026-04-21

## Goal

Produce a fixed, hand-labeled evaluation set of ~40 Claude Code transcripts so every downstream change to the extraction pipeline is measurable.

## Source data

Real ingested transcripts already live at `~/.phileas/ingested-sessions.json`. Sample from there — not from synthetic or hand-written examples. The distribution of noise/signal in real data is the whole point.

## Sampling strategy (40 items)

Stratify the sample so we cover the full distribution of inputs the pipeline actually sees:

| Stratum | Target count | Why |
|---|---|---|
| Pure coding session (English, all tool output) | 8 | Most common workflow |
| Coding + life mix (English) | 5 | Casual convos mid-session |
| Vietnamese conversational | 6 | Language rule validation |
| Mixed VN/EN (code-switching) | 5 | Real usage pattern |
| Short prompts (<200 chars) | 6 | Most likely zero-memory |
| System-reminder heavy / noise | 5 | Tests the "ignore scaffolding" rule |
| Explicit memory requests ("remember X") | 3 | Must-capture cases |
| Empty/trivial ("ok", "continue") | 2 | Must produce zero memories |

Store the sampled transcripts verbatim in `tests/eval/gold/transcripts/<id>.txt` where `<id>` is a short slug (e.g., `001-usd-migration-planning`).

## Label schema (per transcript)

Each labeled case lives at `tests/eval/gold/labels/<id>.yaml`:

```yaml
id: 001-usd-migration-planning
source_session_id: <uuid from ingested-sessions.json>
stratum: coding-english
expected_memories:
  - summary: "ModelProvider.php has USD-based cost fields planned for removal in favor of credit_cost."
    memory_type: knowledge
    importance: 7
    required_substrings: ["ModelProvider", "credit_cost"]  # precision gate
  - summary: "User prefers bundling related migrations into one PR."
    memory_type: behavior
    importance: 6
    required_substrings: ["migration", "PR"]
expected_entities:  # graph track — top-level, not per-memory (see note below)
  - name: ModelProvider.php
    type: File
  - name: credit_cost
    type: Field
expected_relationships:
  - from_name: ModelProvider.php
    edge: HAS_FIELD
    to_name: credit_cost
notes: |
  Rationale for each expected memory goes here. Explain why borderline
  items were or were not included.
```

**Why graph fields are top-level, not per-memory:** the graph merges entities
across memories for a given ingest. If memory A extracts `{ModelProvider.php}`
and memory B extracts `{credit_cost}`, the graph ends up with both — we don't
care which memory produced which entity. Aggregating at the case level is
also more forgiving of acceptable LLM variation (same entity on a different
memory still counts).

**`expected_memories: []` is valid** and expected for the "empty/trivial" and
most "system-reminder heavy" strata. `expected_entities` and
`expected_relationships` default to `[]` too and are informational in the
first baseline (no hard targets set until we have real data).

## Labeling rules

1. **Zero-memory is the default for noise.** If the transcript is a continuation prompt, a tool-call echo, or a skill file being loaded — label zero.
2. **One concept per memory.** If the transcript mentions three distinct facts, write three expected memories.
3. **English summaries only** (aligned with the English-only rule in `fact_derivation.txt:35`). Translate when labeling Vietnamese content.
4. **`required_substrings`** are the small set of tokens a correct summary *must* contain to count as a match. Keeps fuzzy-match scoring grounded.
5. **Importance sanity bounds:** 1-4 never appear (they shouldn't be stored at all); 5-6 = typical knowledge/event; 7-8 = significant; 9-10 = reserved for identity-level facts.
6. **Graph labels are lightweight.** Only list `expected_entities` / `expected_relationships` when the content is genuinely project-specific or identity-forming (class names, people, places, tools). Generic nouns ("code", "user", "project") don't belong — they'd add noise without measurement value. For zero-memory cases, leave the graph fields as `[]`.
7. **Entity matching is case-insensitive on name only.** Types are compared but reported as a separate "type_consistency" metric rather than gating the match, because same-name/different-type is a known failure pattern we want to measure, not obscure behind a binary.
8. **Relationship matching is reported both strict (from+edge+to) and endpoint-only (from+to, any edge).** Strict measures edge-type fidelity; endpoint-only measures whether the graph at least knows these two things are connected.

## Deliverables

- `tests/eval/gold/transcripts/*.txt` — 40 real transcripts
- `tests/eval/gold/labels/*.yaml` — 40 label files
- `tests/eval/gold/sampling.md` — one-page note explaining how transcripts were sampled, any redactions, and a change log for future additions

## Out of scope

- Automated labeling (defeats the purpose — this is the ground truth).
- Covering the `reflect()` or `consolidate()` paths. This gold set is for `extract_memories()` only.
