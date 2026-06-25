---
name: phileas
description: Phileas long-term companion memory. Recall past context BEFORE answering when the prompt references past work, decisions, named projects, people, dates, or asks "what did we / last time / remember when". Stream conversation turns to Phileas with `ingest`; it distills durable memories from them itself.
---

# Phileas — Companion Memory

Phileas is the user's centralized memory layer — three databases (SQLite + ChromaDB + KuzuDB) behind an MCP connector that store facts, find them semantically, and connect entities.

Tool names below are written bare (`recall_recent`, `about`, `recall`, …). They map to whatever prefix your environment exposes them under — `mcp__phileas__*` in local Claude Code, the connector's prefix on claude.ai. Use the matching name from your tool list.

## Recall — load context before answering

Recall when the prompt references past work, decisions, people, dates, named projects, or asks anything like "what did I / we", "last time", "remember when", "before we", "you mentioned". Skip recall entirely when the prompt is purely about the code, task, or conversation already in front of you.

### Query shape — focused terms, not sentences

Phileas reads `recall(query=...)` best as a *focused term phrase* — one concept, 1–4 words. The keyword path OR-matches each token against memory summaries and ranks by coverage: a summary surfaces if it holds *any* token, and ranks higher the more of the query's tokens co-occur in it. A focused phrase floats the memory whose summary carries all its tokens to the top; a verbatim user sentence ("what did the user say about Alex and the Q3 budget") instead drags in filler tokens ("what", "did", "the") that match unrelated memories and dilute the coverage signal — and long natural-language queries score poorly on the semantic path too. So **extract the named entities and concepts from the prompt first**, then issue one tool call per concept and merge the results by `id`: coverage rewards tokens that co-occur, so concepts living in *separate* memories surface far better as separate queries. For *"did Alex bring up the Q3 budget at the planning offsite"*: call `about("Alex")`, `recall("Q3 budget")`, and `recall("planning offsite")` in parallel — not one sentence-shaped `recall()`.

### Pick the tool by query shape

Route by the shape of the question. Call several in parallel when shapes overlap, then merge results by `id`:

- **Time-relative** ("yesterday", "recently", "last week", "last session", "last time we talked") → `recall_recent(days=N)`. Top memories per day, newest first, bounded. Reach for this first when the question has a temporal anchor.
- **Named entity** in the prompt (person, project, tool) → `about(name=...)`. Pass the bare name without a leading `@`. Returns every memory linked to that entity in the graph — the cheapest, most precise "who is X / what about Y" lookup. Bounded ("+N more" footer when a hub entity is capped).
- **Explicit date** ("2026-04-14", "Apr 14") → `list_day_memories(date="YYYY-MM-DD")`. Every active memory anchored to that day.
- **Topic / concept** with no entity or date anchor → `recall(query=<focused term, 1–4 words>)`. Hybrid gather + cross-encoder rerank.
- **Date range** spanning multiple days → `timeline(start=..., end=...)`.
- **Wildcard / cross-topic nudge** (no anchor; you want what the task *wouldn't* surface) → `serendipity(n=3)`. Opt-in, not relevance-gated. Pass ids already in context as `exclude_ids`.

### Pointers in, hydrate on demand

Recall-family tools (`recall`, `recall_recent`, `about`, `timeline`) return cheap **pointers**, not full bodies:

```
[a1b2c3d4] [event] 2026-06-07 · Mara bought a cake in Lisbon last night · Mara, Lisbon
```

That line is `[id8] [type] date · summary · entity tags`. The summary is the whole fact — for most prompts the pointers already answer the question, so **don't fan out `recall()` a dozen times hoping for depth**, and don't dump everything. `recall_recent` and `about` are bounded (a heavy day or hub entity shows a cap / `+N more` note) so they can't overflow the context.

When you genuinely need more than a pointer, drill in — cheapest to most expensive:

- `hydrate(id8)` — the full record of **one** memory: exact timestamps, status/counts, its source turn (the raw it was distilled from), its `thread_id`, and linked entities. The inverse of the pointer trim.
- `thread(thread_id)` — the conversation a memory came from: its raw turns in order, each with the memories it produced. Get `thread_id` from `hydrate` first. The deepest, most expensive view.
- `about(name)` — everything tied to an entity (also bounded).

Rule of thumb: scan pointers → hydrate the one or two that matter → thread only if you need the surrounding conversation. Each hop up the ladder costs more context, so climb it deliberately.

### Use the context

A recalled memory is a prior that shapes how you answer, not content to repeat back. By default it stays unspoken: let it set your stance, assumptions, and tone, so the answer reads as if you simply know the person. Reciting what you remember to show that you remember is what makes a conversation feel bounded and surveilled; a friend who knows you doesn't narrate your own history back at you to prove it.

Name a recalled memory explicitly only when it earns it:

- the user asks about the past ("what did we decide", "last time"), or
- stating the fact materially changes or grounds the answer (a constraint, a decision, a date the user needs).

Otherwise hold it. Never lead with "Based on my memory…", and never list what you know about someone as a preamble. Work the knowledge in silently and surface it only when it's load-bearing.

## Capture — stream turns to Phileas

You do not decide what to remember. Hand conversation turns to Phileas with `ingest`, and it watches from the outside and distills durable memories from them on its own, with its own model. There is no `memorize` step to run, and no judgment call about what is worth keeping; that is Phileas's job now.

### How to ingest

- `ingest(content=<the turn, verbatim>, attribution=<self|other|source>)` captures the turn and extracts from it in the background.
- `attribution` is whose words the turn is, from the user's standpoint:
  - `self` — the user's own words (the default).
  - `other` — someone, or an agent, the user is talking with.
  - `source` — external material the user brought in (a pasted article, a quoted passage).
- To keep a conversation's turns together, call `start_thread(client_key="claude_code:<session_id>")` once, keep the returned `thread_id`, and pass it to every `ingest` for that conversation. The `client_key` makes it resume-safe across a compaction or `--resume`. Omit `thread_id` and the turn stands as its own one-turn thread.

### Which turns to pipe

Pipe the turns that carry something durable about the user — personal facts, decisions and their reasons, preferences, events with a time anchor, patterns, the project archaeology that code and git won't preserve. Let pure task and code chatter pass. You are choosing which turns count as a conversation worth remembering against, not which facts become memories; Phileas decides the latter.

Forward-prescriptive conventions ("always use snake_case", "tests live in `tests/`") are not memory — they belong in `CLAUDE.md`.

### When extraction is off

If no extraction key is configured, ingested turns are still captured and recallable verbatim; they simply wait, un-distilled, until a key is present. Nothing is lost, and you don't ingest any differently.

### Fixing a name

Entity curation stays available for when the user explicitly asks to fix how someone is known: `find_entities(stem)` lists candidates, `alias(name=..., alias=...)` records a user-declared alias, and `merge_entities` folds two nodes that are the same person. Don't guess these; act on them only when the user states the convention.
