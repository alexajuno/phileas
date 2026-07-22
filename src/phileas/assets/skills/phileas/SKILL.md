---
name: phileas
description: Phileas long-term companion memory. Use it two ways. To recall, when the prompt references past work, decisions, named projects, people, dates, or asks "what did we / last time / remember when". To capture, when the user asks to remember, record, note, or save something, above all a decision (a choice and why): call `memorize` directly for a single fact, or run the review-first capture pass with `propose_memory` when they ask to save what's worth keeping from the whole conversation.
---

# Phileas — Companion Memory

Phileas is the user's centralized memory layer — three databases (SQLite + ChromaDB + KuzuDB) behind an MCP connector that store facts, find them semantically, and connect entities.

Tool names below are written bare (`recall_recent`, `about`, `recall`, …). They map to whatever prefix your environment exposes them under — `mcp__phileas__*` in local Claude Code, the connector's prefix on claude.ai. Use the matching name from your tool list.

## Recall — load context before answering

Recall when the prompt references past work, decisions, people, dates, named projects, or asks anything like "what did I / we", "last time", "remember when", "before we", "you mentioned". Skip recall entirely when the prompt is purely about the code, task, or conversation already in front of you.

### Query shape — focused terms, not sentences

Phileas reads `recall(query=...)` best as a *focused term phrase* — one concept, 1–4 words. The keyword path OR-matches each token against memory content and ranks by coverage: a memory surfaces if its content holds *any* token, and ranks higher the more of the query's tokens co-occur in it. A focused phrase floats the memory whose content carries all its tokens to the top; a verbatim user sentence ("what did the user say about Alex and the Q3 budget") instead drags in filler tokens ("what", "did", "the") that match unrelated memories and dilute the coverage signal — and long natural-language queries score poorly on the semantic path too. So **extract the named entities and concepts from the prompt first**, then issue one tool call per concept and merge the results by `id`: coverage rewards tokens that co-occur, so concepts living in *separate* memories surface far better as separate queries. For *"did Alex bring up the Q3 budget at the planning offsite"*: call `about("Alex")`, `recall("Q3 budget")`, and `recall("planning offsite")` in parallel — not one sentence-shaped `recall()`.

### Pick the tool by query shape

Route by the shape of the question. Call several in parallel when shapes overlap, then merge results by `id`:

- **Time-relative** ("yesterday", "recently", "last week", "last session", "last time we talked") → `recall_recent(days=N)`. Top memories per day, newest first, bounded. Reach for this first when the question has a temporal anchor.
- **Named entity** in the prompt (person, project, tool) → `about(name=...)`. Pass the bare name without a leading `@`. Returns every memory linked to that entity in the graph — the cheapest, most precise "who is X / what about Y" lookup. Bounded ("+N more" footer when a hub entity is capped).
- **Explicit date** ("2026-04-14", "Apr 14") → `timeline(start_date="YYYY-MM-DD", window=0)`. Every active memory anchored to that day.
- **Topic / concept** with no entity or date anchor → `recall(query=<focused term, 1–4 words>)`. Hybrid gather + cross-encoder rerank.
- **Date range** spanning multiple days → `timeline(start_date=..., end_date=...)`.
- **Wildcard / cross-topic nudge** (no anchor; you want what the task *wouldn't* surface) → `serendipity(n=3)`. Opt-in, not relevance-gated. Pass ids already in context as `exclude_ids`.

### What a recall returns

Recall-family tools (`recall`, `recall_recent`, `about`, `timeline`) return one line per memory:

```
[a1b2c3d4] [event] 2026-06-07 · Mara bought a cake in Lisbon last night · Mara, Lisbon
```

That line is `[id8] [type] date · content · entity tags`, and the content is the whole fact. What you read is what the memory says, so answer from it: **don't fan out `recall()` a dozen times hoping for depth**, and don't dump everything. Results are bounded by relevance and by output size (a heavy window or hub entity shows a cap / `+N more` note) so a broad query returns the strongest few rather than everything.

The one step past a memory is the conversation it came from:

- `source(source_id)` — the raw turns of a session, in order, plus the memories distilled from them. `recall_recent` prints a session handle (`↳id8`) on each line; `get_source_memories(id)` is the lighter version that lists a session's memories without the turns. Far more expensive than a recall, so reach for it only when the wording of the conversation itself is what you need.

### Use the context

A recalled memory is a prior that shapes how you answer, not content to repeat back. By default it stays unspoken: let it set your stance, assumptions, and tone, so the answer reads as if you simply know the person. Reciting what you remember to show that you remember is what makes a conversation feel bounded and surveilled; a friend who knows you doesn't narrate your own history back at you to prove it.

Name a recalled memory explicitly only when it earns it:

- the user asks about the past ("what did we decide", "last time"), or
- stating the fact materially changes or grounds the answer (a constraint, a decision, a date the user needs).

Otherwise hold it. Never lead with "Based on my memory…", and never list what you know about someone as a preamble. Work the knowledge in silently and surface it only when it's load-bearing.

## Capture — what's worth keeping, and how it lands

Every turn is already saved verbatim by Phileas on its own. You never capture the raw record and never judge which turns to keep; that floor is laid for you. Your job is higher: turn what a conversation taught into durable memory. How that memory lands depends on the mode, and the mode announces itself through the hint you see.

**What is worth keeping** (the same in every mode). Save what would be useful to recall later, from the user or from your own work: a fact or preference the user states about themselves; a decision and its reason; a gotcha or root cause you hit; a wiring or location fact that would otherwise go stale and mislead; a dead end worth not re-walking; a command or recipe that worked. A thing *you* discovered counts on its own, it does not need the user to have endorsed it first. The bar is usefulness, not authorship: the archaeology test. Will this still be useful once the code shows only the result and git shows only the diff? If it survives that, keep it; if it is obvious from the code or the diff, let it go. Never keep what the user waves off, argues down, or passes over. A path rejected is not a memory; the choice landed on is.

**How it lands** — three surfaces, and the hint tells you which is live:

- **The user asks directly** ("remember that ...", "save this"). Call `memorize` now — they have already judged it worth keeping. Works in any mode.
- **A `<phileas-memorize-hint>` at end of a turn** (the per-turn mode). Judge that one turn against the bar and, if something qualifies, `memorize` it, one memory per fact.
- **A `<phileas-capture-hint>` carrying a `thread_id`** (the default, review-first mode). Do not `memorize` mid-conversation. When the user asks to capture or save what's worth keeping from the conversation, review the whole thread and, for each candidate, call `propose_memory(content=..., thread_id=<the id from the hint>, ...)`, one call per fact. Nothing is stored yet: proposals wait in a review queue the user approves from (`phileas memory queue list`, or the web dashboard). After proposing, tell the user how many you queued and where to review them. Don't propose unprompted; a normal turn just answers.

**Writing a memory or a proposal** — the fields are the same for `memorize` and `propose_memory`:

- `content` is the conclusion as the single line recall surfaces. `source_text` is the body: for `memorize` the reasoning and alternatives that `source` reads back; for `propose_memory` a short "why it's worth keeping" the user sees at review.
- `memory_type` is `decision` for a choice-and-why, else `knowledge` / `behavior` / `reflection` / `event` / `profile`.
- Tag `entities` with what the memory governs — for a code decision the repo, the file(s) or dir, and the concept — so a later `about(<file>)` surfaces it. With no entities it is findable only by full-text search. Pick each entity's `type` from exactly this vocabulary, closest bucket wins: Person, Organization, Place, Project, Tool, Object, Animal, Activity, Event, Concept. The type is a collision-resistant bucket, not a rich label; an invented synonym (Company, Topic, Repo) splits the same referent across separate graph nodes. Put richness in the entity's `description`, a brief stable phrase saying which one this is, which also helps the linker keep same-name entities apart.
- If a `memorize` write conflicts with an existing memory, the result ends with a resolve menu; that is how a reversed decision supersedes the one it replaces.

Forward-prescriptive conventions ("always use snake_case", "tests live in `tests/`") are not memory; they belong in `CLAUDE.md`. The decision behind one ("snake_case over camelCase because the linter assumes it") is a memory.

### Fixing a name

Entity curation stays available for when the user explicitly asks to fix how someone is known: `find_entities(stem)` lists candidates, `alias(name=..., alias=...)` records a user-declared alias, and `merge_entities` folds two nodes that are the same person. Don't guess these; act on them only when the user states the convention.
