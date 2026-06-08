---
name: phileas
description: Phileas long-term companion memory. Recall past context BEFORE answering when the prompt references past work, decisions, named projects, people, dates, or asks "what did we / last time / remember when". Memorize new facts when the user shares personal info, makes decisions, expresses preferences, discusses life events, or makes an explicit memory request.
---

# Phileas — Companion Memory

Phileas is the user's centralized memory layer — three databases (SQLite + ChromaDB + KuzuDB) behind an MCP connector that store facts, find them semantically, and connect entities.

Tool names below are written bare (`recall_recent`, `about`, `recall`, …). They map to whatever prefix your environment exposes them under — `mcp__phileas__*` in local Claude Code, the connector's prefix on claude.ai. Use the matching name from your tool list.

## Recall — load context before answering

Recall when the prompt references past work, decisions, people, dates, named projects, or asks anything like "what did I / we", "last time", "remember when", "before we", "you mentioned". Skip recall entirely when the prompt is purely about the code, task, or conversation already in front of you.

### Query shape — focused terms, not sentences

Phileas treats `recall(query=...)` as a *focused term phrase* — one concept, 1–4 words. The keyword path AND-matches every token against memory summaries, so a verbatim user sentence ("what did the user say about Alex and the Q3 budget") AND-matches almost nothing and falls back to graph + semantic alone. So **extract the named entities and concepts from the prompt first**, then issue one tool call per concept and merge the results by `id`. For *"did Alex bring up the Q3 budget at the planning offsite"*: call `about("Alex")`, `recall("Q3 budget")`, and `recall("planning offsite")` in parallel — not one sentence-shaped `recall()`.

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
[a1b2c3d4] [event] 2026-06-07 · trungld bought a cake in Shanghai last night · trungld, Shanghai
```

That line is `[id8] [type] date · summary · entity tags`. The summary is the whole fact — for most prompts the pointers already answer the question, so **don't fan out `recall()` a dozen times hoping for depth**, and don't dump everything. `recall_recent` and `about` are bounded (a heavy day or hub entity shows a cap / `+N more` note) so they can't overflow the context.

When you genuinely need more than a pointer, drill in — cheapest to most expensive:

- `hydrate(id8)` — the full record of **one** memory: exact timestamps, importance/status/counts, the full `source_event_id`, and linked entities. The inverse of the pointer trim.
- `thread(event_id)` — the verbatim originating conversation + every sibling memory from that turn. Get `event_id` from `hydrate` first. The deepest, most expensive view.
- `about(name)` — everything tied to an entity (also bounded).

Rule of thumb: scan pointers → hydrate the one or two that matter → thread only if you need the raw conversation. Each hop up the ladder costs more context, so climb it deliberately.

### Use the context

Treat recalled memories as background context, not as content to recite. Reference them when answering only if directly relevant. Never lead a response with "Based on my memory…" — work the context in naturally.

## Memorize — store new facts

Inline `memorize` (and `memorize_batch` for multiple facts from one turn).

### What to save

Phileas captures what the code alone and git alone will not preserve. **Archaeology test:** will this still be useful when the code shows only the result and git shows only the diff?

- **Personal facts** the user states about themselves, people in their life, or their situation.
- **Preferences** about tools, workflow, tone, collaboration style.
- **Decisions** — especially ones with a stated reason ("we're going with X because Y").
- **Project decision archaeology** — why X over Y, what was rejected, who pushed back, deadline/constraint that forced the call, alternative tried and reverted. The narrative behind the diff.
- **Events** with a time anchor ("shipped v0.1.0 on Apr 4", "trip to Tokyo next month").
- **Patterns** observed over time — recurring frustrations, emotional throughlines, habits.
- **Project state** not derivable from code or git (ownership, blockers, why a design was chosen).

### What NOT to save

- **Forward-prescriptive conventions** ("always use snake_case", "tests live in `tests/`") — those belong in `CLAUDE.md`, which is the right home for rules. Phileas holds the *backward-narrative* archaeology, not the rulebook.
- **How the code works** — re-readable from the repo.
- **Git history, recent commits, who-changed-what** — `git log`/`git blame` are authoritative.
- **Transient task state** (current in-progress step, conversation context, temp debugging notes).
- **Anything already in `CLAUDE.md` or the repo's own docs.**
- **Fix recipes from debugging** — the commit explains the fix; don't mirror it in memory.

### Memory types

Pass `memory_type` as exactly one of these five — anything else stores but won't match recall's type filter:

- `profile` — who the user is: name, identity, core traits.
- `event` — things that happened: dates, milestones, life events.
- `knowledge` — facts, skills, stated preferences, and opinions the user holds (the default).
- `behavior` — recurring patterns and habits: workflows, communication and collaboration style.
- `reflection` — higher-level inferences across memories (usually generated by `reflect`).

Pick the one that best matches how the memory would be recalled later. Emotional throughlines and recurring patterns fold into `behavior` or `reflection` — there is no separate `emotional`, `pattern`, `preference`, or `project` type.

### Dedupe before writing

Before calling `memorize`, do a quick `recall` on the core entity or topic. If a very similar memory already exists:

- **Same fact, same wording** → skip.
- **Same fact, refined or corrected** → call `update()` on the existing memory_id instead of creating a new one.
- **Related but distinct angle** → write the new one; use `relate()` to link them.

### Importance and summary

- `summary` should be one sentence, self-contained — readable without the original turn for context.
- `importance` ranges 0.0–1.0. Reserve ≥0.8 for things that shape how you should act going forward (strong preferences, identity, major life events). Routine facts are 0.4–0.6.

### Language

**Always write `summary` (and any `raw_text` you pass) in English, even when the source turn is in Vietnamese or mixed language.** Translate the user's words; preserve proper nouns (people, places, projects, @mentions, brand names, and Vietnamese terms with no clean English equivalent — keep those in italics or quotes).

*Why:* Phileas embeds with `all-MiniLM-L6-v2`, an English-centric model. Vietnamese-vs-Vietnamese similarity peaks around 0.40–0.49, below the 0.5 recall floor — so non-English memories store cleanly but never surface in recall.

*Examples:*
- Source: "Sếp bảo phải nộp báo cáo trước thứ 6." → Summary: "Boss said the report must be submitted before Friday."
- Source: "Anh ấy nhắc về *tiền đen* trong ngành." → Summary: "He warned about *tiền đen* (off-the-books money) in the industry." (preserve the term, gloss it once)
- Don't store: "user mới biết hả" — translate: "User just learned this."

### Batching

When a single turn yields several distinct memories, prefer `memorize_batch` over N sequential `memorize` calls — it's faster and cheaper.

### Entity tagging

When calling `memorize` with `entities=[...]`, only tag an entity whose presence a future `about(name=<entity>)` query would find useful. A tag says "this memory is *about* this entity," not "this entity appears in this memory."

**The user-entity trap.** Nearly every memory is implicitly authored *by* the user. Tagging `Person:<user>` on every one makes `about('<user>')` return the whole activity log. Only tag the user when the memory is genuinely identity-shaped:

- **Tag `Person:<user>`** on `profile`, `behavior`, and `reflection` memories — things that describe who they are, how they act, or inferences about them.
- **Don't tag `Person:<user>`** on `event` and `knowledge` memories — the user is the implicit narrator; the tag adds noise, not signal.

**Other people and entities** (colleagues, partners, projects, tools) can be tagged freely — they're not implicit narrators, so `about(them)` is a useful retrieval primitive.

### Disambiguating same-name entities

Identity in the graph is an opaque uuid; `name` and `type` are attributes. The linker decides whether a new mention reuses an existing entity or mints a new one. Provide an optional `description` (one short line) on entity records when the name is potentially ambiguous — `Apple` the fruit vs. `Apple` the company, two people both named Alex, etc. Description is written once at entity creation and never overwritten, so it stays a stable disambiguator.

```
{"name": "Apple", "type": "Company", "description": "consumer electronics maker (Tim Cook era)"}
```

Skip `description` when the name is unambiguous in the user's world (their colleagues, their projects). For multi-type referents the same physical thing may carry — `Acme` is a place AND the company that owns it AND a project name — let the linker collapse them onto one uuid by tagging consistently and the migration script handles legacy splits.

### Handle vs. display-name: user-declared aliases

Phileas does **not** auto-pair a username-handle with a display name. Name bridges happen only three ways: diacritic/case folding (automatic and safe — `Ngân` ↔ `Ngan`), an explicit user-declared `alias`, or `merge_entities` to fold already-split nodes. The linker never guesses handle↔name pairings, because handle stems collide across *distinct* people — e.g. `huyenntk` (**N**guyễn Thị Khánh Huyền) and `huyenctk` (**C**hu Thị Khánh Huyền) share the stem `huyen`, so an auto-merge would silently fuse two real people. A miss is recoverable; a wrong merge is not.

So when the user refers to someone by a bare or partial name that may be ambiguous (or when `about(name)` looks like it's returning only a fragment of a person):

1. `find_entities(stem)` — lists every candidate (norm-aware, so `huyen` surfaces `huyenntk`, `huyenctk`, and `Huyền`), with memory counts and descriptions.
2. If more than one plausible match, **ask the user which one** — do not pick for them.
3. Persist their answer with `alias(name=<the unambiguous handle>, alias=<what they call them>)` — e.g. the user says "call Chu's one *huyen chu*" → `alias(name="huyenctk", alias="huyen chu")`. Afterwards `about("huyen chu")` and future mentions resolve to that entity.
4. If a candidate is an orphaned fragment that genuinely belongs to another (e.g. a 1-memory `Huyền` node that is the same person as `huyenctk`), fold it with `merge_entities` rather than aliasing.

The alias is the user's convention, set explicitly — never inferred.

---

## Claude Code only (optional accelerator)

This applies to local Claude Code, not claude.ai web (which has no hooks or local config — there, just route manually with the menu above).

If `recall.mode` in `~/.phileas/config.toml` is `auto` or `always`, the installer can wire a `phileas-hook recall` UserPromptSubmit hook that pre-fires recall before you read the prompt, plus a Stop hook for memorize. When that hook's `<phileas-recall>` output is present, treat it as background and only supplement with focused-term calls from the menu when it missed a concept. With no hook installed, the menu above is the whole path.
