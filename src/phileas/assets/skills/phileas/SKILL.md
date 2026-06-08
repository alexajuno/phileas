---
name: phileas
description: Phileas long-term companion memory. Recall past context BEFORE answering when the prompt references past work, decisions, named projects, people, dates, or asks "what did we / last time / remember when". Memorize new facts when the user shares personal info, makes decisions, expresses preferences, discusses life events, or makes an explicit memory request.
---

# Phileas — Companion Memory

Phileas is the user's centralized memory layer. Three databases (SQLite + ChromaDB + KuzuDB) work together to store facts, find them semantically, and connect entities. The Phileas daemon runs locally and exposes recall + memorize via MCP tools.

For each user message: **recall first → respond → memorize**. Don't reverse the order; recalling after answering means you've already answered without context.

## Hook-driven auto firing (the default path)

When `recall.mode` is `"auto"` or `"always"` (set in `~/.phileas/config.toml`), a `phileas-hook recall` UserPromptSubmit hook fires before you ever read the prompt. The hook is the deterministic firing mechanism — it runs every time, no skill-matcher heuristics involved. What it does depends on `recall.pipeline`:

- `pipeline = "rerank"` (default) → the hook calls the daemon's `recall` (gather + cross-encoder rerank + MMR), formats the top results, and injects a `<phileas-recall>` block at the top of the prompt. Just use it as context.
- `pipeline = "direct"` → the hook injects a static `<phileas-recall-hint>` block with a cognitive routing ladder. **When you see that hint, pick the right phileas tool by query shape and call it directly.** See Step 3 below for the ladder. Skip the call entirely if the prompt is purely about the current code/task/conversation.

In `mode = "auto"` the hook applies a content heuristic and only fires on memory-relevant prompts (past-tense queries, decision phrases, named dates, "remember when"-style cues). In `mode = "always"` it fires on every prompt. In `mode = "never"` the hook is removed.

The skill body below is for the explicit-invocation path (`/phileas …`) and for cases where you want richer or differently-shaped recall than the hook gives you.

## Recall — load context before answering

Trigger when the prompt references past work, decisions, people, dates, named projects, or asks anything like "what did I / we", "last time", "remember when", "before we", "you mentioned".

### Step 1: Read recall config

Phileas recall is configurable. Resolution order (later wins):

1. Built-in defaults: `mode = "auto"`, `format = "pointer"`, `pipeline = "rerank"`, `top_k = 10`.
2. User config: `~/.phileas/config.toml` `[recall]` section.
3. Project config: nearest `.phileas.toml` walking up from the cwd, `[recall]` section.

Read the project config once per session via `cat .phileas.toml 2>/dev/null` (or walk up to repo root if cwd is nested). If neither file exists, use defaults.

### Step 2: Branch on `mode`

In `mode = "auto"` and `mode = "always"`, a `phileas-hook recall` UserPromptSubmit hook is installed and fires before this skill ever sees the prompt — see "Hook-driven auto firing" below. This skill body runs when you reach for it via explicit `/phileas` invocation, when the user asks for deeper context, or when you decide the hook output isn't enough. So the mode-branching here is mostly defensive:

- **`mode = "never"`** → skip recall entirely. The user has opted out for this project. Return without calling any phileas tool.
- **`mode = "auto"`** (default) → continue.
- **`mode = "always"`** → continue.

### Step 3: Branch on `pipeline`

**Query shape contract (applies to both pipelines).** Phileas treats `recall(query=...)` as a *focused term phrase* — one concept, 1–4 words. The keyword path AND-matches every token against memory summaries, so verbatim user sentences ("what did the user say about Alex and the Q3 budget") AND-match almost nothing on keyword and rely entirely on graph + semantic. The right shape: **extract the named entities and concepts from the prompt first**, then issue one tool call per concept and merge the results by `id`. For a prompt like *"did Alex bring up the Q3 budget at the planning offsite"*: call `about(Alex)`, `recall("Q3 budget")`, `recall("planning offsite")` in parallel — not `recall("did Alex bring up the Q3 budget at the planning offsite")`.

- **`pipeline = "rerank"`** (default) → the hook has already fired with the verbatim prompt and surfaced its result. Use that result as background. If the prompt has clear concepts the hook didn't cover, supplement with focused-term `recall()` / `about()` / `recall_recent()` calls per the contract above.
- **`pipeline = "direct"`** → main agent calls phileas tools directly using a cognitive routing ladder. Pick the tool by query shape; call several in parallel when shapes overlap, then merge results by `id`:
  - **Named entity** in prompt (person, project) → `mcp__phileas__about(name=...)`. Pass the bare name without a leading `@`. Returns all memories linked to that entity in the graph. Cheapest, most precise lookup for "who is X / what about Y".
  - **Explicit date** ("2026-04-14", "Apr 14") → `mcp__phileas__list_day_memories(date="YYYY-MM-DD")`. Every active memory anchored to that day.
  - **Time-relative** ("yesterday", "recently", "last week", "last session") → `mcp__phileas__recall_recent(days=N)`. Top memories per day, newest first.
  - **Topic / concept** with no entity or date anchor → `mcp__phileas__recall(query=<focused term, 1–4 words>)`. Full gather + cross-encoder rerank, ~30 best.
  - **Date range** spanning multiple days → `mcp__phileas__timeline(start=..., end=...)`.

### Step 4: Format output

- **`format = "pointer"`** (default) — emit a short pointer-style brief. One or two sentences summarizing the most relevant memories, followed by their short IDs so the user (or a follow-up tool call) can drill in. Example:

  ```
  <phileas-recall>
  Relevant: Apr 14 design call settled on token-bucket rate limiting (id: 5db9ca0d). Apr 17 commit added the limiter middleware (id: 0f91c891). Use mcp__phileas__about or mcp__phileas__timeline for more.
  </phileas-recall>
  ```

- **`format = "inline"`** — emit the full block matching the legacy hook output: `<phileas-recall>` wrapper, one line per memory with id prefix, type, importance, score, created_at, and summary. Use this when the user has explicitly requested verbose recall output.

### Step 5: Use the context

Treat recalled memories as background context, not as content to recite. Reference them when answering only if directly relevant. Never lead a response with "Based on my memory…" — work the context in naturally.

## Memorize — store new facts

Inline `mcp__phileas__memorize` (and `memorize_batch` for multiple facts from one turn).

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

Pass `memory_type` as one of: `personal`, `event`, `project`, `preference`, `pattern`, `emotional`, `reflection`. Pick the one that best matches how the memory would be recalled later.

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

- **Tag `Person:<user>`** on `profile`, `behavior`, `reflection`, `emotional`, `pattern` memories — things that describe who they are, how they act, or their inner state.
- **Don't tag `Person:<user>`** on `event`, `knowledge`, `project`, `feedback`, `preference` memories — the user is the implicit narrator; the tag adds noise, not signal.

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
