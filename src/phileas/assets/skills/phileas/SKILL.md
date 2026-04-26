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
- `pipeline = "agent_summarizer"` → the hook calls `recall_raw` to confirm there's a non-empty candidate pool, then injects a `<phileas-recall-task>` directive. **When you see that directive, dispatch the `phileas-recall` subagent via the `Task` tool** with `subagent_type="phileas-recall"` and `prompt="Query: <verbatim user prompt>"`. The subagent will gather the pool itself, rank it, and return a `<phileas-recall>` block — surface that block as context.

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

- **`pipeline = "rerank"`** (default) → call `mcp__phileas__recall(query=<user_prompt>, top_k=<top_k>)`. This runs the existing gather → cross-encoder rerank → MMR pipeline server-side and returns ~`top_k` memories.
- **`pipeline = "agent_summarizer"`** → call `mcp__phileas__recall_raw(query=<user_prompt>)` to get the full Stage-1 candidate pool (no rerank, typically up to ~1000 candidates). Then invoke the `phileas-recall` subagent via the Task tool (`subagent_type="phileas-recall"`), passing the user prompt and the raw pool in the prompt body, e.g. `Query: <user_prompt>\n\nRaw pool:\n<json or compact list of pool items>`. The subagent returns a `<phileas-recall>` block — emit it directly as your skill output. If `recall_raw` returns an empty list, skip the subagent and emit an empty-pool note in the format from step 4. If the subagent is unavailable for any reason, fall back to `pipeline = "rerank"` and note it.

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

Save facts that will still be useful in a future conversation:

- **Personal facts** the user states about themselves, people in their life, or their situation.
- **Preferences** about tools, workflow, tone, collaboration style.
- **Decisions** — especially ones with a stated reason ("we're going with X because Y").
- **Events** with a time anchor ("shipped v0.1.0 on Apr 4", "trip to Tokyo next month").
- **Patterns** observed over time — recurring frustrations, emotional throughlines, habits.
- **Project state** not derivable from code or git (ownership, blockers, why a design was chosen).

### What NOT to save

- Code conventions, file paths, architecture — re-readable from the repo.
- Git history, recent commits, who-changed-what — `git log`/`git blame` are authoritative.
- Transient task state (current in-progress step, conversation context, temp debugging notes).
- Anything already in `CLAUDE.md` or the repo's own docs.
- Fix recipes from debugging — the commit explains the fix; don't mirror it in memory.

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

### Batching

When a single turn yields several distinct memories, prefer `memorize_batch` over N sequential `memorize` calls — it's faster and cheaper.

### Entity tagging

When calling `memorize` with `entities=[...]`, only tag an entity whose presence a future `about(name=<entity>)` query would find useful. A tag says "this memory is *about* this entity," not "this entity appears in this memory."

**The user-entity trap.** Nearly every memory is implicitly authored *by* the user. Tagging `Person:<user>` on every one makes `about('<user>')` return the whole activity log. Only tag the user when the memory is genuinely identity-shaped:

- **Tag `Person:<user>`** on `profile`, `behavior`, `reflection`, `emotional`, `pattern` memories — things that describe who they are, how they act, or their inner state.
- **Don't tag `Person:<user>`** on `event`, `knowledge`, `project`, `feedback`, `preference` memories — the user is the implicit narrator; the tag adds noise, not signal.

**Other people and entities** (colleagues, partners, projects, tools) can be tagged freely — they're not implicit narrators, so `about(them)` is a useful retrieval primitive.
