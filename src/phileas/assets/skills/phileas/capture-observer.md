## Capture: stream turns, record decisions

There are two ways in. Most of the time you stream the conversation and let Phileas distill it. For a decision the user flags, you write it yourself.

### Ambient capture with `ingest`

Hand conversation turns to Phileas with `ingest`; it distills durable memories from them in the background, with its own model. You do not judge what becomes a memory, you only choose which turns are worth streaming.

- `ingest(content=<the turn, verbatim>, attribution=<self|assistant|source>)`.
- `attribution` is whose words the turn is: `self` (the user's own words, the default), `assistant` (the AI they are talking with), `source` (external material they brought in).
- To keep a conversation's turns together, call `start_thread(client_key="claude_code:<session_id>")` once, keep the returned `thread_id`, and pass it to every `ingest`. The `client_key` makes it resume-safe across a compaction or `--resume`.
- Pipe the turns that carry something durable: personal facts, decisions and their reasons, preferences, time-anchored events, patterns, the project archaeology that code and git will not preserve. Let pure task chatter pass.

### Explicit capture with `memorize`

When the user explicitly asks you to remember or record something, above all a decision (a choice and the reasoning behind it), write it yourself with `memorize` rather than leaving it to the background distill. This is a conclusion you have already judged, so it goes in directly, typed and located:

- `memorize(summary=<the choice, one line>, source_text=<the why: rationale, the alternatives passed over, what it changes>, memory_type="decision", entities=[...])`.
- You compose it from the conversation you are in; no extraction model runs, so it is kept as written. `summary` is the pointer recall surfaces; `source_text` is the body `hydrate` then `thread` drills into.
- Tag `entities` with what the decision governs: the repo, the file(s) or dir, and the concept. A later `about(<file>, memory_type="decision")` then surfaces it when that area comes up again. With no entities it is findable only by full-text search.
- An explicit record request is a `memorize`, not also an `ingest` of the same turn, so the decision cannot land twice.
- If the write conflicts with an existing memory, the result ends with a resolve menu; that is how a reversed decision supersedes the one it replaces.

Forward-prescriptive conventions ("always use snake_case", "tests live in `tests/`") are not memory; they belong in `CLAUDE.md`. The decision behind one ("snake_case over camelCase because the linter assumes it") is a memory.
