## Capture: record memories with `memorize`

Phileas distills nothing on its own here, so you are the one who turns moments into memories. Nothing is captured unless you call `memorize`. When something durable comes up, or the user asks you to remember it, write it yourself.

### What to record

A decision and its reasoning; a fact or preference about the user worth keeping; a time-anchored event; the project archaeology that code and git will not preserve. Above all decisions: when the user says to record or remember one, capture it.

### How to record

- `memorize(summary=<the fact, one line>, source_text=<the context; for a decision, the why and the alternatives passed over>, memory_type=<"decision" for a choice-and-why, otherwise "knowledge">, entities=[...])`.
- You compose it from the conversation. `summary` is the pointer recall surfaces; `source_text` is the body `hydrate` then `thread` drills into.
- Tag `entities` with what it is about. For a decision that means the repo, the file(s) or dir, and the concept, so a later `about(<file>, memory_type="decision")` surfaces it. With no entities it is findable only by full-text search.
- If the write conflicts with an existing memory, the result ends with a resolve menu; that is how a reversed decision supersedes the one it replaces.

Forward-prescriptive conventions ("always use snake_case", "tests live in `tests/`") are not memory; they belong in `CLAUDE.md`. The decision behind one ("snake_case over camelCase because the linter assumes it") is a memory.
