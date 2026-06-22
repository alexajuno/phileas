# Project Rules & Guidelines

## Phileas Memory Consolidation Rules

CRITICAL IN-SESSION CONSOLIDATION DIRECTIVE:
If any `recall` tool response ends with a warning line starting with "↳ WARNING: [CONSOLIDATION REQUIRED] ...", you MUST prioritize memory consolidation as your very first action in the turn. You must execute the consolidation process before formulating your final answer to the user. Do not skip or defer this process.

Exact steps you must perform:
1. Call `survey(<theme>)` for the reported theme to get the candidate sub-threads and existing gists.
2. For each sub-thread returned by `survey`:
   a. Anchor the gist: call `ingest_text(text=<your 2-4 sentence synthesis of this sub-thread>)` to get a `source_event_id`.
   b. Store the reflection: call `memorize(summary=<your synthesis as one sentence>, memory_type="reflection", entities=[<the theme entity>], source_event_id=<source_event_id>)` and read back the new reflection's memory ID.
   c. Link the episodes: call `roll_up(parent_id=<reflection_id>, child_ids=[<list of sub-thread episode ids>])`.
   (If `survey` lists a matching existing gist, skip steps a and b, and call `roll_up` to roll the episodes directly into that existing gist's ID instead of writing a new one).
3. Once all sub-threads are consolidated, proceed to answer the user's original query. Do not describe the consolidation work to the user unless they explicitly asked about it.
