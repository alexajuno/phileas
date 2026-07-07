# Recall eval fixture — provenance

The fixture is the **`mara-eval` profile store** at `~/.phileas-mara-eval`
(`memory.db` + `chroma` + `graph`), built by `seed.py` from the cold-start Mara
corpus (the sibling `coldstart` eval). Like the rest of `evals/`, it is local and
not committed: the binary `chroma`/`kuzu` stores are version-sensitive and would
bloat git, so the fixture is reproduced from text on demand instead.

## What it is

- Source corpus: the sibling `coldstart` eval's session transcripts + extraction JSON (30 sessions, 184 memories, 2026-03-22 → 2026-09-30). Ground truth: `coldstart/persona.md`.
- Seeded chronologically, one Event/thread per session, entity names fed raw (the linker runs for real). See `seed.py`.
- Built shape (for sanity): 184 memories, 184 memory vectors, 30 event vectors, ~249 graph nodes, ~454 edges.

## How the gold set references it

Memory ids are random uuid4 assigned at write time, so they change on every
re-seed. The gold set (`goldset.json`) therefore anchors each relevant memory by
its **content text**, and the A/B runner resolves anchors to live ids against the
seeded store at load. A re-seed does not rot the gold set; only a change to the
corpus text does.

## Rebuild

Run `seed.py` via the project venv python (pass `--reset` to rebuild from
scratch). It refuses to run against any home other than `~/.phileas-mara-eval`,
so it can never touch the real `~/.phileas` graph.

## Versioning

`fixture_version` in `goldset.json` is the corpus fingerprint: the sha256 of the
concatenated, name-sorted extraction files. The runner recomputes it and warns if
the gold set was authored against a different corpus revision.
