# Entity Multi-Type & Disambiguation Redesign

**Date:** 2026-05-01
**Status:** Design — pending decision
**Closes investigation for:** [AA-42](https://linear.app/<workspace>/issue/AA-42) — Entity model collapses multi-type referents (Ownego = Place + Company)

## Problem

Two related failures in the current entity model:

1. **Multi-type fragmentation** — a single real-world referent can legitimately carry multiple types simultaneously. Ownego is `Place` (106 mems) **and** `Company` (10) **and** `Project` (17). Today's schema (`id = "{Type}:{Name}"`) forces these into three separate `Entity` rows with disjoint ABOUT-edge sets. The 2026-04-26 audit found this fragmentation across 66 cross-type clusters touching 34.1% of memories.

2. **Name collision** — distinct referents can share a name. "Apple" the fruit and "Apple" the company are different entities; today they survive only because the LLM happens to extract them with different `type` strings. If both got tagged as `Concept:Apple` or both as `Topic:Apple`, they would silently merge into one row and pollute each other's neighborhoods.

These failures share a root cause: **the schema treats `(type, name)` as identity**, which is wrong on both ends — it splits one referent across multiple rows when types differ, and risks merging unrelated referents when types accidentally match.

## Prior art

This is a textbook **Entity Linking / Named Entity Disambiguation (NED)** problem. The field has converged on one principle:

> **Identity is an opaque id. Names are attributes. Disambiguation happens at extraction time.**

Wikidata is the canonical implementation: every referent has a stable QID (`Q312` = Apple Inc., `Q89` = apple fruit). The string "Apple" is just a label; many QIDs can carry it. Modern systems (BLINK, GENRE, EntGPT) operationalise this as a retrieval problem over a sense inventory, resolved by surrounding context.

References:
- *Knowledge Graphs for Enhancing LLMs in Entity Disambiguation* (arXiv 2505.02737)
- *LLM-empowered knowledge graph construction: A survey* (arXiv 2510.20345)
- BLINK / GENRE / EntGPT — modern entity-linking pipelines

## Options considered

### Option A — Opaque id + `types: list[str]` + extraction-time linking *(recommended)*

Entity table:

```
Entity
  id             UUID    PRIMARY KEY        # opaque, stable, never derived from name
  primary_name   STRING                     # canonical display name
  aliases        STRING                     # JSON list of alternative names
  types          STRING                     # JSON list of type-aspects
  description    STRING                     # one-line disambiguator
  props          STRING                     # existing props blob
```

- **Multi-type referent (Ownego):** one row, `types = ["Place", "Company", "Project"]`. All ABOUT edges land on the same node.
- **Name collision (Apple fruit vs company):** two rows, two uuids, both with `primary_name = "Apple"`. Disambiguated at extraction time by description + context + neighborhood.
- **Type drift (Phileas: Project vs Tool vs Skill):** types accumulate on the same row instead of forking new rows.

### Option B — `types: list[str]` keyed off `name` only

Same multi-type union, but id collapses to `name`. **Rejected** — re-introduces collisions for Apple-fruit vs Apple-company, which Giao explicitly flagged as a non-negotiable.

### Option C — Concept umbrella + `IS_ASPECT_OF` edges

Add a `Concept:Ownego` super-node with REL edges to typed aspects. **Rejected** — doubles graph size, forces every recall path to traverse one extra hop, over-engineered for a personal KG.

### Option D — Accept fragmentation

Do nothing. **Rejected** — costs recall coverage and UI clarity, and the issue is a known degradation that will keep growing as new memories accrue type-drift.

## Recommended schema (Option A details)

```sql
CREATE NODE TABLE Entity (
  id            STRING,          -- uuid4
  primary_name  STRING,
  aliases       STRING DEFAULT '[]',   -- JSON list[string]
  types         STRING DEFAULT '[]',   -- JSON list[string]
  description   STRING DEFAULT '',
  props         STRING DEFAULT '',
  PRIMARY KEY (id)
)
```

Edges unchanged: `ABOUT (Memory→Entity)`, `REL (Entity→Entity, edge_type)`, `MEM_REL (Memory→Memory, edge_type)`.

## Extraction-time linking algorithm

Replaces today's `_resolve_canonical` (which only collapses casing within `(lower(type), lower(name))`).

**Per extracted mention `{name, types, description}`:**

1. **Candidate gather** — `MATCH (e:Entity) WHERE lower(e.primary_name) = lower($name) OR lower($name) IN <aliases lowered>`. Returns 0..N candidates.
2. **Score each candidate** by weighted sum of:
   - `type_overlap` — Jaccard between mention `types` and candidate `types`. Overlap > 0 is a strong positive signal; disjoint types weakly negative (different aspect of same thing is possible — Ownego case — but rarer than collision).
   - `neighborhood_overlap` — fraction of co-occurring entities in this memory that already have ABOUT or REL edges to the candidate.
   - `user_prior` — log-scaled count of existing memories linked to the candidate. Captures "Giao writes about Apple-the-company 50× and Apple-the-fruit 0×" → bias toward the company.
   - `description_similarity` — cosine similarity between mention `description` and candidate `description` (reuse existing embedder).
3. **Decision**:
   - Best candidate score above `LINK_HIGH` → reuse id; union new aliases into `aliases`, union new types into `types`, leave `description` and `primary_name` alone.
   - Best score below `LINK_LOW` or zero candidates → mint new uuid, insert new `Entity` row.
   - Scores between thresholds → mint new entity for safety, log to a review queue (out of scope for v1; can revisit if review queue grows).

Thresholds tuned on a held-out gold set; start with `LINK_HIGH = 0.6, LINK_LOW = 0.3` and iterate.

## Callsite migration

Every `WHERE e.type = $t` becomes `WHERE $t IN <types-as-list>`. Concretely:

| Today | After |
|---|---|
| `_entity_id(type, name)` → string PK | `entity_lookup(name, hint_types, context)` → uuid (entity-linking step) |
| `MATCH (e:Entity) WHERE e.type = $t` | `MATCH (e:Entity) WHERE $t IN <parsed types>` (Kuzu has no native list ops on STRING-encoded JSON; either store types as native LIST[STRING] if Kuzu supports it, or filter in Python after `MATCH`) |
| `find_nodes(type, name)` | `find_nodes(name, type=None)` — type optional, used as a filter not a key |
| `get_top_entities_by_type(t)` | `get_top_entities_by_type(t)` — same signature, query updated to list-membership |
| `link_memory(mid, type, name)` | `link_memory(mid, entity_id)` — caller resolves entity_id first via `entity_lookup` |

Callers in scope (from grep): `engine.py:266-282, 338-340, 1109, 1122-1134, 1197-1201` plus daemon RPC dispatch in `daemon.py:389-405` and proxy in `graph_proxy.py:55-110`.

## Migration plan

One-shot script, daemon stopped, pattern matches existing `migrate_entity_duplicates.py`:

1. **Snapshot** — read all `Entity` rows, all ABOUT edges, all REL edges. ~1k entities, ~5k edges based on current scale.
2. **Cluster cross-type duplicates** — for each `(lower(name))` group with >1 row, decide:
   - Apply `entity_merge_overrides.yml` for known clusters (Ownego, Phileas, Polar, etc.)
   - Heuristic for unflagged: merge if neighborhood Jaccard > 0.3 *and* at least one shared neighbor entity. Otherwise leave as separate (collision case).
   - Default-conservative: when uncertain, do **not** merge. Better to leave a duplicate than to fuse Apple-fruit into Apple-company.
3. **Mint uuids** — new `id` per resolved cluster; `primary_name` = name with most ABOUT edges; `aliases` = union of satellite names + existing aliases; `types` = union of satellite types; `description` = generate via LLM from a sample of linked memories (or leave empty for v1, fill lazily).
4. **Rewrite edges** — re-target ABOUT and REL onto new uuids.
5. **Drop old `Entity` table, swap in new schema, re-insert.**
6. **Verify** — counts match (memories unchanged, ABOUT edge count unchanged, entity count ≤ old count).

Daemon downtime estimate: <1 minute for current data size.

## Risks

- **Kuzu list types** — Kuzu's column types are limited; if native `LIST[STRING]` isn't supported on `Entity.types` we fall back to JSON-string + Python-side filtering. Check first; not a blocker.
- **Description quality at extraction time** — the LLM may produce descriptions that drift across sessions for the same entity. Mitigation: only set `description` once at entity creation; subsequent mentions don't overwrite. Lossy but stable.
- **Neighborhood-overlap score is noisy on cold-start** — for the first few memories about a new entity, neighborhood is empty. Falls back to type + description signals only. Acceptable.
- **Review queue for mid-confidence is deferred** — v1 mints new entity, which means some legitimate merges happen as duplicates initially. Audit script can flag them later.

## Acceptance criteria for shipping (separate Linear issue)

- All current `Entity` rows migrated, no ABOUT/REL edge loss.
- Audit script (`audit_entity_duplicates.py`) reports zero cross-type fragmentation for the seeded override clusters (Ownego, Phileas, Polar, etc.).
- Recall integration tests pass (entity-overlap signal now lights up across former type-fragments).
- Entity explorer UI shows one card per uuid; types displayed as a chip list.
- New session ingestion uses `entity_lookup` for every mention; `_entity_id` string-concat path removed.

## Out of scope

- Wikidata cross-linking (would be useful for famous entities but adds external dependency; revisit later).
- Active review queue UI for mid-confidence linking decisions.
- Cross-language alias resolution (Vietnamese → English) — separate concern, tracked in 2026-04-09 migration doc.

## Decision required

Sign off on Option A, then file an `AA-` implementation issue covering schema swap + linking algorithm + migration. The investigation half of AA-42 closes with this doc.
