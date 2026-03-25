# Phileas v2 — Design Document

## Vision

Phileas becomes the **centralized memory layer** for Claude Code. Not just a passive store, but an intelligent system that automatically captures, organizes, and retrieves everything about the user. Replaces specialized skills (/secretary, /butler, etc.) with a unified memory-powered foundation.

## Architecture Overview

See `architecture-v2.svg` for the visual diagram.

### Storage: Three Embedded Databases

| DB | Role | Tech | Data |
|----|------|------|------|
| **Relational** | Metadata, config, audit | SQLite | Timestamps, importance scores, access tracking, processed session index |
| **Vector** | Semantic search | ChromaDB | Embeddings for similarity retrieval, HNSW indexing |
| **Graph** | Relationships & entities | KuzuDB | People, projects, concepts, and typed edges between them |

All three are **embedded** (library, no server, no Docker). Data linked by shared memory ID.

**Data flow example:** Storing "Giao is building Phileas using Python and KuzuDB":
1. **SQLite** — memory item metadata (text, timestamps, importance, access count)
2. **ChromaDB** — embedding vector for semantic search
3. **KuzuDB** — entities + relationships: `(Giao)-[:builds]->(Phileas)`, `(Phileas)-[:uses]->(Python)`, `(Phileas)-[:uses]->(KuzuDB)`

---

## Memory Lifecycle: Three Tiers

### Tier 1: Working Memory (JSONL Pointers)

- **Source:** Claude Code's own conversation logs at `~/.claude/projects/*/*.jsonl`
- **Storage:** SQLite tracks which session files have been processed (pointers, not copies)
- **No data duplication** — Phileas reads JSONL files directly
- **Purpose:** Raw source material for extraction

### Tier 2: Long-term Memory (Extracted Facts)

- **What:** Individual facts, events, preferences, decisions extracted from Tier 1 or captured live
- **Storage:** All three DBs (SQLite metadata + ChromaDB embeddings + KuzuDB entities/relationships)
- **Fields:** summary, memory_type, importance (1-10), access_count, last_accessed, daily_ref, source_session_id
- **Lifecycle:** Permanent, subject to importance decay and eventual consolidation
- **Memory types:** profile, event, knowledge, behavior, reflection

### Tier 3: Core Memory (Consolidated Knowledge)

- **What:** Merged, distilled facts from clusters of Tier 2 memories
- **Storage:** All three DBs
- **Lifecycle:** Permanent, high importance, rarely decays
- **Purpose:** Stable identity, long-running patterns, deep knowledge about the user
- **Example:** "Giao has been building Phileas for 3 months, evolving from simple SQLite to graph-powered architecture" — synthesized from dozens of session memories

### Flow Between Tiers

```
SessionStart hook
  → scan ~/.claude/projects/*/*.jsonl for unprocessed sessions
  → read user/assistant messages from JSONL
  → extract facts, entities, relationships → Tier 2
  → record session ID as "processed" in SQLite

Mid-conversation (Phileas skill)
  → detect important facts/decisions/preferences
  → store directly to Tier 2 (no need for Tier 1 — Claude Code is live)
  → check for contradictions with existing memories

Weekly / manual consolidation
  → cluster related Tier 2 memories by topic/entity
  → merge clusters into Tier 3 summaries
  → mark originals as "consolidated" (kept but deprioritized)
```

---

## Extraction & Hooks

### SessionStart Hook
Fires when a new Claude Code session begins:
1. Scan for unprocessed JSONL conversation files
2. Claude Code reads them and extracts:
   - **Facts** (profile, event, knowledge, behavior, reflection)
   - **Entities** (people, projects, places, tools)
   - **Relationships** (who → knows/builds/uses/feels → what)
   - **Importance score** (1-10)
3. Store as Tier 2 across all three DBs
4. Mark sessions as processed

### Phileas Skill (Mid-conversation)
Upgraded from v1 to be more proactive:
1. Invoked for any conversation touching personal topics, decisions, preferences
2. Stores directly to Tier 2
3. Detects contradictions before storing
4. Updates access_count when recalled memories are useful

### Consolidation (Weekly / Manual)
1. Cluster related Tier 2 memories using graph neighborhoods + embedding similarity
2. Claude Code summarizes each cluster into Tier 3 consolidated memory
3. Original Tier 2 items get `consolidated_into` reference

**Key constraint:** All intelligence runs inside Claude Code sessions — no external API calls needed.

---

## MCP Tools (API Surface)

### Core Memory Tools

| Tool | Purpose | Used by |
|------|---------|---------|
| `memorize` | Store a fact/event/preference with entity extraction | Phileas skill, SessionStart |
| `recall` | Multi-path retrieval (keyword + semantic + graph) | Any skill needing context |
| `forget` | Mark memory as superseded/archived (soft delete) | Conflict resolution |
| `relate` | Create/update entity relationships in graph | Entity extraction |

### Lifecycle Tools

| Tool | Purpose | Used by |
|------|---------|---------|
| `ingest_session` | Read a JSONL file, extract and store memories | SessionStart hook |
| `consolidate` | Cluster and merge Tier 2 → Tier 3 | Weekly schedule / manual |
| `status` | Memory stats: counts per tier, unprocessed sessions, graph size | Debugging |

### Query Tools

| Tool | Purpose | Replaces |
|------|---------|----------|
| `about` | Everything known about a person/project/topic (graph neighborhood) | /butler, people lookups |
| `timeline` | Temporal query — events in a date range | /secretary schedule queries |
| `profile` | User's identity facts | Self-awareness |

### Retrieval Pipeline (inside `recall`)

```
1. Query Parse     → extract intent + entities from query
2. Multi-path Search → keyword (SQLite) + semantic (ChromaDB) + graph (KuzuDB)
3. Score + Rank    → importance × recency × relevance × access_freq
4. Dedupe + Return → top-k unique results
```

**Scoring formula:**
```
score = (similarity × 0.4) + (importance × 0.3) + (recency_decay × 0.2) + (access_freq × 0.1)
```

---

## JSONL Conversation Format

Claude Code stores conversations at `~/.claude/projects/{project-dir}/{session-uuid}.jsonl`.

Each line is a JSON object with:
- `type`: user | assistant | system | progress | file-history-snapshot | queue-operation
- `message`: `{ role, content }` (for user/assistant types)
- `timestamp`, `sessionId`, `uuid`, `cwd`, `gitBranch`

For extraction, we only need `type: user` and `type: assistant` messages.

---

## Graph Schema (KuzuDB)

### Node Types

| Label | Properties | Examples |
|-------|-----------|----------|
| **Person** | name, handle, relationship_to_user | @alice, @bob, "mom" |
| **Project** | name, status, started_at | Phileas, Zettelgraph, genai-portal |
| **Place** | name, type (city/office/home) | Hanoi, "the office", ASA Coworking |
| **Tool** | name, category | Python, KuzuDB, Claude Code |
| **Topic** | name | career, health, relationships |
| **Memory** | id (links to SQLite memory_item) | The actual memory node |

### Edge Types

| Edge | From → To | Example |
|------|-----------|---------|
| **BUILDS** | Person → Project | Giao → Phileas |
| **USES** | Project → Tool | Phileas → KuzuDB |
| **KNOWS** | Person → Person | Giao → Alice |
| **WORKS_AT** | Person → Place | Giao → "the office" |
| **ABOUT** | Memory → any entity | memory_123 → Alice |
| **RELATES_TO** | Memory → Memory | memory_123 → memory_456 |
| **CONTRADICTS** | Memory → Memory | "lives in Hanoi" → "moved to Saigon" |
| **CONSOLIDATED_INTO** | Memory → Memory | Tier 2 → Tier 3 |

### Example Queries (Cypher)

```cypher
-- Everything about a person
MATCH (p:Person {name: 'Alice'})-[r]-(connected)
RETURN p, r, connected

-- What tools do my projects use?
MATCH (u:Person {handle: '@giao'})-[:BUILDS]->(p:Project)-[:USES]->(t:Tool)
RETURN p.name, t.name

-- Find contradicting memories
MATCH (m1:Memory)-[:CONTRADICTS]->(m2:Memory)
RETURN m1.id, m2.id

-- Neighborhood of a memory (for context)
MATCH (m:Memory {id: $memory_id})-[*1..2]-(connected)
RETURN connected
```

---

## What Phileas Replaces

| Current Skill | How Phileas Handles It |
|---------------|----------------------|
| /secretary | `recall("schedule")` + `timeline(today)` + Google Calendar MCP |
| /butler | `about("belongings")` + `recall("maintenance")` |
| /checklist | `recall("checklist routines")` — stored as knowledge memories |
| ~/life/ manual lookups | `about("person_name")` — entities in graph |
| Claude Code auto-memory | `memorize()` with richer extraction + graph relationships |

---

## Importance Scoring & Decay

### Importance Score (1-10, assigned at extraction time)

| Score | Meaning | Examples |
|-------|---------|----------|
| 9-10 | **Identity** — core facts that define the user | Name, role, long-term goals, close relationships |
| 7-8 | **Significant** — major decisions, life events | Changed jobs, started a project, moved cities |
| 5-6 | **Notable** — preferences, opinions, recurring patterns | Prefers morning work, likes coffee, coding style |
| 3-4 | **Context** — useful but not critical | Tried a restaurant, read an article |
| 1-2 | **Ephemeral** — session-specific, likely stale soon | "Currently debugging X", "waiting for Y" |

Claude Code assigns importance during extraction. Tier 3 memories get boosted to minimum 7.

### Recency Decay

```
recency_score = exp(-decay_rate × days_since_last_access)
```

- `decay_rate = 0.01` — slow decay, 50% after ~70 days unused
- Accessing a memory resets the clock (`last_accessed` updates)
- Tier 3 memories: `decay_rate = 0.001` (near-permanent)
- Importance 9-10: `decay_rate = 0.005` (very slow fade)

### Access Frequency

Every time `recall` returns a useful memory:
- `access_count += 1`
- `last_accessed = now()`

### Combined Scoring (inside `recall`)

```
final_score = (similarity × 0.4) + (importance/10 × 0.3) + (recency_score × 0.2) + (log(access_count+1)/5 × 0.1)
```

A 6-month-old identity fact (importance=10, accessed often) still beats a 1-day-old ephemeral note (importance=2, accessed once).

---

## Migration from v1

### What carries over
- SQLite database structure (extended, not replaced)
- MCP server via FastMCP
- `memorize`, `recall`, `profile` tools (upgraded)
- sentence-transformers embeddings

### What changes
- Add ChromaDB + KuzuDB as new storage backends
- Existing memories migrated: embedded into ChromaDB, entities extracted into KuzuDB
- New schema columns: importance, access_count, last_accessed, tier, consolidated_into
- New tools: `forget`, `relate`, `ingest_session`, `consolidate`, `status`, `about`, `timeline`
- SessionStart hook for automatic JSONL ingestion
- Phileas skill upgraded for proactive mid-conversation extraction

### Migration steps
1. Extend SQLite schema (add new columns with defaults)
2. Install ChromaDB + KuzuDB dependencies
3. Re-embed existing memories into ChromaDB
4. Extract entities from existing memories into KuzuDB graph
5. Assign default importance scores based on memory_type
6. Set up SessionStart hook
7. Upgrade Phileas skill

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Relational DB | SQLite | Already used, embedded, reliable |
| Vector DB | ChromaDB | Embedded, Python-native, HNSW indexing |
| Graph DB | KuzuDB | Embedded (no server), Cypher queries, fast |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Already used, local, no API needed |
| MCP Server | FastMCP (Python) | Already used |
| Hooks | Claude Code SessionStart | Built-in, no extra infra |
