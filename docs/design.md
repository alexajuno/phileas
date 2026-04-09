# Phileas — Design Document

## Vision

Phileas becomes the **centralized memory layer** for Claude Code. Not just a passive store, but an intelligent system that automatically captures, organizes, and retrieves everything about the user. Replaces specialized skills (/secretary, /butler, etc.) with a unified memory-powered foundation.

## Architecture Overview

See `architecture.svg` for the visual diagram.

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
- **Fields:** summary, memory_type, importance (1-10), access_count, last_accessed, daily_ref
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
Upgraded to be more proactive:
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

| Tool | Signature | Purpose |
|------|-----------|---------|
| `memorize` | `(summary, memory_type, importance, category?, daily_ref?, entities?, relationships?)` | Store a fact with optional entity/relationship data. If `entities`/`relationships` provided, writes to KuzuDB in the same call. Otherwise Claude Code calls `relate` separately. |
| `recall` | `(query, top_k=5, memory_type?, min_importance?)` | Multi-path retrieval with scoring. See pipeline below. |
| `forget` | `(memory_id, reason?)` | Sets `status='archived'` + `archived_at` timestamp. `recall` filters these out. Graph edges preserved for audit. |
| `relate` | `(from_name, from_type, edge_type, to_name, to_type, memory_id?)` | Upsert nodes + create edge in KuzuDB. Auto-creates nodes if they don't exist. Idempotent — duplicate edges are no-ops. |

### Lifecycle Tools

| Tool | Signature | Purpose |
|------|-----------|---------|
| `ingest_session` | `(session_path, max_tokens=50000)` | Read a JSONL file, return extracted user/assistant messages as text for Claude Code to process. Marks session as processed only after Claude Code calls `memorize` with the results. Max N=3 sessions per SessionStart to avoid blocking. |
| `consolidate` | `(min_cluster_size=3, max_clusters=10)` | Find clusters of related Tier 2 memories. Returns clusters for Claude Code to summarize, then Claude calls `memorize` with tier=3. Triggered opportunistically when >50 unconsolidated Tier 2 memories exist. |
| `status` | `()` | Memory stats: counts per tier, unprocessed sessions, graph node/edge counts, oldest unaccessed memory. |

### Query Tools

| Tool | Signature | Purpose |
|------|-----------|---------|
| `about` | `(name, entity_type?=None)` | Query KuzuDB for entity + 1-hop neighborhood, return connected memories from SQLite. If `entity_type` omitted, searches all node types. |
| `timeline` | `(start_date, end_date?=None)` | Query memories by `daily_ref` and `happened_at` in date range. Returns chronologically sorted. |
| `recall` | `(query, memory_type="profile")` | Profile memories via ranked recall (replaces old unfiltered `profile()` tool). |

### Retrieval Pipeline (inside `recall`)

```
1. Query Parse     → extract intent + entity names from query text
2. Multi-path Search:
   a. Keyword search (SQLite LIKE) → match on summary text
   b. Semantic search (ChromaDB) → cosine similarity on embeddings
   c. Graph search (KuzuDB) → extract entity names from query,
      find matching nodes, follow ABOUT edges to Memory nodes
3. Score + Rank    → weighted combination (see formula)
4. Dedupe          → merge results by memory ID, keep highest score
5. Filter          → exclude status='archived', apply min_importance
6. Return          → top-k results with scores
```

**Scoring formula:**
```
final_score = (similarity × 0.4) + (importance/10 × 0.3) + (recency_score × 0.2) + (log(access_count+1)/5 × 0.1)
```

### Deduplication

Before storing via `memorize`, check ChromaDB for existing memories with cosine similarity > 0.95. If found, return existing memory ID instead of creating a duplicate. This prevents double-storage from live capture + JSONL ingestion of the same conversation.

### Source of Truth

**SQLite is canonical.** ChromaDB and KuzuDB are derived indexes that can be rebuilt from SQLite data. If they get corrupted or out of sync, a `rebuild_indexes` command re-embeds all memories into ChromaDB and re-extracts entities into KuzuDB.

### Contradiction Detection

Contradictions are detected by **Claude Code, not the MCP server** (no LLM inside the server). The workflow:
1. Claude Code calls `recall` before `memorize` to check for conflicting facts
2. If a contradiction is found, Claude Code calls `relate(memory_A, "Memory", "CONTRADICTS", memory_B, "Memory")`
3. Optionally calls `forget(old_memory_id, reason="superseded by newer info")`

### V1 Tool Changes

| V1 Tool | V2 Status |
|---------|-----------|
| `memorize` | Upgraded (new params) |
| `recall` | Upgraded (multi-path + scoring) |
| `profile` | Removed — use `recall(query, memory_type="profile")` instead |
| `digest` | Removed — replaced by `ingest_session` |
| `categories` | Removed — replaced by `about` + graph queries |

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

## Migration from Current

### What carries over
- SQLite database structure (extended, not replaced)
- MCP server via FastMCP
- `memorize`, `recall` tools (upgraded); `profile` removed in favor of `recall(memory_type="profile")`
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
