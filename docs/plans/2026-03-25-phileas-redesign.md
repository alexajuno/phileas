# Phileas Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Phileas from a simple SQLite memory store into a three-database (SQLite + ChromaDB + KuzuDB) memory system with tiered lifecycle, importance scoring, and hook-driven extraction.

**Architecture:** Three embedded databases — SQLite (canonical metadata), ChromaDB (vector search), KuzuDB (entity graph). Three memory tiers — Working (JSONL pointers), Long-term (extracted facts), Core (consolidated). All intelligence runs inside Claude Code sessions via hooks and skills.

**Tech Stack:** Python 3.14, FastMCP, SQLite, ChromaDB, KuzuDB, sentence-transformers (all-MiniLM-L6-v2)

**Spec:** `docs/design.md`

---

## File Structure

### New files to create

| File | Responsibility |
|------|---------------|
| `src/phileas/graph.py` | KuzuDB wrapper — schema creation, node/edge CRUD, Cypher queries |
| `src/phileas/vector.py` | ChromaDB wrapper — collection management, embed/search/dedup |
| `src/phileas/scoring.py` | Importance scoring, recency decay, combined score calculation |
| `src/phileas/ingest.py` | JSONL parser — read Claude Code conversation files, extract messages |
| `tests/test_models.py` | Tests for data models |
| `tests/test_db.py` | Tests for SQLite operations |
| `tests/test_graph.py` | Tests for KuzuDB operations |
| `tests/test_vector.py` | Tests for ChromaDB operations |
| `tests/test_scoring.py` | Tests for scoring logic |
| `tests/test_ingest.py` | Tests for JSONL parsing |
| `tests/test_engine.py` | Tests for engine (integration across all backends) |
| `tests/test_server.py` | Tests for MCP tool functions |
| `tests/conftest.py` | Shared fixtures (temp DB paths, sample data) |

### Files to modify

| File | Changes |
|------|---------|
| `src/phileas/models.py` | Add new fields (importance, access_count, tier, status, etc.), remove Category/CategoryItem |
| `src/phileas/db.py` | New schema (processed_sessions table, new columns), remove category tables, update queries |
| `src/phileas/engine.py` | Rewrite to orchestrate all three backends, new recall pipeline with scoring |
| `src/phileas/server.py` | New MCP tools (forget, relate, ingest_session, consolidate, status, about, timeline), remove digest/categories |
| `pyproject.toml` | Add chromadb, kuzu, pytest dependencies |

---

### Task 1: Add dependencies and test infrastructure

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add new dependencies to pyproject.toml**

```toml
# In [project] dependencies, replace the current list with:
dependencies = [
    "mcp[cli]",
    "sentence-transformers>=5.3.0",
    "chromadb>=1.0.0",
    "kuzu>=0.8.0",
]

# In [dependency-groups], add pytest:
[dependency-groups]
dev = ["ruff", "pytest"]
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync`
Expected: All packages install successfully

- [ ] **Step 3: Create test conftest with shared fixtures**

Create `tests/conftest.py`:

```python
"""Shared test fixtures for Phileas."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory, cleaned up after test."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sqlite_path(tmp_dir):
    return tmp_dir / "test.db"


@pytest.fixture
def kuzu_path(tmp_dir):
    return tmp_dir / "graph"


@pytest.fixture
def chroma_path(tmp_dir):
    return tmp_dir / "chroma"
```

- [ ] **Step 4: Verify pytest runs**

Run: `uv run pytest tests/ -v --co`
Expected: Collects 0 tests, no errors

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/conftest.py
git commit -m "chore: add chromadb, kuzu, pytest dependencies and test fixtures"
```

---

### Task 2: Update data models

**Files:**
- Modify: `src/phileas/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write tests for updated models**

Create `tests/test_models.py`:

```python
"""Tests for Phileas data models."""

from phileas.models import MemoryItem


def test_memory_item_defaults():
    item = MemoryItem(summary="test fact")
    assert item.summary == "test fact"
    assert item.memory_type == "knowledge"
    assert item.importance == 5
    assert item.access_count == 0
    assert item.tier == 2
    assert item.status == "active"
    assert item.last_accessed is None
    assert item.consolidated_into is None
    assert item.source_session_id is None
    assert item.id  # UUID generated


def test_memory_item_custom_fields():
    item = MemoryItem(
        summary="identity fact",
        memory_type="profile",
        importance=9,
        tier=3,
    )
    assert item.importance == 9
    assert item.tier == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — MemoryItem missing new fields

- [ ] **Step 3: Update models.py**

Rewrite `src/phileas/models.py`:

```python
"""Core data models for Phileas memory system.

Three-tier memory hierarchy:
  Tier 1: JSONL pointers (processed_sessions table)
  Tier 2: Extracted facts (memory_items table + ChromaDB + KuzuDB)
  Tier 3: Consolidated knowledge (same tables, tier=3)
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

MemoryType = Literal[
    "profile",     # who the user is
    "event",       # things that happened
    "knowledge",   # things the user knows or cares about
    "behavior",    # patterns and preferences
    "reflection",  # higher-level inferences
]

MemoryStatus = Literal["active", "archived"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class MemoryItem:
    """A structured memory. The core unit of Phileas."""

    id: str = field(default_factory=_uuid)
    summary: str = ""
    memory_type: MemoryType = "knowledge"
    importance: int = 5  # 1-10 scale
    tier: int = 2  # 2=long-term, 3=consolidated
    status: MemoryStatus = "active"
    access_count: int = 0
    last_accessed: datetime | None = None
    daily_ref: str | None = None
    source_session_id: str | None = None
    consolidated_into: str | None = None  # memory ID of tier-3 parent
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Run ruff**

Run: `uv run ruff check src/phileas/models.py && uv run ruff format src/phileas/models.py`

- [ ] **Step 6: Commit**

```bash
git add src/phileas/models.py tests/test_models.py
git commit -m "refactor: update MemoryItem model with importance, tier, status fields"
```

---

### Task 3: Rewrite SQLite schema and database layer

**Files:**
- Modify: `src/phileas/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write tests for new SQLite operations**

Create `tests/test_db.py`:

```python
"""Tests for SQLite database operations."""

from phileas.db import Database
from phileas.models import MemoryItem


def test_save_and_get_item(sqlite_path):
    db = Database(path=sqlite_path)
    item = MemoryItem(summary="Giao likes coffee", memory_type="behavior", importance=5)
    db.save_item(item)
    loaded = db.get_item(item.id)
    assert loaded is not None
    assert loaded.summary == "Giao likes coffee"
    assert loaded.importance == 5
    db.close()


def test_archive_item(sqlite_path):
    db = Database(path=sqlite_path)
    item = MemoryItem(summary="old fact")
    db.save_item(item)
    db.archive_item(item.id, reason="superseded")
    loaded = db.get_item(item.id)
    assert loaded.status == "archived"
    db.close()


def test_get_active_items_excludes_archived(sqlite_path):
    db = Database(path=sqlite_path)
    active = MemoryItem(summary="active fact")
    archived = MemoryItem(summary="archived fact")
    db.save_item(active)
    db.save_item(archived)
    db.archive_item(archived.id)
    items = db.get_active_items()
    assert len(items) == 1
    assert items[0].id == active.id
    db.close()


def test_keyword_search(sqlite_path):
    db = Database(path=sqlite_path)
    db.save_item(MemoryItem(summary="Giao is building Phileas"))
    db.save_item(MemoryItem(summary="Alice works at a startup"))
    results = db.search_by_keyword("Phileas", top_k=5)
    assert len(results) == 1
    assert "Phileas" in results[0].summary
    db.close()


def test_processed_sessions(sqlite_path):
    db = Database(path=sqlite_path)
    assert not db.is_session_processed("session-123")
    db.mark_session_processed("session-123", "/path/to/file.jsonl")
    assert db.is_session_processed("session-123")
    db.close()


def test_bump_access(sqlite_path):
    db = Database(path=sqlite_path)
    item = MemoryItem(summary="test")
    db.save_item(item)
    db.bump_access(item.id)
    loaded = db.get_item(item.id)
    assert loaded.access_count == 1
    assert loaded.last_accessed is not None
    db.close()


def test_get_items_by_tier(sqlite_path):
    db = Database(path=sqlite_path)
    db.save_item(MemoryItem(summary="long-term", tier=2))
    db.save_item(MemoryItem(summary="core", tier=3))
    tier2 = db.get_items_by_tier(2)
    tier3 = db.get_items_by_tier(3)
    assert len(tier2) == 1
    assert len(tier3) == 1
    assert tier2[0].summary == "long-term"
    db.close()


def test_status_counts(sqlite_path):
    db = Database(path=sqlite_path)
    db.save_item(MemoryItem(summary="a", tier=2))
    db.save_item(MemoryItem(summary="b", tier=2))
    db.save_item(MemoryItem(summary="c", tier=3))
    counts = db.get_counts()
    assert counts["tier2"] == 2
    assert counts["tier3"] == 1
    assert counts["total"] == 3
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL — missing methods

- [ ] **Step 3: Rewrite db.py**

Rewrite `src/phileas/db.py`:

```python
"""SQLite storage backend for Phileas.

Canonical data store. ChromaDB and KuzuDB are derived indexes
that can be rebuilt from this database.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from phileas.models import MemoryItem

DEFAULT_DB_PATH = Path.home() / ".phileas" / "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 5,
    tier INTEGER NOT NULL DEFAULT 2,
    status TEXT NOT NULL DEFAULT 'active',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    daily_ref TEXT,
    source_session_id TEXT,
    consolidated_into TEXT REFERENCES memory_items(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_sessions (
    session_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_status ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_items_tier ON memory_items(tier);
CREATE INDEX IF NOT EXISTS idx_items_type ON memory_items(memory_type);
CREATE INDEX IF NOT EXISTS idx_items_daily_ref ON memory_items(daily_ref);
"""


class Database:
    def __init__(self, path: Path = DEFAULT_DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    # --- Memory Items ---

    def save_item(self, item: MemoryItem) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO memory_items
               (id, summary, memory_type, importance, tier, status,
                access_count, last_accessed, daily_ref, source_session_id,
                consolidated_into, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.id,
                item.summary,
                item.memory_type,
                item.importance,
                item.tier,
                item.status,
                item.access_count,
                item.last_accessed.isoformat() if item.last_accessed else None,
                item.daily_ref,
                item.source_session_id,
                item.consolidated_into,
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
            ),
        )
        self.conn.commit()

    def get_item(self, item_id: str) -> MemoryItem | None:
        row = self.conn.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    def get_active_items(self) -> list[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memory_items WHERE status = 'active' ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_items_by_type(self, memory_type: str) -> list[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memory_items WHERE memory_type = ? AND status = 'active' ORDER BY created_at DESC",
            (memory_type,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_items_by_tier(self, tier: int) -> list[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memory_items WHERE tier = ? AND status = 'active' ORDER BY created_at DESC",
            (tier,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def search_by_keyword(self, query: str, top_k: int = 10) -> list[MemoryItem]:
        """Keyword search using SQLite LIKE. Splits query into words, scores by match count."""
        words = query.lower().split()
        if not words:
            return self.get_active_items()[:top_k]

        conditions = " OR ".join(["LOWER(summary) LIKE ?" for _ in words])
        score_expr = " + ".join(
            ["(CASE WHEN LOWER(summary) LIKE ? THEN 1 ELSE 0 END)" for _ in words]
        )
        params = [f"%{w}%" for w in words]

        rows = self.conn.execute(
            f"""SELECT *, ({score_expr}) as match_count
            FROM memory_items
            WHERE status = 'active' AND ({conditions})
            ORDER BY match_count DESC, created_at DESC
            LIMIT ?""",
            params + params + [top_k],
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def archive_item(self, item_id: str, reason: str | None = None) -> None:
        self.conn.execute(
            "UPDATE memory_items SET status = 'archived', updated_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), item_id),
        )
        self.conn.commit()

    def bump_access(self, item_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE memory_items SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, item_id),
        )
        self.conn.commit()

    def get_counts(self) -> dict:
        row = self.conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN tier = 2 AND status = 'active' THEN 1 ELSE 0 END) as tier2,
                SUM(CASE WHEN tier = 3 AND status = 'active' THEN 1 ELSE 0 END) as tier3,
                SUM(CASE WHEN status = 'archived' THEN 1 ELSE 0 END) as archived
            FROM memory_items"""
        ).fetchone()
        return {"total": row["total"], "tier2": row["tier2"], "tier3": row["tier3"], "archived": row["archived"]}

    # --- Processed Sessions ---

    def is_session_processed(self, session_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM processed_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def mark_session_processed(self, session_id: str, file_path: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO processed_sessions (session_id, file_path, processed_at) VALUES (?, ?, ?)",
            (session_id, file_path, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def get_unprocessed_session_count(self) -> int:
        """Count is calculated by caller scanning JSONL files vs processed_sessions table."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM processed_sessions").fetchone()
        return row["cnt"]

    # --- Timeline ---

    def get_items_by_date_range(self, start_date: str, end_date: str | None = None) -> list[MemoryItem]:
        if end_date:
            rows = self.conn.execute(
                """SELECT * FROM memory_items
                WHERE status = 'active' AND daily_ref >= ? AND daily_ref <= ?
                ORDER BY daily_ref ASC, created_at ASC""",
                (start_date, end_date),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM memory_items
                WHERE status = 'active' AND daily_ref = ?
                ORDER BY created_at ASC""",
                (start_date,),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    # --- Internal ---

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        last_accessed = None
        if row["last_accessed"]:
            last_accessed = datetime.fromisoformat(row["last_accessed"])
        return MemoryItem(
            id=row["id"],
            summary=row["summary"],
            memory_type=row["memory_type"],
            importance=row["importance"],
            tier=row["tier"],
            status=row["status"],
            access_count=row["access_count"],
            last_accessed=last_accessed,
            daily_ref=row["daily_ref"],
            source_session_id=row["source_session_id"],
            consolidated_into=row["consolidated_into"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff**

Run: `uv run ruff check src/phileas/db.py && uv run ruff format src/phileas/db.py`

- [ ] **Step 6: Commit**

```bash
git add src/phileas/db.py tests/test_db.py
git commit -m "refactor: rewrite SQLite schema with importance, tier, status, processed_sessions"
```

---

### Task 4: ChromaDB vector store wrapper

**Files:**
- Create: `src/phileas/vector.py`
- Create: `tests/test_vector.py`

- [ ] **Step 1: Write tests**

Create `tests/test_vector.py`:

```python
"""Tests for ChromaDB vector store."""

from phileas.vector import VectorStore


def test_add_and_search(chroma_path):
    vs = VectorStore(path=chroma_path)
    vs.add("mem-1", "Giao is building Phileas with Python")
    vs.add("mem-2", "Alice works at a robotics startup")
    results = vs.search("Python programming projects", top_k=2)
    assert len(results) > 0
    # mem-1 should rank higher for Python-related query
    assert results[0][0] == "mem-1"
    vs.close()


def test_dedup_check(chroma_path):
    vs = VectorStore(path=chroma_path)
    vs.add("mem-1", "Giao likes coffee in the morning")
    duplicate = vs.find_duplicate("Giao enjoys morning coffee", threshold=0.85)
    assert duplicate is not None
    assert duplicate == "mem-1"
    vs.close()


def test_no_false_dedup(chroma_path):
    vs = VectorStore(path=chroma_path)
    vs.add("mem-1", "Giao likes coffee in the morning")
    result = vs.find_duplicate("Alice works at a robotics startup", threshold=0.85)
    assert result is None
    vs.close()


def test_delete(chroma_path):
    vs = VectorStore(path=chroma_path)
    vs.add("mem-1", "test memory")
    vs.delete("mem-1")
    results = vs.search("test memory", top_k=5)
    assert len(results) == 0
    vs.close()


def test_count(chroma_path):
    vs = VectorStore(path=chroma_path)
    vs.add("mem-1", "first")
    vs.add("mem-2", "second")
    assert vs.count() == 2
    vs.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vector.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement vector.py**

Create `src/phileas/vector.py`:

```python
"""ChromaDB vector store for semantic search.

A derived index — can be rebuilt from SQLite data.
Handles embedding generation internally via sentence-transformers.
"""

from pathlib import Path

import chromadb

DEFAULT_CHROMA_PATH = Path.home() / ".phileas" / "chroma"
COLLECTION_NAME = "memories"


class VectorStore:
    def __init__(self, path: Path = DEFAULT_CHROMA_PATH):
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self):
        pass  # ChromaDB PersistentClient doesn't need explicit close

    def add(self, memory_id: str, text: str) -> None:
        """Add or update a memory embedding."""
        self._collection.upsert(ids=[memory_id], documents=[text])

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search by semantic similarity. Returns [(memory_id, score)]."""
        if self._collection.count() == 0:
            return []
        results = self._collection.query(query_texts=[query], n_results=min(top_k, self._collection.count()))
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        # ChromaDB returns distances (lower = closer for cosine). Convert to similarity.
        return [(id_, 1.0 - dist) for id_, dist in zip(ids, distances)]

    def find_duplicate(self, text: str, threshold: float = 0.95) -> str | None:
        """Check if a near-duplicate exists. Returns memory_id if found."""
        if self._collection.count() == 0:
            return None
        results = self._collection.query(query_texts=[text], n_results=1)
        if not results["ids"] or not results["ids"][0]:
            return None
        dist = results["distances"][0][0]
        similarity = 1.0 - dist
        if similarity >= threshold:
            return results["ids"][0][0]
        return None

    def delete(self, memory_id: str) -> None:
        self._collection.delete(ids=[memory_id])

    def count(self) -> int:
        return self._collection.count()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_vector.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff and commit**

```bash
uv run ruff check src/phileas/vector.py && uv run ruff format src/phileas/vector.py
git add src/phileas/vector.py tests/test_vector.py
git commit -m "feat: add ChromaDB vector store with semantic search and dedup"
```

---

### Task 5: KuzuDB graph store wrapper

**Files:**
- Create: `src/phileas/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write tests**

Create `tests/test_graph.py`:

```python
"""Tests for KuzuDB graph store."""

from phileas.graph import GraphStore


def test_upsert_node_and_query(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao", {"handle": "@giao"})
    nodes = gs.find_nodes("Person", "Giao")
    assert len(nodes) == 1
    assert nodes[0]["name"] == "Giao"
    gs.close()


def test_create_edge(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Phileas")
    gs.create_edge("Person", "Giao", "BUILDS", "Project", "Phileas")
    neighbors = gs.get_neighborhood("Person", "Giao")
    assert len(neighbors) > 0
    gs.close()


def test_duplicate_edge_is_noop(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Phileas")
    gs.create_edge("Person", "Giao", "BUILDS", "Project", "Phileas")
    gs.create_edge("Person", "Giao", "BUILDS", "Project", "Phileas")
    neighbors = gs.get_neighborhood("Person", "Giao")
    # Should still only have one connection, not duplicate
    project_names = [n["name"] for n in neighbors if n.get("_label") == "Project"]
    assert project_names.count("Phileas") == 1
    gs.close()


def test_link_memory_to_entity(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Alice")
    gs.link_memory("mem-123", "Person", "Alice")
    memories = gs.get_memories_about("Person", "Alice")
    assert "mem-123" in memories
    gs.close()


def test_find_nodes_by_name(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Giao-Bot")
    # Searching across all types
    results = gs.search_nodes("Giao")
    assert len(results) >= 1
    gs.close()


def test_get_stats(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Phileas")
    gs.create_edge("Person", "Giao", "BUILDS", "Project", "Phileas")
    stats = gs.get_stats()
    assert stats["nodes"] >= 2
    assert stats["edges"] >= 1
    gs.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL

- [ ] **Step 3: Implement graph.py**

Create `src/phileas/graph.py`. This is the most complex new module — it wraps KuzuDB with the schema from the design doc (Person, Project, Place, Tool, Topic, Memory nodes and typed edges). The implementation needs to:

1. Create the schema on init (node tables + edge tables)
2. Upsert nodes by name (create if not exists)
3. Create edges idempotently
4. Link Memory nodes to entity nodes via ABOUT edges
5. Query neighborhoods (1-hop from a node)
6. Search nodes by name across all types
7. Return memory IDs connected to an entity

Key implementation notes:
- KuzuDB uses Cypher syntax. Node tables are created with `CREATE NODE TABLE IF NOT EXISTS`.
- Edge tables require `CREATE REL TABLE IF NOT EXISTS ... (FROM X TO Y)`.
- Since edges can go between multiple node types (e.g., ABOUT goes from Memory to any entity), you need one REL TABLE per from-to pair, or use REL TABLE GROUP.
- Use `MERGE` for upsert semantics.

```python
"""KuzuDB graph store for entity relationships.

A derived index — can be rebuilt by re-extracting entities from SQLite memories.
Stores people, projects, places, tools, topics, and memory linkages.
"""

from pathlib import Path

import kuzu

DEFAULT_GRAPH_PATH = Path.home() / ".phileas" / "graph"

NODE_TYPES = ["Person", "Project", "Place", "Tool", "Topic", "Memory"]

# Edge definitions: (from_type, edge_name, to_type)
EDGE_TYPES = [
    ("Person", "BUILDS", "Project"),
    ("Person", "KNOWS", "Person"),
    ("Person", "WORKS_AT", "Place"),
    ("Project", "USES", "Tool"),
    ("Memory", "ABOUT_PERSON", "Person"),
    ("Memory", "ABOUT_PROJECT", "Project"),
    ("Memory", "ABOUT_PLACE", "Place"),
    ("Memory", "ABOUT_TOOL", "Tool"),
    ("Memory", "ABOUT_TOPIC", "Topic"),
    ("Memory", "RELATES_TO", "Memory"),
    ("Memory", "CONTRADICTS", "Memory"),
    ("Memory", "CONSOLIDATED_INTO", "Memory"),
]

# Map generic ABOUT to specific edge based on target type
ABOUT_EDGE_MAP = {
    "Person": "ABOUT_PERSON",
    "Project": "ABOUT_PROJECT",
    "Place": "ABOUT_PLACE",
    "Tool": "ABOUT_TOOL",
    "Topic": "ABOUT_TOPIC",
}


class GraphStore:
    def __init__(self, path: Path = DEFAULT_GRAPH_PATH):
        path.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(path))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def close(self):
        pass  # KuzuDB handles cleanup

    def _init_schema(self):
        # Create node tables
        for ntype in NODE_TYPES:
            if ntype == "Memory":
                self._conn.execute(
                    "CREATE NODE TABLE IF NOT EXISTS Memory (id STRING, PRIMARY KEY(id))"
                )
            else:
                self._conn.execute(
                    f"CREATE NODE TABLE IF NOT EXISTS {ntype} "
                    f"(name STRING, props STRING DEFAULT '', PRIMARY KEY(name))"
                )
        # Create edge tables
        for from_t, edge, to_t in EDGE_TYPES:
            self._conn.execute(
                f"CREATE REL TABLE IF NOT EXISTS {edge} "
                f"(FROM {from_t} TO {to_t})"
            )

    def upsert_node(self, node_type: str, name: str, props: dict | None = None) -> None:
        """Create node if not exists. Props stored as JSON string."""
        import json

        props_str = json.dumps(props) if props else ""
        if node_type == "Memory":
            self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": name})
        else:
            self._conn.execute(
                f"MERGE (n:{node_type} {{name: $name}}) SET n.props = $props",
                parameters={"name": name, "props": props_str},
            )

    def create_edge(self, from_type: str, from_name: str, edge_type: str, to_type: str, to_name: str) -> None:
        """Create edge between two nodes. Idempotent."""
        if from_type == "Memory":
            from_match = f"(a:{from_type} {{id: $from_name}})"
        else:
            from_match = f"(a:{from_type} {{name: $from_name}})"
        if to_type == "Memory":
            to_match = f"(b:{to_type} {{id: $to_name}})"
        else:
            to_match = f"(b:{to_type} {{name: $to_name}})"

        # Ensure nodes exist
        self.upsert_node(from_type, from_name)
        self.upsert_node(to_type, to_name)

        # Check if edge exists
        query = f"MATCH {from_match}-[r:{edge_type}]->{to_match} RETURN COUNT(r) AS cnt"
        result = self._conn.execute(query, parameters={"from_name": from_name, "to_name": to_name})
        row = result.get_next()
        if row[0] > 0:
            return  # Already exists

        create_q = f"MATCH {from_match}, {to_match} CREATE (a)-[:{edge_type}]->(b)"
        self._conn.execute(create_q, parameters={"from_name": from_name, "to_name": to_name})

    def link_memory(self, memory_id: str, entity_type: str, entity_name: str) -> None:
        """Link a memory to an entity via ABOUT edge."""
        edge_type = ABOUT_EDGE_MAP.get(entity_type)
        if not edge_type:
            return
        self.create_edge("Memory", memory_id, edge_type, entity_type, entity_name)

    def get_neighborhood(self, node_type: str, name: str, depth: int = 1) -> list[dict]:
        """Get all nodes connected within N hops."""
        if node_type == "Memory":
            match_clause = f"(n:{node_type} {{id: $name}})"
        else:
            match_clause = f"(n:{node_type} {{name: $name}})"
        query = f"MATCH {match_clause}-[*1..{depth}]-(connected) RETURN DISTINCT connected.*"
        result = self._conn.execute(query, parameters={"name": name})
        nodes = []
        while result.has_next():
            row = result.get_next()
            # Parse result columns into dict
            columns = result.get_column_names()
            node = {}
            for i, col in enumerate(columns):
                key = col.replace("connected.", "")
                node[key] = row[i]
            nodes.append(node)
        return nodes

    def get_memories_about(self, entity_type: str, entity_name: str) -> list[str]:
        """Get all memory IDs linked to an entity."""
        edge_type = ABOUT_EDGE_MAP.get(entity_type)
        if not edge_type:
            return []
        query = f"MATCH (m:Memory)-[:{edge_type}]->(e:{entity_type} {{name: $name}}) RETURN m.id"
        result = self._conn.execute(query, parameters={"name": entity_name})
        ids = []
        while result.has_next():
            ids.append(result.get_next()[0])
        return ids

    def search_nodes(self, name_query: str) -> list[dict]:
        """Search nodes by name across all entity types."""
        results = []
        for ntype in NODE_TYPES:
            if ntype == "Memory":
                continue
            query = f"MATCH (n:{ntype}) WHERE n.name CONTAINS $q RETURN n.name AS name, '{ntype}' AS type"
            result = self._conn.execute(query, parameters={"q": name_query})
            while result.has_next():
                row = result.get_next()
                results.append({"name": row[0], "type": row[1]})
        return results

    def get_stats(self) -> dict:
        """Return node and edge counts."""
        node_count = 0
        for ntype in NODE_TYPES:
            result = self._conn.execute(f"MATCH (n:{ntype}) RETURN COUNT(n)")
            node_count += result.get_next()[0]
        edge_count = 0
        for _, etype, _ in EDGE_TYPES:
            try:
                result = self._conn.execute(f"MATCH ()-[r:{etype}]->() RETURN COUNT(r)")
                edge_count += result.get_next()[0]
            except Exception:
                pass
        return {"nodes": node_count, "edges": edge_count}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_graph.py -v`
Expected: All PASS (some tests may need adjustment based on KuzuDB's exact Cypher dialect — fix iteratively)

- [ ] **Step 5: Run ruff and commit**

```bash
uv run ruff check src/phileas/graph.py && uv run ruff format src/phileas/graph.py
git add src/phileas/graph.py tests/test_graph.py
git commit -m "feat: add KuzuDB graph store with entity/relationship CRUD"
```

---

### Task 6: Scoring module

**Files:**
- Create: `src/phileas/scoring.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Write tests**

Create `tests/test_scoring.py`:

```python
"""Tests for memory scoring."""

import math
from datetime import datetime, timedelta, timezone

from phileas.scoring import compute_score, recency_score


def test_recency_score_recent():
    # Just accessed: score should be ~1.0
    score = recency_score(days_since_access=0, importance=5, tier=2)
    assert score > 0.99


def test_recency_score_old():
    # 70 days old, normal decay: should be ~0.5
    score = recency_score(days_since_access=70, importance=5, tier=2)
    assert 0.4 < score < 0.6


def test_recency_score_tier3_slow_decay():
    # Tier 3 memories decay much slower
    tier2_score = recency_score(days_since_access=200, importance=5, tier=2)
    tier3_score = recency_score(days_since_access=200, importance=5, tier=3)
    assert tier3_score > tier2_score


def test_recency_score_high_importance_slow_decay():
    # High importance memories decay slower
    low = recency_score(days_since_access=200, importance=3, tier=2)
    high = recency_score(days_since_access=200, importance=9, tier=2)
    assert high > low


def test_compute_score():
    score = compute_score(
        similarity=0.8,
        importance=8,
        days_since_access=0,
        access_count=5,
        tier=2,
    )
    # All components positive, should be > 0
    assert score > 0
    assert score <= 1.0


def test_high_importance_beats_low():
    high = compute_score(similarity=0.5, importance=10, days_since_access=0, access_count=1, tier=2)
    low = compute_score(similarity=0.5, importance=2, days_since_access=0, access_count=1, tier=2)
    assert high > low


def test_recent_beats_old_same_importance():
    recent = compute_score(similarity=0.5, importance=5, days_since_access=1, access_count=1, tier=2)
    old = compute_score(similarity=0.5, importance=5, days_since_access=365, access_count=1, tier=2)
    assert recent > old
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scoring.py -v`

- [ ] **Step 3: Implement scoring.py**

Create `src/phileas/scoring.py`:

```python
"""Memory scoring: importance, recency decay, access frequency.

Scoring formula:
  final = (similarity × 0.4) + (importance/10 × 0.3) + (recency × 0.2) + (access × 0.1)
"""

import math


def recency_score(days_since_access: float, importance: int = 5, tier: int = 2) -> float:
    """Exponential decay based on time since last access.

    Decay rate varies by tier and importance:
    - Tier 3 (core): 0.001 (near-permanent)
    - Importance 9-10: 0.005 (very slow)
    - Default: 0.01 (50% after ~70 days)
    """
    if tier == 3:
        decay_rate = 0.001
    elif importance >= 9:
        decay_rate = 0.005
    else:
        decay_rate = 0.01
    return math.exp(-decay_rate * days_since_access)


def compute_score(
    similarity: float,
    importance: int,
    days_since_access: float,
    access_count: int,
    tier: int = 2,
) -> float:
    """Combined scoring for retrieval ranking."""
    sim_component = similarity * 0.4
    imp_component = (importance / 10.0) * 0.3
    rec_component = recency_score(days_since_access, importance, tier) * 0.2
    acc_component = (math.log(access_count + 1) / 5.0) * 0.1
    return sim_component + imp_component + rec_component + acc_component
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff and commit**

```bash
uv run ruff check src/phileas/scoring.py && uv run ruff format src/phileas/scoring.py
git add src/phileas/scoring.py tests/test_scoring.py
git commit -m "feat: add memory scoring with importance, recency decay, access frequency"
```

---

### Task 7: JSONL ingestion parser

**Files:**
- Create: `src/phileas/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write tests**

Create `tests/test_ingest.py`:

```python
"""Tests for JSONL conversation ingestion."""

import json
from pathlib import Path

from phileas.ingest import parse_session_jsonl, find_unprocessed_sessions


def _write_jsonl(path: Path, messages: list[dict]):
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


def test_parse_extracts_user_and_assistant(tmp_dir):
    path = tmp_dir / "session.jsonl"
    _write_jsonl(path, [
        {"type": "system", "message": {"role": "system", "content": "You are Claude"}},
        {"type": "user", "message": {"role": "user", "content": "Hello, I like coffee"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Nice!"}]}},
        {"type": "progress", "data": {"type": "hookEvent"}},
    ])
    messages = parse_session_jsonl(path)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, I like coffee"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Nice!"


def test_parse_handles_content_list(tmp_dir):
    """Assistant messages often have content as a list of blocks."""
    path = tmp_dir / "session.jsonl"
    _write_jsonl(path, [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "First part. "},
            {"type": "text", "text": "Second part."},
        ]}},
    ])
    messages = parse_session_jsonl(path)
    assert messages[0]["content"] == "First part. Second part."


def test_find_unprocessed_sessions(tmp_dir):
    # Create fake Claude Code project structure
    project_dir = tmp_dir / ".claude" / "projects" / "test-project"
    project_dir.mkdir(parents=True)
    _write_jsonl(project_dir / "abc-123.jsonl", [
        {"type": "user", "message": {"role": "user", "content": "hi"}, "sessionId": "abc-123"}
    ])
    _write_jsonl(project_dir / "def-456.jsonl", [
        {"type": "user", "message": {"role": "user", "content": "bye"}, "sessionId": "def-456"}
    ])
    processed = {"abc-123"}
    unprocessed = find_unprocessed_sessions(tmp_dir / ".claude" / "projects", processed)
    assert len(unprocessed) == 1
    assert unprocessed[0]["session_id"] == "def-456"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -v`

- [ ] **Step 3: Implement ingest.py**

Create `src/phileas/ingest.py`:

```python
"""JSONL conversation ingestion.

Reads Claude Code conversation logs from ~/.claude/projects/*/*.jsonl.
Extracts user and assistant messages for memory extraction.
"""

import json
from pathlib import Path


def parse_session_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file and extract user/assistant messages.

    Returns list of {"role": str, "content": str}.
    """
    messages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") not in ("user", "assistant"):
                continue

            msg = entry.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Content can be a string or a list of blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "".join(text_parts)

            if content:
                messages.append({"role": role, "content": content})

    return messages


def find_unprocessed_sessions(
    projects_dir: Path, processed_ids: set[str]
) -> list[dict]:
    """Scan Claude Code projects dir for unprocessed session files.

    Returns list of {"session_id": str, "path": Path}.
    """
    if not projects_dir.exists():
        return []

    unprocessed = []
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            session_id = jsonl_file.stem
            if session_id not in processed_ids:
                unprocessed.append({
                    "session_id": session_id,
                    "path": jsonl_file,
                })
    return unprocessed
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff and commit**

```bash
uv run ruff check src/phileas/ingest.py && uv run ruff format src/phileas/ingest.py
git add src/phileas/ingest.py tests/test_ingest.py
git commit -m "feat: add JSONL conversation parser and session scanner"
```

---

### Task 8: Rewrite engine to orchestrate all three backends

**Files:**
- Modify: `src/phileas/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write tests**

Create `tests/test_engine.py`:

```python
"""Tests for the memory engine (integration across all backends)."""

from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore


def _make_engine(tmp_dir):
    db = Database(path=tmp_dir / "test.db")
    vs = VectorStore(path=tmp_dir / "chroma")
    gs = GraphStore(path=tmp_dir / "graph")
    return MemoryEngine(db=db, vector=vs, graph=gs)


def test_store_and_recall(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(summary="Giao likes coffee", memory_type="behavior", importance=5)
    results = engine.recall("coffee")
    assert len(results) > 0
    assert any("coffee" in r["summary"] for r in results)


def test_memorize_with_entities(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(
        summary="Giao is building Phileas",
        memory_type="event",
        importance=7,
        entities=[
            {"name": "Giao", "type": "Person"},
            {"name": "Phileas", "type": "Project"},
        ],
        relationships=[
            {"from_name": "Giao", "from_type": "Person", "edge": "BUILDS", "to_name": "Phileas", "to_type": "Project"},
        ],
    )
    # Should be findable via graph
    about_results = engine.about("Giao")
    assert len(about_results) > 0


def test_forget(tmp_dir):
    engine = _make_engine(tmp_dir)
    result = engine.memorize(summary="old fact", importance=3)
    engine.forget(result["id"])
    results = engine.recall("old fact")
    assert not any(r["id"] == result["id"] for r in results)


def test_dedup_prevents_duplicates(tmp_dir):
    engine = _make_engine(tmp_dir)
    r1 = engine.memorize(summary="Giao likes morning coffee")
    r2 = engine.memorize(summary="Giao enjoys morning coffee")  # Near duplicate
    # Should return existing, not create new
    assert r2["id"] == r1["id"]


def test_timeline(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(summary="meeting with Alice", daily_ref="2026-03-25", memory_type="event")
    engine.memorize(summary="lunch at new place", daily_ref="2026-03-25", memory_type="event")
    engine.memorize(summary="unrelated old thing", daily_ref="2026-03-01", memory_type="event")
    results = engine.timeline("2026-03-25")
    assert len(results) == 2


def test_status(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(summary="fact one")
    engine.memorize(summary="fact two")
    stats = engine.status()
    assert stats["tier2"] == 2
    assert stats["vector_count"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_engine.py -v`

- [ ] **Step 3: Rewrite engine.py**

Rewrite `src/phileas/engine.py`:

```python
"""Memory engine: orchestrates SQLite, ChromaDB, and KuzuDB.

SQLite is the source of truth. ChromaDB and KuzuDB are derived indexes.
All three are updated together on writes, queried in parallel on reads.
"""

from datetime import datetime, timezone

from phileas.db import Database
from phileas.graph import GraphStore
from phileas.models import MemoryItem
from phileas.scoring import compute_score
from phileas.vector import VectorStore


class MemoryEngine:
    def __init__(self, db: Database, vector: VectorStore, graph: GraphStore):
        self.db = db
        self.vector = vector
        self.graph = graph

    def memorize(
        self,
        summary: str,
        memory_type: str = "knowledge",
        importance: int = 5,
        daily_ref: str | None = None,
        source_session_id: str | None = None,
        tier: int = 2,
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
    ) -> dict:
        """Store a memory across all three backends. Returns {"id": str, "summary": str}.

        Checks for duplicates first via ChromaDB similarity.
        """
        from datetime import date

        if daily_ref is None:
            daily_ref = date.today().isoformat()

        # Dedup check
        existing_id = self.vector.find_duplicate(summary, threshold=0.95)
        if existing_id:
            existing = self.db.get_item(existing_id)
            if existing and existing.status == "active":
                return {"id": existing.id, "summary": existing.summary, "deduplicated": True}

        # Create memory item
        item = MemoryItem(
            summary=summary,
            memory_type=memory_type,
            importance=importance,
            tier=tier,
            daily_ref=daily_ref,
            source_session_id=source_session_id,
        )

        # Write to all three backends
        self.db.save_item(item)
        self.vector.add(item.id, summary)

        # Graph: link memory to entities
        if entities:
            for entity in entities:
                self.graph.link_memory(item.id, entity["type"], entity["name"])
        if relationships:
            for rel in relationships:
                self.graph.create_edge(
                    rel["from_type"], rel["from_name"],
                    rel["edge"],
                    rel["to_type"], rel["to_name"],
                )

        return {"id": item.id, "summary": item.summary, "deduplicated": False}

    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str | None = None,
        min_importance: int | None = None,
    ) -> list[dict]:
        """Multi-path retrieval: keyword + semantic + graph, scored and ranked."""
        now = datetime.now(timezone.utc)
        candidates: dict[str, dict] = {}  # id -> {item, similarity}

        # Path 1: Keyword search
        keyword_results = self.db.search_by_keyword(query, top_k=top_k * 3)
        for item in keyword_results:
            if item.id not in candidates:
                candidates[item.id] = {"item": item, "similarity": 0.3}  # Base keyword similarity

        # Path 2: Semantic search
        vector_results = self.vector.search(query, top_k=top_k * 3)
        for mem_id, similarity in vector_results:
            if mem_id in candidates:
                candidates[mem_id]["similarity"] = max(candidates[mem_id]["similarity"], similarity)
            else:
                item = self.db.get_item(mem_id)
                if item and item.status == "active":
                    candidates[mem_id] = {"item": item, "similarity": similarity}

        # Path 3: Graph search — find entities mentioned in query, get connected memories
        graph_nodes = self.graph.search_nodes(query)
        for node in graph_nodes:
            memory_ids = self.graph.get_memories_about(node["type"], node["name"])
            for mem_id in memory_ids:
                if mem_id not in candidates:
                    item = self.db.get_item(mem_id)
                    if item and item.status == "active":
                        candidates[mem_id] = {"item": item, "similarity": 0.5}  # Graph match bonus

        # Score all candidates
        scored = []
        for mem_id, data in candidates.items():
            item = data["item"]

            # Apply filters
            if memory_type and item.memory_type != memory_type:
                continue
            if min_importance and item.importance < min_importance:
                continue

            # Calculate days since last access
            if item.last_accessed:
                delta = now - item.last_accessed
                days = delta.total_seconds() / 86400
            else:
                delta = now - item.created_at
                days = delta.total_seconds() / 86400

            score = compute_score(
                similarity=data["similarity"],
                importance=item.importance,
                days_since_access=days,
                access_count=item.access_count,
                tier=item.tier,
            )
            scored.append({"id": item.id, "summary": item.summary, "type": item.memory_type,
                           "importance": item.importance, "score": score})

        # Sort by score, return top_k
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Bump access counts for returned results
        for result in scored[:top_k]:
            self.db.bump_access(result["id"])

        return scored[:top_k]

    def forget(self, memory_id: str, reason: str | None = None) -> str:
        """Archive a memory. Keeps it in DB but excludes from recall."""
        self.db.archive_item(memory_id, reason)
        self.vector.delete(memory_id)
        return f"Archived memory {memory_id}"

    def relate(
        self,
        from_name: str, from_type: str,
        edge_type: str,
        to_name: str, to_type: str,
        memory_id: str | None = None,
    ) -> str:
        """Create entity relationship in graph."""
        self.graph.create_edge(from_type, from_name, edge_type, to_type, to_name)
        if memory_id:
            self.graph.link_memory(memory_id, to_type, to_name)
        return f"Created {from_name} -[{edge_type}]-> {to_name}"

    def about(self, name: str, entity_type: str | None = None) -> list[dict]:
        """Get everything known about an entity."""
        if entity_type:
            types_to_search = [entity_type]
        else:
            # Search all entity types
            nodes = self.graph.search_nodes(name)
            types_to_search = list({n["type"] for n in nodes})
            if not types_to_search:
                return []

        all_memory_ids = set()
        for etype in types_to_search:
            ids = self.graph.get_memories_about(etype, name)
            all_memory_ids.update(ids)

        results = []
        for mem_id in all_memory_ids:
            item = self.db.get_item(mem_id)
            if item and item.status == "active":
                results.append({"id": item.id, "summary": item.summary, "type": item.memory_type,
                                "importance": item.importance})
        return sorted(results, key=lambda x: x["importance"], reverse=True)

    def timeline(self, start_date: str, end_date: str | None = None) -> list[dict]:
        """Get memories in a date range."""
        items = self.db.get_items_by_date_range(start_date, end_date)
        return [{"id": i.id, "summary": i.summary, "type": i.memory_type,
                 "daily_ref": i.daily_ref} for i in items]

    def status(self) -> dict:
        """Memory system stats."""
        counts = self.db.get_counts()
        graph_stats = self.graph.get_stats()
        return {
            **counts,
            "vector_count": self.vector.count(),
            "graph_nodes": graph_stats["nodes"],
            "graph_edges": graph_stats["edges"],
        }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_engine.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff and commit**

```bash
uv run ruff check src/phileas/engine.py && uv run ruff format src/phileas/engine.py
git add src/phileas/engine.py tests/test_engine.py
git commit -m "refactor: rewrite engine to orchestrate SQLite + ChromaDB + KuzuDB"
```

---

### Task 9: Rewrite MCP server with new tools

**Files:**
- Modify: `src/phileas/server.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: Write tests for MCP tool functions**

Create `tests/test_server.py`:

```python
"""Tests for MCP server tool functions.

Tests the tool functions directly (not via MCP protocol).
"""

from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore


def _make_engine(tmp_dir):
    db = Database(path=tmp_dir / "test.db")
    vs = VectorStore(path=tmp_dir / "chroma")
    gs = GraphStore(path=tmp_dir / "graph")
    return MemoryEngine(db=db, vector=vs, graph=gs)


def test_memorize_tool(tmp_dir):
    """Test that the memorize function stores and returns confirmation."""
    engine = _make_engine(tmp_dir)
    # Directly call what the MCP tool would call
    result = engine.memorize(summary="Giao prefers dark mode", memory_type="behavior", importance=5)
    assert "id" in result
    assert result["summary"] == "Giao prefers dark mode"


def test_recall_tool(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(summary="Giao is learning React")
    results = engine.recall("React")
    assert len(results) > 0


def test_forget_tool(tmp_dir):
    engine = _make_engine(tmp_dir)
    result = engine.memorize(summary="wrong fact")
    msg = engine.forget(result["id"])
    assert "Archived" in msg


def test_profile_tool(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(summary="Name is Giao", memory_type="profile", importance=10)
    engine.memorize(summary="Works as developer", memory_type="profile", importance=9)
    profiles = engine.recall("", memory_type="profile", top_k=10)
    # Should find profile items (may need keyword or type filter)
    profile_items = engine.db.get_items_by_type("profile")
    assert len(profile_items) == 2
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_server.py -v`
Expected: All PASS (tests use engine directly)

- [ ] **Step 3: Rewrite server.py**

Rewrite `src/phileas/server.py` with all new MCP tools:

```python
"""Phileas MCP server.

Centralized memory layer for Claude Code. Stores and retrieves memories
across SQLite (metadata), ChromaDB (embeddings), and KuzuDB (graph).

Tools:
  Core:     memorize, recall, forget, relate
  Query:    about, timeline, profile
  Lifecycle: ingest_session, consolidate, status
"""

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.ingest import find_unprocessed_sessions, parse_session_jsonl
from phileas.vector import VectorStore

mcp = FastMCP(
    "phileas",
    instructions=(
        "Phileas is a centralized memory companion. "
        "Use 'memorize' to store facts about the user, "
        "'recall' to retrieve relevant memories, "
        "'about' for entity lookups, "
        "'ingest_session' to process past conversations."
    ),
)

# Initialize all three backends
db = Database()
vector = VectorStore()
graph = GraphStore()
engine = MemoryEngine(db=db, vector=vector, graph=graph)


@mcp.tool()
def memorize(
    summary: str,
    memory_type: str = "knowledge",
    importance: int = 5,
    daily_ref: str | None = None,
    entities: str | None = None,
    relationships: str | None = None,
) -> str:
    """Store a memory about the user.

    Args:
        summary: What to remember (1-2 sentences).
        memory_type: One of "profile", "event", "knowledge", "behavior", "reflection".
        importance: 1-10 scale. 9-10=identity, 7-8=significant, 5-6=notable, 3-4=context, 1-2=ephemeral.
        daily_ref: Date (YYYY-MM-DD). Defaults to today.
        entities: JSON array of {"name": str, "type": str}. Types: Person, Project, Place, Tool, Topic.
        relationships: JSON array of {"from_name", "from_type", "edge", "to_name", "to_type"}.
    """
    ent_list = json.loads(entities) if entities else None
    rel_list = json.loads(relationships) if relationships else None

    result = engine.memorize(
        summary=summary,
        memory_type=memory_type,
        importance=importance,
        daily_ref=daily_ref,
        entities=ent_list,
        relationships=rel_list,
    )
    if result.get("deduplicated"):
        return f"Already known: {result['summary']}"
    return f"Stored [{memory_type}] (importance={importance}) {result['summary']}"


@mcp.tool()
def recall(query: str, top_k: int = 5, memory_type: str | None = None, min_importance: int | None = None) -> str:
    """Retrieve memories relevant to a query. Uses keyword + semantic + graph search.

    Args:
        query: Natural language or keywords.
        top_k: Max results.
        memory_type: Filter by type.
        min_importance: Minimum importance threshold (1-10).
    """
    results = engine.recall(query, top_k=top_k, memory_type=memory_type, min_importance=min_importance)
    if not results:
        return "No relevant memories found."
    lines = [f"Found {len(results)} memories:"]
    for r in results:
        lines.append(f"  [{r['type']}] (imp={r['importance']}, score={r['score']:.2f}) {r['summary']}")
    return "\n".join(lines)


@mcp.tool()
def forget(memory_id: str, reason: str | None = None) -> str:
    """Archive a memory (soft delete). It won't appear in recall results.

    Args:
        memory_id: The ID of the memory to archive.
        reason: Why it's being archived (e.g., "superseded by newer info").
    """
    return engine.forget(memory_id, reason)


@mcp.tool()
def relate(
    from_name: str, from_type: str,
    edge_type: str,
    to_name: str, to_type: str,
    memory_id: str | None = None,
) -> str:
    """Create a relationship between entities in the knowledge graph.

    Args:
        from_name: Source entity name.
        from_type: Source type (Person, Project, Place, Tool, Topic, Memory).
        edge_type: Relationship (BUILDS, USES, KNOWS, WORKS_AT, RELATES_TO, CONTRADICTS).
        to_name: Target entity name.
        to_type: Target type.
        memory_id: Optionally link a memory to the target entity.
    """
    return engine.relate(from_name, from_type, edge_type, to_name, to_type, memory_id)


@mcp.tool()
def about(name: str, entity_type: str | None = None) -> str:
    """Get everything known about a person, project, or topic.

    Args:
        name: Entity name to look up.
        entity_type: Optionally narrow to Person, Project, Place, Tool, Topic.
    """
    results = engine.about(name, entity_type)
    if not results:
        return f"No memories found about '{name}'."
    lines = [f"About '{name}' ({len(results)} memories):"]
    for r in results:
        lines.append(f"  [{r['type']}] (imp={r['importance']}) {r['summary']}")
    return "\n".join(lines)


@mcp.tool()
def timeline(start_date: str, end_date: str | None = None) -> str:
    """Get memories from a date or date range.

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD). If omitted, returns only start_date.
    """
    results = engine.timeline(start_date, end_date)
    if not results:
        return f"No memories found for {start_date}."
    lines = [f"Timeline ({len(results)} entries):"]
    for r in results:
        lines.append(f"  [{r['daily_ref']}] [{r['type']}] {r['summary']}")
    return "\n".join(lines)


@mcp.tool()
def profile() -> str:
    """Get all profile memories — who the user is."""
    items = engine.db.get_items_by_type("profile")
    if not items:
        return "No profile memories stored yet."
    lines = ["User profile:"]
    for item in items:
        lines.append(f"  - (imp={item.importance}) {item.summary}")
    return "\n".join(lines)


@mcp.tool()
def ingest_session(session_path: str) -> str:
    """Read a Claude Code conversation JSONL file and return messages for extraction.

    Call this from SessionStart hook. Returns conversation text that Claude Code
    should process and call memorize() with the extracted facts.

    Args:
        session_path: Path to the .jsonl file.
    """
    path = Path(session_path)
    if not path.exists():
        return f"File not found: {session_path}"

    session_id = path.stem
    if engine.db.is_session_processed(session_id):
        return f"Session {session_id} already processed."

    messages = parse_session_jsonl(path)
    if not messages:
        engine.db.mark_session_processed(session_id, str(path))
        return "Empty session, marked as processed."

    # Format for Claude Code to extract from
    lines = [f"Session {session_id} ({len(messages)} messages):"]
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"][:500]  # Truncate long messages
        lines.append(f"[{role}] {content}")

    lines.append("\n---\nExtract key facts, entities, and relationships from above.")
    lines.append("Call memorize() for each fact. Then mark as processed.")

    return "\n".join(lines)


@mcp.tool()
def mark_session_done(session_path: str) -> str:
    """Mark a session as processed after extraction is complete.

    Args:
        session_path: Path to the .jsonl file that was processed.
    """
    path = Path(session_path)
    session_id = path.stem
    engine.db.mark_session_processed(session_id, str(path))
    return f"Session {session_id} marked as processed."


@mcp.tool()
def consolidate(min_cluster_size: int = 3, max_clusters: int = 10) -> str:
    """Find clusters of related Tier 2 memories for consolidation.

    Returns clusters of similar memories. Claude Code should summarize each
    cluster and call memorize(tier=3) with the summary, then the originals
    can be marked with consolidated_into.

    Args:
        min_cluster_size: Minimum memories per cluster to report.
        max_clusters: Maximum clusters to return.
    """
    # Get all active tier-2 memories without consolidated_into
    items = [i for i in engine.db.get_items_by_tier(2) if not i.consolidated_into]
    if len(items) < min_cluster_size:
        return f"Only {len(items)} unconsolidated tier-2 memories. Need at least {min_cluster_size}."

    # Use vector similarity to find clusters
    clusters: list[list[dict]] = []
    used_ids: set[str] = set()
    for item in items:
        if item.id in used_ids:
            continue
        # Find similar memories
        similar = engine.vector.search(item.summary, top_k=10)
        cluster_items = [{"id": item.id, "summary": item.summary}]
        used_ids.add(item.id)
        for sim_id, score in similar:
            if sim_id != item.id and sim_id not in used_ids and score > 0.7:
                sim_item = engine.db.get_item(sim_id)
                if sim_item and sim_item.tier == 2 and not sim_item.consolidated_into:
                    cluster_items.append({"id": sim_id, "summary": sim_item.summary})
                    used_ids.add(sim_id)
        if len(cluster_items) >= min_cluster_size:
            clusters.append(cluster_items)
        if len(clusters) >= max_clusters:
            break

    if not clusters:
        return "No clusters found meeting the minimum size."

    lines = [f"Found {len(clusters)} clusters ready for consolidation:"]
    for i, cluster in enumerate(clusters):
        lines.append(f"\nCluster {i+1} ({len(cluster)} memories):")
        for mem in cluster:
            lines.append(f"  - [{mem['id'][:8]}] {mem['summary']}")
    lines.append("\nSummarize each cluster and call memorize(tier=3). Then update originals.")
    return "\n".join(lines)


@mcp.tool()
def status() -> str:
    """Show memory system statistics."""
    stats = engine.status()
    claude_dir = Path.home() / ".claude" / "projects"
    processed_count = engine.db.get_unprocessed_session_count()

    lines = [
        "Phileas Memory Status:",
        f"  Tier 2 (long-term): {stats['tier2']} memories",
        f"  Tier 3 (core): {stats['tier3']} memories",
        f"  Archived: {stats['archived']}",
        f"  Total: {stats['total']}",
        f"  Vector index: {stats['vector_count']} embeddings",
        f"  Graph: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges",
        f"  Processed sessions: {processed_count}",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Run ruff on all files and commit**

```bash
uv run ruff check src/phileas/ && uv run ruff format src/phileas/
git add src/phileas/server.py tests/test_server.py
git commit -m "refactor: rewrite MCP server with full tool suite"
```

---

### Task 10: Delete legacy code and clean up

**Files:**
- Modify: `src/phileas/models.py` (remove Resource, Category, CategoryItem if not already done)
- Delete: `scripts/bootstrap.py` (replaced by ingest_session)

- [ ] **Step 1: Remove bootstrap.py**

```bash
git rm scripts/bootstrap.py
```

- [ ] **Step 2: Verify no remaining imports of removed models**

Run: `uv run ruff check src/phileas/ && uv run pytest tests/ -v`
Expected: All clean, all pass

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove legacy bootstrap script and unused models"
```

---

### Task 11: Run full test suite and verify MCP server starts

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Verify MCP server starts**

Run: `uv run mcp dev src/phileas/server.py`
Expected: Server starts without errors, tools are listed

- [ ] **Step 3: Run ruff on entire codebase**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: All clean

- [ ] **Step 4: Commit any fixes**

```bash
git add -A && git commit -m "chore: fix any remaining lint/test issues"
```

---

### Task 12: Migration — backfill existing memories (if any exist)

This task handles migrating data from the old v1 SQLite schema to the new system.

- [ ] **Step 1: Check if old database exists**

```bash
ls -la ~/.phileas/memory.db
```

If no database exists, skip this task entirely.

- [ ] **Step 2: Write a one-time migration script**

Create `scripts/migrate_v1.py`:

```python
"""One-time migration from v1 schema to new schema.

Reads old memory_items, re-inserts with new fields, embeds into ChromaDB.
"""

import json
import sqlite3
from pathlib import Path

from phileas.db import Database
from phileas.graph import GraphStore
from phileas.models import MemoryItem
from phileas.vector import VectorStore

OLD_DB = Path.home() / ".phileas" / "memory.db.bak"
NEW_DB = Path.home() / ".phileas" / "memory.db"

IMPORTANCE_DEFAULTS = {
    "profile": 9,
    "event": 6,
    "knowledge": 5,
    "behavior": 5,
    "reflection": 7,
}


def migrate():
    # Backup first
    if NEW_DB.exists():
        import shutil
        shutil.copy(NEW_DB, OLD_DB)
        print(f"Backed up to {OLD_DB}")

    # Read old data
    old_conn = sqlite3.connect(str(OLD_DB))
    old_conn.row_factory = sqlite3.Row
    old_items = old_conn.execute("SELECT * FROM memory_items").fetchall()
    print(f"Found {len(old_items)} old memories")

    # Initialize new backends
    db = Database(path=NEW_DB)
    vs = VectorStore()
    gs = GraphStore()

    for row in old_items:
        importance = IMPORTANCE_DEFAULTS.get(row["memory_type"], 5)
        item = MemoryItem(
            id=row["id"],
            summary=row["summary"],
            memory_type=row["memory_type"],
            importance=importance,
            daily_ref=row["daily_ref"] if "daily_ref" in row.keys() else None,
        )
        db.save_item(item)
        vs.add(item.id, item.summary)
        print(f"  Migrated: [{item.memory_type}] {item.summary[:60]}...")

    old_conn.close()
    db.close()
    print(f"\nDone. Migrated {len(old_items)} memories.")


if __name__ == "__main__":
    migrate()
```

- [ ] **Step 3: Run migration**

Run: `uv run python scripts/migrate_v1.py`

- [ ] **Step 4: Verify**

Run: `uv run python -c "from phileas.db import Database; db = Database(); print(db.get_counts()); db.close()"`

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_v1.py
git commit -m "feat: add v1 to new schema migration script"
```
