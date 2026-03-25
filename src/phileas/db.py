"""SQLite storage backend for Phileas.

Simple, local-first. No external database server needed.
Embeddings stored as JSON arrays (brute-force cosine similarity for now).
"""

import json
import math
import sqlite3
from pathlib import Path

from phileas.models import Category, MemoryItem, Resource

DEFAULT_DB_PATH = Path.home() / ".phileas" / "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS resources (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    modality TEXT NOT NULL DEFAULT 'conversation',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    resource_id TEXT REFERENCES resources(id),
    memory_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    embedding TEXT,
    happened_at TEXT,
    daily_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS categories (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    summary TEXT,
    embedding TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS category_items (
    item_id TEXT NOT NULL REFERENCES memory_items(id),
    category_id TEXT NOT NULL REFERENCES categories(id),
    PRIMARY KEY (item_id, category_id)
);
"""


class Database:
    def __init__(self, path: Path = DEFAULT_DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self):
        """Run schema migrations for existing databases."""
        cursor = self.conn.execute("PRAGMA table_info(memory_items)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "daily_ref" not in columns:
            self.conn.execute("ALTER TABLE memory_items ADD COLUMN daily_ref TEXT")
            self.conn.commit()

    def close(self):
        self.conn.close()

    # --- Resources ---

    def save_resource(self, r: Resource) -> None:
        self.conn.execute(
            "INSERT INTO resources (id, content, modality, created_at) VALUES (?, ?, ?, ?)",
            (r.id, r.content, r.modality, r.created_at.isoformat()),
        )
        self.conn.commit()

    def get_resource(self, id: str) -> Resource | None:
        row = self.conn.execute("SELECT * FROM resources WHERE id = ?", (id,)).fetchone()
        if not row:
            return None
        return Resource(
            id=row["id"],
            content=row["content"],
            modality=row["modality"],
        )

    # --- Memory Items ---

    def save_item(self, item: MemoryItem) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO memory_items
               (id, resource_id, memory_type, summary, embedding, happened_at, daily_ref, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.id,
                item.resource_id,
                item.memory_type,
                item.summary,
                json.dumps(item.embedding) if item.embedding else None,
                item.happened_at.isoformat() if item.happened_at else None,
                item.daily_ref,
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
            ),
        )
        self.conn.commit()

    def get_all_items(self) -> list[MemoryItem]:
        rows = self.conn.execute("SELECT * FROM memory_items ORDER BY created_at DESC").fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_items_by_type(self, memory_type: str) -> list[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memory_items WHERE memory_type = ? ORDER BY created_at DESC",
            (memory_type,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def search_items_by_embedding(self, query_embedding: list[float], top_k: int = 10) -> list[MemoryItem]:
        """Brute-force cosine similarity search. Fine for <10k items."""
        items = self.get_all_items()
        scored = []
        for item in items:
            if item.embedding:
                score = _cosine_similarity(query_embedding, item.embedding)
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def search_items_by_keyword(self, query: str, top_k: int = 10) -> list[MemoryItem]:
        """Simple keyword search using SQLite LIKE. Splits query into words."""
        words = query.lower().split()
        if not words:
            return self.get_all_items()[:top_k]

        # Match items containing ANY of the query words
        conditions = " OR ".join(["LOWER(summary) LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words]

        rows = self.conn.execute(
            f"""SELECT *, (
                {" + ".join(["(CASE WHEN LOWER(summary) LIKE ? THEN 1 ELSE 0 END)" for _ in words])}
            ) as match_count
            FROM memory_items
            WHERE {conditions}
            ORDER BY match_count DESC, created_at DESC
            LIMIT ?""",
            params + params + [top_k],
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            resource_id=row["resource_id"],
            memory_type=row["memory_type"],
            summary=row["summary"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            daily_ref=row["daily_ref"] if "daily_ref" in row.keys() else None,
        )

    # --- Categories ---

    def save_category(self, cat: Category) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO categories
               (id, name, description, summary, embedding, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                cat.id,
                cat.name,
                cat.description,
                cat.summary,
                json.dumps(cat.embedding) if cat.embedding else None,
                cat.created_at.isoformat(),
                cat.updated_at.isoformat(),
            ),
        )
        self.conn.commit()

    def get_category_by_name(self, name: str) -> Category | None:
        row = self.conn.execute("SELECT * FROM categories WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        return Category(id=row["id"], name=row["name"], description=row["description"], summary=row["summary"])

    def get_all_categories(self) -> list[Category]:
        rows = self.conn.execute("SELECT * FROM categories ORDER BY name").fetchall()
        return [Category(id=r["id"], name=r["name"], description=r["description"], summary=r["summary"]) for r in rows]

    # --- Category-Item links ---

    def link_item_to_category(self, item_id: str, category_id: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO category_items (item_id, category_id) VALUES (?, ?)",
            (item_id, category_id),
        )
        self.conn.commit()

    def get_items_in_category(self, category_id: str) -> list[MemoryItem]:
        rows = self.conn.execute(
            """SELECT mi.* FROM memory_items mi
               JOIN category_items ci ON mi.id = ci.item_id
               WHERE ci.category_id = ?
               ORDER BY mi.created_at DESC""",
            (category_id,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
