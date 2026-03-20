"""Memory engine: store and retrieve memories.

Two retrieval paths:
  1. Keyword search (no dependencies, always available)
  2. Embedding search (requires sentence-transformers, optional)

LLM extraction is NOT done here — Claude Code handles that via skills/agents
and calls the MCP tools to store pre-extracted memories.
"""

from phileas.db import Database
from phileas.models import Category, MemoryItem, Resource


class MemoryEngine:
    def __init__(self, db: Database, use_embeddings: bool = False):
        self.db = db
        self._embedder = None
        if use_embeddings:
            self._embedder = _load_embedder()

    def store_resource(self, content: str, modality: str = "conversation") -> Resource:
        """L1: Store raw content (immutable)."""
        resource = Resource(content=content, modality=modality)
        self.db.save_resource(resource)
        return resource

    def store_memory(
        self,
        summary: str,
        memory_type: str = "knowledge",
        category_name: str | None = None,
        resource_id: str | None = None,
    ) -> MemoryItem:
        """L2: Store a pre-extracted memory. Claude Code decides what to store."""
        embedding = self._embed(summary) if self._embedder else None

        item = MemoryItem(
            resource_id=resource_id,
            memory_type=memory_type,
            summary=summary,
            embedding=embedding,
        )
        self.db.save_item(item)

        if category_name:
            category = self.db.get_category_by_name(category_name)
            if not category:
                cat_embedding = self._embed(category_name) if self._embedder else None
                category = Category(
                    name=category_name,
                    description=f"Memories about {category_name}",
                    embedding=cat_embedding,
                )
                self.db.save_category(category)
            self.db.link_item_to_category(item.id, category.id)

        return item

    def recall(self, query: str, top_k: int = 10, memory_type: str | None = None) -> list[MemoryItem]:
        """Retrieve relevant memories. Uses embeddings if available, else keyword search."""
        if memory_type:
            return self.db.get_items_by_type(memory_type)[:top_k]

        if self._embedder:
            query_embedding = self._embed(query)
            return self.db.search_items_by_embedding(query_embedding, top_k=top_k)

        return self.db.search_items_by_keyword(query, top_k=top_k)

    def get_user_profile(self) -> list[MemoryItem]:
        return self.db.get_items_by_type("profile")

    def get_all_categories(self) -> list[dict]:
        categories = self.db.get_all_categories()
        result = []
        for cat in categories:
            items = self.db.get_items_in_category(cat.id)
            result.append({
                "name": cat.name,
                "description": cat.description,
                "summary": cat.summary,
                "item_count": len(items),
            })
        return result

    def _embed(self, text: str) -> list[float] | None:
        if self._embedder is None:
            return None
        return self._embedder.encode(text).tolist()


def _load_embedder():
    """Load sentence-transformers model. Fails gracefully if not installed."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None
