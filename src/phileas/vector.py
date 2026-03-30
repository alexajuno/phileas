"""ChromaDB vector store for semantic search.

A derived index — can be rebuilt from SQLite data.
Uses ChromaDB's built-in embedding function.
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

    def get_embeddings(self, memory_ids: list[str]) -> dict[str, list[float]]:
        """Get stored embeddings for given memory IDs. Returns {id: embedding}."""
        if not memory_ids:
            return {}
        result = self._collection.get(ids=memory_ids, include=["embeddings"])
        ids = result["ids"]
        embeddings = result["embeddings"] if result["embeddings"] is not None else []
        return {mid: list(emb) for mid, emb in zip(ids, embeddings) if emb is not None}

    def delete(self, memory_id: str) -> None:
        self._collection.delete(ids=[memory_id])

    def count(self) -> int:
        return self._collection.count()
