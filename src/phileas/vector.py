"""ChromaDB vector store for semantic search.

A derived index — can be rebuilt from SQLite data.
Uses ChromaDB's built-in embedding function.
"""

from pathlib import Path

import chromadb

DEFAULT_CHROMA_PATH = Path.home() / ".phileas" / "chroma"
COLLECTION_NAME = "memories"
RAW_COLLECTION_NAME = "raw_memories"


def _zip_embeddings(chroma_result: dict) -> dict[str, list[float]]:
    """Zip Chroma get() result into {id: embedding} safely.

    Chroma may return embeddings as a numpy ndarray or a Python list, and
    individual entries may be None for missing ids. Truthiness checks on
    numpy arrays raise, so this helper avoids them entirely.

    Always returns native Python floats (not numpy scalars). Downstream MMR
    stages do pairwise dot products in pure Python, and numpy scalars there
    are ~100x slower per op than plain floats.
    """
    ids = chroma_result.get("ids") or []
    raw = chroma_result.get("embeddings")
    if raw is None:
        return {}
    out: dict[str, list[float]] = {}
    for i, mid in enumerate(ids):
        if i >= len(raw):
            break
        emb = raw[i]
        if emb is None:
            continue
        # .tolist() converts numpy arrays + scalars deeply to native Python.
        # Falls back to list() for plain Python iterables that lack .tolist.
        if hasattr(emb, "tolist"):
            out[mid] = emb.tolist()
        else:
            try:
                out[mid] = [float(x) for x in emb]
            except TypeError:
                continue
    return out


class VectorStore:
    def __init__(self, path: Path = DEFAULT_CHROMA_PATH):
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._raw_collection = self._client.get_or_create_collection(
            name=RAW_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self):
        pass  # ChromaDB PersistentClient doesn't need explicit close

    def add(self, memory_id: str, text: str, metadata: dict | None = None) -> None:
        """Add or update a memory embedding."""
        kwargs: dict = {"ids": [memory_id], "documents": [text]}
        if metadata:
            kwargs["metadatas"] = [metadata]
        self._collection.upsert(**kwargs)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search by semantic similarity. Returns [(memory_id, score)]."""
        if self._collection.count() == 0:
            return []
        results = self._collection.query(query_texts=[query], n_results=min(top_k, self._collection.count()))
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        # ChromaDB returns distances (lower = closer for cosine). Convert to similarity.
        return [(id_, 1.0 - dist) for id_, dist in zip(ids, distances)]

    def find_similar(self, text: str, floor: float = 0.70, ceiling: float = 0.95) -> tuple[str, float] | None:
        """Find the most similar memory in the [floor, ceiling) range. Returns (id, similarity) or None."""
        if self._collection.count() == 0:
            return None
        results = self._collection.query(query_texts=[text], n_results=1)
        if not results["ids"] or not results["ids"][0]:
            return None
        dist = results["distances"][0][0]
        similarity = 1.0 - dist
        if floor <= similarity < ceiling:
            return (results["ids"][0][0], similarity)
        return None

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
        """Get stored embeddings for given memory IDs. Returns {id: embedding}.

        Resilient to drift between SQLite and Chroma: if some IDs are missing
        from the vector store, returns embeddings only for the ones that exist
        instead of raising. Without this guard a single orphan memory_item
        brings down all of recall via Chroma's "Error finding id" error.

        Note on numpy: Chroma can return embeddings as a numpy ndarray, so
        any truthiness check (`bool(arr)`, `arr or default`) raises. Always
        use `is None` and explicit length checks here.
        """
        if not memory_ids:
            return {}
        try:
            result = self._collection.get(ids=memory_ids, include=["embeddings"])
            return _zip_embeddings(result)
        except Exception:
            return self._get_embeddings_individually(memory_ids)

    def _get_embeddings_individually(self, memory_ids: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for mid in memory_ids:
            try:
                r = self._collection.get(ids=[mid], include=["embeddings"])
            except Exception:
                continue
            out.update(_zip_embeddings(r))
        return out

    def delete(self, memory_id: str) -> None:
        self._collection.delete(ids=[memory_id])

    def count(self) -> int:
        return self._collection.count()

    # --- Raw text collection ---

    def add_raw(self, memory_id: str, raw_text: str, metadata: dict | None = None) -> None:
        """Store raw verbatim text for a memory."""
        kwargs: dict = {"ids": [memory_id], "documents": [raw_text]}
        if metadata:
            kwargs["metadatas"] = [metadata]
        self._raw_collection.upsert(**kwargs)

    def search_raw(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search raw text by semantic similarity. Returns [(memory_id, score)]."""
        if self._raw_collection.count() == 0:
            return []
        results = self._raw_collection.query(query_texts=[query], n_results=min(top_k, self._raw_collection.count()))
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        return [(id_, 1.0 - dist) for id_, dist in zip(ids, distances)]

    def delete_raw(self, memory_id: str) -> None:
        self._raw_collection.delete(ids=[memory_id])

    def raw_count(self) -> int:
        return self._raw_collection.count()
