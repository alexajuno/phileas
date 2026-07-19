"""ChromaDB vector store for semantic search.

A derived index — can be rebuilt from SQLite data.
Uses ChromaDB's built-in embedding function.
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings

from phileas.config import resolve_home

DEFAULT_CHROMA_PATH = resolve_home() / "chroma"
COLLECTION_NAME = "memories"
SOURCES_COLLECTION_NAME = "sources"


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
        # Disable ChromaDB's phone-home telemetry: it sends usage events to a
        # remote endpoint on writes, which contradicts the local-first promise
        # and makes a write do a DNS lookup that fails when offline.
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._sources_collection = self._client.get_or_create_collection(
            name=SOURCES_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # One-time cleanup: earlier builds kept other verbatim collections
        # ("raw_memories", the per-turn "events" index) alongside this one.
        # Provenance now flows through sources, so drop the leftovers a prior
        # version may have left on disk.
        for stale in ("raw_memories", "events"):
            try:
                self._client.delete_collection(stale)
            except Exception:
                pass

        # Lazy embedding function for text_similarities — same default model
        # Chroma attaches to the collections, instantiated only when the
        # entity linker first asks for a description similarity.
        self._embed_fn = None

    def close(self):
        pass  # ChromaDB PersistentClient doesn't need explicit close

    def text_similarities(self, query: str, texts: list[str]) -> list[float]:
        """Cosine similarity between ``query`` and each of ``texts``, without persisting.

        Backs the entity linker's description signal: candidate sets are tiny
        (a handful of same-name entities), so embedding both sides per lookup
        on the daemon's warm model is cheaper and simpler than maintaining a
        synced per-entity description collection.
        """
        if not texts:
            return []
        if self._embed_fn is None:
            from chromadb.utils import embedding_functions

            self._embed_fn = embedding_functions.DefaultEmbeddingFunction()
        vectors = [[float(x) for x in vec] for vec in self._embed_fn([query, *texts])]
        qvec, rest = vectors[0], vectors[1:]
        qnorm = sum(x * x for x in qvec) ** 0.5
        sims: list[float] = []
        for vec in rest:
            vnorm = sum(x * x for x in vec) ** 0.5
            if qnorm == 0 or vnorm == 0:
                sims.append(0.0)
                continue
            dot = sum(a * b for a, b in zip(qvec, vec))
            sims.append(dot / (qnorm * vnorm))
        return sims

    def add(self, memory_id: str, text: str, metadata: dict | None = None) -> None:
        """Add or update a memory embedding."""
        kwargs: dict = {"ids": [memory_id], "documents": [text]}
        if metadata:
            kwargs["metadatas"] = [metadata]
        self._collection.upsert(**kwargs)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Search by semantic similarity. Returns [(memory_id, score)]."""
        count = self._collection.count()
        if count == 0:
            return []
        n_results = count if top_k is None else min(top_k, count)
        results = self._collection.query(query_texts=[query], n_results=n_results)
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

    # --- Sources collection (session text, keyed by source_id) ---

    def add_source(self, source_id: str, text: str) -> None:
        """Embed a session's text so a query can find the session it came from."""
        self._sources_collection.upsert(ids=[source_id], documents=[text])

    def search_sources(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search session text by semantic similarity. Returns [(source_id, score)]."""
        if self._sources_collection.count() == 0:
            return []
        results = self._sources_collection.query(
            query_texts=[query], n_results=min(top_k, self._sources_collection.count())
        )
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        return [(id_, 1.0 - dist) for id_, dist in zip(ids, distances)]

    def delete_source(self, source_id: str) -> None:
        self._sources_collection.delete(ids=[source_id])

    def source_count(self) -> int:
        return self._sources_collection.count()
