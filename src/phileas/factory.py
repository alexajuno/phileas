"""Construction seam: build a ``MemoryEngine`` and its storage backends from config.

Both the daemon and the CLI wire the same three stores (SQLite, ChromaDB,
KuzuDB) into an engine. Centralizing that wiring here gives one place to make a
backend swappable: a provider registry lands in ``build_engine`` later without
touching the call sites. Today the backends are fixed.

Process lifecycle stays with the caller. ``build_engine`` only constructs and
returns a ready engine; the daemon still owns eager graph-lock acquisition and
model pre-warm, which are about how the process lives, not how it is wired.

Heavy imports (the engine pulls in the embedding model and reranker) are kept
inside ``build_engine`` so ``import phileas`` and ``import phileas.factory`` stay
cheap: nothing loads a model until an engine is actually built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phileas.config import PhileasConfig
    from phileas.engine import MemoryEngine


def _quiet_model_loading() -> None:
    """Lower the sentence-transformers / HuggingFace import chatter to errors.

    The embedding model and reranker log at import and first use; both
    construction paths silence it, so it lives here once.
    """
    import logging

    for name in ("sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)


def build_engine(config: PhileasConfig) -> MemoryEngine:
    """Construct the storage backends and the engine that fuses them.

    The single wiring site for SQLite + ChromaDB + KuzuDB. Hand in a resolved
    config; get back a ready engine. Access ``engine.db`` / ``engine.vector`` /
    ``engine.graph`` for the individual stores when a caller needs them (the
    daemon does, for its post-construction warmup).
    """
    from phileas.db import Database
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    _quiet_model_loading()
    db = Database(path=config.db_path)
    vector = VectorStore(path=config.chroma_path)
    graph = GraphStore(path=config.graph_path)
    return MemoryEngine(db=db, vector=vector, graph=graph, config=config)
