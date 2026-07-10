"""Build a throwaway MemoryEngine over an ephemeral store for the LongMemEval eval.

Each LongMemEval question carries its own haystack of sessions, so the retrieval
eval builds one isolated store per question and tears it down after scoring (the
``rollup`` eval's ``tempfile`` pattern, not the fixed-profile pattern the recall
eval uses). The reranker is a process-global singleton, so it loads once and is
shared across every per-question store; only the vector/graph/db backends are
rebuilt per question.

``build_engine`` refuses any path inside the real ``~/.phileas`` graph, so a stray
call can never write into it, and freezes the store's retrieval-strength write so
repeated runs see byte-identical state (the noise-floor trick the other evals use).
"""
from __future__ import annotations

import logging
from pathlib import Path

FORBIDDEN_HOME = Path.home() / ".phileas"


def _quiet() -> None:
    for n in ("sentence_transformers", "transformers", "huggingface_hub", "chromadb"):
        logging.getLogger(n).setLevel(logging.ERROR)


def build_engine(path: Path):
    """An isolated engine rooted at ``path`` (an ephemeral temp dir)."""
    _quiet()

    from phileas.config import load_config
    from phileas.db import Database
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    path = Path(path).resolve()
    forbidden = FORBIDDEN_HOME.resolve()
    if path == forbidden or forbidden in path.parents:
        raise SystemExit(f"REFUSING: store path {path} is inside the real graph {forbidden}")

    db = Database(path=path / "memory.db")
    vector = VectorStore(path=path / "chroma")
    graph = GraphStore(path=path / "graph")  # direct KuzuDB, daemon-free
    eng = MemoryEngine(db=db, vector=vector, graph=graph, config=load_config(home=path))
    # Freeze the store: neutralize the retrieval-strength write so recall is a pure
    # read and a re-run over the same haystack is identical.
    eng.db.record_retrieval = lambda *a, **k: 0.0  # type: ignore[method-assign]
    return eng


def require_real_model() -> None:
    """Assert the real cross-encoder reranker is loadable (no stub retrieval)."""
    from phileas import reranker

    try:
        reranker._ensure_model()
    except reranker.RerankerUnavailable as exc:
        raise SystemExit(f"REFUSING: real reranker unavailable ({exc}). Retrieval must run the real model.")
    print(f"RERANKER: loaded {reranker._MODEL_NAME}")
