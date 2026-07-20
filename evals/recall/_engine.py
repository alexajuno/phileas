"""Build an isolated MemoryEngine for the recall eval, in its own profile.

Same hard safety as the cold-start eval: refuses to run unless the resolved home
is the dedicated throwaway profile, so a misconfigured PHILEAS_PROFILE can never
write into the real ~/.phileas graph. Uses a profile distinct from cold-start
(``mara-eval`` → ~/.phileas-mara-eval) so the two evals never share a store.
Direct GraphStore (daemon-free), mirroring the CLI's _get_engine.
"""
from __future__ import annotations

import logging
from pathlib import Path

from phileas.config import DEFAULT_PROFILE, resolve_home

EXPECTED_PROFILE = "mara-eval"
EXPECTED_HOME = resolve_home(EXPECTED_PROFILE)


def build_engine():
    for n in ("sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(n).setLevel(logging.ERROR)

    from phileas.config import load_config
    from phileas.db import Database
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    cfg = load_config(profile=EXPECTED_PROFILE)
    home = Path(cfg.home).resolve()
    if home == resolve_home(DEFAULT_PROFILE).resolve():
        raise SystemExit(f"REFUSING: resolved home is the real graph {home}")
    if home.name != EXPECTED_PROFILE:
        raise SystemExit(f"REFUSING: unexpected home {home} (want a {EXPECTED_PROFILE} profile)")

    db = Database(path=cfg.db_path)
    vector = VectorStore(path=cfg.chroma_path)
    graph = GraphStore(path=cfg.graph_path)  # direct KuzuDB, daemon-free
    return MemoryEngine(db=db, vector=vector, graph=graph, config=cfg), cfg
