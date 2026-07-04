"""Build an isolated MemoryEngine for the temporal-deixis eval, in its own profile.

Same hard safety as the sibling recall/coldstart evals: refuses to run unless the
resolved home is the dedicated throwaway profile, so a misconfigured
PHILEAS_PROFILE can never write into the real ~/.phileas graph. Uses a profile
distinct from the others (``temporal-eval`` -> ~/.phileas-temporal-eval) so the
evals never share a store. Direct GraphStore (daemon-free), mirroring the CLI's
_get_engine; because the store is isolated, the real daemon's Kuzu write lock on
~/.phileas never blocks these in-process reads.
"""

from __future__ import annotations

import logging
from pathlib import Path

EXPECTED_PROFILE = "temporal-eval"
EXPECTED_HOME = Path.home() / ".phileas-temporal-eval"
FORBIDDEN_HOME = Path.home() / ".phileas"


def build_engine():
    for n in ("sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(n).setLevel(logging.ERROR)

    from phileas.config import load_config
    from phileas.db import Database
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    # Pin the home explicitly (not via profile->home resolution, which varies by
    # config layer) so the throwaway store is always this fixed directory.
    cfg = load_config(home=EXPECTED_HOME, profile=EXPECTED_PROFILE)
    home = Path(cfg.home).resolve()
    if home == FORBIDDEN_HOME.resolve():
        raise SystemExit(f"REFUSING: resolved home is the real graph {home}")
    if home != EXPECTED_HOME.resolve():
        raise SystemExit(f"REFUSING: unexpected home {home} (want {EXPECTED_HOME})")

    db = Database(path=cfg.db_path)
    vector = VectorStore(path=cfg.chroma_path)
    graph = GraphStore(path=cfg.graph_path)  # direct KuzuDB, daemon-free
    return MemoryEngine(db=db, vector=vector, graph=graph, config=cfg), cfg
