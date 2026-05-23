"""Run each fixture query under both Path 3 modes, back-to-back.

Writes a run marker (a unique tag) to the env-var $PHILEAS_PROBE_TAG so the
compare script can pick out *this* probe's traces from the metrics DB and
ignore the user's normal recall traffic.

Driving via engine.recall directly (not MCP) means we need the daemon's
graph backend to know about lookup_nodes — restart the daemon before
running this if it's been running on an older revision.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from pathlib import Path

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph_proxy import GraphProxy
from phileas.vector import VectorStore
from tests.path3_fixtures import QUERIES


def make_engine() -> MemoryEngine:
    config = load_config()
    db = Database(path=config.db_path)
    vector = VectorStore(path=config.chroma_path)
    graph = GraphProxy()
    return MemoryEngine(db=db, vector=vector, graph=graph, config=config)


def latest_trace_id(metrics_db: Path) -> int:
    """Return the current max(id) in recall_traces. Used as a watermark so
    the compare script can pick out only traces written by this probe run."""
    if not metrics_db.exists():
        return 0
    conn = sqlite3.connect(str(metrics_db))
    try:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM recall_traces").fetchone()
        return int(row[0] if row else 0)
    finally:
        conn.close()


def main() -> int:
    config = load_config()
    tag = os.environ.get("PHILEAS_PROBE_TAG") or f"path3-{uuid.uuid4().hex[:8]}"
    os.environ["PHILEAS_PROBE_TAG"] = tag

    out = Path("tests/path3_runs")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_tag").write_text(tag)
    watermark = latest_trace_id(config.home / "metrics.db")
    (out / "last_watermark").write_text(str(watermark))
    print(f"probe tag: {tag}  (trace_id > {watermark})")

    engine = make_engine()
    start = time.time()
    _ = config  # silence unused — kept for future use

    for mode in ("legacy", "index"):
        os.environ["PHILEAS_PATH3"] = mode
        print(f"\n--- mode={mode} ---")
        for query, label in QUERIES:
            results = engine.recall(query, top_k=10)
            top1 = results[0]["score"] if results else None
            top1_str = f"{top1:.3f}" if top1 is not None else "-"
            print(f"  [{label:25s}] {query!r:40s} -> {len(results)} hits, top1={top1_str}")

    dur = time.time() - start
    print(f"\ndone in {dur:.1f}s. compare with:\n  python tests/path3_compare.py {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
