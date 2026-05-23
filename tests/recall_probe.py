"""Drive engine.recall directly to capture instrumented traces.

Bypasses MCP — the MCP server in the Claude Code session has the
pre-instrumentation engine loaded. We talk to the same KuzuDB through
GraphProxy (daemon) so we don't fight over the file lock.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph_proxy import GraphProxy
from phileas.vector import VectorStore


def make_engine() -> MemoryEngine:
    config = load_config()
    db = Database(path=config.db_path)
    vector = VectorStore(path=config.chroma_path)
    graph = GraphProxy()
    return MemoryEngine(db=db, vector=vector, graph=graph, config=config)


def run_queries(engine: MemoryEngine, queries: list[str], top_k: int = 10) -> None:
    for q in queries:
        results = engine.recall(q, top_k=top_k)
        print(f"\n=== query: {q!r}  ({len(results)} results) ===")
        for r in results[:5]:
            print(f"  [{r['type']:10s}] score={r['score']:.2f}  {r['summary'][:100]}")


def dump_traces(metrics_db: Path, limit: int = 20) -> None:
    conn = sqlite3.connect(str(metrics_db))
    rows = conn.execute(
        """SELECT created_at, source, query, latency_ms, candidate_count, extra
           FROM recall_traces
           WHERE source = 'engine.recall'
           ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    print(f"\n=== last {len(rows)} recall traces ===")
    for created_at, source, query, lat, cand, extra in rows:
        extra_obj = json.loads(extra) if extra else {}
        print(f"\n  {created_at}  q={query!r}")
        print(f"    latency={lat:.0f}ms  candidates={cand}  top_score={extra_obj.get('top_score')}")
        gh = extra_obj.get("result_gather_histogram") or {}
        up = extra_obj.get("result_unique_path_counts") or {}
        if gh:
            print(f"    paths in top-K: {gh}")
        if up:
            print(f"    unique-path top-K: {up}  <- results matched by exactly ONE path")


if __name__ == "__main__":
    config = load_config()
    engine = make_engine()
    queries = sys.argv[1:] or [
        "badminton",
        "what did the user say about badminton",
        "phuongtq preferences",
        "recent decisions about phileas architecture",
        "the user is a software engineer",
        "memory layer design",
    ]
    run_queries(engine, queries)
    dump_traces(config.home / "metrics.db")
