"""Live-daemon probe: constructs MemoryEngine with GraphProxy (the MCP-side
code path) and runs the full recall pipeline against the running daemon.

Unlike scripts/probe_recall.py, nothing is snapshotted — this hits the real
~/.phileas stores via the daemon's graph_read/graph_write endpoints, so it
exercises exactly the code paths the MCP `recall` tool uses.

Usage:
    uv run python scripts/probe_recall_live.py \\
        --query "đố biết chị ở trên mình nhắc đến là ai" --top-k 10

Requires the daemon to be running and loading the current source.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phileas.config import load_config  # noqa: E402
from phileas.db import Database  # noqa: E402
from phileas.engine import MemoryEngine  # noqa: E402
from phileas.graph_proxy import GraphProxy  # noqa: E402
from phileas.llm.query_rewrite import analyze_query  # noqa: E402
from phileas.llm.referent_resolve import (  # noqa: E402
    build_person_candidates,
    resolve_referents,
)
from phileas.vector import VectorStore  # noqa: E402


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


async def _probe_stage0(engine: MemoryEngine, query: str) -> dict:
    analysis = await analyze_query(engine.llm, query)
    hr("stage 0a — analyze_query")
    print(f"queries                      : {analysis['queries']}")
    print(f"needs_referent_resolution    : {analysis['needs_referent_resolution']}")
    print(f"pronoun_hints                : {analysis['pronoun_hints']}")

    if not analysis.get("needs_referent_resolution"):
        print("\n(resolver will not fire)")
        return {"analysis": analysis, "candidates": None, "resolved": []}

    hr("stage 0b — build_person_candidates (via GraphProxy → daemon)")
    candidates = build_person_candidates(engine.graph, engine.db, top_n=15)
    for c in candidates:
        theme = (c["recent_summaries"] or ["—"])[0][:110]
        print(f"  - {c['name']:<24} {c['memory_count']:>4} memories  last {c['last_mentioned'] or '—':<10}  {theme!r}")
    if not candidates:
        print("  (no Person candidates — proxy call probably returned []; check daemon logs)")
        return {"analysis": analysis, "candidates": [], "resolved": []}

    hr("stage 0c — resolve_referents")
    resolved = await resolve_referents(engine.llm, query, analysis.get("pronoun_hints") or [], candidates)
    print(f"resolved names (ranked): {resolved}")
    return {"analysis": analysis, "candidates": candidates, "resolved": resolved}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config()
    print(f"PHILEAS_HOME            : {cfg.home}")
    print(f"LLM provider/model      : {cfg.llm.provider} / {cfg.llm.model}")

    db = Database(path=cfg.db_path)
    vector = VectorStore(path=cfg.chroma_path)
    graph = GraphProxy()
    engine = MemoryEngine(db=db, vector=vector, graph=graph, config=cfg)
    print(f"engine.llm.available    : {engine.llm.available}")
    print(f"daemon graph status     : {graph.status()}")

    hr(f"query: {args.query!r}  (top_k={args.top_k})")
    stage0 = asyncio.run(_probe_stage0(engine, args.query))

    hr("full engine.recall (LLM enabled, GraphProxy)")
    results = engine.recall(args.query, top_k=args.top_k)
    if not results:
        print("(no results)")
    for i, r in enumerate(results, 1):
        print(f"  {i:>2}. [{r['id'][:8]}…] score={r['score']:.2f} {r['type']:<11} {r['summary'][:120]}")

    hr("summary")
    resolved = stage0["resolved"]
    print(f"referent resolver picked: {resolved or '(nothing)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
