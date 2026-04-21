"""Probe the full recall pipeline against a read-only snapshot of ~/.phileas.

Bypasses MCP entirely. Snapshots the live stores into a temp PHILEAS_HOME,
builds MemoryEngine with LLM enabled, and prints every stage-0 decision
so we can see whether referent resolution actually resolves "chị" to
phuongtq on the real graph.

Usage:
    uv run python scripts/probe_recall.py \\
        --query "đố biết chị ở trên mình nhắc đến là ai" \\
        --top-k 10

The daemon keeps holding the live Kuzu lock, so we copy files to a
temp directory and open the copy read-write. Safe to run while the
daemon is up.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phileas.config import load_config  # noqa: E402
from phileas.db import Database  # noqa: E402
from phileas.engine import MemoryEngine  # noqa: E402
from phileas.graph import GraphStore  # noqa: E402
from phileas.llm.query_rewrite import analyze_query  # noqa: E402
from phileas.llm.referent_resolve import (  # noqa: E402
    build_person_candidates,
    resolve_referents,
)
from phileas.vector import VectorStore  # noqa: E402


def snapshot_home(live: Path, snap: Path) -> None:
    """Copy memory.db, chroma/, graph, graph.wal into snap.

    Skips runtime files (daemon.pid/port/log, ingest.lock, etc.) so the
    snapshot opens cleanly without thinking a daemon is already running.
    """
    snap.mkdir(parents=True, exist_ok=True)
    # config.toml — so load_config picks the real provider/model
    if (live / "config.toml").exists():
        shutil.copy2(live / "config.toml", snap / "config.toml")
    # SQLite
    if (live / "memory.db").exists():
        shutil.copy2(live / "memory.db", snap / "memory.db")
    # Chroma (directory)
    if (live / "chroma").exists():
        shutil.copytree(live / "chroma", snap / "chroma", dirs_exist_ok=True)
    # Kuzu (file + WAL)
    if (live / "graph").exists():
        shutil.copy2(live / "graph", snap / "graph")
    if (live / "graph.wal").exists():
        shutil.copy2(live / "graph.wal", snap / "graph.wal")


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


async def _probe_stage0(engine: MemoryEngine, query: str) -> dict:
    """Run stage-0 analysis + referent resolution and return raw outputs."""
    analysis = await analyze_query(engine.llm, query)
    hr("stage 0a — analyze_query")
    print(f"queries                      : {analysis['queries']}")
    print(f"needs_referent_resolution    : {analysis['needs_referent_resolution']}")
    print(f"pronoun_hints                : {analysis['pronoun_hints']}")

    if not analysis.get("needs_referent_resolution"):
        print("\n(analyze_query did not flag the query as ambiguous; resolve_referents will not fire.)")
        return {"analysis": analysis, "candidates": None, "resolved": []}

    hr("stage 0b — build_person_candidates")
    candidates = build_person_candidates(engine.graph, engine.db, top_n=15)
    for c in candidates:
        theme = (c["recent_summaries"] or ["—"])[0][:110]
        print(f"  - {c['name']:<24} {c['memory_count']:>4} memories  last {c['last_mentioned'] or '—':<10}  {theme!r}")
    if not candidates:
        print("  (no Person candidates in the graph; resolution skipped)")
        return {"analysis": analysis, "candidates": [], "resolved": []}

    hr("stage 0c — resolve_referents")
    resolved = await resolve_referents(engine.llm, query, analysis.get("pronoun_hints") or [], candidates)
    print(f"resolved names (ranked): {resolved}")
    return {"analysis": analysis, "candidates": candidates, "resolved": resolved}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="query to probe")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--live-home",
        type=Path,
        default=Path.home() / ".phileas",
        help="source PHILEAS_HOME to snapshot (default ~/.phileas)",
    )
    ap.add_argument(
        "--keep-snapshot",
        action="store_true",
        help="don't delete the temp snapshot dir after the probe",
    )
    args = ap.parse_args()

    snap = Path(tempfile.mkdtemp(prefix="phileas-probe-"))
    print(f"snapshot dir: {snap}")
    try:
        snapshot_home(args.live_home, snap)

        cfg = load_config(home=snap)
        print(f"LLM provider/model      : {cfg.llm.provider} / {cfg.llm.model}")
        print(f"LLM available           : {cfg.llm.provider is not None and cfg.llm.model is not None}")

        db = Database(path=cfg.db_path)
        vector = VectorStore(path=cfg.chroma_path)
        graph = GraphStore(path=cfg.graph_path)
        engine = MemoryEngine(db=db, vector=vector, graph=graph, config=cfg)
        print(f"engine.llm.available    : {engine.llm.available}")

        hr(f"query: {args.query!r}  (top_k={args.top_k})")

        stage0 = asyncio.run(_probe_stage0(engine, args.query))

        hr("full engine.recall (LLM enabled)")
        results = engine.recall(args.query, top_k=args.top_k)
        if not results:
            print("(no results)")
        for i, r in enumerate(results, 1):
            print(f"  {i:>2}. [{r['id'][:8]}…] score={r['score']:.2f} {r['type']:<11} {r['summary'][:120]}")

        hr("summary")
        resolved = stage0["resolved"]
        if resolved:
            print(f"referent resolver picked: {resolved}")
            # Did any of the resolved entities' memories actually surface?
            surfaced = any(r_name.lower() in item["summary"].lower() for r_name in resolved for item in results)
            print(f"resolved-entity content surfaced in results: {surfaced}")
        else:
            print("referent resolver returned no names")

        try:
            db.close()
            vector.close()
            graph.close()
        except Exception:
            pass
        return 0
    finally:
        if args.keep_snapshot:
            print(f"(kept snapshot at {snap})")
        else:
            shutil.rmtree(snap, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
