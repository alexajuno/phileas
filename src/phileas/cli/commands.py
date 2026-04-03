"""CLI commands for Phileas.

Each command is a thin wrapper over MemoryEngine. Business logic lives
in the engine; commands handle argument parsing and output formatting.
"""

from __future__ import annotations

import asyncio
import json
import sys

import click

from phileas.cli.formatter import (
    console,
    print_error,
    print_memories,
    print_memory_detail,
    print_memory_stored,
    print_status,
    print_success,
    print_warning,
)
from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore


def _get_engine() -> MemoryEngine:
    """Create a MemoryEngine from the current config."""
    cfg = load_config()
    db = Database(path=cfg.db_path)
    vector = VectorStore(path=cfg.chroma_path)
    graph = GraphStore(path=cfg.graph_path)
    return MemoryEngine(db=db, vector=vector, graph=graph, config=cfg)


# ------------------------------------------------------------------
# status
# ------------------------------------------------------------------


@click.command()
def status():
    """Show system health and memory statistics."""
    try:
        engine = _get_engine()
        stats = engine.status()
        print_status(stats)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# remember
# ------------------------------------------------------------------


@click.command()
@click.argument("text")
@click.option("--type", "memory_type", default="knowledge", help="Memory type (profile, event, knowledge, behavior, reflection).")
@click.option("--importance", default=5, type=int, help="Importance score 1-10.")
def remember(text: str, memory_type: str, importance: int):
    """Store a memory."""
    try:
        engine = _get_engine()
        result = engine.memorize(
            summary=text,
            memory_type=memory_type,
            importance=importance,
            auto_importance=False,
        )
        print_memory_stored(result)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# recall
# ------------------------------------------------------------------


@click.command()
@click.argument("query")
@click.option("--top-k", default=10, type=int, help="Maximum results to return.")
@click.option("--type", "memory_type", default=None, help="Filter by memory type.")
def recall(query: str, top_k: int, memory_type: str | None):
    """Search memories by query."""
    try:
        engine = _get_engine()
        results = engine.recall(query, top_k=top_k, memory_type=memory_type)
        print_memories(results, title=f"Results for '{query}'")
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# forget
# ------------------------------------------------------------------


@click.command()
@click.argument("memory_id")
@click.option("--reason", default=None, help="Reason for archiving.")
def forget(memory_id: str, reason: str | None):
    """Archive a memory by ID."""
    try:
        engine = _get_engine()
        msg = engine.forget(memory_id, reason=reason)
        print_success(msg)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# update
# ------------------------------------------------------------------


@click.command("update")
@click.argument("memory_id")
@click.argument("summary")
def update_cmd(memory_id: str, summary: str):
    """Update a memory's summary."""
    try:
        engine = _get_engine()
        result = engine.update(memory_id, summary)
        if "error" in result:
            print_error(result["error"])
            raise SystemExit(1)
        print_success(f"Updated [{result['id'][:8]}] {result['summary']}")
        console.print(f"[dim]Old version archived as [{result['snapshot_id'][:8]}][/dim]")
    except SystemExit:
        raise
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# list
# ------------------------------------------------------------------


@click.command("list")
@click.option("--type", "memory_type", default=None, help="Filter by memory type.")
@click.option("--limit", default=20, type=int, help="Maximum items to show.")
def list_cmd(memory_type: str | None, limit: int):
    """Browse memories."""
    try:
        engine = _get_engine()
        if memory_type:
            items = engine.db.get_items_by_type(memory_type)
        else:
            items = engine.db.get_active_items()

        items = items[:limit]
        dicts = [
            {
                "id": item.id,
                "type": item.memory_type,
                "importance": item.importance,
                "summary": item.summary,
                "score": 0.0,
            }
            for item in items
        ]
        title = f"Memories (type={memory_type})" if memory_type else "All active memories"
        print_memories(dicts, title=title)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# show
# ------------------------------------------------------------------


@click.command()
@click.argument("memory_id")
def show(memory_id: str):
    """Show full detail of a memory."""
    try:
        engine = _get_engine()
        item = engine.db.get_item(memory_id)
        if not item:
            print_error(f"Memory {memory_id} not found.")
            raise SystemExit(1)

        print_memory_detail({
            "id": item.id,
            "summary": item.summary,
            "memory_type": item.memory_type,
            "importance": item.importance,
            "tier": item.tier,
            "status": item.status,
            "access_count": item.access_count,
            "daily_ref": item.daily_ref,
            "created_at": item.created_at.isoformat() if item.created_at else "",
            "updated_at": item.updated_at.isoformat() if item.updated_at else "",
        })
    except SystemExit:
        raise
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# ingest
# ------------------------------------------------------------------


@click.command()
@click.argument("source")
def ingest(source: str):
    """Extract memories from a file or text using LLM extraction."""
    try:
        engine = _get_engine()

        if not engine.llm.available:
            print_error("LLM not configured. Set provider and model in config.toml to use ingest.")
            raise SystemExit(1)

        from pathlib import Path

        from phileas.llm.extraction import extract_memories

        path = Path(source)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        else:
            text = source

        memories = asyncio.run(extract_memories(engine.llm, text))

        stored = 0
        for mem in memories:
            result = engine.memorize(
                summary=mem["summary"],
                memory_type=mem.get("memory_type", "knowledge"),
                importance=mem.get("importance", 5),
                entities=mem.get("entities"),
                relationships=mem.get("relationships"),
                auto_importance=False,
            )
            print_memory_stored(result)
            if not result.get("deduplicated"):
                stored += 1

        print_success(f"Ingested {stored} new memories from source.")
    except SystemExit:
        raise
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# consolidate
# ------------------------------------------------------------------


@click.command()
@click.option("--min-cluster", default=3, type=int, help="Minimum cluster size.")
@click.option("--max-clusters", default=10, type=int, help="Maximum clusters to process.")
def consolidate(min_cluster: int, max_clusters: int):
    """Find and merge similar memories using LLM consolidation."""
    try:
        engine = _get_engine()

        if not engine.llm.available:
            print_error("LLM not configured. Set provider and model in config.toml to use consolidate.")
            raise SystemExit(1)

        from phileas.llm.consolidation import consolidate_memories

        # Reuse the same cluster-finding logic from server.py
        tier2_items = engine.db.get_items_by_tier(2)
        unconsolidated = [item for item in tier2_items if item.consolidated_into is None]

        if not unconsolidated:
            print_warning("No unconsolidated tier-2 memories found.")
            return

        if len(unconsolidated) < min_cluster:
            print_warning(f"Only {len(unconsolidated)} unconsolidated memories -- need at least {min_cluster}.")
            return

        # Find clusters using vector similarity
        clusters: list[list[dict]] = []
        used_ids: set[str] = set()

        for item in unconsolidated:
            if item.id in used_ids:
                continue

            similar = engine.vector.search(item.summary, top_k=min_cluster * 3)
            cluster_ids = []
            for mem_id, sim in similar:
                if sim >= 0.7 and mem_id not in used_ids:
                    candidate = engine.db.get_item(mem_id)
                    is_eligible = (
                        candidate
                        and candidate.status == "active"
                        and candidate.tier == 2
                        and candidate.consolidated_into is None
                    )
                    if is_eligible:
                        cluster_ids.append((mem_id, sim))

            if len(cluster_ids) >= min_cluster:
                cluster = []
                for mem_id, _sim in cluster_ids:
                    candidate = engine.db.get_item(mem_id)
                    if candidate:
                        cluster.append({"id": candidate.id, "summary": candidate.summary})
                        used_ids.add(mem_id)
                clusters.append(cluster)

            if len(clusters) >= max_clusters:
                break

        if not clusters:
            print_warning(f"No clusters of size >= {min_cluster} found.")
            return

        console.print(f"Found {len(clusters)} cluster(s). Consolidating...")

        consolidated_count = 0
        for i, cluster in enumerate(clusters, 1):
            result = asyncio.run(consolidate_memories(engine.llm, cluster))
            if result:
                engine.memorize(
                    summary=result["summary"],
                    memory_type="reflection",
                    importance=result["importance"],
                    tier=3,
                    auto_importance=False,
                )
                console.print(f"  Cluster {i}: {result['summary'][:80]}...")
                consolidated_count += 1

        print_success(f"Consolidated {consolidated_count} clusters into tier-3 memories.")
    except SystemExit:
        raise
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# contradictions
# ------------------------------------------------------------------


@click.command()
@click.option("--limit", default=50, type=int, help="Maximum memories to scan.")
def contradictions(limit: int):
    """Scan for conflicting memories using LLM contradiction detection."""
    try:
        engine = _get_engine()

        if not engine.llm.available:
            print_error("LLM not configured. Set provider and model in config.toml to use contradictions.")
            raise SystemExit(1)

        from phileas.llm.contradiction import detect_contradictions

        items = engine.db.get_active_items()[:limit]
        if not items:
            print_warning("No active memories to scan.")
            return

        console.print(f"Scanning {len(items)} memories for contradictions...")

        found = []
        for item in items:
            related = engine.recall(item.summary, top_k=5, _skip_llm=True)
            # Exclude self from related
            related = [r for r in related if r["id"] != item.id]
            if not related:
                continue

            result = asyncio.run(
                detect_contradictions(engine.llm, new_memory=item.summary, existing_memories=related)
            )
            if result.get("contradicts"):
                found.append({
                    "memory_id": item.id,
                    "summary": item.summary,
                    "conflicting_ids": result.get("conflicting_ids", []),
                    "explanation": result.get("explanation", ""),
                })

        if not found:
            print_success("No contradictions found.")
            return

        console.print(f"\n[yellow]Found {len(found)} contradiction(s):[/yellow]")
        for c in found:
            console.print(f"\n  [{c['memory_id'][:8]}] {c['summary']}")
            console.print(f"  [dim]Conflicts with: {', '.join(cid[:8] for cid in c['conflicting_ids'])}[/dim]")
            console.print(f"  [dim]{c['explanation']}[/dim]")
    except SystemExit:
        raise
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# export
# ------------------------------------------------------------------


@click.command("export")
@click.option("--format", "fmt", default="json", type=click.Choice(["json"]), help="Export format.")
@click.option("--output", "-o", default=None, help="Output file path (default: stdout).")
def export_cmd(fmt: str, output: str | None):
    """Export memories as JSON."""
    try:
        engine = _get_engine()
        items = engine.db.get_active_items()

        data = [
            {
                "id": item.id,
                "summary": item.summary,
                "memory_type": item.memory_type,
                "importance": item.importance,
                "tier": item.tier,
                "status": item.status,
                "access_count": item.access_count,
                "daily_ref": item.daily_ref,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            }
            for item in items
        ]

        json_str = json.dumps(data, indent=2)

        if output:
            from pathlib import Path

            Path(output).write_text(json_str, encoding="utf-8")
            print_success(f"Exported {len(data)} memories to {output}")
        else:
            click.echo(json_str)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# serve
# ------------------------------------------------------------------


@click.command()
def serve():
    """Start the Phileas MCP server."""
    try:
        from phileas.server import mcp

        mcp.run()
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# init
# ------------------------------------------------------------------


@click.command("init")
def init_cmd():
    """Set up Phileas interactively."""
    from phileas.cli.wizard import run_wizard
    run_wizard()
