"""CLI commands for Phileas.

Each command is a thin wrapper over MemoryEngine. Business logic lives
in the engine; commands handle argument parsing and output formatting.
"""

from __future__ import annotations

import asyncio
import json

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


def _daemon_call(method: str, params: dict | None = None) -> dict | None:
    """Try calling the daemon. Returns response or None if not running."""
    from phileas.daemon import call

    return call(method, params)


def _get_engine() -> MemoryEngine:
    """Create a MemoryEngine from the current config. Suppresses model loading noise."""
    import logging

    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    cfg = load_config()
    db = Database(path=cfg.db_path)
    vector = VectorStore(path=cfg.chroma_path)
    graph = GraphStore(path=cfg.graph_path)
    return MemoryEngine(db=db, vector=vector, graph=graph, config=cfg)


def _resolve_id(engine: MemoryEngine, short_id: str) -> str | None:
    """Resolve a short ID prefix to a full UUID. Returns None if no match or ambiguous."""
    # Try exact match first
    if engine.db.get_item(short_id):
        return short_id
    # Prefix match
    items = engine.db.get_active_items() + [
        i for i in (engine.db.get_items_by_tier(2) + engine.db.get_items_by_tier(3)) if i.status == "archived"
    ]
    matches = [i.id for i in items if i.id.startswith(short_id)]
    if len(matches) == 1:
        return matches[0]
    return None


# ------------------------------------------------------------------
# status
# ------------------------------------------------------------------


@click.command()
def status():
    """Show system health and memory statistics."""
    try:
        resp = _daemon_call("status")
        if resp and resp.get("ok"):
            print_status(resp["result"])
            return

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
@click.option(
    "--type",
    "memory_type",
    default="knowledge",
    help="Memory type (profile, event, knowledge, behavior, reflection).",
)
@click.option("--importance", default=None, type=int, help="Importance score 1-10 (auto-scored by LLM if omitted).")
def remember(text: str, memory_type: str, importance: int | None):
    """Store a memory."""
    try:
        resp = _daemon_call(
            "memorize",
            {
                "summary": text,
                "memory_type": memory_type,
                "importance": importance,
            },
        )
        if resp and resp.get("ok"):
            result = resp["result"]
            print_memory_stored(result)
            if result.get("contradiction"):
                c = result["contradiction"]
                print_warning(f"Contradiction: {c.get('explanation', '')}")
            return

        engine = _get_engine()
        result = engine.memorize(
            summary=text,
            memory_type=memory_type,
            importance=importance,
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
        resp = _daemon_call("recall", {"query": query, "top_k": top_k, "memory_type": memory_type})
        if resp and resp.get("ok"):
            print_memories(resp["result"], title=f"Results for '{query}'")
            return

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
        resolved = _resolve_id(engine, memory_id)
        if not resolved:
            print_error(f"Memory {memory_id} not found.")
            raise SystemExit(1)
        msg = engine.forget(resolved, reason=reason)
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
        resolved = _resolve_id(engine, memory_id)
        if not resolved:
            print_error(f"Memory {memory_id} not found.")
            raise SystemExit(1)
        result = engine.update(resolved, summary)
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
        resolved = _resolve_id(engine, memory_id)
        if not resolved:
            print_error(f"Memory {memory_id} not found.")
            raise SystemExit(1)
        item = engine.db.get_item(resolved)
        if not item:
            print_error(f"Memory {memory_id} not found.")
            raise SystemExit(1)

        print_memory_detail(
            {
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
            }
        )
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
            )
            print_memory_stored(result)
            stored += 1

        print_success(f"Ingested {stored} memories from source.")
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
# reflect
# ------------------------------------------------------------------


@click.command()
@click.option("--date", default=None, help="Date to reflect on (YYYY-MM-DD). Defaults to today.")
def reflect(date: str | None):
    """Synthesize insights from a day's memories."""
    try:
        resp = _daemon_call("reflect", {"date": date})
        if resp and resp.get("ok"):
            insights = resp["result"]
            if not insights:
                print_error("No insights extracted (not enough data or already reflected).")
                return
            print_success(f"Extracted {len(insights)} insight(s):")
            for ins in insights:
                console.print(f"  [{ins.get('type', 'reflection')}] {ins['summary']}")
            return

        # Fallback: direct engine
        engine = _get_engine()
        insights = engine.reflect(target_date=date)
        if not insights:
            print_error("No insights extracted (not enough data or already reflected).")
            return
        print_success(f"Extracted {len(insights)} insight(s):")
        for ins in insights:
            console.print(f"  [{ins.get('type', 'reflection')}] {ins['summary']}")
    except SystemExit:
        raise
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# infer-graph
# ------------------------------------------------------------------


@click.command("infer-graph")
def infer_graph():
    """Run graph inference to discover relationships and patterns."""
    try:
        resp = _daemon_call("infer_graph")
        if resp and resp.get("ok"):
            result = resp["result"]
            if result["memories_processed"] == 0:
                console.print("No new memories to process.")
                return
            print_success(
                f"Processed {result['memories_processed']} memories: "
                f"{result['edges_added']} edges added, "
                f"{result['insights_stored']} insights stored."
            )
            return

        engine = _get_engine()
        result = engine.infer_graph()
        if result["memories_processed"] == 0:
            console.print("No new memories to process.")
            return
        print_success(
            f"Processed {result['memories_processed']} memories: "
            f"{result['edges_added']} edges added, "
            f"{result['insights_stored']} insights stored."
        )
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

            result = asyncio.run(detect_contradictions(engine.llm, new_memory=item.summary, existing_memories=related))
            if result.get("contradicts"):
                found.append(
                    {
                        "memory_id": item.id,
                        "summary": item.summary,
                        "conflicting_ids": result.get("conflicting_ids", []),
                        "explanation": result.get("explanation", ""),
                    }
                )

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


# ------------------------------------------------------------------
# start / stop (daemon)
# ------------------------------------------------------------------


@click.command()
def start():
    """Start the Phileas daemon (keeps models loaded for fast CLI)."""
    from phileas.daemon import is_running
    from phileas.daemon import start as daemon_start

    port = is_running()
    if port:
        console.print(f"Daemon already running on port {port}.")
        return

    try:
        console.print("Starting Phileas daemon...")
        port = daemon_start()
        console.print(f"[green]Daemon started[/green] on port {port}.")
        console.print("[dim]Models are loaded. CLI commands will be fast now.[/dim]")
    except Exception as exc:
        print_error(f"Failed to start daemon: {exc}")
        raise SystemExit(1)


@click.command()
def stop_cmd():
    """Stop the Phileas daemon."""
    from phileas.daemon import stop as daemon_stop

    if daemon_stop():
        print_success("Daemon stopped.")
    else:
        console.print("Daemon is not running.")


# ------------------------------------------------------------------
# usage
# ------------------------------------------------------------------


@click.command()
@click.option("--recent", default=0, type=int, help="Show N most recent LLM calls.")
def usage(recent: int):
    """Show LLM usage statistics — tokens, cost, requests by operation."""
    try:
        from phileas.config import load_config
        from phileas.llm.usage import UsageTracker

        cfg = load_config()
        usage_db = cfg.home / "usage.db"
        if not usage_db.exists():
            print_error("No usage data yet. Make some LLM calls first (e.g. phileas remember).")
            raise SystemExit(1)

        tracker = UsageTracker(usage_db)

        # Summary
        summary = tracker.get_summary()
        from rich.table import Table

        table = Table(title="LLM Usage Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total requests", str(summary["total_requests"]))
        table.add_row("Successful", str(summary["successful"]))
        table.add_row("Failed", str(summary["failed"]))
        table.add_row("Prompt tokens", f"{summary['total_prompt_tokens']:,}")
        table.add_row("Completion tokens", f"{summary['total_completion_tokens']:,}")
        table.add_row("Total tokens", f"{summary['total_tokens']:,}")
        table.add_row("Total cost", f"${summary['total_cost_usd']:.4f}")
        table.add_row("Avg latency", f"{summary['avg_latency_ms']:.0f}ms")
        console.print(table)

        # Per-operation breakdown
        by_op = tracker.get_by_operation()
        if by_op:
            op_table = Table(title="By Operation")
            op_table.add_column("Operation", style="cyan")
            op_table.add_column("Requests", style="green")
            op_table.add_column("Tokens", style="yellow")
            op_table.add_column("Cost", style="green")
            op_table.add_column("Avg ms", style="dim")
            op_table.add_column("Fails", style="red")
            for row in by_op:
                op_table.add_row(
                    row["operation"],
                    str(row["requests"]),
                    f"{row['total_tokens']:,}",
                    f"${row['cost_usd']:.4f}",
                    f"{row['avg_latency_ms']:.0f}",
                    str(row["failures"]) if row["failures"] else "",
                )
            console.print(op_table)

        # Recent calls
        if recent > 0:
            recent_calls = tracker.get_recent(limit=recent)
            if recent_calls:
                r_table = Table(title=f"Recent {len(recent_calls)} Calls")
                r_table.add_column("Time", style="dim")
                r_table.add_column("Operation", style="cyan")
                r_table.add_column("Model")
                r_table.add_column("Tokens", style="yellow")
                r_table.add_column("Cost", style="green")
                r_table.add_column("ms", style="dim")
                r_table.add_column("OK", style="green")
                for call in recent_calls:
                    ts = call["created_at"][:19].replace("T", " ")
                    ok = "[green]Y[/green]" if call["success"] else f"[red]N[/red] {call.get('error', '')[:30]}"
                    r_table.add_row(
                        ts,
                        call["operation"],
                        call["model"] or "",
                        str(call["total_tokens"]),
                        f"${call['cost_usd']:.4f}",
                        f"{call['latency_ms']:.0f}",
                        ok,
                    )
                console.print(r_table)

        tracker.close()
    except SystemExit:
        raise
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)
